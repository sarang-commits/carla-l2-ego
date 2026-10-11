"""Offline integration tests for the Mission 10A longitudinal controller.

These tests drive the accepted Mission 6A runtime, Mission 7 temporal pipeline,
Mission 8 safety supervisor, and Mission 9 lateral controller for real,
injecting only a Mission 5-shaped perception callable, synthetic telemetry, and
explicit clocks.  The ego speed observation is detached from the very same
immutable :class:`TelemetrySnapshot` the perception stack consumed, so the
snapshot revision that binds all four sources is a real one rather than a
hand-written constant.

CARLA is neither imported nor contacted, and no throttle or brake value produced
here reaches a vehicle: Mission 10A publishes requests only.
"""

from __future__ import annotations

import threading
import unittest
from dataclasses import fields

from control.lateral import (
    LateralControlMode,
    LateralControlReason,
    LateralController,
)
from control.longitudinal import (
    DEFAULT_LONGITUDINAL_CONTROLLER_CONFIG,
    EgoSpeedObservation,
    LongitudinalControlMode,
    LongitudinalControlReason,
    LongitudinalController,
    LongitudinalControllerConfig,
    TargetSpeedMode,
    TargetSpeedOrigin,
    TargetSpeedRequest,
)
from perception.runtime import PerceptionRuntime, PerceptionRuntimeState
from perception.state import (
    LaneBoundaryObservation,
    LaneBoundarySide,
    LaneDetectionState,
    LaneObservation,
    LaneObservationReason,
    NormalizedImagePoint,
)
from perception.temporal import (
    TemporalLaneObservation,
    TemporalLanePipeline,
    TemporalLaneState,
)
from safety.supervisor import (
    ComponentHealth,
    SafetyReason,
    SafetyState,
    SafetySupervisor,
)
from telemetry.state import (
    PIXEL_FORMAT_RGB8,
    CameraFrame,
    CameraFrameMetadata,
    EgoKinematics,
    EulerDegrees,
    SampleStamp,
    TelemetrySnapshot,
    TransformState,
    Vector3,
    speed_from_velocity,
)


WAIT_SECONDS = 5.0
BASE_MONOTONIC = 100.0

# One control step per published frame, fast enough that Mission 8 never calls
# the evidence stale and that every ego sample stays inside the Mission 10A
# freshness window.
STEP_SECONDS = 0.10

CONFIG = DEFAULT_LONGITUDINAL_CONTROLLER_CONFIG


class ConstantClock:
    """Thread-safe constant clock for zero-duration runtime processing."""

    def __init__(self, value: float = BASE_MONOTONIC) -> None:
        self.value = float(value)
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.value


def control_time(frame: int) -> float:
    return BASE_MONOTONIC + frame * STEP_SECONDS


def snapshot(
    frame: int,
    *,
    monotonic: float | None = None,
    velocity: tuple[float, float, float] = (1.0, 0.0, 0.0),
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ego_monotonic: float | None = None,
    ego_frame: int | None = None,
    vehicle_alive: bool = True,
) -> TelemetrySnapshot:
    """One immutable snapshot carrying both a camera frame and ego kinematics."""
    captured = control_time(frame) if monotonic is None else monotonic
    stamp = SampleStamp(
        frame=frame, sim_time=frame * STEP_SECONDS, monotonic=captured
    )
    metadata = CameraFrameMetadata(
        stamp=stamp,
        width_px=1,
        height_px=1,
        horizontal_fov_deg=90.0,
        pixel_format=PIXEL_FORMAT_RGB8,
        row_stride_bytes=3,
    )
    roll, pitch, yaw = rotation
    speed_mps, speed_kph = speed_from_velocity(*velocity)
    selected_ego_frame = frame if ego_frame is None else ego_frame
    ego = EgoKinematics(
        stamp=SampleStamp(
            frame=selected_ego_frame,
            sim_time=selected_ego_frame * STEP_SECONDS,
            monotonic=captured if ego_monotonic is None else ego_monotonic,
        ),
        transform=TransformState(
            location=Vector3(0.0, 0.0, 0.0),
            rotation=EulerDegrees(roll=roll, pitch=pitch, yaw=yaw),
        ),
        velocity=Vector3(*velocity),
        speed_mps=speed_mps,
        speed_kph=speed_kph,
        vehicle_alive=vehicle_alive,
    )
    return TelemetrySnapshot(
        revision=frame,
        captured_monotonic=captured,
        camera=CameraFrame(metadata=metadata, rgb_bytes=b"\x10\x20\x30"),
        ego=ego,
    )


def boundary(
    side: LaneBoundarySide, *, confidence: float = 0.80
) -> LaneBoundaryObservation:
    if side is LaneBoundarySide.LEFT:
        upper_x, lower_x, slope, intercept = 0.34, 0.24, -0.20, 0.44
    else:
        upper_x, lower_x, slope, intercept = 0.66, 0.76, 0.20, 0.56
    return LaneBoundaryObservation(
        side=side,
        endpoints=(
            NormalizedImagePoint(upper_x, 0.50),
            NormalizedImagePoint(lower_x, 1.00),
        ),
        x_intercept=intercept,
        x_slope=slope,
        support_count=12,
        fit_residual=0.01,
        confidence=confidence,
    )


def lane_observation(
    value: TelemetrySnapshot,
    state: LaneDetectionState,
    *,
    offset: float = 0.0,
    heading: float = 0.0,
    confidence: float = 0.80,
) -> LaneObservation:
    """Build one Mission 5-shaped observation with the requested geometry."""
    assert value.camera is not None
    stamp = value.camera.metadata.stamp
    common = dict(
        snapshot_revision=value.revision,
        source_frame=stamp.frame,
        source_simulation_timestamp=stamp.sim_time,
        source_receive_monotonic=stamp.monotonic,
        state=state,
    )
    if state is LaneDetectionState.DETECTED:
        top_x = min(1.0, max(0.0, 0.50 + offset - heading * 0.25))
        bottom_x = min(1.0, max(0.0, 0.50 + offset + heading * 0.25))
        return LaneObservation(
            **common,
            reason=LaneObservationReason.DETECTED,
            left_boundary=boundary(LaneBoundarySide.LEFT, confidence=confidence),
            right_boundary=boundary(LaneBoundarySide.RIGHT, confidence=confidence),
            centerline_endpoints=(
                NormalizedImagePoint(top_x, 0.50),
                NormalizedImagePoint(bottom_x, 1.00),
            ),
            center_offset=offset,
            heading_proxy=heading,
            confidence=confidence,
        )
    if state is LaneDetectionState.PARTIAL:
        return LaneObservation(
            **common,
            reason=LaneObservationReason.SINGLE_PLAUSIBLE_BOUNDARY,
            left_boundary=boundary(LaneBoundarySide.LEFT, confidence=confidence),
            confidence=confidence,
        )
    if state is LaneDetectionState.NOT_DETECTED:
        return LaneObservation(
            **common,
            reason=LaneObservationReason.NO_PLAUSIBLE_BOUNDARY,
            confidence=0.0,
        )
    if state is LaneDetectionState.INPUT_UNUSABLE:
        return LaneObservation(
            **common,
            reason=LaneObservationReason.STATUS_UNUSABLE,
            confidence=0.0,
        )
    raise AssertionError(f"unsupported synthetic lane state: {state!r}")


class ScriptedPerception:
    """Injected Mission 5-shaped callable driven by a per-frame script."""

    def __init__(self, script: dict[int, tuple]) -> None:
        self._script = dict(script)

    def __call__(self, value: TelemetrySnapshot) -> LaneObservation:
        assert value.camera is not None
        frame = value.camera.metadata.stamp.frame
        state, offset, heading, confidence = self._script[frame]
        return lane_observation(
            value, state, offset=offset, heading=heading, confidence=confidence
        )


def detected(offset: float = 0.0, heading: float = 0.0, confidence: float = 0.80):
    return LaneDetectionState.DETECTED, offset, heading, confidence


MISSING = (LaneDetectionState.NOT_DETECTED, 0.0, 0.0, 0.0)
PARTIAL = (LaneDetectionState.PARTIAL, 0.0, 0.0, 0.80)
UNUSABLE = (LaneDetectionState.INPUT_UNUSABLE, 0.0, 0.0, 0.0)


def scenario_target(
    speed: float = 2.0,
    *,
    origin: TargetSpeedOrigin = TargetSpeedOrigin.SCENARIO,
    epoch: int = 0,
    revision: int = 0,
    issued: float = BASE_MONOTONIC,
    valid_until: float = BASE_MONOTONIC + 100.0,
) -> TargetSpeedRequest:
    return TargetSpeedRequest(
        mode=TargetSpeedMode.FORWARD if speed > 0.0 else TargetSpeedMode.STOP,
        target_speed_mps=speed,
        origin=origin,
        source_epoch=epoch,
        source_revision=revision,
        issued_monotonic=issued,
        valid_until_monotonic=valid_until,
    )


class LongitudinalIntegrationTestCase(unittest.TestCase):
    """Shared real-pipeline lifecycle helpers for the Mission 10A suite."""

    def build(self, script: dict[int, tuple], *, config=None):
        pipeline = TemporalLanePipeline(perception_callable=ScriptedPerception(script))
        runtime = PerceptionRuntime(clock=ConstantClock(), perception_callable=pipeline)
        supervisor = SafetySupervisor()
        lateral = LateralController()
        longitudinal = LongitudinalController(
            LongitudinalControllerConfig() if config is None else config
        )
        self.addCleanup(self.stop_runtime, runtime)
        runtime.start()
        return runtime, supervisor, lateral, longitudinal

    @staticmethod
    def stop_runtime(runtime: PerceptionRuntime) -> None:
        if runtime.state is not PerceptionRuntimeState.STOPPED:
            runtime.stop()

    def wait_for_processed(self, runtime: PerceptionRuntime, count: int) -> None:
        with runtime._condition:
            completed = runtime._condition.wait_for(
                lambda: (
                    runtime._inputs_processed >= count
                    or runtime._state is PerceptionRuntimeState.FAULTED
                ),
                timeout=WAIT_SECONDS,
            )
        self.assertTrue(completed, "runtime did not reach the expected disposition")

    def publish(self, runtime: PerceptionRuntime, value: TelemetrySnapshot, count: int):
        self.assertTrue(runtime.submit(value))
        self.wait_for_processed(runtime, count)
        self.assertIsNot(runtime.state, PerceptionRuntimeState.FAULTED, runtime.fault)
        published = runtime.latest_result
        self.assertIsNotNone(published)
        assert published is not None
        self.assertIsInstance(published.observation, TemporalLaneObservation)
        return published

    def step(
        self,
        runtime,
        supervisor,
        lateral,
        longitudinal,
        frame: int,
        *,
        target=None,
        now=None,
        processed=None,
        **snapshot_kwargs,
    ):
        """Publish, supervise, steer, and pace one frame through the real stack."""
        value = snapshot(frame, **snapshot_kwargs)
        published = self.publish(runtime, value, frame if processed is None else processed)
        control_now = control_time(frame) if now is None else now
        verdict = supervisor.update(
            published,
            now=control_now,
            health=ComponentHealth.from_components(runtime),
        )
        steering = lateral.update(published, verdict, now=control_now)
        speed = EgoSpeedObservation.from_telemetry_snapshot(value)
        request = longitudinal.update(
            verdict,
            steering,
            speed,
            scenario_target() if target is None else target,
            now=control_now,
        )
        return published, verdict, steering, request


class RealNominalTrackingTest(LongitudinalIntegrationTestCase):
    def test_a_real_nominal_pipeline_below_target_requests_throttle(self):
        runtime, supervisor, lateral, longitudinal = self.build(
            {frame: detected(0.10) for frame in range(1, 4)}
        )
        for frame in range(1, 4):
            published, verdict, steering, request = self.step(
                runtime, supervisor, lateral, longitudinal, frame,
                velocity=(1.0, 0.0, 0.0),
            )
            self.assertIs(
                published.observation.temporal_estimate.state,
                TemporalLaneState.TRACKING,
            )
            self.assertIs(verdict.state, SafetyState.NOMINAL)
            self.assertIs(verdict.reason, SafetyReason.NOMINAL_TRACKING)
            self.assertIs(steering.mode, LateralControlMode.NOMINAL)
            self.assertTrue(request.longitudinal_allowed, frame)
            self.assertIs(request.mode, LongitudinalControlMode.NOMINAL, frame)
            self.assertIs(request.reason, LongitudinalControlReason.NOMINAL_TRACKING)
            self.assertGreater(request.throttle_request, 0.0, frame)
            self.assertEqual(request.brake_request, 0.0, frame)
            self.assertLessEqual(
                request.throttle_request, CONFIG.nominal_throttle_authority
            )
            self.assertEqual(request.observed_forward_speed_mps, 1.0)
            self.assertEqual(request.snapshot_revision, frame)
            self.assertEqual(request.guidance_frame, frame)

    def test_a_real_pipeline_at_target_requests_neither_pedal(self):
        runtime, supervisor, lateral, longitudinal = self.build(
            {frame: detected(0.10) for frame in range(1, 3)}
        )
        for frame in range(1, 3):
            _, verdict, _, request = self.step(
                runtime, supervisor, lateral, longitudinal, frame,
                velocity=(2.0, 0.0, 0.0),
            )
            self.assertTrue(verdict.autonomy_allowed)
            self.assertTrue(request.longitudinal_allowed)
            self.assertEqual(request.throttle_request, 0.0)
            self.assertEqual(request.brake_request, 0.0)
            self.assertEqual(request.speed_error_mps, 0.0)

    def test_a_real_pipeline_above_target_requests_brake_only(self):
        runtime, supervisor, lateral, longitudinal = self.build(
            {frame: detected(0.10) for frame in range(1, 3)}
        )
        for frame in range(1, 3):
            _, _, _, request = self.step(
                runtime, supervisor, lateral, longitudinal, frame,
                velocity=(2.8, 0.0, 0.0),
            )
            self.assertTrue(request.longitudinal_allowed)
            self.assertEqual(request.throttle_request, 0.0)
            self.assertGreater(request.brake_request, 0.0)
            self.assertLessEqual(request.brake_request, CONFIG.max_service_brake)
            self.assertLess(request.speed_error_mps, 0.0)

    def test_the_ego_stamp_need_not_equal_the_camera_stamp(self):
        runtime, supervisor, lateral, longitudinal = self.build({1: detected(0.10)})
        _, _, steering, request = self.step(
            runtime, supervisor, lateral, longitudinal, 1,
            velocity=(1.0, 0.0, 0.0),
            ego_frame=91,
            ego_monotonic=control_time(1) - 0.02,
        )
        self.assertTrue(request.longitudinal_allowed)
        self.assertEqual(request.guidance_frame, 1)
        self.assertEqual(request.speed_frame, 91)
        self.assertEqual(request.guidance_stamp, steering.source_stamp)


class RealDegradedAuthorityTest(LongitudinalIntegrationTestCase):
    def test_real_mission_8_degraded_evidence_lowers_the_caps(self):
        runtime, supervisor, lateral, longitudinal = self.build(
            {frame: detected(0.10, 0.0, 0.50) for frame in range(1, 3)}
        )
        for frame in range(1, 3):
            _, verdict, steering, request = self.step(
                runtime, supervisor, lateral, longitudinal, frame,
                velocity=(0.0, 0.0, 0.0),
            )
        self.assertIs(verdict.state, SafetyState.DEGRADED)
        self.assertIs(verdict.reason, SafetyReason.DEGRADED_LOW_CONFIDENCE)
        self.assertIs(steering.mode, LateralControlMode.DEGRADED)
        self.assertIs(request.mode, LongitudinalControlMode.DEGRADED)
        self.assertIs(request.reason, LongitudinalControlReason.DEGRADED_TRACKING)
        self.assertEqual(request.throttle_authority, CONFIG.degraded_throttle_authority)
        self.assertEqual(request.speed_authority_mps, CONFIG.degraded_speed_cap_mps)
        self.assertLess(
            request.throttle_authority, CONFIG.nominal_throttle_authority
        )

    def test_real_mission_7_coasting_lowers_the_caps_further(self):
        script = {1: detected(0.10), 2: detected(0.10), 3: MISSING}
        runtime, supervisor, lateral, longitudinal = self.build(script)
        for frame in (1, 2):
            _, _, _, nominal = self.step(
                runtime, supervisor, lateral, longitudinal, frame,
                velocity=(0.0, 0.0, 0.0),
            )
        published, verdict, steering, request = self.step(
            runtime, supervisor, lateral, longitudinal, 3,
            velocity=(0.0, 0.0, 0.0),
        )
        self.assertIs(
            published.observation.temporal_estimate.state, TemporalLaneState.COASTING
        )
        self.assertIs(verdict.reason, SafetyReason.DEGRADED_COASTING)
        self.assertIs(steering.reason, LateralControlReason.DEGRADED_COASTING)
        self.assertIs(request.reason, LongitudinalControlReason.DEGRADED_COASTING)
        self.assertEqual(request.throttle_authority, CONFIG.coasting_throttle_authority)
        self.assertEqual(request.speed_authority_mps, CONFIG.coasting_speed_cap_mps)
        self.assertLess(request.throttle_request, nominal.throttle_request)
        self.assertIs(request.temporal_state, TemporalLaneState.COASTING)


class RealStopRequestTest(LongitudinalIntegrationTestCase):
    def test_mission_9_refusing_partial_evidence_requests_a_controlled_stop(self):
        runtime, supervisor, lateral, longitudinal = self.build({1: PARTIAL})
        _, verdict, steering, request = self.step(
            runtime, supervisor, lateral, longitudinal, 1, velocity=(1.0, 0.0, 0.0)
        )
        # Mission 8 really does permit degraded autonomy for partial evidence.
        self.assertIs(verdict.state, SafetyState.DEGRADED)
        self.assertTrue(verdict.autonomy_allowed)
        # Mission 9 refuses anyway, so Mission 10A must not propel.
        self.assertFalse(steering.lateral_allowed)
        self.assertIs(steering.reason, LateralControlReason.INSUFFICIENT_GEOMETRY)
        self.assertIs(request.mode, LongitudinalControlMode.STOPPING)
        self.assertIs(request.reason, LongitudinalControlReason.LATERAL_STOP)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, CONFIG.soft_stop_brake)
        self.assertTrue(request.stop_required)

    def test_real_mission_8_recovery_requests_a_controlled_stop(self):
        script = {
            1: detected(0.10),
            2: UNUSABLE,
            3: detected(0.10),
            4: detected(0.10),
        }
        runtime, supervisor, lateral, longitudinal = self.build(script)
        for frame in (1, 2):
            self.step(runtime, supervisor, lateral, longitudinal, frame,
                      velocity=(1.0, 0.0, 0.0))
        seen_recovering = False
        for frame in (3, 4):
            _, verdict, _, request = self.step(
                runtime, supervisor, lateral, longitudinal, frame,
                velocity=(1.0, 0.0, 0.0),
            )
            if verdict.state is SafetyState.RECOVERING:
                seen_recovering = True
                self.assertIs(verdict.reason, SafetyReason.RECOVERY_PENDING)
                self.assertIs(request.reason, LongitudinalControlReason.SAFETY_STOP)
                self.assertEqual(request.throttle_request, 0.0)
                self.assertEqual(request.brake_request, CONFIG.soft_stop_brake)
        self.assertTrue(seen_recovering, "Mission 8 never entered recovery")

    def test_real_unlatched_fail_safe_requests_a_controlled_stop(self):
        script = {1: detected(0.10), 2: UNUSABLE}
        runtime, supervisor, lateral, longitudinal = self.build(script)
        self.step(runtime, supervisor, lateral, longitudinal, 1,
                  velocity=(1.0, 0.0, 0.0))
        _, verdict, _, request = self.step(
            runtime, supervisor, lateral, longitudinal, 2, velocity=(1.0, 0.0, 0.0)
        )
        self.assertIs(verdict.state, SafetyState.FAIL_SAFE)
        self.assertIs(verdict.reason, SafetyReason.TEMPORAL_INPUT_UNUSABLE)
        self.assertFalse(verdict.latched)
        self.assertIs(request.reason, LongitudinalControlReason.SAFETY_STOP)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, CONFIG.soft_stop_brake)

    def test_a_real_latched_fail_safe_requests_the_latched_stop(self):
        script = {frame: detected(0.10) for frame in range(1, 4)}
        runtime, supervisor, lateral, longitudinal = self.build(script)
        for frame in (1, 2, 3):
            self.step(runtime, supervisor, lateral, longitudinal, frame,
                      velocity=(1.0, 0.0, 0.0))
        published = runtime.latest_result
        health = ComponentHealth.from_components(runtime)

        # A genuinely regressed supervisor clock is a hard fault, so Mission 8
        # latches for the rest of the epoch.
        latching = supervisor.update(published, now=control_time(3) - 1.0, health=health)
        self.assertIs(latching.state, SafetyState.FAIL_SAFE)
        self.assertIs(latching.reason, SafetyReason.CLOCK_REGRESSED)
        self.assertTrue(latching.latched)

        # Mission 10A then sees the latched decision for the same accepted frame
        # on a forward clock, which is a trusted, identity-bound stop.
        now = control_time(4)
        verdict = supervisor.update(published, now=now, health=health)
        self.assertTrue(verdict.latched)
        steering = lateral.update(published, verdict, now=now)
        self.assertFalse(steering.lateral_allowed)
        speed = EgoSpeedObservation.from_telemetry_snapshot(
            snapshot(3, velocity=(1.0, 0.0, 0.0))
        )
        request = longitudinal.update(
            verdict, steering, speed, scenario_target(), now=now
        )
        self.assertIs(request.mode, LongitudinalControlMode.STOPPING)
        self.assertIs(request.reason, LongitudinalControlReason.FAIL_SAFE_STOP)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, CONFIG.latched_stop_brake)
        self.assertTrue(request.stop_required)

    def test_a_real_expired_target_requests_a_controlled_stop(self):
        runtime, supervisor, lateral, longitudinal = self.build({1: detected(0.10)})
        _, verdict, steering, request = self.step(
            runtime, supervisor, lateral, longitudinal, 1,
            velocity=(1.0, 0.0, 0.0),
            target=scenario_target(
                2.0, issued=BASE_MONOTONIC, valid_until=BASE_MONOTONIC + 0.05
            ),
        )
        self.assertTrue(verdict.autonomy_allowed)
        self.assertTrue(steering.lateral_allowed)
        self.assertIs(request.reason, LongitudinalControlReason.TARGET_EXPIRED)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, CONFIG.soft_stop_brake)


class RealMotionPolicyTest(LongitudinalIntegrationTestCase):
    def test_real_reverse_telemetry_never_requests_throttle(self):
        runtime, supervisor, lateral, longitudinal = self.build({1: detected(0.10)})
        _, verdict, steering, request = self.step(
            runtime, supervisor, lateral, longitudinal, 1, velocity=(-1.0, 0.0, 0.0)
        )
        self.assertTrue(verdict.autonomy_allowed)
        self.assertTrue(steering.lateral_allowed)
        self.assertIs(request.reason, LongitudinalControlReason.REVERSE_MOTION)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, CONFIG.reverse_stop_brake)
        self.assertEqual(request.observed_forward_speed_mps, -1.0)

    def test_a_rotated_reverse_is_still_reverse(self):
        # Yawed 90 degrees, so world -y is this vehicle's own backward axis even
        # though world x velocity is exactly zero.
        runtime, supervisor, lateral, longitudinal = self.build({1: detected(0.10)})
        _, _, _, request = self.step(
            runtime, supervisor, lateral, longitudinal, 1,
            velocity=(0.0, -1.0, 0.0),
            rotation=(0.0, 0.0, 90.0),
        )
        self.assertIs(request.reason, LongitudinalControlReason.REVERSE_MOTION)
        self.assertAlmostEqual(request.observed_forward_speed_mps, -1.0, places=12)

    def test_real_lateral_only_motion_requests_a_controlled_stop(self):
        runtime, supervisor, lateral, longitudinal = self.build({1: detected(0.10)})
        _, _, steering, request = self.step(
            runtime, supervisor, lateral, longitudinal, 1, velocity=(0.0, 1.0, 0.0)
        )
        self.assertTrue(steering.lateral_allowed)
        self.assertIs(
            request.reason, LongitudinalControlReason.EXCESSIVE_LATERAL_MOTION
        )
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, CONFIG.soft_stop_brake)
        self.assertTrue(request.stop_required)

    def test_real_vertical_motion_refuses_neutrally(self):
        runtime, supervisor, lateral, longitudinal = self.build({1: detected(0.10)})
        _, _, steering, request = self.step(
            runtime, supervisor, lateral, longitudinal, 1, velocity=(0.0, 0.0, 1.0)
        )
        self.assertTrue(steering.lateral_allowed)
        self.assertIs(
            request.reason, LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION
        )
        self.assertIs(request.mode, LongitudinalControlMode.DISABLED)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, 0.0)
        self.assertFalse(request.stop_required)

    def test_a_real_zero_target_requests_a_stop_while_still_moving(self):
        runtime, supervisor, lateral, longitudinal = self.build({1: detected(0.10)})
        _, _, _, request = self.step(
            runtime, supervisor, lateral, longitudinal, 1,
            velocity=(1.0, 0.0, 0.0),
            target=scenario_target(0.0),
        )
        self.assertIs(request.reason, LongitudinalControlReason.TARGET_STOP)
        self.assertIs(request.mode, LongitudinalControlMode.STOPPING)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertGreater(request.brake_request, 0.0)
        self.assertEqual(request.effective_target_speed_mps, 0.0)

    def test_a_real_dead_ego_refuses_neutrally(self):
        runtime, supervisor, lateral, longitudinal = self.build({1: detected(0.10)})
        _, _, _, request = self.step(
            runtime, supervisor, lateral, longitudinal, 1,
            velocity=(1.0, 0.0, 0.0),
            vehicle_alive=False,
        )
        self.assertIs(request.reason, LongitudinalControlReason.VEHICLE_NOT_ALIVE)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, 0.0)


class RealIdentityTest(LongitudinalIntegrationTestCase):
    def test_a_decision_for_an_older_frame_cannot_authorize_a_newer_one(self):
        script = {frame: detected(0.10) for frame in range(1, 3)}
        runtime, supervisor, lateral, longitudinal = self.build(script)
        _, older_verdict, _, _ = self.step(
            runtime, supervisor, lateral, longitudinal, 1, velocity=(1.0, 0.0, 0.0)
        )
        self.assertTrue(older_verdict.autonomy_allowed)

        value = snapshot(2, velocity=(1.0, 0.0, 0.0))
        published = self.publish(runtime, value, 2)
        now = control_time(2)
        newer_verdict = supervisor.update(
            published, now=now, health=ComponentHealth.from_components(runtime)
        )
        steering = lateral.update(published, newer_verdict, now=now)
        speed = EgoSpeedObservation.from_telemetry_snapshot(value)
        request = longitudinal.update(
            older_verdict, steering, speed, scenario_target(), now=now
        )
        self.assertIs(request.reason, LongitudinalControlReason.IDENTITY_MISMATCH)
        self.assertIs(request.mode, LongitudinalControlMode.DISABLED)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, 0.0)

    def test_a_speed_sample_from_another_snapshot_revision_is_refused(self):
        script = {frame: detected(0.10) for frame in range(1, 3)}
        runtime, supervisor, lateral, longitudinal = self.build(script)
        value = snapshot(1, velocity=(1.0, 0.0, 0.0))
        published = self.publish(runtime, value, 1)
        now = control_time(1)
        verdict = supervisor.update(
            published, now=now, health=ComponentHealth.from_components(runtime)
        )
        steering = lateral.update(published, verdict, now=now)
        # A real observation, but detached from a different snapshot revision.
        other = EgoSpeedObservation.from_telemetry_snapshot(
            snapshot(2, velocity=(1.0, 0.0, 0.0), monotonic=now)
        )
        request = longitudinal.update(
            verdict, steering, other, scenario_target(), now=now
        )
        self.assertIs(request.reason, LongitudinalControlReason.IDENTITY_MISMATCH)
        self.assertEqual(request.throttle_request, 0.0)

    def test_a_stale_ego_sample_requests_a_controlled_stop(self):
        runtime, supervisor, lateral, longitudinal = self.build({1: detected(0.10)})
        _, _, _, request = self.step(
            runtime, supervisor, lateral, longitudinal, 1,
            velocity=(1.0, 0.0, 0.0),
            ego_monotonic=control_time(1) - 0.90,
        )
        self.assertIs(request.reason, LongitudinalControlReason.SPEED_STALE)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, CONFIG.soft_stop_brake)


class RealEpochTest(LongitudinalIntegrationTestCase):
    def test_a_new_epoch_removes_every_previous_identity_and_order_baseline(self):
        script = {frame: detected(0.10) for frame in range(1, 3)}
        runtime, supervisor, lateral, longitudinal = self.build(script)
        for frame in (1, 2):
            _, _, _, request = self.step(
                runtime, supervisor, lateral, longitudinal, frame,
                velocity=(1.0, 0.0, 0.0),
            )
        self.assertTrue(request.longitudinal_allowed)
        self.assertIsNotNone(longitudinal.metrics.last_accepted_guidance_stamp)
        lifetime = longitudinal.metrics.evaluations

        runtime.stop()
        supervisor.reset()
        lateral.reset()
        cleared = longitudinal.reset()
        self.assertIs(cleared.reason, LongitudinalControlReason.NOT_EVALUATED)
        self.assertEqual(cleared.throttle_request, 0.0)
        self.assertEqual(cleared.brake_request, 0.0)
        metrics = longitudinal.metrics
        self.assertIsNone(metrics.last_accepted_guidance_stamp)
        self.assertIsNone(metrics.last_accepted_speed_stamp)
        self.assertIsNone(metrics.last_accepted_target_origin)
        self.assertEqual(metrics.evaluations, lifetime)

        restarted = PerceptionRuntime(
            clock=ConstantClock(),
            perception_callable=TemporalLanePipeline(
                perception_callable=ScriptedPerception({1: detected(0.10)})
            ),
        )
        self.addCleanup(self.stop_runtime, restarted)
        restarted.start()

        # Frame 1 is acceptable again, under a brand new target origin and epoch
        # that the previous epoch would have refused outright.
        epoch_two_now = control_time(2) + 1.0
        value = snapshot(1, velocity=(1.0, 0.0, 0.0), monotonic=epoch_two_now)
        published = self.publish(restarted, value, 1)
        verdict = supervisor.update(
            published,
            now=epoch_two_now,
            health=ComponentHealth.from_components(restarted),
        )
        steering = lateral.update(published, verdict, now=epoch_two_now)
        speed = EgoSpeedObservation.from_telemetry_snapshot(value)
        resumed = longitudinal.update(
            verdict,
            steering,
            speed,
            scenario_target(
                2.0,
                origin=TargetSpeedOrigin.OPERATOR,
                epoch=7,
                revision=3,
                issued=epoch_two_now,
                valid_until=epoch_two_now + 10.0,
            ),
            now=epoch_two_now,
        )
        self.assertTrue(resumed.longitudinal_allowed)
        self.assertGreater(resumed.throttle_request, 0.0)
        self.assertEqual(resumed.guidance_frame, 1)
        self.assertIs(resumed.target_origin, TargetSpeedOrigin.OPERATOR)
        self.assertEqual(resumed.target_epoch, 7)


class RealVerticalPrecedenceTest(LongitudinalIntegrationTestCase):
    """Vertical motion must dominate real brake-bearing pipeline states."""

    LIFTED = (1.0, 0.0, 1.0)

    def assertVerticalNeutral(self, request):
        self.assertIs(
            request.reason, LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION
        )
        self.assertIs(request.mode, LongitudinalControlMode.DISABLED)
        self.assertFalse(request.stop_required)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, 0.0)

    def test_real_mission_8_recovery_with_vertical_motion_refuses_neutrally(self):
        script = {
            1: detected(0.10),
            2: UNUSABLE,
            3: detected(0.10),
            4: detected(0.10),
        }
        runtime, supervisor, lateral, longitudinal = self.build(script)
        for frame in (1, 2):
            self.step(runtime, supervisor, lateral, longitudinal, frame,
                      velocity=self.LIFTED)
        seen_recovering = False
        for frame in (3, 4):
            _, verdict, _, request = self.step(
                runtime, supervisor, lateral, longitudinal, frame,
                velocity=self.LIFTED,
            )
            if verdict.state is SafetyState.RECOVERING:
                seen_recovering = True
                # Without the lift this is a SAFETY_STOP asking for brake 0.20.
                self.assertVerticalNeutral(request)
        self.assertTrue(seen_recovering, "Mission 8 never entered recovery")

    def test_a_real_latched_fail_safe_with_vertical_motion_refuses_neutrally(self):
        script = {frame: detected(0.10) for frame in range(1, 4)}
        runtime, supervisor, lateral, longitudinal = self.build(script)
        for frame in (1, 2, 3):
            self.step(runtime, supervisor, lateral, longitudinal, frame,
                      velocity=self.LIFTED)
        published = runtime.latest_result
        health = ComponentHealth.from_components(runtime)
        latching = supervisor.update(published, now=control_time(3) - 1.0, health=health)
        self.assertTrue(latching.latched)

        now = control_time(4)
        verdict = supervisor.update(published, now=now, health=health)
        self.assertTrue(verdict.latched)
        steering = lateral.update(published, verdict, now=now)
        speed = EgoSpeedObservation.from_telemetry_snapshot(
            snapshot(3, velocity=self.LIFTED)
        )
        request = longitudinal.update(
            verdict, steering, speed, scenario_target(), now=now
        )
        # The strongest brake request in the policy still loses to the lift.
        self.assertVerticalNeutral(request)

    def test_real_partial_evidence_with_vertical_motion_refuses_neutrally(self):
        runtime, supervisor, lateral, longitudinal = self.build({1: PARTIAL})
        _, verdict, steering, request = self.step(
            runtime, supervisor, lateral, longitudinal, 1, velocity=self.LIFTED
        )
        self.assertTrue(verdict.autonomy_allowed)
        self.assertFalse(steering.lateral_allowed)
        self.assertVerticalNeutral(request)

    def test_a_real_expired_target_with_vertical_motion_refuses_neutrally(self):
        runtime, supervisor, lateral, longitudinal = self.build({1: detected(0.10)})
        _, _, _, request = self.step(
            runtime, supervisor, lateral, longitudinal, 1,
            velocity=self.LIFTED,
            target=scenario_target(
                2.0, issued=BASE_MONOTONIC, valid_until=BASE_MONOTONIC + 0.05
            ),
        )
        self.assertVerticalNeutral(request)

    def test_a_real_stale_ego_sample_with_vertical_motion_refuses_neutrally(self):
        runtime, supervisor, lateral, longitudinal = self.build({1: detected(0.10)})
        _, _, _, request = self.step(
            runtime, supervisor, lateral, longitudinal, 1,
            velocity=self.LIFTED,
            ego_monotonic=control_time(1) - 0.90,
        )
        self.assertVerticalNeutral(request)


class RealUpstreamValidationTest(LongitudinalIntegrationTestCase):
    """Real pipeline records with one forged field cannot authorize anything."""

    @staticmethod
    def bypassed(base, **overrides):
        clone = object.__new__(type(base))
        for field in fields(base):
            object.__setattr__(
                clone,
                field.name,
                overrides[field.name]
                if field.name in overrides
                else getattr(base, field.name),
            )
        return clone

    def prepare(self, frame=1):
        runtime, supervisor, lateral, longitudinal = self.build(
            {frame: detected(0.10)}
        )
        value = snapshot(frame, velocity=(1.0, 0.0, 0.0))
        published = self.publish(runtime, value, frame)
        now = control_time(frame)
        verdict = supervisor.update(
            published, now=now, health=ComponentHealth.from_components(runtime)
        )
        steering = lateral.update(published, verdict, now=now)
        speed = EgoSpeedObservation.from_telemetry_snapshot(value)
        return longitudinal, verdict, steering, speed, now

    def test_a_real_decision_with_a_forged_reason_cannot_authorize_throttle(self):
        longitudinal, verdict, steering, speed, now = self.prepare()
        self.assertIs(verdict.reason, SafetyReason.NOMINAL_TRACKING)
        forged = self.bypassed(verdict, reason=SafetyReason.RUNTIME_FAULT)
        request = longitudinal.update(
            forged, steering, speed, scenario_target(), now=now
        )
        self.assertIs(request.reason, LongitudinalControlReason.EVIDENCE_MALFORMED)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, 0.0)

    def test_a_real_lateral_request_with_forged_raw_steering_is_refused(self):
        longitudinal, verdict, steering, speed, now = self.prepare()
        self.assertTrue(steering.lateral_allowed)
        forged = self.bypassed(steering, raw_steering=float("nan"))
        request = longitudinal.update(
            verdict, forged, speed, scenario_target(), now=now
        )
        self.assertIs(request.reason, LongitudinalControlReason.EVIDENCE_MALFORMED)
        self.assertEqual(request.throttle_request, 0.0)

    def test_a_real_decision_with_forged_confidence_is_refused(self):
        longitudinal, verdict, steering, speed, now = self.prepare()
        forged = self.bypassed(verdict, temporal_confidence=None)
        request = longitudinal.update(
            forged, steering, speed, scenario_target(), now=now
        )
        self.assertIs(request.reason, LongitudinalControlReason.EVIDENCE_MALFORMED)
        self.assertEqual(request.throttle_request, 0.0)

    def test_real_canonicalized_records_equal_their_originals(self):
        longitudinal, verdict, steering, speed, now = self.prepare()
        request = longitudinal.update(
            verdict, steering, speed, scenario_target(), now=now
        )
        self.assertTrue(request.longitudinal_allowed)
        self.assertEqual(request.guidance_stamp, verdict.source_stamp)
        self.assertIsNot(request.guidance_stamp, verdict.source_stamp)
        self.assertIs(request.safety_state, verdict.state)
        self.assertIs(request.lateral_mode, steering.mode)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    unittest.main()
