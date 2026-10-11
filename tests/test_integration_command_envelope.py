"""Offline integration tests for the Mission 10B command safety envelope.

These tests drive the accepted Mission 6A runtime, Mission 7 temporal pipeline,
Mission 8 safety supervisor, Mission 9 lateral controller, and Mission 10A
longitudinal controller for real, injecting only a Mission 5-shaped perception
callable, synthetic telemetry, and explicit clocks.  The ego speed observation is
detached from the very same immutable :class:`TelemetrySnapshot` the perception
stack consumed, so the snapshot revision that binds the whole cut is a real one
rather than a hand-written constant.

Every test asserts the real upstream dispositions -- the Mission 7 temporal
state, the Mission 8 state and reason, the Mission 9 mode and reason, and the
Mission 10A mode and reason -- *before* it asserts anything about Mission 10B, so
a Mission 10B expectation can never quietly pass on top of a pipeline that did
something else.

CARLA is neither imported nor contacted, and no value produced here reaches a
vehicle: Mission 10B publishes one bounded command *envelope* request and applies
nothing.
"""

from __future__ import annotations

import numbers
import threading
import unittest
from dataclasses import fields

import control.command as command_module
from control.command import (
    DEFAULT_COMMAND_ENVELOPE_CONFIG,
    CommandEnvelope,
    CommandEnvelopeConfig,
    CommandEnvelopeMode,
    CommandEnvelopeReason,
    CommandEnvelopeRequest,
    _canonicalize_lateral_request,
    _canonicalize_longitudinal_request,
)
from control.lateral import (
    LateralControlMode,
    LateralControlReason,
    LateralControlRequest,
    LateralController,
)
from control.longitudinal import (
    DEFAULT_LONGITUDINAL_CONTROLLER_CONFIG,
    EgoSpeedObservation,
    LongitudinalControlMode,
    LongitudinalControlReason,
    LongitudinalControlRequest,
    LongitudinalController,
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
from tests.test_command_envelope import (
    REAL_SAME_CUT_LONGITUDINAL,
    real_lateral_shape,
    retained_objects,
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

LONGITUDINAL_CONFIG = DEFAULT_LONGITUDINAL_CONTROLLER_CONFIG
ENVELOPE_CONFIG = DEFAULT_COMMAND_ENVELOPE_CONFIG


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


class Cut:
    """The five real records one control step produces, kept together."""

    __slots__ = ("published", "verdict", "steering", "longitudinal", "envelope",
                 "now", "snapshot")

    def __init__(self, published, verdict, steering, longitudinal, envelope,
                 now, value):
        self.published = published
        self.verdict = verdict
        self.steering = steering
        self.longitudinal = longitudinal
        self.envelope = envelope
        self.now = now
        self.snapshot = value


class EnvelopeIntegrationTestCase(unittest.TestCase):
    """Shared real-pipeline lifecycle helpers for the Mission 10B suite."""

    def build(self, script: dict[int, tuple], *, envelope_config=None):
        pipeline = TemporalLanePipeline(perception_callable=ScriptedPerception(script))
        runtime = PerceptionRuntime(clock=ConstantClock(), perception_callable=pipeline)
        supervisor = SafetySupervisor()
        lateral = LateralController()
        longitudinal = LongitudinalController()
        envelope = CommandEnvelope(
            ENVELOPE_CONFIG if envelope_config is None else envelope_config
        )
        self.addCleanup(self.stop_runtime, runtime)
        runtime.start()
        return runtime, supervisor, lateral, longitudinal, envelope

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
        envelope,
        frame: int,
        *,
        target=None,
        now=None,
        processed=None,
        **snapshot_kwargs,
    ) -> Cut:
        """Publish, supervise, steer, pace, and bound one frame for real."""
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
        bounded = envelope.update(steering, request, now=control_now)
        return Cut(published, verdict, steering, request, bounded, control_now, value)

    # ------------------------------------------------------------- assertions

    def assertUpstreamNominal(self, cut: Cut):
        self.assertIs(
            cut.published.observation.temporal_estimate.state,
            TemporalLaneState.TRACKING,
        )
        self.assertIs(cut.verdict.state, SafetyState.NOMINAL)
        self.assertIs(cut.verdict.reason, SafetyReason.NOMINAL_TRACKING)
        self.assertIs(cut.steering.mode, LateralControlMode.NOMINAL)
        self.assertIs(cut.steering.reason, LateralControlReason.NOMINAL_TRACKING)
        self.assertIs(cut.longitudinal.mode, LongitudinalControlMode.NOMINAL)
        self.assertIs(
            cut.longitudinal.reason, LongitudinalControlReason.NOMINAL_TRACKING
        )

    def assertEnvelopeSafe(self, request):
        self.assertFalse(
            request.throttle_request > 0.0 and request.brake_request > 0.0
        )
        self.assertLessEqual(abs(request.steering_request), request.steering_authority)
        self.assertLessEqual(request.throttle_request, request.throttle_authority)
        self.assertLessEqual(request.brake_request, request.brake_authority)
        self.assertEqual(
            request.shaping_applied,
            request.steering_clamped
            or request.throttle_clamped
            or request.brake_clamped
            or request.throttle_rate_limited
            or request.brake_release_rate_limited
            or request.transition_interlocked,
        )

    def assertEnvelopeNeutral(self, request, reason):
        self.assertIs(request.reason, reason)
        self.assertIs(request.mode, CommandEnvelopeMode.DISABLED)
        self.assertFalse(request.command_allowed)
        self.assertFalse(request.stop_required)
        self.assertFalse(request.evidence_bound)
        self.assertEqual(request.steering_request, 0.0)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, 0.0)
        self.assertEqual(request.steering_authority, 0.0)
        self.assertEqual(request.throttle_authority, 0.0)
        self.assertEqual(request.brake_authority, 0.0)
        self.assertFalse(request.shaping_applied)

    def assertBoundToCut(self, request, cut: Cut):
        self.assertTrue(request.evidence_bound)
        self.assertEqual(request.evaluation_monotonic, cut.now)
        self.assertEqual(request.guidance_stamp, cut.longitudinal.guidance_stamp)
        self.assertEqual(request.speed_stamp, cut.longitudinal.speed_stamp)
        self.assertEqual(request.snapshot_revision, cut.published.snapshot_revision)
        self.assertIs(request.safety_state, cut.verdict.state)
        self.assertIs(request.temporal_state, cut.longitudinal.temporal_state)
        self.assertIs(request.lateral_mode, cut.steering.mode)
        self.assertIs(request.lateral_reason, cut.steering.reason)
        self.assertIs(request.longitudinal_mode, cut.longitudinal.mode)
        self.assertIs(request.longitudinal_reason, cut.longitudinal.reason)


# --------------------------------------------------------------------------
# A. Nominal, degraded, and ordinary braking
# --------------------------------------------------------------------------


class RealNominalCommandTest(EnvelopeIntegrationTestCase):
    def test_a_real_nominal_pipeline_ramps_throttle_from_exactly_zero(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {frame: detected(0.10) for frame in range(1, 5)}
        )
        published = []
        for frame in range(1, 5):
            cut = self.step(
                runtime, supervisor, lateral, longitudinal, envelope, frame,
                velocity=(1.0, 0.0, 0.0),
            )
            self.assertUpstreamNominal(cut)
            self.assertGreater(cut.longitudinal.throttle_request, 0.0)
            self.assertIs(cut.envelope.mode, CommandEnvelopeMode.NOMINAL)
            self.assertTrue(cut.envelope.command_allowed)
            self.assertFalse(cut.envelope.stop_required)
            self.assertEnvelopeSafe(cut.envelope)
            self.assertBoundToCut(cut.envelope, cut)
            published.append(cut.envelope)
        self.assertEqual(published[0].throttle_request, 0.0)
        self.assertTrue(published[0].throttle_rate_limited)
        for earlier, later in zip(published, published[1:]):
            self.assertGreaterEqual(later.throttle_request, earlier.throttle_request)
            self.assertLessEqual(
                later.throttle_request - earlier.throttle_request,
                ENVELOPE_CONFIG.throttle_rise_rate_per_second * STEP_SECONDS + 1e-12,
            )

    def test_a_real_pipeline_at_target_publishes_neutral_pedals(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {frame: detected(0.10) for frame in range(1, 3)}
        )
        for frame in range(1, 3):
            cut = self.step(
                runtime, supervisor, lateral, longitudinal, envelope, frame,
                velocity=(2.0, 0.0, 0.0),
            )
            self.assertUpstreamNominal(cut)
            self.assertEqual(cut.longitudinal.throttle_request, 0.0)
            self.assertEqual(cut.longitudinal.brake_request, 0.0)
            self.assertEqual(cut.envelope.throttle_request, 0.0)
            self.assertEqual(cut.envelope.brake_request, 0.0)
            self.assertTrue(cut.envelope.command_allowed)
            self.assertIs(cut.envelope.reason, CommandEnvelopeReason.NOMINAL_COMMAND)
            self.assertFalse(cut.envelope.shaping_applied)

    def test_real_overspeed_braking_is_published_immediately_and_is_not_a_stop(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {frame: detected(0.10) for frame in range(1, 3)}
        )
        for frame in range(1, 3):
            cut = self.step(
                runtime, supervisor, lateral, longitudinal, envelope, frame,
                velocity=(2.8, 0.0, 0.0),
            )
            self.assertUpstreamNominal(cut)
            self.assertGreater(cut.longitudinal.brake_request, 0.0)
            self.assertFalse(cut.longitudinal.stop_required)
            self.assertEqual(
                cut.envelope.brake_request, cut.longitudinal.brake_request
            )
            self.assertEqual(cut.envelope.throttle_request, 0.0)
            self.assertFalse(cut.envelope.stop_required)
            self.assertIs(cut.envelope.mode, CommandEnvelopeMode.NOMINAL)
            self.assertFalse(cut.envelope.brake_release_rate_limited)

    def test_the_envelope_preserves_the_real_speed_law_evidence(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {1: detected(0.10)}
        )
        cut = self.step(
            runtime, supervisor, lateral, longitudinal, envelope, 1,
            velocity=(1.0, 0.0, 0.0),
        )
        self.assertUpstreamNominal(cut)
        self.assertEqual(
            cut.envelope.effective_target_speed_mps,
            cut.longitudinal.effective_target_speed_mps,
        )
        self.assertEqual(
            cut.envelope.observed_forward_speed_mps,
            cut.longitudinal.observed_forward_speed_mps,
        )
        self.assertEqual(cut.envelope.speed_error_mps, cut.longitudinal.speed_error_mps)
        self.assertEqual(
            cut.envelope.speed_authority_mps, cut.longitudinal.speed_authority_mps
        )


class RealDegradedCommandTest(EnvelopeIntegrationTestCase):
    def test_real_degraded_evidence_lowers_the_published_authorities(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {frame: detected(0.10, 0.0, 0.50) for frame in range(1, 3)}
        )
        for frame in range(1, 3):
            cut = self.step(
                runtime, supervisor, lateral, longitudinal, envelope, frame,
                velocity=(0.0, 0.0, 0.0),
            )
        self.assertIs(cut.verdict.state, SafetyState.DEGRADED)
        self.assertIs(cut.verdict.reason, SafetyReason.DEGRADED_LOW_CONFIDENCE)
        self.assertIs(cut.steering.mode, LateralControlMode.DEGRADED)
        self.assertIs(cut.longitudinal.mode, LongitudinalControlMode.DEGRADED)
        self.assertIs(
            cut.longitudinal.reason, LongitudinalControlReason.DEGRADED_TRACKING
        )
        self.assertIs(cut.envelope.mode, CommandEnvelopeMode.DEGRADED)
        self.assertEqual(
            cut.envelope.throttle_authority,
            LONGITUDINAL_CONFIG.degraded_throttle_authority,
        )
        self.assertEqual(
            cut.envelope.steering_authority, cut.steering.steering_authority
        )
        self.assertEnvelopeSafe(cut.envelope)

    def test_real_coasting_lowers_the_published_authorities_further(self):
        script = {1: detected(0.10), 2: detected(0.10), 3: MISSING}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        for frame in (1, 2):
            nominal = self.step(
                runtime, supervisor, lateral, longitudinal, envelope, frame,
                velocity=(0.0, 0.0, 0.0),
            )
        cut = self.step(
            runtime, supervisor, lateral, longitudinal, envelope, 3,
            velocity=(0.0, 0.0, 0.0),
        )
        self.assertIs(
            cut.published.observation.temporal_estimate.state,
            TemporalLaneState.COASTING,
        )
        self.assertIs(cut.verdict.reason, SafetyReason.DEGRADED_COASTING)
        self.assertIs(cut.steering.reason, LateralControlReason.DEGRADED_COASTING)
        self.assertIs(
            cut.longitudinal.reason, LongitudinalControlReason.DEGRADED_COASTING
        )
        self.assertIs(cut.envelope.mode, CommandEnvelopeMode.DEGRADED)
        self.assertEqual(
            cut.envelope.throttle_authority,
            LONGITUDINAL_CONFIG.coasting_throttle_authority,
        )
        self.assertLess(
            cut.envelope.throttle_authority, nominal.envelope.throttle_authority
        )
        self.assertIs(cut.envelope.temporal_state, TemporalLaneState.COASTING)


# --------------------------------------------------------------------------
# B. Trusted stops and neutral refusals
# --------------------------------------------------------------------------


class RealTrustedStopTest(EnvelopeIntegrationTestCase):
    def test_real_partial_evidence_stops_with_exactly_zero_steering(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {1: PARTIAL}
        )
        cut = self.step(
            runtime, supervisor, lateral, longitudinal, envelope, 1,
            velocity=(1.0, 0.0, 0.0),
        )
        self.assertIs(cut.verdict.state, SafetyState.DEGRADED)
        self.assertTrue(cut.verdict.autonomy_allowed)
        self.assertFalse(cut.steering.lateral_allowed)
        self.assertIs(cut.steering.reason, LateralControlReason.INSUFFICIENT_GEOMETRY)
        self.assertIs(cut.longitudinal.reason, LongitudinalControlReason.LATERAL_STOP)
        self.assertIs(cut.envelope.mode, CommandEnvelopeMode.STOPPING)
        self.assertIs(
            cut.envelope.reason, CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP
        )
        self.assertTrue(cut.envelope.stop_required)
        self.assertTrue(cut.envelope.command_allowed)
        self.assertEqual(cut.envelope.steering_request, 0.0)
        self.assertEqual(cut.envelope.steering_authority, 0.0)
        self.assertFalse(cut.envelope.steering_clamped)
        self.assertEqual(
            cut.envelope.brake_request, LONGITUDINAL_CONFIG.soft_stop_brake
        )
        self.assertEqual(cut.envelope.throttle_request, 0.0)
        self.assertEqual(cut.envelope.throttle_authority, 0.0)
        self.assertBoundToCut(cut.envelope, cut)

    def test_real_mission_8_recovery_becomes_a_trusted_stop(self):
        script = {1: detected(0.10), 2: UNUSABLE, 3: detected(0.10), 4: detected(0.10)}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        for frame in (1, 2):
            self.step(runtime, supervisor, lateral, longitudinal, envelope, frame,
                      velocity=(1.0, 0.0, 0.0))
        seen_recovering = False
        for frame in (3, 4):
            cut = self.step(
                runtime, supervisor, lateral, longitudinal, envelope, frame,
                velocity=(1.0, 0.0, 0.0),
            )
            if cut.verdict.state is SafetyState.RECOVERING:
                seen_recovering = True
                self.assertIs(cut.verdict.reason, SafetyReason.RECOVERY_PENDING)
                self.assertIs(
                    cut.steering.reason, LateralControlReason.SAFETY_NOT_PERMITTED
                )
                self.assertIs(
                    cut.longitudinal.reason, LongitudinalControlReason.SAFETY_STOP
                )
                self.assertIs(cut.envelope.mode, CommandEnvelopeMode.STOPPING)
                self.assertEqual(
                    cut.envelope.brake_request, LONGITUDINAL_CONFIG.soft_stop_brake
                )
                self.assertEqual(cut.envelope.steering_request, 0.0)
        self.assertTrue(seen_recovering, "Mission 8 never entered recovery")

    def test_a_real_unlatched_fail_safe_becomes_a_trusted_stop(self):
        script = {1: detected(0.10), 2: UNUSABLE}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                  velocity=(1.0, 0.0, 0.0))
        cut = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2,
                        velocity=(1.0, 0.0, 0.0))
        self.assertIs(cut.verdict.state, SafetyState.FAIL_SAFE)
        self.assertIs(cut.verdict.reason, SafetyReason.TEMPORAL_INPUT_UNUSABLE)
        self.assertFalse(cut.verdict.latched)
        self.assertIs(cut.longitudinal.reason, LongitudinalControlReason.SAFETY_STOP)
        self.assertIs(cut.envelope.mode, CommandEnvelopeMode.STOPPING)
        self.assertEqual(
            cut.envelope.brake_request, LONGITUDINAL_CONFIG.soft_stop_brake
        )
        self.assertEqual(
            cut.envelope.brake_authority, cut.longitudinal.brake_authority
        )

    def test_a_real_latched_fail_safe_stop_is_republished_exactly(self):
        script = {frame: detected(0.10) for frame in range(1, 4)}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        for frame in (1, 2, 3):
            self.step(runtime, supervisor, lateral, longitudinal, envelope, frame,
                      velocity=(1.0, 0.0, 0.0))
        published = runtime.latest_result
        health = ComponentHealth.from_components(runtime)
        latching = supervisor.update(published, now=control_time(3) - 1.0, health=health)
        self.assertIs(latching.reason, SafetyReason.CLOCK_REGRESSED)
        self.assertTrue(latching.latched)

        now = control_time(4)
        verdict = supervisor.update(published, now=now, health=health)
        self.assertTrue(verdict.latched)
        steering = lateral.update(published, verdict, now=now)
        self.assertFalse(steering.lateral_allowed)
        speed = EgoSpeedObservation.from_telemetry_snapshot(
            snapshot(3, velocity=(1.0, 0.0, 0.0))
        )
        request = longitudinal.update(verdict, steering, speed, scenario_target(),
                                      now=now)
        self.assertIs(request.reason, LongitudinalControlReason.FAIL_SAFE_STOP)
        bounded = envelope.update(steering, request, now=now)
        self.assertIs(bounded.mode, CommandEnvelopeMode.STOPPING)
        self.assertEqual(
            bounded.brake_request, LONGITUDINAL_CONFIG.latched_stop_brake
        )
        self.assertEqual(bounded.brake_request, request.brake_request)
        self.assertEqual(bounded.brake_authority, request.brake_authority)
        self.assertEqual(bounded.steering_request, 0.0)

    def test_a_real_expired_target_becomes_a_trusted_stop_with_live_steering(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {1: detected(0.10)}
        )
        cut = self.step(
            runtime, supervisor, lateral, longitudinal, envelope, 1,
            velocity=(1.0, 0.0, 0.0),
            target=scenario_target(
                2.0, issued=BASE_MONOTONIC, valid_until=BASE_MONOTONIC + 0.05
            ),
        )
        self.assertTrue(cut.verdict.autonomy_allowed)
        self.assertTrue(cut.steering.lateral_allowed)
        self.assertIs(cut.longitudinal.reason, LongitudinalControlReason.TARGET_EXPIRED)
        self.assertIs(cut.envelope.mode, CommandEnvelopeMode.STOPPING)
        self.assertEqual(cut.envelope.steering_request, cut.steering.steering)
        self.assertGreater(cut.envelope.steering_authority, 0.0)
        self.assertEqual(
            cut.envelope.brake_request, LONGITUDINAL_CONFIG.soft_stop_brake
        )

    def test_a_real_stale_ego_sample_becomes_a_trusted_stop(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {1: detected(0.10)}
        )
        cut = self.step(
            runtime, supervisor, lateral, longitudinal, envelope, 1,
            velocity=(1.0, 0.0, 0.0),
            ego_monotonic=control_time(1) - 0.90,
        )
        self.assertIs(cut.longitudinal.reason, LongitudinalControlReason.SPEED_STALE)
        self.assertIs(cut.envelope.mode, CommandEnvelopeMode.STOPPING)
        self.assertEqual(
            cut.envelope.brake_request, LONGITUDINAL_CONFIG.soft_stop_brake
        )
        self.assertIsNone(cut.envelope.effective_target_speed_mps)
        self.assertIsNotNone(cut.envelope.observed_forward_speed_mps)

    def test_a_real_zero_target_stop_keeps_its_full_speed_law_evidence(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {1: detected(0.10)}
        )
        cut = self.step(
            runtime, supervisor, lateral, longitudinal, envelope, 1,
            velocity=(1.0, 0.0, 0.0),
            target=scenario_target(0.0),
        )
        self.assertIs(cut.longitudinal.reason, LongitudinalControlReason.TARGET_STOP)
        self.assertIs(cut.envelope.mode, CommandEnvelopeMode.STOPPING)
        self.assertEqual(cut.envelope.effective_target_speed_mps, 0.0)
        self.assertEqual(
            cut.envelope.speed_error_mps, cut.longitudinal.speed_error_mps
        )
        self.assertEqual(cut.envelope.brake_request, cut.longitudinal.brake_request)

    def test_real_reverse_motion_becomes_a_trusted_stop_and_never_propels(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {1: detected(0.10)}
        )
        cut = self.step(
            runtime, supervisor, lateral, longitudinal, envelope, 1,
            velocity=(-1.0, 0.0, 0.0),
        )
        self.assertTrue(cut.verdict.autonomy_allowed)
        self.assertTrue(cut.steering.lateral_allowed)
        self.assertIs(cut.longitudinal.reason, LongitudinalControlReason.REVERSE_MOTION)
        self.assertIs(cut.envelope.mode, CommandEnvelopeMode.STOPPING)
        self.assertEqual(cut.envelope.throttle_request, 0.0)
        self.assertEqual(
            cut.envelope.brake_request, LONGITUDINAL_CONFIG.reverse_stop_brake
        )
        self.assertEqual(cut.envelope.observed_forward_speed_mps, -1.0)

    def test_real_excessive_lateral_motion_becomes_a_trusted_stop(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {1: detected(0.10)}
        )
        cut = self.step(
            runtime, supervisor, lateral, longitudinal, envelope, 1,
            velocity=(0.0, 1.0, 0.0),
        )
        self.assertIs(
            cut.longitudinal.reason,
            LongitudinalControlReason.EXCESSIVE_LATERAL_MOTION,
        )
        self.assertIs(cut.envelope.mode, CommandEnvelopeMode.STOPPING)
        self.assertEqual(
            cut.envelope.brake_request, LONGITUDINAL_CONFIG.soft_stop_brake
        )


class RealNeutralRefusalTest(EnvelopeIntegrationTestCase):
    def test_real_vertical_motion_refuses_neutrally(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {1: detected(0.10)}
        )
        cut = self.step(
            runtime, supervisor, lateral, longitudinal, envelope, 1,
            velocity=(0.0, 0.0, 1.0),
        )
        self.assertTrue(cut.steering.lateral_allowed)
        self.assertIs(
            cut.longitudinal.reason,
            LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION,
        )
        self.assertIs(cut.longitudinal.mode, LongitudinalControlMode.DISABLED)
        self.assertEnvelopeNeutral(
            cut.envelope, CommandEnvelopeReason.UPSTREAM_REFUSAL
        )

    def test_a_real_dead_ego_refuses_neutrally(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {1: detected(0.10)}
        )
        cut = self.step(
            runtime, supervisor, lateral, longitudinal, envelope, 1,
            velocity=(1.0, 0.0, 0.0),
            vehicle_alive=False,
        )
        self.assertIs(
            cut.longitudinal.reason, LongitudinalControlReason.VEHICLE_NOT_ALIVE
        )
        self.assertIsNone(cut.longitudinal.guidance_stamp)
        self.assertEnvelopeNeutral(
            cut.envelope, CommandEnvelopeReason.UPSTREAM_REFUSAL
        )

    def test_a_real_mission_9_refusal_with_a_refused_mission_10a_is_neutral(self):
        script = {1: detected(0.10), 2: UNUSABLE}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                  velocity=(1.0, 0.0, 0.0))
        cut = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2,
                        velocity=(0.0, 0.0, 1.0))
        self.assertIs(cut.verdict.state, SafetyState.FAIL_SAFE)
        self.assertFalse(cut.steering.lateral_allowed)
        self.assertIs(
            cut.longitudinal.reason,
            LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION,
        )
        self.assertEnvelopeNeutral(
            cut.envelope, CommandEnvelopeReason.UPSTREAM_REFUSAL
        )

    def test_the_initial_evidence_free_pair_is_an_upstream_refusal(self):
        _runtime, _supervisor, lateral, longitudinal, envelope = self.build({})
        request = envelope.update(
            lateral.latest_request, longitudinal.latest_request, now=BASE_MONOTONIC
        )
        self.assertIs(lateral.latest_request.reason, LateralControlReason.NOT_EVALUATED)
        self.assertIs(
            longitudinal.latest_request.reason,
            LongitudinalControlReason.NOT_EVALUATED,
        )
        self.assertEnvelopeNeutral(request, CommandEnvelopeReason.UPSTREAM_REFUSAL)


# --------------------------------------------------------------------------
# C. Identity, ordering, and replay against real records
# --------------------------------------------------------------------------


class RealIdentityTest(EnvelopeIntegrationTestCase):
    def test_a_real_pair_from_two_different_cuts_is_an_identity_mismatch(self):
        script = {frame: detected(0.10) for frame in range(1, 3)}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        first = self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                          velocity=(1.0, 0.0, 0.0))
        second = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2,
                           velocity=(1.0, 0.0, 0.0))
        self.assertUpstreamNominal(first)
        self.assertUpstreamNominal(second)
        fresh = CommandEnvelope()
        request = fresh.update(first.steering, second.longitudinal, now=second.now)
        self.assertEnvelopeNeutral(
            request, CommandEnvelopeReason.CROSS_RECORD_IDENTITY_MISMATCH
        )

    def test_a_real_pair_presented_at_the_wrong_control_time_is_time_invalid(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {1: detected(0.10)}
        )
        cut = self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                        velocity=(1.0, 0.0, 0.0))
        self.assertUpstreamNominal(cut)
        fresh = CommandEnvelope()
        request = fresh.update(
            cut.steering, cut.longitudinal, now=cut.now + 1.0
        )
        self.assertEnvelopeNeutral(request, CommandEnvelopeReason.TIME_INVALID)

    def test_a_real_older_cut_replayed_at_a_newer_time_is_source_rejected(self):
        script = {frame: detected(0.10) for frame in range(1, 3)}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        first = self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                          velocity=(1.0, 0.0, 0.0))
        second = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2,
                           velocity=(1.0, 0.0, 0.0))
        self.assertUpstreamNominal(second)
        self.assertTrue(second.envelope.command_allowed)

        # A second, independent Mission 8/9/10A triple re-evaluates the *older*
        # published result on the newer control clock, which is exactly the
        # replay a stale producer would deliver.
        replay_supervisor = SafetySupervisor()
        replay_lateral = LateralController()
        replay_longitudinal = LongitudinalController()
        now = control_time(3)
        verdict = replay_supervisor.update(
            first.published,
            now=now,
            health=ComponentHealth.from_components(runtime),
        )
        steering = replay_lateral.update(first.published, verdict, now=now)
        speed = EgoSpeedObservation.from_telemetry_snapshot(first.snapshot)
        request = replay_longitudinal.update(
            verdict, steering, speed, scenario_target(), now=now
        )
        self.assertTrue(steering.lateral_allowed)
        self.assertEqual(request.guidance_frame, 1)
        rejected = envelope.update(steering, request, now=now)
        self.assertEnvelopeNeutral(rejected, CommandEnvelopeReason.SOURCE_REJECTED)

    def test_a_real_exact_duplicate_is_idempotent_and_a_changed_one_is_rejected(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {1: detected(0.10)}
        )
        cut = self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                        velocity=(1.0, 0.0, 0.0))
        self.assertUpstreamNominal(cut)
        again = envelope.update(cut.steering, cut.longitudinal, now=cut.now)
        self.assertIs(again, cut.envelope)

        # The same cut, but a genuinely different target contract at the very
        # same control instant, is changed canonical content and is refused.
        other_supervisor = SafetySupervisor()
        other_lateral = LateralController()
        other_longitudinal = LongitudinalController()
        verdict = other_supervisor.update(
            cut.published,
            now=cut.now,
            health=ComponentHealth.from_components(runtime),
        )
        steering = other_lateral.update(cut.published, verdict, now=cut.now)
        speed = EgoSpeedObservation.from_telemetry_snapshot(cut.snapshot)
        request = other_longitudinal.update(
            verdict,
            steering,
            speed,
            scenario_target(2.0, revision=9, issued=BASE_MONOTONIC + 0.01),
            now=cut.now,
        )
        self.assertTrue(request.longitudinal_allowed)
        rejected = envelope.update(steering, request, now=cut.now)
        self.assertEnvelopeNeutral(rejected, CommandEnvelopeReason.SOURCE_REJECTED)

    def test_a_replay_after_a_refusal_never_resurrects_the_real_command(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {1: detected(0.10)}
        )
        cut = self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                        velocity=(1.0, 0.0, 0.0))
        self.assertTrue(cut.envelope.command_allowed)
        refused = envelope.update(
            cut.steering, cut.longitudinal, now=cut.now + 1.0
        )
        self.assertEnvelopeNeutral(refused, CommandEnvelopeReason.TIME_INVALID)
        replay = envelope.update(cut.steering, cut.longitudinal, now=cut.now)
        self.assertEnvelopeNeutral(replay, CommandEnvelopeReason.SOURCE_REJECTED)
        self.assertIs(envelope.latest_request, replay)

    def test_a_real_mission_9_refusal_paired_with_a_permitted_10a_conflicts(self):
        script = {1: detected(0.10), 2: PARTIAL}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        permitted = self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                              velocity=(1.0, 0.0, 0.0))
        refused = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2,
                            velocity=(1.0, 0.0, 0.0))
        self.assertUpstreamNominal(permitted)
        self.assertFalse(refused.steering.lateral_allowed)
        fresh = CommandEnvelope()
        request = fresh.update(
            refused.steering, permitted.longitudinal, now=permitted.now
        )
        self.assertEnvelopeNeutral(request, CommandEnvelopeReason.AXIS_CONFLICT)


class RealEpochTest(EnvelopeIntegrationTestCase):
    def test_a_reset_removes_every_baseline_and_lets_a_real_cut_replay(self):
        script = {frame: detected(0.10) for frame in range(1, 3)}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        first = self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                          velocity=(1.0, 0.0, 0.0))
        second = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2,
                           velocity=(1.0, 0.0, 0.0))
        self.assertUpstreamNominal(second)
        self.assertTrue(second.envelope.command_allowed)
        lifetime = envelope.metrics.evaluations

        cleared = envelope.reset()
        self.assertEnvelopeNeutral(cleared, CommandEnvelopeReason.NOT_EVALUATED)
        self.assertEqual(cleared.controller_epoch, 1)
        self.assertEqual(envelope.metrics.evaluations, lifetime)

        # Frame 1 is acceptable again in the new local epoch.
        resumed = envelope.update(first.steering, first.longitudinal, now=first.now)
        self.assertTrue(resumed.command_allowed)
        self.assertEqual(resumed.guidance_frame, 1)
        self.assertEqual(resumed.controller_epoch, 1)
        # And the first propulsion of the new epoch is still exactly zero.
        self.assertEqual(resumed.throttle_request, 0.0)


# --------------------------------------------------------------------------
# D. Shaping and the interlock through the real pipeline
# --------------------------------------------------------------------------


class RealShapingTest(EnvelopeIntegrationTestCase):
    def test_a_real_safety_stop_during_a_throttle_ramp_zeroes_throttle_at_once(self):
        script = {1: detected(0.10), 2: detected(0.10), 3: detected(0.10), 4: UNUSABLE}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        for frame in (1, 2, 3):
            cut = self.step(runtime, supervisor, lateral, longitudinal, envelope,
                            frame, velocity=(1.0, 0.0, 0.0))
            self.assertUpstreamNominal(cut)
        self.assertGreater(cut.envelope.throttle_request, 0.0)

        stop = self.step(runtime, supervisor, lateral, longitudinal, envelope, 4,
                         velocity=(1.0, 0.0, 0.0))
        self.assertIs(stop.verdict.state, SafetyState.FAIL_SAFE)
        self.assertIs(stop.longitudinal.reason, LongitudinalControlReason.SAFETY_STOP)
        self.assertIs(stop.envelope.mode, CommandEnvelopeMode.STOPPING)
        self.assertEqual(stop.envelope.throttle_request, 0.0)
        self.assertEqual(stop.envelope.throttle_authority, 0.0)
        self.assertEqual(
            stop.envelope.brake_request, LONGITUDINAL_CONFIG.soft_stop_brake
        )

    def test_a_real_stop_then_recovery_releases_dwells_then_rises(self):
        script = {1: detected(0.10), 2: UNUSABLE}
        script.update({frame: detected(0.10) for frame in range(3, 12)})
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                  velocity=(1.0, 0.0, 0.0))
        stop = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2,
                         velocity=(1.0, 0.0, 0.0))
        self.assertIs(stop.envelope.mode, CommandEnvelopeMode.STOPPING)
        self.assertGreater(stop.envelope.brake_request, 0.0)

        seen_interlock = False
        seen_dwell = False
        seen_rise = False
        previous_brake = stop.envelope.brake_request
        for frame in range(3, 12):
            cut = self.step(runtime, supervisor, lateral, longitudinal, envelope,
                            frame, velocity=(1.0, 0.0, 0.0))
            request = cut.envelope
            self.assertEnvelopeSafe(request)
            self.assertFalse(
                request.throttle_request > 0.0 and request.brake_request > 0.0
            )
            if request.transition_interlocked:
                seen_interlock = True
                self.assertEqual(request.throttle_request, 0.0)
                self.assertLessEqual(request.brake_request, previous_brake)
                if request.brake_request == 0.0:
                    seen_dwell = True
            if request.throttle_request > 0.0:
                seen_rise = True
                self.assertEqual(request.brake_request, 0.0)
                self.assertTrue(seen_dwell, "throttle rose before the zero dwell")
            if request.command_allowed:
                previous_brake = request.brake_request
        self.assertTrue(seen_interlock, "the interlock never engaged")
        self.assertTrue(seen_dwell, "the zero-pedal dwell never happened")
        self.assertTrue(seen_rise, "propulsion never resumed")

    def test_a_real_throttle_ramp_never_exceeds_the_configured_rate(self):
        script = {frame: detected(0.10) for frame in range(1, 9)}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        previous = 0.0
        for frame in range(1, 9):
            cut = self.step(runtime, supervisor, lateral, longitudinal, envelope,
                            frame, velocity=(1.0, 0.0, 0.0))
            self.assertUpstreamNominal(cut)
            granted = cut.envelope.throttle_request - previous
            self.assertLessEqual(
                granted,
                ENVELOPE_CONFIG.throttle_rise_rate_per_second * STEP_SECONDS + 1e-12,
            )
            self.assertLessEqual(
                cut.envelope.throttle_request, cut.longitudinal.throttle_request
            )
            previous = cut.envelope.throttle_request

    def test_a_real_ordinary_brake_release_is_rate_limited(self):
        script = {frame: detected(0.10) for frame in range(1, 4)}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        braking = self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                            velocity=(2.8, 0.0, 0.0))
        self.assertGreater(braking.envelope.brake_request, 0.0)
        released = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2,
                             velocity=(2.0, 0.0, 0.0))
        self.assertUpstreamNominal(released)
        self.assertEqual(released.longitudinal.brake_request, 0.0)
        self.assertGreater(released.envelope.brake_request, 0.0)
        self.assertTrue(released.envelope.brake_release_rate_limited)
        self.assertLess(
            released.envelope.brake_request, braking.envelope.brake_request
        )
        self.assertLessEqual(
            braking.envelope.brake_request - released.envelope.brake_request,
            ENVELOPE_CONFIG.brake_release_rate_per_second * STEP_SECONDS + 1e-12,
        )

    def test_a_real_neutral_refusal_destroys_the_shaping_history(self):
        script = {frame: detected(0.10) for frame in range(1, 7)}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        for frame in (1, 2, 3, 4):
            cut = self.step(runtime, supervisor, lateral, longitudinal, envelope,
                            frame, velocity=(1.0, 0.0, 0.0))
        self.assertGreater(cut.envelope.throttle_request, 0.0)
        lifted = self.step(runtime, supervisor, lateral, longitudinal, envelope, 5,
                           velocity=(0.0, 0.0, 1.0))
        self.assertIs(
            lifted.longitudinal.reason,
            LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION,
        )
        self.assertEnvelopeNeutral(
            lifted.envelope, CommandEnvelopeReason.UPSTREAM_REFUSAL
        )
        resumed = self.step(runtime, supervisor, lateral, longitudinal, envelope, 6,
                            velocity=(1.0, 0.0, 0.0))
        self.assertUpstreamNominal(resumed)
        self.assertGreater(resumed.longitudinal.throttle_request, 0.0)
        self.assertEqual(resumed.envelope.throttle_request, 0.0)
        self.assertTrue(resumed.envelope.throttle_rate_limited)


# --------------------------------------------------------------------------
# E. Steering behaviour through the real pipeline
# --------------------------------------------------------------------------


class RealSteeringTest(EnvelopeIntegrationTestCase):
    def test_the_envelope_republishes_real_mission_9_steering_unchanged(self):
        script = {1: detected(0.05), 2: detected(0.10), 3: detected(0.02)}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        for frame in (1, 2, 3):
            cut = self.step(runtime, supervisor, lateral, longitudinal, envelope,
                            frame, velocity=(1.0, 0.0, 0.0))
            self.assertUpstreamNominal(cut)
            self.assertLessEqual(
                abs(cut.steering.steering), ENVELOPE_CONFIG.maximum_steering
            )
            self.assertEqual(cut.envelope.steering_request, cut.steering.steering)
            self.assertFalse(cut.envelope.steering_clamped)

    def test_the_envelope_adds_no_second_steering_slew(self):
        # Mission 9 owns the only steering rate limit.  A mirrored lane in one
        # step is slewed by Mission 9 alone, and Mission 10B republishes exactly
        # the value Mission 9 produced, rate-limited or not.
        script = {1: detected(0.30), 2: detected(-0.30)}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        first = self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                          velocity=(1.0, 0.0, 0.0))
        second = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2,
                           velocity=(1.0, 0.0, 0.0))
        self.assertUpstreamNominal(first)
        self.assertUpstreamNominal(second)
        # Mission 9 really did clamp, then really did rate limit, all by itself.
        self.assertTrue(first.steering.clamped)
        self.assertTrue(second.steering.rate_limited)
        self.assertEqual(first.envelope.steering_request, first.steering.steering)
        self.assertEqual(second.envelope.steering_request, second.steering.steering)
        self.assertFalse(first.envelope.steering_clamped)
        self.assertFalse(second.envelope.steering_clamped)
        # The step Mission 10B published is exactly the step Mission 9 asked for,
        # so no second slew was applied on top of the first.
        self.assertEqual(
            second.envelope.steering_request - first.envelope.steering_request,
            second.steering.steering - first.steering.steering,
        )

    def test_a_steering_reversal_while_braking_is_published_unchanged(self):
        script = {1: detected(0.10), 2: detected(-0.10)}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        first = self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                          velocity=(2.8, 0.0, 0.0))
        second = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2,
                           velocity=(2.8, 0.0, 0.0))
        self.assertUpstreamNominal(second)
        self.assertGreater(first.envelope.brake_request, 0.0)
        self.assertGreater(second.envelope.brake_request, 0.0)
        self.assertLess(second.steering.steering, first.steering.steering)
        self.assertEqual(second.envelope.steering_request, second.steering.steering)

    def test_the_final_steering_ceiling_can_still_subtract_authority(self):
        script = {1: detected(0.20)}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            script, envelope_config=CommandEnvelopeConfig(maximum_steering=0.05)
        )
        cut = self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                        velocity=(1.0, 0.0, 0.0))
        self.assertUpstreamNominal(cut)
        self.assertGreater(cut.steering.steering, 0.05)
        self.assertEqual(cut.envelope.steering_request, 0.05)
        self.assertEqual(cut.envelope.steering_authority, 0.05)
        self.assertTrue(cut.envelope.steering_clamped)
        self.assertTrue(cut.envelope.shaping_applied)
        self.assertIs(cut.envelope.reason, CommandEnvelopeReason.NOMINAL_SHAPED)


# --------------------------------------------------------------------------
# F. R1: real-pipeline evidence shapes, settlement, and public reconstruction
# --------------------------------------------------------------------------


def bypassed(base, **overrides):
    """A record of the exact accepted type with its constructor skipped."""
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


class RealEvidenceShapeTest(EnvelopeIntegrationTestCase):
    def test_every_real_longitudinal_output_has_an_accepted_speed_law_shape(self):
        """Sweep the real pipeline and pin the shape each branch publishes."""
        seen = {}
        scenarios = (
            ("nominal", dict(velocity=(1.0, 0.0, 0.0)), {}),
            ("at target", dict(velocity=(2.0, 0.0, 0.0)), {}),
            ("overspeed", dict(velocity=(2.8, 0.0, 0.0)), {}),
            ("reverse", dict(velocity=(-1.0, 0.0, 0.0)), {}),
            ("lateral motion", dict(velocity=(0.0, 1.0, 0.0)), {}),
            ("vertical motion", dict(velocity=(0.0, 0.0, 1.0)), {}),
            ("dead ego", dict(velocity=(1.0, 0.0, 0.0), vehicle_alive=False), {}),
            (
                "stale speed",
                dict(velocity=(1.0, 0.0, 0.0), ego_monotonic=control_time(1) - 0.9),
                {},
            ),
            (
                "expired target",
                dict(velocity=(1.0, 0.0, 0.0)),
                dict(
                    target=scenario_target(
                        2.0,
                        issued=BASE_MONOTONIC,
                        valid_until=BASE_MONOTONIC + 0.05,
                    )
                ),
            ),
            (
                "target stop",
                dict(velocity=(1.0, 0.0, 0.0)),
                dict(target=scenario_target(0.0)),
            ),
            ("lateral stop", dict(velocity=(1.0, 0.0, 0.0)), {"script": PARTIAL}),
        )
        for label, snapshot_kwargs, extra in scenarios:
            with self.subTest(label):
                script = {1: extra.pop("script", detected(0.10))}
                runtime, supervisor, lateral, longitudinal, envelope = self.build(
                    script
                )
                cut = self.step(
                    runtime, supervisor, lateral, longitudinal, envelope, 1,
                    **snapshot_kwargs, **extra,
                )
                shape = (
                    cut.longitudinal.effective_target_speed_mps is not None,
                    cut.longitudinal.observed_forward_speed_mps is not None,
                    cut.longitudinal.speed_error_mps is not None,
                    cut.longitudinal.speed_authority_mps is not None,
                )
                self.assertIn(
                    shape,
                    (
                        (False, False, False, False),
                        (False, True, False, False),
                        (True, True, True, True),
                    ),
                    f"{label}: {cut.longitudinal.reason} published {shape}",
                )
                seen[cut.longitudinal.reason] = shape
        self.assertGreaterEqual(len(seen), 8)

    def test_a_forged_partial_shape_on_a_real_stop_fails_closed(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {1: detected(0.10)}
        )
        cut = self.step(
            runtime, supervisor, lateral, longitudinal, envelope, 1,
            velocity=(1.0, 0.0, 0.0),
            ego_monotonic=control_time(1) - 0.90,
        )
        self.assertIs(cut.longitudinal.reason, LongitudinalControlReason.SPEED_STALE)
        self.assertIsNone(cut.longitudinal.effective_target_speed_mps)
        forged = bypassed(cut.longitudinal, effective_target_speed_mps=1.0)
        fresh = CommandEnvelope()
        request = fresh.update(cut.steering, forged, now=cut.now)
        self.assertEnvelopeNeutral(
            request, CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE
        )
        self.assertEqual(fresh.metrics.internal_errors, 0)

    def test_every_real_envelope_output_reconstructs_field_for_field(self):
        script = {1: detected(0.10), 2: detected(0.30), 3: PARTIAL, 4: detected(0.10)}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        rebuilt = 0
        for frame in (1, 2, 3, 4):
            cut = self.step(
                runtime, supervisor, lateral, longitudinal, envelope, frame,
                velocity=(1.0, 0.0, 0.0),
            )
            clone = CommandEnvelopeRequest(
                **{
                    field.name: getattr(cut.envelope, field.name)
                    for field in fields(CommandEnvelopeRequest)
                }
            )
            self.assertEqual(clone, cut.envelope)
            rebuilt += 1
        self.assertEqual(rebuilt, 4)

    def test_a_settlement_fault_on_the_real_chain_moves_no_baseline(self):
        script = {frame: detected(0.10) for frame in range(1, 4)}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        first = self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                          velocity=(1.0, 0.0, 0.0))
        self.assertUpstreamNominal(first)
        before = envelope._state

        value = snapshot(2, velocity=(1.0, 0.0, 0.0))
        published = self.publish(runtime, value, 2)
        now = control_time(2)
        verdict = supervisor.update(
            published, now=now, health=ComponentHealth.from_components(runtime)
        )
        steering = lateral.update(published, verdict, now=now)
        speed = EgoSpeedObservation.from_telemetry_snapshot(value)
        request = longitudinal.update(
            verdict, steering, speed, scenario_target(), now=now
        )
        self.assertTrue(request.longitudinal_allowed)

        original = command_module._accepted_metrics
        command_module._accepted_metrics = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("metrics")
        )
        try:
            with self.assertRaises(RuntimeError):
                envelope.update(steering, request, now=now)
        finally:
            command_module._accepted_metrics = original
        after = envelope._state
        self.assertIs(after.evidence, before.evidence)
        self.assertIs(after.accepted, before.accepted)
        self.assertIs(
            after.request.reason, CommandEnvelopeReason.INTERNAL_ERROR
        )
        self.assertEqual(after.request.evaluation_monotonic, now)
        # The very same real cut is still acceptable afterwards.
        recovered = envelope.update(steering, request, now=now)
        self.assertTrue(recovered.command_allowed)
        self.assertEqual(recovered.throttle_request, 0.0)


# --------------------------------------------------------------------------
# G. R2: hostile control time and real Mission 9 producer shapes
# --------------------------------------------------------------------------


@numbers.Real.register
class ExplodingReal:
    """A registered ``numbers.Real`` whose float conversion raises."""

    def __init__(self, exception: BaseException) -> None:
        self._exception = exception

    def __float__(self) -> float:
        raise self._exception


class RealHostileControlTimeTest(EnvelopeIntegrationTestCase):
    def test_a_hostile_control_time_neutralizes_a_real_running_command(self):
        script = {frame: detected(0.10) for frame in range(1, 6)}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        for frame in (1, 2, 3):
            cut = self.step(runtime, supervisor, lateral, longitudinal, envelope,
                            frame, velocity=(1.0, 0.0, 0.0))
            self.assertUpstreamNominal(cut)
        self.assertGreater(cut.envelope.throttle_request, 0.0)

        hostile = envelope.update(
            cut.steering, cut.longitudinal, now=ExplodingReal(RuntimeError("x"))
        )
        self.assertEnvelopeNeutral(hostile, CommandEnvelopeReason.TIME_INVALID)
        self.assertIsNone(hostile.evaluation_monotonic)
        self.assertEqual(envelope.metrics.internal_errors, 0)

        resumed = self.step(runtime, supervisor, lateral, longitudinal, envelope,
                            4, velocity=(1.0, 0.0, 0.0))
        self.assertUpstreamNominal(resumed)
        self.assertGreater(resumed.longitudinal.throttle_request, 0.0)
        self.assertEqual(resumed.envelope.throttle_request, 0.0)
        self.assertTrue(resumed.envelope.throttle_rate_limited)

    def test_a_hostile_control_time_advances_no_real_baseline(self):
        script = {frame: detected(0.10) for frame in range(1, 4)}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        first = self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                          velocity=(1.0, 0.0, 0.0))
        before = envelope._state.evidence
        envelope.update(
            first.steering, first.longitudinal,
            now=ExplodingReal(RuntimeError("x")),
        )
        self.assertIs(envelope._state.evidence, before)
        second = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2,
                           velocity=(1.0, 0.0, 0.0))
        self.assertTrue(second.envelope.command_allowed)


class RealLateralProducerShapeTest(EnvelopeIntegrationTestCase):
    """Every Mission 9 record the real pipeline emits must be accepted."""

    def collect(self):
        seen = {}
        scenarios = (
            ("nominal", {1: detected(0.10)}, 1, dict(velocity=(1.0, 0.0, 0.0))),
            ("low confidence", {1: detected(0.10, 0.0, 0.50)}, 1,
             dict(velocity=(0.0, 0.0, 0.0))),
            ("partial geometry", {1: PARTIAL}, 1, dict(velocity=(1.0, 0.0, 0.0))),
            ("unusable", {1: detected(0.10), 2: UNUSABLE}, 2,
             dict(velocity=(1.0, 0.0, 0.0))),
            ("coasting", {1: detected(0.10), 2: detected(0.10), 3: MISSING}, 3,
             dict(velocity=(0.0, 0.0, 0.0))),
        )
        for label, script, frames, kwargs in scenarios:
            runtime, supervisor, lateral, longitudinal, envelope = self.build(
                script
            )
            for frame in range(1, frames + 1):
                cut = self.step(runtime, supervisor, lateral, longitudinal,
                                envelope, frame, **kwargs)
                seen.setdefault(cut.steering.reason, (label, cut.steering))
        return seen

    def test_every_real_lateral_output_survives_canonicalization(self):
        seen = self.collect()
        self.assertGreaterEqual(len(seen), 4)
        for reason, (label, record) in seen.items():
            with self.subTest(reason=reason, scenario=label):
                canonical = _canonicalize_lateral_request(record)
                self.assertIs(canonical.reason, reason)
                self.assertEqual(canonical, record)
                # The expectation is restated independently from the accepted
                # Mission 9 source, never asked of a Mission 10B helper.
                self.assertTrue(
                    real_lateral_shape(
                        record.reason,
                        record.safety_state,
                        record.temporal_state,
                        record.source_stamp is not None,
                    ),
                    f"the real pipeline emitted a shape the oracle rejects: {record}",
                )

    def test_a_real_safety_refusal_never_carries_a_permitted_state(self):
        script = {1: detected(0.10), 2: UNUSABLE}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                  velocity=(1.0, 0.0, 0.0))
        cut = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2,
                        velocity=(1.0, 0.0, 0.0))
        self.assertIs(
            cut.steering.reason, LateralControlReason.SAFETY_NOT_PERMITTED
        )
        self.assertNotIn(
            cut.steering.safety_state,
            (SafetyState.NOMINAL, SafetyState.DEGRADED),
        )
        self.assertIs(cut.envelope.mode, CommandEnvelopeMode.STOPPING)

    def test_a_real_geometry_refusal_carries_a_permitted_state(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {1: PARTIAL}
        )
        cut = self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                        velocity=(1.0, 0.0, 0.0))
        self.assertIs(
            cut.steering.reason, LateralControlReason.INSUFFICIENT_GEOMETRY
        )
        self.assertIn(
            cut.steering.safety_state,
            (SafetyState.NOMINAL, SafetyState.DEGRADED),
        )
        self.assertIs(cut.envelope.mode, CommandEnvelopeMode.STOPPING)

    def test_a_forged_permitted_state_on_a_real_safety_refusal_is_malformed(self):
        script = {1: detected(0.10), 2: UNUSABLE}
        runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
        self.step(runtime, supervisor, lateral, longitudinal, envelope, 1,
                  velocity=(1.0, 0.0, 0.0))
        cut = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2,
                        velocity=(1.0, 0.0, 0.0))
        forged = bypassed(cut.steering, safety_state=SafetyState.NOMINAL)
        fresh = CommandEnvelope()
        request = fresh.update(forged, cut.longitudinal, now=cut.now)
        self.assertEnvelopeNeutral(
            request, CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE
        )
        self.assertIsNot(request.mode, CommandEnvelopeMode.STOPPING)


# ==========================================================================
# R3  Same-cut reachability against the real pipeline
# ==========================================================================


class RealSameCutPairTest(EnvelopeIntegrationTestCase):
    """Positive fixtures are whatever the real M7->M8->M9->M10A chain produced.

    Nothing here asks Mission 10B what to expect: the pairs are harvested by
    running the accepted controllers, and the relationship they must satisfy is
    restated independently in ``tests.test_command_envelope``.
    """

    SCENARIOS = (
        # label, perception script, frames, per-step snapshot kwargs
        ("nominal tracking", {f: detected(0.10, 0.02) for f in range(1, 5)}, 4, {}),
        (
            "low-confidence tracking",
            {f: detected(0.10, 0.02, 0.50) for f in range(1, 5)},
            4,
            {},
        ),
        ("partial geometry", {1: detected(0.10), 2: PARTIAL, 3: PARTIAL}, 3, {}),
        (
            "coasting",
            {1: detected(0.10), 2: detected(0.10), 3: MISSING, 4: MISSING},
            4,
            {},
        ),
        (
            "coasting without complete scalars",
            {1: detected(0.10), 2: PARTIAL, 3: MISSING, 4: MISSING},
            4,
            {},
        ),
        ("lost", {1: detected(0.10), 2: MISSING, 3: MISSING, 4: MISSING,
                  5: MISSING, 6: MISSING}, 6, {}),
        ("input unusable", {1: detected(0.10), 2: UNUSABLE, 3: UNUSABLE}, 3, {}),
        (
            "reverse motion",
            {f: detected(0.10) for f in range(1, 4)},
            3,
            dict(velocity=(-3.0, 0.0, 0.0)),
        ),
        (
            "excessive lateral motion",
            {f: detected(0.10) for f in range(1, 4)},
            3,
            dict(velocity=(1.0, 5.0, 0.0)),
        ),
        (
            "excessive vertical motion",
            {f: detected(0.10) for f in range(1, 4)},
            3,
            dict(velocity=(1.0, 0.0, 5.0)),
        ),
    )

    def harvest(self):
        """Every genuine cut the scenarios above produce."""
        cuts = []
        for label, script, frames, kwargs in self.SCENARIOS:
            runtime, supervisor, lateral, longitudinal, envelope = self.build(script)
            for frame in range(1, frames + 1):
                cuts.append(
                    (
                        label,
                        self.step(
                            runtime,
                            supervisor,
                            lateral,
                            longitudinal,
                            envelope,
                            frame,
                            **kwargs,
                        ),
                    )
                )
        return cuts

    def test_every_real_cut_satisfies_the_independent_relationship(self):
        cuts = self.harvest()
        self.assertGreaterEqual(len(cuts), 30)
        lateral_reasons = set()
        pairs = set()
        for label, cut in cuts:
            with self.subTest(scenario=label, frame=cut.published.source_stamp.frame):
                record = cut.steering
                self.assertTrue(
                    real_lateral_shape(
                        record.reason,
                        record.safety_state,
                        record.temporal_state,
                        record.source_stamp is not None,
                    ),
                    f"unreachable Mission 9 shape from the real pipeline: {record}",
                )
                lateral_reasons.add(record.reason)
                if cut.longitudinal.guidance_stamp is not None:
                    self.assertIn(
                        cut.longitudinal.reason,
                        REAL_SAME_CUT_LONGITUDINAL[record.reason],
                        "the real pipeline produced a pair the oracle rejects",
                    )
                    pairs.add((record.reason, cut.longitudinal.reason))
        self.assertGreaterEqual(len(lateral_reasons), 5)
        self.assertGreaterEqual(len(pairs), 5)

    def test_every_real_cut_is_accepted_by_the_envelope(self):
        """No genuine pair is ever refused as an impossible one."""
        for label, cut in self.harvest():
            with self.subTest(scenario=label):
                self.assertIsNot(
                    cut.envelope.reason,
                    CommandEnvelopeReason.CROSS_RECORD_IDENTITY_MISMATCH,
                )
                self.assertIsNot(
                    cut.envelope.reason,
                    CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE,
                )
                self.assertIsNot(
                    cut.envelope.reason,
                    CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE,
                )

    def test_a_real_geometry_refusal_becomes_a_real_trusted_lateral_stop(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {1: detected(0.10), 2: PARTIAL}
        )
        self.step(runtime, supervisor, lateral, longitudinal, envelope, 1)
        cut = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2)
        self.assertIs(
            cut.steering.reason, LateralControlReason.INSUFFICIENT_GEOMETRY
        )
        self.assertIs(cut.steering.safety_state, SafetyState.DEGRADED)
        self.assertIs(cut.longitudinal.reason, LongitudinalControlReason.LATERAL_STOP)
        self.assertIs(cut.envelope.mode, CommandEnvelopeMode.STOPPING)
        self.assertIs(
            cut.envelope.reason, CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP
        )
        self.assertGreater(cut.envelope.brake_request, 0.0)
        self.assertEqual(cut.envelope.steering_authority, 0.0)

    def test_a_real_bound_identity_mismatch_never_reaches_a_bound_partner(self):
        """The BLOCKER, driven for real: Mission 10A refuses the same pair."""
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {f: detected(0.10) for f in range(1, 5)}
        )
        self.step(runtime, supervisor, lateral, longitudinal, envelope, 1)
        self.step(runtime, supervisor, lateral, longitudinal, envelope, 2)
        stale = self.step(runtime, supervisor, lateral, longitudinal, envelope, 3)

        value = snapshot(4)
        published = self.publish(runtime, value, 4)
        now = control_time(4)
        # The decision issued for frame 3, handed to Mission 9 with frame 4.
        steering = lateral.update(published, stale.verdict, now=now)
        self.assertIs(steering.reason, LateralControlReason.IDENTITY_MISMATCH)
        self.assertIsNotNone(steering.source_stamp)
        self.assertIs(steering.safety_state, SafetyState.NOMINAL)
        self.assertIs(steering.temporal_state, TemporalLaneState.TRACKING)

        speed = EgoSpeedObservation.from_telemetry_snapshot(value)
        request = longitudinal.update(
            stale.verdict, steering, speed, scenario_target(), now=now
        )
        # Mission 10A re-checks the same axes and reaches the same verdict, so a
        # bound partner never exists for this Mission 9 record.
        self.assertIsNone(request.guidance_stamp)
        self.assertIs(request.reason, LongitudinalControlReason.IDENTITY_MISMATCH)
        bounded = envelope.update(steering, request, now=now)
        self.assertIsNot(bounded.mode, CommandEnvelopeMode.STOPPING)
        self.assertFalse(bounded.command_allowed)

    def test_a_forged_bound_partner_for_that_record_is_refused(self):
        """Force the impossible pair the reviewer reported and prove it fails."""
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {f: detected(0.10) for f in range(1, 5)}
        )
        self.step(runtime, supervisor, lateral, longitudinal, envelope, 1)
        good = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2)
        stale = self.step(runtime, supervisor, lateral, longitudinal, envelope, 3)

        value = snapshot(4)
        published = self.publish(runtime, value, 4)
        now = control_time(4)
        steering = lateral.update(published, stale.verdict, now=now)
        self.assertIs(steering.reason, LateralControlReason.IDENTITY_MISMATCH)

        # A real Mission 10A LATERAL_STOP, retargeted onto the mismatched record.
        forged = bypassed(
            good.longitudinal,
            mode=LongitudinalControlMode.STOPPING,
            reason=LongitudinalControlReason.LATERAL_STOP,
            longitudinal_allowed=False,
            stop_required=True,
            throttle_request=0.0,
            brake_request=0.20,
            throttle_authority=0.0,
            brake_authority=0.40,
            effective_target_speed_mps=None,
            speed_error_mps=None,
            speed_authority_mps=None,
            evaluation_monotonic=now,
            guidance_stamp=steering.source_stamp,
            snapshot_revision=steering.snapshot_revision,
            safety_state=steering.safety_state,
            lateral_mode=steering.mode,
            temporal_state=steering.temporal_state,
        )
        # Every field the envelope binds on now agrees.
        self.assertEqual(steering.source_stamp, forged.guidance_stamp)
        self.assertEqual(steering.snapshot_revision, forged.snapshot_revision)
        self.assertIs(steering.safety_state, forged.safety_state)
        self.assertIs(steering.temporal_state, forged.temporal_state)
        self.assertIs(steering.mode, forged.lateral_mode)

        fresh = CommandEnvelope(ENVELOPE_CONFIG)
        request = fresh.update(steering, forged, now=now)
        self.assertEnvelopeNeutral(
            request, CommandEnvelopeReason.CROSS_RECORD_IDENTITY_MISMATCH
        )
        self.assertIsNot(request.reason, CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP)
        self.assertEqual(request.brake_request, 0.0)
        self.assertEqual(request.brake_authority, 0.0)
        self.assertEqual(fresh.metrics.stop_commands, 0)

    def test_one_mutated_field_on_a_genuine_pair_is_always_refused(self):
        """Mutate exactly one axis of a real cut and require a fail-closed answer."""
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {f: detected(0.10) for f in range(1, 4)}
        )
        self.step(runtime, supervisor, lateral, longitudinal, envelope, 1)
        cut = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2)
        self.assertTrue(cut.envelope.command_allowed)

        mutations = (
            ("lateral reason", dict(reason=LateralControlReason.DEGRADED_TRACKING), {}),
            ("lateral mode", dict(mode=LateralControlMode.DEGRADED), {}),
            ("safety state", dict(safety_state=SafetyState.DEGRADED), {}),
            ("temporal state", dict(temporal_state=TemporalLaneState.COASTING), {}),
            (
                "longitudinal reason",
                {},
                dict(reason=LongitudinalControlReason.DEGRADED_TRACKING),
            ),
            ("longitudinal revision", {}, dict(snapshot_revision=99)),
            (
                "longitudinal safety",
                {},
                dict(safety_state=SafetyState.DEGRADED),
            ),
        )
        for label, lateral_overrides, longitudinal_overrides in mutations:
            with self.subTest(mutation=label):
                fresh = CommandEnvelope(ENVELOPE_CONFIG)
                request = fresh.update(
                    bypassed(cut.steering, **lateral_overrides)
                    if lateral_overrides
                    else cut.steering,
                    bypassed(cut.longitudinal, **longitudinal_overrides)
                    if longitudinal_overrides
                    else cut.longitudinal,
                    now=cut.now,
                )
                self.assertFalse(
                    request.command_allowed,
                    f"a mutated {label} still produced a command",
                )
                self.assertIs(request.mode, CommandEnvelopeMode.DISABLED)


class RealForeignTypeAndFaultTest(EnvelopeIntegrationTestCase):
    """Foreign top-level types and internal faults, against real pipeline state."""

    def warmed(self):
        runtime, supervisor, lateral, longitudinal, envelope = self.build(
            {f: detected(0.10) for f in range(1, 5)}
        )
        self.step(runtime, supervisor, lateral, longitudinal, envelope, 1)
        cut = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2)
        self.assertTrue(cut.envelope.command_allowed)
        return (runtime, supervisor, lateral, longitudinal, envelope), cut

    def test_a_foreign_type_publishes_nothing_against_a_real_warm_envelope(self):
        parts, cut = self.warmed()
        envelope = parts[4]
        warm = envelope.latest_request
        metrics = envelope.metrics
        for value in (None, object(), {"mode": None}, "lateral", (1, 2),
                      type(cut.steering), cut.longitudinal):
            with self.subTest(argument=type(value).__name__):
                with self.assertRaises(TypeError):
                    envelope.update(value, cut.longitudinal, now=cut.now)
                self.assertIs(envelope.latest_request, warm)
                self.assertEqual(envelope.metrics, metrics)
                self.assertTrue(envelope.latest_request.command_allowed)

    def test_real_records_are_exact_and_spoofed_classes_publish_nothing(self):
        class SpoofedLateral:
            @property
            def __class__(self):
                return type(cut.steering)

        parts, cut = self.warmed()
        envelope = parts[4]
        self.assertIs(type(cut.steering), LateralControlRequest)
        self.assertIs(type(cut.longitudinal), LongitudinalControlRequest)
        state = envelope._state
        metrics = envelope.metrics
        with self.assertRaises(TypeError):
            envelope.update(
                SpoofedLateral(), cut.longitudinal, now=cut.now + STEP_SECONDS
            )
        self.assertIs(envelope._state, state)
        self.assertEqual(envelope.metrics, metrics)

    def test_real_traffic_continues_unchanged_after_the_raised_type_error(self):
        parts, _cut = self.warmed()
        runtime, supervisor, lateral, longitudinal, envelope = parts
        with self.assertRaises(TypeError):
            envelope.update(None, None, now=control_time(3))
        after = self.step(runtime, supervisor, lateral, longitudinal, envelope, 3)
        self.assertTrue(after.envelope.command_allowed)

        clean_parts = self.build({f: detected(0.10) for f in range(1, 5)})
        c_runtime, c_supervisor, c_lateral, c_longitudinal, c_envelope = clean_parts
        for frame in (1, 2, 3):
            reference = self.step(
                c_runtime, c_supervisor, c_lateral, c_longitudinal, c_envelope, frame
            )
        self.assertEqual(after.envelope, reference.envelope)
        self.assertEqual(envelope.metrics, c_envelope.metrics)

    def test_an_internal_fault_after_a_real_command_publishes_internal_error(self):
        parts, cut = self.warmed()
        envelope = parts[4]
        before = envelope.metrics

        class Boom(RuntimeError):
            pass

        class FailingMath:
            def __getattr__(self, name):
                return getattr(command_module.math, name)

            @staticmethod
            def isfinite(value):
                raise Boom("injected internal finiteness failure")

        real_math = command_module.math
        command_module.math = FailingMath()
        try:
            with self.assertRaises(Boom):
                envelope.update(cut.steering, cut.longitudinal, now=cut.now + 0.1)
        finally:
            command_module.math = real_math

        visible = envelope.latest_request
        self.assertIs(visible.reason, CommandEnvelopeReason.INTERNAL_ERROR)
        self.assertFalse(visible.command_allowed)
        self.assertEqual(visible.brake_request, 0.0)
        self.assertEqual(visible.throttle_request, 0.0)
        self.assertEqual(
            envelope.metrics.internal_errors, before.internal_errors + 1
        )
        runtime, supervisor, lateral, longitudinal, _envelope = parts
        recovered = self.step(
            runtime, supervisor, lateral, longitudinal, envelope, 3
        )
        self.assertTrue(recovered.envelope.command_allowed)


class RealEvidenceClassificationTest(EnvelopeIntegrationTestCase):
    """Where a raw exception arose decides how a real running envelope reports it.

    The records under test are the genuine Mission 9 and Mission 10A outputs of a
    real warmed pipeline, so these cases pin the classification against the
    evidence the exhibition actually produces rather than hand-built fixtures.
    The oracle is stated by the test: a fault injected into caller-owned
    machinery must read as malformed evidence, and a fault injected into one of
    Mission 10B's own steps must read as an internal error.
    """

    def warmed(self):
        parts = self.build({frame: detected(0.10) for frame in range(1, 6)})
        runtime, supervisor, lateral, longitudinal, envelope = parts
        self.step(runtime, supervisor, lateral, longitudinal, envelope, 1)
        cut = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2)
        self.assertUpstreamNominal(cut)
        self.assertTrue(cut.envelope.command_allowed)
        return parts, cut

    # ------------------------------------------------------------- oracles

    def assertInternalError(self, envelope, before, baseline, warm, caught):
        visible = envelope.latest_request
        self.assertIs(visible.reason, CommandEnvelopeReason.INTERNAL_ERROR)
        self.assertIs(visible.mode, CommandEnvelopeMode.DISABLED)
        self.assertFalse(visible.command_allowed)
        self.assertFalse(visible.stop_required)
        self.assertFalse(visible.evidence_bound)
        for name in ("steering_request", "throttle_request", "brake_request",
                     "steering_authority", "throttle_authority",
                     "brake_authority"):
            self.assertEqual(getattr(visible, name), 0.0, name)
        self.assertFalse(visible.shaping_applied)
        after = envelope.metrics
        self.assertEqual(after.internal_errors, before.internal_errors + 1)
        self.assertEqual(after.refusals, before.refusals + 1)
        self.assertEqual(after.commands, before.commands)
        self.assertEqual(envelope._state.evidence, baseline)
        shaping = envelope._state.shaping
        self.assertEqual(shaping.throttle_anchor, 0.0)
        self.assertEqual(shaping.brake_anchor, 0.0)
        self.assertIsNone(shaping.shaping_monotonic)
        self.assertIsNot(visible, warm)
        for value in retained_objects(envelope):
            self.assertNotIsInstance(value, BaseException)
            self.assertFalse(hasattr(value, "tb_frame"))
        self.assertNotIn(id(caught), {id(v) for v in retained_objects(envelope)})

    def assertMalformed(self, envelope, before, baseline, warm, request, reason):
        self.assertIs(request.reason, reason)
        self.assertIs(request.mode, CommandEnvelopeMode.DISABLED)
        self.assertFalse(request.command_allowed)
        for name in ("steering_request", "throttle_request", "brake_request",
                     "steering_authority", "throttle_authority",
                     "brake_authority"):
            self.assertEqual(getattr(request, name), 0.0, name)
        after = envelope.metrics
        self.assertEqual(after.internal_errors, before.internal_errors)
        self.assertEqual(after.refusals, before.refusals + 1)
        self.assertEqual(after.commands, before.commands)
        self.assertEqual(envelope._state.evidence, baseline)
        self.assertIsNot(request, warm)

    # ------------------------------------------------------------- drivers

    def run_internal(self, name, exception, *, fail_after=0):
        """Break one of Mission 10B's own steps mid-pipeline."""
        parts, _cut = self.warmed()
        runtime, supervisor, lateral, longitudinal, envelope = parts
        before = envelope.metrics
        baseline = envelope._state.evidence
        warm = envelope.latest_request
        original = getattr(command_module, name)
        counters = {"calls": 0}

        def boom(*args, **kwargs):
            counters["calls"] += 1
            if counters["calls"] > fail_after:
                raise exception
            return original(*args, **kwargs)

        setattr(command_module, name, boom)
        try:
            with self.assertRaises(type(exception)) as caught:
                self.step(runtime, supervisor, lateral, longitudinal, envelope, 3)
        finally:
            setattr(command_module, name, original)
        self.assertIs(caught.exception, exception)
        self.assertInternalError(envelope, before, baseline, warm, caught.exception)
        recovered = self.step(
            runtime, supervisor, lateral, longitudinal, envelope, 4
        )
        self.assertTrue(recovered.envelope.command_allowed)

    # ---------------------------------------------------- internal, lateral

    def test_a_real_internal_lateral_fault_is_an_internal_error(self):
        for name, fail_after in (
            ("_canonical_stamp", 0),
            ("_validate_detached_lateral", 0),
            ("_LATERAL_RECONSTRUCTOR", 0),
        ):
            for factory in (TypeError, ValueError):
                with self.subTest(step=name, raises=factory.__name__):
                    self.run_internal(
                        name, factory(f"internal {name} defect"),
                        fail_after=fail_after,
                    )

    def test_a_real_internal_longitudinal_fault_is_an_internal_error(self):
        # The lateral axis canonicalizes its stamp first.
        for name, fail_after in (
            ("_canonical_stamp", 1),
            ("_validate_detached_longitudinal", 0),
            ("_LONGITUDINAL_RECONSTRUCTOR", 0),
        ):
            for factory in (TypeError, ValueError):
                with self.subTest(step=name, raises=factory.__name__):
                    self.run_internal(
                        name, factory(f"internal {name} defect"),
                        fail_after=fail_after,
                    )

    # ----------------------------------------------------- caller-originated

    def test_a_real_caller_property_fault_is_malformed_evidence(self):
        for owner, attribute, reason in (
            (LateralControlRequest, "steering",
             CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE),
            (LongitudinalControlRequest, "throttle_request",
             CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE),
        ):
            for factory in (TypeError, ValueError):
                with self.subTest(field=attribute, raises=factory.__name__):
                    parts, cut = self.warmed()
                    runtime, supervisor, lateral, longitudinal, envelope = parts
                    before = envelope.metrics
                    baseline = envelope._state.evidence
                    warm = envelope.latest_request
                    exception = factory("caller property")
                    descriptor = getattr(owner, attribute)

                    def read(_self, exc=exception):
                        raise exc

                    # The whole pipeline is driven *before* the accepted
                    # property is replaced, because the real Mission 9 and
                    # Mission 10A controllers must assign it to build a record.
                    setattr(owner, attribute, property(read))
                    try:
                        request = envelope.update(
                            cut.steering, cut.longitudinal, now=cut.now
                        )
                    finally:
                        setattr(owner, attribute, descriptor)
                    self.assertMalformed(
                        envelope, before, baseline, warm, request, reason
                    )
                    recovered = self.step(
                        runtime, supervisor, lateral, longitudinal, envelope, 3
                    )
                    self.assertTrue(recovered.envelope.command_allowed)

    def test_a_real_caller_conversion_fault_is_malformed_evidence(self):
        for axis, reason in (
            ("lateral", CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE),
            ("longitudinal",
             CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE),
        ):
            for factory in (TypeError, ValueError):
                with self.subTest(axis=axis, raises=factory.__name__):
                    parts, cut = self.warmed()
                    runtime, supervisor, lateral, longitudinal, envelope = parts
                    before = envelope.metrics
                    baseline = envelope._state.evidence
                    warm = envelope.latest_request
                    hostile = ExplodingReal(factory("caller conversion"))
                    if axis == "lateral":
                        lat = bypassed(cut.steering, steering=hostile)
                        lon = cut.longitudinal
                    else:
                        lat = cut.steering
                        lon = bypassed(cut.longitudinal, throttle_request=hostile)
                    request = envelope.update(lat, lon, now=cut.now)
                    self.assertMalformed(
                        envelope, before, baseline, warm, request, reason
                    )
                    recovered = self.step(
                        runtime, supervisor, lateral, longitudinal, envelope, 3
                    )
                    self.assertTrue(recovered.envelope.command_allowed)

    # ------------------------------------------------------------ contrast

    def test_the_real_pipeline_answers_the_two_faults_differently(self):
        """Same real cut, same exception type, two honest and distinct answers."""
        parts, cut = self.warmed()
        runtime, supervisor, lateral, longitudinal, envelope = parts
        before = envelope.metrics

        caller = envelope.update(
            bypassed(
                cut.steering, steering=ExplodingReal(TypeError("caller"))
            ),
            cut.longitudinal,
            now=cut.now,
        )
        self.assertIs(
            caller.reason, CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE
        )
        self.assertEqual(
            envelope.metrics.internal_errors, before.internal_errors
        )

        defect = TypeError("internal")
        original = command_module._validate_detached_lateral

        def boom(*args, **kwargs):
            raise defect

        command_module._validate_detached_lateral = boom
        try:
            with self.assertRaises(TypeError) as caught:
                envelope.update(cut.steering, cut.longitudinal, now=cut.now)
        finally:
            command_module._validate_detached_lateral = original
        self.assertIs(caught.exception, defect)
        self.assertIs(
            envelope.latest_request.reason, CommandEnvelopeReason.INTERNAL_ERROR
        )
        self.assertEqual(
            envelope.metrics.internal_errors, before.internal_errors + 1
        )
        recovered = self.step(
            runtime, supervisor, lateral, longitudinal, envelope, 3
        )
        self.assertTrue(recovered.envelope.command_allowed)


class RealDetachedContractBoundaryTest(EnvelopeIntegrationTestCase):
    """The detached contract boundary, over records a real pipeline produced.

    Mission 10B proves both accepted upstream contracts itself and reconstructs
    each record once.  These tests pin that against genuine Mission 9 and Mission
    10A outputs: the evidence is real, the fault is injected into accepted
    *upstream* code Mission 10B merely depends on, and the oracle is stated here
    -- an upstream defect is an internal error, never bad caller evidence.
    """

    # The upstream helpers each accepted ``__post_init__`` depends on.
    LATERAL_DEPENDENCIES = (
        "_finite_real",
        "_bounded_real",
        "_builtin_bool",
        "_canonical_stamp",
        "_nonnegative_builtin_int",
    )
    LONGITUDINAL_DEPENDENCIES = LATERAL_DEPENDENCIES + ("_nonnegative_real",)

    def warmed(self):
        parts = self.build({frame: detected(0.10) for frame in range(1, 8)})
        runtime, supervisor, lateral, longitudinal, envelope = parts
        self.step(runtime, supervisor, lateral, longitudinal, envelope, 1)
        cut = self.step(runtime, supervisor, lateral, longitudinal, envelope, 2)
        self.assertUpstreamNominal(cut)
        self.assertTrue(cut.envelope.command_allowed)
        return parts, cut

    def assertReRaised(self, envelope, before, baseline, warm, caught):
        """Everything an unexpected internal failure must publish."""
        visible = envelope.latest_request
        self.assertIs(visible.reason, CommandEnvelopeReason.INTERNAL_ERROR)
        self.assertIs(visible.mode, CommandEnvelopeMode.DISABLED)
        self.assertFalse(visible.command_allowed)
        self.assertFalse(visible.stop_required)
        self.assertFalse(visible.evidence_bound)
        self.assertFalse(visible.shaping_applied)
        for name in ("steering_request", "throttle_request", "brake_request",
                     "steering_authority", "throttle_authority",
                     "brake_authority"):
            self.assertEqual(getattr(visible, name), 0.0, name)
        after = envelope.metrics
        self.assertEqual(after.internal_errors, before.internal_errors + 1)
        self.assertEqual(after.refusals, before.refusals + 1)
        self.assertEqual(after.commands, before.commands)
        self.assertEqual(envelope._state.evidence, baseline)
        shaping = envelope._state.shaping
        self.assertEqual(shaping.throttle_anchor, 0.0)
        self.assertEqual(shaping.brake_anchor, 0.0)
        self.assertIsNone(shaping.shaping_monotonic)
        self.assertIsNot(visible, warm)
        self.assertNotIn(id(caught), {id(v) for v in retained_objects(envelope)})
        for value in retained_objects(envelope):
            self.assertNotIsInstance(value, BaseException)
            self.assertFalse(hasattr(value, "tb_frame"))

    def instrument(self):
        """Counters on every accepted upstream constructor dependency."""
        import control.lateral as lateral_module
        import control.longitudinal as longitudinal_module

        counts = {}
        restore = []
        for module, names in (
            (lateral_module, self.LATERAL_DEPENDENCIES),
            (longitudinal_module, self.LONGITUDINAL_DEPENDENCIES),
        ):
            for name in names:
                key = f"{module.__name__}.{name}"
                counts[key] = 0
                original = getattr(module, name)
                restore.append((module, name, original))

                def counting(*args, _key=key, _original=original, **kwargs):
                    counts[_key] += 1
                    return _original(*args, **kwargs)

                setattr(module, name, counting)
        return counts, restore

    # --------------------------------------------------- upstream dependency

    def run_upstream_fault(self, module_name, helper, exception, *, once):
        import importlib

        module = importlib.import_module(module_name)
        parts, cut = self.warmed()
        runtime, supervisor, lateral, longitudinal, envelope = parts
        before = envelope.metrics
        baseline = envelope._state.evidence
        warm = envelope.latest_request
        original = getattr(module, helper)
        counters = {"calls": 0, "fired": False}

        def boom(*args, **kwargs):
            counters["calls"] += 1
            if not (once and counters["fired"]):
                counters["fired"] = True
                raise exception
            return original(*args, **kwargs)

        # ``cut`` already holds the real Mission 9 and Mission 10A records, so
        # the accepted controllers ran to completion before the fault existed.
        setattr(module, helper, boom)
        try:
            with self.assertRaises(type(exception)) as caught:
                envelope.update(cut.steering, cut.longitudinal, now=cut.now)
        finally:
            setattr(module, helper, original)
        self.assertIs(caught.exception, exception)
        self.assertGreater(counters["calls"], 0)
        self.assertReRaised(envelope, before, baseline, warm, caught.exception)
        self.assertNotIn(
            envelope.latest_request.reason,
            (
                CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE,
                CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE,
            ),
        )
        recovered = self.step(
            runtime, supervisor, lateral, longitudinal, envelope, 3
        )
        self.assertTrue(recovered.envelope.command_allowed)
        self.assertEqual(recovered.envelope.throttle_request, 0.0)
        self.assertEqual(recovered.envelope.brake_request, 0.0)

    def test_a_real_lateral_upstream_dependency_fault_is_an_internal_error(self):
        for helper in self.LATERAL_DEPENDENCIES:
            for factory in (TypeError, ValueError, RuntimeError):
                for once in (False, True):
                    with self.subTest(helper=helper, raises=factory.__name__,
                                      once=once):
                        self.run_upstream_fault(
                            "control.lateral",
                            helper,
                            factory(f"upstream {helper} defect"),
                            once=once,
                        )

    def test_a_real_longitudinal_upstream_dependency_fault_is_an_internal_error(self):
        for helper in self.LONGITUDINAL_DEPENDENCIES:
            for factory in (TypeError, ValueError, RuntimeError):
                for once in (False, True):
                    with self.subTest(helper=helper, raises=factory.__name__,
                                      once=once):
                        self.run_upstream_fault(
                            "control.longitudinal",
                            helper,
                            factory(f"upstream {helper} defect"),
                            once=once,
                        )

    # ------------------------------------------------------ real-record parity

    def test_every_real_upstream_record_satisfies_the_local_contract(self):
        """A genuine pipeline output is never judged malformed."""
        parts = self.build({frame: detected(0.10) for frame in range(1, 8)})
        runtime, supervisor, lateral, longitudinal, envelope = parts
        seen = 0
        for index in range(1, 7):
            cut = self.step(
                runtime, supervisor, lateral, longitudinal, envelope, index
            )
            for record, detach, validate, prove, rebuild, cls in (
                (
                    cut.steering,
                    command_module._detach_lateral,
                    command_module._validate_detached_lateral,
                    command_module._prove_detached_lateral_contract,
                    command_module._reconstruct_lateral,
                    LateralControlRequest,
                ),
                (
                    cut.longitudinal,
                    command_module._detach_longitudinal,
                    command_module._validate_detached_longitudinal,
                    command_module._prove_detached_longitudinal_contract,
                    command_module._reconstruct_longitudinal,
                    LongitudinalControlRequest,
                ),
            ):
                with self.subTest(index=index, axis=cls.__name__):
                    detached = validate(detach(record))
                    prove(detached)          # must not raise on real evidence
                    rebuilt = rebuild(detached)
                    self.assertIs(type(rebuilt), cls)
                    self.assertIsNot(rebuilt, record)
                    # The rebuilt record is an exact, detached equal of the real
                    # one the accepted controller published.
                    for field in fields(cls):
                        self.assertEqual(
                            getattr(rebuilt, field.name),
                            getattr(record, field.name),
                            field.name,
                        )
                    seen += 1
        self.assertGreaterEqual(seen, 12)

    def test_a_real_cut_reconstructs_each_accepted_record_exactly_once(self):
        parts, cut = self.warmed()
        counts, restore = self.instrument()
        try:
            request = parts[4].update(
                cut.steering, cut.longitudinal, now=cut.now
            )
        finally:
            for module, name, original in restore:
                setattr(module, name, original)
        # Both records were accepted: neither axis was refused as malformed and
        # nothing was reported as a defect.
        self.assertNotIn(
            request.reason,
            (
                CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE,
                CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE,
                CommandEnvelopeReason.INTERNAL_ERROR,
            ),
        )
        # One Mission 9 construction: one stamp and three bools.
        self.assertEqual(counts["control.lateral._canonical_stamp"], 1)
        self.assertEqual(counts["control.lateral._builtin_bool"], 3)
        # One Mission 10A construction: two stamps and four bools.
        self.assertEqual(counts["control.longitudinal._canonical_stamp"], 2)
        self.assertEqual(counts["control.longitudinal._builtin_bool"], 4)

    def test_no_upstream_code_runs_for_a_locally_malformed_real_record(self):
        """A real record mutated past its own contract never reaches a constructor."""
        cases = (
            (
                "lateral",
                lambda cut: (
                    bypassed(cut.steering, steering=0.99),
                    cut.longitudinal,
                ),
                CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE,
                ("control.lateral", "control.longitudinal"),
            ),
            (
                "longitudinal",
                lambda cut: (
                    cut.steering,
                    bypassed(
                        cut.longitudinal,
                        throttle_request=0.20,
                        brake_request=0.20,
                    ),
                ),
                CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE,
                ("control.longitudinal",),
            ),
        )
        for axis, mutate, reason, silent in cases:
            with self.subTest(axis=axis):
                parts, cut = self.warmed()
                envelope = parts[4]
                before = envelope.metrics
                lat, lon = mutate(cut)
                counts, restore = self.instrument()
                try:
                    request = envelope.update(lat, lon, now=cut.now)
                finally:
                    for module, name, original in restore:
                        setattr(module, name, original)
                self.assertIs(request.reason, reason)
                self.assertFalse(request.command_allowed)
                self.assertEqual(
                    envelope.metrics.internal_errors, before.internal_errors
                )
                for key, value in counts.items():
                    if key.rsplit(".", 1)[0] in silent:
                        self.assertEqual(value, 0, key)

    def test_a_real_upstream_fault_and_a_real_malformed_record_differ(self):
        import control.lateral as lateral_module

        parts, cut = self.warmed()
        envelope = parts[4]
        malformed = envelope.update(
            bypassed(cut.steering, steering=0.99), cut.longitudinal, now=cut.now
        )
        self.assertIs(
            malformed.reason, CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE
        )
        self.assertEqual(envelope.metrics.internal_errors, 0)

        faulty_parts, faulty_cut = self.warmed()
        faulty = faulty_parts[4]
        before = faulty.metrics
        exception = ValueError("upstream defect")
        original = lateral_module._finite_real

        def boom(*args, **kwargs):
            raise exception

        lateral_module._finite_real = boom
        try:
            with self.assertRaises(ValueError) as caught:
                faulty.update(
                    faulty_cut.steering, faulty_cut.longitudinal, now=faulty_cut.now
                )
        finally:
            lateral_module._finite_real = original
        self.assertIs(caught.exception, exception)
        self.assertIs(
            faulty.latest_request.reason, CommandEnvelopeReason.INTERNAL_ERROR
        )
        self.assertEqual(
            faulty.metrics.internal_errors, before.internal_errors + 1
        )
        for request in (malformed, faulty.latest_request):
            self.assertIs(request.mode, CommandEnvelopeMode.DISABLED)
            self.assertFalse(request.command_allowed)
            self.assertEqual(request.throttle_authority, 0.0)
            self.assertEqual(request.brake_authority, 0.0)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    unittest.main()
