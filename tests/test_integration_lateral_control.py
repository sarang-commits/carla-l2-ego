"""Offline integration tests for the Mission 9 lateral controller.

These tests drive the accepted Mission 6A runtime, Mission 7 temporal pipeline,
and Mission 8 safety supervisor for real, injecting only a Mission 5-shaped
perception callable and explicit clocks.  Every input is an immutable synthetic
telemetry record; CARLA is neither imported nor contacted, and no steering value
produced here reaches a vehicle.
"""

from __future__ import annotations

import threading
import unittest

from control.lateral import (
    LateralControlMode,
    LateralControlReason,
    LateralController,
    LateralControllerConfig,
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
    SampleStamp,
    TelemetrySnapshot,
)


WAIT_SECONDS = 5.0
BASE_MONOTONIC = 100.0

# One control step per published frame, fast enough that Mission 8 never calls
# the evidence stale and slow enough that the Mission 9 slew limit is visible.
STEP_SECONDS = 0.10


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


def snapshot(frame: int, *, monotonic: float | None = None) -> TelemetrySnapshot:
    stamp = SampleStamp(
        frame=frame,
        sim_time=frame * STEP_SECONDS,
        monotonic=control_time(frame) if monotonic is None else monotonic,
    )
    metadata = CameraFrameMetadata(
        stamp=stamp,
        width_px=1,
        height_px=1,
        horizontal_fov_deg=90.0,
        pixel_format=PIXEL_FORMAT_RGB8,
        row_stride_bytes=3,
    )
    return TelemetrySnapshot(
        revision=frame,
        captured_monotonic=stamp.monotonic,
        camera=CameraFrame(metadata=metadata, rgb_bytes=b"\x10\x20\x30"),
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
        state, offset, heading = self._script[frame]
        return lane_observation(value, state, offset=offset, heading=heading)


def detected(offset: float = 0.0, heading: float = 0.0) -> tuple:
    return LaneDetectionState.DETECTED, offset, heading


MISSING = (LaneDetectionState.NOT_DETECTED, 0.0, 0.0)
PARTIAL = (LaneDetectionState.PARTIAL, 0.0, 0.0)
UNUSABLE = (LaneDetectionState.INPUT_UNUSABLE, 0.0, 0.0)


class LateralIntegrationTestCase(unittest.TestCase):
    """Shared real-pipeline lifecycle helpers for the Mission 9 suite."""

    def build(self, script: dict[int, tuple], *, config=None):
        pipeline = TemporalLanePipeline(perception_callable=ScriptedPerception(script))
        runtime = PerceptionRuntime(
            clock=ConstantClock(), perception_callable=pipeline
        )
        supervisor = SafetySupervisor()
        controller = LateralController(
            LateralControllerConfig() if config is None else config
        )
        self.addCleanup(self.stop_runtime, runtime)
        runtime.start()
        return runtime, pipeline, supervisor, controller

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

    def publish(
        self,
        runtime: PerceptionRuntime,
        frame: int,
        *,
        monotonic: float | None = None,
        processed: int | None = None,
    ):
        self.assertTrue(runtime.submit(snapshot(frame, monotonic=monotonic)))
        self.wait_for_processed(runtime, frame if processed is None else processed)
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
        controller,
        frame: int,
        *,
        monotonic: float | None = None,
        processed: int | None = None,
    ):
        """Publish, supervise, and control one frame through the real stack."""
        published = self.publish(
            runtime, frame, monotonic=monotonic, processed=processed
        )
        now = control_time(frame) if monotonic is None else monotonic
        verdict = supervisor.update(
            published,
            now=now,
            health=ComponentHealth.from_components(runtime),
        )
        request = controller.update(published, verdict, now=now)
        return published, verdict, request


class RealNominalTrackingTest(LateralIntegrationTestCase):
    def test_a_real_nominal_pipeline_produces_nominal_steering(self):
        runtime, _, supervisor, controller = self.build(
            {frame: detected(0.20) for frame in range(1, 4)}
        )
        for frame in range(1, 4):
            published, verdict, request = self.step(
                runtime, supervisor, controller, frame
            )
            self.assertIs(
                published.observation.temporal_estimate.state,
                TemporalLaneState.TRACKING,
            )
            self.assertIs(verdict.state, SafetyState.NOMINAL)
            self.assertIs(verdict.reason, SafetyReason.NOMINAL_TRACKING)
            self.assertTrue(request.lateral_allowed, frame)
            self.assertIs(request.mode, LateralControlMode.NOMINAL, frame)
            self.assertIs(request.reason, LateralControlReason.NOMINAL_TRACKING)
            self.assertGreater(request.steering, 0.0, frame)
            self.assertEqual(request.source_frame, frame)

    def test_a_real_centered_lane_requests_near_zero_steering(self):
        runtime, _, supervisor, controller = self.build(
            {frame: detected(0.0, 0.0) for frame in range(1, 3)}
        )
        for frame in range(1, 3):
            _, verdict, request = self.step(runtime, supervisor, controller, frame)
            self.assertTrue(verdict.autonomy_allowed)
            self.assertTrue(request.lateral_allowed)
            self.assertAlmostEqual(request.steering, 0.0, places=12)

    def test_real_opposite_displacements_produce_opposite_steering(self):
        left_runtime, _, left_safety, left_control = self.build({1: detected(-0.20)})
        right_runtime, _, right_safety, right_control = self.build({1: detected(0.20)})
        _, _, left = self.step(left_runtime, left_safety, left_control, 1)
        _, _, right = self.step(right_runtime, right_safety, right_control, 1)
        self.assertLess(left.steering, 0.0)
        self.assertGreater(right.steering, 0.0)
        self.assertEqual(left.steering, -right.steering)

    def test_real_heading_evidence_moves_steering_the_expected_way(self):
        # heading_proxy < 0 means the image centreline moves rightward as rows
        # move up toward road further ahead, so the lane ahead lies right.
        ahead_right_runtime, _, safety_a, control_a = self.build(
            {1: detected(0.0, -0.20)}
        )
        ahead_left_runtime, _, safety_b, control_b = self.build(
            {1: detected(0.0, 0.20)}
        )
        _, _, ahead_right = self.step(ahead_right_runtime, safety_a, control_a, 1)
        _, _, ahead_left = self.step(ahead_left_runtime, safety_b, control_b, 1)
        self.assertGreater(ahead_right.steering, 0.0)
        self.assertLess(ahead_left.steering, 0.0)
        self.assertEqual(ahead_right.steering, -ahead_left.steering)


class RealDegradedAuthorityTest(LateralIntegrationTestCase):
    def test_real_coasting_degrades_authority_below_nominal(self):
        config = LateralControllerConfig(max_steering_rate_per_second=100.0)
        script = {1: detected(0.40), 2: detected(0.40), 3: MISSING}
        runtime, _, supervisor, controller = self.build(script, config=config)
        _, _, first = self.step(runtime, supervisor, controller, 1)
        self.step(runtime, supervisor, controller, 2)
        published, verdict, coasting = self.step(runtime, supervisor, controller, 3)

        estimate = published.observation.temporal_estimate
        self.assertIs(estimate.state, TemporalLaneState.COASTING)
        self.assertIs(published.observation.state, LaneDetectionState.NOT_DETECTED)
        # The raw Mission 5 frame carries no geometry at all this tick.
        self.assertIsNone(published.observation.center_offset)
        self.assertIs(verdict.state, SafetyState.DEGRADED)
        self.assertIs(verdict.reason, SafetyReason.DEGRADED_COASTING)

        self.assertTrue(coasting.lateral_allowed)
        self.assertIs(coasting.mode, LateralControlMode.DEGRADED)
        self.assertIs(coasting.reason, LateralControlReason.DEGRADED_COASTING)
        self.assertEqual(
            coasting.steering_authority, config.degraded_max_steering
        )
        self.assertEqual(first.steering_authority, config.nominal_max_steering)
        self.assertLess(abs(coasting.steering), abs(first.steering))
        # Coasting steered from retained temporal guidance, not from raw pixels.
        self.assertEqual(coasting.lateral_error, estimate.center_offset)
        self.assertGreater(coasting.steering, 0.0)

    def test_mission_8_permitting_partial_evidence_still_cannot_steer(self):
        """A degraded permission is not a licence to invent the missing side."""
        runtime, _, supervisor, controller = self.build({1: PARTIAL})
        published, verdict, request = self.step(runtime, supervisor, controller, 1)
        estimate = published.observation.temporal_estimate
        self.assertIs(estimate.state, TemporalLaneState.TRACKING)
        self.assertIsNone(estimate.center_offset)
        self.assertIsNone(estimate.right_boundary)
        # Mission 8 really does permit degraded autonomy here.
        self.assertIs(verdict.state, SafetyState.DEGRADED)
        self.assertIs(verdict.reason, SafetyReason.DEGRADED_PARTIAL_TRACKING)
        self.assertTrue(verdict.autonomy_allowed)
        # Mission 9 refuses anyway, because there is no lane centre to steer to.
        self.assertFalse(request.lateral_allowed)
        self.assertIs(request.reason, LateralControlReason.INSUFFICIENT_GEOMETRY)
        self.assertEqual(request.steering, 0.0)


class RealRefusalTest(LateralIntegrationTestCase):
    def test_real_temporal_loss_reaches_fail_safe_and_zero_steering(self):
        script = {1: detected(0.40), 2: MISSING, 3: MISSING, 4: MISSING, 5: MISSING}
        runtime, _, supervisor, controller = self.build(script)
        self.step(runtime, supervisor, controller, 1)
        states = []
        for frame in range(2, 6):
            published, verdict, request = self.step(
                runtime, supervisor, controller, frame
            )
            states.append(published.observation.temporal_estimate.state)
            if verdict.state is SafetyState.FAIL_SAFE:
                self.assertIs(verdict.reason, SafetyReason.TEMPORAL_LOST)
                self.assertFalse(request.lateral_allowed)
                self.assertEqual(request.steering, 0.0)
                self.assertIs(
                    request.reason, LateralControlReason.SAFETY_NOT_PERMITTED
                )
        self.assertIn(TemporalLaneState.LOST, states)

    def test_real_unusable_input_yields_zero_steering(self):
        script = {1: detected(0.40), 2: UNUSABLE}
        runtime, _, supervisor, controller = self.build(script)
        self.step(runtime, supervisor, controller, 1)
        published, verdict, request = self.step(runtime, supervisor, controller, 2)
        self.assertIs(
            published.observation.temporal_estimate.state,
            TemporalLaneState.INPUT_UNUSABLE,
        )
        self.assertIs(verdict.state, SafetyState.FAIL_SAFE)
        self.assertIs(verdict.reason, SafetyReason.TEMPORAL_INPUT_UNUSABLE)
        self.assertFalse(request.lateral_allowed)
        self.assertEqual(request.steering, 0.0)

    def test_real_recovering_state_yields_zero_steering(self):
        script = {
            1: detected(0.40),
            2: UNUSABLE,
            3: detected(0.40),
            4: detected(0.40),
        }
        runtime, _, supervisor, controller = self.build(script)
        for frame in (1, 2):
            self.step(runtime, supervisor, controller, frame)
        seen_recovering = False
        for frame in (3, 4):
            _, verdict, request = self.step(runtime, supervisor, controller, frame)
            if verdict.state is SafetyState.RECOVERING:
                seen_recovering = True
                self.assertIs(verdict.reason, SafetyReason.RECOVERY_PENDING)
                self.assertFalse(request.lateral_allowed)
                self.assertEqual(request.steering, 0.0)
                self.assertIs(
                    request.reason, LateralControlReason.SAFETY_NOT_PERMITTED
                )
        self.assertTrue(seen_recovering, "Mission 8 never entered recovery")

    def test_a_real_decision_for_an_older_frame_cannot_authorize_a_newer_one(self):
        script = {frame: detected(0.40) for frame in range(1, 4)}
        runtime, _, supervisor, controller = self.build(script)
        _, older_verdict, _ = self.step(runtime, supervisor, controller, 1)
        self.assertTrue(older_verdict.autonomy_allowed)
        newer = self.publish(runtime, 2)
        request = controller.update(newer, older_verdict, now=control_time(2))
        self.assertFalse(request.lateral_allowed)
        self.assertIs(request.reason, LateralControlReason.IDENTITY_MISMATCH)
        self.assertEqual(request.steering, 0.0)
        self.assertEqual(controller.metrics.identity_rejects, 1)


class RealEpochTest(LateralIntegrationTestCase):
    def test_a_new_epoch_removes_every_previous_command_and_slew_baseline(self):
        config = LateralControllerConfig(
            offset_gain=1.0,
            heading_gain=0.0,
            nominal_max_steering=1.0,
            max_steering_rate_per_second=1.0,
        )
        script = {frame: detected(0.40) for frame in range(1, 3)}
        runtime, _, supervisor, controller = self.build(script, config=config)
        for frame in (1, 2):
            _, _, request = self.step(runtime, supervisor, controller, frame)
        self.assertEqual(request.steering, 0.40)
        self.assertIsNotNone(controller.metrics.last_accepted_source_stamp)

        # A new runtime epoch recreates Mission 7, then resets Mission 8, then
        # Mission 9, in that order.
        runtime.stop()
        supervisor.reset()
        cleared = controller.reset()
        self.assertFalse(cleared.lateral_allowed)
        self.assertIs(cleared.reason, LateralControlReason.NOT_EVALUATED)
        self.assertIsNone(cleared.source_stamp)
        self.assertIsNone(controller.metrics.last_accepted_source_stamp)

        restarted = PerceptionRuntime(
            clock=ConstantClock(),
            perception_callable=TemporalLanePipeline(
                perception_callable=ScriptedPerception({1: detected(-0.40)})
            ),
        )
        self.addCleanup(self.stop_runtime, restarted)
        restarted.start()
        # Frame 1 is acceptable again, and the request jumps straight to the
        # mirrored target: had the previous +0.40 command or its slew baseline
        # survived, this 0.10 s step could have moved by at most 0.10.
        epoch_two_now = control_time(2) + 1.0
        _, verdict, resumed = self.step(
            restarted, supervisor, controller, 1, monotonic=epoch_two_now
        )
        self.assertTrue(verdict.autonomy_allowed)
        self.assertEqual(resumed.steering, -0.40)
        self.assertFalse(resumed.rate_limited)
        self.assertEqual(controller.metrics.last_accepted_source_stamp.frame, 1)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    unittest.main()
