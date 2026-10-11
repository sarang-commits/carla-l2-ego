"""Offline integration tests for the Mission 8 safety supervisor.

These tests drive the accepted Mission 6A runtime, Mission 6B bridge, and
Mission 7 temporal pipeline for real, injecting only a Mission 5-shaped
perception callable and explicit clocks.  Every input is an immutable synthetic
telemetry record; CARLA is neither imported nor contacted.
"""

from __future__ import annotations

import threading
import unittest

from perception.live_bridge import PerceptionLiveBridge, PerceptionLiveBridgeState
from perception.runtime import (
    PerceptionRuntime,
    PerceptionRuntimeConfig,
    PerceptionRuntimeState,
)
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
    TemporalLaneTrackerConfig,
)
from safety.supervisor import (
    ComponentHealth,
    SafetyReason,
    SafetyState,
    SafetySupervisor,
    SafetySupervisorConfig,
)
from telemetry.state import (
    PIXEL_FORMAT_RGB8,
    CameraFrame,
    CameraFrameMetadata,
    SampleStamp,
    TelemetrySnapshot,
)


WAIT_SECONDS = 5.0


class ConstantClock:
    """Thread-safe constant clock for zero-duration runtime processing."""

    def __init__(self, value: float = 100.0) -> None:
        self.value = float(value)
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            self.calls += 1
            return self.value


def snapshot(
    frame: int,
    *,
    sim_time: float | None = None,
    receive_monotonic: float | None = None,
    revision: int | None = None,
    with_camera: bool = True,
) -> TelemetrySnapshot:
    selected_sim_time = frame / 10.0 if sim_time is None else sim_time
    selected_receive = (
        selected_sim_time if receive_monotonic is None else receive_monotonic
    )
    if not with_camera:
        return TelemetrySnapshot(
            revision=frame if revision is None else revision,
            captured_monotonic=selected_receive,
        )
    stamp = SampleStamp(
        frame=frame,
        sim_time=selected_sim_time,
        monotonic=selected_receive,
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
        revision=frame if revision is None else revision,
        captured_monotonic=selected_receive,
        camera=CameraFrame(metadata=metadata, rgb_bytes=b"\x10\x20\x30"),
    )


def boundary(
    side: LaneBoundarySide, *, confidence: float = 0.8
) -> LaneBoundaryObservation:
    if side is LaneBoundarySide.LEFT:
        lower_x, upper_x, slope = 0.24, 0.34, -0.20
        intercept = upper_x + 0.10
    else:
        lower_x, upper_x, slope = 0.76, 0.66, 0.20
        intercept = upper_x - 0.10
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
    confidence: float = 0.8,
) -> LaneObservation:
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
        return LaneObservation(
            **common,
            reason=LaneObservationReason.DETECTED,
            left_boundary=boundary(LaneBoundarySide.LEFT, confidence=confidence),
            right_boundary=boundary(LaneBoundarySide.RIGHT, confidence=confidence),
            centerline_endpoints=(
                NormalizedImagePoint(0.50, 0.50),
                NormalizedImagePoint(0.50, 1.00),
            ),
            center_offset=0.0,
            heading_proxy=0.0,
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

    def __init__(self, states: dict[int, LaneDetectionState], *, raising=()) -> None:
        self._states = dict(states)
        self._raising = frozenset(raising)
        self._lock = threading.Lock()
        self.calls: list[int] = []

    def __call__(self, value: TelemetrySnapshot) -> LaneObservation:
        assert value.camera is not None
        frame = value.camera.metadata.stamp.frame
        with self._lock:
            self.calls.append(frame)
        if frame in self._raising:
            raise RuntimeError(f"synthetic perception failure on frame {frame}")
        return lane_observation(value, self._states[frame])


class BlockingFirstPerception(ScriptedPerception):
    def __init__(self, states, **kwargs) -> None:
        super().__init__(states, **kwargs)
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, value: TelemetrySnapshot) -> LaneObservation:
        assert value.camera is not None
        if value.camera.metadata.stamp.frame == 1:
            self.entered.set()
            if not self.release.wait(WAIT_SECONDS):
                raise RuntimeError("test did not release the first M5 call")
        return super().__call__(value)


class SafetyIntegrationTestCase(unittest.TestCase):
    """Shared runtime lifecycle helpers for the Mission 8 integration suite."""

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
        self, runtime: PerceptionRuntime, value: TelemetrySnapshot, count: int
    ):
        self.assertTrue(runtime.submit(value))
        self.wait_for_processed(runtime, count)
        self.assertIsNot(runtime.state, PerceptionRuntimeState.FAULTED, runtime.fault)
        published = runtime.latest_result
        self.assertIsNotNone(published)
        assert published is not None
        self.assertIsInstance(published.observation, TemporalLaneObservation)
        return published

    @staticmethod
    def stop_runtime(runtime: PerceptionRuntime) -> None:
        if runtime.state is not PerceptionRuntimeState.STOPPED:
            runtime.stop()

    @staticmethod
    def health(runtime: PerceptionRuntime, bridge=None) -> ComponentHealth:
        return ComponentHealth.from_components(runtime, bridge)


class SafetyUnknownHealthIntegrationTest(SafetyIntegrationTestCase):
    """A real, healthy pipeline still cannot permit autonomy unsupervised.

    Component health is evidence the caller must supply.  Omitting it means
    "no new sample", so an epoch that has never seen one holds health unknown.
    """

    def test_healthy_pipeline_without_health_evidence_is_refused(self):
        perception = ScriptedPerception(
            {frame: LaneDetectionState.DETECTED for frame in range(1, 6)}
        )
        pipeline = TemporalLanePipeline(perception_callable=perception)
        runtime = PerceptionRuntime(clock=ConstantClock(), perception_callable=pipeline)
        supervisor = SafetySupervisor()
        try:
            runtime.start()
            for frame in range(1, 6):
                published = self.publish(
                    runtime,
                    snapshot(frame, sim_time=frame / 10.0, receive_monotonic=100.0),
                    frame,
                )
                decision = supervisor.update(published, now=100.0)
                self.assertIs(decision.state, SafetyState.FAIL_SAFE, frame)
                self.assertIs(
                    decision.reason, SafetyReason.COMPONENT_NOT_READY, frame
                )
                self.assertFalse(decision.autonomy_allowed, frame)
                self.assertFalse(decision.latched, frame)
            # The very same evidence is permitted once health is actually
            # observed, so the refusal is the health gate and nothing else.
            self.assertIs(runtime.state, PerceptionRuntimeState.RUNNING)
            supervisor.reset()
            allowed = supervisor.update(
                runtime.latest_result, now=100.0, health=self.health(runtime)
            )
            self.assertIs(allowed.state, SafetyState.NOMINAL)
            self.assertTrue(allowed.autonomy_allowed)
        finally:
            self.stop_runtime(runtime)

    def test_faulted_then_restarted_epoch_without_health_never_permits(self):
        # Mission 6 keeps publishing strictly newer CARLA frames after a
        # restart inside one session.  A caller that never supplies health must
        # not silently regain autonomy across that fault.
        perception = ScriptedPerception(
            {1: LaneDetectionState.DETECTED, 2: LaneDetectionState.DETECTED,
             10: LaneDetectionState.DETECTED, 11: LaneDetectionState.DETECTED},
            raising=(2,),
        )
        pipeline = TemporalLanePipeline(perception_callable=perception)
        runtime = PerceptionRuntime(clock=ConstantClock(), perception_callable=pipeline)
        supervisor = SafetySupervisor()
        try:
            runtime.start()
            published = self.publish(
                runtime, snapshot(1, sim_time=0.1, receive_monotonic=100.0), 1
            )
            self.assertFalse(supervisor.update(published, now=100.0).autonomy_allowed)

            self.assertTrue(runtime.submit(snapshot(2, receive_monotonic=100.0)))
            with runtime._condition:
                faulted = runtime._condition.wait_for(
                    lambda: runtime._state is PerceptionRuntimeState.FAULTED,
                    timeout=WAIT_SECONDS,
                )
            self.assertTrue(faulted, "perception failure did not fault the runtime")
            self.assertFalse(supervisor.evaluate(now=100.0).autonomy_allowed)

            # Documented lifecycle, except that health is still never supplied.
            runtime.stop()
            pipeline.reset()
            runtime.start()
            # The restarted epoch resets the runtime's processed counter.
            for processed, frame in enumerate((10, 11), start=1):
                restarted = self.publish(
                    runtime,
                    snapshot(frame, sim_time=frame / 10.0, receive_monotonic=100.0),
                    processed,
                )
                decision = supervisor.update(restarted, now=100.0)
                self.assertIs(
                    decision.reason, SafetyReason.COMPONENT_NOT_READY, frame
                )
                self.assertFalse(decision.autonomy_allowed, frame)
            self.assertEqual(supervisor.metrics.nominal_decisions, 0)
            self.assertEqual(supervisor.metrics.degraded_decisions, 0)
        finally:
            self.stop_runtime(runtime)


class SafetyPipelineIntegrationTest(SafetyIntegrationTestCase):
    def drive(
        self,
        states: dict[int, LaneDetectionState],
        frames,
        *,
        tracker_config: TemporalLaneTrackerConfig | None = None,
        safety_config: SafetySupervisorConfig | None = None,
    ):
        perception = ScriptedPerception(states)
        pipeline = (
            TemporalLanePipeline(perception_callable=perception)
            if tracker_config is None
            else TemporalLanePipeline(tracker_config, perception_callable=perception)
        )
        runtime = PerceptionRuntime(clock=ConstantClock(), perception_callable=pipeline)
        supervisor = SafetySupervisor(
            SafetySupervisorConfig() if safety_config is None else safety_config
        )
        decisions = []
        try:
            runtime.start()
            for count, frame in enumerate(frames, 1):
                published = self.publish(runtime, snapshot(frame), count)
                decisions.append(
                    supervisor.update(
                        published,
                        now=published.source_stamp.monotonic,
                        health=self.health(runtime),
                    )
                )
            runtime.stop()
        finally:
            self.stop_runtime(runtime)
        return supervisor, decisions

    def test_runtime_temporal_result_reaches_nominal(self):
        supervisor, decisions = self.drive(
            {1: LaneDetectionState.DETECTED, 2: LaneDetectionState.DETECTED},
            (1, 2),
        )
        for decision in decisions:
            self.assertIs(decision.state, SafetyState.NOMINAL)
            self.assertIs(decision.reason, SafetyReason.NOMINAL_TRACKING)
            self.assertTrue(decision.autonomy_allowed)
            self.assertIs(decision.temporal_state, TemporalLaneState.TRACKING)
        self.assertEqual(decisions[-1].source_stamp.frame, 2)
        self.assertEqual(decisions[-1].result_age_seconds, 0.0)
        metrics = supervisor.metrics
        self.assertEqual(metrics.new_results_accepted, 2)
        self.assertEqual(metrics.identity_mismatch_trips, 0)
        self.assertEqual(metrics.malformed_evidence_trips, 0)

    def test_partial_temporal_evidence_is_degraded(self):
        _supervisor, decisions = self.drive(
            {1: LaneDetectionState.DETECTED, 2: LaneDetectionState.PARTIAL},
            (1, 2),
        )
        self.assertIs(decisions[0].state, SafetyState.NOMINAL)
        self.assertIs(decisions[1].state, SafetyState.DEGRADED)
        self.assertIs(decisions[1].reason, SafetyReason.DEGRADED_PARTIAL_TRACKING)
        self.assertTrue(decisions[1].autonomy_allowed)
        self.assertIs(decisions[1].temporal_state, TemporalLaneState.TRACKING)

    def test_brief_temporal_coasting_is_degraded(self):
        _supervisor, decisions = self.drive(
            {1: LaneDetectionState.DETECTED, 2: LaneDetectionState.NOT_DETECTED},
            (1, 2),
        )
        self.assertIs(decisions[0].state, SafetyState.NOMINAL)
        self.assertIs(decisions[1].state, SafetyState.DEGRADED)
        self.assertIs(decisions[1].reason, SafetyReason.DEGRADED_COASTING)
        self.assertTrue(decisions[1].autonomy_allowed)
        self.assertIs(decisions[1].temporal_state, TemporalLaneState.COASTING)
        self.assertLess(decisions[1].temporal_confidence, decisions[0].temporal_confidence)

    def test_temporal_lost_fails_closed(self):
        _supervisor, decisions = self.drive(
            {
                1: LaneDetectionState.DETECTED,
                2: LaneDetectionState.NOT_DETECTED,
                3: LaneDetectionState.NOT_DETECTED,
            },
            (1, 2, 3),
            tracker_config=TemporalLaneTrackerConfig(max_consecutive_misses=1),
        )
        self.assertIs(decisions[1].state, SafetyState.DEGRADED)
        self.assertIs(decisions[2].state, SafetyState.FAIL_SAFE)
        self.assertIs(decisions[2].reason, SafetyReason.TEMPORAL_LOST)
        self.assertFalse(decisions[2].autonomy_allowed)
        self.assertFalse(decisions[2].latched)

    def test_temporal_input_unusable_fails_closed(self):
        _supervisor, decisions = self.drive(
            {1: LaneDetectionState.DETECTED, 2: LaneDetectionState.INPUT_UNUSABLE},
            (1, 2),
        )
        self.assertIs(decisions[1].state, SafetyState.FAIL_SAFE)
        self.assertIs(decisions[1].reason, SafetyReason.TEMPORAL_INPUT_UNUSABLE)
        self.assertFalse(decisions[1].autonomy_allowed)
        self.assertFalse(decisions[1].latched)

    def test_recovery_requires_the_configured_healthy_streak(self):
        states = {
            1: LaneDetectionState.DETECTED,
            2: LaneDetectionState.NOT_DETECTED,
            3: LaneDetectionState.DETECTED,
            4: LaneDetectionState.DETECTED,
            5: LaneDetectionState.DETECTED,
        }
        supervisor, decisions = self.drive(
            states,
            (1, 2, 3, 4, 5),
            tracker_config=TemporalLaneTrackerConfig(max_consecutive_misses=0),
            safety_config=SafetySupervisorConfig(required_healthy_results=2),
        )
        self.assertEqual(
            [decision.state for decision in decisions],
            [
                SafetyState.NOMINAL,
                SafetyState.FAIL_SAFE,
                SafetyState.RECOVERING,
                SafetyState.NOMINAL,
                SafetyState.NOMINAL,
            ],
        )
        self.assertIs(decisions[1].reason, SafetyReason.TEMPORAL_LOST)
        self.assertEqual(decisions[2].recovery_streak, 1)
        self.assertFalse(decisions[2].autonomy_allowed)
        self.assertTrue(decisions[3].autonomy_allowed)
        self.assertEqual(supervisor.metrics.recovery_sequences, 1)


class SafetyFreshnessIntegrationTest(SafetyIntegrationTestCase):
    def test_no_new_runtime_result_expires_safety(self):
        perception = ScriptedPerception({1: LaneDetectionState.DETECTED})
        pipeline = TemporalLanePipeline(perception_callable=perception)
        runtime = PerceptionRuntime(clock=ConstantClock(), perception_callable=pipeline)
        supervisor = SafetySupervisor()
        try:
            runtime.start()
            published = self.publish(runtime, snapshot(1, receive_monotonic=0.0), 1)
            health = self.health(runtime)
            first = supervisor.update(published, now=0.0, health=health)
            self.assertTrue(first.autonomy_allowed)

            # Perception publishes nothing further.  Mission 7 cannot age its
            # own track without a newer observation, so Mission 8 must.
            before = pipeline.latest_estimate
            fresh = supervisor.evaluate(now=0.30, health=health)
            stale = supervisor.evaluate(now=0.40, health=health)
            later = supervisor.evaluate(now=5.00, health=health)
            runtime.stop()

            self.assertIs(fresh.state, SafetyState.NOMINAL)
            self.assertIs(stale.state, SafetyState.FAIL_SAFE)
            self.assertIs(stale.reason, SafetyReason.RESULT_STALE)
            self.assertFalse(stale.autonomy_allowed)
            self.assertIs(later.reason, SafetyReason.RESULT_STALE)
            self.assertFalse(later.autonomy_allowed)
            # Mission 7 state is untouched: the expiry is entirely Mission 8's.
            self.assertIs(pipeline.latest_estimate, before)
            self.assertIs(before.state, TemporalLaneState.TRACKING)
            self.assertEqual(pipeline.metrics.observations_received, 1)
            self.assertEqual(supervisor.metrics.stale_expirations, 1)
            self.assertEqual(supervisor.metrics.updates_received, 1)
        finally:
            self.stop_runtime(runtime)

    def test_runtime_fault_after_last_good_publication_fails_closed(self):
        perception = ScriptedPerception(
            {1: LaneDetectionState.DETECTED, 2: LaneDetectionState.DETECTED},
            raising=(2,),
        )
        pipeline = TemporalLanePipeline(perception_callable=perception)
        runtime = PerceptionRuntime(clock=ConstantClock(), perception_callable=pipeline)
        supervisor = SafetySupervisor()
        try:
            runtime.start()
            published = self.publish(runtime, snapshot(1, receive_monotonic=0.0), 1)
            good = supervisor.update(published, now=0.0, health=self.health(runtime))
            self.assertTrue(good.autonomy_allowed)

            self.assertTrue(runtime.submit(snapshot(2, receive_monotonic=0.1)))
            with runtime._condition:
                faulted = runtime._condition.wait_for(
                    lambda: runtime._state is PerceptionRuntimeState.FAULTED,
                    timeout=WAIT_SECONDS,
                )
            self.assertTrue(faulted, "perception failure did not fault the runtime")
            self.assertIsNotNone(runtime.fault)
            # The last published result is still well inside the freshness bound.
            self.assertIs(runtime.latest_result, published)

            decision = supervisor.evaluate(now=0.05, health=self.health(runtime))
            self.assertIs(decision.state, SafetyState.FAIL_SAFE)
            self.assertIs(decision.reason, SafetyReason.RUNTIME_FAULT)
            self.assertTrue(decision.latched)
            self.assertFalse(decision.autonomy_allowed)
            self.assertLess(decision.result_age_seconds, 0.35)

            runtime.stop()
            # A later good-looking publication cannot clear the hard latch.
            replay = supervisor.update(published, now=0.06)
            self.assertIs(replay.reason, SafetyReason.RUNTIME_FAULT)
            self.assertTrue(replay.latched)
            self.assertEqual(supervisor.metrics.component_fault_trips, 1)
        finally:
            self.stop_runtime(runtime)

    def test_bridge_fault_fails_closed(self):
        perception = ScriptedPerception({1: LaneDetectionState.DETECTED})
        pipeline = TemporalLanePipeline(perception_callable=perception)
        clock = ConstantClock()
        runtime = PerceptionRuntime(clock=clock, perception_callable=pipeline)
        bridge = PerceptionLiveBridge(runtime, clock=clock)
        supervisor = SafetySupervisor()
        try:
            runtime.start()
            self.assertTrue(bridge.submit(snapshot(1, receive_monotonic=0.0)))
            self.wait_for_processed(runtime, 1)
            published = runtime.latest_result
            assert published is not None
            healthy = supervisor.update(
                published, now=0.0, health=self.health(runtime, bridge)
            )
            self.assertTrue(healthy.autonomy_allowed)

            # A structurally invalid snapshot fail-closes the real bridge.
            self.assertFalse(bridge.submit(snapshot(2, with_camera=False)))
            self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)
            self.assertIsNotNone(bridge.fault)

            decision = supervisor.evaluate(
                now=0.05, health=self.health(runtime, bridge)
            )
            self.assertIs(decision.reason, SafetyReason.BRIDGE_FAULT)
            self.assertTrue(decision.latched)
            self.assertFalse(decision.autonomy_allowed)
            self.assertEqual(supervisor.metrics.component_fault_trips, 1)
        finally:
            bridge.close(0.0)
            self.stop_runtime(runtime)


class SafetyOrderingIntegrationTest(SafetyIntegrationTestCase):
    def test_coalesced_frames_create_no_phantom_safety_updates(self):
        perception = BlockingFirstPerception(
            {
                1: LaneDetectionState.DETECTED,
                5: LaneDetectionState.DETECTED,
            }
        )
        pipeline = TemporalLanePipeline(perception_callable=perception)
        runtime = PerceptionRuntime(clock=ConstantClock(), perception_callable=pipeline)
        supervisor = SafetySupervisor()
        seen = []
        try:
            runtime.start()
            self.assertTrue(runtime.submit(snapshot(1, receive_monotonic=0.1)))
            self.assertTrue(perception.entered.wait(WAIT_SECONDS))
            for frame in (2, 3, 4, 5):
                self.assertTrue(
                    runtime.submit(snapshot(frame, receive_monotonic=frame / 10.0))
                )
            self.assertEqual(runtime.metrics.pending_frames_coalesced, 3)

            perception.release.set()
            self.wait_for_processed(runtime, 2)

            # The supervisor only ever sees genuinely published results.
            self.assertEqual(perception.calls, [1, 5])
            published = runtime.latest_result
            assert published is not None
            seen.append(
                supervisor.update(published, now=0.5, health=self.health(runtime))
            )
            # Re-offering the same published result is an idempotent duplicate.
            seen.append(supervisor.update(published, now=0.5))
            runtime.stop()
            self.assertIs(seen[0].state, SafetyState.NOMINAL)
            self.assertIs(seen[1].state, SafetyState.NOMINAL)
            self.assertEqual(seen[1].source_stamp.frame, 5)
            metrics = supervisor.metrics
            self.assertEqual(metrics.new_results_accepted, 1)
            self.assertEqual(metrics.duplicates_rejected, 1)
            self.assertEqual(metrics.out_of_order_rejected, 0)
            self.assertEqual(metrics.identity_mismatch_trips, 0)
        finally:
            perception.release.set()
            self.stop_runtime(runtime)

    def test_cross_epoch_cached_carrier_is_caught_by_identity_gate(self):
        """Reproduce the accepted Mission 7 reuse limitation end to end."""
        perception = ScriptedPerception(
            {
                1: LaneDetectionState.DETECTED,
                10: LaneDetectionState.DETECTED,
                11: LaneDetectionState.DETECTED,
            }
        )
        pipeline = TemporalLanePipeline(perception_callable=perception)
        runtime = PerceptionRuntime(clock=ConstantClock(), perception_callable=pipeline)
        try:
            runtime.start()
            self.publish(runtime, snapshot(10, sim_time=1.0, receive_monotonic=1.0), 1)
            epoch_one = self.publish(
                runtime, snapshot(11, sim_time=1.1, receive_monotonic=1.1), 2
            )
            self.assertEqual(epoch_one.observation.source_frame, 11)
            runtime.stop()

            # The forbidden lifecycle: a new runtime epoch reusing the same
            # Mission 7 pipeline without the required reset.
            runtime.start()
            republished = self.publish(
                runtime, snapshot(1, sim_time=0.1, receive_monotonic=2.0), 1
            )
            runtime.stop()

            # Mission 7 rejected the restarted frame as out of order and
            # replayed its cached prior-epoch carrier under a new outer identity.
            self.assertEqual(republished.source_stamp.frame, 1)
            self.assertIs(republished.observation, epoch_one.observation)
            self.assertEqual(republished.observation.source_frame, 11)
            self.assertEqual(pipeline.metrics.out_of_order_rejected, 1)

            # A brand new supervisor epoch still refuses the incoherent pair,
            # so this cannot be mistaken for an ordering rejection.
            supervisor = SafetySupervisor()
            decision = supervisor.update(
                republished, now=2.0, health=self.health(runtime)
            )
            self.assertIs(decision.state, SafetyState.FAIL_SAFE)
            self.assertIs(decision.reason, SafetyReason.IDENTITY_MISMATCH)
            self.assertTrue(decision.latched)
            self.assertFalse(decision.autonomy_allowed)
            self.assertEqual(supervisor.metrics.identity_mismatch_trips, 1)
            self.assertEqual(supervisor.metrics.new_results_accepted, 0)
            self.assertEqual(supervisor.metrics.out_of_order_rejected, 0)
        finally:
            self.stop_runtime(runtime)

    def test_reset_epoch_restores_clean_startup(self):
        perception = ScriptedPerception(
            {1: LaneDetectionState.DETECTED, 10: LaneDetectionState.DETECTED}
        )
        pipeline = TemporalLanePipeline(perception_callable=perception)
        runtime = PerceptionRuntime(clock=ConstantClock(), perception_callable=pipeline)
        supervisor = SafetySupervisor()
        try:
            runtime.start()
            published = self.publish(
                runtime, snapshot(10, sim_time=1.0, receive_monotonic=1.0), 1
            )
            self.assertTrue(
                supervisor.update(
                    published, now=1.0, health=self.health(runtime)
                ).autonomy_allowed
            )
            runtime.stop()

            # The documented lifecycle: new runtime epoch, fresh temporal
            # pipeline, fresh safety supervisor.
            pipeline.reset()
            cleared = supervisor.reset()
            self.assertIs(cleared.state, SafetyState.INITIALIZING)
            self.assertFalse(cleared.autonomy_allowed)
            self.assertIsNone(supervisor.metrics.last_accepted_source_stamp)

            runtime.start()
            restarted = self.publish(
                runtime, snapshot(1, sim_time=0.1, receive_monotonic=2.0), 1
            )
            decision = supervisor.update(
                restarted, now=2.0, health=self.health(runtime)
            )
            runtime.stop()

            self.assertEqual(restarted.observation.source_frame, 1)
            self.assertIs(decision.state, SafetyState.NOMINAL)
            self.assertTrue(decision.autonomy_allowed)
            self.assertEqual(decision.source_stamp.frame, 1)
            self.assertEqual(supervisor.metrics.identity_mismatch_trips, 0)
            self.assertEqual(supervisor.metrics.out_of_order_rejected, 0)
        finally:
            self.stop_runtime(runtime)

    def test_supervision_adds_no_runtime_or_temporal_resource(self):
        threads_before = set(threading.enumerate())
        perception = ScriptedPerception(
            {frame: LaneDetectionState.DETECTED for frame in range(1, 6)}
        )
        pipeline = TemporalLanePipeline(perception_callable=perception)
        runtime = PerceptionRuntime(
            PerceptionRuntimeConfig(),
            clock=ConstantClock(),
            perception_callable=pipeline,
        )
        supervisor = SafetySupervisor()
        # Constructing the supervisor and its inputs starts nothing.
        self.assertEqual(set(threading.enumerate()), threads_before)
        try:
            runtime.start()
            worker = runtime._worker
            self.assertEqual(
                set(threading.enumerate()) - threads_before, {worker}
            )
            for count, frame in enumerate(range(1, 6), 1):
                published = self.publish(runtime, snapshot(frame), count)
                supervisor.update(
                    published,
                    now=published.source_stamp.monotonic,
                    health=self.health(runtime),
                )
                supervisor.evaluate(now=published.source_stamp.monotonic)
            runtime.stop()

            self.assertEqual(set(threading.enumerate()), threads_before)
            self.assertEqual(runtime.pending_capacity, 1)
            runtime_metrics = runtime.metrics
            self.assertEqual(runtime_metrics.inputs_received, 5)
            self.assertEqual(runtime_metrics.inputs_accepted, 5)
            self.assertEqual(runtime_metrics.inputs_processed, 5)
            self.assertEqual(runtime_metrics.perception_successes, 5)
            self.assertEqual(runtime_metrics.perception_failures, 0)
            self.assertEqual(runtime_metrics.duplicate_inputs_rejected, 0)
            self.assertEqual(runtime_metrics.out_of_order_inputs_rejected, 0)
            pipeline_metrics = pipeline.metrics
            self.assertEqual(pipeline_metrics.observations_received, 5)
            self.assertEqual(pipeline_metrics.observations_accepted, 5)
            self.assertEqual(pipeline_metrics.usable_updates, 5)
            self.assertEqual(pipeline_metrics.duplicates_rejected, 0)
            self.assertEqual(pipeline_metrics.out_of_order_rejected, 0)
            self.assertEqual(supervisor.metrics.new_results_accepted, 5)
            self.assertEqual(supervisor.metrics.evaluations, 10)
        finally:
            self.stop_runtime(runtime)


if __name__ == "__main__":
    unittest.main()
