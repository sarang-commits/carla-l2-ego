"""Offline integration tests for Mission 7 temporal lane tracking.

The tests in this module exercise the opt-in Mission 7 callable through the
accepted Mission 6A runtime and Mission 6B bridge.  Every input is an immutable
synthetic telemetry record; CARLA is neither imported nor contacted.
"""

from __future__ import annotations

import math
import threading
import unittest
from dataclasses import FrozenInstanceError, fields

from perception.live_bridge import PerceptionLiveBridge, PerceptionLiveBridgeState
from perception.runtime import (
    PENDING_CAPACITY,
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


class SequenceClock:
    """Return or raise scripted actions and expose deterministic call gates."""

    def __init__(self, *actions: object) -> None:
        self._actions = tuple(actions)
        self._events = tuple(threading.Event() for _action in actions)
        self._lock = threading.Lock()
        self.calls = 0

    def __call__(self):
        with self._lock:
            index = self.calls
            if index >= len(self._actions):
                raise AssertionError("runtime performed an unexpected clock read")
            action = self._actions[index]
            self.calls += 1
            event = self._events[index]
        event.set()
        if isinstance(action, BaseException):
            raise action
        return action

    def wait_for_call(self, call_number: int) -> bool:
        return self._events[call_number - 1].wait(WAIT_SECONDS)


def snapshot(
    frame: int,
    *,
    sim_time: float | None = None,
    receive_monotonic: float | None = None,
    revision: int | None = None,
) -> TelemetrySnapshot:
    selected_sim_time = frame / 10.0 if sim_time is None else sim_time
    selected_receive = (
        selected_sim_time
        if receive_monotonic is None
        else receive_monotonic
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
    side: LaneBoundarySide,
    *,
    offset: float = 0.0,
    confidence: float = 0.8,
) -> LaneBoundaryObservation:
    if side is LaneBoundarySide.LEFT:
        lower_x, upper_x, slope = 0.24 + offset, 0.34 + offset, -0.20
        intercept = upper_x + 0.10
    else:
        lower_x, upper_x, slope = 0.76 + offset, 0.66 + offset, 0.20
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
    stamped: bool = True,
    offset: float = 0.0,
    heading: float = 0.0,
    confidence: float = 0.8,
) -> LaneObservation:
    assert value.camera is not None
    stamp = value.camera.metadata.stamp
    source = dict(
        source_frame=stamp.frame if stamped else None,
        source_simulation_timestamp=stamp.sim_time if stamped else None,
        source_receive_monotonic=stamp.monotonic if stamped else None,
    )
    common = dict(snapshot_revision=value.revision, state=state, **source)
    if state is LaneDetectionState.DETECTED:
        return LaneObservation(
            **common,
            reason=LaneObservationReason.DETECTED,
            left_boundary=boundary(
                LaneBoundarySide.LEFT,
                offset=offset,
                confidence=confidence,
            ),
            right_boundary=boundary(
                LaneBoundarySide.RIGHT,
                offset=offset,
                confidence=confidence,
            ),
            centerline_endpoints=(
                NormalizedImagePoint(0.50 + offset, 0.50),
                NormalizedImagePoint(0.50 + offset, 1.00),
            ),
            center_offset=offset,
            heading_proxy=heading,
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


class RecordingPerception:
    """Injected M5-shaped callable retaining only bounded test evidence."""

    def __init__(self, builder) -> None:
        self._builder = builder
        self._lock = threading.Lock()
        self.calls: list[int] = []
        self.thread_names: list[str] = []
        self.observations: dict[int, LaneObservation] = {}

    def __call__(self, value: TelemetrySnapshot) -> LaneObservation:
        assert value.camera is not None
        frame = value.camera.metadata.stamp.frame
        result = self._builder(value)
        with self._lock:
            self.calls.append(frame)
            self.thread_names.append(threading.current_thread().name)
            self.observations[frame] = result
        return result


class BlockingFirstPerception(RecordingPerception):
    def __init__(self, builder) -> None:
        super().__init__(builder)
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, value: TelemetrySnapshot) -> LaneObservation:
        assert value.camera is not None
        if value.camera.metadata.stamp.frame == 1:
            self.entered.set()
            if not self.release.wait(WAIT_SECONDS):
                raise RuntimeError("test did not release the first M5 call")
        return super().__call__(value)


class TemporalRuntimeIntegrationTest(unittest.TestCase):
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
        self.assertIsNot(runtime.state, PerceptionRuntimeState.FAULTED, runtime.fault)
        self.assertGreaterEqual(runtime.metrics.inputs_processed, count)

    def submit_and_result(
        self,
        runtime: PerceptionRuntime,
        value: TelemetrySnapshot,
        processed_count: int,
    ):
        self.assertTrue(runtime.submit(value))
        self.wait_for_processed(runtime, processed_count)
        result = runtime.latest_result
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.source_stamp.frame, value.camera.metadata.stamp.frame)
        return result

    @staticmethod
    def stop_runtime(runtime: PerceptionRuntime) -> None:
        if runtime.state is not PerceptionRuntimeState.STOPPED:
            runtime.stop()

    def test_steady_stream_preserves_raw_carrier_and_atomic_temporal_results(self):
        offsets = {1: 0.05, 2: 0.10, 3: 0.15}
        perception = RecordingPerception(
            lambda value: lane_observation(
                value,
                LaneDetectionState.DETECTED,
                offset=offsets[value.camera.metadata.stamp.frame],
                heading=offsets[value.camera.metadata.stamp.frame] / 2.0,
            )
        )
        config = TemporalLaneTrackerConfig(smoothing_alpha=0.5)
        pipeline = TemporalLanePipeline(config, perception_callable=perception)
        runtime = PerceptionRuntime(
            clock=ConstantClock(), perception_callable=pipeline
        )
        results = []
        try:
            runtime.start()
            for count, frame in enumerate((1, 2, 3), 1):
                results.append(
                    self.submit_and_result(runtime, snapshot(frame), count)
                )
            runtime.stop()

            self.assertEqual(perception.calls, [1, 2, 3])
            self.assertEqual(perception.thread_names, ["PerceptionRuntime"] * 3)
            self.assertEqual(runtime.metrics.perception_successes, 3)
            self.assertEqual(runtime.metrics.perception_failures, 0)

            for frame, result in enumerate(results, 1):
                carrier = result.observation
                raw = perception.observations[frame]
                self.assertIsInstance(carrier, TemporalLaneObservation)
                self.assertIsInstance(carrier, LaneObservation)
                for raw_field in fields(LaneObservation):
                    self.assertEqual(
                        getattr(carrier, raw_field.name),
                        getattr(raw, raw_field.name),
                        raw_field.name,
                    )
                estimate = carrier.temporal_estimate
                self.assertEqual(estimate.source_stamp, result.source_stamp)
                self.assertEqual(estimate.snapshot_revision, result.snapshot_revision)
                self.assertIs(estimate.state, TemporalLaneState.TRACKING)
                self.assertIs(estimate.raw_state, LaneDetectionState.DETECTED)
                self.assertEqual(estimate.consecutive_misses, 0)
                self.assertFalse(estimate.extrapolated)

            self.assertIs(
                results[-1].observation.temporal_estimate,
                pipeline.latest_estimate,
            )
            self.assertEqual(
                [
                    result.observation.temporal_estimate.source_stamp.frame
                    for result in results
                ],
                [1, 2, 3],
            )
            # The raw current field remains unsmoothed; temporal geometry is
            # carried separately and does not rewrite Mission 5 evidence.
            self.assertEqual(results[-1].observation.center_offset, 0.15)
            self.assertAlmostEqual(pipeline.latest_estimate.center_offset, 0.1125)
            with self.assertRaises(FrozenInstanceError):
                results[-1].observation.confidence = 0.0
            self.assertFalse(hasattr(results[-1].observation, "__dict__"))
        finally:
            self.stop_runtime(runtime)

    def test_brief_dropout_expires_by_miss_bound_and_recovers(self):
        states = {
            1: LaneDetectionState.DETECTED,
            2: LaneDetectionState.NOT_DETECTED,
            3: LaneDetectionState.NOT_DETECTED,
            4: LaneDetectionState.DETECTED,
        }
        perception = RecordingPerception(
            lambda value: lane_observation(
                value,
                states[value.camera.metadata.stamp.frame],
                offset=0.10,
            )
        )
        pipeline = TemporalLanePipeline(
            TemporalLaneTrackerConfig(
                max_consecutive_misses=1,
                max_coast_age_seconds=1.0,
                coast_confidence_decay=0.5,
                minimum_coast_confidence=0.01,
            ),
            perception_callable=perception,
        )
        runtime = PerceptionRuntime(
            clock=ConstantClock(), perception_callable=pipeline
        )
        try:
            runtime.start()
            estimates = [
                self.submit_and_result(runtime, snapshot(frame), frame)
                .observation.temporal_estimate
                for frame in (1, 2, 3, 4)
            ]
            runtime.stop()

            self.assertEqual(
                [estimate.state for estimate in estimates],
                [
                    TemporalLaneState.TRACKING,
                    TemporalLaneState.COASTING,
                    TemporalLaneState.LOST,
                    TemporalLaneState.TRACKING,
                ],
            )
            tracking, coasting, lost, recovered = estimates
            self.assertEqual(coasting.consecutive_misses, 1)
            self.assertTrue(coasting.extrapolated)
            self.assertIsNotNone(coasting.left_boundary)
            self.assertIsNotNone(coasting.right_boundary)
            self.assertLess(coasting.confidence, tracking.confidence)
            self.assertEqual(lost.consecutive_misses, 2)
            self.assertFalse(lost.extrapolated)
            self.assertIsNone(lost.left_boundary)
            self.assertIsNone(lost.right_boundary)
            self.assertIsNone(lost.center_offset)
            self.assertIsNone(lost.heading_proxy)
            self.assertEqual(lost.confidence, 0.0)
            self.assertEqual(recovered.consecutive_misses, 0)
            self.assertFalse(recovered.extrapolated)
            self.assertGreater(recovered.confidence, 0.0)
        finally:
            self.stop_runtime(runtime)

    def test_coast_age_boundary_is_inclusive_then_expires(self):
        perception = RecordingPerception(
            lambda value: lane_observation(
                value,
                (
                    LaneDetectionState.DETECTED
                    if value.camera.metadata.stamp.frame == 1
                    else LaneDetectionState.NOT_DETECTED
                ),
            )
        )
        pipeline = TemporalLanePipeline(
            TemporalLaneTrackerConfig(
                max_consecutive_misses=10,
                max_coast_age_seconds=0.20,
                coast_confidence_decay=0.99,
                minimum_coast_confidence=0.01,
            ),
            perception_callable=perception,
        )
        runtime = PerceptionRuntime(
            clock=ConstantClock(), perception_callable=pipeline
        )
        try:
            runtime.start()
            first = self.submit_and_result(
                runtime, snapshot(1, sim_time=0.0), 1
            )
            boundary_result = self.submit_and_result(
                runtime, snapshot(2, sim_time=0.20), 2
            )
            expired = self.submit_and_result(
                runtime,
                snapshot(3, sim_time=math.nextafter(0.20, math.inf)),
                3,
            )
            runtime.stop()

            self.assertIs(
                first.observation.temporal_estimate.state,
                TemporalLaneState.TRACKING,
            )
            self.assertIs(
                boundary_result.observation.temporal_estimate.state,
                TemporalLaneState.COASTING,
            )
            self.assertEqual(
                boundary_result.observation.temporal_estimate.track_age_seconds,
                0.20,
            )
            self.assertIs(
                expired.observation.temporal_estimate.state,
                TemporalLaneState.LOST,
            )
            self.assertGreater(
                expired.observation.temporal_estimate.track_age_seconds,
                0.20,
            )
        finally:
            self.stop_runtime(runtime)

    def test_stamped_and_unstamped_input_unusable_surface_and_recover(self):
        for stamped in (True, False):
            with self.subTest(stamped=stamped):
                def build(value, stamped=stamped):
                    frame = value.camera.metadata.stamp.frame
                    if frame == 2:
                        return lane_observation(
                            value,
                            LaneDetectionState.INPUT_UNUSABLE,
                            stamped=stamped,
                        )
                    return lane_observation(
                        value,
                        LaneDetectionState.DETECTED,
                        offset=0.05 * frame,
                    )

                perception = RecordingPerception(build)
                pipeline = TemporalLanePipeline(
                    perception_callable=perception
                )
                runtime = PerceptionRuntime(
                    clock=ConstantClock(), perception_callable=pipeline
                )
                try:
                    runtime.start()
                    tracked = self.submit_and_result(runtime, snapshot(1), 1)
                    unusable = self.submit_and_result(runtime, snapshot(2), 2)
                    recovered = self.submit_and_result(runtime, snapshot(3), 3)
                    runtime.stop()

                    raw_unusable = perception.observations[2]
                    if stamped:
                        self.assertEqual(raw_unusable.source_frame, 2)
                    else:
                        self.assertIsNone(raw_unusable.source_frame)
                        self.assertIsNone(raw_unusable.source_simulation_timestamp)
                        self.assertIsNone(raw_unusable.source_receive_monotonic)
                    carrier = unusable.observation
                    estimate = carrier.temporal_estimate
                    self.assertEqual(carrier.source_frame, raw_unusable.source_frame)
                    self.assertIs(estimate.state, TemporalLaneState.INPUT_UNUSABLE)
                    self.assertIs(
                        estimate.raw_state, LaneDetectionState.INPUT_UNUSABLE
                    )
                    # Pipeline ordering uses the canonical snapshot camera
                    # stamp even when Mission 5 deliberately omits it.
                    self.assertEqual(estimate.source_stamp, unusable.source_stamp)
                    self.assertEqual(estimate.source_stamp.frame, 2)
                    self.assertEqual(estimate.confidence, 0.0)
                    self.assertIsNone(estimate.left_boundary)
                    self.assertIsNone(estimate.right_boundary)
                    self.assertFalse(estimate.extrapolated)
                    self.assertIs(
                        tracked.observation.temporal_estimate.state,
                        TemporalLaneState.TRACKING,
                    )
                    self.assertIs(
                        recovered.observation.temporal_estimate.state,
                        TemporalLaneState.TRACKING,
                    )
                    self.assertEqual(
                        recovered.observation.temporal_estimate.source_stamp.frame,
                        3,
                    )
                finally:
                    self.stop_runtime(runtime)

    def test_runtime_reorder_gate_precedes_pipeline_without_mutation(self):
        perception = RecordingPerception(
            lambda value: lane_observation(
                value, LaneDetectionState.DETECTED, offset=0.10
            )
        )
        pipeline = TemporalLanePipeline(perception_callable=perception)
        runtime = PerceptionRuntime(
            clock=ConstantClock(), perception_callable=pipeline
        )
        try:
            runtime.start()
            first = self.submit_and_result(
                runtime, snapshot(10, sim_time=1.0), 1
            )
            first_estimate = pipeline.latest_estimate

            self.assertFalse(
                runtime.submit(snapshot(10, sim_time=1.0, revision=100))
            )
            self.assertFalse(
                runtime.submit(snapshot(10, sim_time=1.1, revision=101))
            )
            self.assertFalse(
                runtime.submit(snapshot(9, sim_time=0.9, revision=102))
            )
            self.assertFalse(
                runtime.submit(snapshot(11, sim_time=0.5, revision=103))
            )

            self.assertEqual(perception.calls, [10])
            self.assertIs(pipeline.latest_estimate, first_estimate)
            self.assertIs(runtime.latest_result, first)

            final = self.submit_and_result(
                runtime, snapshot(11, sim_time=1.1), 2
            )
            runtime.stop()

            self.assertEqual(perception.calls, [10, 11])
            self.assertEqual(final.observation.temporal_estimate.source_stamp.frame, 11)
            metrics = runtime.metrics
            self.assertEqual(metrics.inputs_received, 6)
            self.assertEqual(metrics.inputs_accepted, 2)
            self.assertEqual(metrics.duplicate_inputs_rejected, 1)
            self.assertEqual(metrics.out_of_order_inputs_rejected, 3)
            self.assertEqual(metrics.inputs_processed, 2)
        finally:
            self.stop_runtime(runtime)

    def test_runtime_coalescing_keeps_latest_order_and_one_temporal_miss(self):
        perception = BlockingFirstPerception(
            lambda value: lane_observation(
                value,
                (
                    LaneDetectionState.DETECTED
                    if value.camera.metadata.stamp.frame == 1
                    else LaneDetectionState.NOT_DETECTED
                ),
            )
        )
        pipeline = TemporalLanePipeline(
            TemporalLaneTrackerConfig(
                max_consecutive_misses=1,
                max_coast_age_seconds=1.0,
            ),
            perception_callable=perception,
        )
        runtime = PerceptionRuntime(
            clock=ConstantClock(), perception_callable=pipeline
        )
        try:
            runtime.start()
            self.assertTrue(runtime.submit(snapshot(1, sim_time=0.1)))
            self.assertTrue(perception.entered.wait(WAIT_SECONDS))
            for frame in (2, 3, 4, 5):
                self.assertTrue(runtime.submit(snapshot(frame, sim_time=frame / 10)))
                self.assertTrue(runtime.has_pending_input)

            mid = runtime.metrics
            self.assertEqual(PENDING_CAPACITY, 1)
            self.assertEqual(runtime.pending_capacity, 1)
            self.assertEqual(mid.inputs_accepted, 5)
            self.assertEqual(mid.pending_frames_coalesced, 3)
            self.assertEqual(mid.source_frames_inferred_dropped, 0)

            perception.release.set()
            self.wait_for_processed(runtime, 2)
            runtime.stop()

            self.assertEqual(perception.calls, [1, 5])
            self.assertEqual(runtime.latest_result.source_stamp.frame, 5)
            estimate = runtime.latest_result.observation.temporal_estimate
            self.assertEqual(estimate.source_stamp.frame, 5)
            self.assertIs(estimate.state, TemporalLaneState.COASTING)
            self.assertEqual(estimate.consecutive_misses, 1)
            self.assertTrue(estimate.extrapolated)
            self.assertEqual(runtime.metrics.inputs_processed, 2)
            self.assertFalse(runtime.has_pending_input)
        finally:
            perception.release.set()
            self.stop_runtime(runtime)

    def test_finish_clock_fault_withholds_result_and_reset_clears_pipeline(self):
        perception = RecordingPerception(
            lambda value: lane_observation(
                value, LaneDetectionState.DETECTED, offset=0.05
            )
        )
        clock = SequenceClock(
            10.0,
            10.0,
            RuntimeError("processing-finish clock unavailable"),
        )
        pipeline = TemporalLanePipeline(perception_callable=perception)
        runtime = PerceptionRuntime(clock=clock, perception_callable=pipeline)
        try:
            runtime.start()
            self.assertTrue(runtime.submit(snapshot(1, sim_time=0.1)))
            with runtime._condition:
                faulted = runtime._condition.wait_for(
                    lambda: runtime._state is PerceptionRuntimeState.FAULTED,
                    timeout=WAIT_SECONDS,
                )
            self.assertTrue(faulted, "finish-clock failure did not fault runtime")

            self.assertEqual(perception.calls, [1])
            self.assertEqual(clock.calls, 3)
            self.assertIsNone(runtime.latest_result)
            self.assertIsNotNone(runtime.fault)
            self.assertEqual(runtime.fault.stage, "clock")
            self.assertIs(
                pipeline.latest_estimate.state,
                TemporalLaneState.TRACKING,
            )
            self.assertEqual(pipeline.latest_estimate.source_stamp.frame, 1)
            self.assertEqual(pipeline.metrics.observations_received, 1)
            self.assertEqual(pipeline.metrics.observations_accepted, 1)
            self.assertEqual(pipeline.metrics.usable_updates, 1)

            metrics = runtime.metrics
            self.assertEqual(metrics.inputs_received, 1)
            self.assertEqual(metrics.inputs_accepted, 1)
            self.assertEqual(metrics.inputs_processed, 0)
            self.assertEqual(metrics.inputs_abandoned, 1)
            self.assertEqual(metrics.perception_successes, 0)
            self.assertEqual(metrics.perception_failures, 0)

            runtime.stop()
            reset = pipeline.reset()
            self.assertIs(reset, pipeline.latest_estimate)
            self.assertIs(reset.state, TemporalLaneState.UNINITIALIZED)
            self.assertIsNone(reset.source_stamp)
            self.assertEqual(pipeline.metrics.resets, 1)
        finally:
            self.stop_runtime(runtime)

    def test_runtime_stale_rejection_precedes_pipeline_and_temporal_accounting(self):
        perception = RecordingPerception(
            lambda value: lane_observation(
                value, LaneDetectionState.NOT_DETECTED
            )
        )
        clock = SequenceClock(0.0, 1.0)
        pipeline = TemporalLanePipeline(perception_callable=perception)
        initial_estimate = pipeline.latest_estimate
        initial_metrics = pipeline.metrics
        runtime = PerceptionRuntime(
            PerceptionRuntimeConfig(max_pending_age_seconds=0.25),
            clock=clock,
            perception_callable=pipeline,
        )
        try:
            runtime.start()
            self.assertTrue(runtime.submit(snapshot(1, sim_time=0.1)))
            self.assertTrue(
                clock.wait_for_call(2),
                "worker did not sample the stale processing-start time",
            )
            # Stop joins the worker after its already-claimed stale
            # disposition, making every metric below quiescent and exact.
            runtime.stop()

            self.assertEqual(clock.calls, 2)
            self.assertEqual(perception.calls, [])
            self.assertIs(pipeline.latest_estimate, initial_estimate)
            self.assertEqual(pipeline.metrics, initial_metrics)
            self.assertIs(
                pipeline.latest_estimate.state,
                TemporalLaneState.UNINITIALIZED,
            )
            self.assertEqual(pipeline.metrics.observations_received, 0)
            self.assertEqual(pipeline.metrics.coast_updates, 0)
            self.assertEqual(pipeline.metrics.lost_updates, 0)

            metrics = runtime.metrics
            self.assertEqual(metrics.inputs_received, 1)
            self.assertEqual(metrics.inputs_accepted, 1)
            self.assertEqual(metrics.stale_inputs_rejected, 1)
            self.assertEqual(metrics.inputs_processed, 0)
            self.assertEqual(metrics.inputs_abandoned, 0)
            self.assertEqual(metrics.last_pending_age_seconds, 1.0)
            self.assertIsNone(runtime.latest_result)
            self.assertFalse(runtime.has_pending_input)
            self.assertIs(runtime.state, PerceptionRuntimeState.STOPPED)
        finally:
            self.stop_runtime(runtime)

    def test_restart_requires_explicit_pipeline_reset_and_adds_no_worker(self):
        perception = RecordingPerception(
            lambda value: lane_observation(
                value, LaneDetectionState.DETECTED, offset=0.05
            )
        )
        threads_before = set(threading.enumerate())
        pipeline = TemporalLanePipeline(perception_callable=perception)
        runtime = PerceptionRuntime(
            clock=ConstantClock(), perception_callable=pipeline
        )
        self.assertEqual(set(threading.enumerate()), threads_before)
        self.assertEqual(PENDING_CAPACITY, 1)
        self.assertEqual(runtime.pending_capacity, 1)
        first_worker = None
        second_worker = None
        try:
            runtime.start()
            first_worker = runtime._worker
            self.assertIsNotNone(first_worker)
            assert first_worker is not None
            self.assertFalse(first_worker.daemon)
            self.assertEqual(first_worker.name, "PerceptionRuntime")
            self.assertEqual(
                set(threading.enumerate()) - threads_before,
                {first_worker},
            )
            self.submit_and_result(runtime, snapshot(10, sim_time=1.0), 1)
            runtime.stop()
            self.assertFalse(first_worker.is_alive())
            self.assertEqual(pipeline.latest_estimate.source_stamp.frame, 10)

            # Runtime ordering resets per epoch; state hidden in its callable
            # has an independent explicit lifecycle.
            pipeline.reset()
            self.assertIs(
                pipeline.latest_estimate.state,
                TemporalLaneState.UNINITIALIZED,
            )
            self.assertIsNone(pipeline.latest_estimate.source_stamp)

            runtime.start()
            second_worker = runtime._worker
            self.assertIsNotNone(second_worker)
            self.assertIsNot(second_worker, first_worker)
            self.assertEqual(
                set(threading.enumerate()) - threads_before,
                {second_worker},
            )
            result = self.submit_and_result(
                runtime, snapshot(1, sim_time=0.1), 1
            )
            runtime.stop()

            self.assertEqual(perception.calls, [10, 1])
            self.assertEqual(perception.thread_names, ["PerceptionRuntime"] * 2)
            self.assertEqual(result.source_stamp.frame, 1)
            self.assertIs(
                result.observation.temporal_estimate.state,
                TemporalLaneState.TRACKING,
            )
            self.assertEqual(runtime.metrics.inputs_received, 1)
            self.assertEqual(runtime.metrics.inputs_accepted, 1)
            self.assertEqual(runtime.metrics.inputs_processed, 1)
            self.assertFalse(second_worker.is_alive())
            self.assertEqual(set(threading.enumerate()), threads_before)
        finally:
            self.stop_runtime(runtime)

    def test_real_bridge_and_runtime_contracts_remain_unchanged(self):
        perception = RecordingPerception(
            lambda value: lane_observation(
                value, LaneDetectionState.DETECTED, offset=0.02
            )
        )
        clock = ConstantClock()
        pipeline = TemporalLanePipeline(perception_callable=perception)
        runtime = PerceptionRuntime(clock=clock, perception_callable=pipeline)
        bridge = PerceptionLiveBridge(runtime, clock=clock)
        try:
            runtime.start()
            first = snapshot(1, sim_time=0.1, receive_monotonic=1.0)
            self.assertTrue(bridge.submit(first))
            self.wait_for_processed(runtime, 1)
            self.assertFalse(
                bridge.submit(
                    snapshot(
                        1,
                        sim_time=0.1,
                        receive_monotonic=9.0,
                        revision=99,
                    )
                )
            )
            second = snapshot(2, sim_time=0.2, receive_monotonic=2.0)
            self.assertTrue(bridge.submit(second))
            self.wait_for_processed(runtime, 2)

            bridge.close_admission()
            self.assertIs(bridge.state, PerceptionLiveBridgeState.CLOSING)
            bridge.drain(0.0)
            self.assertIs(bridge.state, PerceptionLiveBridgeState.CLOSED)
            runtime.stop()

            bridge_metrics = bridge.metrics
            self.assertEqual(bridge_metrics.submissions_received, 3)
            self.assertEqual(bridge_metrics.submissions_admitted, 3)
            self.assertEqual(bridge_metrics.runtime_submissions_accepted, 2)
            self.assertEqual(bridge_metrics.runtime_submissions_rejected, 1)
            self.assertEqual(bridge_metrics.submission_failures, 0)
            self.assertEqual(bridge_metrics.current_in_flight, 0)
            self.assertEqual(bridge_metrics.last_submitted_source_stamp.frame, 2)
            self.assertEqual(
                bridge_metrics.submissions_admitted,
                bridge_metrics.runtime_submissions_accepted
                + bridge_metrics.runtime_submissions_rejected
                + bridge_metrics.submission_failures,
            )

            runtime_metrics = runtime.metrics
            self.assertEqual(runtime_metrics.inputs_received, 3)
            self.assertEqual(runtime_metrics.inputs_accepted, 2)
            self.assertEqual(runtime_metrics.duplicate_inputs_rejected, 1)
            self.assertEqual(runtime_metrics.out_of_order_inputs_rejected, 0)
            self.assertEqual(runtime_metrics.inputs_processed, 2)
            self.assertEqual(runtime_metrics.perception_successes, 2)
            self.assertEqual(runtime_metrics.perception_failures, 0)
            self.assertEqual(
                runtime_metrics.inputs_received,
                runtime_metrics.inputs_accepted
                + runtime_metrics.duplicate_inputs_rejected
                + runtime_metrics.out_of_order_inputs_rejected,
            )
            self.assertIs(runtime.state, PerceptionRuntimeState.STOPPED)
            self.assertFalse(runtime.worker_alive)
            self.assertIsInstance(
                runtime.latest_result.observation, TemporalLaneObservation
            )
            self.assertIs(
                runtime.latest_result.observation.temporal_estimate,
                pipeline.latest_estimate,
            )
            self.assertEqual(perception.calls, [1, 2])
        finally:
            if bridge.state is PerceptionLiveBridgeState.OPEN:
                bridge.close(0.0)
            self.stop_runtime(runtime)


if __name__ == "__main__":
    unittest.main()
