"""Deterministic offline tests for the Mission 6A bounded perception runtime."""

from __future__ import annotations

import ast
import builtins
import dataclasses
import fractions
import gc
import io
import math
import os
import pathlib
import subprocess
import sys
import textwrap
import threading
import unittest
import weakref
from dataclasses import FrozenInstanceError, fields, replace
from unittest import mock

import perception.runtime as runtime_module
from perception.runtime import (
    PENDING_CAPACITY,
    PerceptionRuntime,
    PerceptionRuntimeConfig,
    PerceptionRuntimeFault,
    PerceptionRuntimeMetrics,
    PerceptionRuntimeResult,
    PerceptionRuntimeState,
)
from perception.state import (
    LaneDetectionState,
    LaneObservation,
    LaneObservationReason,
)
from telemetry.freshness import (
    STATUS_TEXT_LIMIT,
    OperationalHealth,
    SensorFreshness,
    SensorKind,
    SensorLifecycle,
    SensorReadiness,
    SensorStatus,
)
from telemetry.state import (
    PIXEL_FORMAT_RGB8,
    CameraFrame,
    CameraFrameMetadata,
    SampleStamp,
    TelemetrySnapshot,
)


WAIT_TIMEOUT_SECONDS = 5.0
_DEFAULT = object()


class FakeClock:
    """A thread-safe manually advanced monotonic clock."""

    def __init__(self, value: float = 0.0) -> None:
        self._value = float(value)
        self._lock = threading.Lock()
        self.calls: list[tuple[str, float]] = []

    def __call__(self) -> float:
        with self._lock:
            value = self._value
            self.calls.append((threading.current_thread().name, value))
            return value

    @property
    def value(self) -> float:
        with self._lock:
            return self._value

    def set(self, value: float) -> None:
        with self._lock:
            self._value = float(value)

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._value += seconds


class WorkerStartGateClock:
    """Hold the first worker clock read after it has claimed the pending item."""

    def __init__(self, ingestion: float, processing_start: float) -> None:
        self.ingestion = float(ingestion)
        self.processing_start = float(processing_start)
        self.worker_entered = threading.Event()
        self.release_worker = threading.Event()
        self._worker_calls = 0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        if threading.current_thread().name != "PerceptionRuntime":
            return self.ingestion
        with self._lock:
            self._worker_calls += 1
            first = self._worker_calls == 1
        if first:
            self.worker_entered.set()
            if not self.release_worker.wait(WAIT_TIMEOUT_SECONDS):
                raise RuntimeError("test did not release the worker clock")
        return self.processing_start


class NthWorkerCallClock(FakeClock):
    """Signal a selected worker clock read without delaying it."""

    def __init__(self, selected_call: int, value: float = 0.0) -> None:
        super().__init__(value)
        self.selected_call = selected_call
        self.selected_call_entered = threading.Event()
        self._worker_call_count = 0

    def __call__(self) -> float:
        value = super().__call__()
        if threading.current_thread().name == "PerceptionRuntime":
            with self._lock:
                self._worker_call_count += 1
                selected = self._worker_call_count == self.selected_call
            if selected:
                self.selected_call_entered.set()
        return value


class FinishFailClock:
    """Fail the processing-finish clock read, then permit fault evidence reads."""

    def __init__(self) -> None:
        self.worker_calls = 0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        if threading.current_thread().name != "PerceptionRuntime":
            return 0.0
        with self._lock:
            self.worker_calls += 1
            call = self.worker_calls
        if call == 1:
            return 0.0
        if call == 2:
            raise RuntimeError("finish clock unavailable")
        return 0.125


class ScriptedWorkerClock:
    """Return deterministic producer time and scripted worker outcomes."""

    def __init__(self, *worker_actions: object, producer_value: float = 0.0) -> None:
        self._worker_actions = iter(worker_actions)
        self._producer_value = producer_value
        self._lock = threading.Lock()

    def __call__(self) -> object:
        if threading.current_thread().name != "PerceptionRuntime":
            return self._producer_value
        with self._lock:
            action = next(self._worker_actions)
        if isinstance(action, BaseException):
            raise action
        return action


def camera_status(accepted_count: int = 1, **changes) -> SensorStatus:
    value = SensorStatus(
        name="rgb_camera",
        kind=SensorKind.CONTINUOUS,
        lifecycle=SensorLifecycle.ATTACHED,
        readiness=SensorReadiness.READY,
        freshness=SensorFreshness.FRESH,
        operational_health=OperationalHealth.HEALTHY,
        evaluated_monotonic=100.0,
        age_seconds=0.0,
        expected_interval=0.1,
        stale_after=0.5,
        accepted_count=accepted_count,
        event_count=None,
        error_count=0,
        reason=None,
    )
    return replace(value, **changes)


def snapshot(
    frame: int = 1,
    *,
    sim_time: float | None = None,
    receive_monotonic: float | None = None,
    sequence: int | None = 1,
    revision: int | None = None,
    statuses: object = _DEFAULT,
) -> TelemetrySnapshot:
    selected_sim_time = frame / 8.0 if sim_time is None else sim_time
    selected_receive = (
        selected_sim_time if receive_monotonic is None else receive_monotonic
    )
    stamp = SampleStamp(
        frame=frame,
        sim_time=selected_sim_time,
        monotonic=selected_receive,
    )
    camera = CameraFrame(
        metadata=CameraFrameMetadata(
            stamp=stamp,
            width_px=1,
            height_px=1,
            horizontal_fov_deg=90.0,
            pixel_format=PIXEL_FORMAT_RGB8,
            row_stride_bytes=3,
            transform=None,
        ),
        rgb_bytes=b"\x01\x02\x03",
    )
    if statuses is _DEFAULT:
        selected_statuses = (
            () if sequence is None else (camera_status(sequence),)
        )
    else:
        selected_statuses = statuses
    return TelemetrySnapshot(
        revision=frame if revision is None else revision,
        captured_monotonic=selected_receive,
        camera=camera,
        statuses=selected_statuses,
    )


def observation(value: TelemetrySnapshot) -> LaneObservation:
    assert value.camera is not None
    stamp = value.camera.metadata.stamp
    return LaneObservation(
        snapshot_revision=value.revision,
        source_frame=stamp.frame,
        source_simulation_timestamp=stamp.sim_time,
        source_receive_monotonic=stamp.monotonic,
        state=LaneDetectionState.NOT_DETECTED,
        reason=LaneObservationReason.NO_PLAUSIBLE_BOUNDARY,
    )


def wait_for_runtime(runtime: PerceptionRuntime, predicate) -> bool:
    """Wait on the runtime's own condition; the timeout is only a deadlock guard."""

    with runtime._condition:
        return runtime._condition.wait_for(predicate, timeout=WAIT_TIMEOUT_SECONDS)


def cleanup_runtime(runtime: PerceptionRuntime, *release_events: threading.Event) -> None:
    """Release test gates and ensure even a failed lifecycle assertion cannot leak."""

    for event in release_events:
        event.set()
    if runtime.worker_alive and runtime.state is PerceptionRuntimeState.STOPPED:
        # This is only a failure-path recovery for the overlapping-stop
        # regression.  A correct runtime never needs it.
        with runtime._condition:
            runtime._state = PerceptionRuntimeState.RUNNING
    if runtime.state is not PerceptionRuntimeState.STOPPED:
        try:
            runtime.stop()
        except TimeoutError:
            for event in release_events:
                event.set()
            runtime.stop()


def run_isolated_python(code: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def import_safety_probe_code(import_statement: str, after_import: str = "") -> str:
    formatted_after_import = after_import.strip().replace("\n", "\n        ")
    return textwrap.dedent(
        f"""
        import _thread
        import builtins
        import io
        import os
        import sys
        import threading
        import time

        forbidden_roots = {{"numpy", "carla", "pygame", "cv2", "scipy"}}
        assert not [
            name for name in sys.modules
            if name.partition(".")[0] in forbidden_roots
        ]
        environment_before = dict(os.environ)
        environment_identity = id(os.environ)
        threads_before = tuple(sorted(t.ident for t in threading.enumerate()))
        filesystem_calls = []
        clock_calls = []
        thread_calls = []
        putenv_calls = []
        unsetenv_calls = []

        original_putenv = os.putenv
        original_unsetenv = os.unsetenv
        def recording_putenv(name, value):
            putenv_calls.append((repr(name), repr(value)))
            return original_putenv(name, value)
        def recording_unsetenv(name):
            unsetenv_calls.append(repr(name))
            return original_unsetenv(name)
        os.putenv = recording_putenv
        os.unsetenv = recording_unsetenv

        def wrapper(calls, label, original):
            def wrapped(*args, **kwargs):
                calls.append(label)
                return original(*args, **kwargs)
            return wrapped

        for owner, name in (
            (builtins, "open"),
            (io, "open"),
            (os, "open"),
            (os, "listdir"),
            (os, "scandir"),
            (os, "stat"),
            (os, "lstat"),
            (os, "access"),
        ):
            setattr(owner, name, wrapper(
                filesystem_calls,
                f"{{owner.__name__}}.{{name}}",
                getattr(owner, name),
            ))
        for name in (
            "time", "time_ns", "monotonic", "monotonic_ns",
            "perf_counter", "perf_counter_ns", "process_time",
            "process_time_ns", "thread_time", "thread_time_ns",
        ):
            if hasattr(time, name):
                setattr(time, name, wrapper(clock_calls, f"time.{{name}}", getattr(time, name)))
        threading.Thread.start = wrapper(
            thread_calls, "threading.Thread.start", threading.Thread.start
        )
        _thread.start_new_thread = wrapper(
            thread_calls, "_thread.start_new_thread", _thread.start_new_thread
        )

        {import_statement}
        {formatted_after_import}

        threads_after = tuple(sorted(t.ident for t in threading.enumerate()))
        assert not [
            name for name in sys.modules
            if name.partition(".")[0] in forbidden_roots
        ]
        assert not filesystem_calls, filesystem_calls
        assert not clock_calls, clock_calls
        assert not thread_calls, thread_calls
        assert not putenv_calls, putenv_calls
        assert not unsetenv_calls, unsetenv_calls
        assert id(os.environ) == environment_identity
        assert dict(os.environ) == environment_before
        assert threads_after == threads_before, (threads_before, threads_after)
        print("IMPORT_SAFE")
        """
    )


class RuntimeTestCase(unittest.TestCase):
    def assert_event(self, event: threading.Event, label: str) -> None:
        self.assertTrue(event.wait(WAIT_TIMEOUT_SECONDS), label)

    def assert_runtime_wait(self, runtime: PerceptionRuntime, predicate, label: str) -> None:
        self.assertTrue(wait_for_runtime(runtime, predicate), label)

    def assert_accounting_partition(self, metrics: PerceptionRuntimeMetrics) -> None:
        self.assertEqual(
            metrics.inputs_received,
            metrics.inputs_accepted
            + metrics.duplicate_inputs_rejected
            + metrics.out_of_order_inputs_rejected,
        )
        self.assertEqual(
            metrics.inputs_accepted,
            metrics.inputs_processed
            + metrics.stale_inputs_rejected
            + metrics.pending_frames_coalesced
            + metrics.pending_inputs_discarded
            + metrics.inputs_abandoned,
        )
        self.assertEqual(
            metrics.inputs_processed,
            metrics.perception_successes + metrics.perception_failures,
        )


class ConfigurationTest(RuntimeTestCase):
    def test_defaults_are_valid_frozen_and_define_a_ten_hz_period(self):
        config = PerceptionRuntimeConfig()
        self.assertEqual(config.target_hz, 10.0)
        self.assertEqual(config.processing_period_seconds, 0.1)
        self.assertEqual(config.max_pending_age_seconds, 0.25)
        self.assertEqual(config.shutdown_timeout_seconds, 2.0)
        self.assertEqual(PENDING_CAPACITY, 1)
        with self.assertRaises(FrozenInstanceError):
            config.target_hz = 20.0

    def test_target_frequency_rejects_bool_zero_negative_nonfinite_and_bounds(self):
        invalid = (
            True,
            False,
            0,
            -1,
            float("nan"),
            float("inf"),
            -float("inf"),
            "10",
            None,
            runtime_module.MIN_TARGET_HZ / 2.0,
            runtime_module.MAX_TARGET_HZ * 2.0,
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                PerceptionRuntimeConfig(target_hz=value)

    def test_pending_age_rejects_bool_zero_negative_nonfinite_and_excess(self):
        invalid = (
            True,
            False,
            0,
            -0.1,
            float("nan"),
            float("inf"),
            -float("inf"),
            "0.25",
            None,
            runtime_module.MAX_RUNTIME_INTERVAL_SECONDS + 1.0,
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                PerceptionRuntimeConfig(max_pending_age_seconds=value)

    def test_shutdown_timeout_has_the_same_strict_numeric_contract(self):
        invalid = (
            True,
            False,
            0,
            -0.1,
            float("nan"),
            float("inf"),
            -float("inf"),
            "2",
            None,
            runtime_module.MAX_RUNTIME_INTERVAL_SECONDS + 1.0,
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                PerceptionRuntimeConfig(shutdown_timeout_seconds=value)

    def test_valid_integral_values_are_normalized_to_builtin_floats(self):
        config = PerceptionRuntimeConfig(
            target_hz=8,
            max_pending_age_seconds=1,
            shutdown_timeout_seconds=3,
        )
        self.assertIs(type(config.target_hz), float)
        self.assertIs(type(config.max_pending_age_seconds), float)
        self.assertIs(type(config.shutdown_timeout_seconds), float)
        self.assertEqual(config.processing_period_seconds, 0.125)

    def test_huge_integers_follow_the_value_error_numeric_contract(self):
        huge = 10**400
        config_cases = (
            {"target_hz": huge},
            {"max_pending_age_seconds": huge},
            {"shutdown_timeout_seconds": huge},
        )
        for values in config_cases:
            with self.subTest(values=tuple(values)), self.assertRaises(ValueError):
                PerceptionRuntimeConfig(**values)

        value = snapshot()
        for field_name in ("sim_time", "monotonic"):
            malformed_stamp = replace(
                value.camera.metadata.stamp,
                **{field_name: huge},
            )
            malformed = replace(
                value,
                camera=replace(
                    value.camera,
                    metadata=replace(value.camera.metadata, stamp=malformed_stamp),
                ),
            )
            runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=observation)
            try:
                runtime.start()
                with self.subTest(field_name=field_name), self.assertRaises(ValueError):
                    runtime.submit(malformed)
                self.assertEqual(runtime.metrics.inputs_received, 0)
            finally:
                cleanup_runtime(runtime)

        base = PerceptionRuntimeResult(
            observation=observation(value),
            source_stamp=value.camera.metadata.stamp,
            source_sequence=1,
            snapshot_revision=1,
            ingestion_monotonic=0.0,
            processing_start_monotonic=0.0,
            processing_finish_monotonic=0.0,
            pending_age_seconds=0.0,
            processing_latency_seconds=0.0,
        )
        with self.assertRaises(ValueError):
            replace(base, ingestion_monotonic=huge)

    def test_runtime_constructor_validates_dependencies_without_calling_them(self):
        calls = []

        def clock():
            calls.append("clock")
            return 0.0

        def perceive(value):
            calls.append(value)
            return observation(value)

        runtime = PerceptionRuntime(clock=clock, perception_callable=perceive)
        self.assertEqual(calls, [])

    def test_config_subclass_is_canonicalized_before_locked_use(self):
        accesses = []
        under_lock = threading.Event()
        runtime_holder = []
        config_fields = {
            "target_hz",
            "max_pending_age_seconds",
            "shutdown_timeout_seconds",
        }

        class HookConfig(PerceptionRuntimeConfig):
            __slots__ = ()

            def __getattribute__(self, name):
                if name in config_fields:
                    runtime = runtime_holder[0] if runtime_holder else None
                    locked = runtime is not None and runtime._lock.locked()
                    accesses.append((name, locked))
                    if locked:
                        under_lock.set()
                return object.__getattribute__(self, name)

        hostile = HookConfig(
            target_hz=8.0,
            max_pending_age_seconds=0.5,
            shutdown_timeout_seconds=1.0,
        )
        accesses.clear()
        runtime = PerceptionRuntime(
            hostile,
            clock=FakeClock(),
            perception_callable=observation,
        )
        runtime_holder.append(runtime)
        try:
            self.assertIs(type(runtime.config), PerceptionRuntimeConfig)
            self.assertIsNot(runtime.config, hostile)
            runtime.start()
            self.assertTrue(runtime.submit(snapshot()))
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._inputs_processed == 1,
                "canonicalized configuration did not support processing",
            )
            runtime.stop()
            self.assertTrue(accesses)
            self.assertEqual({name for name, _ in accesses}, config_fields)
            self.assertFalse(any(locked for _, locked in accesses))
            self.assertFalse(under_lock.is_set())

            canonical = PerceptionRuntimeConfig()
            canonical_runtime = PerceptionRuntime(
                canonical,
                clock=FakeClock(),
                perception_callable=observation,
            )
            self.assertIs(canonical_runtime.config, canonical)
        finally:
            cleanup_runtime(runtime)
        self.assertIs(runtime.config, runtime.config)
        with self.assertRaises(TypeError):
            PerceptionRuntime(config=object())
        with self.assertRaises(TypeError):
            PerceptionRuntime(clock=None)
        with self.assertRaises(TypeError):
            PerceptionRuntime(perception_callable=None)


class ConstructionAndLifecycleTest(RuntimeTestCase):
    def test_construction_creates_no_thread_and_reads_no_clock(self):
        clock = FakeClock(10.0)
        threads_before = tuple(threading.enumerate())
        with mock.patch.object(runtime_module.threading, "Thread", wraps=threading.Thread) as factory:
            runtime = PerceptionRuntime(clock=clock, perception_callable=observation)
        self.assertEqual(factory.call_count, 0)
        self.assertEqual(clock.calls, [])
        self.assertEqual(tuple(threading.enumerate()), threads_before)
        self.assertIs(runtime.state, PerceptionRuntimeState.STOPPED)
        self.assertFalse(runtime.worker_alive)
        self.assertFalse(runtime.has_pending_input)
        self.assertIsNone(runtime.latest_result)
        self.assertIsNone(runtime.fault)

    def test_start_is_explicit_idempotent_and_creates_one_non_daemon_worker(self):
        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=observation)
        try:
            runtime.start()
            worker = runtime._worker
            self.assertIsNotNone(worker)
            self.assertTrue(worker.is_alive())
            self.assertFalse(worker.daemon)
            self.assertEqual(worker.name, "PerceptionRuntime")
            runtime.start()
            self.assertIs(runtime._worker, worker)
            self.assertEqual(
                sum(t is worker for t in threading.enumerate()),
                1,
            )
            self.assertIs(runtime.state, PerceptionRuntimeState.RUNNING)
        finally:
            cleanup_runtime(runtime)
        self.assertFalse(worker.is_alive())

    def test_idle_stop_and_repeated_stop_are_idempotent(self):
        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=observation)
        runtime.stop()
        runtime.start()
        worker = runtime._worker
        runtime.stop()
        metrics = runtime.metrics
        runtime.stop()
        self.assertIs(runtime.state, PerceptionRuntimeState.STOPPED)
        self.assertFalse(runtime.worker_alive)
        self.assertFalse(worker.is_alive())
        self.assertEqual(runtime.metrics, metrics)

    def test_clean_restart_uses_a_new_worker_and_resets_epoch_state(self):
        clock = FakeClock(2.0)
        called = threading.Event()
        calls = []

        def perceive(value):
            calls.append(value.camera.metadata.stamp.frame)
            called.set()
            return observation(value)

        runtime = PerceptionRuntime(clock=clock, perception_callable=perceive)
        try:
            runtime.start()
            first_worker = runtime._worker
            self.assertTrue(runtime.submit(snapshot(10, sequence=1)))
            self.assert_event(called, "first epoch perception did not run")
            runtime.stop()
            self.assertEqual(runtime.metrics.inputs_processed, 1)
            self.assertEqual(runtime.latest_result.source_stamp.frame, 10)
            self.assertFalse(first_worker.is_alive())

            called.clear()
            runtime.start()
            second_worker = runtime._worker
            self.assertIsNot(second_worker, first_worker)
            self.assertIsNone(runtime.latest_result)
            self.assertIsNone(runtime.fault)
            self.assertEqual(runtime.metrics.inputs_received, 0)
            # Ordering belongs to the execution epoch, so the same canonical
            # source identity may seed a deliberate restart.
            self.assertTrue(runtime.submit(snapshot(10, sequence=1)))
            self.assert_event(called, "second epoch perception did not run")
            runtime.stop()
            self.assertEqual(calls, [10, 10])
            self.assertFalse(second_worker.is_alive())
        finally:
            cleanup_runtime(runtime)

    def test_stop_discards_one_pending_item_and_none_leaks_into_restart(self):
        entered = threading.Event()
        release = threading.Event()
        stop_done = threading.Event()
        calls = []

        def perceive(value):
            frame = value.camera.metadata.stamp.frame
            calls.append(frame)
            if frame == 1:
                entered.set()
                if not release.wait(WAIT_TIMEOUT_SECONDS):
                    raise RuntimeError("test perception gate was not released")
            return observation(value)

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=perceive)
        stopper = None
        try:
            runtime.start()
            self.assertTrue(runtime.submit(snapshot(1, sequence=1)))
            self.assert_event(entered, "worker did not claim the first input")
            self.assertTrue(runtime.submit(snapshot(2, sequence=2)))
            stopper = threading.Thread(
                target=lambda: (runtime.stop(), stop_done.set()),
                name="runtime-test-stopper",
            )
            stopper.start()
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._state is PerceptionRuntimeState.STOPPING,
                "runtime did not enter STOPPING",
            )
            self.assertFalse(runtime.has_pending_input)
            self.assertEqual(runtime.metrics.pending_inputs_discarded, 1)
            release.set()
            self.assert_event(stop_done, "stop did not complete after release")
            stopper.join(WAIT_TIMEOUT_SECONDS)
            self.assertFalse(stopper.is_alive())
            self.assertEqual(calls, [1])

            runtime.start()
            self.assertFalse(runtime.has_pending_input)
            self.assertEqual(runtime.metrics.inputs_received, 0)
            self.assertTrue(runtime.submit(snapshot(3, sequence=1)))
            self.assert_runtime_wait(
                runtime,
                lambda: (
                    runtime._latest_result is not None
                    and runtime._latest_result.source_stamp.frame == 3
                ),
                "restart did not publish frame 3",
            )
            runtime.stop()
            self.assertEqual(calls, [1, 3])
        finally:
            release.set()
            if stopper is not None:
                stopper.join(WAIT_TIMEOUT_SECONDS)
            cleanup_runtime(runtime, release)

    def test_stop_during_active_perception_waits_without_deadlock(self):
        entered = threading.Event()
        release = threading.Event()
        stopped = threading.Event()

        def perceive(value):
            entered.set()
            if not release.wait(WAIT_TIMEOUT_SECONDS):
                raise RuntimeError("test perception gate was not released")
            return observation(value)

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=perceive)
        stopper = threading.Thread(
            target=lambda: (runtime.stop(), stopped.set()),
            name="active-stop-test",
        )
        try:
            runtime.start()
            runtime.submit(snapshot())
            self.assert_event(entered, "perception did not block")
            stopper.start()
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._state is PerceptionRuntimeState.STOPPING,
                "runtime did not enter STOPPING",
            )
            self.assertFalse(stopped.is_set())
            self.assertTrue(runtime.worker_alive)
            release.set()
            self.assert_event(stopped, "active stop did not complete")
            stopper.join(WAIT_TIMEOUT_SECONDS)
            self.assertFalse(stopper.is_alive())
            self.assertIs(runtime.state, PerceptionRuntimeState.STOPPED)
            self.assertEqual(runtime.metrics.inputs_processed, 1)
        finally:
            release.set()
            if stopper.ident is not None:
                stopper.join(WAIT_TIMEOUT_SECONDS)
            cleanup_runtime(runtime, release)

    def test_shutdown_timeout_remains_stopping_and_a_retry_finishes(self):
        entered = threading.Event()
        release = threading.Event()

        def perceive(value):
            entered.set()
            if not release.wait(WAIT_TIMEOUT_SECONDS):
                raise RuntimeError("test perception gate was not released")
            return observation(value)

        runtime = PerceptionRuntime(
            PerceptionRuntimeConfig(shutdown_timeout_seconds=0.01),
            clock=FakeClock(),
            perception_callable=perceive,
        )
        try:
            runtime.start()
            runtime.submit(snapshot())
            self.assert_event(entered, "perception did not block")
            with self.assertRaises(TimeoutError):
                runtime.stop()
            self.assertIs(runtime.state, PerceptionRuntimeState.STOPPING)
            self.assertTrue(runtime.worker_alive)
            with self.assertRaises(RuntimeError):
                runtime.start()
            release.set()
            runtime.stop()
            self.assertIs(runtime.state, PerceptionRuntimeState.STOPPED)
            self.assertFalse(runtime.worker_alive)
            runtime.start()
            self.assertTrue(runtime.submit(snapshot(2, sequence=1)))
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._inputs_processed == 1,
                "runtime did not process after timeout recovery and restart",
            )
            runtime.stop()
            self.assertIs(runtime.state, PerceptionRuntimeState.STOPPED)
        finally:
            cleanup_runtime(runtime, release)

    def test_three_and_five_simultaneous_stop_callers_finalize_once(self):
        for caller_count in (3, 5):
            with self.subTest(caller_count=caller_count):
                entered = threading.Event()
                release = threading.Event()
                barrier = threading.Barrier(caller_count + 1)
                errors = []

                def perceive(value):
                    entered.set()
                    if not release.wait(WAIT_TIMEOUT_SECONDS):
                        raise RuntimeError("test perception gate was not released")
                    return observation(value)

                def stop_together():
                    try:
                        barrier.wait()
                        runtime.stop()
                    except Exception as exc:
                        errors.append(exc)

                runtime = PerceptionRuntime(
                    clock=FakeClock(), perception_callable=perceive
                )
                callers = [
                    threading.Thread(target=stop_together, name=f"stop-{index}")
                    for index in range(caller_count)
                ]
                try:
                    runtime.start()
                    self.assertTrue(runtime.submit(snapshot()))
                    self.assert_event(entered, "perception did not block")
                    for caller in callers:
                        caller.start()
                    barrier.wait()
                    self.assert_runtime_wait(
                        runtime,
                        lambda: runtime._state is PerceptionRuntimeState.STOPPING,
                        "concurrent stops did not begin shutdown",
                    )
                    release.set()
                    for caller in callers:
                        caller.join(WAIT_TIMEOUT_SECONDS)
                    self.assertFalse(any(caller.is_alive() for caller in callers))
                    self.assertEqual(errors, [])
                    self.assertIs(runtime.state, PerceptionRuntimeState.STOPPED)
                    self.assertFalse(runtime.worker_alive)
                    self.assertEqual(runtime.metrics.inputs_processed, 1)
                    self.assert_accounting_partition(runtime.metrics)
                finally:
                    release.set()
                    for caller in callers:
                        if caller.ident is not None:
                            caller.join(WAIT_TIMEOUT_SECONDS)
                    cleanup_runtime(runtime, release)

    def test_faulted_runtime_must_be_stopped_before_restart(self):
        def fail(_value):
            raise RuntimeError("perception failed")

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=fail)
        try:
            runtime.start()
            runtime.submit(snapshot())
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._state is PerceptionRuntimeState.FAULTED,
                "runtime did not fault",
            )
            with self.assertRaises(RuntimeError):
                runtime.start()
            self.assertIsNotNone(runtime.fault)
            runtime.stop()
            self.assertIs(runtime.state, PerceptionRuntimeState.STOPPED)
            self.assertFalse(runtime.worker_alive)
            runtime.start()
            self.assertIs(runtime.state, PerceptionRuntimeState.RUNNING)
            self.assertIsNone(runtime.fault)
            self.assertEqual(runtime.metrics.perception_failures, 0)
        finally:
            cleanup_runtime(runtime)

    def test_simultaneous_starts_create_only_one_worker(self):
        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=observation)
        barrier = threading.Barrier(3)
        errors = []

        def start_runtime():
            try:
                barrier.wait()
                runtime.start()
            except Exception as exc:
                errors.append(exc)

        starters = [threading.Thread(target=start_runtime) for _ in range(2)]
        try:
            for thread in starters:
                thread.start()
            barrier.wait()
            for thread in starters:
                thread.join(WAIT_TIMEOUT_SECONDS)
            self.assertFalse(any(thread.is_alive() for thread in starters))
            self.assertEqual(errors, [])
            worker = runtime._worker
            self.assertIsNotNone(worker)
            self.assertTrue(worker.is_alive())
            self.assertEqual(sum(t is worker for t in threading.enumerate()), 1)
        finally:
            cleanup_runtime(runtime)

    def test_late_duplicate_stop_cannot_clobber_a_restarted_epoch(self):
        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=observation)
        runtime.start()
        old_worker = runtime._worker
        late_join_returned = threading.Event()
        allow_late_finalize = threading.Event()
        early_done = threading.Event()
        late_done = threading.Event()
        errors = []
        original_join = threading.Thread.join

        def controlled_join(thread, timeout=None):
            result = original_join(thread, timeout)
            if (
                thread is old_worker
                and threading.current_thread().name == "late-stop"
            ):
                late_join_returned.set()
                if not allow_late_finalize.wait(WAIT_TIMEOUT_SECONDS):
                    raise RuntimeError("late stop was not released")
            return result

        def invoke_stop(done):
            try:
                runtime.stop()
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()

        late = threading.Thread(
            target=lambda: invoke_stop(late_done), name="late-stop"
        )
        early = threading.Thread(
            target=lambda: invoke_stop(early_done), name="early-stop"
        )
        replacement = None
        try:
            with mock.patch.object(threading.Thread, "join", controlled_join):
                late.start()
                self.assert_event(
                    late_join_returned,
                    "late stop did not pause after joining the old worker",
                )
                early.start()
                self.assert_event(early_done, "early stop did not finalize")
                self.assertIs(runtime.state, PerceptionRuntimeState.STOPPED)

                runtime.start()
                replacement = runtime._worker
                self.assertIsNot(replacement, old_worker)
                self.assertTrue(replacement.is_alive())
                # A duplicated/delayed ownership finalizer from the old epoch
                # must be identity-guarded just like the delayed stop caller.
                runtime._finalize_worker_exit(old_worker, None)
                self.assertIs(runtime._worker, replacement)
                self.assertIs(runtime.state, PerceptionRuntimeState.RUNNING)
                allow_late_finalize.set()
                self.assert_event(late_done, "late stop did not return")
                original_join(late, WAIT_TIMEOUT_SECONDS)
                original_join(early, WAIT_TIMEOUT_SECONDS)

            self.assertEqual(errors, [])
            self.assertIs(runtime._worker, replacement)
            self.assertIs(runtime.state, PerceptionRuntimeState.RUNNING)
            self.assertTrue(runtime.worker_alive)
            runtime.stop()
            self.assertFalse(replacement.is_alive())
        finally:
            allow_late_finalize.set()
            if late.ident is not None:
                original_join(late, WAIT_TIMEOUT_SECONDS)
            if early.ident is not None:
                original_join(early, WAIT_TIMEOUT_SECONDS)
            cleanup_runtime(runtime, allow_late_finalize)

    def test_stopped_submission_does_not_read_the_clock(self):
        clock = FakeClock(7.0)
        runtime = PerceptionRuntime(clock=clock, perception_callable=observation)
        self.assertFalse(runtime.submit(snapshot()))
        self.assertEqual(clock.calls, [])
        self.assertEqual(runtime.metrics.inputs_received, 0)

    def test_ingestion_clock_fault_propagates_without_shared_state_change(self):
        cases = (
            (RuntimeError("ingestion unavailable"), RuntimeError),
            (float("nan"), ValueError),
        )
        for first_action, expected_type in cases:
            with self.subTest(first_action=type(first_action).__name__):
                actions = iter((first_action,))
                safe_clock = FakeClock()

                def clock():
                    try:
                        action = next(actions)
                    except StopIteration:
                        return safe_clock()
                    if isinstance(action, Exception):
                        raise action
                    return action

                runtime = PerceptionRuntime(
                    clock=clock, perception_callable=observation
                )
                try:
                    runtime.start()
                    before = runtime.metrics
                    with self.assertRaises(expected_type):
                        runtime.submit(snapshot())
                    self.assertEqual(runtime.metrics, before)
                    self.assertIs(runtime.state, PerceptionRuntimeState.RUNNING)
                    self.assertFalse(runtime.has_pending_input)
                    self.assertTrue(runtime.submit(snapshot(2, sequence=1)))
                    self.assert_runtime_wait(
                        runtime,
                        lambda: runtime._inputs_processed == 1,
                        "runtime was not usable after ingestion clock failure",
                    )
                    self.assert_accounting_partition(runtime.metrics)
                finally:
                    cleanup_runtime(runtime)

    def test_thread_start_failure_restores_a_clean_stopped_state(self):
        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=observation)
        with mock.patch.object(
            threading.Thread,
            "start",
            side_effect=RuntimeError("thread start failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "thread start failed"):
                runtime.start()
        self.assertIs(runtime.state, PerceptionRuntimeState.STOPPED)
        self.assertIsNone(runtime._worker)
        self.assertFalse(runtime.worker_alive)
        runtime.start()
        runtime.stop()


class OneSlotBufferTest(RuntimeTestCase):
    def run_blocked_burst(self, final_frame: int):
        entered = threading.Event()
        release = threading.Event()
        newest_entered = threading.Event()
        calls = []

        def perceive(value):
            frame = value.camera.metadata.stamp.frame
            calls.append(frame)
            if len(calls) == 1:
                entered.set()
                if not release.wait(WAIT_TIMEOUT_SECONDS):
                    raise RuntimeError("test perception gate was not released")
            else:
                newest_entered.set()
            return observation(value)

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=perceive)
        try:
            runtime.start()
            self.assertTrue(runtime.submit(snapshot(1, sequence=1)))
            self.assert_event(entered, "first frame was not claimed")
            for frame in range(2, final_frame + 1):
                self.assertTrue(runtime.submit(snapshot(frame, sequence=frame)))
                self.assertTrue(runtime.has_pending_input)
                self.assertNotIsInstance(runtime._pending, (list, tuple, dict, set))
            mid_metrics = runtime.metrics
            release.set()
            self.assert_event(newest_entered, "newest pending frame was not processed")
            runtime.stop()
            return runtime, calls, mid_metrics
        finally:
            cleanup_runtime(runtime, release)

    def test_three_pending_replacements_keep_only_the_newest(self):
        runtime, calls, mid = self.run_blocked_burst(4)
        self.assertEqual(calls, [1, 4])
        self.assertEqual(mid.inputs_received, 4)
        self.assertEqual(mid.inputs_accepted, 4)
        self.assertEqual(mid.pending_frames_coalesced, 2)
        self.assertEqual(mid.source_frames_inferred_dropped, 0)
        self.assertEqual(runtime.metrics.inputs_processed, 2)
        self.assertEqual(runtime.latest_result.source_stamp.frame, 4)

    def test_ten_thousand_submissions_do_not_accumulate_a_backlog(self):
        runtime, calls, mid = self.run_blocked_burst(10_000)
        self.assertEqual(PENDING_CAPACITY, 1)
        self.assertEqual(calls, [1, 10_000])
        self.assertEqual(mid.inputs_received, 10_000)
        self.assertEqual(mid.inputs_accepted, 10_000)
        self.assertEqual(mid.pending_frames_coalesced, 9_998)
        self.assertEqual(runtime.metrics.inputs_processed, 2)
        self.assertFalse(runtime.has_pending_input)

    def test_rejected_inputs_never_displace_the_newest_valid_pending_item(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def perceive(value):
            calls.append(value.camera.metadata.stamp.frame)
            if len(calls) == 1:
                entered.set()
                if not release.wait(WAIT_TIMEOUT_SECONDS):
                    raise RuntimeError("test perception gate was not released")
            return observation(value)

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=perceive)
        try:
            runtime.start()
            runtime.submit(snapshot(1, sequence=1))
            self.assert_event(entered, "first frame was not claimed")
            self.assertTrue(runtime.submit(snapshot(2, sequence=2)))
            self.assertFalse(runtime.submit(snapshot(2, sequence=2, revision=99)))
            self.assertFalse(runtime.submit(snapshot(1, sequence=1, revision=100)))
            self.assertTrue(runtime.submit(snapshot(3, sequence=3)))
            release.set()
            self.assert_runtime_wait(
                runtime,
                lambda: (
                    runtime._latest_result is not None
                    and runtime._latest_result.source_stamp.frame == 3
                ),
                "newest valid frame was not published",
            )
            runtime.stop()
            self.assertEqual(calls, [1, 3])
            metrics = runtime.metrics
            self.assertEqual(metrics.pending_frames_coalesced, 1)
            self.assertEqual(metrics.duplicate_inputs_rejected, 1)
            self.assertEqual(metrics.out_of_order_inputs_rejected, 1)
        finally:
            cleanup_runtime(runtime, release)


class OrderingTest(RuntimeTestCase):
    def test_duplicate_and_all_order_regressions_are_classified_exactly(self):
        entered = threading.Event()
        release = threading.Event()

        def perceive(value):
            entered.set()
            if not release.wait(WAIT_TIMEOUT_SECONDS):
                raise RuntimeError("test perception gate was not released")
            return observation(value)

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=perceive)
        try:
            runtime.start()
            self.assertTrue(runtime.submit(snapshot(10, sequence=1, sim_time=10.0)))
            self.assert_event(entered, "baseline was not claimed")
            # Receive-monotonic and snapshot revision are evidence, not frame
            # identity: equal frame+simulation time is still a duplicate.
            self.assertFalse(
                runtime.submit(
                    snapshot(
                        10,
                        sequence=1,
                        sim_time=10.0,
                        receive_monotonic=99.0,
                        revision=500,
                    )
                )
            )
            self.assertFalse(runtime.submit(snapshot(10, sequence=2, sim_time=11.0)))
            self.assertFalse(runtime.submit(snapshot(9, sequence=2, sim_time=9.0)))
            self.assertFalse(runtime.submit(snapshot(11, sequence=2, sim_time=9.5)))
            metrics = runtime.metrics
            self.assertEqual(metrics.inputs_received, 5)
            self.assertEqual(metrics.inputs_accepted, 1)
            self.assertEqual(metrics.duplicate_inputs_rejected, 1)
            self.assertEqual(metrics.out_of_order_inputs_rejected, 3)
        finally:
            release.set()
            cleanup_runtime(runtime, release)

    def test_monotonic_sequence_is_accepted_and_latest_wins(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def perceive(value):
            calls.append(value.camera.metadata.stamp.frame)
            if len(calls) == 1:
                entered.set()
                if not release.wait(WAIT_TIMEOUT_SECONDS):
                    raise RuntimeError("test perception gate was not released")
            return observation(value)

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=perceive)
        try:
            runtime.start()
            runtime.submit(snapshot(1, sequence=1))
            self.assert_event(entered, "baseline was not claimed")
            self.assertTrue(runtime.submit(snapshot(2, sequence=2)))
            self.assertTrue(runtime.submit(snapshot(3, sequence=3)))
            release.set()
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._inputs_processed == 2,
                "newest monotonic input was not processed",
            )
            runtime.stop()
            self.assertEqual(calls, [1, 3])
            self.assertEqual(runtime.metrics.pending_frames_coalesced, 1)
        finally:
            cleanup_runtime(runtime, release)

    def test_observable_source_sequence_gap_is_counted_exactly(self):
        entered = threading.Event()
        release = threading.Event()

        def perceive(value):
            entered.set()
            if not release.wait(WAIT_TIMEOUT_SECONDS):
                raise RuntimeError("test perception gate was not released")
            return observation(value)

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=perceive)
        try:
            runtime.start()
            runtime.submit(snapshot(10, sequence=10))
            self.assert_event(entered, "baseline was not claimed")
            self.assertTrue(runtime.submit(snapshot(20, sequence=13)))
            self.assertTrue(runtime.submit(snapshot(21, sequence=14)))
            metrics = runtime.metrics
            self.assertEqual(metrics.source_frames_inferred_dropped, 2)
            self.assertEqual(metrics.pending_frames_coalesced, 1)
        finally:
            release.set()
            cleanup_runtime(runtime, release)

    def test_runtime_replacement_is_never_reported_as_source_loss(self):
        entered = threading.Event()
        release = threading.Event()

        def perceive(value):
            entered.set()
            if not release.wait(WAIT_TIMEOUT_SECONDS):
                raise RuntimeError("test perception gate was not released")
            return observation(value)

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=perceive)
        try:
            runtime.start()
            runtime.submit(snapshot(1, sequence=1))
            self.assert_event(entered, "baseline was not claimed")
            for frame in (2, 3, 4):
                self.assertTrue(runtime.submit(snapshot(frame, sequence=frame)))
            metrics = runtime.metrics
            self.assertEqual(metrics.pending_frames_coalesced, 2)
            self.assertEqual(metrics.source_frames_inferred_dropped, 0)
        finally:
            release.set()
            cleanup_runtime(runtime, release)

    def test_canonical_producer_count_advances_only_with_camera_acceptance(self):
        from telemetry.aggregator import TelemetryAggregator

        aggregator = TelemetryAggregator()
        aggregator.register("rgb_camera")
        frame_10 = snapshot(10).camera
        duplicate_10 = snapshot(10, receive_monotonic=99.0).camera
        frame_11 = snapshot(11).camera

        self.assertTrue(aggregator.submit_camera_frame(frame_10))
        first_health = next(
            value
            for value in aggregator.raw_snapshot().sensors
            if value.name == "rgb_camera"
        )
        self.assertEqual(first_health.messages_received, 1)

        self.assertFalse(aggregator.submit_camera_frame(duplicate_10))
        duplicate_health = next(
            value
            for value in aggregator.raw_snapshot().sensors
            if value.name == "rgb_camera"
        )
        self.assertEqual(duplicate_health.messages_received, 1)

        self.assertTrue(aggregator.submit_camera_frame(frame_11))
        next_health = next(
            value
            for value in aggregator.raw_snapshot().sensors
            if value.name == "rgb_camera"
        )
        self.assertEqual(next_health.messages_received, 2)

    def test_hostile_real_ordering_methods_never_run_under_runtime_lock(self):
        comparisons_under_lock = threading.Event()

        class HostileReal(fractions.Fraction):
            runtime = None

            def _record_lock_context(self):
                acquired = self.runtime._lock.acquire(blocking=False)
                if acquired:
                    self.runtime._lock.release()
                else:
                    comparisons_under_lock.set()

            def __eq__(self, other):
                self._record_lock_context()
                return super().__eq__(other)

            def __lt__(self, other):
                self._record_lock_context()
                return super().__lt__(other)

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=observation)
        try:
            runtime.start()
            self.assertTrue(runtime.submit(snapshot(10, sim_time=1.0, sequence=10)))
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._inputs_processed == 1,
                "baseline was not processed",
            )
            hostile = HostileReal(1, 1)
            hostile.runtime = runtime
            self.assertFalse(
                runtime.submit(
                    snapshot(10, sim_time=hostile, sequence=10, revision=11)
                )
            )
            self.assertFalse(comparisons_under_lock.is_set())
            self.assertIs(type(runtime._last_accepted_stamp.sim_time), float)
        finally:
            cleanup_runtime(runtime)

    def test_samplestamp_subclass_is_canonicalized_before_locked_ordering(self):
        accesses = []
        under_lock = threading.Event()
        runtime_holder = []

        class HookStamp(SampleStamp):
            __slots__ = ("__weakref__",)

            def __getattribute__(self, name):
                if name in {"frame", "sim_time", "monotonic"}:
                    runtime = runtime_holder[0] if runtime_holder else None
                    locked = runtime is not None and runtime._lock.locked()
                    accesses.append((name, locked))
                    if locked:
                        under_lock.set()
                return object.__getattribute__(self, name)

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=observation)
        runtime_holder.append(runtime)
        hostile = HookStamp(frame=10, sim_time=10.0, monotonic=10.0)
        hostile_reference = weakref.ref(hostile)
        base = snapshot(
            10,
            sim_time=10.0,
            receive_monotonic=10.0,
            sequence=10,
        )
        assert base.camera is not None
        hostile_snapshot = replace(
            base,
            camera=replace(
                base.camera,
                metadata=replace(base.camera.metadata, stamp=hostile),
            ),
        )
        try:
            runtime.start()
            self.assertTrue(runtime.submit(hostile_snapshot))
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._inputs_processed == 1,
                "hostile-subclass submission was not processed",
            )

            first_result = runtime.latest_result
            self.assertIsNotNone(first_result)
            assert first_result is not None
            self.assertIs(type(first_result.source_stamp), SampleStamp)
            self.assertIs(type(first_result.source_stamp.frame), int)
            self.assertIs(type(first_result.source_stamp.sim_time), float)
            self.assertIs(type(first_result.source_stamp.monotonic), float)
            self.assertIsNot(first_result.source_stamp, hostile)
            self.assertIs(type(runtime._last_accepted_stamp), SampleStamp)
            self.assertIs(
                type(runtime.metrics.last_processed_source_stamp), SampleStamp
            )

            self.assertFalse(
                runtime.submit(
                    snapshot(
                        10,
                        sim_time=10.0,
                        receive_monotonic=99.0,
                        sequence=10,
                        revision=11,
                    )
                )
            )
            self.assertFalse(
                runtime.submit(snapshot(9, sim_time=9.0, sequence=9, revision=12))
            )
            canonical = snapshot(
                11,
                sim_time=11.0,
                receive_monotonic=11.0,
                sequence=11,
            )
            assert canonical.camera is not None
            canonical_stamp = canonical.camera.metadata.stamp
            self.assertTrue(runtime.submit(canonical))
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._inputs_processed == 2,
                "canonical submission after hostile evidence was not processed",
            )
            runtime.stop()

            result = runtime.latest_result
            self.assertIsNotNone(result)
            assert result is not None
            self.assertIs(result.source_stamp, canonical_stamp)
            self.assertIs(runtime.metrics.last_processed_source_stamp, canonical_stamp)
            self.assertEqual(result.source_stamp.frame, 11)
            metrics = runtime.metrics
            self.assertEqual(metrics.inputs_received, 4)
            self.assertEqual(metrics.inputs_accepted, 2)
            self.assertEqual(metrics.inputs_processed, 2)
            self.assertEqual(metrics.duplicate_inputs_rejected, 1)
            self.assertEqual(metrics.out_of_order_inputs_rejected, 1)
            self.assertTrue(accesses)
            self.assertFalse(any(locked for _, locked in accesses))
            self.assertFalse(under_lock.is_set())

            hostile = None
            hostile_snapshot = None
            gc.collect()
            self.assertIsNone(hostile_reference())
        finally:
            cleanup_runtime(runtime)

    def test_missing_or_ambiguous_source_sequence_disables_gap_inference(self):
        entered = threading.Event()
        release = threading.Event()

        def perceive(value):
            entered.set()
            if not release.wait(WAIT_TIMEOUT_SECONDS):
                raise RuntimeError("test perception gate was not released")
            return observation(value)

        duplicate_statuses = (camera_status(100), camera_status(100))
        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=perceive)
        try:
            runtime.start()
            runtime.submit(snapshot(1, sequence=None))
            self.assert_event(entered, "baseline was not claimed")
            self.assertTrue(runtime.submit(snapshot(10, statuses=duplicate_statuses)))
            self.assertEqual(runtime.metrics.source_frames_inferred_dropped, 0)
        finally:
            release.set()
            cleanup_runtime(runtime, release)

    def test_structurally_invalid_identity_raises_without_changing_metrics(self):
        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=observation)
        runtime.start()
        try:
            with self.assertRaises(TypeError):
                runtime.submit(object())
            with self.assertRaises(ValueError):
                runtime.submit(TelemetrySnapshot(revision=1, captured_monotonic=0.0))
            malformed = replace(
                snapshot(),
                camera=replace(
                    snapshot().camera,
                    metadata=replace(
                        snapshot().camera.metadata,
                        stamp=SampleStamp(True, 0.0, 0.0),
                    ),
                ),
            )
            with self.assertRaises(ValueError):
                runtime.submit(malformed)
            self.assertEqual(runtime.metrics.inputs_received, 0)
            self.assertEqual(runtime.metrics.inputs_accepted, 0)
        finally:
            cleanup_runtime(runtime)


class FreshnessTest(RuntimeTestCase):
    def run_pending_age_case(self, age: float, expect_perception: bool):
        # Use zero as the baseline so nextafter(0.25, +inf) survives subtraction
        # exactly instead of being rounded while added to a larger timestamp.
        clock = WorkerStartGateClock(ingestion=0.0, processing_start=age)
        called = threading.Event()

        def perceive(value):
            called.set()
            return observation(value)

        runtime = PerceptionRuntime(
            PerceptionRuntimeConfig(max_pending_age_seconds=0.25),
            clock=clock,
            perception_callable=perceive,
        )
        try:
            runtime.start()
            self.assertTrue(runtime.submit(snapshot()))
            self.assert_event(clock.worker_entered, "worker did not claim pending input")
            clock.release_worker.set()
            if expect_perception:
                self.assert_event(called, "fresh input did not reach perception")
            runtime.stop()
            return runtime, called
        finally:
            cleanup_runtime(runtime, clock.release_worker)

    def test_fresh_pending_input_is_processed(self):
        runtime, called = self.run_pending_age_case(0.125, True)
        self.assertTrue(called.is_set())
        self.assertEqual(runtime.metrics.stale_inputs_rejected, 0)
        self.assertEqual(runtime.metrics.inputs_processed, 1)
        self.assertEqual(runtime.metrics.last_pending_age_seconds, 0.125)

    def test_exact_pending_age_threshold_is_still_fresh(self):
        runtime, called = self.run_pending_age_case(0.25, True)
        self.assertTrue(called.is_set())
        self.assertEqual(runtime.metrics.stale_inputs_rejected, 0)
        self.assertEqual(runtime.metrics.inputs_processed, 1)
        self.assertEqual(runtime.metrics.last_pending_age_seconds, 0.25)

    def test_one_representable_value_above_threshold_is_rejected_before_perception(self):
        age = math.nextafter(0.25, math.inf)
        runtime, called = self.run_pending_age_case(age, False)
        self.assertFalse(called.is_set())
        metrics = runtime.metrics
        self.assertEqual(metrics.inputs_received, 1)
        self.assertEqual(metrics.inputs_accepted, 1)
        self.assertEqual(metrics.inputs_processed, 0)
        self.assertEqual(metrics.stale_inputs_rejected, 1)
        self.assertEqual(metrics.perception_successes, 0)
        self.assertEqual(metrics.last_pending_age_seconds, age)
        self.assertIsNone(runtime.latest_result)

    def test_pending_frame_can_age_out_while_prior_perception_is_blocked(self):
        # Worker reads: start frame 1, finish frame 1, start frame 2.  The third
        # read proves frame 2 has been atomically claimed before stop begins.
        clock = NthWorkerCallClock(selected_call=3, value=0.0)
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def perceive(value):
            calls.append(value.camera.metadata.stamp.frame)
            if len(calls) == 1:
                entered.set()
                if not release.wait(WAIT_TIMEOUT_SECONDS):
                    raise RuntimeError("test perception gate was not released")
            return observation(value)

        runtime = PerceptionRuntime(clock=clock, perception_callable=perceive)
        try:
            runtime.start()
            runtime.submit(snapshot(1, sequence=1))
            self.assert_event(entered, "first frame was not claimed")
            runtime.submit(snapshot(2, sequence=2))
            clock.advance(0.5)
            release.set()
            self.assert_event(
                clock.selected_call_entered,
                "aged pending frame was not claimed",
            )
            runtime.stop()
            metrics = runtime.metrics
            self.assertEqual(calls, [1])
            self.assertEqual(metrics.inputs_accepted, 2)
            self.assertEqual(metrics.inputs_processed, 1)
            self.assertEqual(metrics.stale_inputs_rejected, 1)
            self.assertEqual(metrics.pending_frames_coalesced, 0)
            self.assertEqual(metrics.pending_inputs_discarded, 0)
        finally:
            cleanup_runtime(runtime, release)


class DeadlineAndResultTest(RuntimeTestCase):
    def test_initial_result_is_absent_and_success_preserves_exact_evidence(self):
        clock = FakeClock(10.0)
        called = threading.Event()
        value = snapshot(
            42,
            sequence=7,
            sim_time=12.5,
            receive_monotonic=9.0,
            revision=99,
        )
        expected = observation(value)

        def perceive(received):
            self.assertIs(received, value)
            clock.advance(0.0625)
            called.set()
            return expected

        runtime = PerceptionRuntime(
            PerceptionRuntimeConfig(target_hz=8.0),
            clock=clock,
            perception_callable=perceive,
        )
        self.assertIsNone(runtime.latest_result)
        try:
            runtime.start()
            self.assertTrue(runtime.submit(value))
            self.assert_event(called, "perception did not run")
            runtime.stop()
            result = runtime.latest_result
            self.assertIsInstance(result, PerceptionRuntimeResult)
            self.assertIs(result.observation, expected)
            self.assertIs(result.source_stamp, value.camera.metadata.stamp)
            self.assertEqual(result.source_sequence, 7)
            self.assertEqual(result.snapshot_revision, 99)
            self.assertEqual(result.ingestion_monotonic, 10.0)
            self.assertEqual(result.processing_start_monotonic, 10.0)
            self.assertEqual(result.processing_finish_monotonic, 10.0625)
            self.assertEqual(result.pending_age_seconds, 0.0)
            self.assertEqual(result.processing_latency_seconds, 0.0625)
            with self.assertRaises(FrozenInstanceError):
                result.pending_age_seconds = 1.0
        finally:
            cleanup_runtime(runtime)

    def test_deadline_below_equal_and_above_period_have_exact_semantics(self):
        cases = (
            (0.0625, 0),
            (0.125, 0),
            (0.25, 1),
        )
        for latency, expected_misses in cases:
            with self.subTest(latency=latency):
                clock = FakeClock()
                called = threading.Event()
                expected = observation(snapshot())

                def perceive(value, latency=latency):
                    clock.advance(latency)
                    called.set()
                    return expected

                runtime = PerceptionRuntime(
                    PerceptionRuntimeConfig(target_hz=8),
                    clock=clock,
                    perception_callable=perceive,
                )
                try:
                    runtime.start()
                    runtime.submit(snapshot())
                    self.assert_event(called, "deadline case did not run")
                    runtime.stop()
                    self.assertEqual(runtime.metrics.deadline_misses, expected_misses)
                    self.assertEqual(
                        runtime.metrics.last_processing_latency_seconds,
                        latency,
                    )
                    self.assertIs(runtime.latest_result.observation, expected)
                finally:
                    cleanup_runtime(runtime)

    def test_latest_result_replacement_and_latency_metrics_are_atomic(self):
        clock = FakeClock()
        latencies = (0.25, 0.0625)
        calls = []

        def perceive(value):
            index = len(calls)
            calls.append(value.camera.metadata.stamp.frame)
            clock.advance(latencies[index])
            return observation(value)

        runtime = PerceptionRuntime(
            PerceptionRuntimeConfig(target_hz=8),
            clock=clock,
            perception_callable=perceive,
        )
        try:
            runtime.start()
            runtime.submit(snapshot(1, sequence=1))
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._perception_successes == 1,
                "first result was not published",
            )
            first_result = runtime.latest_result
            first_metrics = runtime.metrics
            runtime.submit(snapshot(2, sequence=2))
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._perception_successes == 2,
                "second result was not published",
            )
            second_result = runtime.latest_result
            runtime.stop()
            self.assertEqual(calls, [1, 2])
            self.assertIsNot(first_result, second_result)
            self.assertEqual(first_result.source_stamp.frame, 1)
            self.assertEqual(second_result.source_stamp.frame, 2)
            self.assertEqual(first_metrics.perception_successes, 1)
            self.assertEqual(runtime.metrics.perception_successes, 2)
            self.assertEqual(runtime.metrics.last_processing_latency_seconds, 0.0625)
            self.assertEqual(runtime.metrics.maximum_processing_latency_seconds, 0.25)
            self.assertEqual(runtime.metrics.deadline_misses, 1)
        finally:
            cleanup_runtime(runtime)

    def test_runtime_retains_no_snapshot_payload_or_result_history(self):
        expected_result_fields = (
            "observation",
            "source_stamp",
            "source_sequence",
            "snapshot_revision",
            "ingestion_monotonic",
            "processing_start_monotonic",
            "processing_finish_monotonic",
            "pending_age_seconds",
            "processing_latency_seconds",
        )
        self.assertEqual(
            tuple(field.name for field in fields(PerceptionRuntimeResult)),
            expected_result_fields,
        )
        names = set(PerceptionRuntime.__slots__)
        self.assertNotIn("_results", names)
        self.assertFalse(any("history" in name or "queue" in name for name in names))
        self.assertIn("_pending", names)
        self.assertIn("_latest_result", names)

    def test_restart_releases_previous_epoch_result_behaviorally(self):
        class WeakObservation(LaneObservation):
            pass

        retained = []

        def perceive(value):
            stamp = value.camera.metadata.stamp
            result = WeakObservation(
                snapshot_revision=value.revision,
                source_frame=stamp.frame,
                source_simulation_timestamp=stamp.sim_time,
                source_receive_monotonic=stamp.monotonic,
                state=LaneDetectionState.NOT_DETECTED,
                reason=LaneObservationReason.NO_PLAUSIBLE_BOUNDARY,
            )
            retained.append(weakref.ref(result))
            return result

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=perceive)
        try:
            runtime.start()
            self.assertTrue(runtime.submit(snapshot()))
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._inputs_processed == 1,
                "first epoch result was not published",
            )
            runtime.stop()
            self.assertIsNotNone(retained[0]())
            runtime.start()
            gc.collect()
            self.assertIsNone(retained[0]())
            self.assertIsNone(runtime.latest_result)
        finally:
            cleanup_runtime(runtime)

    def test_result_rejects_inconsistent_timing_evidence(self):
        value = snapshot()
        result = PerceptionRuntimeResult(
            observation=observation(value),
            source_stamp=value.camera.metadata.stamp,
            source_sequence=1,
            snapshot_revision=1,
            ingestion_monotonic=1.0,
            processing_start_monotonic=1.25,
            processing_finish_monotonic=1.375,
            pending_age_seconds=0.25,
            processing_latency_seconds=0.125,
        )
        with self.assertRaises(ValueError):
            replace(result, pending_age_seconds=0.5)
        with self.assertRaises(ValueError):
            replace(result, processing_latency_seconds=-0.125)
        with self.assertRaises(ValueError):
            replace(result, processing_finish_monotonic=float("nan"))

    def test_metrics_records_are_frozen_atomic_snapshots(self):
        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=observation)
        before = runtime.metrics
        self.assertIsInstance(before, PerceptionRuntimeMetrics)
        runtime.start()
        try:
            runtime.submit(snapshot())
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._inputs_processed == 1,
                "input was not processed",
            )
            after = runtime.metrics
            self.assertEqual(before.inputs_received, 0)
            self.assertEqual(after.inputs_received, 1)
            with self.assertRaises(FrozenInstanceError):
                after.inputs_received = 2
            for field in fields(after):
                value = getattr(after, field.name)
                if field.name.endswith((
                    "received", "accepted", "processed", "rejected", "dropped",
                    "coalesced", "discarded", "abandoned", "successes",
                    "failures", "misses",
                )):
                    self.assertGreaterEqual(value, 0, field.name)
        finally:
            cleanup_runtime(runtime)


class ConcurrencyTest(RuntimeTestCase):
    def test_perception_callable_runs_outside_the_runtime_lock(self):
        called = threading.Event()
        lock_was_free = []
        holder = {}

        def perceive(value):
            runtime = holder["runtime"]
            acquired = runtime._lock.acquire(blocking=False)
            lock_was_free.append(acquired)
            if acquired:
                runtime._lock.release()
            called.set()
            return observation(value)

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=perceive)
        holder["runtime"] = runtime
        try:
            runtime.start()
            runtime.submit(snapshot())
            self.assert_event(called, "perception did not run")
            runtime.stop()
            self.assertEqual(lock_was_free, [True])
        finally:
            cleanup_runtime(runtime)

    def test_blocked_perception_does_not_block_a_fast_producer(self):
        entered = threading.Event()
        release = threading.Event()
        producer_done = threading.Event()
        newest_entered = threading.Event()
        calls = []
        producer_results = []

        def perceive(value):
            frame = value.camera.metadata.stamp.frame
            calls.append(frame)
            if len(calls) == 1:
                entered.set()
                if not release.wait(WAIT_TIMEOUT_SECONDS):
                    raise RuntimeError("test perception gate was not released")
            else:
                newest_entered.set()
            return observation(value)

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=perceive)

        def produce():
            producer_results.extend(
                runtime.submit(snapshot(frame, sequence=frame))
                for frame in range(2, 2_001)
            )
            producer_done.set()

        producer = threading.Thread(target=produce, name="runtime-fast-producer")
        try:
            runtime.start()
            runtime.submit(snapshot(1, sequence=1))
            self.assert_event(entered, "perception did not block")
            producer.start()
            self.assert_event(
                producer_done,
                "producer could not submit while perception was blocked",
            )
            producer.join(WAIT_TIMEOUT_SECONDS)
            self.assertFalse(producer.is_alive())
            self.assertFalse(release.is_set())
            self.assertTrue(all(producer_results))
            self.assertTrue(runtime.has_pending_input)
            self.assertEqual(runtime.metrics.pending_frames_coalesced, 1_998)
            release.set()
            self.assert_event(newest_entered, "newest pending input did not run")
            runtime.stop()
            self.assertEqual(calls, [1, 2_000])
        finally:
            release.set()
            if producer.ident is not None:
                producer.join(WAIT_TIMEOUT_SECONDS)
            cleanup_runtime(runtime, release)

    def test_concurrent_out_of_order_producers_still_end_at_newest_identity(self):
        entered = threading.Event()
        release = threading.Event()
        barrier = threading.Barrier(3)
        calls = []

        def perceive(value):
            calls.append(value.camera.metadata.stamp.frame)
            if len(calls) == 1:
                entered.set()
                if not release.wait(WAIT_TIMEOUT_SECONDS):
                    raise RuntimeError("test perception gate was not released")
            return observation(value)

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=perceive)

        def submit_frame(frame):
            barrier.wait()
            runtime.submit(snapshot(frame, sequence=frame))

        producers = [
            threading.Thread(target=submit_frame, args=(100,)),
            threading.Thread(target=submit_frame, args=(101,)),
        ]
        try:
            runtime.start()
            runtime.submit(snapshot(1, sequence=1))
            self.assert_event(entered, "baseline was not claimed")
            for producer in producers:
                producer.start()
            barrier.wait()
            for producer in producers:
                producer.join(WAIT_TIMEOUT_SECONDS)
            release.set()
            self.assert_runtime_wait(
                runtime,
                lambda: (
                    runtime._latest_result is not None
                    and runtime._latest_result.source_stamp.frame == 101
                ),
                "concurrent submissions did not preserve newest identity",
            )
            runtime.stop()
            self.assertEqual(calls[-1], 101)
        finally:
            release.set()
            for producer in producers:
                if producer.ident is not None:
                    producer.join(WAIT_TIMEOUT_SECONDS)
            cleanup_runtime(runtime, release)


class FaultBehaviorTest(RuntimeTestCase):
    def test_infrastructure_faults_abandon_claimed_input_exactly_once(self):
        cases = (
            (
                "start_raise",
                (RuntimeError("start unavailable"),),
                "clock",
                False,
            ),
            ("start_regress", (-1.0,), "clock", False),
            ("start_nonfinite", (float("nan"),), "clock", False),
            (
                "finish_raise",
                (0.0, RuntimeError("finish unavailable")),
                "clock",
                False,
            ),
            ("finish_regress", (0.0, -1.0), "clock", False),
            ("result_construction", (0.0, 0.0), "result", True),
        )
        for name, actions, expected_stage, fail_result in cases:
            with self.subTest(name=name):
                runtime = PerceptionRuntime(
                    clock=ScriptedWorkerClock(*actions),
                    perception_callable=observation,
                )
                patcher = None
                if fail_result:
                    patcher = mock.patch.object(
                        runtime_module,
                        "PerceptionRuntimeResult",
                        side_effect=RuntimeError("result construction failed"),
                    )
                    patcher.start()
                try:
                    runtime.start()
                    self.assertTrue(runtime.submit(snapshot()))
                    self.assert_runtime_wait(
                        runtime,
                        lambda: runtime._state is PerceptionRuntimeState.FAULTED,
                        f"{name} did not fault the runtime",
                    )
                    worker = runtime._worker
                    worker.join(WAIT_TIMEOUT_SECONDS)
                    self.assertFalse(worker.is_alive())
                    metrics = runtime.metrics
                    self.assertEqual(metrics.inputs_accepted, 1)
                    self.assertEqual(metrics.inputs_abandoned, 1)
                    self.assertEqual(metrics.inputs_processed, 0)
                    self.assertEqual(metrics.stale_inputs_rejected, 0)
                    self.assertEqual(metrics.pending_frames_coalesced, 0)
                    self.assertEqual(metrics.pending_inputs_discarded, 0)
                    self.assertEqual(metrics.perception_successes, 0)
                    self.assertEqual(metrics.perception_failures, 0)
                    self.assertIsNone(metrics.last_processing_latency_seconds)
                    self.assertEqual(runtime.fault.stage, expected_stage)
                    self.assertIsNone(runtime.latest_result)
                    self.assert_accounting_partition(metrics)
                finally:
                    if patcher is not None:
                        patcher.stop()
                    cleanup_runtime(runtime)

    def test_infrastructure_fault_keeps_pending_discard_separate(self):
        entered = threading.Event()
        release = threading.Event()

        def perceive(value):
            entered.set()
            if not release.wait(WAIT_TIMEOUT_SECONDS):
                raise RuntimeError("test perception gate was not released")
            return observation(value)

        runtime = PerceptionRuntime(
            clock=FinishFailClock(), perception_callable=perceive
        )
        try:
            runtime.start()
            self.assertTrue(runtime.submit(snapshot(1, sequence=1)))
            self.assert_event(entered, "perception did not block")
            self.assertTrue(runtime.submit(snapshot(2, sequence=2)))
            release.set()
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._state is PerceptionRuntimeState.FAULTED,
                "finish clock did not fault",
            )
            metrics = runtime.metrics
            self.assertEqual(metrics.inputs_accepted, 2)
            self.assertEqual(metrics.inputs_abandoned, 1)
            self.assertEqual(metrics.pending_inputs_discarded, 1)
            self.assertEqual(metrics.pending_frames_coalesced, 0)
            self.assert_accounting_partition(metrics)
        finally:
            cleanup_runtime(runtime, release)

    def test_prior_success_then_infrastructure_fault_clears_last_latency(self):
        clock = ScriptedWorkerClock(
            0.0,
            0.125,
            0.125,
            RuntimeError("finish unavailable"),
        )
        runtime = PerceptionRuntime(clock=clock, perception_callable=observation)
        try:
            runtime.start()
            self.assertTrue(runtime.submit(snapshot(1, sequence=1)))
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._perception_successes == 1,
                "first success was not published",
            )
            first_result = runtime.latest_result
            self.assertEqual(runtime.metrics.last_processing_latency_seconds, 0.125)
            self.assertTrue(runtime.submit(snapshot(2, sequence=2)))
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._state is PerceptionRuntimeState.FAULTED,
                "second finish clock did not fault",
            )
            metrics = runtime.metrics
            self.assertIs(runtime.latest_result, first_result)
            self.assertEqual(metrics.inputs_processed, 1)
            self.assertEqual(metrics.perception_successes, 1)
            self.assertEqual(metrics.inputs_abandoned, 1)
            self.assertIsNone(metrics.last_processing_latency_seconds)
            self.assertEqual(metrics.maximum_processing_latency_seconds, 0.125)
            self.assertEqual(metrics.last_processed_source_stamp.frame, 1)
            self.assert_accounting_partition(metrics)
        finally:
            cleanup_runtime(runtime)

    def test_perception_exception_faults_with_exact_bounded_evidence(self):
        clock = FakeClock(4.0)

        def fail(_value):
            clock.advance(0.25)
            raise RuntimeError("lane inference failed")

        runtime = PerceptionRuntime(
            PerceptionRuntimeConfig(target_hz=8),
            clock=clock,
            perception_callable=fail,
        )
        try:
            runtime.start()
            runtime.submit(snapshot(7, sequence=3, revision=70))
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._state is PerceptionRuntimeState.FAULTED,
                "perception exception did not fault the runtime",
            )
            worker = runtime._worker
            worker.join(WAIT_TIMEOUT_SECONDS)
            self.assertFalse(worker.is_alive())
            fault = runtime.fault
            metrics = runtime.metrics
            self.assertIsInstance(fault, PerceptionRuntimeFault)
            self.assertEqual(fault.stage, "perception")
            self.assertEqual(fault.exception_type, "builtins.RuntimeError")
            self.assertEqual(fault.message, "lane inference failed")
            self.assertEqual(fault.source_stamp.frame, 7)
            self.assertEqual(fault.source_sequence, 3)
            self.assertEqual(fault.snapshot_revision, 70)
            self.assertEqual(fault.occurred_monotonic, 4.25)
            self.assertEqual(metrics.inputs_processed, 1)
            self.assertEqual(metrics.perception_failures, 1)
            self.assertEqual(metrics.perception_successes, 0)
            self.assertEqual(metrics.deadline_misses, 1)
            self.assertIsNone(runtime.latest_result)
            with self.assertRaises(FrozenInstanceError):
                fault.message = "changed"
        finally:
            cleanup_runtime(runtime)

    def test_perception_fault_discards_exactly_one_newer_pending_input(self):
        entered = threading.Event()
        release = threading.Event()

        def fail(_value):
            entered.set()
            if not release.wait(WAIT_TIMEOUT_SECONDS):
                raise RuntimeError("test perception gate was not released")
            raise ValueError("bad frame")

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=fail)
        try:
            runtime.start()
            runtime.submit(snapshot(1, sequence=1))
            self.assert_event(entered, "failing perception did not block")
            runtime.submit(snapshot(2, sequence=2))
            release.set()
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._state is PerceptionRuntimeState.FAULTED,
                "runtime did not fault",
            )
            metrics = runtime.metrics
            self.assertEqual(metrics.inputs_accepted, 2)
            self.assertEqual(metrics.inputs_processed, 1)
            self.assertEqual(metrics.perception_failures, 1)
            self.assertEqual(metrics.pending_inputs_discarded, 1)
            self.assertEqual(metrics.pending_frames_coalesced, 0)
            self.assertFalse(runtime.has_pending_input)
        finally:
            cleanup_runtime(runtime, release)

    def test_wrong_perception_return_type_is_a_perception_failure(self):
        runtime = PerceptionRuntime(
            clock=FakeClock(), perception_callable=lambda _value: object()
        )
        try:
            runtime.start()
            runtime.submit(snapshot())
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._state is PerceptionRuntimeState.FAULTED,
                "wrong return type did not fault",
            )
            self.assertEqual(runtime.fault.stage, "perception")
            self.assertIn("LaneObservation", runtime.fault.message)
            self.assertEqual(runtime.metrics.inputs_processed, 1)
            self.assertEqual(runtime.metrics.perception_failures, 1)
            self.assertIsNone(runtime.latest_result)
        finally:
            cleanup_runtime(runtime)

    def test_finish_clock_failure_is_a_clock_fault_not_a_perception_failure(self):
        clock = FinishFailClock()
        returned = threading.Event()

        def perceive(value):
            returned.set()
            return observation(value)

        runtime = PerceptionRuntime(clock=clock, perception_callable=perceive)
        try:
            runtime.start()
            runtime.submit(snapshot())
            self.assert_event(returned, "perception did not return")
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._state is PerceptionRuntimeState.FAULTED,
                "finish clock failure did not fault",
            )
            self.assertEqual(runtime.fault.stage, "clock")
            self.assertIn("finish clock unavailable", runtime.fault.message)
            self.assertEqual(runtime.metrics.inputs_processed, 0)
            self.assertEqual(runtime.metrics.inputs_abandoned, 1)
            self.assertEqual(runtime.metrics.perception_failures, 0)
            self.assertEqual(runtime.metrics.perception_successes, 0)
            self.assertIsNone(runtime.latest_result)
            self.assert_accounting_partition(runtime.metrics)
        finally:
            cleanup_runtime(runtime)

    def test_fault_message_is_bounded_and_unprintable_exception_is_safe(self):
        class UnprintableError(Exception):
            def __str__(self):
                raise RuntimeError("cannot render exception")

        cases = (
            (RuntimeError("x" * (STATUS_TEXT_LIMIT * 4)), "..."),
            (UnprintableError(), "<unprintable exception>"),
        )
        for error, expected_suffix in cases:
            with self.subTest(error=type(error).__name__):
                def fail(_value, error=error):
                    raise error

                runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=fail)
                try:
                    runtime.start()
                    runtime.submit(snapshot())
                    self.assert_runtime_wait(
                        runtime,
                        lambda: runtime._state is PerceptionRuntimeState.FAULTED,
                        "diagnostic exception did not fault safely",
                    )
                    message = runtime.fault.message
                    self.assertLessEqual(len(message), STATUS_TEXT_LIMIT)
                    self.assertTrue(message.endswith(expected_suffix), message)
                finally:
                    cleanup_runtime(runtime)

    def test_exception_type_is_bounded_and_hostile_metadata_cannot_kill_worker(self):
        long_error = type("E" * (STATUS_TEXT_LIMIT * 4), (Exception,), {})

        class HostileText:
            def __str__(self):
                raise RuntimeError("type metadata cannot render")

        class HostileMetadataError(Exception):
            pass

        HostileMetadataError.__module__ = HostileText()
        for error in (long_error("long type"), HostileMetadataError("ordinary message")):
            with self.subTest(error=type(error).__qualname__):
                def fail(_value, error=error):
                    raise error

                runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=fail)
                try:
                    runtime.start()
                    runtime.submit(snapshot())
                    self.assert_runtime_wait(
                        runtime,
                        lambda: runtime._state is PerceptionRuntimeState.FAULTED,
                        "fault evidence formatting killed the worker silently",
                    )
                    worker = runtime._worker
                    worker.join(WAIT_TIMEOUT_SECONDS)
                    self.assertFalse(worker.is_alive())
                    self.assertIsNotNone(runtime.fault)
                    self.assertIs(type(runtime.fault.exception_type), str)
                    self.assertLessEqual(
                        len(runtime.fault.exception_type), STATUS_TEXT_LIMIT
                    )
                    self.assertEqual(runtime.metrics.perception_failures, 1)
                finally:
                    cleanup_runtime(runtime)

    def test_prior_success_remains_latest_when_next_invocation_faults(self):
        calls = []

        def perceive(value):
            calls.append(value.camera.metadata.stamp.frame)
            if len(calls) == 2:
                raise RuntimeError("second call failed")
            return observation(value)

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=perceive)
        try:
            runtime.start()
            runtime.submit(snapshot(1, sequence=1))
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._perception_successes == 1,
                "first success was not published",
            )
            first = runtime.latest_result
            runtime.submit(snapshot(2, sequence=2))
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._state is PerceptionRuntimeState.FAULTED,
                "second invocation did not fault",
            )
            self.assertIs(runtime.latest_result, first)
            self.assertEqual(runtime.latest_result.source_stamp.frame, 1)
            self.assertEqual(runtime.metrics.inputs_processed, 2)
            self.assertEqual(runtime.metrics.perception_successes, 1)
            self.assertEqual(runtime.metrics.perception_failures, 1)
        finally:
            cleanup_runtime(runtime)


class UnexpectedWorkerExitTest(RuntimeTestCase):
    def test_perception_baseexception_faults_truthfully_and_requires_recovery(self):
        result = run_isolated_python(
            textwrap.dedent(
                """
                from perception.runtime import PerceptionRuntime, PerceptionRuntimeState
                from tests.test_perception_runtime import FakeClock, observation, snapshot

                class Fatal(BaseException):
                    pass

                calls = 0
                def perceive(value):
                    global calls
                    calls += 1
                    if calls == 1:
                        raise Fatal("perception fatal")
                    return observation(value)

                runtime = PerceptionRuntime(
                    clock=FakeClock(), perception_callable=perceive
                )
                runtime.start()
                assert runtime.submit(snapshot(1, sequence=1))
                worker = runtime._worker
                worker.join(5.0)
                assert not worker.is_alive()
                assert not runtime.worker_alive
                assert runtime.state is PerceptionRuntimeState.FAULTED
                assert runtime.fault.stage == "worker"
                assert runtime.fault.exception_type == "<uncaught BaseException>"
                metrics = runtime.metrics
                assert metrics.inputs_accepted == 1
                assert metrics.inputs_abandoned == 1
                assert metrics.pending_inputs_discarded == 0
                assert not runtime.submit(snapshot(2, sequence=2))
                assert runtime.metrics == metrics
                assert not runtime.has_pending_input
                try:
                    runtime.start()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("faulted runtime restarted without stop")
                runtime.stop()
                assert runtime.state is PerceptionRuntimeState.STOPPED
                assert runtime._worker is None
                runtime.start()
                assert runtime.submit(snapshot(2, sequence=1))
                with runtime._condition:
                    assert runtime._condition.wait_for(
                        lambda: runtime._inputs_processed == 1, 5.0
                    )
                runtime.stop()
                print("PERCEPTION_BASEEXCEPTION_RECOVERED")
                """
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Fatal: perception fatal", result.stderr)
        self.assertIn("PERCEPTION_BASEEXCEPTION_RECOVERED", result.stdout)

    def test_clock_baseexception_faults_truthfully_and_rejects_submission(self):
        result = run_isolated_python(
            textwrap.dedent(
                """
                import threading
                from perception.runtime import PerceptionRuntime, PerceptionRuntimeState
                from tests.test_perception_runtime import FakeClock, observation, snapshot

                class Fatal(BaseException):
                    pass

                class Clock:
                    def __call__(self):
                        if threading.current_thread().name == "PerceptionRuntime":
                            raise Fatal("clock fatal")
                        return 0.0

                runtime = PerceptionRuntime(
                    clock=Clock(), perception_callable=observation
                )
                runtime.start()
                assert runtime.submit(snapshot(1, sequence=1))
                worker = runtime._worker
                worker.join(5.0)
                assert not worker.is_alive()
                assert runtime.state is PerceptionRuntimeState.FAULTED
                assert not runtime.worker_alive
                assert runtime.metrics.inputs_abandoned == 1
                assert not runtime.submit(snapshot(2, sequence=2))
                assert not runtime.has_pending_input
                runtime.stop()
                runtime._clock = FakeClock()
                runtime.start()
                assert runtime.submit(snapshot(2, sequence=1))
                with runtime._condition:
                    assert runtime._condition.wait_for(
                        lambda: runtime._inputs_processed == 1, 5.0
                    )
                runtime.stop()
                print("CLOCK_BASEEXCEPTION_RECOVERED")
                """
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Fatal: clock fatal", result.stderr)
        self.assertIn("CLOCK_BASEEXCEPTION_RECOVERED", result.stdout)

    def test_baseexception_exit_racing_stop_and_pending_replacement(self):
        class Fatal(BaseException):
            pass

        entered = threading.Event()
        release = threading.Event()
        hook_called = threading.Event()
        stop_done = threading.Event()
        stop_errors = []

        def perceive(_value):
            entered.set()
            if not release.wait(WAIT_TIMEOUT_SECONDS):
                raise RuntimeError("test perception gate was not released")
            raise Fatal("fatal during stop")

        def stop_runtime():
            try:
                runtime.stop()
            except Exception as exc:
                stop_errors.append(exc)
            finally:
                stop_done.set()

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=perceive)
        stopper = threading.Thread(target=stop_runtime, name="fatal-stop-race")
        try:
            with mock.patch.object(
                threading, "excepthook", side_effect=lambda _args: hook_called.set()
            ):
                runtime.start()
                self.assertTrue(runtime.submit(snapshot(1, sequence=1)))
                self.assert_event(entered, "fatal perception did not block")
                self.assertTrue(runtime.submit(snapshot(2, sequence=2)))
                stopper.start()
                self.assert_runtime_wait(
                    runtime,
                    lambda: runtime._state is PerceptionRuntimeState.STOPPING,
                    "stop did not race the fatal worker",
                )
                release.set()
                self.assert_event(hook_called, "uncaught BaseException missed excepthook")
                self.assert_event(stop_done, "stop did not recover fatal worker")
                stopper.join(WAIT_TIMEOUT_SECONDS)
            metrics = runtime.metrics
            self.assertEqual(stop_errors, [])
            self.assertIs(runtime.state, PerceptionRuntimeState.STOPPED)
            self.assertFalse(runtime.worker_alive)
            self.assertEqual(metrics.inputs_accepted, 2)
            self.assertEqual(metrics.inputs_abandoned, 1)
            self.assertEqual(metrics.pending_inputs_discarded, 1)
            self.assertEqual(metrics.pending_frames_coalesced, 0)
            self.assertEqual(runtime.fault.stage, "worker")
            self.assert_accounting_partition(metrics)
        finally:
            release.set()
            if stopper.ident is not None:
                stopper.join(WAIT_TIMEOUT_SECONDS)
            cleanup_runtime(runtime, release)

    def test_unexpected_exit_retains_no_exception_traceback_or_payload(self):
        class Fatal(BaseException):
            pass

        class Canary:
            pass

        references = {}
        hook_called = threading.Event()
        entered = threading.Event()
        release = threading.Event()

        def fail(_value):
            entered.set()
            if not release.wait(WAIT_TIMEOUT_SECONDS):
                raise RuntimeError("test perception gate was not released")
            payload = Canary()
            error = Fatal("retention canary")
            error.payload = payload
            references["exception"] = weakref.ref(error)
            references["traceback_payload"] = weakref.ref(payload)
            raise error

        active_rgb_payload = Canary()
        pending_rgb_payload = Canary()
        references["active_rgb_payload"] = weakref.ref(active_rgb_payload)
        references["pending_rgb_payload"] = weakref.ref(pending_rgb_payload)
        active_value = snapshot(1, sequence=1)
        active_value = replace(
            active_value,
            camera=replace(active_value.camera, rgb_bytes=active_rgb_payload),
        )
        pending_value = snapshot(2, sequence=2)
        pending_value = replace(
            pending_value,
            camera=replace(pending_value.camera, rgb_bytes=pending_rgb_payload),
        )
        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=fail)
        original_excepthook = threading.excepthook
        threading.excepthook = lambda _args: hook_called.set()
        try:
            runtime.start()
            self.assertTrue(runtime.submit(active_value))
            self.assert_event(entered, "fatal perception did not block")
            self.assertTrue(runtime.submit(pending_value))
            active_value = None
            pending_value = None
            active_rgb_payload = None
            pending_rgb_payload = None
            release.set()
            worker = runtime._worker
            worker.join(WAIT_TIMEOUT_SECONDS)
            self.assertFalse(worker.is_alive())
            self.assert_event(hook_called, "uncaught BaseException missed excepthook")
        finally:
            release.set()
            threading.excepthook = original_excepthook
        self.assertIs(runtime.state, PerceptionRuntimeState.FAULTED)
        self.assertEqual(runtime.metrics.inputs_abandoned, 1)
        self.assertEqual(runtime.metrics.pending_inputs_discarded, 1)
        self.assert_accounting_partition(runtime.metrics)
        runtime.stop()
        runtime._perception_callable = observation
        runtime.start()
        runtime.stop()
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references.values()))


class AccountingInvariantTest(RuntimeTestCase):
    def test_terminal_metrics_partition_every_valid_input_coherently(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def perceive(value):
            calls.append(value.camera.metadata.stamp.frame)
            if len(calls) == 1:
                entered.set()
                if not release.wait(WAIT_TIMEOUT_SECONDS):
                    raise RuntimeError("test perception gate was not released")
            return observation(value)

        runtime = PerceptionRuntime(clock=FakeClock(), perception_callable=perceive)
        try:
            runtime.start()
            runtime.submit(snapshot(1, sequence=1))
            self.assert_event(entered, "baseline was not claimed")
            runtime.submit(snapshot(2, sequence=2))
            runtime.submit(snapshot(2, sequence=2, revision=20))
            runtime.submit(snapshot(1, sequence=1, revision=21))
            runtime.submit(snapshot(5, sequence=5))
            release.set()
            self.assert_runtime_wait(
                runtime,
                lambda: runtime._inputs_processed == 2,
                "terminal pending input was not processed",
            )
            runtime.stop()
            metrics = runtime.metrics
            self.assertEqual(
                metrics.inputs_received,
                metrics.inputs_accepted
                + metrics.duplicate_inputs_rejected
                + metrics.out_of_order_inputs_rejected,
            )
            self.assertEqual(
                metrics.inputs_accepted,
                metrics.inputs_processed
                + metrics.stale_inputs_rejected
                + metrics.pending_frames_coalesced
                + metrics.pending_inputs_discarded
                + metrics.inputs_abandoned,
            )
            self.assertEqual(
                metrics.inputs_processed,
                metrics.perception_successes + metrics.perception_failures,
            )
            self.assertEqual(metrics.source_frames_inferred_dropped, 2)
            self.assertEqual(metrics.pending_frames_coalesced, 1)
        finally:
            cleanup_runtime(runtime, release)


class ImportSafetyTest(RuntimeTestCase):
    def test_fresh_direct_runtime_import_has_no_side_effects(self):
        result = run_isolated_python(
            import_safety_probe_code("import perception.runtime")
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "IMPORT_SAFE")

    def test_fresh_package_import_has_no_side_effects(self):
        result = run_isolated_python(import_safety_probe_code("import perception"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "IMPORT_SAFE")

    def test_fresh_construction_creates_no_thread_or_clock_read(self):
        after_import = textwrap.dedent(
            """
            calls = []
            def clock():
                calls.append("clock")
                return 0.0
            runtime = perception.runtime.PerceptionRuntime(
                clock=clock,
                perception_callable=lambda value: value,
            )
            assert calls == []
            assert not runtime.worker_alive
            assert runtime.state is perception.runtime.PerceptionRuntimeState.STOPPED
            """
        )
        result = run_isolated_python(
            import_safety_probe_code(
                "import perception.runtime; import perception",
                after_import,
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "IMPORT_SAFE")

    def test_explicit_fake_start_stop_terminates_cleanly_without_heavy_imports(self):
        code = textwrap.dedent(
            """
            import os
            import sys
            import threading
            environment_before = dict(os.environ)
            threads_before = tuple(sorted(t.ident for t in threading.enumerate()))
            import perception.runtime as runtime_module
            from perception.state import LaneDetectionState, LaneObservation, LaneObservationReason
            from telemetry.freshness import OperationalHealth, SensorFreshness, SensorKind, SensorLifecycle, SensorReadiness, SensorStatus
            from telemetry.state import PIXEL_FORMAT_RGB8, CameraFrame, CameraFrameMetadata, SampleStamp, TelemetrySnapshot
            forbidden = {"numpy", "carla", "pygame", "cv2", "scipy"}
            assert not [n for n in sys.modules if n.partition(".")[0] in forbidden]
            clock_calls = []
            def clock():
                clock_calls.append(threading.current_thread().name)
                return 1.0
            called = threading.Event()
            def perceive(value):
                stamp = value.camera.metadata.stamp
                called.set()
                return LaneObservation(value.revision, stamp.frame, stamp.sim_time, stamp.monotonic, LaneDetectionState.NOT_DETECTED, LaneObservationReason.NO_PLAUSIBLE_BOUNDARY)
            stamp = SampleStamp(1, 0.125, 0.125)
            frame = CameraFrame(CameraFrameMetadata(stamp, 1, 1, 90.0, PIXEL_FORMAT_RGB8, 3), b"123")
            status = SensorStatus("rgb_camera", SensorKind.CONTINUOUS, SensorLifecycle.ATTACHED, SensorReadiness.READY, SensorFreshness.FRESH, OperationalHealth.HEALTHY, 1.0, 0.0, 0.1, 0.5, 1, None, 0)
            value = TelemetrySnapshot(1, 1.0, camera=frame, statuses=(status,))
            runtime = runtime_module.PerceptionRuntime(clock=clock, perception_callable=perceive)
            assert not runtime.worker_alive
            runtime.start()
            assert runtime.submit(value)
            assert called.wait(5.0)
            runtime.stop()
            assert runtime.state is runtime_module.PerceptionRuntimeState.STOPPED
            assert not runtime.worker_alive
            assert runtime.metrics.inputs_processed == 1
            assert len(clock_calls) == 3
            assert dict(os.environ) == environment_before
            assert tuple(sorted(t.ident for t in threading.enumerate())) == threads_before
            assert not [n for n in sys.modules if n.partition(".")[0] in forbidden]
            print("START_STOP_SAFE")
            """
        )
        result = run_isolated_python(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "START_STOP_SAFE")

    def test_runtime_source_has_no_queue_sleep_heavy_dependency_or_control(self):
        path = pathlib.Path(runtime_module.__file__)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.partition(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.partition(".")[0])
        self.assertTrue(
            imported_roots.isdisjoint({"numpy", "carla", "pygame", "cv2", "scipy"})
        )
        self.assertNotIn("queue.Queue", source)
        self.assertNotIn("deque", source)
        self.assertNotIn("sleep(", source)
        for control_term in (
            "VehicleControl",
            "apply_control",
            "spawn_actor",
            "load_world",
            "autopilot",
            "throttle",
            "brake",
        ):
            self.assertNotIn(control_term, source)

    def test_runtime_module_globals_exclude_forbidden_dependencies(self):
        names = set(runtime_module.__dict__)
        for forbidden in ("numpy", "carla", "pygame", "cv2", "scipy"):
            self.assertNotIn(forbidden, names)


if __name__ == "__main__":
    unittest.main()
