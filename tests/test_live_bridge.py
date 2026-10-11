"""Focused CARLA-free tests for the bounded live perception bridge."""

from __future__ import annotations

import ast
import gc
import os
import pathlib
import subprocess
import sys
import textwrap
import threading
import time
import unittest
import weakref
from dataclasses import FrozenInstanceError

import perception.live_bridge as bridge_module
from perception.live_bridge import (
    MAX_BRIDGE_DRAIN_TIMEOUT_SECONDS,
    PerceptionLiveBridge,
    PerceptionLiveBridgeState,
)
from telemetry.state import (
    PIXEL_FORMAT_RGB8,
    CameraFrame,
    CameraFrameMetadata,
    SampleStamp,
    TelemetrySnapshot,
)


WAIT_SECONDS = 5.0


def snapshot(
    frame: int = 1,
    *,
    sim_time: float | None = None,
    received: float | None = None,
    revision: int | None = None,
    payload: object = b"\x00\x00\x00",
) -> TelemetrySnapshot:
    simulation_timestamp = frame / 10.0 if sim_time is None else sim_time
    receive_timestamp = simulation_timestamp if received is None else received
    value_revision = frame if revision is None else revision
    stamp = SampleStamp(frame, simulation_timestamp, receive_timestamp)
    metadata = CameraFrameMetadata(
        stamp=stamp,
        width_px=1,
        height_px=1,
        horizontal_fov_deg=90.0,
        pixel_format=PIXEL_FORMAT_RGB8,
        row_stride_bytes=3,
    )
    return TelemetrySnapshot(
        revision=value_revision,
        captured_monotonic=receive_timestamp,
        camera=CameraFrame(metadata=metadata, rgb_bytes=payload),
    )


class ConstantClock:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value


class SequenceClock:
    def __init__(self, *values: object) -> None:
        self._values = iter(values)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return next(self._values)


class StubRuntime:
    def __init__(self, handler=None) -> None:
        self.calls = 0
        self._handler = handler or (lambda _snapshot: True)

    def submit(self, value):
        self.calls += 1
        return self._handler(value)


class CountingRuntime:
    """A bounded fake which deliberately retains no submitted values."""

    def __init__(self, result: bool = True) -> None:
        self.calls = 0
        self.result = result

    def submit(self, _value):
        self.calls += 1
        return self.result


class LiveBridgeConstructionAndSubmissionTest(unittest.TestCase):
    def test_construction_is_open_and_reads_neither_clock_nor_runtime(self):
        clock = ConstantClock(1.0)
        runtime = CountingRuntime()
        threads_before = tuple(thread.ident for thread in threading.enumerate())

        bridge = PerceptionLiveBridge(runtime, clock=clock)

        self.assertIs(bridge.state, PerceptionLiveBridgeState.OPEN)
        self.assertEqual(clock.calls, 0)
        self.assertEqual(runtime.calls, 0)
        self.assertEqual(
            tuple(thread.ident for thread in threading.enumerate()), threads_before
        )
        self.assertEqual(bridge.metrics.submissions_received, 0)
        self.assertIsNone(bridge.fault)

    def test_runtime_accept_and_reject_are_normal_bounded_outcomes(self):
        outcomes = iter((True, False))
        runtime = StubRuntime(lambda _value: next(outcomes))
        clock = SequenceClock(1.25, 2.75)
        bridge = PerceptionLiveBridge(runtime, clock=clock)

        self.assertTrue(bridge.submit(snapshot(1, received=1.0)))
        self.assertFalse(bridge.submit(snapshot(2, received=2.0)))

        metrics = bridge.metrics
        self.assertIs(bridge.state, PerceptionLiveBridgeState.OPEN)
        self.assertEqual(metrics.submissions_received, 2)
        self.assertEqual(metrics.submissions_admitted, 2)
        self.assertEqual(metrics.runtime_submissions_accepted, 1)
        self.assertEqual(metrics.runtime_submissions_rejected, 1)
        self.assertEqual(metrics.submission_failures, 0)
        self.assertEqual(metrics.current_in_flight, 0)
        self.assertEqual(metrics.maximum_in_flight, 1)
        self.assertEqual(metrics.last_callback_to_submit_latency_seconds, 0.75)
        self.assertEqual(metrics.maximum_callback_to_submit_latency_seconds, 0.75)
        self.assertEqual(metrics.last_submitted_source_stamp.frame, 2)
        self.assertIsNone(bridge.fault)

    def test_reentrant_runtime_submission_cannot_deadlock_bridge_lock(self):
        clock = SequenceClock(2.0, 3.0)
        second = snapshot(2, received=0.0)
        reentered = False
        bridge = None

        def submit_reentrantly(_value):
            nonlocal reentered
            if not reentered:
                reentered = True
                assert bridge is not None
                self.assertTrue(bridge.submit(second))
            return True

        runtime = StubRuntime(submit_reentrantly)
        bridge = PerceptionLiveBridge(runtime, clock=clock)

        self.assertTrue(bridge.submit(snapshot(1, received=0.0)))

        metrics = bridge.metrics
        self.assertEqual(runtime.calls, 2)
        self.assertEqual(metrics.submissions_admitted, 2)
        self.assertEqual(metrics.runtime_submissions_accepted, 2)
        self.assertEqual(metrics.maximum_in_flight, 2)
        self.assertEqual(metrics.current_in_flight, 0)

    def test_invalid_snapshot_is_contained_and_fail_closes_without_clock_read(self):
        runtime = CountingRuntime()
        clock = ConstantClock(1.0)
        bridge = PerceptionLiveBridge(runtime, clock=clock)

        self.assertFalse(bridge.submit(object()))

        self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)
        self.assertEqual(runtime.calls, 0)
        self.assertEqual(clock.calls, 0)
        self.assertEqual(bridge.metrics.submission_failures, 1)
        self.assertEqual(bridge.metrics.current_in_flight, 0)
        self.assertEqual(bridge.fault.stage, "snapshot")

    def test_non_boolean_runtime_result_is_an_infrastructure_fault(self):
        runtime = StubRuntime(lambda _value: 1)
        clock = ConstantClock(1.0)
        bridge = PerceptionLiveBridge(runtime, clock=clock)

        self.assertFalse(bridge.submit(snapshot(received=0.0)))

        self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)
        self.assertEqual(clock.calls, 0)
        self.assertEqual(bridge.metrics.submission_failures, 1)
        self.assertEqual(bridge.fault.stage, "runtime_result")


class LiveBridgeLockSafetyTest(unittest.TestCase):
    def test_runtime_submit_clock_and_snapshot_access_all_run_outside_lock(self):
        probes = []
        bridge = None

        class ProbingRuntime:
            def submit(self, _value):
                assert bridge is not None
                acquired = bridge._lock.acquire(blocking=False)
                probes.append(("runtime", acquired))
                if acquired:
                    bridge._lock.release()
                return True

        class ProbingClock:
            def __call__(self):
                assert bridge is not None
                acquired = bridge._lock.acquire(blocking=False)
                probes.append(("clock", acquired))
                if acquired:
                    bridge._lock.release()
                return 1.0

        class ProbingSnapshot(TelemetrySnapshot):
            def __getattribute__(self, name):
                if name in {"revision", "camera"} and bridge is not None:
                    acquired = bridge._lock.acquire(blocking=False)
                    probes.append(("snapshot", acquired))
                    if acquired:
                        bridge._lock.release()
                return super().__getattribute__(name)

        original = snapshot(received=0.0)
        value = ProbingSnapshot(
            revision=original.revision,
            captured_monotonic=original.captured_monotonic,
            camera=original.camera,
        )
        bridge = PerceptionLiveBridge(ProbingRuntime(), clock=ProbingClock())

        self.assertTrue(bridge.submit(value))
        self.assertTrue(probes)
        self.assertTrue(all(acquired for _stage, acquired in probes), probes)
        self.assertEqual({stage for stage, _acquired in probes}, {
            "snapshot", "runtime", "clock"
        })

    def test_hostile_exception_formatting_is_outside_lock_and_reentry_is_closed(self):
        lock_probe = []
        reentrant_results = []
        bridge = None

        class HostileError(Exception):
            def __str__(self):
                assert bridge is not None
                acquired = bridge._lock.acquire(blocking=False)
                lock_probe.append(acquired)
                if acquired:
                    bridge._lock.release()
                reentrant_results.append(bridge.submit(object()))
                return "hostile detail"

        def fail(_value):
            raise HostileError()

        bridge = PerceptionLiveBridge(
            StubRuntime(fail), clock=lambda: 1.0
        )

        self.assertFalse(bridge.submit(snapshot(received=0.0)))

        self.assertEqual(lock_probe, [True])
        self.assertEqual(reentrant_results, [False])
        metrics = bridge.metrics
        self.assertEqual(metrics.submissions_received, 2)
        self.assertEqual(metrics.submissions_admitted, 1)
        self.assertEqual(metrics.submissions_rejected_closed, 1)
        self.assertEqual(bridge.fault.message, "hostile detail")

    def test_closed_rejection_does_not_touch_snapshot_runtime_or_clock(self):
        runtime = CountingRuntime()
        clock = ConstantClock(10.0)
        bridge = PerceptionLiveBridge(runtime, clock=clock)
        bridge.close(0.0)

        class Untouchable:
            def __getattribute__(self, _name):
                raise AssertionError("closed bridge inspected rejected input")

        self.assertFalse(bridge.submit(Untouchable()))
        self.assertEqual(runtime.calls, 0)
        self.assertEqual(clock.calls, 0)
        self.assertEqual(bridge.metrics.submissions_rejected_closed, 1)


class LiveBridgeLifecycleTest(unittest.TestCase):
    def test_close_before_submission_is_idempotent_and_one_shot(self):
        runtime = CountingRuntime()
        bridge = PerceptionLiveBridge(runtime, clock=ConstantClock(1.0))

        bridge.close(0.0)
        bridge.close(0.0)

        self.assertIs(bridge.state, PerceptionLiveBridgeState.CLOSED)
        self.assertFalse(bridge.submit(snapshot(received=0.0)))
        self.assertEqual(runtime.calls, 0)
        self.assertEqual(bridge.metrics.submissions_rejected_closed, 1)

    def test_explicit_admission_close_and_drain_are_separate_cleanup_stages(self):
        bridge = PerceptionLiveBridge(
            CountingRuntime(), clock=ConstantClock(1.0)
        )

        bridge.close_admission()
        self.assertIs(bridge.state, PerceptionLiveBridgeState.CLOSING)
        self.assertFalse(bridge.submit(snapshot(received=0.0)))
        bridge.drain(0.0)

        self.assertIs(bridge.state, PerceptionLiveBridgeState.CLOSED)
        self.assertEqual(bridge.metrics.submissions_rejected_closed, 1)

    def test_invalid_close_timeout_leaves_open_state_and_metrics_unchanged(self):
        for invalid in (
            -0.001,
            float("inf"),
            float("nan"),
            True,
            MAX_BRIDGE_DRAIN_TIMEOUT_SECONDS + 0.001,
        ):
            with self.subTest(invalid=invalid):
                bridge = PerceptionLiveBridge(
                    CountingRuntime(), clock=ConstantClock(1.0)
                )
                before = bridge.metrics
                with self.assertRaises(ValueError):
                    bridge.close(invalid)
                self.assertIs(bridge.state, PerceptionLiveBridgeState.OPEN)
                self.assertEqual(bridge.metrics, before)

    def test_timeout_preserves_closing_and_later_close_retry_succeeds(self):
        entered = threading.Event()
        release = threading.Event()

        def block(_value):
            entered.set()
            if not release.wait(WAIT_SECONDS):
                raise RuntimeError("test gate was not released")
            return True

        bridge = PerceptionLiveBridge(
            StubRuntime(block), clock=ConstantClock(1.0)
        )
        submit_result = []
        submitter = threading.Thread(
            target=lambda: submit_result.append(
                bridge.submit(snapshot(received=0.0))
            ),
            name="bridge-submit-timeout-test",
        )
        submitter.start()
        self.assertTrue(entered.wait(WAIT_SECONDS))

        with self.assertRaises(TimeoutError):
            bridge.close(0.01)
        self.assertIs(bridge.state, PerceptionLiveBridgeState.CLOSING)
        self.assertEqual(bridge.metrics.current_in_flight, 1)
        self.assertFalse(bridge.submit(snapshot(2, received=0.0)))

        release.set()
        submitter.join(WAIT_SECONDS)
        self.assertFalse(submitter.is_alive())
        self.assertEqual(submit_result, [True])
        self.assertEqual(bridge.metrics.current_in_flight, 0)
        self.assertIs(bridge.state, PerceptionLiveBridgeState.CLOSING)

        bridge.close(0.0)
        self.assertIs(bridge.state, PerceptionLiveBridgeState.CLOSED)

    def test_close_waiter_is_released_after_runtime_baseexception(self):
        class Fatal(BaseException):
            pass

        entered = threading.Event()
        release = threading.Event()
        submit_errors = []
        close_errors = []

        def fail_fatally(_value):
            entered.set()
            if not release.wait(WAIT_SECONDS):
                raise RuntimeError("test gate was not released")
            raise Fatal("fatal runtime exit")

        bridge = PerceptionLiveBridge(
            StubRuntime(fail_fatally), clock=ConstantClock(1.0)
        )

        def submit_value():
            try:
                bridge.submit(snapshot(received=0.0))
            except BaseException as exc:
                submit_errors.append(exc)

        def close_bridge():
            try:
                bridge.close(WAIT_SECONDS)
            except Exception as exc:
                close_errors.append(exc)

        submitter = threading.Thread(target=submit_value, name="fatal-submitter")
        closer = threading.Thread(target=close_bridge, name="fatal-drainer")
        submitter.start()
        self.assertTrue(entered.wait(WAIT_SECONDS))
        closer.start()
        deadline = time.monotonic() + WAIT_SECONDS
        while bridge.state is not PerceptionLiveBridgeState.CLOSING:
            if time.monotonic() >= deadline:
                self.fail("close did not reach CLOSING")
            time.sleep(0.001)
        release.set()
        submitter.join(WAIT_SECONDS)
        closer.join(WAIT_SECONDS)

        self.assertFalse(submitter.is_alive())
        self.assertFalse(closer.is_alive())
        self.assertEqual(len(submit_errors), 1)
        self.assertIsInstance(submit_errors[0], Fatal)
        self.assertEqual(close_errors, [])
        self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)
        self.assertEqual(bridge.metrics.submission_failures, 1)
        self.assertEqual(bridge.metrics.current_in_flight, 0)
        self.assertEqual(bridge.fault.exception_type, "<uncaught BaseException>")


class LiveBridgeFaultAndClockTest(unittest.TestCase):
    def test_faulted_state_is_sticky_through_every_cleanup_entry(self):
        faulting = StubRuntime(
            lambda _value: (_ for _ in ()).throw(RuntimeError("submit failed"))
        )
        bridge = PerceptionLiveBridge(faulting, clock=ConstantClock(1.0))

        self.assertFalse(bridge.submit(snapshot(received=0.0)))
        original_fault = bridge.fault
        self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)

        bridge.close_admission()
        self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)
        bridge.drain(0.0)
        self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)
        bridge.close(0.0)
        self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)

        self.assertIs(bridge.fault, original_fault)
        self.assertEqual(bridge.metrics.current_in_flight, 0)
        self.assertFalse(bridge.submit(snapshot(2, received=0.0)))
        self.assertEqual(bridge.metrics.submissions_rejected_closed, 1)

    def test_concurrent_fault_and_active_submission_drain_without_state_erasure(self):
        entered = threading.Event()
        release = threading.Event()

        def submit(value):
            frame = value.camera.metadata.stamp.frame
            if frame == 1:
                entered.set()
                if not release.wait(WAIT_SECONDS):
                    raise RuntimeError("active submission was not released")
                return True
            raise RuntimeError("concurrent infrastructure fault")

        bridge = PerceptionLiveBridge(
            StubRuntime(submit), clock=ConstantClock(10.0)
        )
        active_results = []
        active = threading.Thread(
            target=lambda: active_results.append(
                bridge.submit(snapshot(1, received=1.0))
            )
        )
        active.start()
        self.assertTrue(entered.wait(WAIT_SECONDS))
        self.assertFalse(bridge.submit(snapshot(2, received=2.0)))
        original_fault = bridge.fault
        self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)
        self.assertEqual(bridge.metrics.current_in_flight, 1)

        drain_errors = []

        def drain():
            try:
                bridge.drain(WAIT_SECONDS)
            except BaseException as exc:
                drain_errors.append(type(exc).__name__)

        waiter = threading.Thread(target=drain)
        waiter.start()
        bridge.close_admission()
        self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)
        release.set()
        active.join(WAIT_SECONDS)
        waiter.join(WAIT_SECONDS)

        self.assertFalse(active.is_alive())
        self.assertFalse(waiter.is_alive())
        self.assertEqual(drain_errors, [])
        self.assertEqual(active_results, [True])
        self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)
        self.assertIs(bridge.fault, original_fault)
        metrics = bridge.metrics
        self.assertEqual(metrics.submissions_received, 2)
        self.assertEqual(metrics.submissions_admitted, 2)
        self.assertEqual(metrics.runtime_submissions_accepted, 1)
        self.assertEqual(metrics.submission_failures, 1)
        self.assertEqual(metrics.current_in_flight, 0)

    def test_faulted_drain_timeout_and_retry_preserve_original_evidence(self):
        entered = threading.Event()
        release = threading.Event()

        def submit(value):
            if value.camera.metadata.stamp.frame == 1:
                entered.set()
                if not release.wait(WAIT_SECONDS):
                    raise RuntimeError("active submission was not released")
                return True
            raise RuntimeError("original fault")

        bridge = PerceptionLiveBridge(
            StubRuntime(submit), clock=ConstantClock(10.0)
        )
        active = threading.Thread(
            target=lambda: bridge.submit(snapshot(1, received=1.0))
        )
        active.start()
        self.assertTrue(entered.wait(WAIT_SECONDS))
        self.assertFalse(bridge.submit(snapshot(2, received=2.0)))
        original_fault = bridge.fault

        with self.assertRaises(TimeoutError):
            bridge.drain(0.0)
        self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)
        self.assertIs(bridge.fault, original_fault)
        self.assertEqual(bridge.metrics.current_in_flight, 1)

        release.set()
        active.join(WAIT_SECONDS)
        self.assertFalse(active.is_alive())
        bridge.drain(0.0)
        bridge.close(0.0)

        self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)
        self.assertIs(bridge.fault, original_fault)
        self.assertEqual(bridge.metrics.current_in_flight, 0)

    def test_ordinary_runtime_exception_is_contained_bounded_and_fail_closed(self):
        class Payload:
            pass

        class Failure(Exception):
            pass

        references = []

        def fail(_value):
            payload = Payload()
            references.append(weakref.ref(payload))
            error = Failure("x" * 10_000)
            error.payload = payload
            raise error

        runtime = StubRuntime(fail)
        bridge = PerceptionLiveBridge(runtime, clock=ConstantClock(1.0))

        self.assertFalse(bridge.submit(snapshot(received=0.0)))
        self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)
        self.assertFalse(bridge.submit(snapshot(2, received=0.0)))

        fault = bridge.fault
        metrics = bridge.metrics
        self.assertEqual(fault.stage, "runtime_submit")
        self.assertLessEqual(len(fault.exception_type), 192)
        self.assertLessEqual(len(fault.message), 192)
        self.assertEqual(metrics.submission_failures, 1)
        self.assertEqual(metrics.submissions_rejected_closed, 1)
        self.assertEqual(metrics.current_in_flight, 0)
        gc.collect()
        self.assertIsNone(references[0]())

    def test_direct_baseexception_propagates_after_fail_close_and_accounting(self):
        class Fatal(BaseException):
            pass

        def fail(_value):
            raise Fatal("must propagate")

        bridge = PerceptionLiveBridge(
            StubRuntime(fail), clock=ConstantClock(1.0)
        )

        with self.assertRaises(Fatal):
            bridge.submit(snapshot(received=0.0))

        self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)
        self.assertEqual(bridge.metrics.submission_failures, 1)
        self.assertEqual(bridge.metrics.current_in_flight, 0)
        self.assertEqual(bridge.fault.exception_type, "<uncaught BaseException>")
        original_fault = bridge.fault
        bridge.close_admission()
        bridge.drain(0.0)
        bridge.close(0.0)
        self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)
        self.assertIs(bridge.fault, original_fault)

    def test_nonfinite_and_regressing_completion_clocks_fault_deterministically(self):
        cases = (
            (float("nan"), "finite real number"),
            (float("inf"), "finite real number"),
            (0.5, "regressed"),
        )
        for completion, detail in cases:
            with self.subTest(completion=completion):
                bridge = PerceptionLiveBridge(
                    CountingRuntime(), clock=ConstantClock(completion)
                )
                self.assertFalse(bridge.submit(snapshot(received=1.0)))
                self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)
                self.assertEqual(bridge.metrics.runtime_submissions_accepted, 1)
                self.assertEqual(bridge.metrics.submission_failures, 0)
                self.assertEqual(
                    bridge.metrics.submissions_admitted,
                    bridge.metrics.runtime_submissions_accepted
                    + bridge.metrics.runtime_submissions_rejected
                    + bridge.metrics.submission_failures,
                )
                self.assertIsNone(
                    bridge.metrics.last_callback_to_submit_latency_seconds
                )
                self.assertEqual(bridge.fault.stage, "clock")
                self.assertIn(detail, bridge.fault.message)

    def test_post_accept_clock_faults_keep_one_terminal_partition_and_wake_waiters(self):
        class CompletionClock:
            def __init__(self, outcome):
                self.outcome = outcome
                self.entered = threading.Event()
                self.release = threading.Event()

            def __call__(self):
                self.entered.set()
                if not self.release.wait(WAIT_SECONDS):
                    raise RuntimeError("completion clock was not released")
                if isinstance(self.outcome, Exception):
                    raise self.outcome
                return self.outcome

        cases = (
            ("exception", RuntimeError("completion clock failed")),
            ("nan", float("nan")),
            ("positive_infinity", float("inf")),
            ("negative_infinity", float("-inf")),
        )
        for label, outcome in cases:
            with self.subTest(label=label):
                clock = CompletionClock(outcome)
                runtime = CountingRuntime()
                bridge = PerceptionLiveBridge(runtime, clock=clock)
                submit_results = []
                waiter_errors = []
                submitter = threading.Thread(
                    target=lambda: submit_results.append(
                        bridge.submit(snapshot(received=0.0))
                    )
                )

                def drain():
                    try:
                        bridge.drain(WAIT_SECONDS)
                    except BaseException as exc:
                        waiter_errors.append(type(exc).__name__)

                submitter.start()
                self.assertTrue(clock.entered.wait(WAIT_SECONDS))
                waiter = threading.Thread(target=drain)
                waiter.start()
                clock.release.set()
                submitter.join(WAIT_SECONDS)
                waiter.join(WAIT_SECONDS)

                self.assertFalse(submitter.is_alive())
                self.assertFalse(waiter.is_alive())
                self.assertEqual(submit_results, [False])
                self.assertEqual(waiter_errors, [])
                self.assertEqual(runtime.calls, 1)
                self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)
                self.assertFalse(bridge.submit(snapshot(2, received=0.0)))
                metrics = bridge.metrics
                self.assertEqual(metrics.submissions_admitted, 1)
                self.assertEqual(metrics.runtime_submissions_accepted, 1)
                self.assertEqual(metrics.runtime_submissions_rejected, 0)
                self.assertEqual(metrics.submission_failures, 0)
                self.assertEqual(
                    metrics.submissions_admitted,
                    metrics.runtime_submissions_accepted
                    + metrics.runtime_submissions_rejected
                    + metrics.submission_failures,
                )
                self.assertEqual(metrics.current_in_flight, 0)
                self.assertEqual(metrics.submissions_rejected_closed, 1)
                self.assertEqual(bridge.fault.stage, "clock")
                self.assertLessEqual(len(bridge.fault.exception_type), 192)
                self.assertLessEqual(len(bridge.fault.message), 192)
                original_fault = bridge.fault
                bridge.close_admission()
                bridge.drain(0.0)
                bridge.close(0.0)
                self.assertIs(bridge.state, PerceptionLiveBridgeState.FAULTED)
                self.assertIs(bridge.fault, original_fault)

    def test_metrics_and_fault_are_frozen_scalar_records(self):
        bridge = PerceptionLiveBridge(
            StubRuntime(lambda _value: (_ for _ in ()).throw(RuntimeError("bad"))),
            clock=ConstantClock(1.0),
        )
        bridge.submit(snapshot(received=0.0))

        with self.assertRaises(FrozenInstanceError):
            bridge.metrics.submissions_received = 99
        with self.assertRaises(FrozenInstanceError):
            bridge.fault.message = "changed"


class LiveBridgeBoundednessTest(unittest.TestCase):
    def test_many_submissions_keep_only_scalar_metrics_and_no_history(self):
        runtime = CountingRuntime()
        bridge = PerceptionLiveBridge(runtime, clock=ConstantClock(1.0))
        value = snapshot(received=1.0)

        for _index in range(10_000):
            self.assertTrue(bridge.submit(value))

        metrics = bridge.metrics
        self.assertEqual(runtime.calls, 10_000)
        self.assertEqual(metrics.submissions_received, 10_000)
        self.assertEqual(metrics.submissions_admitted, 10_000)
        self.assertEqual(metrics.runtime_submissions_accepted, 10_000)
        self.assertEqual(metrics.current_in_flight, 0)
        self.assertEqual(metrics.maximum_in_flight, 1)
        self.assertIsNone(bridge.fault)
        for slot in bridge.__slots__:
            retained = getattr(bridge, slot)
            self.assertNotIsInstance(retained, (list, dict, set), slot)

    def test_bridge_does_not_retain_submitted_snapshot_or_mutable_payload(self):
        class Payload:
            pass

        payload = Payload()
        payload_reference = weakref.ref(payload)
        value = snapshot(received=0.0, payload=payload)
        bridge = PerceptionLiveBridge(
            CountingRuntime(), clock=ConstantClock(1.0)
        )

        self.assertTrue(bridge.submit(value))
        value = None
        payload = None
        gc.collect()

        self.assertIsNone(payload_reference())
        self.assertEqual(bridge.metrics.last_submitted_source_stamp.frame, 1)

    def test_concurrent_submissions_have_exact_current_and_maximum_in_flight(self):
        condition = threading.Condition()
        entered = 0
        release = threading.Event()

        def block(_value):
            nonlocal entered
            with condition:
                entered += 1
                condition.notify_all()
            if not release.wait(WAIT_SECONDS):
                raise RuntimeError("test gate was not released")
            return True

        bridge = PerceptionLiveBridge(
            StubRuntime(block), clock=ConstantClock(1.0)
        )
        threads = [
            threading.Thread(
                target=bridge.submit,
                args=(snapshot(index + 1, received=0.0),),
                name=f"concurrent-bridge-submit-{index}",
            )
            for index in range(4)
        ]
        for thread in threads:
            thread.start()
        with condition:
            self.assertTrue(
                condition.wait_for(lambda: entered == len(threads), WAIT_SECONDS)
            )
        self.assertEqual(bridge.metrics.current_in_flight, len(threads))
        self.assertEqual(bridge.metrics.maximum_in_flight, len(threads))

        release.set()
        for thread in threads:
            thread.join(WAIT_SECONDS)
            self.assertFalse(thread.is_alive())

        metrics = bridge.metrics
        self.assertEqual(metrics.current_in_flight, 0)
        self.assertEqual(metrics.runtime_submissions_accepted, len(threads))
        self.assertEqual(metrics.submission_failures, 0)


class LiveBridgeImportSafetyTest(unittest.TestCase):
    def test_fresh_import_is_heavy_dependency_clock_and_thread_safe(self):
        code = textwrap.dedent(
            """
            import os
            import sys
            import threading
            environment_before = dict(os.environ)
            threads_before = tuple(sorted(t.ident for t in threading.enumerate()))
            import perception.live_bridge as module
            forbidden = {"numpy", "carla", "pygame", "cv2", "scipy"}
            assert not [n for n in sys.modules if n.partition(".")[0] in forbidden]
            assert dict(os.environ) == environment_before
            assert tuple(sorted(t.ident for t in threading.enumerate())) == threads_before
            calls = []
            class Runtime:
                def submit(self, value):
                    raise AssertionError("construction called runtime")
            bridge = module.PerceptionLiveBridge(
                Runtime(), clock=lambda: calls.append("clock")
            )
            assert calls == []
            assert bridge.state is module.PerceptionLiveBridgeState.OPEN
            assert tuple(sorted(t.ident for t in threading.enumerate())) == threads_before
            print("LIVE_BRIDGE_IMPORT_SAFE")
            """
        )
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=pathlib.Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=WAIT_SECONDS,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "LIVE_BRIDGE_IMPORT_SAFE")

    def test_source_has_no_worker_queue_history_or_heavy_dependency(self):
        path = pathlib.Path(bridge_module.__file__)
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
        self.assertNotIn("threading.Thread", source)
        self.assertNotIn("sleep(", source)
        for prohibited_slot in ("history", "snapshots", "frames", "results"):
            self.assertNotIn(f'"_{prohibited_slot}"', source)


if __name__ == "__main__":
    unittest.main()
