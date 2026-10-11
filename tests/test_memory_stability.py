"""Deterministic offline tests for Mission 4G.6 memory stability.

Every test here runs with fake native backends, fake PDH backends, fake
clocks, fake events, and fake samplers.  No test starts CARLA, sleeps on the
wall clock, or depends on a live process.
"""

import ast
import contextlib
import ctypes
import dataclasses
import inspect
import math
import sys
import threading
import time
import types
import unittest
import unittest.mock
from dataclasses import FrozenInstanceError
from pathlib import Path

from tests import integration_sensor_suite as harness
from tests import windows_memory_sampler as sampler_module
from tests.test_repeated_live_sensor_suite_validation import (
    successful_cycle_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLER_PATH = Path(sampler_module.__file__).resolve()

THRESHOLD_DEFAULTS = {
    harness.CARLA_MEMORY_WORKING_SET_MAX_PEAK_GROWTH_BYTES: "1000000000",
    harness.CARLA_MEMORY_WORKING_SET_MAX_FINAL_GROWTH_BYTES: "500000000",
    harness.CARLA_MEMORY_WORKING_SET_MAX_FINAL_GROWTH_PERCENT: "50",
    harness.CARLA_MEMORY_PRIVATE_COMMIT_MAX_PEAK_GROWTH_BYTES: "1000000000",
    harness.CARLA_MEMORY_PRIVATE_COMMIT_MAX_FINAL_GROWTH_BYTES: "500000000",
    harness.CARLA_MEMORY_PRIVATE_COMMIT_MAX_FINAL_GROWTH_PERCENT: "50",
}


def memory_environ(*, gate="1", soak="120", host=None, cycles=None, **extra):
    environ = {
        harness.CARLA_INTEGRATION: "1",
        harness.CARLA_INTEGRATION_SOAK_SECONDS: soak,
    }
    if host is not None:
        environ[harness.CARLA_HOST] = host
    if cycles is not None:
        environ[harness.CARLA_INTEGRATION_CYCLES] = cycles
    if gate is not None:
        environ[harness.CARLA_INTEGRATION_MEMORY_STABILITY] = gate
    environ.update(THRESHOLD_DEFAULTS)
    environ.update(extra)
    return environ


def parse_memory(environ):
    config = harness.parse_integration_config(environ)
    mode = harness.select_execution_mode(config)
    return harness.parse_memory_stability_config(environ, config, mode)


def limits(**overrides):
    values = dict(
        working_set_max_peak_growth_bytes=10**9,
        working_set_max_final_growth_bytes=10**9,
        working_set_max_final_growth_percent=10000.0,
        private_commit_max_peak_growth_bytes=10**9,
        private_commit_max_final_growth_bytes=10**9,
        private_commit_max_final_growth_percent=10000.0,
    )
    values.update(overrides)
    return harness.MemoryThresholds(**values)


def reading(working_set=1000, private_commit=2000, pagefile=None):
    return harness.ProcessCounterReading(
        working_set_bytes=working_set,
        private_commit_bytes=private_commit,
        pagefile_usage_bytes=(
            private_commit if pagefile is None else pagefile
        ),
    )


class FakeClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


class AutoAdvanceEvent:
    """Event fake whose timed waits advance a fake clock deterministically."""

    def __init__(self, clock, false_waits, *, extra_advance=None):
        self.clock = clock
        self.false_waits = false_waits
        self.extra_advance = dict(extra_advance or {})
        self.wait_timeouts = []
        self.calls = 0
        self._set = False

    def is_set(self):
        return self._set

    def set(self):
        self._set = True

    def wait(self, timeout=None):
        self.calls += 1
        self.wait_timeouts.append(timeout)
        if self._set or self.calls > self.false_waits:
            return True
        self.clock.value += timeout + self.extra_advance.get(self.calls, 0.0)
        return False


class ScriptedEvent:
    """Event fake driven by an explicit script of wait outcomes."""

    def __init__(self, script=()):
        self.script = list(script)
        self.wait_timeouts = []
        self._set = False

    def is_set(self):
        return self._set

    def set(self):
        self._set = True

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self._set:
            return True
        if not self.script:
            return True
        entry = self.script.pop(0)
        if callable(entry):
            return entry(timeout)
        return entry


class FunctionSampler:
    """Sampler fake producing readings (or exceptions) per call index."""

    def __init__(self, produce):
        self.produce = produce
        self.calls = 0
        self.closed = 0

    def sample(self):
        self.calls += 1
        result = self.produce(self.calls)
        if isinstance(result, BaseException):
            raise result
        return result

    def close(self):
        self.closed += 1


class FakeGpu:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.closed = 0

    def sample(self):
        self.calls += 1
        entry = self.script.pop(0) if self.script else self._last
        self._last = entry
        if isinstance(entry, BaseException):
            raise entry
        return entry

    def close(self):
        self.closed += 1


class FakeThread:
    def __init__(self, target):
        self.target = target
        self.started = 0
        self.joined = 0

    def start(self):
        self.started += 1
        self.target()

    def join(self, timeout=None):
        self.joined += 1
        self.join_timeout = timeout

    def is_alive(self):
        return False


def synchronous_native_calls(description, operation, timeout_seconds):
    """Deterministic in-thread stand-in for the bounded native executor."""
    return operation()


class ScriptedNativeExecutor:
    """Native-executor fake driven by per-call outcomes.

    Each entry is "run" (execute the operation synchronously) or "timeout"
    (simulate a native call that never returned).  Exhausted scripts run.
    """

    def __init__(self, script=()):
        self.script = list(script)
        self.calls = []

    def __call__(self, description, operation, timeout_seconds):
        self.calls.append(description)
        entry = self.script.pop(0) if self.script else "run"
        if entry == "timeout":
            raise harness.MemoryNativeCallTimeout(
                f"{description} did not complete within "
                f"{timeout_seconds} seconds"
            )
        return operation()


def build_monitor(
    *,
    sampler,
    clock=None,
    event=None,
    gpu=None,
    thresholds=None,
    thread_factory=None,
    native_executor=None,
    pid=4242,
    process_creation_time_100ns=123456789,
    executable_path="D:\\Sim\\CarlaUE4-Win64-Shipping.exe",
    executable_basename="CarlaUE4-Win64-Shipping.exe",
):
    return harness.MemoryStabilityMonitor(
        sampler=sampler,
        thresholds=thresholds if thresholds is not None else limits(),
        gpu_sampler=gpu,
        pid=pid,
        process_creation_time_100ns=process_creation_time_100ns,
        executable_path=executable_path,
        executable_basename=executable_basename,
        monotonic=clock if clock is not None else FakeClock(),
        stop_event=event if event is not None else ScriptedEvent(),
        thread_factory=(
            thread_factory if thread_factory is not None else FakeThread
        ),
        native_call_executor=(
            native_executor
            if native_executor is not None
            else synchronous_native_calls
        ),
    )


def run_monitor(
    *,
    produce=None,
    false_waits=12,
    extra_advance=None,
    thresholds=None,
    gpu=None,
    start_clock=100.0,
    final_advance=None,
):
    """Take a baseline and run the sampling loop fully deterministically."""
    clock = FakeClock(start_clock)
    event = AutoAdvanceEvent(clock, false_waits, extra_advance=extra_advance)
    sampler = FunctionSampler(produce or (lambda index: reading()))
    monitor = build_monitor(
        sampler=sampler,
        clock=clock,
        event=event,
        gpu=gpu,
        thresholds=thresholds,
    )
    monitor.take_baseline()
    monitor._run_sampling_loop()
    if final_advance is not None:
        clock.value += final_advance
        monitor.take_final_sample()
    return monitor, clock, event, sampler


class FakeMonitor:
    """Duck-typed monitor for outer-wrapper orchestration tests."""

    def __init__(
        self,
        *,
        report=None,
        override=None,
        block_reasons=(),
        baseline_error=None,
        stop_error=None,
        join_error=None,
        final_error=None,
        close_error=None,
        journal=None,
    ):
        self.report = (
            report
            if report is not None
            else memory_stability_report()
        )
        self.override = override
        self.block_reasons = list(block_reasons)
        self.baseline_error = baseline_error
        self.stop_error = stop_error
        self.join_error = join_error
        self.final_error = final_error
        self.close_error = close_error
        self.journal = journal if journal is not None else []

    def take_baseline(self):
        self.journal.append("baseline")
        if self.baseline_error is not None:
            raise self.baseline_error

    def start(self):
        self.journal.append("start")

    def request_stop(self):
        self.journal.append("stop")
        if self.stop_error is not None:
            raise self.stop_error

    def join(self):
        self.journal.append("join")
        if self.join_error is not None:
            raise self.join_error

    def take_final_sample(self):
        self.journal.append("final")
        if self.final_error is not None:
            raise self.final_error

    def close(self):
        self.journal.append("close")
        if self.close_error is not None:
            raise self.close_error

    def admission_block_reason(self):
        if self.block_reasons:
            return self.block_reasons.pop(0)
        return None

    def build_report(self):
        self.journal.append("report")
        return self.report, self.override


def stable_metric(counter, *, start, count=25, span=120.0, **overrides):
    """One realistic fully-evaluated passing metric for a STABLE report."""
    final = start + count - 1
    values = dict(
        counter=counter,
        sample_count=count,
        span_seconds=span,
        start_bytes=start,
        peak_bytes=final,
        peak_elapsed_seconds=span,
        final_bytes=final,
        growth_bytes=final - start,
        peak_growth_bytes=final - start,
        growth_percent=(final - start) / start * 100.0,
        peak_growth_percent=(final - start) / start * 100.0,
        slope_bytes_per_minute=12.0,
        failed_thresholds=(),
        unevaluated_thresholds=(),
        passed=True,
    )
    values.update(overrides)
    return harness.MemoryMetricReport(**values)


def memory_stability_report(
    verdict=harness.MemoryStabilityVerdict.STABLE, **overrides
):
    """A realistic, internally coherent memory report for the verdict."""
    samples = tuple(
        harness.ProcessMemorySample(5.0 * index, 1000 + index, 2000 + index)
        for index in range(25)
    )
    values = dict(
        verdict=verdict,
        monitoring_enabled=True,
        identity=harness.ProcessIdentityStatus.PRESERVED,
        mandatory_sampling_failure=False,
        mandatory_shutdown_failure=False,
        pid=4242,
        process_creation_time_100ns=123456789,
        executable_path="D:\\Sim\\CarlaUE4-Win64-Shipping.exe",
        executable_basename="CarlaUE4-Win64-Shipping.exe",
        total_sample_count=25,
        retained_sample_count=25,
        omitted_sample_count=0,
        missed_interval_count=0,
        samples=samples,
        working_set=stable_metric("working_set", start=1000),
        private_commit=stable_metric("private_commit", start=2000),
        thresholds=limits(),
        gpu_status=harness.MemoryCounterAvailability.NOT_SAMPLED,
        gpu_final_dedicated_bytes=None,
        gpu_peak_dedicated_bytes=None,
        diagnostics=(),
        omitted_diagnostic_count=0,
    )
    values.update(overrides)
    return harness.MemoryStabilityReport(**values)


# ======================================================================
# Configuration contract
# ======================================================================


class MemoryConfigGateTest(unittest.TestCase):
    def test_absent_gate_disables_without_reading_thresholds(self):
        environ = {
            harness.CARLA_INTEGRATION: "1",
            harness.CARLA_INTEGRATION_SOAK_SECONDS: "120",
        }
        config = parse_memory(environ)
        self.assertIs(config.enabled, False)
        self.assertIsNone(config.thresholds)
        self.assertIsNone(config.server_pid)

    def test_exact_zero_disables_even_with_malformed_thresholds(self):
        environ = memory_environ(gate="0")
        environ[harness.CARLA_MEMORY_WORKING_SET_MAX_PEAK_GROWTH_BYTES] = "junk"
        config = parse_memory(environ)
        self.assertIs(config.enabled, False)

    def test_exact_one_enables_in_timed_soak_mode(self):
        config = parse_memory(memory_environ())
        self.assertIs(config.enabled, True)
        self.assertEqual(
            config.thresholds,
            harness.MemoryThresholds(
                working_set_max_peak_growth_bytes=1000000000,
                working_set_max_final_growth_bytes=500000000,
                working_set_max_final_growth_percent=50.0,
                private_commit_max_peak_growth_bytes=1000000000,
                private_commit_max_final_growth_bytes=500000000,
                private_commit_max_final_growth_percent=50.0,
                working_set_max_slope_bytes_per_minute=None,
                private_commit_max_slope_bytes_per_minute=None,
            ),
        )
        self.assertIsNone(config.server_pid)

    def test_invalid_gate_tokens_are_rejected(self):
        for token in ("2", " 1", "1 ", "01", "true", "TRUE", "", "yes"):
            with self.subTest(token=token):
                with self.assertRaisesRegex(
                    ValueError, "CARLA_INTEGRATION_MEMORY_STABILITY"
                ):
                    parse_memory(memory_environ(gate=token))

    def test_enabled_gate_requires_timed_soak_mode(self):
        for environ in (
            memory_environ(soak="0"),
            {
                harness.CARLA_INTEGRATION: "1",
                harness.CARLA_INTEGRATION_MEMORY_STABILITY: "1",
                **THRESHOLD_DEFAULTS,
            },
        ):
            with self.subTest(environ=environ):
                with self.assertRaisesRegex(ValueError, "timed-soak"):
                    parse_memory(environ)

    def test_enabled_gate_requires_local_host(self):
        for host in ("localhost", "127.0.0.1", "::1"):
            with self.subTest(host=host, accepted=True):
                self.assertTrue(
                    parse_memory(memory_environ(host=host)).enabled
                )
        for host in (
            "Localhost",
            "LOCALHOST",
            "127.0.0.2",
            "192.168.1.5",
            "example.com",
            "::2",
        ):
            with self.subTest(host=host, accepted=False):
                with self.assertRaisesRegex(ValueError, "CARLA_HOST"):
                    parse_memory(memory_environ(host=host))

    def test_memory_host_uses_the_raw_value_without_normalization(self):
        # The admission contract is strict raw-input: no whitespace stripping
        # and no case folding, even though legacy host parsing strips.
        for host in (
            " localhost ",
            "localhost ",
            " localhost",
            "\tlocalhost",
            "localhost\n",
            "Localhost",
            "LOCALHOST",
            "127.0.0.1 ",
            " ::1",
        ):
            with self.subTest(host=host):
                with self.assertRaisesRegex(
                    ValueError, "raw CARLA_HOST value"
                ):
                    parse_memory(memory_environ(host=host))

    def test_raw_host_rule_gates_only_the_memory_contract(self):
        environ = memory_environ(gate=None, host=" localhost ")
        config = harness.parse_integration_config(environ)
        self.assertEqual(config.host, "localhost")
        self.assertFalse(parse_memory(environ).enabled)

    def test_absent_host_defaults_to_localhost_and_is_accepted(self):
        self.assertTrue(parse_memory(memory_environ()).enabled)

    def test_each_missing_required_threshold_is_rejected_by_name(self):
        for name in THRESHOLD_DEFAULTS:
            with self.subTest(missing=name):
                environ = memory_environ()
                del environ[name]
                with self.assertRaisesRegex(ValueError, name):
                    parse_memory(environ)

    def test_partial_threshold_group_lists_every_missing_variable(self):
        environ = memory_environ()
        for name in THRESHOLD_DEFAULTS:
            del environ[name]
        with self.assertRaises(ValueError) as context:
            parse_memory(environ)
        for name in THRESHOLD_DEFAULTS:
            self.assertIn(name, str(context.exception))

    def test_byte_limits_accept_exact_boundaries(self):
        environ = memory_environ()
        environ[harness.CARLA_MEMORY_WORKING_SET_MAX_PEAK_GROWTH_BYTES] = "0"
        environ[
            harness.CARLA_MEMORY_PRIVATE_COMMIT_MAX_PEAK_GROWTH_BYTES
        ] = str(2**63 - 1)
        config = parse_memory(environ)
        self.assertEqual(config.thresholds.working_set_max_peak_growth_bytes, 0)
        self.assertEqual(
            config.thresholds.private_commit_max_peak_growth_bytes, 2**63 - 1
        )

    def test_malformed_byte_limits_are_rejected(self):
        variable = harness.CARLA_MEMORY_WORKING_SET_MAX_FINAL_GROWTH_BYTES
        for value in (
            "-1",
            "+1",
            "1.5",
            " 1",
            "1 ",
            "",
            "abc",
            "0x10",
            "1_000",
            "1e3",
            str(2**63),
            "nan",
            "inf",
        ):
            with self.subTest(value=value):
                environ = memory_environ(**{variable: value})
                with self.assertRaisesRegex(ValueError, variable):
                    parse_memory(environ)

    def test_percent_boundaries_and_malformed_values(self):
        variable = harness.CARLA_MEMORY_WORKING_SET_MAX_FINAL_GROWTH_PERCENT
        for value, expected in (("0", 0.0), ("10000", 10000.0), ("2.5", 2.5)):
            with self.subTest(value=value, accepted=True):
                config = parse_memory(memory_environ(**{variable: value}))
                self.assertEqual(
                    config.thresholds.working_set_max_final_growth_percent,
                    expected,
                )
        for value in (
            "10000.1",
            "-0.1",
            "-1",
            "+5",
            " 5",
            "5 ",
            "5_0",
            "nan",
            "inf",
            "-inf",
            "",
            "abc",
        ):
            with self.subTest(value=value, accepted=False):
                with self.assertRaisesRegex(ValueError, variable):
                    parse_memory(memory_environ(**{variable: value}))

    def test_optional_slopes_parse_individually_and_strictly(self):
        ws_slope = harness.CARLA_MEMORY_WORKING_SET_MAX_SLOPE_BYTES_PER_MINUTE
        pc_slope = (
            harness.CARLA_MEMORY_PRIVATE_COMMIT_MAX_SLOPE_BYTES_PER_MINUTE
        )
        config = parse_memory(memory_environ(**{ws_slope: "12.5"}))
        self.assertEqual(
            config.thresholds.working_set_max_slope_bytes_per_minute, 12.5
        )
        self.assertIsNone(
            config.thresholds.private_commit_max_slope_bytes_per_minute
        )
        config = parse_memory(memory_environ(**{pc_slope: "0"}))
        self.assertEqual(
            config.thresholds.private_commit_max_slope_bytes_per_minute, 0.0
        )
        for value in ("-1", "+1", "nan", "inf", " 3", "3 ", "", "abc"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, ws_slope):
                    parse_memory(memory_environ(**{ws_slope: value}))

    def test_server_pid_parses_strictly(self):
        for value, expected in (("1", 1), ("4294967295", 4294967295)):
            with self.subTest(value=value, accepted=True):
                config = parse_memory(
                    memory_environ(**{harness.CARLA_SERVER_PID: value})
                )
                self.assertEqual(config.server_pid, expected)
        for value in ("0", "4294967296", "-5", "+5", "1.5", " 1", "1 ", "", "abc"):
            with self.subTest(value=value, accepted=False):
                with self.assertRaisesRegex(
                    ValueError, harness.CARLA_SERVER_PID
                ):
                    parse_memory(
                        memory_environ(**{harness.CARLA_SERVER_PID: value})
                    )

    def test_type_validation_of_config_and_mode_arguments(self):
        config = harness.parse_integration_config(memory_environ())
        with self.assertRaises(TypeError):
            harness.parse_memory_stability_config(
                memory_environ(), None, harness.ExecutionMode.TIMED_SOAK
            )
        with self.assertRaises(TypeError):
            harness.parse_memory_stability_config(
                memory_environ(), config, "TIMED_SOAK"
            )

    def test_configuration_error_precedes_every_live_boundary(self):
        environ = memory_environ(
            **{harness.CARLA_MEMORY_WORKING_SET_MAX_PEAK_GROWTH_BYTES: "bad"}
        )
        importer = unittest.mock.Mock(name="importer")
        factory = unittest.mock.Mock(name="monitor_factory")
        with unittest.mock.patch.object(
            harness, "run_repeated_sensor_suite_validation_cycles"
        ) as runner:
            with self.assertRaisesRegex(
                ValueError,
                harness.CARLA_MEMORY_WORKING_SET_MAX_PEAK_GROWTH_BYTES,
            ):
                harness.run_repeated_sensor_suite_validation_with_memory_stability(
                    environ, importer=importer, monitor_factory=factory
                )
        runner.assert_not_called()
        factory.assert_not_called()
        importer.assert_not_called()


# ======================================================================
# Process identity and native sampling
# ======================================================================


class FakeNativeBackend:
    def __init__(
        self,
        processes=(),
        *,
        image_paths=None,
        creation_time=111222333,
        snapshot_error=None,
        open_errors=(),
        image_error=None,
        creation_error=None,
        close_error=None,
        wait_results=None,
        counter_results=None,
    ):
        self.processes = list(processes)
        self.image_paths = dict(image_paths or {})
        self.creation_time = creation_time
        self.snapshot_error = snapshot_error
        self.open_errors = dict(open_errors)
        self.image_error = image_error
        self.creation_error = creation_error
        self.close_error = close_error
        self.wait_results = list(wait_results or [])
        self.counter_results = list(counter_results or [])
        self.snapshot_calls = 0
        self.open_calls = []
        self.closed_handles = []
        self.wait_calls = 0

    def snapshot_processes(self):
        self.snapshot_calls += 1
        if self.snapshot_error is not None:
            raise self.snapshot_error
        return tuple(self.processes)

    def open_process(self, pid):
        self.open_calls.append(pid)
        error = self.open_errors.get(pid)
        if error is not None:
            raise error
        return f"handle-{pid}"

    def close_handle(self, handle):
        self.closed_handles.append(handle)
        if self.close_error is not None:
            raise self.close_error

    def query_image_path(self, handle):
        if self.image_error is not None:
            raise self.image_error
        pid = int(handle.rsplit("-", 1)[-1])
        return self.image_paths.get(
            pid, f"D:\\Sim\\{sampler_module.SHIPPING_PROCESS_IMAGE_NAME}"
        )

    def query_creation_time(self, handle):
        if self.creation_error is not None:
            raise self.creation_error
        return self.creation_time

    def wait_zero(self, handle):
        self.wait_calls += 1
        if self.wait_results:
            result = self.wait_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return sampler_module.WAIT_TIMEOUT

    def read_memory_counters(self, handle):
        if self.counter_results:
            result = self.counter_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return sampler_module.ProcessMemoryCounters(1000, 2000, 2000)


SHIPPING = sampler_module.SHIPPING_PROCESS_IMAGE_NAME


class ProcessIdentityTest(unittest.TestCase):
    def test_zero_shipping_processes_is_a_resolution_error(self):
        backend = FakeNativeBackend([(4, "System"), (900, "explorer.exe")])
        with self.assertRaisesRegex(
            sampler_module.ProcessResolutionError, "no CarlaUE4"
        ):
            sampler_module.WindowsProcessMemorySampler.open(backend=backend)
        self.assertEqual(backend.open_calls, [])

    def test_exactly_one_shipping_process_is_opened_with_identity(self):
        backend = FakeNativeBackend(
            [(4, "System"), (777, SHIPPING), (900, "explorer.exe")]
        )
        sampler = sampler_module.WindowsProcessMemorySampler.open(
            backend=backend
        )
        self.assertEqual(backend.open_calls, [777])
        self.assertEqual(
            sampler.identity,
            sampler_module.ProcessIdentity(
                pid=777,
                image_path=f"D:\\Sim\\{SHIPPING}",
                creation_time_100ns=111222333,
            ),
        )
        self.assertFalse(sampler.closed)

    def test_case_insensitive_executable_name_matches(self):
        backend = FakeNativeBackend(
            [(777, SHIPPING.lower())],
            image_paths={777: f"d:\\sim\\{SHIPPING.upper()}"},
        )
        sampler = sampler_module.WindowsProcessMemorySampler.open(
            backend=backend
        )
        self.assertEqual(sampler.identity.pid, 777)

    def test_multiple_shipping_processes_require_explicit_pid(self):
        backend = FakeNativeBackend([(777, SHIPPING), (888, SHIPPING)])
        with self.assertRaisesRegex(
            sampler_module.ProcessResolutionError, "CARLA_SERVER_PID"
        ):
            sampler_module.WindowsProcessMemorySampler.open(backend=backend)
        self.assertEqual(backend.open_calls, [])

    def test_explicit_pid_selects_between_multiple_processes(self):
        backend = FakeNativeBackend([(777, SHIPPING), (888, SHIPPING)])
        sampler = sampler_module.WindowsProcessMemorySampler.open(
            pid=888, backend=backend
        )
        self.assertEqual(sampler.identity.pid, 888)
        self.assertEqual(backend.open_calls, [888])

    def test_explicit_pid_not_running_is_rejected(self):
        backend = FakeNativeBackend([(777, SHIPPING)])
        with self.assertRaisesRegex(
            sampler_module.ProcessResolutionError, "not a running process"
        ):
            sampler_module.WindowsProcessMemorySampler.open(
                pid=999, backend=backend
            )

    def test_explicit_pid_with_wrong_executable_is_rejected(self):
        backend = FakeNativeBackend([(777, SHIPPING), (900, "explorer.exe")])
        with self.assertRaisesRegex(
            sampler_module.ProcessResolutionError, "not a running process|not "
        ):
            sampler_module.WindowsProcessMemorySampler.open(
                pid=900, backend=backend
            )
        self.assertEqual(backend.open_calls, [])

    def test_invalid_explicit_pid_values_are_rejected(self):
        backend = FakeNativeBackend([(777, SHIPPING)])
        for pid in (0, -1, True, "777", 1.0):
            with self.subTest(pid=pid):
                with self.assertRaises(
                    sampler_module.ProcessResolutionError
                ):
                    sampler_module.WindowsProcessMemorySampler.open(
                        pid=pid, backend=backend
                    )

    def test_inaccessible_process_is_an_access_error(self):
        backend = FakeNativeBackend(
            [(777, SHIPPING)], open_errors={777: OSError("access denied")}
        )
        with self.assertRaisesRegex(
            sampler_module.ProcessAccessError, "access denied"
        ):
            sampler_module.WindowsProcessMemorySampler.open(backend=backend)
        self.assertEqual(backend.closed_handles, [])

    def test_image_query_failure_closes_the_handle(self):
        backend = FakeNativeBackend(
            [(777, SHIPPING)], image_error=OSError("query failed")
        )
        with self.assertRaises(sampler_module.ProcessAccessError):
            sampler_module.WindowsProcessMemorySampler.open(backend=backend)
        self.assertEqual(backend.closed_handles, ["handle-777"])

    def test_wrong_handle_image_closes_the_handle(self):
        backend = FakeNativeBackend(
            [(777, SHIPPING)],
            image_paths={777: "C:\\Windows\\explorer.exe"},
        )
        with self.assertRaisesRegex(
            sampler_module.ProcessResolutionError, "image is not"
        ):
            sampler_module.WindowsProcessMemorySampler.open(backend=backend)
        self.assertEqual(backend.closed_handles, ["handle-777"])

    def test_path_validation_primary_survives_checked_close_failure(self):
        primary = sampler_module.ProcessResolutionError(
            "path validation failed"
        )
        backend = FakeNativeBackend(
            [(777, SHIPPING)],
            image_error=primary,
            close_error=OSError("CloseHandle failed with Windows error 6"),
        )
        with self.assertRaises(
            sampler_module.ProcessResolutionError
        ) as caught:
            sampler_module.WindowsProcessMemorySampler.open(backend=backend)
        self.assertIs(caught.exception, primary)
        self.assertEqual(backend.open_calls, [777])
        self.assertEqual(backend.closed_handles, ["handle-777"])
        notes = getattr(primary, "__notes__", ())
        self.assertEqual(len(notes), 1)
        self.assertIn("rollback close also failed", notes[0])
        self.assertIn("Windows error 6", notes[0])
        self.assertLessEqual(
            len(notes[0]), sampler_module.MAX_NATIVE_ERROR_EVIDENCE_CHARACTERS
        )

    def test_creation_time_failure_closes_the_handle(self):
        backend = FakeNativeBackend(
            [(777, SHIPPING)], creation_error=OSError("times failed")
        )
        with self.assertRaises(sampler_module.ProcessAccessError):
            sampler_module.WindowsProcessMemorySampler.open(backend=backend)
        self.assertEqual(backend.closed_handles, ["handle-777"])

    def test_creation_token_primary_survives_checked_close_failure(self):
        primary = sampler_module.ProcessAccessError(
            "creation token validation failed"
        )
        backend = FakeNativeBackend(
            [(777, SHIPPING)],
            creation_error=primary,
            close_error=OSError("CloseHandle failed with Windows error 6"),
        )
        with self.assertRaises(sampler_module.ProcessAccessError) as caught:
            sampler_module.WindowsProcessMemorySampler.open(backend=backend)
        self.assertIs(caught.exception, primary)
        self.assertEqual(backend.open_calls, [777])
        self.assertEqual(backend.closed_handles, ["handle-777"])
        notes = getattr(primary, "__notes__", ())
        self.assertEqual(len(notes), 1)
        self.assertIn("creation token validation failed", str(primary))
        self.assertIn("Windows error 6", notes[0])

    def test_snapshot_failure_is_a_resolution_error(self):
        backend = FakeNativeBackend(snapshot_error=OSError("snapshot failed"))
        with self.assertRaisesRegex(
            sampler_module.ProcessResolutionError, "snapshot failed"
        ):
            sampler_module.WindowsProcessMemorySampler.open(backend=backend)

    def test_sample_reads_counters_while_the_process_is_alive(self):
        backend = FakeNativeBackend(
            [(777, SHIPPING)],
            counter_results=[
                sampler_module.ProcessMemoryCounters(1111, 2222, 2222)
            ],
        )
        sampler = sampler_module.WindowsProcessMemorySampler.open(
            backend=backend
        )
        counters = sampler.sample()
        self.assertEqual(counters.working_set_bytes, 1111)
        self.assertEqual(counters.private_commit_bytes, 2222)
        self.assertEqual(backend.wait_calls, 1)

    def test_process_exit_is_detected_by_zero_timeout_wait(self):
        backend = FakeNativeBackend(
            [(777, SHIPPING)],
            wait_results=[sampler_module.WAIT_OBJECT_0],
        )
        sampler = sampler_module.WindowsProcessMemorySampler.open(
            backend=backend
        )
        with self.assertRaisesRegex(
            sampler_module.ProcessExitedError, "777"
        ):
            sampler.sample()

    def test_unexpected_wait_states_are_counter_read_errors(self):
        for state in (
            sampler_module.WAIT_ABANDONED,
            sampler_module.WAIT_FAILED,
        ):
            with self.subTest(state=state):
                backend = FakeNativeBackend(
                    [(777, SHIPPING)], wait_results=[state]
                )
                sampler = sampler_module.WindowsProcessMemorySampler.open(
                    backend=backend
                )
                with self.assertRaises(sampler_module.CounterReadError):
                    sampler.sample()

    def test_counter_read_failure_is_wrapped_and_never_retried(self):
        backend = FakeNativeBackend(
            [(777, SHIPPING)], counter_results=[OSError("read failed")]
        )
        sampler = sampler_module.WindowsProcessMemorySampler.open(
            backend=backend
        )
        with self.assertRaisesRegex(
            sampler_module.CounterReadError, "read failed"
        ):
            sampler.sample()
        self.assertEqual(backend.wait_calls, 1)

    def test_pid_reuse_never_reacquires_a_replacement_process(self):
        backend = FakeNativeBackend(
            [(777, SHIPPING)],
            wait_results=[
                sampler_module.WAIT_TIMEOUT,
                sampler_module.WAIT_OBJECT_0,
                sampler_module.WAIT_OBJECT_0,
            ],
        )
        sampler = sampler_module.WindowsProcessMemorySampler.open(
            backend=backend
        )
        sampler.sample()
        # A replacement process appears under the same numeric PID.
        backend.processes = [(777, SHIPPING)]
        with self.assertRaises(sampler_module.ProcessExitedError):
            sampler.sample()
        self.assertTrue(sampler.has_exited())
        # The retained handle keeps answering for the original process; the
        # sampler never enumerates or opens anything again.
        self.assertEqual(backend.snapshot_calls, 1)
        self.assertEqual(backend.open_calls, [777])

    def test_close_is_deterministic_and_idempotent(self):
        backend = FakeNativeBackend([(777, SHIPPING)])
        sampler = sampler_module.WindowsProcessMemorySampler.open(
            backend=backend
        )
        sampler.close()
        sampler.close()
        self.assertEqual(backend.closed_handles, ["handle-777"])
        with self.assertRaises(sampler_module.CounterReadError):
            sampler.sample()
        with self.assertRaises(sampler_module.CounterReadError):
            sampler.has_exited()

    def test_context_manager_closes_the_handle(self):
        backend = FakeNativeBackend([(777, SHIPPING)])
        with sampler_module.WindowsProcessMemorySampler.open(
            backend=backend
        ) as sampler:
            self.assertFalse(sampler.closed)
        self.assertEqual(backend.closed_handles, ["handle-777"])


# ======================================================================
# Native structure mapping
# ======================================================================


class MemoryStructureTest(unittest.TestCase):
    def require_windows_x64_abi(self):
        if sys.platform != "win32" or ctypes.sizeof(ctypes.c_void_p) != 8:
            self.skipTest("PROCESSENTRY32W literal ABI assertions require x64 Windows")

    def test_counters_map_working_set_and_private_usage_exactly(self):
        structure = sampler_module.PROCESS_MEMORY_COUNTERS_EX()
        structure.WorkingSetSize = 123456789
        structure.PrivateUsage = 987654321
        structure.PagefileUsage = 987654321
        counters = sampler_module.counters_from_memory_structure(structure)
        self.assertEqual(counters.working_set_bytes, 123456789)
        self.assertEqual(counters.private_commit_bytes, 987654321)
        self.assertEqual(counters.pagefile_usage_bytes, 987654321)

    def test_negative_stub_counters_are_rejected(self):
        stub = types.SimpleNamespace(
            WorkingSetSize=-1, PrivateUsage=5, PagefileUsage=5
        )
        with self.assertRaises(sampler_module.CounterReadError):
            sampler_module.counters_from_memory_structure(stub)

    def test_memory_counters_structure_layout(self):
        size_t = ctypes.sizeof(ctypes.c_size_t)
        expected = 8 + 9 * size_t
        self.assertEqual(
            ctypes.sizeof(sampler_module.PROCESS_MEMORY_COUNTERS_EX), expected
        )
        self.assertEqual(
            sampler_module.PROCESS_MEMORY_COUNTERS_EX.PrivateUsage.offset,
            8 + 8 * size_t,
        )
        self.assertEqual(
            sampler_module.PROCESS_MEMORY_COUNTERS_EX.WorkingSetSize.offset,
            8 + size_t,
        )

    def test_process_entry_structure_layout(self):
        entry = sampler_module.PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(sampler_module.PROCESSENTRY32W)
        entry.th32ProcessID = 4242
        entry.szExeFile = SHIPPING
        self.assertEqual(entry.th32ProcessID, 4242)
        self.assertEqual(entry.szExeFile, SHIPPING)

    def test_process_entry_x64_windows_layout_uses_literal_abi_values(self):
        self.require_windows_x64_abi()
        structure = sampler_module.PROCESSENTRY32W
        self.assertEqual(ctypes.sizeof(structure), 568)
        self.assertEqual(ctypes.alignment(structure), 8)
        expected_offsets = {
            "dwSize": 0,
            "cntUsage": 4,
            "th32ProcessID": 8,
            "th32DefaultHeapID": 16,
            "th32ModuleID": 24,
            "cntThreads": 28,
            "th32ParentProcessID": 32,
            "pcPriClassBase": 36,
            "dwFlags": 40,
            "szExeFile": 44,
        }
        self.assertEqual(
            {
                field_name: getattr(structure, field_name).offset
                for field_name in expected_offsets
            },
            expected_offsets,
        )

    def test_process_entry_x64_windows_primitives_are_independent(self):
        self.require_windows_x64_abi()
        fields = dict(sampler_module.PROCESSENTRY32W._fields_)
        dword_fields = (
            "dwSize",
            "cntUsage",
            "th32ProcessID",
            "th32ModuleID",
            "cntThreads",
            "th32ParentProcessID",
            "dwFlags",
        )
        for field_name in dword_fields:
            with self.subTest(field=field_name):
                self.assertIs(fields[field_name], ctypes.c_uint32)
                self.assertEqual(ctypes.sizeof(fields[field_name]), 4)
        heap_id_type = fields["th32DefaultHeapID"]
        self.assertIs(heap_id_type, ctypes.c_size_t)
        self.assertEqual(ctypes.sizeof(heap_id_type), 8)
        self.assertEqual(heap_id_type(-1).value, 0xFFFFFFFFFFFFFFFF)
        self.assertIs(fields["pcPriClassBase"], ctypes.c_int32)
        self.assertEqual(ctypes.sizeof(fields["pcPriClassBase"]), 4)
        self.assertEqual(fields["pcPriClassBase"](-1).value, -1)
        executable_array = fields["szExeFile"]
        self.assertIs(executable_array._type_, ctypes.c_wchar)
        self.assertEqual(ctypes.sizeof(ctypes.c_wchar), 2)
        self.assertEqual(executable_array._length_, 260)
        self.assertEqual(sampler_module.MAX_PROCESS_NAME_CHARACTERS, 260)

    def test_pdh_formatted_array_item_layout_uses_native_pointer_and_union(self):
        item_fields = dict(sampler_module.PDH_FMT_COUNTERVALUE_ITEM_W._fields_)
        self.assertIs(item_fields["szName"], ctypes.c_wchar_p)
        self.assertIs(
            item_fields["FmtValue"], sampler_module.PDH_FMT_COUNTERVALUE
        )
        value_fields = dict(
            sampler_module._PDH_FMT_COUNTERVALUE_UNION._fields_
        )
        self.assertIs(value_fields["largeValue"], ctypes.c_longlong)
        self.assertEqual(
            sampler_module.PDH_FMT_COUNTERVALUE_ITEM_W.FmtValue.offset,
            ctypes.sizeof(ctypes.c_void_p),
        )
        self.assertEqual(
            sampler_module.PROCESSENTRY32W.szExeFile.size,
            260 * ctypes.sizeof(ctypes.c_wchar),
        )

    def test_filetime_conversion_combines_both_words(self):
        filetime = sampler_module.FILETIME()
        filetime.dwLowDateTime = 0x89ABCDEF
        filetime.dwHighDateTime = 0x01234567
        self.assertEqual(
            sampler_module.filetime_to_100ns(filetime), 0x0123456789ABCDEF
        )

    def test_pagefile_alias_consistency_diagnostic(self):
        consistent = sampler_module.ProcessMemoryCounters(1, 500, 500)
        inconsistent = sampler_module.ProcessMemoryCounters(1, 500, 600)
        check = sampler_module.WindowsProcessMemorySampler
        self.assertTrue(check.pagefile_alias_consistent(consistent))
        self.assertFalse(check.pagefile_alias_consistent(inconsistent))
        with self.assertRaises(TypeError):
            check.pagefile_alias_consistent("counters")

    def test_image_base_name_handles_both_separators(self):
        self.assertEqual(
            sampler_module.image_base_name("D:\\a\\b\\App.exe"), "App.exe"
        )
        self.assertEqual(
            sampler_module.image_base_name("D:/a/b/App.exe"), "App.exe"
        )
        self.assertEqual(sampler_module.image_base_name("App.exe"), "App.exe")


# ======================================================================
# Production ctypes backends behind deterministic mocked DLL exports
# ======================================================================


class FakeNativeFunction:
    """Callable DLL export that enforces its production ctypes ABI."""

    def __init__(
        self, name, implementation, *, expected_argtypes=None, expected_restype=None
    ):
        self._name = name
        self._implementation = implementation
        self._expected_argtypes = expected_argtypes
        self._expected_restype = expected_restype
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        if self._expected_argtypes is not None:
            if self.argtypes != self._expected_argtypes:
                raise AssertionError(
                    f"{self._name}.argtypes {self.argtypes!r} != "
                    f"{self._expected_argtypes!r}"
                )
            if len(args) != len(self._expected_argtypes):
                raise AssertionError(
                    f"{self._name} received {len(args)} arguments; "
                    f"expected {len(self._expected_argtypes)}"
                )
        if self._expected_restype is not None and (
            self.restype is not self._expected_restype
        ):
            raise AssertionError(
                f"{self._name}.restype {self.restype!r} != "
                f"{self._expected_restype!r}"
            )
        return self._implementation(*args)


class FakeDll:
    """One fake loaded DLL exposing exactly the exports it was given."""

    def __init__(self, exports, specs=None):
        specs = specs or {}
        self._exports = {
            name: FakeNativeFunction(
                name,
                implementation,
                expected_argtypes=specs.get(name, (None, None))[0],
                expected_restype=specs.get(name, (None, None))[1],
            )
            for name, implementation in exports.items()
        }

    def __getattr__(self, name):
        try:
            return self._exports[name]
        except KeyError:
            raise AttributeError(name) from None


class Kernel32Scenario:
    """Deterministic kernel32/psapi behavior behind the production backend."""

    SNAPSHOT_HANDLE = 3001
    PROCESS_HANDLE = 4001

    def __init__(
        self,
        processes=((777, SHIPPING),),
        *,
        image_path=None,
        snapshot_fails=False,
        enumeration_end_code=sampler_module.ERROR_NO_MORE_FILES,
        open_fails=False,
        image_fails=False,
        times_fails=False,
        creation=(0x33221100, 0x00000042),
        wait_results=(),
        memory_fails=False,
        working_set=11_000_000,
        private_usage=22_000_000,
        close_results=(),
        include_k32_memory_info=True,
    ):
        self.processes = list(processes)
        self.image_path = image_path
        self.snapshot_fails = snapshot_fails
        self.enumeration_end_code = enumeration_end_code
        self.open_fails = open_fails
        self.image_fails = image_fails
        self.times_fails = times_fails
        self.creation = creation
        self.wait_results = list(wait_results)
        self.memory_fails = memory_fails
        self.working_set = working_set
        self.private_usage = private_usage
        self.close_results = list(close_results)
        self.include_k32_memory_info = include_k32_memory_info
        self.last_error = 0
        self.snapshot_calls = []
        self.entry_sizes = []
        self.entry_object_types = []
        self.open_calls = []
        self.image_calls = []
        self.wait_calls = []
        self.memory_calls = []
        self.closed_handles = []
        self.psapi_loaded = 0
        self._cursor = 0

    def kernel32_exports(self):
        exports = {
            "CreateToolhelp32Snapshot": self._create_snapshot,
            "Process32FirstW": self._first,
            "Process32NextW": self._next,
            "OpenProcess": self._open_process,
            "CloseHandle": self._close_handle,
            "QueryFullProcessImageNameW": self._query_image,
            "GetProcessTimes": self._get_times,
            "WaitForSingleObject": self._wait,
        }
        if self.include_k32_memory_info:
            exports["K32GetProcessMemoryInfo"] = self._memory_info
        return exports

    def psapi_exports(self):
        return {"GetProcessMemoryInfo": self._memory_info}

    def _create_snapshot(self, flags, pid):
        self.snapshot_calls.append((flags, pid))
        if self.snapshot_fails:
            self.last_error = 5
            return sampler_module.INVALID_HANDLE_VALUE
        return self.SNAPSHOT_HANDLE

    def _fill_entry(self, entry_ref, index):
        entry = entry_ref._obj
        pid, name = self.processes[index]
        entry.th32ProcessID = pid
        entry.szExeFile = name

    def _first(self, snapshot, entry_ref):
        self._validate_process_entry_call(snapshot, entry_ref)
        self.entry_sizes.append(int(entry_ref._obj.dwSize))
        self._cursor = 0
        if not self.processes:
            self.last_error = self.enumeration_end_code
            return 0
        self._fill_entry(entry_ref, 0)
        return 1

    def _next(self, snapshot, entry_ref):
        self._validate_process_entry_call(snapshot, entry_ref)
        self._cursor += 1
        if self._cursor >= len(self.processes):
            self.last_error = self.enumeration_end_code
            return 0
        self._fill_entry(entry_ref, self._cursor)
        return 1

    def _validate_process_entry_call(self, snapshot, entry_ref):
        if snapshot != self.SNAPSHOT_HANDLE:
            raise AssertionError("Toolhelp call used the wrong snapshot HANDLE")
        if not isinstance(entry_ref._obj, sampler_module.PROCESSENTRY32W):
            raise AssertionError("Toolhelp call used the wrong structure pointer")
        self.entry_object_types.append(type(entry_ref._obj))
        if entry_ref._obj.dwSize != ctypes.sizeof(
            sampler_module.PROCESSENTRY32W
        ):
            raise AssertionError("PROCESSENTRY32W.dwSize was not exact")

    def _open_process(self, access, inherit, pid):
        self.open_calls.append((access, inherit, pid))
        if self.open_fails:
            self.last_error = 5
            return 0
        return self.PROCESS_HANDLE

    def _close_handle(self, handle):
        self.closed_handles.append(handle)
        result = self.close_results.pop(0) if self.close_results else 1
        if not result:
            self.last_error = 6
        return result

    def _query_image(self, handle, flags, buffer, size_ref):
        self.image_calls.append((handle, flags, len(buffer)))
        if not isinstance(size_ref._obj, ctypes.c_uint32):
            raise AssertionError("image-path length pointer is not DWORD*")
        if size_ref._obj.value != sampler_module.MAX_IMAGE_PATH_CHARACTERS:
            raise AssertionError("image-path input length is not exact")
        if len(buffer) != sampler_module.MAX_IMAGE_PATH_CHARACTERS:
            raise AssertionError("image-path buffer length is not bounded exactly")
        if self.image_fails:
            self.last_error = 24
            return 0
        path = (
            self.image_path
            if self.image_path is not None
            else f"D:\\Sim\\{SHIPPING}"
        )
        buffer.value = path
        if buffer[len(path)] != "\x00":
            raise AssertionError("image-path buffer is not null terminated")
        size_ref._obj.value = len(path)
        return 1

    def _get_times(self, handle, creation_ref, exit_ref, kernel_ref, user_ref):
        for reference in (creation_ref, exit_ref, kernel_ref, user_ref):
            if not isinstance(reference._obj, sampler_module.FILETIME):
                raise AssertionError("GetProcessTimes requires FILETIME pointers")
        if self.times_fails:
            self.last_error = 87
            return 0
        low, high = self.creation
        creation_ref._obj.dwLowDateTime = low
        creation_ref._obj.dwHighDateTime = high
        return 1

    def _wait(self, handle, timeout):
        self.wait_calls.append((handle, timeout))
        if self.wait_results:
            return self.wait_results.pop(0)
        return sampler_module.WAIT_TIMEOUT

    def _memory_info(self, handle, counters_ref, cb):
        if not isinstance(
            counters_ref._obj, sampler_module.PROCESS_MEMORY_COUNTERS_EX
        ):
            raise AssertionError(
                "GetProcessMemoryInfo requires PROCESS_MEMORY_COUNTERS_EX*"
            )
        counters = counters_ref._obj
        self.memory_calls.append((handle, int(counters.cb), int(cb)))
        if self.memory_fails:
            self.last_error = 299
            return 0
        counters.WorkingSetSize = self.working_set
        counters.PrivateUsage = self.private_usage
        counters.PagefileUsage = self.private_usage
        return 1


@contextlib.contextmanager
def production_kernel_backend(scenario):
    """Yield a real WindowsNativeBackend bound to the scenario's fake DLLs."""
    entry_pointer = ctypes.POINTER(sampler_module.PROCESSENTRY32W)
    filetime_pointer = ctypes.POINTER(sampler_module.FILETIME)
    counters_pointer = ctypes.POINTER(
        sampler_module.PROCESS_MEMORY_COUNTERS_EX
    )
    specs = {
        "CreateToolhelp32Snapshot": (
            (ctypes.c_uint32, ctypes.c_uint32), ctypes.c_void_p
        ),
        "Process32FirstW": (
            (ctypes.c_void_p, entry_pointer), ctypes.c_int32
        ),
        "Process32NextW": (
            (ctypes.c_void_p, entry_pointer), ctypes.c_int32
        ),
        "OpenProcess": (
            (ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32),
            ctypes.c_void_p,
        ),
        "CloseHandle": ((ctypes.c_void_p,), ctypes.c_int32),
        "QueryFullProcessImageNameW": (
            (
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_wchar_p,
                ctypes.POINTER(ctypes.c_uint32),
            ),
            ctypes.c_int32,
        ),
        "GetProcessTimes": (
            (
                ctypes.c_void_p,
                filetime_pointer,
                filetime_pointer,
                filetime_pointer,
                filetime_pointer,
            ),
            ctypes.c_int32,
        ),
        "WaitForSingleObject": (
            (ctypes.c_void_p, ctypes.c_uint32), ctypes.c_uint32
        ),
        "K32GetProcessMemoryInfo": (
            (ctypes.c_void_p, counters_pointer, ctypes.c_uint32),
            ctypes.c_int32,
        ),
        "GetProcessMemoryInfo": (
            (ctypes.c_void_p, counters_pointer, ctypes.c_uint32),
            ctypes.c_int32,
        ),
    }
    kernel32 = FakeDll(scenario.kernel32_exports(), specs)
    psapi = FakeDll(scenario.psapi_exports(), specs)

    def load(name, *args, **kwargs):
        if name == "psapi":
            scenario.psapi_loaded += 1
        return {"kernel32": kernel32, "psapi": psapi}[name]

    with unittest.mock.patch.object(
        ctypes, "WinDLL", side_effect=load
    ), unittest.mock.patch.object(
        ctypes, "get_last_error", side_effect=lambda: scenario.last_error
    ):
        yield sampler_module.WindowsNativeBackend()


class ProductionNativeBackendTest(unittest.TestCase):
    """Execute the real ctypes wrapper methods over mocked DLL exports."""

    def test_all_kernel_and_psapi_declarations_use_exact_windows_types(self):
        scenario = Kernel32Scenario()
        with production_kernel_backend(scenario) as backend:
            functions = (
                backend._kernel32.CreateToolhelp32Snapshot,
                backend._kernel32.Process32FirstW,
                backend._kernel32.Process32NextW,
                backend._kernel32.OpenProcess,
                backend._kernel32.CloseHandle,
                backend._kernel32.QueryFullProcessImageNameW,
                backend._kernel32.GetProcessTimes,
                backend._kernel32.WaitForSingleObject,
                backend._get_memory_info,
            )
            for function in functions:
                self.assertEqual(function.argtypes, function._expected_argtypes)
                self.assertIs(function.restype, function._expected_restype)

        scenario = Kernel32Scenario(include_k32_memory_info=False)
        with production_kernel_backend(scenario) as backend:
            self.assertEqual(
                backend._get_memory_info.argtypes,
                backend._get_memory_info._expected_argtypes,
            )
            self.assertIs(
                backend._get_memory_info.restype,
                backend._get_memory_info._expected_restype,
            )

    def test_snapshot_enumerates_with_exact_structure_size_and_closes(self):
        scenario = Kernel32Scenario(
            [(4, "System"), (777, SHIPPING), (900, "explorer.exe")]
        )
        with production_kernel_backend(scenario) as backend:
            entries = backend.snapshot_processes()
        self.assertEqual(
            entries, ((4, "System"), (777, SHIPPING), (900, "explorer.exe"))
        )
        self.assertEqual(
            scenario.snapshot_calls, [(sampler_module.TH32CS_SNAPPROCESS, 0)]
        )
        self.assertEqual(
            scenario.entry_sizes,
            [ctypes.sizeof(sampler_module.PROCESSENTRY32W)],
        )
        self.assertEqual(
            scenario.closed_handles, [Kernel32Scenario.SNAPSHOT_HANDLE]
        )

    def test_toolhelp_apis_receive_exact_process_entry_pointer_type(self):
        if sys.platform != "win32" or ctypes.sizeof(ctypes.c_void_p) != 8:
            self.skipTest("PROCESSENTRY32W pointer ABI requires x64 Windows")
        scenario = Kernel32Scenario([(777, SHIPPING), (900, "other.exe")])
        with production_kernel_backend(scenario) as backend:
            first_pointer = backend._kernel32.Process32FirstW.argtypes[1]
            next_pointer = backend._kernel32.Process32NextW.argtypes[1]
            self.assertIs(first_pointer._type_, sampler_module.PROCESSENTRY32W)
            self.assertIs(next_pointer._type_, sampler_module.PROCESSENTRY32W)
            backend.snapshot_processes()
        self.assertEqual(
            scenario.entry_object_types,
            [
                sampler_module.PROCESSENTRY32W,
                sampler_module.PROCESSENTRY32W,
                sampler_module.PROCESSENTRY32W,
            ],
        )

    def test_snapshot_creation_failure_raises_with_last_error(self):
        scenario = Kernel32Scenario(snapshot_fails=True)
        with production_kernel_backend(scenario) as backend:
            with self.assertRaisesRegex(OSError, "CreateToolhelp32Snapshot"):
                backend.snapshot_processes()
        self.assertEqual(scenario.closed_handles, [])

    def test_enumeration_ending_without_no_more_files_raises_and_closes(self):
        scenario = Kernel32Scenario(
            [(777, SHIPPING)], enumeration_end_code=1450
        )
        with production_kernel_backend(scenario) as backend:
            with self.assertRaisesRegex(OSError, "1450"):
                backend.snapshot_processes()
        self.assertEqual(
            scenario.closed_handles, [Kernel32Scenario.SNAPSHOT_HANDLE]
        )

    def test_empty_snapshot_ends_cleanly_on_no_more_files(self):
        scenario = Kernel32Scenario([])
        with production_kernel_backend(scenario) as backend:
            self.assertEqual(backend.snapshot_processes(), ())
        self.assertEqual(
            scenario.closed_handles, [Kernel32Scenario.SNAPSHOT_HANDLE]
        )

    def test_successful_enumeration_with_snapshot_close_failure_is_fatal(self):
        scenario = Kernel32Scenario(close_results=[0])
        with production_kernel_backend(scenario) as backend:
            with self.assertRaisesRegex(OSError, r"CloseHandle.*6"):
                backend.snapshot_processes()
        self.assertEqual(
            scenario.closed_handles, [Kernel32Scenario.SNAPSHOT_HANDLE]
        )

    def test_empty_enumeration_with_snapshot_close_failure_is_fatal(self):
        scenario = Kernel32Scenario([], close_results=[0])
        with production_kernel_backend(scenario) as backend:
            with self.assertRaisesRegex(OSError, r"CloseHandle.*6"):
                backend.snapshot_processes()
        self.assertEqual(
            scenario.closed_handles, [Kernel32Scenario.SNAPSHOT_HANDLE]
        )

    def test_enumeration_and_close_failure_preserve_both_errors_once(self):
        scenario = Kernel32Scenario(
            enumeration_end_code=1450, close_results=[0]
        )
        with production_kernel_backend(scenario) as backend:
            with self.assertRaisesRegex(OSError, "1450") as raised:
                backend.snapshot_processes()
        self.assertEqual(
            scenario.closed_handles, [Kernel32Scenario.SNAPSHOT_HANDLE]
        )
        notes = getattr(raised.exception, "__notes__", ())
        self.assertEqual(len(notes), 1)
        self.assertIn("snapshot close also failed", notes[0])
        self.assertIn("CloseHandle failed with Windows error 6", notes[0])
        self.assertLessEqual(
            len(notes[0]), sampler_module.MAX_NATIVE_ERROR_EVIDENCE_CHARACTERS
        )

    def test_monitor_factory_refuses_snapshot_close_failure(self):
        scenario = Kernel32Scenario(close_results=[0])
        config = parse_memory(memory_environ())
        with production_kernel_backend(scenario) as backend, \
             unittest.mock.patch.object(
                 sampler_module, "WindowsNativeBackend", return_value=backend
             ), unittest.mock.patch.object(
                 harness,
                 "_run_native_call_bounded",
                 side_effect=synchronous_native_calls,
             ), unittest.mock.patch.object(
                 harness.MemoryStabilityMonitor, "__init__", return_value=None
             ) as monitor_init:
            factory = harness._default_memory_monitor_factory(
                lambda _name: sampler_module
            )
            with self.assertRaises(sampler_module.ProcessResolutionError):
                factory(config)
        monitor_init.assert_not_called()
        self.assertEqual(scenario.open_calls, [])
        self.assertEqual(
            scenario.closed_handles, [Kernel32Scenario.SNAPSHOT_HANDLE]
        )

    def test_open_process_requests_limited_query_and_synchronize_rights(self):
        scenario = Kernel32Scenario()
        with production_kernel_backend(scenario) as backend:
            handle = backend.open_process(777)
        self.assertEqual(handle, Kernel32Scenario.PROCESS_HANDLE)
        self.assertEqual(
            scenario.open_calls,
            [(sampler_module.PROCESS_OPEN_ACCESS, 0, 777)],
        )

    def test_open_process_failure_raises_with_pid_and_last_error(self):
        scenario = Kernel32Scenario(open_fails=True)
        with production_kernel_backend(scenario) as backend:
            with self.assertRaisesRegex(OSError, r"OpenProcess\(777\).*5"):
                backend.open_process(777)

    def test_image_query_uses_a_bounded_buffer_and_returns_the_path(self):
        scenario = Kernel32Scenario(image_path="C:\\Games\\" + SHIPPING)
        with production_kernel_backend(scenario) as backend:
            path = backend.query_image_path(4001)
        self.assertEqual(path, "C:\\Games\\" + SHIPPING)
        handle, flags, buffer_length = scenario.image_calls[0]
        self.assertEqual(handle, 4001)
        self.assertEqual(flags, 0)
        self.assertEqual(
            buffer_length, sampler_module.MAX_IMAGE_PATH_CHARACTERS
        )

    def test_image_query_failure_raises(self):
        scenario = Kernel32Scenario(image_fails=True)
        with production_kernel_backend(scenario) as backend:
            with self.assertRaisesRegex(
                OSError, "QueryFullProcessImageNameW"
            ):
                backend.query_image_path(4001)

    def test_creation_time_combines_both_filetime_words(self):
        scenario = Kernel32Scenario(creation=(0x89ABCDEF, 0x01234567))
        with production_kernel_backend(scenario) as backend:
            value = backend.query_creation_time(4001)
        self.assertEqual(value, (0x01234567 << 32) | 0x89ABCDEF)

    def test_creation_time_failure_raises(self):
        scenario = Kernel32Scenario(times_fails=True)
        with production_kernel_backend(scenario) as backend:
            with self.assertRaisesRegex(OSError, "GetProcessTimes"):
                backend.query_creation_time(4001)

    def test_wait_zero_polls_with_zero_timeout_and_returns_the_state(self):
        for state in (
            sampler_module.WAIT_TIMEOUT,
            sampler_module.WAIT_OBJECT_0,
            sampler_module.WAIT_ABANDONED,
            sampler_module.WAIT_FAILED,
        ):
            with self.subTest(state=state):
                scenario = Kernel32Scenario(wait_results=[state])
                with production_kernel_backend(scenario) as backend:
                    self.assertEqual(backend.wait_zero(4001), state)
                self.assertEqual(scenario.wait_calls, [(4001, 0)])

    def test_memory_counters_pass_exact_cb_and_structure_size(self):
        scenario = Kernel32Scenario()
        with production_kernel_backend(scenario) as backend:
            counters = backend.read_memory_counters(4001)
        expected_size = ctypes.sizeof(
            sampler_module.PROCESS_MEMORY_COUNTERS_EX
        )
        self.assertEqual(
            scenario.memory_calls, [(4001, expected_size, expected_size)]
        )
        self.assertEqual(counters.working_set_bytes, 11_000_000)
        self.assertEqual(counters.private_commit_bytes, 22_000_000)
        self.assertEqual(counters.pagefile_usage_bytes, 22_000_000)

    def test_memory_counter_failure_raises_with_last_error(self):
        scenario = Kernel32Scenario(memory_fails=True)
        with production_kernel_backend(scenario) as backend:
            with self.assertRaisesRegex(OSError, "GetProcessMemoryInfo.*299"):
                backend.read_memory_counters(4001)

    def test_missing_k32_export_falls_back_to_psapi(self):
        scenario = Kernel32Scenario(include_k32_memory_info=False)
        with production_kernel_backend(scenario) as backend:
            counters = backend.read_memory_counters(4001)
        self.assertEqual(scenario.psapi_loaded, 1)
        self.assertEqual(counters.working_set_bytes, 11_000_000)

    def test_close_handle_failure_raises(self):
        scenario = Kernel32Scenario(close_results=[0])
        with production_kernel_backend(scenario) as backend:
            with self.assertRaisesRegex(OSError, "CloseHandle"):
                backend.close_handle(4001)
        self.assertEqual(scenario.closed_handles, [4001])

    def test_full_sampler_flow_leaks_no_handles_and_matches_exactly(self):
        scenario = Kernel32Scenario(
            [(4, "System"), (777, SHIPPING), (900, "explorer.exe")],
            wait_results=[
                sampler_module.WAIT_TIMEOUT,
                sampler_module.WAIT_TIMEOUT,
            ],
        )
        with production_kernel_backend(scenario) as backend:
            sampler = sampler_module.WindowsProcessMemorySampler.open(
                backend=backend
            )
            counters = sampler.sample()
            sampler.close()
            sampler.close()  # double-close prevention at the sampler layer
        self.assertEqual(scenario.open_calls[0][2], 777)
        self.assertEqual(
            sampler.identity,
            sampler_module.ProcessIdentity(
                pid=777,
                image_path=f"D:\\Sim\\{SHIPPING}",
                creation_time_100ns=(0x00000042 << 32) | 0x33221100,
            ),
        )
        self.assertEqual(counters.working_set_bytes, 11_000_000)
        self.assertEqual(counters.private_commit_bytes, 22_000_000)
        # Exactly one snapshot handle and one process handle, each closed
        # exactly once: no leaked handles on the complete flow.
        self.assertEqual(
            scenario.closed_handles,
            [
                Kernel32Scenario.SNAPSHOT_HANDLE,
                Kernel32Scenario.PROCESS_HANDLE,
            ],
        )

    def test_full_sampler_flow_detects_exit_through_the_retained_handle(self):
        scenario = Kernel32Scenario(
            wait_results=[sampler_module.WAIT_OBJECT_0]
        )
        with production_kernel_backend(scenario) as backend:
            sampler = sampler_module.WindowsProcessMemorySampler.open(
                backend=backend
            )
            with self.assertRaises(sampler_module.ProcessExitedError):
                sampler.sample()
            sampler.close()
        self.assertEqual(len(scenario.memory_calls), 0)

    def test_wait_failure_through_the_production_backend_is_classified(self):
        scenario = Kernel32Scenario(
            wait_results=[sampler_module.WAIT_FAILED]
        )
        with production_kernel_backend(scenario) as backend:
            sampler = sampler_module.WindowsProcessMemorySampler.open(
                backend=backend
            )
            with self.assertRaises(sampler_module.CounterReadError):
                sampler.sample()
            sampler.close()

    def test_open_flow_closes_the_process_handle_on_image_mismatch(self):
        scenario = Kernel32Scenario(image_path="D:\\Other\\other.exe")
        with production_kernel_backend(scenario) as backend:
            with self.assertRaises(sampler_module.ProcessResolutionError):
                sampler_module.WindowsProcessMemorySampler.open(
                    backend=backend
                )
        self.assertIn(
            Kernel32Scenario.PROCESS_HANDLE, scenario.closed_handles
        )


class PdhScenario:
    """Deterministic pdh.dll behavior behind the production PDH backend."""

    QUERY_HANDLE = 9101
    COUNTER_HANDLE = 9202

    def __init__(
        self,
        items=(),
        *,
        open_status=0,
        add_status=0,
        collect_statuses=(),
        sizing_status=None,
        read_status=0,
        close_status=0,
        include_english=True,
    ):
        self.items = list(items)
        self.open_status = open_status
        self.add_status = add_status
        self.collect_statuses = list(collect_statuses)
        self.sizing_status = sizing_status
        self.read_status = read_status
        self.close_status = close_status
        self.include_english = include_english
        self.open_calls = []
        self.add_calls = []
        self.collect_calls = 0
        self.array_calls = []
        self.closed_queries = []
        self._keepalive = []

    def exports(self):
        exports = {
            "PdhOpenQueryW": self._open,
            "PdhCollectQueryData": self._collect,
            "PdhGetFormattedCounterArrayW": self._array,
            "PdhCloseQuery": self._close,
        }
        if self.include_english:
            exports["PdhAddEnglishCounterW"] = self._add
        else:
            exports["PdhAddCounterW"] = self._add
        return exports

    def _open(self, source, userdata, query_ref):
        if not isinstance(query_ref._obj, ctypes.c_void_p):
            raise AssertionError("PdhOpenQueryW requires PDH_HQUERY*")
        self.open_calls.append((source, userdata))
        if self.open_status:
            return self.open_status
        query_ref._obj.value = self.QUERY_HANDLE
        return 0

    def _add(self, query, path, userdata, counter_ref):
        if not isinstance(counter_ref._obj, ctypes.c_void_p):
            raise AssertionError("PdhAddCounterW requires PDH_HCOUNTER*")
        if not isinstance(path, str) or not path or "\x00" in path:
            raise AssertionError("PDH counter path was not null-terminated text")
        self.add_calls.append((getattr(query, "value", query), path))
        if self.add_status:
            return self.add_status
        counter_ref._obj.value = self.COUNTER_HANDLE
        return 0

    def _collect(self, query):
        self.collect_calls += 1
        if self.collect_statuses:
            return self.collect_statuses.pop(0)
        return 0

    def _array(self, counter, fmt, size_ref, count_ref, raw):
        if not isinstance(size_ref._obj, ctypes.c_uint32):
            raise AssertionError("formatted array size requires DWORD*")
        if not isinstance(count_ref._obj, ctypes.c_uint32):
            raise AssertionError("formatted array count requires DWORD*")
        if fmt != sampler_module.PDH_FMT_LARGE:
            raise AssertionError("formatted array did not request LARGE values")
        self.array_calls.append(
            (getattr(counter, "value", counter), fmt, raw is not None)
        )
        item_type = sampler_module.PDH_FMT_COUNTERVALUE_ITEM_W
        if raw is None:
            if self.sizing_status is not None:
                return self.sizing_status
            size_ref._obj.value = ctypes.sizeof(item_type) * len(self.items)
            count_ref._obj.value = len(self.items)
            return sampler_module.PDH_MORE_DATA
        if self.read_status:
            return self.read_status
        required = ctypes.sizeof(item_type) * len(self.items)
        if len(raw) < required:
            raise AssertionError("formatted array buffer is too short")
        buffer = (item_type * max(1, len(self.items)))()
        for index, (name, counter_status, value) in enumerate(self.items):
            buffer[index].szName = name
            buffer[index].FmtValue.CStatus = counter_status
            buffer[index].FmtValue.value.largeValue = value
        self._keepalive.append(buffer)
        ctypes.memmove(
            raw, buffer, min(len(raw), ctypes.sizeof(item_type) * len(self.items))
        )
        count_ref._obj.value = len(self.items)
        return 0

    def _close(self, query):
        self.closed_queries.append(getattr(query, "value", query))
        return self.close_status


@contextlib.contextmanager
def production_pdh_backend(scenario):
    """Yield a real WindowsPdhBackend bound to the scenario's fake pdh.dll."""
    handle_pointer = ctypes.POINTER(ctypes.c_void_p)
    specs = {
        "PdhOpenQueryW": (
            (ctypes.c_wchar_p, ctypes.c_size_t, handle_pointer),
            ctypes.c_uint32,
        ),
        "PdhAddEnglishCounterW": (
            (
                ctypes.c_void_p,
                ctypes.c_wchar_p,
                ctypes.c_size_t,
                handle_pointer,
            ),
            ctypes.c_uint32,
        ),
        "PdhAddCounterW": (
            (
                ctypes.c_void_p,
                ctypes.c_wchar_p,
                ctypes.c_size_t,
                handle_pointer,
            ),
            ctypes.c_uint32,
        ),
        "PdhCollectQueryData": ((ctypes.c_void_p,), ctypes.c_uint32),
        "PdhGetFormattedCounterArrayW": (
            (
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_void_p,
            ),
            ctypes.c_uint32,
        ),
        "PdhCloseQuery": ((ctypes.c_void_p,), ctypes.c_uint32),
    }
    pdh = FakeDll(scenario.exports(), specs)
    with unittest.mock.patch.object(
        ctypes, "WinDLL", side_effect=lambda name, **kwargs: {"pdh": pdh}[name]
    ):
        yield sampler_module.WindowsPdhBackend()


VALID_PDH_STATUS = sampler_module.PDH_CSTATUS_VALID_DATA


class ProductionPdhBackendTest(unittest.TestCase):
    """Execute the real PDH ctypes wrapper methods over mocked exports."""

    def test_all_pdh_declarations_use_exact_windows_types(self):
        for include_english in (True, False):
            with self.subTest(include_english=include_english):
                scenario = PdhScenario(include_english=include_english)
                with production_pdh_backend(scenario) as backend:
                    functions = (
                        backend._pdh.PdhOpenQueryW,
                        backend._add_counter,
                        backend._pdh.PdhCollectQueryData,
                        backend._pdh.PdhGetFormattedCounterArrayW,
                        backend._pdh.PdhCloseQuery,
                    )
                    for function in functions:
                        self.assertEqual(
                            function.argtypes, function._expected_argtypes
                        )
                        self.assertIs(
                            function.restype, function._expected_restype
                        )

    def test_open_query_returns_the_native_query_handle(self):
        scenario = PdhScenario()
        with production_pdh_backend(scenario) as backend:
            query = backend.open_query()
        self.assertEqual(query.value, PdhScenario.QUERY_HANDLE)
        self.assertEqual(scenario.open_calls, [(None, 0)])

    def test_open_query_failure_raises_with_hex_status(self):
        scenario = PdhScenario(open_status=0xC0000BB8)
        with production_pdh_backend(scenario) as backend:
            with self.assertRaisesRegex(
                sampler_module.PdhQueryError, "0xC0000BB8"
            ):
                backend.open_query()

    def test_add_counter_binds_the_wildcard_gpu_path(self):
        scenario = PdhScenario()
        with production_pdh_backend(scenario) as backend:
            query = backend.open_query()
            counter = backend.add_english_counter(
                query, sampler_module.GPU_PROCESS_MEMORY_COUNTER_PATH
            )
        self.assertEqual(counter.value, PdhScenario.COUNTER_HANDLE)
        self.assertEqual(
            scenario.add_calls,
            [
                (
                    PdhScenario.QUERY_HANDLE,
                    sampler_module.GPU_PROCESS_MEMORY_COUNTER_PATH,
                )
            ],
        )

    def test_add_counter_failure_raises_with_the_path(self):
        scenario = PdhScenario(add_status=0xC0000BB9)
        with production_pdh_backend(scenario) as backend:
            query = backend.open_query()
            with self.assertRaisesRegex(
                sampler_module.PdhQueryError, "GPU Process Memory"
            ):
                backend.add_english_counter(
                    query, sampler_module.GPU_PROCESS_MEMORY_COUNTER_PATH
                )

    def test_missing_english_export_falls_back_to_add_counter(self):
        scenario = PdhScenario(include_english=False)
        with production_pdh_backend(scenario) as backend:
            query = backend.open_query()
            counter = backend.add_english_counter(query, "\\Some(*)\\Path")
        self.assertEqual(counter.value, PdhScenario.COUNTER_HANDLE)
        self.assertEqual(scenario.add_calls[0][1], "\\Some(*)\\Path")

    def test_collect_translates_no_data_and_other_failures(self):
        scenario = PdhScenario(collect_statuses=[0])
        with production_pdh_backend(scenario) as backend:
            query = backend.open_query()
            backend.collect(query)
        self.assertEqual(scenario.collect_calls, 1)

        scenario = PdhScenario(
            collect_statuses=[sampler_module.PDH_NO_DATA]
        )
        with production_pdh_backend(scenario) as backend:
            query = backend.open_query()
            with self.assertRaises(sampler_module.PdhNoDataError):
                backend.collect(query)

        scenario = PdhScenario(collect_statuses=[0xC0000BBB])
        with production_pdh_backend(scenario) as backend:
            query = backend.open_query()
            with self.assertRaisesRegex(
                sampler_module.PdhQueryError, "PdhCollectQueryData"
            ):
                backend.collect(query)

    def test_formatted_array_uses_the_two_call_sizing_protocol(self):
        scenario = PdhScenario(
            [
                ("pid_42_luid_0x0_0x1_phys_0", VALID_PDH_STATUS, 1024),
                ("pid_42_luid_0x0_0x2_phys_0", VALID_PDH_STATUS, 2048),
            ]
        )
        with production_pdh_backend(scenario) as backend:
            readings = backend.formatted_counter_array(
                ctypes.c_void_p(PdhScenario.COUNTER_HANDLE)
            )
        self.assertEqual(
            readings,
            (
                ("pid_42_luid_0x0_0x1_phys_0", VALID_PDH_STATUS, 1024),
                ("pid_42_luid_0x0_0x2_phys_0", VALID_PDH_STATUS, 2048),
            ),
        )
        self.assertEqual(
            scenario.array_calls,
            [
                (PdhScenario.COUNTER_HANDLE, sampler_module.PDH_FMT_LARGE, False),
                (PdhScenario.COUNTER_HANDLE, sampler_module.PDH_FMT_LARGE, True),
            ],
        )

    def test_sizing_no_data_and_unexpected_statuses_are_classified(self):
        scenario = PdhScenario(sizing_status=sampler_module.PDH_NO_DATA)
        with production_pdh_backend(scenario) as backend:
            with self.assertRaises(sampler_module.PdhNoDataError):
                backend.formatted_counter_array(ctypes.c_void_p(1))

        scenario = PdhScenario(sizing_status=0xC0000BBC)
        with production_pdh_backend(scenario) as backend:
            with self.assertRaisesRegex(
                sampler_module.PdhQueryError, "sizing failed"
            ):
                backend.formatted_counter_array(ctypes.c_void_p(1))

    def test_sizing_success_with_zero_items_returns_empty(self):
        scenario = PdhScenario(sizing_status=0)
        with production_pdh_backend(scenario) as backend:
            self.assertEqual(
                backend.formatted_counter_array(ctypes.c_void_p(1)), ()
            )

    def test_read_call_failure_is_classified(self):
        scenario = PdhScenario(
            [("pid_42_luid_0x0_0x1_phys_0", VALID_PDH_STATUS, 1024)],
            read_status=0xC0000BBD,
        )
        with production_pdh_backend(scenario) as backend:
            with self.assertRaisesRegex(
                sampler_module.PdhQueryError, "read failed"
            ):
                backend.formatted_counter_array(ctypes.c_void_p(1))

    def test_close_query_success_and_failure_classification(self):
        scenario = PdhScenario()
        with production_pdh_backend(scenario) as backend:
            query = backend.open_query()
            backend.close_query(query)
        self.assertEqual(scenario.closed_queries, [PdhScenario.QUERY_HANDLE])

        scenario = PdhScenario(close_status=0xC0000BBE)
        with production_pdh_backend(scenario) as backend:
            query = backend.open_query()
            with self.assertRaisesRegex(
                sampler_module.PdhQueryError, "PdhCloseQuery"
            ):
                backend.close_query(query)

    def gpu_sampler(self, scenario, pid=42):
        with production_pdh_backend(scenario) as backend:
            sampler = sampler_module.GpuDedicatedMemorySampler(
                pid, backend=backend
            )
            return sampler

    def test_gpu_flow_matches_exact_pid_prefix_without_collisions(self):
        scenario = PdhScenario(
            [
                ("pid_42_luid_0x0_0x1_phys_0", VALID_PDH_STATUS, 1000),
                ("pid_420_luid_0x0_0x1_phys_0", VALID_PDH_STATUS, 5000),
                ("pid_4_luid_0x0_0x1_phys_0", VALID_PDH_STATUS, 7000),
                ("pid_42_luid_0x0_0x2_phys_0", VALID_PDH_STATUS, 200),
                ("pid_42_luid_0x0_0x2_phys_0", VALID_PDH_STATUS, 999),
            ]
        )
        sampler = self.gpu_sampler(scenario)
        self.assertIs(
            sampler.open(), sampler_module.GpuAvailability.AVAILABLE
        )
        sample = sampler.sample()
        # pid_420 and pid_4 are prefix collisions, never matched; the
        # duplicated pid_42 luid instance is deduplicated to its first value.
        self.assertEqual(sample.dedicated_bytes, 1200)
        self.assertEqual(sample.instance_count, 2)
        sampler.close()
        sampler.close()
        self.assertEqual(
            scenario.closed_queries, [PdhScenario.QUERY_HANDLE]
        )

    def test_gpu_flow_reports_no_activity_without_matching_instances(self):
        scenario = PdhScenario(
            [("pid_999_luid_0x0_0x1_phys_0", VALID_PDH_STATUS, 4096)]
        )
        sampler = self.gpu_sampler(scenario)
        sampler.open()
        self.assertIs(
            sampler.sample().status,
            sampler_module.GpuAvailability.NO_ACTIVITY,
        )
        sampler.close()

    def test_gpu_flow_flags_malformed_values_as_unreliable(self):
        scenario = PdhScenario(
            [("pid_42_luid_0x0_0x1_phys_0", VALID_PDH_STATUS, -5)]
        )
        sampler = self.gpu_sampler(scenario)
        sampler.open()
        self.assertIs(
            sampler.sample().status,
            sampler_module.GpuAvailability.UNRELIABLE,
        )
        sampler.close()

    def test_gpu_flow_flags_invalid_counter_status_beside_valid_data(self):
        scenario = PdhScenario(
            [
                ("pid_42_luid_0x0_0x1_phys_0", 0xC0000BBF, 4096),
                ("pid_42_luid_0x0_0x2_phys_0", VALID_PDH_STATUS, 512),
            ]
        )
        sampler = self.gpu_sampler(scenario)
        sampler.open()
        sample = sampler.sample()
        self.assertIs(
            sample.status, sampler_module.GpuAvailability.AVAILABLE
        )
        self.assertEqual(sample.dedicated_bytes, 512)
        sampler.close()

    def test_gpu_open_failure_closes_the_query_and_preserves_failure(self):
        scenario = PdhScenario(add_status=0xC0000BB9)
        sampler = self.gpu_sampler(scenario)
        with self.assertRaises(sampler_module.PdhQueryError):
            sampler.open()
        # The successfully opened query is still closed on the failed path.
        self.assertEqual(
            scenario.closed_queries, [PdhScenario.QUERY_HANDLE]
        )
        sampler.close()
        self.assertEqual(
            scenario.closed_queries, [PdhScenario.QUERY_HANDLE]
        )


# ======================================================================
# Scheduler
# ======================================================================


class MonitorSchedulerTest(unittest.TestCase):
    def test_baseline_is_sample_zero_and_must_precede_start(self):
        sampler = FunctionSampler(lambda index: reading(1000 + index, 2000))
        monitor = build_monitor(sampler=sampler, clock=FakeClock(100.0))
        with self.assertRaisesRegex(RuntimeError, "take_baseline"):
            monitor.start()
        monitor.take_baseline()
        self.assertEqual(sampler.calls, 1)
        report, override = monitor.build_report()
        self.assertIsNone(override)
        self.assertEqual(report.total_sample_count, 1)
        self.assertEqual(report.samples[0].elapsed_seconds, 0.0)
        with self.assertRaisesRegex(RuntimeError, "already taken"):
            monitor.take_baseline()

    def test_exact_five_second_absolute_cadence(self):
        monitor, _clock, event, _sampler = run_monitor(false_waits=4)
        self.assertEqual(event.wait_timeouts, [5.0] * 5)
        report, _override = monitor.build_report()
        self.assertEqual(
            tuple(sample.elapsed_seconds for sample in report.samples),
            (0.0, 5.0, 10.0, 15.0, 20.0),
        )
        self.assertEqual(report.missed_interval_count, 0)

    def test_late_wake_takes_one_sample_and_moves_to_future_deadline(self):
        # Wait 2 overshoots its 5-second timeout by 12 seconds, waking at
        # elapsed 22.0: exactly one sample is taken there, the two missed
        # deadlines (15, 20) are skipped and counted, and the next wait
        # targets the future deadline 25 with a 3-second timeout.
        monitor, _clock, event, _sampler = run_monitor(
            false_waits=3, extra_advance={2: 12.0}
        )
        report, _override = monitor.build_report()
        self.assertEqual(
            tuple(sample.elapsed_seconds for sample in report.samples),
            (0.0, 5.0, 22.0, 25.0),
        )
        self.assertEqual(report.missed_interval_count, 2)
        self.assertEqual(event.wait_timeouts, [5.0, 5.0, 3.0, 5.0])

    def test_already_late_entry_samples_without_waiting(self):
        clock = FakeClock(100.0)
        event = ScriptedEvent([True])
        sampler = FunctionSampler(lambda index: reading())
        monitor = build_monitor(sampler=sampler, clock=clock, event=event)
        monitor.take_baseline()
        clock.value = 113.0
        monitor._run_sampling_loop()
        report, _override = monitor.build_report()
        # The overdue wake samples immediately at 13.0 without any wait for
        # that sample, skips the missed deadline 10, then waits toward the
        # future deadline 15 with the remaining 2 seconds.
        self.assertEqual(
            tuple(sample.elapsed_seconds for sample in report.samples),
            (0.0, 13.0),
        )
        self.assertEqual(report.missed_interval_count, 1)
        self.assertEqual(event.wait_timeouts, [2.0])

    def test_stop_during_wait_exits_without_extra_sample(self):
        monitor, _clock, _event, sampler = run_monitor(false_waits=2)
        self.assertEqual(sampler.calls, 3)  # baseline + two cadence samples

    def test_preset_stop_event_exits_before_any_cadence_sample(self):
        clock = FakeClock(100.0)
        event = ScriptedEvent()
        event.set()
        sampler = FunctionSampler(lambda index: reading())
        monitor = build_monitor(sampler=sampler, clock=clock, event=event)
        monitor.take_baseline()
        monitor._run_sampling_loop()
        self.assertEqual(sampler.calls, 1)
        self.assertEqual(event.wait_timeouts, [])

    def test_final_sample_is_taken_after_the_loop(self):
        monitor, _clock, _event, sampler = run_monitor(
            false_waits=2, final_advance=2.5
        )
        report, _override = monitor.build_report()
        self.assertEqual(sampler.calls, 4)
        self.assertEqual(report.samples[-1].elapsed_seconds, 12.5)

    def test_final_sample_with_identical_timestamp_is_coalesced(self):
        monitor, _clock, _event, sampler = run_monitor(
            false_waits=2, final_advance=0.0
        )
        report, _override = monitor.build_report()
        self.assertEqual(sampler.calls, 3)
        self.assertEqual(report.total_sample_count, 3)
        self.assertEqual(report.samples[-1].elapsed_seconds, 10.0)

    def test_final_sample_runs_at_most_once(self):
        monitor, clock, _event, sampler = run_monitor(false_waits=1)
        clock.value += 1.0
        monitor.take_final_sample()
        clock.value += 1.0
        monitor.take_final_sample()
        self.assertEqual(sampler.calls, 3)

    def test_final_sample_is_skipped_after_process_exit(self):
        exit_error = harness.MemoryMonitoredProcessExited("gone")
        monitor, clock, _event, sampler = run_monitor(
            produce=lambda index: exit_error if index == 2 else reading(),
            false_waits=3,
        )
        calls_before = sampler.calls
        clock.value += 30.0
        monitor.take_final_sample()
        self.assertEqual(sampler.calls, calls_before)

    def test_sampler_failure_stops_the_loop_without_retry(self):
        monitor, _clock, event, sampler = run_monitor(
            produce=lambda index: (
                RuntimeError("boom") if index == 3 else reading()
            ),
            false_waits=10,
        )
        self.assertEqual(sampler.calls, 3)
        self.assertEqual(len(event.wait_timeouts), 2)
        self.assertIs(
            monitor.admission_block_reason(),
            harness.TerminationReason.MEMORY_SAMPLING_FAILURE,
        )
        report, override = monitor.build_report()
        self.assertIs(
            override, harness.TerminationReason.MEMORY_SAMPLING_FAILURE
        )
        self.assertTrue(
            any("boom" in text for text in report.diagnostics)
        )

    def test_process_exit_in_loop_blocks_next_admission(self):
        exit_error = harness.MemoryMonitoredProcessExited("exited")
        monitor, _clock, _event, sampler = run_monitor(
            produce=lambda index: exit_error if index == 4 else reading(),
            false_waits=10,
        )
        self.assertEqual(sampler.calls, 4)
        self.assertIs(
            monitor.admission_block_reason(),
            harness.TerminationReason.MEMORY_PROCESS_DISAPPEARED,
        )
        report, override = monitor.build_report()
        self.assertIs(
            override, harness.TerminationReason.MEMORY_PROCESS_DISAPPEARED
        )
        self.assertIs(report.identity, harness.ProcessIdentityStatus.EXITED)

    def test_non_positive_counters_are_a_sampling_failure(self):
        for bad in (reading(working_set=0), reading(private_commit=-5)):
            with self.subTest(bad=bad):
                monitor, _clock, _event, _sampler = run_monitor(
                    produce=lambda index, bad=bad: (
                        bad if index == 2 else reading()
                    ),
                    false_waits=5,
                )
                self.assertIs(
                    monitor.admission_block_reason(),
                    harness.TerminationReason.MEMORY_SAMPLING_FAILURE,
                )

    def test_clock_failure_in_loop_is_recorded_not_raised(self):
        clock = FakeClock(100.0)
        sampler = FunctionSampler(lambda index: reading())
        monitor = build_monitor(
            sampler=sampler, clock=clock, event=ScriptedEvent([False])
        )
        monitor.take_baseline()
        clock.value = math.nan
        monitor._run_sampling_loop()
        self.assertIs(
            monitor.admission_block_reason(),
            harness.TerminationReason.MEMORY_SAMPLING_FAILURE,
        )

    def test_baseline_failure_raises_after_classification(self):
        sampler = FunctionSampler(
            lambda index: harness.MemoryMonitoredProcessExited("gone")
        )
        monitor = build_monitor(sampler=sampler)
        with self.assertRaisesRegex(RuntimeError, "baseline"):
            monitor.take_baseline()
        self.assertIs(
            monitor.admission_block_reason(),
            harness.TerminationReason.MEMORY_PROCESS_DISAPPEARED,
        )

    def test_start_uses_thread_factory_and_join_delegates(self):
        threads = []

        def factory(target):
            thread = FakeThread(target)
            threads.append(thread)
            return thread

        clock = FakeClock(100.0)
        sampler = FunctionSampler(lambda index: reading())
        monitor = build_monitor(
            sampler=sampler,
            clock=clock,
            event=ScriptedEvent([True]),
            thread_factory=factory,
        )
        monitor.take_baseline()
        monitor.start()
        with self.assertRaisesRegex(RuntimeError, "already started"):
            monitor.start()
        monitor.join()
        self.assertEqual(len(threads), 1)
        self.assertEqual(threads[0].started, 1)
        self.assertEqual(threads[0].joined, 1)

    def test_close_closes_sampler_and_gpu_exactly_once(self):
        gpu = FakeGpu(
            [
                harness.GpuEvidenceReading(
                    harness.MemoryCounterAvailability.NO_ACTIVITY
                )
            ]
        )
        sampler = FunctionSampler(lambda index: reading())
        monitor = build_monitor(sampler=sampler, gpu=gpu)
        monitor.take_baseline()
        monitor.close()
        monitor.close()
        self.assertEqual(sampler.closed, 1)
        self.assertEqual(gpu.closed, 1)


class SchedulerAdvancementTest(unittest.TestCase):
    """Constant-time missed-deadline advancement after clock jumps."""

    def advance(self, deadline, now, interval=5.0):
        return harness._advance_cadence_deadline(deadline, now, interval)

    def test_on_time_wake_advances_one_interval_without_missed(self):
        self.assertEqual(self.advance(105.0, 104.0), (110.0, 0))
        self.assertEqual(self.advance(105.0, 109.9), (110.0, 0))

    def test_two_missed_intervals_are_counted_exactly(self):
        self.assertEqual(self.advance(110.0, 122.0), (125.0, 2))

    def test_exact_cadence_equality_stays_strictly_future(self):
        self.assertEqual(self.advance(10.0, 15.0), (20.0, 1))
        self.assertEqual(self.advance(10.0, 20.0), (25.0, 2))
        self.assertEqual(self.advance(10.0, 25.0), (30.0, 3))

    def test_clock_rollback_keeps_the_next_deadline_without_missed(self):
        self.assertEqual(self.advance(105.0, 90.0), (110.0, 0))

    def test_a_billion_missed_intervals_compute_arithmetically(self):
        deadline, missed = self.advance(105.0, 5_000_000_105.0)
        self.assertEqual(missed, 1_000_000_000)
        self.assertEqual(deadline, 5_000_000_110.0)
        self.assertIs(type(missed), int)

    def test_extremely_large_finite_jump_stays_bounded_and_minimal(self):
        now = 1e18
        deadline, missed = self.advance(0.0, now)
        self.assertIs(type(missed), int)
        self.assertGreater(deadline, now)
        self.assertEqual(missed, 200_000_000_000_000_000)
        # Five seconds is below the float ULP here, so the bounded fallback is
        # the least representable value strictly above now.
        self.assertEqual(deadline, math.nextafter(now, math.inf))

    def test_1e25_jump_uses_the_exact_input_float_rational_count(self):
        deadline, missed = self.advance(0.0, 1e25, 5.0)
        self.assertEqual(missed, 2_000_000_000_000_000_181_193_932)
        self.assertIs(type(missed), int)
        self.assertEqual(deadline, math.nextafter(1e25, math.inf))

    def test_sub_ulp_interval_at_large_origin_uses_bounded_fallback(self):
        for origin, interval in ((1e25, 5.0), (1e16, 0.25)):
            with self.subTest(origin=origin, interval=interval):
                deadline, missed = self.advance(origin, origin, interval)
                self.assertEqual(missed, 0)
                self.assertEqual(deadline, math.nextafter(origin, math.inf))

    def test_near_maximum_finite_value_uses_last_finite_deadline(self):
        maximum = sys.float_info.max
        origin = math.nextafter(maximum, 0.0)
        deadline, missed = self.advance(origin, origin, 1.0)
        self.assertEqual((deadline, missed), (maximum, 0))

    def test_infinite_arithmetic_fallback_is_rejected(self):
        with self.assertRaisesRegex(OverflowError, "representable"):
            self.advance(sys.float_info.max, sys.float_info.max, 5.0)
        with self.assertRaisesRegex(OverflowError, "representable"):
            self.advance(-sys.float_info.max, sys.float_info.max, 5.0)

    def test_non_positive_and_non_finite_intervals_are_rejected(self):
        for bad in (0.0, -1.0, math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.advance(105.0, 105.0, bad)

    def test_advancement_contains_no_iteration_construct(self):
        tree = ast.parse(inspect.getsource(harness._advance_cadence_deadline))
        iteration_nodes = (
            ast.For,
            ast.While,
            ast.AsyncFor,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
        )
        self.assertFalse(
            any(isinstance(node, iteration_nodes) for node in ast.walk(tree))
        )

    def test_non_finite_clock_values_are_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.advance(105.0, bad)
                with self.assertRaises(ValueError):
                    self.advance(bad, 105.0)

    def test_loop_counts_a_billion_missed_intervals_exactly(self):
        monitor, _clock, event, _sampler = run_monitor(
            false_waits=2, extra_advance={1: 5_000_000_000.0}
        )
        report, _override = monitor.build_report()
        self.assertEqual(report.missed_interval_count, 1_000_000_000)
        self.assertEqual(event.wait_timeouts, [5.0, 5.0, 5.0])
        self.assertIsNone(monitor.admission_block_reason())

    def test_stop_requested_during_the_late_wake_path_skips_the_sample(self):
        clock = FakeClock(100.0)

        class LateStopEvent:
            def __init__(self):
                self._set = False
                self.wait_timeouts = []

            def is_set(self):
                return self._set

            def set(self):
                self._set = True

            def wait(self, timeout=None):
                self.wait_timeouts.append(timeout)
                # The stop request lands while this wake oversleeps far
                # beyond its deadline.
                clock.value += timeout + 42.0
                self._set = True
                return False

        event = LateStopEvent()
        sampler = FunctionSampler(lambda index: reading())
        monitor = build_monitor(sampler=sampler, clock=clock, event=event)
        monitor.take_baseline()
        monitor._run_sampling_loop()
        self.assertEqual(sampler.calls, 1)
        report, _override = monitor.build_report()
        self.assertEqual(report.total_sample_count, 1)
        self.assertEqual(report.missed_interval_count, 0)

    def test_stop_set_during_false_wait_is_checked_before_clock_read(self):
        clock = FakeClock(100.0)

        class FalseWaitStopEvent(ScriptedEvent):
            def wait(self, timeout=None):
                self.wait_timeouts.append(timeout)
                self._set = True
                return False

        event = FalseWaitStopEvent()
        sampler = FunctionSampler(lambda index: reading())
        monitor = build_monitor(sampler=sampler, clock=clock, event=event)
        monitor.take_baseline()
        monitor._run_sampling_loop()
        self.assertEqual(sampler.calls, 1)
        self.assertEqual(event.wait_timeouts, [5.0])

    def test_stop_set_during_already_late_clock_read_prevents_sample(self):
        event = ScriptedEvent()

        class LateStoppingClock:
            def __init__(self):
                self.calls = 0

            def __call__(self):
                self.calls += 1
                if self.calls == 2:
                    event.set()
                    return 110.0
                return 100.0

        clock = LateStoppingClock()
        sampler = FunctionSampler(lambda index: reading())
        monitor = build_monitor(sampler=sampler, clock=clock, event=event)
        monitor.take_baseline()
        monitor._run_sampling_loop()
        self.assertEqual(sampler.calls, 1)
        self.assertEqual(event.wait_timeouts, [])

    def test_stop_set_by_cadence_advancement_prevents_sample(self):
        clock = FakeClock(110.0)
        event = ScriptedEvent()
        sampler = FunctionSampler(lambda index: reading())
        monitor = build_monitor(sampler=sampler, clock=clock, event=event)
        monitor.take_baseline()
        clock.value = 120.0
        original = harness._advance_cadence_deadline

        def advance_then_stop(*args):
            result = original(*args)
            event.set()
            return result

        with unittest.mock.patch.object(
            harness,
            "_advance_cadence_deadline",
            side_effect=advance_then_stop,
        ):
            monitor._run_sampling_loop()
        self.assertEqual(sampler.calls, 1)
        report, _override = monitor.build_report()
        self.assertEqual(report.missed_interval_count, 0)

    def test_stop_at_immediate_pre_sample_check_prevents_sample(self):
        clock = FakeClock(110.0)

        class StopOnFourthCheck(ScriptedEvent):
            def __init__(self):
                super().__init__()
                self.checks = 0

            def is_set(self):
                self.checks += 1
                if self.checks >= 4:
                    self._set = True
                return self._set

        event = StopOnFourthCheck()
        sampler = FunctionSampler(lambda index: reading())
        monitor = build_monitor(sampler=sampler, clock=clock, event=event)
        monitor.take_baseline()
        clock.value = 120.0
        monitor._run_sampling_loop()
        self.assertEqual(sampler.calls, 1)
        self.assertEqual(event.checks, 4)

    def test_infinite_clock_in_loop_is_a_sampling_failure(self):
        clock = FakeClock(100.0)
        sampler = FunctionSampler(lambda index: reading())
        monitor = build_monitor(
            sampler=sampler, clock=clock, event=ScriptedEvent([False])
        )
        monitor.take_baseline()
        clock.value = math.inf
        monitor._run_sampling_loop()
        self.assertIs(
            monitor.admission_block_reason(),
            harness.TerminationReason.MEMORY_SAMPLING_FAILURE,
        )


# ======================================================================
# Bounded native execution and shutdown
# ======================================================================


class BoundedNativeExecutionTest(unittest.TestCase):
    def test_default_executor_returns_the_operation_result(self):
        result = harness._run_native_call_bounded("probe", lambda: 1234, 5.0)
        self.assertEqual(result, 1234)

    def test_default_executor_delivers_operation_exceptions_unchanged(self):
        error = RuntimeError("native failure")

        def failing():
            raise error

        with self.assertRaises(RuntimeError) as caught:
            harness._run_native_call_bounded("probe", failing, 5.0)
        self.assertIs(caught.exception, error)

    def test_default_executor_abandons_a_never_returning_call(self):
        gate = threading.Event()
        entered = threading.Event()
        finished = threading.Event()

        def blocked():
            entered.set()
            gate.wait()  # a stand-in for a native call that never returns
            finished.set()
            return "late result"

        try:
            started = time.perf_counter()
            with self.assertRaisesRegex(
                harness.MemoryNativeCallTimeout, "did not complete"
            ):
                harness._run_native_call_bounded("probe", blocked, 0.05)
            elapsed = time.perf_counter() - started
            self.assertTrue(entered.wait(1.0))
            self.assertFalse(finished.is_set())
            self.assertLess(elapsed, 1.0)
        finally:
            gate.set()
            self.assertTrue(finished.wait(1.0))

    def test_real_hung_native_call_cannot_mutate_or_reuse_abandoned_state(self):
        gate = threading.Event()
        entered = threading.Event()
        finished = threading.Event()

        class BlockingSampler(FunctionSampler):
            def sample(self):
                self.calls += 1
                if self.calls == 1:
                    return reading()
                entered.set()
                gate.wait()
                finished.set()
                return reading(9999, 9999)

        sampler = BlockingSampler(lambda _index: reading())
        monitor = build_monitor(
            sampler=sampler,
            native_executor=harness._run_native_call_bounded,
        )
        monitor._native_call_timeout_seconds = 0.05
        monitor.take_baseline()
        try:
            started = time.perf_counter()
            self.assertFalse(monitor._take_sample("cadence sampling", 5.0))
            elapsed = time.perf_counter() - started
            self.assertTrue(entered.wait(1.0))
            self.assertFalse(finished.is_set())
            self.assertLess(elapsed, 1.0)
            report, override = monitor.build_report()
            self.assertEqual(report.total_sample_count, 1)
            self.assertTrue(report.mandatory_sampling_failure)
            self.assertIs(
                override, harness.TerminationReason.MEMORY_SAMPLING_FAILURE
            )
            monitor.take_final_sample()
            monitor.close()
            self.assertEqual(sampler.calls, 2)
            self.assertEqual(sampler.closed, 0)
        finally:
            gate.set()
            self.assertTrue(finished.wait(1.0))
        # The native worker's late return was discarded by its timed-out
        # controller and could not append a sample afterward.
        report, _override = monitor.build_report()
        self.assertEqual(report.total_sample_count, 1)

    def test_process_sample_timeout_is_a_mandatory_sampling_failure(self):
        executor = ScriptedNativeExecutor(["run", "timeout"])
        clock = FakeClock(100.0)
        event = AutoAdvanceEvent(clock, 10)
        sampler = FunctionSampler(lambda index: reading())
        monitor = build_monitor(
            sampler=sampler, clock=clock, event=event, native_executor=executor
        )
        monitor.take_baseline()
        monitor._run_sampling_loop()
        # The timed-out call was never delivered and never retried.
        self.assertEqual(sampler.calls, 1)
        self.assertIs(
            monitor.admission_block_reason(),
            harness.TerminationReason.MEMORY_SAMPLING_FAILURE,
        )
        report, override = monitor.build_report()
        self.assertIs(
            override, harness.TerminationReason.MEMORY_SAMPLING_FAILURE
        )
        self.assertTrue(report.mandatory_sampling_failure)
        self.assertIsNot(report.verdict, harness.MemoryStabilityVerdict.STABLE)

    def test_abandoned_process_handle_is_never_closed_or_resampled(self):
        executor = ScriptedNativeExecutor(["run", "timeout"])
        clock = FakeClock(100.0)
        event = AutoAdvanceEvent(clock, 10)
        sampler = FunctionSampler(lambda index: reading())
        monitor = build_monitor(
            sampler=sampler, clock=clock, event=event, native_executor=executor
        )
        monitor.take_baseline()
        monitor._run_sampling_loop()
        clock.value += 3.0
        monitor.take_final_sample()  # refused: the handle is abandoned
        monitor.close()
        self.assertEqual(sampler.calls, 1)
        self.assertEqual(sampler.closed, 0)
        report, _override = monitor.build_report()
        self.assertTrue(report.mandatory_shutdown_failure)
        self.assertIs(
            report.verdict, harness.MemoryStabilityVerdict.INDETERMINATE
        )
        self.assertTrue(
            any("abandoned" in text for text in report.diagnostics)
        )
        self.assertTrue(
            any("close refused" in text for text in report.diagnostics)
        )

    def test_baseline_timeout_raises_and_classifies(self):
        executor = ScriptedNativeExecutor(["timeout"])
        sampler = FunctionSampler(lambda index: reading())
        monitor = build_monitor(sampler=sampler, native_executor=executor)
        with self.assertRaisesRegex(RuntimeError, "baseline"):
            monitor.take_baseline()
        self.assertIs(
            monitor.admission_block_reason(),
            harness.TerminationReason.MEMORY_SAMPLING_FAILURE,
        )
        monitor.close()
        self.assertEqual(sampler.closed, 0)

    def test_final_sample_timeout_is_recorded_and_blocks_stable(self):
        executor = ScriptedNativeExecutor(["run", "run", "run", "timeout"])
        clock = FakeClock(100.0)
        event = AutoAdvanceEvent(clock, 2)
        sampler = FunctionSampler(lambda index: reading())
        monitor = build_monitor(
            sampler=sampler, clock=clock, event=event, native_executor=executor
        )
        monitor.take_baseline()
        monitor._run_sampling_loop()
        clock.value += 2.0
        monitor.take_final_sample()
        report, override = monitor.build_report()
        self.assertTrue(report.mandatory_sampling_failure)
        self.assertIs(
            override, harness.TerminationReason.MEMORY_SAMPLING_FAILURE
        )
        monitor.close()
        self.assertEqual(sampler.closed, 0)

    def test_gpu_sample_timeout_stays_advisory(self):
        executor = ScriptedNativeExecutor(["run", "timeout"])
        gpu = FakeGpu(
            [
                harness.GpuEvidenceReading(
                    harness.MemoryCounterAvailability.AVAILABLE, 4096
                )
            ]
        )
        sampler = FunctionSampler(lambda index: reading())
        monitor = build_monitor(
            sampler=sampler, gpu=gpu, native_executor=executor
        )
        monitor.take_baseline()
        report, override = monitor.build_report()
        self.assertIsNone(override)
        self.assertFalse(report.mandatory_sampling_failure)
        self.assertIs(
            report.gpu_status, harness.MemoryCounterAvailability.UNRELIABLE
        )
        monitor.close()
        # The abandoned GPU query is never closed, yet the mandatory
        # process sampler still closes and the failure stays advisory.
        self.assertEqual(gpu.closed, 0)
        self.assertEqual(sampler.closed, 1)
        report, _override = monitor.build_report()
        self.assertFalse(report.mandatory_shutdown_failure)

    def test_gpu_timeout_never_prevents_a_stable_verdict(self):
        script = ["run"] * 20
        script[1] = "timeout"  # the baseline GPU observation hangs
        executor = ScriptedNativeExecutor(script)
        clock = FakeClock(100.0)
        event = AutoAdvanceEvent(clock, 12)
        gpu = FakeGpu(
            [
                harness.GpuEvidenceReading(
                    harness.MemoryCounterAvailability.AVAILABLE, 4096
                )
            ]
        )
        sampler = FunctionSampler(lambda index: reading())
        monitor = build_monitor(
            sampler=sampler,
            clock=clock,
            event=event,
            gpu=gpu,
            native_executor=executor,
        )
        monitor.take_baseline()
        monitor._run_sampling_loop()
        monitor.close()
        report, override = monitor.build_report()
        self.assertIsNone(override)
        self.assertIs(report.verdict, harness.MemoryStabilityVerdict.STABLE)
        self.assertIs(
            report.gpu_status, harness.MemoryCounterAvailability.UNRELIABLE
        )

    def test_join_timeout_abandons_resources_and_prevents_stable(self):
        gate = threading.Event()
        entered = threading.Event()
        finished = threading.Event()

        class BlockingSampler(FunctionSampler):
            def sample(self):
                self.calls += 1
                if self.calls == 1:
                    return reading()
                entered.set()
                gate.wait()
                finished.set()
                return reading(9999, 9999)

        sampler = BlockingSampler(lambda _index: reading())
        monitor = harness.MemoryStabilityMonitor(
            sampler=sampler,
            thresholds=limits(),
            pid=4242,
            process_creation_time_100ns=123456789,
            executable_path=f"D:\\Sim\\{SHIPPING}",
            executable_basename=SHIPPING,
            monotonic=FakeClock(100.0),
            stop_event=ScriptedEvent([False]),
            thread_factory=harness._default_memory_thread_factory,
            native_call_timeout_seconds=2.0,
            join_timeout_seconds=0.05,
        )
        monitor.take_baseline()
        monitor.start()
        self.assertTrue(entered.wait(1.0))
        worker = monitor._thread
        self.assertTrue(worker.daemon)
        try:
            monitor.request_stop()
            started = time.perf_counter()
            monitor.join()
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 1.0)
            self.assertTrue(worker.is_alive())
            monitor.join()  # a timed-out join is never retried
            self.assertTrue(worker.is_alive())
            monitor.take_final_sample()
            monitor.close()
            self.assertEqual(sampler.calls, 2)
            self.assertEqual(sampler.closed, 0)
            report, _override = monitor.build_report()
            self.assertEqual(report.total_sample_count, 1)
            self.assertTrue(report.mandatory_shutdown_failure)
            self.assertIsNot(
                report.verdict, harness.MemoryStabilityVerdict.STABLE
            )
        finally:
            gate.set()
            self.assertTrue(finished.wait(1.0))
            worker.join(1.0)
        self.assertFalse(worker.is_alive())
        # The late process reading was discarded after join abandonment.
        report, _override = monitor.build_report()
        self.assertEqual(report.total_sample_count, 1)

    def test_close_while_the_loop_thread_still_runs_is_refused(self):
        class RunningThread:
            def __init__(self, target):
                pass

            def start(self):
                pass

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return True

        sampler = FunctionSampler(lambda index: reading())
        monitor = build_monitor(sampler=sampler, thread_factory=RunningThread)
        monitor.take_baseline()
        monitor.start()
        monitor.close()  # close without a successful join first
        self.assertEqual(sampler.closed, 0)
        report, _override = monitor.build_report()
        self.assertTrue(report.mandatory_shutdown_failure)

    def test_monitor_validates_bounded_execution_configuration(self):
        sampler = FunctionSampler(lambda index: reading())
        with self.assertRaises(TypeError):
            build_monitor(sampler=sampler, native_executor=123)
        for bad in (0.0, -1.0, math.nan, math.inf, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    harness.MemoryStabilityMonitor(
                        sampler=sampler,
                        thresholds=limits(),
                        native_call_timeout_seconds=bad,
                    )
                with self.assertRaises(ValueError):
                    harness.MemoryStabilityMonitor(
                        sampler=sampler,
                        thresholds=limits(),
                        join_timeout_seconds=bad,
                    )

    def test_default_factory_bounds_the_native_open_calls(self):
        source = inspect.getsource(harness._default_memory_monitor_factory)
        self.assertIn("_run_native_call_bounded", source)
        self.assertNotIn(
            "WindowsProcessMemorySampler.open(", source.replace(
                "lambda: module.WindowsProcessMemorySampler.open(", ""
            )
        )


# ======================================================================
# Startup, cleanup, and handle ownership
# ======================================================================


class StartupCleanupTest(unittest.TestCase):
    def ready_monitor(self, **kwargs):
        sampler = kwargs.pop(
            "sampler", FunctionSampler(lambda index: reading())
        )
        monitor = build_monitor(sampler=sampler, **kwargs)
        monitor.take_baseline()
        return monitor, sampler

    def test_thread_construction_failure_is_classified_and_closes(self):
        def factory(target):
            raise RuntimeError("no thread")

        monitor, sampler = self.ready_monitor(thread_factory=factory)
        with self.assertRaisesRegex(RuntimeError, "no thread"):
            monitor.start()
        self.assertIs(
            monitor.admission_block_reason(),
            harness.TerminationReason.MEMORY_SAMPLING_FAILURE,
        )
        monitor.join()  # nothing was constructed successfully: no join
        monitor.close()
        self.assertEqual(sampler.closed, 1)
        report, override = monitor.build_report()
        self.assertIs(
            override, harness.TerminationReason.MEMORY_SAMPLING_FAILURE
        )
        self.assertTrue(
            any("thread construction" in text for text in report.diagnostics)
        )

    def test_thread_start_failure_never_joins_the_unstarted_thread(self):
        class UnstartableThread:
            def __init__(self, target):
                self.join_calls = 0

            def start(self):
                raise RuntimeError("start refused")

            def join(self, timeout=None):
                self.join_calls += 1

            def is_alive(self):
                return False

        threads = []

        def factory(target):
            thread = UnstartableThread(target)
            threads.append(thread)
            return thread

        monitor, sampler = self.ready_monitor(thread_factory=factory)
        with self.assertRaisesRegex(RuntimeError, "start refused"):
            monitor.start()
        monitor.join()
        self.assertEqual(threads[0].join_calls, 0)
        monitor.close()
        self.assertEqual(sampler.closed, 1)
        report, _override = monitor.build_report()
        self.assertTrue(
            any("thread start" in text for text in report.diagnostics)
        )
        self.assertTrue(report.mandatory_sampling_failure)

    def test_stop_failure_is_recorded_and_prevents_stable(self):
        class BrokenStopEvent(ScriptedEvent):
            def set(self):
                raise RuntimeError("set failed")

        monitor, sampler = self.ready_monitor(event=BrokenStopEvent())
        monitor.request_stop()  # never raises; recorded as evidence
        monitor.close()
        self.assertEqual(sampler.closed, 1)
        report, _override = monitor.build_report()
        self.assertTrue(report.mandatory_shutdown_failure)
        self.assertIsNot(report.verdict, harness.MemoryStabilityVerdict.STABLE)
        self.assertTrue(
            any("stop request failed" in text for text in report.diagnostics)
        )

    def test_process_close_failure_is_mandatory_evidence(self):
        class FailingCloseSampler(FunctionSampler):
            def close(self):
                super().close()
                raise RuntimeError("close boom")

        monitor, _clock, _event, _sampler = run_monitor(false_waits=12)
        # Re-run the stable scenario with a sampler whose close fails.
        clock = FakeClock(100.0)
        event = AutoAdvanceEvent(clock, 12)
        sampler = FailingCloseSampler(lambda index: reading())
        monitor = build_monitor(sampler=sampler, clock=clock, event=event)
        monitor.take_baseline()
        monitor._run_sampling_loop()
        monitor.close()
        monitor.close()
        self.assertEqual(sampler.closed, 1)  # attempted exactly once
        report, _override = monitor.build_report()
        self.assertTrue(report.mandatory_shutdown_failure)
        self.assertIs(
            report.verdict, harness.MemoryStabilityVerdict.INDETERMINATE
        )
        self.assertTrue(
            any(
                "process sampler close failed" in text
                for text in report.diagnostics
            )
        )

    def test_gpu_close_failure_stays_advisory_and_keeps_stable(self):
        class FailingCloseGpu(FakeGpu):
            def close(self):
                super().close()
                raise RuntimeError("gpu close boom")

        gpu = FailingCloseGpu(
            [
                harness.GpuEvidenceReading(
                    harness.MemoryCounterAvailability.NO_ACTIVITY
                )
            ]
        )
        monitor, _clock, _event, sampler = run_monitor(
            false_waits=12, gpu=gpu
        )
        monitor.close()
        self.assertEqual(gpu.closed, 1)
        self.assertEqual(sampler.closed, 1)  # gpu failure never blocks it
        report, _override = monitor.build_report()
        self.assertFalse(report.mandatory_shutdown_failure)
        self.assertIs(report.verdict, harness.MemoryStabilityVerdict.STABLE)
        self.assertTrue(
            any(
                "gpu sampler close failed" in text
                for text in report.diagnostics
            )
        )

    def test_gpu_base_exceptions_never_skip_process_close(self):
        for error in (
            KeyboardInterrupt("gpu interrupt"),
            SystemExit("gpu exit"),
        ):
            with self.subTest(error=type(error).__name__):
                class InterruptedGpu(FakeGpu):
                    def close(self):
                        super().close()
                        raise error

                gpu = InterruptedGpu(
                    [
                        harness.GpuEvidenceReading(
                            harness.MemoryCounterAvailability.NO_ACTIVITY
                        )
                    ]
                )
                monitor, _clock, _event, sampler = run_monitor(
                    false_waits=12, gpu=gpu
                )
                with self.assertRaises(type(error)) as caught:
                    monitor.close()
                self.assertIs(caught.exception, error)
                self.assertEqual(gpu.closed, 1)
                self.assertEqual(sampler.closed, 1)
                # Both per-resource stages were finalized despite the
                # interruption, so cleanup cannot run twice.
                monitor.close()
                self.assertEqual(gpu.closed, 1)
                self.assertEqual(sampler.closed, 1)

    def test_gpu_base_exception_preserves_identity_with_process_failure(self):
        interrupt = KeyboardInterrupt("first cleanup interrupt")

        class InterruptedGpu(FakeGpu):
            def close(self):
                super().close()
                raise interrupt

        class FailingCloseSampler(FunctionSampler):
            def close(self):
                super().close()
                raise RuntimeError("process close boom")

        gpu = InterruptedGpu(
            [
                harness.GpuEvidenceReading(
                    harness.MemoryCounterAvailability.NO_ACTIVITY
                )
            ]
        )
        sampler = FailingCloseSampler(lambda index: reading())
        monitor = build_monitor(sampler=sampler, gpu=gpu)
        monitor.take_baseline()
        with self.assertRaises(KeyboardInterrupt) as caught:
            monitor.close()
        self.assertIs(caught.exception, interrupt)
        self.assertEqual(gpu.closed, 1)
        self.assertEqual(sampler.closed, 1)
        self.assertTrue(monitor._gpu_close_finalized)
        self.assertTrue(monitor._process_close_finalized)
        report, _override = monitor.build_report()
        self.assertTrue(report.mandatory_shutdown_failure)
        self.assertIsNot(report.verdict, harness.MemoryStabilityVerdict.STABLE)
        diagnostics = "\n".join(report.diagnostics)
        self.assertIn("gpu sampler close interrupted", diagnostics)
        self.assertIn("process sampler close failed", diagnostics)

    def test_production_pdh_close_failure_reaches_monitor_diagnostics(self):
        scenario = PdhScenario(close_status=0xC0000BC0)
        with production_pdh_backend(scenario) as backend:
            native_gpu = sampler_module.GpuDedicatedMemorySampler(
                42, backend=backend
            )
            self.assertIs(
                native_gpu.open(), sampler_module.GpuAvailability.AVAILABLE
            )
            gpu = harness._WindowsGpuSamplerAdapter(native_gpu)
            monitor, _clock, _event, sampler = run_monitor(
                false_waits=12, gpu=gpu
            )
            monitor.close()
        self.assertEqual(
            scenario.closed_queries, [PdhScenario.QUERY_HANDLE]
        )
        self.assertEqual(sampler.closed, 1)
        report, _override = monitor.build_report()
        self.assertFalse(report.mandatory_shutdown_failure)
        self.assertIs(report.verdict, harness.MemoryStabilityVerdict.STABLE)
        self.assertTrue(
            any(
                "PdhCloseQuery failed with 0xC0000BC0" in text
                for text in report.diagnostics
            )
        )

    def test_multiple_simultaneous_cleanup_failures_are_all_recorded(self):
        class BrokenStopEvent(ScriptedEvent):
            def set(self):
                raise RuntimeError("set failed")

        class FailingCloseSampler(FunctionSampler):
            def close(self):
                super().close()
                raise RuntimeError("process close boom")

        class FailingCloseGpu(FakeGpu):
            def close(self):
                super().close()
                raise RuntimeError("gpu close boom")

        gpu = FailingCloseGpu(
            [
                harness.GpuEvidenceReading(
                    harness.MemoryCounterAvailability.NO_ACTIVITY
                )
            ]
        )
        sampler = FailingCloseSampler(lambda index: reading())
        monitor = build_monitor(
            sampler=sampler, gpu=gpu, event=BrokenStopEvent()
        )
        monitor.take_baseline()
        monitor.request_stop()
        monitor.close()
        self.assertEqual(sampler.closed, 1)
        self.assertEqual(gpu.closed, 1)
        report, _override = monitor.build_report()
        self.assertTrue(report.mandatory_shutdown_failure)
        texts = "\n".join(report.diagnostics)
        self.assertIn("stop request failed", texts)
        self.assertIn("gpu sampler close failed", texts)
        self.assertIn("process sampler close failed", texts)

    def test_wrapper_runs_every_cleanup_stage_and_raises_the_first_failure(
        self,
    ):
        stop_error = RuntimeError("stop boom")
        monitor = FakeMonitor(stop_error=stop_error)
        clock = FakeClock(0.0)

        def one_cycle(*_args, **_kwargs):
            clock.value = 130.0
            return successful_cycle_report()

        with unittest.mock.patch.object(
            harness,
            "run_sensor_suite_validation_cycle",
            side_effect=one_cycle,
        ):
            with self.assertRaises(RuntimeError) as caught:
                harness.run_repeated_sensor_suite_validation_with_memory_stability(
                    memory_environ(),
                    importer=unittest.mock.Mock(name="importer"),
                    monotonic=clock,
                    monitor_factory=lambda memory_config: monitor,
                )
        self.assertIs(caught.exception, stop_error)
        # The failing stop stage never prevented the later safe stages.
        self.assertIn("join", monitor.journal)
        self.assertIn("final", monitor.journal)
        self.assertIn("close", monitor.journal)
        self.assertIn("report", monitor.journal)
        self.assertLess(monitor.journal.index("join"), monitor.journal.index("final"))
        self.assertLess(monitor.journal.index("final"), monitor.journal.index("close"))

    def test_stop_failure_still_allows_safe_final_sample(self):
        class BrokenStopEvent(ScriptedEvent):
            def set(self):
                raise RuntimeError("set failed")

        clock = FakeClock(100.0)
        sampler = FunctionSampler(lambda index: reading(1000 + index, 2000))
        monitor = build_monitor(
            sampler=sampler, clock=clock, event=BrokenStopEvent()
        )
        monitor.take_baseline()
        monitor.request_stop()
        clock.value = 101.0
        monitor.take_final_sample()
        monitor.close()
        self.assertEqual(sampler.calls, 2)
        self.assertEqual(sampler.closed, 1)
        report, _override = monitor.build_report()
        self.assertTrue(report.mandatory_shutdown_failure)
        self.assertEqual(report.samples[-1].elapsed_seconds, 1.0)

    def test_final_sample_failure_does_not_prevent_process_close(self):
        sampler = FunctionSampler(
            lambda index: reading() if index == 1 else RuntimeError("final boom")
        )
        clock = FakeClock(100.0)
        monitor = build_monitor(sampler=sampler, clock=clock)
        monitor.take_baseline()
        clock.value = 101.0
        monitor.take_final_sample()
        monitor.close()
        self.assertEqual(sampler.calls, 2)
        self.assertEqual(sampler.closed, 1)
        report, override = monitor.build_report()
        self.assertTrue(report.mandatory_sampling_failure)
        self.assertIs(
            override, harness.TerminationReason.MEMORY_SAMPLING_FAILURE
        )
        self.assertTrue(any("final boom" in text for text in report.diagnostics))

    def test_wrapper_final_failure_closes_and_builds_before_raising(self):
        final_error = RuntimeError("final stage failed")
        monitor = FakeMonitor(final_error=final_error)
        clock = FakeClock(0.0)

        def finish_soak(*_args, **_kwargs):
            clock.value = 130.0
            return successful_cycle_report()

        with unittest.mock.patch.object(
            harness,
            "run_sensor_suite_validation_cycle",
            side_effect=finish_soak,
        ):
            with self.assertRaises(RuntimeError) as caught:
                harness.run_repeated_sensor_suite_validation_with_memory_stability(
                    memory_environ(),
                    monotonic=clock,
                    monitor_factory=lambda memory_config: monitor,
                )
        self.assertIs(caught.exception, final_error)
        self.assertLess(monitor.journal.index("final"), monitor.journal.index("close"))
        self.assertLess(monitor.journal.index("close"), monitor.journal.index("report"))

    def test_original_base_exception_outranks_final_stage_failure(self):
        original = KeyboardInterrupt("original")
        monitor = FakeMonitor(final_error=SystemExit("final"))
        with unittest.mock.patch.object(
            harness,
            "run_sensor_suite_validation_cycle",
            side_effect=original,
        ):
            with self.assertRaises(KeyboardInterrupt) as caught:
                harness.run_repeated_sensor_suite_validation_with_memory_stability(
                    memory_environ(),
                    monotonic=FakeClock(0.0),
                    monitor_factory=lambda memory_config: monitor,
                )
        self.assertIs(caught.exception, original)
        self.assertIn("final", monitor.journal)
        self.assertIn("close", monitor.journal)
        self.assertIn("report", monitor.journal)

    def test_escaping_base_exception_with_every_cleanup_stage_failing(self):
        interrupt = KeyboardInterrupt("operator interrupt")
        monitor = FakeMonitor(
            stop_error=RuntimeError("stop boom"),
            join_error=RuntimeError("join boom"),
            close_error=RuntimeError("close boom"),
        )
        with unittest.mock.patch.object(
            harness,
            "run_sensor_suite_validation_cycle",
            side_effect=interrupt,
        ):
            with self.assertRaises(KeyboardInterrupt) as caught:
                harness.run_repeated_sensor_suite_validation_with_memory_stability(
                    memory_environ(),
                    monotonic=FakeClock(0.0),
                    monitor_factory=lambda memory_config: monitor,
                )
        self.assertIs(caught.exception, interrupt)
        # Every cleanup stage was still attempted despite its predecessors.
        self.assertIn("stop", monitor.journal)
        self.assertIn("join", monitor.journal)
        self.assertIn("close", monitor.journal)


# ======================================================================
# Bounded retention
# ======================================================================


class RetentionTest(unittest.TestCase):
    def run_total(self, total_samples, produce=None):
        monitor, _clock, _event, sampler = run_monitor(
            produce=produce, false_waits=total_samples - 1
        )
        report, _override = monitor.build_report()
        return report, sampler

    def test_boundary_totals_255_256_and_257(self):
        for total in (255, 256, 257):
            with self.subTest(total=total):
                report, _sampler = self.run_total(total)
                retained = min(total, 256)
                self.assertEqual(report.total_sample_count, total)
                self.assertEqual(report.retained_sample_count, retained)
                self.assertEqual(report.omitted_sample_count, total - retained)
                self.assertEqual(len(report.samples), retained)

    def test_exact_first_sixteen_and_final_240_ordering(self):
        report, _sampler = self.run_total(300)
        elapsed = tuple(sample.elapsed_seconds for sample in report.samples)
        expected_first = tuple(5.0 * index for index in range(16))
        expected_final = tuple(5.0 * index for index in range(60, 300))
        self.assertEqual(elapsed[:16], expected_first)
        self.assertEqual(elapsed[16:], expected_final)
        self.assertEqual(report.omitted_sample_count, 44)
        self.assertEqual(elapsed, tuple(sorted(elapsed)))

    def test_large_synthetic_run_keeps_exact_counts(self):
        report, _sampler = self.run_total(701)
        self.assertEqual(report.total_sample_count, 701)
        self.assertEqual(report.retained_sample_count, 256)
        self.assertEqual(report.omitted_sample_count, 445)
        self.assertEqual(report.samples[-1].elapsed_seconds, 3500.0)

    def test_metrics_include_omitted_middle_peak(self):
        # The peak lands on sample index 30 (elapsed 150), which is omitted
        # from retention for a 300-sample run, yet metrics must keep it.
        def produce(index):
            if index == 31:
                return reading(working_set=9_000_000, private_commit=2000)
            return reading(working_set=1000 + index, private_commit=2000)

        report, _sampler = self.run_total(300, produce=produce)
        omitted_elapsed = 150.0
        self.assertNotIn(
            omitted_elapsed,
            tuple(sample.elapsed_seconds for sample in report.samples),
        )
        self.assertEqual(report.working_set.peak_bytes, 9_000_000)
        self.assertEqual(report.working_set.peak_elapsed_seconds, 150.0)
        self.assertEqual(report.working_set.sample_count, 300)
        self.assertEqual(report.working_set.final_bytes, 1300)

    def test_retention_helper_orders_first_and_rolling_final(self):
        retention = harness._BoundedMemorySampleRetention()
        for index in range(400):
            retention.add(
                harness.ProcessMemorySample(float(index), 1, 1, None)
            )
        snapshot = retention.snapshot()
        self.assertEqual(retention.total_count, 400)
        self.assertEqual(len(snapshot), 256)
        self.assertEqual(
            tuple(sample.elapsed_seconds for sample in snapshot[:16]),
            tuple(float(index) for index in range(16)),
        )
        self.assertEqual(
            tuple(sample.elapsed_seconds for sample in snapshot[16:]),
            tuple(float(index) for index in range(160, 400)),
        )

    def test_diagnostics_are_bounded_with_exact_omitted_count(self):
        monitor = build_monitor(sampler=FunctionSampler(lambda index: reading()))
        for index in range(12):
            monitor._record_sampling_failure(
                "probe", RuntimeError(f"failure {index}")
            )
        report, _override = monitor.build_report()
        self.assertEqual(len(report.diagnostics), 8)
        self.assertEqual(report.omitted_diagnostic_count, 4)
        self.assertTrue(
            all(
                len(text) <= harness.MAX_CLEANUP_ERROR_LENGTH
                for text in report.diagnostics
            )
        )

    def test_pagefile_alias_inconsistency_is_noted_once(self):
        monitor, _clock, _event, _sampler = run_monitor(
            produce=lambda index: reading(pagefile=999999), false_waits=4
        )
        report, _override = monitor.build_report()
        alias_notes = [
            text for text in report.diagnostics if "PagefileUsage" in text
        ]
        self.assertEqual(len(alias_notes), 1)
        # The alias diagnostic never turns into a failure or extra metric.
        self.assertIsNone(monitor.admission_block_reason())


# ======================================================================
# Metrics and thresholds
# ======================================================================


def fed_accumulator(samples):
    accumulator = harness._MemoryCounterAccumulator()
    for elapsed, value in samples:
        accumulator.add(elapsed, value)
    return accumulator


def metric(samples, **threshold_overrides):
    values = dict(
        max_peak_growth_bytes=10**9,
        max_final_growth_bytes=10**9,
        max_final_growth_percent=10000.0,
        max_slope_bytes_per_minute=None,
    )
    values.update(threshold_overrides)
    return harness._memory_metric_report(
        "working_set", fed_accumulator(samples), **values
    )


def linear_samples(count, *, start=1000, step=10, interval=5.0):
    return tuple(
        (interval * index, start + step * index) for index in range(count)
    )


class MetricsTest(unittest.TestCase):
    def test_growth_remains_signed_for_shrinking_memory(self):
        report = metric(((0.0, 1000), (60.0, 800)))
        self.assertEqual(report.growth_bytes, -200)
        self.assertEqual(report.peak_growth_bytes, 0)
        self.assertEqual(report.growth_percent, -20.0)
        self.assertEqual(report.final_bytes, 800)
        self.assertTrue(report.passed)

    def test_peak_metrics_use_the_earliest_peak(self):
        report = metric(
            ((0.0, 1000), (5.0, 1500), (10.0, 1200), (15.0, 1500))
        )
        self.assertEqual(report.peak_bytes, 1500)
        self.assertEqual(report.peak_elapsed_seconds, 5.0)
        self.assertEqual(report.peak_growth_bytes, 500)
        self.assertEqual(report.peak_growth_percent, 50.0)

    def test_threshold_equality_passes_and_one_byte_over_fails(self):
        samples = ((0.0, 1000), (60.0, 1100))
        equal = metric(samples, max_final_growth_bytes=100)
        self.assertTrue(equal.passed)
        over = metric(samples, max_final_growth_bytes=99)
        self.assertEqual(over.failed_thresholds, ("max_final_growth_bytes",))
        self.assertFalse(over.passed)

        peak_equal = metric(samples, max_peak_growth_bytes=100)
        self.assertTrue(peak_equal.passed)
        peak_over = metric(samples, max_peak_growth_bytes=99)
        self.assertIn("max_peak_growth_bytes", peak_over.failed_thresholds)

    def test_percentage_boundaries(self):
        samples = ((0.0, 1000), (60.0, 1100))
        self.assertTrue(metric(samples, max_final_growth_percent=10.0).passed)
        failed = metric(samples, max_final_growth_percent=9.999)
        self.assertEqual(
            failed.failed_thresholds, ("max_final_growth_percent",)
        )

    def test_percentage_adjacent_float_boundaries_are_exact(self):
        samples = ((0.0, 40), (60.0, 41))  # exactly 2.5 percent
        equal = metric(samples, max_final_growth_percent=2.5)
        below_actual = metric(
            samples,
            max_final_growth_percent=math.nextafter(2.5, -math.inf),
        )
        above_actual = metric(
            samples,
            max_final_growth_percent=math.nextafter(2.5, math.inf),
        )
        self.assertTrue(equal.passed)
        self.assertFalse(below_actual.passed)
        self.assertEqual(
            below_actual.failed_thresholds,
            ("max_final_growth_percent",),
        )
        self.assertTrue(above_actual.passed)

    def test_percentage_2_4_exceeds_the_adjacent_lower_float(self):
        report = metric(
            ((0.0, 1000), (60.0, 1024)),
            max_final_growth_percent=2.3999999999999995,
        )
        self.assertEqual(report.growth_percent, 2.4)
        self.assertEqual(
            report.failed_thresholds, ("max_final_growth_percent",)
        )
        self.assertFalse(report.passed)

    def test_zero_baseline_produces_none_percentages(self):
        report = metric(((0.0, 0), (60.0, 100)))
        self.assertIsNone(report.growth_percent)
        self.assertIsNone(report.peak_growth_percent)
        self.assertEqual(
            report.unevaluated_thresholds, ("max_final_growth_percent",)
        )
        self.assertFalse(report.passed)

    def test_large_integer_counters_keep_exact_growth(self):
        base = 2**62
        report = metric(((0.0, base), (60.0, base + 3)))
        self.assertEqual(report.growth_bytes, 3)
        self.assertEqual(report.peak_growth_bytes, 3)
        self.assertEqual(report.start_bytes, base)

    def test_slope_requires_five_samples_and_sixty_seconds(self):
        four = linear_samples(4, interval=30.0)
        self.assertIsNone(metric(four).slope_bytes_per_minute)
        five_short = linear_samples(5, interval=14.9)
        self.assertIsNone(metric(five_short).slope_bytes_per_minute)
        five_exact = linear_samples(5, interval=15.0)
        self.assertIsNotNone(metric(five_exact).slope_bytes_per_minute)

    def test_linear_data_yields_exact_slope(self):
        samples = linear_samples(13, step=10, interval=5.0)
        report = metric(samples)
        self.assertAlmostEqual(report.slope_bytes_per_minute, 120.0)

    def test_constant_data_yields_zero_slope(self):
        samples = tuple((5.0 * index, 5000) for index in range(13))
        report = metric(samples)
        self.assertAlmostEqual(report.slope_bytes_per_minute, 0.0)

    def test_online_slope_is_stable_with_large_offsets(self):
        base = 10**15
        samples = tuple(
            (5.0 * index, base + 7 * index) for index in range(25)
        )
        report = metric(samples)
        self.assertAlmostEqual(
            report.slope_bytes_per_minute, 84.0, places=6
        )

    def test_slope_threshold_equality_passes_and_above_fails(self):
        samples = linear_samples(13, step=10, interval=5.0)  # 120 bytes/min
        passing = metric(samples, max_slope_bytes_per_minute=120.0)
        self.assertNotIn(
            "max_slope_bytes_per_minute", passing.failed_thresholds
        )
        failing = metric(samples, max_slope_bytes_per_minute=119.0)
        self.assertIn(
            "max_slope_bytes_per_minute", failing.failed_thresholds
        )

    def test_configured_slope_without_enough_data_is_unevaluated(self):
        samples = ((0.0, 1000), (30.0, 1010))
        report = metric(samples, max_slope_bytes_per_minute=10.0)
        self.assertIn(
            "max_slope_bytes_per_minute", report.unevaluated_thresholds
        )
        self.assertFalse(report.passed)

    def test_empty_accumulator_produces_no_report(self):
        self.assertIsNone(metric(()))


class HighOffsetSlopeTest(unittest.TestCase):
    """OLS must stay byte-exact above the float exact-integer range."""

    @staticmethod
    def offset_samples(base, *, step=1, interval=15.0, count=5):
        return tuple(
            (interval * index, base + step * index) for index in range(count)
        )

    def test_small_growth_above_two_to_the_62_fails_a_one_byte_slope(self):
        # The review's required regression: 2**62 + [0..4] over 60 seconds
        # is exactly 4 bytes/minute and must fail a 1 byte/minute limit,
        # never flatten to 0.0 and pass.
        report = metric(
            self.offset_samples(2**62), max_slope_bytes_per_minute=1.0
        )
        self.assertEqual(report.slope_bytes_per_minute, 4.0)
        self.assertIn(
            "max_slope_bytes_per_minute", report.failed_thresholds
        )
        self.assertFalse(report.passed)

    def test_growth_reaching_two_to_the_63_boundary_is_exact(self):
        report = metric(
            self.offset_samples(2**63 - 5), max_slope_bytes_per_minute=1.0
        )
        self.assertEqual(report.slope_bytes_per_minute, 4.0)
        self.assertIn(
            "max_slope_bytes_per_minute", report.failed_thresholds
        )
        self.assertEqual(report.final_bytes, 2**63 - 1)

    def test_growth_at_the_two_to_the_53_boundary_is_exact(self):
        report = metric(self.offset_samples(2**53))
        self.assertEqual(report.slope_bytes_per_minute, 4.0)

    def test_constant_data_at_high_offset_yields_zero_slope(self):
        report = metric(self.offset_samples(2**62, step=0))
        self.assertEqual(report.slope_bytes_per_minute, 0.0)

    def test_slope_equality_at_high_offset_still_passes(self):
        report = metric(
            self.offset_samples(2**62), max_slope_bytes_per_minute=4.0
        )
        self.assertNotIn(
            "max_slope_bytes_per_minute", report.failed_thresholds
        )
        self.assertTrue(report.passed)

    def test_negative_slope_at_high_offset_stays_signed_and_passes(self):
        base = 2**62
        samples = tuple(
            (15.0 * index, base + 4 - index) for index in range(5)
        )
        report = metric(samples, max_slope_bytes_per_minute=1.0)
        self.assertEqual(report.slope_bytes_per_minute, -4.0)
        self.assertNotIn(
            "max_slope_bytes_per_minute", report.failed_thresholds
        )
        self.assertEqual(report.growth_bytes, -4)

    def test_duplicate_timestamps_are_handled_safely(self):
        base = 2**62
        samples = (
            (0.0, base),
            (0.0, base + 1),
            (30.0, base + 2),
            (60.0, base + 3),
            (60.0, base + 4),
        )
        report = metric(samples)
        self.assertEqual(report.slope_bytes_per_minute, 3.0)

    def test_entirely_duplicated_timestamps_yield_no_slope(self):
        accumulator = harness._MemoryCounterAccumulator()
        for value in range(5):
            accumulator.add(10.0, 2**62 + value)
        self.assertIsNone(accumulator.slope_bytes_per_minute())


# ======================================================================
# GPU evidence
# ======================================================================


class FakePdhBackend:
    def __init__(
        self,
        *,
        open_error=None,
        add_error=None,
        close_error=None,
        collect_errors=(),
        arrays=(),
    ):
        self.open_error = open_error
        self.add_error = add_error
        self.close_error = close_error
        self.collect_errors = list(collect_errors)
        self.arrays = list(arrays)
        self.paths = []
        self.closed_queries = []
        self.open_calls = 0

    def open_query(self):
        self.open_calls += 1
        if self.open_error is not None:
            raise self.open_error
        return "query"

    def add_english_counter(self, query, path):
        self.paths.append(path)
        if self.add_error is not None:
            raise self.add_error
        return "counter"

    def collect(self, query):
        if self.collect_errors:
            error = self.collect_errors.pop(0)
            if error is not None:
                raise error

    def formatted_counter_array(self, counter):
        if not self.arrays:
            return ()
        entry = self.arrays.pop(0)
        if isinstance(entry, BaseException):
            raise entry
        return entry

    def close_query(self, query):
        self.closed_queries.append(query)
        if self.close_error is not None:
            raise self.close_error


VALID = sampler_module.PDH_CSTATUS_VALID_DATA


class GpuEvidenceTest(unittest.TestCase):
    def open_sampler(self, backend, pid=4242):
        gpu = sampler_module.GpuDedicatedMemorySampler(pid, backend=backend)
        status = gpu.open()
        return gpu, status

    def test_pdh_open_failure_is_preserved_without_published_state(self):
        error = OSError("no pdh")
        backend = FakePdhBackend(open_error=error)
        gpu = sampler_module.GpuDedicatedMemorySampler(4242, backend=backend)
        with self.assertRaises(OSError) as caught:
            gpu.open()
        self.assertIs(caught.exception, error)
        observed = gpu.sample()
        self.assertIs(
            observed.status, sampler_module.GpuAvailability.UNSUPPORTED
        )
        self.assertIsNone(observed.dedicated_bytes)

    def test_counter_add_failure_closes_query_and_preserves_identity(self):
        error = sampler_module.PdhQueryError("bad path")
        backend = FakePdhBackend(
            add_error=error
        )
        gpu = sampler_module.GpuDedicatedMemorySampler(4242, backend=backend)
        with self.assertRaises(sampler_module.PdhQueryError) as caught:
            gpu.open()
        self.assertIs(caught.exception, error)
        self.assertEqual(backend.closed_queries, ["query"])
        self.assertIsNone(gpu._query)
        self.assertIsNone(gpu._counter)
        gpu.close()
        self.assertEqual(backend.closed_queries, ["query"])

    def test_counter_add_base_exceptions_roll_back_once_and_are_not_swallowed(self):
        for error in (
            RuntimeError("add failed"),
            KeyboardInterrupt("add interrupted"),
            SystemExit("add exited"),
        ):
            with self.subTest(error=type(error).__name__):
                backend = FakePdhBackend(add_error=error)
                gpu = sampler_module.GpuDedicatedMemorySampler(
                    4242, backend=backend
                )
                with self.assertRaises(type(error)) as caught:
                    gpu.open()
                self.assertIs(caught.exception, error)
                self.assertEqual(backend.closed_queries, ["query"])
                self.assertIsNone(gpu._query)
                self.assertIsNone(gpu._counter)
                self.assertIs(
                    gpu.sample().status,
                    sampler_module.GpuAvailability.UNSUPPORTED,
                )
                gpu.close()
                self.assertEqual(backend.closed_queries, ["query"])

    def test_counter_add_primary_survives_failed_rollback_close(self):
        primary = KeyboardInterrupt("add interrupted")
        close_failure = sampler_module.PdhQueryError("close failed")
        backend = FakePdhBackend(
            add_error=primary, close_error=close_failure
        )
        gpu = sampler_module.GpuDedicatedMemorySampler(4242, backend=backend)
        with self.assertRaises(KeyboardInterrupt) as caught:
            gpu.open()
        self.assertIs(caught.exception, primary)
        self.assertEqual(backend.closed_queries, ["query"])
        notes = getattr(primary, "__notes__", ())
        self.assertEqual(len(notes), 1)
        self.assertIn("rollback close also failed", notes[0])
        self.assertIn("close failed", notes[0])
        self.assertLessEqual(
            len(notes[0]), sampler_module.MAX_NATIVE_ERROR_EVIDENCE_CHARACTERS
        )
        self.assertIsNone(gpu._query)
        self.assertIsNone(gpu._counter)

    def test_wildcard_gpu_process_memory_path_is_used(self):
        backend = FakePdhBackend(arrays=[()])
        self.open_sampler(backend)
        self.assertEqual(
            backend.paths,
            [sampler_module.GPU_PROCESS_MEMORY_COUNTER_PATH],
        )

    def test_no_matching_pid_instance_is_no_activity(self):
        backend = FakePdhBackend(
            arrays=[
                (
                    ("pid_999_luid_0x0_0x1_phys_0", VALID, 4096),
                    ("pid_1000_luid_0x0_0x1_phys_0", VALID, 8192),
                )
            ]
        )
        gpu, _status = self.open_sampler(backend, pid=4242)
        observed = gpu.sample()
        self.assertIs(
            observed.status, sampler_module.GpuAvailability.NO_ACTIVITY
        )
        self.assertIsNone(observed.dedicated_bytes)

    def test_multiple_adapter_instances_are_summed_uniquely(self):
        backend = FakePdhBackend(
            arrays=[
                (
                    ("pid_4242_luid_0x0_0x1_phys_0", VALID, 1000),
                    ("pid_4242_luid_0x0_0x2_phys_1", VALID, 500),
                    ("pid_4242_luid_0x0_0x1_phys_0", VALID, 777),
                    ("pid_999_luid_0x0_0x1_phys_0", VALID, 4096),
                )
            ]
        )
        gpu, _status = self.open_sampler(backend)
        observed = gpu.sample()
        self.assertIs(
            observed.status, sampler_module.GpuAvailability.AVAILABLE
        )
        # The duplicate LUID instance counts once; the foreign pid never.
        self.assertEqual(observed.dedicated_bytes, 1500)
        self.assertEqual(observed.instance_count, 2)

    def test_zero_total_reports_zero_or_unified_memory(self):
        backend = FakePdhBackend(
            arrays=[(("pid_4242_luid_0x0_0x1_phys_0", VALID, 0),)]
        )
        gpu, _status = self.open_sampler(backend)
        observed = gpu.sample()
        self.assertIs(
            observed.status,
            sampler_module.GpuAvailability.ZERO_OR_UNIFIED_MEMORY,
        )
        self.assertEqual(observed.dedicated_bytes, 0)

    def test_malformed_values_alone_are_unreliable(self):
        backend = FakePdhBackend(
            arrays=[
                (
                    ("pid_4242_luid_0x0_0x1_phys_0", 0xC0000BBB, 1000),
                    ("pid_4242_luid_0x0_0x2_phys_0", VALID, -5),
                )
            ]
        )
        gpu, _status = self.open_sampler(backend)
        observed = gpu.sample()
        self.assertIs(
            observed.status, sampler_module.GpuAvailability.UNRELIABLE
        )
        self.assertIsNone(observed.dedicated_bytes)

    def test_malformed_values_beside_valid_instances_sum_the_valid(self):
        backend = FakePdhBackend(
            arrays=[
                (
                    ("pid_4242_luid_0x0_0x1_phys_0", VALID, 2048),
                    ("pid_4242_luid_0x0_0x2_phys_0", 0xC0000BBB, 77),
                )
            ]
        )
        gpu, _status = self.open_sampler(backend)
        observed = gpu.sample()
        self.assertIs(
            observed.status, sampler_module.GpuAvailability.AVAILABLE
        )
        self.assertEqual(observed.dedicated_bytes, 2048)

    def test_collect_no_data_is_no_activity_and_errors_are_unreliable(self):
        backend = FakePdhBackend(
            collect_errors=[
                sampler_module.PdhNoDataError("empty"),
                sampler_module.PdhQueryError("broken"),
            ],
            arrays=[()],
        )
        gpu, _status = self.open_sampler(backend)
        self.assertIs(
            gpu.sample().status, sampler_module.GpuAvailability.NO_ACTIVITY
        )
        self.assertIs(
            gpu.sample().status, sampler_module.GpuAvailability.UNRELIABLE
        )

    def test_close_releases_the_query_once(self):
        backend = FakePdhBackend(arrays=[()])
        gpu, _status = self.open_sampler(backend)
        gpu.close()
        gpu.close()
        self.assertEqual(backend.closed_queries, ["query"])
        self.assertIs(
            gpu.sample().status, sampler_module.GpuAvailability.UNSUPPORTED
        )

    def test_invalid_pid_construction_is_rejected(self):
        for pid in (0, -1, True, "4242"):
            with self.subTest(pid=pid):
                with self.assertRaises(ValueError):
                    sampler_module.GpuDedicatedMemorySampler(pid)

    def test_monitor_preserves_previous_valid_evidence_on_failure(self):
        gpu = FakeGpu(
            [
                harness.GpuEvidenceReading(
                    harness.MemoryCounterAvailability.AVAILABLE, 500
                ),
                harness.GpuEvidenceReading(
                    harness.MemoryCounterAvailability.AVAILABLE, 900
                ),
                RuntimeError("query failed"),
            ]
        )
        monitor, _clock, _event, _sampler = run_monitor(
            false_waits=2, gpu=gpu
        )
        report, _override = monitor.build_report()
        self.assertIs(
            report.gpu_status, harness.MemoryCounterAvailability.UNRELIABLE
        )
        self.assertEqual(report.gpu_final_dedicated_bytes, 900)
        self.assertEqual(report.gpu_peak_dedicated_bytes, 900)
        # The failed reading is never replaced with a fabricated zero.
        self.assertIsNone(report.samples[-1].gpu_dedicated_bytes)
        self.assertEqual(report.samples[0].gpu_dedicated_bytes, 500)

    def test_gpu_status_never_gates_the_stability_verdict(self):
        for status in (
            harness.MemoryCounterAvailability.UNSUPPORTED,
            harness.MemoryCounterAvailability.NO_ACTIVITY,
            harness.MemoryCounterAvailability.UNRELIABLE,
            harness.MemoryCounterAvailability.ZERO_OR_UNIFIED_MEMORY,
        ):
            with self.subTest(status=status):
                gpu = FakeGpu(
                    [
                        harness.GpuEvidenceReading(
                            status,
                            0
                            if status
                            is harness.MemoryCounterAvailability.ZERO_OR_UNIFIED_MEMORY
                            else None,
                        )
                    ]
                )
                monitor, _clock, _event, _sampler = run_monitor(
                    false_waits=12, gpu=gpu
                )
                report, override = monitor.build_report()
                self.assertIs(
                    report.verdict, harness.MemoryStabilityVerdict.STABLE
                )
                self.assertIsNone(override)

    def test_gpu_adapter_translates_module_statuses(self):
        module_sample = sampler_module.GpuDedicatedSample(
            sampler_module.GpuAvailability.ZERO_OR_UNIFIED_MEMORY, 0, 1
        )
        adapter = harness._WindowsGpuSamplerAdapter(
            types.SimpleNamespace(
                sample=lambda: module_sample, close=lambda: None
            )
        )
        observed = adapter.sample()
        self.assertIs(
            observed.status,
            harness.MemoryCounterAvailability.ZERO_OR_UNIFIED_MEMORY,
        )
        self.assertEqual(observed.dedicated_bytes, 0)


# ======================================================================
# Verdict evaluation
# ======================================================================


class VerdictTest(unittest.TestCase):
    def test_stable_requires_minimum_samples_and_span(self):
        monitor, _clock, _event, _sampler = run_monitor(false_waits=12)
        report, override = monitor.build_report()
        self.assertIs(report.verdict, harness.MemoryStabilityVerdict.STABLE)
        self.assertIsNone(override)
        self.assertIs(
            report.identity, harness.ProcessIdentityStatus.PRESERVED
        )
        self.assertEqual(report.total_sample_count, 13)
        self.assertEqual(report.working_set.span_seconds, 60.0)

    def test_too_few_samples_or_short_span_is_indeterminate(self):
        for false_waits in (3, 11):
            with self.subTest(false_waits=false_waits):
                monitor, _clock, _event, _sampler = run_monitor(
                    false_waits=false_waits
                )
                report, override = monitor.build_report()
                self.assertIs(
                    report.verdict,
                    harness.MemoryStabilityVerdict.INDETERMINATE,
                )
                self.assertIsNone(override)

    def test_final_growth_breach_is_unstable_with_override(self):
        monitor, _clock, _event, _sampler = run_monitor(
            produce=lambda index: reading(working_set=1000 + 50 * index),
            false_waits=12,
            thresholds=limits(working_set_max_final_growth_bytes=100),
        )
        report, override = monitor.build_report()
        self.assertIs(report.verdict, harness.MemoryStabilityVerdict.UNSTABLE)
        self.assertIs(
            override, harness.TerminationReason.MEMORY_STABILITY_FAILURE
        )
        self.assertIn(
            "max_final_growth_bytes", report.working_set.failed_thresholds
        )

    def test_peak_breach_is_irreversible_and_blocks_admission(self):
        def produce(index):
            if index == 3:
                return reading(working_set=5000)
            return reading(working_set=1000)

        monitor, _clock, _event, _sampler = run_monitor(
            produce=produce,
            false_waits=12,
            thresholds=limits(working_set_max_peak_growth_bytes=1000),
        )
        self.assertIs(
            monitor.admission_block_reason(),
            harness.TerminationReason.MEMORY_STABILITY_FAILURE,
        )
        report, override = monitor.build_report()
        self.assertIs(report.verdict, harness.MemoryStabilityVerdict.UNSTABLE)
        self.assertIs(
            override, harness.TerminationReason.MEMORY_STABILITY_FAILURE
        )

    def test_slope_breach_alone_is_unstable_only_via_configured_threshold(self):
        produce = lambda index: reading(working_set=1000 + 10 * index)
        unlimited, _clock, _event, _sampler = run_monitor(
            produce=produce, false_waits=12
        )
        report, override = unlimited.build_report()
        self.assertIs(report.verdict, harness.MemoryStabilityVerdict.STABLE)

        limited, _clock2, _event2, _sampler2 = run_monitor(
            produce=produce,
            false_waits=12,
            thresholds=limits(working_set_max_slope_bytes_per_minute=1.0),
        )
        report, override = limited.build_report()
        self.assertIs(report.verdict, harness.MemoryStabilityVerdict.UNSTABLE)
        self.assertIs(
            override, harness.TerminationReason.MEMORY_STABILITY_FAILURE
        )

    def test_admission_reason_precedence_exit_then_sampling_then_peak(self):
        monitor = build_monitor(sampler=FunctionSampler(lambda index: reading()))
        with monitor._lock:
            monitor._process_exited = True
            monitor._sampling_failed = True
            monitor._peak_breached = True
        self.assertIs(
            monitor.admission_block_reason(),
            harness.TerminationReason.MEMORY_PROCESS_DISAPPEARED,
        )
        with monitor._lock:
            monitor._process_exited = False
        self.assertIs(
            monitor.admission_block_reason(),
            harness.TerminationReason.MEMORY_SAMPLING_FAILURE,
        )
        with monitor._lock:
            monitor._sampling_failed = False
        self.assertIs(
            monitor.admission_block_reason(),
            harness.TerminationReason.MEMORY_STABILITY_FAILURE,
        )

    def test_process_exit_outranks_a_recorded_peak_breach(self):
        def produce(index):
            if index == 2:
                return reading(working_set=10**7)
            if index == 4:
                return harness.MemoryMonitoredProcessExited("gone")
            return reading()

        monitor, _clock, _event, _sampler = run_monitor(
            produce=produce,
            false_waits=12,
            thresholds=limits(working_set_max_peak_growth_bytes=10),
        )
        report, override = monitor.build_report()
        self.assertIs(
            report.verdict, harness.MemoryStabilityVerdict.INDETERMINATE
        )
        self.assertIs(
            override, harness.TerminationReason.MEMORY_PROCESS_DISAPPEARED
        )

    def test_monitor_requires_memory_thresholds(self):
        with self.assertRaises(TypeError):
            harness.MemoryStabilityMonitor(
                sampler=FunctionSampler(lambda index: reading()),
                thresholds=None,
            )


# ======================================================================
# Outer-runner admission guard
# ======================================================================


class AdmissionGuardTest(unittest.TestCase):
    def run_guarded(self, reports, guard, *, environ=None):
        with unittest.mock.patch.object(
            harness,
            "run_sensor_suite_validation_cycle",
            side_effect=list(reports),
        ) as inner:
            aggregate = harness.run_repeated_sensor_suite_validation_cycles(
                environ
                if environ is not None
                else memory_environ(gate=None),
                importer=unittest.mock.Mock(name="importer"),
                monotonic=FakeClock(50.0),
                disappearance_timeout_seconds=2.5,
                admission_guard=guard,
            )
        return aggregate, inner

    def test_guard_veto_before_first_admission_runs_no_cycle(self):
        guard_calls = []

        def guard():
            guard_calls.append(True)
            return harness.TerminationReason.MEMORY_PROCESS_DISAPPEARED

        aggregate, inner = self.run_guarded((), guard)
        inner.assert_not_called()
        self.assertEqual(len(guard_calls), 1)
        self.assertEqual(aggregate.attempted_cycle_count, 0)
        self.assertIs(
            aggregate.termination_reason,
            harness.TerminationReason.MEMORY_PROCESS_DISAPPEARED,
        )
        self.assertIs(aggregate.success, False)

    def test_current_cycle_finishes_before_guard_termination(self):
        reasons = [None, harness.TerminationReason.MEMORY_SAMPLING_FAILURE]
        aggregate, inner = self.run_guarded(
            (successful_cycle_report(),), lambda: reasons.pop(0)
        )
        self.assertEqual(inner.call_count, 1)
        self.assertEqual(aggregate.completed_cycle_count, 1)
        self.assertEqual(aggregate.passed_cycle_count, 1)
        self.assertIs(
            aggregate.termination_reason,
            harness.TerminationReason.MEMORY_SAMPLING_FAILURE,
        )
        self.assertIs(aggregate.success, False)
        self.assertTrue(aggregate.cycles[0].success)

    def test_cycle_failure_outranks_any_pending_guard_reason(self):
        guard_calls = []

        def guard():
            guard_calls.append(True)
            if len(guard_calls) > 1:
                return harness.TerminationReason.MEMORY_STABILITY_FAILURE
            return None

        failed = successful_cycle_report(
            primary_error=harness.ValidationFailure("readiness", "failed", 1)
        )
        aggregate, inner = self.run_guarded((failed,), guard)
        self.assertEqual(inner.call_count, 1)
        self.assertEqual(len(guard_calls), 1)
        self.assertIs(
            aggregate.termination_reason,
            harness.TerminationReason.FUNCTIONAL_FAILURE,
        )

    def test_guard_reason_outranks_soak_duration(self):
        clock = FakeClock(0.0)
        reasons = [None, harness.TerminationReason.MEMORY_STABILITY_FAILURE]

        def finish_beyond_deadline(*_args, **_kwargs):
            clock.value = 130.0
            return successful_cycle_report()

        with unittest.mock.patch.object(
            harness,
            "run_sensor_suite_validation_cycle",
            side_effect=finish_beyond_deadline,
        ) as inner:
            aggregate = harness.run_repeated_sensor_suite_validation_cycles(
                memory_environ(gate=None),
                monotonic=clock,
                admission_guard=lambda: reasons.pop(0),
            )
        self.assertEqual(inner.call_count, 1)
        self.assertIs(
            aggregate.termination_reason,
            harness.TerminationReason.MEMORY_STABILITY_FAILURE,
        )

    def test_non_callable_guard_is_rejected_before_live_work(self):
        with unittest.mock.patch.object(
            harness, "run_sensor_suite_validation_cycle"
        ) as inner:
            with self.assertRaises(TypeError):
                harness.run_repeated_sensor_suite_validation_cycles(
                    memory_environ(gate=None), admission_guard="guard"
                )
        inner.assert_not_called()

    def test_guard_returning_wrong_type_raises(self):
        with unittest.mock.patch.object(
            harness, "run_sensor_suite_validation_cycle"
        ) as inner:
            with self.assertRaises(TypeError):
                harness.run_repeated_sensor_suite_validation_cycles(
                    memory_environ(gate=None),
                    monotonic=FakeClock(),
                    admission_guard=lambda: "STOP",
                )
        inner.assert_not_called()


# ======================================================================
# Outer wrapper integration
# ======================================================================


class WrapperIntegrationTest(unittest.TestCase):
    def run_wrapper(
        self,
        *,
        environ=None,
        reports=(),
        monitor=None,
        monitor_factory=None,
        clock=None,
        importer=None,
    ):
        active_environ = environ if environ is not None else memory_environ()
        factory = monitor_factory
        if factory is None and monitor is not None:
            factory = lambda memory_config: monitor
        with unittest.mock.patch.object(
            harness,
            "run_sensor_suite_validation_cycle",
            side_effect=list(reports),
        ) as inner:
            aggregate = (
                harness.run_repeated_sensor_suite_validation_with_memory_stability(
                    active_environ,
                    importer=(
                        importer
                        if importer is not None
                        else unittest.mock.Mock(name="importer")
                    ),
                    monotonic=clock if clock is not None else FakeClock(10.0),
                    disappearance_timeout_seconds=2.5,
                    monitor_factory=factory,
                )
            )
        return aggregate, inner

    def timed_clock(self):
        # One wrapper-entry read, one deadline construction, one admission
        # read, then the post-cycle admission rejection and final read.
        clock = FakeClock(0.0)
        return clock

    def successful_soak(self, monitor):
        clock = FakeClock(0.0)

        def one_cycle(*_args, **_kwargs):
            clock.value = 130.0
            return successful_cycle_report()

        with unittest.mock.patch.object(
            harness,
            "run_sensor_suite_validation_cycle",
            side_effect=one_cycle,
        ) as inner:
            aggregate = (
                harness.run_repeated_sensor_suite_validation_with_memory_stability(
                    memory_environ(),
                    importer=unittest.mock.Mock(name="importer"),
                    monotonic=clock,
                    monitor_factory=lambda memory_config: monitor,
                )
            )
        return aggregate, inner

    def run_delegate(self, environ, factory=None):
        clock = FakeClock(0.0)

        def one_cycle(*_args, **_kwargs):
            clock.value = 130.0
            return successful_cycle_report()

        with unittest.mock.patch.object(
            harness,
            "run_sensor_suite_validation_cycle",
            side_effect=one_cycle,
        ):
            if factory is None:
                return harness.run_repeated_sensor_suite_validation_cycles(
                    environ, monotonic=clock
                )
            return (
                harness.run_repeated_sensor_suite_validation_with_memory_stability(
                    environ, monotonic=clock, monitor_factory=factory
                )
            )

    def test_disabled_mode_delegates_identically(self):
        environ = memory_environ(gate=None)
        factory = unittest.mock.Mock(name="factory")
        direct = self.run_delegate(environ)
        wrapped = self.run_delegate(environ, factory=factory)
        factory.assert_not_called()
        self.assertIs(
            wrapped.memory.verdict,
            harness.MemoryStabilityVerdict.NOT_REQUESTED,
        )
        self.assertEqual(wrapped, direct)

    def test_disabled_gate_zero_also_delegates(self):
        aggregate = self.run_delegate(
            memory_environ(gate="0"),
            factory=unittest.mock.Mock(
                side_effect=AssertionError("factory must not run")
            ),
        )
        self.assertIs(
            aggregate.memory.verdict,
            harness.MemoryStabilityVerdict.NOT_REQUESTED,
        )
        self.assertIs(aggregate.success, True)

    def test_enabled_stable_run_attaches_report_and_keeps_success(self):
        monitor = FakeMonitor()
        aggregate, inner = self.successful_soak(monitor)
        self.assertEqual(inner.call_count, 1)
        self.assertIs(
            aggregate.termination_reason,
            harness.TerminationReason.SOAK_DURATION_REACHED,
        )
        self.assertIs(aggregate.success, True)
        self.assertIs(
            aggregate.memory.verdict, harness.MemoryStabilityVerdict.STABLE
        )
        self.assertTrue(
            harness._repeated_sensor_suite_validation_report_succeeded(
                aggregate
            )
        )
        self.assertEqual(
            monitor.journal,
            ["baseline", "start", "stop", "join", "final", "close", "report"],
        )

    def test_baseline_precedes_first_cycle_admission(self):
        journal = []
        monitor = FakeMonitor(journal=journal)

        def one_cycle(*_args, **_kwargs):
            journal.append("cycle")
            return successful_cycle_report()

        clock = FakeClock(0.0)

        def cycle_and_finish(*args, **kwargs):
            result = one_cycle(*args, **kwargs)
            clock.value = 130.0
            return result

        with unittest.mock.patch.object(
            harness,
            "run_sensor_suite_validation_cycle",
            side_effect=cycle_and_finish,
        ):
            harness.run_repeated_sensor_suite_validation_with_memory_stability(
                memory_environ(),
                monotonic=clock,
                monitor_factory=lambda memory_config: monitor,
            )
        self.assertEqual(journal[:3], ["baseline", "start", "cycle"])

    def test_runner_receives_the_monitor_admission_guard(self):
        monitor = FakeMonitor()
        with unittest.mock.patch.object(
            harness,
            "run_repeated_sensor_suite_validation_cycles",
            return_value=replace_success(),
        ) as runner:
            harness.run_repeated_sensor_suite_validation_with_memory_stability(
                memory_environ(),
                monotonic=FakeClock(0.0),
                monitor_factory=lambda memory_config: monitor,
            )
        self.assertEqual(
            runner.call_args.kwargs["admission_guard"],
            monitor.admission_block_reason,
        )

    def test_unstable_memory_overrides_soak_duration_termination(self):
        monitor = FakeMonitor(
            report=memory_stability_report(
                harness.MemoryStabilityVerdict.UNSTABLE
            ),
            override=harness.TerminationReason.MEMORY_STABILITY_FAILURE,
        )
        aggregate, _inner = self.successful_soak(monitor)
        self.assertIs(
            aggregate.termination_reason,
            harness.TerminationReason.MEMORY_STABILITY_FAILURE,
        )
        self.assertIs(aggregate.success, False)
        self.assertFalse(
            harness._repeated_sensor_suite_validation_report_succeeded(
                aggregate
            )
        )

    def test_indeterminate_memory_without_override_fails_softly(self):
        monitor = FakeMonitor(
            report=memory_stability_report(
                harness.MemoryStabilityVerdict.INDETERMINATE
            ),
            override=None,
        )
        aggregate, _inner = self.successful_soak(monitor)
        self.assertIs(
            aggregate.termination_reason,
            harness.TerminationReason.SOAK_DURATION_REACHED,
        )
        self.assertIs(aggregate.success, False)

    def test_cycle_failures_retain_precedence_over_memory_evidence(self):
        cases = (
            (
                successful_cycle_report(
                    primary_error=harness.ValidationFailure(
                        "readiness", "failed", 1
                    )
                ),
                harness.TerminationReason.FUNCTIONAL_FAILURE,
            ),
            (
                successful_cycle_report(cleanup_errors=("suite failed",)),
                harness.TerminationReason.CLEANUP_FAILURE,
            ),
            (
                successful_cycle_report(
                    final_disappearance=(
                        harness.CleanupVerification.FAILED_ACTORS_REMAIN
                    ),
                    remaining_owned_actor_ids=(901,),
                ),
                harness.TerminationReason.DISAPPEARANCE_FAILURE,
            ),
        )
        for failed_report, expected in cases:
            with self.subTest(expected=expected):
                monitor = FakeMonitor(
                    report=memory_stability_report(
                        harness.MemoryStabilityVerdict.UNSTABLE
                    ),
                    override=(
                        harness.TerminationReason.MEMORY_STABILITY_FAILURE
                    ),
                )
                aggregate, _inner = self.run_wrapper(
                    reports=(failed_report,),
                    monitor=monitor,
                    clock=FakeClock(0.0),
                )
                self.assertIs(aggregate.termination_reason, expected)
                self.assertIs(aggregate.success, False)
                self.assertIs(
                    aggregate.memory.verdict,
                    harness.MemoryStabilityVerdict.UNSTABLE,
                )

    def test_guard_termination_is_not_overridden_by_build_report(self):
        monitor = FakeMonitor(
            report=memory_stability_report(
                harness.MemoryStabilityVerdict.INDETERMINATE
            ),
            override=harness.TerminationReason.MEMORY_SAMPLING_FAILURE,
            block_reasons=[
                None,
                harness.TerminationReason.MEMORY_PROCESS_DISAPPEARED,
            ],
        )
        aggregate, inner = self.run_wrapper(
            reports=(successful_cycle_report(),),
            monitor=monitor,
            clock=FakeClock(0.0),
        )
        self.assertEqual(inner.call_count, 1)
        self.assertIs(
            aggregate.termination_reason,
            harness.TerminationReason.MEMORY_PROCESS_DISAPPEARED,
        )

    def test_factory_failure_aborts_before_any_cycle(self):
        def failing_factory(memory_config):
            raise sampler_module.ProcessResolutionError("no process")

        aggregate, inner = self.run_wrapper(
            monitor_factory=failing_factory, clock=FakeClock(5.0)
        )
        inner.assert_not_called()
        self.assertEqual(aggregate.attempted_cycle_count, 0)
        self.assertIs(
            aggregate.termination_reason,
            harness.TerminationReason.MEMORY_SAMPLING_FAILURE,
        )
        self.assertIs(aggregate.success, False)
        self.assertIs(
            aggregate.memory.verdict,
            harness.MemoryStabilityVerdict.INDETERMINATE,
        )
        self.assertIs(
            aggregate.memory.identity,
            harness.ProcessIdentityStatus.UNAVAILABLE,
        )
        self.assertTrue(
            any("no process" in text for text in aggregate.memory.diagnostics)
        )
        self.assertIsNone(aggregate.final_disappearance)

    def test_default_factory_process_rollback_failure_is_mandatory_evidence(self):
        backend = FakeNativeBackend(
            [(777, SHIPPING)],
            image_paths={777: "D:\\Other\\other.exe"},
            close_error=OSError("CloseHandle failed with Windows error 6"),
        )
        gpu_constructor = unittest.mock.Mock(name="gpu_constructor")
        fake_module = types.SimpleNamespace(
            WindowsProcessMemorySampler=types.SimpleNamespace(
                open=lambda pid=None: (
                    sampler_module.WindowsProcessMemorySampler.open(
                        pid=pid, backend=backend
                    )
                )
            ),
            GpuDedicatedMemorySampler=gpu_constructor,
            ProcessExitedError=sampler_module.ProcessExitedError,
            image_base_name=sampler_module.image_base_name,
        )
        factory = harness._default_memory_monitor_factory(
            lambda _name: fake_module
        )
        with unittest.mock.patch.object(
            harness,
            "_run_native_call_bounded",
            side_effect=synchronous_native_calls,
        ), unittest.mock.patch.object(
            harness.MemoryStabilityMonitor, "__init__", return_value=None
        ) as monitor_init:
            aggregate = (
                harness.run_repeated_sensor_suite_validation_with_memory_stability(
                    memory_environ(),
                    monotonic=FakeClock(0.0),
                    monitor_factory=factory,
                )
            )
        monitor_init.assert_not_called()
        gpu_constructor.assert_not_called()
        self.assertEqual(backend.open_calls, [777])
        self.assertEqual(backend.closed_handles, ["handle-777"])
        self.assertEqual(aggregate.attempted_cycle_count, 0)
        self.assertIs(
            aggregate.termination_reason,
            harness.TerminationReason.MEMORY_SAMPLING_FAILURE,
        )
        self.assertTrue(aggregate.memory.mandatory_sampling_failure)
        self.assertIs(
            aggregate.memory.verdict,
            harness.MemoryStabilityVerdict.INDETERMINATE,
        )
        self.assertTrue(
            any(
                "CloseHandle failed with Windows error 6" in text
                for text in aggregate.memory.diagnostics
            )
        )

    def test_baseline_failure_uses_monitor_classification(self):
        monitor = FakeMonitor(
            report=memory_stability_report(
                harness.MemoryStabilityVerdict.INDETERMINATE,
                identity=harness.ProcessIdentityStatus.EXITED,
            ),
            override=harness.TerminationReason.MEMORY_PROCESS_DISAPPEARED,
            baseline_error=RuntimeError("memory baseline sampling failed"),
        )
        aggregate, inner = self.run_wrapper(
            monitor=monitor, clock=FakeClock(5.0)
        )
        inner.assert_not_called()
        self.assertIs(
            aggregate.termination_reason,
            harness.TerminationReason.MEMORY_PROCESS_DISAPPEARED,
        )
        self.assertIn("close", monitor.journal)

    def test_escaping_exceptions_preserve_identity_and_join_monitor(self):
        class AbortSignal(BaseException):
            pass

        for exception in (
            unittest.SkipTest("skip"),
            RuntimeError("disconnect"),
            KeyboardInterrupt("interrupt"),
            SystemExit(23),
            AbortSignal("abort"),
        ):
            with self.subTest(exception=type(exception).__name__):
                monitor = FakeMonitor()
                with unittest.mock.patch.object(
                    harness,
                    "run_sensor_suite_validation_cycle",
                    side_effect=exception,
                ):
                    with self.assertRaises(type(exception)) as caught:
                        harness.run_repeated_sensor_suite_validation_with_memory_stability(
                            memory_environ(),
                            monotonic=FakeClock(0.0),
                            monitor_factory=lambda memory_config: monitor,
                        )
                self.assertIs(caught.exception, exception)
                self.assertIn("stop", monitor.journal)
                self.assertIn("join", monitor.journal)
                self.assertIn("close", monitor.journal)
                self.assertIn("final", monitor.journal)
                self.assertLess(
                    monitor.journal.index("join"),
                    monitor.journal.index("final"),
                )
                self.assertLess(
                    monitor.journal.index("final"),
                    monitor.journal.index("close"),
                )

    def test_shutdown_failure_never_masks_the_escaping_exception(self):
        interrupt = KeyboardInterrupt("operator interrupt")
        monitor = FakeMonitor(close_error=RuntimeError("close failed"))
        with unittest.mock.patch.object(
            harness,
            "run_sensor_suite_validation_cycle",
            side_effect=interrupt,
        ):
            with self.assertRaises(KeyboardInterrupt) as caught:
                harness.run_repeated_sensor_suite_validation_with_memory_stability(
                    memory_environ(),
                    monotonic=FakeClock(0.0),
                    monitor_factory=lambda memory_config: monitor,
                )
        self.assertIs(caught.exception, interrupt)

    def test_default_factory_imports_the_sampler_lazily(self):
        fake_sampler = unittest.mock.Mock(name="process_sampler")
        fake_sampler.identity = sampler_module.ProcessIdentity(
            4242, f"D:\\Sim\\{SHIPPING}", 5
        )
        fake_sampler.sample.return_value = (
            sampler_module.ProcessMemoryCounters(1000, 2000, 2000)
        )
        fake_module = types.SimpleNamespace(
            WindowsProcessMemorySampler=types.SimpleNamespace(
                open=lambda pid=None: fake_sampler
            ),
            GpuDedicatedMemorySampler=unittest.mock.Mock(
                side_effect=OSError("no pdh")
            ),
            ProcessExitedError=sampler_module.ProcessExitedError,
            image_base_name=sampler_module.image_base_name,
        )
        importer = unittest.mock.Mock(return_value=fake_module)
        factory = harness._default_memory_monitor_factory(importer)
        monitor = factory(parse_memory(memory_environ()))
        importer.assert_called_once_with("tests.windows_memory_sampler")
        self.assertIsInstance(monitor, harness.MemoryStabilityMonitor)
        monitor.take_baseline()
        report, _override = monitor.build_report()
        self.assertEqual(report.pid, 4242)
        self.assertEqual(report.process_creation_time_100ns, 5)
        self.assertEqual(report.executable_path, f"D:\\Sim\\{SHIPPING}")
        self.assertEqual(report.executable_basename, SHIPPING)
        self.assertEqual(report.total_sample_count, 1)
        self.assertIs(
            report.gpu_status,
            harness.MemoryCounterAvailability.UNSUPPORTED,
        )
        self.assertTrue(
            any("no pdh" in text for text in report.diagnostics)
        )

    def test_gpu_startup_rollback_close_diagnostic_is_advisory_and_stable(self):
        primary = RuntimeError("counter add failed")
        close_failure = sampler_module.PdhQueryError("rollback close failed")
        backend = FakePdhBackend(
            add_error=primary, close_error=close_failure
        )
        native_gpu = sampler_module.GpuDedicatedMemorySampler(
            4242, backend=backend
        )
        process_sampler = unittest.mock.Mock(name="process_sampler")
        process_sampler.identity = sampler_module.ProcessIdentity(
            4242, f"D:\\Sim\\{SHIPPING}", 5
        )
        process_sampler.sample.return_value = (
            sampler_module.ProcessMemoryCounters(1000, 2000, 2000)
        )
        fake_module = types.SimpleNamespace(
            WindowsProcessMemorySampler=types.SimpleNamespace(
                open=lambda pid=None: process_sampler
            ),
            GpuDedicatedMemorySampler=lambda pid: native_gpu,
            ProcessExitedError=sampler_module.ProcessExitedError,
            image_base_name=sampler_module.image_base_name,
        )
        with unittest.mock.patch.object(
            harness,
            "_run_native_call_bounded",
            side_effect=synchronous_native_calls,
        ):
            monitor = harness._default_memory_monitor_factory(
                lambda _name: fake_module
            )(parse_memory(memory_environ()))
        clock = FakeClock(100.0)
        monitor._monotonic = clock
        monitor._stop_event = AutoAdvanceEvent(clock, 12)
        monitor.take_baseline()
        monitor._run_sampling_loop()
        monitor.close()
        report, override = monitor.build_report()
        self.assertIsNone(override)
        self.assertIs(report.verdict, harness.MemoryStabilityVerdict.STABLE)
        self.assertIs(
            report.gpu_status,
            harness.MemoryCounterAvailability.UNSUPPORTED,
        )
        diagnostics = "\n".join(report.diagnostics)
        self.assertIn("counter add failed", diagnostics)
        self.assertIn("rollback close failed", diagnostics)
        self.assertTrue(
            all(
                len(text) <= harness.MAX_CLEANUP_ERROR_LENGTH
                for text in report.diagnostics
            )
        )
        self.assertEqual(backend.open_calls, 1)
        self.assertEqual(backend.closed_queries, ["query"])
        process_sampler.close.assert_called_once_with()

    def test_gpu_startup_diagnostics_obey_retention_and_omission_counts(self):
        startup_failure = OSError("gpu unavailable")
        for index in range(10):
            startup_failure.add_note(f"secondary startup evidence {index}")
        backend = FakePdhBackend(open_error=startup_failure)
        native_gpu = sampler_module.GpuDedicatedMemorySampler(
            4242, backend=backend
        )
        process_sampler = unittest.mock.Mock(name="process_sampler")
        process_sampler.identity = sampler_module.ProcessIdentity(
            4242, f"D:\\Sim\\{SHIPPING}", 5
        )
        fake_module = types.SimpleNamespace(
            WindowsProcessMemorySampler=types.SimpleNamespace(
                open=lambda pid=None: process_sampler
            ),
            GpuDedicatedMemorySampler=lambda pid: native_gpu,
            ProcessExitedError=sampler_module.ProcessExitedError,
            image_base_name=sampler_module.image_base_name,
        )
        with unittest.mock.patch.object(
            harness,
            "_run_native_call_bounded",
            side_effect=synchronous_native_calls,
        ):
            monitor = harness._default_memory_monitor_factory(
                lambda _name: fake_module
            )(parse_memory(memory_environ()))
        report, _override = monitor.build_report()
        self.assertEqual(
            len(report.diagnostics),
            harness.MAX_RETAINED_MEMORY_DIAGNOSTICS,
        )
        self.assertEqual(report.omitted_diagnostic_count, 3)
        self.assertEqual(backend.open_calls, 1)
        monitor.close()

    def test_factory_gpu_open_base_exception_still_closes_process_sampler(self):
        interrupt = KeyboardInterrupt("gpu add interrupted")
        backend = FakePdhBackend(add_error=interrupt)
        gpu = sampler_module.GpuDedicatedMemorySampler(4242, backend=backend)
        process_sampler = unittest.mock.Mock(name="process_sampler")
        process_sampler.identity = sampler_module.ProcessIdentity(
            4242, f"D:\\Sim\\{SHIPPING}", 5
        )
        fake_module = types.SimpleNamespace(
            WindowsProcessMemorySampler=types.SimpleNamespace(
                open=lambda pid=None: process_sampler
            ),
            GpuDedicatedMemorySampler=lambda pid: gpu,
            ProcessExitedError=sampler_module.ProcessExitedError,
            image_base_name=sampler_module.image_base_name,
        )
        with unittest.mock.patch.object(
            harness,
            "_run_native_call_bounded",
            side_effect=synchronous_native_calls,
        ):
            factory = harness._default_memory_monitor_factory(
                lambda _name: fake_module
            )
            with self.assertRaises(KeyboardInterrupt) as caught:
                factory(parse_memory(memory_environ()))
        self.assertIs(caught.exception, interrupt)
        self.assertEqual(backend.closed_queries, ["query"])
        process_sampler.close.assert_called_once_with()

    def test_factory_gpu_base_exception_never_skips_process_rollback(self):
        original = RuntimeError("monitor construction failed")
        gpu_interrupt = KeyboardInterrupt("gpu close interrupted")
        process_sampler = unittest.mock.Mock(name="process_sampler")
        process_sampler.identity = sampler_module.ProcessIdentity(
            4242, f"D:\\Sim\\{SHIPPING}", 5
        )

        class InterruptingGpu:
            def __init__(self, pid):
                self.pid = pid
                self.close_calls = 0

            def open(self):
                return sampler_module.GpuAvailability.AVAILABLE

            def close(self):
                self.close_calls += 1
                raise gpu_interrupt

        gpu_instances = []

        def make_gpu(pid):
            instance = InterruptingGpu(pid)
            gpu_instances.append(instance)
            return instance

        fake_module = types.SimpleNamespace(
            WindowsProcessMemorySampler=types.SimpleNamespace(
                open=lambda pid=None: process_sampler
            ),
            GpuDedicatedMemorySampler=make_gpu,
            ProcessExitedError=sampler_module.ProcessExitedError,
            image_base_name=sampler_module.image_base_name,
        )
        with unittest.mock.patch.object(
            harness, "_run_native_call_bounded", side_effect=synchronous_native_calls
        ), unittest.mock.patch.object(
            harness.MemoryStabilityMonitor, "__init__", side_effect=original
        ):
            factory = harness._default_memory_monitor_factory(
                lambda _name: fake_module
            )
            with self.assertRaises(RuntimeError) as caught:
                factory(parse_memory(memory_environ()))
        self.assertIs(caught.exception, original)
        self.assertEqual(gpu_instances[0].close_calls, 1)
        process_sampler.close.assert_called_once_with()
        notes = getattr(original, "__notes__", ())
        self.assertTrue(any("KeyboardInterrupt" in note for note in notes))

    def test_process_adapter_translates_exit_exceptions(self):
        module_error = sampler_module.ProcessExitedError("gone")
        fake_sampler = types.SimpleNamespace(
            sample=unittest.mock.Mock(side_effect=module_error),
            close=lambda: None,
        )
        adapter = harness._WindowsProcessSamplerAdapter(
            sampler_module, fake_sampler
        )
        with self.assertRaises(harness.MemoryMonitoredProcessExited):
            adapter.sample()

    def test_pre_cycle_failure_aggregate_shape(self):
        config = harness.parse_integration_config(memory_environ())
        report = memory_stability_report(
            harness.MemoryStabilityVerdict.INDETERMINATE
        )
        aggregate = harness._memory_failure_aggregate(
            config,
            3.25,
            harness.TerminationReason.MEMORY_SAMPLING_FAILURE,
            report,
        )
        self.assertIs(
            aggregate.execution_mode, harness.ExecutionMode.TIMED_SOAK
        )
        self.assertIsNone(aggregate.requested_cycle_count)
        self.assertEqual(aggregate.requested_soak_seconds, 120.0)
        self.assertEqual(aggregate.elapsed_seconds, 3.25)
        self.assertEqual(aggregate.cycles, ())
        self.assertIs(aggregate.memory, report)


def replace_success():
    """A successful timed-soak aggregate produced by the real outer runner."""
    clock = FakeClock(0.0)

    def one_cycle(*_args, **_kwargs):
        clock.value = 130.0
        return successful_cycle_report()

    with unittest.mock.patch.object(
        harness, "run_sensor_suite_validation_cycle", side_effect=one_cycle
    ):
        return harness.run_repeated_sensor_suite_validation_cycles(
            memory_environ(gate=None), monotonic=clock
        )


# ======================================================================
# Report and source contract
# ======================================================================


class ReportContractTest(unittest.TestCase):
    def test_not_requested_report_is_the_default_aggregate_field(self):
        aggregate = replace_success()
        self.assertIs(aggregate.memory, harness.NOT_REQUESTED_MEMORY_REPORT)
        self.assertIs(
            aggregate.memory.verdict,
            harness.MemoryStabilityVerdict.NOT_REQUESTED,
        )
        self.assertIs(
            aggregate.memory.identity,
            harness.ProcessIdentityStatus.NOT_TRACKED,
        )
        self.assertIs(
            aggregate.memory.gpu_status,
            harness.MemoryCounterAvailability.NOT_SAMPLED,
        )

    def test_boundary_check_accepts_not_requested_and_stable_only(self):
        aggregate = replace_success()
        self.assertTrue(
            harness._repeated_sensor_suite_validation_report_succeeded(
                aggregate
            )
        )
        stable = dataclasses.replace(
            aggregate, memory=memory_stability_report()
        )
        self.assertTrue(
            harness._repeated_sensor_suite_validation_report_succeeded(stable)
        )
        for verdict in (
            harness.MemoryStabilityVerdict.UNSTABLE,
            harness.MemoryStabilityVerdict.INDETERMINATE,
        ):
            with self.subTest(verdict=verdict):
                failed = dataclasses.replace(
                    aggregate, memory=memory_stability_report(verdict)
                )
                self.assertFalse(
                    harness._repeated_sensor_suite_validation_report_succeeded(
                        failed
                    )
                )

    def test_memory_records_are_frozen_and_slotted(self):
        values = (
            harness.ProcessMemorySample(0.0, 1, 2, None),
            harness.ProcessCounterReading(1, 2, 2),
            harness.GpuEvidenceReading(
                harness.MemoryCounterAvailability.AVAILABLE, 1
            ),
            limits(),
            harness.MemoryStabilityConfig(enabled=False),
            memory_stability_report(),
            sampler_module.ProcessIdentity(1, "x", 2),
            sampler_module.ProcessMemoryCounters(1, 2, 2),
            sampler_module.GpuDedicatedSample(
                sampler_module.GpuAvailability.AVAILABLE, 1, 1
            ),
        )
        for value in values:
            with self.subTest(value=type(value).__name__):
                self.assertFalse(hasattr(value, "__dict__"))
                field = dataclasses.fields(value)[0].name
                with self.assertRaises(FrozenInstanceError):
                    setattr(value, field, None)

    def test_monitored_aggregate_walk_is_bounded_and_carla_free(self):
        monitor, _clock, _event, _sampler = run_monitor(false_waits=12)
        memory_report, _override = monitor.build_report()
        aggregate = dataclasses.replace(
            replace_success(), memory=memory_report
        )

        seen = set()

        def walk(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            self.assertNotIsInstance(value, BaseException)
            self.assertNotIsInstance(value, types.TracebackType)
            self.assertNotIsInstance(value, (bytes, bytearray, memoryview))
            self.assertNotIsInstance(value, (list, dict, set))
            self.assertFalse(callable(value), f"callable {value!r} retained")
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                for field in dataclasses.fields(value):
                    walk(getattr(value, field.name))
            elif isinstance(value, tuple):
                for item in value:
                    walk(item)

        walk(aggregate)
        self.assertLessEqual(
            len(aggregate.memory.samples), harness.MAX_RETAINED_MEMORY_SAMPLES
        )
        self.assertLessEqual(
            len(aggregate.memory.diagnostics),
            harness.MAX_RETAINED_MEMORY_DIAGNOSTICS,
        )

    def test_single_cycle_runner_remains_memory_unaware(self):
        source = inspect.getsource(harness.run_sensor_suite_validation_cycle)
        for forbidden in (
            "admission_guard",
            "MemoryStability",
            "ProcessMemorySample",
            "windows_memory_sampler",
            "MemoryMonitoredProcessExited",
            "_take_sample",
        ):
            self.assertNotIn(forbidden, source)

    def test_outer_runner_gains_only_the_neutral_admission_guard(self):
        source = inspect.getsource(
            harness.run_repeated_sensor_suite_validation_cycles
        )
        self.assertIn("admission_guard", source)
        for forbidden in (
            "MemoryStabilityMonitor",
            "ProcessMemorySample",
            "windows_memory_sampler",
            "take_baseline",
            "build_report",
        ):
            self.assertNotIn(forbidden, source)

    def test_monitor_exists_only_around_the_timed_soak_wrapper(self):
        wrapper_source = inspect.getsource(
            harness.run_repeated_sensor_suite_validation_with_memory_stability
        )
        self.assertIn("take_baseline", wrapper_source)
        self.assertIn("admission_guard=monitor.admission_block_reason",
                      wrapper_source)
        self.assertIn("finally", wrapper_source)

    def test_exactly_one_live_unittest_entry_remains(self):
        live_methods = []
        for value in vars(harness).values():
            if isinstance(value, type) and issubclass(value, unittest.TestCase):
                live_methods.extend(
                    name for name in vars(value) if name.startswith("test_")
                )
        self.assertEqual(
            live_methods, ["test_connection_and_single_owned_ego_cycle"]
        )

    def test_sampler_module_has_no_prohibited_dependencies(self):
        tree = ast.parse(SAMPLER_PATH.read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                roots.add(
                    "<relative-import>"
                    if node.level
                    else (node.module or "").split(".")[0]
                )
        self.assertTrue(roots)
        for root in roots:
            self.assertIn(root, sys.stdlib_module_names)
        source = SAMPLER_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "psutil",
            "import wmi",
            "subprocess",
            "PowerShell",
            "powershell",
            "nvidia-smi",
            "pynvml",
            "pip install",
            "D3DKMT",
            "time.sleep",
            "socket.",
            "CARLA_Latest",
            "CARLA_0.9.16_STABLE",
        ):
            self.assertNotIn(forbidden, source)

    def test_sampler_module_import_binds_no_native_libraries(self):
        with unittest.mock.patch.object(
            ctypes, "WinDLL", side_effect=AssertionError("bound at import")
        ):
            import importlib

            importlib.reload(sampler_module)
        # Restore the module for later tests in this process.
        import importlib

        importlib.reload(sampler_module)

    def test_no_hardcoded_installation_paths_in_harness_memory_logic(self):
        harness_source = Path(harness.__file__).read_text(encoding="utf-8")
        self.assertNotIn("CARLA_Latest", harness_source)
        self.assertNotIn("CARLA_0.9.16_STABLE", harness_source)


# ======================================================================
# STABLE report invariants at the live acceptance boundary
# ======================================================================


class StableReportInvariantTest(unittest.TestCase):
    def assert_rejected(self, memory_report):
        self.assertFalse(
            harness._stable_memory_report_coherent(memory_report)
        )
        aggregate = dataclasses.replace(
            replace_success(), memory=memory_report
        )
        self.assertFalse(
            harness._repeated_sensor_suite_validation_report_succeeded(
                aggregate
            )
        )

    def test_realistic_stable_report_is_coherent_and_accepted(self):
        report = memory_stability_report()
        self.assertTrue(harness._stable_memory_report_coherent(report))
        aggregate = dataclasses.replace(replace_success(), memory=report)
        self.assertTrue(
            harness._repeated_sensor_suite_validation_report_succeeded(
                aggregate
            )
        )

    def test_a_real_monitor_stable_report_passes_the_validator(self):
        monitor, _clock, _event, _sampler = run_monitor(false_waits=12)
        monitor.close()
        report, _override = monitor.build_report()
        self.assertIs(report.verdict, harness.MemoryStabilityVerdict.STABLE)
        self.assertTrue(harness._stable_memory_report_coherent(report))

    def test_stored_percentage_rounding_never_changes_exact_decision(self):
        base = memory_stability_report()
        cases = (
            (
                "working_set",
                2.4,
                math.nextafter(2.4, math.inf),
            ),
            (
                "private_commit",
                1.2,
                math.nextafter(1.2, math.inf),
            ),
        )
        for counter, displayed, passing_limit in cases:
            with self.subTest(counter=counter, decision="pass"):
                thresholds = dataclasses.replace(
                    base.thresholds,
                    **{f"{counter}_max_final_growth_percent": passing_limit},
                )
                stored = dataclasses.replace(
                    getattr(base, counter),
                    growth_percent=displayed + 1e-13,
                )
                report = dataclasses.replace(
                    base, thresholds=thresholds, **{counter: stored}
                )
                self.assertTrue(
                    harness._stable_memory_report_coherent(report)
                )

            with self.subTest(counter=counter, decision="fail"):
                failing_limit = math.nextafter(displayed, -math.inf)
                thresholds = dataclasses.replace(
                    base.thresholds,
                    **{f"{counter}_max_final_growth_percent": failing_limit},
                )
                stored = dataclasses.replace(
                    getattr(base, counter), growth_percent=failing_limit
                )
                self.assert_rejected(
                    dataclasses.replace(
                        base, thresholds=thresholds, **{counter: stored}
                    )
                )

    def test_known_2_4_threshold_probe_rejects_both_metrics(self):
        base = memory_stability_report()
        for counter, limit in (
            ("working_set", 2.3999999999999995),
            ("private_commit", 1.1999999999999997),
        ):
            with self.subTest(counter=counter):
                thresholds = dataclasses.replace(
                    base.thresholds,
                    **{f"{counter}_max_final_growth_percent": limit},
                )
                self.assert_rejected(
                    dataclasses.replace(base, thresholds=thresholds)
                )

    def test_monitor_never_emits_stable_with_invalid_identity_evidence(self):
        cases = (
            dict(process_creation_time_100ns=None),
            dict(executable_basename="CarlaUE4.exe"),
            dict(executable_path="D:\\Sim\\CarlaUE4.exe"),
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                monitor, _clock, _event, _sampler = run_monitor(
                    false_waits=12
                )
                for name, value in overrides.items():
                    setattr(monitor, f"_{name}", value)
                monitor.close()
                report, _override = monitor.build_report()
                self.assertIs(
                    report.verdict,
                    harness.MemoryStabilityVerdict.INDETERMINATE,
                )

    def test_non_stable_verdicts_are_never_coherent(self):
        for verdict in (
            harness.MemoryStabilityVerdict.NOT_REQUESTED,
            harness.MemoryStabilityVerdict.UNSTABLE,
            harness.MemoryStabilityVerdict.INDETERMINATE,
        ):
            with self.subTest(verdict=verdict):
                self.assertFalse(
                    harness._stable_memory_report_coherent(
                        memory_stability_report(verdict)
                    )
                )

    def test_each_identity_and_failure_invariant_is_enforced(self):
        cases = {
            "monitoring_not_enabled": dict(monitoring_enabled=False),
            "identity_exited": dict(
                identity=harness.ProcessIdentityStatus.EXITED
            ),
            "identity_unavailable": dict(
                identity=harness.ProcessIdentityStatus.UNAVAILABLE
            ),
            "identity_not_tracked": dict(
                identity=harness.ProcessIdentityStatus.NOT_TRACKED
            ),
            "sampling_failure_present": dict(
                mandatory_sampling_failure=True
            ),
            "shutdown_failure_present": dict(
                mandatory_shutdown_failure=True
            ),
            "missing_pid": dict(pid=None),
            "zero_pid": dict(pid=0),
            "boolean_pid": dict(pid=True),
            "oversized_pid": dict(pid=harness.MAX_MEMORY_PID + 1),
            "missing_creation_time": dict(process_creation_time_100ns=None),
            "boolean_creation_time": dict(process_creation_time_100ns=True),
            "negative_creation_time": dict(process_creation_time_100ns=-1),
            "oversized_creation_time": dict(
                process_creation_time_100ns=(
                    harness.MAX_PROCESS_CREATION_TIME_100NS + 1
                )
            ),
            "missing_executable": dict(executable_path=None),
            "empty_executable": dict(executable_path=""),
            "wrong_executable": dict(
                executable_path="D:\\Sim\\CarlaUE4.exe",
                executable_basename="CarlaUE4.exe",
            ),
            "wrong_executable_case": dict(
                executable_path="D:\\Sim\\carlaue4-win64-shipping.exe",
                executable_basename="carlaue4-win64-shipping.exe",
            ),
            "path_basename_disagreement": dict(
                executable_basename="Other.exe"
            ),
            "embedded_nul_path": dict(
                executable_path=(
                    "D:\\Sim\\bad\0path\\CarlaUE4-Win64-Shipping.exe"
                )
            ),
            "trailing_nul_path": dict(
                executable_path=(
                    "D:\\Sim\\CarlaUE4-Win64-Shipping.exe\0"
                )
            ),
            "control_character_path": dict(
                executable_path=(
                    "D:\\Sim\\bad\x1fpath\\CarlaUE4-Win64-Shipping.exe"
                )
            ),
            "invalid_component_path": dict(
                executable_path=(
                    "D:\\Sim\\bad?path\\CarlaUE4-Win64-Shipping.exe"
                )
            ),
            "relative_path": dict(
                executable_path="Sim\\CarlaUE4-Win64-Shipping.exe"
            ),
            "overlength_path": dict(
                executable_path=(
                    "D:\\"
                    + "a" * harness.MAX_MEMORY_EXECUTABLE_PATH_CHARACTERS
                    + "\\CarlaUE4-Win64-Shipping.exe"
                )
            ),
            "missing_thresholds": dict(thresholds=None),
        }
        for name, overrides in cases.items():
            with self.subTest(case=name):
                self.assert_rejected(memory_stability_report(**overrides))

    def test_threshold_field_types_and_ranges_are_recomputed(self):
        base = limits()
        cases = {
            "wrong_threshold_object": "not thresholds",
            "boolean_byte": dataclasses.replace(
                base, working_set_max_peak_growth_bytes=True
            ),
            "negative_byte": dataclasses.replace(
                base, working_set_max_final_growth_bytes=-1
            ),
            "oversized_byte": dataclasses.replace(
                base,
                private_commit_max_peak_growth_bytes=(
                    harness.MAX_MEMORY_BYTE_LIMIT + 1
                ),
            ),
            "integer_percent": dataclasses.replace(
                base, working_set_max_final_growth_percent=50
            ),
            "negative_percent": dataclasses.replace(
                base, private_commit_max_final_growth_percent=-0.1
            ),
            "nonfinite_percent": dataclasses.replace(
                base, working_set_max_final_growth_percent=math.inf
            ),
            "oversized_percent": dataclasses.replace(
                base,
                private_commit_max_final_growth_percent=(
                    harness.MAX_MEMORY_PERCENT_LIMIT + 0.1
                ),
            ),
            "integer_slope": dataclasses.replace(
                base, working_set_max_slope_bytes_per_minute=5
            ),
            "negative_slope": dataclasses.replace(
                base, private_commit_max_slope_bytes_per_minute=-1.0
            ),
            "nonfinite_slope": dataclasses.replace(
                base, working_set_max_slope_bytes_per_minute=math.nan
            ),
        }
        for name, thresholds in cases.items():
            with self.subTest(case=name):
                self.assert_rejected(
                    memory_stability_report(thresholds=thresholds)
                )

    def test_threshold_breaches_cannot_be_overridden_by_stored_pass(self):
        cases = {
            "peak_bytes": limits(
                working_set_max_peak_growth_bytes=23
            ),
            "final_bytes": limits(
                working_set_max_final_growth_bytes=23
            ),
            "final_percent": limits(
                working_set_max_final_growth_percent=2.39
            ),
            "slope": limits(
                working_set_max_slope_bytes_per_minute=11.99
            ),
        }
        for name, thresholds in cases.items():
            with self.subTest(case=name):
                report = memory_stability_report(thresholds=thresholds)
                self.assertTrue(report.working_set.passed)
                self.assertEqual(report.working_set.failed_thresholds, ())
                self.assert_rejected(report)

    def test_threshold_equality_recomputed_as_passing(self):
        samples = tuple(
            harness.ProcessMemorySample(
                5.0 * index, 960 + index, 1920 + index
            )
            for index in range(25)
        )
        report = memory_stability_report(
            samples=samples,
            working_set=stable_metric("working_set", start=960),
            private_commit=stable_metric("private_commit", start=1920),
            thresholds=limits(
                working_set_max_peak_growth_bytes=24,
                working_set_max_final_growth_bytes=24,
                working_set_max_final_growth_percent=2.5,
                working_set_max_slope_bytes_per_minute=12.0,
                private_commit_max_peak_growth_bytes=24,
                private_commit_max_final_growth_bytes=24,
                private_commit_max_final_growth_percent=1.25,
                private_commit_max_slope_bytes_per_minute=12.0,
            )
        )
        self.assertTrue(harness._stable_memory_report_coherent(report))

    def test_metric_invariants_are_enforced(self):
        cases = {
            "missing_working_set": dict(working_set=None),
            "missing_private_commit": dict(private_commit=None),
            "failed_thresholds": dict(
                working_set=stable_metric(
                    "working_set",
                    start=1000,
                    failed_thresholds=("max_final_growth_bytes",),
                )
            ),
            "unevaluated_thresholds": dict(
                working_set=stable_metric(
                    "working_set",
                    start=1000,
                    unevaluated_thresholds=("max_final_growth_percent",),
                )
            ),
            "not_passed": dict(
                working_set=stable_metric(
                    "working_set", start=1000, passed=False
                )
            ),
            "wrong_counter_name": dict(
                working_set=stable_metric("private_commit", start=1000)
            ),
            "too_few_samples": dict(
                working_set=stable_metric(
                    "working_set", start=1000, sample_count=4
                )
            ),
            "short_span": dict(
                working_set=stable_metric(
                    "working_set", start=1000, span_seconds=59.9
                )
            ),
            "incoherent_growth": dict(
                working_set=stable_metric(
                    "working_set", start=1000, growth_bytes=999_999
                )
            ),
            "incoherent_peak_growth": dict(
                working_set=stable_metric(
                    "working_set", start=1000, peak_growth_bytes=-7
                )
            ),
            "incoherent_growth_percent": dict(
                working_set=stable_metric(
                    "working_set", start=1000, growth_percent=99.0
                )
            ),
            "incoherent_peak_percent": dict(
                working_set=stable_metric(
                    "working_set", start=1000, peak_growth_percent=99.0
                )
            ),
            "wrong_observed_peak": dict(
                working_set=stable_metric(
                    "working_set",
                    start=1000,
                    peak_bytes=1023,
                    peak_growth_bytes=23,
                )
            ),
            "wrong_peak_timestamp": dict(
                working_set=stable_metric(
                    "working_set", start=1000, peak_elapsed_seconds=115.0
                )
            ),
            "incoherent_slope": dict(
                working_set=stable_metric(
                    "working_set", start=1000, slope_bytes_per_minute=11.0
                )
            ),
            "missing_percentages": dict(
                working_set=stable_metric(
                    "working_set",
                    start=1000,
                    growth_percent=None,
                    peak_growth_percent=None,
                )
            ),
            "sample_count_mismatch": dict(
                working_set=stable_metric(
                    "working_set", start=1000, sample_count=26
                )
            ),
        }
        for name, overrides in cases.items():
            with self.subTest(case=name):
                self.assert_rejected(memory_stability_report(**overrides))

    def test_configured_slope_threshold_requires_an_evaluated_slope(self):
        thresholds = limits(working_set_max_slope_bytes_per_minute=10.0)
        self.assert_rejected(memory_stability_report(thresholds=thresholds))
        evaluated = memory_stability_report(
            thresholds=limits(working_set_max_slope_bytes_per_minute=12.0),
            working_set=stable_metric(
                "working_set", start=1000, slope_bytes_per_minute=12.0
            ),
        )
        self.assertTrue(harness._stable_memory_report_coherent(evaluated))

    def test_zero_baseline_percentage_cannot_form_stable_evidence(self):
        samples = tuple(
            harness.ProcessMemorySample(
                5.0 * index, index, 2000 + index
            )
            for index in range(25)
        )
        metric = stable_metric(
            "working_set",
            start=1,
            start_bytes=0,
            final_bytes=24,
            peak_bytes=24,
            growth_bytes=24,
            peak_growth_bytes=24,
            growth_percent=None,
            peak_growth_percent=None,
        )
        self.assert_rejected(
            memory_stability_report(samples=samples, working_set=metric)
        )

    def test_retention_count_invariants_are_enforced(self):
        cases = {
            "retained_mismatch": dict(retained_sample_count=24),
            "omitted_mismatch": dict(omitted_sample_count=1),
            "negative_missed": dict(missed_interval_count=-1),
            "empty_samples": dict(samples=()),
            "boolean_total": dict(total_sample_count=True),
            "omitted_without_overflow": dict(
                total_sample_count=30,
                omitted_sample_count=5,
            ),
        }
        for name, overrides in cases.items():
            with self.subTest(case=name):
                self.assert_rejected(memory_stability_report(**overrides))

    def test_sample_shape_invariants_are_enforced(self):
        base = memory_stability_report()
        decreasing = base.samples[:24] + (
            harness.ProcessMemorySample(0.5, 1024, 2024),
        )
        duplicate = base.samples[:24] + (
            harness.ProcessMemorySample(115.0, 1024, 2024),
        )
        three_equal = base.samples[:21] + tuple(
            harness.ProcessMemorySample(105.0, 1021 + index, 2021 + index)
            for index in range(3)
        ) + base.samples[24:]
        nonfinite = {
            name: base.samples[:24]
            + (harness.ProcessMemorySample(value, 1024, 2024),)
            for name, value in (
                ("nan_elapsed", math.nan),
                ("positive_infinite_elapsed", math.inf),
                ("negative_infinite_elapsed", -math.inf),
                ("boolean_elapsed", True),
                ("integer_elapsed", 120),
            )
        }
        zero_bytes = base.samples[:24] + (
            harness.ProcessMemorySample(120.0, 0, 2024),
        )
        wrong_type = base.samples[:24] + (("elapsed", 1, 2),)
        negative_gpu = base.samples[:24] + (
            harness.ProcessMemorySample(120.0, 1024, 2024, -1),
        )
        cases = (
            ("non_monotonic_elapsed", decreasing),
            ("duplicate_elapsed", duplicate),
            ("three_equal_elapsed", three_equal),
            ("non_positive_counter", zero_bytes),
            ("wrong_sample_type", wrong_type),
            ("negative_gpu_bytes", negative_gpu),
        ) + tuple(nonfinite.items())
        for name, samples in cases:
            with self.subTest(case=name):
                self.assert_rejected(
                    memory_stability_report(samples=samples)
                )

    def test_gpu_evidence_stays_advisory_but_well_formed(self):
        for status in harness.MemoryCounterAvailability:
            with self.subTest(status=status):
                self.assertTrue(
                    harness._stable_memory_report_coherent(
                        memory_stability_report(gpu_status=status)
                    )
                )
        cases = {
            "final_without_peak": dict(gpu_final_dedicated_bytes=10),
            "peak_below_final": dict(
                gpu_final_dedicated_bytes=10, gpu_peak_dedicated_bytes=5
            ),
            "negative_peak": dict(gpu_peak_dedicated_bytes=-1),
            "non_enum_status": dict(gpu_status="AVAILABLE"),
        }
        for name, overrides in cases.items():
            with self.subTest(case=name):
                self.assert_rejected(memory_stability_report(**overrides))

    def test_diagnostic_invariants_are_enforced(self):
        cases = {
            "too_many_diagnostics": dict(
                diagnostics=tuple(
                    f"diagnostic {index}"
                    for index in range(
                        harness.MAX_RETAINED_MEMORY_DIAGNOSTICS + 1
                    )
                )
            ),
            "non_text_diagnostic": dict(diagnostics=(42,)),
            "oversized_diagnostic": dict(
                diagnostics=(
                    "x" * (harness.MAX_CLEANUP_ERROR_LENGTH + 1),
                )
            ),
            "empty_with_omitted": dict(
                diagnostics=(), omitted_diagnostic_count=1
            ),
            "seven_with_omitted_before_cap": dict(
                diagnostics=tuple(f"diagnostic {index}" for index in range(7)),
                omitted_diagnostic_count=1,
            ),
            "negative_omitted": dict(omitted_diagnostic_count=-1),
            "boolean_omitted": dict(omitted_diagnostic_count=True),
            "empty_diagnostic": dict(diagnostics=("",)),
            "control_diagnostic": dict(diagnostics=("bad\0diagnostic",)),
        }
        for name, overrides in cases.items():
            with self.subTest(case=name):
                self.assert_rejected(memory_stability_report(**overrides))

    def test_diagnostic_retention_at_capacity_is_coherent(self):
        diagnostics = tuple(
            f"diagnostic {index}"
            for index in range(harness.MAX_RETAINED_MEMORY_DIAGNOSTICS)
        )
        for omitted in (0, 1, 37):
            with self.subTest(omitted=omitted):
                self.assertTrue(
                    harness._stable_memory_report_coherent(
                        memory_stability_report(
                            diagnostics=diagnostics,
                            omitted_diagnostic_count=omitted,
                        )
                    )
                )

    def test_tampered_not_requested_reports_are_rejected(self):
        for name, overrides in {
            "sampled": dict(total_sample_count=3),
            "identity": dict(
                identity=harness.ProcessIdentityStatus.PRESERVED
            ),
            "sampling_failure": dict(mandatory_sampling_failure=True),
            "shutdown_failure": dict(mandatory_shutdown_failure=True),
            "metric_attached": dict(
                working_set=stable_metric("working_set", start=1000)
            ),
        }.items():
            with self.subTest(case=name):
                tampered = dataclasses.replace(
                    harness.NOT_REQUESTED_MEMORY_REPORT, **overrides
                )
                aggregate = dataclasses.replace(
                    replace_success(), memory=tampered
                )
                self.assertFalse(
                    harness._repeated_sensor_suite_validation_report_succeeded(
                        aggregate
                    )
                )

    def test_cycle_count_mode_never_accepts_a_memory_verdict(self):
        clock = FakeClock(0.0)
        with unittest.mock.patch.object(
            harness,
            "run_sensor_suite_validation_cycle",
            side_effect=lambda *a, **k: successful_cycle_report(),
        ):
            aggregate = harness.run_repeated_sensor_suite_validation_cycles(
                memory_environ(gate=None, soak="0", cycles="1"),
                monotonic=clock,
            )
        self.assertTrue(
            harness._repeated_sensor_suite_validation_report_succeeded(
                aggregate
            )
        )
        stable = dataclasses.replace(
            aggregate, memory=memory_stability_report()
        )
        self.assertFalse(
            harness._repeated_sensor_suite_validation_report_succeeded(stable)
        )


if __name__ == "__main__":
    unittest.main()
