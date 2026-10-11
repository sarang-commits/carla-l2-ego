"""Focused offline tests for Mission 4G.5 timed SensorSuite soak execution."""

import dataclasses
import inspect
import math
import types
import unittest
import unittest.mock
from dataclasses import FrozenInstanceError

from tests import integration_sensor_suite as harness
from tests.test_repeated_live_sensor_suite_validation import successful_cycle_report


_MISSING = object()


def enabled_environ(*, cycles=_MISSING, soak_seconds=_MISSING):
    environ = {harness.CARLA_INTEGRATION: "1"}
    if cycles is not _MISSING:
        environ[harness.CARLA_INTEGRATION_CYCLES] = str(cycles)
    if soak_seconds is not _MISSING:
        environ[harness.CARLA_INTEGRATION_SOAK_SECONDS] = str(soak_seconds)
    return environ


class SequenceClock:
    def __init__(self, values):
        self.values = list(values)
        self.index = 0

    def __call__(self):
        if not self.values:
            raise AssertionError("clock has no values")
        index = min(self.index, len(self.values) - 1)
        self.index += 1
        return self.values[index]


class MutableClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


def timed_clock_values(cycle_count, *, duration=120.0):
    # Deadline construction; then shared admission/start and finish per cycle;
    # finally rejected admission and the final elapsed-time read.
    return [0.0] + [0.0, 0.0] * cycle_count + [duration, duration]


class RunnerMixin:
    def run_outer(self, reports, environ, clock):
        with unittest.mock.patch.object(
            harness,
            "run_sensor_suite_validation_cycle",
            side_effect=list(reports),
        ) as inner:
            report = harness.run_repeated_sensor_suite_validation_cycles(
                environ,
                importer=unittest.mock.Mock(name="importer"),
                monotonic=clock,
                disappearance_timeout_seconds=2.5,
            )
        return report, inner

    def run_timed_cycles(self, cycle_count):
        reports = tuple(
            successful_cycle_report(ego_id=1000 + index)
            for index in range(1, cycle_count + 1)
        )
        return self.run_outer(
            reports,
            enabled_environ(soak_seconds=120),
            SequenceClock(timed_clock_values(cycle_count)),
        )


class ModeSelectionTest(RunnerMixin, unittest.TestCase):
    def test_absent_variables_select_default_one_cycle_count_mode(self):
        config = harness.parse_integration_config({})
        self.assertEqual(config.cycles, 1)
        self.assertIs(config.cycles_explicit, False)
        self.assertIs(
            harness.select_execution_mode(config), harness.ExecutionMode.CYCLE_COUNT
        )

        aggregate, inner = self.run_outer(
            (successful_cycle_report(),), enabled_environ(), MutableClock()
        )
        self.assertEqual(inner.call_count, 1)
        self.assertIs(aggregate.execution_mode, harness.ExecutionMode.CYCLE_COUNT)
        self.assertEqual(aggregate.requested_cycle_count, 1)
        self.assertEqual(aggregate.requested_soak_seconds, 0.0)

    def test_explicit_cycles_with_absent_or_zero_soak_select_count_mode(self):
        for environ in (
            enabled_environ(cycles=3),
            enabled_environ(cycles=3, soak_seconds=0),
        ):
            with self.subTest(environ=environ):
                config = harness.parse_integration_config(environ)
                self.assertIs(config.cycles_explicit, True)
                self.assertIs(
                    harness.select_execution_mode(config),
                    harness.ExecutionMode.CYCLE_COUNT,
                )

    def test_positive_soak_without_explicit_cycles_selects_timed_soak(self):
        for value in (120, 3600):
            with self.subTest(value=value):
                config = harness.parse_integration_config(
                    enabled_environ(soak_seconds=value)
                )
                self.assertIs(config.cycles_explicit, False)
                self.assertIs(
                    harness.select_execution_mode(config),
                    harness.ExecutionMode.TIMED_SOAK,
                )

    def test_positive_soak_and_explicit_cycles_are_rejected_before_live_work(self):
        environ = enabled_environ(cycles=2, soak_seconds=120)
        names = (
            "run_sensor_suite_validation_cycle",
            "load_live_dependencies",
            "load_sensor_suite_dependencies",
            "connect_client",
            "verify_versions",
            "acquire_world",
            "spawn_owned_ego",
        )
        patches = [unittest.mock.patch.object(harness, name) for name in names]
        mocks = [patch.start() for patch in patches]
        try:
            with self.assertRaisesRegex(ValueError, r"positive .*SOAK.*explicit .*CYCLES"):
                harness.run_repeated_sensor_suite_validation_cycles(environ)
        finally:
            for patch in reversed(patches):
                patch.stop()
        for boundary in mocks:
            boundary.assert_not_called()

    def test_existing_invalid_soak_values_remain_rejected(self):
        for value in ("-1", "0.1", "119.999", "3600.1", "nan", "inf", "abc", ""):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "SOAK_SECONDS"):
                    harness.parse_integration_config(
                        enabled_environ(soak_seconds=value)
                    )


class SoakClockAndAdmissionTest(RunnerMixin, unittest.TestCase):
    def test_clamped_clock_rolls_back_and_expiration_stays_sticky(self):
        clock = harness._ClampedMonotonic(
            SequenceClock((10.0, 12.0, 9.0, 15.0, 14.0))
        )
        deadline = harness.Deadline(5.0, clock)
        self.assertEqual(deadline.remaining(), 3.0)
        self.assertEqual(deadline.remaining(), 3.0)
        self.assertIs(deadline.expired(), True)
        self.assertIs(deadline.expired(), True)

    def test_exact_deadline_before_first_admission_runs_no_cycle(self):
        clock = SequenceClock((0.0, 120.0, 120.0))
        aggregate, inner = self.run_outer(
            (),
            enabled_environ(soak_seconds=120),
            clock,
        )
        inner.assert_not_called()
        self.assertEqual(clock.index, 3)
        self.assertEqual(
            (
                aggregate.attempted_cycle_count,
                aggregate.completed_cycle_count,
                aggregate.passed_cycle_count,
            ),
            (0, 0, 0),
        )
        self.assertIs(
            aggregate.termination_reason,
            harness.TerminationReason.SOAK_DURATION_REACHED,
        )
        self.assertIs(aggregate.success, False)
        self.assertIsNone(aggregate.first_primary_error)
        self.assertIsNone(aggregate.final_disappearance)
        self.assertEqual(aggregate.cycles, ())

    def test_admission_sample_is_reused_as_cycle_start_without_an_extra_read(self):
        clock = SequenceClock((0.0, 119.999, 120.0, 120.0, 120.0))
        raw_reads_at_invocation = []

        def record_raw_read_count(*_args, **_kwargs):
            raw_reads_at_invocation.append(clock.index)
            return successful_cycle_report()

        with unittest.mock.patch.object(
            harness,
            "run_sensor_suite_validation_cycle",
            side_effect=record_raw_read_count,
        ) as inner:
            aggregate = harness.run_repeated_sensor_suite_validation_cycles(
                enabled_environ(soak_seconds=120), monotonic=clock
            )

        self.assertEqual(inner.call_count, 1)
        self.assertEqual(raw_reads_at_invocation, [2])
        self.assertAlmostEqual(aggregate.cycles[0].elapsed_seconds, 0.001)
        self.assertEqual(
            (
                aggregate.attempted_cycle_count,
                aggregate.completed_cycle_count,
                aggregate.passed_cycle_count,
            ),
            (1, 1, 1),
        )
        self.assertIs(
            aggregate.termination_reason,
            harness.TerminationReason.SOAK_DURATION_REACHED,
        )

    def test_one_admitted_cycle_finishes_after_expiry_and_no_next_cycle_starts(self):
        clock = MutableClock()

        def finish_after_deadline(*_args, **_kwargs):
            clock.value = 130.0
            return successful_cycle_report()

        with unittest.mock.patch.object(
            harness,
            "run_sensor_suite_validation_cycle",
            side_effect=finish_after_deadline,
        ) as inner:
            aggregate = harness.run_repeated_sensor_suite_validation_cycles(
                enabled_environ(soak_seconds=120), monotonic=clock
            )
        self.assertEqual(inner.call_count, 1)
        self.assertEqual(aggregate.elapsed_seconds, 130.0)
        self.assertIs(aggregate.success, True)
        self.assertIs(
            aggregate.termination_reason,
            harness.TerminationReason.SOAK_DURATION_REACHED,
        )

    def test_several_cycles_run_back_to_back_until_admission_expires(self):
        clock = MutableClock()
        calls = 0

        def advance_cycle(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            clock.value += 50.0
            return successful_cycle_report(ego_id=100 + calls)

        with unittest.mock.patch.object(
            harness, "run_sensor_suite_validation_cycle", side_effect=advance_cycle
        ) as inner:
            aggregate = harness.run_repeated_sensor_suite_validation_cycles(
                enabled_environ(soak_seconds=120), monotonic=clock
            )
        self.assertEqual(inner.call_count, 3)
        self.assertEqual(aggregate.completed_cycle_count, 3)
        self.assertEqual(aggregate.elapsed_seconds, 150.0)
        self.assertTrue(aggregate.success)

    def test_summary_processing_counts_toward_elapsed_soak_time(self):
        clock = MutableClock()
        original = harness._summarize_repeated_cycle

        def summarize_then_advance(*args, **kwargs):
            result = original(*args, **kwargs)
            clock.value = 125.0
            return result

        with (
            unittest.mock.patch.object(
                harness,
                "run_sensor_suite_validation_cycle",
                return_value=successful_cycle_report(),
            ) as inner,
            unittest.mock.patch.object(
                harness, "_summarize_repeated_cycle", side_effect=summarize_then_advance
            ),
        ):
            aggregate = harness.run_repeated_sensor_suite_validation_cycles(
                enabled_environ(soak_seconds=120), monotonic=clock
            )
        self.assertEqual(inner.call_count, 1)
        self.assertEqual(aggregate.elapsed_seconds, 125.0)

    def test_inner_receives_shared_clock_but_no_total_deadline(self):
        aggregate, inner = self.run_timed_cycles(1)
        self.assertTrue(aggregate.success)
        call = inner.call_args
        self.assertIsInstance(call.kwargs["monotonic"], harness._ClampedMonotonic)
        self.assertNotIn("deadline", call.kwargs)
        self.assertNotIn("total_soak_deadline", call.kwargs)


class CycleCountCompatibilityTest(RunnerMixin, unittest.TestCase):
    def test_one_and_multiple_cycles_end_at_exact_count_limit(self):
        for count in (1, 3, 20):
            with self.subTest(count=count):
                reports = tuple(
                    successful_cycle_report(ego_id=100 + index)
                    for index in range(count)
                )
                aggregate, inner = self.run_outer(
                    reports, enabled_environ(cycles=count), MutableClock()
                )
                self.assertEqual(inner.call_count, count)
                self.assertEqual(aggregate.completed_cycle_count, count)
                self.assertEqual(aggregate.requested_cycle_count, count)
                self.assertIs(
                    aggregate.termination_reason,
                    harness.TerminationReason.CYCLE_LIMIT_REACHED,
                )
                self.assertTrue(aggregate.success)


class FailureAndInterruptionTest(RunnerMixin, unittest.TestCase):
    def test_failure_reason_precedence_and_no_future_cycle(self):
        functional = successful_cycle_report(
            primary_error=harness.ValidationFailure("readiness", "failed", 1),
            cleanup_errors=("suite cleanup failed",),
            final_disappearance=harness.CleanupVerification.UNVERIFIED_DISCONNECTED,
            remaining_owned_actor_ids=(900,),
        )
        cleanup = successful_cycle_report(cleanup_errors=("suite cleanup failed",))
        disappearance = successful_cycle_report(
            final_disappearance=harness.CleanupVerification.FAILED_ACTORS_REMAIN,
            remaining_owned_actor_ids=(901,),
        )
        unverified = successful_cycle_report(
            final_disappearance=harness.CleanupVerification.UNVERIFIED_DISCONNECTED
        )
        cases = (
            (functional, harness.TerminationReason.FUNCTIONAL_FAILURE),
            (cleanup, harness.TerminationReason.CLEANUP_FAILURE),
            (disappearance, harness.TerminationReason.DISAPPEARANCE_FAILURE),
            (unverified, harness.TerminationReason.DISAPPEARANCE_FAILURE),
        )
        for failed, expected in cases:
            with self.subTest(expected=expected):
                aggregate, inner = self.run_outer(
                    (failed, successful_cycle_report()),
                    enabled_environ(soak_seconds=120),
                    MutableClock(),
                )
                self.assertEqual(inner.call_count, 1)
                self.assertIs(aggregate.termination_reason, expected)
                self.assertFalse(aggregate.success)

    def test_first_cycle_disappearance_failure_reports_exact_live_signature(self):
        # The verified 2026-07-20 live failure: one completed timed-soak cycle,
        # destroys reported success, yet all six owned actors stayed visible.
        failed = successful_cycle_report(
            ego_id=24,
            sensor_actor_ids=(25, 26, 27, 28, 29),
            final_disappearance=harness.CleanupVerification.FAILED_ACTORS_REMAIN,
            remaining_owned_actor_ids=(25, 26, 27, 28, 29, 24),
        )
        aggregate, inner = self.run_outer(
            (failed, successful_cycle_report()),
            enabled_environ(soak_seconds=120),
            MutableClock(),
        )
        self.assertEqual(inner.call_count, 1)
        self.assertIs(aggregate.execution_mode, harness.ExecutionMode.TIMED_SOAK)
        self.assertIs(
            aggregate.termination_reason,
            harness.TerminationReason.DISAPPEARANCE_FAILURE,
        )
        self.assertIsNone(aggregate.requested_cycle_count)
        self.assertEqual(aggregate.requested_soak_seconds, 120.0)
        self.assertEqual(aggregate.attempted_cycle_count, 1)
        self.assertEqual(aggregate.completed_cycle_count, 1)
        self.assertEqual(aggregate.passed_cycle_count, 0)
        self.assertEqual(aggregate.failed_cycle_index, 1)
        self.assertIsNone(aggregate.first_primary_error)
        self.assertEqual(aggregate.supplemental_cleanup_errors, ())
        self.assertEqual(aggregate.omitted_cycle_summary_count, 0)
        self.assertEqual(aggregate.omitted_cleanup_error_count, 0)
        self.assertEqual(
            aggregate.remaining_owned_actors,
            tuple(
                harness.ActorKey(1, actor_id)
                for actor_id in (25, 26, 27, 28, 29, 24)
            ),
        )
        self.assertFalse(aggregate.success)
        self.assertFalse(
            harness._repeated_sensor_suite_validation_report_succeeded(aggregate)
        )

    def test_failure_after_duration_expires_inside_admitted_cycle_wins(self):
        clock = MutableClock()
        failed = successful_cycle_report(
            primary_error=harness.ValidationFailure("sensor_readiness", "failed", 1)
        )

        def fail_after_expiry(*_args, **_kwargs):
            clock.value = 130.0
            return failed

        with unittest.mock.patch.object(
            harness, "run_sensor_suite_validation_cycle", side_effect=fail_after_expiry
        ) as inner:
            aggregate = harness.run_repeated_sensor_suite_validation_cycles(
                enabled_environ(soak_seconds=120), monotonic=clock
            )
        self.assertEqual(inner.call_count, 1)
        self.assertIs(
            aggregate.termination_reason, harness.TerminationReason.FUNCTIONAL_FAILURE
        )

    def test_escaping_exceptions_propagate_by_exact_identity(self):
        class AbortSignal(BaseException):
            pass

        for exception in (
            unittest.SkipTest("skip"),
            RuntimeError("disconnect"),
            KeyboardInterrupt("interrupt"),
            SystemExit(19),
            AbortSignal("abort"),
        ):
            with self.subTest(exception=type(exception).__name__):
                with unittest.mock.patch.object(
                    harness,
                    "run_sensor_suite_validation_cycle",
                    side_effect=exception,
                ) as inner:
                    with self.assertRaises(type(exception)) as caught:
                        harness.run_repeated_sensor_suite_validation_cycles(
                            enabled_environ(soak_seconds=120),
                            monotonic=MutableClock(),
                        )
                self.assertIs(caught.exception, exception)
                self.assertEqual(inner.call_count, 1)


class BoundedRetentionTest(RunnerMixin, unittest.TestCase):
    def test_boundary_counts_retain_exact_first_and_final_regions(self):
        for count in (0, 1, 8, 24, 31, 32, 33, 100):
            with self.subTest(count=count):
                aggregate, inner = self.run_timed_cycles(count)
                self.assertEqual(inner.call_count, count)
                self.assertEqual(aggregate.completed_cycle_count, count)
                expected_indexes = list(range(1, count + 1))
                if count > 32:
                    expected_indexes = list(range(1, 9)) + list(
                        range(count - 23, count + 1)
                    )
                self.assertEqual(
                    [summary.cycle_index for summary in aggregate.cycles],
                    expected_indexes,
                )
                self.assertEqual(
                    aggregate.omitted_cycle_summary_count,
                    max(0, count - 32),
                )
                self.assertEqual(
                    len({summary.cycle_index for summary in aggregate.cycles}),
                    len(aggregate.cycles),
                )
                self.assertIs(aggregate.success, count > 0)

    def test_thirty_one_cycles_are_all_retained_in_chronological_unique_order(self):
        aggregate, inner = self.run_timed_cycles(31)
        retained_indexes = tuple(
            summary.cycle_index for summary in aggregate.cycles
        )

        self.assertEqual(inner.call_count, 31)
        self.assertEqual(
            (
                aggregate.attempted_cycle_count,
                aggregate.completed_cycle_count,
                aggregate.passed_cycle_count,
            ),
            (31, 31, 31),
        )
        self.assertEqual(len(aggregate.cycles), 31)
        self.assertEqual(retained_indexes, tuple(range(1, 32)))
        self.assertEqual(retained_indexes, tuple(sorted(retained_indexes)))
        self.assertEqual(len(set(retained_indexes)), 31)
        self.assertEqual(aggregate.omitted_cycle_summary_count, 0)

    def test_failed_final_summary_is_retained_after_many_successes(self):
        reports = [
            successful_cycle_report(ego_id=1000 + index) for index in range(1, 40)
        ]
        reports.append(
            successful_cycle_report(
                ego_id=1040,
                primary_error=harness.ValidationFailure("readiness", "failed", 1),
            )
        )
        aggregate, inner = self.run_outer(
            reports,
            enabled_environ(soak_seconds=120),
            MutableClock(),
        )
        self.assertEqual(inner.call_count, 40)
        self.assertEqual(aggregate.completed_cycle_count, 40)
        self.assertEqual(aggregate.cycles[-1].cycle_index, 40)
        self.assertFalse(aggregate.cycles[-1].success)
        self.assertEqual(aggregate.omitted_cycle_summary_count, 8)

    def test_cleanup_diagnostics_are_bounded_counted_and_copied(self):
        errors = tuple(RuntimeError("x" * 1000) for _ in range(12))
        failed = successful_cycle_report(cleanup_errors=errors)
        aggregate, _inner = self.run_outer(
            (failed,), enabled_environ(cycles=1), MutableClock()
        )
        self.assertEqual(len(aggregate.cycles[0].cleanup_errors), 8)
        self.assertEqual(len(aggregate.supplemental_cleanup_errors), 8)
        self.assertEqual(aggregate.omitted_cleanup_error_count, 4)
        self.assertTrue(
            all(
                type(value) is str
                and len(value) <= harness.MAX_CLEANUP_ERROR_LENGTH
                for value in aggregate.supplemental_cleanup_errors
            )
        )
        self.assertFalse(
            any(
                isinstance(value, BaseException)
                for value in aggregate.supplemental_cleanup_errors
            )
        )


class ReportAndSourceContractTest(RunnerMixin, unittest.TestCase):
    def test_report_is_frozen_slotted_tuple_based_and_carla_free(self):
        inner_report = successful_cycle_report()
        aggregate, _inner = self.run_outer(
            (inner_report,), enabled_environ(cycles=1), MutableClock()
        )
        with self.assertRaises(FrozenInstanceError):
            aggregate.success = False
        self.assertFalse(hasattr(aggregate, "__dict__"))
        self.assertIs(type(aggregate.cycles), tuple)
        self.assertIs(type(aggregate.supplemental_cleanup_errors), tuple)
        self.assertIs(type(aggregate.remaining_owned_actors), tuple)

        forbidden_ids = {id(inner_report)}
        seen = set()

        def walk(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            self.assertNotIn(id(value), forbidden_ids)
            self.assertNotIsInstance(value, BaseException)
            self.assertNotIsInstance(value, types.TracebackType)
            self.assertNotIsInstance(value, (bytes, bytearray, memoryview))
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                for field in dataclasses.fields(value):
                    walk(getattr(value, field.name))
            elif isinstance(value, tuple):
                for item in value:
                    walk(item)

        walk(aggregate)

    def test_single_cycle_remains_unaware_of_outer_soak_contract(self):
        source = inspect.getsource(harness.run_sensor_suite_validation_cycle)
        for forbidden in (
            "ExecutionMode",
            "TerminationReason",
            "total_soak_deadline",
            "FIRST_RETAINED_CYCLE_SUMMARIES",
            "FINAL_RETAINED_CYCLE_SUMMARIES",
            "evaluate_memory_growth",
            "MemorySample",
            "MemoryStability",
            "admission_guard",
            "windows_memory_sampler",
            "tracemalloc",
            "gc.collect",
            "time.sleep",
        ):
            self.assertNotIn(forbidden, source)

    def test_outer_source_has_no_memory_collection_sleep_or_unrelated_live_scope(self):
        source = inspect.getsource(harness.run_repeated_sensor_suite_validation_cycles)
        for forbidden in (
            "MemorySample",
            "MemoryGrowthResult",
            "evaluate_memory_growth",
            "tracemalloc",
            "gc.collect",
            "psutil",
            "getrusage",
            "GetProcessMemoryInfo",
            "time.sleep",
            "get_actors(",
            "apply_settings",
            "get_trafficmanager",
            "apply_control",
            "pygame",
            "spectator",
            "autopilot",
            "recording",
        ):
            self.assertNotIn(forbidden, source)

    def test_exactly_one_live_unittest_entry_remains(self):
        live_methods = []
        for value in vars(harness).values():
            if isinstance(value, type) and issubclass(value, unittest.TestCase):
                live_methods.extend(
                    name for name in vars(value) if name.startswith("test_")
                )
        self.assertEqual(live_methods, ["test_connection_and_single_owned_ego_cycle"])

    def test_requested_soak_and_elapsed_are_finite_floats(self):
        aggregate, _inner = self.run_timed_cycles(1)
        self.assertIs(type(aggregate.requested_soak_seconds), float)
        self.assertIs(type(aggregate.elapsed_seconds), float)
        self.assertTrue(math.isfinite(aggregate.elapsed_seconds))
        self.assertTrue(
            harness._repeated_sensor_suite_validation_report_succeeded(aggregate)
        )


if __name__ == "__main__":
    unittest.main()
