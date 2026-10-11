"""Focused offline tests for Mission 4G.4 repeated-cycle orchestration."""

import ast
import dataclasses
import inspect
import textwrap
import types
import unittest
import unittest.mock
from dataclasses import FrozenInstanceError

from tests import integration_sensor_suite as harness


def enabled_environ(*, cycles=None, soak_seconds=None):
    environ = {harness.CARLA_INTEGRATION: "1"}
    if cycles is not None:
        environ[harness.CARLA_INTEGRATION_CYCLES] = str(cycles)
    if soak_seconds is not None:
        environ[harness.CARLA_INTEGRATION_SOAK_SECONDS] = str(soak_seconds)
    return environ


def successful_cycle_report(*, ego_id=101, sensor_actor_ids=None, **overrides):
    if sensor_actor_ids is None:
        sensor_actor_ids = tuple(ego_id + 100 + index for index in range(5))
    names = harness.EXPECTED_SENSOR_NAMES
    types_ = harness.EXPECTED_SENSOR_TYPES
    values = {
        "versions": harness.VersionReport("0.9.16", "0.9.16", True),
        "world_mode": harness.WorldMode.ASYNCHRONOUS,
        "policy": harness.ProgressionPolicy.PASSIVE_WAIT,
        "ego_id": ego_id,
        "ego_type_id": "vehicle.tesla.model3",
        "ego_role_valid": True,
        "sensor_identities": tuple(
            harness.SensorIdentityRecord(name, actor_id, type_id)
            for name, actor_id, type_id in zip(
                names, sensor_actor_ids, types_, strict=True
            )
        ),
        "attachment_success": True,
        "tick_registration_success": True,
        "readiness_success": True,
        "sensor_health": tuple(
            harness.SensorHealthReport(
                name=name,
                attached=True,
                messages_received=(
                    0 if name in ("collision", "lane_invasion") else 1
                ),
                error_count=0,
                callback_error_present=False,
            )
            for name in names
        ),
        "ego_telemetry_present": True,
        "gnss_present": True,
        "imu_present": True,
        "camera_present": True,
        "camera": harness.CameraValidationReport(
            harness.EXPECTED_CAMERA_WIDTH,
            harness.EXPECTED_CAMERA_HEIGHT,
            harness.EXPECTED_CAMERA_FOV,
            harness.EXPECTED_CAMERA_PIXEL_FORMAT,
            harness.EXPECTED_CAMERA_ROW_STRIDE,
            harness.EXPECTED_CAMERA_WIDTH
            * harness.EXPECTED_CAMERA_HEIGHT
            * 3,
            True,
        ),
        "primary_error": None,
        "cleanup_errors": (),
        "final_disappearance": harness.CleanupVerification.CONFIRMED,
        "remaining_owned_actor_ids": (),
    }
    values.update(overrides)
    return harness.SensorSuiteValidationReport(**values)


class RepeatedRunnerTestCase(unittest.TestCase):
    def run_reports(self, reports, *, cycles=None, environ=None):
        if environ is None:
            environ = enabled_environ(cycles=cycles)
        importer = unittest.mock.Mock(name="importer")
        monotonic = unittest.mock.Mock(name="monotonic", return_value=42.0)
        with unittest.mock.patch.object(
            harness,
            "run_sensor_suite_validation_cycle",
            side_effect=list(reports),
        ) as inner:
            aggregate = harness.run_repeated_sensor_suite_validation_cycles(
                environ,
                importer=importer,
                monotonic=monotonic,
                disappearance_timeout_seconds=2.75,
            )
        return aggregate, inner, environ, importer, monotonic

    def assert_forwarded(self, calls, environ, importer, monotonic):
        for call in calls:
            if call.args:
                self.assertIs(call.args[0], environ)
            else:
                self.assertIs(call.kwargs["environ"], environ)
            self.assertIs(call.kwargs["importer"], importer)
            self.assertIs(call.kwargs["monotonic"], monotonic)
            self.assertEqual(call.kwargs["disappearance_timeout_seconds"], 2.75)


class SuccessfulRepetitionTest(RepeatedRunnerTestCase):
    def test_default_one_cycle_is_one_complete_delegated_lifecycle(self):
        inner_report = successful_cycle_report()
        aggregate, inner, environ, importer, monotonic = self.run_reports(
            (inner_report,), environ=enabled_environ()
        )

        self.assertEqual(
            (
                aggregate.requested_cycle_count,
                aggregate.attempted_cycle_count,
                aggregate.completed_cycle_count,
                aggregate.passed_cycle_count,
            ),
            (1, 1, 1, 1),
        )
        self.assertIs(aggregate.success, True)
        self.assertIs(aggregate.execution_mode, harness.ExecutionMode.CYCLE_COUNT)
        self.assertIs(
            aggregate.termination_reason,
            harness.TerminationReason.CYCLE_LIMIT_REACHED,
        )
        self.assertEqual(aggregate.requested_soak_seconds, 0.0)
        self.assertEqual(aggregate.omitted_cycle_summary_count, 0)
        self.assertEqual(aggregate.omitted_cleanup_error_count, 0)
        self.assertIs(
            aggregate.final_disappearance, harness.CleanupVerification.CONFIRMED
        )
        self.assertIsNone(aggregate.first_primary_error)
        self.assertIsNone(aggregate.failed_cycle_index)
        self.assertEqual(aggregate.supplemental_cleanup_errors, ())
        self.assertEqual(aggregate.remaining_owned_actors, ())
        self.assertEqual(len(aggregate.cycles), 1)
        summary = aggregate.cycles[0]
        self.assertEqual(summary.cycle_index, 1)
        self.assertIs(summary.success, True)
        self.assertEqual(summary.ego_id, inner_report.ego_id)
        self.assertEqual(
            summary.sensor_actor_ids,
            tuple(item.actor_id for item in inner_report.sensor_identities),
        )
        self.assertIs(summary.readiness_success, True)
        self.assertIsNone(summary.primary_error)
        self.assertEqual(summary.cleanup_errors, ())
        self.assertIs(
            summary.disappearance, harness.CleanupVerification.CONFIRMED
        )
        self.assertEqual(summary.remaining_actor_ids, ())
        self.assertIs(type(summary.elapsed_seconds), float)
        self.assertIs(type(aggregate.elapsed_seconds), float)
        self.assertEqual(inner.call_count, 1)
        self.assert_forwarded(inner.call_args_list, environ, importer, monotonic)

    def test_three_cycles_delegate_a_fresh_complete_cycle_each_time(self):
        reports = tuple(
            successful_cycle_report(ego_id=100 + cycle) for cycle in range(1, 4)
        )
        aggregate, inner, environ, importer, monotonic = self.run_reports(
            reports, cycles=3
        )

        self.assertEqual(inner.call_count, 3)
        self.assertEqual(
            (
                aggregate.requested_cycle_count,
                aggregate.attempted_cycle_count,
                aggregate.completed_cycle_count,
                aggregate.passed_cycle_count,
            ),
            (3, 3, 3, 3),
        )
        self.assertEqual(
            tuple(summary.cycle_index for summary in aggregate.cycles),
            (1, 2, 3),
        )
        self.assertEqual(
            tuple(summary.ego_id for summary in aggregate.cycles),
            (101, 102, 103),
        )
        self.assertTrue(all(summary.success for summary in aggregate.cycles))
        self.assertIs(aggregate.success, True)
        self.assert_forwarded(inner.call_args_list, environ, importer, monotonic)

    def test_twenty_cycle_upper_boundary_runs_exactly_twenty_cycles(self):
        reports = tuple(
            successful_cycle_report(ego_id=1000 + cycle)
            for cycle in range(1, 21)
        )
        aggregate, inner, _environ, _importer, _monotonic = self.run_reports(
            reports, cycles=20
        )

        self.assertEqual(inner.call_count, 20)
        self.assertEqual(aggregate.requested_cycle_count, 20)
        self.assertEqual(aggregate.attempted_cycle_count, 20)
        self.assertEqual(aggregate.completed_cycle_count, 20)
        self.assertEqual(aggregate.passed_cycle_count, 20)
        self.assertEqual(len(aggregate.cycles), 20)
        self.assertEqual(aggregate.cycles[-1].cycle_index, 20)
        self.assertIs(aggregate.success, True)


class ConfigurationBoundaryTest(unittest.TestCase):
    def test_disabled_outer_runner_skips_before_import_or_inner_cycle(self):
        importer = unittest.mock.Mock(name="importer")
        with unittest.mock.patch.object(
            harness, "run_sensor_suite_validation_cycle"
        ) as inner:
            with self.assertRaisesRegex(
                unittest.SkipTest, f"^{harness.DISABLED_MESSAGE}$"
            ):
                harness.run_repeated_sensor_suite_validation_cycles(
                    {}, importer=importer
                )
        inner.assert_not_called()
        importer.assert_not_called()

    def test_nonzero_soak_is_rejected_before_all_live_boundaries(self):
        environ = enabled_environ(cycles=3, soak_seconds=120)
        importer = unittest.mock.Mock(name="importer")
        patch_names = (
            "run_sensor_suite_validation_cycle",
            "load_live_dependencies",
            "connect_client",
            "acquire_world",
            "load_sensor_suite_dependencies",
        )
        patches = [unittest.mock.patch.object(harness, name) for name in patch_names]
        mocks = []
        try:
            for patch in patches:
                mocks.append(patch.start())
            with self.assertRaisesRegex(
                ValueError, r"(?i)soak.*(?:combined|cycles)"
            ):
                harness.run_repeated_sensor_suite_validation_cycles(
                    environ, importer=importer
                )
        finally:
            for patch in reversed(patches):
                patch.stop()

        importer.assert_not_called()
        for boundary in mocks:
            boundary.assert_not_called()


class FailFastTest(RepeatedRunnerTestCase):
    def test_functional_failure_in_first_middle_or_final_cycle_stops_repetition(self):
        for failed_cycle in (1, 2, 3):
            with self.subTest(failed_cycle=failed_cycle):
                reports = []
                for cycle in range(1, 4):
                    failure = None
                    if cycle == failed_cycle:
                        # The complete inner API is always cycle 1; the outer
                        # summary must rebind that diagnostic to its run index.
                        failure = harness.ValidationFailure(
                            "sensor_readiness", "required telemetry missing", 1
                        )
                    reports.append(
                        successful_cycle_report(
                            ego_id=100 + cycle, primary_error=failure
                        )
                    )

                aggregate, inner, _env, _importer, _clock = self.run_reports(
                    reports, cycles=3
                )

                self.assertEqual(inner.call_count, failed_cycle)
                self.assertEqual(aggregate.requested_cycle_count, 3)
                self.assertEqual(aggregate.attempted_cycle_count, failed_cycle)
                self.assertEqual(aggregate.completed_cycle_count, failed_cycle)
                self.assertEqual(aggregate.passed_cycle_count, failed_cycle - 1)
                self.assertEqual(len(aggregate.cycles), failed_cycle)
                self.assertEqual(aggregate.failed_cycle_index, failed_cycle)
                self.assertIs(aggregate.success, False)
                self.assertEqual(
                    aggregate.first_primary_error.phase, "sensor_readiness"
                )
                self.assertEqual(
                    aggregate.first_primary_error.cycle, failed_cycle
                )
                self.assertIs(aggregate.cycles[-1].success, False)

    def test_cleanup_disappearance_unverified_or_remaining_ids_stop_next_cycle(self):
        cases = (
            (
                "cleanup",
                {"cleanup_errors": ("suite RuntimeError: cleanup failed",)},
                (),
            ),
            (
                "disappearance",
                {
                    "final_disappearance": (
                        harness.CleanupVerification.FAILED_ACTORS_REMAIN
                    )
                },
                (),
            ),
            (
                "unverified",
                {
                    "final_disappearance": (
                        harness.CleanupVerification.UNVERIFIED_DISCONNECTED
                    )
                },
                (),
            ),
            (
                "remaining",
                {"remaining_owned_actor_ids": (919,)},
                (harness.ActorKey(2, 919),),
            ),
        )
        for name, overrides, expected_remaining in cases:
            with self.subTest(name=name):
                reports = (
                    successful_cycle_report(ego_id=101),
                    successful_cycle_report(ego_id=102, **overrides),
                    successful_cycle_report(ego_id=103),
                )
                aggregate, inner, _env, _importer, _clock = self.run_reports(
                    reports, cycles=3
                )

                self.assertEqual(inner.call_count, 2)
                self.assertEqual(
                    (
                        aggregate.attempted_cycle_count,
                        aggregate.completed_cycle_count,
                        aggregate.passed_cycle_count,
                    ),
                    (2, 2, 1),
                )
                self.assertEqual(aggregate.failed_cycle_index, 2)
                self.assertEqual(len(aggregate.cycles), 2)
                self.assertIs(aggregate.cycles[-1].success, False)
                self.assertEqual(
                    aggregate.remaining_owned_actors, expected_remaining
                )
                self.assertIs(aggregate.success, False)
                if name == "cleanup":
                    self.assertTrue(
                        any(
                            "cleanup failed" in error
                            for error in aggregate.supplemental_cleanup_errors
                        )
                    )


class ExceptionPropagationTest(unittest.TestCase):
    class AbortSignal(BaseException):
        pass

    def test_escaping_exceptions_and_interruptions_propagate_by_identity(self):
        exceptions = (
            unittest.SkipTest("unsafe synchronous world"),
            RuntimeError("connection setup failed"),
            KeyboardInterrupt("operator interrupt"),
            SystemExit(17),
            self.AbortSignal("nonordinary abort"),
        )
        for exception in exceptions:
            with self.subTest(exception=type(exception).__name__):
                environ = enabled_environ(cycles=3)
                with unittest.mock.patch.object(
                    harness,
                    "run_sensor_suite_validation_cycle",
                    side_effect=exception,
                ) as inner:
                    with self.assertRaises(type(exception)) as caught:
                        harness.run_repeated_sensor_suite_validation_cycles(
                            environ, monotonic=lambda: 10.0
                        )
                self.assertIs(caught.exception, exception)
                self.assertEqual(inner.call_count, 1)

    def test_base_exception_between_cycles_releases_report_and_stops_next_cycle(self):
        interrupt = self.AbortSignal("between completed cycles")
        first_report = successful_cycle_report()
        environ = enabled_environ(cycles=3)
        clock_calls = 0

        def interrupt_before_second_cycle():
            nonlocal clock_calls
            clock_calls += 1
            # run start, cycle 1 start, cycle 1 finish, cycle 2 start
            if clock_calls == 4:
                raise interrupt
            return 10.0

        with unittest.mock.patch.object(
            harness,
            "run_sensor_suite_validation_cycle",
            return_value=first_report,
        ) as inner:
            try:
                harness.run_repeated_sensor_suite_validation_cycles(
                    environ, monotonic=interrupt_before_second_cycle
                )
            except self.AbortSignal as caught:
                self.assertIs(caught, interrupt)
                runner_frames = []
                traceback = caught.__traceback__
                while traceback is not None:
                    if (
                        traceback.tb_frame.f_code
                        is harness.run_repeated_sensor_suite_validation_cycles.__code__
                    ):
                        runner_frames.append(traceback.tb_frame)
                    traceback = traceback.tb_next
                self.assertEqual(len(runner_frames), 1)
                runner_locals = runner_frames[0].f_locals
                self.assertIsNone(runner_locals.get("cycle_report"))
                self.assertTrue(
                    all(value is not first_report for value in runner_locals.values())
                )
                # Avoid making this test itself retain frames after inspection.
                caught.__traceback__ = None
                runner_frames.clear()
                traceback = None
            else:
                self.fail("between-cycle BaseException did not propagate")
        self.assertEqual(inner.call_count, 1)


class ActorIdentityAndReportSafetyTest(RepeatedRunnerTestCase):
    def test_numeric_actor_id_reuse_is_cycle_local(self):
        sensor_ids = (201, 202, 203, 204, 205)
        reports = (
            successful_cycle_report(ego_id=101, sensor_actor_ids=sensor_ids),
            successful_cycle_report(ego_id=101, sensor_actor_ids=sensor_ids),
        )
        aggregate, _inner, _env, _importer, _clock = self.run_reports(
            reports, cycles=2
        )

        self.assertIs(aggregate.success, True)
        self.assertEqual(
            tuple(summary.ego_id for summary in aggregate.cycles), (101, 101)
        )
        self.assertEqual(aggregate.cycles[0].sensor_actor_ids, sensor_ids)
        self.assertEqual(aggregate.cycles[1].sensor_actor_ids, sensor_ids)
        first = harness.ActorKey(1, 101)
        second = harness.ActorKey(2, 101)
        self.assertNotEqual(first, second)
        self.assertEqual(len({first, second}), 2)

    def test_report_values_are_frozen_slotted_and_tuple_based(self):
        aggregate, _inner, _env, _importer, _clock = self.run_reports(
            (successful_cycle_report(),), cycles=1
        )
        summary = aggregate.cycles[0]
        actor_key = harness.ActorKey(1, 101)

        for value, attribute, replacement in (
            (actor_key, "actor_id", 999),
            (summary, "success", False),
            (aggregate, "success", False),
        ):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(value, attribute, replacement)
                self.assertFalse(hasattr(value, "__dict__"))

        self.assertIs(type(aggregate.cycles), tuple)
        self.assertIs(type(aggregate.supplemental_cleanup_errors), tuple)
        self.assertIs(type(aggregate.remaining_owned_actors), tuple)
        self.assertIs(type(aggregate.omitted_cycle_summary_count), int)
        self.assertIs(type(aggregate.omitted_cleanup_error_count), int)
        self.assertIs(type(summary.sensor_actor_ids), tuple)
        self.assertIs(type(summary.cleanup_errors), tuple)
        self.assertIs(type(summary.remaining_actor_ids), tuple)

    def test_aggregate_copies_bounded_values_and_retains_no_inner_report(self):
        class CarlaProxy:
            pass

        class RetainingText(str):
            def __new__(cls, value, retained):
                instance = super().__new__(cls, value)
                instance.retained = retained
                return instance

        proxy = CarlaProxy()
        long_phase = RetainingText("p" * 500, proxy)
        long_message = RetainingText("m" * 2000, proxy)
        long_cleanup = RetainingText("c" * 2000, proxy)
        inner_report = successful_cycle_report(
            primary_error=harness.ValidationFailure(
                long_phase, long_message, 1
            ),
            cleanup_errors=(long_cleanup,),
        )
        aggregate, _inner, _env, _importer, _clock = self.run_reports(
            (inner_report,), cycles=1
        )

        self.assertIs(type(aggregate.cycles[0]), harness.RepeatedCycleSummary)
        self.assertLessEqual(
            len(aggregate.cycles[0].primary_error.phase),
            harness.MAX_PRIMARY_PHASE_LENGTH,
        )
        self.assertLessEqual(
            len(aggregate.cycles[0].primary_error.message),
            harness.MAX_PRIMARY_MESSAGE_LENGTH,
        )
        self.assertTrue(
            all(
                type(error) is str
                and len(error) <= harness.MAX_CLEANUP_ERROR_LENGTH
                for error in aggregate.cycles[0].cleanup_errors
            )
        )
        self.assertTrue(
            all(
                type(error) is str
                and len(error) <= harness.MAX_CLEANUP_ERROR_LENGTH
                for error in aggregate.supplemental_cleanup_errors
            )
        )

        forbidden_ids = {id(inner_report), id(proxy)}
        seen = set()

        def walk(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            self.assertNotIn(id(value), forbidden_ids)
            self.assertNotIsInstance(value, RetainingText)
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


class RepeatedSourceContractTest(unittest.TestCase):
    def test_inner_cycle_remains_single_cycle_and_outer_only_orchestrates(self):
        inner_source = inspect.getsource(harness.run_sensor_suite_validation_cycle)
        for forbidden in (
            "CARLA_INTEGRATION_CYCLES",
            "CARLA_INTEGRATION_SOAK_SECONDS",
            "config.cycles",
            "config.soak_seconds",
            "run_repeated_sensor_suite_validation_cycles",
            "tracemalloc",
            "gc.collect",
            "time.sleep",
        ):
            self.assertNotIn(forbidden, inner_source)

        outer_source = inspect.getsource(
            harness.run_repeated_sensor_suite_validation_cycles
        )
        self.assertIn("config.cycles", outer_source)
        self.assertIn("config.soak_seconds", outer_source)
        self.assertIn("run_sensor_suite_validation_cycle", outer_source)

        tree = ast.parse(textwrap.dedent(outer_source))
        referenced_names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        referenced_attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        forbidden_names = {
            "MemorySample",
            "MemoryGrowthResult",
            "evaluate_memory_growth",
            "MemoryStabilityMonitor",
            "WindowsProcessMemorySampler",
            "ctypes",
            "tracemalloc",
            "gc",
            "collect",
            "psutil",
            "resource",
            "getrusage",
            "GetProcessMemoryInfo",
            "load_live_dependencies",
            "connect_client",
            "acquire_world",
            "load_sensor_suite_dependencies",
        }
        self.assertFalse(
            (referenced_names | referenced_attributes) & forbidden_names
        )

    def test_exactly_one_live_unittest_entry_remains(self):
        live_methods = []
        for value in vars(harness).values():
            if isinstance(value, type) and issubclass(value, unittest.TestCase):
                live_methods.extend(
                    name for name in vars(value) if name.startswith("test_")
                )
        self.assertEqual(len(live_methods), 1)


if __name__ == "__main__":
    unittest.main()
