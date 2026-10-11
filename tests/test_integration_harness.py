"""Deterministic offline tests for the Mission 4G.1 harness contract."""

import ast
import json
import math
import os
import subprocess
import sys
import unittest
import unittest.mock
import unicodedata
from dataclasses import FrozenInstanceError
from pathlib import Path

from tests import integration_sensor_suite as harness
from tests.integration_sensor_suite import (
    CARLA_HOST,
    CARLA_INTEGRATION,
    CARLA_INTEGRATION_CYCLES,
    CARLA_INTEGRATION_SOAK_SECONDS,
    CARLA_INTEGRATION_TICK_OWNER,
    CARLA_PORT,
    CARLA_TIMEOUT_SECONDS,
    DISABLED_MESSAGE,
    EXPECTED_SENSOR_NAMES,
    REQUIRED_CARLA_VERSION,
    CarlaVersion,
    CleanupOutcome,
    CleanupVerification,
    Deadline,
    IntegrationConfig,
    LiveDependencies,
    MemoryGrowthResult,
    MemorySample,
    OwnedEgo,
    PhaseTimeout,
    ProgressionPolicy,
    ValidationFailure,
    WorldMode,
    acquire_world,
    capture_owned_ego,
    configure_ego_blueprint,
    connect_client,
    coordinate_cleanup,
    decide_world_progression,
    destroy_owned_ego,
    evaluate_memory_growth,
    format_cleanup_outcome,
    inspect_world_mode,
    integration_enabled,
    load_live_dependencies,
    new_cycle_state,
    parse_carla_version,
    parse_integration_config,
    progress_world_once,
    require_integration_opt_in,
    require_progression_policy,
    run_connection_spawn_cycle,
    spawn_owned_ego,
    verify_ego_disappearance,
    verify_ego_identity,
    verify_versions,
    versions_compatible,
)

BANNED_IMPORTS = (
    "carla",
    "pygame",
    "numpy",
    "sensors",
    "telemetry",
    "spawn_ego_vehicle",
    "manual_drive",
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HARNESS_PATH = Path(harness.__file__).resolve()


class FakeClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


class CountingClock(FakeClock):
    def __init__(self, value=0.0):
        super().__init__(value)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return super().__call__()


def module_scope_import_roots(tree):
    roots = []

    def visit(statements):
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, ast.Import):
                roots.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                roots.append(
                    "<relative-import>"
                    if node.level
                    else (node.module or "").split(".")[0]
                )
            for field in ("body", "orelse", "finalbody"):
                visit(getattr(node, field, ()))
            for handler in getattr(node, "handlers", ()):
                visit(handler.body)

    visit(tree.body)
    return roots


class ConfigErrorMixin:
    def assert_config_error(self, environ, variable, accepted_fragment):
        with self.assertRaises(ValueError) as context:
            parse_integration_config(environ)
        message = str(context.exception)
        self.assertIn(variable, message)
        self.assertIn(accepted_fragment, message)


class ImportSafetyTest(unittest.TestCase):
    def test_module_scope_imports_are_stdlib_only(self):
        tree = ast.parse(HARNESS_PATH.read_text(encoding="utf-8"))
        roots = module_scope_import_roots(tree)
        self.assertTrue(roots)
        for root in roots:
            self.assertIn(root, sys.stdlib_module_names)
        for banned in BANNED_IMPORTS:
            self.assertNotIn(banned, roots)

    def test_fresh_python_b_import_has_no_live_side_effects(self):
        probe = r'''
import json, os, sys
violations = []
write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
def audit(event, args):
    if event == "open":
        path, mode, flags = args
        mode_text = mode if isinstance(mode, str) else ""
        if any(char in mode_text for char in "wax+") or (
            isinstance(flags, int) and flags & write_flags
        ):
            violations.append("filesystem-write:" + str(path))
    elif event in {"os.remove", "os.rename", "os.mkdir", "os.rmdir"}:
        violations.append("filesystem-mutation:" + event)
    elif event.startswith("socket."):
        violations.append("network:" + event)
sys.addaudithook(audit)
import tests.integration_sensor_suite
banned = ["carla", "pygame", "numpy", "sensors", "telemetry",
          "spawn_ego_vehicle", "manual_drive"]
leaked = sorted(name for name in banned if name in sys.modules)
print(json.dumps({"leaked": leaked, "violations": violations}))
'''
        environment = os.environ.copy()
        for name in (
            CARLA_INTEGRATION,
            CARLA_HOST,
            CARLA_PORT,
            CARLA_TIMEOUT_SECONDS,
            CARLA_INTEGRATION_CYCLES,
            CARLA_INTEGRATION_SOAK_SECONDS,
            CARLA_INTEGRATION_TICK_OWNER,
            harness.CARLA_INTEGRATION_MEMORY_STABILITY,
            harness.CARLA_MEMORY_WORKING_SET_MAX_PEAK_GROWTH_BYTES,
            harness.CARLA_MEMORY_WORKING_SET_MAX_FINAL_GROWTH_BYTES,
            harness.CARLA_MEMORY_WORKING_SET_MAX_FINAL_GROWTH_PERCENT,
            harness.CARLA_MEMORY_PRIVATE_COMMIT_MAX_PEAK_GROWTH_BYTES,
            harness.CARLA_MEMORY_PRIVATE_COMMIT_MAX_FINAL_GROWTH_BYTES,
            harness.CARLA_MEMORY_PRIVATE_COMMIT_MAX_FINAL_GROWTH_PERCENT,
            harness.CARLA_MEMORY_WORKING_SET_MAX_SLOPE_BYTES_PER_MINUTE,
            harness.CARLA_MEMORY_PRIVATE_COMMIT_MAX_SLOPE_BYTES_PER_MINUTE,
            harness.CARLA_SERVER_PID,
        ):
            environment.pop(name, None)
        result = subprocess.run(
            [sys.executable, "-B", "-c", probe],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(evidence, {"leaked": [], "violations": []})

    def test_fresh_disabled_unittest_run_has_no_live_side_effects(self):
        probe = r'''
import io, json, os, sys, unittest
violations = []
banned = {"carla", "pygame", "numpy", "sensors", "telemetry",
          "spawn_ego_vehicle", "manual_drive"}
write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
def audit(event, args):
    if event == "open":
        path, mode, flags = args
        mode_text = mode if isinstance(mode, str) else ""
        if any(char in mode_text for char in "wax+") or (
            isinstance(flags, int) and flags & write_flags
        ):
            violations.append("filesystem-write:" + str(path))
    elif event in {"os.remove", "os.rename", "os.mkdir", "os.rmdir"}:
        violations.append("filesystem-mutation:" + event)
    elif event.startswith("socket."):
        violations.append("network:" + event)
    elif event == "import" and args and str(args[0]).split(".")[0] in banned:
        violations.append("live-import:" + str(args[0]))
sys.addaudithook(audit)
suite = unittest.defaultTestLoader.loadTestsFromName(
    "tests.integration_sensor_suite"
)
stream = io.StringIO()
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
leaked = sorted(name for name in banned if name in sys.modules)
print(json.dumps({
    "leaked": leaked,
    "output": stream.getvalue(),
    "skipped": len(result.skipped),
    "successful": result.wasSuccessful(),
    "tests_run": result.testsRun,
    "violations": violations,
}))
'''
        environment = os.environ.copy()
        for name in (
            CARLA_INTEGRATION,
            CARLA_HOST,
            CARLA_PORT,
            CARLA_TIMEOUT_SECONDS,
            CARLA_INTEGRATION_CYCLES,
            CARLA_INTEGRATION_SOAK_SECONDS,
            CARLA_INTEGRATION_TICK_OWNER,
            harness.CARLA_INTEGRATION_MEMORY_STABILITY,
            harness.CARLA_MEMORY_WORKING_SET_MAX_PEAK_GROWTH_BYTES,
            harness.CARLA_MEMORY_WORKING_SET_MAX_FINAL_GROWTH_BYTES,
            harness.CARLA_MEMORY_WORKING_SET_MAX_FINAL_GROWTH_PERCENT,
            harness.CARLA_MEMORY_PRIVATE_COMMIT_MAX_PEAK_GROWTH_BYTES,
            harness.CARLA_MEMORY_PRIVATE_COMMIT_MAX_FINAL_GROWTH_BYTES,
            harness.CARLA_MEMORY_PRIVATE_COMMIT_MAX_FINAL_GROWTH_PERCENT,
            harness.CARLA_MEMORY_WORKING_SET_MAX_SLOPE_BYTES_PER_MINUTE,
            harness.CARLA_MEMORY_PRIVATE_COMMIT_MAX_SLOPE_BYTES_PER_MINUTE,
            harness.CARLA_SERVER_PID,
        ):
            environment.pop(name, None)
        result = subprocess.run(
            [sys.executable, "-B", "-c", probe],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = json.loads(result.stdout.splitlines()[-1])
        self.assertTrue(evidence["successful"], evidence["output"])
        self.assertEqual(evidence["tests_run"], 1)
        self.assertEqual(evidence["skipped"], 1)
        self.assertEqual(evidence["output"].count(DISABLED_MESSAGE), 1)
        self.assertEqual(evidence["leaked"], [])
        self.assertEqual(evidence["violations"], [])

    def test_harness_source_has_no_live_or_memory_collection_calls(self):
        # "Client(" is intentionally no longer forbidden: Mission 4G.2 builds
        # the client inside connect_client, strictly behind the opt-in gate.
        source = HARNESS_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "tracemalloc",
            "gc.collect",
            "socket.",
            "pygame.",
            "time.sleep",
        ):
            self.assertNotIn(forbidden, source)


class OptInTest(unittest.TestCase):
    def test_exact_one_enables(self):
        self.assertTrue(integration_enabled({CARLA_INTEGRATION: "1"}))

    def test_unset_disables(self):
        self.assertFalse(integration_enabled({}))

    def test_empty_disables(self):
        self.assertFalse(integration_enabled({CARLA_INTEGRATION: ""}))

    def test_whitespace_disables_without_trimming(self):
        for value in (" 1", "1 ", " 1 ", "\t1"):
            with self.subTest(value=value):
                self.assertFalse(integration_enabled({CARLA_INTEGRATION: value}))

    def test_true_spellings_disable(self):
        for value in ("true", "TRUE"):
            with self.subTest(value=value):
                self.assertFalse(integration_enabled({CARLA_INTEGRATION: value}))

    def test_zero_padded_one_disables(self):
        self.assertFalse(integration_enabled({CARLA_INTEGRATION: "01"}))

    def test_yes_disables(self):
        self.assertFalse(integration_enabled({CARLA_INTEGRATION: "yes"}))

    def test_injected_mapping_does_not_fall_back_to_process_environment(self):
        with unittest.mock.patch.dict(os.environ, {CARLA_INTEGRATION: "1"}):
            self.assertFalse(integration_enabled({}))

    def test_none_mapping_reads_process_environment(self):
        with unittest.mock.patch.dict(os.environ, {CARLA_INTEGRATION: "1"}, clear=True):
            self.assertTrue(integration_enabled())
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(integration_enabled())


class GateBeforeParseTest(unittest.TestCase):
    def assert_disabled_ignores(self, variable, value):
        with self.assertRaises(unittest.SkipTest) as context:
            require_integration_opt_in(
                {CARLA_INTEGRATION: "0", variable: value}
            )
        self.assertEqual(str(context.exception), DISABLED_MESSAGE)

    def test_disabled_raises_exact_skip_type_and_message(self):
        with self.assertRaises(unittest.SkipTest) as context:
            require_integration_opt_in({})
        self.assertIs(type(context.exception), unittest.SkipTest)
        self.assertEqual(str(context.exception), DISABLED_MESSAGE)

    def test_malformed_host_is_ignored_while_disabled(self):
        self.assert_disabled_ignores(CARLA_HOST, " ")

    def test_malformed_port_is_ignored_while_disabled(self):
        self.assert_disabled_ignores(CARLA_PORT, "not-a-port")

    def test_malformed_timeout_is_ignored_while_disabled(self):
        self.assert_disabled_ignores(CARLA_TIMEOUT_SECONDS, "nan")

    def test_malformed_cycles_are_ignored_while_disabled(self):
        self.assert_disabled_ignores(CARLA_INTEGRATION_CYCLES, "0")

    def test_malformed_soak_is_ignored_while_disabled(self):
        self.assert_disabled_ignores(CARLA_INTEGRATION_SOAK_SECONDS, "1")

    def test_malformed_tick_owner_is_ignored_while_disabled(self):
        self.assert_disabled_ignores(CARLA_INTEGRATION_TICK_OWNER, "true")

    def test_enabled_path_returns_configuration(self):
        config = require_integration_opt_in(
            {CARLA_INTEGRATION: "1", CARLA_PORT: "3000"}
        )
        self.assertEqual(config.port, 3000)


class HostConfigTest(ConfigErrorMixin, unittest.TestCase):
    def test_default_host(self):
        self.assertEqual(parse_integration_config({}).host, "localhost")

    def test_host_is_trimmed(self):
        self.assertEqual(
            parse_integration_config({CARLA_HOST: "  example.local  "}).host,
            "example.local",
        )

    def test_empty_and_whitespace_hosts_are_rejected(self):
        for value in ("", " ", "\t"):
            with self.subTest(value=value):
                self.assert_config_error({CARLA_HOST: value}, CARLA_HOST, "1..255")

    def test_one_character_host_is_accepted(self):
        self.assertEqual(parse_integration_config({CARLA_HOST: "x"}).host, "x")

    def test_255_character_host_is_accepted(self):
        value = "x" * 255
        self.assertEqual(parse_integration_config({CARLA_HOST: value}).host, value)

    def test_256_character_host_is_rejected(self):
        self.assert_config_error({CARLA_HOST: "x" * 256}, CARLA_HOST, "1..255")


class PortConfigTest(ConfigErrorMixin, unittest.TestCase):
    def test_default_port(self):
        self.assertEqual(parse_integration_config({}).port, 2000)

    def test_port_bounds_are_accepted(self):
        self.assertEqual(parse_integration_config({CARLA_PORT: "1"}).port, 1)
        self.assertEqual(parse_integration_config({CARLA_PORT: "65535"}).port, 65535)

    def test_out_of_range_ports_are_rejected(self):
        for value in ("0", "-1", "65536"):
            with self.subTest(value=value):
                self.assert_config_error({CARLA_PORT: value}, CARLA_PORT, "1..65535")

    def test_malformed_empty_and_decimal_ports_are_rejected(self):
        for value in ("abc", "", "2.5"):
            with self.subTest(value=value):
                self.assert_config_error({CARLA_PORT: value}, CARLA_PORT, "1..65535")


class TimeoutConfigTest(ConfigErrorMixin, unittest.TestCase):
    def test_default_timeout(self):
        self.assertEqual(parse_integration_config({}).timeout_seconds, 10.0)

    def test_timeout_bounds_are_accepted(self):
        self.assertEqual(
            parse_integration_config({CARLA_TIMEOUT_SECONDS: "2.0"}).timeout_seconds,
            2.0,
        )
        self.assertEqual(
            parse_integration_config({CARLA_TIMEOUT_SECONDS: "120"}).timeout_seconds,
            120.0,
        )

    def test_zero_and_negative_timeouts_are_rejected(self):
        for value in ("0", "-1"):
            with self.subTest(value=value):
                self.assert_config_error(
                    {CARLA_TIMEOUT_SECONDS: value}, CARLA_TIMEOUT_SECONDS, "2.0..120.0"
                )

    def test_nonfinite_timeouts_are_rejected(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                self.assert_config_error(
                    {CARLA_TIMEOUT_SECONDS: value}, CARLA_TIMEOUT_SECONDS, "2.0..120.0"
                )

    def test_malformed_and_above_range_timeouts_are_rejected(self):
        for value in ("abc", "", "120.1"):
            with self.subTest(value=value):
                self.assert_config_error(
                    {CARLA_TIMEOUT_SECONDS: value}, CARLA_TIMEOUT_SECONDS, "2.0..120.0"
                )


class CycleConfigTest(ConfigErrorMixin, unittest.TestCase):
    def test_default_cycles(self):
        config = parse_integration_config({})
        self.assertEqual(config.cycles, 1)
        self.assertIs(config.cycles_explicit, False)

    def test_cycle_bounds_are_accepted(self):
        self.assertEqual(
            parse_integration_config({CARLA_INTEGRATION_CYCLES: "1"}).cycles, 1
        )
        self.assertEqual(
            parse_integration_config({CARLA_INTEGRATION_CYCLES: "20"}).cycles, 20
        )
        self.assertIs(
            parse_integration_config(
                {CARLA_INTEGRATION_CYCLES: "1"}
            ).cycles_explicit,
            True,
        )

    def test_out_of_range_cycles_are_rejected(self):
        for value in ("0", "-1", "21"):
            with self.subTest(value=value):
                self.assert_config_error(
                    {CARLA_INTEGRATION_CYCLES: value},
                    CARLA_INTEGRATION_CYCLES,
                    "1..20",
                )

    def test_malformed_and_decimal_cycles_are_rejected(self):
        for value in ("abc", "", "2.5"):
            with self.subTest(value=value):
                self.assert_config_error(
                    {CARLA_INTEGRATION_CYCLES: value},
                    CARLA_INTEGRATION_CYCLES,
                    "1..20",
                )


class SoakConfigTest(ConfigErrorMixin, unittest.TestCase):
    def test_default_and_explicit_zero_disable_soak(self):
        self.assertEqual(parse_integration_config({}).soak_seconds, 0.0)
        for value in ("0", "0.0", "-0"):
            with self.subTest(value=value):
                self.assertEqual(
                    parse_integration_config(
                        {CARLA_INTEGRATION_SOAK_SECONDS: value}
                    ).soak_seconds,
                    0.0,
                )

    def test_soak_bounds_are_accepted(self):
        self.assertEqual(
            parse_integration_config(
                {CARLA_INTEGRATION_SOAK_SECONDS: "120"}
            ).soak_seconds,
            120.0,
        )
        self.assertEqual(
            parse_integration_config(
                {CARLA_INTEGRATION_SOAK_SECONDS: "3600"}
            ).soak_seconds,
            3600.0,
        )

    def test_negative_interior_and_above_range_soak_are_rejected(self):
        for value in ("-1", "0.1", "119.999", "3600.1"):
            with self.subTest(value=value):
                self.assert_config_error(
                    {CARLA_INTEGRATION_SOAK_SECONDS: value},
                    CARLA_INTEGRATION_SOAK_SECONDS,
                    "120.0..3600.0",
                )

    def test_nonfinite_and_malformed_soak_are_rejected(self):
        for value in ("nan", "inf", "-inf", "abc", ""):
            with self.subTest(value=value):
                self.assert_config_error(
                    {CARLA_INTEGRATION_SOAK_SECONDS: value},
                    CARLA_INTEGRATION_SOAK_SECONDS,
                    "120.0..3600.0",
                )


class TickOwnerConfigTest(ConfigErrorMixin, unittest.TestCase):
    def test_unset_and_zero_are_false(self):
        self.assertFalse(parse_integration_config({}).tick_owner)
        self.assertFalse(
            parse_integration_config({CARLA_INTEGRATION_TICK_OWNER: "0"}).tick_owner
        )

    def test_one_is_true(self):
        self.assertTrue(
            parse_integration_config({CARLA_INTEGRATION_TICK_OWNER: "1"}).tick_owner
        )

    def test_padded_and_all_other_values_are_rejected(self):
        for value in (" 1", "1 ", "true", "TRUE", "yes", "01", "", "2"):
            with self.subTest(value=value):
                self.assert_config_error(
                    {CARLA_INTEGRATION_TICK_OWNER: value},
                    CARLA_INTEGRATION_TICK_OWNER,
                    "0, 1",
                )


class IntegrationConfigTest(unittest.TestCase):
    def test_config_has_correct_values(self):
        config = parse_integration_config(
            {
                CARLA_HOST: " host ",
                CARLA_PORT: "3000",
                CARLA_TIMEOUT_SECONDS: "5",
                CARLA_INTEGRATION_CYCLES: "3",
                CARLA_INTEGRATION_SOAK_SECONDS: "120",
                CARLA_INTEGRATION_TICK_OWNER: "1",
            }
        )
        self.assertEqual(
            config, IntegrationConfig("host", 3000, 5.0, 3, 120.0, True, True)
        )

    def test_config_is_frozen_and_slotted(self):
        config = IntegrationConfig()
        self.assertFalse(hasattr(config, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            config.port = 9999

    def test_injected_empty_mapping_ignores_process_configuration(self):
        with unittest.mock.patch.dict(
            os.environ, {CARLA_HOST: "process-host", CARLA_PORT: "4000"}
        ):
            self.assertEqual(parse_integration_config({}), IntegrationConfig())


class DeadlineTest(unittest.TestCase):
    def test_start_and_expiry_are_fixed(self):
        clock = FakeClock(10.0)
        deadline = Deadline(5.0, clock)
        self.assertEqual(deadline.start, 10.0)
        self.assertEqual(deadline.expiry, 15.0)
        clock.value = 11.0
        self.assertEqual(deadline.expiry, 15.0)

    def test_remaining_decreases(self):
        clock = FakeClock(10.0)
        deadline = Deadline(5.0, clock)
        clock.value = 11.0
        self.assertEqual(deadline.remaining(), 4.0)
        clock.value = 14.5
        self.assertEqual(deadline.remaining(), 0.5)

    def test_remaining_never_becomes_negative(self):
        clock = FakeClock(0.0)
        deadline = Deadline(1.0, clock)
        clock.value = 50.0
        self.assertEqual(deadline.remaining(), 0.0)

    def test_clock_rollback_is_clamped_and_expiration_cannot_reverse(self):
        clock = FakeClock(10.0)
        deadline = Deadline(5.0, clock)
        clock.value = 12.0
        self.assertEqual(deadline.remaining(), 3.0)
        clock.value = 9.0
        self.assertEqual(deadline.remaining(), 3.0)
        clock.value = 15.0
        self.assertTrue(deadline.expired())
        clock.value = 14.0
        self.assertTrue(deadline.expired())
        self.assertEqual(deadline.remaining(), 0.0)

    def test_raise_if_expired_still_raises_after_rollback(self):
        clock = FakeClock(10.0)
        deadline = Deadline(5.0, clock)
        clock.value = 15.0
        with self.assertRaises(PhaseTimeout):
            deadline.raise_if_expired("first")
        clock.value = 11.0
        with self.assertRaises(PhaseTimeout) as context:
            deadline.raise_if_expired("after_rollback")
        self.assertEqual(context.exception.phase, "after_rollback")

    def test_each_public_operation_reads_clock_once(self):
        clock = CountingClock(0.0)
        deadline = Deadline(5.0, clock)
        self.assertEqual(clock.calls, 1)
        deadline.expired()
        self.assertEqual(clock.calls, 2)
        deadline.remaining()
        self.assertEqual(clock.calls, 3)
        deadline.raise_if_expired("phase")
        self.assertEqual(clock.calls, 4)

    def test_pre_boundary_is_not_expired(self):
        clock = FakeClock(0.0)
        deadline = Deadline(5.0, clock)
        clock.value = math.nextafter(5.0, 0.0)
        self.assertFalse(deadline.expired())

    def test_exact_boundary_is_expired(self):
        clock = FakeClock(0.0)
        deadline = Deadline(5.0, clock)
        clock.value = 5.0
        self.assertTrue(deadline.expired())
        self.assertEqual(deadline.remaining(), 0.0)

    def test_exception_at_boundary_preserves_phase_and_timeout(self):
        clock = FakeClock(0.0)
        deadline = Deadline(5.0, clock)
        clock.value = 5.0
        with self.assertRaises(PhaseTimeout) as context:
            deadline.raise_if_expired("camera_readiness")
        error = context.exception
        self.assertEqual(error.phase, "camera_readiness")
        self.assertEqual(error.timeout_seconds, 5.0)
        self.assertEqual(
            str(error), "phase 'camera_readiness' exceeded 5.000 seconds"
        )

    def test_positive_pre_boundary_does_not_raise(self):
        clock = FakeClock(0.0)
        deadline = Deadline(5.0, clock)
        clock.value = 4.999
        deadline.raise_if_expired("phase")

    def test_invalid_timeout_values_are_rejected(self):
        for value in (
            0,
            -1,
            float("nan"),
            float("inf"),
            float("-inf"),
            True,
            "5",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Deadline(value, FakeClock())

    def test_finite_start_and_timeout_cannot_create_infinite_expiry(self):
        with self.assertRaises(ValueError) as context:
            Deadline(1e308, FakeClock(1e308))
        self.assertIn("expiry", str(context.exception))

    def test_positive_timeout_must_advance_expiry(self):
        with self.assertRaises(ValueError) as context:
            Deadline(1.0, FakeClock(float(2**53)))
        self.assertIn("expiry", str(context.exception))

    def test_nonfinite_clock_is_rejected_at_construction(self):
        for value in (float("nan"), float("inf"), float("-inf"), "0", True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Deadline(1.0, FakeClock(value))

    def test_nonfinite_clock_is_rejected_after_construction(self):
        clock = FakeClock(0.0)
        deadline = Deadline(1.0, clock)
        clock.value = float("nan")
        with self.assertRaises(ValueError):
            deadline.remaining()


class WorldPolicyTest(unittest.TestCase):
    def test_world_mode_values_exist(self):
        self.assertEqual(WorldMode.ASYNCHRONOUS.value, "ASYNCHRONOUS")
        self.assertEqual(WorldMode.SYNCHRONOUS.value, "SYNCHRONOUS")

    def test_async_world_always_passively_waits(self):
        self.assertIs(
            decide_world_progression(False, False), ProgressionPolicy.PASSIVE_WAIT
        )
        self.assertIs(
            decide_world_progression(False, True), ProgressionPolicy.PASSIVE_WAIT
        )
        self.assertIs(
            decide_world_progression(WorldMode.ASYNCHRONOUS, False),
            ProgressionPolicy.PASSIVE_WAIT,
        )

    def test_synchronous_owner_ticks(self):
        self.assertIs(
            decide_world_progression(True, True), ProgressionPolicy.OWNED_TICK
        )
        self.assertIs(
            decide_world_progression(WorldMode.SYNCHRONOUS, True),
            ProgressionPolicy.OWNED_TICK,
        )

    def test_synchronous_non_owner_skips(self):
        self.assertIs(decide_world_progression(True, False), ProgressionPolicy.SKIP)

    def test_non_boolean_inputs_are_rejected(self):
        with self.assertRaises(TypeError):
            decide_world_progression(1, False)


class LiveModuleStructureTest(unittest.TestCase):
    def test_module_name_is_excluded_from_ordinary_discovery_pattern(self):
        self.assertEqual(HARNESS_PATH.name, "integration_sensor_suite.py")
        self.assertFalse(HARNESS_PATH.name.startswith("test_"))

    def test_obsolete_placeholder_is_absent(self):
        self.assertFalse(hasattr(harness, "SensorSuiteLiveIntegrationTest"))
        self.assertFalse(hasattr(harness, "NOT_IMPLEMENTED_MESSAGE"))

    def test_direct_disabled_execution_runs_exactly_one_skipped_test(self):
        environment = os.environ.copy()
        environment.pop(CARLA_INTEGRATION, None)
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "unittest",
                "tests.integration_sensor_suite",
                "-v",
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("Ran 1 test", output)
        self.assertIn("OK (skipped=1)", output)
        self.assertEqual(output.count(DISABLED_MESSAGE), 1)
        self.assertNotIn("not implemented", output.lower())


class ValueObjectTest(unittest.TestCase):
    def test_validation_failure_is_frozen_and_slotted(self):
        failure = ValidationFailure("phase", "message", 2)
        self.assertFalse(hasattr(failure, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            failure.phase = "other"

    def test_cleanup_outcome_is_frozen_and_slotted(self):
        outcome = CleanupOutcome(None, (), CleanupVerification.CONFIRMED)
        self.assertFalse(hasattr(outcome, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            outcome.cleanup_errors = ("error",)


class CleanupTest(unittest.TestCase):
    def assert_no_control_characters(self, text):
        controls = [
            character
            for character in text
            if unicodedata.category(character) == "Cc"
        ]
        self.assertEqual(controls, [])

    def test_diagnostic_constants_have_explicit_values(self):
        self.assertEqual(harness.MAX_PRIMARY_PHASE_LENGTH, 64)
        self.assertEqual(harness.MAX_PRIMARY_MESSAGE_LENGTH, 256)
        self.assertEqual(harness.MAX_CLEANUP_ERROR_LENGTH, 256)
        self.assertEqual(harness.MAX_DISPLAYED_CLEANUP_ERRORS, 8)
        self.assertEqual(harness.MAX_COMPLETE_DIAGNOSTIC_LENGTH, 2048)

    def test_exact_suite_ego_final_order(self):
        calls = []
        outcome = coordinate_cleanup(
            None,
            lambda: calls.append("suite"),
            lambda: calls.append("ego"),
            lambda: (calls.append("final"), CleanupVerification.CONFIRMED)[1],
        )
        self.assertEqual(calls, ["suite", "ego", "final"])
        self.assertIs(outcome.verification, CleanupVerification.CONFIRMED)

    def test_suite_exception_does_not_skip_later_steps(self):
        calls = []

        def suite():
            calls.append("suite")
            raise RuntimeError("suite boom")

        outcome = coordinate_cleanup(
            None,
            suite,
            lambda: calls.append("ego"),
            lambda: (calls.append("final"), CleanupVerification.CONFIRMED)[1],
        )
        self.assertEqual(calls, ["suite", "ego", "final"])
        self.assertEqual(len(outcome.cleanup_errors), 1)
        self.assertIn("suite RuntimeError: suite boom", outcome.cleanup_errors[0])

    def test_ego_exception_does_not_skip_final(self):
        calls = []

        def ego():
            calls.append("ego")
            raise ValueError("ego boom")

        outcome = coordinate_cleanup(
            None,
            lambda: calls.append("suite"),
            ego,
            lambda: (calls.append("final"), CleanupVerification.FAILED_ACTORS_REMAIN)[1],
        )
        self.assertEqual(calls, ["suite", "ego", "final"])
        self.assertIs(
            outcome.verification, CleanupVerification.FAILED_ACTORS_REMAIN
        )

    def test_all_cleanup_errors_are_retained_as_strings(self):
        def fail(message):
            def action():
                raise RuntimeError(message)

            return action

        outcome = coordinate_cleanup(
            None, fail("suite"), fail("ego"), fail("final")
        )
        self.assertEqual(len(outcome.cleanup_errors), 3)
        self.assertTrue(all(type(error) is str for error in outcome.cleanup_errors))
        self.assertIs(
            outcome.verification, CleanupVerification.UNVERIFIED_DISCONNECTED
        )

    def test_primary_failure_is_preserved(self):
        failure = ValidationFailure("camera", "not ready", 4)
        outcome = coordinate_cleanup(
            failure, final_cleanup=lambda: CleanupVerification.CONFIRMED
        )
        self.assertIs(outcome.primary_failure, failure)

    def test_missing_callables_are_safe_and_conservative(self):
        outcome = coordinate_cleanup(None)
        self.assertEqual(outcome.cleanup_errors, ())
        self.assertIs(
            outcome.verification, CleanupVerification.UNVERIFIED_DISCONNECTED
        )

    def test_missing_final_step_never_confirms(self):
        outcome = coordinate_cleanup(None, lambda: None, lambda: None)
        self.assertIs(
            outcome.verification, CleanupVerification.UNVERIFIED_DISCONNECTED
        )

    def test_failed_final_step_never_confirms(self):
        def final():
            raise ConnectionError("disconnected")

        outcome = coordinate_cleanup(None, final_cleanup=final)
        self.assertIs(
            outcome.verification, CleanupVerification.UNVERIFIED_DISCONNECTED
        )
        self.assertEqual(len(outcome.cleanup_errors), 1)

    def test_invalid_final_return_is_an_error_and_unverified(self):
        outcome = coordinate_cleanup(None, final_cleanup=lambda: None)
        self.assertIs(
            outcome.verification, CleanupVerification.UNVERIFIED_DISCONNECTED
        )
        self.assertIn("final TypeError", outcome.cleanup_errors[0])

    def test_all_verification_states_are_represented(self):
        self.assertEqual(
            {state.value for state in CleanupVerification},
            {
                "CONFIRMED",
                "FAILED_ACTORS_REMAIN",
                "UNVERIFIED_DISCONNECTED",
            },
        )

    def test_normal_short_diagnostic_semantics_are_preserved(self):
        outcome = CleanupOutcome(
            ValidationFailure("camera", "no frame", 2),
            ("suite RuntimeError: boom",),
            CleanupVerification.FAILED_ACTORS_REMAIN,
        )
        expected = (
            "primary=cycle 2 camera: no frame; "
            "verification=FAILED_ACTORS_REMAIN; "
            "cleanup_errors=suite RuntimeError: boom"
        )
        self.assertEqual(format_cleanup_outcome(outcome), expected)
        self.assertEqual(format_cleanup_outcome(outcome), expected)

    def test_cleanup_error_strings_are_bounded_and_single_line(self):
        def suite():
            raise RuntimeError("line one\n" + "x" * 1000)

        outcome = coordinate_cleanup(None, suite_cleanup=suite)
        self.assertEqual(len(outcome.cleanup_errors), 1)
        self.assertLessEqual(
            len(outcome.cleanup_errors[0]), harness.MAX_CLEANUP_ERROR_LENGTH
        )
        self.assertNotIn("\n", outcome.cleanup_errors[0])

    def test_displayed_cleanup_errors_stop_at_explicit_limit(self):
        limit = harness.MAX_DISPLAYED_CLEANUP_ERRORS
        errors = tuple(f"[cleanup-{index}]" for index in range(limit + 1))
        formatted = format_cleanup_outcome(
            CleanupOutcome(None, errors, CleanupVerification.CONFIRMED)
        )
        for index in range(limit):
            self.assertIn(f"[cleanup-{index}]", formatted)
        self.assertNotIn(f"[cleanup-{limit}]", formatted)
        self.assertIn("(+1 more)", formatted)

    def test_omitted_cleanup_error_count_is_accurate(self):
        limit = harness.MAX_DISPLAYED_CLEANUP_ERRORS
        errors = tuple(f"error-{index}" for index in range(limit + 37))
        formatted = format_cleanup_outcome(
            CleanupOutcome(None, errors, CleanupVerification.CONFIRMED)
        )
        self.assertIn("(+37 more)", formatted)
        self.assertNotIn("(+36 more)", formatted)

    def test_one_thousand_cleanup_errors_are_bounded_and_counted(self):
        errors = tuple(f"<cleanup-{index:04d}>" for index in range(1000))
        formatted = format_cleanup_outcome(
            CleanupOutcome(None, errors, CleanupVerification.CONFIRMED)
        )
        for index in range(harness.MAX_DISPLAYED_CLEANUP_ERRORS):
            self.assertIn(f"<cleanup-{index:04d}>", formatted)
        self.assertNotIn("<cleanup-0008>", formatted)
        self.assertNotIn("<cleanup-0999>", formatted)
        self.assertTrue(formatted.endswith("(+992 more)"))
        self.assertLessEqual(
            len(formatted), harness.MAX_COMPLETE_DIAGNOSTIC_LENGTH
        )

    def test_nul_and_escape_are_sanitized(self):
        outcome = CleanupOutcome(
            ValidationFailure("cam\0era", "bad\x1b[31m frame"),
            ("suite\0error\x1b",),
            CleanupVerification.CONFIRMED,
        )
        formatted = format_cleanup_outcome(outcome)
        self.assert_no_control_characters(formatted)
        self.assertIn("primary=cam era: bad [31m frame", formatted)
        self.assertIn("cleanup_errors=suite error", formatted)

    def test_carriage_return_line_feed_and_tab_are_sanitized(self):
        outcome = CleanupOutcome(
            ValidationFailure("camera\r\n\treadiness", "no\r\n\tframe"),
            ("suite\r\n\tcleanup failed",),
            CleanupVerification.FAILED_ACTORS_REMAIN,
        )
        formatted = format_cleanup_outcome(outcome)
        self.assert_no_control_characters(formatted)
        self.assertIn("primary=camera readiness: no frame", formatted)
        self.assertIn("cleanup_errors=suite cleanup failed", formatted)

    def test_every_unicode_cc_character_is_sanitized(self):
        controls = "".join(
            chr(codepoint)
            for codepoint in (*range(32), *range(127, 160))
        )
        self.assertTrue(
            all(unicodedata.category(character) == "Cc" for character in controls)
        )
        formatted = format_cleanup_outcome(
            CleanupOutcome(
                ValidationFailure(
                    f"left{controls}right",
                    f"before{controls}after",
                ),
                (f"suite{controls}failed",),
                CleanupVerification.CONFIRMED,
            )
        )
        self.assert_no_control_characters(formatted)
        self.assertIn("primary=left right: before after", formatted)
        self.assertIn("cleanup_errors=suite failed", formatted)

    def test_very_long_primary_phase_and_message_use_separate_bounds(self):
        formatted = format_cleanup_outcome(
            CleanupOutcome(
                ValidationFailure("p" * 10_000, "m" * 10_000),
                (),
                CleanupVerification.CONFIRMED,
            )
        )
        bounded_phase = "p" * (harness.MAX_PRIMARY_PHASE_LENGTH - 3) + "..."
        bounded_message = "m" * (harness.MAX_PRIMARY_MESSAGE_LENGTH - 3) + "..."
        self.assertIn(
            f"primary={bounded_phase}: {bounded_message}; verification=",
            formatted,
        )

    def test_unprintable_exception_cannot_skip_later_cleanup(self):
        calls = []

        class UnprintableError(Exception):
            def __str__(self):
                raise RuntimeError("string conversion failed")

        def suite():
            calls.append("suite")
            raise UnprintableError()

        outcome = coordinate_cleanup(
            None,
            suite,
            lambda: calls.append("ego"),
            lambda: (calls.append("final"), CleanupVerification.CONFIRMED)[1],
        )
        self.assertEqual(calls, ["suite", "ego", "final"])
        self.assertIs(outcome.verification, CleanupVerification.CONFIRMED)
        self.assertEqual(len(outcome.cleanup_errors), 1)
        self.assertIn("suite UnprintableError", outcome.cleanup_errors[0])
        self.assertIn("unprintable: RuntimeError", outcome.cleanup_errors[0])

    def test_formatter_reaches_but_never_exceeds_exact_output_bound(self):
        outcome = CleanupOutcome(
            ValidationFailure("p" * 10_000, "m" * 10_000),
            tuple(
                "x" * 10_000
                for _ in range(harness.MAX_DISPLAYED_CLEANUP_ERRORS)
            ),
            CleanupVerification.CONFIRMED,
        )
        formatted = format_cleanup_outcome(outcome)
        self.assertEqual(len(formatted), harness.MAX_COMPLETE_DIAGNOSTIC_LENGTH)
        self.assertTrue(formatted.endswith("..."))


class CycleStateTest(unittest.TestCase):
    def test_new_state_has_exact_empty_defaults(self):
        state = new_cycle_state(3)
        self.assertEqual(state.cycle_number, 3)
        self.assertIsNone(state.ego_id)
        self.assertEqual(state.sensor_ids, ())
        self.assertIsNone(state.snapshot_reference)
        self.assertIsNone(state.camera_reference)
        self.assertIsNone(state.primary_failure)
        self.assertEqual(state.cleanup_errors, ())

    def test_cycle_states_are_independent(self):
        first = new_cycle_state(1)
        second = new_cycle_state(2)
        first.ego_id = 99
        first.sensor_ids += (100,)
        first.cleanup_errors += ("error",)
        self.assertIsNone(second.ego_id)
        self.assertEqual(second.sensor_ids, ())
        self.assertEqual(second.cleanup_errors, ())

    def test_cycle_state_is_slotted_and_has_no_mutable_defaults(self):
        state = new_cycle_state(1)
        self.assertFalse(hasattr(state, "__dict__"))
        self.assertIs(type(state.sensor_ids), tuple)
        self.assertIs(type(state.cleanup_errors), tuple)


class MemoryGrowthTest(unittest.TestCase):
    def evaluate(self, baseline, samples, fixed, slope):
        return evaluate_memory_growth(baseline, samples, fixed, slope)

    def test_stable_memory_passes(self):
        result = self.evaluate(
            1000,
            (MemorySample(0.0, 1000), MemorySample(60.0, 1000)),
            0,
            0.0,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, "pass")
        self.assertEqual(result.final_delta_bytes, 0)
        self.assertEqual(result.maximum_delta_bytes, 0)
        self.assertEqual(result.slope_bytes_per_minute, 0.0)

    def test_large_counters_with_zero_growth_pass_zero_slope_allowance(self):
        baseline = 2**200
        result = self.evaluate(
            baseline,
            (
                MemorySample(0.0, baseline),
                MemorySample(60.0, baseline),
                MemorySample(120.0, baseline),
            ),
            0,
            0.0,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, "pass")
        self.assertEqual(result.final_delta_bytes, 0)
        self.assertEqual(result.maximum_delta_bytes, 0)
        self.assertEqual(result.slope_bytes_per_minute, 0.0)

    def test_bounded_noise_passes(self):
        result = self.evaluate(
            1000,
            (
                MemorySample(0.0, 1000),
                MemorySample(60.0, 1010),
                MemorySample(120.0, 1005),
            ),
            10,
            3.0,
        )
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.slope_bytes_per_minute, 2.5)

    def test_large_absolute_counters_preserve_small_growth(self):
        baseline = 2**60
        result = self.evaluate(
            baseline,
            (
                MemorySample(0.0, baseline),
                MemorySample(60.0, baseline + 60),
            ),
            60,
            0.0,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "slope allowance exceeded")
        self.assertEqual(result.slope_bytes_per_minute, 60.0)

    def test_large_counter_exact_slope_threshold_passes(self):
        baseline = 2**60
        result = self.evaluate(
            baseline,
            (
                MemorySample(0.0, baseline),
                MemorySample(60.0, baseline + 60),
            ),
            60,
            60.0,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.slope_bytes_per_minute, 60.0)

    def test_extremely_close_timestamps_do_not_underflow_denominator(self):
        result = self.evaluate(
            0,
            (
                MemorySample(0.0, 0),
                MemorySample(math.nextafter(0.0, 1.0), 1),
            ),
            1,
            0.0,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "slope allowance exceeded")
        self.assertEqual(result.slope_bytes_per_minute, math.inf)

    def test_nonzero_and_large_timestamp_origins_are_stable(self):
        ordinary = self.evaluate(
            1000,
            (
                MemorySample(1_000_000_000.0, 1000),
                MemorySample(1_000_000_060.0, 1060),
            ),
            60,
            60.0,
        )
        self.assertTrue(ordinary.passed)
        self.assertEqual(ordinary.slope_bytes_per_minute, 60.0)

        start = 1e300
        large = self.evaluate(
            0,
            (
                MemorySample(start, 0),
                MemorySample(math.nextafter(start, math.inf), 1),
            ),
            1,
            1.0,
        )
        self.assertTrue(large.passed)
        self.assertGreater(large.slope_bytes_per_minute, 0.0)

    def test_decreasing_values_preserve_negative_deltas(self):
        result = self.evaluate(
            1000,
            (MemorySample(0.0, 900), MemorySample(60.0, 800)),
            0,
            0.0,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.final_delta_bytes, -200)
        self.assertEqual(result.maximum_delta_bytes, -100)
        self.assertEqual(result.slope_bytes_per_minute, -100.0)

    def test_exact_fixed_threshold_passes(self):
        result = self.evaluate(
            1000,
            (MemorySample(0.0, 1100), MemorySample(60.0, 1100)),
            100,
            0.0,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.maximum_delta_bytes, 100)

    def test_exact_slope_threshold_passes_with_tolerance(self):
        result = self.evaluate(
            1000,
            (MemorySample(0.0, 1000), MemorySample(60.0, 1060)),
            100,
            60.0,
        )
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.slope_bytes_per_minute, 60.0, places=9)

    def test_fixed_threshold_failure(self):
        result = self.evaluate(
            1000,
            (MemorySample(0.0, 1101), MemorySample(60.0, 1101)),
            100,
            0.0,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "fixed growth allowance exceeded")

    def test_slope_failure(self):
        result = self.evaluate(
            1000,
            (MemorySample(0.0, 1000), MemorySample(60.0, 1061)),
            100,
            60.0,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "slope allowance exceeded")

    def test_both_thresholds_failure(self):
        result = self.evaluate(
            1000,
            (MemorySample(0.0, 1000), MemorySample(60.0, 1200)),
            100,
            100.0,
        )
        self.assertFalse(result.passed)
        self.assertEqual(
            result.reason, "fixed growth allowance and slope allowance exceeded"
        )

    def test_decreasing_and_duplicate_timestamps_are_rejected(self):
        for samples in (
            (MemorySample(1.0, 1), MemorySample(0.0, 1)),
            (MemorySample(1.0, 1), MemorySample(1.0, 1)),
        ):
            with self.subTest(samples=samples):
                with self.assertRaises(ValueError):
                    self.evaluate(0, samples, 0, 0.0)

    def test_nonfinite_timestamps_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf"), "1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.evaluate(
                        0,
                        (MemorySample(0.0, 1), MemorySample(value, 1)),
                        0,
                        0.0,
                    )

    def test_invalid_sample_bytes_are_rejected(self):
        for value in (-1, 1.5, "1", True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.evaluate(
                        0,
                        (MemorySample(0.0, 1), MemorySample(1.0, value)),
                        0,
                        0.0,
                    )

    def test_invalid_baselines_are_rejected(self):
        for value in (-1, 1.5, "1", True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.evaluate(
                        value,
                        (MemorySample(0.0, 1), MemorySample(1.0, 1)),
                        0,
                        0.0,
                    )

    def test_insufficient_samples_are_rejected(self):
        for samples in ((), (MemorySample(0.0, 1),)):
            with self.subTest(samples=samples):
                with self.assertRaises(ValueError):
                    self.evaluate(0, samples, 0, 0.0)

    def test_invalid_fixed_allowances_are_rejected(self):
        for value in (-1, 1.5, "1", True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.evaluate(
                        0,
                        (MemorySample(0.0, 1), MemorySample(1.0, 1)),
                        value,
                        0.0,
                    )

    def test_invalid_slope_allowances_are_rejected(self):
        for value in (
            -1,
            float("nan"),
            float("inf"),
            float("-inf"),
            "x",
            "1",
            True,
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.evaluate(
                        0,
                        (MemorySample(0.0, 1), MemorySample(1.0, 1)),
                        0,
                        value,
                    )

    def test_result_and_sample_are_frozen_slotted_value_objects(self):
        sample = MemorySample(0.0, 1)
        result = MemoryGrowthResult(True, 0, 0, 0.0, "pass")
        self.assertFalse(hasattr(sample, "__dict__"))
        self.assertFalse(hasattr(result, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            sample.current_bytes = 2
        with self.assertRaises(FrozenInstanceError):
            result.passed = False


class ExpectedSensorsTest(unittest.TestCase):
    def test_expected_sensor_order_is_immutable_and_exact(self):
        self.assertEqual(
            EXPECTED_SENSOR_NAMES,
            ("gnss", "imu", "collision", "lane_invasion", "rgb_camera"),
        )
        self.assertIs(type(EXPECTED_SENSOR_NAMES), tuple)


# -- Mission 4G.2: strict live-path fakes (no CARLA, no network) ---------------


class SequenceClock:
    def __init__(self, *values):
        self.values = list(values)
        self.last = values[-1] if values else 0.0
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.values:
            self.last = self.values.pop(0)
        return self.last


class StrictSettings:
    def __init__(self, synchronous_mode=False):
        object.__setattr__(self, "synchronous_mode", synchronous_mode)

    def __setattr__(self, name, value):
        raise AssertionError(f"world setting mutation is forbidden: {name}={value!r}")


class StrictBlueprint:
    def __init__(self, calls, blueprint_id="vehicle.tesla.model3", role=True):
        self.calls = calls
        self.id = blueprint_id
        self.attributes = {}
        self.supports_role_name = role

    def has_attribute(self, name):
        self.calls.append(("blueprint.has_attribute", name))
        return name == "role_name" and self.supports_role_name

    def set_attribute(self, name, value):
        self.calls.append(("blueprint.set_attribute", name, value))
        if name != "role_name":
            raise AssertionError(f"unexpected blueprint attribute: {name}")
        self.attributes[name] = value


class StrictBlueprintLibrary:
    def __init__(self, calls):
        self.calls = calls

    def filter(self, pattern):
        raise AssertionError(
            f"harness duplicated blueprint selection with filter({pattern!r})"
        )


class StrictMap:
    def __init__(self, calls, spawn_points):
        self.calls = calls
        self.spawn_points = tuple(spawn_points)

    def get_spawn_points(self):
        self.calls.append(("get_spawn_points",))
        return list(self.spawn_points)


class StrictActor:
    def __init__(
        self,
        calls=None,
        *,
        actor_id=101,
        type_id="vehicle.tesla.model3",
        role_name="hero",
        id_error=None,
        type_error=None,
        attributes_error=None,
        is_alive=False,
        is_alive_error=None,
        destroy_result=True,
        destroy_error=None,
        remove_even_on_failure=False,
        falsey=False,
    ):
        self.calls = [] if calls is None else calls
        self._actor_id = actor_id
        self._type_id = type_id
        self._attributes = {"role_name": role_name}
        self._id_error = id_error
        self._type_error = type_error
        self._attributes_error = attributes_error
        self._is_alive = is_alive
        self._is_alive_error = is_alive_error
        self._destroy_result = destroy_result
        self._destroy_error = destroy_error
        self.remove_even_on_failure = remove_even_on_failure
        self.falsey = falsey
        self.world = None
        self.id_reads = 0
        self.type_reads = 0
        self.attributes_reads = 0
        self.is_alive_reads = 0
        self.identity_read_order = []
        self.destroy_calls = 0
        self.control_calls = 0

    def __bool__(self):
        return not self.falsey

    @property
    def id(self):
        self.id_reads += 1
        self.identity_read_order.append("id")
        if self._id_error is not None:
            raise self._id_error
        return self._actor_id

    @property
    def type_id(self):
        self.type_reads += 1
        self.identity_read_order.append("type_id")
        if self._type_error is not None:
            raise self._type_error
        return self._type_id

    @property
    def attributes(self):
        self.attributes_reads += 1
        self.identity_read_order.append("attributes")
        if self._attributes_error is not None:
            raise self._attributes_error
        return self._attributes

    @property
    def is_alive(self):
        self.is_alive_reads += 1
        self.identity_read_order.append("is_alive")
        if self._is_alive_error is not None:
            raise self._is_alive_error
        return self._is_alive

    def destroy(self):
        self.destroy_calls += 1
        self.calls.append(("actor.destroy", self._actor_id))
        if self._destroy_error is not None:
            raise self._destroy_error
        if self.world is not None and (
            self._destroy_result is True or self.remove_even_on_failure
        ):
            self.world.registry.pop(self._actor_id, None)
        return self._destroy_result

    def apply_control(self, _control):
        self.control_calls += 1
        raise AssertionError("the harness must not control any actor")


class StrictWorldSnapshot:
    """One frame's frozen actor membership, queried only via has_actor."""

    def __init__(self, world, present_ids):
        self._world = world
        self._present_ids = frozenset(present_ids)

    def has_actor(self, actor_id):
        self._world.calls.append(("snapshot.has_actor", actor_id))
        self._world.snapshot_queried_ids.append(actor_id)
        return actor_id in self._present_ids


class StrictWorld:
    def __init__(
        self,
        calls=None,
        *,
        synchronous=False,
        blueprint_library=None,
        spawn_points=("sp0",),
        spawn_plan=(),
        lookup_plan=None,
        snapshot_plan=None,
        progression_error=None,
        remove_after_progressions=0,
        remove_actor_id=101,
        on_progress=None,
    ):
        self.calls = [] if calls is None else calls
        self.settings = StrictSettings(synchronous)
        self.blueprint_library = (
            StrictBlueprintLibrary(self.calls)
            if blueprint_library is None
            else blueprint_library
        )
        self.map = StrictMap(self.calls, spawn_points)
        self.spawn_plan = list(spawn_plan)
        self.lookup_plan = None if lookup_plan is None else list(lookup_plan)
        self.snapshot_plan = None if snapshot_plan is None else list(snapshot_plan)
        self.progression_error = progression_error
        self.remove_after_progressions = remove_after_progressions
        self.remove_actor_id = remove_actor_id
        self.on_progress = on_progress
        self.registry = {}
        self.spawned = []
        self.queried_ids = []
        self.snapshot_queried_ids = []
        self.progressions = 0

    def get_settings(self):
        self.calls.append(("get_settings",))
        return self.settings

    def get_blueprint_library(self):
        self.calls.append(("get_blueprint_library",))
        return self.blueprint_library

    def get_map(self):
        self.calls.append(("get_map",))
        return self.map

    def try_spawn_actor(self, blueprint, spawn_point):
        self.calls.append(("try_spawn_actor", blueprint, spawn_point))
        planned = self.spawn_plan.pop(0) if self.spawn_plan else None
        if isinstance(planned, BaseException):
            raise planned
        if planned is None:
            return None
        planned.world = self
        if type(planned._actor_id) is int and planned._actor_id > 0:
            self.registry[planned._actor_id] = planned
        self.spawned.append(planned)
        return planned

    def get_actor(self, actor_id):
        self.calls.append(("get_actor", actor_id))
        self.queried_ids.append(actor_id)
        if self.lookup_plan is not None and self.lookup_plan:
            planned = self.lookup_plan.pop(0)
            if isinstance(planned, BaseException):
                raise planned
            return planned
        return self.registry.get(actor_id)

    def get_snapshot(self):
        self.calls.append(("get_snapshot",))
        if self.snapshot_plan is not None and self.snapshot_plan:
            planned = self.snapshot_plan.pop(0)
            if isinstance(planned, BaseException):
                raise planned
            return StrictWorldSnapshot(self, planned)
        return StrictWorldSnapshot(self, self.registry)

    def _progress(self, method, timeout):
        self.calls.append((method, timeout))
        if self.progression_error is not None:
            raise self.progression_error
        self.progressions += 1
        if self.on_progress is not None:
            self.on_progress()
        if (
            self.remove_after_progressions
            and self.progressions >= self.remove_after_progressions
        ):
            self.registry.pop(self.remove_actor_id, None)
        return f"{method}-result"

    def tick(self, timeout):
        return self._progress("tick", timeout)

    def wait_for_tick(self, timeout):
        return self._progress("wait_for_tick", timeout)

    def get_actors(self):
        raise AssertionError("actor enumeration is forbidden")

    def apply_settings(self, _settings):
        raise AssertionError("world settings mutation is forbidden")

    def load_world(self, *_args):
        raise AssertionError("load_world is forbidden")

    def reload_world(self, *_args):
        raise AssertionError("reload_world is forbidden")

    def get_spectator(self):
        raise AssertionError("spectator access is forbidden")


class StrictClient:
    def __init__(
        self,
        calls,
        world,
        *,
        client_version="0.9.16",
        server_version="0.9.16",
        errors=None,
    ):
        self.calls = calls
        self.world = world
        self.client_version = client_version
        self.server_version = server_version
        self.errors = {} if errors is None else dict(errors)

    def _raise(self, phase):
        error = self.errors.get(phase)
        if error is not None:
            raise error

    def set_timeout(self, seconds):
        self.calls.append(("set_timeout", seconds))
        self._raise("set_timeout")

    def get_client_version(self):
        self.calls.append(("get_client_version",))
        self._raise("get_client_version")
        return self.client_version

    def get_server_version(self):
        self.calls.append(("get_server_version",))
        self._raise("get_server_version")
        return self.server_version

    def get_world(self):
        self.calls.append(("get_world",))
        self._raise("get_world")
        return self.world

    def load_world(self, *_args):
        raise AssertionError("client.load_world is forbidden")

    def reload_world(self, *_args):
        raise AssertionError("client.reload_world is forbidden")


class StrictCarlaModule:
    def __init__(self, calls, client, error=None):
        self.calls = calls
        self.client = client
        self.error = error
        self.client_calls = 0

    def Client(self, host, port):
        self.client_calls += 1
        self.calls.append(("Client", host, port))
        if self.error is not None:
            raise self.error
        return self.client


class ExactPicker:
    def __init__(self, calls, blueprint, error=None):
        self.calls = calls
        self.blueprint = blueprint
        self.error = error
        self.arguments = []

    def __call__(self, blueprint_library):
        self.calls.append(("pick_vehicle_blueprint", blueprint_library))
        self.arguments.append(blueprint_library)
        if self.error is not None:
            raise self.error
        return self.blueprint


class StrictSpawnModule:
    def __init__(self, picker):
        self.pick_vehicle_blueprint = picker


class RecordingImporter:
    def __init__(self, modules=None, errors=None, calls=None):
        self.modules = {} if modules is None else dict(modules)
        self.errors = {} if errors is None else dict(errors)
        self.requests = []
        self.calls = calls

    def __call__(self, name):
        self.requests.append(name)
        if self.calls is not None:
            self.calls.append(("import", name))
        if name in self.errors:
            raise self.errors[name]
        if name not in self.modules:
            raise ImportError(f"no injected module named {name}")
        return self.modules[name]


class LiveRig:
    def __init__(
        self,
        *,
        synchronous=False,
        actor=None,
        role_supported=True,
        spawn_points=("sp0",),
        spawn_plan=None,
        client_version="0.9.16",
        server_version="0.9.16",
        client_errors=None,
        client_construction_error=None,
        world_kwargs=None,
    ):
        self.calls = []
        self.actor = (
            StrictActor(
                self.calls,
                role_name="hero" if role_supported else None,
            )
            if actor is None
            else actor
        )
        self.actor.calls = self.calls
        self.library = StrictBlueprintLibrary(self.calls)
        self.blueprint = StrictBlueprint(self.calls, role=role_supported)
        kwargs = {} if world_kwargs is None else dict(world_kwargs)
        self.world = StrictWorld(
            self.calls,
            synchronous=synchronous,
            blueprint_library=self.library,
            spawn_points=spawn_points,
            spawn_plan=[self.actor] if spawn_plan is None else spawn_plan,
            **kwargs,
        )
        self.client = StrictClient(
            self.calls,
            self.world,
            client_version=client_version,
            server_version=server_version,
            errors=client_errors,
        )
        self.carla = StrictCarlaModule(
            self.calls, self.client, error=client_construction_error
        )
        self.picker = ExactPicker(self.calls, self.blueprint)
        self.spawn_module = StrictSpawnModule(self.picker)
        self.importer = RecordingImporter(
            {"carla": self.carla, "spawn_ego_vehicle": self.spawn_module},
            calls=self.calls,
        )


ENABLED_ENVIRON = {CARLA_INTEGRATION: "1"}


def run_cycle(rig, environ=None, clock=None, timeout=5.0):
    return run_connection_spawn_cycle(
        dict(ENABLED_ENVIRON) if environ is None else dict(environ),
        importer=rig.importer,
        monotonic=clock if clock is not None else FakeClock(0.0),
        disappearance_timeout_seconds=timeout,
    )


class LiveDependencyLoaderTest(unittest.TestCase):
    def test_disabled_skips_before_importing(self):
        importer = RecordingImporter()
        with self.assertRaises(unittest.SkipTest) as context:
            load_live_dependencies({}, importer=importer)
        self.assertEqual(str(context.exception), DISABLED_MESSAGE)
        self.assertEqual(importer.requests, [])

    def test_near_miss_opt_in_skips_before_importing(self):
        for value in ("01", "true", " 1", "1 ", ""):
            with self.subTest(value=value):
                importer = RecordingImporter()
                with self.assertRaises(unittest.SkipTest):
                    load_live_dependencies(
                        {CARLA_INTEGRATION: value}, importer=importer
                    )
                self.assertEqual(importer.requests, [])

    def test_enabled_imports_expected_names_and_exposes_exact_picker(self):
        rig = LiveRig()
        dependencies = load_live_dependencies(
            ENABLED_ENVIRON, importer=rig.importer
        )
        self.assertEqual(rig.importer.requests, ["carla", "spawn_ego_vehicle"])
        self.assertIs(dependencies.carla, rig.carla)
        self.assertIs(dependencies.pick_vehicle_blueprint, rig.picker)

    def test_dependency_loading_does_not_construct_a_client(self):
        rig = LiveRig()
        load_live_dependencies(ENABLED_ENVIRON, importer=rig.importer)
        self.assertEqual(rig.carla.client_calls, 0)
        self.assertEqual(
            rig.calls,
            [("import", "carla"), ("import", "spawn_ego_vehicle")],
        )

    def test_import_failures_are_bounded_and_preserve_the_cause(self):
        for name in ("carla", "spawn_ego_vehicle"):
            with self.subTest(name=name):
                error = ImportError("bad\r\n" + "x" * 1000)
                modules = {"carla": object(), "spawn_ego_vehicle": object()}
                importer = RecordingImporter(modules, {name: error})
                with self.assertRaises(RuntimeError) as context:
                    load_live_dependencies(ENABLED_ENVIRON, importer=importer)
                self.assertIs(context.exception.__cause__, error)
                self.assertLessEqual(
                    len(str(context.exception)), harness.MAX_PRIMARY_MESSAGE_LENGTH
                )
                self.assertNotIn("\n", str(context.exception))
                self.assertIn(name, str(context.exception))

    def test_missing_or_noncallable_picker_is_a_bounded_error(self):
        for spawn_module in (object(), StrictSpawnModule(None)):
            with self.subTest(spawn_module=spawn_module):
                importer = RecordingImporter(
                    {"carla": object(), "spawn_ego_vehicle": spawn_module}
                )
                with self.assertRaises(RuntimeError) as context:
                    load_live_dependencies(ENABLED_ENVIRON, importer=importer)
                self.assertIsInstance(context.exception.__cause__, (AttributeError, TypeError))
                self.assertIn("pick_vehicle_blueprint", str(context.exception))

    def test_dependencies_record_is_frozen(self):
        dependencies = LiveDependencies(carla=object(), pick_vehicle_blueprint=lambda _: None)
        with self.assertRaises(FrozenInstanceError):
            dependencies.carla = None


class VersionCheckTest(unittest.TestCase):
    def test_exact_and_ordinary_suffix_versions_are_parsed(self):
        self.assertEqual(parse_carla_version("0.9.16"), CarlaVersion(0, 9, 16))
        for value in (
            "0.9.16-dirty",
            "0.9.16-1-g8854e62",
            "0.9.16+build.7",
            "0.9.16+windows-x64",
        ):
            with self.subTest(value=value):
                parsed = parse_carla_version(value)
                self.assertEqual((parsed.major, parsed.minor, parsed.patch), (0, 9, 16))
                self.assertEqual(parsed.suffix, value[len("0.9.16"):])

    def test_anchored_parser_rejects_malformed_and_boundary_values(self):
        malformed = (
            "0.9.16.1",
            "0.9",
            "0.9.16foo",
            "0.9.16-",
            "0.9.16+",
            "0.9.16\n",
            " 0.9.16",
            "0.9.16 ",
            "00.9.16",
            "٠.٩.١٦",
            "",
            "   ",
            "malformed",
            None,
            916,
            object(),
        )
        for value in malformed:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_carla_version(value)

    def test_client_and_server_are_independently_required_to_be_0916(self):
        accepted = ("0.9.16", "0.9.16-dirty", "0.9.16+build.1")
        for client_version in accepted:
            for server_version in accepted:
                with self.subTest(client=client_version, server=server_version):
                    self.assertTrue(
                        versions_compatible(client_version, server_version)
                    )
        self.assertFalse(versions_compatible("0.9.15", "0.9.16"))
        self.assertFalse(versions_compatible("0.9.16", "0.9.17"))
        self.assertFalse(versions_compatible("0.9.15", "0.9.15"))
        self.assertFalse(versions_compatible("0.9.160", "0.9.160"))
        for bad_version in ("0.9.15", "0.9.17", "0.9.160"):
            with self.subTest(bad_client=bad_version):
                self.assertFalse(versions_compatible(bad_version, "0.9.16"))
            with self.subTest(bad_server=bad_version):
                self.assertFalse(versions_compatible("0.9.16", bad_version))

    def test_verify_versions_accepts_suffixes_and_returns_frozen_report(self):
        rig = LiveRig(
            client_version="0.9.16-1-g8854e62",
            server_version="0.9.16+server.4",
        )
        report = verify_versions(rig.client)
        self.assertEqual(report.client_version, "0.9.16-1-g8854e62")
        self.assertEqual(report.server_version, "0.9.16+server.4")
        self.assertTrue(report.compatible)
        with self.assertRaises(FrozenInstanceError):
            report.compatible = False

    def test_valid_long_suffix_is_not_truncated_in_the_verified_report(self):
        version = "0.9.16+" + "build" * 100
        rig = LiveRig(client_version=version, server_version=version)
        report = verify_versions(rig.client)
        self.assertEqual(report.client_version, version)
        self.assertEqual(report.server_version, version)
        self.assertTrue(report.compatible)

    def test_client_only_server_only_and_both_mismatches_are_named(self):
        cases = (
            ("0.9.15", "0.9.16", ("client version",)),
            ("0.9.16", "0.9.17", ("server version",)),
            ("0.9.15", "0.9.15", ("client version", "server version")),
        )
        for client_version, server_version, fragments in cases:
            with self.subTest(client=client_version, server=server_version):
                rig = LiveRig(
                    client_version=client_version,
                    server_version=server_version,
                )
                with self.assertRaises(RuntimeError) as context:
                    verify_versions(rig.client)
                message = str(context.exception)
                self.assertIn("version_validation", message)
                self.assertIn(REQUIRED_CARLA_VERSION, message)
                for fragment in fragments:
                    self.assertIn(fragment, message)
                self.assertIn(client_version, message)
                self.assertIn(server_version, message)
                self.assertEqual(
                    [call[0] for call in rig.calls],
                    ["get_client_version", "get_server_version"],
                )

    def test_malformed_control_and_long_values_have_safe_bounded_diagnostics(self):
        rig = LiveRig(
            client_version="bad\r\n\x1b" + "x" * 1000,
            server_version=None,
        )
        with self.assertRaises(RuntimeError) as context:
            verify_versions(rig.client)
        message = str(context.exception)
        self.assertLessEqual(len(message), harness.MAX_PRIMARY_MESSAGE_LENGTH)
        self.assertNotIn("x" * harness.VERSION_TEXT_LIMIT, message)
        self.assertFalse(
            any(unicodedata.category(character) == "Cc" for character in message)
        )

    def test_version_query_failures_are_phase_specific_and_keep_causes(self):
        cases = (
            ("get_client_version", "client_version_query"),
            ("get_server_version", "server_version_query"),
        )
        for method, phase in cases:
            with self.subTest(method=method):
                error = RuntimeError("rpc\r\nfailed")
                rig = LiveRig(client_errors={method: error})
                with self.assertRaises(RuntimeError) as context:
                    verify_versions(rig.client)
                self.assertIs(context.exception.__cause__, error)
                self.assertIn(phase, str(context.exception))
                self.assertNotIn("\n", str(context.exception))


class ConnectClientTest(unittest.TestCase):
    def test_client_uses_config_and_sets_timeout_first(self):
        rig = LiveRig()
        config = IntegrationConfig(host="h", port=4000, timeout_seconds=7.0)
        returned = connect_client(rig.carla, config)
        self.assertIs(returned, rig.client)
        self.assertEqual(
            rig.calls,
            [("Client", "h", 4000), ("set_timeout", 7.0)],
        )

    def test_constructor_timeout_and_world_failures_are_phase_specific(self):
        cases = (
            ("constructor", "client_construction"),
            ("set_timeout", "timeout_configuration"),
            ("get_world", "world_acquisition"),
        )
        for failing_step, phase in cases:
            with self.subTest(failing_step=failing_step):
                error = RuntimeError("boom\n" + "x" * 1000)
                rig = LiveRig(
                    client_construction_error=(
                        error if failing_step == "constructor" else None
                    ),
                    client_errors=(
                        {failing_step: error}
                        if failing_step in {"set_timeout", "get_world"}
                        else None
                    ),
                )
                with self.assertRaises(RuntimeError) as context:
                    client = connect_client(rig.carla, IntegrationConfig())
                    acquire_world(client)
                self.assertIs(context.exception.__cause__, error)
                self.assertIn(phase, str(context.exception))
                self.assertLessEqual(
                    len(str(context.exception)), harness.MAX_PRIMARY_MESSAGE_LENGTH
                )
                self.assertNotIn("\n", str(context.exception))


class WorldModeInspectionTest(unittest.TestCase):
    def test_asynchronous_mode_is_reported(self):
        self.assertIs(
            inspect_world_mode(StrictWorld(synchronous=False)),
            WorldMode.ASYNCHRONOUS,
        )

    def test_synchronous_mode_is_reported(self):
        self.assertIs(
            inspect_world_mode(StrictWorld(synchronous=True)),
            WorldMode.SYNCHRONOUS,
        )

    def test_inspection_is_read_only(self):
        world = StrictWorld(synchronous=True)
        inspect_world_mode(world)
        self.assertEqual(world.calls, [("get_settings",)])
        self.assertTrue(world.settings.synchronous_mode)

    def test_settings_failures_are_phase_specific(self):
        world = StrictWorld(synchronous="yes")
        with self.assertRaises(RuntimeError) as context:
            inspect_world_mode(world)
        self.assertIn("settings_inspection", str(context.exception))
        self.assertIsInstance(context.exception.__cause__, TypeError)

    def test_get_settings_rpc_failure_is_phase_specific_and_preserves_cause(self):
        error = RuntimeError("settings RPC unavailable")
        world = StrictWorld()

        def fail_get_settings():
            world.calls.append(("get_settings",))
            raise error

        world.get_settings = fail_get_settings
        with self.assertRaises(RuntimeError) as context:
            inspect_world_mode(world)
        self.assertIn("settings_inspection", str(context.exception))
        self.assertIs(context.exception.__cause__, error)
        self.assertEqual(world.calls, [("get_settings",)])


class ProgressionRequirementTest(unittest.TestCase):
    def test_asynchronous_world_waits_passively(self):
        self.assertIs(
            require_progression_policy(WorldMode.ASYNCHRONOUS, False),
            ProgressionPolicy.PASSIVE_WAIT,
        )

    def test_synchronous_owner_ticks(self):
        self.assertIs(
            require_progression_policy(WorldMode.SYNCHRONOUS, True),
            ProgressionPolicy.OWNED_TICK,
        )

    def test_unowned_synchronous_world_skips_with_exact_message(self):
        with self.assertRaises(unittest.SkipTest) as context:
            require_progression_policy(WorldMode.SYNCHRONOUS, False)
        self.assertEqual(
            str(context.exception), harness.SYNCHRONOUS_UNOWNED_MESSAGE
        )


class ProgressWorldOnceTest(unittest.TestCase):
    def progression_calls(self, world):
        return [
            call
            for call in world.calls
            if call[0] in ("wait_for_tick", "tick")
        ]

    def test_passive_wait_calls_wait_once_with_one_second_cap(self):
        world = StrictWorld()
        result = progress_world_once(
            world,
            ProgressionPolicy.PASSIVE_WAIT,
            Deadline(5.0, SequenceClock(0.0, 0.0, 0.0)),
        )
        self.assertEqual(result, "wait_for_tick-result")
        self.assertEqual(self.progression_calls(world), [("wait_for_tick", 1.0)])

    def test_owned_tick_calls_tick_once_with_one_second_cap(self):
        world = StrictWorld()
        result = progress_world_once(
            world,
            ProgressionPolicy.OWNED_TICK,
            Deadline(5.0, SequenceClock(0.0, 0.0, 0.0)),
        )
        self.assertEqual(result, "tick-result")
        self.assertEqual(self.progression_calls(world), [("tick", 1.0)])

    def test_timeout_is_capped_by_remaining_deadline(self):
        world = StrictWorld()
        progress_world_once(
            world,
            ProgressionPolicy.PASSIVE_WAIT,
            Deadline(1.0, SequenceClock(0.0, 0.0, 0.75)),
        )
        self.assertEqual(self.progression_calls(world), [("wait_for_tick", 0.25)])

    def test_expired_deadline_causes_no_rpc(self):
        world = StrictWorld()
        deadline = Deadline(1.0, SequenceClock(0.0, 1.0))
        with self.assertRaises(PhaseTimeout):
            progress_world_once(world, ProgressionPolicy.PASSIVE_WAIT, deadline)
        self.assertEqual(self.progression_calls(world), [])

    def test_precision_absorbed_expiry_causes_no_rpc(self):
        world = StrictWorld()
        deadline = Deadline(
            1.0,
            SequenceClock(0.0, math.nextafter(1.0, 0.0), 1.0),
        )
        with self.assertRaises(PhaseTimeout):
            progress_world_once(world, ProgressionPolicy.OWNED_TICK, deadline)
        self.assertEqual(self.progression_calls(world), [])

    def test_skip_raises_before_touching_deadline_or_world(self):
        world = StrictWorld()

        class ForbiddenDeadline:
            def __getattribute__(self, name):
                raise AssertionError(f"deadline was touched: {name}")

        with self.assertRaises(unittest.SkipTest):
            progress_world_once(world, ProgressionPolicy.SKIP, ForbiddenDeadline())
        self.assertEqual(self.progression_calls(world), [])

    def test_rpc_exception_is_preserved_and_never_retried(self):
        error = RuntimeError("rpc failed")
        world = StrictWorld(progression_error=error)
        with self.assertRaises(harness.ProgressionRPCError) as context:
            progress_world_once(
                world,
                ProgressionPolicy.PASSIVE_WAIT,
                Deadline(5.0, SequenceClock(0.0, 0.0, 0.0)),
            )
        self.assertIs(context.exception.__cause__, error)
        self.assertIn("world_progression_rpc", str(context.exception))
        self.assertEqual(len(self.progression_calls(world)), 1)

    def test_every_issued_timeout_is_finite_and_strictly_positive(self):
        for policy in (
            ProgressionPolicy.PASSIVE_WAIT,
            ProgressionPolicy.OWNED_TICK,
        ):
            with self.subTest(policy=policy):
                world = StrictWorld()
                progress_world_once(
                    world,
                    policy,
                    Deadline(0.5, SequenceClock(0.0, 0.0, 0.25)),
                )
                timeout = self.progression_calls(world)[0][1]
                self.assertTrue(math.isfinite(timeout))
                self.assertGreater(timeout, 0.0)
                self.assertLessEqual(timeout, 0.25)


class EgoBlueprintTest(unittest.TestCase):
    def test_role_name_is_literal_hero_when_supported(self):
        calls = []
        blueprint = StrictBlueprint(calls)
        self.assertTrue(configure_ego_blueprint(blueprint))
        self.assertEqual(blueprint.attributes["role_name"], "hero")
        self.assertEqual(
            calls,
            [
                ("blueprint.has_attribute", "role_name"),
                ("blueprint.set_attribute", "role_name", "hero"),
            ],
        )

    def test_unsupported_role_name_is_skipped_without_error(self):
        blueprint = StrictBlueprint([], "vehicle.x", role=False)
        self.assertFalse(configure_ego_blueprint(blueprint))
        self.assertEqual(blueprint.attributes, {})

    def test_harness_contains_no_duplicated_blueprint_picker(self):
        self.assertFalse(hasattr(harness, "pick_ego_blueprint"))
        self.assertFalse(hasattr(harness, "PREFERRED_EGO_BLUEPRINT"))


class EgoSpawnTest(unittest.TestCase):
    def attempts(self, world):
        return [call for call in world.calls if call[0] == "try_spawn_actor"]

    def test_points_are_attempted_in_order_and_first_success_stops(self):
        calls = []
        actor = StrictActor(calls)
        never_spawned = StrictActor(calls, actor_id=202)
        world = StrictWorld(
            calls,
            spawn_points=("a", "b", "c"),
            spawn_plan=[None, actor, never_spawned],
        )
        blueprint = StrictBlueprint(calls)
        owned = spawn_owned_ego(
            world,
            blueprint,
            Deadline(5.0, FakeClock(0.0)),
            role_name_required=True,
        )
        self.assertIs(owned.actor, actor)
        self.assertEqual([attempt[2] for attempt in self.attempts(world)], ["a", "b"])
        self.assertEqual(world.spawned, [actor])
        self.assertNotIn(never_spawned, world.spawned)
        self.assertEqual(actor.id_reads, 1)

    def test_deadline_is_checked_before_first_and_every_later_attempt(self):
        calls = []
        clock = CountingClock(0.0)
        world = StrictWorld(
            calls,
            spawn_points=("a", "b"),
            spawn_plan=[None, StrictActor(calls)],
        )
        spawn_owned_ego(
            world, StrictBlueprint(calls), Deadline(5.0, clock)
        )
        self.assertEqual(clock.calls, 3)
        self.assertEqual(len(self.attempts(world)), 2)

    def test_expiry_before_attempt_zero_starts_no_spawn_rpc(self):
        calls = []
        world = StrictWorld(calls, spawn_points=("a",), spawn_plan=[StrictActor(calls)])
        deadline = Deadline(1.0, SequenceClock(0.0, 1.0))
        with self.assertRaises(PhaseTimeout) as context:
            spawn_owned_ego(world, StrictBlueprint(calls), deadline)
        self.assertEqual(context.exception.phase, "ego_spawn")
        self.assertEqual(self.attempts(world), [])

    def test_expiry_between_attempts_prevents_the_next_rpc(self):
        calls = []
        world = StrictWorld(
            calls,
            spawn_points=("a", "b"),
            spawn_plan=[None, StrictActor(calls)],
        )
        deadline = Deadline(1.0, SequenceClock(0.0, 0.0, 1.0))
        with self.assertRaises(PhaseTimeout):
            spawn_owned_ego(world, StrictBlueprint(calls), deadline)
        self.assertEqual([attempt[2] for attempt in self.attempts(world)], ["a"])

    def test_falsey_non_none_actor_is_owned_and_stops_iteration(self):
        calls = []
        actor = StrictActor(calls, falsey=True)
        world = StrictWorld(
            calls,
            spawn_points=("a", "b"),
            spawn_plan=[actor, StrictActor(calls, actor_id=202)],
        )
        owned = spawn_owned_ego(
            world, StrictBlueprint(calls), Deadline(5.0, FakeClock(0.0))
        )
        self.assertIs(owned.actor, actor)
        self.assertEqual(len(self.attempts(world)), 1)

    def test_empty_spawn_points_fail_cleanly_without_rpc(self):
        calls = []
        world = StrictWorld(calls, spawn_points=())
        with self.assertRaises(RuntimeError) as context:
            spawn_owned_ego(
                world, StrictBlueprint(calls), Deadline(5.0, FakeClock(0.0))
            )
        self.assertIn("no spawn points", str(context.exception))
        self.assertEqual(self.attempts(world), [])

    def test_all_occupied_fails_cleanly_after_exactly_one_rpc_per_point(self):
        calls = []
        world = StrictWorld(
            calls,
            spawn_points=("a", "b"),
            spawn_plan=[None, None],
        )
        with self.assertRaises(RuntimeError) as context:
            spawn_owned_ego(
                world, StrictBlueprint(calls), Deadline(5.0, FakeClock(0.0))
            )
        self.assertIn("all 2", str(context.exception))
        self.assertEqual(len(self.attempts(world)), 2)

    def test_spawn_exception_is_phase_specific_with_preserved_cause(self):
        calls = []
        error = RuntimeError("spawn rpc down")
        world = StrictWorld(calls, spawn_points=("a",), spawn_plan=[error])
        with self.assertRaises(RuntimeError) as context:
            spawn_owned_ego(
                world, StrictBlueprint(calls), Deadline(5.0, FakeClock(0.0))
            )
        self.assertIs(context.exception.__cause__, error)
        self.assertIn("ego_spawn", str(context.exception))
        self.assertEqual(len(self.attempts(world)), 1)

    def test_map_and_spawn_point_acquisition_failures_are_phase_specific(self):
        for step in ("get_map", "get_spawn_points"):
            with self.subTest(step=step):
                calls = []
                error = RuntimeError(f"{step} rpc down")
                world = StrictWorld(calls)
                if step == "get_map":
                    def fail_get_map():
                        raise error

                    world.get_map = fail_get_map
                else:
                    class FailingMap:
                        def get_spawn_points(self):
                            raise error

                    world.map = FailingMap()
                with self.assertRaises(RuntimeError) as context:
                    spawn_owned_ego(
                        world,
                        StrictBlueprint(calls),
                        Deadline(5.0, FakeClock(0.0)),
                    )
                self.assertIs(context.exception.__cause__, error)
                self.assertIn("ego_spawn", str(context.exception))
                self.assertEqual(self.attempts(world), [])

    def test_spawning_uses_no_worker_thread(self):
        calls = []
        actor = StrictActor(calls)
        world = StrictWorld(calls, spawn_plan=[actor])
        with unittest.mock.patch(
            "threading.Thread",
            side_effect=AssertionError("worker thread forbidden"),
        ) as thread_constructor:
            owned = spawn_owned_ego(
                world,
                StrictBlueprint(calls),
                Deadline(5.0, FakeClock(0.0)),
            )
        self.assertIs(owned.actor, actor)
        thread_constructor.assert_not_called()


class EgoIdentityTest(unittest.TestCase):
    def captured(self, role_required=True, **actor_kwargs):
        actor = StrictActor(**actor_kwargs)
        return actor, capture_owned_ego(actor, role_required)

    def test_valid_positive_builtin_id_vehicle_type_and_hero_pass(self):
        actor, owned = self.captured(actor_id=7, role_name="hero")
        self.assertIsNone(verify_ego_identity(owned))
        self.assertEqual(owned.actor_id, 7)
        self.assertEqual(actor.id_reads, 1)

    def test_zero_negative_bool_missing_and_non_builtin_ids_fail(self):
        class IntSubclass(int):
            pass

        for actor_id in (0, -1, True, None, "7", IntSubclass(7)):
            with self.subTest(actor_id=actor_id):
                _actor, owned = self.captured(actor_id=actor_id)
                failure = verify_ego_identity(owned)
                self.assertEqual(failure.phase, "ego_identity")
                self.assertIn("positive built-in integer", failure.message)

    def test_unavailable_id_is_bounded_identity_failure(self):
        actor, owned = self.captured(id_error=RuntimeError("id\n" + "x" * 1000))
        failure = verify_ego_identity(owned)
        self.assertIn("actor id unavailable", failure.message)
        self.assertLessEqual(
            len(failure.message), harness.MAX_PRIMARY_MESSAGE_LENGTH
        )
        self.assertEqual(actor.id_reads, 1)

    def test_non_vehicle_and_non_string_type_ids_fail_without_coercion(self):
        class VehicleLookingObject:
            def __str__(self):
                return "vehicle.fake"

        for type_id in ("sensor.other.gnss", "vehicle", "", None, VehicleLookingObject()):
            with self.subTest(type_id=type_id):
                _actor, owned = self.captured(type_id=type_id)
                failure = verify_ego_identity(owned)
                self.assertIn("not a vehicle string", failure.message)

    def test_literal_hero_is_required_only_when_supported(self):
        _actor, wrong = self.captured(role_name="autopilot")
        failure = verify_ego_identity(wrong)
        self.assertIn("role_name", failure.message)
        self.assertIn("'hero'", failure.message)
        _actor, unsupported = self.captured(
            role_required=False, role_name="autopilot"
        )
        self.assertIsNone(verify_ego_identity(unsupported))

    def test_non_string_role_cannot_masquerade_as_literal_hero(self):
        class HeroMasquerader:
            def __eq__(self, other):
                return other == "hero"

        _actor, owned = self.captured(role_name=HeroMasquerader())
        failure = verify_ego_identity(owned)
        self.assertIn("role_name", failure.message)

    def test_str_subclass_cannot_override_vehicle_prefix_validation(self):
        class LyingStr(str):
            def startswith(self, _prefix, *_args):
                return True

        _actor, owned = self.captured(type_id=LyingStr("walker.pedestrian"))
        failure = verify_ego_identity(owned)
        self.assertIn("not a vehicle string", failure.message)

    def test_stale_is_alive_false_is_non_authoritative(self):
        actor, owned = self.captured(is_alive=False)
        self.assertIsNone(verify_ego_identity(owned))
        self.assertEqual(actor.is_alive_reads, 0)

    def test_exception_throwing_is_alive_is_never_read(self):
        actor, owned = self.captured(
            is_alive_error=AssertionError("is_alive must not be read")
        )
        self.assertIsNone(verify_ego_identity(owned))
        self.assertEqual(actor.is_alive_reads, 0)

    def test_failure_message_is_bounded_and_sanitized(self):
        _actor, owned = self.captured(
            type_id="walker.\x1b[31m" + "x" * 500,
            role_name=None,
        )
        failure = verify_ego_identity(owned)
        self.assertLessEqual(
            len(failure.message), harness.MAX_PRIMARY_MESSAGE_LENGTH
        )
        self.assertNotIn("\x1b", failure.message)


class DestroyOwnedEgoTest(unittest.TestCase):
    def outcome_for(self, actor):
        owned = capture_owned_ego(actor, True)
        verification_calls = []
        outcome = coordinate_cleanup(
            None,
            ego_cleanup=lambda: destroy_owned_ego(owned),
            final_cleanup=lambda: (
                verification_calls.append("verify"),
                CleanupVerification.CONFIRMED,
            )[1],
        )
        return outcome, verification_calls

    def test_true_false_and_exception_each_destroy_once_and_always_verify(self):
        cases = (
            (StrictActor(destroy_result=True), False),
            (StrictActor(destroy_result=False), True),
            (
                StrictActor(destroy_error=RuntimeError("destroy rpc failed")),
                True,
            ),
        )
        for actor, expects_error in cases:
            with self.subTest(expects_error=expects_error):
                outcome, verification_calls = self.outcome_for(actor)
                self.assertEqual(actor.destroy_calls, 1)
                self.assertEqual(verification_calls, ["verify"])
                self.assertEqual(bool(outcome.cleanup_errors), expects_error)
                self.assertIs(
                    outcome.verification, CleanupVerification.CONFIRMED
                )

    def test_only_literal_true_is_success(self):
        for result in (False, None, 0, 1, "true"):
            with self.subTest(result=result):
                actor = StrictActor(destroy_result=result)
                outcome, _calls = self.outcome_for(actor)
                self.assertEqual(actor.destroy_calls, 1)
                self.assertIn("destroy() returned", outcome.cleanup_errors[0])


class EgoDisappearanceTest(unittest.TestCase):
    def deadline(self, clock=None, timeout=5.0):
        return Deadline(timeout, clock if clock is not None else FakeClock(0.0))

    def register(self, world, actor):
        actor.world = world
        world.registry[actor._actor_id] = actor

    def progression_calls(self, world):
        return [call for call in world.calls if call[0] in ("wait_for_tick", "tick")]

    def test_immediate_absence_confirms_with_one_authoritative_snapshot(self):
        world = StrictWorld()
        verification = verify_ego_disappearance(
            world, 999, ProgressionPolicy.PASSIVE_WAIT, self.deadline()
        )
        self.assertIs(verification, CleanupVerification.CONFIRMED)
        self.assertEqual(world.snapshot_queried_ids, [999])
        self.assertEqual(world.queried_ids, [])
        self.assertEqual(self.progression_calls(world), [])

    def test_absence_confirms_even_when_deadline_is_already_expired(self):
        world = StrictWorld()
        verification = verify_ego_disappearance(
            world,
            7,
            ProgressionPolicy.PASSIVE_WAIT,
            Deadline(1.0, SequenceClock(0.0, 1.0)),
        )
        self.assertIs(verification, CleanupVerification.CONFIRMED)
        self.assertEqual(world.snapshot_queried_ids, [7])

    def test_delayed_absence_uses_passive_wait_and_owned_id_only(self):
        calls = []
        actor = StrictActor(calls, actor_id=7, is_alive=False)
        unrelated = StrictActor(calls, actor_id=55, role_name="hero")
        world = StrictWorld(
            calls,
            remove_after_progressions=2,
            remove_actor_id=7,
        )
        self.register(world, actor)
        self.register(world, unrelated)
        verification = verify_ego_disappearance(
            world, 7, ProgressionPolicy.PASSIVE_WAIT, self.deadline()
        )
        self.assertIs(verification, CleanupVerification.CONFIRMED)
        self.assertEqual(world.snapshot_queried_ids, [7, 7, 7])
        self.assertEqual(world.queried_ids, [])
        self.assertEqual(
            [call[0] for call in self.progression_calls(world)],
            ["wait_for_tick", "wait_for_tick"],
        )
        self.assertIn(55, world.registry)
        self.assertEqual(actor.is_alive_reads, 0)
        self.assertEqual(actor.destroy_calls, 0)
        self.assertEqual(unrelated.destroy_calls, 0)

    def test_delayed_absence_in_owned_world_uses_tick_with_timeout(self):
        actor = StrictActor(actor_id=7)
        world = StrictWorld(
            remove_after_progressions=1,
            remove_actor_id=7,
        )
        self.register(world, actor)
        verification = verify_ego_disappearance(
            world, 7, ProgressionPolicy.OWNED_TICK, self.deadline()
        )
        self.assertIs(verification, CleanupVerification.CONFIRMED)
        self.assertEqual(self.progression_calls(world), [("tick", 1.0)])

    def test_only_literal_false_membership_counts_as_absent(self):
        class JunkSnapshot:
            def __init__(self, result):
                self._result = result

            def has_actor(self, _actor_id):
                return self._result

        class JunkSnapshotWorld(StrictWorld):
            def __init__(self, results):
                super().__init__()
                self._results = list(results)

            def get_snapshot(self):
                self.calls.append(("get_snapshot",))
                return JunkSnapshot(self._results.pop(0))

        world = JunkSnapshotWorld([0, None, False])
        verification = verify_ego_disappearance(
            world, 7, ProgressionPolicy.PASSIVE_WAIT, self.deadline()
        )
        self.assertIs(verification, CleanupVerification.CONFIRMED)
        self.assertEqual(
            [call for call in world.calls if call[0] == "get_snapshot"],
            [("get_snapshot",)] * 3,
        )
        self.assertEqual(
            self.progression_calls(world),
            [("wait_for_tick", 1.0), ("wait_for_tick", 1.0)],
        )

    def test_actor_remaining_until_deadline_reports_failed_actors_remain(self):
        clock = FakeClock(0.0)
        actor = StrictActor(actor_id=7)
        world = StrictWorld(on_progress=lambda: setattr(clock, "value", clock.value + 0.6))
        self.register(world, actor)
        verification = verify_ego_disappearance(
            world, 7, ProgressionPolicy.PASSIVE_WAIT, self.deadline(clock, 1.0)
        )
        self.assertIs(
            verification, CleanupVerification.FAILED_ACTORS_REMAIN
        )
        self.assertTrue(self.progression_calls(world))
        self.assertTrue(
            all(0.0 < call[1] <= 1.0 for call in self.progression_calls(world))
        )

    def test_lookup_and_progression_disconnects_are_never_confirmed(self):
        lookup_world = StrictWorld(snapshot_plan=[RuntimeError("snapshot down")])
        lookup = verify_ego_disappearance(
            lookup_world, 7, ProgressionPolicy.PASSIVE_WAIT, self.deadline()
        )
        self.assertIs(lookup, CleanupVerification.UNVERIFIED_DISCONNECTED)

        actor = StrictActor(actor_id=7)
        progression_world = StrictWorld(progression_error=RuntimeError("tick down"))
        self.register(progression_world, actor)
        progression = verify_ego_disappearance(
            progression_world,
            7,
            ProgressionPolicy.PASSIVE_WAIT,
            self.deadline(),
        )
        self.assertIs(
            progression, CleanupVerification.UNVERIFIED_DISCONNECTED
        )
        self.assertNotIn(CleanupVerification.CONFIRMED, (lookup, progression))

    def test_unexpected_local_programming_error_is_not_called_disconnect(self):
        error = AssertionError("bad fake/programming defect")
        world = StrictWorld(snapshot_plan=[error])
        with self.assertRaises(AssertionError) as context:
            verify_ego_disappearance(
                world, 7, ProgressionPolicy.PASSIVE_WAIT, self.deadline()
            )
        self.assertIs(context.exception, error)

    def test_pre_rpc_host_clock_failure_is_not_called_disconnect(self):
        error = RuntimeError("host clock defect")
        readings = iter((0.0, 0.0))

        def clock():
            try:
                return next(readings)
            except StopIteration:
                raise error

        actor = StrictActor(actor_id=7)
        world = StrictWorld()
        self.register(world, actor)
        with self.assertRaises(RuntimeError) as context:
            verify_ego_disappearance(
                world,
                7,
                ProgressionPolicy.PASSIVE_WAIT,
                Deadline(1.0, clock),
            )
        self.assertIs(context.exception, error)
        self.assertEqual(self.progression_calls(world), [])

    def test_local_progression_assertion_is_not_called_disconnect(self):
        error = AssertionError("local progression defect")
        actor = StrictActor(actor_id=7)
        world = StrictWorld(progression_error=error)
        self.register(world, actor)
        with self.assertRaises(AssertionError) as context:
            verify_ego_disappearance(
                world, 7, ProgressionPolicy.PASSIVE_WAIT, self.deadline()
            )
        self.assertIs(context.exception, error)
        self.assertEqual(len(self.progression_calls(world)), 1)

    def test_invalid_or_missing_copied_id_is_unverified_without_lookup(self):
        for actor_id in (None, True, 0, -1, "7"):
            with self.subTest(actor_id=actor_id):
                world = StrictWorld()
                verification = verify_ego_disappearance(
                    world,
                    actor_id,
                    ProgressionPolicy.PASSIVE_WAIT,
                    self.deadline(),
                )
                self.assertIs(
                    verification, CleanupVerification.UNVERIFIED_DISCONNECTED
                )
                self.assertEqual(world.snapshot_queried_ids, [])
                self.assertEqual(world.calls, [])

    def test_cached_lookup_enumeration_and_destroy_are_never_used(self):
        world = StrictWorld()
        verification = verify_ego_disappearance(
            world, 7, ProgressionPolicy.PASSIVE_WAIT, self.deadline()
        )
        self.assertIs(verification, CleanupVerification.CONFIRMED)
        self.assertEqual(
            [call[0] for call in world.calls],
            ["get_snapshot", "snapshot.has_actor"],
        )
        self.assertEqual(world.queried_ids, [])


class ConnectionSpawnCycleTest(unittest.TestCase):
    def test_happy_path_uses_exact_picker_and_confirms_cleanup(self):
        rig = LiveRig()
        report = run_cycle(rig)
        self.assertEqual(report.versions.client_version, "0.9.16")
        self.assertTrue(report.versions.compatible)
        self.assertIs(report.world_mode, WorldMode.ASYNCHRONOUS)
        self.assertIs(report.policy, ProgressionPolicy.PASSIVE_WAIT)
        self.assertEqual(len(rig.world.spawned), 1)
        self.assertIs(rig.world.spawned[0], rig.actor)
        self.assertEqual(report.ego_id, 101)
        self.assertEqual(report.ego_type_id, "vehicle.tesla.model3")
        self.assertIsNone(report.outcome.primary_failure)
        self.assertEqual(report.outcome.cleanup_errors, ())
        self.assertIs(report.outcome.verification, CleanupVerification.CONFIRMED)
        self.assertEqual(rig.actor.destroy_calls, 1)
        self.assertEqual(rig.actor.id_reads, 1)
        self.assertEqual(rig.actor.is_alive_reads, 0)
        self.assertEqual(rig.importer.requests, ["carla", "spawn_ego_vehicle"])
        self.assertEqual(rig.picker.arguments, [rig.library])
        self.assertEqual(rig.blueprint.attributes["role_name"], "hero")
        spawn_call = next(
            call for call in rig.calls if call[0] == "try_spawn_actor"
        )
        self.assertIs(spawn_call[1], rig.blueprint)
        self.assertEqual(
            rig.actor.identity_read_order,
            ["id", "type_id", "attributes"],
        )

    def test_complete_connection_and_spawn_call_order(self):
        rig = LiveRig()
        run_cycle(rig)
        self.assertEqual(
            [call[0] for call in rig.calls],
            [
                "import",
                "import",
                "Client",
                "set_timeout",
                "get_client_version",
                "get_server_version",
                "get_world",
                "get_settings",
                "get_blueprint_library",
                "pick_vehicle_blueprint",
                "blueprint.has_attribute",
                "blueprint.set_attribute",
                "get_map",
                "get_spawn_points",
                "try_spawn_actor",
                "actor.destroy",
                "get_snapshot",
                "snapshot.has_actor",
            ],
        )
        self.assertEqual(rig.calls[3], ("set_timeout", 10.0))

    def test_disabled_environ_skips_before_import(self):
        rig = LiveRig()
        with self.assertRaises(unittest.SkipTest):
            run_connection_spawn_cycle(
                {}, importer=rig.importer, monotonic=FakeClock(0.0)
            )
        self.assertEqual(rig.importer.requests, [])
        self.assertEqual(rig.carla.client_calls, 0)
        self.assertEqual(rig.calls, [])

    def test_malformed_enabled_config_fails_before_import_or_client(self):
        rig = LiveRig()
        with self.assertRaises(ValueError):
            run_cycle(
                rig,
                environ={CARLA_INTEGRATION: "1", CARLA_PORT: "not-a-port"},
            )
        self.assertEqual(rig.importer.requests, [])
        self.assertEqual(rig.carla.client_calls, 0)
        self.assertEqual(rig.calls, [])

    def test_unowned_synchronous_world_skips_before_blueprint_or_spawn(self):
        rig = LiveRig(synchronous=True)
        with self.assertRaises(unittest.SkipTest) as context:
            run_cycle(rig)
        self.assertEqual(str(context.exception), harness.SYNCHRONOUS_UNOWNED_MESSAGE)
        self.assertEqual(rig.world.spawned, [])
        forbidden = {
            "get_blueprint_library",
            "pick_vehicle_blueprint",
            "get_map",
            "get_spawn_points",
            "try_spawn_actor",
            "tick",
            "wait_for_tick",
        }
        self.assertTrue(forbidden.isdisjoint(call[0] for call in rig.calls))

    def test_owned_synchronous_world_uses_owned_tick(self):
        actor = StrictActor(destroy_result=False)
        rig = LiveRig(
            synchronous=True,
            actor=actor,
            world_kwargs={"remove_after_progressions": 1},
        )
        report = run_cycle(
            rig,
            environ={CARLA_INTEGRATION: "1", CARLA_INTEGRATION_TICK_OWNER: "1"},
        )
        self.assertIs(report.policy, ProgressionPolicy.OWNED_TICK)
        self.assertIs(report.outcome.verification, CleanupVerification.CONFIRMED)
        self.assertEqual(
            [call[0] for call in rig.calls if call[0] in ("tick", "wait_for_tick")],
            ["tick"],
        )

    def test_equal_wrong_versions_fail_before_world_acquisition(self):
        rig = LiveRig(client_version="0.9.15", server_version="0.9.15")
        with self.assertRaises(RuntimeError):
            run_cycle(rig)
        self.assertEqual(
            [call[0] for call in rig.calls],
            [
                "import",
                "import",
                "Client",
                "set_timeout",
                "get_client_version",
                "get_server_version",
            ],
        )
        self.assertEqual(rig.world.spawned, [])

    def test_connection_phase_failures_are_bounded_and_stop_in_order(self):
        cases = (
            ("constructor", "client_construction"),
            ("set_timeout", "timeout_configuration"),
            ("get_client_version", "client_version_query"),
            ("get_server_version", "server_version_query"),
            ("get_world", "world_acquisition"),
        )
        for step, phase in cases:
            with self.subTest(step=step):
                error = RuntimeError("rpc down")
                rig = LiveRig(
                    client_construction_error=error if step == "constructor" else None,
                    client_errors=(
                        {step: error} if step != "constructor" else None
                    ),
                )
                with self.assertRaises(RuntimeError) as context:
                    run_cycle(rig)
                self.assertIn(phase, str(context.exception))
                self.assertIs(context.exception.__cause__, error)
                self.assertEqual(rig.world.spawned, [])

    def test_progression_policy_selection_failure_is_phase_specific(self):
        rig = LiveRig()
        with unittest.mock.patch.object(
            harness,
            "decide_world_progression",
            side_effect=ValueError("bad policy"),
        ):
            with self.assertRaises(RuntimeError) as context:
                run_cycle(rig)
        self.assertIn("progression_policy_selection", str(context.exception))
        self.assertNotIn("get_blueprint_library", [call[0] for call in rig.calls])

    def test_existing_unrelated_hero_is_never_queried_adopted_or_destroyed(self):
        rig = LiveRig()
        unrelated = StrictActor(rig.calls, actor_id=55, role_name="hero")
        unrelated.world = rig.world
        rig.world.registry[55] = unrelated
        report = run_cycle(rig)
        self.assertIsNone(report.outcome.primary_failure)
        self.assertEqual(rig.world.snapshot_queried_ids, [101])
        self.assertIn(55, rig.world.registry)
        self.assertEqual(unrelated.destroy_calls, 0)
        self.assertEqual(unrelated.control_calls, 0)
        self.assertEqual(unrelated.id_reads, 0)

    def test_spawn_failure_records_primary_without_false_cleanup_confirmation(self):
        rig = LiveRig(spawn_points=("a",), spawn_plan=[None])
        report = run_cycle(rig)
        self.assertEqual(report.outcome.primary_failure.phase, "ego_spawn")
        self.assertIsNone(report.ego_id)
        self.assertEqual(rig.world.spawned, [])
        self.assertEqual(rig.actor.destroy_calls, 0)
        self.assertIs(
            report.outcome.verification,
            CleanupVerification.UNVERIFIED_DISCONNECTED,
        )

    def test_spawn_deadline_is_created_by_orchestration(self):
        rig = LiveRig(spawn_points=("a",), spawn_plan=[rig_actor := StrictActor()])
        clock = SequenceClock(0.0, 10.0)
        report = run_cycle(rig, clock=clock)
        self.assertEqual(report.outcome.primary_failure.phase, "ego_spawn")
        self.assertEqual(rig.world.spawned, [])
        self.assertEqual(rig_actor.destroy_calls, 0)

    def test_invalid_owned_id_still_destroys_proxy_and_never_confirms(self):
        actor = StrictActor(actor_id=0)
        rig = LiveRig(actor=actor)
        report = run_cycle(rig)
        self.assertIsNotNone(report.outcome.primary_failure)
        self.assertEqual(report.outcome.primary_failure.phase, "ego_identity")
        self.assertEqual(actor.destroy_calls, 1)
        self.assertIsNone(report.ego_id)
        self.assertEqual(rig.world.snapshot_queried_ids, [])
        self.assertIs(
            report.outcome.verification,
            CleanupVerification.UNVERIFIED_DISCONNECTED,
        )

    def test_unsupported_role_attribute_is_valid_through_orchestration(self):
        rig = LiveRig(role_supported=False)
        report = run_cycle(rig)
        self.assertIsNone(report.outcome.primary_failure)
        self.assertEqual(rig.blueprint.attributes, {})
        self.assertEqual(rig.actor.attributes_reads, 0)
        self.assertIs(
            report.outcome.verification, CleanupVerification.CONFIRMED
        )

    def test_type_and_attribute_read_errors_become_identity_failures_and_cleanup(self):
        actors = (
            StrictActor(type_error=RuntimeError("type unavailable")),
            StrictActor(attributes_error=RuntimeError("attributes unavailable")),
        )
        for actor in actors:
            with self.subTest(actor=actor):
                rig = LiveRig(actor=actor)
                report = run_cycle(rig)
                self.assertEqual(
                    report.outcome.primary_failure.phase, "ego_identity"
                )
                self.assertEqual(actor.destroy_calls, 1)
                self.assertEqual(rig.world.snapshot_queried_ids, [101])
                self.assertIs(
                    report.outcome.verification,
                    CleanupVerification.CONFIRMED,
                )

    def test_type_and_attribute_keyboard_interrupts_cleanup_then_propagate(self):
        for field in ("type", "attributes"):
            with self.subTest(field=field):
                interrupt = KeyboardInterrupt()
                actor = StrictActor(
                    type_error=interrupt if field == "type" else None,
                    attributes_error=(
                        interrupt if field == "attributes" else None
                    ),
                )
                rig = LiveRig(actor=actor)
                with self.assertRaises(KeyboardInterrupt) as context:
                    run_cycle(rig)
                self.assertIs(context.exception, interrupt)
                self.assertEqual(actor.destroy_calls, 1)
                self.assertEqual(rig.world.snapshot_queried_ids, [101])

    def test_interrupt_immediately_after_spawn_return_still_retains_and_cleans(self):
        rig = LiveRig()
        interrupt = KeyboardInterrupt()
        rpc_returned = False
        injected = False
        original_try_spawn_actor = rig.world.try_spawn_actor

        def marked_try_spawn_actor(blueprint, spawn_point):
            nonlocal rpc_returned
            actor = original_try_spawn_actor(blueprint, spawn_point)
            rpc_returned = True
            return actor

        def inject_after_rpc(frame, event, _arg):
            nonlocal injected
            if (
                event == "line"
                and frame.f_code is spawn_owned_ego.__code__
                and rpc_returned
                and not injected
            ):
                injected = True
                raise interrupt
            return inject_after_rpc

        rig.world.try_spawn_actor = marked_try_spawn_actor
        sys.settrace(inject_after_rpc)
        try:
            with self.assertRaises(KeyboardInterrupt) as context:
                run_cycle(rig)
        finally:
            sys.settrace(None)
        self.assertTrue(injected)
        self.assertIs(context.exception, interrupt)
        self.assertEqual(rig.world.spawned, [rig.actor])
        self.assertEqual(rig.actor.destroy_calls, 1)
        self.assertEqual(rig.world.snapshot_queried_ids, [101])

    def test_interrupt_inside_cleanup_coordinator_resumes_remaining_steps(self):
        for phase in ("before_destroy", "after_destroy"):
            with self.subTest(phase=phase):
                rig = LiveRig()
                interrupt = KeyboardInterrupt()
                injected = False

                def inject_in_coordinator(frame, event, _arg):
                    nonlocal injected
                    ready = (
                        bool(rig.world.spawned)
                        and (
                            rig.actor.destroy_calls == 0
                            if phase == "before_destroy"
                            else rig.actor.destroy_calls == 1
                            and not rig.world.snapshot_queried_ids
                        )
                    )
                    if (
                        event == "line"
                        and frame.f_code is coordinate_cleanup.__code__
                        and ready
                        and not injected
                    ):
                        injected = True
                        raise interrupt
                    return inject_in_coordinator

                sys.settrace(inject_in_coordinator)
                try:
                    with self.assertRaises(KeyboardInterrupt) as context:
                        run_cycle(rig)
                finally:
                    sys.settrace(None)
                self.assertTrue(injected)
                self.assertIs(context.exception, interrupt)
                self.assertEqual(rig.actor.destroy_calls, 1)
                self.assertEqual(rig.world.snapshot_queried_ids, [101])

    def test_unexpected_capture_failure_after_spawn_still_cleans_owned_proxy(self):
        for error in (RuntimeError("capture defect"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                rig = LiveRig()
                with unittest.mock.patch.object(
                    harness, "capture_owned_ego", side_effect=error
                ):
                    if isinstance(error, KeyboardInterrupt):
                        with self.assertRaises(KeyboardInterrupt) as context:
                            run_cycle(rig)
                        self.assertIs(context.exception, error)
                    else:
                        report = run_cycle(rig)
                        self.assertEqual(
                            report.outcome.primary_failure.phase,
                            "post_spawn_validation",
                        )
                        self.assertIs(
                            report.outcome.verification,
                            CleanupVerification.UNVERIFIED_DISCONNECTED,
                        )
                self.assertEqual(rig.world.spawned, [rig.actor])
                self.assertEqual(rig.actor.destroy_calls, 1)
                self.assertEqual(rig.world.snapshot_queried_ids, [])

    def test_unexpected_capture_record_still_cleans_exact_owned_proxy(self):
        rig = LiveRig()
        with unittest.mock.patch.object(
            harness, "capture_owned_ego", return_value=object()
        ):
            report = run_cycle(rig)
        self.assertEqual(
            report.outcome.primary_failure.phase,
            "post_spawn_validation",
        )
        self.assertEqual(rig.world.spawned, [rig.actor])
        self.assertEqual(rig.actor.destroy_calls, 1)
        self.assertEqual(rig.world.snapshot_queried_ids, [])
        self.assertIs(
            report.outcome.verification,
            CleanupVerification.UNVERIFIED_DISCONNECTED,
        )

    def test_mismatched_capture_proxy_is_primary_failure_and_never_destroyed(self):
        rig = LiveRig()
        unrelated = StrictActor(rig.calls, actor_id=202, role_name="hero")
        mismatched = harness.OwnedEgo(
            actor=unrelated,
            actor_id=202,
            type_id="vehicle.unrelated",
            role_name="hero",
            role_name_required=True,
        )
        with unittest.mock.patch.object(
            harness, "capture_owned_ego", return_value=mismatched
        ):
            report = run_cycle(rig)
        self.assertEqual(
            report.outcome.primary_failure.phase,
            "post_spawn_validation",
        )
        self.assertEqual(rig.actor.destroy_calls, 1)
        self.assertEqual(unrelated.destroy_calls, 0)
        self.assertEqual(rig.world.snapshot_queried_ids, [])
        self.assertIs(
            report.outcome.verification,
            CleanupVerification.UNVERIFIED_DISCONNECTED,
        )

    def test_destroy_true_false_and_exception_all_run_verification_once(self):
        cases = (
            StrictActor(destroy_result=True),
            StrictActor(destroy_result=False, remove_even_on_failure=True),
            StrictActor(
                destroy_error=RuntimeError("destroy rpc failed")
            ),
        )
        for actor in cases:
            with self.subTest(
                destroy_result=actor._destroy_result,
                destroy_error=actor._destroy_error,
            ):
                world_kwargs = (
                    {"remove_after_progressions": 1}
                    if actor._destroy_error is not None
                    else None
                )
                rig = LiveRig(actor=actor, world_kwargs=world_kwargs)
                report = run_cycle(rig)
                self.assertEqual(actor.destroy_calls, 1)
                self.assertEqual(rig.world.snapshot_queried_ids, [101, 101] if actor._destroy_error else [101])
                self.assertIs(
                    report.outcome.verification,
                    CleanupVerification.CONFIRMED,
                )
                self.assertEqual(
                    bool(report.outcome.cleanup_errors),
                    actor._destroy_result is not True or actor._destroy_error is not None,
                )

    def test_destroy_false_with_persistent_actor_reports_both_cleanup_failures(self):
        clock = FakeClock(0.0)
        actor = StrictActor(destroy_result=False)
        rig = LiveRig(
            actor=actor,
            world_kwargs={
                "on_progress": lambda: setattr(
                    clock, "value", clock.value + 0.6
                )
            },
        )
        report = run_cycle(rig, clock=clock, timeout=1.0)
        self.assertEqual(actor.destroy_calls, 1)
        self.assertEqual(len(report.outcome.cleanup_errors), 1)
        self.assertIs(
            report.outcome.verification,
            CleanupVerification.FAILED_ACTORS_REMAIN,
        )
        self.assertGreaterEqual(len(rig.world.snapshot_queried_ids), 2)
        self.assertEqual(set(rig.world.snapshot_queried_ids), {101})

    def test_destroy_false_with_lookup_or_progression_disconnect_is_unverified(self):
        cases = (
            {"snapshot_plan": [RuntimeError("snapshot disconnected")]},
            {"progression_error": RuntimeError("progression disconnected")},
        )
        for world_kwargs in cases:
            with self.subTest(world_kwargs=tuple(world_kwargs)):
                actor = StrictActor(destroy_result=False)
                rig = LiveRig(actor=actor, world_kwargs=world_kwargs)
                report = run_cycle(rig)
                self.assertEqual(actor.destroy_calls, 1)
                self.assertEqual(len(report.outcome.cleanup_errors), 1)
                self.assertIs(
                    report.outcome.verification,
                    CleanupVerification.UNVERIFIED_DISCONNECTED,
                )

    def test_primary_failure_is_preserved_when_cleanup_also_fails(self):
        actor = StrictActor(destroy_result=False, remove_even_on_failure=True)
        rig = LiveRig(actor=actor)
        primary = ValidationFailure("validation", "original", 1)
        with unittest.mock.patch.object(
            harness, "verify_ego_identity", return_value=primary
        ):
            report = run_cycle(rig)
        self.assertIs(report.outcome.primary_failure, primary)
        self.assertEqual(len(report.outcome.cleanup_errors), 1)
        self.assertIs(
            report.outcome.verification, CleanupVerification.CONFIRMED
        )

    def test_post_spawn_runtime_assertion_and_unexpected_errors_all_cleanup(self):
        for error in (
            RuntimeError("runtime"),
            AssertionError("assertion"),
            ValueError("unexpected"),
        ):
            with self.subTest(error=type(error).__name__):
                rig = LiveRig()
                with unittest.mock.patch.object(
                    harness, "verify_ego_identity", side_effect=error
                ):
                    report = run_cycle(rig)
                self.assertEqual(
                    report.outcome.primary_failure.phase,
                    "post_spawn_validation",
                )
                self.assertEqual(rig.actor.destroy_calls, 1)
                self.assertEqual(rig.world.snapshot_queried_ids, [101])
                self.assertIs(
                    report.outcome.verification,
                    CleanupVerification.CONFIRMED,
                )

    def test_keyboard_interrupt_propagates_after_destroy_and_verification(self):
        actor = StrictActor(destroy_result=False, remove_even_on_failure=True)
        rig = LiveRig(actor=actor)
        interrupt = KeyboardInterrupt()
        with unittest.mock.patch.object(
            harness, "verify_ego_identity", side_effect=interrupt
        ):
            with self.assertRaises(KeyboardInterrupt) as context:
                run_cycle(rig)
        self.assertIs(context.exception, interrupt)
        self.assertEqual(actor.destroy_calls, 1)
        self.assertEqual(rig.world.snapshot_queried_ids, [101])

    def test_destroy_keyboard_interrupt_is_deferred_until_after_verification(self):
        interrupt = KeyboardInterrupt()
        actor = StrictActor(destroy_error=interrupt)
        rig = LiveRig(actor=actor, world_kwargs={"snapshot_plan": [()]})
        with self.assertRaises(KeyboardInterrupt) as context:
            run_cycle(rig)
        self.assertIs(context.exception, interrupt)
        self.assertEqual(actor.destroy_calls, 1)
        self.assertEqual(rig.world.snapshot_queried_ids, [101])

    def test_destroy_system_exit_is_deferred_until_after_verification(self):
        exit_signal = SystemExit(23)
        actor = StrictActor(destroy_error=exit_signal)
        rig = LiveRig(actor=actor, world_kwargs={"snapshot_plan": [()]})
        with self.assertRaises(SystemExit) as context:
            run_cycle(rig)
        self.assertIs(context.exception, exit_signal)
        self.assertEqual(actor.destroy_calls, 1)
        self.assertEqual(rig.world.snapshot_queried_ids, [101])

    def test_cleanup_interrupt_does_not_replace_primary_keyboard_interrupt(self):
        primary_interrupt = KeyboardInterrupt()
        cleanup_interrupt = KeyboardInterrupt()
        actor = StrictActor(destroy_error=cleanup_interrupt)
        rig = LiveRig(actor=actor, world_kwargs={"snapshot_plan": [()]})
        with unittest.mock.patch.object(
            harness, "verify_ego_identity", side_effect=primary_interrupt
        ):
            with self.assertRaises(KeyboardInterrupt) as context:
                run_cycle(rig)
        self.assertIs(context.exception, primary_interrupt)
        self.assertEqual(actor.destroy_calls, 1)
        self.assertEqual(rig.world.snapshot_queried_ids, [101])

    def test_id_getter_interrupt_also_cleans_before_propagating(self):
        interrupt = KeyboardInterrupt()
        actor = StrictActor(id_error=interrupt)
        rig = LiveRig(actor=actor)
        with self.assertRaises(KeyboardInterrupt) as context:
            run_cycle(rig)
        self.assertIs(context.exception, interrupt)
        self.assertEqual(actor.destroy_calls, 1)
        self.assertEqual(rig.world.snapshot_queried_ids, [])

    def test_local_verification_defect_is_cleanup_error_not_false_success(self):
        rig = LiveRig()
        with unittest.mock.patch.object(
            harness,
            "verify_ego_disappearance",
            side_effect=AssertionError("local defect"),
        ):
            report = run_cycle(rig)
        self.assertIsNone(report.outcome.primary_failure)
        self.assertEqual(len(report.outcome.cleanup_errors), 1)
        self.assertIn("final AssertionError", report.outcome.cleanup_errors[0])
        self.assertIs(
            report.outcome.verification,
            CleanupVerification.UNVERIFIED_DISCONNECTED,
        )

    def test_report_is_frozen(self):
        report = run_cycle(LiveRig())
        with self.assertRaises(FrozenInstanceError):
            report.ego_id = None


class LiveEntryContractTest(unittest.TestCase):
    def report(self, **overrides):
        names = harness.EXPECTED_SENSOR_NAMES
        types = harness.EXPECTED_SENSOR_TYPES
        values = {
            "versions": harness.VersionReport("0.9.16", "0.9.16", True),
            "world_mode": WorldMode.ASYNCHRONOUS,
            "policy": ProgressionPolicy.PASSIVE_WAIT,
            "ego_id": 101,
            "ego_type_id": "vehicle.tesla.model3",
            "ego_role_valid": True,
            "sensor_identities": tuple(
                harness.SensorIdentityRecord(name, 201 + index, type_id)
                for index, (name, type_id) in enumerate(zip(names, types))
            ),
            "attachment_success": True,
            "tick_registration_success": True,
            "readiness_success": True,
            "sensor_health": tuple(
                harness.SensorHealthReport(
                    name=name,
                    attached=True,
                    messages_received=(0 if name in ("collision", "lane_invasion") else 1),
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
                640, 360, 90.0, "RGB8", 1920, 640 * 360 * 3, True
            ),
            "primary_error": None,
            "cleanup_errors": (),
            "final_disappearance": CleanupVerification.CONFIRMED,
            "remaining_owned_actor_ids": (),
        }
        values.update(overrides)
        return harness.SensorSuiteValidationReport(**values)

    def invoke(self, report):
        test = harness.ConnectionSpawnLiveTest(
            "test_connection_and_single_owned_ego_cycle"
        )
        with (
            unittest.mock.patch.dict(
                os.environ, {CARLA_INTEGRATION: "1"}, clear=True
            ),
            unittest.mock.patch.object(
                harness, "run_sensor_suite_validation_cycle", return_value=report
            ),
        ):
            return test.test_connection_and_single_owned_ego_cycle()

    def test_valid_report_passes_the_live_assertion(self):
        self.assertIsNone(self.invoke(self.report()))

    def test_incompatible_versions_and_unsafe_policies_fail(self):
        reports = (
            self.report(
                versions=harness.VersionReport("0.9.16", "0.9.16", False)
            ),
            self.report(
                versions=harness.VersionReport("0.9.15", "0.9.15", True)
            ),
            self.report(policy=ProgressionPolicy.SKIP),
            self.report(
                world_mode=WorldMode.SYNCHRONOUS,
                policy=ProgressionPolicy.PASSIVE_WAIT,
            ),
            self.report(
                world_mode=WorldMode.ASYNCHRONOUS,
                policy=ProgressionPolicy.OWNED_TICK,
            ),
        )
        for report in reports:
            with self.subTest(report=report):
                with self.assertRaises(AssertionError):
                    self.invoke(report)

    def test_missing_or_malformed_ego_identity_fails(self):
        class VehicleStringSubclass(str):
            pass

        reports = (
            self.report(ego_id=None),
            self.report(ego_id=True),
            self.report(ego_type_id=None),
            self.report(ego_type_id="walker.pedestrian"),
            self.report(ego_type_id=VehicleStringSubclass("vehicle.fake")),
        )
        for report in reports:
            with self.subTest(report=report):
                with self.assertRaises(AssertionError):
                    self.invoke(report)

    def test_failed_or_unverified_cleanup_and_primary_failure_fail(self):
        reports = (
            self.report(primary_error=ValidationFailure("validation", "failed")),
            self.report(cleanup_errors=("ego RuntimeError: failed",)),
            self.report(
                final_disappearance=CleanupVerification.FAILED_ACTORS_REMAIN,
                remaining_owned_actor_ids=(101,),
            ),
            self.report(
                final_disappearance=CleanupVerification.UNVERIFIED_DISCONNECTED
            ),
        )
        for report in reports:
            with self.subTest(report=report):
                with self.assertRaises(AssertionError):
                    self.invoke(report)

    def test_sensor_topology_attachment_health_readiness_and_camera_failures_fail(self):
        bad_health = list(self.report().sensor_health)
        bad_health[0] = harness.SensorHealthReport(
            "gnss", True, 1, 1, True
        )
        reports = (
            self.report(sensor_identities=self.report().sensor_identities[:-1]),
            self.report(attachment_success=False),
            self.report(tick_registration_success=False),
            self.report(readiness_success=False),
            self.report(sensor_health=tuple(bad_health)),
            self.report(camera=None),
            self.report(ego_telemetry_present=False),
            self.report(gnss_present=False),
            self.report(imu_present=False),
            self.report(camera_present=False),
        )
        for report in reports:
            with self.subTest(report=report):
                with self.assertRaises(AssertionError):
                    self.invoke(report)


if __name__ == "__main__":
    unittest.main()
