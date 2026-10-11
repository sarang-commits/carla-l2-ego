"""Fake-only tests for the inert Mission 6B live observation harness."""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace

from tests import integration_perception_runtime as harness


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return self.now

    def advance(self, seconds: float = 0.01) -> None:
        self.now += seconds


def fake_stamp(frame: int, monotonic: float):
    return SimpleNamespace(
        frame=frame,
        sim_time=float(frame) / 20.0,
        monotonic=monotonic,
    )


def runtime_metrics():
    return SimpleNamespace(
        inputs_received=0,
        inputs_accepted=0,
        inputs_processed=0,
        duplicate_inputs_rejected=0,
        out_of_order_inputs_rejected=0,
        stale_inputs_rejected=0,
        source_frames_inferred_dropped=0,
        pending_frames_coalesced=0,
        pending_inputs_discarded=0,
        inputs_abandoned=0,
        perception_successes=0,
        perception_failures=0,
        deadline_misses=0,
        last_processed_source_stamp=None,
    )


def bridge_metrics():
    return SimpleNamespace(
        submissions_received=0,
        submissions_admitted=0,
        runtime_submissions_accepted=0,
        runtime_submissions_rejected=0,
        submissions_rejected_closed=0,
        submission_failures=0,
        current_in_flight=0,
        maximum_in_flight=0,
        last_callback_to_submit_latency_seconds=None,
        maximum_callback_to_submit_latency_seconds=None,
        last_submitted_source_stamp=None,
    )


class FakeRuntime:
    def __init__(self, serial: int) -> None:
        self.serial = serial
        self.state = SimpleNamespace(value="stopped")
        self.metrics = runtime_metrics()
        self.fault = None
        self.alive = False


class FakeBridge:
    def __init__(self, serial: int, runtime: FakeRuntime) -> None:
        self.serial = serial
        self.runtime = runtime
        self.state = SimpleNamespace(value="open")
        self.metrics = bridge_metrics()
        self.fault = None


class FakeSuite:
    def __init__(self, serial: int, bridge: FakeBridge) -> None:
        self.serial = serial
        self.bridge = bridge
        self.sensor_ids = tuple(serial * 1000 + offset for offset in range(1, 6))


class FakeEgo:
    def __init__(self, serial: int) -> None:
        self.serial = serial
        self.actor_id = serial * 100 + 1


class FakeOperations:
    """No fake method imports or resembles CARLA; only lifecycle is modeled."""

    def __init__(
        self,
        clock: FakeClock,
        *,
        progress_error: BaseException | None = None,
        drain_error: BaseException | None = None,
        stop_error: BaseException | None = None,
        remaining_actor_ids=(),
        worker_leak: bool = False,
        runtime_accept: bool = True,
        perception_input_usable: bool = True,
        seconds_per_progress: float = 0.01,
    ) -> None:
        self.clock = clock
        self.progress_error = progress_error
        self.drain_error = drain_error
        self.stop_error = stop_error
        self.remaining_actor_ids = tuple(remaining_actor_ids)
        self.worker_leak = worker_leak
        self.runtime_accept = runtime_accept
        self.perception_input_usable_result = perception_input_usable
        self.seconds_per_progress = seconds_per_progress
        self.events: list[tuple[int, str]] = []
        self.contexts: list[dict] = []
        self.runtime_serials: list[int] = []
        self.bridge_serials: list[int] = []
        self.suite_serials: list[int] = []
        self._progress_error_raised = False
        self._drain_error_raised = False
        self._stop_error_raised = False

    def _event(self, owner, name: str) -> None:
        serial = owner["serial"] if isinstance(owner, dict) else owner.serial
        self.events.append((serial, name))

    def prepare_cycle(self, _config):
        serial = len(self.contexts) + 1
        context = {"serial": serial}
        self.contexts.append(context)
        self._event(context, "prepare_cycle")
        return context

    def spawn_ego(self, context, _config):
        self._event(context, "spawn_ego")
        ego = FakeEgo(context["serial"])
        context["ego"] = ego
        return ego

    @staticmethod
    def recover_owned_ego(context):
        return context.get("ego")

    @staticmethod
    def ego_actor_id(ego):
        return ego.actor_id

    def create_runtime(self, _config):
        context = self.contexts[-1]
        self._event(context, "create_runtime")
        runtime = FakeRuntime(context["serial"])
        context["runtime"] = runtime
        self.runtime_serials.append(runtime.serial)
        return runtime

    def create_bridge(self, runtime, _config, monotonic):
        self.assert_same_clock(monotonic)
        self._event(runtime, "create_bridge")
        bridge = FakeBridge(runtime.serial, runtime)
        self.contexts[-1]["bridge"] = bridge
        self.bridge_serials.append(bridge.serial)
        return bridge

    def create_suite(self, bridge, _config, monotonic):
        self.assert_same_clock(monotonic)
        self._event(bridge, "create_suite")
        suite = FakeSuite(bridge.serial, bridge)
        self.contexts[-1]["suite"] = suite
        self.suite_serials.append(suite.serial)
        return suite

    def assert_same_clock(self, value) -> None:
        if value is not self.clock:
            raise AssertionError("harness did not preserve the injected clock basis")

    def attach_suite(self, suite, _context, _ego, _config):
        self._event(suite, "attach_suite")

    def start_runtime(self, runtime):
        self._event(runtime, "start_runtime")
        runtime.alive = True
        runtime.state = SimpleNamespace(value="running")

    def progress_once(self, context, deadline):
        self._event(context, "progress_once")
        if self.progress_error is not None and not self._progress_error_raised:
            self._progress_error_raised = True
            raise self.progress_error
        deadline.raise_if_expired("fake_progress")
        self.clock.advance(self.seconds_per_progress)
        bridge = context["bridge"]
        runtime = context["runtime"]
        metrics = bridge.metrics
        metrics.submissions_received += 1
        metrics.submissions_admitted += 1
        if self.runtime_accept:
            metrics.runtime_submissions_accepted += 1
        else:
            metrics.runtime_submissions_rejected += 1
        metrics.maximum_in_flight = max(metrics.maximum_in_flight, 1)
        metrics.last_callback_to_submit_latency_seconds = 0.001
        metrics.maximum_callback_to_submit_latency_seconds = 0.001
        stamp = fake_stamp(metrics.submissions_admitted, self.clock.now)
        metrics.last_submitted_source_stamp = stamp
        if self.runtime_accept:
            runtime.metrics.inputs_received += 1
            runtime.metrics.inputs_accepted += 1
            runtime.metrics.inputs_processed += 1
            runtime.metrics.perception_successes += 1
            runtime.metrics.last_processed_source_stamp = stamp

    def sensor_actor_ids(self, suite):
        self._event(suite, "sensor_actor_ids")
        return suite.sensor_ids

    def perception_input_usable(self, _runtime):
        return self.perception_input_usable_result

    def close_bridge_admission(self, bridge):
        self._event(bridge, "close_bridge_admission")
        bridge.state = SimpleNamespace(value="closing")

    def drain_bridge(self, bridge, _timeout_seconds):
        self._event(bridge, "drain_bridge")
        if self.drain_error is not None and not self._drain_error_raised:
            self._drain_error_raised = True
            bridge.metrics.current_in_flight = 1
            raise self.drain_error
        bridge.metrics.current_in_flight = 0
        bridge.state = SimpleNamespace(value="closed")

    def destroy_suite(self, suite):
        self._event(suite, "destroy_suite")

    def stop_runtime(self, runtime):
        self._event(runtime, "stop_runtime")
        if self.stop_error is not None and not self._stop_error_raised:
            self._stop_error_raised = True
            # Model a worker which finished immediately after the bounded stop
            # timed out; the timeout remains a cleanup failure in the report.
            runtime.alive = self.worker_leak
            raise self.stop_error
        runtime.alive = self.worker_leak
        runtime.state = SimpleNamespace(value="stopped")

    def destroy_ego(self, ego):
        self._event(ego, "destroy_ego")

    def verify_disappearance(
        self, context, actor_ids, _timeout_seconds, monotonic
    ):
        self.assert_same_clock(monotonic)
        self._event(context, "verify_disappearance")
        if self.remaining_actor_ids:
            return harness.DisappearanceEvidence(
                verified=False,
                remaining_actor_ids=self.remaining_actor_ids,
            )
        if set(actor_ids) != {
            context["ego"].actor_id,
            *context["suite"].sensor_ids,
        }:
            raise AssertionError("not every owned actor id was verified")
        return harness.DisappearanceEvidence(True, ())

    def release_references(self, context, _suite, _bridge, _ego):
        self._event(context, "release_references")
        # Keep only scalar serial evidence in the fake itself.
        context.pop("ego", None)
        context.pop("suite", None)
        context.pop("bridge", None)

    @staticmethod
    def worker_alive(runtime):
        return runtime.alive


def config(**overrides):
    values = dict(
        observation_timeout_seconds=10.0,
        cycles=1,
        frame_target=3,
        duration_seconds=None,
        shutdown_timeout_seconds=0.1,
    )
    values.update(overrides)
    return harness.PerceptionIntegrationConfig(**values)


def cleanup_event_names(operations: FakeOperations, cycle: int = 1):
    cleanup_names = {
        "close_bridge_admission",
        "drain_bridge",
        "destroy_suite",
        "stop_runtime",
        "destroy_ego",
        "verify_disappearance",
        "release_references",
    }
    return [
        name
        for serial, name in operations.events
        if serial == cycle and name in cleanup_names
    ]


EXPECTED_OPERATION_CLEANUP_ORDER = [
    "close_bridge_admission",
    "drain_bridge",
    "destroy_suite",
    "stop_runtime",
    "destroy_ego",
    "verify_disappearance",
    "release_references",
]


class ImportAndGateSafetyTest(unittest.TestCase):
    def test_fresh_import_is_inert_and_forbidden_dependency_free(self):
        code = textwrap.dedent(
            """
            import os
            import socket
            import sys
            import threading

            sys.path.insert(0, os.getcwd())
            environment_before = dict(os.environ)
            threads_before = tuple(sorted(t.ident for t in threading.enumerate()))
            real_socket = socket.socket
            real_start = threading.Thread.start
            socket.socket = lambda *a, **k: (_ for _ in ()).throw(AssertionError("socket"))
            threading.Thread.start = lambda *a, **k: (_ for _ in ()).throw(AssertionError("thread"))
            try:
                import tests.integration_perception_runtime as module
            finally:
                socket.socket = real_socket
                threading.Thread.start = real_start
            forbidden = {"carla", "numpy", "pygame", "cv2", "scipy"}
            assert not [name for name in sys.modules if name.partition(".")[0] in forbidden]
            assert "perception.runtime" not in sys.modules
            assert "perception.live_bridge" not in sys.modules
            assert "sensors.suite" not in sys.modules
            assert dict(os.environ) == environment_before
            assert tuple(sorted(t.ident for t in threading.enumerate())) == threads_before
            assert module.integration_enabled({}) is False
            print("IMPORT_SAFE")
            """
        )
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20.0,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "IMPORT_SAFE")

    def test_source_has_no_eager_heavy_or_project_import(self):
        source = pathlib.Path(harness.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        roots = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                roots.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.partition(".")[0])
        self.assertTrue(
            roots.isdisjoint(
                {"carla", "numpy", "pygame", "cv2", "scipy", "perception", "sensors"}
            )
        )

    def test_gate_is_exact_and_precedes_parsing_or_operations(self):
        class ExplodingOperations:
            def __getattribute__(self, _name):
                raise AssertionError("disabled harness touched operations")

        for value in (None, "", "0", "true", "01", 1):
            environment = {harness.CARLA_PORT: "invalid"}
            if value is not None:
                environment[harness.CARLA_INTEGRATION] = value
            with self.subTest(value=value), self.assertRaises(unittest.SkipTest):
                harness.run_live_perception_integration(
                    environment, operations=ExplodingOperations()
                )

    def test_enabled_fake_path_never_calls_importer(self):
        clock = FakeClock()
        operations = FakeOperations(clock)

        def importer(_name):
            raise AssertionError("injected fake path tried to import live code")

        report = harness.run_live_perception_integration(
            {
                harness.CARLA_INTEGRATION: "1",
                harness.CARLA_PERCEPTION_FRAME_TARGET: "1",
            },
            operations=operations,
            importer=importer,
            monotonic=clock,
        )
        self.assertTrue(report.success)


class ConfigurationTest(unittest.TestCase):
    def test_frame_target_and_duration_modes_are_bounded_and_exclusive(self):
        frame = harness.parse_integration_config(
            {harness.CARLA_PERCEPTION_FRAME_TARGET: "10000"}
        )
        self.assertIs(frame.observation_mode, harness.ObservationMode.FRAME_TARGET)
        self.assertEqual(frame.frame_target, 10_000)
        duration = harness.parse_integration_config(
            {harness.CARLA_PERCEPTION_DURATION_SECONDS: "120"}
        )
        self.assertIs(duration.observation_mode, harness.ObservationMode.DURATION)
        self.assertIsNone(duration.frame_target)
        self.assertEqual(duration.duration_seconds, 120.0)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            harness.parse_integration_config(
                {
                    harness.CARLA_PERCEPTION_DURATION_SECONDS: "120",
                    harness.CARLA_PERCEPTION_FRAME_TARGET: "2",
                }
            )

    def test_secondary_bounds_reject_nonfinite_and_oversized_values(self):
        cases = (
            (harness.CARLA_INTEGRATION_CYCLES, "21"),
            (harness.CARLA_PERCEPTION_FRAME_TARGET, "10001"),
            (harness.CARLA_PERCEPTION_DURATION_SECONDS, "nan"),
            (harness.CARLA_PERCEPTION_TARGET_HZ, "inf"),
            (harness.CARLA_PERCEPTION_SHUTDOWN_TIMEOUT_SECONDS, "0"),
        )
        for name, value in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                harness.parse_integration_config({name: value})


class LifecycleTest(unittest.TestCase):
    def test_hostile_recovery_descriptor_cannot_skip_cleanup_tree(self):
        class HostileRecoveryOperations(FakeOperations):
            def spawn_ego(self, context, _config):
                self._event(context, "spawn_ego")
                raise RuntimeError("spawn interrupted before ownership return")

            @property
            def recover_owned_ego(self):
                raise RuntimeError("hostile recovery descriptor")

        clock = FakeClock()
        operations = HostileRecoveryOperations(clock)
        report = harness.run_perception_cycles(
            config(), operations, monotonic=clock
        )
        cycle = report.retained_cycle_reports[0]
        self.assertFalse(report.success)
        self.assertEqual(
            tuple(step.stage for step in cycle.cleanup_steps),
            harness.CLEANUP_ORDER,
        )
        self.assertIn("ego", {fault.component for fault in cycle.faults})

    def test_default_spawn_reraises_deferred_baseexception_after_retention(self):
        class Fatal(BaseException):
            pass

        fatal = Fatal("post-spawn interrupt")
        actor = object()
        owned = SimpleNamespace(actor=actor, pending_base_exception=fatal)

        class HarnessStub:
            class Deadline:
                def __init__(self, _timeout, _clock):
                    pass

            @staticmethod
            def spawn_owned_ego(
                _world,
                _blueprint,
                _deadline,
                *,
                role_name_required,
                ownership_sink,
            ):
                self.assertFalse(role_name_required)
                ownership_sink(actor)
                return owned

            @staticmethod
            def verify_ego_identity(_owned):
                raise AssertionError("pending BaseException was not reraised first")

        operations = harness._DefaultLiveOperations(
            HarnessStub,
            runtime_module=None,
            bridge_module=None,
            suite_module=None,
            environ={},
            monotonic=FakeClock(),
        )
        context = harness._DefaultCycleContext(
            client=object(),
            world=object(),
            policy=object(),
            blueprint=object(),
            role_name_required=False,
        )

        with self.assertRaises(Fatal) as caught:
            operations.spawn_ego(context, config())

        self.assertIs(caught.exception, fatal)
        self.assertIs(context.retained_actor, actor)
        self.assertIs(context.owned_ego, owned)

    def test_default_suite_cleanup_requires_tick_and_complete_sensor_results(self):
        complete_results = tuple(
            SimpleNamespace(
                name=name,
                stopped=True,
                destroyed=True,
                errors=(),
            )
            for name in harness.EXPECTED_SENSOR_NAMES
        )
        reports = (
            SimpleNamespace(
                errors=(),
                tick_unregistered=False,
                results=complete_results,
            ),
            SimpleNamespace(errors=(), tick_unregistered=True, results=()),
        )
        for report in reports:
            with self.subTest(report=report):
                suite = SimpleNamespace(destroy=lambda report=report: report)
                with self.assertRaises(RuntimeError):
                    harness._DefaultLiveOperations.destroy_suite(suite)

    def test_success_uses_fresh_runtime_bridge_and_suite_each_cycle(self):
        clock = FakeClock()
        operations = FakeOperations(clock)
        report = harness.run_perception_cycles(
            config(cycles=3, frame_target=2), operations, monotonic=clock
        )
        self.assertTrue(report.success)
        self.assertEqual(report.attempted_cycle_count, 3)
        self.assertEqual(operations.runtime_serials, [1, 2, 3])
        self.assertEqual(operations.bridge_serials, [1, 2, 3])
        self.assertEqual(operations.suite_serials, [1, 2, 3])
        for cycle in range(1, 4):
            names = [
                name for serial, name in operations.events if serial == cycle
            ]
            self.assertLess(names.index("spawn_ego"), names.index("create_runtime"))
            self.assertLess(names.index("create_runtime"), names.index("create_bridge"))
            self.assertLess(names.index("create_bridge"), names.index("create_suite"))
            self.assertLess(names.index("create_suite"), names.index("attach_suite"))
            self.assertLess(names.index("attach_suite"), names.index("start_runtime"))
            self.assertLess(
                names.index("close_bridge_admission"),
                names.index("sensor_actor_ids"),
            )
            self.assertEqual(
                cleanup_event_names(operations, cycle),
                EXPECTED_OPERATION_CLEANUP_ORDER,
            )
        self.assertTrue(
            all(
                tuple(step.stage for step in cycle.cleanup_steps)
                == harness.CLEANUP_ORDER
                for cycle in report.retained_cycle_reports
            )
        )

    def test_functional_failure_still_reaches_every_cleanup_stage(self):
        clock = FakeClock()
        operations = FakeOperations(clock, progress_error=RuntimeError("observe"))
        report = harness.run_perception_cycles(config(), operations, monotonic=clock)
        cycle = report.retained_cycle_reports[0]
        self.assertFalse(report.success)
        self.assertEqual(
            cleanup_event_names(operations), EXPECTED_OPERATION_CLEANUP_ORDER
        )
        self.assertEqual(
            tuple(step.stage for step in cycle.cleanup_steps), harness.CLEANUP_ORDER
        )
        self.assertTrue(all(step.succeeded for step in cycle.cleanup_steps))
        self.assertEqual(cycle.faults[0].component, "observation")

    def test_bridge_drain_timeout_does_not_skip_suite_runtime_or_ego_cleanup(self):
        clock = FakeClock()
        operations = FakeOperations(
            clock, drain_error=TimeoutError("bridge still in flight")
        )
        report = harness.run_perception_cycles(config(), operations, monotonic=clock)
        cycle = report.retained_cycle_reports[0]
        self.assertFalse(report.success)
        self.assertEqual(
            cleanup_event_names(operations),
            [
                "close_bridge_admission",
                "drain_bridge",
                "destroy_suite",
                "drain_bridge",
                "stop_runtime",
                "destroy_ego",
                "verify_disappearance",
                "release_references",
            ],
        )
        by_stage = {step.stage: step for step in cycle.cleanup_steps}
        self.assertFalse(by_stage[harness.CleanupStage.DRAIN_BRIDGE].succeeded)
        self.assertTrue(by_stage[harness.CleanupStage.DESTROY_SENSOR_SUITE].succeeded)
        self.assertTrue(by_stage[harness.CleanupStage.DESTROY_EGO].succeeded)
        self.assertTrue(
            by_stage[harness.CleanupStage.VERIFY_ACTOR_DISAPPEARANCE].succeeded
        )

    def test_runtime_stop_timeout_does_not_skip_ego_or_disappearance(self):
        clock = FakeClock()
        operations = FakeOperations(
            clock, stop_error=TimeoutError("runtime join timed out")
        )
        report = harness.run_perception_cycles(config(), operations, monotonic=clock)
        cycle = report.retained_cycle_reports[0]
        self.assertFalse(report.success)
        self.assertEqual(
            cleanup_event_names(operations),
            [
                "close_bridge_admission",
                "drain_bridge",
                "destroy_suite",
                "stop_runtime",
                "destroy_ego",
                "verify_disappearance",
                "stop_runtime",
                "release_references",
            ],
        )
        by_stage = {step.stage: step for step in cycle.cleanup_steps}
        self.assertFalse(
            by_stage[harness.CleanupStage.STOP_PERCEPTION_RUNTIME].succeeded
        )
        self.assertTrue(by_stage[harness.CleanupStage.DESTROY_EGO].succeeded)
        self.assertTrue(
            by_stage[harness.CleanupStage.VERIFY_ACTOR_DISAPPEARANCE].succeeded
        )
        self.assertFalse(cycle.runtime_worker_alive_after_cleanup)

    def test_actor_disappearance_failure_is_preserved(self):
        clock = FakeClock()
        operations = FakeOperations(clock, remaining_actor_ids=(101, 1005))
        report = harness.run_perception_cycles(config(), operations, monotonic=clock)
        cycle = report.retained_cycle_reports[0]
        self.assertFalse(report.success)
        self.assertFalse(cycle.disappearance_verified)
        self.assertEqual(cycle.remaining_actor_ids, (101, 1005))
        self.assertIn(
            "actor_disappearance", {fault.component for fault in cycle.faults}
        )

    def test_all_runtime_rejections_cannot_produce_a_live_success(self):
        clock = FakeClock()
        operations = FakeOperations(
            clock, runtime_accept=False, seconds_per_progress=0.01
        )
        report = harness.run_perception_cycles(
            config(frame_target=1, observation_timeout_seconds=0.03),
            operations,
            monotonic=clock,
        )
        cycle = report.retained_cycle_reports[0]
        self.assertFalse(report.success)
        self.assertFalse(cycle.observation_completed)
        self.assertGreater(cycle.accepted_camera_snapshots, 0)
        self.assertEqual(cycle.bridge_metrics.runtime_submissions_accepted, 0)
        self.assertEqual(cycle.runtime_metrics.inputs_accepted, 0)
        self.assertIn("observation", {fault.component for fault in cycle.faults})
        self.assertEqual(
            cleanup_event_names(operations), EXPECTED_OPERATION_CLEANUP_ORDER
        )

    def test_unusable_required_rgb_evidence_cannot_produce_live_success(self):
        clock = FakeClock()
        operations = FakeOperations(
            clock,
            perception_input_usable=False,
            seconds_per_progress=0.01,
        )
        report = harness.run_perception_cycles(
            config(frame_target=1, observation_timeout_seconds=0.03),
            operations,
            monotonic=clock,
        )
        cycle = report.retained_cycle_reports[0]

        self.assertFalse(report.success)
        self.assertFalse(cycle.observation_completed)
        self.assertGreater(cycle.runtime_metrics.perception_successes, 0)
        self.assertIn("sensor_health", {fault.component for fault in cycle.faults})
        self.assertEqual(
            cleanup_event_names(operations), EXPECTED_OPERATION_CLEANUP_ORDER
        )

    def test_faulted_bridge_report_is_truthful_while_cleanup_continues(self):
        class FaultingOperations(FakeOperations):
            def progress_once(self, context, deadline):
                super().progress_once(context, deadline)
                bridge = context["bridge"]
                bridge.state = SimpleNamespace(value="faulted")
                bridge.fault = SimpleNamespace(
                    stage="runtime_submit",
                    exception_type="builtins.RuntimeError",
                    message="submit failed",
                )

            def close_bridge_admission(self, bridge):
                self._event(bridge, "close_bridge_admission")

            def drain_bridge(self, bridge, _timeout_seconds):
                self._event(bridge, "drain_bridge")
                bridge.metrics.current_in_flight = 0

        clock = FakeClock()
        operations = FaultingOperations(clock)
        report = harness.run_perception_cycles(
            config(frame_target=1), operations, monotonic=clock
        )
        cycle = report.retained_cycle_reports[0]

        self.assertFalse(report.success)
        self.assertFalse(cycle.success)
        self.assertEqual(cycle.bridge_state, "faulted")
        self.assertIn("bridge", {fault.component for fault in cycle.faults})
        self.assertEqual(
            cleanup_event_names(operations), EXPECTED_OPERATION_CLEANUP_ORDER
        )
        self.assertTrue(all(step.succeeded for step in cycle.cleanup_steps))

    def test_final_worker_leak_is_a_failed_verdict(self):
        clock = FakeClock()
        operations = FakeOperations(clock, worker_leak=True)
        report = harness.run_perception_cycles(config(), operations, monotonic=clock)
        cycle = report.retained_cycle_reports[0]
        self.assertFalse(report.success)
        self.assertTrue(cycle.runtime_worker_alive_after_cleanup)
        self.assertFalse(cycle.cleanup_steps[-1].succeeded)
        self.assertEqual(
            cycle.cleanup_steps[-1].stage,
            harness.CleanupStage.VERIFY_RUNTIME_WORKER,
        )


class BoundedReportingTest(unittest.TestCase):
    def test_ten_thousand_observations_retain_only_first_and_final_samples(self):
        clock = FakeClock()
        operations = FakeOperations(clock, seconds_per_progress=0.001)
        report = harness.run_perception_cycles(
            config(
                frame_target=10_000,
                observation_timeout_seconds=20.0,
            ),
            operations,
            monotonic=clock,
        )
        cycle = report.retained_cycle_reports[0]
        self.assertTrue(report.success)
        self.assertEqual(cycle.accepted_camera_snapshots, 10_000)
        self.assertLessEqual(
            len(cycle.samples),
            harness.FIRST_RETAINED_SAMPLES + harness.FINAL_RETAINED_SAMPLES,
        )
        self.assertGreater(cycle.omitted_sample_count, 9_000)
        self.assertEqual(cycle.bridge_metrics.current_in_flight, 0)
        self.assertEqual(cycle.runtime_metrics.inputs_processed, 10_000)

    def test_cycle_report_retention_is_first_plus_final_only(self):
        clock = FakeClock()
        operations = FakeOperations(clock)
        report = harness.run_perception_cycles(
            config(cycles=20, frame_target=1), operations, monotonic=clock
        )
        self.assertTrue(report.success)
        self.assertEqual(report.attempted_cycle_count, 20)
        self.assertEqual(len(report.retained_cycle_reports), 12)
        self.assertEqual(report.omitted_cycle_report_count, 8)
        self.assertEqual(
            tuple(item.cycle_number for item in report.retained_cycle_reports),
            (1, 2, 3, 4, 13, 14, 15, 16, 17, 18, 19, 20),
        )


if __name__ == "__main__":
    unittest.main()
