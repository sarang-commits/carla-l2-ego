"""Strict offline regressions for Mission 4G.3 (no CARLA server required)."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import types
import unittest
import unittest.mock
from dataclasses import FrozenInstanceError, replace

from sensors.collision import COLLISION_BLUEPRINT
from sensors.gnss import GNSS_BLUEPRINT
from sensors.imu import IMU_BLUEPRINT
from sensors.lane_invasion import LANE_INVASION_BLUEPRINT
from sensors.rgb_camera import RGB_CAMERA_BLUEPRINT
from sensors.suite import SensorSuite
from telemetry.state import (
    CameraFrame,
    CameraFrameMetadata,
    EgoKinematics,
    GnssState,
    ImuState,
    SampleStamp,
    SensorHealth,
    TelemetrySnapshot,
    TransformState,
    Vector3,
    EulerDegrees,
)
from tests import integration_sensor_suite as harness
from tests.fakes import FakeVehicle, FakeWorld
from tests.test_integration_harness import (
    ENABLED_ENVIRON,
    FakeClock,
    LiveRig,
    StrictActor,
)


EXPECTED_NAMES = ("gnss", "imu", "collision", "lane_invasion", "rgb_camera")
EXPECTED_TYPES = (
    GNSS_BLUEPRINT,
    IMU_BLUEPRINT,
    COLLISION_BLUEPRINT,
    LANE_INVASION_BLUEPRINT,
    RGB_CAMERA_BLUEPRINT,
)
SENSOR_IDS = (1001, 1002, 1003, 1004, 1005)
RGB_PAYLOAD = bytes(640 * 360 * 3)


def require_harness_api(testcase, name):
    value = getattr(harness, name, None)
    testcase.assertIsNotNone(value, f"Mission 4G.3 API {name!r} is missing")
    return value


def ready_snapshot(*, error_name=None, camera=None):
    stamp = SampleStamp(7, 1.25, 2.5)
    transform = TransformState(
        Vector3(1.0, 2.0, 3.0), EulerDegrees(0.0, 0.0, 0.0)
    )
    ego = EgoKinematics(
        stamp, transform, Vector3(0.0, 0.0, 0.0), 0.0, 0.0, True
    )
    gnss = GnssState(stamp, 1.0, 2.0, 3.0, transform)
    imu = ImuState(
        stamp,
        Vector3(0.0, 0.0, 9.81),
        Vector3(0.0, 0.0, 0.0),
        0.0,
        0.0,
        transform,
    )
    if camera is None:
        camera = CameraFrame(
            CameraFrameMetadata(
                stamp=stamp,
                width_px=640,
                height_px=360,
                horizontal_fov_deg=90.0,
                pixel_format="RGB8",
                row_stride_bytes=1920,
                transform=transform,
            ),
            RGB_PAYLOAD,
        )
    counts = {"gnss": 1, "imu": 1, "collision": 0, "lane_invasion": 0, "rgb_camera": 1}
    health = tuple(
        SensorHealth(
            name=name,
            attached=True,
            messages_received=counts[name],
            error_count=1 if name == error_name else 0,
            last_error="callback failed" if name == error_name else None,
        )
        for name in reversed(EXPECTED_NAMES)
    )
    return TelemetrySnapshot(
        revision=4,
        captured_monotonic=2.5,
        ego=ego,
        gnss=gnss,
        imu=imu,
        sensors=health,
        camera=camera,
    )


class StrictVehicle:
    id = 77
    type_id = "vehicle.test.strict"

    @property
    def is_alive(self):
        raise AssertionError("SensorSuite must not read actor.is_alive")


class SensorTopologyApiTest(unittest.TestCase):
    def attach(self, **world_kwargs):
        world = FakeWorld(**world_kwargs)
        suite = SensorSuite()
        report = suite.attach(world, StrictVehicle())
        return world, suite, report

    def test_public_topology_is_frozen_carla_free_and_exact(self):
        _world, suite, _report = self.attach()
        topology = suite.topology_snapshot()
        self.assertFalse(hasattr(topology, "__dict__"))
        self.assertEqual(tuple(item.name for item in topology.identities), EXPECTED_NAMES)
        self.assertEqual(tuple(item.actor_id for item in topology.identities), SENSOR_IDS)
        self.assertEqual(tuple(item.type_id for item in topology.identities), EXPECTED_TYPES)
        self.assertTrue(all(item.identity_errors == () for item in topology.identities))
        for item in topology.identities:
            self.assertFalse(hasattr(item, "__dict__"))
            self.assertIs(type(item.name), str)
            self.assertIs(type(item.actor_id), int)
            self.assertIs(type(item.type_id), str)
            with self.assertRaises(FrozenInstanceError):
                item.actor_id = 9
        suite.destroy()

    def test_exact_five_blueprints_order_parent_rigidity_and_mounts(self):
        world, suite, report = self.attach()
        self.assertEqual(tuple(call[0] for call in world.spawn_calls), EXPECTED_TYPES)
        self.assertEqual(tuple(result.name for result in report.results), EXPECTED_NAMES)
        self.assertTrue(all(call[2] is not None for call in world.spawn_calls))
        parent = world.spawn_calls[0][2]
        self.assertTrue(all(call[2] is parent for call in world.spawn_calls))
        rigid = world.spawn_calls[0][3]
        self.assertTrue(all(call[3] is rigid for call in world.spawn_calls))
        for _name, transform, _parent, _rigid, _actor in world.spawn_calls[:4]:
            self.assertEqual(
                (
                    transform.location.x,
                    transform.location.y,
                    transform.location.z,
                    transform.rotation.pitch,
                    transform.rotation.yaw,
                    transform.rotation.roll,
                ),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
        camera = world.spawn_calls[4][1]
        for actual, expected in zip(
            (
                camera.location.x,
                camera.location.y,
                camera.location.z,
                camera.rotation.pitch,
                camera.rotation.yaw,
                camera.rotation.roll,
            ),
            (1.5, 0.0, 1.7, -5.0, 0.0, 0.0),
        ):
            self.assertAlmostEqual(actual, expected, places=5)
        blueprint = world.blueprints[RGB_CAMERA_BLUEPRINT]
        self.assertEqual(blueprint.attributes["image_size_x"], "640")
        self.assertEqual(blueprint.attributes["image_size_y"], "360")
        self.assertEqual(blueprint.attributes["fov"], "90.0")
        suite.destroy()

    def test_partial_failure_at_every_blueprint_spawn_and_listen_stage(self):
        for blueprint in EXPECTED_TYPES:
            with self.subTest(stage="blueprint", blueprint=blueprint):
                world, suite, report = self.attach(missing_blueprints=(blueprint,))
                record_names = tuple(x.name for x in suite.topology_snapshot().identities)
                self.assertNotIn(EXPECTED_NAMES[EXPECTED_TYPES.index(blueprint)], record_names)
                self.assertEqual(sum(r.attached is False for r in report.results), 1)
                suite.destroy()
            with self.subTest(stage="spawn", blueprint=blueprint):
                world, suite, report = self.attach(spawn_fail=(blueprint,))
                record_names = tuple(x.name for x in suite.topology_snapshot().identities)
                self.assertNotIn(EXPECTED_NAMES[EXPECTED_TYPES.index(blueprint)], record_names)
                self.assertEqual(sum(r.attached is False for r in report.results), 1)
                suite.destroy()
            with self.subTest(stage="listen", blueprint=blueprint):
                world, suite, report = self.attach(fail_listen=(blueprint,))
                self.assertEqual(
                    tuple(x.name for x in suite.topology_snapshot().identities),
                    EXPECTED_NAMES,
                )
                self.assertEqual(sum(r.attached is False for r in report.results), 1)
                suite.destroy()

    def test_tick_registration_success_and_failure_are_explicit(self):
        _world, suite, report = self.attach()
        self.assertIs(report.tick_registered, True)
        self.assertIsNone(report.tick_error)
        suite.destroy()

        _world, suite, report = self.attach(fail_on_tick=True)
        self.assertIs(report.tick_registered, False)
        self.assertIsInstance(report.tick_error, str)
        self.assertTrue(report.all_attached)
        suite.destroy()


class IdentityWorld(FakeWorld):
    def __init__(self, identity_by_blueprint):
        super().__init__()
        self.identity_by_blueprint = dict(identity_by_blueprint)

    def spawn_actor(self, blueprint, transform, attach_to=None, attachment_type=None):
        actor = super().spawn_actor(blueprint, transform, attach_to, attachment_type)
        actor.id, actor.type_id = self.identity_by_blueprint.get(
            blueprint.id, (actor.id, actor.type_id)
        )
        return actor


class SensorIdentityValidationTest(unittest.TestCase):
    def topology(self, changes=None):
        changes = {} if changes is None else dict(changes)
        identity = {
            blueprint: (actor_id, blueprint)
            for blueprint, actor_id in zip(EXPECTED_TYPES, SENSOR_IDS)
        }
        identity.update(changes)
        world = IdentityWorld(identity)
        suite = SensorSuite()
        suite.attach(world, FakeVehicle(actor_id=77))
        return suite, suite.topology_snapshot()

    def validate(self, topology, ego_id=77):
        validate = require_harness_api(self, "validate_sensor_topology")
        return validate(topology, ego_id)

    def test_valid_topology_copies_five_records(self):
        suite, topology = self.topology()
        copied = self.validate(topology)
        self.assertEqual(tuple(x.name for x in copied), EXPECTED_NAMES)
        self.assertEqual(tuple(x.actor_id for x in copied), SENSOR_IDS)
        suite.destroy()

    def test_invalid_bool_missing_nonbuiltin_duplicate_conflicting_and_ego_ids_fail(self):
        class IntSubclass(int):
            pass

        cases = (
            {GNSS_BLUEPRINT: (True, GNSS_BLUEPRINT)},
            {GNSS_BLUEPRINT: (None, GNSS_BLUEPRINT)},
            {GNSS_BLUEPRINT: (IntSubclass(1001), GNSS_BLUEPRINT)},
            {IMU_BLUEPRINT: (1001, IMU_BLUEPRINT)},
            {GNSS_BLUEPRINT: (77, GNSS_BLUEPRINT)},
            {GNSS_BLUEPRINT: (1001, IMU_BLUEPRINT)},
            {GNSS_BLUEPRINT: (1001, None)},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                suite, topology = self.topology(changes)
                with self.assertRaises(ValueError):
                    self.validate(topology)
                suite.destroy()


class ReadinessAndCameraValidationTest(unittest.TestCase):
    def evaluate(self, snapshot):
        evaluate = require_harness_api(self, "evaluate_sensor_readiness")
        return evaluate(snapshot)

    def test_ready_with_zero_event_messages_and_health_compared_by_name(self):
        evaluation = self.evaluate(ready_snapshot())
        self.assertIs(evaluation.ready, True)
        self.assertIsNone(evaluation.failure)

    def test_each_required_sample_can_be_incomplete(self):
        base = ready_snapshot()
        cases = (
            replace(base, ego=None),
            replace(base, gnss=None),
            replace(base, imu=None),
            replace(base, camera=None),
        )
        for snapshot in cases:
            with self.subTest(snapshot=snapshot):
                evaluation = self.evaluate(snapshot)
                self.assertIs(evaluation.ready, False)
                self.assertIsNone(evaluation.failure)

    def test_dead_ego_bad_health_or_callback_error_is_fatal(self):
        base = ready_snapshot()
        dead_ego = replace(base.ego, vehicle_alive=False)
        missing_health = replace(base, sensors=base.sensors[1:])
        detached = replace(
            base,
            sensors=tuple(
                replace(item, attached=False) if item.name == "imu" else item
                for item in base.sensors
            ),
        )
        for snapshot in (
            replace(base, ego=dead_ego),
            missing_health,
            detached,
            ready_snapshot(error_name="gnss"),
        ):
            with self.subTest(snapshot=snapshot):
                evaluation = self.evaluate(snapshot)
                self.assertIs(evaluation.ready, False)
                self.assertIsNotNone(evaluation.failure)

    def test_callback_error_still_fails_after_a_later_valid_sample(self):
        snapshot = ready_snapshot(error_name="gnss")
        gnss_health = next(x for x in snapshot.sensors if x.name == "gnss")
        self.assertGreaterEqual(gnss_health.messages_received, 1)
        self.assertIsNotNone(snapshot.gnss)
        self.assertIsNotNone(self.evaluate(snapshot).failure)

    def test_camera_metadata_payload_transform_and_reference_identity(self):
        validate = require_harness_api(self, "validate_camera_snapshot")
        snapshot = ready_snapshot()
        state = harness.new_cycle_state(1)
        state.snapshot_reference = snapshot
        state.camera_reference = snapshot.camera
        report = validate(snapshot, state)
        self.assertEqual(
            (
                report.width_px,
                report.height_px,
                report.horizontal_fov_deg,
                report.pixel_format,
                report.row_stride_bytes,
                report.payload_length,
                report.transform_present,
            ),
            (640, 360, 90.0, "RGB8", 1920, 640 * 360 * 3, True),
        )

    def test_each_bad_camera_field_and_wrong_reference_fails(self):
        validate = require_harness_api(self, "validate_camera_snapshot")
        base = ready_snapshot()
        metadata = base.camera.metadata
        bad_cameras = (
            replace(base.camera, metadata=replace(metadata, width_px=639)),
            replace(base.camera, metadata=replace(metadata, height_px=359)),
            replace(base.camera, metadata=replace(metadata, horizontal_fov_deg=89.0)),
            replace(base.camera, metadata=replace(metadata, pixel_format="BGRA8")),
            replace(base.camera, metadata=replace(metadata, row_stride_bytes=1919)),
            replace(base.camera, metadata=replace(metadata, transform=None)),
            replace(base.camera, rgb_bytes=RGB_PAYLOAD[:-1]),
        )
        for camera in bad_cameras:
            snapshot = replace(base, camera=camera)
            state = harness.new_cycle_state(1)
            state.camera_reference = snapshot.camera
            with self.subTest(camera=camera):
                with self.assertRaises(ValueError):
                    validate(snapshot, state)
        state = harness.new_cycle_state(1)
        state.camera_reference = object()
        with self.assertRaises(ValueError):
            validate(base, state)


class _FrameSnapshot:
    """One frozen frame's membership, observed only through has_actor."""

    def __init__(self, world, present_ids):
        self._world = world
        self._present_ids = frozenset(present_ids)

    def has_actor(self, actor_id):
        self._world.queries.append(actor_id)
        return actor_id in self._present_ids


class SnapshotWorld:
    def __init__(
        self,
        registry,
        clock=None,
        remove_plan=(),
        snapshot_error=None,
        progress_error=None,
    ):
        self.registry = dict(registry)
        self.clock = clock
        self.remove_plan = list(remove_plan)
        self.snapshot_error = snapshot_error
        self.progress_error = progress_error
        self.queries = []
        self.snapshot_reads = 0
        self.progress_calls = []
        self.progressions = 0

    def get_snapshot(self):
        if self.snapshot_error is not None:
            raise self.snapshot_error
        self.snapshot_reads += 1
        return _FrameSnapshot(self, self.registry)

    def get_actor(self, _actor_id):
        raise AssertionError("cached actor lookup is forbidden")

    def wait_for_tick(self, timeout):
        return self._progress("wait_for_tick", timeout)

    def tick(self, timeout):
        return self._progress("tick", timeout)

    def _progress(self, method, timeout):
        if self.progress_error is not None:
            raise self.progress_error
        self.progress_calls.append(method)
        self.progressions += 1
        if self.remove_plan:
            self.registry.pop(self.remove_plan.pop(0), None)
        if self.clock is not None:
            self.clock.value += min(timeout, 0.6)
        return self.progressions

    def get_actors(self):
        raise AssertionError("enumeration is forbidden")


class MultiActorDisappearanceTest(unittest.TestCase):
    def verify(self, world, ids, clock=None, timeout=1.0, policy=None):
        verify = require_harness_api(self, "verify_owned_actors_disappearance")
        clock = FakeClock(0.0) if clock is None else clock
        return verify(
            world,
            ids,
            harness.ProgressionPolicy.PASSIVE_WAIT if policy is None else policy,
            harness.Deadline(timeout, clock),
        )

    def test_all_absent_are_checked_once_and_confirm_at_expired_boundary(self):
        clock = FakeClock(5.0)
        deadline = harness.Deadline(1.0, clock)
        clock.value = 6.0
        world = SnapshotWorld({})
        verify = require_harness_api(self, "verify_owned_actors_disappearance")
        result = verify(
            world,
            SENSOR_IDS + (77,),
            harness.ProgressionPolicy.PASSIVE_WAIT,
            deadline,
        )
        self.assertIs(result.verification, harness.CleanupVerification.CONFIRMED)
        self.assertEqual(world.queries, list(SENSOR_IDS + (77,)))
        self.assertEqual(world.snapshot_reads, 1)
        self.assertEqual(world.progressions, 0)
        self.assertEqual(result.remaining_actor_ids, ())

    def test_staggered_removal_queries_only_still_unconfirmed_ids(self):
        clock = FakeClock(0.0)
        world = SnapshotWorld(
            {actor_id: object() for actor_id in (1, 2, 3)},
            clock=clock,
            remove_plan=(1, 2, 3),
        )
        result = self.verify(world, (1, 2, 3), clock=clock, timeout=5.0)
        self.assertIs(result.verification, harness.CleanupVerification.CONFIRMED)
        self.assertEqual(world.queries.count(1), 2)
        self.assertEqual(world.queries.count(2), 3)
        self.assertEqual(world.queries.count(3), 4)
        self.assertEqual(world.progressions, 3)

    def test_persistent_actors_report_exact_live_remaining_order(self):
        clock = FakeClock(0.0)
        owned_ids = (25, 26, 27, 28, 29, 24)
        world = SnapshotWorld(
            {actor_id: object() for actor_id in owned_ids}, clock=clock
        )
        result = self.verify(world, owned_ids, clock=clock)
        self.assertIs(
            result.verification,
            harness.CleanupVerification.FAILED_ACTORS_REMAIN,
        )
        self.assertEqual(result.remaining_actor_ids, owned_ids)
        self.assertGreaterEqual(world.progressions, 1)

    def test_only_literal_false_membership_counts_as_absent(self):
        class JunkMembershipWorld(SnapshotWorld):
            def __init__(self, results, clock):
                super().__init__({}, clock=clock)
                self._results = list(results)

            def get_snapshot(self):
                self.snapshot_reads += 1
                world = self
                result = self._results.pop(0)

                class _JunkSnapshot:
                    def has_actor(self, actor_id):
                        world.queries.append(actor_id)
                        return result

                return _JunkSnapshot()

        clock = FakeClock(0.0)
        world = JunkMembershipWorld([0, None, False], clock)
        result = self.verify(world, (9,), clock=clock, timeout=5.0)
        self.assertIs(result.verification, harness.CleanupVerification.CONFIRMED)
        self.assertEqual(world.snapshot_reads, 3)
        self.assertEqual(world.progressions, 2)

    def test_exact_deadline_equality_fails_without_a_second_admission_read(self):
        clock = FakeClock(0.0)
        world = SnapshotWorld({1: object()}, clock=clock)
        result = self.verify(world, (1,), clock=clock, timeout=0.6)
        self.assertIs(
            result.verification,
            harness.CleanupVerification.FAILED_ACTORS_REMAIN,
        )
        self.assertEqual(result.remaining_actor_ids, (1,))
        self.assertEqual(world.progressions, 1)
        self.assertEqual(world.snapshot_reads, 2)

    def test_owned_tick_world_progresses_only_through_tick(self):
        clock = FakeClock(0.0)
        world = SnapshotWorld(
            {1: object()}, clock=clock, remove_plan=(1,)
        )
        result = self.verify(
            world,
            (1,),
            clock=clock,
            timeout=5.0,
            policy=harness.ProgressionPolicy.OWNED_TICK,
        )
        self.assertIs(result.verification, harness.CleanupVerification.CONFIRMED)
        self.assertEqual(world.progress_calls, ["tick"])

    def test_passive_wait_world_never_issues_an_unowned_tick(self):
        clock = FakeClock(0.0)
        world = SnapshotWorld(
            {1: object()}, clock=clock, remove_plan=(1,)
        )
        result = self.verify(world, (1,), clock=clock, timeout=5.0)
        self.assertIs(result.verification, harness.CleanupVerification.CONFIRMED)
        self.assertEqual(world.progress_calls, ["wait_for_tick"])

    def test_snapshot_and_progression_disconnect_are_unverified(self):
        cases = (
            SnapshotWorld({1: object()}, snapshot_error=RuntimeError("lost")),
            SnapshotWorld({1: object()}, progress_error=RuntimeError("lost")),
        )
        for world in cases:
            with self.subTest(world=world):
                result = self.verify(world, (1,))
                self.assertIs(
                    result.verification,
                    harness.CleanupVerification.UNVERIFIED_DISCONNECTED,
                )
                self.assertEqual(result.remaining_actor_ids, (1,))


class FakeSuiteScenario:
    def __init__(self, snapshots=None):
        self.snapshots = list(snapshots or (ready_snapshot(),))
        self.constructor_calls = 0
        self.attach_calls = []
        self.destroy_calls = 0
        self.snapshot_calls = 0
        self.tick_registered = True
        self.attach_fail_name = None
        self.topology = tuple(
            types.SimpleNamespace(
                name=name,
                actor_id=actor_id,
                type_id=type_id,
                identity_errors=(),
            )
            for name, actor_id, type_id in zip(EXPECTED_NAMES, SENSOR_IDS, EXPECTED_TYPES)
        )
        self.cleanup_errors = ()
        self.keep_ids = set()
        self.events = []
        self.on_construct = None
        self.construct_error = None
        self.attach_error = None
        self.topology_error = None
        self.snapshot_error = None
        self.destroy_errors = []


def fake_suite_class(scenario):
    class FakeSensorSuite:
        def __init__(self):
            scenario.constructor_calls += 1
            if scenario.on_construct is not None:
                scenario.on_construct()
            if scenario.construct_error is not None:
                raise scenario.construct_error
            self.world = None

        def attach(self, world, vehicle, pre_operation_guard=None):
            scenario.attach_calls.append((world, vehicle, pre_operation_guard))
            self.world = world
            for identity in scenario.topology:
                if type(identity.actor_id) is int and identity.actor_id > 0:
                    world.registry[identity.actor_id] = object()
            if scenario.attach_error is not None:
                # Sensor actors were already created before the interruption.
                raise scenario.attach_error
            results = tuple(
                types.SimpleNamespace(
                    name=name,
                    attached=(name != scenario.attach_fail_name),
                    error=("attach failed" if name == scenario.attach_fail_name else None),
                )
                for name in EXPECTED_NAMES
            )
            return types.SimpleNamespace(
                results=results,
                all_attached=all(item.attached is True for item in results),
                tick_registered=scenario.tick_registered,
                tick_error=None if scenario.tick_registered else "on_tick failed",
            )

        def topology_snapshot(self):
            if scenario.topology_error is not None:
                raise scenario.topology_error
            return types.SimpleNamespace(identities=scenario.topology)

        def snapshot(self):
            scenario.snapshot_calls += 1
            if scenario.snapshot_error is not None:
                raise scenario.snapshot_error
            if len(scenario.snapshots) > 1:
                return scenario.snapshots.pop(0)
            return scenario.snapshots[0]

        def destroy(self):
            scenario.destroy_calls += 1
            scenario.events.append("suite.destroy")
            if scenario.destroy_errors:
                raise scenario.destroy_errors.pop(0)
            if self.world is not None:
                for identity in scenario.topology:
                    if identity.actor_id not in scenario.keep_ids:
                        self.world.registry.pop(identity.actor_id, None)
            results = tuple(
                types.SimpleNamespace(
                    name=name,
                    stopped=True,
                    destroyed=True,
                    errors=(),
                )
                for name in EXPECTED_NAMES
            )
            return types.SimpleNamespace(
                tick_unregistered=True,
                results=results,
                errors=scenario.cleanup_errors,
            )

    return FakeSensorSuite


def run_fake_cycle(
    scenario,
    *,
    synchronous=False,
    tick_owner=False,
    clock=None,
    actor=None,
    world_kwargs=None,
    environ_extra=None,
):
    actor = StrictActor() if actor is None else actor
    rig = LiveRig(synchronous=synchronous, actor=actor, world_kwargs=world_kwargs)
    sensor_class = fake_suite_class(scenario)
    rig.importer.modules["sensors.suite"] = types.SimpleNamespace(SensorSuite=sensor_class)
    environ = dict(ENABLED_ENVIRON)
    if tick_owner:
        environ[harness.CARLA_INTEGRATION_TICK_OWNER] = "1"
    if environ_extra:
        environ.update(environ_extra)
    run = require_harness_api(unittest.TestCase(), "run_sensor_suite_validation_cycle")
    report = run(
        environ,
        importer=rig.importer,
        monotonic=FakeClock(0.0) if clock is None else clock,
        disappearance_timeout_seconds=1.0,
    )
    return rig, report


class LazySensorDependencyTest(unittest.TestCase):
    def test_loader_exposes_exact_constructor_without_constructing(self):
        scenario = FakeSuiteScenario()
        cls = fake_suite_class(scenario)
        importer = lambda name: types.SimpleNamespace(SensorSuite=cls)
        load = require_harness_api(self, "load_sensor_suite_dependencies")
        dependencies = load(ENABLED_ENVIRON, importer=importer)
        self.assertIs(dependencies.SensorSuite, cls)
        self.assertEqual(scenario.constructor_calls, 0)

    def test_unowned_synchronous_skip_precedes_sensor_import_and_construction(self):
        scenario = FakeSuiteScenario()
        rig = LiveRig(synchronous=True)
        cls = fake_suite_class(scenario)
        rig.importer.modules["sensors.suite"] = types.SimpleNamespace(SensorSuite=cls)
        run = require_harness_api(self, "run_sensor_suite_validation_cycle")
        with self.assertRaises(unittest.SkipTest):
            run(ENABLED_ENVIRON, importer=rig.importer, monotonic=FakeClock(0.0))
        self.assertEqual(rig.importer.requests, ["carla", "spawn_ego_vehicle"])
        self.assertEqual(scenario.constructor_calls, 0)

    def test_disabled_cycle_skips_before_any_import(self):
        rig = LiveRig()
        run = require_harness_api(self, "run_sensor_suite_validation_cycle")
        with self.assertRaisesRegex(unittest.SkipTest, f"^{harness.DISABLED_MESSAGE}$"):
            run({}, importer=rig.importer)
        self.assertEqual(rig.importer.requests, [])


class SensorValidationCycleTest(unittest.TestCase):
    def test_happy_async_and_owned_sync_cycles_validate_and_clean_six_actors(self):
        for synchronous, tick_owner, policy in (
            (False, False, harness.ProgressionPolicy.PASSIVE_WAIT),
            (True, True, harness.ProgressionPolicy.OWNED_TICK),
        ):
            with self.subTest(synchronous=synchronous):
                scenario = FakeSuiteScenario()
                rig, report = run_fake_cycle(
                    scenario, synchronous=synchronous, tick_owner=tick_owner
                )
                self.assertIs(report.policy, policy)
                self.assertIs(report.attachment_success, True)
                self.assertIs(report.tick_registration_success, True)
                self.assertIs(report.readiness_success, True)
                self.assertEqual(tuple(x.actor_id for x in report.sensor_identities), SENSOR_IDS)
                self.assertIsNone(report.primary_error)
                self.assertEqual(report.cleanup_errors, ())
                self.assertIs(
                    report.final_disappearance,
                    harness.CleanupVerification.CONFIRMED,
                )
                self.assertEqual(report.remaining_owned_actor_ids, ())
                self.assertEqual(scenario.destroy_calls, 1)
                self.assertEqual(rig.actor.destroy_calls, 1)
                self.assertEqual(
                    set(rig.world.snapshot_queried_ids), set(SENSOR_IDS + (101,))
                )

    def test_immediate_and_staggered_readiness_use_zero_or_one_rpc_per_iteration(self):
        immediate = FakeSuiteScenario((ready_snapshot(),))
        rig, _report = run_fake_cycle(immediate)
        self.assertEqual(rig.world.progressions, 0)

        incomplete = replace(ready_snapshot(), gnss=None)
        staggered = FakeSuiteScenario((incomplete, incomplete, ready_snapshot()))
        rig, report = run_fake_cycle(staggered)
        self.assertIs(report.readiness_success, True)
        self.assertEqual(rig.world.progressions, 2)

    def test_permanent_missing_required_samples_time_out_under_one_deadline(self):
        fields = ("ego", "gnss", "imu", "camera")
        for field in fields:
            with self.subTest(field=field):
                clock = FakeClock(0.0)
                snapshot = replace(ready_snapshot(), **{field: None})
                scenario = FakeSuiteScenario((snapshot,))
                actor = StrictActor()
                rig = LiveRig(actor=actor, world_kwargs={"on_progress": lambda: setattr(clock, "value", clock.value + 0.6)})
                cls = fake_suite_class(scenario)
                rig.importer.modules["sensors.suite"] = types.SimpleNamespace(SensorSuite=cls)
                run = require_harness_api(self, "run_sensor_suite_validation_cycle")
                report = run(
                    ENABLED_ENVIRON,
                    importer=rig.importer,
                    monotonic=clock,
                    disappearance_timeout_seconds=1.0,
                )
                self.assertEqual(report.primary_error.phase, "sensor_readiness")
                self.assertIs(report.readiness_success, False)
                self.assertEqual(scenario.destroy_calls, 1)
                self.assertEqual(actor.destroy_calls, 1)

    def test_attachment_tick_topology_callback_and_camera_failures_stay_primary(self):
        scenarios = []
        attach = FakeSuiteScenario()
        attach.attach_fail_name = "imu"
        scenarios.append((attach, "sensor_attachment"))
        tick = FakeSuiteScenario()
        tick.tick_registered = False
        scenarios.append((tick, "tick_registration"))
        topology = FakeSuiteScenario()
        topology.topology = topology.topology[:-1]
        scenarios.append((topology, "sensor_topology"))
        callback = FakeSuiteScenario((ready_snapshot(error_name="gnss"),))
        scenarios.append((callback, "sensor_health"))
        bad_camera = replace(
            ready_snapshot().camera,
            metadata=replace(ready_snapshot().camera.metadata, width_px=1),
        )
        camera = FakeSuiteScenario((ready_snapshot(camera=bad_camera),))
        scenarios.append((camera, "camera_validation"))
        for scenario, phase in scenarios:
            with self.subTest(phase=phase):
                rig, report = run_fake_cycle(scenario)
                self.assertEqual(report.primary_error.phase, phase)
                self.assertEqual(scenario.destroy_calls, 1)
                self.assertEqual(rig.actor.destroy_calls, 1)

    def test_report_is_frozen_and_retains_no_proxies_snapshots_or_payload(self):
        scenario = FakeSuiteScenario()
        rig, report = run_fake_cycle(scenario)
        with self.assertRaises(FrozenInstanceError):
            report.readiness_success = False

        forbidden = {id(rig.client), id(rig.world), id(rig.actor)}

        def walk(value):
            self.assertNotIn(id(value), forbidden)
            self.assertNotIsInstance(value, TelemetrySnapshot)
            self.assertNotIsInstance(value, CameraFrame)
            self.assertNotIsInstance(value, bytes)
            if dataclasses.is_dataclass(value):
                for field in dataclasses.fields(value):
                    walk(getattr(value, field.name))
            elif isinstance(value, (tuple, list)):
                for item in value:
                    walk(item)

        walk(report)

    def test_cycle_clears_temporary_snapshot_and_camera_references(self):
        scenario = FakeSuiteScenario()
        observed_states = []
        original = harness.new_cycle_state

        def recording_state(cycle_number):
            state = original(cycle_number)
            observed_states.append(state)
            return state

        run = require_harness_api(self, "run_sensor_suite_validation_cycle")
        rig = LiveRig()
        rig.importer.modules["sensors.suite"] = types.SimpleNamespace(
            SensorSuite=fake_suite_class(scenario)
        )
        with unittest.mock.patch.object(harness, "new_cycle_state", recording_state):
            run(ENABLED_ENVIRON, importer=rig.importer, monotonic=FakeClock(0.0))
        self.assertEqual(len(observed_states), 1)
        self.assertIsNone(observed_states[0].snapshot_reference)
        self.assertIsNone(observed_states[0].camera_reference)


class DisappearanceCorrectionCycleTest(unittest.TestCase):
    """The Mission 4G.5 live pattern: destroys succeed but frames lag."""

    OWNED_IDS = SENSOR_IDS + (101,)

    def advancing_clock(self):
        clock = FakeClock(0.0)
        return clock, {
            "on_progress": lambda: setattr(clock, "value", clock.value + 0.6)
        }

    def test_lagging_frames_confirm_after_bounded_progression(self):
        scenario = FakeSuiteScenario()
        rig, report = run_fake_cycle(
            scenario,
            world_kwargs={
                "snapshot_plan": [self.OWNED_IDS, self.OWNED_IDS]
            },
        )
        self.assertIsNone(report.primary_error)
        self.assertEqual(report.cleanup_errors, ())
        self.assertEqual(scenario.destroy_calls, 1)
        self.assertEqual(rig.actor.destroy_calls, 1)
        self.assertIs(
            report.final_disappearance, harness.CleanupVerification.CONFIRMED
        )
        self.assertEqual(report.remaining_owned_actor_ids, ())
        self.assertEqual(rig.world.progressions, 2)

    def test_never_clearing_frames_report_bounded_disappearance_failure(self):
        clock, world_kwargs = self.advancing_clock()
        world_kwargs["snapshot_plan"] = [
            self.OWNED_IDS, self.OWNED_IDS, self.OWNED_IDS
        ]
        scenario = FakeSuiteScenario()
        rig, report = run_fake_cycle(
            scenario, clock=clock, world_kwargs=world_kwargs
        )
        self.assertIsNone(report.primary_error)
        self.assertEqual(report.cleanup_errors, ())
        self.assertEqual(scenario.destroy_calls, 1)
        self.assertEqual(rig.actor.destroy_calls, 1)
        self.assertIs(
            report.final_disappearance,
            harness.CleanupVerification.FAILED_ACTORS_REMAIN,
        )
        self.assertEqual(
            report.remaining_owned_actor_ids, SENSOR_IDS + (101,)
        )
        self.assertEqual(rig.world.progressions, 2)

    def test_cleanup_error_and_remaining_actor_evidence_are_both_preserved(self):
        clock, world_kwargs = self.advancing_clock()
        scenario = FakeSuiteScenario()
        scenario.destroy_errors = [RuntimeError("suite cleanup failed")]
        scenario.keep_ids = {SENSOR_IDS[-1]}
        rig, report = run_fake_cycle(
            scenario, clock=clock, world_kwargs=world_kwargs
        )
        self.assertIsNone(report.primary_error)
        self.assertTrue(
            any(
                "suite cleanup failed" in error
                for error in report.cleanup_errors
            )
        )
        self.assertIs(
            report.final_disappearance,
            harness.CleanupVerification.FAILED_ACTORS_REMAIN,
        )
        self.assertEqual(
            report.remaining_owned_actor_ids, (SENSOR_IDS[-1],)
        )


class ProductionCleanupHardeningTest(unittest.TestCase):
    def attach(self, **kwargs):
        world = FakeWorld(**kwargs)
        suite = SensorSuite()
        suite.attach(world, FakeVehicle())
        return world, suite

    def test_false_and_nonliteral_destroy_results_are_errors_and_never_resubmitted(self):
        for result in (False, 1, "true", None):
            with self.subTest(result=result):
                world, suite = self.attach(destroy_results={GNSS_BLUEPRINT: result})
                first = suite.destroy()
                second = suite.destroy()
                gnss = next(x for x in first.results if x.name == "gnss")
                self.assertTrue(gnss.errors)
                self.assertIs(gnss.destroyed, False)
                destroys = [x for x in world.events if x == ("destroy", GNSS_BLUEPRINT)]
                self.assertEqual(len(destroys), 1)
                self.assertEqual(second.results, ())

    def test_destroy_exception_continues_all_sensors_and_is_resumable(self):
        world, suite = self.attach(fail_destroy=(GNSS_BLUEPRINT,))
        report = suite.destroy()
        self.assertEqual(
            [target for kind, target in world.events if kind == "destroy"],
            list(reversed(EXPECTED_TYPES)),
        )
        self.assertTrue(next(x for x in report.results if x.name == "gnss").errors)
        suite.destroy()
        self.assertEqual(
            len([x for x in world.events if x == ("destroy", GNSS_BLUEPRINT)]), 1
        )

    def test_baseexception_is_deferred_until_all_sensors_are_destroyed_then_cached(self):
        world, suite = self.attach()
        interrupt = KeyboardInterrupt()
        target = world.spawned[IMU_BLUEPRINT]
        calls = 0

        def interrupted_destroy():
            nonlocal calls
            calls += 1
            world.events.append(("destroy", IMU_BLUEPRINT))
            raise interrupt

        target.destroy = interrupted_destroy
        with self.assertRaises(KeyboardInterrupt) as context:
            suite.destroy()
        self.assertIs(context.exception, interrupt)
        self.assertEqual(
            set(target for kind, target in world.events if kind == "destroy"),
            set(EXPECTED_TYPES),
        )
        cached = suite.destroy()
        self.assertEqual(calls, 1)
        self.assertTrue(next(x for x in cached.results if x.name == "imu").errors)


class SensorDependencyDiagnosticsTest(unittest.TestCase):
    def test_disabled_gate_skips_before_any_import(self):
        load = require_harness_api(self, "load_sensor_suite_dependencies")
        imported = []
        with self.assertRaisesRegex(unittest.SkipTest, f"^{harness.DISABLED_MESSAGE}$"):
            load({}, importer=imported.append)
        self.assertEqual(imported, [])

    def test_import_failure_diagnostic_names_module_and_preserves_cause(self):
        load = require_harness_api(self, "load_sensor_suite_dependencies")
        cause = ImportError("suite import exploded")

        def importer(name):
            raise cause

        with self.assertRaises(RuntimeError) as context:
            load(ENABLED_ENVIRON, importer=importer)
        self.assertIn("sensors.suite", str(context.exception))
        self.assertIn("suite import exploded", str(context.exception))
        self.assertIs(context.exception.__cause__, cause)

    def test_missing_or_noncallable_public_api_fails_with_bounded_message(self):
        incomplete = type(
            "NoDestroySuite",
            (),
            {
                "attach": lambda self: None,
                "snapshot": lambda self: None,
                "topology_snapshot": lambda self: None,
            },
        )
        modules = (
            types.SimpleNamespace(),
            types.SimpleNamespace(SensorSuite="not callable"),
            types.SimpleNamespace(SensorSuite=incomplete),
        )
        load = require_harness_api(self, "load_sensor_suite_dependencies")
        for module in modules:
            with self.subTest(module=module):
                with self.assertRaises(RuntimeError) as context:
                    load(ENABLED_ENVIRON, importer=lambda name: module)
                message = str(context.exception)
                self.assertIn(harness.SENSOR_SUITE_DEPENDENCY_MESSAGE, message)
                self.assertLessEqual(
                    len(message), harness.MAX_PRIMARY_MESSAGE_LENGTH
                )

    def test_cycle_records_dependency_failure_and_still_cleans_the_ego(self):
        rig = LiveRig()
        rig.importer.errors["sensors.suite"] = ImportError("boom")
        run = require_harness_api(self, "run_sensor_suite_validation_cycle")
        report = run(
            ENABLED_ENVIRON,
            importer=rig.importer,
            monotonic=FakeClock(0.0),
            disappearance_timeout_seconds=1.0,
        )
        self.assertEqual(
            report.primary_error.phase, "sensor_suite_dependency_loading"
        )
        self.assertEqual(rig.actor.destroy_calls, 1)
        self.assertEqual(rig.world.snapshot_queried_ids, [101])
        self.assertIs(
            report.final_disappearance, harness.CleanupVerification.CONFIRMED
        )


class ConfigFailWorld(FakeWorld):
    """Blueprint configuration raises for one target sensor blueprint."""

    def __init__(self, fail_blueprint):
        super().__init__()
        self._fail_blueprint = fail_blueprint

    def get_blueprint_library(self):
        library = super().get_blueprint_library()
        original_find = library.find
        fail_id = self._fail_blueprint

        def find(bp_id):
            blueprint = original_find(bp_id)
            if bp_id == fail_id:
                def failing_set_attribute(name, value):
                    raise RuntimeError(f"configuration failed for {bp_id}")

                blueprint.set_attribute = failing_set_attribute
            return blueprint

        library.find = find
        return library


class SensorConfigurationFailureTest(unittest.TestCase):
    def test_configuration_failure_at_each_sensor_spawns_nothing_for_it(self):
        for blueprint in EXPECTED_TYPES:
            with self.subTest(blueprint=blueprint):
                world = ConfigFailWorld(blueprint)
                suite = SensorSuite()
                report = suite.attach(world, FakeVehicle())
                failed = [r for r in report.results if r.attached is False]
                self.assertEqual(len(failed), 1)
                self.assertEqual(
                    failed[0].name,
                    EXPECTED_NAMES[EXPECTED_TYPES.index(blueprint)],
                )
                self.assertNotIn(blueprint, world.spawned)
                self.assertNotIn(
                    failed[0].name,
                    tuple(x.name for x in suite.topology_snapshot().identities),
                )
                suite.destroy()


class SteppingSpawnWorld(FakeWorld):
    """Advance the injected host clock during each spawn RPC."""

    def __init__(self, clock, step):
        super().__init__()
        self._clock = clock
        self._step = step

    def spawn_actor(self, blueprint, transform, attach_to=None, attachment_type=None):
        self._clock.value += self._step
        return super().spawn_actor(blueprint, transform, attach_to, attachment_type)


class AttachDeadlineGuardTest(unittest.TestCase):
    def attach_with_deadline(self, step):
        clock = FakeClock(0.0)
        deadline = harness.Deadline(1.0, clock)
        world = SteppingSpawnWorld(clock, step)
        suite = SensorSuite()
        report = suite.attach(
            world, FakeVehicle(), pre_operation_guard=deadline.raise_if_expired
        )
        return world, suite, report

    def test_expiry_during_first_spawn_starts_no_further_live_operation(self):
        world, suite, report = self.attach_with_deadline(2.0)
        self.assertEqual(
            tuple(call[0] for call in world.spawn_calls), (GNSS_BLUEPRINT,)
        )
        self.assertEqual(set(world.blueprints), {GNSS_BLUEPRINT})
        self.assertEqual(
            tuple(r.attached for r in report.results),
            (False, False, False, False, False),
        )
        self.assertIs(report.tick_registered, False)
        self.assertEqual(world.tick_callbacks, {})
        identities = suite.topology_snapshot().identities
        self.assertEqual(tuple(x.name for x in identities), ("gnss",))
        cleanup = suite.destroy()
        gnss = next(r for r in cleanup.results if r.name == "gnss")
        self.assertIs(gnss.destroyed, True)
        self.assertIs(world.spawned[GNSS_BLUEPRINT].destroyed, True)
        self.assertNotIn(("stop", GNSS_BLUEPRINT), world.events)

    def test_expiry_between_sensors_keeps_earlier_attachments_and_cleans_all(self):
        world, suite, report = self.attach_with_deadline(0.4)
        self.assertEqual(
            tuple(call[0] for call in world.spawn_calls),
            (GNSS_BLUEPRINT, IMU_BLUEPRINT, COLLISION_BLUEPRINT),
        )
        self.assertEqual(
            tuple(r.attached for r in report.results),
            (True, True, False, False, False),
        )
        self.assertIs(report.tick_registered, False)
        identities = suite.topology_snapshot().identities
        self.assertEqual(
            tuple(x.name for x in identities), ("gnss", "imu", "collision")
        )
        suite.destroy()
        destroyed = [t for kind, t in world.events if kind == "destroy"]
        self.assertEqual(
            set(destroyed),
            {GNSS_BLUEPRINT, IMU_BLUEPRINT, COLLISION_BLUEPRINT},
        )
        self.assertEqual(len(destroyed), 3)
        stopped = [t for kind, t in world.events if kind == "stop"]
        self.assertEqual(set(stopped), {GNSS_BLUEPRINT, IMU_BLUEPRINT})


class SensorPhaseDeadlineTest(unittest.TestCase):
    def test_ready_snapshot_observed_at_expired_boundary_still_passes(self):
        clock = FakeClock(0.0)
        scenario = FakeSuiteScenario()
        scenario.on_construct = lambda: setattr(clock, "value", clock.value + 5.0)
        rig, report = run_fake_cycle(
            scenario,
            clock=clock,
            environ_extra={harness.CARLA_TIMEOUT_SECONDS: "2.0"},
        )
        self.assertIs(report.readiness_success, True)
        self.assertIsNone(report.primary_error)
        self.assertEqual(rig.world.progressions, 0)
        self.assertIs(
            report.final_disappearance, harness.CleanupVerification.CONFIRMED
        )

    def test_incomplete_snapshot_at_expired_boundary_times_out_without_rpc(self):
        clock = FakeClock(0.0)
        scenario = FakeSuiteScenario((replace(ready_snapshot(), gnss=None),))
        scenario.on_construct = lambda: setattr(clock, "value", clock.value + 5.0)
        rig, report = run_fake_cycle(
            scenario,
            clock=clock,
            environ_extra={harness.CARLA_TIMEOUT_SECONDS: "2.0"},
        )
        self.assertEqual(report.primary_error.phase, "sensor_readiness")
        self.assertIs(report.readiness_success, False)
        self.assertEqual(rig.world.progressions, 0)
        self.assertEqual(scenario.destroy_calls, 1)
        self.assertEqual(rig.actor.destroy_calls, 1)

    def test_partially_improving_snapshots_never_extend_the_fixed_deadline(self):
        clock = FakeClock(0.0)
        improving = (
            replace(ready_snapshot(), gnss=None),
            replace(ready_snapshot(), camera=None),
            replace(ready_snapshot(), camera=None),
        )
        scenario = FakeSuiteScenario(improving)
        rig, report = run_fake_cycle(
            scenario,
            clock=clock,
            world_kwargs={
                "on_progress": lambda: setattr(clock, "value", clock.value + 0.6)
            },
            environ_extra={harness.CARLA_TIMEOUT_SECONDS: "2.0"},
        )
        self.assertEqual(report.primary_error.phase, "sensor_readiness")
        self.assertEqual(rig.world.progressions, 4)

    def test_disconnect_during_readiness_is_primary_and_cleanup_confirms(self):
        scenario = FakeSuiteScenario((replace(ready_snapshot(), gnss=None),))
        rig, report = run_fake_cycle(
            scenario,
            world_kwargs={"progression_error": RuntimeError("connection lost")},
        )
        self.assertEqual(report.primary_error.phase, "sensor_readiness")
        self.assertIs(report.readiness_success, False)
        self.assertEqual(scenario.destroy_calls, 1)
        self.assertEqual(rig.actor.destroy_calls, 1)
        self.assertIs(
            report.final_disappearance, harness.CleanupVerification.CONFIRMED
        )

    def test_primary_failure_and_ego_cleanup_error_stay_separate(self):
        scenario = FakeSuiteScenario((ready_snapshot(error_name="gnss"),))
        actor = StrictActor(destroy_result=False, remove_even_on_failure=True)
        _rig, report = run_fake_cycle(scenario, actor=actor)
        self.assertEqual(report.primary_error.phase, "sensor_health")
        self.assertTrue(
            any(error.startswith("ego destroy") for error in report.cleanup_errors)
        )
        self.assertIs(
            report.final_disappearance, harness.CleanupVerification.CONFIRMED
        )


class CycleInterruptionTest(unittest.TestCase):
    """KeyboardInterrupt at each harness boundary: full cleanup, same object."""

    def run_interrupted(
        self, scenario, interrupt, *, actor=None, world_kwargs=None
    ):
        run = require_harness_api(self, "run_sensor_suite_validation_cycle")
        actor = StrictActor() if actor is None else actor
        rig = LiveRig(actor=actor, world_kwargs=world_kwargs)
        rig.importer.modules["sensors.suite"] = types.SimpleNamespace(
            SensorSuite=fake_suite_class(scenario)
        )
        with self.assertRaises(KeyboardInterrupt) as context:
            run(
                dict(ENABLED_ENVIRON),
                importer=rig.importer,
                monotonic=FakeClock(0.0),
                disappearance_timeout_seconds=1.0,
            )
        self.assertIs(context.exception, interrupt)
        return rig

    def test_interrupt_during_construction_still_cleans_and_verifies_ego(self):
        interrupt = KeyboardInterrupt("construction")
        scenario = FakeSuiteScenario()
        scenario.construct_error = interrupt
        rig = self.run_interrupted(scenario, interrupt)
        self.assertEqual(scenario.constructor_calls, 1)
        self.assertEqual(scenario.destroy_calls, 0)
        self.assertEqual(rig.actor.destroy_calls, 1)
        self.assertEqual(rig.world.snapshot_queried_ids, [101])

    def test_interrupt_during_attach_cleans_created_sensors_and_ego(self):
        interrupt = KeyboardInterrupt("attach")
        scenario = FakeSuiteScenario()
        scenario.attach_error = interrupt
        rig = self.run_interrupted(scenario, interrupt)
        self.assertEqual(scenario.destroy_calls, 1)
        self.assertEqual(rig.actor.destroy_calls, 1)
        self.assertEqual(set(rig.world.snapshot_queried_ids), set(SENSOR_IDS + (101,)))

    def test_interrupt_during_topology_capture_stays_conservative(self):
        interrupt = KeyboardInterrupt("topology")
        scenario = FakeSuiteScenario()
        scenario.topology_error = interrupt
        rig = self.run_interrupted(scenario, interrupt)
        self.assertEqual(scenario.destroy_calls, 1)
        self.assertEqual(rig.actor.destroy_calls, 1)
        self.assertEqual(rig.world.snapshot_queried_ids, [101])

    def test_interrupt_during_readiness_snapshot_cleans_all_six_actors(self):
        interrupt = KeyboardInterrupt("readiness")
        scenario = FakeSuiteScenario()
        scenario.snapshot_error = interrupt
        rig = self.run_interrupted(scenario, interrupt)
        self.assertEqual(scenario.destroy_calls, 1)
        self.assertEqual(rig.actor.destroy_calls, 1)
        self.assertEqual(set(rig.world.snapshot_queried_ids), set(SENSOR_IDS + (101,)))

    def test_interrupt_during_suite_destroy_is_deferred_and_cleanup_resumes(self):
        interrupt = KeyboardInterrupt("suite destroy")
        scenario = FakeSuiteScenario()
        scenario.destroy_errors = [interrupt]
        rig = self.run_interrupted(scenario, interrupt)
        self.assertEqual(scenario.destroy_calls, 2)
        self.assertEqual(rig.actor.destroy_calls, 1)
        self.assertEqual(set(rig.world.snapshot_queried_ids), set(SENSOR_IDS + (101,)))

    def test_interrupt_during_ego_destroy_submits_once_and_still_verifies(self):
        interrupt = KeyboardInterrupt("ego destroy")
        scenario = FakeSuiteScenario()
        actor = StrictActor(destroy_error=interrupt)
        rig = self.run_interrupted(
            scenario,
            interrupt,
            actor=actor,
            world_kwargs={"remove_after_progressions": 1, "remove_actor_id": 101},
        )
        self.assertEqual(rig.actor.destroy_calls, 1)
        self.assertEqual(scenario.destroy_calls, 1)
        self.assertIn(101, rig.world.snapshot_queried_ids)

    def test_interrupt_during_final_snapshot_is_deferred_and_retried(self):
        interrupt = KeyboardInterrupt("final snapshot")
        scenario = FakeSuiteScenario()
        rig = self.run_interrupted(
            scenario, interrupt, world_kwargs={"snapshot_plan": [interrupt]}
        )
        self.assertEqual(rig.actor.destroy_calls, 1)
        self.assertEqual(scenario.destroy_calls, 1)
        self.assertGreater(len(rig.world.snapshot_queried_ids), 1)

    def test_interrupt_during_final_progression_stays_unverified(self):
        interrupt = KeyboardInterrupt("final progression")
        scenario = FakeSuiteScenario()
        scenario.keep_ids = {SENSOR_IDS[-1]}
        rig = self.run_interrupted(
            scenario, interrupt, world_kwargs={"progression_error": interrupt}
        )
        self.assertEqual(rig.actor.destroy_calls, 1)
        self.assertEqual(scenario.destroy_calls, 1)
        self.assertIn(SENSOR_IDS[-1], rig.world.snapshot_queried_ids)


class InterruptSpawnWorld(FakeWorld):
    """Raise a retained BaseException when spawning one target blueprint."""

    def __init__(self, interrupt, spawn_target):
        super().__init__()
        self._interrupt = interrupt
        self._spawn_target = spawn_target

    def spawn_actor(self, blueprint, transform, attach_to=None, attachment_type=None):
        if blueprint.id == self._spawn_target:
            raise self._interrupt
        return super().spawn_actor(blueprint, transform, attach_to, attachment_type)


class InterruptListenWorld(FakeWorld):
    """Raise a retained BaseException from listen() of one target sensor."""

    def __init__(self, interrupt, listen_target):
        super().__init__()
        self._interrupt = interrupt
        self._listen_target = listen_target

    def spawn_actor(self, blueprint, transform, attach_to=None, attachment_type=None):
        actor = super().spawn_actor(blueprint, transform, attach_to, attachment_type)
        if blueprint.id == self._listen_target:
            original_listen = actor.listen
            interrupt = self._interrupt

            def interrupted_listen(callback):
                # Model CARLA registering the callback before the failure.
                original_listen(callback)
                raise interrupt

            actor.listen = interrupted_listen
        return actor


class InterruptIdActor:
    """Proxy whose ``id`` read raises; everything else delegates."""

    def __init__(self, inner, interrupt):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_interrupt", interrupt)

    def __getattr__(self, name):
        if name == "id":
            raise object.__getattribute__(self, "_interrupt")
        return getattr(object.__getattribute__(self, "_inner"), name)


class InterruptIdWorld(FakeWorld):
    def __init__(self, interrupt, id_target):
        super().__init__()
        self._interrupt = interrupt
        self._id_target = id_target

    def spawn_actor(self, blueprint, transform, attach_to=None, attachment_type=None):
        actor = super().spawn_actor(blueprint, transform, attach_to, attachment_type)
        if blueprint.id == self._id_target:
            return InterruptIdActor(actor, self._interrupt)
        return actor


class SuiteInterruptionBoundariesTest(unittest.TestCase):
    """KeyboardInterrupt at each production boundary keeps cleanup resumable."""

    def test_interrupt_during_each_spawn_preserves_prior_actors_and_cleanup(self):
        for index, target in enumerate(EXPECTED_TYPES):
            with self.subTest(target=target):
                interrupt = KeyboardInterrupt(target)
                world = InterruptSpawnWorld(interrupt, target)
                suite = SensorSuite()
                with self.assertRaises(KeyboardInterrupt) as context:
                    suite.attach(world, FakeVehicle())
                self.assertIs(context.exception, interrupt)
                identities = suite.topology_snapshot().identities
                self.assertEqual(
                    tuple(x.name for x in identities), EXPECTED_NAMES[:index]
                )
                report = suite.destroy()
                destroyed = [t for kind, t in world.events if kind == "destroy"]
                self.assertEqual(set(destroyed), set(EXPECTED_TYPES[:index]))
                self.assertEqual(len(destroyed), index)
                for result in report.results:
                    if result.name in EXPECTED_NAMES[:index]:
                        self.assertIs(result.destroyed, True)

    def test_interrupt_after_spawn_during_identity_read_keeps_the_actor_owned(self):
        interrupt = KeyboardInterrupt("identity")
        world = InterruptIdWorld(interrupt, IMU_BLUEPRINT)
        suite = SensorSuite()
        with self.assertRaises(KeyboardInterrupt) as context:
            suite.attach(world, FakeVehicle())
        self.assertIs(context.exception, interrupt)
        identities = suite.topology_snapshot().identities
        imu = next(x for x in identities if x.name == "imu")
        self.assertIsNone(imu.actor_id)
        self.assertTrue(imu.identity_errors)
        suite.destroy()
        destroyed = [t for kind, t in world.events if kind == "destroy"]
        self.assertEqual(set(destroyed), {GNSS_BLUEPRINT, IMU_BLUEPRINT})

    def test_interrupt_during_listen_registration_still_stops_and_destroys(self):
        interrupt = KeyboardInterrupt("listen")
        world = InterruptListenWorld(interrupt, COLLISION_BLUEPRINT)
        suite = SensorSuite()
        with self.assertRaises(KeyboardInterrupt) as context:
            suite.attach(world, FakeVehicle())
        self.assertIs(context.exception, interrupt)
        identities = suite.topology_snapshot().identities
        self.assertEqual(
            tuple(x.name for x in identities), ("gnss", "imu", "collision")
        )
        suite.destroy()
        stopped = [t for kind, t in world.events if kind == "stop"]
        self.assertIn(COLLISION_BLUEPRINT, stopped)
        destroyed = [t for kind, t in world.events if kind == "destroy"]
        self.assertEqual(
            set(destroyed),
            {GNSS_BLUEPRINT, IMU_BLUEPRINT, COLLISION_BLUEPRINT},
        )

    def test_interrupt_between_sensors_via_guard_spawns_nothing_further(self):
        interrupt = KeyboardInterrupt("between sensors")

        def guard(phase):
            if phase == "collision.carla_import":
                raise interrupt

        world = FakeWorld()
        suite = SensorSuite()
        with self.assertRaises(KeyboardInterrupt) as context:
            suite.attach(world, FakeVehicle(), pre_operation_guard=guard)
        self.assertIs(context.exception, interrupt)
        self.assertEqual(
            tuple(call[0] for call in world.spawn_calls),
            (GNSS_BLUEPRINT, IMU_BLUEPRINT),
        )
        suite.destroy()
        destroyed = [t for kind, t in world.events if kind == "destroy"]
        self.assertEqual(set(destroyed), {GNSS_BLUEPRINT, IMU_BLUEPRINT})

    def test_interrupt_during_stop_still_destroys_every_sensor_once(self):
        interrupt = KeyboardInterrupt("stop")
        world = FakeWorld()
        suite = SensorSuite()
        suite.attach(world, FakeVehicle())
        target = world.spawned[IMU_BLUEPRINT]

        def interrupted_stop():
            world.events.append(("stop", IMU_BLUEPRINT))
            raise interrupt

        target.stop = interrupted_stop
        with self.assertRaises(KeyboardInterrupt) as context:
            suite.destroy()
        self.assertIs(context.exception, interrupt)
        destroyed = [t for kind, t in world.events if kind == "destroy"]
        self.assertEqual(set(destroyed), set(EXPECTED_TYPES))
        self.assertEqual(len(destroyed), 5)
        for blueprint in EXPECTED_TYPES:
            self.assertEqual(
                len([e for e in world.events if e == ("stop", blueprint)]), 1
            )
        cached = suite.destroy()
        imu = next(x for x in cached.results if x.name == "imu")
        self.assertIs(imu.stopped, False)
        self.assertTrue(imu.errors)
        self.assertEqual(
            len([e for e in world.events if e[0] == "destroy"]), 5
        )

    def test_interrupt_during_each_sensor_destroy_continues_and_caches(self):
        for target_type, target_name in zip(EXPECTED_TYPES, EXPECTED_NAMES):
            with self.subTest(target=target_name):
                world = FakeWorld()
                suite = SensorSuite()
                suite.attach(world, FakeVehicle())
                interrupt = KeyboardInterrupt(target_name)
                target = world.spawned[target_type]

                def interrupted_destroy():
                    world.events.append(("destroy", target_type))
                    raise interrupt

                target.destroy = interrupted_destroy
                with self.assertRaises(KeyboardInterrupt) as context:
                    suite.destroy()
                self.assertIs(context.exception, interrupt)
                destroyed = [t for kind, t in world.events if kind == "destroy"]
                self.assertEqual(set(destroyed), set(EXPECTED_TYPES))
                self.assertEqual(len(destroyed), 5)
                cached = suite.destroy()
                failed = next(
                    x for x in cached.results if x.name == target_name
                )
                self.assertIs(failed.destroyed, False)
                self.assertTrue(failed.errors)
                self.assertEqual(
                    len([e for e in world.events if e[0] == "destroy"]), 5
                )


class LateCallbackSuppressionTest(unittest.TestCase):
    def test_late_malformed_callback_after_shutdown_mutates_nothing(self):
        world = FakeWorld()
        suite = SensorSuite()
        suite.attach(world, FakeVehicle())
        imu_actor = world.spawned[IMU_BLUEPRINT]
        suite.destroy()
        before = suite.snapshot()
        imu_actor.emit(object())  # malformed: parse would raise if attempted
        after = suite.snapshot()
        self.assertEqual(after.revision, before.revision)
        self.assertEqual(
            [(h.name, h.error_count, h.messages_received) for h in after.sensors],
            [(h.name, h.error_count, h.messages_received) for h in before.sensors],
        )
        self.assertIsNone(after.camera)


class SourceAndLiveEntryContractTest(unittest.TestCase):
    def test_harness_keeps_stdlib_module_imports_and_forbids_manual_drive(self):
        tree = ast.parse(inspect.getsource(harness))
        banned = {
            "carla",
            "numpy",
            "pygame",
            "sensors",
            "telemetry",
            "manual_drive",
            "spawn_ego_vehicle",
        }
        module_roots = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                module_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module_roots.add((node.module or "").split(".")[0])
        self.assertFalse(module_roots & banned)
        self.assertNotIn("manual_drive", inspect.getsource(harness))

    def test_new_cycle_has_no_cycle_soak_memory_control_or_sleep_scope_leakage(self):
        run = require_harness_api(self, "run_sensor_suite_validation_cycle")
        source = inspect.getsource(run)
        for forbidden in (
            "CARLA_INTEGRATION_CYCLES",
            "CARLA_INTEGRATION_SOAK_SECONDS",
            "tracemalloc",
            "gc.collect",
            "time.sleep",
            "apply_control",
            "autopilot",
            "spectator",
        ):
            self.assertNotIn(forbidden, source)

    def test_exactly_one_live_unittest_method_remains(self):
        methods = []
        for _name, value in vars(harness).items():
            if isinstance(value, type) and issubclass(value, unittest.TestCase):
                methods.extend(
                    (value, name)
                    for name in vars(value)
                    if name.startswith("test_")
                )
        self.assertEqual(len(methods), 1)


if __name__ == "__main__":
    unittest.main()
