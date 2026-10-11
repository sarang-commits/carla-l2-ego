"""Unit tests for SensorSuite attach/callback/cleanup behavior (fakes only)."""

import gc
import unittest
import weakref

from sensors.collision import COLLISION_BLUEPRINT
from sensors.gnss import GNSS_BLUEPRINT
from sensors.imu import IMU_BLUEPRINT
from sensors.lane_invasion import LANE_INVASION_BLUEPRINT
from sensors.rgb_camera import RGB_CAMERA_BLUEPRINT
from sensors.suite import SensorSuite
from tests.fakes import FakeImuMeasurement, FakeTransform, FakeVector, FakeVehicle, FakeWorld


def make_imu_measurement(frame=5, timestamp=1.0):
    return FakeImuMeasurement(
        frame=frame,
        timestamp=timestamp,
        accelerometer=FakeVector(0.0, 0.0, 9.81),
        gyroscope=FakeVector(0.0, 0.0, 0.0),
        compass=0.0,
        transform=FakeTransform(),
    )


def attach_suite(events=None, warning_sink=None, **world_kwargs):
    world = FakeWorld(events=events, **world_kwargs)
    suite = SensorSuite(warning_sink=warning_sink)
    report = suite.attach(world, FakeVehicle())
    return world, suite, report


class AttachTest(unittest.TestCase):
    def test_missing_gnss_blueprint_does_not_prevent_imu(self):
        _world, suite, report = attach_suite(missing_blueprints=(GNSS_BLUEPRINT,))
        by_name = {r.name: r for r in report.results}
        self.assertFalse(by_name["gnss"].attached)
        self.assertIsNotNone(by_name["gnss"].error)
        self.assertTrue(by_name["imu"].attached)
        self.assertTrue(report.any_attached)
        self.assertFalse(report.all_attached)
        suite.destroy()

    def test_warning_sink_exceptions_are_isolated(self):
        def exploding_sink(_message):
            raise RuntimeError("sink boom")

        # The gnss attach failure triggers a warning; the raising sink must
        # not break attach.
        _world, suite, report = attach_suite(
            warning_sink=exploding_sink,
            missing_blueprints=(GNSS_BLUEPRINT,),
        )
        by_name = {r.name: r for r in report.results}
        self.assertFalse(by_name["gnss"].attached)
        self.assertTrue(by_name["imu"].attached)
        suite.destroy()

    def test_attach_is_one_shot(self):
        world, suite, _report = attach_suite()
        with self.assertRaises(RuntimeError):
            suite.attach(world, FakeVehicle())
        suite.destroy()

    def test_non_vehicle_actor_is_rejected(self):
        world = FakeWorld()
        suite = SensorSuite()
        with self.assertRaises(ValueError):
            suite.attach(world, FakeVehicle(type_id="sensor.other.gnss"))

    def test_default_attach_order_ends_with_camera_fifth(self):
        _world, suite, report = attach_suite()
        names = [r.name for r in report.results]
        self.assertEqual(
            names, ["gnss", "imu", "collision", "lane_invasion", "rgb_camera"]
        )
        self.assertTrue(report.all_attached)
        suite.destroy()


class CallbackIsolationTest(unittest.TestCase):
    def test_callback_exception_in_one_sensor_does_not_stop_other(self):
        world, suite, _report = attach_suite()
        gnss_actor = world.spawned[GNSS_BLUEPRINT]
        imu_actor = world.spawned[IMU_BLUEPRINT]

        # A broken GNSS measurement makes parse raise; it must be isolated.
        gnss_actor.emit(None)
        # The IMU still delivers normally.
        imu_actor.emit(make_imu_measurement())

        snap = suite.snapshot()
        self.assertIsNone(snap.gnss)          # GNSS never produced a record
        self.assertIsNotNone(snap.imu)        # IMU is unaffected
        health = {h.name: h for h in snap.sensors}
        self.assertGreaterEqual(health["gnss"].error_count, 1)
        self.assertGreaterEqual(health["imu"].messages_received, 1)
        suite.destroy()

    def test_callbacks_after_shutdown_are_ignored(self):
        world, suite, _report = attach_suite()
        imu_actor = world.spawned[IMU_BLUEPRINT]
        suite.destroy()
        revision_before = suite.snapshot().revision
        imu_actor.emit(make_imu_measurement())  # late callback
        self.assertEqual(suite.snapshot().revision, revision_before)
        self.assertIsNone(suite.snapshot().imu)


class CleanupTest(unittest.TestCase):
    def test_cleanup_order_unregister_then_stop_then_destroy(self):
        events = []
        _world, suite, _report = attach_suite(events=events)
        suite.destroy()
        kinds = [kind for kind, _ in events]
        self.assertEqual(kinds[0], "tick_unregister")
        stop_indices = [i for i, k in enumerate(kinds) if k == "stop"]
        destroy_indices = [i for i, k in enumerate(kinds) if k == "destroy"]
        self.assertTrue(stop_indices and destroy_indices)
        # Every stop happens before every destroy.
        self.assertLess(max(stop_indices), min(destroy_indices))

    def test_destroy_uses_reverse_creation_order(self):
        events = []
        _world, suite, _report = attach_suite(events=events)
        suite.destroy()
        stopped = [target for kind, target in events if kind == "stop"]
        destroyed = [target for kind, target in events if kind == "destroy"]
        # Default creation order is [gnss, imu, collision, lane_invasion,
        # rgb_camera]; listeners stop, then actors are destroyed, both in
        # reverse creation order.
        reverse_order = [
            RGB_CAMERA_BLUEPRINT,
            LANE_INVASION_BLUEPRINT,
            COLLISION_BLUEPRINT,
            IMU_BLUEPRINT,
            GNSS_BLUEPRINT,
        ]
        self.assertEqual(stopped, reverse_order)
        self.assertEqual(destroyed, reverse_order)
        # All listeners stop before any actor is destroyed (stop-all, destroy-all).
        phases = [kind for kind, _ in events if kind in ("stop", "destroy")]
        self.assertEqual(
            phases,
            ["stop"] * len(reverse_order) + ["destroy"] * len(reverse_order),
        )

    def test_destroy_is_idempotent(self):
        events = []
        _world, suite, _report = attach_suite(events=events)
        suite.destroy()
        event_count = len(events)
        second = suite.destroy()
        self.assertEqual(len(events), event_count)  # nothing happened again
        self.assertEqual(second.results, ())

    def test_cleanup_continues_after_individual_failures(self):
        events = []
        _world, suite, _report = attach_suite(
            events=events,
            fail_stop=(IMU_BLUEPRINT,),
            fail_destroy=(GNSS_BLUEPRINT,),
        )
        report = suite.destroy()  # must not raise
        destroyed = [target for kind, target in events if kind == "destroy"]
        self.assertIn(IMU_BLUEPRINT, destroyed)
        self.assertIn(GNSS_BLUEPRINT, destroyed)
        by_name = {r.name: r for r in report.results}
        self.assertTrue(by_name["imu"].errors)   # imu stop failed
        self.assertTrue(by_name["gnss"].errors)  # gnss destroy failed


class LifecycleTest(unittest.TestCase):
    def test_no_strong_callback_cycle_keeps_suite_alive(self):
        world = FakeWorld()
        suite = SensorSuite()
        suite.attach(world, FakeVehicle())
        ref = weakref.ref(suite)
        del suite
        gc.collect()
        self.assertIsNone(ref())


if __name__ == "__main__":
    unittest.main()
