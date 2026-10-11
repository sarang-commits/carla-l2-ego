"""Deterministic Mission 4I freshness and degradation contracts."""

import math
import unittest
from dataclasses import FrozenInstanceError, fields

from sensors.collision import COLLISION_BLUEPRINT
from sensors.gnss import GNSS_BLUEPRINT
from sensors.imu import IMU_BLUEPRINT
from sensors.lane_invasion import LANE_INVASION_BLUEPRINT
from sensors.rgb_camera import RGB_CAMERA_BLUEPRINT
from sensors.suite import SensorSuite, SensorSuiteConfig
from telemetry.aggregator import TelemetryAggregator

from telemetry.freshness import (
    GNSS_FRESHNESS,
    IMU_FRESHNESS,
    RGB_CAMERA_FRESHNESS,
    ContinuousFreshnessPolicy,
    OperationalHealth,
    SensorEvidence,
    SensorFreshness,
    SensorFreshnessConfig,
    SensorKind,
    SensorLifecycle,
    SensorReadiness,
    evaluate_sensor_status,
)
from telemetry.state import GnssState, SampleStamp
from tests.fakes import (
    FakeGnssMeasurement,
    FakeImuMeasurement,
    FakeTransform,
    FakeVector,
    FakeVehicle,
    FakeWorld,
)
from tests.test_events import collision_measurement, lane_measurement


def continuous_evidence(**changes):
    values = dict(
        name="gnss",
        kind=SensorKind.CONTINUOUS,
        enabled=True,
        attach_attempted=True,
        attached=True,
        attachment_failed=False,
        destroyed=False,
        attachment_monotonic=10.0,
        accepted_count=0,
        last_receive_monotonic=None,
        active_error=False,
        error_count=0,
    )
    values.update(changes)
    return SensorEvidence(**values)


def event_evidence(**changes):
    values = dict(
        name="collision",
        kind=SensorKind.EVENT,
        enabled=True,
        attach_attempted=True,
        attached=True,
        attachment_failed=False,
        destroyed=False,
        attachment_monotonic=10.0,
        accepted_count=0,
        last_receive_monotonic=None,
        active_error=False,
        error_count=0,
        event_count=0,
    )
    values.update(changes)
    return SensorEvidence(**values)


class NumericClock:
    def __init__(self, value=100.0):
        self.value = value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value


class FakeCameraImage:
    def __init__(self, frame=1, timestamp=1.0, raw_data=b"\x00\x00\x00\xff"):
        self.frame = frame
        self.timestamp = timestamp
        self.width = 1
        self.height = 1
        self.fov = 90.0
        self.raw_data = raw_data
        self.transform = FakeTransform()


def attach_suite(clock=None, config=None, **world_kwargs):
    clock = clock or NumericClock()
    world = FakeWorld(**world_kwargs)
    suite = SensorSuite(config=config, monotonic_clock=clock)
    report = suite.attach(world, FakeVehicle())
    return clock, world, suite, report


def status_by_name(snapshot, name):
    return next(status for status in snapshot.statuses if status.name == name)


def continuous_measurement(name, frame, timestamp):
    if name == "gnss":
        return FakeGnssMeasurement(
            frame=frame,
            timestamp=timestamp,
            latitude=1.0,
            longitude=2.0,
            altitude=3.0,
            transform=FakeTransform(),
        )
    if name == "imu":
        return FakeImuMeasurement(
            frame=frame,
            timestamp=timestamp,
            accelerometer=FakeVector(0.0, 0.0, 9.81),
            gyroscope=FakeVector(0.0, 0.0, 0.0),
            compass=0.0,
            transform=FakeTransform(),
        )
    if name == "rgb_camera":
        return FakeCameraImage(frame=frame, timestamp=timestamp)
    raise KeyError(name)


CONTINUOUS_ACTORS = {
    "gnss": GNSS_BLUEPRINT,
    "imu": IMU_BLUEPRINT,
    "rgb_camera": RGB_CAMERA_BLUEPRINT,
}


class PolicyValidationTest(unittest.TestCase):
    def test_default_policies_match_contract(self):
        self.assertEqual((GNSS_FRESHNESS.expected_interval, GNSS_FRESHNESS.stale_after, GNSS_FRESHNESS.startup_grace), (0.10, 0.50, 1.00))
        self.assertEqual((IMU_FRESHNESS.expected_interval, IMU_FRESHNESS.stale_after, IMU_FRESHNESS.startup_grace), (0.05, 0.25, 0.50))
        self.assertEqual((RGB_CAMERA_FRESHNESS.expected_interval, RGB_CAMERA_FRESHNESS.stale_after, RGB_CAMERA_FRESHNESS.startup_grace), (0.10, 0.50, 1.00))

    def test_bool_and_string_are_rejected(self):
        for value in (True, False, "0.1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ContinuousFreshnessPolicy(value, 0.5, 1.0)

    def test_zero_and_negative_are_rejected(self):
        for value in (0, -0.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ContinuousFreshnessPolicy(value, 0.5, 1.0)

    def test_nan_and_infinity_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ContinuousFreshnessPolicy(0.1, value, 1.0)

    def test_contradictory_bounds_are_rejected(self):
        for values in ((0.6, 0.5, 1.0), (0.1, 1.1, 1.0), (0.1, 0.5, 10.1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ContinuousFreshnessPolicy(*values)

    def test_event_sensor_rejects_continuous_policy(self):
        with self.assertRaises(ValueError):
            SensorFreshnessConfig(collision=GNSS_FRESHNESS)

    def test_continuous_sensor_requires_explicit_positive_period(self):
        with self.assertRaises(ValueError):
            SensorFreshnessConfig(gnss=None)
        with self.assertRaises(ValueError):
            ContinuousFreshnessPolicy(0.0, 0.5, 1.0)


class ContinuousEvaluationTest(unittest.TestCase):
    def test_attached_without_sample_waits_within_grace(self):
        status = evaluate_sensor_status(continuous_evidence(), GNSS_FRESHNESS, 10.75)
        self.assertIs(status.readiness, SensorReadiness.WAITING_FOR_FIRST_SAMPLE)
        self.assertIs(status.freshness, SensorFreshness.WAITING_FOR_FIRST_SAMPLE)
        self.assertIs(status.operational_health, OperationalHealth.HEALTHY)

    def test_startup_grace_equality_still_waits(self):
        status = evaluate_sensor_status(continuous_evidence(), GNSS_FRESHNESS, 11.0)
        self.assertIs(status.freshness, SensorFreshness.WAITING_FOR_FIRST_SAMPLE)

    def test_beyond_startup_grace_is_stale_but_still_waiting(self):
        status = evaluate_sensor_status(continuous_evidence(), GNSS_FRESHNESS, math.nextafter(11.0, math.inf))
        self.assertIs(status.readiness, SensorReadiness.WAITING_FOR_FIRST_SAMPLE)
        self.assertIs(status.freshness, SensorFreshness.STALE)

    def test_first_sample_is_fresh(self):
        status = evaluate_sensor_status(continuous_evidence(accepted_count=1, last_receive_monotonic=20.0), GNSS_FRESHNESS, 20.0)
        self.assertIs(status.readiness, SensorReadiness.READY)
        self.assertIs(status.freshness, SensorFreshness.FRESH)

    def test_stale_threshold_equality_is_fresh(self):
        status = evaluate_sensor_status(continuous_evidence(accepted_count=1, last_receive_monotonic=20.0), IMU_FRESHNESS, 20.25)
        self.assertIs(status.freshness, SensorFreshness.FRESH)

    def test_one_representable_value_above_threshold_is_stale(self):
        evaluated = math.nextafter(20.25, math.inf)
        status = evaluate_sensor_status(continuous_evidence(accepted_count=1, last_receive_monotonic=20.0), IMU_FRESHNESS, evaluated)
        self.assertIs(status.freshness, SensorFreshness.STALE)

    def test_negative_age_is_clamped_to_zero(self):
        status = evaluate_sensor_status(continuous_evidence(accepted_count=1, last_receive_monotonic=21.0), GNSS_FRESHNESS, 20.0)
        self.assertEqual(status.age_seconds, 0.0)
        self.assertIs(status.freshness, SensorFreshness.FRESH)

    def test_active_callback_error_degrades_without_erasing_freshness(self):
        status = evaluate_sensor_status(continuous_evidence(accepted_count=1, last_receive_monotonic=20.0, active_error=True, error_count=1, active_error_reason="callback error"), GNSS_FRESHNESS, 20.1)
        self.assertIs(status.freshness, SensorFreshness.FRESH)
        self.assertIs(status.operational_health, OperationalHealth.DEGRADED)

    def test_clock_rollback_degrades_with_bounded_reason(self):
        status = evaluate_sensor_status(continuous_evidence(accepted_count=1, last_receive_monotonic=20.0), GNSS_FRESHNESS, 20.0, clock_rollback=True)
        self.assertIs(status.operational_health, OperationalHealth.DEGRADED)
        self.assertIn("rollback", status.reason)
        self.assertLessEqual(len(status.reason), 192)

    def test_nonfinite_receive_timestamp_is_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate_sensor_status(continuous_evidence(accepted_count=1, last_receive_monotonic=value), GNSS_FRESHNESS, 20.0)


class EventEvaluationTest(unittest.TestCase):
    def test_zero_events_are_healthy_after_indefinite_silence(self):
        status = evaluate_sensor_status(event_evidence(), None, 1.0e9)
        self.assertIs(status.freshness, SensorFreshness.NOT_APPLICABLE)
        self.assertIs(status.readiness, SensorReadiness.NOT_APPLICABLE)
        self.assertIs(status.operational_health, OperationalHealth.HEALTHY)
        self.assertEqual(status.event_count, 0)

    def test_event_arrival_updates_count_without_freshness(self):
        status = evaluate_sensor_status(event_evidence(accepted_count=3, event_count=3, last_receive_monotonic=99.0), None, 500.0)
        self.assertEqual(status.event_count, 3)
        self.assertIs(status.freshness, SensorFreshness.NOT_APPLICABLE)

    def test_event_callback_error_degrades(self):
        status = evaluate_sensor_status(event_evidence(active_error=True, error_count=1), None, 100.0)
        self.assertIs(status.operational_health, OperationalHealth.DEGRADED)

    def test_later_valid_event_recovers_active_health_and_keeps_history(self):
        status = evaluate_sensor_status(event_evidence(accepted_count=1, event_count=1, active_error=False, error_count=1), None, 100.0)
        self.assertIs(status.operational_health, OperationalHealth.HEALTHY)
        self.assertEqual(status.error_count, 1)


class LifecycleAndRecordTest(unittest.TestCase):
    def test_disabled_is_not_available(self):
        status = evaluate_sensor_status(continuous_evidence(enabled=False, attached=False), GNSS_FRESHNESS, 1.0)
        self.assertIs(status.lifecycle, SensorLifecycle.NOT_AVAILABLE)

    def test_pre_attach_is_not_attached(self):
        status = evaluate_sensor_status(continuous_evidence(attach_attempted=False, attached=False, attachment_monotonic=None), GNSS_FRESHNESS, 1.0)
        self.assertIs(status.lifecycle, SensorLifecycle.NOT_ATTACHED)

    def test_attach_failure_is_failed(self):
        status = evaluate_sensor_status(continuous_evidence(attached=False, attachment_failed=True), GNSS_FRESHNESS, 1.0)
        self.assertIs(status.lifecycle, SensorLifecycle.ATTACHMENT_FAILED)
        self.assertIs(status.operational_health, OperationalHealth.FAILED)

    def test_destroyed_is_final(self):
        status = evaluate_sensor_status(continuous_evidence(destroyed=True, accepted_count=1, last_receive_monotonic=1.0), GNSS_FRESHNESS, 1.0)
        self.assertIs(status.lifecycle, SensorLifecycle.DESTROYED)
        self.assertIs(status.freshness, SensorFreshness.NOT_APPLICABLE)

    def test_status_is_frozen_and_slotted(self):
        status = evaluate_sensor_status(continuous_evidence(), GNSS_FRESHNESS, 10.0)
        self.assertFalse(hasattr(status, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            status.age_seconds = 9.0


class SuiteSnapshotContractTest(unittest.TestCase):
    def test_pre_attach_and_disabled_lifecycle_projection(self):
        clock = NumericClock(10.0)
        suite = SensorSuite(
            SensorSuiteConfig(enable_imu=False), monotonic_clock=clock
        )
        snapshot = suite.snapshot()
        self.assertIs(
            status_by_name(snapshot, "gnss").lifecycle,
            SensorLifecycle.NOT_ATTACHED,
        )
        self.assertIs(
            status_by_name(snapshot, "imu").lifecycle,
            SensorLifecycle.NOT_AVAILABLE,
        )

    def test_snapshot_reads_clock_once_and_shares_evaluation_time(self):
        clock, _world, suite, _report = attach_suite()
        clock.calls = 0
        clock.value = 123.5
        snapshot = suite.snapshot()
        self.assertEqual(clock.calls, 1)
        self.assertEqual(snapshot.captured_monotonic, 123.5)
        self.assertEqual(
            {status.evaluated_monotonic for status in snapshot.statuses},
            {123.5},
        )
        suite.destroy()

    def test_status_collection_is_tuple_only_and_contains_no_live_objects(self):
        _clock, _world, suite, _report = attach_suite()
        snapshot = suite.snapshot()
        self.assertIs(type(snapshot.statuses), tuple)
        for status in snapshot.statuses:
            self.assertFalse(hasattr(status, "__dict__"))
            for field in fields(status):
                value = getattr(status, field.name)
                self.assertNotIn("lock", type(value).__name__.lower())
                self.assertFalse(callable(value))
                self.assertFalse(hasattr(value, "listen"))
        suite.destroy()

    def test_old_snapshots_remain_unchanged(self):
        clock, world, suite, _report = attach_suite()
        before = suite.snapshot()
        clock.value = 101.0
        world.spawned[GNSS_BLUEPRINT].emit(continuous_measurement("gnss", 1, 1.0))
        after = suite.snapshot()
        self.assertEqual(status_by_name(before, "gnss").accepted_count, 0)
        self.assertEqual(status_by_name(after, "gnss").accepted_count, 1)
        suite.destroy()

    def test_clock_rollback_clamps_degrades_and_recovers(self):
        clock, _world, suite, _report = attach_suite()
        clock.value = 120.0
        coherent = suite.snapshot()
        clock.value = 119.0
        rolled_back = suite.snapshot()
        self.assertEqual(rolled_back.captured_monotonic, 120.0)
        self.assertTrue(
            all(status.age_seconds is None or status.age_seconds >= 0.0 for status in rolled_back.statuses)
        )
        self.assertTrue(
            all(
                status.operational_health is OperationalHealth.DEGRADED
                for status in rolled_back.statuses
            )
        )
        clock.value = 120.0
        recovered = suite.snapshot()
        self.assertTrue(
            all(
                status.operational_health is OperationalHealth.HEALTHY
                for status in recovered.statuses
            )
        )
        self.assertEqual(coherent.captured_monotonic, 120.0)
        suite.destroy()

    def test_destroyed_suite_never_projects_historical_freshness(self):
        clock, world, suite, _report = attach_suite()
        world.spawned[GNSS_BLUEPRINT].emit(continuous_measurement("gnss", 1, 1.0))
        self.assertIs(status_by_name(suite.snapshot(), "gnss").freshness, SensorFreshness.FRESH)
        suite.destroy()
        destroyed = status_by_name(suite.snapshot(), "gnss")
        self.assertIs(destroyed.lifecycle, SensorLifecycle.DESTROYED)
        self.assertIs(destroyed.freshness, SensorFreshness.NOT_APPLICABLE)


class ContinuousSuiteContractTest(unittest.TestCase):
    def test_full_boundary_and_stale_recovery_contract_for_all_continuous_sensors(self):
        policies = {
            "gnss": GNSS_FRESHNESS,
            "imu": IMU_FRESHNESS,
            "rgb_camera": RGB_CAMERA_FRESHNESS,
        }
        for name, policy in policies.items():
            with self.subTest(sensor=name):
                clock, world, suite, _report = attach_suite()
                actor = world.spawned[CONTINUOUS_ACTORS[name]]
                actor.emit(continuous_measurement(name, 10, 1.0))
                first = status_by_name(suite.snapshot(), name)
                self.assertIs(first.freshness, SensorFreshness.FRESH)

                clock.value = 100.0 + policy.stale_after
                equality = status_by_name(suite.snapshot(), name)
                self.assertIs(equality.freshness, SensorFreshness.FRESH)

                clock.value = math.nextafter(clock.value, math.inf)
                stale = status_by_name(suite.snapshot(), name)
                self.assertIs(stale.freshness, SensorFreshness.STALE)

                actor.emit(continuous_measurement(name, 11, 1.1))
                recovered = status_by_name(suite.snapshot(), name)
                self.assertIs(recovered.freshness, SensorFreshness.FRESH)
                self.assertIs(recovered.operational_health, OperationalHealth.HEALTHY)
                suite.destroy()

    def test_rejected_duplicate_and_old_source_never_refresh_age(self):
        for name in CONTINUOUS_ACTORS:
            with self.subTest(sensor=name):
                clock, world, suite, _report = attach_suite()
                actor = world.spawned[CONTINUOUS_ACTORS[name]]
                actor.emit(continuous_measurement(name, 10, 1.0))
                clock.value = 101.0
                before = status_by_name(suite.snapshot(), name)

                actor.emit(continuous_measurement(name, 10, 1.0))
                duplicate = status_by_name(suite.snapshot(), name)
                self.assertEqual(duplicate.accepted_count, 1)
                self.assertEqual(duplicate.age_seconds, before.age_seconds)

                actor.emit(continuous_measurement(name, 9, 0.9))
                old = status_by_name(suite.snapshot(), name)
                self.assertEqual(old.accepted_count, 1)
                self.assertEqual(old.age_seconds, before.age_seconds)
                self.assertIs(old.operational_health, OperationalHealth.DEGRADED)
                suite.destroy()

    def test_callback_error_then_valid_sample_recovers_active_health(self):
        for name in CONTINUOUS_ACTORS:
            with self.subTest(sensor=name):
                clock, world, suite, _report = attach_suite()
                actor = world.spawned[CONTINUOUS_ACTORS[name]]
                actor.emit(None)
                degraded = status_by_name(suite.snapshot(), name)
                self.assertIs(degraded.operational_health, OperationalHealth.DEGRADED)
                self.assertEqual(degraded.error_count, 1)

                actor.emit(continuous_measurement(name, 1, 1.0))
                recovered = status_by_name(suite.snapshot(), name)
                self.assertIs(recovered.operational_health, OperationalHealth.HEALTHY)
                self.assertEqual(recovered.error_count, 1)
                self.assertIs(recovered.freshness, SensorFreshness.FRESH)
                suite.destroy()

    def test_nonfinite_receive_timestamp_is_not_published(self):
        clock, world, suite, _report = attach_suite()
        clock.value = math.nan
        world.spawned[GNSS_BLUEPRINT].emit(continuous_measurement("gnss", 1, 1.0))
        clock.value = 100.0
        snapshot = suite.snapshot()
        self.assertIsNone(snapshot.gnss)
        status = status_by_name(snapshot, "gnss")
        self.assertEqual(status.accepted_count, 0)
        self.assertIs(status.operational_health, OperationalHealth.DEGRADED)
        suite.destroy()

    def test_source_regression_recovers_only_after_coherent_progression(self):
        for name in CONTINUOUS_ACTORS:
            with self.subTest(sensor=name):
                _clock, world, suite, _report = attach_suite()
                actor = world.spawned[CONTINUOUS_ACTORS[name]]
                actor.emit(continuous_measurement(name, 10, 1.0))
                actor.emit(continuous_measurement(name, 9, 0.9))
                degraded = status_by_name(suite.snapshot(), name)
                self.assertIs(degraded.operational_health, OperationalHealth.DEGRADED)
                self.assertEqual(degraded.accepted_count, 1)

                actor.emit(continuous_measurement(name, 11, 1.1))
                recovered = status_by_name(suite.snapshot(), name)
                self.assertIs(recovered.operational_health, OperationalHealth.HEALTHY)
                self.assertEqual(recovered.accepted_count, 2)
                self.assertGreaterEqual(recovered.error_count, 1)
                suite.destroy()

    def test_camera_newer_frame_with_regressed_source_time_is_rejected(self):
        _clock, world, suite, _report = attach_suite()
        actor = world.spawned[RGB_CAMERA_BLUEPRINT]
        actor.emit(FakeCameraImage(frame=10, timestamp=2.0))
        actor.emit(FakeCameraImage(frame=11, timestamp=1.9))
        snapshot = suite.snapshot()
        self.assertEqual(snapshot.camera.metadata.stamp.frame, 10)
        status = status_by_name(snapshot, "rgb_camera")
        self.assertEqual(status.accepted_count, 1)
        self.assertIs(status.operational_health, OperationalHealth.DEGRADED)
        suite.destroy()


class EventSuiteContractTest(unittest.TestCase):
    def test_zero_events_remain_healthy_after_arbitrary_silence(self):
        clock, _world, suite, _report = attach_suite()
        clock.value = 1.0e9
        snapshot = suite.snapshot()
        for name in ("collision", "lane_invasion"):
            status = status_by_name(snapshot, name)
            self.assertEqual(status.event_count, 0)
            self.assertIs(status.operational_health, OperationalHealth.HEALTHY)
            self.assertIs(status.freshness, SensorFreshness.NOT_APPLICABLE)
        suite.destroy()

    def test_event_arrival_error_and_valid_event_recovery(self):
        cases = (
            ("collision", COLLISION_BLUEPRINT, collision_measurement),
            ("lane_invasion", LANE_INVASION_BLUEPRINT, lane_measurement),
        )
        for name, blueprint, measurement in cases:
            with self.subTest(sensor=name):
                clock, world, suite, _report = attach_suite()
                actor = world.spawned[blueprint]
                actor.emit(measurement(frame=1))
                first = status_by_name(suite.snapshot(), name)
                self.assertEqual(first.event_count, 1)
                actor.emit(None)
                degraded = status_by_name(suite.snapshot(), name)
                self.assertIs(degraded.operational_health, OperationalHealth.DEGRADED)
                actor.emit(measurement(frame=2, timestamp=2.0))
                recovered = status_by_name(suite.snapshot(), name)
                self.assertIs(recovered.operational_health, OperationalHealth.HEALTHY)
                self.assertEqual(recovered.event_count, 2)
                self.assertEqual(recovered.error_count, 1)
                suite.destroy()

    def test_lane_zero_events_error_is_degraded_then_first_event_recovers(self):
        _clock, world, suite, _report = attach_suite()
        actor = world.spawned[LANE_INVASION_BLUEPRINT]
        actor.emit(None)
        degraded = status_by_name(suite.snapshot(), "lane_invasion")
        self.assertEqual(degraded.event_count, 0)
        self.assertIs(degraded.freshness, SensorFreshness.NOT_APPLICABLE)
        self.assertIs(degraded.operational_health, OperationalHealth.DEGRADED)

        actor.emit(lane_measurement(frame=1))
        recovered = status_by_name(suite.snapshot(), "lane_invasion")
        self.assertEqual(recovered.event_count, 1)
        self.assertEqual(recovered.error_count, 1)
        self.assertIs(recovered.operational_health, OperationalHealth.HEALTHY)
        suite.destroy()


class RawEvidenceContractTest(unittest.TestCase):
    def test_error_text_and_reason_are_bounded(self):
        aggregator = TelemetryAggregator()
        aggregator.register("gnss")
        aggregator.record_error("gnss", "secret:" + "x" * 1000, 10.0, "y" * 1000)
        health = aggregator.snapshot().sensors[0]
        self.assertLessEqual(len(health.last_error), 192)
        self.assertLessEqual(len(health.active_error_reason), 192)
        self.assertEqual(health.last_error_monotonic, 10.0)

    def test_nonfinite_simulation_timestamp_is_not_published(self):
        aggregator = TelemetryAggregator()
        aggregator.register("gnss")
        aggregator.mark_attached("gnss", True, 1.0)
        accepted = aggregator.submit_gnss(
            GnssState(SampleStamp(1, math.nan, 3.0), 1.0, 2.0, 3.0)
        )
        self.assertFalse(accepted)
        snapshot = aggregator.snapshot()
        self.assertIsNone(snapshot.gnss)
        self.assertTrue(snapshot.sensors[0].active_error)

    def test_valid_sample_clears_active_error_but_keeps_cumulative_history(self):
        aggregator = TelemetryAggregator()
        aggregator.register("gnss")
        aggregator.mark_attached("gnss", True, 1.0)
        aggregator.record_error("gnss", "boom", 2.0)
        accepted = aggregator.submit_gnss(
            GnssState(SampleStamp(1, 1.0, 3.0), 1.0, 2.0, 3.0)
        )
        self.assertTrue(accepted)
        health = aggregator.snapshot().sensors[0]
        self.assertFalse(health.active_error)
        self.assertEqual(health.error_count, 1)
        self.assertEqual(health.last_error, "boom")

    def test_failed_attachment_cannot_be_rewritten_by_late_accepted_sample(self):
        clock, world, suite, report = attach_suite(
            fail_listen=(RGB_CAMERA_BLUEPRINT,)
        )
        self.assertFalse(next(r for r in report.results if r.name == "rgb_camera").attached)
        actor = world.spawned[RGB_CAMERA_BLUEPRINT]
        actor.emit(FakeCameraImage(frame=1))
        status = status_by_name(suite.snapshot(), "rgb_camera")
        self.assertIs(status.lifecycle, SensorLifecycle.ATTACHMENT_FAILED)
        self.assertIs(status.operational_health, OperationalHealth.FAILED)
        suite.destroy()

    def test_destroy_failure_still_produces_final_destroyed_lifecycle(self):
        _clock, _world, suite, _report = attach_suite(
            fail_destroy=(GNSS_BLUEPRINT,)
        )
        cleanup = suite.destroy()
        self.assertTrue(next(r for r in cleanup.results if r.name == "gnss").errors)
        self.assertIs(
            status_by_name(suite.snapshot(), "gnss").lifecycle,
            SensorLifecycle.DESTROYED,
        )


if __name__ == "__main__":
    unittest.main()
