"""Unit tests for telemetry records and the aggregator (no CARLA needed)."""

import math
import unittest
from dataclasses import FrozenInstanceError

from telemetry.aggregator import TelemetryAggregator
from telemetry.state import (
    ActorRef,
    CollisionEvent,
    EgoKinematics,
    GnssState,
    ImuState,
    LaneInvasionEvent,
    LaneMarkingInfo,
    SampleStamp,
    TransformState,
    Vector3,
    EulerDegrees,
    normalize_heading,
    speed_from_velocity,
)


def make_stamp(frame=1, sim=0.1, mono=0.0):
    return SampleStamp(frame=frame, sim_time=sim, monotonic=mono)


def make_gnss(lat=1.0, lon=2.0, alt=3.0, frame=1):
    return GnssState(
        stamp=make_stamp(frame=frame),
        latitude=lat,
        longitude=lon,
        altitude=alt,
        transform=None,
    )


def make_imu(frame=1):
    return ImuState(
        stamp=make_stamp(frame=frame),
        accelerometer=Vector3(0.0, 0.0, 9.81),
        gyroscope=Vector3(0.0, 0.0, 0.0),
        compass_rad=0.0,
        heading_deg=0.0,
        transform=None,
    )


class RecordImmutabilityTest(unittest.TestCase):
    def test_records_are_immutable(self):
        vector = Vector3(1.0, 2.0, 3.0)
        with self.assertRaises(FrozenInstanceError):
            vector.x = 9.0
        stamp = make_stamp()
        with self.assertRaises(FrozenInstanceError):
            stamp.frame = 5
        gnss = make_gnss()
        with self.assertRaises(FrozenInstanceError):
            gnss.latitude = 0.0


class MathHelperTest(unittest.TestCase):
    def test_speed_conversion_is_correct(self):
        mps, kph = speed_from_velocity(3.0, 4.0, 0.0)
        self.assertAlmostEqual(mps, 5.0)
        self.assertAlmostEqual(kph, 18.0)
        # 1 m/s is exactly 3.6 km/h.
        mps2, kph2 = speed_from_velocity(1.0, 0.0, 0.0)
        self.assertAlmostEqual(mps2, 1.0)
        self.assertAlmostEqual(kph2, 3.6)
        # 3D magnitude.
        mps3, _ = speed_from_velocity(1.0, 2.0, 2.0)
        self.assertAlmostEqual(mps3, 3.0)

    def test_heading_normalization_is_correct(self):
        self.assertAlmostEqual(normalize_heading(0.0), 0.0)
        self.assertAlmostEqual(normalize_heading(math.pi), 180.0)
        # Beyond 2*pi wraps back into range: 450 deg -> 90 deg.
        self.assertAlmostEqual(normalize_heading(2 * math.pi + math.pi / 2), 90.0, places=6)
        # Negative angles wrap into [0, 360): -90 deg -> 270 deg.
        negative = normalize_heading(-math.pi / 2)
        self.assertAlmostEqual(negative, 270.0, places=6)
        self.assertTrue(0.0 <= negative < 360.0)


class AggregatorTest(unittest.TestCase):
    def test_initial_gnss_and_imu_are_none(self):
        agg = TelemetryAggregator()
        snap = agg.snapshot()
        self.assertIsNone(snap.gnss)
        self.assertIsNone(snap.imu)
        self.assertIsNone(snap.ego)
        self.assertEqual(snap.revision, 0)

    def test_revision_increments_on_accepted_updates(self):
        agg = TelemetryAggregator()
        r0 = agg.snapshot().revision
        agg.submit_gnss(make_gnss())
        r1 = agg.snapshot().revision
        agg.submit_imu(make_imu())
        r2 = agg.snapshot().revision
        ego = EgoKinematics(
            stamp=make_stamp(),
            transform=TransformState(Vector3(0, 0, 0), EulerDegrees(0, 0, 0)),
            velocity=Vector3(0, 0, 0),
            speed_mps=0.0,
            speed_kph=0.0,
            vehicle_alive=True,
        )
        agg.update_ego(ego)
        r3 = agg.snapshot().revision
        self.assertEqual(r1, r0 + 1)
        self.assertEqual(r2, r1 + 1)
        self.assertEqual(r3, r2 + 1)

    def test_post_shutdown_updates_are_ignored(self):
        agg = TelemetryAggregator()
        agg.submit_gnss(make_gnss(lat=1.0))
        revision_before = agg.snapshot().revision
        gnss_before = agg.snapshot().gnss

        agg.begin_shutdown()
        agg.submit_gnss(make_gnss(lat=99.0))
        agg.submit_imu(make_imu())

        self.assertFalse(agg.accepting)
        self.assertEqual(agg.snapshot().revision, revision_before)
        self.assertEqual(agg.snapshot().gnss, gnss_before)
        self.assertIsNone(agg.snapshot().imu)


def make_collision(frame=1, other_id=200, monotonic=0.0):
    return CollisionEvent(
        stamp=SampleStamp(frame=frame, sim_time=0.1, monotonic=monotonic),
        self_actor=ActorRef(10, "vehicle.tesla.model3"),
        other_actor=ActorRef(other_id, "vehicle.audi.tt"),
        normal_impulse=Vector3(1.0, 0.0, 0.0),
        impulse_magnitude=1.0,
        transform=None,
    )


def make_lane_invasion(frame=1, sim_time=1.0):
    return LaneInvasionEvent(
        stamp=SampleStamp(frame=frame, sim_time=sim_time, monotonic=0.0),
        crossed_markings=(LaneMarkingInfo("Solid", "White", "NONE", 0.15),),
        transform=None,
    )


class EventSequenceTest(unittest.TestCase):
    """Mission 4F preparation: monotonic accepted-event counters."""

    def test_initial_sequences_are_zero(self):
        snap = TelemetryAggregator().snapshot()
        self.assertEqual(snap.collision_sequence, 0)
        self.assertEqual(snap.lane_invasion_sequence, 0)

    def test_collision_sequence_increments_only_for_accepted(self):
        agg = TelemetryAggregator()
        self.assertTrue(agg.submit_collision(make_collision(frame=1)))
        self.assertEqual(agg.snapshot().collision_sequence, 1)
        # An exact duplicate (only host receive time differs) is not counted.
        self.assertFalse(agg.submit_collision(make_collision(frame=1, monotonic=9.9)))
        self.assertEqual(agg.snapshot().collision_sequence, 1)
        self.assertTrue(agg.submit_collision(make_collision(frame=2)))
        self.assertEqual(agg.snapshot().collision_sequence, 2)

    def test_lane_sequence_increments_only_for_accepted(self):
        agg = TelemetryAggregator()
        self.assertTrue(
            agg.submit_lane_invasion(make_lane_invasion(frame=1, sim_time=1.0))
        )
        self.assertEqual(agg.snapshot().lane_invasion_sequence, 1)
        # A debounced repeat (same signature 0.1 s later) is not counted.
        self.assertFalse(
            agg.submit_lane_invasion(make_lane_invasion(frame=2, sim_time=1.1))
        )
        self.assertEqual(agg.snapshot().lane_invasion_sequence, 1)
        self.assertTrue(
            agg.submit_lane_invasion(make_lane_invasion(frame=3, sim_time=1.4))
        )
        self.assertEqual(agg.snapshot().lane_invasion_sequence, 2)

    def test_sequences_do_not_increment_after_shutdown(self):
        agg = TelemetryAggregator()
        agg.submit_collision(make_collision(frame=1))
        agg.submit_lane_invasion(make_lane_invasion(frame=1, sim_time=1.0))
        agg.begin_shutdown()
        agg.submit_collision(make_collision(frame=2))
        agg.submit_lane_invasion(make_lane_invasion(frame=2, sim_time=9.0))
        snap = agg.snapshot()
        self.assertEqual(snap.collision_sequence, 1)
        self.assertEqual(snap.lane_invasion_sequence, 1)

    def test_sequences_stay_monotonic_across_history_eviction(self):
        agg = TelemetryAggregator()
        for i in range(200):
            agg.submit_collision(make_collision(frame=i, other_id=200 + i))
        snap = agg.snapshot()
        self.assertEqual(len(snap.collisions), 128)   # history stays bounded
        self.assertEqual(snap.collision_sequence, 200)  # counter keeps going


if __name__ == "__main__":
    unittest.main()
