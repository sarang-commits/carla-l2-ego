"""Unit tests for GNSS/IMU parsing: copying, no clamping, no CARLA retention."""

import gc
import math
import unittest
import weakref

from sensors.gnss import parse_gnss
from sensors.imu import parse_imu
from telemetry.state import ImuState, TransformState
from tests.fakes import (
    FakeGnssMeasurement,
    FakeImuMeasurement,
    FakeRotation,
    FakeTransform,
    FakeVector,
)


class GnssParseTest(unittest.TestCase):
    def test_gnss_values_are_copied(self):
        measurement = FakeGnssMeasurement(
            frame=7,
            timestamp=1.5,
            latitude=12.34,
            longitude=56.78,
            altitude=9.0,
            transform=FakeTransform(FakeVector(1.0, 2.0, 3.0), FakeRotation(yaw=90.0)),
        )
        state = parse_gnss(measurement)
        self.assertEqual(state.latitude, 12.34)
        self.assertEqual(state.longitude, 56.78)
        self.assertEqual(state.altitude, 9.0)
        self.assertEqual(state.stamp.frame, 7)
        self.assertEqual(state.stamp.sim_time, 1.5)
        self.assertEqual(state.transform.location.x, 1.0)
        self.assertEqual(state.transform.rotation.yaw, 90.0)

    def test_carla_objects_are_not_retained(self):
        measurement = FakeGnssMeasurement(
            frame=1,
            timestamp=0.5,
            latitude=1.0,
            longitude=2.0,
            altitude=3.0,
            transform=FakeTransform(FakeVector(4.0, 5.0, 6.0)),
        )
        state = parse_gnss(measurement)
        # The record holds only immutable telemetry types.
        self.assertIsInstance(state.transform, TransformState)
        # And the CARLA-shaped measurement is not referenced anywhere.
        ref = weakref.ref(measurement)
        del measurement
        gc.collect()
        self.assertIsNone(ref())


class ImuParseTest(unittest.TestCase):
    def test_imu_values_are_copied_without_clamping(self):
        big = 1.0e9
        measurement = FakeImuMeasurement(
            frame=3,
            timestamp=2.0,
            accelerometer=FakeVector(big, -big, 12345.678),
            gyroscope=FakeVector(0.1, -0.2, 0.3),
            compass=math.pi,  # 180 degrees
            transform=FakeTransform(),
        )
        state = parse_imu(measurement)
        self.assertIsInstance(state, ImuState)
        # No clamping: extreme values survive verbatim.
        self.assertEqual(state.accelerometer.x, big)
        self.assertEqual(state.accelerometer.y, -big)
        self.assertEqual(state.accelerometer.z, 12345.678)
        self.assertEqual(state.gyroscope.x, 0.1)
        self.assertEqual(state.gyroscope.y, -0.2)
        self.assertEqual(state.gyroscope.z, 0.3)
        self.assertAlmostEqual(state.compass_rad, math.pi)
        self.assertAlmostEqual(state.heading_deg, 180.0)

    def test_imu_carla_objects_are_not_retained(self):
        measurement = FakeImuMeasurement(
            frame=1,
            timestamp=0.1,
            accelerometer=FakeVector(1.0, 2.0, 3.0),
            gyroscope=FakeVector(0.0, 0.0, 0.0),
            compass=0.0,
            transform=FakeTransform(),
        )
        parse_imu(measurement)
        ref = weakref.ref(measurement)
        del measurement
        gc.collect()
        self.assertIsNone(ref())


if __name__ == "__main__":
    unittest.main()
