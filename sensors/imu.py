"""IMU sensor: sensor.other.imu, noiseless, rigidly fixed at vehicle origin.

Values are copied verbatim -- accelerometer stays in m/s^2, gyroscope in
rad/s, compass in radians -- and are never clamped.
"""

from __future__ import annotations

import time

from sensors.base import BaseSensor, set_blueprint_attribute
from telemetry.state import ImuState, SampleStamp, Vector3, normalize_heading, transform_state

IMU_BLUEPRINT = "sensor.other.imu"
IMU_NAME = "imu"

# Every IMU noise stddev/bias is disabled and the seed pinned to 0.
_IMU_NOISE_ATTRS = (
    "noise_accel_stddev_x",
    "noise_accel_stddev_y",
    "noise_accel_stddev_z",
    "noise_gyro_bias_x",
    "noise_gyro_bias_y",
    "noise_gyro_bias_z",
    "noise_gyro_stddev_x",
    "noise_gyro_stddev_y",
    "noise_gyro_stddev_z",
)


def parse_imu(measurement, receive_monotonic=None) -> ImuState:
    """Copy a CARLA IMU measurement into an immutable record (no clamping)."""
    accel = measurement.accelerometer
    gyro = measurement.gyroscope
    compass_rad = float(measurement.compass)
    return ImuState(
        stamp=SampleStamp(
            frame=int(measurement.frame),
            sim_time=float(measurement.timestamp),
            monotonic=(
                time.monotonic() if receive_monotonic is None else receive_monotonic
            ),
        ),
        accelerometer=Vector3(float(accel.x), float(accel.y), float(accel.z)),
        gyroscope=Vector3(float(gyro.x), float(gyro.y), float(gyro.z)),
        compass_rad=compass_rad,
        heading_deg=normalize_heading(compass_rad),
        transform=transform_state(measurement.transform),
    )


class ImuSensor(BaseSensor):
    name = IMU_NAME
    blueprint_id = IMU_BLUEPRINT

    def __init__(self, aggregator, tick: float = 0.05, monotonic_clock=None) -> None:
        super().__init__(aggregator, tick, monotonic_clock)

    def configure_blueprint(self, blueprint) -> None:
        for attr in _IMU_NOISE_ATTRS:
            set_blueprint_attribute(blueprint, attr, 0.0)
        set_blueprint_attribute(blueprint, "noise_seed", 0)

    def parse(self, measurement, receive_monotonic=None) -> ImuState:
        return parse_imu(measurement, receive_monotonic)

    def submit(self, state: ImuState) -> None:
        self._aggregator.submit_imu(state)
