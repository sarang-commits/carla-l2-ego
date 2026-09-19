"""GNSS sensor: sensor.other.gnss, noiseless, rigidly fixed at vehicle origin."""

from __future__ import annotations

import time

from sensors.base import BaseSensor, set_blueprint_attribute
from telemetry.state import GnssState, SampleStamp, transform_state

GNSS_BLUEPRINT = "sensor.other.gnss"
GNSS_NAME = "gnss"

# Every GNSS noise bias/stddev is disabled and the seed pinned to 0.
_GNSS_NOISE_ATTRS = (
    "noise_alt_bias",
    "noise_alt_stddev",
    "noise_lat_bias",
    "noise_lat_stddev",
    "noise_lon_bias",
    "noise_lon_stddev",
)


def parse_gnss(measurement, receive_monotonic=None) -> GnssState:
    """Copy a CARLA GNSS measurement into an immutable record (no retention)."""
    return GnssState(
        stamp=SampleStamp(
            frame=int(measurement.frame),
            sim_time=float(measurement.timestamp),
            monotonic=(
                time.monotonic() if receive_monotonic is None else receive_monotonic
            ),
        ),
        latitude=float(measurement.latitude),
        longitude=float(measurement.longitude),
        altitude=float(measurement.altitude),
        transform=transform_state(measurement.transform),
    )


class GnssSensor(BaseSensor):
    name = GNSS_NAME
    blueprint_id = GNSS_BLUEPRINT

    def __init__(self, aggregator, tick: float = 0.10, monotonic_clock=None) -> None:
        super().__init__(aggregator, tick, monotonic_clock)

    def configure_blueprint(self, blueprint) -> None:
        for attr in _GNSS_NOISE_ATTRS:
            set_blueprint_attribute(blueprint, attr, 0.0)
        set_blueprint_attribute(blueprint, "noise_seed", 0)

    def parse(self, measurement, receive_monotonic=None) -> GnssState:
        return parse_gnss(measurement, receive_monotonic)

    def submit(self, state: GnssState) -> None:
        self._aggregator.submit_gnss(state)
