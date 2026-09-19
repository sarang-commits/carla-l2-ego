"""Lane-invasion sensor: sensor.other.lane_invasion.

This is *marking-crossing* telemetry -- it reports which lane markings the ego
crossed, as reported by the simulator -- NOT camera-based lane detection.
Event-driven, rigidly attached at the ego origin. Crossed markings are copied
into immutable records with stable enum-to-string names and returned in a
deterministic sorted order, with duplicates preserved.
"""

from __future__ import annotations

import time

from sensors.base import BaseSensor
from telemetry.state import LaneInvasionEvent, LaneMarkingInfo, SampleStamp, transform_state

LANE_INVASION_BLUEPRINT = "sensor.other.lane_invasion"
LANE_INVASION_NAME = "lane_invasion"


def enum_name(value) -> str:
    """Stable enum-to-string: prefer ``.name``, else strip an ``Enum.`` prefix.

    Works for real CARLA enums (which expose ``.name``), plain strings, and any
    object whose ``str()`` looks like ``carla.LaneMarkingType.Solid``.
    """
    name = getattr(value, "name", None)
    if name is not None:
        return str(name)
    return str(value).rsplit(".", 1)[-1]


def lane_marking_info(marking) -> LaneMarkingInfo:
    """Copy one crossed lane marking into an immutable record."""
    return LaneMarkingInfo(
        type=enum_name(marking.type),
        color=enum_name(marking.color),
        lane_change=enum_name(marking.lane_change),
        width=float(marking.width),
    )


def parse_lane_invasion(measurement, receive_monotonic=None) -> LaneInvasionEvent:
    """Copy a CARLA lane-invasion event into an immutable record (no retention).

    Markings are sorted by (type, color, lane_change, width) for a deterministic
    order; the sort is stable so duplicate markings are preserved.
    """
    markings = tuple(
        sorted(
            (lane_marking_info(m) for m in measurement.crossed_lane_markings),
            key=lambda mi: (mi.type, mi.color, mi.lane_change, mi.width),
        )
    )
    return LaneInvasionEvent(
        stamp=SampleStamp(
            frame=int(measurement.frame),
            sim_time=float(measurement.timestamp),
            monotonic=(
                time.monotonic() if receive_monotonic is None else receive_monotonic
            ),
        ),
        crossed_markings=markings,
        transform=transform_state(measurement.transform),
    )


class LaneInvasionSensor(BaseSensor):
    name = LANE_INVASION_NAME
    blueprint_id = LANE_INVASION_BLUEPRINT

    def __init__(self, aggregator, tick: float = 0.0, monotonic_clock=None) -> None:
        # Event sensor: sensor_tick does not apply.
        super().__init__(aggregator, tick, monotonic_clock)

    def configure_blueprint(self, blueprint) -> None:
        # Nothing to configure: no noise model, no tick.
        pass

    def parse(self, measurement, receive_monotonic=None) -> LaneInvasionEvent:
        return parse_lane_invasion(measurement, receive_monotonic)

    def submit(self, state: LaneInvasionEvent) -> None:
        self._aggregator.submit_lane_invasion(state)
