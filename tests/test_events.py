"""Unit tests for Mission 4D: collision and lane-invasion event telemetry.

No CARLA server or package is required. Collision/lane measurement fakes are
defined here (the generic world/vehicle/transform fakes are reused from
tests.fakes) so tests/fakes.py stays untouched.
"""

import gc
import unittest
import weakref
from dataclasses import FrozenInstanceError

from sensors.collision import COLLISION_BLUEPRINT, parse_collision
from sensors.gnss import GNSS_BLUEPRINT
from sensors.imu import IMU_BLUEPRINT
from sensors.lane_invasion import LANE_INVASION_BLUEPRINT, enum_name, parse_lane_invasion
from sensors.rgb_camera import RGB_CAMERA_BLUEPRINT
from sensors.suite import SensorSuite, SensorSuiteConfig
from telemetry.aggregator import (
    LANE_DEBOUNCE_CACHE,
    TelemetryAggregator,
)
from telemetry.state import ActorRef, LaneMarkingInfo, TransformState, vector_magnitude
from tests.fakes import FakeTransform, FakeVector, FakeVehicle, FakeWorld


# -- local CARLA-shaped fakes for events ----------------------------------


class FakeActor:
    def __init__(self, actor_id, type_id):
        self.id = actor_id
        self.type_id = type_id


class FakeCollisionMeasurement:
    def __init__(self, frame, timestamp, actor, other_actor, normal_impulse, transform=None):
        self.frame = frame
        self.timestamp = timestamp
        self.actor = actor
        self.other_actor = other_actor
        self.normal_impulse = normal_impulse
        self.transform = transform if transform is not None else FakeTransform()


class FakeEnum:
    """Enum-like object exposing .name; its str() carries a dotted prefix."""

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"carla.Enum.{self.name}"


class FakePrefixedEnum:
    """Enum-like object with no .name, only a dotted str() (fallback path)."""

    def __init__(self, text):
        self._text = text

    def __str__(self):
        return self._text


class FakeLaneMarking:
    def __init__(self, type, color, lane_change, width):
        self.type = type
        self.color = color
        self.lane_change = lane_change
        self.width = width


class FakeLaneInvasionMeasurement:
    def __init__(self, frame, timestamp, crossed_lane_markings, transform=None):
        self.frame = frame
        self.timestamp = timestamp
        self.crossed_lane_markings = crossed_lane_markings
        self.transform = transform if transform is not None else FakeTransform()


# -- builders --------------------------------------------------------------


def collision_measurement(
    frame=1,
    timestamp=0.5,
    other_id=200,
    other_type="vehicle.audi.tt",
    impulse=(3.0, 4.0, 12.0),
    ego_id=10,
    ego_type="vehicle.tesla.model3",
):
    return FakeCollisionMeasurement(
        frame=frame,
        timestamp=timestamp,
        actor=FakeActor(ego_id, ego_type),
        other_actor=FakeActor(other_id, other_type),
        normal_impulse=FakeVector(*impulse),
        transform=FakeTransform(),
    )


def marking(mtype="Solid", color="White", lane_change="NONE", width=0.15):
    return FakeLaneMarking(FakeEnum(mtype), FakeEnum(color), FakeEnum(lane_change), width)


def lane_measurement(frame=1, timestamp=1.0, markings=None, transform=None):
    if markings is None:
        markings = [marking()]
    return FakeLaneInvasionMeasurement(frame, timestamp, markings, transform)


# -- collision: conversion & records --------------------------------------


class CollisionParseTest(unittest.TestCase):
    def test_impulse_magnitude_3_4_12_equals_13(self):
        event = parse_collision(collision_measurement(impulse=(3.0, 4.0, 12.0)))
        self.assertEqual(event.impulse_magnitude, 13.0)
        self.assertEqual(vector_magnitude(3.0, 4.0, 12.0), 13.0)

    def test_collision_conversion_copies_fields(self):
        event = parse_collision(
            collision_measurement(
                frame=7,
                timestamp=1.5,
                other_id=99,
                other_type="walker.pedestrian.0001",
                impulse=(1.0, 2.0, 2.0),
            )
        )
        self.assertEqual(event.stamp.frame, 7)
        self.assertEqual(event.stamp.sim_time, 1.5)
        self.assertEqual(event.self_actor, ActorRef(10, "vehicle.tesla.model3"))
        self.assertEqual(event.other_actor, ActorRef(99, "walker.pedestrian.0001"))
        self.assertEqual(event.normal_impulse.x, 1.0)
        self.assertEqual(event.normal_impulse.z, 2.0)
        self.assertAlmostEqual(event.impulse_magnitude, 3.0)
        self.assertIsInstance(event.transform, TransformState)

    def test_collision_event_is_immutable(self):
        event = parse_collision(collision_measurement())
        with self.assertRaises(FrozenInstanceError):
            event.impulse_magnitude = 0.0
        with self.assertRaises(FrozenInstanceError):
            event.other_actor = ActorRef(1, "x")

    def test_collision_retains_no_carla_objects(self):
        measurement = collision_measurement()
        other_actor = measurement.other_actor
        event = parse_collision(measurement)
        self.assertIsInstance(event.other_actor, ActorRef)
        measurement_ref = weakref.ref(measurement)
        actor_ref = weakref.ref(other_actor)
        del measurement, other_actor
        gc.collect()
        self.assertIsNone(measurement_ref())
        self.assertIsNone(actor_ref())


# -- collision: aggregator dedup / history / gate -------------------------


class CollisionAggregatorTest(unittest.TestCase):
    def test_exact_duplicate_is_suppressed(self):
        agg = TelemetryAggregator()
        measurement = collision_measurement()
        # Parsed twice: same content, different host receive time.
        first = agg.submit_collision(parse_collision(measurement))
        second = agg.submit_collision(parse_collision(measurement))
        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(len(agg.snapshot().collisions), 1)

    def test_distinct_actors_impulses_and_frames_are_preserved(self):
        agg = TelemetryAggregator()
        agg.submit_collision(parse_collision(collision_measurement(frame=1, other_id=200)))
        agg.submit_collision(parse_collision(collision_measurement(frame=1, other_id=201)))
        agg.submit_collision(
            parse_collision(collision_measurement(frame=1, other_id=201, impulse=(5.0, 0.0, 0.0)))
        )
        agg.submit_collision(
            parse_collision(collision_measurement(frame=2, other_id=201, impulse=(5.0, 0.0, 0.0)))
        )
        collisions = agg.snapshot().collisions
        self.assertEqual(len(collisions), 4)  # different actor, impulse, and frame

    def test_history_is_bounded_to_128(self):
        agg = TelemetryAggregator()
        for i in range(200):
            agg.submit_collision(parse_collision(collision_measurement(frame=i)))
        collisions = agg.snapshot().collisions
        self.assertEqual(len(collisions), 128)
        self.assertEqual(collisions[-1].stamp.frame, 199)  # newest kept
        self.assertEqual(collisions[0].stamp.frame, 72)     # oldest dropped

    def test_collision_rejected_after_shutdown(self):
        agg = TelemetryAggregator()
        agg.begin_shutdown()
        accepted = agg.submit_collision(parse_collision(collision_measurement()))
        self.assertFalse(accepted)
        self.assertEqual(len(agg.snapshot().collisions), 0)


# -- lane: conversion, ordering, records ----------------------------------


class LaneParseTest(unittest.TestCase):
    def test_enum_conversion_prefers_name(self):
        event = parse_lane_invasion(
            lane_measurement(markings=[marking("Solid", "White", "NONE", 0.15)])
        )
        info = event.crossed_markings[0]
        self.assertEqual(info.type, "Solid")     # from .name, not str()
        self.assertEqual(info.color, "White")
        self.assertEqual(info.lane_change, "NONE")
        self.assertEqual(info.width, 0.15)

    def test_enum_name_handles_plain_and_prefixed(self):
        self.assertEqual(enum_name("Broken"), "Broken")
        self.assertEqual(enum_name(FakeEnum("Grass")), "Grass")
        self.assertEqual(enum_name(FakePrefixedEnum("carla.LaneMarkingType.Curb")), "Curb")

    def test_markings_sorted_deterministically_preserving_duplicates(self):
        markings = [
            marking("Solid", "Yellow", "NONE", 0.20),
            marking("Broken", "White", "Right", 0.10),
            marking("Broken", "White", "Right", 0.10),  # exact duplicate
        ]
        event = parse_lane_invasion(lane_measurement(markings=markings))
        types = [m.type for m in event.crossed_markings]
        self.assertEqual(types, ["Broken", "Broken", "Solid"])  # sorted, dup kept
        self.assertEqual(event.crossed_markings[0], event.crossed_markings[1])
        # Deterministic: a shuffled input yields the same ordered tuple.
        shuffled = [markings[2], markings[0], markings[1]]
        event2 = parse_lane_invasion(lane_measurement(markings=shuffled))
        self.assertEqual(event.crossed_markings, event2.crossed_markings)

    def test_lane_event_is_immutable(self):
        event = parse_lane_invasion(lane_measurement())
        with self.assertRaises(FrozenInstanceError):
            event.crossed_markings = ()

    def test_lane_retains_no_carla_objects(self):
        measurement = lane_measurement()
        event = parse_lane_invasion(measurement)
        self.assertIsInstance(event.crossed_markings[0], LaneMarkingInfo)
        measurement_ref = weakref.ref(measurement)
        del measurement
        gc.collect()
        self.assertIsNone(measurement_ref())


# -- lane: aggregator debounce / bounds / gate ----------------------------


class LaneAggregatorTest(unittest.TestCase):
    def test_debounce_suppresses_identical_within_window(self):
        agg = TelemetryAggregator()
        sig = [marking("Solid", "White", "NONE", 0.15)]
        self.assertTrue(
            agg.submit_lane_invasion(parse_lane_invasion(lane_measurement(frame=1, timestamp=1.0, markings=sig)))
        )
        # 0.1s later (< 0.25) -> debounced.
        self.assertFalse(
            agg.submit_lane_invasion(parse_lane_invasion(lane_measurement(frame=2, timestamp=1.1, markings=sig)))
        )
        self.assertEqual(len(agg.snapshot().lane_invasions), 1)
        # 0.3s after the accepted one (>= 0.25) -> accepted again.
        self.assertTrue(
            agg.submit_lane_invasion(parse_lane_invasion(lane_measurement(frame=3, timestamp=1.3, markings=sig)))
        )
        self.assertEqual(len(agg.snapshot().lane_invasions), 2)

    def test_distinct_signatures_are_not_debounced(self):
        agg = TelemetryAggregator()
        agg.submit_lane_invasion(
            parse_lane_invasion(lane_measurement(timestamp=1.0, markings=[marking("Solid")]))
        )
        agg.submit_lane_invasion(
            parse_lane_invasion(lane_measurement(timestamp=1.0, markings=[marking("Broken")]))
        )
        self.assertEqual(len(agg.snapshot().lane_invasions), 2)

    def test_debounce_state_cleared_when_time_moves_backwards(self):
        agg = TelemetryAggregator()
        sig = [marking("Solid", "White", "NONE", 0.15)]
        agg.submit_lane_invasion(parse_lane_invasion(lane_measurement(frame=1, timestamp=5.0, markings=sig)))
        # Time jumps back (world reset): the identical signature is accepted.
        agg.submit_lane_invasion(parse_lane_invasion(lane_measurement(frame=2, timestamp=1.0, markings=sig)))
        self.assertEqual(len(agg.snapshot().lane_invasions), 2)
        # Debounce is active again from the new timeline.
        agg.submit_lane_invasion(parse_lane_invasion(lane_measurement(frame=3, timestamp=1.1, markings=sig)))
        self.assertEqual(len(agg.snapshot().lane_invasions), 2)

    def test_debounce_cache_is_bounded_to_32(self):
        agg = TelemetryAggregator()
        for i in range(40):
            markings = [marking("Solid", "White", "NONE", 0.10 + i * 0.001)]  # distinct width
            agg.submit_lane_invasion(
                parse_lane_invasion(lane_measurement(frame=i, timestamp=0.0, markings=markings))
            )
        # All 40 distinct signatures were accepted into the (128) history...
        self.assertEqual(len(agg.snapshot().lane_invasions), 40)
        # ...but the debounce cache never exceeds its bound.
        self.assertEqual(len(agg._lane_debounce), LANE_DEBOUNCE_CACHE)

    def test_history_is_bounded_to_128(self):
        agg = TelemetryAggregator()
        for i in range(200):
            markings = [marking("Solid", "White", "NONE", float(i))]  # distinct signature
            agg.submit_lane_invasion(
                parse_lane_invasion(lane_measurement(frame=i, timestamp=0.0, markings=markings))
            )
        self.assertEqual(len(agg.snapshot().lane_invasions), 128)

    def test_lane_rejected_after_shutdown(self):
        agg = TelemetryAggregator()
        agg.begin_shutdown()
        accepted = agg.submit_lane_invasion(parse_lane_invasion(lane_measurement()))
        self.assertFalse(accepted)
        self.assertEqual(len(agg.snapshot().lane_invasions), 0)


# -- suite integration: isolation, shutdown, cleanup order ----------------


def attach_event_suite(events=None, **world_kwargs):
    config = SensorSuiteConfig(enable_collision=True, enable_lane_invasion=True)
    world = FakeWorld(events=events, **world_kwargs)
    suite = SensorSuite(config=config)
    suite.attach(world, FakeVehicle())
    return world, suite


class EventSuiteTest(unittest.TestCase):
    def test_callback_failure_in_one_event_sensor_isolated(self):
        world, suite = attach_event_suite()
        collision_actor = world.spawned[COLLISION_BLUEPRINT]
        lane_actor = world.spawned[LANE_INVASION_BLUEPRINT]

        collision_actor.emit(None)                 # parse raises -> isolated
        lane_actor.emit(lane_measurement())        # unaffected

        snap = suite.snapshot()
        self.assertEqual(len(snap.collisions), 0)
        self.assertEqual(len(snap.lane_invasions), 1)
        health = {h.name: h for h in snap.sensors}
        self.assertGreaterEqual(health["collision"].error_count, 1)
        self.assertGreaterEqual(health["lane_invasion"].messages_received, 1)
        suite.destroy()

    def test_event_callbacks_after_shutdown_are_ignored(self):
        world, suite = attach_event_suite()
        collision_actor = world.spawned[COLLISION_BLUEPRINT]
        suite.destroy()
        revision_before = suite.snapshot().revision
        collision_actor.emit(collision_measurement())  # late callback
        self.assertEqual(suite.snapshot().revision, revision_before)
        self.assertEqual(len(suite.snapshot().collisions), 0)

    def test_cleanup_stops_all_then_destroys_all_in_reverse_order(self):
        events = []
        _world, suite = attach_event_suite(events=events)
        suite.destroy()
        kinds = [kind for kind, _ in events]
        self.assertEqual(kinds[0], "tick_unregister")
        stop_indices = [i for i, k in enumerate(kinds) if k == "stop"]
        destroy_indices = [i for i, k in enumerate(kinds) if k == "destroy"]
        self.assertLess(max(stop_indices), min(destroy_indices))
        destroyed = [target for kind, target in events if kind == "destroy"]
        # Creation order [gnss, imu, collision, lane_invasion, rgb_camera]
        # (the camera is default-enabled since Mission 4E) -> reversed.
        self.assertEqual(
            destroyed,
            [
                RGB_CAMERA_BLUEPRINT,
                LANE_INVASION_BLUEPRINT,
                COLLISION_BLUEPRINT,
                IMU_BLUEPRINT,
                GNSS_BLUEPRINT,
            ],
        )


if __name__ == "__main__":
    unittest.main()
