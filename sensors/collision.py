"""Collision sensor: sensor.other.collision.

Event-driven (fires on contact, not on a tick), rigidly attached at the ego
origin. Each CARLA collision event is copied into an immutable, CARLA-free
record -- actor identities and the impulse vector are copied out, so no CARLA
actor, event, or transform is retained.
"""

from __future__ import annotations

import time

from sensors.base import BaseSensor
from telemetry.state import (
    ActorRef,
    CollisionEvent,
    SampleStamp,
    Vector3,
    transform_state,
    vector_magnitude,
)

COLLISION_BLUEPRINT = "sensor.other.collision"
COLLISION_NAME = "collision"


def _actor_ref(actor) -> ActorRef:
    """Copy an actor's id and type out; never retain the actor."""
    return ActorRef(id=int(actor.id), type_id=str(actor.type_id))


def parse_collision(measurement, receive_monotonic=None) -> CollisionEvent:
    """Copy a CARLA collision event into an immutable record (no retention)."""
    impulse = measurement.normal_impulse
    return CollisionEvent(
        stamp=SampleStamp(
            frame=int(measurement.frame),
            sim_time=float(measurement.timestamp),
            monotonic=(
                time.monotonic() if receive_monotonic is None else receive_monotonic
            ),
        ),
        self_actor=_actor_ref(measurement.actor),
        other_actor=_actor_ref(measurement.other_actor),
        normal_impulse=Vector3(float(impulse.x), float(impulse.y), float(impulse.z)),
        impulse_magnitude=vector_magnitude(impulse.x, impulse.y, impulse.z),
        transform=transform_state(measurement.transform),
    )


class CollisionSensor(BaseSensor):
    name = COLLISION_NAME
    blueprint_id = COLLISION_BLUEPRINT

    def __init__(self, aggregator, tick: float = 0.0, monotonic_clock=None) -> None:
        # Event sensor: sensor_tick does not apply (collision blueprints have
        # no such attribute); tick is accepted only for a uniform constructor.
        super().__init__(aggregator, tick, monotonic_clock)

    def configure_blueprint(self, blueprint) -> None:
        # Nothing to configure: no noise model, no tick.
        pass

    def parse(self, measurement, receive_monotonic=None) -> CollisionEvent:
        return parse_collision(measurement, receive_monotonic)

    def submit(self, state: CollisionEvent) -> None:
        self._aggregator.submit_collision(state)
