"""CARLA-shaped fakes for unit tests. No CARLA package or server required.

These duck-type just enough of the CARLA API surface that the sensor/suite
code exercises: blueprint library, world, sensor actors, world/actor
snapshots, and GNSS/IMU measurements. A shared ``events`` list records
stop/destroy/tick-unregister calls so tests can assert cleanup ordering.
"""

from __future__ import annotations


class FakeVector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class FakeRotation:
    def __init__(self, roll=0.0, pitch=0.0, yaw=0.0):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw


class FakeTransform:
    def __init__(self, location=None, rotation=None):
        self.location = location if location is not None else FakeVector()
        self.rotation = rotation if rotation is not None else FakeRotation()


class FakeTimestamp:
    def __init__(self, elapsed_seconds=0.0, frame=0):
        self.elapsed_seconds = elapsed_seconds
        self.frame = frame


class FakeGnssMeasurement:
    def __init__(self, frame, timestamp, latitude, longitude, altitude, transform=None):
        self.frame = frame
        self.timestamp = timestamp
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.transform = transform if transform is not None else FakeTransform()


class FakeImuMeasurement:
    def __init__(self, frame, timestamp, accelerometer, gyroscope, compass, transform=None):
        self.frame = frame
        self.timestamp = timestamp
        self.accelerometer = accelerometer
        self.gyroscope = gyroscope
        self.compass = compass
        self.transform = transform if transform is not None else FakeTransform()


class FakeBlueprint:
    def __init__(self, bp_id):
        self.id = bp_id
        self.attributes = {}

    def has_attribute(self, name):
        return True

    def set_attribute(self, name, value):
        self.attributes[name] = value


class FakeBlueprintLibrary:
    def __init__(self, missing=(), registry=None):
        self._missing = set(missing)
        self._registry = registry

    def find(self, bp_id):
        if bp_id in self._missing:
            raise IndexError(f"blueprint {bp_id!r} not found")
        blueprint = FakeBlueprint(bp_id)
        if self._registry is not None:
            self._registry[bp_id] = blueprint      # let tests inspect attrs
        return blueprint


class FakeSensorActor:
    def __init__(
        self,
        actor_id,
        blueprint_id,
        events,
        fail_stop=False,
        fail_destroy=False,
        fail_listen=False,
        immediate=None,
        destroy_result=True,
    ):
        self.id = actor_id
        self.blueprint_id = blueprint_id
        self.type_id = blueprint_id
        self.is_alive = True
        self._events = events
        self._fail_stop = fail_stop
        self._fail_destroy = fail_destroy
        self._fail_listen = fail_listen
        self._immediate = immediate
        self._destroy_result = destroy_result
        self._callback = None
        self.stopped = False
        self.destroyed = False

    def listen(self, callback):
        # Records/activates the callback first, then optionally raises: this
        # models a sensor that becomes potentially-listening before listen()
        # fails, so cleanup must still stop() it.
        self._callback = callback
        if self._fail_listen:
            raise RuntimeError(f"listen failed for {self.blueprint_id}")
        if self._immediate is not None:
            # Simulate a measurement arriving during listen() itself.
            callback(self._immediate)

    def emit(self, measurement):
        """Simulate the CARLA server delivering a measurement."""
        if self._callback is not None:
            self._callback(measurement)

    def stop(self):
        self._events.append(("stop", self.blueprint_id))
        self.stopped = True
        if self._fail_stop:
            raise RuntimeError(f"stop failed for {self.blueprint_id}")

    def destroy(self):
        self._events.append(("destroy", self.blueprint_id))
        self.destroyed = True
        if self._fail_destroy:
            raise RuntimeError(f"destroy failed for {self.blueprint_id}")
        if self._destroy_result is True:
            self.is_alive = False
        return self._destroy_result


class FakeVehicle:
    def __init__(self, actor_id=10, type_id="vehicle.tesla.model3", is_alive=True):
        self.id = actor_id
        self.type_id = type_id
        self.is_alive = is_alive


class FakeActorSnapshot:
    def __init__(self, transform, velocity):
        self._transform = transform
        self._velocity = velocity

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return self._velocity


class FakeWorldSnapshot:
    def __init__(self, frame, elapsed_seconds, actors=None):
        self.frame = frame
        self.timestamp = FakeTimestamp(elapsed_seconds, frame)
        self._actors = actors or {}

    def find(self, actor_id):
        return self._actors.get(actor_id)


class FakeWorld:
    def __init__(
        self,
        events=None,
        missing_blueprints=(),
        spawn_fail=(),
        fail_stop=(),
        fail_destroy=(),
        fail_listen=(),
        immediate_emit=None,
        destroy_results=None,
        fail_on_tick=False,
    ):
        self.events = events if events is not None else []
        self._missing = tuple(missing_blueprints)
        self._spawn_fail = set(spawn_fail)
        self._fail_stop = set(fail_stop)
        self._fail_destroy = set(fail_destroy)
        self._fail_listen = set(fail_listen)
        self._immediate_emit = dict(immediate_emit or {})
        self._destroy_results = dict(destroy_results or {})
        self._fail_on_tick = fail_on_tick
        self.spawned = {}
        self.blueprints = {}
        self.spawn_transforms = {}
        self.spawn_calls = []
        self.tick_callbacks = {}
        self._next_actor_id = 1000
        self._next_tick_id = 1

    def get_blueprint_library(self):
        return FakeBlueprintLibrary(missing=self._missing, registry=self.blueprints)

    def spawn_actor(self, blueprint, transform, attach_to=None, attachment_type=None):
        if blueprint.id in self._spawn_fail:
            raise RuntimeError(f"spawn failed for {blueprint.id}")
        self._next_actor_id += 1
        actor = FakeSensorActor(
            self._next_actor_id,
            blueprint.id,
            self.events,
            fail_stop=blueprint.id in self._fail_stop,
            fail_destroy=blueprint.id in self._fail_destroy,
            fail_listen=blueprint.id in self._fail_listen,
            immediate=self._immediate_emit.get(blueprint.id),
            destroy_result=self._destroy_results.get(blueprint.id, True),
        )
        self.spawned[blueprint.id] = actor
        self.spawn_transforms[blueprint.id] = transform
        self.spawn_calls.append(
            (blueprint.id, transform, attach_to, attachment_type, actor)
        )
        return actor

    def on_tick(self, callback):
        if self._fail_on_tick:
            raise RuntimeError("on_tick failed")
        tick_id = self._next_tick_id
        self._next_tick_id += 1
        self.tick_callbacks[tick_id] = callback
        return tick_id

    def remove_on_tick(self, tick_id):
        self.events.append(("tick_unregister", tick_id))
        self.tick_callbacks.pop(tick_id, None)

    def tick(self, world_snapshot):
        for callback in list(self.tick_callbacks.values()):
            callback(world_snapshot)
