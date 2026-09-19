"""Base lifecycle for a single CARLA sensor actor.

Each concrete sensor owns exactly one CARLA sensor actor and drives it
through attach -> listen -> stop -> destroy. The listen callback holds only a
*weak* reference back to this wrapper, so a live C++ sensor never keeps the
Python wrapper alive (no strong callback cycle). Callback failures are caught
(Exception, not BaseException) and recorded in the aggregator so that one
sensor's bad frame cannot disturb another sensor.

The wrapper only ever destroys its own sensor actor -- never the ego vehicle.
"""

from __future__ import annotations

import time
import weakref
from dataclasses import dataclass


IDENTITY_ERROR_LIMIT = 192


def _identity_error(field: str, value) -> str:
    """Return bounded primitive-only identity failure evidence."""
    if isinstance(value, BaseException):
        try:
            detail = str(value)
        except BaseException:
            detail = "<unprintable>"
        detail = " ".join(detail.split())
        text = f"{field} unavailable: {type(value).__name__}: {detail}"
    else:
        text = f"{field} must be an exact built-in value, got {type(value).__name__}"
    if len(text) <= IDENTITY_ERROR_LIMIT:
        return text
    return text[: IDENTITY_ERROR_LIMIT - 3] + "..."


@dataclass(frozen=True, slots=True)
class SensorActorIdentity:
    """Copied identity evidence for one successfully created sensor actor."""

    name: str
    actor_id: int | None
    type_id: str | None
    identity_errors: tuple[str, ...] = ()


def set_blueprint_attribute(blueprint, name: str, value) -> bool:
    """Set an attribute if the blueprint supports it. Returns whether it did."""
    if blueprint.has_attribute(name):
        blueprint.set_attribute(name, str(value))
        return True
    return False


class BaseSensor:
    # Subclasses set these.
    name: str = ""
    blueprint_id: str = ""

    def __init__(self, aggregator, tick: float, monotonic_clock=None) -> None:
        self._aggregator = aggregator
        self.tick = tick
        self._monotonic_clock = monotonic_clock or time.monotonic
        if not callable(self._monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        self._actor = None
        self._listening = False
        self._destroyed = False
        self._identity: SensorActorIdentity | None = None
        self._actor_created = False
        self._stop_attempted = False
        self._stop_succeeded = False
        self._stop_error: BaseException | None = None
        self._destroy_attempted = False
        self._destroy_succeeded = False
        self._destroy_error: BaseException | None = None
        self._destroy_result = None

    # -- to be provided by concrete sensors --------------------------------

    def configure_blueprint(self, blueprint) -> None:
        """Set sensor-specific (noise) attributes. sensor_tick is set here."""
        raise NotImplementedError

    def parse(self, measurement, receive_monotonic=None):
        """Convert a CARLA measurement into an immutable telemetry record."""
        raise NotImplementedError

    def submit(self, state) -> None:
        """Hand a parsed record to the aggregator."""
        raise NotImplementedError

    def make_spawn_transform(self, carla_module):
        """Relative transform at which the sensor attaches to its parent.

        Default: the vehicle origin (identity transform). Sensors with a
        physical mount offset (e.g. the RGB camera) override this.
        """
        return carla_module.Transform()

    # -- lifecycle ---------------------------------------------------------

    def attach(self, world, vehicle, pre_operation_guard=None):
        if self._destroyed:
            raise RuntimeError(f"{self.name} sensor has already been destroyed")
        if self._actor is not None:
            raise RuntimeError(f"{self.name} sensor is already attached")

        guard = pre_operation_guard
        if guard is not None and not callable(guard):
            raise TypeError("pre_operation_guard must be callable or None")

        # Imported lazily: only the real attach path needs the CARLA package,
        # so pure parsing/aggregator unit tests never import carla at all.
        self._check_operation(guard, f"{self.name}.carla_import")
        import carla

        self._check_operation(guard, f"{self.name}.blueprint_library")
        blueprint_library = world.get_blueprint_library()
        self._check_operation(guard, f"{self.name}.blueprint_lookup")
        blueprint = blueprint_library.find(self.blueprint_id)
        self._check_operation(guard, f"{self.name}.blueprint_configuration")
        set_blueprint_attribute(blueprint, "sensor_tick", self.tick)
        self.configure_blueprint(blueprint)

        self._check_operation(guard, f"{self.name}.spawn")
        actor = world.spawn_actor(
            blueprint,
            self.make_spawn_transform(carla),
            attach_to=vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )
        self._actor = actor
        self._actor_created = True
        # ``id`` and ``type_id`` are local proxy attributes, not new RPCs.
        # Cache them immediately so even expiry/interrupt before listen keeps
        # public ownership evidence for the actor that was just created.
        self._capture_identity(actor)

        weak_self = weakref.ref(self)
        # Mark as listening *before* listen(): a sensor can register/activate
        # its callback and then have listen() raise, leaving the actor
        # potentially delivering data. Cleanup must still stop() it. Only a
        # sensor whose listen() was never invoked (e.g. spawn failed) stays
        # un-listening, so its cleanup skips stop().
        self._check_operation(guard, f"{self.name}.listen")
        self._listening = True
        actor.listen(lambda data: BaseSensor._dispatch(weak_self, data))
        return actor

    @staticmethod
    def _check_operation(guard, phase: str) -> None:
        if guard is not None:
            guard(phase)

    def _capture_identity(self, actor) -> None:
        errors = []
        actor_id = None
        type_id = None
        try:
            raw_id = actor.id
        except BaseException as exc:
            errors.append(_identity_error("actor id", exc))
            self._identity = SensorActorIdentity(
                self.name, None, None, tuple(errors)
            )
            if not isinstance(exc, Exception):
                raise
        else:
            if type(raw_id) is int and raw_id > 0:
                actor_id = raw_id
            else:
                errors.append(_identity_error("actor id", raw_id))

        try:
            raw_type_id = actor.type_id
        except BaseException as exc:
            errors.append(_identity_error("type_id", exc))
            self._identity = SensorActorIdentity(
                self.name, actor_id, None, tuple(errors)
            )
            if not isinstance(exc, Exception):
                raise
        else:
            if type(raw_type_id) is str:
                type_id = raw_type_id
            else:
                errors.append(_identity_error("type_id", raw_type_id))

        self._identity = SensorActorIdentity(
            self.name, actor_id, type_id, tuple(errors)
        )

    def identity_snapshot(self) -> SensorActorIdentity | None:
        """Return cached primitive identity evidence without touching CARLA."""
        return self._identity

    @staticmethod
    def _dispatch(weak_self, measurement) -> None:
        self = weak_self()
        if self is None:
            return
        if not self._aggregator.accepting:
            return
        receive_monotonic = None
        try:
            receive_monotonic = self._monotonic_clock()
            state = self.parse(
                measurement, receive_monotonic=receive_monotonic
            )                                       # parse OUTSIDE any lock
        except Exception as exc:                     # noqa: BLE001 - isolate
            self._aggregator.record_error(
                self.name, repr(exc), receive_monotonic, "callback error"
            )
            return
        try:
            self.submit(state)
        except Exception as exc:                     # noqa: BLE001 - isolate
            self._aggregator.record_error(
                self.name, repr(exc), receive_monotonic, "callback error"
            )

    def stop(self) -> bool:
        """Submit listener stop at most once and retain its recorded outcome."""
        if self._stop_attempted:
            return self._stop_succeeded
        if not self._listening or self._actor is None:
            self._stop_attempted = True
            return False

        actor = self._actor
        self._stop_attempted = True
        try:
            actor.stop()
        except BaseException as exc:
            self._stop_error = exc
            raise
        else:
            self._listening = False
            self._stop_succeeded = True
            return True

    def destroy(self) -> bool:
        """Submit actor destruction at most once; literal ``True`` is success."""
        if self._destroy_attempted:
            return False
        actor = self._actor
        if actor is None:
            self._destroy_attempted = True
            self._destroyed = True
            return False

        self._destroy_attempted = True
        try:
            result = actor.destroy()
            self._destroy_result = result
            if result is not True:
                error = RuntimeError(
                    f"destroy() returned {result!r}; expected literal True"
                )
                self._destroy_error = error
                raise error
        except BaseException as exc:
            if self._destroy_error is None:
                self._destroy_error = exc
            self._destroyed = True
            self._actor = None
            raise
        else:
            self._destroy_succeeded = True
            self._destroyed = True
            self._actor = None
            return True
