"""SensorSuite: owns the configured sensors and one world.on_tick callback.

The caller owns the world and the ego vehicle; the suite owns only the sensor
actors it creates and its tick registration. Sensors attach independently, so
one sensor's failure never blocks the other. attach() is one-shot; destroy()
is idempotent and tears down in the required order:

    close callback gate -> unregister world tick -> stop listeners -> destroy
    actors (reverse creation order)

Ego kinematics come from the per-frame world snapshot (coherent state), never
derived from GNSS or IMU.
"""

from __future__ import annotations

import time
import threading
import weakref
from dataclasses import dataclass, replace

from sensors.base import SensorActorIdentity
from sensors.collision import CollisionSensor
from sensors.gnss import GnssSensor
from sensors.imu import ImuSensor
from sensors.lane_invasion import LaneInvasionSensor
from sensors.rgb_camera import RgbCameraConfig, RgbCameraSensor
from telemetry.aggregator import TelemetryAggregator
from telemetry.freshness import (
    ContinuousFreshnessPolicy,
    SensorEvidence,
    SensorFreshnessConfig,
    SensorKind,
    evaluate_sensor_status,
    finite_monotonic,
)
from telemetry.state import (
    EgoKinematics,
    SampleStamp,
    TelemetrySnapshot,
    Vector3,
    speed_from_velocity,
    transform_state,
)

TICK_SOURCE = "world_tick"


@dataclass(frozen=True, slots=True)
class SensorSuiteConfig:
    gnss_tick: float = 0.10
    imu_tick: float = 0.05
    enable_gnss: bool = True
    enable_imu: bool = True
    enable_collision: bool = True
    enable_lane_invasion: bool = True
    enable_rgb_camera: bool = True
    rgb_camera: RgbCameraConfig = RgbCameraConfig()
    freshness: SensorFreshnessConfig = SensorFreshnessConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.freshness, SensorFreshnessConfig):
            raise ValueError("freshness must be a SensorFreshnessConfig")


def _canonical_snapshot_status_specs(config, specs, rgb_enabled: bool):
    """Copy caller configuration into exact immutable snapshot evidence.

    Every attribute and policy hook runs during construction, before the suite
    owns any runtime-critical lock.  Snapshot evaluation retains only built-in
    booleans and exact base policy records, never caller subclasses.
    """
    freshness = config.freshness
    canonical = []
    for name, kind, enabled_field in specs:
        enabled = (
            rgb_enabled
            if name == "rgb_camera"
            else bool(getattr(config, enabled_field))
        )
        policy = freshness.policy_for(name)
        if policy is not None:
            if not isinstance(policy, ContinuousFreshnessPolicy):
                raise ValueError(
                    f"sensor {name!r} requires a continuous freshness policy or None"
                )
            policy = ContinuousFreshnessPolicy(
                expected_interval=policy.expected_interval,
                stale_after=policy.stale_after,
                startup_grace=policy.startup_grace,
            )
        canonical.append((name, kind, bool(enabled), policy))
    return tuple(canonical)


@dataclass(frozen=True, slots=True)
class SensorAttachResult:
    name: str
    attached: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AttachReport:
    results: tuple[SensorAttachResult, ...]
    tick_registered: bool = False
    tick_error: str | None = None

    @property
    def any_attached(self) -> bool:
        return any(r.attached for r in self.results)

    @property
    def all_attached(self) -> bool:
        return bool(self.results) and all(r.attached for r in self.results)


@dataclass(frozen=True, slots=True)
class SensorCleanupResult:
    name: str
    stopped: bool
    destroyed: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CleanupReport:
    tick_unregistered: bool
    results: tuple[SensorCleanupResult, ...]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SensorTopologySnapshot:
    """Immutable, CARLA-free identities cached at successful actor creation."""

    identities: tuple[SensorActorIdentity, ...]


def build_ego(stamp: SampleStamp, actor_snapshot) -> EgoKinematics:
    """Build ego kinematics from a per-frame actor snapshot (coherent state)."""
    transform = actor_snapshot.get_transform()
    velocity = actor_snapshot.get_velocity()
    speed_mps, speed_kph = speed_from_velocity(velocity.x, velocity.y, velocity.z)
    return EgoKinematics(
        stamp=stamp,
        transform=transform_state(transform),
        velocity=Vector3(float(velocity.x), float(velocity.y), float(velocity.z)),
        speed_mps=speed_mps,
        speed_kph=speed_kph,
        vehicle_alive=True,
    )


class SensorSuite:
    _STATUS_SPECS = (
        ("gnss", SensorKind.CONTINUOUS, "enable_gnss"),
        ("imu", SensorKind.CONTINUOUS, "enable_imu"),
        ("collision", SensorKind.EVENT, "enable_collision"),
        ("lane_invasion", SensorKind.EVENT, "enable_lane_invasion"),
        ("rgb_camera", SensorKind.CONTINUOUS, "enable_rgb_camera"),
    )

    def __init__(
        self,
        config: SensorSuiteConfig | None = None,
        warning_sink=None,
        monotonic_clock=None,
        camera_snapshot_consumer=None,
    ) -> None:
        selected_config = config if config is not None else SensorSuiteConfig()
        rgb_enabled = bool(selected_config.enable_rgb_camera)
        if not rgb_enabled and camera_snapshot_consumer is not None:
            raise ValueError("camera snapshot consumer requires RGB camera")
        snapshot_status_specs = _canonical_snapshot_status_specs(
            selected_config,
            self._STATUS_SPECS,
            rgb_enabled,
        )
        enabled = {
            name: sensor_enabled
            for name, _kind, sensor_enabled, _policy in snapshot_status_specs
        }
        self._config = selected_config
        self._snapshot_status_specs = snapshot_status_specs
        self._warning_sink = warning_sink
        self._monotonic_clock = monotonic_clock or time.monotonic
        if not callable(self._monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        self._aggregator = TelemetryAggregator()
        self._snapshot_lock = threading.Lock()
        self._snapshot_observation_generation = 0
        self._snapshot_comparison_floor = 0
        self._last_effective_monotonic: float | None = None
        self._camera_snapshot_consumer_lock = threading.Lock()
        self._camera_snapshot_consumer = None
        self._camera_sensor: RgbCameraSensor | None = None

        # Creation order matters: cleanup destroys in reverse.
        self._sensors: list = []
        if enabled["gnss"]:
            self._sensors.append(
                GnssSensor(
                    self._aggregator,
                    self._config.gnss_tick,
                    monotonic_clock=self._monotonic_clock,
                )
            )
        if enabled["imu"]:
            self._sensors.append(
                ImuSensor(
                    self._aggregator,
                    self._config.imu_tick,
                    monotonic_clock=self._monotonic_clock,
                )
            )
        if enabled["collision"]:
            self._sensors.append(
                CollisionSensor(
                    self._aggregator, monotonic_clock=self._monotonic_clock
                )
            )
        if enabled["lane_invasion"]:
            self._sensors.append(
                LaneInvasionSensor(
                    self._aggregator, monotonic_clock=self._monotonic_clock
                )
            )
        if enabled["rgb_camera"]:
            camera_consumer = None
            if camera_snapshot_consumer is not None:
                self._camera_snapshot_consumer = _resolve_snapshot_consumer(
                    camera_snapshot_consumer
                )
                weak_self = weakref.ref(self)

                def dispatch_camera_snapshot(snapshot, weak_self=weak_self):
                    return SensorSuite._dispatch_camera_snapshot(
                        weak_self, snapshot
                    )

                camera_consumer = dispatch_camera_snapshot
            self._camera_sensor = RgbCameraSensor(
                self._aggregator,
                self._config.rgb_camera,
                monotonic_clock=self._monotonic_clock,
                snapshot_consumer=camera_consumer,
            )
            self._sensors.append(self._camera_sensor)

        self._world = None
        self._vehicle_id: int | None = None
        self._tick_id = None
        self._tick_registered = False
        self._tick_unregister_attempted = False
        self._tick_unregister_succeeded = False
        self._tick_unregister_error: BaseException | None = None
        self._attached = False
        self._destroyed = False
        self._shutdown_started = False
        self._cleanup_report: CleanupReport | None = None
        self._cleanup_report_delivered = False
        self._cleanup_status = {
            sensor.name: {
                "stopped": False,
                "destroyed": False,
                "errors": [],
                "seen_errors": set(),
            }
            for sensor in self._sensors
        }

    # -- public API --------------------------------------------------------

    def attach(self, world, vehicle, pre_operation_guard=None) -> AttachReport:
        if self._destroyed:
            raise RuntimeError("SensorSuite has been destroyed")
        if self._attached:
            raise RuntimeError("SensorSuite.attach() is one-shot")

        if pre_operation_guard is not None and not callable(pre_operation_guard):
            raise TypeError("pre_operation_guard must be callable or None")
        vehicle_id = self._validate_vehicle(vehicle)
        self._world = world
        self._vehicle_id = vehicle_id
        # One-shot even if a BaseException interrupts a later attachment step.
        self._attached = True

        # Health exists for every configured sensor before any attach/listen,
        # so a callback firing during listen() always finds its slot.
        for sensor in self._sensors:
            self._aggregator.register(sensor.name)

        results = []
        for sensor in self._sensors:
            try:
                sensor.attach(
                    world,
                    vehicle,
                    pre_operation_guard=pre_operation_guard,
                )
            except Exception as exc:                 # noqa: BLE001 - per-sensor
                attachment_monotonic = self._read_monotonic()
                self._aggregator.mark_attached(
                    sensor.name, False, attachment_monotonic
                )
                self._warn(f"sensor {sensor.name!r} failed to attach: {exc!r}")
                results.append(SensorAttachResult(sensor.name, False, repr(exc)))
            else:
                attachment_monotonic = self._read_monotonic()
                self._aggregator.mark_attached(
                    sensor.name, True, attachment_monotonic
                )
                results.append(SensorAttachResult(sensor.name, True, None))

        weak_self = weakref.ref(self)
        try:
            if pre_operation_guard is not None:
                pre_operation_guard("world_tick_registration")
            tick_id = world.on_tick(
                lambda snapshot: SensorSuite._on_tick(weak_self, snapshot)
            )
            if type(tick_id) is not int or tick_id < 0:
                raise RuntimeError(
                    "world.on_tick() did not return a built-in non-negative integer"
                )
            self._tick_id = tick_id
            self._tick_registered = True
        except Exception as exc:                     # noqa: BLE001
            self._tick_id = None
            self._tick_registered = False
            self._warn(f"failed to register world tick: {exc!r}")
            tick_error = repr(exc)
        else:
            tick_error = None

        return AttachReport(
            tuple(results),
            tick_registered=self._tick_registered,
            tick_error=tick_error,
        )

    def snapshot(self) -> TelemetrySnapshot:
        raw = self._aggregator.raw_snapshot()
        return self._snapshot_with_statuses(raw)

    def _snapshot_with_statuses(
        self, raw: TelemetrySnapshot
    ) -> TelemetrySnapshot:
        """Decorate exact raw evidence through one ordered freshness timeline.

        A fixed-size generation reservation defines logical call-start order.
        The caller-controlled clock then runs with no suite lock held.  The
        comparison floor published by each commit separates generations that
        were already reserved at that point from calls provably begun later.
        An already-reserved call is an overlap regardless of which generation
        commits first; it may advance effective time but cannot prove rollback.
        A generation at or above the floor still detects and clamps a genuine
        sequential regression.  Failed or reentrant clock calls retain no
        reservation object and strand no waiter: skipped scalar generations
        are deliberately harmless.
        """
        # Reservation touches only trusted scalar state and completes before
        # the hostile injected clock is invoked.  No queue or sample history is
        # retained, and reverse completion needs no waiting or handoff.
        with self._snapshot_lock:
            observation_generation = self._snapshot_observation_generation
            self._snapshot_observation_generation += 1

        observed = self._read_monotonic()
        with self._snapshot_lock:
            previous = self._last_effective_monotonic
            provably_nonoverlapping = (
                observation_generation >= self._snapshot_comparison_floor
            )
            if previous is None:
                rollback = False
                effective = observed
            elif provably_nonoverlapping:
                rollback = observed < previous
                effective = previous if rollback else observed
            else:
                rollback = False
                effective = observed if observed > previous else previous
            self._last_effective_monotonic = effective
            # Every smaller generation was reserved before this commit.  It
            # therefore overlaps the state just published even if it returns
            # later with a smaller clock sample.
            self._snapshot_comparison_floor = self._snapshot_observation_generation
            statuses = self._evaluate_statuses(raw, effective, rollback)
        return replace(
            raw,
            captured_monotonic=effective,
            statuses=statuses,
        )

    def topology_snapshot(self) -> SensorTopologySnapshot:
        """Return cached identities in configured creation order, with no RPCs."""
        return SensorTopologySnapshot(
            tuple(
                identity
                for sensor in self._sensors
                if (identity := sensor.identity_snapshot()) is not None
            )
        )

    def destroy(self) -> CleanupReport:
        if self._cleanup_report is not None:
            if not self._cleanup_report_delivered:
                self._cleanup_report_delivered = True
                return self._cleanup_report
            return CleanupReport(tick_unregistered=False, results=())

        pending: tuple[BaseException, object] | None = None

        def defer(exc: BaseException) -> None:
            nonlocal pending
            if pending is None and not isinstance(exc, Exception):
                pending = (exc, exc.__traceback__)

        if not self._shutdown_started:
            self._shutdown_started = True
            self._aggregator.begin_shutdown()
            if self._camera_sensor is not None:
                self._camera_sensor.release_snapshot_consumer()
            self._release_camera_snapshot_consumer()

        try:
            self._unregister_tick()
        except BaseException as exc:
            self._record_cleanup_error(None, "tick unregister", exc)
            defer(exc)

        # Stop every potentially-listening sensor before any destroy call.
        for sensor in reversed(self._sensors):
            try:
                sensor.stop()
            except BaseException as exc:
                self._record_cleanup_error(sensor, "stop", exc)
                defer(exc)
            self._sync_sensor_cleanup_status(sensor)

        for sensor in reversed(self._sensors):
            try:
                sensor.destroy()
            except BaseException as exc:
                self._record_cleanup_error(sensor, "destroy", exc)
                defer(exc)
            self._sync_sensor_cleanup_status(sensor)

        self._destroyed = True
        results = tuple(
            SensorCleanupResult(
                name=sensor.name,
                stopped=self._cleanup_status[sensor.name]["stopped"],
                destroyed=self._cleanup_status[sensor.name]["destroyed"],
                errors=tuple(self._cleanup_status[sensor.name]["errors"]),
            )
            for sensor in self._sensors
        )
        suite_errors = tuple(
            self._cleanup_status["__suite__"]["errors"]
            if "__suite__" in self._cleanup_status
            else ()
        )
        self._cleanup_report = CleanupReport(
            tick_unregistered=self._tick_unregister_succeeded,
            results=results,
            errors=suite_errors,
        )
        if pending is not None:
            exc, traceback = pending
            raise exc.with_traceback(traceback)
        self._cleanup_report_delivered = True
        return self._cleanup_report

    # -- internals ---------------------------------------------------------

    def _unregister_tick(self) -> bool:
        if self._tick_unregister_attempted:
            return self._tick_unregister_succeeded
        if not self._tick_registered or self._world is None or self._tick_id is None:
            self._tick_unregister_attempted = True
            return False
        self._tick_unregister_attempted = True
        world = self._world
        tick_id = self._tick_id
        try:
            world.remove_on_tick(tick_id)
            self._tick_unregister_succeeded = True
            return True
        except BaseException as exc:
            self._tick_unregister_error = exc
            raise
        finally:
            # The outcome has now been recorded, so retaining the handle is
            # no longer needed and the call must never be submitted again.
            self._tick_id = None

    def _validate_vehicle(self, vehicle) -> int:
        actor_id = vehicle.id
        if type(actor_id) is not int or actor_id <= 0:
            raise ValueError("ego actor id must be a positive built-in integer")
        type_id = vehicle.type_id
        if type(type_id) is not str or not type_id.startswith("vehicle."):
            raise ValueError(f"actor {type_id!r} is not a vehicle")
        return actor_id

    def _record_cleanup_error(self, sensor, operation: str, exc: BaseException) -> None:
        name = "__suite__" if sensor is None else sensor.name
        if name not in self._cleanup_status:
            self._cleanup_status[name] = {
                "stopped": False,
                "destroyed": False,
                "errors": [],
                "seen_errors": set(),
            }
        marker = (operation, id(exc))
        status = self._cleanup_status[name]
        if marker not in status["seen_errors"]:
            status["seen_errors"].add(marker)
            status["errors"].append(
                f"{operation}: {type(exc).__name__}: {self._safe_error_text(exc)}"
            )
        try:
            self._warn(f"{operation} failed for {name!r}: {exc!r}")
        except BaseException:
            # Diagnostics must never prevent remaining owned cleanup.
            pass

    def _sync_sensor_cleanup_status(self, sensor) -> None:
        status = self._cleanup_status[sensor.name]
        status["stopped"] = sensor._stop_succeeded
        status["destroyed"] = sensor._destroy_succeeded
        if sensor._stop_error is not None:
            self._record_cleanup_error(sensor, "stop", sensor._stop_error)
        if sensor._destroy_error is not None:
            self._record_cleanup_error(sensor, "destroy", sensor._destroy_error)

    @staticmethod
    def _safe_error_text(exc: BaseException) -> str:
        try:
            text = str(exc)
        except BaseException:
            text = "<unprintable>"
        text = " ".join(text.split()) or "<no message>"
        return text if len(text) <= 192 else text[:189] + "..."

    def _warn(self, message: str) -> None:
        if self._warning_sink is None:
            return
        try:
            self._warning_sink(message)
        except Exception:                            # noqa: BLE001 - isolate sink
            pass

    @staticmethod
    def _on_tick(weak_self, world_snapshot) -> None:
        self = weak_self()
        if self is None:
            return
        self._handle_tick(world_snapshot)

    @staticmethod
    def _dispatch_camera_snapshot(
        weak_self, raw: TelemetrySnapshot
    ) -> bool:
        """Weak callback adapter; retains neither suite nor external consumer."""
        self = weak_self()
        if self is None:
            return False
        return self._consume_camera_snapshot(raw)

    def _consume_camera_snapshot(self, raw: TelemetrySnapshot) -> bool:
        with self._camera_snapshot_consumer_lock:
            if self._camera_snapshot_consumer is None:
                return False
        snapshot = self._snapshot_with_statuses(raw)
        # A reentrant clock or concurrent teardown may have closed the seam
        # while statuses were being projected.  Recheck without retaining the
        # external object across that caller-controlled clock invocation.
        with self._camera_snapshot_consumer_lock:
            consumer = self._camera_snapshot_consumer
        if consumer is None:
            return False
        # External bridge/hook invocation is deliberately after every suite
        # lock has been released.
        return consumer(snapshot)

    def _release_camera_snapshot_consumer(self) -> None:
        consumer = None
        with self._camera_snapshot_consumer_lock:
            consumer = self._camera_snapshot_consumer
            self._camera_snapshot_consumer = None
        # Delay the last possible decref until after lock release so hostile
        # finalizers and weakref callbacks cannot execute under our lock.
        del consumer

    def _handle_tick(self, world_snapshot) -> None:
        try:
            stamp = SampleStamp(
                frame=int(world_snapshot.frame),
                sim_time=float(world_snapshot.timestamp.elapsed_seconds),
                monotonic=self._read_monotonic(),
            )
            actor_snapshot = world_snapshot.find(self._vehicle_id)
            if actor_snapshot is None:
                self._aggregator.mark_vehicle_missing(stamp)
                return
            ego = build_ego(stamp, actor_snapshot)
        except Exception as exc:                     # noqa: BLE001
            self._aggregator.record_error(TICK_SOURCE, repr(exc))
            return
        self._aggregator.update_ego(ego)

    def _read_monotonic(self) -> float:
        return finite_monotonic(self._monotonic_clock(), "monotonic clock result")

    def _evaluate_statuses(
        self, raw: TelemetrySnapshot, evaluated: float, rollback: bool
    ):
        health_by_name = {health.name: health for health in raw.sensors}
        statuses = []
        for name, kind, enabled, policy in self._snapshot_status_specs:
            health = health_by_name.get(name)
            evidence = SensorEvidence(
                name=name,
                kind=kind,
                enabled=enabled,
                attach_attempted=False if health is None else health.attach_attempted,
                attached=False if health is None else health.attached,
                attachment_failed=False if health is None else health.attachment_failed,
                destroyed=self._destroyed,
                attachment_monotonic=(
                    None if health is None else health.attachment_monotonic
                ),
                accepted_count=0 if health is None else health.messages_received,
                last_receive_monotonic=(
                    None if health is None else health.last_receive_monotonic
                ),
                active_error=False if health is None else health.active_error,
                error_count=0 if health is None else health.error_count,
                active_error_reason=(
                    None if health is None else health.active_error_reason
                ),
                event_count=(
                    (0 if health is None else health.messages_received)
                    if kind is SensorKind.EVENT
                    else None
                ),
            )
            statuses.append(
                evaluate_sensor_status(
                    evidence,
                    policy,
                    evaluated,
                    clock_rollback=rollback,
                )
            )
        return tuple(statuses)


def _resolve_snapshot_consumer(consumer):
    """Normalize a callable or bridge-shaped consumer outside suite locks."""
    try:
        submit = consumer.submit
    except AttributeError:
        if callable(consumer):
            return consumer
        raise TypeError(
            "camera_snapshot_consumer must be callable or expose callable submit()"
        ) from None
    if not callable(submit):
        raise TypeError(
            "camera_snapshot_consumer must be callable or expose callable submit()"
        )
    return submit
