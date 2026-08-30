"""Missions 3 + 4F: manual keyboard driving with live sensor telemetry.

Spawns a single ego vehicle (preferring vehicle.tesla.model3, with a safe
fallback), attaches the Mission 4 SensorSuite to it, and lets you drive with
the keyboard while a small pygame window shows live control state, compact
sensor readiness, and short-lived collision / lane-crossing warnings.

Telemetry is sampled from the suite's immutable snapshots at most 10 times
per second (host-monotonic gated); the 60 FPS control loop, steering ramp,
and every key mapping are unchanged from Mission 3. All exits -- Esc, window
close, Ctrl+C, CARLA disconnect, unexpected errors -- funnel through one
cleanup path: sensor suite first, then the ego vehicle, then pygame.

Controls:
    W / Up        throttle
    S / Down      brake
    A / Left      steer left
    D / Right     steer right
    Space         handbrake
    R             toggle reverse
    Esc / close   exit
"""

import math
import sys
import time
from dataclasses import dataclass

import carla
import pygame

from sensors.suite import SensorSuite
from telemetry.freshness import (
    OperationalHealth,
    SensorFreshness,
    SensorKind,
    SensorLifecycle,
    SensorReadiness,
    SensorStatus,
)
from spawn_ego_vehicle import (
    CARLA_LAUNCH_PATH,
    HOST,
    PORT,
    TIMEOUT_SECONDS,
    move_spectator_behind,
    pick_vehicle_blueprint,
    spawn_at_free_spawn_point,
)

FPS = 60
THROTTLE_LEVEL = 0.7
BRAKE_LEVEL = 1.0
MAX_STEER = 0.7
STEER_RATE = 1.2         # steer units per second while a steering key is held
STEER_RETURN_RATE = 2.0  # steer units per second back toward center when released

SNAPSHOT_INTERVAL_SECONDS = 0.10   # sample suite telemetry at most 10 Hz
WARNING_SECONDS = 2.0              # HUD event warnings live this long (host time)

WINDOW_SIZE = (460, 400)
WINDOW_TITLE = "CARLA Manual Drive"
HUD_TEXT_COLOR = (230, 230, 230)
WARNING_COLOR = (235, 190, 90)
HUD_INSTRUCTIONS = [
    "W/Up: throttle    S/Down: brake",
    "A/Left, D/Right: steer",
    "Space: handbrake  R: toggle reverse",
    "Esc or close window: quit",
]


def speed_kmh(vehicle) -> float:
    velocity = vehicle.get_velocity()
    return 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def update_steer(steer: float, steer_input: int, dt: float) -> float:
    """Ramp steering gradually; return it toward center when no key is held."""
    if steer_input != 0:
        steer += STEER_RATE * dt * steer_input
    elif steer > 0.0:
        steer = max(0.0, steer - STEER_RETURN_RATE * dt)
    else:
        steer = min(0.0, steer + STEER_RETURN_RATE * dt)
    return max(-MAX_STEER, min(MAX_STEER, steer))


# -- Mission 4F: pure telemetry/HUD helpers (unit-tested without CARLA) -----


@dataclass(frozen=True, slots=True)
class SensorHud:
    """Compact evaluated statuses only; never telemetry payloads."""

    gnss: SensorStatus | None = None
    imu: SensorStatus | None = None
    camera: SensorStatus | None = None
    collision: SensorStatus | None = None
    lane_invasion: SensorStatus | None = None

    @staticmethod
    def _ready(status: SensorStatus | None) -> bool:
        return bool(
            status is not None
            and status.lifecycle is SensorLifecycle.ATTACHED
            and status.readiness is SensorReadiness.READY
            and status.freshness is SensorFreshness.FRESH
            and status.operational_health is OperationalHealth.HEALTHY
        )

    @staticmethod
    def _armed(status: SensorStatus | None) -> bool:
        return bool(
            status is not None
            and status.lifecycle is SensorLifecycle.ATTACHED
            and status.operational_health is not OperationalHealth.FAILED
        )

    @property
    def gnss_ready(self) -> bool:
        return self._ready(self.gnss)

    @property
    def imu_ready(self) -> bool:
        return self._ready(self.imu)

    @property
    def camera_ready(self) -> bool:
        return self._ready(self.camera)

    @property
    def collision_armed(self) -> bool:
        return self._armed(self.collision)

    @property
    def lane_armed(self) -> bool:
        return self._armed(self.lane_invasion)


def attachment_summary(report) -> str:
    """One concise line from an AttachReport, printed exactly once."""
    total = len(report.results)
    attached = sum(1 for result in report.results if result.attached)
    summary = f"Sensors: {attached}/{total} attached"
    unavailable = [result.name for result in report.results if not result.attached]
    if unavailable:
        summary += "; unavailable: " + ", ".join(unavailable)
    return summary


def attached_sensor_names(report) -> frozenset[str]:
    """Compact, immutable attachment truth derived from AttachReport."""
    return frozenset(result.name for result in report.results if result.attached)


def extract_hud_state(snapshot, attached_names) -> SensorHud:
    """Project already-evaluated statuses without recalculating freshness."""
    del attached_names  # retained for call compatibility; suite status is authoritative
    by_name = {status.name: status for status in snapshot.statuses}
    return SensorHud(
        gnss=by_name.get("gnss"),
        imu=by_name.get("imu"),
        camera=by_name.get("rgb_camera"),
        collision=by_name.get("collision"),
        lane_invasion=by_name.get("lane_invasion"),
    )


def sensor_hud_lines(hud: SensorHud) -> list[str]:
    return [
        f"GNSS: {_sensor_status_label(hud.gnss, SensorKind.CONTINUOUS)}  "
        f"IMU: {_sensor_status_label(hud.imu, SensorKind.CONTINUOUS)}  "
        f"CAMERA: {_sensor_status_label(hud.camera, SensorKind.CONTINUOUS)}",
        f"COLLISION: {_sensor_status_label(hud.collision, SensorKind.EVENT)}  "
        f"LANE SENSOR: {_sensor_status_label(hud.lane_invasion, SensorKind.EVENT)}",
    ]


def _bounded_age(age: float | None) -> str:
    if age is None or not math.isfinite(age) or age > 999.9:
        return ">999.9s"
    return f"{max(0.0, age):.1f}s"


def _sensor_status_label(
    status: SensorStatus | None, expected_kind: SensorKind
) -> str:
    if status is None or status.lifecycle is SensorLifecycle.NOT_AVAILABLE:
        return "UNAVAILABLE"
    if status.lifecycle is SensorLifecycle.DESTROYED:
        return "DESTROYED"
    if (
        status.lifecycle is SensorLifecycle.ATTACHMENT_FAILED
        or status.operational_health is OperationalHealth.FAILED
    ):
        return "FAILED"
    if status.lifecycle is SensorLifecycle.NOT_ATTACHED:
        return "WAITING" if expected_kind is SensorKind.CONTINUOUS else "UNAVAILABLE"
    if status.operational_health is OperationalHealth.DEGRADED:
        return "DEGRADED"
    if expected_kind is SensorKind.EVENT:
        count = status.event_count
        if isinstance(count, bool) or type(count) is not int or count < 0:
            count = 0
        return f"HEALTHY({min(count, 999999999)})"
    if status.freshness is SensorFreshness.STALE:
        return f"STALE {_bounded_age(status.age_seconds)}"
    if (
        status.freshness is SensorFreshness.WAITING_FOR_FIRST_SAMPLE
        or status.readiness is SensorReadiness.WAITING_FOR_FIRST_SAMPLE
    ):
        return "WAITING"
    if status.freshness is SensorFreshness.FRESH:
        return "FRESH"
    return "WAITING"


def format_collision_warning(event, new_count: int) -> str:
    text = f"Collision: {event.other_actor.type_id}"
    if new_count > 1:
        text += f" ({new_count} new)"
    return text


def format_lane_warning(event, new_count: int) -> str:
    names = ", ".join(marking.type for marking in event.crossed_markings)
    text = f"Lane crossed: {names}"
    if new_count > 1:
        text += f" ({new_count} new)"
    return text


class IntervalGate:
    """Host-monotonic rate limiter: at most one due() per interval.

    The deadline advances along a fixed grid (skipping missed intervals in
    one step), so late frames never trigger catch-up bursts and the sampling
    cadence does not drift over time.
    """

    def __init__(self, interval: float) -> None:
        self._interval = float(interval)
        self._deadline: float | None = None

    def due(self, now: float) -> bool:
        if self._deadline is None:
            self._deadline = now + self._interval
            return True
        if now < self._deadline:
            return False
        missed = int((now - self._deadline) // self._interval)
        self._deadline += (missed + 1) * self._interval
        return True


class EventWarnings:
    """Sequence-driven, time-limited HUD warnings (bounded: one per kind).

    Tracks the accepted-event counters (collision_sequence and
    lane_invasion_sequence) independently; a higher sequence posts one compact
    text for the *latest* event in the bounded history, shown for
    WARNING_SECONDS of host-monotonic time. Only the text is retained --
    never the event or the snapshot -- and expiry ignores simulation time.
    """

    def __init__(self) -> None:
        self._last_collision_sequence = 0
        self._last_lane_sequence = 0
        self._collision_warning: tuple[str, float] | None = None
        self._lane_warning: tuple[str, float] | None = None

    def observe(self, snapshot, now: float) -> None:
        if snapshot.collision_sequence > self._last_collision_sequence:
            new_count = snapshot.collision_sequence - self._last_collision_sequence
            self._last_collision_sequence = snapshot.collision_sequence
            if snapshot.collisions:
                self._collision_warning = (
                    format_collision_warning(snapshot.collisions[-1], new_count),
                    now + WARNING_SECONDS,
                )
        if snapshot.lane_invasion_sequence > self._last_lane_sequence:
            new_count = snapshot.lane_invasion_sequence - self._last_lane_sequence
            self._last_lane_sequence = snapshot.lane_invasion_sequence
            if snapshot.lane_invasions:
                self._lane_warning = (
                    format_lane_warning(snapshot.lane_invasions[-1], new_count),
                    now + WARNING_SECONDS,
                )

    def active(self, now: float) -> list[str]:
        texts = []
        for slot in ("_collision_warning", "_lane_warning"):
            entry = getattr(self, slot)
            if entry is None:
                continue
            text, expires_at = entry
            if now >= expires_at:
                setattr(self, slot, None)       # release the text promptly
            else:
                texts.append(text)
        return texts


def sample_telemetry(
    suite, event_warnings: EventWarnings, now: float, attached_names
) -> SensorHud:
    """Take one snapshot, distill HUD state, release the snapshot."""
    snapshot = suite.snapshot()
    event_warnings.observe(snapshot, now)
    return extract_hud_state(snapshot, attached_names)


# -- Mission 4F: common cleanup path ----------------------------------------


@dataclass
class SessionState:
    """Cleanup references, initialized before anything can fail."""

    vehicle: object | None = None
    suite: object | None = None


_CLEANUP_ERROR_DETAIL_LIMIT = 240


def _cleanup_failure(stage: str, exc: BaseException) -> str:
    """Return one bounded diagnostic without trusting exception formatting."""
    try:
        detail = repr(exc)
    except BaseException as formatting_error:
        detail = (
            f"<{type(exc).__name__} repr failed: "
            f"{type(formatting_error).__name__}>"
        )
    if len(detail) > _CLEANUP_ERROR_DETAIL_LIMIT:
        detail = detail[: _CLEANUP_ERROR_DETAIL_LIMIT - 3] + "..."
    return f"{stage}: {detail}"


def _attempt_cleanup_stages(suite, vehicle, quit_display):
    """Attempt every initialized stage and defer the first non-Exception."""
    failures: list[str] = []
    deferred: tuple[BaseException, object] | None = None

    def record(stage: str, exc: BaseException) -> None:
        nonlocal deferred
        failures.append(_cleanup_failure(stage, exc))
        if deferred is None and not isinstance(exc, Exception):
            deferred = (exc, exc.__traceback__)

    if suite is not None:
        try:
            suite.destroy()
        except BaseException as exc:              # keep later stages reachable
            record("sensor suite", exc)
    if vehicle is not None:
        try:
            if not vehicle.destroy():
                failures.append("vehicle: destroy() returned False")
        except BaseException as exc:              # keep display reachable
            record("vehicle", exc)
    try:
        quit_display()
    except BaseException as exc:                  # report after all attempts
        record("display", exc)
    return failures, deferred


def _add_cleanup_note(exc: BaseException, failures) -> None:
    warning = cleanup_warning(failures)
    if warning is None:
        return
    try:
        exc.add_note(warning)
    except BaseException:
        # Diagnostic attachment must not replace the selected primary failure.
        pass


def perform_cleanup(suite, vehicle, quit_display) -> list[str]:
    """Sensor suite first, then the ego vehicle, then the display.

    Every initialized stage is attempted exactly once. Ordinary Exceptions are
    returned as diagnostics. The first non-Exception BaseException is deferred
    until all stages finish, then re-raised by exact identity with its original
    traceback. SensorSuite.destroy() continues to own all sensor internals.
    """
    failures, deferred = _attempt_cleanup_stages(suite, vehicle, quit_display)
    if deferred is not None:
        exc, traceback = deferred
        _add_cleanup_note(exc, failures)
        raise exc.with_traceback(traceback)
    return failures


def cleanup_warning(failures) -> str | None:
    """At most one consolidated warning line (None when all clean)."""
    if not failures:
        return None
    return "Cleanup warning: " + "; ".join(failures)


def run_with_cleanup(action, state: SessionState, quit_display, sink=print) -> int:
    """Run the driving session and guarantee the common cleanup path.

    Normal return (Esc / window close), KeyboardInterrupt (Ctrl+C), and
    RuntimeError (CARLA disconnect, spawn failure, pygame.error) all reach
    cleanup; any other exception still gets cleanup, then propagates.
    """
    code = 0
    action_failed = False
    action_failure: tuple[BaseException, object] | None = None
    cleanup_failures: list[str] = []
    cleanup_base_exception: tuple[BaseException, object] | None = None
    try:
        try:
            action()
        except KeyboardInterrupt:
            action_failed = True
            sink("Interrupted; shutting down.")
        except RuntimeError as exc:
            action_failed = True
            # Loop already exited, so no further controls are issued. One line,
            # no automatic reconnect or respawn.
            sink(f"ERROR: {exc}")
            code = 1
    except BaseException as exc:
        action_failed = True
        action_failure = (exc, exc.__traceback__)
    finally:
        cleanup_failures, cleanup_base_exception = _attempt_cleanup_stages(
            state.suite, state.vehicle, quit_display
        )

    warning = cleanup_warning(cleanup_failures)
    if warning is not None:
        sink(warning)
    elif state.vehicle is not None:
        sink("Cleanup complete.")

    if action_failure is not None:
        exc, traceback = action_failure
        _add_cleanup_note(exc, cleanup_failures)
        raise exc.with_traceback(traceback)
    if cleanup_base_exception is not None and not action_failed:
        exc, traceback = cleanup_base_exception
        _add_cleanup_note(exc, cleanup_failures)
        raise exc.with_traceback(traceback)
    return code


# -- rendering and control loop ----------------------------------------------


def draw_hud(screen, font, vehicle, control, sensor_hud: SensorHud, warning_lines) -> None:
    screen.fill((20, 20, 25))
    lines = [
        f"Speed:    {speed_kmh(vehicle):6.1f} km/h",
        f"Throttle: {control.throttle:.2f}    Brake: {control.brake:.2f}",
        f"Steer:    {control.steer:+.2f}",
        "Reverse:  {}    Handbrake: {}".format(
            "ON " if control.reverse else "OFF",
            "ON" if control.hand_brake else "OFF",
        ),
        "",
    ] + sensor_hud_lines(sensor_hud)
    y = 14
    for line in lines:
        screen.blit(font.render(line, True, HUD_TEXT_COLOR), (14, y))
        y += 26
    for line in warning_lines:
        screen.blit(font.render(line, True, WARNING_COLOR), (14, y))
        y += 26
    y += 26
    for line in HUD_INSTRUCTIONS:
        screen.blit(font.render(line, True, HUD_TEXT_COLOR), (14, y))
        y += 26
    pygame.display.flip()


def drive_loop(world, vehicle, screen, clock, font, suite, attached_names) -> None:
    steer = 0.0
    reverse = False
    snapshot_gate = IntervalGate(SNAPSHOT_INTERVAL_SECONDS)
    event_warnings = EventWarnings()
    sensor_hud = SensorHud()
    while True:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return
                if event.key == pygame.K_r:
                    reverse = not reverse

        keys = pygame.key.get_pressed()
        steer_input = int(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - int(
            keys[pygame.K_a] or keys[pygame.K_LEFT]
        )
        steer = update_steer(steer, steer_input, dt)

        control = carla.VehicleControl(
            throttle=THROTTLE_LEVEL if keys[pygame.K_w] or keys[pygame.K_UP] else 0.0,
            brake=BRAKE_LEVEL if keys[pygame.K_s] or keys[pygame.K_DOWN] else 0.0,
            steer=round(steer, 3),
            hand_brake=bool(keys[pygame.K_SPACE]),
            reverse=reverse,
        )
        vehicle.apply_control(control)

        move_spectator_behind(world, vehicle)

        now = time.monotonic()
        if suite is not None and snapshot_gate.due(now):
            sensor_hud = sample_telemetry(
                suite, event_warnings, now, attached_names
            )
        draw_hud(screen, font, vehicle, control, sensor_hud, event_warnings.active(now))


def main() -> int:
    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT_SECONDS)

    try:
        world = client.get_world()
    except RuntimeError as exc:
        print(f"ERROR: could not connect to the CARLA server: {exc}", file=sys.stderr)
        print("Is the simulator running? Launch it manually with:", file=sys.stderr)
        print(f"  {CARLA_LAUNCH_PATH}", file=sys.stderr)
        return 1

    blueprint = pick_vehicle_blueprint(world.get_blueprint_library())
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")

    state = SessionState()

    def session() -> None:
        pygame.init()
        screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption(WINDOW_TITLE)
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 24)

        vehicle, spawn_point = spawn_at_free_spawn_point(world, blueprint)
        state.vehicle = vehicle
        loc = spawn_point.location
        print(f"Vehicle ID:     {vehicle.id}")
        print(f"Blueprint:      {vehicle.type_id}")
        print(f"Spawn location: x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}")
        move_spectator_behind(world, vehicle)

        state.suite = SensorSuite()
        report = state.suite.attach(world, vehicle)
        print(attachment_summary(report))          # exactly one summary line
        attached_names = attached_sensor_names(report)

        print("Driving started. Press Esc (or close the window) to quit.")
        drive_loop(
            world, vehicle, screen, clock, font, state.suite, attached_names
        )

    return run_with_cleanup(session, state, pygame.quit)


if __name__ == "__main__":
    sys.exit(main())
