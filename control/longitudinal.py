"""Bounded, deterministic longitudinal speed *intent* from published evidence.

Mission 10A answers exactly one question: given the Mission 8 permission, the
Mission 9 lateral request issued for that very same evidence, a detached
observation of how fast the ego is actually moving *along its own forward axis*,
and an explicit target speed, what bounded normalized throttle and brake may be
*requested* right now?

It is an intent producer, not an actuator.  It builds no ``carla.VehicleControl``,
calls no ``apply_control``, owns no CARLA actor, and emits no combined vehicle
command.  It creates no thread, queue, timer, socket, or clock: the caller
supplies host-monotonic time explicitly, exactly as Missions 8 and 9 require.

Scope boundary
--------------
Mission 10A publishes normalized throttle and brake *requests* only.  It does
**not** own the final command safety envelope, the final steering/throttle/brake
arbitration, the final command rate shaping, or any mapping onto a CARLA
control.  In particular it keeps **no throttle-rate or brake-rate baseline of
any kind**: a first valid request may reach its full Mission 10A authority in a
single step, and Mission 10B will later impose every first-command and slew
limit.  Mission 10A may only clamp a desired request to its own current
controller authority.

Signed body-frame speed model
-----------------------------
``EgoKinematics.speed_mps`` published by the telemetry layer is the unsigned
total 3-D magnitude of the world velocity.  It cannot distinguish driving
forward from rolling backward, from sliding sideways, or from falling, so it is
retained here strictly as a diagnostic and is **never** the control variable.

The control variable is the signed projection of the world velocity onto the
vehicle's own forward axis.  From the detached roll ``phi``, pitch ``theta``, and
yaw ``psi`` in degrees, converted once to radians, the body basis is

    forward = ( cos(psi) * cos(theta),
                sin(psi) * cos(theta),
                sin(theta) )

    right   = ( cos(psi) * sin(theta) * sin(phi) - sin(psi) * cos(phi),
                sin(psi) * sin(theta) * sin(phi) + cos(psi) * cos(phi),
                -cos(theta) * sin(phi) )

    up      = ( -cos(psi) * sin(theta) * cos(phi) - sin(psi) * sin(phi),
                -sin(psi) * sin(theta) * cos(phi) + cos(psi) * sin(phi),
                cos(theta) * cos(phi) )

and the signed body components are plain dot products against the world
velocity ``v``:

    forward_speed_mps  = dot(v, forward)
    lateral_speed_mps  = dot(v, right)
    vertical_speed_mps = dot(v, up)
    speed_mps          = norm(v)

Each basis vector is analytically a unit vector and the three are pairwise
orthogonal, which the tests prove rather than assume:

    |forward|^2 = cos^2(theta) * (cos^2(psi) + sin^2(psi)) + sin^2(theta) = 1
    |right|^2   = sin^2(phi) * (sin^2(theta) + cos^2(theta)) + cos^2(phi) = 1
    |up|^2      = cos^2(phi) * (sin^2(theta) + cos^2(theta)) + sin^2(phi) = 1

    dot(forward, right) = sin(theta) * cos(theta) * sin(phi) - sin(theta) * cos(theta) * sin(phi) = 0
    dot(forward, up)    = -sin(theta) * cos(theta) * cos(phi) + sin(theta) * cos(theta) * cos(phi) = 0
    dot(right, up)      = 0   (the sin(phi)cos(phi), sin(theta)sin^2(phi) and
                               sin(theta)cos^2(phi) groups each cancel exactly)

Because the basis is orthonormal, the three signed components are the velocity's
coordinates in that basis and must reconstruct it exactly:

    v = forward_speed * forward + lateral_speed * right + vertical_speed * up

Sign meaning, stated per axis rather than by naming a handedness convention:

* ``forward_speed_mps > 0`` means the ego is moving toward where its nose
  points.  ``< 0`` means it is moving backward, i.e. reversing or rolling back.
* ``lateral_speed_mps > 0`` means motion toward the vehicle's own right-hand
  side.  ``< 0`` means motion toward its left.
* ``vertical_speed_mps > 0`` means motion along the vehicle's own local up axis.
  ``< 0`` means motion downward relative to the vehicle body.

At ``phi = theta = psi = 0`` the basis is exactly ``forward = (1, 0, 0)``,
``right = (0, 1, 0)``, ``up = (0, 0, 1)``; at ``psi = 90`` degrees the nose points
along world ``+y`` and the vehicle's right-hand side points along world ``-x``.

Negative forward speed is valid evidence, not malformed evidence.  Mission 10A
never requests propulsion while it is moving backward: it requests a bounded
controlled stop instead, and implements no reverse driving of any kind.

Boundedness
-----------
Retained state is O(1): one immutable request, one accepted guidance stamp, one
accepted speed stamp, one accepted target identity, one last evaluation time,
one detached config, and bounded lifetime counters.  There is no command, frame,
decision, or pedal history, no queue, and no log.  The controller is
single-consumer and lock-free; its metrics are diagnostics read by that same
consumer and are explicitly **not** claimed to be concurrently atomic.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from enum import Enum

from control.lateral import (
    LateralControlMode,
    LateralControlReason,
    LateralControlRequest,
)
from perception.temporal import TemporalLaneState
from safety.supervisor import SafetyDecision, SafetyReason, SafetyState
from telemetry.state import (
    EulerDegrees,
    SampleStamp,
    TelemetrySnapshot,
    Vector3,
)


# The absolute construction ceiling on any requested target speed.  It is a
# hard structural bound on the *contract*, not a tuned vehicle limit.
HARD_TARGET_CEILING_MPS = 30.0

# Structural bounds on configurable gains and times, mirroring the accepted
# Mission 9 convention of bounding policy numbers rather than trusting callers.
MAX_LONGITUDINAL_GAIN = 100.0
MAX_LONGITUDINAL_AGE_SECONDS = 60.0
MAX_LONGITUDINAL_SPEED_LIMIT_MPS = 100.0

# Narrow float-roundoff tolerances for the body-frame identities.  These are
# roundoff budgets for a handful of products and sums of magnitude <= 1, whose
# error is a few units in the last place of 1.0 (~2e-16); 1e-12 is several
# orders of magnitude of headroom and still far tighter than any physically
# meaningful speed difference.
_BASIS_REL_TOL = 1e-12
_BASIS_ABS_TOL = 1e-12


__all__ = (
    "DEFAULT_LONGITUDINAL_CONTROLLER_CONFIG",
    "HARD_TARGET_CEILING_MPS",
    "MAX_LONGITUDINAL_AGE_SECONDS",
    "MAX_LONGITUDINAL_GAIN",
    "MAX_LONGITUDINAL_SPEED_LIMIT_MPS",
    "EgoSpeedObservation",
    "LongitudinalControlMode",
    "LongitudinalControlReason",
    "LongitudinalControlRequest",
    "LongitudinalController",
    "LongitudinalControllerConfig",
    "LongitudinalControllerMetrics",
    "TargetSpeedMode",
    "TargetSpeedOrigin",
    "TargetSpeedRequest",
)


def _finite_real(value: object, name: str) -> float:
    """Canonicalize one caller number into a detached, finite built-in float.

    A ``bool`` is rejected outright even though Python calls it an integer, and
    negative zero is folded onto ``+0.0`` so a sign that carries no magnitude
    can never reach a comparison or a published field.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result + 0.0


def _bounded_real(value: object, name: str, *, low: float, high: float) -> float:
    result = _finite_real(value, name)
    if not low <= result <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return result


def _nonnegative_real(value: object, name: str) -> float:
    result = _finite_real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _nonnegative_builtin_int(value: object, name: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative built-in integer")
    return value


def _builtin_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a built-in bool")
    return value


def _canonical_stamp(value: object, name: str = "source_stamp") -> SampleStamp:
    """Detach one stamp into exact built-in scalars before it is retained."""
    if not isinstance(value, SampleStamp):
        raise ValueError(f"{name} must be a SampleStamp")
    return SampleStamp(
        frame=_nonnegative_builtin_int(value.frame, f"{name}.frame"),
        sim_time=_finite_real(value.sim_time, f"{name}.sim_time"),
        monotonic=_finite_real(value.monotonic, f"{name}.monotonic"),
    )


def _same_stamp(first: SampleStamp, second: SampleStamp) -> bool:
    return (
        first.frame == second.frame
        and first.sim_time == second.sim_time
        and first.monotonic == second.monotonic
    )


def body_basis(
    roll_degrees: float, pitch_degrees: float, yaw_degrees: float
) -> tuple[tuple[float, float, float], ...]:
    """Return the ``(forward, right, up)`` body basis for one detached rotation.

    The three Euler angles are converted to radians exactly once and each
    trigonometric value is evaluated exactly once, so the returned triad is
    internally consistent by construction.  The formulas are the ones documented
    in the module docstring; their unit norms, pairwise orthogonality, and
    per-axis sign meanings are proven by the Mission 10A tests.
    """
    roll = math.radians(roll_degrees)
    pitch = math.radians(pitch_degrees)
    yaw = math.radians(yaw_degrees)
    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    forward = (
        cos_yaw * cos_pitch,
        sin_yaw * cos_pitch,
        sin_pitch,
    )
    right = (
        cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll,
        sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll,
        -cos_pitch * sin_roll,
    )
    up = (
        -cos_yaw * sin_pitch * cos_roll - sin_yaw * sin_roll,
        -sin_yaw * sin_pitch * cos_roll + cos_yaw * sin_roll,
        cos_pitch * cos_roll,
    )
    return forward, right, up


def _dot(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> float:
    return (
        first[0] * second[0] + first[1] * second[1] + first[2] * second[2]
    ) + 0.0


def _verified_body_components(
    rotation: EulerDegrees, velocity: Vector3, speed_mps: float
) -> tuple[float, float, float]:
    """Derive and prove the signed body components for one detached sample.

    Every structural property the control law depends on is checked here rather
    than assumed: the basis is finite, each axis is a unit vector, the axes are
    pairwise orthogonal, and the three signed components reconstruct the world
    velocity they came from.  A record that fails any of these is malformed
    evidence, never a licence to steer a pedal request from an unproven
    projection.
    """
    forward, right, up = body_basis(rotation.roll, rotation.pitch, rotation.yaw)
    for axis, name in ((forward, "forward"), (right, "right"), (up, "up")):
        for component in axis:
            if not math.isfinite(component):
                raise ValueError(f"{name} basis axis must be finite")
        if not math.isclose(
            math.sqrt(_dot(axis, axis)),
            1.0,
            rel_tol=_BASIS_REL_TOL,
            abs_tol=_BASIS_ABS_TOL,
        ):
            raise ValueError(f"{name} basis axis must be a unit vector")
    for first, second, name in (
        (forward, right, "forward/right"),
        (forward, up, "forward/up"),
        (right, up, "right/up"),
    ):
        if abs(_dot(first, second)) > _BASIS_ABS_TOL:
            raise ValueError(f"{name} basis axes must be orthogonal")

    world = (velocity.x, velocity.y, velocity.z)
    forward_speed = _dot(world, forward)
    lateral_speed = _dot(world, right)
    vertical_speed = _dot(world, up)
    for value, name in (
        (forward_speed, "forward_speed_mps"),
        (lateral_speed, "lateral_speed_mps"),
        (vertical_speed, "vertical_speed_mps"),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

    # An orthonormal basis makes the three components the velocity's coordinates
    # in that basis, so they must rebuild it to within float roundoff.
    tolerance = _BASIS_ABS_TOL * max(1.0, speed_mps)
    for index in range(3):
        rebuilt = (
            forward_speed * forward[index]
            + lateral_speed * right[index]
            + vertical_speed * up[index]
        )
        if not math.isfinite(rebuilt) or abs(rebuilt - world[index]) > tolerance:
            raise ValueError("body components must reconstruct the world velocity")
    return forward_speed, lateral_speed, vertical_speed


@dataclass(frozen=True, slots=True)
class EgoSpeedObservation:
    """One immutable, detached observation of how the ego is actually moving.

    The record is built from a telemetry snapshot but retains none of it: the
    stamp, world velocity, and rotation are copied into fresh accepted immutable
    values and every number is canonicalized exactly once.  ``speed_mps`` is the
    unsigned total magnitude and is a diagnostic only; the signed
    ``forward_speed_mps`` is the sole longitudinal control variable.
    """

    source_stamp: SampleStamp
    snapshot_revision: int
    snapshot_captured_monotonic: float
    world_velocity: Vector3
    rotation_degrees: EulerDegrees
    forward_speed_mps: float
    lateral_speed_mps: float
    vertical_speed_mps: float
    speed_mps: float
    vehicle_alive: bool

    def __post_init__(self) -> None:
        stamp = _canonical_stamp(self.source_stamp)
        revision = _nonnegative_builtin_int(
            self.snapshot_revision, "snapshot_revision"
        )
        captured = _finite_real(
            self.snapshot_captured_monotonic, "snapshot_captured_monotonic"
        )
        if not isinstance(self.world_velocity, Vector3):
            raise ValueError("world_velocity must be a Vector3")
        velocity = Vector3(
            x=_finite_real(self.world_velocity.x, "world_velocity.x"),
            y=_finite_real(self.world_velocity.y, "world_velocity.y"),
            z=_finite_real(self.world_velocity.z, "world_velocity.z"),
        )
        if not isinstance(self.rotation_degrees, EulerDegrees):
            raise ValueError("rotation_degrees must be an EulerDegrees")
        rotation = EulerDegrees(
            roll=_finite_real(self.rotation_degrees.roll, "rotation_degrees.roll"),
            pitch=_finite_real(self.rotation_degrees.pitch, "rotation_degrees.pitch"),
            yaw=_finite_real(self.rotation_degrees.yaw, "rotation_degrees.yaw"),
        )
        forward = _finite_real(self.forward_speed_mps, "forward_speed_mps")
        lateral = _finite_real(self.lateral_speed_mps, "lateral_speed_mps")
        vertical = _finite_real(self.vertical_speed_mps, "vertical_speed_mps")
        speed = _nonnegative_real(self.speed_mps, "speed_mps")
        alive = _builtin_bool(self.vehicle_alive, "vehicle_alive")

        magnitude = math.sqrt(
            velocity.x * velocity.x
            + velocity.y * velocity.y
            + velocity.z * velocity.z
        )
        if not math.isclose(
            speed, magnitude, rel_tol=_BASIS_REL_TOL, abs_tol=_BASIS_ABS_TOL
        ):
            raise ValueError("speed_mps must equal the world velocity magnitude")
        derived = _verified_body_components(rotation, velocity, speed)
        for value, expected, name in (
            (forward, derived[0], "forward_speed_mps"),
            (lateral, derived[1], "lateral_speed_mps"),
            (vertical, derived[2], "vertical_speed_mps"),
        ):
            if not math.isclose(
                value,
                expected,
                rel_tol=_BASIS_REL_TOL,
                abs_tol=_BASIS_ABS_TOL * max(1.0, speed),
            ):
                raise ValueError(f"{name} must match its body-frame projection")

        object.__setattr__(self, "source_stamp", stamp)
        object.__setattr__(self, "snapshot_revision", revision)
        object.__setattr__(self, "snapshot_captured_monotonic", captured)
        object.__setattr__(self, "world_velocity", velocity)
        object.__setattr__(self, "rotation_degrees", rotation)
        object.__setattr__(self, "forward_speed_mps", forward)
        object.__setattr__(self, "lateral_speed_mps", lateral)
        object.__setattr__(self, "vertical_speed_mps", vertical)
        object.__setattr__(self, "speed_mps", speed)
        object.__setattr__(self, "vehicle_alive", alive)

    @classmethod
    def from_telemetry_snapshot(
        cls, snapshot: TelemetrySnapshot
    ) -> EgoSpeedObservation | None:
        """Detach one signed speed observation from a published snapshot.

        Returns ``None`` for exactly one reason: the snapshot carries no ego
        kinematics at all.  Anything else that cannot be read, canonicalized, or
        proven self-consistent raises :class:`ValueError`, because a longitudinal
        request must never be derived from evidence that could not be verified.
        Neither the snapshot nor any object it owns is retained.
        """
        if not isinstance(snapshot, TelemetrySnapshot):
            raise ValueError("snapshot must be a TelemetrySnapshot")
        try:
            ego = snapshot.ego
            if ego is None:
                return None
            revision = snapshot.revision
            captured = snapshot.captured_monotonic
            stamp = _canonical_stamp(ego.stamp, "ego.stamp")
            transform = ego.transform
            if transform is None:
                raise ValueError("ego.transform must be present")
            rotation = transform.rotation
            if not isinstance(rotation, EulerDegrees):
                raise ValueError("ego.transform.rotation must be an EulerDegrees")
            detached_rotation = EulerDegrees(
                roll=_finite_real(rotation.roll, "rotation.roll"),
                pitch=_finite_real(rotation.pitch, "rotation.pitch"),
                yaw=_finite_real(rotation.yaw, "rotation.yaw"),
            )
            raw_velocity = ego.velocity
            if not isinstance(raw_velocity, Vector3):
                raise ValueError("ego.velocity must be a Vector3")
            velocity = Vector3(
                x=_finite_real(raw_velocity.x, "velocity.x"),
                y=_finite_real(raw_velocity.y, "velocity.y"),
                z=_finite_real(raw_velocity.z, "velocity.z"),
            )
            source_speed = _nonnegative_real(ego.speed_mps, "ego.speed_mps")
            source_kph = _nonnegative_real(ego.speed_kph, "ego.speed_kph")
            alive = _builtin_bool(ego.vehicle_alive, "ego.vehicle_alive")

            magnitude = math.sqrt(
                velocity.x * velocity.x
                + velocity.y * velocity.y
                + velocity.z * velocity.z
            )
            if not math.isfinite(magnitude):
                raise ValueError("world velocity magnitude must be finite")
            magnitude = magnitude + 0.0
            # The published unsigned speed and its kph twin must agree with the
            # velocity they claim to summarize, or the record is incoherent and
            # nothing derived from it may be trusted.
            if not math.isclose(
                source_speed,
                magnitude,
                rel_tol=_BASIS_REL_TOL,
                abs_tol=_BASIS_ABS_TOL,
            ):
                raise ValueError("ego.speed_mps must equal its velocity magnitude")
            if not math.isclose(
                source_kph,
                source_speed * 3.6,
                rel_tol=_BASIS_REL_TOL,
                abs_tol=_BASIS_ABS_TOL,
            ):
                raise ValueError("ego.speed_kph must equal speed_mps * 3.6")
            forward, lateral, vertical = _verified_body_components(
                detached_rotation, velocity, magnitude
            )
            observation = cls(
                source_stamp=stamp,
                snapshot_revision=_nonnegative_builtin_int(
                    revision, "snapshot.revision"
                ),
                snapshot_captured_monotonic=_finite_real(
                    captured, "snapshot.captured_monotonic"
                ),
                world_velocity=velocity,
                rotation_degrees=detached_rotation,
                forward_speed_mps=forward,
                lateral_speed_mps=lateral,
                vertical_speed_mps=vertical,
                speed_mps=magnitude,
                vehicle_alive=alive,
            )
        except ValueError:
            raise
        except Exception as exc:  # hostile attribute or property
            raise ValueError("snapshot ego evidence could not be read") from exc
        return observation


class TargetSpeedMode(Enum):
    """What a target speed asks for, independent of its magnitude."""

    FORWARD = "forward"
    STOP = "stop"


class TargetSpeedOrigin(Enum):
    """Who issued a target speed.  A change of origin requires a reset."""

    OPERATOR = "operator"
    SCENARIO = "scenario"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class TargetSpeedRequest:
    """One immutable, explicitly bounded and explicitly dated target speed.

    Mission 10A never invents a target.  The caller states one, states who issued
    it, states which epoch and revision it belongs to, and states the monotonic
    window in which it may be honoured.  Outside that window the controller
    requests a controlled stop rather than continuing on a stale intent.
    """

    mode: TargetSpeedMode
    target_speed_mps: float
    origin: TargetSpeedOrigin
    source_epoch: int
    source_revision: int
    issued_monotonic: float
    valid_until_monotonic: float

    def __post_init__(self) -> None:
        if type(self.mode) is not TargetSpeedMode:
            raise ValueError("mode must be a TargetSpeedMode")
        if type(self.origin) is not TargetSpeedOrigin:
            raise ValueError("origin must be a TargetSpeedOrigin")
        speed = _nonnegative_real(self.target_speed_mps, "target_speed_mps")
        if speed > HARD_TARGET_CEILING_MPS:
            raise ValueError(
                f"target_speed_mps must not exceed {HARD_TARGET_CEILING_MPS}"
            )
        if (self.mode is TargetSpeedMode.STOP) is not (speed == 0.0):
            raise ValueError("STOP means exactly zero target speed and nothing else")
        epoch = _nonnegative_builtin_int(self.source_epoch, "source_epoch")
        revision = _nonnegative_builtin_int(self.source_revision, "source_revision")
        issued = _finite_real(self.issued_monotonic, "issued_monotonic")
        valid_until = _finite_real(
            self.valid_until_monotonic, "valid_until_monotonic"
        )
        if issued > valid_until:
            raise ValueError("issued_monotonic must not follow valid_until_monotonic")
        object.__setattr__(self, "target_speed_mps", speed)
        object.__setattr__(self, "source_epoch", epoch)
        object.__setattr__(self, "source_revision", revision)
        object.__setattr__(self, "issued_monotonic", issued)
        object.__setattr__(self, "valid_until_monotonic", valid_until)


class LongitudinalControlMode(Enum):
    """How much longitudinal authority this request carries, if any."""

    DISABLED = "disabled"
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    STOPPING = "stopping"


class LongitudinalControlReason(Enum):
    """One bounded explanation for the current longitudinal request."""

    NOT_EVALUATED = "not_evaluated"
    NOMINAL_TRACKING = "nominal_tracking"
    DEGRADED_TRACKING = "degraded_tracking"
    DEGRADED_COASTING = "degraded_coasting"
    TARGET_STOP = "target_stop"
    SAFETY_STOP = "safety_stop"
    FAIL_SAFE_STOP = "fail_safe_stop"
    LATERAL_STOP = "lateral_stop"
    SPEED_UNAVAILABLE = "speed_unavailable"
    SPEED_STALE = "speed_stale"
    TARGET_EXPIRED = "target_expired"
    VEHICLE_NOT_ALIVE = "vehicle_not_alive"
    REVERSE_MOTION = "reverse_motion"
    EXCESSIVE_LATERAL_MOTION = "excessive_lateral_motion"
    EXCESSIVE_VERTICAL_MOTION = "excessive_vertical_motion"
    IDENTITY_MISMATCH = "identity_mismatch"
    EVIDENCE_MALFORMED = "evidence_malformed"
    SOURCE_REJECTED = "source_rejected"
    TIME_INVALID = "time_invalid"


# Every reason belongs to exactly one mode, so a request can never advertise
# propulsion authority under a fail-closed explanation, and can never present a
# neutral refusal as though it were a trusted controlled stop.
_REASON_MODE: dict[LongitudinalControlReason, LongitudinalControlMode] = {
    LongitudinalControlReason.NOT_EVALUATED: LongitudinalControlMode.DISABLED,
    LongitudinalControlReason.NOMINAL_TRACKING: LongitudinalControlMode.NOMINAL,
    LongitudinalControlReason.DEGRADED_TRACKING: LongitudinalControlMode.DEGRADED,
    LongitudinalControlReason.DEGRADED_COASTING: LongitudinalControlMode.DEGRADED,
    LongitudinalControlReason.TARGET_STOP: LongitudinalControlMode.STOPPING,
    LongitudinalControlReason.SAFETY_STOP: LongitudinalControlMode.STOPPING,
    LongitudinalControlReason.FAIL_SAFE_STOP: LongitudinalControlMode.STOPPING,
    LongitudinalControlReason.LATERAL_STOP: LongitudinalControlMode.STOPPING,
    LongitudinalControlReason.SPEED_UNAVAILABLE: LongitudinalControlMode.DISABLED,
    LongitudinalControlReason.SPEED_STALE: LongitudinalControlMode.STOPPING,
    LongitudinalControlReason.TARGET_EXPIRED: LongitudinalControlMode.STOPPING,
    LongitudinalControlReason.VEHICLE_NOT_ALIVE: LongitudinalControlMode.DISABLED,
    LongitudinalControlReason.REVERSE_MOTION: LongitudinalControlMode.STOPPING,
    LongitudinalControlReason.EXCESSIVE_LATERAL_MOTION: (
        LongitudinalControlMode.STOPPING
    ),
    LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION: (
        LongitudinalControlMode.DISABLED
    ),
    LongitudinalControlReason.IDENTITY_MISMATCH: LongitudinalControlMode.DISABLED,
    LongitudinalControlReason.EVIDENCE_MALFORMED: LongitudinalControlMode.DISABLED,
    LongitudinalControlReason.SOURCE_REJECTED: LongitudinalControlMode.DISABLED,
    LongitudinalControlReason.TIME_INVALID: LongitudinalControlMode.DISABLED,
}

# Propulsion is permitted in exactly two modes and nowhere else.
_PERMITTED_MODES = frozenset(
    (LongitudinalControlMode.NOMINAL, LongitudinalControlMode.DEGRADED)
)

# Mission 8 permits autonomy in exactly these two states.
_PERMITTED_SAFETY_STATES = frozenset((SafetyState.NOMINAL, SafetyState.DEGRADED))

# Mission 9 carries lateral authority in exactly these two modes.
_PERMITTED_LATERAL_MODES = frozenset(
    (LateralControlMode.NOMINAL, LateralControlMode.DEGRADED)
)

# The only Mission 7 states that can carry guidance at all.  Every other
# temporal state reaches Mission 10A only as a refusal or a controlled stop.
_GUIDANCE_STATES = frozenset(
    (TemporalLaneState.TRACKING, TemporalLaneState.COASTING)
)


@dataclass(frozen=True, slots=True)
class LongitudinalControllerConfig:
    """One small, exact longitudinal policy for a controller epoch.

    Every default is a deliberately conservative offline exhibition value, not a
    tuned or measured vehicle parameter.  Nothing here has been validated against
    a real vehicle, a real road, or a running simulator.

    The speed caps bound how fast each authority level may ever ask to go; the
    throttle authorities bound the normalized pedal request each level may ever
    make; and ``max_service_brake`` bounds every brake request without exception.
    The ordering constraints are enforced rather than documented, so a degraded
    or coasting epoch can never obtain nominal authority by configuration.
    """

    nominal_speed_cap_mps: float = 3.0
    degraded_speed_cap_mps: float = 1.5
    coasting_speed_cap_mps: float = 1.0

    nominal_throttle_authority: float = 0.35
    degraded_throttle_authority: float = 0.15
    coasting_throttle_authority: float = 0.10

    max_service_brake: float = 0.40

    speed_deadband_mps: float = 0.10
    throttle_kp_per_mps: float = 0.30
    brake_kp_per_mps: float = 0.40

    minimum_moving_stop_brake: float = 0.15
    soft_stop_brake: float = 0.20
    latched_stop_brake: float = 0.35
    reverse_stop_brake: float = 0.25
    zero_hold_brake: float = 0.10

    stop_threshold_mps: float = 0.10

    component_zero_epsilon_mps: float = 1e-9
    reverse_motion_threshold_mps: float = 0.05
    lateral_speed_limit_mps: float = 0.50
    vertical_speed_limit_mps: float = 0.25

    maximum_ego_speed_age_seconds: float = 0.25
    maximum_snapshot_age_seconds: float = 0.25
    maximum_target_age_seconds: float = 1.0

    hard_target_ceiling_mps: float = HARD_TARGET_CEILING_MPS

    def __post_init__(self) -> None:
        ceiling = _finite_real(self.hard_target_ceiling_mps, "hard_target_ceiling_mps")
        if not 0.0 < ceiling <= HARD_TARGET_CEILING_MPS:
            raise ValueError(
                f"hard_target_ceiling_mps must be in (0.0, {HARD_TARGET_CEILING_MPS}]"
            )
        nominal_cap = _finite_real(self.nominal_speed_cap_mps, "nominal_speed_cap_mps")
        degraded_cap = _finite_real(
            self.degraded_speed_cap_mps, "degraded_speed_cap_mps"
        )
        coasting_cap = _finite_real(
            self.coasting_speed_cap_mps, "coasting_speed_cap_mps"
        )
        if not 0.0 < coasting_cap < degraded_cap < nominal_cap <= ceiling:
            raise ValueError(
                "speed caps must satisfy "
                "0 < coasting < degraded < nominal <= hard_target_ceiling"
            )

        nominal_throttle = _finite_real(
            self.nominal_throttle_authority, "nominal_throttle_authority"
        )
        degraded_throttle = _finite_real(
            self.degraded_throttle_authority, "degraded_throttle_authority"
        )
        coasting_throttle = _finite_real(
            self.coasting_throttle_authority, "coasting_throttle_authority"
        )
        if not 0.0 < coasting_throttle < degraded_throttle < nominal_throttle <= 1.0:
            raise ValueError(
                "throttle authorities must satisfy "
                "0 < coasting < degraded < nominal <= 1"
            )

        max_brake = _finite_real(self.max_service_brake, "max_service_brake")
        if not 0.0 < max_brake <= 1.0:
            raise ValueError("max_service_brake must be in (0.0, 1.0]")
        brakes = {}
        for name in (
            "minimum_moving_stop_brake",
            "soft_stop_brake",
            "latched_stop_brake",
            "reverse_stop_brake",
            "zero_hold_brake",
        ):
            # Every one of these names a *stop or hold intent*, and a stop intent
            # that requests exactly nothing is not a stop.  Strict positivity is
            # what lets `stop_required=True` mean bounded deceleration was really
            # asked for.  Ordinary proportional braking may still compute zero
            # when no braking is needed; that is regulation, not a stop request.
            value = _finite_real(getattr(self, name), name)
            if not 0.0 < value <= max_brake:
                raise ValueError(
                    f"{name} must be in (0.0, max_service_brake]"
                )
            brakes[name] = value

        deadband = _nonnegative_real(self.speed_deadband_mps, "speed_deadband_mps")
        throttle_kp = _finite_real(self.throttle_kp_per_mps, "throttle_kp_per_mps")
        if not 0.0 < throttle_kp <= MAX_LONGITUDINAL_GAIN:
            raise ValueError(
                f"throttle_kp_per_mps must be in (0.0, {MAX_LONGITUDINAL_GAIN}]"
            )
        brake_kp = _finite_real(self.brake_kp_per_mps, "brake_kp_per_mps")
        if not 0.0 < brake_kp <= MAX_LONGITUDINAL_GAIN:
            raise ValueError(
                f"brake_kp_per_mps must be in (0.0, {MAX_LONGITUDINAL_GAIN}]"
            )

        epsilon = _finite_real(
            self.component_zero_epsilon_mps, "component_zero_epsilon_mps"
        )
        reverse_threshold = _finite_real(
            self.reverse_motion_threshold_mps, "reverse_motion_threshold_mps"
        )
        stop_threshold = _finite_real(
            self.stop_threshold_mps, "stop_threshold_mps"
        )
        if not 0.0 < epsilon < reverse_threshold < stop_threshold:
            raise ValueError(
                "motion thresholds must satisfy "
                "0 < component_zero_epsilon < reverse_motion_threshold "
                "< stop_threshold"
            )

        limits = {}
        for name in ("lateral_speed_limit_mps", "vertical_speed_limit_mps"):
            value = _nonnegative_real(getattr(self, name), name)
            if value > MAX_LONGITUDINAL_SPEED_LIMIT_MPS:
                raise ValueError(
                    f"{name} must not exceed {MAX_LONGITUDINAL_SPEED_LIMIT_MPS}"
                )
            limits[name] = value

        ages = {}
        for name in (
            "maximum_ego_speed_age_seconds",
            "maximum_snapshot_age_seconds",
            "maximum_target_age_seconds",
        ):
            value = _nonnegative_real(getattr(self, name), name)
            if value > MAX_LONGITUDINAL_AGE_SECONDS:
                raise ValueError(
                    f"{name} must not exceed {MAX_LONGITUDINAL_AGE_SECONDS}"
                )
            ages[name] = value

        object.__setattr__(self, "hard_target_ceiling_mps", ceiling)
        object.__setattr__(self, "nominal_speed_cap_mps", nominal_cap)
        object.__setattr__(self, "degraded_speed_cap_mps", degraded_cap)
        object.__setattr__(self, "coasting_speed_cap_mps", coasting_cap)
        object.__setattr__(self, "nominal_throttle_authority", nominal_throttle)
        object.__setattr__(self, "degraded_throttle_authority", degraded_throttle)
        object.__setattr__(self, "coasting_throttle_authority", coasting_throttle)
        object.__setattr__(self, "max_service_brake", max_brake)
        object.__setattr__(self, "speed_deadband_mps", deadband)
        object.__setattr__(self, "throttle_kp_per_mps", throttle_kp)
        object.__setattr__(self, "brake_kp_per_mps", brake_kp)
        object.__setattr__(self, "component_zero_epsilon_mps", epsilon)
        object.__setattr__(self, "reverse_motion_threshold_mps", reverse_threshold)
        object.__setattr__(self, "stop_threshold_mps", stop_threshold)
        for name, value in brakes.items():
            object.__setattr__(self, name, value)
        for name, value in limits.items():
            object.__setattr__(self, name, value)
        for name, value in ages.items():
            object.__setattr__(self, name, value)


DEFAULT_LONGITUDINAL_CONTROLLER_CONFIG = LongitudinalControllerConfig()


@dataclass(frozen=True, slots=True)
class LongitudinalControlRequest:
    """One immutable longitudinal intent and the evidence behind it.

    ``longitudinal_allowed`` is the single question a downstream Mission 10B
    command layer must obey before any propulsion reaches a vehicle command, and
    ``stop_required`` distinguishes a *trusted controlled-stop request* from a
    neutral refusal that asks for nothing at all.  Mission 10A applies neither:
    ``throttle_request`` and ``brake_request`` are normalized desired values in
    ``[0, 1]``, never pedal commands, and they carry no rate shaping whatsoever.
    """

    mode: LongitudinalControlMode
    reason: LongitudinalControlReason
    longitudinal_allowed: bool
    stop_required: bool = False

    throttle_request: float = 0.0
    brake_request: float = 0.0

    effective_target_speed_mps: float | None = None
    observed_forward_speed_mps: float | None = None
    speed_error_mps: float | None = None

    speed_authority_mps: float | None = None
    throttle_authority: float = 0.0
    brake_authority: float = 0.0

    throttle_clamped: bool = False
    brake_clamped: bool = False

    evaluation_monotonic: float | None = None

    guidance_stamp: SampleStamp | None = None
    speed_stamp: SampleStamp | None = None
    snapshot_revision: int | None = None

    target_origin: TargetSpeedOrigin | None = None
    target_epoch: int | None = None
    target_revision: int | None = None
    target_issued_monotonic: float | None = None
    target_valid_until_monotonic: float | None = None

    safety_state: SafetyState | None = None
    lateral_mode: LateralControlMode | None = None
    temporal_state: TemporalLaneState | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, LongitudinalControlReason):
            raise ValueError("reason must be a LongitudinalControlReason")
        if not isinstance(self.mode, LongitudinalControlMode):
            raise ValueError("mode must be a LongitudinalControlMode")
        if self.mode is not _REASON_MODE[self.reason]:
            raise ValueError("mode does not match its longitudinal control reason")
        allowed = _builtin_bool(self.longitudinal_allowed, "longitudinal_allowed")
        if allowed is not (self.mode in _PERMITTED_MODES):
            raise ValueError("longitudinal_allowed does not match the control mode")
        stop_required = _builtin_bool(self.stop_required, "stop_required")
        if stop_required is not (self.mode is LongitudinalControlMode.STOPPING):
            raise ValueError("stop_required does not match the control mode")

        throttle = _bounded_real(self.throttle_request, "throttle_request", low=0.0, high=1.0)
        brake = _bounded_real(self.brake_request, "brake_request", low=0.0, high=1.0)
        throttle_authority = _bounded_real(
            self.throttle_authority, "throttle_authority", low=0.0, high=1.0
        )
        brake_authority = _bounded_real(
            self.brake_authority, "brake_authority", low=0.0, high=1.0
        )
        throttle_clamped = _builtin_bool(self.throttle_clamped, "throttle_clamped")
        brake_clamped = _builtin_bool(self.brake_clamped, "brake_clamped")

        if throttle > 0.0 and brake > 0.0:
            raise ValueError("throttle and brake can never both be requested")
        if throttle > throttle_authority:
            raise ValueError("throttle cannot exceed its own authority")
        if brake > brake_authority:
            raise ValueError("brake cannot exceed its own authority")
        if not allowed and throttle != 0.0:
            raise ValueError("a request without propulsion permission cannot throttle")
        if self.mode is LongitudinalControlMode.DISABLED and (
            throttle != 0.0
            or brake != 0.0
            or throttle_authority != 0.0
            or brake_authority != 0.0
        ):
            raise ValueError("a disabled request cannot carry any pedal authority")
        if self.mode is LongitudinalControlMode.STOPPING and (
            throttle != 0.0 or throttle_authority != 0.0
        ):
            raise ValueError("a stopping request cannot carry throttle authority")

        effective_target = self.effective_target_speed_mps
        if effective_target is not None:
            effective_target = _nonnegative_real(
                effective_target, "effective_target_speed_mps"
            )
            if effective_target > HARD_TARGET_CEILING_MPS:
                raise ValueError(
                    "effective_target_speed_mps must respect the hard ceiling"
                )
        observed_forward = self.observed_forward_speed_mps
        if observed_forward is not None:
            observed_forward = _finite_real(
                observed_forward, "observed_forward_speed_mps"
            )
        speed_error = self.speed_error_mps
        if speed_error is not None:
            speed_error = _finite_real(speed_error, "speed_error_mps")
        speed_authority = self.speed_authority_mps
        if speed_authority is not None:
            speed_authority = _nonnegative_real(
                speed_authority, "speed_authority_mps"
            )
            if speed_authority > HARD_TARGET_CEILING_MPS:
                raise ValueError("speed_authority_mps must respect the hard ceiling")

        evaluated = self.evaluation_monotonic
        if evaluated is not None:
            evaluated = _finite_real(evaluated, "evaluation_monotonic")
        guidance_stamp = (
            None
            if self.guidance_stamp is None
            else _canonical_stamp(self.guidance_stamp, "guidance_stamp")
        )
        speed_stamp = (
            None
            if self.speed_stamp is None
            else _canonical_stamp(self.speed_stamp, "speed_stamp")
        )
        revision = self.snapshot_revision
        if revision is not None:
            revision = _nonnegative_builtin_int(revision, "snapshot_revision")

        if self.target_origin is not None and type(self.target_origin) is not (
            TargetSpeedOrigin
        ):
            raise ValueError("target_origin must be a TargetSpeedOrigin or None")
        target_epoch = self.target_epoch
        if target_epoch is not None:
            target_epoch = _nonnegative_builtin_int(target_epoch, "target_epoch")
        target_revision = self.target_revision
        if target_revision is not None:
            target_revision = _nonnegative_builtin_int(
                target_revision, "target_revision"
            )
        target_issued = self.target_issued_monotonic
        if target_issued is not None:
            target_issued = _finite_real(target_issued, "target_issued_monotonic")
        target_valid_until = self.target_valid_until_monotonic
        if target_valid_until is not None:
            target_valid_until = _finite_real(
                target_valid_until, "target_valid_until_monotonic"
            )
        if (
            target_issued is not None
            and target_valid_until is not None
            and target_issued > target_valid_until
        ):
            raise ValueError("target issue time cannot follow its expiry")

        if self.safety_state is not None and not isinstance(
            self.safety_state, SafetyState
        ):
            raise ValueError("safety_state must be a SafetyState or None")
        if self.lateral_mode is not None and not isinstance(
            self.lateral_mode, LateralControlMode
        ):
            raise ValueError("lateral_mode must be a LateralControlMode or None")
        if self.temporal_state is not None and not isinstance(
            self.temporal_state, TemporalLaneState
        ):
            raise ValueError("temporal_state must be a TemporalLaneState or None")

        # Any request that positively asks for deceleration, and any request that
        # carries propulsion, must be bound to the exact evidence cut it came
        # from.  A neutral refusal is the only shape allowed to be evidence-free.
        identity = (
            evaluated,
            guidance_stamp,
            speed_stamp,
            revision,
            self.target_origin,
            target_epoch,
            target_revision,
            target_issued,
            target_valid_until,
            self.safety_state,
            self.lateral_mode,
            self.temporal_state,
        )
        # A STOPPING request is a positive, trusted demand for deceleration, so
        # it must actually ask for some, and it must be bound to the exact
        # evidence cut that justified it.  Identity is required because the mode
        # is STOPPING, never merely because the brake value happens to be
        # nonzero: there is no "stop required but nothing requested" state.
        if stop_required and brake <= 0.0:
            raise ValueError("a stopping request must request positive brake")
        if (allowed or brake > 0.0 or stop_required) and any(
            value is None for value in identity
        ):
            raise ValueError(
                "a propelling or stopping request requires complete identity binding"
            )
        if allowed or self.reason is LongitudinalControlReason.TARGET_STOP:
            if any(
                value is None
                for value in (
                    effective_target,
                    observed_forward,
                    speed_error,
                    speed_authority,
                )
            ):
                raise ValueError("an evaluated control law requires complete evidence")
            if self.safety_state not in _PERMITTED_SAFETY_STATES:
                raise ValueError("an evaluated law requires a permitted safety state")
            if self.lateral_mode not in _PERMITTED_LATERAL_MODES:
                raise ValueError("an evaluated law requires a permitted lateral mode")
            if self.temporal_state not in _GUIDANCE_STATES:
                raise ValueError("an evaluated law requires a guidance-bearing state")
        if allowed:
            if self.mode is LongitudinalControlMode.NOMINAL and (
                self.safety_state is not SafetyState.NOMINAL
                or self.lateral_mode is not LateralControlMode.NOMINAL
                or self.temporal_state is not TemporalLaneState.TRACKING
            ):
                raise ValueError(
                    "nominal longitudinal authority requires fully nominal evidence"
                )
            if effective_target is not None and speed_authority is not None:
                if effective_target > speed_authority:
                    raise ValueError("effective target cannot exceed its speed authority")
        else:
            if throttle_clamped:
                raise ValueError("a refused request cannot report throttle clamping")
            if brake == 0.0 and brake_clamped:
                raise ValueError("a request with no brake cannot report brake clamping")

        if self.reason is LongitudinalControlReason.NOT_EVALUATED and any(
            value is not None
            for value in (
                *identity,
                effective_target,
                observed_forward,
                speed_error,
                speed_authority,
            )
        ):
            raise ValueError("an unevaluated request cannot expose any evidence")

        object.__setattr__(self, "longitudinal_allowed", allowed)
        object.__setattr__(self, "stop_required", stop_required)
        object.__setattr__(self, "throttle_request", throttle)
        object.__setattr__(self, "brake_request", brake)
        object.__setattr__(self, "effective_target_speed_mps", effective_target)
        object.__setattr__(self, "observed_forward_speed_mps", observed_forward)
        object.__setattr__(self, "speed_error_mps", speed_error)
        object.__setattr__(self, "speed_authority_mps", speed_authority)
        object.__setattr__(self, "throttle_authority", throttle_authority)
        object.__setattr__(self, "brake_authority", brake_authority)
        object.__setattr__(self, "throttle_clamped", throttle_clamped)
        object.__setattr__(self, "brake_clamped", brake_clamped)
        object.__setattr__(self, "evaluation_monotonic", evaluated)
        object.__setattr__(self, "guidance_stamp", guidance_stamp)
        object.__setattr__(self, "speed_stamp", speed_stamp)
        object.__setattr__(self, "snapshot_revision", revision)
        object.__setattr__(self, "target_epoch", target_epoch)
        object.__setattr__(self, "target_revision", target_revision)
        object.__setattr__(self, "target_issued_monotonic", target_issued)
        object.__setattr__(
            self, "target_valid_until_monotonic", target_valid_until
        )

    @property
    def guidance_frame(self) -> int | None:
        """The CARLA camera frame this request was bound to, if any."""
        stamp = self.guidance_stamp
        return None if stamp is None else stamp.frame

    @property
    def speed_frame(self) -> int | None:
        """The CARLA ego frame this request measured speed from, if any."""
        stamp = self.speed_stamp
        return None if stamp is None else stamp.frame


@dataclass(frozen=True, slots=True)
class LongitudinalControllerMetrics:
    """One immutable view of the bounded scalar controller counters.

    These are lifetime diagnostics.  No control behavior is derived from them,
    and they are read from the single owning consumer, not concurrently.
    """

    evaluations: int
    permitted_requests: int
    refused_requests: int
    nominal_requests: int
    degraded_requests: int
    stop_requests: int
    throttle_requests: int
    brake_requests: int
    target_stops: int
    safety_stops: int
    fail_safe_stops: int
    lateral_stops: int
    target_expiries: int
    speed_stale_stops: int
    speed_unavailable_rejects: int
    vehicle_not_alive_rejects: int
    reverse_motion_stops: int
    lateral_motion_stops: int
    vertical_motion_rejects: int
    identity_rejects: int
    malformed_rejects: int
    source_rejects: int
    time_rejects: int
    throttle_clamp_activations: int
    brake_clamp_activations: int
    resets: int
    last_accepted_guidance_stamp: SampleStamp | None
    last_accepted_speed_stamp: SampleStamp | None
    last_accepted_target_origin: TargetSpeedOrigin | None
    last_accepted_target_epoch: int | None
    last_accepted_target_revision: int | None


class _EvidenceError(Exception):
    """Internal marker for evidence Mission 10A must fail closed on."""


class _MalformedEvidence(_EvidenceError):
    pass


class _IdentityMismatch(_EvidenceError):
    pass


class _TimeInvalid(_EvidenceError):
    pass


def _canonicalize_safety_decision(decision: SafetyDecision) -> SafetyDecision:
    """Rebuild one exact, detached :class:`SafetyDecision` from a caller record.

    Every caller-controlled attribute is read *exactly once* into a local, and
    the locals are then handed to the real accepted constructor.  That
    constructor re-proves the whole Mission 8 contract -- the reason/state map,
    the permission/state agreement, the latched/``FAIL_SAFE`` rule, the
    optional-evidence shape for ``INITIALIZING``, built-in bool and
    non-negative-integer types, unit-range confidence, finite non-negative age,
    and the accepted enum types -- so a record that bypassed that constructor,
    or a subclass that contradicts it in a field Mission 10A never reads,
    cannot survive.  The returned record is a fresh exact base instance whose
    nested stamp the constructor has already detached, so no caller object is
    retained.
    """
    if not isinstance(decision, SafetyDecision):
        raise _MalformedEvidence("decision must be a SafetyDecision")
    state = decision.state
    reason = decision.reason
    autonomy_allowed = decision.autonomy_allowed
    evaluation_monotonic = decision.evaluation_monotonic
    source_stamp = decision.source_stamp
    snapshot_revision = decision.snapshot_revision
    result_age_seconds = decision.result_age_seconds
    temporal_state = decision.temporal_state
    temporal_confidence = decision.temporal_confidence
    recovery_streak = decision.recovery_streak
    latched = decision.latched

    # Enum identity is checked here as well as in the constructor so a
    # look-alike can never reach the reason/state lookup.
    if type(state) is not SafetyState:
        raise _MalformedEvidence("decision state must be a SafetyState")
    if type(reason) is not SafetyReason:
        raise _MalformedEvidence("decision reason must be a SafetyReason")
    if temporal_state is not None and type(temporal_state) is not TemporalLaneState:
        raise _MalformedEvidence("decision temporal state must be a TemporalLaneState")

    canonical = SafetyDecision(
        state=state,
        reason=reason,
        autonomy_allowed=autonomy_allowed,
        evaluation_monotonic=evaluation_monotonic,
        source_stamp=source_stamp,
        snapshot_revision=snapshot_revision,
        result_age_seconds=result_age_seconds,
        temporal_state=temporal_state,
        temporal_confidence=temporal_confidence,
        recovery_streak=recovery_streak,
        latched=latched,
    )
    # Mission 8 publishes stamp, revision, temporal state, and confidence
    # together from one accepted result or omits all four; a record carrying
    # only some of them never came from the accepted supervisor.
    published = (
        canonical.source_stamp,
        canonical.snapshot_revision,
        canonical.temporal_state,
        canonical.temporal_confidence,
    )
    if any(value is not None for value in published) and any(
        value is None for value in published
    ):
        raise _MalformedEvidence("decision result evidence must be complete or absent")
    return canonical


def _canonicalize_lateral_request(
    lateral: LateralControlRequest,
) -> LateralControlRequest:
    """Rebuild one exact, detached :class:`LateralControlRequest`.

    As with the Mission 8 record, every caller-controlled attribute is read
    exactly once and the real accepted constructor re-proves the whole Mission 9
    contract: the reason/mode map, the permission/mode agreement, bounded
    steering inside its own authority, finite ``raw_steering``, bounded lane
    errors, built-in bool shaping flags, the completeness a permitted request
    requires, the nominal-mode safety rule, the neutrality a refused request
    requires, and the accepted enum types.  A malformed field invalidates the
    record even when Mission 10A performs no arithmetic on it.
    """
    if not isinstance(lateral, LateralControlRequest):
        raise _MalformedEvidence("lateral must be a LateralControlRequest")
    mode = lateral.mode
    reason = lateral.reason
    lateral_allowed = lateral.lateral_allowed
    steering = lateral.steering
    raw_steering = lateral.raw_steering
    steering_authority = lateral.steering_authority
    clamped = lateral.clamped
    rate_limited = lateral.rate_limited
    evaluation_monotonic = lateral.evaluation_monotonic
    source_stamp = lateral.source_stamp
    snapshot_revision = lateral.snapshot_revision
    lateral_error = lateral.lateral_error
    heading_error = lateral.heading_error
    safety_state = lateral.safety_state
    temporal_state = lateral.temporal_state

    if type(mode) is not LateralControlMode:
        raise _MalformedEvidence("lateral mode must be a LateralControlMode")
    if type(reason) is not LateralControlReason:
        raise _MalformedEvidence("lateral reason must be a LateralControlReason")
    if safety_state is not None and type(safety_state) is not SafetyState:
        raise _MalformedEvidence("lateral safety state must be a SafetyState")
    if temporal_state is not None and type(temporal_state) is not TemporalLaneState:
        raise _MalformedEvidence("lateral temporal state must be a TemporalLaneState")

    return LateralControlRequest(
        mode=mode,
        reason=reason,
        lateral_allowed=lateral_allowed,
        steering=steering,
        raw_steering=raw_steering,
        steering_authority=steering_authority,
        clamped=clamped,
        rate_limited=rate_limited,
        evaluation_monotonic=evaluation_monotonic,
        source_stamp=source_stamp,
        snapshot_revision=snapshot_revision,
        lateral_error=lateral_error,
        heading_error=heading_error,
        safety_state=safety_state,
        temporal_state=temporal_state,
    )


@dataclass(frozen=True, slots=True)
class _AcceptedEvidence:
    """The only per-update evidence the controller reads, all canonical."""

    guidance_stamp: SampleStamp
    speed_stamp: SampleStamp
    snapshot_revision: int
    snapshot_captured_monotonic: float
    vehicle_alive: bool

    safety_state: SafetyState
    safety_allowed: bool
    safety_latched: bool

    lateral_mode: LateralControlMode
    lateral_allowed: bool
    lateral_coasting: bool
    temporal_state: TemporalLaneState

    forward_speed_mps: float
    lateral_speed_mps: float
    vertical_speed_mps: float
    total_speed_mps: float

    target_mode: TargetSpeedMode
    target_speed_mps: float
    target_origin: TargetSpeedOrigin
    target_epoch: int
    target_revision: int
    target_issued_monotonic: float
    target_valid_until_monotonic: float


class LongitudinalController:
    """Single-consumer, lock-free longitudinal speed intent for one epoch.

    ``update`` consumes the Mission 8 decision, the Mission 9 lateral request
    issued for that same evidence cut, a detached :class:`EgoSpeedObservation`,
    and an explicit :class:`TargetSpeedRequest`, and returns the current
    immutable :class:`LongitudinalControlRequest`.  It creates no thread, queue,
    timer, or clock: the caller supplies the one canonical host-monotonic ``now``
    that Missions 8 and 9 were already given, and all three must agree exactly.

    Missions 8 and 9 remain the sole authorities on whether motion is permitted.
    This class never re-decides that, never overrides it, and can only subtract
    authority: it additionally refuses whenever the four evidence sources do not
    describe one coherent cut, whenever any of them is stale or replayed, and
    whenever the ego's own signed motion contradicts forward propulsion.
    """

    __slots__ = (
        "_brake_clamp_activations",
        "_brake_requests",
        "_config",
        "_degraded_requests",
        "_evaluations",
        "_fail_safe_stops",
        "_identity_rejects",
        "_last_evaluation_monotonic",
        "_last_guidance_stamp",
        "_last_guidance_revision",
        "_last_speed_stamp",
        "_last_speed_revision",
        "_last_speed_captured",
        "_last_target_epoch",
        "_last_target_issued",
        "_last_target_mode",
        "_last_target_origin",
        "_last_target_revision",
        "_last_target_speed",
        "_last_target_valid_until",
        "_lateral_motion_stops",
        "_lateral_stops",
        "_malformed_rejects",
        "_nominal_requests",
        "_permitted_requests",
        "_refused_requests",
        "_request",
        "_resets",
        "_reverse_motion_stops",
        "_safety_stops",
        "_source_rejects",
        "_speed_stale_stops",
        "_speed_unavailable_rejects",
        "_stop_requests",
        "_target_expiries",
        "_target_stops",
        "_throttle_clamp_activations",
        "_throttle_requests",
        "_time_rejects",
        "_vehicle_not_alive_rejects",
        "_vertical_motion_rejects",
    )

    def __init__(
        self,
        config: LongitudinalControllerConfig = DEFAULT_LONGITUDINAL_CONTROLLER_CONFIG,
    ) -> None:
        if not isinstance(config, LongitudinalControllerConfig):
            raise TypeError("config must be a LongitudinalControllerConfig")
        self._config = _detached_config(config)
        self._evaluations = 0
        self._permitted_requests = 0
        self._refused_requests = 0
        self._nominal_requests = 0
        self._degraded_requests = 0
        self._stop_requests = 0
        self._throttle_requests = 0
        self._brake_requests = 0
        self._target_stops = 0
        self._safety_stops = 0
        self._fail_safe_stops = 0
        self._lateral_stops = 0
        self._target_expiries = 0
        self._speed_stale_stops = 0
        self._speed_unavailable_rejects = 0
        self._vehicle_not_alive_rejects = 0
        self._reverse_motion_stops = 0
        self._lateral_motion_stops = 0
        self._vertical_motion_rejects = 0
        self._identity_rejects = 0
        self._malformed_rejects = 0
        self._source_rejects = 0
        self._time_rejects = 0
        self._throttle_clamp_activations = 0
        self._brake_clamp_activations = 0
        self._resets = 0
        self._clear_state()

    @property
    def config(self) -> LongitudinalControllerConfig:
        return _detached_config(self._config)

    @property
    def latest_request(self) -> LongitudinalControlRequest:
        return self._request

    @property
    def longitudinal_allowed(self) -> bool:
        return self._request.longitudinal_allowed

    @property
    def metrics(self) -> LongitudinalControllerMetrics:
        guidance = self._last_guidance_stamp
        speed = self._last_speed_stamp
        return LongitudinalControllerMetrics(
            evaluations=self._evaluations,
            permitted_requests=self._permitted_requests,
            refused_requests=self._refused_requests,
            nominal_requests=self._nominal_requests,
            degraded_requests=self._degraded_requests,
            stop_requests=self._stop_requests,
            throttle_requests=self._throttle_requests,
            brake_requests=self._brake_requests,
            target_stops=self._target_stops,
            safety_stops=self._safety_stops,
            fail_safe_stops=self._fail_safe_stops,
            lateral_stops=self._lateral_stops,
            target_expiries=self._target_expiries,
            speed_stale_stops=self._speed_stale_stops,
            speed_unavailable_rejects=self._speed_unavailable_rejects,
            vehicle_not_alive_rejects=self._vehicle_not_alive_rejects,
            reverse_motion_stops=self._reverse_motion_stops,
            lateral_motion_stops=self._lateral_motion_stops,
            vertical_motion_rejects=self._vertical_motion_rejects,
            identity_rejects=self._identity_rejects,
            malformed_rejects=self._malformed_rejects,
            source_rejects=self._source_rejects,
            time_rejects=self._time_rejects,
            throttle_clamp_activations=self._throttle_clamp_activations,
            brake_clamp_activations=self._brake_clamp_activations,
            resets=self._resets,
            last_accepted_guidance_stamp=(
                None if guidance is None else _canonical_stamp(guidance)
            ),
            last_accepted_speed_stamp=(
                None if speed is None else _canonical_stamp(speed)
            ),
            last_accepted_target_origin=self._last_target_origin,
            last_accepted_target_epoch=self._last_target_epoch,
            last_accepted_target_revision=self._last_target_revision,
        )

    def reset(self) -> LongitudinalControlRequest:
        """Begin a fresh controller epoch; lifetime counters remain truthful.

        No previous request, evaluation time, guidance baseline, speed baseline,
        or target identity crosses the boundary, and the published request
        returns to neutral and refused.  Repeated calls are idempotent apart from
        the reset counter.  Mission 10A holds no pedal-rate baseline to clear,
        because it owns no rate shaping at all.
        """
        self._resets += 1
        self._clear_state()
        return self._request

    def _clear_state(self) -> None:
        self._last_evaluation_monotonic: float | None = None
        self._last_guidance_stamp: SampleStamp | None = None
        self._last_guidance_revision: int | None = None
        self._last_speed_stamp: SampleStamp | None = None
        self._last_speed_revision: int | None = None
        self._last_speed_captured: float | None = None
        self._last_target_origin: TargetSpeedOrigin | None = None
        self._last_target_epoch: int | None = None
        self._last_target_revision: int | None = None
        self._last_target_issued: float | None = None
        self._last_target_valid_until: float | None = None
        self._last_target_mode: TargetSpeedMode | None = None
        self._last_target_speed: float | None = None
        self._request = LongitudinalControlRequest(
            mode=LongitudinalControlMode.DISABLED,
            reason=LongitudinalControlReason.NOT_EVALUATED,
            longitudinal_allowed=False,
        )

    def update(
        self,
        decision: SafetyDecision,
        lateral: LateralControlRequest,
        speed: EgoSpeedObservation | None,
        target: TargetSpeedRequest,
        *,
        now: float,
    ) -> LongitudinalControlRequest:
        """Produce one bounded longitudinal intent at control time ``now``.

        ``decision`` and ``lateral`` must have been produced for one evidence cut
        at exactly this ``now``; ``speed`` must be the detached observation for
        the same snapshot revision; ``target`` must be current.  All four are read
        but never retained.  No baseline of any kind advances unless the whole
        update succeeds, so a hostile property or a rejected input can never
        leave the controller partially mutated.
        """
        if not isinstance(decision, SafetyDecision):
            raise TypeError("decision must be a SafetyDecision")
        if not isinstance(lateral, LateralControlRequest):
            raise TypeError("lateral must be a LateralControlRequest")
        if speed is not None and not isinstance(speed, EgoSpeedObservation):
            raise TypeError("speed must be an EgoSpeedObservation or None")
        if not isinstance(target, TargetSpeedRequest):
            raise TypeError("target must be a TargetSpeedRequest")
        self._evaluations += 1
        config = self._config

        # 1. Control clock.  A clock that is not finite, or that has gone
        #    backwards, is never silently accepted as the start of a new epoch:
        #    the evaluation baseline survives untouched so the rollback stays
        #    visible to the next call.
        try:
            evaluated = _finite_real(now, "now")
        except ValueError:
            return self._refuse(LongitudinalControlReason.TIME_INVALID, None)
        previous_evaluation = self._last_evaluation_monotonic
        if previous_evaluation is not None and evaluated < previous_evaluation:
            return self._refuse(LongitudinalControlReason.TIME_INVALID, None)

        # 2. Ego speed presence, before anything is derived from it.
        if speed is None:
            return self._refuse(
                LongitudinalControlReason.SPEED_UNAVAILABLE, evaluated
            )

        # 3. Read and canonicalize every caller-controlled attribute exactly
        #    once, so a hostile property cannot answer one value to a gate and a
        #    different one to the control law.
        try:
            evidence = self._read_evidence(decision, lateral, speed, target, evaluated)
        except _TimeInvalid:
            return self._refuse(LongitudinalControlReason.TIME_INVALID, evaluated)
        except _IdentityMismatch:
            return self._refuse(LongitudinalControlReason.IDENTITY_MISMATCH, evaluated)
        except Exception:
            return self._refuse(LongitudinalControlReason.EVIDENCE_MALFORMED, evaluated)

        # 4. A dead ego is not evidence of anything a controller may act on.
        if not evidence.vehicle_alive:
            return self._refuse(
                LongitudinalControlReason.VEHICLE_NOT_ALIVE, evaluated
            )

        # 5. Source order for guidance, speed, and target.  A rejected source
        #    never moves any baseline.
        rejected = self._order_violation(evidence, evaluated, previous_evaluation)
        if rejected:
            return self._refuse(LongitudinalControlReason.SOURCE_REJECTED, evaluated)

        # From here on every outcome is identity-bound, so each one may publish
        # the full binding and each one commits the accepted baselines.
        forward = evidence.forward_speed_mps
        if -config.component_zero_epsilon_mps <= forward < 0.0:
            # A projection of a stationary vehicle onto its own forward axis can
            # land a few ulps below zero.  That is float noise, not reverse.
            forward = 0.0

        # 6. Excessive vertical motion, before *any* branch that could ask for
        #    deceleration.  A vehicle moving along its own up axis has untrusted
        #    road contact, so its braking authority is untrusted too: requesting
        #    a specific brake value here would assert confidence this controller
        #    does not have.  The only honest answer is to ask for nothing, and
        #    that answer must outrank every stop branch below, including a
        #    latched Mission 8 fail-safe.  The earlier neutral gates stay ahead
        #    of it, so an unreadable, unbindable, or out-of-order record never
        #    gets to use a vertical scalar to justify any output at all.
        if abs(evidence.vertical_speed_mps) > config.vertical_speed_limit_mps:
            return self._neutral(
                LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION,
                evidence,
                evaluated,
                forward,
            )

        # 7. Mission 8 and Mission 9 permission, before any pedal value exists.
        if evidence.safety_latched:
            return self._stop(
                LongitudinalControlReason.FAIL_SAFE_STOP,
                evidence,
                evaluated,
                forward,
                config.latched_stop_brake,
            )
        if not evidence.safety_allowed:
            return self._stop(
                LongitudinalControlReason.SAFETY_STOP,
                evidence,
                evaluated,
                forward,
                config.soft_stop_brake,
            )
        if not evidence.lateral_allowed:
            return self._stop(
                LongitudinalControlReason.LATERAL_STOP,
                evidence,
                evaluated,
                forward,
                config.soft_stop_brake,
            )

        # 8. Freshness of the evidence the law would otherwise use.  Both are
        #    coherent, identity-bound inputs that have simply aged out, so both
        #    ask for a controlled stop rather than refusing neutrally.
        if (
            evaluated - evidence.target_valid_until_monotonic > 0.0
            or evaluated - evidence.target_issued_monotonic
            > config.maximum_target_age_seconds
        ):
            return self._stop(
                LongitudinalControlReason.TARGET_EXPIRED,
                evidence,
                evaluated,
                forward,
                config.soft_stop_brake,
            )
        if (
            evaluated - evidence.speed_stamp.monotonic
            > config.maximum_ego_speed_age_seconds
            or evaluated - evidence.snapshot_captured_monotonic
            > config.maximum_snapshot_age_seconds
        ):
            return self._stop(
                LongitudinalControlReason.SPEED_STALE,
                evidence,
                evaluated,
                forward,
                config.soft_stop_brake,
            )

        # 9. The remaining signed motion policy, in strict precedence.  The
        #    vertical anomaly was already settled at step 6, ahead of every
        #    brake-bearing branch above.
        if abs(evidence.lateral_speed_mps) > config.lateral_speed_limit_mps:
            return self._stop(
                LongitudinalControlReason.EXCESSIVE_LATERAL_MOTION,
                evidence,
                evaluated,
                forward,
                config.soft_stop_brake,
            )
        if forward < -config.reverse_motion_threshold_mps:
            return self._stop(
                LongitudinalControlReason.REVERSE_MOTION,
                evidence,
                evaluated,
                forward,
                config.reverse_stop_brake,
            )
        if forward < 0.0:
            # Backward but below the reverse threshold: hold, do not propel.
            return self._stop(
                LongitudinalControlReason.REVERSE_MOTION,
                evidence,
                evaluated,
                forward,
                config.zero_hold_brake,
            )

        # 10. Mode, authority, and the ordinary signed-forward proportional law.
        return self._control(evidence, evaluated, forward)

    # ---------------------------------------------------------------- evidence

    def _read_evidence(
        self,
        decision: SafetyDecision,
        lateral: LateralControlRequest,
        speed: EgoSpeedObservation,
        target: TargetSpeedRequest,
        evaluated: float,
    ) -> _AcceptedEvidence:
        """Canonicalize all four sources and prove they describe one cut.

        The Mission 8 and Mission 9 records are not merely spot-checked on the
        fields Mission 10A happens to consume: each is fully reconstructed
        through its own accepted constructor, so every intrinsic upstream
        invariant is re-proven and a malformed field invalidates the whole
        record even when no Mission 10A arithmetic reads it.
        """
        # --- Mission 8 -----------------------------------------------------
        canonical_decision = _canonicalize_safety_decision(decision)
        safety_state = canonical_decision.state
        safety_allowed = canonical_decision.autonomy_allowed
        safety_latched = canonical_decision.latched
        decision_evaluated = canonical_decision.evaluation_monotonic
        decision_stamp = canonical_decision.source_stamp
        decision_revision = canonical_decision.snapshot_revision
        decision_age = canonical_decision.result_age_seconds
        decision_temporal = canonical_decision.temporal_state
        if decision_stamp is None or decision_revision is None:
            raise _IdentityMismatch("a bindable decision requires stamp and revision")
        if decision_evaluated is None:
            raise _IdentityMismatch("a bindable decision requires an evaluation time")
        if decision_temporal is None:
            raise _IdentityMismatch("a bindable decision requires a temporal state")

        # --- Mission 9 -----------------------------------------------------
        canonical_lateral = _canonicalize_lateral_request(lateral)
        lateral_mode = canonical_lateral.mode
        lateral_reason = canonical_lateral.reason
        lateral_allowed = canonical_lateral.lateral_allowed
        lateral_evaluated = canonical_lateral.evaluation_monotonic
        lateral_stamp = canonical_lateral.source_stamp
        lateral_revision = canonical_lateral.snapshot_revision
        lateral_safety_state = canonical_lateral.safety_state
        lateral_temporal = canonical_lateral.temporal_state
        if lateral_stamp is None or lateral_revision is None:
            raise _IdentityMismatch("a bindable lateral request requires its evidence")
        if lateral_evaluated is None:
            raise _IdentityMismatch("a bindable lateral request requires its time")
        if lateral_safety_state is None:
            raise _IdentityMismatch("a bindable lateral request requires a state")
        if lateral_temporal is None:
            raise _IdentityMismatch("a bindable lateral request requires a temporal state")

        # --- Ego speed ------------------------------------------------------
        speed_stamp = _canonical_stamp(speed.source_stamp, "speed.source_stamp")
        speed_revision = _nonnegative_builtin_int(
            speed.snapshot_revision, "speed.snapshot_revision"
        )
        speed_captured = _finite_real(
            speed.snapshot_captured_monotonic, "speed.snapshot_captured_monotonic"
        )
        raw_velocity = speed.world_velocity
        if not isinstance(raw_velocity, Vector3):
            raise _MalformedEvidence("speed world_velocity must be a Vector3")
        velocity = Vector3(
            x=_finite_real(raw_velocity.x, "speed.world_velocity.x"),
            y=_finite_real(raw_velocity.y, "speed.world_velocity.y"),
            z=_finite_real(raw_velocity.z, "speed.world_velocity.z"),
        )
        raw_rotation = speed.rotation_degrees
        if not isinstance(raw_rotation, EulerDegrees):
            raise _MalformedEvidence("speed rotation_degrees must be an EulerDegrees")
        rotation = EulerDegrees(
            roll=_finite_real(raw_rotation.roll, "speed.rotation_degrees.roll"),
            pitch=_finite_real(raw_rotation.pitch, "speed.rotation_degrees.pitch"),
            yaw=_finite_real(raw_rotation.yaw, "speed.rotation_degrees.yaw"),
        )
        claimed_forward = _finite_real(speed.forward_speed_mps, "speed.forward_speed_mps")
        claimed_lateral = _finite_real(speed.lateral_speed_mps, "speed.lateral_speed_mps")
        claimed_vertical = _finite_real(
            speed.vertical_speed_mps, "speed.vertical_speed_mps"
        )
        total_speed = _nonnegative_real(speed.speed_mps, "speed.speed_mps")
        vehicle_alive = _builtin_bool(speed.vehicle_alive, "speed.vehicle_alive")
        magnitude = math.sqrt(
            velocity.x * velocity.x
            + velocity.y * velocity.y
            + velocity.z * velocity.z
        )
        if not math.isclose(
            total_speed, magnitude, rel_tol=_BASIS_REL_TOL, abs_tol=_BASIS_ABS_TOL
        ):
            raise _MalformedEvidence("speed_mps must equal its velocity magnitude")
        # The controller re-derives the projection instead of trusting the
        # observation, so a forged subclass cannot smuggle in a forward speed the
        # rotation and velocity it publishes do not actually support.
        derived = _verified_body_components(rotation, velocity, total_speed)
        for claimed, expected, name in (
            (claimed_forward, derived[0], "forward_speed_mps"),
            (claimed_lateral, derived[1], "lateral_speed_mps"),
            (claimed_vertical, derived[2], "vertical_speed_mps"),
        ):
            if not math.isclose(
                claimed,
                expected,
                rel_tol=_BASIS_REL_TOL,
                abs_tol=_BASIS_ABS_TOL * max(1.0, total_speed),
            ):
                raise _MalformedEvidence(f"{name} contradicts its own projection")

        # --- Target ---------------------------------------------------------
        target_mode = target.mode
        if type(target_mode) is not TargetSpeedMode:
            raise _MalformedEvidence("target mode must be a TargetSpeedMode")
        target_origin = target.origin
        if type(target_origin) is not TargetSpeedOrigin:
            raise _MalformedEvidence("target origin must be a TargetSpeedOrigin")
        target_speed = _nonnegative_real(target.target_speed_mps, "target_speed_mps")
        # The *structural* ceiling is what no accepted target can ever exceed, so
        # a value above it can only come from a forged record.  A target above
        # the merely *configured* ceiling or the mode cap is legitimate and is
        # clamped by the control law, not refused.
        if target_speed > HARD_TARGET_CEILING_MPS:
            raise _MalformedEvidence("target speed exceeds the structural ceiling")
        if (target_mode is TargetSpeedMode.STOP) is not (target_speed == 0.0):
            raise _MalformedEvidence("target mode contradicts its magnitude")
        target_epoch = _nonnegative_builtin_int(target.source_epoch, "source_epoch")
        target_revision = _nonnegative_builtin_int(
            target.source_revision, "source_revision"
        )
        target_issued = _finite_real(target.issued_monotonic, "issued_monotonic")
        target_valid_until = _finite_real(
            target.valid_until_monotonic, "valid_until_monotonic"
        )
        if target_issued > target_valid_until:
            raise _MalformedEvidence("target issue time follows its expiry")

        # --- The one canonical evaluation epoch ------------------------------
        if decision_evaluated != evaluated or lateral_evaluated != evaluated:
            raise _IdentityMismatch("every source must share the caller's control time")

        # --- Independent freshness recomputation -----------------------------
        recomputed_age = evaluated - decision_stamp.monotonic
        if not math.isfinite(recomputed_age) or recomputed_age < 0.0:
            raise _TimeInvalid("decision age must be finite and non-negative")
        if decision_age is None:
            raise _TimeInvalid("a bindable decision must publish its result age")
        decision_age = _finite_real(decision_age, "decision.result_age_seconds")
        if not math.isclose(
            decision_age, recomputed_age, rel_tol=0.0, abs_tol=_BASIS_ABS_TOL
        ):
            raise _TimeInvalid("published decision age contradicts the control clock")
        speed_age = evaluated - speed_stamp.monotonic
        snapshot_age = evaluated - speed_captured
        target_age = evaluated - target_issued
        for value, name in (
            (speed_age, "ego speed age"),
            (snapshot_age, "snapshot age"),
            (target_age, "target age"),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise _TimeInvalid(f"{name} must be finite and non-negative")

        # --- Cross-source identity binding -----------------------------------
        if not _same_stamp(decision_stamp, lateral_stamp):
            raise _IdentityMismatch("decision and lateral describe different frames")
        if decision_revision != lateral_revision or decision_revision != speed_revision:
            raise _IdentityMismatch("sources describe different snapshot revisions")
        if safety_state is not lateral_safety_state:
            raise _IdentityMismatch("decision and lateral disagree on safety state")
        if decision_temporal is not lateral_temporal:
            raise _IdentityMismatch("decision and lateral disagree on temporal state")

        return _AcceptedEvidence(
            guidance_stamp=decision_stamp,
            speed_stamp=speed_stamp,
            snapshot_revision=decision_revision,
            snapshot_captured_monotonic=speed_captured,
            vehicle_alive=vehicle_alive,
            safety_state=safety_state,
            safety_allowed=safety_allowed,
            safety_latched=safety_latched,
            lateral_mode=lateral_mode,
            lateral_allowed=lateral_allowed,
            lateral_coasting=(
                lateral_reason is LateralControlReason.DEGRADED_COASTING
            ),
            temporal_state=lateral_temporal,
            forward_speed_mps=claimed_forward,
            lateral_speed_mps=claimed_lateral,
            vertical_speed_mps=claimed_vertical,
            total_speed_mps=total_speed,
            target_mode=target_mode,
            target_speed_mps=target_speed,
            target_origin=target_origin,
            target_epoch=target_epoch,
            target_revision=target_revision,
            target_issued_monotonic=target_issued,
            target_valid_until_monotonic=target_valid_until,
        )

    # ------------------------------------------------------------------ order

    def _order_violation(
        self,
        evidence: _AcceptedEvidence,
        evaluated: float,
        previous_evaluation: float | None,
    ) -> bool:
        """Return ``True`` when any source is a replay, a reorder, or a mutation.

        Mission 6 duplicate semantics are preserved exactly: a duplicate is the
        same frame *and* the same simulation time, a higher frame at an equal
        simulation time is legitimate, and anything older or reordered is not.  A
        duplicate whose monotonic identity or bound revision has changed is an
        identity-mutated replay and is rejected rather than re-evaluated.
        """
        guidance = evidence.guidance_stamp
        baseline = self._last_guidance_stamp
        guidance_duplicate = False
        if baseline is not None:
            guidance_duplicate = (
                guidance.frame == baseline.frame
                and guidance.sim_time == baseline.sim_time
            )
            if guidance_duplicate:
                if (
                    guidance.monotonic != baseline.monotonic
                    or evidence.snapshot_revision != self._last_guidance_revision
                ):
                    return True
            elif (
                guidance.frame <= baseline.frame
                or guidance.sim_time < baseline.sim_time
            ):
                return True

        speed = evidence.speed_stamp
        speed_baseline = self._last_speed_stamp
        speed_duplicate = False
        if speed_baseline is not None:
            speed_duplicate = (
                speed.frame == speed_baseline.frame
                and speed.sim_time == speed_baseline.sim_time
            )
            if speed_duplicate:
                if (
                    speed.monotonic != speed_baseline.monotonic
                    or evidence.snapshot_revision != self._last_speed_revision
                    or evidence.snapshot_captured_monotonic
                    != self._last_speed_captured
                ):
                    return True
            elif (
                speed.frame <= speed_baseline.frame
                or speed.sim_time < speed_baseline.sim_time
            ):
                return True

        target_duplicate = False
        if self._last_target_origin is not None:
            # A different issuer or a different epoch is a new controller
            # contract, and a new contract requires an explicit reset.
            if (
                evidence.target_origin is not self._last_target_origin
                or evidence.target_epoch != self._last_target_epoch
            ):
                return True
            target_duplicate = (
                evidence.target_revision == self._last_target_revision
                and evidence.target_issued_monotonic == self._last_target_issued
            )
            if target_duplicate:
                if (
                    evidence.target_valid_until_monotonic
                    != self._last_target_valid_until
                    or evidence.target_mode is not self._last_target_mode
                    or evidence.target_speed_mps != self._last_target_speed
                ):
                    return True
            elif (
                evidence.target_revision <= self._last_target_revision
                or evidence.target_issued_monotonic < self._last_target_issued
            ):
                return True

        # A control clock that has not advanced may only re-evaluate evidence
        # that has not changed either.  New evidence at an unchanged instant is
        # an incoherent presentation, not a faster control loop.
        if previous_evaluation is not None and evaluated == previous_evaluation:
            if not (guidance_duplicate and speed_duplicate and target_duplicate):
                return True
        return False

    # --------------------------------------------------------------- outcomes

    def _control(
        self,
        evidence: _AcceptedEvidence,
        evaluated: float,
        forward: float,
    ) -> LongitudinalControlRequest:
        """Select mode and authority, then run the proportional speed law."""
        config = self._config
        if (
            evidence.safety_state is SafetyState.NOMINAL
            and evidence.lateral_mode is LateralControlMode.NOMINAL
            and evidence.temporal_state is TemporalLaneState.TRACKING
            and not evidence.lateral_coasting
        ):
            reason = LongitudinalControlReason.NOMINAL_TRACKING
            speed_cap = config.nominal_speed_cap_mps
            throttle_authority = config.nominal_throttle_authority
        elif (
            evidence.lateral_coasting
            or evidence.temporal_state is TemporalLaneState.COASTING
        ):
            reason = LongitudinalControlReason.DEGRADED_COASTING
            speed_cap = config.coasting_speed_cap_mps
            throttle_authority = config.coasting_throttle_authority
        else:
            reason = LongitudinalControlReason.DEGRADED_TRACKING
            speed_cap = config.degraded_speed_cap_mps
            throttle_authority = config.degraded_throttle_authority

        effective_target = min(
            evidence.target_speed_mps, speed_cap, config.hard_target_ceiling_mps
        )
        speed_error = effective_target - forward
        throttle = 0.0
        brake = 0.0
        throttle_clamped = False
        brake_clamped = False

        if effective_target == 0.0:
            # A commanded stop is not propulsion at any authority: it is a
            # trusted controlled stop with no throttle authority at all.
            if forward <= config.stop_threshold_mps:
                brake = config.zero_hold_brake
            else:
                brake = max(
                    config.minimum_moving_stop_brake,
                    config.brake_kp_per_mps * forward,
                )
            if brake > config.max_service_brake:
                brake = config.max_service_brake
                brake_clamped = True
            request = LongitudinalControlRequest(
                mode=LongitudinalControlMode.STOPPING,
                reason=LongitudinalControlReason.TARGET_STOP,
                longitudinal_allowed=False,
                stop_required=True,
                throttle_request=0.0,
                brake_request=brake,
                effective_target_speed_mps=effective_target,
                observed_forward_speed_mps=forward,
                speed_error_mps=speed_error,
                speed_authority_mps=speed_cap,
                throttle_authority=0.0,
                brake_authority=config.max_service_brake,
                brake_clamped=brake_clamped,
                **self._binding(evidence, evaluated),
            )
            return self._settle(request, evidence, evaluated)

        if speed_error > config.speed_deadband_mps:
            throttle = config.throttle_kp_per_mps * (
                speed_error - config.speed_deadband_mps
            )
            if throttle > throttle_authority:
                throttle = throttle_authority
                throttle_clamped = True
        elif speed_error < -config.speed_deadband_mps:
            brake = config.brake_kp_per_mps * (
                -speed_error - config.speed_deadband_mps
            )
            if brake > config.max_service_brake:
                brake = config.max_service_brake
                brake_clamped = True

        request = LongitudinalControlRequest(
            mode=_REASON_MODE[reason],
            reason=reason,
            longitudinal_allowed=True,
            stop_required=False,
            throttle_request=throttle,
            brake_request=brake,
            effective_target_speed_mps=effective_target,
            observed_forward_speed_mps=forward,
            speed_error_mps=speed_error,
            speed_authority_mps=speed_cap,
            throttle_authority=throttle_authority,
            brake_authority=config.max_service_brake,
            throttle_clamped=throttle_clamped,
            brake_clamped=brake_clamped,
            **self._binding(evidence, evaluated),
        )
        return self._settle(request, evidence, evaluated)

    def _stop(
        self,
        reason: LongitudinalControlReason,
        evidence: _AcceptedEvidence,
        evaluated: float,
        forward: float,
        brake: float,
    ) -> LongitudinalControlRequest:
        """Publish one trusted, identity-bound controlled-stop request."""
        config = self._config
        brake_clamped = False
        if brake > config.max_service_brake:
            brake = config.max_service_brake
            brake_clamped = True
        request = LongitudinalControlRequest(
            mode=LongitudinalControlMode.STOPPING,
            reason=reason,
            longitudinal_allowed=False,
            stop_required=True,
            throttle_request=0.0,
            brake_request=brake,
            observed_forward_speed_mps=forward,
            throttle_authority=0.0,
            brake_authority=config.max_service_brake,
            brake_clamped=brake_clamped,
            **self._binding(evidence, evaluated),
        )
        return self._settle(request, evidence, evaluated)

    def _neutral(
        self,
        reason: LongitudinalControlReason,
        evidence: _AcceptedEvidence,
        evaluated: float,
        forward: float,
    ) -> LongitudinalControlRequest:
        """Publish one identity-bound neutral refusal that asks for nothing."""
        request = LongitudinalControlRequest(
            mode=LongitudinalControlMode.DISABLED,
            reason=reason,
            longitudinal_allowed=False,
            stop_required=False,
            throttle_request=0.0,
            brake_request=0.0,
            observed_forward_speed_mps=forward,
            **self._binding(evidence, evaluated),
        )
        return self._settle(request, evidence, evaluated)

    def _refuse(
        self,
        reason: LongitudinalControlReason,
        evaluated: float | None,
    ) -> LongitudinalControlRequest:
        """Publish one neutral refusal and advance no baseline whatsoever.

        This is the answer whenever the evidence could not be read, could not be
        bound, could not be ordered, or arrived on an untrustworthy clock.  It
        requests neither throttle nor brake, it never leaves a previously
        permitted request standing as the apparent current answer, and it leaves
        every accepted baseline exactly where it was.
        """
        request = LongitudinalControlRequest(
            mode=LongitudinalControlMode.DISABLED,
            reason=reason,
            longitudinal_allowed=False,
            evaluation_monotonic=evaluated,
        )
        self._count(request)
        self._request = request
        return request

    def _binding(
        self, evidence: _AcceptedEvidence, evaluated: float
    ) -> dict[str, object]:
        """The exact identity every non-neutral outcome must publish."""
        return dict(
            evaluation_monotonic=evaluated,
            guidance_stamp=evidence.guidance_stamp,
            speed_stamp=evidence.speed_stamp,
            snapshot_revision=evidence.snapshot_revision,
            target_origin=evidence.target_origin,
            target_epoch=evidence.target_epoch,
            target_revision=evidence.target_revision,
            target_issued_monotonic=evidence.target_issued_monotonic,
            target_valid_until_monotonic=evidence.target_valid_until_monotonic,
            safety_state=evidence.safety_state,
            lateral_mode=evidence.lateral_mode,
            temporal_state=evidence.temporal_state,
        )

    def _settle(
        self,
        request: LongitudinalControlRequest,
        evidence: _AcceptedEvidence,
        evaluated: float,
    ) -> LongitudinalControlRequest:
        """Commit one fully constructed request and its accepted baselines.

        Nothing before this point has mutated an order, identity, target, or
        clock baseline, so an exception raised anywhere earlier leaves the
        controller exactly as it was.
        """
        self._count(request)
        self._last_guidance_stamp = evidence.guidance_stamp
        self._last_guidance_revision = evidence.snapshot_revision
        self._last_speed_stamp = evidence.speed_stamp
        self._last_speed_revision = evidence.snapshot_revision
        self._last_speed_captured = evidence.snapshot_captured_monotonic
        self._last_target_origin = evidence.target_origin
        self._last_target_epoch = evidence.target_epoch
        self._last_target_revision = evidence.target_revision
        self._last_target_issued = evidence.target_issued_monotonic
        self._last_target_valid_until = evidence.target_valid_until_monotonic
        self._last_target_mode = evidence.target_mode
        self._last_target_speed = evidence.target_speed_mps
        self._last_evaluation_monotonic = evaluated
        self._request = request
        return request

    def _count(self, request: LongitudinalControlRequest) -> None:
        """Record one published request in the bounded lifetime counters."""
        reason = request.reason
        if request.longitudinal_allowed:
            self._permitted_requests += 1
            if request.mode is LongitudinalControlMode.NOMINAL:
                self._nominal_requests += 1
            else:
                self._degraded_requests += 1
        else:
            self._refused_requests += 1
        if request.stop_required:
            self._stop_requests += 1
        if request.throttle_request > 0.0:
            self._throttle_requests += 1
        if request.brake_request > 0.0:
            self._brake_requests += 1
        if request.throttle_clamped:
            self._throttle_clamp_activations += 1
        if request.brake_clamped:
            self._brake_clamp_activations += 1

        if reason is LongitudinalControlReason.TARGET_STOP:
            self._target_stops += 1
        elif reason is LongitudinalControlReason.SAFETY_STOP:
            self._safety_stops += 1
        elif reason is LongitudinalControlReason.FAIL_SAFE_STOP:
            self._fail_safe_stops += 1
        elif reason is LongitudinalControlReason.LATERAL_STOP:
            self._lateral_stops += 1
        elif reason is LongitudinalControlReason.TARGET_EXPIRED:
            self._target_expiries += 1
        elif reason is LongitudinalControlReason.SPEED_STALE:
            self._speed_stale_stops += 1
        elif reason is LongitudinalControlReason.SPEED_UNAVAILABLE:
            self._speed_unavailable_rejects += 1
        elif reason is LongitudinalControlReason.VEHICLE_NOT_ALIVE:
            self._vehicle_not_alive_rejects += 1
        elif reason is LongitudinalControlReason.REVERSE_MOTION:
            self._reverse_motion_stops += 1
        elif reason is LongitudinalControlReason.EXCESSIVE_LATERAL_MOTION:
            self._lateral_motion_stops += 1
        elif reason is LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION:
            self._vertical_motion_rejects += 1
        elif reason is LongitudinalControlReason.IDENTITY_MISMATCH:
            self._identity_rejects += 1
        elif reason is LongitudinalControlReason.EVIDENCE_MALFORMED:
            self._malformed_rejects += 1
        elif reason is LongitudinalControlReason.SOURCE_REJECTED:
            self._source_rejects += 1
        elif reason is LongitudinalControlReason.TIME_INVALID:
            self._time_rejects += 1


def _detached_config(
    config: LongitudinalControllerConfig,
) -> LongitudinalControllerConfig:
    """Rebuild one config from its scalars so no caller record is retained."""
    return LongitudinalControllerConfig(
        nominal_speed_cap_mps=config.nominal_speed_cap_mps,
        degraded_speed_cap_mps=config.degraded_speed_cap_mps,
        coasting_speed_cap_mps=config.coasting_speed_cap_mps,
        nominal_throttle_authority=config.nominal_throttle_authority,
        degraded_throttle_authority=config.degraded_throttle_authority,
        coasting_throttle_authority=config.coasting_throttle_authority,
        max_service_brake=config.max_service_brake,
        speed_deadband_mps=config.speed_deadband_mps,
        throttle_kp_per_mps=config.throttle_kp_per_mps,
        brake_kp_per_mps=config.brake_kp_per_mps,
        minimum_moving_stop_brake=config.minimum_moving_stop_brake,
        soft_stop_brake=config.soft_stop_brake,
        latched_stop_brake=config.latched_stop_brake,
        reverse_stop_brake=config.reverse_stop_brake,
        zero_hold_brake=config.zero_hold_brake,
        stop_threshold_mps=config.stop_threshold_mps,
        component_zero_epsilon_mps=config.component_zero_epsilon_mps,
        reverse_motion_threshold_mps=config.reverse_motion_threshold_mps,
        lateral_speed_limit_mps=config.lateral_speed_limit_mps,
        vertical_speed_limit_mps=config.vertical_speed_limit_mps,
        maximum_ego_speed_age_seconds=config.maximum_ego_speed_age_seconds,
        maximum_snapshot_age_seconds=config.maximum_snapshot_age_seconds,
        maximum_target_age_seconds=config.maximum_target_age_seconds,
        hard_target_ceiling_mps=config.hard_target_ceiling_mps,
    )
