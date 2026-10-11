"""Deterministic unit tests for the Mission 10A longitudinal controller.

Every input here is an immutable synthetic record built from the accepted
Mission 5/6/7/8/9 public types.  No CARLA package, server, socket, worker, or
wall clock is involved: control time is always supplied explicitly.

The body-frame expectations are deliberately *not* computed with the production
closed-form basis.  They are built from an independently composed rotation
matrix ``Rz(yaw) . Ry(-pitch) . Rx(-roll)`` whose columns must equal the
production forward/right/up axes, so a sign error in either derivation shows up
as a disagreement rather than cancelling out.
"""

from __future__ import annotations

import gc
import math
import numbers
import unittest
import weakref
from dataclasses import FrozenInstanceError, fields

from control.lateral import (
    LateralControlMode,
    LateralControlReason,
    LateralControlRequest,
)
from control.longitudinal import (
    DEFAULT_LONGITUDINAL_CONTROLLER_CONFIG,
    HARD_TARGET_CEILING_MPS,
    EgoSpeedObservation,
    LongitudinalControlMode,
    LongitudinalControlReason,
    LongitudinalControlRequest,
    LongitudinalController,
    LongitudinalControllerConfig,
    TargetSpeedMode,
    TargetSpeedOrigin,
    TargetSpeedRequest,
    _REASON_MODE,
    _canonicalize_lateral_request,
    _canonicalize_safety_decision,
    body_basis,
)
from perception.temporal import TemporalLaneState
from safety.supervisor import SafetyDecision, SafetyReason, SafetyState
from telemetry.state import (
    EgoKinematics,
    EulerDegrees,
    SampleStamp,
    TelemetrySnapshot,
    TransformState,
    Vector3,
)


BASE = 100.0
GUIDANCE_MONOTONIC = BASE - 0.05
CONFIG = DEFAULT_LONGITUDINAL_CONTROLLER_CONFIG

# A spread of rotations including the axis-aligned golden cases, sign mirrors,
# and awkward combined attitudes.
ANGLE_SWEEP = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 90.0),
    (0.0, 0.0, 180.0),
    (0.0, 0.0, -90.0),
    (0.0, 0.0, 37.0),
    (0.0, 30.0, 0.0),
    (0.0, -30.0, 0.0),
    (25.0, 0.0, 0.0),
    (-25.0, 0.0, 0.0),
    (12.0, -7.0, 143.0),
    (-88.0, 41.0, -172.0),
    (179.0, 89.0, 359.0),
)


# --------------------------------------------------------------------------
# Independent reference mathematics
# --------------------------------------------------------------------------


def _matmul(left, right):
    return tuple(
        tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def reference_basis(roll_deg: float, pitch_deg: float, yaw_deg: float):
    """Return (forward, right, up) from an independently composed matrix.

    The production module publishes a closed form.  This builds the same triad
    the long way round, as the columns of ``Rz(yaw) . Ry(-pitch) . Rx(-roll)``,
    so the two derivations can be compared instead of one being assumed.
    """
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    rx = (
        (1.0, 0.0, 0.0),
        (0.0, math.cos(roll), math.sin(roll)),
        (0.0, -math.sin(roll), math.cos(roll)),
    )
    ry = (
        (math.cos(pitch), 0.0, -math.sin(pitch)),
        (0.0, 1.0, 0.0),
        (math.sin(pitch), 0.0, math.cos(pitch)),
    )
    rz = (
        (math.cos(yaw), -math.sin(yaw), 0.0),
        (math.sin(yaw), math.cos(yaw), 0.0),
        (0.0, 0.0, 1.0),
    )
    matrix = _matmul(rz, _matmul(ry, rx))
    return tuple(tuple(matrix[row][col] for row in range(3)) for col in range(3))


def reference_components(velocity, roll=0.0, pitch=0.0, yaw=0.0):
    forward, right, up = reference_basis(roll, pitch, yaw)
    return tuple(
        sum(velocity[i] * axis[i] for i in range(3)) for axis in (forward, right, up)
    )


# --------------------------------------------------------------------------
# Hostile inputs
# --------------------------------------------------------------------------


@numbers.Real.register
class DriftingReal:
    """A hostile ``numbers.Real`` whose float value changes between reads."""

    def __init__(self, *values: float) -> None:
        self._values = list(values)

    def __float__(self) -> float:
        if len(self._values) > 1:
            return float(self._values.pop(0))
        return float(self._values[0])


class SneakyFloat(float):
    """A float subclass that must be canonicalized into a built-in float."""


class ConfigSubclass(LongitudinalControllerConfig):
    """A caller-owned config subtype that must never be retained as-is."""


def unvalidated(base, **overrides):
    """Return a validation-bypassing clone of one accepted frozen record.

    Mission 10A must survive evidence that no accepted constructor would ever
    build.  Subclassing and assigning the slots directly is the only way to hand
    it such a record without weakening the upstream validators.
    """
    subclass = type(f"Unvalidated{type(base).__name__}", (type(base),), {})
    clone = object.__new__(subclass)
    for field in fields(base):
        object.__setattr__(
            clone,
            field.name,
            overrides[field.name]
            if field.name in overrides
            else getattr(base, field.name),
        )
    return clone


def bypassed(base, **overrides):
    """Return a record of the **exact** accepted type with its constructor skipped.

    ``unvalidated`` produces a subclass, which a type check could in principle
    notice.  This produces an instance whose ``type(...) is`` the accepted class
    itself, so nothing short of re-proving the upstream invariants can reject it.
    """
    clone = object.__new__(type(base))
    for field in fields(base):
        object.__setattr__(
            clone,
            field.name,
            overrides[field.name]
            if field.name in overrides
            else getattr(base, field.name),
        )
    return clone


def exploding(base, name: str, exception: BaseException):
    """Return a clone whose ``name`` attribute raises when it is read."""
    clone = unvalidated(base)

    def raiser(self):
        raise exception

    setattr(type(clone), name, property(raiser))
    return clone


# --------------------------------------------------------------------------
# Accepted-record builders
# --------------------------------------------------------------------------


def observation(
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
    *,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    frame: int = 5,
    sim_time: float | None = None,
    revision: int = 5,
    monotonic: float = BASE,
    captured: float = BASE,
    alive: bool = True,
) -> EgoSpeedObservation:
    """Build one exact observation whose components come from the reference."""
    forward, lateral, vertical = reference_components(
        (vx, vy, vz), roll, pitch, yaw
    )
    return EgoSpeedObservation(
        source_stamp=SampleStamp(
            frame=frame,
            sim_time=frame / 10.0 if sim_time is None else sim_time,
            monotonic=monotonic,
        ),
        snapshot_revision=revision,
        snapshot_captured_monotonic=captured,
        world_velocity=Vector3(vx, vy, vz),
        rotation_degrees=EulerDegrees(roll=roll, pitch=pitch, yaw=yaw),
        forward_speed_mps=forward,
        lateral_speed_mps=lateral,
        vertical_speed_mps=vertical,
        speed_mps=math.sqrt(vx * vx + vy * vy + vz * vz),
        vehicle_alive=alive,
    )


def telemetry(
    vx: float = 1.0,
    vy: float = 0.0,
    vz: float = 0.0,
    *,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    frame: int = 5,
    revision: int = 5,
    monotonic: float = BASE,
    captured: float = BASE,
    alive: bool = True,
    speed_mps: float | None = None,
    speed_kph: float | None = None,
    ego: object = ...,
) -> TelemetrySnapshot:
    """Build one Mission 4-shaped telemetry snapshot carrying ego kinematics."""
    magnitude = math.sqrt(vx * vx + vy * vy + vz * vz)
    if ego is ...:
        ego = EgoKinematics(
            stamp=SampleStamp(
                frame=frame, sim_time=frame / 10.0, monotonic=monotonic
            ),
            transform=TransformState(
                location=Vector3(0.0, 0.0, 0.0),
                rotation=EulerDegrees(roll=roll, pitch=pitch, yaw=yaw),
            ),
            velocity=Vector3(vx, vy, vz),
            speed_mps=magnitude if speed_mps is None else speed_mps,
            speed_kph=magnitude * 3.6 if speed_kph is None else speed_kph,
            vehicle_alive=alive,
        )
    return TelemetrySnapshot(
        revision=revision, captured_monotonic=captured, ego=ego
    )


def decision(
    *,
    state: SafetyState = SafetyState.NOMINAL,
    reason: SafetyReason = SafetyReason.NOMINAL_TRACKING,
    frame: int = 5,
    sim_time: float | None = None,
    revision: int = 5,
    monotonic: float = GUIDANCE_MONOTONIC,
    latched: bool = False,
    temporal: TemporalLaneState = TemporalLaneState.TRACKING,
    now: float = BASE,
    age: float | None = None,
) -> SafetyDecision:
    stamp = SampleStamp(
        frame=frame,
        sim_time=frame / 10.0 if sim_time is None else sim_time,
        monotonic=monotonic,
    )
    return SafetyDecision(
        state=state,
        reason=reason,
        autonomy_allowed=state in (SafetyState.NOMINAL, SafetyState.DEGRADED),
        evaluation_monotonic=now,
        source_stamp=stamp,
        snapshot_revision=revision,
        result_age_seconds=(now - stamp.monotonic) if age is None else age,
        temporal_state=temporal,
        temporal_confidence=0.80,
        latched=latched,
    )


def lateral(
    *,
    mode: LateralControlMode = LateralControlMode.NOMINAL,
    reason: LateralControlReason = LateralControlReason.NOMINAL_TRACKING,
    frame: int = 5,
    sim_time: float | None = None,
    revision: int = 5,
    monotonic: float = GUIDANCE_MONOTONIC,
    safety: SafetyState = SafetyState.NOMINAL,
    temporal: TemporalLaneState = TemporalLaneState.TRACKING,
    now: float = BASE,
) -> LateralControlRequest:
    stamp = SampleStamp(
        frame=frame,
        sim_time=frame / 10.0 if sim_time is None else sim_time,
        monotonic=monotonic,
    )
    allowed = mode in (LateralControlMode.NOMINAL, LateralControlMode.DEGRADED)
    kwargs = dict(
        mode=mode,
        reason=reason,
        lateral_allowed=allowed,
        evaluation_monotonic=now,
        source_stamp=stamp,
        snapshot_revision=revision,
        safety_state=safety,
        temporal_state=temporal,
    )
    if allowed:
        kwargs.update(
            steering=0.10,
            raw_steering=0.10,
            steering_authority=0.35,
            lateral_error=0.05,
            heading_error=0.0,
        )
    return LateralControlRequest(**kwargs)


def refused_lateral(
    reason: LateralControlReason,
    *,
    safety: SafetyState = SafetyState.NOMINAL,
    temporal: TemporalLaneState = TemporalLaneState.TRACKING,
    **kwargs,
) -> LateralControlRequest:
    return lateral(
        mode=LateralControlMode.DISABLED,
        reason=reason,
        safety=safety,
        temporal=temporal,
        **kwargs,
    )


def target(
    speed: float = 2.0,
    *,
    origin: TargetSpeedOrigin = TargetSpeedOrigin.TEST,
    epoch: int = 1,
    revision: int = 1,
    issued: float = BASE - 0.20,
    valid_until: float = BASE + 5.0,
) -> TargetSpeedRequest:
    return TargetSpeedRequest(
        mode=TargetSpeedMode.FORWARD if speed > 0.0 else TargetSpeedMode.STOP,
        target_speed_mps=speed,
        origin=origin,
        source_epoch=epoch,
        source_revision=revision,
        issued_monotonic=issued,
        valid_until_monotonic=valid_until,
    )


def retained_ids(controller: LongitudinalController) -> set[int]:
    """Every object id reachable from the controller's own retained slots."""
    seen: set[int] = set()
    pending = [
        getattr(controller, name)
        for name in LongitudinalController.__slots__
        if hasattr(controller, name)
    ]
    while pending:
        value = pending.pop()
        if value is None or isinstance(value, (int, float, str, bool)):
            continue
        if id(value) in seen:
            continue
        seen.add(id(value))
        for field in getattr(type(value), "__slots__", ()):
            if hasattr(value, field):
                pending.append(getattr(value, field))
    return seen


class LongitudinalTestCase(unittest.TestCase):
    """Shared assertions for the Mission 10A suite."""

    def assertNeutral(self, request, reason):
        self.assertIs(request.reason, reason)
        self.assertIs(request.mode, LongitudinalControlMode.DISABLED)
        self.assertFalse(request.longitudinal_allowed)
        self.assertFalse(request.stop_required)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, 0.0)

    def assertStop(self, request, reason, brake=None):
        self.assertIs(request.reason, reason)
        self.assertIs(request.mode, LongitudinalControlMode.STOPPING)
        self.assertFalse(request.longitudinal_allowed)
        self.assertTrue(request.stop_required)
        self.assertEqual(request.throttle_request, 0.0)
        if brake is not None:
            self.assertEqual(request.brake_request, brake)

    def drive(self, controller=None, *, speed=None, dec=None, lat=None, tgt=None,
              now=BASE):
        controller = LongitudinalController() if controller is None else controller
        return controller.update(
            decision() if dec is None else dec,
            lateral() if lat is None else lat,
            observation(1.0) if speed is None else speed,
            target() if tgt is None else tgt,
            now=now,
        )


# --------------------------------------------------------------------------
# A. Body frame and the ego speed adapter
# --------------------------------------------------------------------------


class BodyBasisTest(LongitudinalTestCase):
    def test_basis_matches_an_independent_rotation_matrix(self):
        for roll, pitch, yaw in ANGLE_SWEEP:
            produced = body_basis(roll, pitch, yaw)
            expected = reference_basis(roll, pitch, yaw)
            for axis, (made, want) in enumerate(zip(produced, expected)):
                for component, (a, b) in enumerate(zip(made, want)):
                    self.assertAlmostEqual(
                        a, b, places=12, msg=(roll, pitch, yaw, axis, component)
                    )

    def test_every_basis_axis_is_a_unit_vector(self):
        for roll, pitch, yaw in ANGLE_SWEEP:
            for axis in body_basis(roll, pitch, yaw):
                norm = math.sqrt(sum(value * value for value in axis))
                self.assertAlmostEqual(norm, 1.0, places=12, msg=(roll, pitch, yaw))

    def test_basis_axes_are_pairwise_orthogonal(self):
        for roll, pitch, yaw in ANGLE_SWEEP:
            forward, right, up = body_basis(roll, pitch, yaw)
            for first, second in ((forward, right), (forward, up), (right, up)):
                dot = sum(a * b for a, b in zip(first, second))
                self.assertAlmostEqual(dot, 0.0, places=12, msg=(roll, pitch, yaw))

    def test_zero_rotation_is_the_identity_basis(self):
        forward, right, up = body_basis(0.0, 0.0, 0.0)
        self.assertEqual(forward, (1.0, 0.0, 0.0))
        self.assertEqual(right[1], 1.0)
        self.assertEqual(up[2], 1.0)

    def test_golden_yaw_cases_point_the_documented_way(self):
        cases = {
            0.0: ((1.0, 0.0), (0.0, 1.0)),
            90.0: ((0.0, 1.0), (-1.0, 0.0)),
            180.0: ((-1.0, 0.0), (0.0, -1.0)),
            -90.0: ((0.0, -1.0), (1.0, 0.0)),
        }
        for yaw, ((fx, fy), (rx, ry)) in cases.items():
            forward, right, _ = body_basis(0.0, 0.0, yaw)
            self.assertAlmostEqual(forward[0], fx, places=12, msg=yaw)
            self.assertAlmostEqual(forward[1], fy, places=12, msg=yaw)
            self.assertAlmostEqual(right[0], rx, places=12, msg=yaw)
            self.assertAlmostEqual(right[1], ry, places=12, msg=yaw)

    def test_positive_pitch_lifts_the_forward_axis(self):
        forward, _, up = body_basis(0.0, 90.0, 0.0)
        self.assertAlmostEqual(forward[2], 1.0, places=12)
        self.assertAlmostEqual(up[0], -1.0, places=12)

    def test_positive_roll_rotates_right_toward_world_down(self):
        _, right, up = body_basis(90.0, 0.0, 0.0)
        self.assertAlmostEqual(right[2], -1.0, places=12)
        self.assertAlmostEqual(up[1], 1.0, places=12)


class EgoSpeedAdapterTest(LongitudinalTestCase):
    def test_absent_ego_returns_none(self):
        snapshot = TelemetrySnapshot(revision=3, captured_monotonic=BASE)
        self.assertIsNone(EgoSpeedObservation.from_telemetry_snapshot(snapshot))

    def test_a_dead_ego_is_observed_rather_than_discarded(self):
        result = EgoSpeedObservation.from_telemetry_snapshot(
            telemetry(1.0, alive=False)
        )
        self.assertIsNotNone(result)
        self.assertFalse(result.vehicle_alive)
        self.assertEqual(result.forward_speed_mps, 1.0)

    def test_forward_velocity_is_a_positive_forward_speed(self):
        result = EgoSpeedObservation.from_telemetry_snapshot(telemetry(3.0))
        self.assertEqual(result.forward_speed_mps, 3.0)
        self.assertEqual(result.lateral_speed_mps, 0.0)
        self.assertEqual(result.vertical_speed_mps, 0.0)
        self.assertEqual(result.speed_mps, 3.0)

    def test_reverse_velocity_is_a_negative_forward_speed(self):
        result = EgoSpeedObservation.from_telemetry_snapshot(telemetry(-3.0))
        self.assertEqual(result.forward_speed_mps, -3.0)
        self.assertEqual(result.speed_mps, 3.0)

    def test_lateral_only_velocity_has_no_forward_component(self):
        result = EgoSpeedObservation.from_telemetry_snapshot(telemetry(0.0, 2.0, 0.0))
        self.assertEqual(result.forward_speed_mps, 0.0)
        self.assertEqual(result.lateral_speed_mps, 2.0)
        self.assertEqual(result.speed_mps, 2.0)

    def test_vertical_only_velocity_has_no_forward_component(self):
        result = EgoSpeedObservation.from_telemetry_snapshot(telemetry(0.0, 0.0, 2.0))
        self.assertEqual(result.forward_speed_mps, 0.0)
        self.assertEqual(result.lateral_speed_mps, 0.0)
        self.assertEqual(result.vertical_speed_mps, 2.0)

    def test_mixed_velocity_decomposes_onto_every_axis(self):
        result = EgoSpeedObservation.from_telemetry_snapshot(
            telemetry(1.0, 2.0, 3.0, roll=12.0, pitch=-7.0, yaw=143.0)
        )
        forward, lateral_speed, vertical = reference_components(
            (1.0, 2.0, 3.0), 12.0, -7.0, 143.0
        )
        self.assertAlmostEqual(result.forward_speed_mps, forward, places=12)
        self.assertAlmostEqual(result.lateral_speed_mps, lateral_speed, places=12)
        self.assertAlmostEqual(result.vertical_speed_mps, vertical, places=12)

    def test_yaw_cases_project_world_motion_correctly(self):
        cases = {0.0: 4.0, 90.0: 0.0, 180.0: -4.0, -90.0: 0.0}
        for yaw, expected in cases.items():
            result = EgoSpeedObservation.from_telemetry_snapshot(
                telemetry(4.0, 0.0, 0.0, yaw=yaw)
            )
            self.assertAlmostEqual(result.forward_speed_mps, expected, places=12, msg=yaw)

    def test_yaw_ninety_makes_world_y_the_forward_axis(self):
        result = EgoSpeedObservation.from_telemetry_snapshot(
            telemetry(0.0, 4.0, 0.0, yaw=90.0)
        )
        self.assertAlmostEqual(result.forward_speed_mps, 4.0, places=12)
        self.assertAlmostEqual(result.lateral_speed_mps, 0.0, places=12)

    def test_nonzero_pitch_projects_world_climb_onto_forward(self):
        result = EgoSpeedObservation.from_telemetry_snapshot(
            telemetry(0.0, 0.0, 5.0, pitch=90.0)
        )
        self.assertAlmostEqual(result.forward_speed_mps, 5.0, places=12)

    def test_nonzero_roll_moves_world_z_onto_the_lateral_axis(self):
        result = EgoSpeedObservation.from_telemetry_snapshot(
            telemetry(0.0, 0.0, 5.0, roll=90.0)
        )
        self.assertAlmostEqual(result.lateral_speed_mps, -5.0, places=12)
        self.assertAlmostEqual(result.vertical_speed_mps, 0.0, places=12)

    def test_mirrored_velocity_mirrors_every_signed_component(self):
        forward_case = EgoSpeedObservation.from_telemetry_snapshot(
            telemetry(1.0, 2.0, 3.0, roll=12.0, pitch=-7.0, yaw=143.0)
        )
        mirrored = EgoSpeedObservation.from_telemetry_snapshot(
            telemetry(-1.0, -2.0, -3.0, roll=12.0, pitch=-7.0, yaw=143.0)
        )
        self.assertAlmostEqual(
            mirrored.forward_speed_mps, -forward_case.forward_speed_mps, places=12
        )
        self.assertAlmostEqual(
            mirrored.lateral_speed_mps, -forward_case.lateral_speed_mps, places=12
        )
        self.assertAlmostEqual(
            mirrored.vertical_speed_mps, -forward_case.vertical_speed_mps, places=12
        )
        self.assertAlmostEqual(mirrored.speed_mps, forward_case.speed_mps, places=12)

    def test_components_reconstruct_the_world_velocity(self):
        for roll, pitch, yaw in ANGLE_SWEEP:
            result = EgoSpeedObservation.from_telemetry_snapshot(
                telemetry(1.5, -2.5, 0.75, roll=roll, pitch=pitch, yaw=yaw)
            )
            forward, right, up = reference_basis(roll, pitch, yaw)
            for index, expected in enumerate((1.5, -2.5, 0.75)):
                rebuilt = (
                    result.forward_speed_mps * forward[index]
                    + result.lateral_speed_mps * right[index]
                    + result.vertical_speed_mps * up[index]
                )
                self.assertAlmostEqual(rebuilt, expected, places=11)

    def test_total_speed_is_the_unsigned_magnitude(self):
        result = EgoSpeedObservation.from_telemetry_snapshot(telemetry(3.0, 4.0, 0.0))
        self.assertEqual(result.speed_mps, 5.0)
        self.assertGreater(result.speed_mps, 0.0)

    def test_disagreeing_source_speed_is_rejected(self):
        with self.assertRaises(ValueError):
            EgoSpeedObservation.from_telemetry_snapshot(telemetry(3.0, speed_mps=9.0))

    def test_disagreeing_source_kph_is_rejected(self):
        with self.assertRaises(ValueError):
            EgoSpeedObservation.from_telemetry_snapshot(telemetry(3.0, speed_kph=9.0))

    def test_non_finite_velocity_is_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError, msg=bad):
                EgoSpeedObservation.from_telemetry_snapshot(
                    telemetry(bad, speed_mps=1.0, speed_kph=3.6)
                )

    def test_non_finite_rotation_is_rejected(self):
        with self.assertRaises(ValueError):
            EgoSpeedObservation.from_telemetry_snapshot(telemetry(1.0, yaw=float("nan")))

    def test_bool_velocity_component_is_rejected(self):
        snapshot = telemetry(1.0)
        forged = unvalidated(snapshot.ego, velocity=Vector3(True, 0.0, 0.0))
        with self.assertRaises(ValueError):
            EgoSpeedObservation.from_telemetry_snapshot(telemetry(ego=forged))

    def test_vehicle_alive_must_be_an_exact_builtin_bool(self):
        snapshot = telemetry(1.0)
        forged = unvalidated(snapshot.ego, vehicle_alive=1)
        with self.assertRaises(ValueError):
            EgoSpeedObservation.from_telemetry_snapshot(telemetry(ego=forged))

    def test_negative_revision_is_rejected(self):
        with self.assertRaises(ValueError):
            EgoSpeedObservation.from_telemetry_snapshot(telemetry(1.0, revision=-1))

    def test_bool_revision_is_rejected(self):
        with self.assertRaises(ValueError):
            EgoSpeedObservation.from_telemetry_snapshot(telemetry(1.0, revision=True))

    def test_float_subclass_is_canonicalized_to_a_builtin_float(self):
        result = EgoSpeedObservation.from_telemetry_snapshot(
            telemetry(SneakyFloat(2.0), speed_mps=SneakyFloat(2.0))
        )
        self.assertIs(type(result.forward_speed_mps), float)
        self.assertIs(type(result.world_velocity.x), float)
        self.assertIs(type(result.speed_mps), float)

    def test_a_drifting_number_cannot_change_between_reads(self):
        snapshot = telemetry(1.0)
        forged = unvalidated(
            snapshot.ego, velocity=Vector3(DriftingReal(1.0, 90.0), 0.0, 0.0)
        )
        result = EgoSpeedObservation.from_telemetry_snapshot(telemetry(ego=forged))
        self.assertEqual(result.world_velocity.x, 1.0)
        self.assertEqual(result.forward_speed_mps, 1.0)
        self.assertEqual(result.speed_mps, 1.0)

    def test_a_hostile_property_becomes_a_value_error(self):
        snapshot = telemetry(1.0)
        forged = exploding(snapshot.ego, "velocity", RuntimeError("boom"))
        with self.assertRaises(ValueError):
            EgoSpeedObservation.from_telemetry_snapshot(telemetry(ego=forged))

    def test_a_hostile_snapshot_property_becomes_a_value_error(self):
        snapshot = telemetry(1.0)
        forged = exploding(snapshot, "ego", RuntimeError("boom"))
        with self.assertRaises(ValueError):
            EgoSpeedObservation.from_telemetry_snapshot(forged)

    def test_a_missing_transform_is_rejected(self):
        snapshot = telemetry(1.0)
        forged = unvalidated(snapshot.ego, transform=None)
        with self.assertRaises(ValueError):
            EgoSpeedObservation.from_telemetry_snapshot(telemetry(ego=forged))

    def test_a_non_snapshot_is_rejected(self):
        with self.assertRaises(ValueError):
            EgoSpeedObservation.from_telemetry_snapshot(object())

    def test_no_source_object_is_retained(self):
        snapshot = telemetry(1.0)
        result = EgoSpeedObservation.from_telemetry_snapshot(snapshot)
        self.assertIsNot(result.source_stamp, snapshot.ego.stamp)
        self.assertIsNot(result.world_velocity, snapshot.ego.velocity)
        self.assertIsNot(
            result.rotation_degrees, snapshot.ego.transform.rotation
        )

    def test_negative_zero_is_canonicalized_to_positive_zero(self):
        result = EgoSpeedObservation.from_telemetry_snapshot(
            telemetry(-0.0, -0.0, -0.0)
        )
        self.assertEqual(math.copysign(1.0, result.forward_speed_mps), 1.0)
        self.assertEqual(math.copysign(1.0, result.world_velocity.x), 1.0)

    def test_the_observation_is_frozen_and_slotted(self):
        result = observation(1.0)
        with self.assertRaises(FrozenInstanceError):
            result.forward_speed_mps = 5.0
        self.assertFalse(hasattr(result, "__dict__"))

    def test_direct_construction_rejects_an_unsupported_projection(self):
        good = observation(1.0)
        with self.assertRaises(ValueError):
            EgoSpeedObservation(
                source_stamp=good.source_stamp,
                snapshot_revision=good.snapshot_revision,
                snapshot_captured_monotonic=good.snapshot_captured_monotonic,
                world_velocity=good.world_velocity,
                rotation_degrees=good.rotation_degrees,
                forward_speed_mps=99.0,
                lateral_speed_mps=good.lateral_speed_mps,
                vertical_speed_mps=good.vertical_speed_mps,
                speed_mps=good.speed_mps,
                vehicle_alive=True,
            )

    def test_direct_construction_rejects_a_speed_that_is_not_the_magnitude(self):
        good = observation(1.0)
        with self.assertRaises(ValueError):
            EgoSpeedObservation(
                source_stamp=good.source_stamp,
                snapshot_revision=good.snapshot_revision,
                snapshot_captured_monotonic=good.snapshot_captured_monotonic,
                world_velocity=good.world_velocity,
                rotation_degrees=good.rotation_degrees,
                forward_speed_mps=good.forward_speed_mps,
                lateral_speed_mps=good.lateral_speed_mps,
                vertical_speed_mps=good.vertical_speed_mps,
                speed_mps=7.0,
                vehicle_alive=True,
            )


# --------------------------------------------------------------------------
# B. Target-speed contract
# --------------------------------------------------------------------------


class TargetSpeedContractTest(LongitudinalTestCase):
    def test_forward_mode_requires_a_positive_speed(self):
        with self.assertRaises(ValueError):
            TargetSpeedRequest(
                mode=TargetSpeedMode.FORWARD,
                target_speed_mps=0.0,
                origin=TargetSpeedOrigin.OPERATOR,
                source_epoch=0,
                source_revision=0,
                issued_monotonic=1.0,
                valid_until_monotonic=2.0,
            )

    def test_stop_mode_requires_exactly_zero(self):
        with self.assertRaises(ValueError):
            TargetSpeedRequest(
                mode=TargetSpeedMode.STOP,
                target_speed_mps=0.5,
                origin=TargetSpeedOrigin.OPERATOR,
                source_epoch=0,
                source_revision=0,
                issued_monotonic=1.0,
                valid_until_monotonic=2.0,
            )

    def test_stop_and_forward_round_trip(self):
        self.assertIs(target(0.0).mode, TargetSpeedMode.STOP)
        self.assertIs(target(1.0).mode, TargetSpeedMode.FORWARD)

    def test_negative_target_is_rejected(self):
        with self.assertRaises(ValueError):
            target(-1.0)

    def test_target_above_the_hard_ceiling_is_rejected(self):
        with self.assertRaises(ValueError):
            target(HARD_TARGET_CEILING_MPS + 0.001)

    def test_target_exactly_at_the_ceiling_is_accepted(self):
        self.assertEqual(
            target(HARD_TARGET_CEILING_MPS).target_speed_mps,
            HARD_TARGET_CEILING_MPS,
        )

    def test_non_finite_target_is_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError, msg=bad):
                target(bad)

    def test_bool_target_is_rejected(self):
        with self.assertRaises(ValueError):
            target(True)

    def test_bool_epoch_or_revision_is_rejected(self):
        with self.assertRaises(ValueError):
            target(1.0, epoch=True)
        with self.assertRaises(ValueError):
            target(1.0, revision=True)

    def test_negative_epoch_or_revision_is_rejected(self):
        with self.assertRaises(ValueError):
            target(1.0, epoch=-1)
        with self.assertRaises(ValueError):
            target(1.0, revision=-1)

    def test_issue_after_expiry_is_rejected(self):
        with self.assertRaises(ValueError):
            target(1.0, issued=10.0, valid_until=9.0)

    def test_equal_issue_and_expiry_is_accepted(self):
        request = target(1.0, issued=10.0, valid_until=10.0)
        self.assertEqual(request.issued_monotonic, request.valid_until_monotonic)

    def test_hostile_numbers_are_canonicalized_to_builtin_floats(self):
        request = target(SneakyFloat(2.0), issued=SneakyFloat(1.0))
        self.assertIs(type(request.target_speed_mps), float)
        self.assertIs(type(request.issued_monotonic), float)

    def test_a_drifting_target_speed_is_read_exactly_once(self):
        request = TargetSpeedRequest(
            mode=TargetSpeedMode.FORWARD,
            target_speed_mps=DriftingReal(2.0, 999.0),
            origin=TargetSpeedOrigin.TEST,
            source_epoch=0,
            source_revision=0,
            issued_monotonic=1.0,
            valid_until_monotonic=2.0,
        )
        self.assertEqual(request.target_speed_mps, 2.0)
        self.assertEqual(request.target_speed_mps, 2.0)

    def test_enum_members_must_be_the_exact_accepted_types(self):
        with self.assertRaises(ValueError):
            TargetSpeedRequest(
                mode="forward",
                target_speed_mps=1.0,
                origin=TargetSpeedOrigin.TEST,
                source_epoch=0,
                source_revision=0,
                issued_monotonic=1.0,
                valid_until_monotonic=2.0,
            )
        with self.assertRaises(ValueError):
            TargetSpeedRequest(
                mode=TargetSpeedMode.FORWARD,
                target_speed_mps=1.0,
                origin="test",
                source_epoch=0,
                source_revision=0,
                issued_monotonic=1.0,
                valid_until_monotonic=2.0,
            )

    def test_the_target_is_frozen_and_slotted(self):
        request = target(1.0)
        with self.assertRaises(FrozenInstanceError):
            request.target_speed_mps = 5.0
        self.assertFalse(hasattr(request, "__dict__"))

    def test_negative_zero_target_is_canonicalized(self):
        request = TargetSpeedRequest(
            mode=TargetSpeedMode.STOP,
            target_speed_mps=-0.0,
            origin=TargetSpeedOrigin.TEST,
            source_epoch=0,
            source_revision=0,
            issued_monotonic=1.0,
            valid_until_monotonic=2.0,
        )
        self.assertEqual(math.copysign(1.0, request.target_speed_mps), 1.0)


# --------------------------------------------------------------------------
# C. Configuration
# --------------------------------------------------------------------------


class ConfigurationTest(LongitudinalTestCase):
    def test_documented_defaults(self):
        self.assertEqual(CONFIG.nominal_speed_cap_mps, 3.0)
        self.assertEqual(CONFIG.degraded_speed_cap_mps, 1.5)
        self.assertEqual(CONFIG.coasting_speed_cap_mps, 1.0)
        self.assertEqual(CONFIG.nominal_throttle_authority, 0.35)
        self.assertEqual(CONFIG.degraded_throttle_authority, 0.15)
        self.assertEqual(CONFIG.coasting_throttle_authority, 0.10)
        self.assertEqual(CONFIG.max_service_brake, 0.40)
        self.assertEqual(CONFIG.speed_deadband_mps, 0.10)
        self.assertEqual(CONFIG.throttle_kp_per_mps, 0.30)
        self.assertEqual(CONFIG.brake_kp_per_mps, 0.40)
        self.assertEqual(CONFIG.stop_threshold_mps, 0.10)
        self.assertEqual(CONFIG.hard_target_ceiling_mps, HARD_TARGET_CEILING_MPS)

    def test_speed_cap_ordering_is_enforced(self):
        with self.assertRaises(ValueError):
            LongitudinalControllerConfig(coasting_speed_cap_mps=2.0)
        with self.assertRaises(ValueError):
            LongitudinalControllerConfig(nominal_speed_cap_mps=1.0)
        with self.assertRaises(ValueError):
            LongitudinalControllerConfig(coasting_speed_cap_mps=0.0)

    def test_throttle_authority_ordering_is_enforced(self):
        with self.assertRaises(ValueError):
            LongitudinalControllerConfig(coasting_throttle_authority=0.20)
        with self.assertRaises(ValueError):
            LongitudinalControllerConfig(nominal_throttle_authority=1.5)
        with self.assertRaises(ValueError):
            LongitudinalControllerConfig(coasting_throttle_authority=0.0)

    def test_brakes_are_bounded_by_the_service_brake(self):
        with self.assertRaises(ValueError):
            LongitudinalControllerConfig(latched_stop_brake=0.9)
        with self.assertRaises(ValueError):
            LongitudinalControllerConfig(max_service_brake=1.5)
        with self.assertRaises(ValueError):
            LongitudinalControllerConfig(soft_stop_brake=-0.1)

    def test_motion_threshold_ordering_is_enforced(self):
        with self.assertRaises(ValueError):
            LongitudinalControllerConfig(reverse_motion_threshold_mps=0.5)
        with self.assertRaises(ValueError):
            LongitudinalControllerConfig(component_zero_epsilon_mps=0.0)
        with self.assertRaises(ValueError):
            LongitudinalControllerConfig(stop_threshold_mps=0.01)

    def test_non_finite_and_bool_configuration_is_rejected(self):
        for bad in (float("nan"), float("inf"), True):
            with self.assertRaises(ValueError, msg=bad):
                LongitudinalControllerConfig(speed_deadband_mps=bad)

    def test_gains_must_be_strictly_positive(self):
        with self.assertRaises(ValueError):
            LongitudinalControllerConfig(throttle_kp_per_mps=0.0)
        with self.assertRaises(ValueError):
            LongitudinalControllerConfig(brake_kp_per_mps=0.0)

    def test_ceiling_above_the_hard_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            LongitudinalControllerConfig(
                hard_target_ceiling_mps=HARD_TARGET_CEILING_MPS + 1.0
            )

    def test_the_controller_does_not_retain_a_caller_config(self):
        supplied = ConfigSubclass()
        controller = LongitudinalController(supplied)
        self.assertIsNot(controller.config, supplied)
        self.assertIs(type(controller.config), LongitudinalControllerConfig)
        self.assertNotIn(id(supplied), retained_ids(controller))

    def test_a_non_config_is_rejected(self):
        with self.assertRaises(TypeError):
            LongitudinalController(object())

    def test_the_config_is_frozen_and_slotted(self):
        with self.assertRaises(FrozenInstanceError):
            CONFIG.max_service_brake = 1.0
        self.assertFalse(hasattr(CONFIG, "__dict__"))


# --------------------------------------------------------------------------
# D. Request invariants
# --------------------------------------------------------------------------


class RequestInvariantTest(LongitudinalTestCase):
    def neutral(self, **overrides):
        base = dict(
            mode=LongitudinalControlMode.DISABLED,
            reason=LongitudinalControlReason.IDENTITY_MISMATCH,
            longitudinal_allowed=False,
        )
        base.update(overrides)
        return LongitudinalControlRequest(**base)

    def test_every_reason_maps_to_exactly_one_mode(self):
        self.assertEqual(set(_REASON_MODE), set(LongitudinalControlReason))
        for reason, mode in _REASON_MODE.items():
            self.assertIsInstance(mode, LongitudinalControlMode)

    def test_mode_must_match_its_reason(self):
        with self.assertRaises(ValueError):
            LongitudinalControlRequest(
                mode=LongitudinalControlMode.NOMINAL,
                reason=LongitudinalControlReason.IDENTITY_MISMATCH,
                longitudinal_allowed=True,
            )

    def test_permission_must_match_the_mode(self):
        with self.assertRaises(ValueError):
            self.neutral(longitudinal_allowed=True)

    def test_stop_required_must_match_the_mode(self):
        with self.assertRaises(ValueError):
            self.neutral(stop_required=True)

    def test_a_disabled_request_cannot_brake(self):
        with self.assertRaises(ValueError):
            self.neutral(brake_request=0.2, brake_authority=0.4)

    def test_a_disabled_request_cannot_throttle(self):
        with self.assertRaises(ValueError):
            self.neutral(throttle_request=0.2, throttle_authority=0.4)

    def test_a_stopping_request_cannot_throttle(self):
        with self.assertRaises(ValueError):
            LongitudinalControlRequest(
                mode=LongitudinalControlMode.STOPPING,
                reason=LongitudinalControlReason.SAFETY_STOP,
                longitudinal_allowed=False,
                stop_required=True,
                throttle_request=0.1,
                throttle_authority=0.4,
            )

    def test_throttle_and_brake_can_never_both_be_positive(self):
        with self.assertRaises(ValueError):
            LongitudinalControlRequest(
                mode=LongitudinalControlMode.NOMINAL,
                reason=LongitudinalControlReason.NOMINAL_TRACKING,
                longitudinal_allowed=True,
                throttle_request=0.1,
                brake_request=0.1,
                throttle_authority=0.35,
                brake_authority=0.40,
                effective_target_speed_mps=1.0,
                observed_forward_speed_mps=0.0,
                speed_error_mps=1.0,
                speed_authority_mps=3.0,
                **self.binding(),
            )

    def binding(self):
        return dict(
            evaluation_monotonic=BASE,
            guidance_stamp=SampleStamp(frame=5, sim_time=0.5, monotonic=BASE),
            speed_stamp=SampleStamp(frame=5, sim_time=0.5, monotonic=BASE),
            snapshot_revision=5,
            target_origin=TargetSpeedOrigin.TEST,
            target_epoch=1,
            target_revision=1,
            target_issued_monotonic=BASE - 1.0,
            target_valid_until_monotonic=BASE + 1.0,
            safety_state=SafetyState.NOMINAL,
            lateral_mode=LateralControlMode.NOMINAL,
            temporal_state=TemporalLaneState.TRACKING,
        )

    def permitted(self, **overrides):
        base = dict(
            mode=LongitudinalControlMode.NOMINAL,
            reason=LongitudinalControlReason.NOMINAL_TRACKING,
            longitudinal_allowed=True,
            throttle_request=0.1,
            throttle_authority=0.35,
            brake_authority=0.40,
            effective_target_speed_mps=1.0,
            observed_forward_speed_mps=0.0,
            speed_error_mps=1.0,
            speed_authority_mps=3.0,
            **self.binding(),
        )
        base.update(overrides)
        return LongitudinalControlRequest(**base)

    def test_a_valid_permitted_request_is_accepted(self):
        request = self.permitted()
        self.assertTrue(request.longitudinal_allowed)
        self.assertEqual(request.throttle_request, 0.1)

    def test_throttle_cannot_exceed_its_authority(self):
        with self.assertRaises(ValueError):
            self.permitted(throttle_request=0.5)

    def test_brake_cannot_exceed_its_authority(self):
        with self.assertRaises(ValueError):
            self.permitted(throttle_request=0.0, brake_request=0.9, brake_authority=0.4)

    def test_requests_must_lie_in_the_unit_interval(self):
        with self.assertRaises(ValueError):
            self.permitted(throttle_request=-0.1)
        with self.assertRaises(ValueError):
            self.permitted(throttle_request=1.5, throttle_authority=1.0)

    def test_nominal_authority_requires_fully_nominal_evidence(self):
        with self.assertRaises(ValueError):
            self.permitted(safety_state=SafetyState.DEGRADED)
        with self.assertRaises(ValueError):
            self.permitted(lateral_mode=LateralControlMode.DEGRADED)
        with self.assertRaises(ValueError):
            self.permitted(temporal_state=TemporalLaneState.COASTING)

    def test_a_permitted_request_requires_complete_law_evidence(self):
        with self.assertRaises(ValueError):
            self.permitted(speed_error_mps=None)
        with self.assertRaises(ValueError):
            self.permitted(effective_target_speed_mps=None)

    def test_a_braking_request_requires_complete_identity(self):
        with self.assertRaises(ValueError):
            LongitudinalControlRequest(
                mode=LongitudinalControlMode.STOPPING,
                reason=LongitudinalControlReason.SAFETY_STOP,
                longitudinal_allowed=False,
                stop_required=True,
                brake_request=0.2,
                brake_authority=0.4,
            )

    def test_an_unevaluated_request_cannot_expose_evidence(self):
        with self.assertRaises(ValueError):
            LongitudinalControlRequest(
                mode=LongitudinalControlMode.DISABLED,
                reason=LongitudinalControlReason.NOT_EVALUATED,
                longitudinal_allowed=False,
                evaluation_monotonic=BASE,
            )

    def test_non_finite_values_are_rejected(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=bad):
                self.permitted(observed_forward_speed_mps=bad)

    def test_bool_as_number_is_rejected(self):
        with self.assertRaises(ValueError):
            self.permitted(throttle_request=True)

    def test_effective_target_cannot_exceed_its_speed_authority(self):
        with self.assertRaises(ValueError):
            self.permitted(effective_target_speed_mps=5.0, speed_authority_mps=3.0)

    def test_a_refused_request_cannot_claim_throttle_clamping(self):
        with self.assertRaises(ValueError):
            self.neutral(throttle_clamped=True)

    def test_the_request_is_frozen_and_slotted(self):
        request = self.neutral()
        with self.assertRaises(FrozenInstanceError):
            request.throttle_request = 1.0
        self.assertFalse(hasattr(request, "__dict__"))

    def test_convenience_frames_follow_the_bound_stamps(self):
        request = self.permitted()
        self.assertEqual(request.guidance_frame, 5)
        self.assertEqual(request.speed_frame, 5)
        self.assertIsNone(self.neutral().guidance_frame)


# --------------------------------------------------------------------------
# E. The proportional control law
# --------------------------------------------------------------------------


class ControlLawTest(LongitudinalTestCase):
    def test_below_target_requests_bounded_throttle(self):
        request = self.drive(speed=observation(1.0), tgt=target(2.0))
        self.assertTrue(request.longitudinal_allowed)
        self.assertIs(request.reason, LongitudinalControlReason.NOMINAL_TRACKING)
        self.assertEqual(request.speed_error_mps, 1.0)
        self.assertAlmostEqual(
            request.throttle_request,
            CONFIG.throttle_kp_per_mps * (1.0 - CONFIG.speed_deadband_mps),
            places=12,
        )
        self.assertEqual(request.brake_request, 0.0)

    def test_target_reached_requests_neither_pedal(self):
        request = self.drive(speed=observation(2.0), tgt=target(2.0))
        self.assertTrue(request.longitudinal_allowed)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, 0.0)
        self.assertEqual(request.speed_error_mps, 0.0)

    def test_forward_overspeed_requests_brake_only(self):
        request = self.drive(speed=observation(2.5), tgt=target(2.0))
        self.assertTrue(request.longitudinal_allowed)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertAlmostEqual(
            request.brake_request,
            CONFIG.brake_kp_per_mps * (0.5 - CONFIG.speed_deadband_mps),
            places=12,
        )

    def test_target_zero_while_moving_requests_a_stop(self):
        request = self.drive(speed=observation(1.0), tgt=target(0.0))
        self.assertStop(request, LongitudinalControlReason.TARGET_STOP)
        self.assertEqual(
            request.brake_request,
            min(CONFIG.max_service_brake, max(CONFIG.minimum_moving_stop_brake, 0.4)),
        )
        self.assertEqual(request.effective_target_speed_mps, 0.0)

    def test_target_zero_at_rest_holds_with_the_zero_hold_brake(self):
        request = self.drive(speed=observation(0.0), tgt=target(0.0))
        self.assertStop(
            request, LongitudinalControlReason.TARGET_STOP, CONFIG.zero_hold_brake
        )

    def test_target_zero_barely_moving_uses_the_minimum_moving_brake(self):
        request = self.drive(speed=observation(0.2), tgt=target(0.0))
        self.assertStop(
            request,
            LongitudinalControlReason.TARGET_STOP,
            CONFIG.minimum_moving_stop_brake,
        )

    def test_reverse_beyond_the_threshold_uses_the_reverse_brake(self):
        request = self.drive(speed=observation(-1.0), tgt=target(2.0))
        self.assertStop(
            request, LongitudinalControlReason.REVERSE_MOTION, CONFIG.reverse_stop_brake
        )

    def test_reverse_within_the_threshold_only_holds(self):
        request = self.drive(speed=observation(-0.01), tgt=target(2.0))
        self.assertStop(
            request, LongitudinalControlReason.REVERSE_MOTION, CONFIG.zero_hold_brake
        )

    def test_a_tiny_negative_projection_is_canonicalized_to_zero(self):
        request = self.drive(speed=observation(-1e-12), tgt=target(2.0))
        self.assertTrue(request.longitudinal_allowed)
        self.assertEqual(request.observed_forward_speed_mps, 0.0)
        self.assertGreater(request.throttle_request, 0.0)

    def test_rolling_backward_never_requests_throttle(self):
        for backward in (-1e-6, -0.02, -0.05, -0.5, -12.0):
            request = self.drive(speed=observation(backward), tgt=target(3.0))
            self.assertEqual(request.throttle_request, 0.0, msg=backward)
            self.assertFalse(request.longitudinal_allowed, msg=backward)

    def test_excessive_lateral_motion_requests_a_controlled_stop(self):
        request = self.drive(speed=observation(1.0, 2.0, 0.0), tgt=target(2.0))
        self.assertStop(
            request,
            LongitudinalControlReason.EXCESSIVE_LATERAL_MOTION,
            CONFIG.soft_stop_brake,
        )

    def test_excessive_vertical_motion_refuses_neutrally(self):
        request = self.drive(speed=observation(1.0, 0.0, 2.0), tgt=target(2.0))
        self.assertNeutral(
            request, LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION
        )

    def test_vertical_precedes_lateral_when_both_are_exceeded(self):
        request = self.drive(speed=observation(1.0, 2.0, 2.0), tgt=target(2.0))
        self.assertNeutral(
            request, LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION
        )

    def test_the_law_uses_the_forward_component_not_the_total_speed(self):
        # Total speed here is 5.0, far above the 2.0 target, but the *forward*
        # speed is only 1.0, so the correct answer is throttle, not brake.
        config = LongitudinalControllerConfig(lateral_speed_limit_mps=10.0)
        controller = LongitudinalController(config)
        speed = observation(1.0, 4.898979485566356, 0.0)
        self.assertAlmostEqual(speed.speed_mps, 5.0, places=9)
        request = self.drive(controller, speed=speed, tgt=target(2.0))
        self.assertTrue(request.longitudinal_allowed)
        self.assertEqual(request.observed_forward_speed_mps, 1.0)
        self.assertGreater(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, 0.0)

    def test_a_rotated_vehicle_uses_its_own_forward_axis(self):
        # Moving along world +y at 1 m/s while yawed 90 degrees is pure forward
        # motion for this vehicle, even though world x velocity is zero.
        speed = observation(0.0, 1.0, 0.0, yaw=90.0)
        request = self.drive(speed=speed, tgt=target(2.0))
        self.assertAlmostEqual(request.observed_forward_speed_mps, 1.0, places=12)
        self.assertGreater(request.throttle_request, 0.0)

    def test_the_throttle_deadband_boundary_is_exclusive(self):
        config = LongitudinalControllerConfig(speed_deadband_mps=0.25)
        controller = LongitudinalController(config)
        # 2.0 - 1.75 is exactly 0.25 in binary floating point.
        request = self.drive(controller, speed=observation(1.75), tgt=target(2.0))
        self.assertEqual(request.speed_error_mps, 0.25)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, 0.0)

    def test_one_ulp_past_the_deadband_produces_throttle(self):
        config = LongitudinalControllerConfig(speed_deadband_mps=0.25)
        controller = LongitudinalController(config)
        forward = math.nextafter(1.75, 0.0)
        request = self.drive(controller, speed=observation(forward), tgt=target(2.0))
        self.assertGreater(request.speed_error_mps, 0.25)
        self.assertGreater(request.throttle_request, 0.0)

    def test_the_brake_deadband_boundary_is_exclusive(self):
        config = LongitudinalControllerConfig(speed_deadband_mps=0.25)
        controller = LongitudinalController(config)
        request = self.drive(controller, speed=observation(2.25), tgt=target(2.0))
        self.assertEqual(request.speed_error_mps, -0.25)
        self.assertEqual(request.brake_request, 0.0)

    def test_one_ulp_past_the_brake_deadband_produces_brake(self):
        config = LongitudinalControllerConfig(speed_deadband_mps=0.25)
        controller = LongitudinalController(config)
        forward = math.nextafter(2.25, 3.0)
        request = self.drive(controller, speed=observation(forward), tgt=target(2.0))
        self.assertLess(request.speed_error_mps, -0.25)
        self.assertGreater(request.brake_request, 0.0)

    def test_throttle_clamps_to_the_mode_authority(self):
        request = self.drive(speed=observation(0.0), tgt=target(30.0))
        self.assertEqual(request.throttle_request, CONFIG.nominal_throttle_authority)
        self.assertTrue(request.throttle_clamped)
        self.assertEqual(request.effective_target_speed_mps, CONFIG.nominal_speed_cap_mps)

    def test_brake_clamps_to_the_service_brake(self):
        request = self.drive(speed=observation(20.0), tgt=target(2.0))
        self.assertEqual(request.brake_request, CONFIG.max_service_brake)
        self.assertTrue(request.brake_clamped)

    def test_throttle_and_brake_stay_mutually_exclusive_across_a_sweep(self):
        for hundredths in range(-300, 301, 7):
            forward = hundredths / 100.0
            request = self.drive(speed=observation(forward), tgt=target(2.0))
            self.assertFalse(
                request.throttle_request > 0.0 and request.brake_request > 0.0,
                msg=forward,
            )
            self.assertLessEqual(request.throttle_request, request.throttle_authority)
            self.assertLessEqual(request.brake_request, CONFIG.max_service_brake)

    def test_degraded_evidence_receives_degraded_authority(self):
        request = self.drive(
            dec=decision(state=SafetyState.DEGRADED,
                         reason=SafetyReason.DEGRADED_LOW_CONFIDENCE),
            lat=lateral(mode=LateralControlMode.DEGRADED,
                        reason=LateralControlReason.DEGRADED_TRACKING,
                        safety=SafetyState.DEGRADED),
            speed=observation(0.0),
            tgt=target(30.0),
        )
        self.assertIs(request.reason, LongitudinalControlReason.DEGRADED_TRACKING)
        self.assertEqual(request.throttle_request, CONFIG.degraded_throttle_authority)
        self.assertEqual(
            request.effective_target_speed_mps, CONFIG.degraded_speed_cap_mps
        )

    def test_coasting_evidence_receives_coasting_authority(self):
        request = self.drive(
            dec=decision(state=SafetyState.DEGRADED,
                         reason=SafetyReason.DEGRADED_COASTING,
                         temporal=TemporalLaneState.COASTING),
            lat=lateral(mode=LateralControlMode.DEGRADED,
                        reason=LateralControlReason.DEGRADED_COASTING,
                        safety=SafetyState.DEGRADED,
                        temporal=TemporalLaneState.COASTING),
            speed=observation(0.0),
            tgt=target(30.0),
        )
        self.assertIs(request.reason, LongitudinalControlReason.DEGRADED_COASTING)
        self.assertEqual(request.throttle_request, CONFIG.coasting_throttle_authority)
        self.assertEqual(
            request.effective_target_speed_mps, CONFIG.coasting_speed_cap_mps
        )

    def test_authorities_are_strictly_ordered(self):
        self.assertLess(
            CONFIG.coasting_throttle_authority, CONFIG.degraded_throttle_authority
        )
        self.assertLess(
            CONFIG.degraded_throttle_authority, CONFIG.nominal_throttle_authority
        )
        self.assertLess(CONFIG.coasting_speed_cap_mps, CONFIG.degraded_speed_cap_mps)
        self.assertLess(CONFIG.degraded_speed_cap_mps, CONFIG.nominal_speed_cap_mps)

    def test_the_effective_target_never_exceeds_the_hard_ceiling(self):
        config = LongitudinalControllerConfig(
            nominal_speed_cap_mps=HARD_TARGET_CEILING_MPS,
            degraded_speed_cap_mps=2.0,
            coasting_speed_cap_mps=1.0,
        )
        controller = LongitudinalController(config)
        request = self.drive(controller, speed=observation(0.0), tgt=target(30.0))
        self.assertLessEqual(
            request.effective_target_speed_mps, HARD_TARGET_CEILING_MPS
        )

    def test_a_target_above_the_configured_ceiling_is_clamped_not_refused(self):
        """A lower configured ceiling bounds the law; it does not fail closed."""
        config = LongitudinalControllerConfig(
            hard_target_ceiling_mps=2.5, nominal_speed_cap_mps=2.5
        )
        controller = LongitudinalController(config)
        request = self.drive(
            controller, speed=observation(0.0), tgt=target(HARD_TARGET_CEILING_MPS)
        )
        self.assertTrue(request.longitudinal_allowed)
        self.assertEqual(request.effective_target_speed_mps, 2.5)

    def test_a_forged_target_beyond_the_structural_ceiling_is_malformed(self):
        forged = unvalidated(target(2.0), target_speed_mps=1e6)
        request = self.drive(tgt=forged)
        self.assertNeutral(request, LongitudinalControlReason.EVIDENCE_MALFORMED)

    def test_reverse_motion_outranks_a_zero_target(self):
        """Both answers are stops; the reverse classification is the stronger."""
        request = self.drive(speed=observation(-1.0), tgt=target(0.0))
        self.assertStop(
            request, LongitudinalControlReason.REVERSE_MOTION, CONFIG.reverse_stop_brake
        )
        self.assertGreater(CONFIG.reverse_stop_brake, CONFIG.zero_hold_brake)

    def test_there_is_no_rate_shaping_between_consecutive_steps(self):
        """A first valid request may reach full Mission 10A authority at once."""
        controller = LongitudinalController()
        first = self.drive(controller, speed=observation(0.0), tgt=target(30.0))
        self.assertEqual(first.throttle_request, CONFIG.nominal_throttle_authority)
        # An immediate swing to full braking is not slewed either.
        second = controller.update(
            decision(frame=6, now=BASE + 0.1),
            lateral(frame=6, now=BASE + 0.1),
            observation(20.0, frame=6, monotonic=BASE + 0.1, captured=BASE + 0.1),
            target(2.0, revision=2, issued=BASE + 0.05),
            now=BASE + 0.1,
        )
        self.assertEqual(second.brake_request, CONFIG.max_service_brake)


# --------------------------------------------------------------------------
# F. Gates and identity
# --------------------------------------------------------------------------


class GateAndIdentityTest(LongitudinalTestCase):
    def test_initializing_safety_cannot_be_bound(self):
        verdict = SafetyDecision(
            state=SafetyState.INITIALIZING,
            reason=SafetyReason.STARTUP_NO_RESULT,
            autonomy_allowed=False,
            evaluation_monotonic=BASE,
        )
        request = self.drive(
            dec=verdict,
            lat=refused_lateral(LateralControlReason.SAFETY_NOT_PERMITTED),
        )
        self.assertNeutral(request, LongitudinalControlReason.IDENTITY_MISMATCH)

    def test_recovering_safety_requests_a_controlled_stop(self):
        request = self.drive(
            dec=decision(state=SafetyState.RECOVERING,
                         reason=SafetyReason.RECOVERY_PENDING),
            lat=refused_lateral(LateralControlReason.SAFETY_NOT_PERMITTED,
                                safety=SafetyState.RECOVERING),
        )
        self.assertStop(
            request, LongitudinalControlReason.SAFETY_STOP, CONFIG.soft_stop_brake
        )

    def test_unlatched_fail_safe_requests_a_soft_stop(self):
        request = self.drive(
            dec=decision(state=SafetyState.FAIL_SAFE,
                         reason=SafetyReason.TEMPORAL_LOST,
                         temporal=TemporalLaneState.LOST),
            lat=refused_lateral(LateralControlReason.SAFETY_NOT_PERMITTED,
                                safety=SafetyState.FAIL_SAFE,
                                temporal=TemporalLaneState.LOST),
        )
        self.assertStop(
            request, LongitudinalControlReason.SAFETY_STOP, CONFIG.soft_stop_brake
        )

    def test_latched_fail_safe_requests_the_latched_stop(self):
        request = self.drive(
            dec=decision(state=SafetyState.FAIL_SAFE,
                         reason=SafetyReason.RUNTIME_FAULT,
                         latched=True),
            lat=refused_lateral(LateralControlReason.SAFETY_NOT_PERMITTED,
                                safety=SafetyState.FAIL_SAFE),
        )
        self.assertStop(
            request,
            LongitudinalControlReason.FAIL_SAFE_STOP,
            CONFIG.latched_stop_brake,
        )

    def test_lateral_refusal_requests_a_controlled_stop(self):
        request = self.drive(
            lat=refused_lateral(LateralControlReason.INSUFFICIENT_GEOMETRY)
        )
        self.assertStop(
            request, LongitudinalControlReason.LATERAL_STOP, CONFIG.soft_stop_brake
        )

    def test_every_lateral_refusal_reason_blocks_propulsion(self):
        for reason in (
            LateralControlReason.INSUFFICIENT_GEOMETRY,
            LateralControlReason.SAFETY_NOT_PERMITTED,
            LateralControlReason.IDENTITY_MISMATCH,
            LateralControlReason.SOURCE_REJECTED,
        ):
            request = self.drive(lat=refused_lateral(reason))
            self.assertEqual(request.throttle_request, 0.0, msg=reason)
            self.assertFalse(request.longitudinal_allowed, msg=reason)

    def test_a_frame_mismatch_refuses_neutrally(self):
        request = self.drive(dec=decision(frame=5), lat=lateral(frame=6, sim_time=0.5))
        self.assertNeutral(request, LongitudinalControlReason.IDENTITY_MISMATCH)

    def test_a_simulation_time_mismatch_refuses_neutrally(self):
        request = self.drive(lat=lateral(sim_time=9.0))
        self.assertNeutral(request, LongitudinalControlReason.IDENTITY_MISMATCH)

    def test_a_monotonic_mismatch_refuses_neutrally(self):
        request = self.drive(lat=lateral(monotonic=GUIDANCE_MONOTONIC - 0.01))
        self.assertNeutral(request, LongitudinalControlReason.IDENTITY_MISMATCH)

    def test_a_guidance_revision_mismatch_refuses_neutrally(self):
        request = self.drive(lat=lateral(revision=6))
        self.assertNeutral(request, LongitudinalControlReason.IDENTITY_MISMATCH)

    def test_a_speed_revision_mismatch_refuses_neutrally(self):
        request = self.drive(speed=observation(1.0, revision=6))
        self.assertNeutral(request, LongitudinalControlReason.IDENTITY_MISMATCH)

    def test_an_evaluation_time_mismatch_refuses_neutrally(self):
        request = self.drive(lat=lateral(now=BASE - 0.01))
        self.assertNeutral(request, LongitudinalControlReason.IDENTITY_MISMATCH)
        request = self.drive(dec=decision(now=BASE - 0.01))
        self.assertNeutral(request, LongitudinalControlReason.IDENTITY_MISMATCH)

    def test_a_safety_state_mismatch_refuses_neutrally(self):
        # Both records are individually well formed; they simply disagree about
        # which safety state the shared evidence cut produced.
        request = self.drive(
            dec=decision(state=SafetyState.DEGRADED,
                         reason=SafetyReason.DEGRADED_LOW_CONFIDENCE),
            lat=lateral(),
        )
        self.assertNeutral(request, LongitudinalControlReason.IDENTITY_MISMATCH)

    def test_a_temporal_state_mismatch_refuses_neutrally(self):
        request = self.drive(lat=lateral(temporal=TemporalLaneState.COASTING))
        self.assertNeutral(request, LongitudinalControlReason.IDENTITY_MISMATCH)

    def test_a_disagreeing_published_result_age_is_time_invalid(self):
        request = self.drive(dec=decision(age=9.0))
        self.assertNeutral(request, LongitudinalControlReason.TIME_INVALID)

    def test_camera_and_ego_stamps_need_not_match(self):
        request = self.drive(speed=observation(1.0, frame=41, sim_time=4.1))
        self.assertTrue(request.longitudinal_allowed)
        self.assertEqual(request.guidance_frame, 5)
        self.assertEqual(request.speed_frame, 41)

    def test_stale_ego_speed_requests_a_controlled_stop(self):
        speed = observation(1.0, monotonic=BASE - 0.9, captured=BASE)
        request = self.drive(speed=speed)
        self.assertStop(
            request, LongitudinalControlReason.SPEED_STALE, CONFIG.soft_stop_brake
        )

    def test_a_stale_snapshot_requests_a_controlled_stop(self):
        speed = observation(1.0, monotonic=BASE, captured=BASE - 0.9)
        request = self.drive(speed=speed)
        self.assertStop(request, LongitudinalControlReason.SPEED_STALE)

    def test_an_expired_target_requests_a_controlled_stop(self):
        request = self.drive(tgt=target(2.0, valid_until=BASE - 0.001))
        self.assertStop(
            request, LongitudinalControlReason.TARGET_EXPIRED, CONFIG.soft_stop_brake
        )

    def test_an_over_aged_target_requests_a_controlled_stop(self):
        request = self.drive(
            tgt=target(2.0, issued=BASE - 5.0, valid_until=BASE + 5.0)
        )
        self.assertStop(request, LongitudinalControlReason.TARGET_EXPIRED)

    def test_a_target_issued_in_the_future_is_time_invalid(self):
        request = self.drive(
            tgt=target(2.0, issued=BASE + 1.0, valid_until=BASE + 5.0)
        )
        self.assertNeutral(request, LongitudinalControlReason.TIME_INVALID)

    def test_a_dead_vehicle_refuses_neutrally(self):
        request = self.drive(speed=observation(1.0, alive=False))
        self.assertNeutral(request, LongitudinalControlReason.VEHICLE_NOT_ALIVE)

    def test_absent_speed_refuses_neutrally(self):
        controller = LongitudinalController()
        request = controller.update(decision(), lateral(), None, target(), now=BASE)
        self.assertNeutral(request, LongitudinalControlReason.SPEED_UNAVAILABLE)

    def test_forged_lateral_steering_beyond_authority_is_malformed(self):
        forged = unvalidated(lateral(), steering=0.9)
        request = self.drive(lat=forged)
        self.assertNeutral(request, LongitudinalControlReason.EVIDENCE_MALFORMED)

    def test_a_forged_refused_lateral_that_still_steers_is_malformed(self):
        forged = unvalidated(
            refused_lateral(LateralControlReason.INSUFFICIENT_GEOMETRY),
            steering=0.2,
            steering_authority=0.35,
        )
        request = self.drive(lat=forged)
        self.assertNeutral(request, LongitudinalControlReason.EVIDENCE_MALFORMED)

    def test_a_forged_permission_that_contradicts_its_state_is_malformed(self):
        forged = unvalidated(
            decision(state=SafetyState.FAIL_SAFE, reason=SafetyReason.TEMPORAL_LOST,
                     temporal=TemporalLaneState.LOST),
            autonomy_allowed=True,
        )
        request = self.drive(
            dec=forged,
            lat=refused_lateral(LateralControlReason.SAFETY_NOT_PERMITTED,
                                safety=SafetyState.FAIL_SAFE,
                                temporal=TemporalLaneState.LOST),
        )
        self.assertNeutral(request, LongitudinalControlReason.EVIDENCE_MALFORMED)

    def test_a_forged_speed_projection_is_rejected_by_the_controller(self):
        forged = unvalidated(observation(1.0), forward_speed_mps=25.0)
        request = self.drive(speed=forged)
        self.assertNeutral(request, LongitudinalControlReason.EVIDENCE_MALFORMED)

    def test_no_refusal_ever_emits_throttle(self):
        cases = [
            dict(lat=refused_lateral(LateralControlReason.INSUFFICIENT_GEOMETRY)),
            dict(dec=decision(state=SafetyState.RECOVERING,
                              reason=SafetyReason.RECOVERY_PENDING),
                 lat=refused_lateral(LateralControlReason.SAFETY_NOT_PERMITTED,
                                     safety=SafetyState.RECOVERING)),
            dict(lat=lateral(revision=9)),
            dict(speed=observation(1.0, alive=False)),
            dict(tgt=target(2.0, issued=BASE - 2.0, valid_until=BASE - 1.0)),
            dict(speed=observation(-3.0)),
            dict(speed=observation(1.0, 3.0, 0.0)),
            dict(speed=observation(1.0, 0.0, 3.0)),
        ]
        for case in cases:
            request = self.drive(**case)
            self.assertEqual(request.throttle_request, 0.0, msg=case)
            self.assertFalse(request.longitudinal_allowed, msg=case)


# --------------------------------------------------------------------------
# G. Source ordering
# --------------------------------------------------------------------------


class SequenceTestCase(LongitudinalTestCase):
    """Drives frame sequences whose stamps depend only on the frame number.

    Holding every stamp a pure function of ``frame`` is what makes an *exact*
    duplicate expressible: re-presenting frame 5 later must reuse frame 5's
    stamps unchanged, so only the control clock moves.
    """

    @staticmethod
    def frame_monotonic(frame: int) -> float:
        """Every frame's stamp sits just before the default control clock.

        Later frames carry later stamps, but all of them precede ``BASE``, so a
        decision built for any of them has a non-negative published age and the
        ego sample is always inside the freshness window.
        """
        return BASE - 0.05 + (frame - 5) * 0.01

    def step(self, controller, frame, *, now=BASE, revision=None, sim_time=None,
             stamp=None, target_request=None, speed=None):
        revision = frame if revision is None else revision
        stamp_monotonic = self.frame_monotonic(frame) if stamp is None else stamp
        guidance_monotonic = stamp_monotonic - 0.01
        return controller.update(
            decision(frame=frame, sim_time=sim_time, revision=revision,
                     monotonic=guidance_monotonic, now=now),
            lateral(frame=frame, sim_time=sim_time, revision=revision,
                    monotonic=guidance_monotonic, now=now),
            observation(1.0, frame=frame, revision=revision,
                        monotonic=stamp_monotonic, captured=stamp_monotonic)
            if speed is None else speed,
            target() if target_request is None else target_request,
            now=now,
        )


class OrderingTest(SequenceTestCase):
    def test_an_exact_duplicate_is_re_evaluated(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE)
        again = self.step(controller, 5, now=BASE + 0.05)
        self.assertTrue(again.longitudinal_allowed)
        self.assertEqual(controller.metrics.source_rejects, 0)

    def test_a_duplicate_with_a_changed_monotonic_is_rejected(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE)
        mutated = controller.update(
            decision(frame=5, revision=5, monotonic=BASE - 0.02, now=BASE + 0.05),
            lateral(frame=5, revision=5, monotonic=BASE - 0.02, now=BASE + 0.05),
            observation(1.0, frame=5, revision=5, monotonic=BASE, captured=BASE),
            target(),
            now=BASE + 0.05,
        )
        self.assertNeutral(mutated, LongitudinalControlReason.SOURCE_REJECTED)

    def test_a_duplicate_with_a_changed_speed_stamp_is_rejected(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE)
        mutated = controller.update(
            decision(frame=5, revision=5, monotonic=BASE - 0.01, now=BASE + 0.05),
            lateral(frame=5, revision=5, monotonic=BASE - 0.01, now=BASE + 0.05),
            observation(1.0, frame=5, revision=5, monotonic=BASE + 0.02,
                        captured=BASE),
            target(),
            now=BASE + 0.05,
        )
        self.assertNeutral(mutated, LongitudinalControlReason.SOURCE_REJECTED)

    def test_a_duplicate_with_a_changed_revision_is_rejected(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE, revision=5)
        mutated = self.step(controller, 5, now=BASE + 0.05, revision=6)
        self.assertNeutral(mutated, LongitudinalControlReason.SOURCE_REJECTED)

    def test_a_lower_frame_is_rejected(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE)
        older = self.step(controller, 4, now=BASE + 0.05)
        self.assertNeutral(older, LongitudinalControlReason.SOURCE_REJECTED)

    def test_an_equal_frame_with_a_changed_sim_time_is_rejected(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE)
        changed = self.step(controller, 5, now=BASE + 0.05, sim_time=9.9)
        self.assertNeutral(changed, LongitudinalControlReason.SOURCE_REJECTED)

    def test_a_higher_frame_with_an_equal_sim_time_is_accepted(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE, sim_time=0.5)
        newer = self.step(controller, 6, now=BASE + 0.05, sim_time=0.5)
        self.assertTrue(newer.longitudinal_allowed)

    def test_a_regressed_simulation_time_is_rejected(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE, sim_time=0.5)
        regressed = self.step(controller, 6, now=BASE + 0.05, sim_time=0.4)
        self.assertNeutral(regressed, LongitudinalControlReason.SOURCE_REJECTED)

    def test_an_older_speed_stamp_is_rejected(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE)
        stale = controller.update(
            decision(frame=6, revision=6, monotonic=BASE + 0.04, now=BASE + 0.05),
            lateral(frame=6, revision=6, monotonic=BASE + 0.04, now=BASE + 0.05),
            observation(1.0, frame=4, revision=6, monotonic=BASE + 0.05,
                        captured=BASE + 0.05),
            target(),
            now=BASE + 0.05,
        )
        self.assertNeutral(stale, LongitudinalControlReason.SOURCE_REJECTED)

    def test_a_target_duplicate_is_accepted(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE)
        again = self.step(controller, 6, now=BASE + 0.05)
        self.assertTrue(again.longitudinal_allowed)

    def test_a_target_rollback_is_rejected(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE, target_request=target(2.0, revision=4))
        rolled = self.step(
            controller, 6, now=BASE + 0.05,
            target_request=target(2.0, revision=3, issued=BASE - 0.30),
        )
        self.assertNeutral(rolled, LongitudinalControlReason.SOURCE_REJECTED)

    def test_a_target_with_an_equal_revision_but_new_issue_time_is_rejected(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE, target_request=target(2.0, revision=4))
        changed = self.step(
            controller, 6, now=BASE + 0.05,
            target_request=target(2.0, revision=4, issued=BASE - 0.10),
        )
        self.assertNeutral(changed, LongitudinalControlReason.SOURCE_REJECTED)

    def test_a_mutated_target_duplicate_is_rejected(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE, target_request=target(2.0))
        mutated = self.step(controller, 6, now=BASE + 0.05,
                            target_request=target(2.5))
        self.assertNeutral(mutated, LongitudinalControlReason.SOURCE_REJECTED)

    def test_a_target_origin_change_without_reset_is_rejected(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE,
                  target_request=target(2.0, origin=TargetSpeedOrigin.OPERATOR))
        switched = self.step(
            controller, 6, now=BASE + 0.05,
            target_request=target(2.0, origin=TargetSpeedOrigin.SCENARIO, revision=2),
        )
        self.assertNeutral(switched, LongitudinalControlReason.SOURCE_REJECTED)

    def test_a_target_epoch_change_without_reset_is_rejected(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE, target_request=target(2.0, epoch=1))
        switched = self.step(
            controller, 6, now=BASE + 0.05,
            target_request=target(2.0, epoch=2, revision=2),
        )
        self.assertNeutral(switched, LongitudinalControlReason.SOURCE_REJECTED)

    def test_a_higher_target_revision_is_accepted(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE, target_request=target(2.0, revision=1))
        newer = self.step(
            controller, 6, now=BASE + 0.05,
            target_request=target(2.5, revision=2, issued=BASE - 0.10),
        )
        self.assertTrue(newer.longitudinal_allowed)

    def test_a_rejection_does_not_move_any_baseline(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE)
        baseline = controller.metrics.last_accepted_guidance_stamp
        self.step(controller, 4, now=BASE + 0.05)
        self.assertEqual(
            controller.metrics.last_accepted_guidance_stamp.frame, baseline.frame
        )
        # The legitimate next frame is still accepted afterwards.
        resumed = self.step(controller, 6, now=BASE + 0.10)
        self.assertTrue(resumed.longitudinal_allowed)


# --------------------------------------------------------------------------
# H. Time, first command, and reset
# --------------------------------------------------------------------------


class TimeAndResetTest(SequenceTestCase):
    def test_a_first_command_may_reach_full_mission_10a_authority(self):
        controller = LongitudinalController()
        request = controller.update(
            decision(), lateral(), observation(0.0), target(30.0), now=BASE
        )
        self.assertEqual(request.throttle_request, CONFIG.nominal_throttle_authority)

    def test_an_equal_time_duplicate_is_allowed(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE)
        again = self.step(controller, 5, now=BASE)
        self.assertTrue(again.longitudinal_allowed)

    def test_equal_time_with_changed_evidence_is_rejected(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE)
        changed = self.step(controller, 6, now=BASE)
        self.assertNeutral(changed, LongitudinalControlReason.SOURCE_REJECTED)

    def test_a_regressed_control_clock_refuses(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE)
        regressed = self.step(controller, 6, now=BASE - 1.0, stamp=BASE - 1.05)
        self.assertNeutral(regressed, LongitudinalControlReason.TIME_INVALID)

    def test_non_finite_control_clocks_refuse(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            controller = LongitudinalController()
            request = controller.update(
                decision(), lateral(), observation(1.0), target(), now=bad
            )
            self.assertNeutral(request, LongitudinalControlReason.TIME_INVALID)
            self.assertIsNone(request.evaluation_monotonic)

    def test_an_invalid_clock_is_followed_by_a_normal_evaluation(self):
        controller = LongitudinalController()
        controller.update(
            decision(), lateral(), observation(1.0), target(), now=float("nan")
        )
        request = self.step(controller, 5, now=BASE)
        self.assertTrue(request.longitudinal_allowed)

    def test_a_rollback_is_never_laundered_into_a_new_epoch(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE)
        self.step(controller, 6, now=BASE - 5.0, stamp=BASE - 5.05)
        # A second attempt at the same rolled-back instant is still refused.
        again = self.step(controller, 6, now=BASE - 5.0, stamp=BASE - 5.05)
        self.assertNeutral(again, LongitudinalControlReason.TIME_INVALID)

    def test_a_non_finite_clock_cannot_launder_a_rollback(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE)
        controller.update(
            decision(), lateral(), observation(1.0), target(), now=float("nan")
        )
        rolled = self.step(controller, 6, now=BASE - 5.0, stamp=BASE - 5.05)
        self.assertNeutral(rolled, LongitudinalControlReason.TIME_INVALID)

    def test_reset_publishes_a_neutral_unevaluated_request(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE)
        cleared = controller.reset()
        self.assertNeutral(cleared, LongitudinalControlReason.NOT_EVALUATED)
        self.assertIsNone(cleared.evaluation_monotonic)
        self.assertIsNone(cleared.guidance_stamp)
        self.assertIs(controller.latest_request, cleared)

    def test_reset_clears_every_behavioral_baseline(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE)
        controller.reset()
        metrics = controller.metrics
        self.assertIsNone(metrics.last_accepted_guidance_stamp)
        self.assertIsNone(metrics.last_accepted_speed_stamp)
        self.assertIsNone(metrics.last_accepted_target_origin)
        self.assertIsNone(metrics.last_accepted_target_epoch)
        self.assertIsNone(metrics.last_accepted_target_revision)

    def test_reset_preserves_truthful_lifetime_counters(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE)
        before = controller.metrics
        controller.reset()
        after = controller.metrics
        self.assertEqual(after.evaluations, before.evaluations)
        self.assertEqual(after.permitted_requests, before.permitted_requests)
        self.assertEqual(after.resets, before.resets + 1)

    def test_repeated_reset_is_idempotent_apart_from_its_counter(self):
        controller = LongitudinalController()
        first = controller.reset()
        second = controller.reset()
        self.assertEqual(first, second)
        self.assertEqual(controller.metrics.resets, 2)

    def test_an_old_frame_is_acceptable_again_after_reset(self):
        controller = LongitudinalController()
        self.step(controller, 9, now=BASE)
        controller.reset()
        resumed = self.step(controller, 1, now=BASE + 0.05)
        self.assertTrue(resumed.longitudinal_allowed)

    def test_a_new_target_origin_is_acceptable_after_reset(self):
        controller = LongitudinalController()
        self.step(controller, 5, now=BASE,
                  target_request=target(2.0, origin=TargetSpeedOrigin.OPERATOR))
        controller.reset()
        resumed = self.step(
            controller, 1, now=BASE + 0.05,
            target_request=target(2.0, origin=TargetSpeedOrigin.SCENARIO),
        )
        self.assertTrue(resumed.longitudinal_allowed)

    def test_no_stale_pedal_request_crosses_a_reset(self):
        controller = LongitudinalController()
        moving = self.step(controller, 5, now=BASE)
        self.assertGreater(moving.throttle_request, 0.0)
        cleared = controller.reset()
        self.assertEqual(cleared.throttle_request, 0.0)
        self.assertEqual(cleared.brake_request, 0.0)
        self.assertEqual(cleared.throttle_authority, 0.0)
        self.assertEqual(cleared.brake_authority, 0.0)


# --------------------------------------------------------------------------
# I. Transactionality and hostile input
# --------------------------------------------------------------------------


class TransactionalityTest(LongitudinalTestCase):
    def hostile_cases(self):
        return (
            ("decision", exploding(decision(), "state", RuntimeError("boom"))),
            ("decision", exploding(decision(), "source_stamp", KeyError("k"))),
            ("lateral", exploding(lateral(), "mode", RuntimeError("boom"))),
            ("lateral", exploding(lateral(), "snapshot_revision", ZeroDivisionError())),
            ("speed", exploding(observation(1.0), "world_velocity", RuntimeError())),
            ("speed", exploding(observation(1.0), "vehicle_alive", MemoryError())),
            ("target", exploding(target(), "target_speed_mps", RuntimeError())),
            ("target", exploding(target(), "origin", AttributeError())),
        )

    def call(self, controller, kind, hostile, *, now=BASE):
        return controller.update(
            hostile if kind == "decision" else decision(),
            hostile if kind == "lateral" else lateral(),
            hostile if kind == "speed" else observation(1.0),
            hostile if kind == "target" else target(),
            now=now,
        )

    def test_every_hostile_property_fails_closed(self):
        for kind, hostile in self.hostile_cases():
            controller = LongitudinalController()
            request = self.call(controller, kind, hostile)
            self.assertFalse(request.longitudinal_allowed, msg=kind)
            self.assertEqual(request.throttle_request, 0.0, msg=kind)
            self.assertEqual(request.brake_request, 0.0, msg=kind)
            self.assertIs(request.mode, LongitudinalControlMode.DISABLED, msg=kind)

    def test_a_hostile_read_never_moves_a_baseline(self):
        for kind, hostile in self.hostile_cases():
            controller = LongitudinalController()
            controller.update(
                decision(), lateral(), observation(1.0), target(), now=BASE
            )
            before = controller.metrics
            self.call(controller, kind, hostile, now=BASE + 0.05)
            after = controller.metrics
            self.assertEqual(
                after.last_accepted_guidance_stamp,
                before.last_accepted_guidance_stamp,
                msg=kind,
            )
            self.assertEqual(
                after.last_accepted_target_revision,
                before.last_accepted_target_revision,
                msg=kind,
            )

    def test_a_hostile_read_replaces_the_previously_permitted_request(self):
        controller = LongitudinalController()
        permitted = controller.update(
            decision(), lateral(), observation(1.0), target(), now=BASE
        )
        self.assertTrue(permitted.longitudinal_allowed)
        hostile = exploding(lateral(), "mode", RuntimeError("boom"))
        refused = self.call(controller, "lateral", hostile, now=BASE + 0.05)
        self.assertIs(controller.latest_request, refused)
        self.assertFalse(controller.latest_request.longitudinal_allowed)

    def test_no_exception_or_traceback_is_retained(self):
        controller = LongitudinalController()
        hostile = exploding(observation(1.0), "world_velocity", RuntimeError("boom"))
        self.call(controller, "speed", hostile)
        pending = [
            getattr(controller, name)
            for name in LongitudinalController.__slots__
            if hasattr(controller, name)
        ]
        for value in pending:
            self.assertNotIsInstance(value, BaseException)
            self.assertFalse(hasattr(value, "__traceback__"), msg=repr(value))

    def test_no_caller_evidence_object_is_retained(self):
        controller = LongitudinalController()
        verdict = decision()
        guidance = lateral()
        speed = observation(1.0)
        goal = target()
        controller.update(verdict, guidance, speed, goal, now=BASE)
        retained = retained_ids(controller)
        for caller_object in (verdict, guidance, speed, goal):
            self.assertNotIn(id(caller_object), retained, msg=type(caller_object))
        self.assertNotIn(id(verdict.source_stamp), retained)
        self.assertNotIn(id(speed.source_stamp), retained)
        self.assertNotIn(id(speed.world_velocity), retained)

    def test_the_published_request_does_not_alias_caller_stamps(self):
        controller = LongitudinalController()
        verdict = decision()
        speed = observation(1.0)
        request = controller.update(verdict, lateral(), speed, target(), now=BASE)
        self.assertIsNot(request.guidance_stamp, verdict.source_stamp)
        self.assertIsNot(request.speed_stamp, speed.source_stamp)
        self.assertEqual(request.guidance_stamp, verdict.source_stamp)

    def test_metrics_do_not_alias_retained_state(self):
        controller = LongitudinalController()
        controller.update(decision(), lateral(), observation(1.0), target(), now=BASE)
        first = controller.metrics
        second = controller.metrics
        self.assertIsNot(
            first.last_accepted_guidance_stamp, second.last_accepted_guidance_stamp
        )
        self.assertEqual(
            first.last_accepted_guidance_stamp, second.last_accepted_guidance_stamp
        )

    def test_wrong_argument_types_raise_type_error(self):
        controller = LongitudinalController()
        with self.assertRaises(TypeError):
            controller.update(object(), lateral(), observation(1.0), target(), now=BASE)
        with self.assertRaises(TypeError):
            controller.update(decision(), object(), observation(1.0), target(), now=BASE)
        with self.assertRaises(TypeError):
            controller.update(decision(), lateral(), object(), target(), now=BASE)
        with self.assertRaises(TypeError):
            controller.update(decision(), lateral(), observation(1.0), object(), now=BASE)

    def test_retained_state_stays_bounded_over_a_long_stream(self):
        controller = LongitudinalController()
        gc.collect()
        for frame in range(1, 400):
            now = BASE + frame * 0.05
            controller.update(
                decision(frame=frame, revision=frame, monotonic=now - 0.01, now=now),
                lateral(frame=frame, revision=frame, monotonic=now - 0.01, now=now),
                observation(1.0, frame=frame, revision=frame, monotonic=now,
                            captured=now),
                target(),
                now=now,
            )
        self.assertLessEqual(len(retained_ids(controller)), 24)
        self.assertEqual(controller.metrics.evaluations, 399)


# --------------------------------------------------------------------------
# R1 / MAJOR 1. Complete upstream record validation
# --------------------------------------------------------------------------


class UpstreamCanonicalizationTest(LongitudinalTestCase):
    """A malformed M8/M9 field invalidates the whole record.

    Mission 10A re-proves every intrinsic upstream invariant, not merely the
    subset it performs arithmetic on, so a constructor-bypassed record cannot
    authorize propulsion through a field this module never reads.
    """

    def assertFailsClosed(self, controller, request):
        self.assertIs(request.mode, LongitudinalControlMode.DISABLED)
        self.assertIs(request.reason, LongitudinalControlReason.EVIDENCE_MALFORMED)
        self.assertFalse(request.longitudinal_allowed)
        self.assertFalse(request.stop_required)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, 0.0)
        self.assertIs(controller.latest_request, request)

    def probe(self, *, dec=None, lat=None):
        """Run one malformed record and assert the full fail-closed contract."""
        controller = LongitudinalController()
        # Establish a real baseline first, so "no baseline movement" is testable.
        good = controller.update(
            decision(), lateral(), observation(1.0), target(), now=BASE
        )
        self.assertTrue(good.longitudinal_allowed)
        before = controller.metrics
        request = controller.update(
            decision() if dec is None else dec,
            lateral() if lat is None else lat,
            observation(1.0, frame=6, revision=6, monotonic=BASE + 0.05,
                        captured=BASE + 0.05),
            target(revision=2, issued=BASE - 0.10),
            now=BASE + 0.05,
        )
        self.assertFailsClosed(controller, request)
        after = controller.metrics
        self.assertEqual(
            after.last_accepted_guidance_stamp, before.last_accepted_guidance_stamp
        )
        self.assertEqual(
            after.last_accepted_speed_stamp, before.last_accepted_speed_stamp
        )
        self.assertEqual(
            after.last_accepted_target_revision, before.last_accepted_target_revision
        )
        # A later well-formed cut is still acceptable.
        resumed = controller.update(
            decision(frame=7, revision=7, monotonic=BASE + 0.09, now=BASE + 0.10),
            lateral(frame=7, revision=7, monotonic=BASE + 0.09, now=BASE + 0.10),
            observation(1.0, frame=7, revision=7, monotonic=BASE + 0.10,
                        captured=BASE + 0.10),
            target(revision=3, issued=BASE + 0.05),
            now=BASE + 0.10,
        )
        self.assertTrue(resumed.longitudinal_allowed)
        return controller

    # ---- Mission 8 -------------------------------------------------------

    def test_state_and_reason_contradiction_is_rejected(self):
        self.probe(dec=unvalidated(decision(), reason=SafetyReason.RUNTIME_FAULT))
        self.probe(dec=bypassed(decision(), reason=SafetyReason.RESULT_STALE))

    def test_permission_and_state_contradiction_is_rejected(self):
        self.probe(dec=unvalidated(decision(), autonomy_allowed=False))
        self.probe(dec=bypassed(decision(), state=SafetyState.FAIL_SAFE))

    def test_latched_outside_fail_safe_is_rejected(self):
        self.probe(dec=unvalidated(decision(), latched=True))
        self.probe(dec=bypassed(decision(), latched=True))

    def test_non_builtin_bool_permission_is_rejected(self):
        self.probe(dec=bypassed(decision(), autonomy_allowed=1))
        self.probe(dec=bypassed(decision(), latched=0))

    def test_missing_temporal_confidence_is_rejected(self):
        self.probe(dec=bypassed(decision(), temporal_confidence=None))

    def test_out_of_range_temporal_confidence_is_rejected(self):
        for bad in (5.0, -0.5, float("nan")):
            self.probe(dec=bypassed(decision(), temporal_confidence=bad))

    def test_non_finite_or_negative_result_age_is_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf"), -1.0):
            self.probe(dec=bypassed(decision(), result_age_seconds=bad))

    def test_negative_or_bool_recovery_streak_is_rejected(self):
        self.probe(dec=bypassed(decision(), recovery_streak=-3))
        self.probe(dec=bypassed(decision(), recovery_streak=True))

    def test_partial_published_evidence_is_rejected(self):
        """M8 publishes stamp, revision, temporal state, and confidence together."""
        self.probe(dec=bypassed(decision(), temporal_state=None))
        self.probe(dec=bypassed(decision(), snapshot_revision=None))

    def test_non_enum_decision_members_are_rejected(self):
        self.probe(dec=bypassed(decision(), state="nominal"))
        self.probe(dec=bypassed(decision(), reason="nominal_tracking"))
        self.probe(dec=bypassed(decision(), temporal_state="tracking"))

    def test_initializing_decision_carrying_evidence_is_rejected(self):
        self.probe(dec=bypassed(decision(), state=SafetyState.INITIALIZING,
                                reason=SafetyReason.STARTUP_NO_RESULT,
                                autonomy_allowed=False))

    def test_hostile_decision_properties_fail_closed(self):
        for name in ("state", "reason", "autonomy_allowed", "latched",
                     "temporal_confidence", "recovery_streak",
                     "result_age_seconds", "snapshot_revision"):
            self.probe(dec=exploding(decision(), name, RuntimeError(name)))

    # ---- Mission 9 -------------------------------------------------------

    def test_lateral_mode_and_reason_contradiction_is_rejected(self):
        self.probe(lat=unvalidated(
            lateral(), reason=LateralControlReason.SAFETY_NOT_PERMITTED))
        self.probe(lat=bypassed(
            lateral(), reason=LateralControlReason.INSUFFICIENT_GEOMETRY))

    def test_lateral_permission_and_mode_contradiction_is_rejected(self):
        self.probe(lat=bypassed(lateral(), lateral_allowed=False))
        self.probe(lat=bypassed(lateral(), mode=LateralControlMode.DISABLED))

    def test_non_finite_raw_steering_is_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            self.probe(lat=bypassed(lateral(), raw_steering=bad))

    def test_steering_beyond_its_authority_is_rejected(self):
        self.probe(lat=bypassed(lateral(), steering=0.9))
        self.probe(lat=bypassed(lateral(), steering_authority=0.05))

    def test_out_of_range_steering_is_rejected(self):
        self.probe(lat=bypassed(lateral(), steering=2.0, steering_authority=3.0))

    def test_malformed_shaping_flags_are_rejected(self):
        self.probe(lat=bypassed(lateral(), clamped=1))
        self.probe(lat=bypassed(lateral(), rate_limited=0))

    def test_out_of_range_lane_errors_are_rejected(self):
        self.probe(lat=bypassed(lateral(), lateral_error=9.0))
        self.probe(lat=bypassed(lateral(), heading_error=-4.0))

    def test_incomplete_permitted_lateral_evidence_is_rejected(self):
        self.probe(lat=bypassed(lateral(), heading_error=None))
        self.probe(lat=bypassed(lateral(), lateral_error=None))

    def test_nominal_lateral_mode_without_nominal_safety_is_rejected(self):
        self.probe(lat=bypassed(lateral(), safety_state=SafetyState.DEGRADED))

    def test_a_refused_lateral_request_that_steers_is_rejected(self):
        refused = refused_lateral(LateralControlReason.INSUFFICIENT_GEOMETRY)
        self.probe(lat=bypassed(refused, steering=0.2, steering_authority=0.35))
        self.probe(lat=bypassed(refused, clamped=True))
        self.probe(lat=bypassed(refused, lateral_error=0.1))

    def test_non_enum_lateral_members_are_rejected(self):
        self.probe(lat=bypassed(lateral(), mode="nominal"))
        self.probe(lat=bypassed(lateral(), reason="nominal_tracking"))
        self.probe(lat=bypassed(lateral(), temporal_state="tracking"))
        self.probe(lat=bypassed(lateral(), safety_state="nominal"))

    def test_unevaluated_lateral_request_carrying_evidence_is_rejected(self):
        self.probe(lat=bypassed(
            lateral(), mode=LateralControlMode.DISABLED,
            reason=LateralControlReason.NOT_EVALUATED, lateral_allowed=False,
            steering=0.0, raw_steering=0.0, steering_authority=0.0,
            lateral_error=None, heading_error=None))

    def test_hostile_lateral_properties_fail_closed(self):
        for name in ("mode", "reason", "lateral_allowed", "steering",
                     "raw_steering", "steering_authority", "clamped",
                     "rate_limited", "lateral_error", "heading_error",
                     "safety_state", "temporal_state"):
            self.probe(lat=exploding(lateral(), name, RuntimeError(name)))

    def test_the_canonical_records_are_fresh_exact_base_instances(self):
        """Canonicalization must not retain, or return, the caller's record."""
        verdict = decision()
        guidance = lateral()
        canonical_decision = _canonicalize_safety_decision(verdict)
        canonical_lateral = _canonicalize_lateral_request(guidance)
        self.assertIsNot(canonical_decision, verdict)
        self.assertIsNot(canonical_lateral, guidance)
        self.assertIs(type(canonical_decision), SafetyDecision)
        self.assertIs(type(canonical_lateral), LateralControlRequest)
        self.assertIsNot(canonical_decision.source_stamp, verdict.source_stamp)
        self.assertIsNot(canonical_lateral.source_stamp, guidance.source_stamp)
        self.assertEqual(canonical_decision, verdict)
        self.assertEqual(canonical_lateral, guidance)

    def test_a_subclass_is_canonicalized_into_the_exact_base_type(self):
        subclassed = unvalidated(decision())
        self.assertIsNot(type(subclassed), SafetyDecision)
        canonical = _canonicalize_safety_decision(subclassed)
        self.assertIs(type(canonical), SafetyDecision)

    def test_a_drifting_upstream_number_is_read_exactly_once(self):
        forged = bypassed(decision(), temporal_confidence=DriftingReal(0.8, 99.0))
        canonical = _canonicalize_safety_decision(forged)
        self.assertEqual(canonical.temporal_confidence, 0.8)
        self.assertEqual(canonical.temporal_confidence, 0.8)


# --------------------------------------------------------------------------
# R1 / MAJOR 2. Excessive vertical motion dominates every brake-bearing branch
# --------------------------------------------------------------------------


class VerticalPrecedenceTest(LongitudinalTestCase):
    """Vertical motion means road contact, and therefore braking, is untrusted.

    Once the evidence is readable, bindable, and in order, an excessive vertical
    component must outrank every branch that would otherwise request brake --
    including a latched Mission 8 fail-safe -- because asking for a specific
    deceleration would assert confidence the controller does not have.
    """

    LIFT = 1.0

    def assertVerticalNeutral(self, request):
        self.assertIs(
            request.reason, LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION
        )
        self.assertIs(request.mode, LongitudinalControlMode.DISABLED)
        self.assertFalse(request.longitudinal_allowed)
        self.assertFalse(request.stop_required)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, 0.0)
        self.assertEqual(request.throttle_authority, 0.0)
        self.assertEqual(request.brake_authority, 0.0)

    def test_vertical_motion_outranks_every_stop_bearing_branch(self):
        stale = observation(1.0, 0.0, self.LIFT, monotonic=BASE - 0.90)
        stale_snapshot = observation(1.0, 0.0, self.LIFT, captured=BASE - 0.90)
        cases = {
            "M8 RECOVERING": dict(
                dec=decision(state=SafetyState.RECOVERING,
                             reason=SafetyReason.RECOVERY_PENDING),
                lat=refused_lateral(LateralControlReason.SAFETY_NOT_PERMITTED,
                                    safety=SafetyState.RECOVERING)),
            "M8 FAIL_SAFE": dict(
                dec=decision(state=SafetyState.FAIL_SAFE,
                             reason=SafetyReason.TEMPORAL_LOST,
                             temporal=TemporalLaneState.LOST),
                lat=refused_lateral(LateralControlReason.SAFETY_NOT_PERMITTED,
                                    safety=SafetyState.FAIL_SAFE,
                                    temporal=TemporalLaneState.LOST)),
            "M8 latched": dict(
                dec=decision(state=SafetyState.FAIL_SAFE,
                             reason=SafetyReason.RUNTIME_FAULT, latched=True),
                lat=refused_lateral(LateralControlReason.SAFETY_NOT_PERMITTED,
                                    safety=SafetyState.FAIL_SAFE)),
            "M9 disabled": dict(
                lat=refused_lateral(LateralControlReason.INSUFFICIENT_GEOMETRY)),
            "stale ego speed": dict(speed=stale),
            "stale snapshot": dict(speed=stale_snapshot),
            "expired target": dict(
                tgt=target(2.0, issued=BASE - 2.0, valid_until=BASE - 1.0)),
            "over-aged target": dict(
                tgt=target(2.0, issued=BASE - 5.0, valid_until=BASE + 5.0)),
            "excessive lateral too": dict(
                speed=observation(1.0, 2.0, self.LIFT)),
            "reverse too": dict(speed=observation(-2.0, 0.0, self.LIFT)),
            "target zero": dict(tgt=target(0.0)),
            "forward overspeed": dict(speed=observation(9.0, 0.0, self.LIFT)),
        }
        for label, kwargs in cases.items():
            kwargs.setdefault("speed", observation(1.0, 0.0, self.LIFT))
            with self.subTest(label):
                self.assertVerticalNeutral(self.drive(**kwargs))

    def test_vertical_motion_outranks_a_latched_stop_specifically(self):
        """The single most authoritative brake request still loses to it."""
        without_lift = self.drive(
            dec=decision(state=SafetyState.FAIL_SAFE,
                         reason=SafetyReason.RUNTIME_FAULT, latched=True),
            lat=refused_lateral(LateralControlReason.SAFETY_NOT_PERMITTED,
                                safety=SafetyState.FAIL_SAFE),
            speed=observation(1.0))
        self.assertStop(
            without_lift, LongitudinalControlReason.FAIL_SAFE_STOP,
            CONFIG.latched_stop_brake)
        with_lift = self.drive(
            dec=decision(state=SafetyState.FAIL_SAFE,
                         reason=SafetyReason.RUNTIME_FAULT, latched=True),
            lat=refused_lateral(LateralControlReason.SAFETY_NOT_PERMITTED,
                                safety=SafetyState.FAIL_SAFE),
            speed=observation(1.0, 0.0, self.LIFT))
        self.assertVerticalNeutral(with_lift)

    def test_earlier_neutral_gates_still_precede_the_vertical_check(self):
        """Untrusted identity must never use a vertical scalar to say anything."""
        lifted = observation(1.0, 0.0, self.LIFT)
        initializing = SafetyDecision(
            state=SafetyState.INITIALIZING,
            reason=SafetyReason.STARTUP_NO_RESULT,
            autonomy_allowed=False,
            evaluation_monotonic=BASE)
        cases = {
            "M8 INITIALIZING": (
                dict(dec=initializing,
                     lat=refused_lateral(
                         LateralControlReason.SAFETY_NOT_PERMITTED),
                     speed=lifted),
                LongitudinalControlReason.IDENTITY_MISMATCH),
            "identity mismatch": (
                dict(lat=lateral(revision=9), speed=lifted),
                LongitudinalControlReason.IDENTITY_MISMATCH),
            "malformed evidence": (
                dict(lat=bypassed(lateral(), raw_steering=float("nan")),
                     speed=lifted),
                LongitudinalControlReason.EVIDENCE_MALFORMED),
            "dead vehicle": (
                dict(speed=observation(1.0, 0.0, self.LIFT, alive=False)),
                LongitudinalControlReason.VEHICLE_NOT_ALIVE),
        }
        for label, (kwargs, expected) in cases.items():
            with self.subTest(label):
                request = self.drive(**kwargs)
                self.assertNeutral(request, expected)

    def test_absent_speed_still_precedes_the_vertical_check(self):
        controller = LongitudinalController()
        request = controller.update(decision(), lateral(), None, target(), now=BASE)
        self.assertNeutral(request, LongitudinalControlReason.SPEED_UNAVAILABLE)

    def test_the_vertical_limit_boundary_is_exclusive(self):
        at_limit = self.drive(
            speed=observation(1.0, 0.0, CONFIG.vertical_speed_limit_mps))
        self.assertTrue(at_limit.longitudinal_allowed)
        self.assertIs(at_limit.reason, LongitudinalControlReason.NOMINAL_TRACKING)

    def test_one_ulp_above_the_vertical_limit_refuses(self):
        just_over = math.nextafter(CONFIG.vertical_speed_limit_mps, 1.0)
        self.assertVerticalNeutral(self.drive(speed=observation(1.0, 0.0, just_over)))

    def test_a_downward_component_is_treated_the_same_as_upward(self):
        self.assertVerticalNeutral(self.drive(speed=observation(1.0, 0.0, -self.LIFT)))

    def test_a_vertical_refusal_still_advances_the_order_baseline(self):
        """It is accepted, bound evidence, so ordering must stay current."""
        controller = LongitudinalController()
        self.drive(controller, speed=observation(1.0, 0.0, self.LIFT))
        self.assertIsNotNone(controller.metrics.last_accepted_guidance_stamp)
        replayed = controller.update(
            decision(frame=4, revision=4, monotonic=BASE - 0.06, now=BASE + 0.05),
            lateral(frame=4, revision=4, monotonic=BASE - 0.06, now=BASE + 0.05),
            observation(1.0, frame=4, revision=4, monotonic=BASE,
                        captured=BASE),
            target(),
            now=BASE + 0.05)
        self.assertNeutral(replayed, LongitudinalControlReason.SOURCE_REJECTED)


# --------------------------------------------------------------------------
# R1 / MAJOR 3. STOPPING always means a real, bounded, identity-bound stop
# --------------------------------------------------------------------------


class StopRequiredIntegrityTest(LongitudinalTestCase):
    """``stop_required`` positively asks for deceleration, or it is not asserted."""

    def binding(self):
        return dict(
            evaluation_monotonic=BASE,
            guidance_stamp=SampleStamp(frame=5, sim_time=0.5, monotonic=BASE),
            speed_stamp=SampleStamp(frame=5, sim_time=0.5, monotonic=BASE),
            snapshot_revision=5,
            target_origin=TargetSpeedOrigin.TEST,
            target_epoch=1,
            target_revision=1,
            target_issued_monotonic=BASE - 1.0,
            target_valid_until_monotonic=BASE + 1.0,
            safety_state=SafetyState.NOMINAL,
            lateral_mode=LateralControlMode.NOMINAL,
            temporal_state=TemporalLaneState.TRACKING,
        )

    def stopping(self, **overrides):
        base = dict(
            mode=LongitudinalControlMode.STOPPING,
            reason=LongitudinalControlReason.SAFETY_STOP,
            longitudinal_allowed=False,
            stop_required=True,
            throttle_request=0.0,
            brake_request=0.20,
            brake_authority=0.40,
            observed_forward_speed_mps=1.0,
            **self.binding(),
        )
        base.update(overrides)
        return LongitudinalControlRequest(**base)

    def test_a_well_formed_stopping_request_is_accepted(self):
        request = self.stopping()
        self.assertTrue(request.stop_required)
        self.assertGreater(request.brake_request, 0.0)

    def test_stopping_with_zero_brake_is_rejected(self):
        with self.assertRaises(ValueError):
            self.stopping(brake_request=0.0)

    def test_stop_required_without_the_stopping_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            LongitudinalControlRequest(
                mode=LongitudinalControlMode.DISABLED,
                reason=LongitudinalControlReason.IDENTITY_MISMATCH,
                longitudinal_allowed=False,
                stop_required=True)

    def test_stopping_without_stop_required_is_rejected(self):
        with self.assertRaises(ValueError):
            self.stopping(stop_required=False)

    def test_stopping_with_throttle_is_rejected(self):
        with self.assertRaises(ValueError):
            self.stopping(throttle_request=0.1, throttle_authority=0.35)

    def test_stopping_requires_every_identity_field(self):
        for field in self.binding():
            with self.subTest(field):
                with self.assertRaises(ValueError):
                    self.stopping(**{field: None})

    def test_stopping_cannot_carry_a_neutral_refusal_reason(self):
        for reason in (
            LongitudinalControlReason.IDENTITY_MISMATCH,
            LongitudinalControlReason.EVIDENCE_MALFORMED,
            LongitudinalControlReason.SOURCE_REJECTED,
            LongitudinalControlReason.TIME_INVALID,
            LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION,
            LongitudinalControlReason.VEHICLE_NOT_ALIVE,
            LongitudinalControlReason.SPEED_UNAVAILABLE,
            LongitudinalControlReason.NOT_EVALUATED,
        ):
            with self.subTest(reason):
                with self.assertRaises(ValueError):
                    self.stopping(reason=reason)

    def test_every_stop_or_hold_brake_must_be_strictly_positive(self):
        for name in (
            "minimum_moving_stop_brake",
            "soft_stop_brake",
            "latched_stop_brake",
            "reverse_stop_brake",
            "zero_hold_brake",
        ):
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    LongitudinalControllerConfig(**{name: 0.0})
                with self.assertRaises(ValueError):
                    LongitudinalControllerConfig(**{name: -0.1})

    def test_max_service_brake_must_be_strictly_positive(self):
        with self.assertRaises(ValueError):
            LongitudinalControllerConfig(max_service_brake=0.0)

    def test_every_controller_stop_carries_a_real_bounded_braked_stop(self):
        """Sweep every branch that publishes STOPPING and check the contract."""
        cases = {
            "safety stop": dict(
                dec=decision(state=SafetyState.RECOVERING,
                             reason=SafetyReason.RECOVERY_PENDING),
                lat=refused_lateral(LateralControlReason.SAFETY_NOT_PERMITTED,
                                    safety=SafetyState.RECOVERING)),
            "latched stop": dict(
                dec=decision(state=SafetyState.FAIL_SAFE,
                             reason=SafetyReason.RUNTIME_FAULT, latched=True),
                lat=refused_lateral(LateralControlReason.SAFETY_NOT_PERMITTED,
                                    safety=SafetyState.FAIL_SAFE)),
            "lateral stop": dict(
                lat=refused_lateral(LateralControlReason.INSUFFICIENT_GEOMETRY)),
            "target expired": dict(
                tgt=target(2.0, issued=BASE - 2.0, valid_until=BASE - 1.0)),
            "speed stale": dict(speed=observation(1.0, monotonic=BASE - 0.9)),
            "lateral anomaly": dict(speed=observation(1.0, 2.0, 0.0)),
            "reverse strong": dict(speed=observation(-2.0)),
            "reverse hold": dict(speed=observation(-0.01)),
            "target stop moving": dict(tgt=target(0.0), speed=observation(1.0)),
            "target stop at rest": dict(tgt=target(0.0), speed=observation(0.0)),
        }
        for label, kwargs in cases.items():
            with self.subTest(label):
                request = self.drive(**kwargs)
                self.assertIs(request.mode, LongitudinalControlMode.STOPPING)
                self.assertTrue(request.stop_required)
                self.assertGreater(request.brake_request, 0.0)
                self.assertLessEqual(request.brake_request, CONFIG.max_service_brake)
                self.assertLessEqual(request.brake_request, request.brake_authority)
                self.assertEqual(request.throttle_request, 0.0)
                self.assertEqual(request.throttle_authority, 0.0)
                self.assertFalse(request.longitudinal_allowed)
                for name in ("guidance_stamp", "speed_stamp", "snapshot_revision",
                             "target_origin", "target_epoch", "target_revision",
                             "target_issued_monotonic",
                             "target_valid_until_monotonic", "safety_state",
                             "lateral_mode", "temporal_state",
                             "evaluation_monotonic"):
                    self.assertIsNotNone(getattr(request, name), f"{label}:{name}")

    def test_ordinary_overspeed_braking_is_not_a_stop_request(self):
        request = self.drive(speed=observation(2.8), tgt=target(2.0))
        self.assertTrue(request.longitudinal_allowed)
        self.assertGreater(request.brake_request, 0.0)
        self.assertFalse(request.stop_required)
        self.assertIs(request.mode, LongitudinalControlMode.NOMINAL)

    def test_a_neutral_refusal_never_asserts_a_stop(self):
        for kwargs, expected in (
            (dict(speed=observation(1.0, 0.0, 1.0)),
             LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION),
            (dict(speed=observation(1.0, alive=False)),
             LongitudinalControlReason.VEHICLE_NOT_ALIVE),
            (dict(lat=lateral(revision=9)),
             LongitudinalControlReason.IDENTITY_MISMATCH),
        ):
            with self.subTest(expected):
                request = self.drive(**kwargs)
                self.assertNeutral(request, expected)
                self.assertFalse(request.stop_required)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    unittest.main()
