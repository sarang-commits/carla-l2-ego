"""Deterministic unit tests for the Mission 10B command safety envelope.

Every input here is an immutable synthetic record built from the accepted
Mission 7/8/9/10A public types, or a deliberately hostile record that no accepted
constructor would ever build.  No CARLA package, server, socket, worker, or wall
clock is involved: control time is always supplied explicitly.

The shaping expectations use a 0.25 s control step and the default rates, so
every anchor, allowance, and published pedal value below is exactly representable
in IEEE-754 double precision and is compared exactly rather than approximately.
"""

from __future__ import annotations

import ast
import contextlib
import gc
import inspect
import math
import numbers
import os
import subprocess
import sys
import textwrap
import unittest
import weakref
from dataclasses import FrozenInstanceError, fields, replace

from control.command import (
    DEFAULT_COMMAND_ENVELOPE_CONFIG,
    MAX_ENVELOPE_RATE_PER_SECOND,
    MAX_SHAPING_DELTA_SECONDS,
    CommandEnvelope,
    CommandEnvelopeConfig,
    CommandEnvelopeMetrics,
    CommandEnvelopeMode,
    CommandEnvelopeReason,
    CommandEnvelopeRequest,
    _LATERAL_REASON_MODE,
    _LONGITUDINAL_REASON_MODE,
    _MalformedEvidence,
    _REASON_MODE,
    _SPEED_LAW_SHAPE,
    _Transition,
    _canonicalize_lateral_request,
    _canonicalize_longitudinal_request,
    _clamp_symmetric,
    _release_limited,
    _rise_limited,
)
from control.lateral import (
    LateralControlMode,
    LateralControlReason,
    LateralControlRequest,
)
from control.longitudinal import (
    HARD_TARGET_CEILING_MPS,
    LongitudinalControlMode,
    LongitudinalControlReason,
    LongitudinalControlRequest,
    TargetSpeedOrigin,
)
from perception.temporal import TemporalLaneState
from safety.supervisor import SafetyState
from telemetry.state import SampleStamp


BASE = 100.0
STEP = 0.25
CONFIG = DEFAULT_COMMAND_ENVELOPE_CONFIG

# Upstream authorities the accepted Mission 9 and Mission 10A defaults publish.
LATERAL_AUTHORITY = 0.35
THROTTLE_AUTHORITY = 0.35
BRAKE_AUTHORITY = 0.40


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


class SneakyInt(int):
    """An int subclass that must never be accepted where an exact int is due."""


class HostileIndex:
    """An object that would answer ``__index__`` if anything ever asked."""

    def __index__(self) -> int:  # pragma: no cover - must never be called
        raise AssertionError("__index__ must never be consulted")


class ConfigSubclass(CommandEnvelopeConfig):
    """A caller-owned config subtype that must never be retained as-is."""


def unvalidated(base, **overrides):
    """Return a validation-bypassing clone of one accepted frozen record.

    Mission 10B must survive evidence that no accepted constructor would ever
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
    """Return a record of the **exact** accepted type with its constructor skipped."""
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


def drifting(base, name: str, *values):
    """Return a clone whose ``name`` attribute answers a new value each read."""
    clone = unvalidated(base)
    remaining = list(values)

    def getter(self):
        if len(remaining) > 1:
            return remaining.pop(0)
        return remaining[0]

    setattr(type(clone), name, property(getter))
    return clone


def counting(base, name: str):
    """Return ``(clone, counter)`` where reading ``name`` increments the counter."""
    clone = unvalidated(base)
    stored = getattr(base, name)
    counter = [0]

    def getter(self):
        counter[0] += 1
        return stored

    setattr(type(clone), name, property(getter))
    return clone, counter


# --------------------------------------------------------------------------
# Accepted-record builders
# --------------------------------------------------------------------------


def cut(index: int = 0, *, now: float | None = None):
    """One coherent evidence cut: control time, both stamps, and a revision."""
    control_now = BASE + index * STEP if now is None else now
    frame = 5 + index
    sim_time = frame / 4.0
    guidance = SampleStamp(
        frame=frame, sim_time=sim_time, monotonic=control_now - 0.05
    )
    speed = SampleStamp(frame=frame, sim_time=sim_time, monotonic=control_now - 0.02)
    return control_now, guidance, speed, frame


def lateral(
    index: int = 0,
    *,
    now: float | None = None,
    mode: LateralControlMode = LateralControlMode.NOMINAL,
    reason: LateralControlReason = LateralControlReason.NOMINAL_TRACKING,
    safety: SafetyState | None = SafetyState.NOMINAL,
    temporal: TemporalLaneState | None = TemporalLaneState.TRACKING,
    steering: float = 0.10,
    authority: float = LATERAL_AUTHORITY,
    stamp: object = ...,
    revision: object = ...,
    evaluated: object = ...,
    clamped: bool = False,
    rate_limited: bool = False,
) -> LateralControlRequest:
    """One accepted Mission 9 request for the cut at ``index``."""
    control_now, guidance, _speed, frame = cut(index, now=now)
    allowed = mode in (LateralControlMode.NOMINAL, LateralControlMode.DEGRADED)
    kwargs = dict(
        mode=mode,
        reason=reason,
        lateral_allowed=allowed,
        evaluation_monotonic=control_now if evaluated is ... else evaluated,
        source_stamp=guidance if stamp is ... else stamp,
        snapshot_revision=frame if revision is ... else revision,
        safety_state=safety,
        temporal_state=temporal,
    )
    if allowed:
        kwargs.update(
            steering=steering,
            raw_steering=steering,
            steering_authority=authority,
            clamped=clamped,
            rate_limited=rate_limited,
            lateral_error=0.05,
            heading_error=0.0,
        )
    return LateralControlRequest(**kwargs)


# The safety state each bound Mission 9 refusal genuinely publishes, so a fixture
# that does not care which one it gets still gets a reachable record.  Read off
# ``control/lateral.py``: the geometry branch is only ever reached under a
# degraded permission, the safety branch only when the gate failed, and the
# source-order branch before the gate was read at all.
DEFAULT_REFUSAL_SAFETY = {
    LateralControlReason.INSUFFICIENT_GEOMETRY: SafetyState.DEGRADED,
    LateralControlReason.SAFETY_NOT_PERMITTED: SafetyState.FAIL_SAFE,
    LateralControlReason.SOURCE_REJECTED: None,
    LateralControlReason.IDENTITY_MISMATCH: SafetyState.NOMINAL,
    LateralControlReason.EVIDENCE_MALFORMED: SafetyState.NOMINAL,
}


def refused_lateral(
    index: int = 0,
    *,
    reason: LateralControlReason = LateralControlReason.INSUFFICIENT_GEOMETRY,
    **kwargs,
) -> LateralControlRequest:
    kwargs.setdefault("safety", DEFAULT_REFUSAL_SAFETY.get(reason, SafetyState.NOMINAL))
    return lateral(index, mode=LateralControlMode.DISABLED, reason=reason, **kwargs)


def binding(
    index: int = 0,
    *,
    now: float | None = None,
    safety: SafetyState | None = SafetyState.NOMINAL,
    lateral_mode: LateralControlMode | None = LateralControlMode.NOMINAL,
    temporal: TemporalLaneState | None = TemporalLaneState.TRACKING,
    stamp: object = ...,
    speed_stamp: object = ...,
    revision: object = ...,
    evaluated: object = ...,
    origin: TargetSpeedOrigin | None = TargetSpeedOrigin.TEST,
    epoch: object = 1,
    target_revision: object = 1,
    issued: object = BASE - 0.5,
    valid_until: object = BASE + 50.0,
) -> dict[str, object]:
    """The exact twelve-field Mission 10A identity binding for one cut."""
    control_now, guidance, speed, frame = cut(index, now=now)
    return dict(
        evaluation_monotonic=control_now if evaluated is ... else evaluated,
        guidance_stamp=guidance if stamp is ... else stamp,
        speed_stamp=speed if speed_stamp is ... else speed_stamp,
        snapshot_revision=frame if revision is ... else revision,
        target_origin=origin,
        target_epoch=epoch,
        target_revision=target_revision,
        target_issued_monotonic=issued,
        target_valid_until_monotonic=valid_until,
        safety_state=safety,
        lateral_mode=lateral_mode,
        temporal_state=temporal,
    )


def permitted(
    index: int = 0,
    *,
    throttle: float = 0.20,
    brake: float = 0.0,
    target: float = 2.0,
    observed: float = 1.0,
    speed_authority: float = 3.0,
    mode: LongitudinalControlMode = LongitudinalControlMode.NOMINAL,
    reason: LongitudinalControlReason = LongitudinalControlReason.NOMINAL_TRACKING,
    throttle_authority: float = THROTTLE_AUTHORITY,
    brake_authority: float = BRAKE_AUTHORITY,
    throttle_clamped: bool = False,
    brake_clamped: bool = False,
    error: object = ...,
    **kwargs,
) -> LongitudinalControlRequest:
    """One accepted, permitted Mission 10A request for the cut at ``index``."""
    return LongitudinalControlRequest(
        mode=mode,
        reason=reason,
        longitudinal_allowed=True,
        stop_required=False,
        throttle_request=throttle,
        brake_request=brake,
        effective_target_speed_mps=target,
        observed_forward_speed_mps=observed,
        speed_error_mps=(target - observed) if error is ... else error,
        speed_authority_mps=speed_authority,
        throttle_authority=throttle_authority,
        brake_authority=brake_authority,
        throttle_clamped=throttle_clamped,
        brake_clamped=brake_clamped,
        **binding(index, **kwargs),
    )


def stopping(
    index: int = 0,
    *,
    brake: float = 0.20,
    reason: LongitudinalControlReason = LongitudinalControlReason.SPEED_STALE,
    observed: float = 1.0,
    brake_authority: float = BRAKE_AUTHORITY,
    brake_clamped: bool = False,
    **kwargs,
) -> LongitudinalControlRequest:
    """One accepted, trusted Mission 10A controlled stop for the cut.

    A lateral stop answers a Mission 9 geometry refusal, which is only ever
    reached under a degraded Mission 8 permission, so that is the binding this
    builder pairs with it unless the caller says otherwise.
    """
    if reason is LongitudinalControlReason.LATERAL_STOP:
        kwargs.setdefault("safety", SafetyState.DEGRADED)
    return LongitudinalControlRequest(
        mode=LongitudinalControlMode.STOPPING,
        reason=reason,
        longitudinal_allowed=False,
        stop_required=True,
        throttle_request=0.0,
        brake_request=brake,
        observed_forward_speed_mps=observed,
        throttle_authority=0.0,
        brake_authority=brake_authority,
        brake_clamped=brake_clamped,
        **binding(index, **kwargs),
    )


def target_stop(
    index: int = 0, *, brake: float = 0.10, observed: float = 0.0, **kwargs
) -> LongitudinalControlRequest:
    """The one Mission 10A stop that also publishes complete speed-law evidence."""
    return LongitudinalControlRequest(
        mode=LongitudinalControlMode.STOPPING,
        reason=LongitudinalControlReason.TARGET_STOP,
        longitudinal_allowed=False,
        stop_required=True,
        throttle_request=0.0,
        brake_request=brake,
        effective_target_speed_mps=0.0,
        observed_forward_speed_mps=observed,
        speed_error_mps=0.0 - observed,
        speed_authority_mps=3.0,
        throttle_authority=0.0,
        brake_authority=BRAKE_AUTHORITY,
        **binding(index, **kwargs),
    )


def bound_neutral(index: int = 0, *, observed: float = 1.0, **kwargs):
    """The one Mission 10A neutral refusal that still publishes its binding."""
    return LongitudinalControlRequest(
        mode=LongitudinalControlMode.DISABLED,
        reason=LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION,
        longitudinal_allowed=False,
        stop_required=False,
        observed_forward_speed_mps=observed,
        **binding(index, **kwargs),
    )


def unbound_refusal(
    index: int = 0,
    *,
    reason: LongitudinalControlReason = LongitudinalControlReason.SPEED_UNAVAILABLE,
    now: float | None = None,
    evaluated: object = ...,
) -> LongitudinalControlRequest:
    """One Mission 10A refusal that carries no identity binding at all."""
    control_now, _guidance, _speed, _frame = cut(index, now=now)
    return LongitudinalControlRequest(
        mode=LongitudinalControlMode.DISABLED,
        reason=reason,
        longitudinal_allowed=False,
        evaluation_monotonic=control_now if evaluated is ... else evaluated,
    )


def bound_provenance(**overrides):
    """The detached provenance every actionable envelope publishes."""
    base = dict(
        evidence_bound=True,
        evaluation_monotonic=BASE,
        guidance_stamp=SampleStamp(frame=5, sim_time=1.25, monotonic=BASE - 0.05),
        speed_stamp=SampleStamp(frame=5, sim_time=1.25, monotonic=BASE - 0.02),
        snapshot_revision=5,
        target_origin=TargetSpeedOrigin.TEST,
        target_epoch=1,
        target_revision=1,
        target_issued_monotonic=BASE - 0.5,
        target_valid_until_monotonic=BASE + 50.0,
        safety_state=SafetyState.NOMINAL,
        temporal_state=TemporalLaneState.TRACKING,
        lateral_mode=LateralControlMode.NOMINAL,
        lateral_reason=LateralControlReason.NOMINAL_TRACKING,
        longitudinal_mode=LongitudinalControlMode.NOMINAL,
        longitudinal_reason=LongitudinalControlReason.NOMINAL_TRACKING,
        observed_forward_speed_mps=1.0,
    )
    base.update(overrides)
    return base


def command_provenance(*, degraded: bool = False, **overrides):
    """A reachable permitted-pair provenance, nominal or degraded."""
    if degraded:
        overrides.setdefault("safety_state", SafetyState.DEGRADED)
        overrides.setdefault("lateral_mode", LateralControlMode.DEGRADED)
        overrides.setdefault(
            "lateral_reason", LateralControlReason.DEGRADED_TRACKING
        )
        overrides.setdefault(
            "longitudinal_mode", LongitudinalControlMode.DEGRADED
        )
        overrides.setdefault(
            "longitudinal_reason", LongitudinalControlReason.DEGRADED_TRACKING
        )
    return bound_provenance(**overrides)


def stop_provenance(**overrides):
    """A reachable trusted-stop provenance with a permitted Mission 9 record."""
    overrides.setdefault("longitudinal_mode", LongitudinalControlMode.STOPPING)
    overrides.setdefault(
        "longitudinal_reason", LongitudinalControlReason.SPEED_STALE
    )
    if overrides["longitudinal_reason"] is LongitudinalControlReason.LATERAL_STOP:
        # The only Mission 9 record a lateral stop can answer.
        overrides.setdefault("safety_state", SafetyState.DEGRADED)
        overrides.setdefault("lateral_mode", LateralControlMode.DISABLED)
        overrides.setdefault(
            "lateral_reason", LateralControlReason.INSUFFICIENT_GEOMETRY
        )
    return bound_provenance(**overrides)


def replay(index: int, now: float, **longitudinal_kwargs):
    """The same evidence cut, re-presented unchanged at a later control time.

    Only the control instant moves: both stamps, the revision, and the target
    identity are exactly the ones the cut published the first time, which is
    what makes the pair a legitimate duplicate rather than a mutated replay.
    """
    _first, guidance, speed, _frame = cut(index)
    return (
        lateral(index, now=now, stamp=guidance),
        permitted(
            index,
            now=now,
            stamp=guidance,
            speed_stamp=speed,
            **longitudinal_kwargs,
        ),
    )


NOT_EVALUATED_LATERAL = LateralControlRequest(
    mode=LateralControlMode.DISABLED,
    reason=LateralControlReason.NOT_EVALUATED,
    lateral_allowed=False,
)
NOT_EVALUATED_LONGITUDINAL = LongitudinalControlRequest(
    mode=LongitudinalControlMode.DISABLED,
    reason=LongitudinalControlReason.NOT_EVALUATED,
    longitudinal_allowed=False,
)


# --------------------------------------------------------------------------
# Independent upstream oracles (R3)
# --------------------------------------------------------------------------
#
# These restate the accepted Mission 9 and Mission 10A branch behavior from
# ``control/lateral.py``, ``control/longitudinal.py``, ``safety/supervisor.py``,
# and ``perception/temporal.py``.  They are deliberately written out here rather
# than imported from ``control.command``: a test that asks the implementation
# what to expect proves nothing about whether the implementation is right.

# Mission 8's constructor makes ``autonomy_allowed`` exactly
# ``state in {NOMINAL, DEGRADED}`` and lets only a ``FAIL_SAFE`` decision latch,
# so Mission 9's permission gate fails on exactly these three states.
REAL_UNPERMITTED_SAFETY = (
    SafetyState.INITIALIZING,
    SafetyState.RECOVERING,
    SafetyState.FAIL_SAFE,
)
REAL_PERMITTED_SAFETY = (SafetyState.NOMINAL, SafetyState.DEGRADED)

# ``TemporalLaneObservation.__post_init__`` refuses to wrap an ``UNINITIALIZED``
# estimate at all, and Mission 9 reads its evidence only out of that carrier, so
# no bound Mission 9 record can carry ``UNINITIALIZED``.
REAL_BOUND_TEMPORAL = (
    TemporalLaneState.TRACKING,
    TemporalLaneState.COASTING,
    TemporalLaneState.LOST,
    TemporalLaneState.INPUT_UNUSABLE,
)
REAL_GUIDANCE_TEMPORAL = (TemporalLaneState.TRACKING, TemporalLaneState.COASTING)

# The exact bound Mission 9 (safety, temporal) shapes, branch by branch.
REAL_BOUND_LATERAL_SHAPES = {
    # `_refuse(SOURCE_REJECTED, evaluated, evidence=evidence)` -- no safety state
    # is passed, because the source-order gate precedes the Mission 8 gate.
    LateralControlReason.SOURCE_REJECTED: ((None,), REAL_BOUND_TEMPORAL),
    # `_refuse(SAFETY_NOT_PERMITTED, ..., safety_state=state or None)`.
    LateralControlReason.SAFETY_NOT_PERMITTED: (
        REAL_UNPERMITTED_SAFETY + (None,),
        REAL_BOUND_TEMPORAL,
    ),
    # The late Mission 8-gate-passed branches.
    LateralControlReason.IDENTITY_MISMATCH: (
        REAL_PERMITTED_SAFETY,
        REAL_BOUND_TEMPORAL,
    ),
    LateralControlReason.EVIDENCE_MALFORMED: (
        REAL_PERMITTED_SAFETY,
        REAL_BOUND_TEMPORAL,
    ),
    # Past the gate and past the identity comparison, so the decision agrees
    # about the temporal state.  Mission 8 is only ever NOMINAL over TRACKING
    # with a DETECTED raw state, and Mission 7 requires complete geometry for
    # exactly that pair, so a nominal cut always has guidance and this branch is
    # only ever reached under DEGRADED.
    LateralControlReason.INSUFFICIENT_GEOMETRY: (
        (SafetyState.DEGRADED,),
        REAL_GUIDANCE_TEMPORAL,
    ),
    LateralControlReason.NOMINAL_TRACKING: (
        (SafetyState.NOMINAL,),
        (TemporalLaneState.TRACKING,),
    ),
    # Mission 7 forbids an extrapolated TRACKING estimate, so nominal safety over
    # tracking always selects NOMINAL_TRACKING instead of this branch.
    LateralControlReason.DEGRADED_TRACKING: (
        (SafetyState.DEGRADED,),
        (TemporalLaneState.TRACKING,),
    ),
    # Mission 8 only ever calls a COASTING estimate degraded.
    LateralControlReason.DEGRADED_COASTING: (
        (SafetyState.DEGRADED,),
        (TemporalLaneState.COASTING,),
    ),
}

# The two Mission 9 branches reached before any evidence exists.
REAL_UNBOUND_LATERAL_REASONS = (
    LateralControlReason.IDENTITY_MISMATCH,
    LateralControlReason.EVIDENCE_MALFORMED,
)
# Mission 9's clock gate always refuses with ``evaluated=None``, and
# ``_clear_state`` publishes NOT_EVALUATED with nothing at all.
REAL_EVIDENCE_FREE_LATERAL_REASONS = (
    LateralControlReason.NOT_EVALUATED,
    LateralControlReason.TIME_INVALID,
)


def real_lateral_shape(reason, safety_state, temporal_state, bound) -> bool:
    """Independently: can accepted Mission 9 publish this exact shape?"""
    if reason in REAL_EVIDENCE_FREE_LATERAL_REASONS:
        return not bound and safety_state is None and temporal_state is None
    if not bound:
        return (
            reason in REAL_UNBOUND_LATERAL_REASONS
            and safety_state is None
            and temporal_state is None
        )
    states, temporals = REAL_BOUND_LATERAL_SHAPES[reason]
    return safety_state in states and temporal_state in temporals


# The bound Mission 10A outcomes each Mission 9 reason can accompany on one real
# cut, restated from the gate order in ``LongitudinalController.update``:
#
#   1 TIME_INVALID | 2 SPEED_UNAVAILABLE | 3 _read_evidence refusals
#   4 VEHICLE_NOT_ALIVE | 5 SOURCE_REJECTED          <- all unbound
#   6 EXCESSIVE_VERTICAL_MOTION                      <- bound, outranks all stops
#   7 FAIL_SAFE_STOP -> SAFETY_STOP -> LATERAL_STOP
#   8 TARGET_EXPIRED -> SPEED_STALE
#   9 EXCESSIVE_LATERAL_MOTION -> REVERSE_MOTION
#  10 the speed law: TARGET_STOP / NOMINAL_ / DEGRADED_ TRACKING / COASTING
#
# `_read_evidence` binds nothing unless the Mission 9 record carries a stamp, a
# revision, an evaluation time, a safety state, and a temporal state, and unless
# the decision agrees with it on the stamp, revision, safety state, and temporal
# state.  So every Mission 9 reason with an incomplete binding, and
# SOURCE_REJECTED (which publishes no safety state), has no bound partner at all.
# A bound IDENTITY_MISMATCH or EVIDENCE_MALFORMED is Mission 9 *stating* the
# decision does not describe this evidence; Mission 10A re-checks the same axes
# and must agree, so those have no bound partner either.
REAL_PERMITTED_PAIR_LONGITUDINAL = frozenset(
    (
        LongitudinalControlReason.NOMINAL_TRACKING,
        LongitudinalControlReason.DEGRADED_TRACKING,
        LongitudinalControlReason.DEGRADED_COASTING,
        LongitudinalControlReason.TARGET_STOP,
        LongitudinalControlReason.TARGET_EXPIRED,
        LongitudinalControlReason.SPEED_STALE,
        LongitudinalControlReason.EXCESSIVE_LATERAL_MOTION,
        LongitudinalControlReason.REVERSE_MOTION,
        LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION,
    )
)

REAL_SAME_CUT_LONGITUDINAL = {
    LateralControlReason.NOT_EVALUATED: frozenset(),
    LateralControlReason.TIME_INVALID: frozenset(),
    LateralControlReason.SOURCE_REJECTED: frozenset(),
    LateralControlReason.IDENTITY_MISMATCH: frozenset(),
    LateralControlReason.EVIDENCE_MALFORMED: frozenset(),
    LateralControlReason.SAFETY_NOT_PERMITTED: frozenset(
        (
            LongitudinalControlReason.SAFETY_STOP,
            LongitudinalControlReason.FAIL_SAFE_STOP,
            LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION,
        )
    ),
    LateralControlReason.INSUFFICIENT_GEOMETRY: frozenset(
        (
            LongitudinalControlReason.LATERAL_STOP,
            LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION,
        )
    ),
    LateralControlReason.NOMINAL_TRACKING: REAL_PERMITTED_PAIR_LONGITUDINAL,
    LateralControlReason.DEGRADED_TRACKING: REAL_PERMITTED_PAIR_LONGITUDINAL,
    LateralControlReason.DEGRADED_COASTING: REAL_PERMITTED_PAIR_LONGITUDINAL,
}


def retained_objects(envelope: CommandEnvelope) -> list[object]:
    """Every object transitively reachable from the envelope's own slots."""
    seen: dict[int, object] = {}
    pending = [
        getattr(envelope, name)
        for name in CommandEnvelope.__slots__
        if hasattr(envelope, name)
    ]
    while pending:
        value = pending.pop()
        if value is None or isinstance(value, (int, float, str, bool)):
            continue
        if id(value) in seen:
            continue
        seen[id(value)] = value
        for field in getattr(type(value), "__slots__", ()):
            if hasattr(value, field):
                pending.append(getattr(value, field))
        if isinstance(value, tuple):
            pending.extend(value)
    return list(seen.values())


def retained_ids(envelope: CommandEnvelope) -> set[int]:
    """Every object id reachable from the envelope's own retained slots."""
    return {id(value) for value in retained_objects(envelope)}


class EnvelopeTestCase(unittest.TestCase):
    """Shared assertions and drivers for the Mission 10B suite."""

    def assertNeutral(self, request, reason):
        self.assertIs(request.reason, reason)
        self.assertIs(request.mode, CommandEnvelopeMode.DISABLED)
        self.assertFalse(request.command_allowed)
        self.assertFalse(request.stop_required)
        self.assertFalse(request.evidence_bound)
        self.assertFalse(request.shaping_applied)
        for name in (
            "steering_request",
            "throttle_request",
            "brake_request",
            "steering_authority",
            "throttle_authority",
            "brake_authority",
        ):
            value = getattr(request, name)
            self.assertEqual(value, 0.0, name)
            self.assertEqual(math.copysign(1.0, value), 1.0, name)
        for name in (
            "steering_clamped",
            "throttle_clamped",
            "brake_clamped",
            "throttle_rate_limited",
            "brake_release_rate_limited",
            "transition_interlocked",
        ):
            self.assertFalse(getattr(request, name), name)
        for name in (
            "guidance_stamp",
            "speed_stamp",
            "snapshot_revision",
            "target_origin",
            "target_epoch",
            "target_revision",
            "target_issued_monotonic",
            "target_valid_until_monotonic",
            "safety_state",
            "temporal_state",
            "lateral_mode",
            "lateral_reason",
            "longitudinal_mode",
            "longitudinal_reason",
            "effective_target_speed_mps",
            "observed_forward_speed_mps",
            "speed_error_mps",
            "speed_authority_mps",
        ):
            self.assertIsNone(getattr(request, name), name)

    def assertBoundIdentity(self, request):
        self.assertTrue(request.evidence_bound)
        self.assertIsNotNone(request.evaluation_monotonic)
        for name in (
            "guidance_stamp",
            "speed_stamp",
            "snapshot_revision",
            "target_origin",
            "target_epoch",
            "target_revision",
            "target_issued_monotonic",
            "target_valid_until_monotonic",
            "safety_state",
            "temporal_state",
            "lateral_mode",
            "lateral_reason",
            "longitudinal_mode",
            "longitudinal_reason",
            "observed_forward_speed_mps",
        ):
            self.assertIsNotNone(getattr(request, name), name)

    def assertSafeAxes(self, request):
        self.assertFalse(
            request.throttle_request > 0.0 and request.brake_request > 0.0
        )
        self.assertLessEqual(abs(request.steering_request), request.steering_authority)
        self.assertLessEqual(request.throttle_request, request.throttle_authority)
        self.assertLessEqual(request.brake_request, request.brake_authority)
        for name in (
            "steering_request",
            "throttle_request",
            "brake_request",
            "steering_authority",
            "throttle_authority",
            "brake_authority",
        ):
            value = getattr(request, name)
            self.assertTrue(math.isfinite(value), name)
            self.assertIs(type(value), float, name)

    def drive(self, envelope=None, *, lat=None, lon=None, index=0, now=None):
        envelope = CommandEnvelope() if envelope is None else envelope
        control_now = BASE + index * STEP if now is None else now
        return envelope.update(
            lateral(index) if lat is None else lat,
            permitted(index) if lon is None else lon,
            now=control_now,
        )


# --------------------------------------------------------------------------
# A. Vocabulary
# --------------------------------------------------------------------------


class VocabularyTest(EnvelopeTestCase):
    def test_every_required_mode_exists(self):
        self.assertEqual(
            sorted(member.name for member in CommandEnvelopeMode),
            ["DEGRADED", "DISABLED", "NOMINAL", "STOPPING"],
        )

    def test_every_required_reason_exists(self):
        self.assertEqual(
            sorted(member.name for member in CommandEnvelopeReason),
            [
                "AXIS_CONFLICT",
                "CROSS_RECORD_IDENTITY_MISMATCH",
                "DEGRADED_COMMAND",
                "DEGRADED_SHAPED",
                "DEGRADED_TRANSITION_INTERLOCK",
                "INTERNAL_ERROR",
                "MALFORMED_LATERAL_EVIDENCE",
                "MALFORMED_LONGITUDINAL_EVIDENCE",
                "NOMINAL_COMMAND",
                "NOMINAL_SHAPED",
                "NOMINAL_TRANSITION_INTERLOCK",
                "NOT_EVALUATED",
                "SOURCE_REJECTED",
                "TIME_INVALID",
                "TRUSTED_UPSTREAM_STOP",
                "UPSTREAM_REFUSAL",
            ],
        )

    def test_every_reason_maps_to_exactly_one_mode(self):
        self.assertEqual(set(_REASON_MODE), set(CommandEnvelopeReason))
        for reason, mode in _REASON_MODE.items():
            self.assertIsInstance(mode, CommandEnvelopeMode)

    def test_the_disabled_reasons_are_exactly_the_fail_closed_set(self):
        disabled = {
            reason
            for reason, mode in _REASON_MODE.items()
            if mode is CommandEnvelopeMode.DISABLED
        }
        self.assertEqual(
            sorted(reason.name for reason in disabled),
            [
                "AXIS_CONFLICT",
                "CROSS_RECORD_IDENTITY_MISMATCH",
                "INTERNAL_ERROR",
                "MALFORMED_LATERAL_EVIDENCE",
                "MALFORMED_LONGITUDINAL_EVIDENCE",
                "NOT_EVALUATED",
                "SOURCE_REJECTED",
                "TIME_INVALID",
                "UPSTREAM_REFUSAL",
            ],
        )

    def test_stopping_has_exactly_one_reason(self):
        stopping_reasons = [
            reason
            for reason, mode in _REASON_MODE.items()
            if mode is CommandEnvelopeMode.STOPPING
        ]
        self.assertEqual(
            stopping_reasons, [CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP]
        )

    def test_nominal_and_degraded_have_exactly_three_reasons_each(self):
        for mode in (CommandEnvelopeMode.NOMINAL, CommandEnvelopeMode.DEGRADED):
            with self.subTest(mode):
                reasons = [
                    reason
                    for reason, mapped in _REASON_MODE.items()
                    if mapped is mode
                ]
                self.assertEqual(len(reasons), 3)

    def test_every_reason_constructs_a_request_with_its_own_mode(self):
        for reason, mode in _REASON_MODE.items():
            with self.subTest(reason):
                request = self.build_for(reason, mode)
                self.assertIs(request.mode, mode)
                self.assertIs(request.reason, reason)

    def test_no_reason_constructs_a_request_with_a_foreign_mode(self):
        for reason, mode in _REASON_MODE.items():
            for other in CommandEnvelopeMode:
                if other is mode:
                    continue
                with self.subTest(reason=reason, mode=other):
                    with self.assertRaises(ValueError):
                        self.build_for(reason, other)

    def build_for(self, reason, mode):
        """One reachable fixture per reason, built the way Mission 10B builds it."""
        if mode is CommandEnvelopeMode.DISABLED:
            return CommandEnvelopeRequest(
                mode=mode, reason=reason, command_allowed=False
            )
        if mode is CommandEnvelopeMode.STOPPING:
            return CommandEnvelopeRequest(
                mode=mode,
                reason=reason,
                command_allowed=True,
                stop_required=True,
                brake_request=0.20,
                brake_authority=BRAKE_AUTHORITY,
                steering_authority=LATERAL_AUTHORITY,
                **stop_provenance(),
            )
        interlocked = reason in (
            CommandEnvelopeReason.NOMINAL_TRANSITION_INTERLOCK,
            CommandEnvelopeReason.DEGRADED_TRANSITION_INTERLOCK,
        )
        shaped = interlocked or reason in (
            CommandEnvelopeReason.NOMINAL_SHAPED,
            CommandEnvelopeReason.DEGRADED_SHAPED,
        )
        if mode is CommandEnvelopeMode.NOMINAL:
            provenance = command_provenance()
            authority = THROTTLE_AUTHORITY
        else:
            provenance = command_provenance(degraded=True)
            authority = 0.15
        return CommandEnvelopeRequest(
            mode=mode,
            reason=reason,
            command_allowed=True,
            # An interlocked envelope suppresses throttle, so the reachable
            # fixture for that reason carries none.
            throttle_request=0.0 if interlocked else 0.10,
            throttle_authority=authority,
            steering_authority=CONFIG.maximum_steering,
            brake_authority=CONFIG.maximum_ordinary_brake,
            transition_interlocked=interlocked,
            throttle_rate_limited=shaped and not interlocked,
            shaping_applied=shaped,
            effective_target_speed_mps=2.0,
            speed_error_mps=1.0,
            speed_authority_mps=3.0,
            **provenance,
        )


# --------------------------------------------------------------------------
# B. Configuration
# --------------------------------------------------------------------------


class ConfigurationTest(EnvelopeTestCase):
    def test_defaults_are_exactly_the_documented_offline_placeholders(self):
        self.assertEqual(CONFIG.maximum_steering, 0.35)
        self.assertEqual(CONFIG.maximum_throttle, 0.35)
        self.assertEqual(CONFIG.maximum_ordinary_brake, 0.40)
        self.assertEqual(CONFIG.throttle_rise_rate_per_second, 0.50)
        self.assertEqual(CONFIG.brake_release_rate_per_second, 1.00)
        self.assertEqual(CONFIG.maximum_shaping_delta_seconds, 0.25)

    def test_the_config_is_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            CONFIG.maximum_steering = 0.5

    def test_the_config_uses_slots_and_has_no_instance_dict(self):
        self.assertFalse(hasattr(CommandEnvelopeConfig(), "__dict__"))

    def test_every_ceiling_rejects_zero_and_negatives(self):
        for name in (
            "maximum_steering",
            "maximum_throttle",
            "maximum_ordinary_brake",
        ):
            for value in (0.0, -0.0, -0.1, -1.0):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        CommandEnvelopeConfig(**{name: value})

    def test_every_ceiling_rejects_above_one(self):
        for name in (
            "maximum_steering",
            "maximum_throttle",
            "maximum_ordinary_brake",
        ):
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    CommandEnvelopeConfig(**{name: 1.0 + 2.0 ** -52})

    def test_every_ceiling_accepts_exactly_one(self):
        for name in (
            "maximum_steering",
            "maximum_throttle",
            "maximum_ordinary_brake",
        ):
            with self.subTest(name):
                config = CommandEnvelopeConfig(**{name: 1.0})
                self.assertEqual(getattr(config, name), 1.0)

    def test_every_rate_rejects_zero_and_negatives(self):
        for name in (
            "throttle_rise_rate_per_second",
            "brake_release_rate_per_second",
        ):
            for value in (0.0, -1.0):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        CommandEnvelopeConfig(**{name: value})

    def test_every_rate_rejects_above_its_structural_bound(self):
        for name in (
            "throttle_rise_rate_per_second",
            "brake_release_rate_per_second",
        ):
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    CommandEnvelopeConfig(
                        **{name: MAX_ENVELOPE_RATE_PER_SECOND * 1.5}
                    )
                config = CommandEnvelopeConfig(**{name: MAX_ENVELOPE_RATE_PER_SECOND})
                self.assertEqual(getattr(config, name), MAX_ENVELOPE_RATE_PER_SECOND)

    def test_the_shaping_delta_is_strictly_positive_and_bounded(self):
        with self.assertRaises(ValueError):
            CommandEnvelopeConfig(maximum_shaping_delta_seconds=0.0)
        with self.assertRaises(ValueError):
            CommandEnvelopeConfig(
                maximum_shaping_delta_seconds=MAX_SHAPING_DELTA_SECONDS * 2.0
            )
        config = CommandEnvelopeConfig(
            maximum_shaping_delta_seconds=MAX_SHAPING_DELTA_SECONDS
        )
        self.assertEqual(
            config.maximum_shaping_delta_seconds, MAX_SHAPING_DELTA_SECONDS
        )

    def test_every_field_rejects_a_bool(self):
        for field in fields(CommandEnvelopeConfig):
            with self.subTest(field.name):
                with self.assertRaises(ValueError):
                    CommandEnvelopeConfig(**{field.name: True})

    def test_every_field_rejects_non_finite_values(self):
        for field in fields(CommandEnvelopeConfig):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=field.name, value=value):
                    with self.assertRaises(ValueError):
                        CommandEnvelopeConfig(**{field.name: value})

    def test_every_field_rejects_a_non_real(self):
        for field in fields(CommandEnvelopeConfig):
            with self.subTest(field.name):
                with self.assertRaises(ValueError):
                    CommandEnvelopeConfig(**{field.name: "0.5"})

    def test_a_float_subclass_is_canonicalized_into_a_builtin_float(self):
        config = CommandEnvelopeConfig(maximum_steering=SneakyFloat(0.25))
        self.assertIs(type(config.maximum_steering), float)
        self.assertEqual(config.maximum_steering, 0.25)

    def test_a_hostile_real_is_converted_exactly_once(self):
        config = CommandEnvelopeConfig(
            maximum_steering=DriftingReal(0.25, 0.90, 0.90)
        )
        self.assertEqual(config.maximum_steering, 0.25)
        self.assertIs(type(config.maximum_steering), float)

    def test_negative_zero_is_never_retained_by_the_config(self):
        # A negative zero cannot pass any of the strictly positive gates, so the
        # canonicalization is proven on the one value that can carry it through.
        config = CommandEnvelopeConfig(maximum_shaping_delta_seconds=0.25)
        self.assertEqual(math.copysign(1.0, config.maximum_shaping_delta_seconds), 1.0)

    def test_the_envelope_detaches_a_caller_config_subclass(self):
        supplied = ConfigSubclass(maximum_steering=0.20)
        envelope = CommandEnvelope(supplied)
        self.assertIs(type(envelope.config), CommandEnvelopeConfig)
        self.assertEqual(envelope.config.maximum_steering, 0.20)
        self.assertNotIn(id(supplied), retained_ids(envelope))

    def test_the_envelope_detaches_a_constructor_bypassed_config(self):
        hostile = bypassed(CommandEnvelopeConfig(), maximum_steering=5.0)
        with self.assertRaises(ValueError):
            CommandEnvelope(hostile)

    def test_the_config_property_returns_a_fresh_equal_record(self):
        envelope = CommandEnvelope()
        first = envelope.config
        second = envelope.config
        self.assertIsNot(first, second)
        self.assertEqual(first, second)

    def test_a_non_config_is_a_type_error(self):
        for value in (None, 0.35, "config", object()):
            with self.subTest(value):
                with self.assertRaises(TypeError):
                    CommandEnvelope(value)


# --------------------------------------------------------------------------
# C. Output record invariants
# --------------------------------------------------------------------------


class RequestInvariantTest(EnvelopeTestCase):
    def provenance(self, **overrides):
        return bound_provenance(**overrides)

    def command(self, **overrides):
        base = dict(
            mode=CommandEnvelopeMode.NOMINAL,
            reason=CommandEnvelopeReason.NOMINAL_COMMAND,
            command_allowed=True,
            throttle_request=0.10,
            throttle_authority=THROTTLE_AUTHORITY,
            brake_authority=CONFIG.maximum_ordinary_brake,
            steering_request=0.10,
            steering_authority=CONFIG.maximum_steering,
            effective_target_speed_mps=2.0,
            speed_error_mps=1.0,
            speed_authority_mps=3.0,
            **command_provenance(),
        )
        base.update(overrides)
        return CommandEnvelopeRequest(**base)

    def stop(self, **overrides):
        base = dict(
            mode=CommandEnvelopeMode.STOPPING,
            reason=CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP,
            command_allowed=True,
            stop_required=True,
            brake_request=0.20,
            brake_authority=BRAKE_AUTHORITY,
            # The bound Mission 9 record is permitted, so its steering authority
            # survives the stop and is strictly positive.
            steering_authority=LATERAL_AUTHORITY,
            **stop_provenance(),
        )
        base.update(overrides)
        return CommandEnvelopeRequest(**base)

    def test_a_well_formed_command_is_accepted(self):
        request = self.command()
        self.assertTrue(request.command_allowed)
        self.assertFalse(request.stop_required)
        self.assertBoundIdentity(request)
        self.assertSafeAxes(request)

    def test_the_request_is_frozen_and_slotted(self):
        request = self.command()
        with self.assertRaises(FrozenInstanceError):
            request.throttle_request = 0.9
        self.assertFalse(hasattr(request, "__dict__"))

    def test_the_record_publishes_exactly_the_required_field_set(self):
        self.assertEqual(
            tuple(field.name for field in fields(CommandEnvelopeRequest)),
            (
                "mode",
                "reason",
                "command_allowed",
                "stop_required",
                "steering_request",
                "throttle_request",
                "brake_request",
                "steering_authority",
                "throttle_authority",
                "brake_authority",
                "steering_clamped",
                "throttle_clamped",
                "brake_clamped",
                "throttle_rate_limited",
                "brake_release_rate_limited",
                "transition_interlocked",
                "shaping_applied",
                "evaluation_monotonic",
                "controller_epoch",
                "evidence_bound",
                "guidance_stamp",
                "speed_stamp",
                "snapshot_revision",
                "target_origin",
                "target_epoch",
                "target_revision",
                "target_issued_monotonic",
                "target_valid_until_monotonic",
                "safety_state",
                "temporal_state",
                "lateral_mode",
                "lateral_reason",
                "longitudinal_mode",
                "longitudinal_reason",
                "effective_target_speed_mps",
                "observed_forward_speed_mps",
                "speed_error_mps",
                "speed_authority_mps",
            ),
        )

    def test_command_allowed_must_match_the_mode(self):
        with self.assertRaises(ValueError):
            self.command(command_allowed=False)
        with self.assertRaises(ValueError):
            CommandEnvelopeRequest(
                mode=CommandEnvelopeMode.DISABLED,
                reason=CommandEnvelopeReason.UPSTREAM_REFUSAL,
                command_allowed=True,
            )

    def test_stop_required_must_match_the_mode(self):
        with self.assertRaises(ValueError):
            self.stop(stop_required=False)
        with self.assertRaises(ValueError):
            self.command(stop_required=True)

    def test_evidence_bound_must_match_the_mode(self):
        with self.assertRaises(ValueError):
            self.command(evidence_bound=False)
        with self.assertRaises(ValueError):
            CommandEnvelopeRequest(
                mode=CommandEnvelopeMode.DISABLED,
                reason=CommandEnvelopeReason.UPSTREAM_REFUSAL,
                command_allowed=False,
                evidence_bound=True,
            )

    def test_shaping_applied_is_the_exact_disjunction_of_the_six_flags(self):
        fixtures = {
            "steering_clamped": dict(steering_request=LATERAL_AUTHORITY),
            "throttle_clamped": dict(throttle_request=THROTTLE_AUTHORITY),
            "brake_clamped": dict(
                throttle_request=0.0,
                brake_request=BRAKE_AUTHORITY,
            ),
            "throttle_rate_limited": dict(throttle_request=0.0),
            "brake_release_rate_limited": dict(
                throttle_request=0.0,
                brake_request=0.10,
            ),
            "transition_interlocked": dict(throttle_request=0.0),
        }
        for name, fixture in fixtures.items():
            interlocked = name == "transition_interlocked"
            with self.subTest(flag=name, shaping=True):
                request = self.command(
                    **fixture,
                    **{name: True},
                    shaping_applied=True,
                    reason=(
                        CommandEnvelopeReason.NOMINAL_TRANSITION_INTERLOCK
                        if interlocked
                        else CommandEnvelopeReason.NOMINAL_SHAPED
                    ),
                )
                self.assertTrue(request.shaping_applied)
            with self.subTest(flag=name, shaping=False):
                with self.assertRaises(ValueError):
                    self.command(
                        **fixture,
                        **{name: True},
                        shaping_applied=False,
                    )

    def test_shaping_applied_cannot_be_claimed_without_any_flag(self):
        with self.assertRaises(ValueError):
            self.command(shaping_applied=True)

    def test_every_reachable_pedal_flag_shape_is_constructible(self):
        reachable = {
            (False, False, False, False, False): {},
            (True, False, False, False, False): {
                "throttle_request": THROTTLE_AUTHORITY,
            },
            (False, True, False, False, False): {
                "throttle_request": 0.0,
                "brake_request": BRAKE_AUTHORITY,
            },
            (False, False, True, False, False): {"throttle_request": 0.0},
            (True, False, True, False, False): {"throttle_request": 0.0},
            (False, False, False, True, False): {
                "throttle_request": 0.0,
                "brake_request": 0.10,
            },
            (False, False, False, False, True): {"throttle_request": 0.0},
            (True, False, False, False, True): {"throttle_request": 0.0},
            (False, False, False, True, True): {
                "throttle_request": 0.0,
                "brake_request": 0.10,
            },
            (True, False, False, True, True): {
                "throttle_request": 0.0,
                "brake_request": 0.10,
            },
        }
        names = (
            "throttle_clamped",
            "brake_clamped",
            "throttle_rate_limited",
            "brake_release_rate_limited",
            "transition_interlocked",
        )
        for shape, axes in reachable.items():
            for steering_clamped in (False, True):
                with self.subTest(shape=shape, steering=steering_clamped):
                    flags = dict(zip(names, shape))
                    flags["steering_clamped"] = steering_clamped
                    shaped = any(flags.values())
                    reason = (
                        CommandEnvelopeReason.NOMINAL_TRANSITION_INTERLOCK
                        if flags["transition_interlocked"]
                        else CommandEnvelopeReason.NOMINAL_SHAPED
                        if shaped
                        else CommandEnvelopeReason.NOMINAL_COMMAND
                    )
                    request = self.command(
                        **axes,
                        **flags,
                        steering_request=(
                            LATERAL_AUTHORITY if steering_clamped else 0.10
                        ),
                        shaping_applied=shaped,
                        reason=reason,
                    )
                    self.assertEqual(request.shaping_applied, shaped)

    def test_every_other_pedal_flag_shape_is_rejected(self):
        reachable = {
            (False, False, False, False, False),
            (True, False, False, False, False),
            (False, True, False, False, False),
            (False, False, True, False, False),
            (True, False, True, False, False),
            (False, False, False, True, False),
            (False, False, False, False, True),
            (True, False, False, False, True),
            (False, False, False, True, True),
            (True, False, False, True, True),
        }
        names = (
            "throttle_clamped",
            "brake_clamped",
            "throttle_rate_limited",
            "brake_release_rate_limited",
            "transition_interlocked",
        )
        for mask in range(32):
            shape = tuple(bool(mask & (1 << bit)) for bit in range(5))
            if shape in reachable:
                continue
            for steering_clamped in (False, True):
                with self.subTest(shape=shape, steering=steering_clamped):
                    flags = dict(zip(names, shape))
                    flags["steering_clamped"] = steering_clamped
                    with self.assertRaises(ValueError):
                        self.command(
                            **flags,
                            throttle_request=0.0,
                            steering_request=(
                                LATERAL_AUTHORITY if steering_clamped else 0.10
                            ),
                            shaping_applied=True,
                            reason=(
                                CommandEnvelopeReason.NOMINAL_TRANSITION_INTERLOCK
                                if flags["transition_interlocked"]
                                else CommandEnvelopeReason.NOMINAL_SHAPED
                            ),
                        )

    def test_shaping_flags_prove_their_axis_relationships(self):
        mutations = (
            dict(
                steering_clamped=True,
                steering_request=0.10,
                shaping_applied=True,
                reason=CommandEnvelopeReason.NOMINAL_SHAPED,
            ),
            dict(
                throttle_clamped=True,
                throttle_request=0.10,
                shaping_applied=True,
                reason=CommandEnvelopeReason.NOMINAL_SHAPED,
            ),
            dict(
                brake_clamped=True,
                throttle_request=0.0,
                brake_request=0.10,
                shaping_applied=True,
                reason=CommandEnvelopeReason.NOMINAL_SHAPED,
            ),
            dict(
                brake_release_rate_limited=True,
                throttle_request=0.0,
                brake_request=0.0,
                shaping_applied=True,
                reason=CommandEnvelopeReason.NOMINAL_SHAPED,
            ),
            dict(
                throttle_rate_limited=True,
                throttle_request=THROTTLE_AUTHORITY,
                shaping_applied=True,
                reason=CommandEnvelopeReason.NOMINAL_SHAPED,
            ),
            dict(
                transition_interlocked=True,
                throttle_request=0.0,
                brake_request=0.10,
                shaping_applied=True,
                reason=CommandEnvelopeReason.NOMINAL_TRANSITION_INTERLOCK,
            ),
            dict(
                transition_interlocked=True,
                brake_release_rate_limited=True,
                throttle_request=0.0,
                brake_request=0.0,
                shaping_applied=True,
                reason=CommandEnvelopeReason.NOMINAL_TRANSITION_INTERLOCK,
            ),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    self.command(**mutation)

    def test_real_outputs_reconstruct_under_the_complete_flag_contract(self):
        envelope = CommandEnvelope()
        outputs = [
            envelope.update(lateral(0), permitted(0), now=BASE),
            envelope.update(
                lateral(1),
                permitted(1, throttle=0.0, brake=0.30, target=1.0, observed=2.0),
                now=BASE + STEP,
            ),
            envelope.update(
                lateral(2), permitted(2, throttle=0.30), now=BASE + 2 * STEP
            ),
            envelope.update(
                lateral(3), permitted(3, throttle=0.30), now=BASE + 3 * STEP
            ),
        ]
        for request in outputs:
            with self.subTest(flags=tuple(
                getattr(request, name)
                for name in (
                    "steering_clamped",
                    "throttle_clamped",
                    "brake_clamped",
                    "throttle_rate_limited",
                    "brake_release_rate_limited",
                    "transition_interlocked",
                )
            )):
                self.assertEqual(replace(request), request)

    def test_no_axis_may_exceed_its_own_authority(self):
        with self.assertRaises(ValueError):
            self.command(steering_request=0.90, steering_authority=0.35)
        with self.assertRaises(ValueError):
            self.command(steering_request=-0.90, steering_authority=0.35)
        with self.assertRaises(ValueError):
            self.command(throttle_request=0.90, throttle_authority=0.35)
        with self.assertRaises(ValueError):
            self.stop(brake_request=0.90, brake_authority=0.40)

    def test_one_ulp_of_authority_overshoot_is_rejected(self):
        authority = 0.35
        overshoot = math.nextafter(authority, 1.0)
        with self.assertRaises(ValueError):
            self.command(throttle_request=overshoot, throttle_authority=authority)
        request = self.command(throttle_request=authority, throttle_authority=authority)
        self.assertEqual(request.throttle_request, authority)

    def test_throttle_and_brake_can_never_both_be_positive(self):
        with self.assertRaises(ValueError):
            self.command(
                throttle_request=0.10,
                brake_request=0.10,
                brake_authority=CONFIG.maximum_ordinary_brake,
            )

    def test_a_stopping_envelope_carries_no_throttle_authority(self):
        with self.assertRaises(ValueError):
            self.stop(throttle_authority=0.10)
        with self.assertRaises(ValueError):
            self.stop(throttle_request=0.10, throttle_authority=0.35)

    def test_a_stopping_envelope_must_request_positive_brake(self):
        with self.assertRaises(ValueError):
            self.stop(brake_request=0.0)

    def test_a_refused_envelope_cannot_carry_any_axis(self):
        for name in (
            "steering_request",
            "throttle_request",
            "brake_request",
            "steering_authority",
            "throttle_authority",
            "brake_authority",
        ):
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    CommandEnvelopeRequest(
                        mode=CommandEnvelopeMode.DISABLED,
                        reason=CommandEnvelopeReason.UPSTREAM_REFUSAL,
                        command_allowed=False,
                        **{name: 0.10},
                    )

    def test_a_refused_envelope_cannot_report_shaping(self):
        with self.assertRaises(ValueError):
            CommandEnvelopeRequest(
                mode=CommandEnvelopeMode.DISABLED,
                reason=CommandEnvelopeReason.UPSTREAM_REFUSAL,
                command_allowed=False,
                steering_clamped=True,
                shaping_applied=True,
            )

    def test_a_refused_envelope_cannot_expose_any_provenance(self):
        for name, value in self.provenance(evidence_bound=False).items():
            if name == "evidence_bound":
                continue
            if name == "evaluation_monotonic":
                continue
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    CommandEnvelopeRequest(
                        mode=CommandEnvelopeMode.DISABLED,
                        reason=CommandEnvelopeReason.UPSTREAM_REFUSAL,
                        command_allowed=False,
                        **{name: value},
                    )

    def test_a_refusal_may_still_publish_its_own_evaluation_time(self):
        request = CommandEnvelopeRequest(
            mode=CommandEnvelopeMode.DISABLED,
            reason=CommandEnvelopeReason.UPSTREAM_REFUSAL,
            command_allowed=False,
            evaluation_monotonic=BASE,
            controller_epoch=3,
        )
        self.assertEqual(request.evaluation_monotonic, BASE)
        self.assertEqual(request.controller_epoch, 3)
        self.assertNeutral(request, CommandEnvelopeReason.UPSTREAM_REFUSAL)

    def test_an_actionable_envelope_requires_every_identity_field(self):
        for name in self.provenance():
            if name in ("evidence_bound", "observed_forward_speed_mps"):
                continue
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    self.stop(**{name: None})

    def test_an_actionable_envelope_requires_the_observed_forward_speed(self):
        with self.assertRaises(ValueError):
            self.stop(observed_forward_speed_mps=None)

    def test_speed_law_evidence_must_be_complete_or_absent(self):
        for name in (
            "effective_target_speed_mps",
            "speed_error_mps",
            "speed_authority_mps",
        ):
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    self.command(**{name: None})
                with self.assertRaises(ValueError):
                    self.stop(**{name: 1.0})

    def test_a_stop_may_omit_the_whole_speed_law_triple(self):
        request = self.stop()
        self.assertIsNone(request.effective_target_speed_mps)
        self.assertIsNone(request.speed_error_mps)
        self.assertIsNone(request.speed_authority_mps)
        self.assertEqual(request.observed_forward_speed_mps, 1.0)

    def test_a_stop_may_carry_the_whole_speed_law_triple(self):
        # Exactly one accepted stopping branch publishes the complete law, and
        # the record only accepts the triple for that branch.
        request = self.stop(
            longitudinal_reason=LongitudinalControlReason.TARGET_STOP,
            effective_target_speed_mps=0.0,
            observed_forward_speed_mps=1.0,
            speed_error_mps=-1.0,
            speed_authority_mps=3.0,
        )
        self.assertEqual(request.effective_target_speed_mps, 0.0)
        with self.assertRaises(ValueError):
            self.stop(
                effective_target_speed_mps=0.0,
                speed_error_mps=-1.0,
                speed_authority_mps=3.0,
            )

    def test_a_permitted_envelope_requires_its_speed_law_evidence(self):
        with self.assertRaises(ValueError):
            self.command(
                effective_target_speed_mps=None,
                speed_error_mps=None,
                speed_authority_mps=None,
            )

    def test_an_effective_target_cannot_exceed_its_speed_authority(self):
        with self.assertRaises(ValueError):
            self.command(effective_target_speed_mps=4.0, speed_authority_mps=3.0)

    def test_the_speed_law_respects_the_hard_structural_ceiling(self):
        with self.assertRaises(ValueError):
            self.command(
                effective_target_speed_mps=HARD_TARGET_CEILING_MPS + 1.0,
                speed_authority_mps=HARD_TARGET_CEILING_MPS + 1.0,
            )

    def test_every_axis_rejects_a_bool(self):
        for name in (
            "steering_request",
            "throttle_request",
            "brake_request",
            "steering_authority",
            "throttle_authority",
            "brake_authority",
        ):
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    self.command(**{name: True})

    def test_every_flag_rejects_a_non_builtin_bool(self):
        for name in (
            "command_allowed",
            "stop_required",
            "evidence_bound",
            "steering_clamped",
            "throttle_clamped",
            "brake_clamped",
            "throttle_rate_limited",
            "brake_release_rate_limited",
            "transition_interlocked",
            "shaping_applied",
        ):
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    self.command(**{name: 1})

    def test_every_axis_rejects_non_finite_values(self):
        for name in ("steering_request", "throttle_request", "brake_request"):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        self.command(**{name: value})

    def test_negative_zero_is_folded_on_every_published_float(self):
        request = self.command(
            steering_request=-0.0,
            throttle_request=-0.0,
            observed_forward_speed_mps=2.0,
            speed_error_mps=-0.0,
            effective_target_speed_mps=2.0,
        )
        for name in (
            "steering_request",
            "throttle_request",
            "brake_request",
            "speed_error_mps",
        ):
            value = getattr(request, name)
            self.assertEqual(math.copysign(1.0, value), 1.0, name)
        stop = self.stop(
            longitudinal_reason=LongitudinalControlReason.TARGET_STOP,
            effective_target_speed_mps=-0.0,
            observed_forward_speed_mps=-0.0,
            speed_error_mps=-0.0,
            speed_authority_mps=3.0,
        )
        for name in (
            "effective_target_speed_mps",
            "observed_forward_speed_mps",
            "speed_error_mps",
        ):
            value = getattr(stop, name)
            self.assertEqual(math.copysign(1.0, value), 1.0, name)

    def test_the_controller_epoch_must_be_a_non_negative_builtin_int(self):
        for value in (-1, True, 1.0, SneakyInt(2), HostileIndex()):
            with self.subTest(value):
                with self.assertRaises(ValueError):
                    self.command(controller_epoch=value)

    def test_the_snapshot_revision_must_be_a_non_negative_builtin_int(self):
        for value in (-1, True, 5.0, SneakyInt(5), HostileIndex()):
            with self.subTest(value):
                with self.assertRaises(ValueError):
                    self.command(snapshot_revision=value)

    def test_every_provenance_enum_is_checked_for_its_exact_class(self):
        for name in (
            "safety_state",
            "temporal_state",
            "lateral_mode",
            "lateral_reason",
            "longitudinal_mode",
            "longitudinal_reason",
            "target_origin",
        ):
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    self.command(**{name: CommandEnvelopeMode.NOMINAL})

    def test_the_target_window_must_not_be_inverted(self):
        with self.assertRaises(ValueError):
            self.command(
                target_issued_monotonic=BASE + 1.0,
                target_valid_until_monotonic=BASE - 1.0,
            )

    def test_stamps_are_rebuilt_as_exact_base_instances(self):
        hostile = unvalidated(
            SampleStamp(frame=5, sim_time=1.25, monotonic=BASE - 0.05)
        )
        request = self.command(guidance_stamp=hostile)
        self.assertIs(type(request.guidance_stamp), SampleStamp)
        self.assertIsNot(request.guidance_stamp, hostile)
        self.assertEqual(request.guidance_stamp.frame, 5)

    def test_a_malformed_stamp_member_is_rejected(self):
        for member, value in (
            ("frame", -1),
            ("frame", 1.0),
            ("sim_time", float("nan")),
            ("monotonic", float("inf")),
        ):
            with self.subTest(member=member, value=value):
                hostile = bypassed(
                    SampleStamp(frame=5, sim_time=1.25, monotonic=BASE),
                    **{member: value},
                )
                with self.assertRaises(ValueError):
                    self.command(guidance_stamp=hostile)

    def test_the_frame_helpers_expose_the_bound_frames(self):
        request = self.command()
        self.assertEqual(request.guidance_frame, 5)
        self.assertEqual(request.speed_frame, 5)
        neutral = CommandEnvelopeRequest(
            mode=CommandEnvelopeMode.DISABLED,
            reason=CommandEnvelopeReason.NOT_EVALUATED,
            command_allowed=False,
        )
        self.assertIsNone(neutral.guidance_frame)
        self.assertIsNone(neutral.speed_frame)

    def test_the_mode_and_reason_must_be_the_exact_envelope_enums(self):
        with self.assertRaises(ValueError):
            self.command(mode=LongitudinalControlMode.NOMINAL)
        with self.assertRaises(ValueError):
            self.command(reason=LongitudinalControlReason.NOMINAL_TRACKING)


# --------------------------------------------------------------------------
# D. Mission 9 canonicalization
# --------------------------------------------------------------------------


class LateralCanonicalizationTest(EnvelopeTestCase):
    def test_a_valid_request_is_rebuilt_as_a_fresh_exact_base_record(self):
        supplied = lateral()
        canonical = _canonicalize_lateral_request(supplied)
        self.assertIs(type(canonical), LateralControlRequest)
        self.assertIsNot(canonical, supplied)
        self.assertEqual(canonical, supplied)

    def test_a_subclass_is_rebuilt_as_the_exact_base_type(self):
        supplied = unvalidated(lateral())
        canonical = _canonicalize_lateral_request(supplied)
        self.assertIs(type(canonical), LateralControlRequest)

    def test_the_nested_stamp_is_rebuilt_and_never_retained(self):
        stamp = SampleStamp(frame=5, sim_time=1.25, monotonic=BASE - 0.05)
        canonical = _canonicalize_lateral_request(lateral(stamp=stamp))
        self.assertIs(type(canonical.source_stamp), SampleStamp)
        self.assertIsNot(canonical.source_stamp, stamp)
        self.assertEqual(canonical.source_stamp, stamp)

    def test_a_foreign_type_is_malformed(self):
        for value in (None, object(), permitted()):
            with self.subTest(value):
                with self.assertRaises(_MalformedEvidence):
                    _canonicalize_lateral_request(value)

    def test_every_public_field_is_read_exactly_once(self):
        for field in fields(LateralControlRequest):
            with self.subTest(field.name):
                clone, counter = counting(lateral(), field.name)
                _canonicalize_lateral_request(clone)
                self.assertEqual(counter[0], 1, field.name)

    def test_every_raising_property_fails_closed(self):
        for field in fields(LateralControlRequest):
            with self.subTest(field.name):
                clone = exploding(lateral(), field.name, RuntimeError("boom"))
                with self.assertRaises(_MalformedEvidence):
                    _canonicalize_lateral_request(clone)

    def test_a_changing_numeric_property_cannot_answer_twice(self):
        clone = drifting(lateral(), "steering", 0.10, 0.30, 0.30)
        canonical = _canonicalize_lateral_request(clone)
        self.assertEqual(canonical.steering, 0.10)

    def test_a_hostile_real_is_converted_exactly_once(self):
        clone = bypassed(lateral(), steering=DriftingReal(0.10, 0.90, 0.90))
        canonical = _canonicalize_lateral_request(clone)
        self.assertEqual(canonical.steering, 0.10)
        self.assertIs(type(canonical.steering), float)

    def test_a_float_subclass_becomes_a_builtin_float(self):
        clone = bypassed(lateral(), steering=SneakyFloat(0.10))
        canonical = _canonicalize_lateral_request(clone)
        self.assertIs(type(canonical.steering), float)

    def test_negative_zero_is_folded(self):
        clone = bypassed(lateral(steering=0.0), steering=-0.0, raw_steering=-0.0)
        canonical = _canonicalize_lateral_request(clone)
        self.assertEqual(math.copysign(1.0, canonical.steering), 1.0)
        self.assertEqual(math.copysign(1.0, canonical.raw_steering), 1.0)

    def test_a_constructor_bypassed_record_is_re_proven(self):
        for overrides in (
            dict(steering=0.90),
            dict(steering_authority=2.0),
            dict(steering=float("nan")),
            dict(raw_steering=float("inf")),
            dict(lateral_allowed=False),
            dict(lateral_error=2.0),
            dict(heading_error=-2.0),
            dict(snapshot_revision=-1),
            dict(evaluation_monotonic=float("nan")),
            dict(safety_state=SafetyState.FAIL_SAFE),
            dict(temporal_state=TemporalLaneState.LOST),
            dict(clamped=1),
            dict(rate_limited="yes"),
        ):
            with self.subTest(overrides):
                clone = bypassed(lateral(), **overrides)
                with self.assertRaises(_MalformedEvidence):
                    _canonicalize_lateral_request(clone)

    def test_a_wrong_enum_object_is_malformed(self):
        for name, value in (
            ("mode", LongitudinalControlMode.NOMINAL),
            ("reason", LongitudinalControlReason.NOMINAL_TRACKING),
            ("safety_state", TemporalLaneState.TRACKING),
            ("temporal_state", SafetyState.NOMINAL),
            ("mode", "nominal"),
        ):
            with self.subTest(name=name, value=value):
                clone = bypassed(lateral(), **{name: value})
                with self.assertRaises(_MalformedEvidence):
                    _canonicalize_lateral_request(clone)

    def test_bool_is_never_accepted_as_a_numeric(self):
        for name in ("steering", "raw_steering", "steering_authority"):
            with self.subTest(name):
                clone = bypassed(lateral(), **{name: True})
                with self.assertRaises(_MalformedEvidence):
                    _canonicalize_lateral_request(clone)

    def test_an_int_subclass_is_never_accepted_as_a_revision(self):
        clone = bypassed(lateral(), snapshot_revision=SneakyInt(5))
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_lateral_request(clone)

    def test_index_is_never_consulted_for_a_revision(self):
        clone = bypassed(lateral(), snapshot_revision=HostileIndex())
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_lateral_request(clone)

    def test_a_mutable_stamp_is_read_once_and_detached(self):
        stamp = SampleStamp(frame=5, sim_time=1.25, monotonic=BASE - 0.05)
        hostile = drifting(stamp, "frame", 5, 99, 99)
        canonical = _canonicalize_lateral_request(lateral(stamp=hostile))
        self.assertEqual(canonical.source_stamp.frame, 5)
        self.assertIs(type(canonical.source_stamp), SampleStamp)

    def test_partial_source_evidence_is_malformed(self):
        for overrides in (
            dict(snapshot_revision=None),
            dict(temporal_state=None),
            dict(source_stamp=None),
        ):
            with self.subTest(overrides):
                clone = bypassed(
                    refused_lateral(reason=LateralControlReason.SOURCE_REJECTED,
                                    safety=None),
                    **overrides,
                )
                with self.assertRaises(_MalformedEvidence):
                    _canonicalize_lateral_request(clone)

    def test_a_safety_state_without_source_evidence_is_malformed(self):
        clone = bypassed(
            refused_lateral(reason=LateralControlReason.EVIDENCE_MALFORMED),
            source_stamp=None,
            snapshot_revision=None,
            temporal_state=None,
        )
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_lateral_request(clone)

    def test_a_time_refusal_must_expose_nothing_at_all(self):
        accepted = LateralControlRequest(
            mode=LateralControlMode.DISABLED,
            reason=LateralControlReason.TIME_INVALID,
            lateral_allowed=False,
        )
        self.assertIsNotNone(_canonicalize_lateral_request(accepted))
        clone = bypassed(accepted, evaluation_monotonic=BASE)
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_lateral_request(clone)

    def test_an_evaluated_refusal_requires_its_time(self):
        clone = bypassed(
            refused_lateral(reason=LateralControlReason.INSUFFICIENT_GEOMETRY),
            evaluation_monotonic=None,
        )
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_lateral_request(clone)

    def test_a_source_rejection_never_carries_a_safety_state(self):
        good = refused_lateral(
            reason=LateralControlReason.SOURCE_REJECTED, safety=None
        )
        self.assertIsNotNone(_canonicalize_lateral_request(good))
        clone = bypassed(good, safety_state=SafetyState.NOMINAL)
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_lateral_request(clone)

    def test_a_geometry_refusal_requires_a_permitted_safety_state(self):
        clone = bypassed(
            refused_lateral(reason=LateralControlReason.INSUFFICIENT_GEOMETRY),
            safety_state=SafetyState.FAIL_SAFE,
        )
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_lateral_request(clone)

    def test_a_permitted_request_requires_positive_authority(self):
        clone = bypassed(lateral(), steering=0.0, steering_authority=0.0)
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_lateral_request(clone)

    def test_coasting_and_tracking_reasons_must_match_the_temporal_state(self):
        clone = bypassed(
            lateral(
                mode=LateralControlMode.DEGRADED,
                reason=LateralControlReason.DEGRADED_COASTING,
                safety=SafetyState.DEGRADED,
                temporal=TemporalLaneState.COASTING,
            ),
            temporal_state=TemporalLaneState.TRACKING,
        )
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_lateral_request(clone)
        other = bypassed(
            lateral(
                mode=LateralControlMode.DEGRADED,
                reason=LateralControlReason.DEGRADED_TRACKING,
                safety=SafetyState.DEGRADED,
            ),
            temporal_state=TemporalLaneState.COASTING,
        )
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_lateral_request(other)

    def test_the_accepted_unevaluated_record_survives_canonicalization(self):
        canonical = _canonicalize_lateral_request(NOT_EVALUATED_LATERAL)
        self.assertIs(canonical.reason, LateralControlReason.NOT_EVALUATED)
        self.assertIsNot(canonical, NOT_EVALUATED_LATERAL)

    def test_canonicalization_does_not_mutate_the_caller_record(self):
        supplied = lateral()
        before = tuple(getattr(supplied, f.name) for f in fields(supplied))
        _canonicalize_lateral_request(supplied)
        after = tuple(getattr(supplied, f.name) for f in fields(supplied))
        self.assertEqual(before, after)


# --------------------------------------------------------------------------
# E. Mission 10A canonicalization
# --------------------------------------------------------------------------


class LongitudinalCanonicalizationTest(EnvelopeTestCase):
    def test_a_valid_request_is_rebuilt_as_a_fresh_exact_base_record(self):
        supplied = permitted()
        canonical = _canonicalize_longitudinal_request(supplied)
        self.assertIs(type(canonical), LongitudinalControlRequest)
        self.assertIsNot(canonical, supplied)
        self.assertEqual(canonical, supplied)

    def test_a_subclass_is_rebuilt_as_the_exact_base_type(self):
        canonical = _canonicalize_longitudinal_request(unvalidated(permitted()))
        self.assertIs(type(canonical), LongitudinalControlRequest)

    def test_a_foreign_type_is_malformed(self):
        for value in (None, object(), lateral()):
            with self.subTest(value):
                with self.assertRaises(_MalformedEvidence):
                    _canonicalize_longitudinal_request(value)

    def test_every_public_field_is_read_exactly_once(self):
        for field in fields(LongitudinalControlRequest):
            with self.subTest(field.name):
                clone, counter = counting(permitted(), field.name)
                _canonicalize_longitudinal_request(clone)
                self.assertEqual(counter[0], 1, field.name)

    def test_every_raising_property_fails_closed(self):
        for field in fields(LongitudinalControlRequest):
            with self.subTest(field.name):
                clone = exploding(permitted(), field.name, RuntimeError("boom"))
                with self.assertRaises(_MalformedEvidence):
                    _canonicalize_longitudinal_request(clone)

    def test_both_nested_stamps_are_rebuilt_and_never_retained(self):
        supplied = permitted()
        canonical = _canonicalize_longitudinal_request(supplied)
        self.assertIsNot(canonical.guidance_stamp, supplied.guidance_stamp)
        self.assertIsNot(canonical.speed_stamp, supplied.speed_stamp)
        self.assertIs(type(canonical.guidance_stamp), SampleStamp)
        self.assertIs(type(canonical.speed_stamp), SampleStamp)

    def test_a_changing_numeric_property_cannot_answer_twice(self):
        clone = drifting(permitted(), "throttle_request", 0.20, 0.90, 0.90)
        canonical = _canonicalize_longitudinal_request(clone)
        self.assertEqual(canonical.throttle_request, 0.20)

    def test_a_hostile_real_is_converted_exactly_once(self):
        clone = bypassed(
            permitted(), throttle_request=DriftingReal(0.20, 0.90, 0.90)
        )
        canonical = _canonicalize_longitudinal_request(clone)
        self.assertEqual(canonical.throttle_request, 0.20)

    def test_a_constructor_bypassed_record_is_re_proven(self):
        for overrides in (
            dict(throttle_request=0.90),
            dict(brake_request=0.90),
            dict(throttle_request=0.10, brake_request=0.10),
            dict(throttle_authority=2.0),
            dict(brake_authority=-1.0),
            dict(longitudinal_allowed=False),
            dict(stop_required=True),
            dict(effective_target_speed_mps=HARD_TARGET_CEILING_MPS + 1.0),
            dict(speed_authority_mps=-1.0),
            dict(snapshot_revision=-1),
            dict(target_epoch=-1),
            dict(target_revision=1.0),
            dict(target_issued_monotonic=float("nan")),
            dict(evaluation_monotonic=float("inf")),
            dict(throttle_clamped=1),
            dict(brake_clamped=None),
        ):
            with self.subTest(overrides):
                clone = bypassed(permitted(), **overrides)
                with self.assertRaises(_MalformedEvidence):
                    _canonicalize_longitudinal_request(clone)

    def test_a_wrong_enum_object_is_malformed(self):
        for name, value in (
            ("mode", LateralControlMode.NOMINAL),
            ("reason", LateralControlReason.NOMINAL_TRACKING),
            ("target_origin", SafetyState.NOMINAL),
            ("safety_state", TargetSpeedOrigin.TEST),
            ("lateral_mode", LongitudinalControlMode.NOMINAL),
            ("temporal_state", SafetyState.NOMINAL),
        ):
            with self.subTest(name=name, value=value):
                clone = bypassed(permitted(), **{name: value})
                with self.assertRaises(_MalformedEvidence):
                    _canonicalize_longitudinal_request(clone)

    def test_bool_is_never_accepted_as_a_numeric(self):
        for name in (
            "throttle_request",
            "brake_request",
            "throttle_authority",
            "brake_authority",
            "effective_target_speed_mps",
            "observed_forward_speed_mps",
            "speed_error_mps",
            "speed_authority_mps",
            "target_issued_monotonic",
        ):
            with self.subTest(name):
                clone = bypassed(permitted(), **{name: True})
                with self.assertRaises(_MalformedEvidence):
                    _canonicalize_longitudinal_request(clone)

    def test_an_int_subclass_is_never_accepted_where_an_exact_int_is_due(self):
        for name in ("snapshot_revision", "target_epoch", "target_revision"):
            with self.subTest(name):
                clone = bypassed(permitted(), **{name: SneakyInt(1)})
                with self.assertRaises(_MalformedEvidence):
                    _canonicalize_longitudinal_request(clone)

    def test_a_bound_reason_without_a_binding_is_malformed(self):
        clone = bypassed(permitted(), guidance_stamp=None)
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_longitudinal_request(clone)

    def test_an_unbound_reason_with_a_binding_is_malformed(self):
        clone = bypassed(
            unbound_refusal(),
            guidance_stamp=SampleStamp(frame=5, sim_time=1.25, monotonic=BASE),
        )
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_longitudinal_request(clone)

    def test_a_bound_record_requires_every_binding_field(self):
        for name in (
            "speed_stamp",
            "snapshot_revision",
            "target_origin",
            "target_epoch",
            "target_revision",
            "target_issued_monotonic",
            "target_valid_until_monotonic",
            "safety_state",
            "lateral_mode",
            "temporal_state",
            "observed_forward_speed_mps",
            "evaluation_monotonic",
        ):
            with self.subTest(name):
                clone = bypassed(stopping(), **{name: None})
                with self.assertRaises(_MalformedEvidence):
                    _canonicalize_longitudinal_request(clone)

    def test_an_unbound_refusal_cannot_expose_an_observed_speed(self):
        clone = bypassed(unbound_refusal(), observed_forward_speed_mps=1.0)
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_longitudinal_request(clone)

    def test_an_evaluated_unbound_refusal_requires_its_time(self):
        clone = bypassed(unbound_refusal(), evaluation_monotonic=None)
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_longitudinal_request(clone)

    def test_the_time_refusal_may_omit_its_time(self):
        record = unbound_refusal(
            reason=LongitudinalControlReason.TIME_INVALID, evaluated=None
        )
        canonical = _canonicalize_longitudinal_request(record)
        self.assertIsNone(canonical.evaluation_monotonic)

    def test_the_accepted_unevaluated_record_survives_canonicalization(self):
        canonical = _canonicalize_longitudinal_request(NOT_EVALUATED_LONGITUDINAL)
        self.assertIs(canonical.reason, LongitudinalControlReason.NOT_EVALUATED)

    def test_speed_law_evidence_must_match_its_own_reason(self):
        clone = bypassed(stopping(), effective_target_speed_mps=1.0,
                         speed_error_mps=0.0, speed_authority_mps=3.0)
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_longitudinal_request(clone)
        other = bypassed(permitted(), effective_target_speed_mps=None,
                         speed_error_mps=None, speed_authority_mps=None)
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_longitudinal_request(other)

    def test_the_speed_error_must_equal_its_own_law(self):
        clone = bypassed(permitted(), speed_error_mps=0.5)
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_longitudinal_request(clone)

    def test_a_target_stop_is_exactly_a_zero_effective_target(self):
        self.assertIsNotNone(_canonicalize_longitudinal_request(target_stop()))
        clone = bypassed(
            target_stop(), effective_target_speed_mps=1.0, speed_error_mps=1.0
        )
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_longitudinal_request(clone)

    def test_a_permitted_record_cannot_publish_a_zero_effective_target(self):
        clone = bypassed(
            permitted(), effective_target_speed_mps=0.0, speed_error_mps=-1.0
        )
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_longitudinal_request(clone)

    def test_a_permitted_record_requires_both_pedal_authorities(self):
        for overrides in (
            dict(throttle_request=0.0, throttle_authority=0.0),
            dict(brake_authority=0.0),
        ):
            with self.subTest(overrides):
                clone = bypassed(permitted(), **overrides)
                with self.assertRaises(_MalformedEvidence):
                    _canonicalize_longitudinal_request(clone)

    def test_a_gated_reason_requires_fully_permitted_evidence(self):
        for reason in (
            LongitudinalControlReason.TARGET_EXPIRED,
            LongitudinalControlReason.SPEED_STALE,
            LongitudinalControlReason.EXCESSIVE_LATERAL_MOTION,
            LongitudinalControlReason.REVERSE_MOTION,
        ):
            for overrides in (
                dict(safety_state=SafetyState.FAIL_SAFE,
                     lateral_mode=LateralControlMode.DISABLED),
                dict(lateral_mode=LateralControlMode.DISABLED),
                dict(temporal_state=TemporalLaneState.LOST),
            ):
                with self.subTest(reason=reason, overrides=overrides):
                    clone = bypassed(stopping(reason=reason), **overrides)
                    with self.assertRaises(_MalformedEvidence):
                        _canonicalize_longitudinal_request(clone)

    def test_a_lateral_stop_requires_a_refused_lateral_axis(self):
        good = stopping(
            reason=LongitudinalControlReason.LATERAL_STOP,
            lateral_mode=LateralControlMode.DISABLED,
        )
        self.assertIsNotNone(_canonicalize_longitudinal_request(good))
        clone = bypassed(good, lateral_mode=LateralControlMode.NOMINAL)
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_longitudinal_request(clone)

    def test_a_safety_stop_requires_an_unpermitted_safety_state(self):
        good = stopping(
            reason=LongitudinalControlReason.SAFETY_STOP,
            safety=SafetyState.RECOVERING,
            lateral_mode=LateralControlMode.DISABLED,
        )
        self.assertIsNotNone(_canonicalize_longitudinal_request(good))
        clone = bypassed(good, safety_state=SafetyState.NOMINAL)
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_longitudinal_request(clone)

    def test_a_fail_safe_stop_requires_the_fail_safe_state(self):
        good = stopping(
            reason=LongitudinalControlReason.FAIL_SAFE_STOP,
            safety=SafetyState.FAIL_SAFE,
            lateral_mode=LateralControlMode.DISABLED,
        )
        self.assertIsNotNone(_canonicalize_longitudinal_request(good))
        clone = bypassed(good, safety_state=SafetyState.RECOVERING)
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_longitudinal_request(clone)

    def test_an_unpermitted_safety_state_cannot_carry_a_permitted_lateral_mode(self):
        clone = bypassed(
            bound_neutral(),
            safety_state=SafetyState.FAIL_SAFE,
            lateral_mode=LateralControlMode.NOMINAL,
        )
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_longitudinal_request(clone)

    def test_the_bound_neutral_refusal_is_accepted(self):
        canonical = _canonicalize_longitudinal_request(bound_neutral())
        self.assertIs(
            canonical.reason, LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION
        )
        self.assertIsNotNone(canonical.guidance_stamp)

    def test_canonicalization_does_not_mutate_the_caller_record(self):
        supplied = permitted()
        before = tuple(getattr(supplied, f.name) for f in fields(supplied))
        _canonicalize_longitudinal_request(supplied)
        after = tuple(getattr(supplied, f.name) for f in fields(supplied))
        self.assertEqual(before, after)


# --------------------------------------------------------------------------
# F. Clock, binding, and precedence
# --------------------------------------------------------------------------


class ClockTest(EnvelopeTestCase):
    def test_a_non_finite_now_is_time_invalid(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value):
                envelope = CommandEnvelope()
                request = envelope.update(lateral(), permitted(), now=value)
                self.assertNeutral(request, CommandEnvelopeReason.TIME_INVALID)
                self.assertIsNone(request.evaluation_monotonic)

    def test_a_non_real_now_is_time_invalid(self):
        envelope = CommandEnvelope()
        request = envelope.update(lateral(), permitted(), now="now")
        self.assertNeutral(request, CommandEnvelopeReason.TIME_INVALID)

    def test_a_bool_now_is_time_invalid(self):
        envelope = CommandEnvelope()
        request = envelope.update(lateral(), permitted(), now=True)
        self.assertNeutral(request, CommandEnvelopeReason.TIME_INVALID)

    def test_a_regressed_now_is_time_invalid_and_publishes_no_time(self):
        envelope = CommandEnvelope()
        self.drive(envelope, index=1)
        request = envelope.update(lateral(0), permitted(0), now=BASE)
        self.assertNeutral(request, CommandEnvelopeReason.TIME_INVALID)
        self.assertIsNone(request.evaluation_monotonic)

    def test_one_ulp_of_regression_is_time_invalid(self):
        envelope = CommandEnvelope()
        self.drive(envelope, index=0)
        earlier = math.nextafter(BASE, 0.0)
        request = envelope.update(lateral(0), permitted(0), now=earlier)
        self.assertNeutral(request, CommandEnvelopeReason.TIME_INVALID)

    def test_upstream_times_that_agree_but_differ_from_now_are_time_invalid(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            lateral(0), permitted(0), now=BASE + 1.0
        )
        self.assertNeutral(request, CommandEnvelopeReason.TIME_INVALID)
        self.assertEqual(request.evaluation_monotonic, BASE + 1.0)

    def test_upstream_times_that_disagree_are_an_identity_mismatch(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            lateral(0, evaluated=BASE + 1.0), permitted(0), now=BASE
        )
        self.assertNeutral(
            request, CommandEnvelopeReason.CROSS_RECORD_IDENTITY_MISMATCH
        )

    def test_a_lateral_record_with_no_time_against_a_bound_stop_mismatches(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            NOT_EVALUATED_LATERAL,
            stopping(0, reason=LongitudinalControlReason.LATERAL_STOP,
                     lateral_mode=LateralControlMode.DISABLED),
            now=BASE,
        )
        self.assertNeutral(
            request, CommandEnvelopeReason.CROSS_RECORD_IDENTITY_MISMATCH
        )

    def test_a_type_error_is_raised_for_a_foreign_input(self):
        envelope = CommandEnvelope()
        with self.assertRaises(TypeError):
            envelope.update(permitted(), permitted(), now=BASE)
        with self.assertRaises(TypeError):
            envelope.update(lateral(), lateral(), now=BASE)


class PrecedenceTest(EnvelopeTestCase):
    def test_malformed_lateral_evidence_outranks_everything_else(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            bypassed(lateral(), steering=0.90),
            bypassed(permitted(), throttle_request=0.90),
            now=BASE,
        )
        self.assertNeutral(
            request, CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE
        )

    def test_malformed_longitudinal_evidence_is_classified_separately(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            lateral(), bypassed(permitted(), throttle_request=0.90), now=BASE
        )
        self.assertNeutral(
            request, CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE
        )

    def test_the_evidence_free_initial_pair_is_an_upstream_refusal(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            NOT_EVALUATED_LATERAL, NOT_EVALUATED_LONGITUDINAL, now=BASE
        )
        self.assertNeutral(request, CommandEnvelopeReason.UPSTREAM_REFUSAL)
        self.assertEqual(request.evaluation_monotonic, BASE)

    def test_every_unbound_upstream_refusal_is_an_upstream_refusal(self):
        for reason in (
            LongitudinalControlReason.NOT_EVALUATED,
            LongitudinalControlReason.TIME_INVALID,
            LongitudinalControlReason.SPEED_UNAVAILABLE,
            LongitudinalControlReason.IDENTITY_MISMATCH,
            LongitudinalControlReason.EVIDENCE_MALFORMED,
            LongitudinalControlReason.VEHICLE_NOT_ALIVE,
            LongitudinalControlReason.SOURCE_REJECTED,
        ):
            with self.subTest(reason):
                envelope = CommandEnvelope()
                evaluated = None if reason in (
                    LongitudinalControlReason.NOT_EVALUATED,
                ) else ...
                record = unbound_refusal(reason=reason, evaluated=evaluated)
                request = envelope.update(lateral(), record, now=BASE)
                self.assertNeutral(
                    request, CommandEnvelopeReason.UPSTREAM_REFUSAL
                )

    def test_a_bound_neutral_upstream_refusal_is_an_upstream_refusal(self):
        envelope = CommandEnvelope()
        request = envelope.update(lateral(), bound_neutral(), now=BASE)
        self.assertNeutral(request, CommandEnvelopeReason.UPSTREAM_REFUSAL)

    def test_a_permitted_longitudinal_with_a_refused_lateral_is_an_axis_conflict(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            refused_lateral(), permitted(), now=BASE
        )
        self.assertNeutral(request, CommandEnvelopeReason.AXIS_CONFLICT)

    def test_the_axis_conflict_outranks_the_mode_equality_mismatch(self):
        # The pair below disagrees on the lateral mode *and* on permission; the
        # permission disagreement is the more specific, more useful answer.
        envelope = CommandEnvelope()
        request = envelope.update(
            refused_lateral(reason=LateralControlReason.INSUFFICIENT_GEOMETRY),
            permitted(lateral_mode=LateralControlMode.NOMINAL),
            now=BASE,
        )
        self.assertIs(request.reason, CommandEnvelopeReason.AXIS_CONFLICT)

    def test_a_trusted_stop_outranks_a_bound_neutral_and_a_command(self):
        envelope = CommandEnvelope()
        request = envelope.update(lateral(), stopping(), now=BASE)
        self.assertIs(request.mode, CommandEnvelopeMode.STOPPING)
        self.assertIs(request.reason, CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP)


class BindingTest(EnvelopeTestCase):
    def mismatch(self, *, lat=None, lon=None):
        envelope = CommandEnvelope()
        request = envelope.update(
            lateral() if lat is None else lat,
            permitted() if lon is None else lon,
            now=BASE,
        )
        self.assertNeutral(
            request, CommandEnvelopeReason.CROSS_RECORD_IDENTITY_MISMATCH
        )
        return request

    def test_a_different_guidance_stamp_is_a_mismatch(self):
        self.mismatch(
            lat=lateral(stamp=SampleStamp(frame=9, sim_time=2.25, monotonic=BASE))
        )

    def test_one_differing_stamp_member_is_a_mismatch(self):
        _now, guidance, _speed, _frame = cut(0)
        for member, value in (
            ("frame", 99),
            ("sim_time", 9.5),
            ("monotonic", BASE - 0.06),
        ):
            with self.subTest(member):
                self.mismatch(
                    lat=lateral(
                        stamp=SampleStamp(
                            **{
                                **dict(
                                    frame=guidance.frame,
                                    sim_time=guidance.sim_time,
                                    monotonic=guidance.monotonic,
                                ),
                                member: value,
                            }
                        )
                    )
                )

    def test_a_different_snapshot_revision_is_a_mismatch(self):
        self.mismatch(lat=lateral(revision=99))

    def test_a_different_lateral_mode_is_a_mismatch(self):
        self.mismatch(
            lat=lateral(mode=LateralControlMode.DEGRADED,
                        reason=LateralControlReason.DEGRADED_TRACKING,
                        safety=SafetyState.DEGRADED),
            lon=permitted(safety=SafetyState.DEGRADED,
                          lateral_mode=LateralControlMode.NOMINAL,
                          mode=LongitudinalControlMode.DEGRADED,
                          reason=LongitudinalControlReason.DEGRADED_TRACKING,
                          throttle=0.05, throttle_authority=0.15),
        )

    def test_a_different_safety_state_is_a_mismatch(self):
        self.mismatch(
            lat=lateral(mode=LateralControlMode.DEGRADED,
                        reason=LateralControlReason.DEGRADED_TRACKING,
                        safety=SafetyState.DEGRADED),
            lon=permitted(safety=SafetyState.NOMINAL,
                          lateral_mode=LateralControlMode.DEGRADED,
                          mode=LongitudinalControlMode.DEGRADED,
                          reason=LongitudinalControlReason.DEGRADED_TRACKING,
                          throttle=0.05, throttle_authority=0.15),
        )

    def test_a_different_temporal_state_is_a_mismatch(self):
        self.mismatch(
            lat=lateral(mode=LateralControlMode.DEGRADED,
                        reason=LateralControlReason.DEGRADED_COASTING,
                        safety=SafetyState.DEGRADED,
                        temporal=TemporalLaneState.COASTING),
            lon=permitted(safety=SafetyState.DEGRADED,
                          lateral_mode=LateralControlMode.DEGRADED,
                          temporal=TemporalLaneState.TRACKING,
                          mode=LongitudinalControlMode.DEGRADED,
                          reason=LongitudinalControlReason.DEGRADED_TRACKING,
                          throttle=0.05, throttle_authority=0.15),
        )

    def test_a_nominal_pair_whose_longitudinal_calls_itself_degraded_mismatches(self):
        self.mismatch(
            lon=permitted(
                mode=LongitudinalControlMode.DEGRADED,
                reason=LongitudinalControlReason.DEGRADED_TRACKING,
                throttle=0.05,
                throttle_authority=0.15,
            )
        )

    def test_a_coasting_cut_whose_longitudinal_calls_itself_tracking_mismatches(self):
        self.mismatch(
            lat=lateral(mode=LateralControlMode.DEGRADED,
                        reason=LateralControlReason.DEGRADED_COASTING,
                        safety=SafetyState.DEGRADED,
                        temporal=TemporalLaneState.COASTING),
            lon=permitted(safety=SafetyState.DEGRADED,
                          lateral_mode=LateralControlMode.DEGRADED,
                          temporal=TemporalLaneState.COASTING,
                          mode=LongitudinalControlMode.DEGRADED,
                          reason=LongitudinalControlReason.DEGRADED_TRACKING,
                          throttle=0.05, throttle_authority=0.15),
        )

    def test_a_tracking_cut_whose_longitudinal_calls_itself_coasting_mismatches(self):
        self.mismatch(
            lat=lateral(mode=LateralControlMode.DEGRADED,
                        reason=LateralControlReason.DEGRADED_TRACKING,
                        safety=SafetyState.DEGRADED),
            lon=permitted(safety=SafetyState.DEGRADED,
                          lateral_mode=LateralControlMode.DEGRADED,
                          mode=LongitudinalControlMode.DEGRADED,
                          reason=LongitudinalControlReason.DEGRADED_COASTING,
                          throttle=0.05, throttle_authority=0.10),
        )

    def test_the_accepted_degraded_coasting_pair_binds(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            lateral(mode=LateralControlMode.DEGRADED,
                    reason=LateralControlReason.DEGRADED_COASTING,
                    safety=SafetyState.DEGRADED,
                    temporal=TemporalLaneState.COASTING,
                    authority=0.15),
            permitted(safety=SafetyState.DEGRADED,
                      lateral_mode=LateralControlMode.DEGRADED,
                      temporal=TemporalLaneState.COASTING,
                      mode=LongitudinalControlMode.DEGRADED,
                      reason=LongitudinalControlReason.DEGRADED_COASTING,
                      throttle=0.05, throttle_authority=0.10,
                      target=1.0, speed_authority=1.0),
            now=BASE,
        )
        self.assertIs(request.mode, CommandEnvelopeMode.DEGRADED)
        self.assertBoundIdentity(request)

    def test_a_trusted_stop_binds_against_a_refused_lateral_request(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            refused_lateral(reason=LateralControlReason.INSUFFICIENT_GEOMETRY),
            stopping(reason=LongitudinalControlReason.LATERAL_STOP,
                     lateral_mode=LateralControlMode.DISABLED),
            now=BASE,
        )
        self.assertIs(request.mode, CommandEnvelopeMode.STOPPING)
        self.assertEqual(request.steering_request, 0.0)
        self.assertEqual(request.steering_authority, 0.0)
        self.assertFalse(request.steering_clamped)

    def test_a_trusted_stop_binding_still_requires_a_matching_lateral_mode(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            refused_lateral(reason=LateralControlReason.INSUFFICIENT_GEOMETRY),
            stopping(lateral_mode=LateralControlMode.NOMINAL),
            now=BASE,
        )
        self.assertNeutral(
            request, CommandEnvelopeReason.CROSS_RECORD_IDENTITY_MISMATCH
        )

    def test_no_mismatch_ever_becomes_a_trusted_stop(self):
        cases = (
            dict(lat=lateral(revision=99)),
            dict(lat=lateral(evaluated=BASE + 1.0)),
            dict(lat=refused_lateral()),
            dict(lat=lateral(safety=SafetyState.DEGRADED,
                             mode=LateralControlMode.DEGRADED,
                             reason=LateralControlReason.DEGRADED_TRACKING)),
        )
        for kwargs in cases:
            with self.subTest(kwargs):
                envelope = CommandEnvelope()
                request = envelope.update(
                    kwargs["lat"], permitted(), now=BASE
                )
                self.assertIsNot(request.mode, CommandEnvelopeMode.STOPPING)
                self.assertFalse(request.stop_required)
                self.assertEqual(request.brake_request, 0.0)


# --------------------------------------------------------------------------
# G. Ordering, duplicates, and fingerprints
# --------------------------------------------------------------------------


class OrderingTest(EnvelopeTestCase):
    def test_the_exact_same_time_duplicate_is_idempotent(self):
        envelope = CommandEnvelope()
        first = self.drive(envelope, index=0)
        again = envelope.update(lateral(0), permitted(0), now=BASE)
        self.assertIs(again, first)
        self.assertEqual(envelope.metrics.idempotent_replays, 1)
        self.assertEqual(envelope.metrics.commands, 1)

    def test_changed_content_at_an_unchanged_time_is_source_rejected(self):
        envelope = CommandEnvelope()
        self.drive(envelope, index=0)
        request = envelope.update(lateral(0, steering=0.20), permitted(0), now=BASE)
        self.assertNeutral(request, CommandEnvelopeReason.SOURCE_REJECTED)

    def test_a_changed_longitudinal_field_at_an_unchanged_time_is_rejected(self):
        envelope = CommandEnvelope()
        self.drive(envelope, index=0)
        request = envelope.update(
            lateral(0), permitted(0, throttle=0.25), now=BASE
        )
        self.assertNeutral(request, CommandEnvelopeReason.SOURCE_REJECTED)

    def test_a_same_time_replay_after_a_refusal_never_resurrects_the_command(self):
        envelope = CommandEnvelope()
        accepted = self.drive(envelope, index=0)
        self.assertGreater(accepted.steering_request, 0.0)
        refused = envelope.update(lateral(0, steering=0.20), permitted(0), now=BASE)
        self.assertNeutral(refused, CommandEnvelopeReason.SOURCE_REJECTED)
        replay = envelope.update(lateral(0), permitted(0), now=BASE)
        self.assertNeutral(replay, CommandEnvelopeReason.SOURCE_REJECTED)
        self.assertIs(envelope.latest_request, replay)

    def test_an_older_guidance_frame_is_source_rejected(self):
        envelope = CommandEnvelope()
        self.drive(envelope, index=2)
        request = envelope.update(
            lateral(1, now=BASE + 3 * STEP),
            permitted(1, now=BASE + 3 * STEP),
            now=BASE + 3 * STEP,
        )
        self.assertNeutral(request, CommandEnvelopeReason.SOURCE_REJECTED)

    def test_a_repeated_guidance_frame_at_a_newer_time_is_accepted(self):
        envelope = CommandEnvelope()
        self.drive(envelope, index=0)
        later = BASE + STEP
        lat, lon = replay(0, later)
        request = envelope.update(lat, lon, now=later)
        self.assertTrue(request.command_allowed)

    def test_a_mutated_duplicate_stamp_is_source_rejected(self):
        envelope = CommandEnvelope()
        self.drive(envelope, index=0)
        later = BASE + STEP
        _first, guidance, speed, _frame = cut(0)
        mutated = SampleStamp(
            frame=guidance.frame,
            sim_time=guidance.sim_time,
            monotonic=guidance.monotonic + 0.001,
        )
        request = envelope.update(
            lateral(0, now=later, stamp=mutated),
            permitted(0, now=later, stamp=mutated, speed_stamp=speed),
            now=later,
        )
        self.assertNeutral(request, CommandEnvelopeReason.SOURCE_REJECTED)

    def test_a_duplicate_frame_with_a_changed_revision_is_source_rejected(self):
        envelope = CommandEnvelope()
        self.drive(envelope, index=0)
        later = BASE + STEP
        _first, guidance, speed, _frame = cut(0)
        request = envelope.update(
            lateral(0, now=later, stamp=guidance, revision=99),
            permitted(0, now=later, stamp=guidance, speed_stamp=speed,
                      revision=99),
            now=later,
        )
        self.assertNeutral(request, CommandEnvelopeReason.SOURCE_REJECTED)

    def test_a_regressed_simulation_time_is_source_rejected(self):
        envelope = CommandEnvelope()
        self.drive(envelope, index=1)
        later = BASE + 2 * STEP
        _now, guidance, speed, frame = cut(1)
        older = SampleStamp(
            frame=guidance.frame + 1,
            sim_time=guidance.sim_time - 1.0,
            monotonic=later - 0.05,
        )
        request = envelope.update(
            lateral(1, now=later, evaluated=later, stamp=older, revision=99),
            permitted(1, now=later, evaluated=later, stamp=older, revision=99,
                      speed_stamp=older),
            now=later,
        )
        self.assertNeutral(request, CommandEnvelopeReason.SOURCE_REJECTED)

    def test_a_changed_target_issuer_is_source_rejected(self):
        envelope = CommandEnvelope()
        self.drive(envelope, index=0)
        request = envelope.update(
            lateral(1), permitted(1, origin=TargetSpeedOrigin.OPERATOR),
            now=BASE + STEP,
        )
        self.assertNeutral(request, CommandEnvelopeReason.SOURCE_REJECTED)

    def test_a_changed_target_epoch_is_source_rejected(self):
        envelope = CommandEnvelope()
        self.drive(envelope, index=0)
        request = envelope.update(
            lateral(1), permitted(1, epoch=2), now=BASE + STEP
        )
        self.assertNeutral(request, CommandEnvelopeReason.SOURCE_REJECTED)

    def test_a_regressed_target_revision_is_source_rejected(self):
        envelope = CommandEnvelope()
        self.drive(envelope, index=0, lon=permitted(0, target_revision=5))
        request = envelope.update(
            lateral(1), permitted(1, target_revision=4, issued=BASE),
            now=BASE + STEP,
        )
        self.assertNeutral(request, CommandEnvelopeReason.SOURCE_REJECTED)

    def test_a_mutated_target_duplicate_is_source_rejected(self):
        envelope = CommandEnvelope()
        self.drive(envelope, index=0)
        request = envelope.update(
            lateral(1), permitted(1, valid_until=BASE + 99.0), now=BASE + STEP
        )
        self.assertNeutral(request, CommandEnvelopeReason.SOURCE_REJECTED)

    def test_an_unchanged_target_is_a_legitimate_duplicate(self):
        envelope = CommandEnvelope()
        self.drive(envelope, index=0)
        request = self.drive(envelope, index=1)
        self.assertTrue(request.command_allowed)

    def test_a_newer_target_revision_and_issue_time_is_accepted(self):
        envelope = CommandEnvelope()
        self.drive(envelope, index=0)
        request = envelope.update(
            lateral(1), permitted(1, target_revision=2, issued=BASE),
            now=BASE + STEP,
        )
        self.assertTrue(request.command_allowed)

    def test_a_rejected_source_advances_no_baseline(self):
        envelope = CommandEnvelope()
        accepted = self.drive(envelope, index=0)
        envelope.update(lateral(1), permitted(1, epoch=2), now=BASE + STEP)
        # The rejected epoch never became the baseline, so the original epoch is
        # still the one that binds.
        request = self.drive(envelope, index=2)
        self.assertTrue(request.command_allowed)
        self.assertEqual(request.target_epoch, 1)

    def test_a_refusal_clears_shaping_but_not_the_evidence_baseline(self):
        envelope = CommandEnvelope()
        for index in range(4):
            self.drive(envelope, index=index)
        ramped = envelope.latest_request.throttle_request
        self.assertGreater(ramped, 0.0)
        envelope.update(lateral(4), bound_neutral(4), now=BASE + 4 * STEP)
        request = self.drive(envelope, index=5)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertTrue(request.throttle_rate_limited)
        # The order baseline survived, so an older frame is still rejected.
        later = BASE + 6 * STEP
        older = envelope.update(
            lateral(3, now=later), permitted(3, now=later), now=later
        )
        self.assertNeutral(older, CommandEnvelopeReason.SOURCE_REJECTED)


# --------------------------------------------------------------------------
# H. Authority and arbitration
# --------------------------------------------------------------------------


class AuthorityTest(EnvelopeTestCase):
    def test_each_authority_is_the_smaller_of_config_and_upstream(self):
        envelope = CommandEnvelope(
            CommandEnvelopeConfig(
                maximum_steering=0.20,
                maximum_throttle=0.10,
                maximum_ordinary_brake=0.30,
            )
        )
        request = envelope.update(lateral(), permitted(throttle=0.0), now=BASE)
        self.assertEqual(request.steering_authority, 0.20)
        self.assertEqual(request.throttle_authority, 0.10)
        self.assertEqual(request.brake_authority, 0.30)

    def test_a_smaller_upstream_authority_wins(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            lateral(authority=0.05, steering=0.0),
            permitted(throttle=0.0, throttle_authority=0.10, brake_authority=0.20),
            now=BASE,
        )
        self.assertEqual(request.steering_authority, 0.05)
        self.assertEqual(request.throttle_authority, 0.10)
        self.assertEqual(request.brake_authority, 0.20)

    def test_steering_is_clamped_into_its_final_authority(self):
        envelope = CommandEnvelope(CommandEnvelopeConfig(maximum_steering=0.05))
        request = envelope.update(lateral(steering=0.30), permitted(), now=BASE)
        self.assertEqual(request.steering_request, 0.05)
        self.assertTrue(request.steering_clamped)
        self.assertTrue(request.shaping_applied)

    def test_negative_steering_is_clamped_symmetrically(self):
        envelope = CommandEnvelope(CommandEnvelopeConfig(maximum_steering=0.05))
        request = envelope.update(lateral(steering=-0.30), permitted(), now=BASE)
        self.assertEqual(request.steering_request, -0.05)
        self.assertTrue(request.steering_clamped)

    def test_steering_exactly_at_authority_is_not_clamped(self):
        envelope = CommandEnvelope(CommandEnvelopeConfig(maximum_steering=0.30))
        request = envelope.update(lateral(steering=0.30), permitted(), now=BASE)
        self.assertEqual(request.steering_request, 0.30)
        self.assertFalse(request.steering_clamped)

    def test_one_ulp_above_authority_is_clamped(self):
        authority = 0.30
        envelope = CommandEnvelope(CommandEnvelopeConfig(maximum_steering=authority))
        request = envelope.update(
            lateral(steering=math.nextafter(authority, 1.0)), permitted(), now=BASE
        )
        self.assertEqual(request.steering_request, authority)
        self.assertTrue(request.steering_clamped)

    def test_throttle_is_clamped_into_its_final_authority(self):
        envelope = CommandEnvelope(CommandEnvelopeConfig(maximum_throttle=0.05))
        request = envelope.update(lateral(), permitted(throttle=0.30), now=BASE)
        self.assertTrue(request.throttle_clamped)
        self.assertLessEqual(request.throttle_request, 0.05)

    def test_brake_is_clamped_into_the_ordinary_ceiling(self):
        envelope = CommandEnvelope(
            CommandEnvelopeConfig(maximum_ordinary_brake=0.10)
        )
        request = envelope.update(
            lateral(), permitted(throttle=0.0, brake=0.30, target=1.0,
                                 observed=2.0), now=BASE
        )
        self.assertEqual(request.brake_request, 0.10)
        self.assertEqual(request.brake_authority, 0.10)
        self.assertTrue(request.brake_clamped)

    def test_an_unclamped_pair_reports_no_shaping_at_all(self):
        envelope = CommandEnvelope()
        request = envelope.update(lateral(), permitted(throttle=0.0), now=BASE)
        self.assertFalse(request.shaping_applied)
        self.assertIs(request.reason, CommandEnvelopeReason.NOMINAL_COMMAND)

    def test_ordinary_braking_is_not_a_stop_request(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            lateral(), permitted(throttle=0.0, brake=0.20, target=1.0, observed=2.0),
            now=BASE,
        )
        self.assertIs(request.mode, CommandEnvelopeMode.NOMINAL)
        self.assertFalse(request.stop_required)
        self.assertEqual(request.brake_request, 0.20)

    def test_the_envelope_only_ever_subtracts_authority(self):
        envelope = CommandEnvelope()
        request = envelope.update(lateral(), permitted(throttle=0.0), now=BASE)
        self.assertLessEqual(request.steering_authority, LATERAL_AUTHORITY)
        self.assertLessEqual(request.throttle_authority, THROTTLE_AUTHORITY)
        self.assertLessEqual(request.brake_authority, BRAKE_AUTHORITY)
        self.assertLessEqual(request.steering_authority, CONFIG.maximum_steering)
        self.assertLessEqual(request.throttle_authority, CONFIG.maximum_throttle)
        self.assertLessEqual(
            request.brake_authority, CONFIG.maximum_ordinary_brake
        )

    def test_the_degraded_mode_is_taken_from_the_longitudinal_record(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            lateral(mode=LateralControlMode.DEGRADED,
                    reason=LateralControlReason.DEGRADED_TRACKING,
                    safety=SafetyState.DEGRADED, authority=0.15),
            permitted(safety=SafetyState.DEGRADED,
                      lateral_mode=LateralControlMode.DEGRADED,
                      mode=LongitudinalControlMode.DEGRADED,
                      reason=LongitudinalControlReason.DEGRADED_TRACKING,
                      throttle=0.05, throttle_authority=0.15,
                      target=1.5, speed_authority=1.5),
            now=BASE,
        )
        self.assertIs(request.mode, CommandEnvelopeMode.DEGRADED)
        self.assertIn(
            request.reason,
            (
                CommandEnvelopeReason.DEGRADED_COMMAND,
                CommandEnvelopeReason.DEGRADED_SHAPED,
            ),
        )


# --------------------------------------------------------------------------
# I. Throttle and brake shaping
# --------------------------------------------------------------------------


class ShapingHelperTest(EnvelopeTestCase):
    def test_rise_passes_decreases_and_equalities_through(self):
        self.assertEqual(_rise_limited(0.1, 0.3, 0.5, 0.25), (0.1, False))
        self.assertEqual(_rise_limited(0.3, 0.3, 0.5, 0.25), (0.3, False))

    def test_rise_bounds_an_increase(self):
        value, limited = _rise_limited(0.5, 0.0, 0.5, 0.25)
        self.assertEqual(value, 0.125)
        self.assertTrue(limited)

    def test_rise_with_zero_delta_grants_nothing(self):
        self.assertEqual(_rise_limited(0.5, 0.0, 0.5, 0.0), (0.0, True))

    def test_release_passes_increases_and_equalities_through(self):
        self.assertEqual(_release_limited(0.3, 0.1, 1.0, 0.25), (0.3, False))
        self.assertEqual(_release_limited(0.3, 0.3, 1.0, 0.25), (0.3, False))

    def test_release_bounds_a_decrease(self):
        value, limited = _release_limited(0.0, 0.5, 1.0, 0.25)
        self.assertEqual(value, 0.25)
        self.assertTrue(limited)

    def test_release_never_undershoots_the_candidate(self):
        value, limited = _release_limited(0.0, 0.10, 1.0, 0.25)
        self.assertEqual(value, 0.0)
        self.assertFalse(limited)

    def test_the_symmetric_clamp_reports_movement_truthfully(self):
        self.assertEqual(_clamp_symmetric(0.1, 0.3), (0.1, False))
        self.assertEqual(_clamp_symmetric(0.4, 0.3), (0.3, True))
        self.assertEqual(_clamp_symmetric(-0.4, 0.3), (-0.3, True))


class ThrottleShapingTest(EnvelopeTestCase):
    def ramp(self, envelope, steps, *, throttle=0.30):
        seen = []
        for index in range(steps):
            request = envelope.update(
                lateral(index),
                permitted(index, throttle=throttle),
                now=BASE + index * STEP,
            )
            seen.append(request)
        return seen

    def test_the_first_propulsion_of_an_epoch_is_exactly_zero(self):
        envelope = CommandEnvelope()
        first = self.ramp(envelope, 1)[0]
        self.assertEqual(first.throttle_request, 0.0)
        self.assertTrue(first.throttle_rate_limited)
        self.assertIs(first.reason, CommandEnvelopeReason.NOMINAL_SHAPED)

    def test_throttle_rises_at_exactly_the_configured_rate(self):
        envelope = CommandEnvelope()
        seen = self.ramp(envelope, 4)
        self.assertEqual(
            [request.throttle_request for request in seen],
            [0.0, 0.125, 0.25, 0.30],
        )
        self.assertEqual(
            [request.throttle_rate_limited for request in seen],
            [True, True, True, False],
        )

    def test_a_throttle_decrease_is_immediate(self):
        envelope = CommandEnvelope()
        self.ramp(envelope, 4)
        request = envelope.update(
            lateral(4), permitted(4, throttle=0.05), now=BASE + 4 * STEP
        )
        self.assertEqual(request.throttle_request, 0.05)
        self.assertFalse(request.throttle_rate_limited)

    def test_an_authority_reduction_is_immediate(self):
        envelope = CommandEnvelope()
        self.ramp(envelope, 4)
        self.assertEqual(envelope.latest_request.throttle_request, 0.30)
        # Mission 10A itself clamps its request to its own reduced authority,
        # and the envelope must adopt the lower value at once rather than
        # slewing down to it.
        request = envelope.update(
            lateral(4),
            permitted(4, throttle=0.10, throttle_authority=0.10),
            now=BASE + 4 * STEP,
        )
        self.assertEqual(request.throttle_request, 0.10)
        self.assertEqual(request.throttle_authority, 0.10)
        self.assertFalse(request.throttle_rate_limited)

    def test_a_brake_demand_removes_throttle_immediately(self):
        envelope = CommandEnvelope()
        self.ramp(envelope, 4)
        request = envelope.update(
            lateral(4),
            permitted(4, throttle=0.0, brake=0.20, target=1.0, observed=2.0),
            now=BASE + 4 * STEP,
        )
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, 0.20)

    def test_a_trusted_stop_removes_throttle_immediately(self):
        envelope = CommandEnvelope()
        self.ramp(envelope, 4)
        request = envelope.update(
            refused_lateral(4, reason=LateralControlReason.INSUFFICIENT_GEOMETRY),
            stopping(4, reason=LongitudinalControlReason.LATERAL_STOP,
                     lateral_mode=LateralControlMode.DISABLED),
            now=BASE + 4 * STEP,
        )
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.throttle_authority, 0.0)
        self.assertEqual(request.brake_request, 0.20)

    def test_a_refusal_removes_throttle_immediately(self):
        envelope = CommandEnvelope()
        self.ramp(envelope, 4)
        request = envelope.update(lateral(4), bound_neutral(4), now=BASE + 4 * STEP)
        self.assertNeutral(request, CommandEnvelopeReason.UPSTREAM_REFUSAL)

    def test_the_first_propulsion_after_a_refusal_is_zero_again(self):
        envelope = CommandEnvelope()
        self.ramp(envelope, 4)
        envelope.update(lateral(4), bound_neutral(4), now=BASE + 4 * STEP)
        request = envelope.update(
            lateral(5), permitted(5, throttle=0.30), now=BASE + 5 * STEP
        )
        self.assertEqual(request.throttle_request, 0.0)
        self.assertTrue(request.throttle_rate_limited)

    def test_one_ulp_of_time_grants_only_the_rate_times_that_ulp(self):
        envelope = CommandEnvelope()
        self.ramp(envelope, 2)
        anchor = envelope.latest_request.throttle_request
        previous = BASE + STEP
        later = math.nextafter(previous, math.inf)
        request = envelope.update(
            lateral(2, now=later),
            permitted(2, now=later, throttle=0.30),
            now=later,
        )
        self.assertEqual(
            request.throttle_request,
            anchor + CONFIG.throttle_rise_rate_per_second * (later - previous),
        )
        granted = request.throttle_request - anchor
        self.assertGreaterEqual(granted, 0.0)
        self.assertLess(granted, 1e-13)
        self.assertTrue(request.throttle_rate_limited)

    def test_a_large_gap_is_capped_by_the_shaping_delta(self):
        envelope = CommandEnvelope()
        self.ramp(envelope, 1)
        later = BASE + 100.0
        request = envelope.update(
            lateral(1, now=later, evaluated=later),
            permitted(1, now=later, evaluated=later, throttle=0.30),
            now=later,
        )
        self.assertEqual(
            request.throttle_request,
            CONFIG.throttle_rise_rate_per_second
            * CONFIG.maximum_shaping_delta_seconds,
        )

    def test_the_rise_never_exceeds_the_candidate(self):
        envelope = CommandEnvelope()
        seen = self.ramp(envelope, 6, throttle=0.05)
        for request in seen:
            self.assertLessEqual(request.throttle_request, 0.05)


class BrakeShapingTest(EnvelopeTestCase):
    def brake_step(self, envelope, index, brake, **kwargs):
        return envelope.update(
            lateral(index),
            permitted(index, throttle=0.0, brake=brake, target=1.0, observed=2.0,
                      **kwargs),
            now=BASE + index * STEP,
        )

    def test_an_ordinary_brake_increase_is_immediate(self):
        envelope = CommandEnvelope()
        request = self.brake_step(envelope, 0, 0.30)
        self.assertEqual(request.brake_request, 0.30)
        self.assertFalse(request.brake_release_rate_limited)

    def test_a_further_brake_increase_is_immediate(self):
        envelope = CommandEnvelope()
        self.brake_step(envelope, 0, 0.10)
        request = self.brake_step(envelope, 1, 0.35)
        self.assertEqual(request.brake_request, 0.35)
        self.assertFalse(request.brake_release_rate_limited)

    def test_a_brake_release_is_rate_limited(self):
        envelope = CommandEnvelope()
        self.brake_step(envelope, 0, 0.375)
        request = self.brake_step(envelope, 1, 0.0)
        self.assertEqual(request.brake_request, 0.125)
        self.assertTrue(request.brake_release_rate_limited)

    def test_a_brake_release_reaches_zero_and_stops_being_limited(self):
        envelope = CommandEnvelope()
        self.brake_step(envelope, 0, 0.375)
        first = self.brake_step(envelope, 1, 0.0)
        second = self.brake_step(envelope, 2, 0.0)
        self.assertEqual(first.brake_request, 0.125)
        self.assertEqual(second.brake_request, 0.0)
        self.assertFalse(second.brake_release_rate_limited)

    def test_a_partial_release_is_honoured_when_it_is_within_the_rate(self):
        envelope = CommandEnvelope()
        self.brake_step(envelope, 0, 0.375)
        request = self.brake_step(envelope, 1, 0.25)
        self.assertEqual(request.brake_request, 0.25)
        self.assertFalse(request.brake_release_rate_limited)

    def test_a_refusal_zeroes_brake_immediately_and_clears_its_history(self):
        envelope = CommandEnvelope()
        self.brake_step(envelope, 0, 0.375)
        refused = envelope.update(lateral(1), bound_neutral(1), now=BASE + STEP)
        self.assertNeutral(refused, CommandEnvelopeReason.UPSTREAM_REFUSAL)
        request = self.brake_step(envelope, 2, 0.0)
        self.assertEqual(request.brake_request, 0.0)
        self.assertFalse(request.brake_release_rate_limited)

    def test_no_old_brake_is_inherited_across_a_refusal(self):
        envelope = CommandEnvelope()
        self.brake_step(envelope, 0, 0.375)
        envelope.update(lateral(1), bound_neutral(1), now=BASE + STEP)
        request = envelope.update(
            lateral(2), permitted(2, throttle=0.30), now=BASE + 2 * STEP
        )
        self.assertEqual(request.brake_request, 0.0)
        self.assertFalse(request.transition_interlocked)
        self.assertEqual(request.throttle_request, 0.0)

    def test_the_release_anchor_is_first_brought_inside_the_authority(self):
        envelope = CommandEnvelope()
        self.brake_step(envelope, 0, 0.375)
        request = self.brake_step(envelope, 1, 0.0, brake_authority=0.20)
        self.assertEqual(request.brake_authority, 0.20)
        self.assertEqual(request.brake_request, 0.0)


# --------------------------------------------------------------------------
# J. Brake-to-throttle interlock
# --------------------------------------------------------------------------


class InterlockTest(EnvelopeTestCase):
    def slow_config(self):
        return CommandEnvelopeConfig(brake_release_rate_per_second=0.50)

    def test_throttle_is_suppressed_while_brake_history_is_positive(self):
        envelope = CommandEnvelope(self.slow_config())
        envelope.update(
            lateral(0),
            permitted(0, throttle=0.0, brake=0.375, target=1.0, observed=2.0),
            now=BASE,
        )
        request = envelope.update(
            lateral(1), permitted(1, throttle=0.30), now=BASE + STEP
        )
        self.assertTrue(request.transition_interlocked)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, 0.25)
        self.assertIs(
            request.reason, CommandEnvelopeReason.NOMINAL_TRANSITION_INTERLOCK
        )

    def test_the_update_reaching_zero_brake_also_publishes_zero_throttle(self):
        envelope = CommandEnvelope(self.slow_config())
        envelope.update(
            lateral(0),
            permitted(0, throttle=0.0, brake=0.375, target=1.0, observed=2.0),
            now=BASE,
        )
        first = envelope.update(
            lateral(1), permitted(1, throttle=0.30), now=BASE + STEP
        )
        self.assertEqual(first.brake_request, 0.25)
        second = envelope.update(
            lateral(2), permitted(2, throttle=0.30), now=BASE + 2 * STEP
        )
        self.assertEqual(second.brake_request, 0.125)
        third = envelope.update(
            lateral(3), permitted(3, throttle=0.30), now=BASE + 3 * STEP
        )
        self.assertEqual(third.brake_request, 0.0)
        self.assertEqual(third.throttle_request, 0.0)
        self.assertTrue(third.transition_interlocked)

    def test_only_a_later_accepted_update_may_start_the_rise(self):
        envelope = CommandEnvelope()
        envelope.update(
            lateral(0),
            permitted(0, throttle=0.0, brake=0.20, target=1.0, observed=2.0),
            now=BASE,
        )
        dwell = envelope.update(
            lateral(1), permitted(1, throttle=0.30), now=BASE + STEP
        )
        self.assertEqual(dwell.brake_request, 0.0)
        self.assertEqual(dwell.throttle_request, 0.0)
        self.assertTrue(dwell.transition_interlocked)
        rise = envelope.update(
            lateral(2), permitted(2, throttle=0.30), now=BASE + 2 * STEP
        )
        self.assertFalse(rise.transition_interlocked)
        self.assertEqual(rise.throttle_request, 0.125)
        self.assertEqual(rise.brake_request, 0.0)

    def test_new_brake_cancels_a_propulsion_transition(self):
        envelope = CommandEnvelope(self.slow_config())
        envelope.update(
            lateral(0),
            permitted(0, throttle=0.0, brake=0.375, target=1.0, observed=2.0),
            now=BASE,
        )
        interlocked = envelope.update(
            lateral(1), permitted(1, throttle=0.30), now=BASE + STEP
        )
        self.assertTrue(interlocked.transition_interlocked)
        request = envelope.update(
            lateral(2),
            permitted(2, throttle=0.0, brake=0.375, target=1.0, observed=2.0),
            now=BASE + 2 * STEP,
        )
        self.assertFalse(request.transition_interlocked)
        self.assertEqual(request.brake_request, 0.375)
        self.assertEqual(request.throttle_request, 0.0)

    def test_a_trusted_stop_cancels_a_propulsion_transition(self):
        envelope = CommandEnvelope(self.slow_config())
        envelope.update(
            lateral(0),
            permitted(0, throttle=0.0, brake=0.375, target=1.0, observed=2.0),
            now=BASE,
        )
        envelope.update(lateral(1), permitted(1, throttle=0.30), now=BASE + STEP)
        request = envelope.update(
            refused_lateral(2, reason=LateralControlReason.INSUFFICIENT_GEOMETRY),
            stopping(2, brake=0.375,
                     reason=LongitudinalControlReason.LATERAL_STOP,
                     lateral_mode=LateralControlMode.DISABLED),
            now=BASE + 2 * STEP,
        )
        self.assertIs(request.mode, CommandEnvelopeMode.STOPPING)
        self.assertEqual(request.brake_request, 0.375)
        self.assertFalse(request.transition_interlocked)

    def test_a_refusal_clears_the_interlock_and_requires_a_propulsion_rearm(self):
        envelope = CommandEnvelope(self.slow_config())
        envelope.update(
            lateral(0),
            permitted(0, throttle=0.0, brake=0.375, target=1.0, observed=2.0),
            now=BASE,
        )
        envelope.update(lateral(1), permitted(1, throttle=0.30), now=BASE + STEP)
        envelope.update(lateral(2), bound_neutral(2), now=BASE + 2 * STEP)
        request = envelope.update(
            lateral(3), permitted(3, throttle=0.30), now=BASE + 3 * STEP
        )
        self.assertFalse(request.transition_interlocked)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.brake_request, 0.0)

    def test_throttle_to_brake_needs_no_interlock(self):
        envelope = CommandEnvelope()
        for index in range(3):
            envelope.update(
                lateral(index), permitted(index, throttle=0.30),
                now=BASE + index * STEP,
            )
        request = envelope.update(
            lateral(3),
            permitted(3, throttle=0.0, brake=0.30, target=1.0, observed=2.0),
            now=BASE + 3 * STEP,
        )
        self.assertFalse(request.transition_interlocked)
        self.assertEqual(request.brake_request, 0.30)
        self.assertEqual(request.throttle_request, 0.0)

    def test_a_stop_then_throttle_sequence_releases_dwells_then_rises(self):
        envelope = CommandEnvelope()
        envelope.update(
            refused_lateral(0, reason=LateralControlReason.INSUFFICIENT_GEOMETRY),
            stopping(0, brake=0.375,
                     reason=LongitudinalControlReason.LATERAL_STOP,
                     lateral_mode=LateralControlMode.DISABLED),
            now=BASE,
        )
        first = envelope.update(
            lateral(1), permitted(1, throttle=0.30), now=BASE + STEP
        )
        self.assertTrue(first.transition_interlocked)
        self.assertEqual(first.brake_request, 0.125)
        second = envelope.update(
            lateral(2), permitted(2, throttle=0.30), now=BASE + 2 * STEP
        )
        self.assertTrue(second.transition_interlocked)
        self.assertEqual(second.brake_request, 0.0)
        self.assertEqual(second.throttle_request, 0.0)
        third = envelope.update(
            lateral(3), permitted(3, throttle=0.30), now=BASE + 3 * STEP
        )
        self.assertFalse(third.transition_interlocked)
        self.assertEqual(third.throttle_request, 0.125)

    def test_no_update_ever_publishes_both_pedals(self):
        envelope = CommandEnvelope(self.slow_config())
        script = [
            dict(throttle=0.0, brake=0.35, target=1.0, observed=2.0),
            dict(throttle=0.30),
            dict(throttle=0.30),
            dict(throttle=0.0, brake=0.20, target=1.0, observed=2.0),
            dict(throttle=0.30),
            dict(throttle=0.0),
            dict(throttle=0.30),
        ]
        for index, kwargs in enumerate(script):
            request = envelope.update(
                lateral(index), permitted(index, **kwargs),
                now=BASE + index * STEP,
            )
            with self.subTest(index):
                self.assertSafeAxes(request)
                self.assertFalse(
                    request.throttle_request > 0.0 and request.brake_request > 0.0
                )

    def test_the_transition_state_is_a_bounded_enum(self):
        self.assertEqual(
            sorted(member.name for member in _Transition),
            ["IDLE", "RELEASING_BRAKE", "ZERO_DWELL"],
        )

    def test_a_reset_then_brake_applies_immediately(self):
        envelope = CommandEnvelope()
        for index in range(3):
            envelope.update(
                lateral(index), permitted(index, throttle=0.30),
                now=BASE + index * STEP,
            )
        envelope.reset()
        request = envelope.update(
            lateral(3),
            permitted(3, throttle=0.0, brake=0.35, target=1.0, observed=2.0),
            now=BASE + 3 * STEP,
        )
        self.assertEqual(request.brake_request, 0.35)

    def test_a_reset_then_trusted_stop_applies_immediately(self):
        envelope = CommandEnvelope()
        envelope.reset()
        request = envelope.update(
            refused_lateral(0, reason=LateralControlReason.INSUFFICIENT_GEOMETRY),
            stopping(0, brake=0.35, reason=LongitudinalControlReason.LATERAL_STOP,
                     lateral_mode=LateralControlMode.DISABLED),
            now=BASE,
        )
        self.assertEqual(request.brake_request, 0.35)
        self.assertTrue(request.stop_required)


# --------------------------------------------------------------------------
# K. Steering and trusted stopping
# --------------------------------------------------------------------------


class SteeringTest(EnvelopeTestCase):
    def test_the_envelope_adds_no_second_steering_slew(self):
        envelope = CommandEnvelope()
        first = envelope.update(lateral(0, steering=0.0), permitted(0), now=BASE)
        self.assertEqual(first.steering_request, 0.0)
        second = envelope.update(
            lateral(1, steering=0.30), permitted(1), now=BASE + STEP
        )
        self.assertEqual(second.steering_request, 0.30)
        self.assertFalse(second.steering_clamped)

    def test_a_full_steering_reversal_is_published_in_one_step(self):
        envelope = CommandEnvelope()
        envelope.update(lateral(0, steering=0.30), permitted(0), now=BASE)
        request = envelope.update(
            lateral(1, steering=-0.30), permitted(1), now=BASE + STEP
        )
        self.assertEqual(request.steering_request, -0.30)

    def test_steering_reversal_while_braking_is_unaffected_by_pedal_shaping(self):
        envelope = CommandEnvelope()
        envelope.update(
            lateral(0, steering=0.30),
            permitted(0, throttle=0.0, brake=0.30, target=1.0, observed=2.0),
            now=BASE,
        )
        request = envelope.update(
            lateral(1, steering=-0.30),
            permitted(1, throttle=0.0, brake=0.30, target=1.0, observed=2.0),
            now=BASE + STEP,
        )
        self.assertEqual(request.steering_request, -0.30)
        self.assertEqual(request.brake_request, 0.30)

    def test_the_envelope_holds_no_steering_baseline_across_a_refusal(self):
        envelope = CommandEnvelope()
        envelope.update(lateral(0, steering=0.30), permitted(0), now=BASE)
        envelope.update(lateral(1), bound_neutral(1), now=BASE + STEP)
        request = envelope.update(
            lateral(2, steering=0.30), permitted(2), now=BASE + 2 * STEP
        )
        self.assertEqual(request.steering_request, 0.30)

    def test_permitted_steering_survives_a_trusted_stop(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            lateral(0, steering=0.30),
            stopping(0, reason=LongitudinalControlReason.SPEED_STALE),
            now=BASE,
        )
        self.assertIs(request.mode, CommandEnvelopeMode.STOPPING)
        self.assertEqual(request.steering_request, 0.30)
        self.assertEqual(request.steering_authority, CONFIG.maximum_steering)

    def test_steering_is_still_clamped_during_a_trusted_stop(self):
        envelope = CommandEnvelope(CommandEnvelopeConfig(maximum_steering=0.10))
        request = envelope.update(
            lateral(0, steering=0.30),
            stopping(0, reason=LongitudinalControlReason.SPEED_STALE),
            now=BASE,
        )
        self.assertEqual(request.steering_request, 0.10)
        self.assertTrue(request.steering_clamped)
        self.assertTrue(request.shaping_applied)

    def test_a_lateral_stop_publishes_exactly_zero_steering(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            refused_lateral(0, reason=LateralControlReason.INSUFFICIENT_GEOMETRY),
            stopping(0, reason=LongitudinalControlReason.LATERAL_STOP,
                     lateral_mode=LateralControlMode.DISABLED),
            now=BASE,
        )
        self.assertEqual(request.steering_request, 0.0)
        self.assertEqual(request.steering_authority, 0.0)
        self.assertFalse(request.steering_clamped)
        self.assertFalse(request.shaping_applied)


class TrustedStopTest(EnvelopeTestCase):
    def test_the_stop_brake_is_republished_exactly(self):
        for brake in (0.10, 0.15, 0.20, 0.25, 0.35, 0.40):
            with self.subTest(brake):
                envelope = CommandEnvelope()
                request = envelope.update(
                    lateral(0), stopping(0, brake=brake), now=BASE
                )
                self.assertEqual(request.brake_request, brake)
                self.assertEqual(request.brake_authority, BRAKE_AUTHORITY)

    def test_the_ordinary_ceiling_never_weakens_a_trusted_stop(self):
        envelope = CommandEnvelope(
            CommandEnvelopeConfig(maximum_ordinary_brake=0.05)
        )
        request = envelope.update(lateral(0), stopping(0, brake=0.40), now=BASE)
        self.assertEqual(request.brake_request, 0.40)
        self.assertEqual(request.brake_authority, BRAKE_AUTHORITY)
        self.assertFalse(request.brake_clamped)

    def test_a_trusted_stop_is_immediate_as_the_first_update_of_an_epoch(self):
        envelope = CommandEnvelope()
        request = envelope.update(lateral(0), stopping(0, brake=0.35), now=BASE)
        self.assertEqual(request.brake_request, 0.35)
        self.assertFalse(request.brake_release_rate_limited)

    def test_a_trusted_stop_carries_no_throttle_authority(self):
        envelope = CommandEnvelope()
        request = envelope.update(lateral(0), stopping(0), now=BASE)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.throttle_authority, 0.0)

    def test_every_trusted_stop_reason_becomes_one_envelope_reason(self):
        for reason, kwargs in (
            (LongitudinalControlReason.SAFETY_STOP,
             dict(safety=SafetyState.RECOVERING,
                  lateral_mode=LateralControlMode.DISABLED)),
            (LongitudinalControlReason.FAIL_SAFE_STOP,
             dict(safety=SafetyState.FAIL_SAFE,
                  lateral_mode=LateralControlMode.DISABLED)),
            # A lateral stop is Mission 10A answering a Mission 9 geometry
            # refusal, which is only ever reached under a degraded permission.
            (LongitudinalControlReason.LATERAL_STOP,
             dict(safety=SafetyState.DEGRADED,
                  lateral_mode=LateralControlMode.DISABLED)),
            (LongitudinalControlReason.TARGET_EXPIRED, {}),
            (LongitudinalControlReason.SPEED_STALE, {}),
            (LongitudinalControlReason.EXCESSIVE_LATERAL_MOTION, {}),
            (LongitudinalControlReason.REVERSE_MOTION, {}),
        ):
            with self.subTest(reason):
                envelope = CommandEnvelope()
                lateral_mode = kwargs.get(
                    "lateral_mode", LateralControlMode.NOMINAL
                )
                safety = kwargs.get("safety", SafetyState.NOMINAL)
                if lateral_mode is LateralControlMode.DISABLED:
                    lat = refused_lateral(
                        0,
                        reason=(
                            LateralControlReason.INSUFFICIENT_GEOMETRY
                            if reason is LongitudinalControlReason.LATERAL_STOP
                            else LateralControlReason.SAFETY_NOT_PERMITTED
                        ),
                        safety=safety,
                    )
                else:
                    lat = lateral(0)
                request = envelope.update(
                    lat, stopping(0, reason=reason, **kwargs), now=BASE
                )
                self.assertIs(request.mode, CommandEnvelopeMode.STOPPING)
                self.assertIs(
                    request.reason, CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP
                )
                self.assertTrue(request.stop_required)
                self.assertGreater(request.brake_request, 0.0)
                self.assertBoundIdentity(request)

    def test_a_target_stop_preserves_its_complete_speed_law_evidence(self):
        envelope = CommandEnvelope()
        request = envelope.update(lateral(0), target_stop(0), now=BASE)
        self.assertIs(request.mode, CommandEnvelopeMode.STOPPING)
        self.assertEqual(request.effective_target_speed_mps, 0.0)
        self.assertEqual(request.speed_authority_mps, 3.0)
        self.assertEqual(request.observed_forward_speed_mps, 0.0)
        self.assertEqual(request.speed_error_mps, 0.0)

    def test_other_stops_publish_the_observed_speed_and_nothing_more(self):
        envelope = CommandEnvelope()
        request = envelope.update(lateral(0), stopping(0, observed=2.5), now=BASE)
        self.assertEqual(request.observed_forward_speed_mps, 2.5)
        self.assertIsNone(request.effective_target_speed_mps)
        self.assertIsNone(request.speed_error_mps)
        self.assertIsNone(request.speed_authority_mps)

    def test_a_permitted_pair_preserves_the_full_speed_law_evidence(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            lateral(0), permitted(0, target=2.0, observed=1.0), now=BASE
        )
        self.assertEqual(request.effective_target_speed_mps, 2.0)
        self.assertEqual(request.observed_forward_speed_mps, 1.0)
        self.assertEqual(request.speed_error_mps, 1.0)
        self.assertEqual(request.speed_authority_mps, 3.0)

    def test_the_published_provenance_matches_both_upstream_records(self):
        envelope = CommandEnvelope()
        lat = lateral(0)
        lon = permitted(0)
        request = envelope.update(lat, lon, now=BASE)
        self.assertEqual(request.guidance_stamp, lon.guidance_stamp)
        self.assertEqual(request.speed_stamp, lon.speed_stamp)
        self.assertEqual(request.snapshot_revision, lon.snapshot_revision)
        self.assertIs(request.target_origin, lon.target_origin)
        self.assertEqual(request.target_epoch, lon.target_epoch)
        self.assertEqual(request.target_revision, lon.target_revision)
        self.assertEqual(
            request.target_issued_monotonic, lon.target_issued_monotonic
        )
        self.assertEqual(
            request.target_valid_until_monotonic, lon.target_valid_until_monotonic
        )
        self.assertIs(request.safety_state, lon.safety_state)
        self.assertIs(request.temporal_state, lon.temporal_state)
        self.assertIs(request.lateral_mode, lat.mode)
        self.assertIs(request.lateral_reason, lat.reason)
        self.assertIs(request.longitudinal_mode, lon.mode)
        self.assertIs(request.longitudinal_reason, lon.reason)


# --------------------------------------------------------------------------
# L. Transactionality, internal faults, and reset
# --------------------------------------------------------------------------


class TransactionalityTest(EnvelopeTestCase):
    def test_an_internal_fault_publishes_a_neutral_and_re_raises(self):
        import control.command as module

        envelope = CommandEnvelope()
        for index in range(3):
            envelope.update(
                lateral(index), permitted(index, throttle=0.30),
                now=BASE + index * STEP,
            )
        original = module._fingerprint
        boom = RuntimeError("internal")

        def exploder(*args, **kwargs):
            raise boom

        module._fingerprint = exploder
        try:
            with self.assertRaises(RuntimeError) as caught:
                envelope.update(lateral(3), permitted(3), now=BASE + 3 * STEP)
        finally:
            module._fingerprint = original
        self.assertIs(caught.exception, boom)
        self.assertNeutral(
            envelope.latest_request, CommandEnvelopeReason.INTERNAL_ERROR
        )
        self.assertEqual(envelope.metrics.internal_errors, 1)

    def test_an_internal_fault_clears_the_shaping_history(self):
        import control.command as module

        envelope = CommandEnvelope()
        for index in range(4):
            envelope.update(
                lateral(index), permitted(index, throttle=0.30),
                now=BASE + index * STEP,
            )
        self.assertGreater(envelope.latest_request.throttle_request, 0.0)
        original = module._provenance
        module._provenance = lambda *a, **k: (_ for _ in ()).throw(ValueError("x"))
        try:
            with self.assertRaises(ValueError):
                envelope.update(lateral(4), permitted(4), now=BASE + 4 * STEP)
        finally:
            module._provenance = original
        request = envelope.update(
            lateral(5), permitted(5, throttle=0.30), now=BASE + 5 * STEP
        )
        self.assertEqual(request.throttle_request, 0.0)
        self.assertTrue(request.throttle_rate_limited)

    def test_an_internal_fault_leaves_the_evidence_baseline_untouched(self):
        import control.command as module

        envelope = CommandEnvelope()
        envelope.update(lateral(0), permitted(0), now=BASE)
        original = module._fingerprint
        module._fingerprint = lambda *a, **k: (_ for _ in ()).throw(ValueError("x"))
        try:
            with self.assertRaises(ValueError):
                envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        finally:
            module._fingerprint = original
        # The failed update advanced nothing, so the very same cut that was
        # accepted before it is still the newest one the order gate knows.
        later = BASE + 2 * STEP
        lat, lon = replay(0, later)
        again = envelope.update(lat, lon, now=later)
        self.assertTrue(again.command_allowed)

    def test_no_exception_or_traceback_is_retained(self):
        import control.command as module

        envelope = CommandEnvelope()
        original = module._fingerprint
        module._fingerprint = lambda *a, **k: (_ for _ in ()).throw(ValueError("x"))
        try:
            with self.assertRaises(ValueError):
                envelope.update(lateral(0), permitted(0), now=BASE)
        finally:
            module._fingerprint = original
        for value in retained_objects(envelope):
            self.assertNotIsInstance(value, BaseException)
            self.assertFalse(hasattr(value, "__traceback__"))

    def test_a_malformed_pair_leaves_the_envelope_completely_unmutated(self):
        envelope = CommandEnvelope()
        accepted = envelope.update(
            lateral(0), permitted(0, throttle=0.30), now=BASE
        )
        before = envelope.metrics
        envelope.update(
            bypassed(lateral(1), steering=0.90), permitted(1), now=BASE + STEP
        )
        after = envelope.metrics
        self.assertEqual(before.commands, after.commands)
        self.assertGreater(accepted.steering_request, 0.0)
        # The accepted baseline survived the malformed update untouched.
        later = BASE + 2 * STEP
        lat, lon = replay(0, later, throttle=0.30)
        again = envelope.update(lat, lon, now=later)
        self.assertTrue(again.command_allowed)

    def test_the_caller_records_are_never_retained(self):
        envelope = CommandEnvelope()
        lat = lateral(0)
        lon = permitted(0)
        envelope.update(lat, lon, now=BASE)
        retained = retained_ids(envelope)
        self.assertNotIn(id(lat), retained)
        self.assertNotIn(id(lon), retained)
        self.assertNotIn(id(lat.source_stamp), retained)
        self.assertNotIn(id(lon.guidance_stamp), retained)

    def test_the_caller_records_become_collectable(self):
        # Exact accepted records are slotted and cannot be weak-referenced, so
        # rejected caller-owned subclasses provide the observable retention
        # probe for the exact public type boundary.
        envelope = CommandEnvelope()
        lat = unvalidated(lateral(0))
        lon = unvalidated(permitted(0))
        lateral_ref = weakref.ref(lat)
        longitudinal_ref = weakref.ref(lon)
        with self.assertRaises(TypeError):
            envelope.update(lat, lon, now=BASE)
        del lat, lon
        gc.collect()
        self.assertIsNone(lateral_ref())
        self.assertIsNone(longitudinal_ref())


class ResetTest(EnvelopeTestCase):
    def test_a_new_envelope_starts_neutral_at_epoch_zero(self):
        envelope = CommandEnvelope()
        self.assertNeutral(
            envelope.latest_request, CommandEnvelopeReason.NOT_EVALUATED
        )
        self.assertEqual(envelope.latest_request.controller_epoch, 0)
        self.assertFalse(envelope.command_allowed)

    def test_reset_increments_the_epoch(self):
        envelope = CommandEnvelope()
        first = envelope.reset()
        second = envelope.reset()
        self.assertEqual(first.controller_epoch, 1)
        self.assertEqual(second.controller_epoch, 2)

    def test_reset_publishes_a_neutral_unevaluated_request(self):
        envelope = CommandEnvelope()
        envelope.update(lateral(0), permitted(0), now=BASE)
        request = envelope.reset()
        self.assertNeutral(request, CommandEnvelopeReason.NOT_EVALUATED)
        self.assertIs(envelope.latest_request, request)

    def test_reset_keeps_the_lifetime_counters_truthful(self):
        envelope = CommandEnvelope()
        envelope.update(lateral(0), permitted(0), now=BASE)
        envelope.update(lateral(1), bound_neutral(1), now=BASE + STEP)
        envelope.reset()
        metrics = envelope.metrics
        self.assertEqual(metrics.evaluations, 2)
        self.assertEqual(metrics.commands, 1)
        self.assertEqual(metrics.refusals, 1)
        self.assertEqual(metrics.resets, 1)
        self.assertEqual(metrics.controller_epoch, 1)

    def test_reset_discards_the_order_baseline_so_old_identities_return(self):
        envelope = CommandEnvelope()
        envelope.update(lateral(3), permitted(3), now=BASE + 3 * STEP)
        envelope.reset()
        request = envelope.update(
            lateral(0, now=BASE + 4 * STEP, evaluated=BASE + 4 * STEP),
            permitted(0, now=BASE + 4 * STEP, evaluated=BASE + 4 * STEP),
            now=BASE + 4 * STEP,
        )
        self.assertTrue(request.command_allowed)

    def test_reset_discards_the_evaluation_baseline(self):
        envelope = CommandEnvelope()
        envelope.update(lateral(3), permitted(3), now=BASE + 3 * STEP)
        envelope.reset()
        request = envelope.update(lateral(0), permitted(0), now=BASE)
        self.assertTrue(request.command_allowed)

    def test_no_axis_survives_a_reset(self):
        envelope = CommandEnvelope()
        for index in range(4):
            envelope.update(
                lateral(index), permitted(index, throttle=0.30),
                now=BASE + index * STEP,
            )
        self.assertGreater(envelope.latest_request.throttle_request, 0.0)
        envelope.reset()
        self.assertEqual(envelope.latest_request.throttle_request, 0.0)
        request = envelope.update(
            lateral(4), permitted(4, throttle=0.30), now=BASE + 4 * STEP
        )
        self.assertEqual(request.throttle_request, 0.0)
        self.assertTrue(request.throttle_rate_limited)

    def test_no_brake_survives_a_reset(self):
        envelope = CommandEnvelope()
        envelope.update(
            lateral(0),
            permitted(0, throttle=0.0, brake=0.40, target=1.0, observed=2.0),
            now=BASE,
        )
        envelope.reset()
        request = envelope.update(
            lateral(1), permitted(1, throttle=0.0), now=BASE + STEP
        )
        self.assertEqual(request.brake_request, 0.0)
        self.assertFalse(request.brake_release_rate_limited)

    def test_repeated_resets_are_idempotent_apart_from_the_counters(self):
        envelope = CommandEnvelope()
        first = envelope.reset()
        second = envelope.reset()
        self.assertNeutral(first, CommandEnvelopeReason.NOT_EVALUATED)
        self.assertNeutral(second, CommandEnvelopeReason.NOT_EVALUATED)
        self.assertEqual(envelope.metrics.resets, 2)

    def test_the_epoch_is_published_on_every_envelope(self):
        envelope = CommandEnvelope()
        envelope.reset()
        envelope.reset()
        command = envelope.update(lateral(0), permitted(0), now=BASE)
        self.assertEqual(command.controller_epoch, 2)
        refusal = envelope.update(lateral(1), bound_neutral(1), now=BASE + STEP)
        self.assertEqual(refusal.controller_epoch, 2)


# --------------------------------------------------------------------------
# M. Determinism, boundedness, and scope
# --------------------------------------------------------------------------


SCRIPT = (
    dict(kind="command", throttle=0.30),
    dict(kind="command", throttle=0.30),
    dict(kind="brake", brake=0.30),
    dict(kind="command", throttle=0.30),
    dict(kind="command", throttle=0.30),
    dict(kind="stop", brake=0.35),
    dict(kind="command", throttle=0.30),
    dict(kind="neutral"),
    dict(kind="command", throttle=0.30),
    dict(kind="mismatch"),
    dict(kind="command", throttle=0.30),
)


def run_script(envelope):
    published = []
    for index, item in enumerate(SCRIPT):
        now = BASE + index * STEP
        kind = item["kind"]
        if kind == "command":
            lat, lon = lateral(index), permitted(index, throttle=item["throttle"])
        elif kind == "brake":
            lat = lateral(index)
            lon = permitted(index, throttle=0.0, brake=item["brake"],
                            target=1.0, observed=2.0)
        elif kind == "stop":
            lat = refused_lateral(
                index, reason=LateralControlReason.INSUFFICIENT_GEOMETRY
            )
            lon = stopping(index, brake=item["brake"],
                           reason=LongitudinalControlReason.LATERAL_STOP,
                           lateral_mode=LateralControlMode.DISABLED)
        elif kind == "neutral":
            lat, lon = lateral(index), bound_neutral(index)
        else:
            lat, lon = lateral(index, revision=999), permitted(index)
        published.append(envelope.update(lat, lon, now=now))
    return published


class DeterminismTest(EnvelopeTestCase):
    def test_two_envelopes_replay_one_script_identically(self):
        first = run_script(CommandEnvelope())
        second = run_script(CommandEnvelope())
        self.assertEqual(first, second)

    def test_every_published_envelope_in_the_script_is_safe(self):
        for index, request in enumerate(run_script(CommandEnvelope())):
            with self.subTest(index):
                self.assertSafeAxes(request)
                if request.mode is CommandEnvelopeMode.DISABLED:
                    self.assertNeutral(request, request.reason)
                else:
                    self.assertBoundIdentity(request)

    def test_the_retained_object_graph_does_not_grow(self):
        envelope = CommandEnvelope()
        sizes = []
        for cycle in range(6):
            for index in range(6):
                position = cycle * 6 + index
                envelope.update(
                    lateral(position),
                    permitted(position, throttle=0.30),
                    now=BASE + position * STEP,
                )
            sizes.append(len(retained_ids(envelope)))
        self.assertEqual(len(set(sizes)), 1, sizes)

    def test_the_envelope_declares_exactly_one_config_and_one_state_slot(self):
        envelope = CommandEnvelope()
        self.assertFalse(hasattr(envelope, "__dict__"))
        self.assertEqual(CommandEnvelope.__slots__, ("_config", "_state"))
        for name in CommandEnvelope.__slots__:
            value = getattr(envelope, name)
            self.assertNotIsInstance(value, (list, dict, set))

    def test_the_whole_controller_state_is_one_frozen_record(self):
        envelope = CommandEnvelope()
        envelope.update(lateral(0), permitted(0), now=BASE)
        state = envelope._state
        self.assertEqual(
            tuple(field.name for field in fields(state)),
            ("request", "accepted", "evidence", "shaping", "metrics"),
        )
        for value in (state, state.evidence, state.shaping, state.metrics):
            with self.subTest(type(value).__name__):
                self.assertFalse(hasattr(value, "__dict__"))
                first = fields(value)[0].name
                with self.assertRaises(FrozenInstanceError):
                    setattr(value, first, 0.0)

    def test_every_reachable_retained_object_is_immutable(self):
        envelope = CommandEnvelope()
        for index in range(4):
            envelope.update(
                lateral(index), permitted(index, throttle=0.30),
                now=BASE + index * STEP,
            )
        for value in retained_objects(envelope):
            self.assertNotIsInstance(value, (list, dict, set, bytearray))
            self.assertNotIsInstance(value, BaseException)

    def test_the_metrics_record_is_immutable(self):
        envelope = CommandEnvelope()
        metrics = envelope.metrics
        self.assertIsInstance(metrics, CommandEnvelopeMetrics)
        with self.assertRaises(FrozenInstanceError):
            metrics.evaluations = 99

    def test_ten_thousand_updates_stay_bounded_and_safe(self):
        envelope = CommandEnvelope()
        worst_overshoot = 0.0
        for index in range(2000):
            now = BASE + index * STEP
            if index % 7 == 0:
                request = envelope.update(
                    lateral(index), bound_neutral(index), now=now
                )
            elif index % 5 == 0:
                request = envelope.update(
                    refused_lateral(
                        index, reason=LateralControlReason.INSUFFICIENT_GEOMETRY
                    ),
                    stopping(index,
                             reason=LongitudinalControlReason.LATERAL_STOP,
                             lateral_mode=LateralControlMode.DISABLED),
                    now=now,
                )
            elif index % 3 == 0:
                request = envelope.update(
                    lateral(index),
                    permitted(index, throttle=0.0, brake=0.30, target=1.0,
                              observed=2.0),
                    now=now,
                )
            else:
                request = envelope.update(
                    lateral(index), permitted(index, throttle=0.30), now=now
                )
            worst_overshoot = max(
                worst_overshoot,
                request.throttle_request - request.throttle_authority,
                request.brake_request - request.brake_authority,
                abs(request.steering_request) - request.steering_authority,
            )
            self.assertFalse(
                request.throttle_request > 0.0 and request.brake_request > 0.0
            )
        self.assertLessEqual(worst_overshoot, 0.0)
        self.assertEqual(envelope.metrics.evaluations, 2000)


# --------------------------------------------------------------------------
# N. R1: exact branch-specific speed-law shapes
# --------------------------------------------------------------------------


SPEED_LAW_FIELDS = (
    "effective_target_speed_mps",
    "observed_forward_speed_mps",
    "speed_error_mps",
    "speed_authority_mps",
)


def producer_speed_law_shape(reason):
    """The presence pattern accepted Mission 10A actually publishes.

    This oracle is derived from the accepted Mission 10A *source branches*, not
    from the Mission 10B table under test: `_refuse` publishes no speed evidence,
    `_neutral` and `_stop` publish the observed forward speed alone, and only
    `_control` publishes the complete quadruple.
    """
    unbound = {
        LongitudinalControlReason.NOT_EVALUATED,
        LongitudinalControlReason.TIME_INVALID,
        LongitudinalControlReason.SPEED_UNAVAILABLE,
        LongitudinalControlReason.IDENTITY_MISMATCH,
        LongitudinalControlReason.EVIDENCE_MALFORMED,
        LongitudinalControlReason.VEHICLE_NOT_ALIVE,
        LongitudinalControlReason.SOURCE_REJECTED,
    }
    complete = {
        LongitudinalControlReason.NOMINAL_TRACKING,
        LongitudinalControlReason.DEGRADED_TRACKING,
        LongitudinalControlReason.DEGRADED_COASTING,
        LongitudinalControlReason.TARGET_STOP,
    }
    if reason in unbound:
        return (False, False, False, False)
    if reason in complete:
        return (True, True, True, True)
    return (False, True, False, False)


def longitudinal_for_reason(reason, index=0, **overrides):
    """One accepted-shaped Mission 10A record for any reason, or ``None``."""
    if reason in (
        LongitudinalControlReason.NOMINAL_TRACKING,
        LongitudinalControlReason.DEGRADED_TRACKING,
        LongitudinalControlReason.DEGRADED_COASTING,
    ):
        if reason is LongitudinalControlReason.NOMINAL_TRACKING:
            return permitted(index, **overrides)
        if reason is LongitudinalControlReason.DEGRADED_TRACKING:
            return permitted(
                index,
                safety=SafetyState.DEGRADED,
                lateral_mode=LateralControlMode.DEGRADED,
                mode=LongitudinalControlMode.DEGRADED,
                reason=reason,
                throttle=0.05,
                throttle_authority=0.15,
                **overrides,
            )
        return permitted(
            index,
            safety=SafetyState.DEGRADED,
            lateral_mode=LateralControlMode.DEGRADED,
            temporal=TemporalLaneState.COASTING,
            mode=LongitudinalControlMode.DEGRADED,
            reason=reason,
            throttle=0.05,
            throttle_authority=0.10,
            **overrides,
        )
    if reason is LongitudinalControlReason.TARGET_STOP:
        return target_stop(index, **overrides)
    if reason is LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION:
        return bound_neutral(index, **overrides)
    if reason is LongitudinalControlReason.LATERAL_STOP:
        return stopping(
            index, reason=reason, lateral_mode=LateralControlMode.DISABLED,
            **overrides,
        )
    if reason is LongitudinalControlReason.SAFETY_STOP:
        return stopping(
            index, reason=reason, safety=SafetyState.RECOVERING,
            lateral_mode=LateralControlMode.DISABLED, **overrides,
        )
    if reason is LongitudinalControlReason.FAIL_SAFE_STOP:
        return stopping(
            index, reason=reason, safety=SafetyState.FAIL_SAFE,
            lateral_mode=LateralControlMode.DISABLED, **overrides,
        )
    if reason in (
        LongitudinalControlReason.TARGET_EXPIRED,
        LongitudinalControlReason.SPEED_STALE,
        LongitudinalControlReason.EXCESSIVE_LATERAL_MOTION,
        LongitudinalControlReason.REVERSE_MOTION,
    ):
        return stopping(index, reason=reason, **overrides)
    if reason is LongitudinalControlReason.NOT_EVALUATED:
        return NOT_EVALUATED_LONGITUDINAL
    if reason is LongitudinalControlReason.TIME_INVALID:
        return unbound_refusal(index, reason=reason, evaluated=None)
    return unbound_refusal(index, reason=reason)


class SpeedLawShapeTest(EnvelopeTestCase):
    def test_the_shape_table_covers_every_longitudinal_reason(self):
        self.assertEqual(set(_SPEED_LAW_SHAPE), set(LongitudinalControlReason))

    def test_the_shape_table_matches_the_accepted_producer_branches(self):
        for reason in LongitudinalControlReason:
            with self.subTest(reason):
                self.assertEqual(
                    _SPEED_LAW_SHAPE[reason], producer_speed_law_shape(reason)
                )

    def test_every_accepted_producer_shape_is_canonicalized(self):
        for reason in LongitudinalControlReason:
            with self.subTest(reason):
                record = longitudinal_for_reason(reason)
                canonical = _canonicalize_longitudinal_request(record)
                self.assertIs(canonical.reason, reason)
                shape = tuple(
                    getattr(canonical, name) is not None
                    for name in SPEED_LAW_FIELDS
                )
                self.assertEqual(shape, producer_speed_law_shape(reason))

    def test_every_other_presence_combination_is_malformed(self):
        values = {
            "effective_target_speed_mps": 2.0,
            "observed_forward_speed_mps": 1.0,
            "speed_error_mps": 1.0,
            "speed_authority_mps": 3.0,
        }
        checked = 0
        for reason in LongitudinalControlReason:
            record = longitudinal_for_reason(reason)
            allowed = producer_speed_law_shape(reason)
            for mask in range(16):
                shape = tuple(bool(mask & (1 << bit)) for bit in range(4))
                if shape == allowed:
                    continue
                overrides = {
                    name: (values[name] if present else None)
                    for name, present in zip(SPEED_LAW_FIELDS, shape)
                }
                forged = bypassed(record, **overrides)
                checked += 1
                with self.subTest(reason=reason, shape=shape):
                    with self.assertRaises(_MalformedEvidence):
                        _canonicalize_longitudinal_request(forged)
        self.assertEqual(checked, 19 * 15)

    def test_a_partial_shape_on_a_stop_never_reaches_output_construction(self):
        envelope = CommandEnvelope()
        forged = bypassed(
            stopping(0, reason=LongitudinalControlReason.SPEED_STALE),
            effective_target_speed_mps=1.0,
        )
        request = envelope.update(lateral(0), forged, now=BASE)
        self.assertNeutral(
            request, CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE
        )
        self.assertEqual(envelope.metrics.internal_errors, 0)

    def test_every_partial_shape_on_a_stop_is_a_neutral_refusal(self):
        record = stopping(0, reason=LongitudinalControlReason.SPEED_STALE)
        values = {
            "effective_target_speed_mps": 2.0,
            "observed_forward_speed_mps": 1.0,
            "speed_error_mps": 1.0,
            "speed_authority_mps": 3.0,
        }
        for mask in range(16):
            shape = tuple(bool(mask & (1 << bit)) for bit in range(4))
            if shape == (False, True, False, False):
                continue
            overrides = {
                name: (values[name] if present else None)
                for name, present in zip(SPEED_LAW_FIELDS, shape)
            }
            with self.subTest(shape=shape):
                envelope = CommandEnvelope()
                request = envelope.update(
                    lateral(0), bypassed(record, **overrides), now=BASE
                )
                self.assertNeutral(
                    request, CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE
                )
                self.assertEqual(envelope.metrics.internal_errors, 0)

    def test_a_forbidden_partial_shape_advances_no_baseline(self):
        envelope = CommandEnvelope()
        accepted = self.drive(envelope, index=0)
        self.assertTrue(accepted.command_allowed)
        forged = bypassed(
            stopping(1, reason=LongitudinalControlReason.SPEED_STALE),
            speed_authority_mps=3.0,
        )
        refused = envelope.update(lateral(1), forged, now=BASE + STEP)
        self.assertNeutral(
            refused, CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE
        )
        # The order baseline survived, so the same cut is still acceptable.
        later = BASE + 2 * STEP
        lat, lon = replay(0, later)
        again = envelope.update(lat, lon, now=later)
        self.assertTrue(again.command_allowed)

    def test_the_complete_law_arithmetic_is_only_required_where_it_exists(self):
        # A stop that publishes the observed speed alone has no law to reconcile.
        canonical = _canonicalize_longitudinal_request(
            stopping(0, reason=LongitudinalControlReason.SPEED_STALE, observed=7.5)
        )
        self.assertEqual(canonical.observed_forward_speed_mps, 7.5)
        self.assertIsNone(canonical.speed_error_mps)
        # A branch that does publish the law must reconcile it exactly.
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_longitudinal_request(
                bypassed(target_stop(0, observed=1.0), speed_error_mps=0.0)
            )

    def test_legitimate_speed_law_values_survive_canonicalization(self):
        for target, observed in (
            (2.0, 1.0),
            (0.5, 0.5),
            (HARD_TARGET_CEILING_MPS, 0.0),
            (1.0, -0.0),
            (1.0, math.nextafter(1.0, 0.0)),
            (1e-9, 1e-9),
        ):
            with self.subTest(target=target, observed=observed):
                record = permitted(
                    0,
                    target=target,
                    observed=observed,
                    speed_authority=HARD_TARGET_CEILING_MPS,
                )
                canonical = _canonicalize_longitudinal_request(record)
                self.assertEqual(
                    canonical.speed_error_mps, (target - observed) + 0.0
                )

    def test_a_one_ulp_speed_error_perturbation_is_malformed(self):
        record = permitted(0, target=2.0, observed=1.0)
        forged = bypassed(record, speed_error_mps=math.nextafter(1.0, 2.0))
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_longitudinal_request(forged)

    def test_a_valid_pair_still_recovers_after_a_forbidden_shape(self):
        envelope = CommandEnvelope()
        forged = bypassed(
            stopping(0, reason=LongitudinalControlReason.SPEED_STALE),
            speed_error_mps=1.0,
        )
        envelope.update(lateral(0), forged, now=BASE)
        request = envelope.update(
            lateral(1), permitted(1, throttle=0.30), now=BASE + STEP
        )
        self.assertTrue(request.command_allowed)
        self.assertEqual(request.throttle_request, 0.0)


# --------------------------------------------------------------------------
# O. R1: atomic settlement
# --------------------------------------------------------------------------


SETTLEMENT_TARGETS = (
    "_accepted_metrics",
    "_EvidenceState",
    "_ShapingState",
    "_ControllerState",
    "_provenance",
    "_fingerprint",
)


def accepted_paths():
    """One pair per accepted Mission 10B outcome, with a warm-up script."""
    return {
        "nominal throttle": ([dict(throttle=0.30)], dict(throttle=0.30)),
        "degraded command": (
            [dict(throttle=0.30)],
            dict(
                safety=SafetyState.DEGRADED,
                lateral_mode=LateralControlMode.DEGRADED,
                mode=LongitudinalControlMode.DEGRADED,
                reason=LongitudinalControlReason.DEGRADED_TRACKING,
                throttle=0.05,
                throttle_authority=0.15,
            ),
        ),
        "ordinary brake": (
            [dict(throttle=0.30)],
            dict(throttle=0.0, brake=0.375, target=1.0, observed=2.0),
        ),
        "transition interlock": (
            [dict(throttle=0.0, brake=0.375, target=1.0, observed=2.0)],
            dict(throttle=0.30),
        ),
    }


class AtomicSettlementTest(EnvelopeTestCase):
    def lateral_for(self, index, kwargs):
        if kwargs.get("safety") is SafetyState.DEGRADED:
            return lateral(
                index,
                mode=LateralControlMode.DEGRADED,
                reason=LateralControlReason.DEGRADED_TRACKING,
                safety=SafetyState.DEGRADED,
                authority=0.15,
            )
        return lateral(index)

    def warm(self, envelope, script):
        index = 0
        for kwargs in script:
            envelope.update(
                self.lateral_for(index, kwargs),
                permitted(index, **kwargs),
                now=BASE + index * STEP,
            )
            index += 1
        return index

    def inject(self, name, envelope, index, lat, lon):
        import control.command as module

        original = getattr(module, name)

        def exploder(*args, **kwargs):
            raise RuntimeError(f"{name} failed")

        setattr(module, name, exploder)
        try:
            with self.assertRaises(RuntimeError):
                envelope.update(lat, lon, now=BASE + index * STEP)
        finally:
            setattr(module, name, original)

    def test_no_accepted_baseline_moves_when_settlement_faults(self):
        for label, (script, final) in accepted_paths().items():
            for name in SETTLEMENT_TARGETS:
                with self.subTest(path=label, target=name):
                    envelope = CommandEnvelope()
                    index = self.warm(envelope, script)
                    before = envelope._state
                    self.inject(
                        name,
                        envelope,
                        index,
                        self.lateral_for(index, final),
                        permitted(index, **final),
                    )
                    after = envelope._state
                    self.assertIs(after.evidence, before.evidence)
                    self.assertIs(after.accepted, before.accepted)
                    self.assertIs(
                        after.request.reason, CommandEnvelopeReason.INTERNAL_ERROR
                    )
                    self.assertIsNot(after.request, before.request)
                    self.assertIs(after.shaping.shaping_monotonic, None)
                    self.assertEqual(after.metrics.commands, before.metrics.commands)
                    self.assertEqual(
                        after.metrics.internal_errors,
                        before.metrics.internal_errors + 1,
                    )

    def test_a_trusted_stop_settlement_fault_moves_no_baseline(self):
        envelope = CommandEnvelope()
        envelope.update(lateral(0), permitted(0, throttle=0.30), now=BASE)
        before = envelope._state
        self.inject(
            "_accepted_metrics",
            envelope,
            1,
            refused_lateral(1),
            stopping(1, reason=LongitudinalControlReason.LATERAL_STOP,
                     lateral_mode=LateralControlMode.DISABLED),
        )
        after = envelope._state
        self.assertIs(after.evidence, before.evidence)
        self.assertIs(after.accepted, before.accepted)
        self.assertIs(after.request.reason, CommandEnvelopeReason.INTERNAL_ERROR)

    def test_a_metrics_fault_never_advances_the_guidance_baseline(self):
        import control.command as module

        envelope = CommandEnvelope()
        envelope.update(lateral(0), permitted(0, throttle=0.30), now=BASE)
        baseline = envelope._state.evidence.guidance_stamp
        original = module._accepted_metrics
        module._accepted_metrics = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("metrics")
        )
        try:
            with self.assertRaises(RuntimeError):
                envelope.update(
                    lateral(1), permitted(1, throttle=0.30), now=BASE + STEP
                )
        finally:
            module._accepted_metrics = original
        self.assertEqual(envelope._state.evidence.guidance_stamp, baseline)

    def test_the_previous_accepted_output_is_not_the_visible_one(self):
        envelope = CommandEnvelope()
        accepted = envelope.update(lateral(0), permitted(0, throttle=0.30), now=BASE)
        self.inject(
            "_accepted_metrics", envelope, 1, lateral(1),
            permitted(1, throttle=0.30),
        )
        self.assertIsNot(envelope.latest_request, accepted)
        self.assertNeutral(
            envelope.latest_request, CommandEnvelopeReason.INTERNAL_ERROR
        )

    def test_the_envelope_recovers_after_a_settlement_fault(self):
        envelope = CommandEnvelope()
        envelope.update(lateral(0), permitted(0, throttle=0.30), now=BASE)
        self.inject(
            "_accepted_metrics", envelope, 1, lateral(1),
            permitted(1, throttle=0.30),
        )
        request = envelope.update(
            lateral(2), permitted(2, throttle=0.30), now=BASE + 2 * STEP
        )
        self.assertTrue(request.command_allowed)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertTrue(request.throttle_rate_limited)

    def test_the_accepted_metrics_helper_is_pure(self):
        import control.command as module

        envelope = CommandEnvelope()
        request = envelope.update(lateral(0), permitted(0), now=BASE)
        before = envelope.metrics
        produced = module._accepted_metrics(before, request)
        self.assertIsNot(produced, before)
        self.assertEqual(envelope.metrics, before)
        self.assertEqual(produced.commands, before.commands + 1)
        self.assertEqual(produced.evaluations, before.evaluations + 1)

    def test_a_reset_replaces_the_whole_state_at_once(self):
        envelope = CommandEnvelope()
        envelope.update(lateral(0), permitted(0, throttle=0.30), now=BASE)
        before = envelope._state
        envelope.reset()
        after = envelope._state
        self.assertIsNot(after, before)
        self.assertIsNone(after.accepted)
        self.assertIs(after.evidence.guidance_stamp, None)
        self.assertIs(after.shaping.shaping_monotonic, None)
        self.assertEqual(after.metrics.evaluations, before.metrics.evaluations)
        self.assertEqual(after.metrics.resets, before.metrics.resets + 1)

    def test_an_idempotent_replay_replaces_only_the_counters(self):
        envelope = CommandEnvelope()
        first = envelope.update(lateral(0), permitted(0), now=BASE)
        before = envelope._state
        again = envelope.update(lateral(0), permitted(0), now=BASE)
        after = envelope._state
        self.assertIs(again, first)
        self.assertIs(after.evidence, before.evidence)
        self.assertIs(after.shaping, before.shaping)
        self.assertIs(after.request, before.request)
        self.assertEqual(
            after.metrics.idempotent_replays, before.metrics.idempotent_replays + 1
        )


# --------------------------------------------------------------------------
# P. R1: complete public output invariants
# --------------------------------------------------------------------------


class PublicProvenanceTest(EnvelopeTestCase):
    def command(self, **overrides):
        base = dict(
            mode=CommandEnvelopeMode.NOMINAL,
            reason=CommandEnvelopeReason.NOMINAL_COMMAND,
            command_allowed=True,
            throttle_request=0.10,
            throttle_authority=THROTTLE_AUTHORITY,
            brake_authority=CONFIG.maximum_ordinary_brake,
            steering_request=0.10,
            steering_authority=CONFIG.maximum_steering,
            effective_target_speed_mps=2.0,
            speed_error_mps=1.0,
            speed_authority_mps=3.0,
            **command_provenance(),
        )
        base.update(overrides)
        return CommandEnvelopeRequest(**base)

    def test_the_reported_contradictory_command_is_rejected(self):
        with self.assertRaises(ValueError):
            self.command(
                safety_state=SafetyState.FAIL_SAFE,
                lateral_mode=LateralControlMode.DISABLED,
                lateral_reason=LateralControlReason.SAFETY_NOT_PERMITTED,
                longitudinal_mode=LongitudinalControlMode.STOPPING,
                longitudinal_reason=LongitudinalControlReason.FAIL_SAFE_STOP,
                speed_error_mps=999.0,
            )

    def test_the_mirrored_reason_mode_tables_match_the_accepted_upstream(self):
        import control.lateral as lateral_module
        import control.longitudinal as longitudinal_module

        self.assertEqual(_LATERAL_REASON_MODE, lateral_module._REASON_MODE)
        self.assertEqual(
            _LONGITUDINAL_REASON_MODE, longitudinal_module._REASON_MODE
        )

    def test_each_upstream_mode_must_match_its_own_upstream_reason(self):
        for name, value in (
            ("lateral_reason", LateralControlReason.INSUFFICIENT_GEOMETRY),
            ("lateral_mode", LateralControlMode.DEGRADED),
            ("longitudinal_reason", LongitudinalControlReason.SAFETY_STOP),
            ("longitudinal_mode", LongitudinalControlMode.DEGRADED),
        ):
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    self.command(**{name: value})

    def test_the_envelope_mode_must_match_its_longitudinal_mode(self):
        with self.assertRaises(ValueError):
            self.command(mode=CommandEnvelopeMode.DEGRADED,
                         reason=CommandEnvelopeReason.DEGRADED_COMMAND)

    def test_a_lateral_refusal_without_source_evidence_can_never_bind(self):
        for reason in (
            LateralControlReason.NOT_EVALUATED,
            LateralControlReason.TIME_INVALID,
        ):
            with self.subTest(reason):
                with self.assertRaises(ValueError):
                    self.command(
                        lateral_mode=LateralControlMode.DISABLED,
                        lateral_reason=reason,
                        steering_request=0.0,
                        steering_authority=0.0,
                    )

    def test_an_unpermitted_safety_state_cannot_carry_a_permitted_lateral(self):
        with self.assertRaises(ValueError):
            self.command(safety_state=SafetyState.FAIL_SAFE)

    def test_the_mode_selection_must_recompute_to_the_published_reason(self):
        with self.assertRaises(ValueError):
            self.command(
                temporal_state=TemporalLaneState.COASTING,
                lateral_reason=LateralControlReason.DEGRADED_COASTING,
                lateral_mode=LateralControlMode.DEGRADED,
            )
        with self.assertRaises(ValueError):
            self.command(
                safety_state=SafetyState.DEGRADED,
                lateral_mode=LateralControlMode.DEGRADED,
                lateral_reason=LateralControlReason.DEGRADED_TRACKING,
            )

    def test_a_permitted_lateral_reason_must_agree_about_coasting(self):
        with self.assertRaises(ValueError):
            CommandEnvelopeRequest(
                mode=CommandEnvelopeMode.DEGRADED,
                reason=CommandEnvelopeReason.DEGRADED_COMMAND,
                command_allowed=True,
                throttle_request=0.05,
                throttle_authority=0.15,
                brake_authority=CONFIG.maximum_ordinary_brake,
                effective_target_speed_mps=1.0,
                speed_error_mps=0.0,
                speed_authority_mps=1.0,
                **command_provenance(
                    degraded=True,
                    lateral_reason=LateralControlReason.DEGRADED_COASTING,
                ),
            )

    def test_every_stopping_reason_relationship_is_enforced(self):
        for reason, bad in (
            (LongitudinalControlReason.FAIL_SAFE_STOP, SafetyState.RECOVERING),
            (LongitudinalControlReason.SAFETY_STOP, SafetyState.NOMINAL),
        ):
            with self.subTest(reason):
                with self.assertRaises(ValueError):
                    CommandEnvelopeRequest(
                        mode=CommandEnvelopeMode.STOPPING,
                        reason=CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP,
                        command_allowed=True,
                        stop_required=True,
                        brake_request=0.20,
                        brake_authority=BRAKE_AUTHORITY,
                        **stop_provenance(
                            longitudinal_reason=reason,
                            safety_state=bad,
                            lateral_mode=LateralControlMode.DISABLED,
                            lateral_reason=(
                                LateralControlReason.SAFETY_NOT_PERMITTED
                            ),
                        ),
                    )

    def test_a_lateral_stop_requires_a_refused_lateral_axis(self):
        # A lateral stop is Mission 10A answering a *refused* Mission 9 record,
        # so a permitted lateral axis alongside one is unreachable.
        with self.assertRaises(ValueError):
            CommandEnvelopeRequest(
                mode=CommandEnvelopeMode.STOPPING,
                reason=CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP,
                command_allowed=True,
                stop_required=True,
                brake_request=0.20,
                brake_authority=BRAKE_AUTHORITY,
                steering_authority=LATERAL_AUTHORITY,
                **stop_provenance(
                    longitudinal_reason=LongitudinalControlReason.LATERAL_STOP,
                    lateral_mode=LateralControlMode.NOMINAL,
                    lateral_reason=LateralControlReason.NOMINAL_TRACKING,
                ),
            )

    def test_a_refused_lateral_axis_cannot_carry_steering(self):
        with self.assertRaises(ValueError):
            CommandEnvelopeRequest(
                mode=CommandEnvelopeMode.STOPPING,
                reason=CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP,
                command_allowed=True,
                stop_required=True,
                steering_request=0.10,
                steering_authority=0.35,
                brake_request=0.20,
                brake_authority=BRAKE_AUTHORITY,
                **stop_provenance(
                    longitudinal_reason=LongitudinalControlReason.LATERAL_STOP,
                    lateral_mode=LateralControlMode.DISABLED,
                    lateral_reason=LateralControlReason.INSUFFICIENT_GEOMETRY,
                ),
            )

    def test_a_trusted_stop_is_never_shaped_beyond_a_steering_clamp(self):
        for name in (
            "throttle_clamped",
            "brake_clamped",
            "throttle_rate_limited",
            "brake_release_rate_limited",
            "transition_interlocked",
        ):
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    CommandEnvelopeRequest(
                        mode=CommandEnvelopeMode.STOPPING,
                        reason=CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP,
                        command_allowed=True,
                        stop_required=True,
                        brake_request=0.20,
                        brake_authority=BRAKE_AUTHORITY,
                        shaping_applied=True,
                        **{name: True},
                        **stop_provenance(),
                    )

    def test_an_interlocked_envelope_cannot_carry_throttle(self):
        with self.assertRaises(ValueError):
            self.command(
                reason=CommandEnvelopeReason.NOMINAL_TRANSITION_INTERLOCK,
                transition_interlocked=True,
                shaping_applied=True,
                throttle_request=0.10,
            )

    def test_the_envelope_reason_must_match_its_own_shaping_flags(self):
        with self.assertRaises(ValueError):
            self.command(
                reason=CommandEnvelopeReason.NOMINAL_SHAPED,
                shaping_applied=False,
            )
        with self.assertRaises(ValueError):
            self.command(
                reason=CommandEnvelopeReason.NOMINAL_COMMAND,
                throttle_rate_limited=True,
                shaping_applied=True,
            )

    def test_the_speed_law_shape_must_match_the_longitudinal_branch(self):
        with self.assertRaises(ValueError):
            CommandEnvelopeRequest(
                mode=CommandEnvelopeMode.STOPPING,
                reason=CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP,
                command_allowed=True,
                stop_required=True,
                brake_request=0.20,
                brake_authority=BRAKE_AUTHORITY,
                effective_target_speed_mps=2.0,
                speed_error_mps=1.0,
                speed_authority_mps=3.0,
                **stop_provenance(),
            )

    def test_the_complete_law_arithmetic_is_enforced_on_the_output(self):
        with self.assertRaises(ValueError):
            self.command(speed_error_mps=999.0)
        with self.assertRaises(ValueError):
            self.command(effective_target_speed_mps=4.0, speed_error_mps=3.0)

    def test_a_permitted_envelope_requires_a_positive_effective_target(self):
        with self.assertRaises(ValueError):
            self.command(
                effective_target_speed_mps=0.0,
                observed_forward_speed_mps=1.0,
                speed_error_mps=-1.0,
            )

    def test_every_real_producer_output_still_constructs(self):
        """Every accepted Mission 10B output is rebuilt field for field."""
        rebuilt = 0
        for request in run_script(CommandEnvelope()):
            clone = CommandEnvelopeRequest(
                **{field.name: getattr(request, field.name)
                   for field in fields(CommandEnvelopeRequest)}
            )
            self.assertEqual(clone, request)
            rebuilt += 1
        self.assertGreater(rebuilt, 0)

    def test_a_generated_provenance_cross_product_rejects_the_unreachable(self):
        """Mutating one provenance field at a time must never widen the surface."""
        accepted = 0
        rejected = 0
        base = command_provenance()
        mutations = {
            "safety_state": list(SafetyState),
            "temporal_state": list(TemporalLaneState),
            "lateral_mode": list(LateralControlMode),
            "lateral_reason": list(LateralControlReason),
            "longitudinal_mode": list(LongitudinalControlMode),
            "longitudinal_reason": list(LongitudinalControlReason),
        }
        for name, values in mutations.items():
            for value in values:
                overrides = dict(base)
                overrides[name] = value
                try:
                    request = self.command(**overrides)
                except ValueError:
                    rejected += 1
                    continue
                accepted += 1
                # Anything that survives must be a coherent nominal command.
                self.assertIs(request.mode, CommandEnvelopeMode.NOMINAL)
                self.assertIs(
                    request.longitudinal_reason,
                    LongitudinalControlReason.NOMINAL_TRACKING,
                )
                self.assertIs(request.safety_state, SafetyState.NOMINAL)
                self.assertIs(request.lateral_mode, LateralControlMode.NOMINAL)
        self.assertEqual(accepted, len(mutations))
        self.assertEqual(rejected, sum(len(v) for v in mutations.values()) - accepted)


# --------------------------------------------------------------------------
# Q. R1: trustworthy internal-error time
# --------------------------------------------------------------------------


class InternalErrorTimeTest(EnvelopeTestCase):
    def explode(self, name="_fingerprint"):
        import control.command as module

        original = getattr(module, name)
        setattr(
            module, name,
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        return module, name, original

    def test_a_fault_after_canonicalization_publishes_the_canonical_time(self):
        envelope = CommandEnvelope()
        module, name, original = self.explode()
        try:
            with self.assertRaises(RuntimeError):
                envelope.update(lateral(0), permitted(0), now=BASE)
        finally:
            setattr(module, name, original)
        published = envelope.latest_request
        self.assertIs(published.reason, CommandEnvelopeReason.INTERNAL_ERROR)
        self.assertEqual(published.evaluation_monotonic, BASE)

    def test_negative_zero_is_canonicalized_before_publication(self):
        envelope = CommandEnvelope()
        module, name, original = self.explode()
        try:
            with self.assertRaises(RuntimeError):
                envelope.update(
                    lateral(0, now=-0.0), permitted(0, now=-0.0), now=-0.0
                )
        finally:
            setattr(module, name, original)
        published = envelope.latest_request
        self.assertEqual(published.evaluation_monotonic, 0.0)
        self.assertEqual(
            math.copysign(1.0, published.evaluation_monotonic), 1.0
        )

    def test_a_fault_without_a_trustworthy_time_publishes_none(self):
        import control.command as module

        # ``now`` is never canonicalized here, so the refusal path itself is the
        # only code that runs; faulting its record constructor leaves the
        # envelope with no trustworthy time to publish.
        envelope = CommandEnvelope()
        original = module.CommandEnvelopeRequest
        module.CommandEnvelopeRequest = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        try:
            with self.assertRaises(RuntimeError):
                envelope.update(lateral(0), permitted(0), now=float("nan"))
        finally:
            module.CommandEnvelopeRequest = original
        published = envelope.latest_request
        self.assertIs(published.reason, CommandEnvelopeReason.INTERNAL_ERROR)
        self.assertIsNone(published.evaluation_monotonic)

    def test_the_fail_closed_path_survives_a_broken_state_constructor(self):
        import control.command as module

        envelope = CommandEnvelope()
        envelope.update(lateral(0), permitted(0, throttle=0.30), now=BASE)
        before = envelope._state
        original = module._ControllerState
        module._ControllerState = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        try:
            with self.assertRaises(RuntimeError):
                envelope.update(
                    lateral(1), permitted(1, throttle=0.30), now=BASE + STEP
                )
        finally:
            module._ControllerState = original
        self.assertNeutral(
            envelope.latest_request, CommandEnvelopeReason.INTERNAL_ERROR
        )
        self.assertIs(envelope._state.evidence, before.evidence)
        self.assertIs(envelope._state.accepted, before.accepted)

    def test_the_internal_error_still_carries_no_upstream_provenance(self):
        envelope = CommandEnvelope()
        module, name, original = self.explode()
        try:
            with self.assertRaises(RuntimeError):
                envelope.update(lateral(0), permitted(0), now=BASE)
        finally:
            setattr(module, name, original)
        published = envelope.latest_request
        self.assertNeutral(published, CommandEnvelopeReason.INTERNAL_ERROR)
        self.assertEqual(published.evaluation_monotonic, BASE)

    def test_the_internal_error_clears_shaping_and_keeps_the_baseline(self):
        envelope = CommandEnvelope()
        for index in range(3):
            envelope.update(
                lateral(index), permitted(index, throttle=0.30),
                now=BASE + index * STEP,
            )
        before = envelope._state.evidence
        module, name, original = self.explode()
        try:
            with self.assertRaises(RuntimeError):
                envelope.update(
                    lateral(3), permitted(3, throttle=0.30), now=BASE + 3 * STEP
                )
        finally:
            setattr(module, name, original)
        self.assertIs(envelope._state.evidence, before)
        self.assertIsNone(envelope._state.shaping.shaping_monotonic)
        request = envelope.update(
            lateral(4), permitted(4, throttle=0.30), now=BASE + 4 * STEP
        )
        self.assertEqual(request.throttle_request, 0.0)


# --------------------------------------------------------------------------
# R. R2: fail-closed caller-controlled control time
# --------------------------------------------------------------------------


@numbers.Real.register
class ExplodingReal:
    """A registered ``numbers.Real`` whose float conversion raises."""

    def __init__(self, exception: BaseException) -> None:
        self._exception = exception

    def __float__(self) -> float:
        raise self._exception


class HostileClass:
    """An object whose ``__class__`` raises when a type check consults it."""

    @property
    def __class__(self):
        raise RuntimeError("hostile __class__")


HOSTILE_TIMES = (
    ExplodingReal(RuntimeError("hostile")),
    ExplodingReal(ValueError("hostile")),
    ExplodingReal(TypeError("hostile")),
    ExplodingReal(ArithmeticError("hostile")),
    ExplodingReal(MemoryError("hostile")),
    ExplodingReal(AttributeError("hostile")),
    DriftingReal(float("nan"), 1.0, 1.0),
    SneakyFloat(float("inf")),
    float("nan"),
    float("inf"),
    float("-inf"),
    True,
    False,
    "100.0",
    None,
    object(),
    [],
    HostileClass(),
)


class HostileControlTimeTest(EnvelopeTestCase):
    def test_a_raising_float_conversion_is_time_invalid_not_an_escape(self):
        envelope = CommandEnvelope()
        accepted = envelope.update(lateral(0), permitted(0, throttle=0.30), now=BASE)
        self.assertTrue(accepted.command_allowed)
        request = envelope.update(
            lateral(1), permitted(1), now=ExplodingReal(RuntimeError("hostile"))
        )
        self.assertNeutral(request, CommandEnvelopeReason.TIME_INVALID)
        self.assertIsNone(request.evaluation_monotonic)
        self.assertIs(envelope.latest_request, request)

    def test_every_hostile_control_time_publishes_a_neutral_refusal(self):
        for value in HOSTILE_TIMES:
            with self.subTest(repr(value)):
                envelope = CommandEnvelope()
                envelope.update(lateral(0), permitted(0, throttle=0.30), now=BASE)
                request = envelope.update(lateral(1), permitted(1), now=value)
                self.assertNeutral(request, CommandEnvelopeReason.TIME_INVALID)
                self.assertEqual(envelope.metrics.internal_errors, 0)

    def test_a_hostile_control_time_removes_the_previous_command(self):
        envelope = CommandEnvelope()
        for index in range(4):
            envelope.update(
                lateral(index), permitted(index, throttle=0.30),
                now=BASE + index * STEP,
            )
        self.assertGreater(envelope.latest_request.throttle_request, 0.0)
        request = envelope.update(
            lateral(4), permitted(4), now=ExplodingReal(RuntimeError("x"))
        )
        self.assertNeutral(request, CommandEnvelopeReason.TIME_INVALID)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertEqual(request.steering_request, 0.0)

    def test_a_hostile_control_time_clears_the_shaping_epoch(self):
        envelope = CommandEnvelope()
        for index in range(4):
            envelope.update(
                lateral(index), permitted(index, throttle=0.30),
                now=BASE + index * STEP,
            )
        envelope.update(lateral(4), permitted(4), now=ExplodingReal(RuntimeError("x")))
        self.assertIsNone(envelope._state.shaping.shaping_monotonic)
        request = envelope.update(
            lateral(5), permitted(5, throttle=0.30), now=BASE + 5 * STEP
        )
        self.assertTrue(request.command_allowed)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertTrue(request.throttle_rate_limited)

    def test_a_hostile_control_time_advances_no_accepted_baseline(self):
        envelope = CommandEnvelope()
        envelope.update(lateral(0), permitted(0, throttle=0.30), now=BASE)
        before = envelope._state.evidence
        envelope.update(lateral(1), permitted(1), now=ExplodingReal(RuntimeError("x")))
        self.assertIs(envelope._state.evidence, before)
        later = BASE + STEP
        lat, lon = replay(0, later)
        again = envelope.update(lat, lon, now=later)
        self.assertTrue(again.command_allowed)

    def test_a_hostile_control_time_keeps_the_counters_truthful(self):
        envelope = CommandEnvelope()
        envelope.update(lateral(0), permitted(0, throttle=0.30), now=BASE)
        before = envelope.metrics
        envelope.update(lateral(1), permitted(1), now=ExplodingReal(RuntimeError("x")))
        after = envelope.metrics
        self.assertEqual(after.evaluations, before.evaluations + 1)
        self.assertEqual(after.refusals, before.refusals + 1)
        self.assertEqual(after.commands, before.commands)
        self.assertEqual(after.internal_errors, 0)

    def test_a_hostile_control_time_retains_no_object_or_exception(self):
        envelope = CommandEnvelope()
        hostile = ExplodingReal(RuntimeError("x"))
        envelope.update(lateral(0), permitted(0), now=hostile)
        retained = retained_ids(envelope)
        self.assertNotIn(id(hostile), retained)
        for value in retained_objects(envelope):
            self.assertNotIsInstance(value, BaseException)

    def test_a_hostile_control_time_is_not_an_internal_programming_error(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            lateral(0), permitted(0), now=ExplodingReal(RuntimeError("x"))
        )
        self.assertIsNot(request.reason, CommandEnvelopeReason.INTERNAL_ERROR)
        self.assertIs(request.reason, CommandEnvelopeReason.TIME_INVALID)

    def test_an_internal_fault_after_a_valid_time_still_re_raises(self):
        import control.command as module

        envelope = CommandEnvelope()
        original = module._fingerprint
        module._fingerprint = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("internal")
        )
        try:
            with self.assertRaises(RuntimeError):
                envelope.update(lateral(0), permitted(0), now=BASE)
        finally:
            module._fingerprint = original
        self.assertNeutral(
            envelope.latest_request, CommandEnvelopeReason.INTERNAL_ERROR
        )
        self.assertEqual(envelope.latest_request.evaluation_monotonic, BASE)
        self.assertEqual(envelope.metrics.internal_errors, 1)

    def test_a_keyboard_interrupt_from_the_clock_still_propagates(self):
        envelope = CommandEnvelope()
        with self.assertRaises(KeyboardInterrupt):
            envelope.update(
                lateral(0), permitted(0), now=ExplodingReal(KeyboardInterrupt())
            )

    def test_negative_zero_is_still_a_perfectly_good_control_time(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            lateral(0, now=-0.0), permitted(0, now=-0.0), now=-0.0
        )
        self.assertTrue(request.command_allowed)
        self.assertEqual(request.evaluation_monotonic, 0.0)
        self.assertEqual(math.copysign(1.0, request.evaluation_monotonic), 1.0)

    def test_alternating_hostile_and_valid_traffic_stays_safe(self):
        envelope = CommandEnvelope()
        for index in range(12):
            now = BASE + index * STEP
            if index % 3 == 2:
                request = envelope.update(
                    lateral(index), permitted(index),
                    now=ExplodingReal(RuntimeError("x")),
                )
                self.assertNeutral(request, CommandEnvelopeReason.TIME_INVALID)
            else:
                request = envelope.update(
                    lateral(index), permitted(index, throttle=0.30), now=now
                )
                self.assertTrue(request.command_allowed)
            self.assertSafeAxes(request)
        self.assertEqual(envelope.metrics.internal_errors, 0)


# --------------------------------------------------------------------------
# S. R2: complete Mission 9 producer shape and pair reachability
# --------------------------------------------------------------------------


def lateral_producer_shapes():
    """Every accepted Mission 9 publishing branch, read off `control/lateral.py`.

    Each entry is ``(reason, bound, allowed safety states, allowed temporal
    states)`` where ``None`` in a set means the branch publishes nothing for that
    field at all.  The R3 reconstruction settles the temporal state too: the R2
    version of this table left it free and therefore admitted 19 combinations no
    accepted producer emits.  Restated here from the independent oracle rather
    than from any Mission 10B helper.
    """
    unbound = ((None,), (None,))
    shapes = [
        (LateralControlReason.NOT_EVALUATED, False, *unbound),
        (LateralControlReason.TIME_INVALID, False, *unbound),
        (LateralControlReason.IDENTITY_MISMATCH, False, *unbound),
        (LateralControlReason.EVIDENCE_MALFORMED, False, *unbound),
    ]
    for reason, (states, temporals) in REAL_BOUND_LATERAL_SHAPES.items():
        shapes.append((reason, True, states, temporals))
    return tuple(shapes)


def build_lateral(reason, bound, safety, temporal=TemporalLaneState.TRACKING):
    """One Mission 9 record of the requested shape, or ``None`` if impossible."""
    mode = _LATERAL_REASON_MODE[reason]
    allowed = mode in (LateralControlMode.NOMINAL, LateralControlMode.DEGRADED)
    _now, guidance, _speed, frame = cut(0)
    if reason is LateralControlReason.DEGRADED_COASTING:
        temporal = TemporalLaneState.COASTING
    kwargs = dict(
        mode=mode,
        reason=reason,
        lateral_allowed=allowed,
        evaluation_monotonic=(
            None if reason in (
                LateralControlReason.NOT_EVALUATED,
                LateralControlReason.TIME_INVALID,
            ) else BASE
        ),
        source_stamp=guidance if bound else None,
        snapshot_revision=frame if bound else None,
        safety_state=safety,
        temporal_state=(temporal if bound else None),
    )
    if allowed:
        kwargs.update(
            steering=0.10,
            raw_steering=0.10,
            steering_authority=LATERAL_AUTHORITY,
            lateral_error=0.05,
            heading_error=0.0,
        )
    return LateralControlRequest(**kwargs)


class LateralProducerShapeTest(EnvelopeTestCase):
    def test_every_accepted_producer_shape_is_canonicalized(self):
        checked = 0
        for reason, bound, states, temporals in lateral_producer_shapes():
            for safety in states:
                for temporal in temporals:
                    with self.subTest(
                        reason=reason,
                        bound=bound,
                        safety=safety,
                        temporal=temporal,
                    ):
                        record = build_lateral(
                            reason,
                            bound,
                            safety,
                            TemporalLaneState.TRACKING
                            if temporal is None
                            else temporal,
                        )
                        canonical = _canonicalize_lateral_request(record)
                        self.assertIs(canonical.reason, reason)
                        checked += 1
        self.assertEqual(checked, 45)

    def test_every_other_reason_state_binding_combination_is_malformed(self):
        checked = 0
        for reason in LateralControlReason:
            for bound in (False, True):
                for safety in (None,) + tuple(SafetyState):
                    for temporal in (None,) + tuple(TemporalLaneState):
                        if real_lateral_shape(reason, safety, temporal, bound):
                            continue
                        try:
                            record = build_lateral(
                                reason,
                                bound,
                                safety,
                                TemporalLaneState.TRACKING
                                if temporal is None
                                else temporal,
                            )
                        except ValueError:
                            continue  # the accepted constructor refuses it
                        if bound and record.temporal_state is not temporal:
                            continue  # the builder pinned a different estimate
                        if not bound and temporal is not None:
                            continue  # an unbound record never carries one
                        checked += 1
                        with self.subTest(
                            reason=reason,
                            bound=bound,
                            safety=safety,
                            temporal=temporal,
                        ):
                            with self.assertRaises(_MalformedEvidence):
                                _canonicalize_lateral_request(record)
        self.assertGreater(checked, 30)

    def test_the_reported_safety_refusal_with_a_permitted_state_is_malformed(self):
        record = refused_lateral(
            0,
            reason=LateralControlReason.SAFETY_NOT_PERMITTED,
            safety=SafetyState.NOMINAL,
        )
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_lateral_request(record)

    def test_the_reported_impossible_pair_never_becomes_a_trusted_stop(self):
        envelope = CommandEnvelope()
        request = envelope.update(
            refused_lateral(
                0,
                reason=LateralControlReason.SAFETY_NOT_PERMITTED,
                safety=SafetyState.NOMINAL,
            ),
            stopping(
                0,
                reason=LongitudinalControlReason.LATERAL_STOP,
                lateral_mode=LateralControlMode.DISABLED,
            ),
            now=BASE,
        )
        self.assertNeutral(
            request, CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE
        )
        self.assertIsNot(request.mode, CommandEnvelopeMode.STOPPING)
        self.assertEqual(request.brake_request, 0.0)

    def test_a_source_rejection_with_a_safety_state_is_malformed(self):
        record = bypassed(
            refused_lateral(
                0, reason=LateralControlReason.SOURCE_REJECTED, safety=None
            ),
            safety_state=SafetyState.NOMINAL,
        )
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_lateral_request(record)

    def test_an_unbound_evidence_refusal_cannot_carry_a_safety_state(self):
        record = bypassed(
            refused_lateral(
                0, reason=LateralControlReason.EVIDENCE_MALFORMED, safety=None,
                stamp=None, revision=None, temporal=None,
            ),
            safety_state=SafetyState.NOMINAL,
        )
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_lateral_request(record)

    def test_a_bound_evidence_refusal_requires_a_permitted_state(self):
        good = refused_lateral(
            0, reason=LateralControlReason.IDENTITY_MISMATCH,
            safety=SafetyState.DEGRADED,
        )
        self.assertIsNotNone(_canonicalize_lateral_request(good))
        bad = bypassed(good, safety_state=SafetyState.FAIL_SAFE)
        with self.assertRaises(_MalformedEvidence):
            _canonicalize_lateral_request(bad)

    def test_a_malformed_lateral_record_advances_no_baseline(self):
        envelope = CommandEnvelope()
        accepted = self.drive(envelope, index=0)
        self.assertTrue(accepted.command_allowed)
        refused = envelope.update(
            refused_lateral(
                1,
                reason=LateralControlReason.SAFETY_NOT_PERMITTED,
                safety=SafetyState.NOMINAL,
            ),
            stopping(1, reason=LongitudinalControlReason.LATERAL_STOP,
                     lateral_mode=LateralControlMode.DISABLED),
            now=BASE + STEP,
        )
        self.assertNeutral(
            refused, CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE
        )
        later = BASE + 2 * STEP
        lat, lon = replay(0, later)
        again = envelope.update(lat, lon, now=later)
        self.assertTrue(again.command_allowed)


class PairReachabilityTest(EnvelopeTestCase):
    def stop_record(self, **overrides):
        base = dict(
            mode=CommandEnvelopeMode.STOPPING,
            reason=CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP,
            command_allowed=True,
            stop_required=True,
            brake_request=0.20,
            brake_authority=BRAKE_AUTHORITY,
        )
        provenance = stop_provenance(
            longitudinal_reason=LongitudinalControlReason.LATERAL_STOP,
            lateral_mode=LateralControlMode.DISABLED,
            lateral_reason=LateralControlReason.INSUFFICIENT_GEOMETRY,
        )
        provenance.update(overrides)
        base.update(provenance)
        return CommandEnvelopeRequest(**base)

    def test_the_reported_public_constructor_tuple_is_rejected(self):
        with self.assertRaises(ValueError):
            self.stop_record(
                lateral_reason=LateralControlReason.SOURCE_REJECTED,
                safety_state=SafetyState.NOMINAL,
            )

    def test_a_legitimate_lateral_stop_still_constructs(self):
        request = self.stop_record()
        self.assertIs(request.mode, CommandEnvelopeMode.STOPPING)
        self.assertEqual(request.steering_request, 0.0)

    def test_every_lateral_reason_crossed_with_every_stopping_reason(self):
        accepted = 0
        rejected = 0
        for lateral_reason in LateralControlReason:
            for longitudinal_reason in LongitudinalControlReason:
                if _LONGITUDINAL_REASON_MODE[longitudinal_reason] is not (
                    LongitudinalControlMode.STOPPING
                ):
                    continue
                for safety in SafetyState:
                    with self.subTest(
                        lateral=lateral_reason,
                        longitudinal=longitudinal_reason,
                        safety=safety,
                    ):
                        lateral_mode = _LATERAL_REASON_MODE[lateral_reason]
                        try:
                            request = self.stop_record(
                                lateral_reason=lateral_reason,
                                lateral_mode=lateral_mode,
                                longitudinal_reason=longitudinal_reason,
                                safety_state=safety,
                                steering_request=0.0,
                                steering_authority=0.0,
                            )
                        except ValueError:
                            rejected += 1
                            continue
                        accepted += 1
                        # Anything accepted must satisfy the independently
                        # restated upstream relationships.  The oracle below is
                        # written from ``control/lateral.py`` and
                        # ``control/longitudinal.py`` directly; no Mission 10B
                        # helper is consulted to decide what to expect.
                        self.assertTrue(
                            real_lateral_shape(
                                lateral_reason,
                                safety,
                                TemporalLaneState.TRACKING,
                                True,
                            ),
                            "an unreachable Mission 9 shape was accepted",
                        )
                        self.assertIn(
                            longitudinal_reason,
                            REAL_SAME_CUT_LONGITUDINAL[lateral_reason],
                            "an unreachable same-cut pair was accepted",
                        )
        self.assertGreater(accepted, 0)
        self.assertGreater(rejected, accepted)

    def test_every_real_pipeline_output_still_reconstructs(self):
        rebuilt = 0
        for request in run_script(CommandEnvelope()):
            clone = CommandEnvelopeRequest(
                **{field.name: getattr(request, field.name)
                   for field in fields(CommandEnvelopeRequest)}
            )
            self.assertEqual(clone, request)
            rebuilt += 1
        self.assertGreater(rebuilt, 0)


# --------------------------------------------------------------------------
# T. R2: exact Mission 10A unbound refusal time shapes
# --------------------------------------------------------------------------


class UnboundRefusalTimeTest(EnvelopeTestCase):
    def test_the_table_covers_every_unbound_reason(self):
        import control.command as module

        self.assertEqual(
            set(module._UNBOUND_TIME_SHAPE), module._UNBOUND_LONGITUDINAL_REASONS
        )

    def test_each_unbound_reason_accepts_exactly_its_own_time_shape(self):
        import control.command as module

        for reason, allowed in module._UNBOUND_TIME_SHAPE.items():
            for present in (False, True):
                with self.subTest(reason=reason, present=present):
                    if reason is LongitudinalControlReason.NOT_EVALUATED and present:
                        # The accepted Mission 10A constructor forbids it, so the
                        # record cannot even be rebuilt.
                        record = bypassed(
                            NOT_EVALUATED_LONGITUDINAL, evaluation_monotonic=BASE
                        )
                        with self.assertRaises(_MalformedEvidence):
                            _canonicalize_longitudinal_request(record)
                        continue
                    record = unbound_refusal(
                        0, reason=reason, evaluated=(BASE if present else None)
                    )
                    if present in allowed:
                        canonical = _canonicalize_longitudinal_request(record)
                        self.assertIs(canonical.reason, reason)
                    else:
                        with self.assertRaises(_MalformedEvidence):
                            _canonicalize_longitudinal_request(record)

    def test_the_time_refusal_really_does_publish_both_shapes(self):
        """Accepted Mission 10A reaches TIME_INVALID from two different gates.

        The two clock gates in `LongitudinalController.update` refuse with no
        time at all; the `_TimeInvalid` raised out of `_read_evidence` refuses
        with the canonical control time already in hand.  Both are real, so
        Mission 10B accepts both and invents nothing.
        """
        without = unbound_refusal(
            0, reason=LongitudinalControlReason.TIME_INVALID, evaluated=None
        )
        withtime = unbound_refusal(
            0, reason=LongitudinalControlReason.TIME_INVALID, evaluated=BASE
        )
        self.assertIsNone(
            _canonicalize_longitudinal_request(without).evaluation_monotonic
        )
        self.assertEqual(
            _canonicalize_longitudinal_request(withtime).evaluation_monotonic, BASE
        )

    def test_a_time_bearing_refusal_without_its_time_is_malformed(self):
        for reason in (
            LongitudinalControlReason.SPEED_UNAVAILABLE,
            LongitudinalControlReason.IDENTITY_MISMATCH,
            LongitudinalControlReason.EVIDENCE_MALFORMED,
            LongitudinalControlReason.VEHICLE_NOT_ALIVE,
            LongitudinalControlReason.SOURCE_REJECTED,
        ):
            with self.subTest(reason):
                record = unbound_refusal(0, reason=reason, evaluated=None)
                with self.assertRaises(_MalformedEvidence):
                    _canonicalize_longitudinal_request(record)

    def test_a_negative_zero_refusal_time_is_folded(self):
        record = unbound_refusal(
            0, reason=LongitudinalControlReason.SPEED_UNAVAILABLE, evaluated=-0.0
        )
        canonical = _canonicalize_longitudinal_request(record)
        self.assertEqual(canonical.evaluation_monotonic, 0.0)
        self.assertEqual(
            math.copysign(1.0, canonical.evaluation_monotonic), 1.0
        )

    def test_a_non_finite_refusal_time_is_malformed(self):
        for value in (float("nan"), float("inf"), float("-inf"), True, "x"):
            with self.subTest(value):
                record = bypassed(
                    unbound_refusal(
                        0, reason=LongitudinalControlReason.SPEED_UNAVAILABLE
                    ),
                    evaluation_monotonic=value,
                )
                with self.assertRaises(_MalformedEvidence):
                    _canonicalize_longitudinal_request(record)

    def test_a_malformed_refusal_time_advances_no_baseline(self):
        envelope = CommandEnvelope()
        accepted = self.drive(envelope, index=0)
        self.assertTrue(accepted.command_allowed)
        forged = unbound_refusal(
            1, reason=LongitudinalControlReason.SPEED_UNAVAILABLE, evaluated=None
        )
        refused = envelope.update(lateral(1), forged, now=BASE + STEP)
        self.assertNeutral(
            refused, CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE
        )
        later = BASE + 2 * STEP
        lat, lon = replay(0, later)
        again = envelope.update(lat, lon, now=later)
        self.assertTrue(again.command_allowed)


class ScopeTest(EnvelopeTestCase):
    def test_the_module_imports_exactly_the_allowed_names(self):
        import control.command as module

        with open(module.__file__, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        imported: set[str] = set()
        deferred = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    imported.add(node.module.split(".")[0])
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for inner in ast.walk(node):
                    if isinstance(inner, (ast.Import, ast.ImportFrom)):
                        deferred.append(node.name)
        self.assertEqual(deferred, [])
        self.assertEqual(
            imported,
            {
                "__future__",
                "control",
                "dataclasses",
                "enum",
                "math",
                "numbers",
                "perception",
                "safety",
                "telemetry",
            },
        )

    def test_the_module_names_no_actuator_or_worker_surface(self):
        import control.command as module

        with open(module.__file__, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        identifiers: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                identifiers.add(node.name)
            elif isinstance(node, ast.arg):
                identifiers.add(node.arg)
            elif isinstance(node, ast.keyword) and node.arg is not None:
                identifiers.add(node.arg)
        lowered = {name.lower() for name in identifiers}
        for forbidden in (
            "carla",
            "vehiclecontrol",
            "apply_control",
            "autopilot",
            "numpy",
            "pygame",
            "scipy",
            "cv2",
            "socket",
            "subprocess",
            "asyncio",
            "thread",
            "queue",
            "timer",
            "getenv",
            "environ",
            "open",
        ):
            matches = sorted(name for name in lowered if forbidden in name)
            self.assertEqual(matches, [], f"{forbidden}: {matches}")

    def test_the_module_namespace_holds_no_forbidden_dependency(self):
        import control.command as module

        names = set(module.__dict__)
        for forbidden in (
            "carla",
            "numpy",
            "pygame",
            "scipy",
            "cv2",
            "socket",
            "subprocess",
            "asyncio",
            "threading",
            "multiprocessing",
            "os",
            "sys",
            "time",
        ):
            self.assertNotIn(forbidden, names)

    def test_a_fresh_isolated_import_creates_no_thread_file_or_socket(self):
        code = (
            "import sys, threading\n"
            "before = threading.active_count()\n"
            "import control.command as module\n"
            "bad = [n for n in ('carla','pygame','numpy','scipy','cv2','socket',"
            "'subprocess','asyncio','multiprocessing','manual_drive') "
            "if n in sys.modules]\n"
            "assert not bad, bad\n"
            "assert threading.active_count() == before, threading.active_count()\n"
            "envelope = module.CommandEnvelope()\n"
            "assert envelope.latest_request.reason is "
            "module.CommandEnvelopeReason.NOT_EVALUATED\n"
            "assert threading.active_count() == before\n"
            "print('IMPORT_SAFE')\n"
        )
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "IMPORT_SAFE")

    def test_the_control_package_surface_is_unchanged(self):
        import control

        self.assertEqual(
            control.__all__,
            (
                "DEFAULT_LATERAL_CONTROLLER_CONFIG",
                "MAX_LATERAL_GAIN",
                "MAX_STEERING_RATE_PER_SECOND",
                "LateralControlMode",
                "LateralControlReason",
                "LateralControlRequest",
                "LateralController",
                "LateralControllerConfig",
                "LateralControllerMetrics",
            ),
        )

    def test_no_public_package_name_exposes_a_longitudinal_command(self):
        import control

        for name in control.__all__:
            lowered = name.lower()
            for forbidden in ("throttle", "brake", "speed", "accel"):
                self.assertNotIn(forbidden, lowered, name)

    def test_the_module_exports_exactly_its_public_vocabulary(self):
        import control.command as module

        self.assertEqual(
            module.__all__,
            (
                "DEFAULT_COMMAND_ENVELOPE_CONFIG",
                "MAX_ENVELOPE_RATE_PER_SECOND",
                "MAX_SHAPING_DELTA_SECONDS",
                "CommandEnvelope",
                "CommandEnvelopeConfig",
                "CommandEnvelopeMetrics",
                "CommandEnvelopeMode",
                "CommandEnvelopeReason",
                "CommandEnvelopeRequest",
            ),
        )

    def test_the_output_record_is_named_a_request_because_nothing_is_applied(self):
        self.assertEqual(CommandEnvelopeRequest.__name__, "CommandEnvelopeRequest")


# ==========================================================================
# R3.A  Complete Mission 9 producer reconstruction
# ==========================================================================


class LateralProducerMatrixTest(EnvelopeTestCase):
    """Every (reason, safety, temporal, bound) shape, against the independent
    oracle written from ``control/lateral.py`` rather than from Mission 10B."""

    def accepted(self, reason, safety, temporal, bound):
        """Does the production canonicalizer accept this Mission 9 shape?"""
        record = bypassed(
            NOT_EVALUATED_LATERAL,
            mode=_LATERAL_REASON_MODE[reason],
            reason=reason,
            lateral_allowed=_LATERAL_REASON_MODE[reason]
            in (LateralControlMode.NOMINAL, LateralControlMode.DEGRADED),
            steering=0.0,
            raw_steering=0.0,
            steering_authority=(
                LATERAL_AUTHORITY
                if _LATERAL_REASON_MODE[reason]
                in (LateralControlMode.NOMINAL, LateralControlMode.DEGRADED)
                else 0.0
            ),
            clamped=False,
            rate_limited=False,
            evaluation_monotonic=(
                None if reason in REAL_EVIDENCE_FREE_LATERAL_REASONS else BASE
            ),
            source_stamp=(
                SampleStamp(frame=5, sim_time=1.25, monotonic=BASE - 0.05)
                if bound
                else None
            ),
            snapshot_revision=5 if bound else None,
            lateral_error=(
                0.05
                if _LATERAL_REASON_MODE[reason]
                in (LateralControlMode.NOMINAL, LateralControlMode.DEGRADED)
                else None
            ),
            heading_error=(
                0.0
                if _LATERAL_REASON_MODE[reason]
                in (LateralControlMode.NOMINAL, LateralControlMode.DEGRADED)
                else None
            ),
            safety_state=safety,
            temporal_state=temporal,
        )
        try:
            _canonicalize_lateral_request(record)
        except _MalformedEvidence:
            return False
        return True

    def test_every_shape_agrees_with_the_independent_oracle(self):
        checked = 0
        for reason in LateralControlReason:
            for safety in (None, *SafetyState):
                for temporal in (None, *TemporalLaneState):
                    for bound in (False, True):
                        with self.subTest(
                            reason=reason,
                            safety=safety,
                            temporal=temporal,
                            bound=bound,
                        ):
                            self.assertEqual(
                                self.accepted(reason, safety, temporal, bound),
                                real_lateral_shape(reason, safety, temporal, bound),
                            )
                            checked += 1
        self.assertEqual(checked, len(LateralControlReason) * 6 * 6 * 2)

    def test_exactly_forty_five_producer_shapes_are_reachable(self):
        reachable = [
            (reason, safety, temporal, bound)
            for reason in LateralControlReason
            for safety in (None, *SafetyState)
            for temporal in (None, *TemporalLaneState)
            for bound in (False, True)
            if real_lateral_shape(reason, safety, temporal, bound)
        ]
        self.assertEqual(len(reachable), 45)
        self.assertEqual(sum(1 for shape in reachable if shape[3]), 41)
        self.assertEqual(sum(1 for shape in reachable if not shape[3]), 4)

    def test_no_bound_record_may_carry_an_uninitialized_estimate(self):
        for reason in LateralControlReason:
            for safety in (None, *SafetyState):
                with self.subTest(reason=reason, safety=safety):
                    self.assertFalse(
                        self.accepted(
                            reason, safety, TemporalLaneState.UNINITIALIZED, True
                        )
                    )

    def test_a_source_rejection_never_carries_a_safety_state(self):
        for safety in SafetyState:
            with self.subTest(safety=safety):
                self.assertFalse(
                    self.accepted(
                        LateralControlReason.SOURCE_REJECTED,
                        safety,
                        TemporalLaneState.TRACKING,
                        True,
                    )
                )
        self.assertTrue(
            self.accepted(
                LateralControlReason.SOURCE_REJECTED,
                None,
                TemporalLaneState.TRACKING,
                True,
            )
        )

    def test_a_safety_refusal_carries_only_unpermitted_states_or_absence(self):
        for safety in (None, *SafetyState):
            expected = safety in (None, *REAL_UNPERMITTED_SAFETY)
            with self.subTest(safety=safety):
                self.assertEqual(
                    self.accepted(
                        LateralControlReason.SAFETY_NOT_PERMITTED,
                        safety,
                        TemporalLaneState.LOST,
                        True,
                    ),
                    expected,
                )

    def test_a_geometry_refusal_is_degraded_over_guidance_evidence_only(self):
        for safety in (None, *SafetyState):
            for temporal in TemporalLaneState:
                expected = (
                    safety is SafetyState.DEGRADED
                    and temporal in REAL_GUIDANCE_TEMPORAL
                )
                with self.subTest(safety=safety, temporal=temporal):
                    self.assertEqual(
                        self.accepted(
                            LateralControlReason.INSUFFICIENT_GEOMETRY,
                            safety,
                            temporal,
                            True,
                        ),
                        expected,
                    )

    def test_degraded_permitted_reasons_are_never_nominal_safety(self):
        for reason, temporal in (
            (LateralControlReason.DEGRADED_TRACKING, TemporalLaneState.TRACKING),
            (LateralControlReason.DEGRADED_COASTING, TemporalLaneState.COASTING),
        ):
            with self.subTest(reason=reason):
                self.assertFalse(
                    self.accepted(reason, SafetyState.NOMINAL, temporal, True)
                )
                self.assertTrue(
                    self.accepted(reason, SafetyState.DEGRADED, temporal, True)
                )

    def test_unbound_forms_stay_unbound_and_expose_no_state(self):
        for reason in LateralControlReason:
            with self.subTest(reason=reason):
                expected = reason in (
                    *REAL_EVIDENCE_FREE_LATERAL_REASONS,
                    *REAL_UNBOUND_LATERAL_REASONS,
                )
                self.assertEqual(self.accepted(reason, None, None, False), expected)

    def test_the_production_table_states_the_same_relationships(self):
        """Pin the production table against the independent restatement."""
        import control.command as module

        self.assertEqual(
            set(module._BOUND_TEMPORAL_STATES), set(REAL_BOUND_TEMPORAL)
        )
        self.assertEqual(
            set(module._UNPERMITTED_SAFETY_STATES), set(REAL_UNPERMITTED_SAFETY)
        )
        self.assertEqual(
            set(module._BOUND_LATERAL_SHAPES), set(REAL_BOUND_LATERAL_SHAPES)
        )
        for reason, (states, temporals) in REAL_BOUND_LATERAL_SHAPES.items():
            with self.subTest(reason=reason):
                produced_states, produced_temporals = module._BOUND_LATERAL_SHAPES[
                    reason
                ]
                self.assertEqual(set(produced_states), set(states))
                self.assertEqual(set(produced_temporals), set(temporals))


# ==========================================================================
# R3.B  Same-cut M9 -> M10A pair reachability
# ==========================================================================


class SameCutPairMatrixTest(EnvelopeTestCase):
    """The confirmed BLOCKER: an impossible pair must never become actionable."""

    def envelope_for(self, lateral_record, longitudinal_record, now=BASE):
        envelope = CommandEnvelope(CONFIG)
        return envelope, envelope.update(lateral_record, longitudinal_record, now=now)

    def bound_pair(
        self,
        lateral_reason,
        longitudinal_reason,
        *,
        safety=SafetyState.NOMINAL,
        temporal=TemporalLaneState.TRACKING,
        index=0,
    ):
        """One field-consistent pair for any two reasons, reachable or not.

        Both records are built with their accepted constructors *bypassed*, so
        combinations no producer emits still reach Mission 10B and are judged
        there rather than being quietly filtered out by an upstream validator.
        """
        control_now, guidance, speed_stamp, frame = cut(index)
        lateral_mode = _LATERAL_REASON_MODE[lateral_reason]
        allowed = lateral_mode in (
            LateralControlMode.NOMINAL,
            LateralControlMode.DEGRADED,
        )
        lateral_record = bypassed(
            NOT_EVALUATED_LATERAL,
            mode=lateral_mode,
            reason=lateral_reason,
            lateral_allowed=allowed,
            steering=0.10 if allowed else 0.0,
            raw_steering=0.10 if allowed else 0.0,
            steering_authority=LATERAL_AUTHORITY if allowed else 0.0,
            clamped=False,
            rate_limited=False,
            evaluation_monotonic=control_now,
            source_stamp=guidance,
            snapshot_revision=frame,
            lateral_error=0.05 if allowed else None,
            heading_error=0.0 if allowed else None,
            safety_state=safety,
            temporal_state=temporal,
        )
        longitudinal_mode = _LONGITUDINAL_REASON_MODE[longitudinal_reason]
        kwargs = dict(
            binding(index, safety=safety, lateral_mode=lateral_mode, temporal=temporal),
            mode=longitudinal_mode,
            reason=longitudinal_reason,
            longitudinal_allowed=longitudinal_mode
            in (LongitudinalControlMode.NOMINAL, LongitudinalControlMode.DEGRADED),
            stop_required=longitudinal_mode is LongitudinalControlMode.STOPPING,
            observed_forward_speed_mps=1.0,
        )
        if longitudinal_mode is LongitudinalControlMode.STOPPING:
            kwargs.update(brake_request=0.20, brake_authority=BRAKE_AUTHORITY)
        elif longitudinal_mode is not LongitudinalControlMode.DISABLED:
            kwargs.update(
                throttle_request=0.20,
                throttle_authority=THROTTLE_AUTHORITY,
                brake_authority=BRAKE_AUTHORITY,
                effective_target_speed_mps=2.0,
                speed_error_mps=1.0,
                speed_authority_mps=3.0,
            )
        if longitudinal_reason in (
            LongitudinalControlReason.NOMINAL_TRACKING,
            LongitudinalControlReason.DEGRADED_TRACKING,
            LongitudinalControlReason.DEGRADED_COASTING,
            LongitudinalControlReason.TARGET_STOP,
        ):
            kwargs.update(
                effective_target_speed_mps=(
                    0.0
                    if longitudinal_reason is LongitudinalControlReason.TARGET_STOP
                    else 2.0
                ),
                speed_error_mps=(
                    -1.0
                    if longitudinal_reason is LongitudinalControlReason.TARGET_STOP
                    else 1.0
                ),
                speed_authority_mps=3.0,
            )
        return lateral_record, bypassed(NOT_EVALUATED_LONGITUDINAL, **kwargs)

    # ------------------------------------------------- the reported blocker

    def test_the_reported_impossible_pair_never_becomes_a_trusted_stop(self):
        lateral_record, longitudinal_record = self.bound_pair(
            LateralControlReason.IDENTITY_MISMATCH,
            LongitudinalControlReason.LATERAL_STOP,
        )
        envelope, request = self.envelope_for(lateral_record, longitudinal_record)
        self.assertNeutral(
            request, CommandEnvelopeReason.CROSS_RECORD_IDENTITY_MISMATCH
        )
        self.assertIsNot(request.reason, CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP)
        self.assertFalse(request.command_allowed)
        self.assertFalse(request.stop_required)
        for value in (
            request.steering_request,
            request.throttle_request,
            request.brake_request,
            request.steering_authority,
            request.throttle_authority,
            request.brake_authority,
        ):
            self.assertEqual(value, 0.0)
        self.assertIsNone(request.guidance_stamp)
        self.assertIsNone(request.lateral_reason)
        self.assertIsNone(request.longitudinal_reason)
        self.assertFalse(request.shaping_applied)
        self.assertEqual(envelope.metrics.commands, 0)
        self.assertEqual(envelope.metrics.stop_commands, 0)
        self.assertEqual(envelope.metrics.refusals, 1)

    def test_the_reported_pair_advances_no_baseline_and_clears_shaping(self):
        envelope = CommandEnvelope(CONFIG)
        first = envelope.update(lateral(0), permitted(0), now=BASE)
        self.assertTrue(first.command_allowed)
        state = envelope._state
        baseline = state.evidence
        lateral_record, longitudinal_record = self.bound_pair(
            LateralControlReason.IDENTITY_MISMATCH,
            LongitudinalControlReason.LATERAL_STOP,
            index=1,
        )
        envelope.update(lateral_record, longitudinal_record, now=BASE + STEP)
        after = envelope._state
        self.assertEqual(after.evidence, baseline)
        self.assertIs(after.accepted, state.accepted)
        self.assertEqual(after.shaping.throttle_anchor, 0.0)
        self.assertEqual(after.shaping.brake_anchor, 0.0)
        self.assertIsNone(after.shaping.shaping_monotonic)
        self.assertIs(after.shaping.transition, _Transition.IDLE)

    def test_a_later_valid_pair_recovers_safely_after_the_refusal(self):
        envelope = CommandEnvelope(CONFIG)
        lateral_record, longitudinal_record = self.bound_pair(
            LateralControlReason.IDENTITY_MISMATCH,
            LongitudinalControlReason.LATERAL_STOP,
        )
        envelope.update(lateral_record, longitudinal_record, now=BASE)
        recovered = envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertTrue(recovered.command_allowed)
        self.assertIs(recovered.mode, CommandEnvelopeMode.NOMINAL)
        self.assertEqual(recovered.brake_request, 0.0)

    # ------------------------------------------------ the required families

    def test_bound_identity_mismatch_never_pairs_with_any_actionable_stop(self):
        for longitudinal_reason in LongitudinalControlReason:
            if _LONGITUDINAL_REASON_MODE[longitudinal_reason] is not (
                LongitudinalControlMode.STOPPING
            ):
                continue
            for lateral_reason in (
                LateralControlReason.IDENTITY_MISMATCH,
                LateralControlReason.EVIDENCE_MALFORMED,
            ):
                with self.subTest(lateral=lateral_reason, longitudinal=longitudinal_reason):
                    pair = self.bound_pair(lateral_reason, longitudinal_reason)
                    _envelope, request = self.envelope_for(*pair)
                    self.assertFalse(request.command_allowed)
                    self.assertIs(request.mode, CommandEnvelopeMode.DISABLED)

    def test_source_rejected_never_pairs_with_any_actionable_stop(self):
        for longitudinal_reason in LongitudinalControlReason:
            if _LONGITUDINAL_REASON_MODE[longitudinal_reason] is not (
                LongitudinalControlMode.STOPPING
            ):
                continue
            with self.subTest(longitudinal=longitudinal_reason):
                pair = self.bound_pair(
                    LateralControlReason.SOURCE_REJECTED,
                    longitudinal_reason,
                    safety=None,
                )
                _envelope, request = self.envelope_for(*pair)
                self.assertFalse(request.command_allowed)

    def test_a_safety_refusal_pairs_only_with_its_own_stop_families(self):
        for longitudinal_reason in LongitudinalControlReason:
            if _LONGITUDINAL_REASON_MODE[longitudinal_reason] is not (
                LongitudinalControlMode.STOPPING
            ):
                continue
            expected = (
                longitudinal_reason
                in REAL_SAME_CUT_LONGITUDINAL[
                    LateralControlReason.SAFETY_NOT_PERMITTED
                ]
            )
            with self.subTest(longitudinal=longitudinal_reason):
                pair = self.bound_pair(
                    LateralControlReason.SAFETY_NOT_PERMITTED,
                    longitudinal_reason,
                    safety=SafetyState.FAIL_SAFE,
                    temporal=TemporalLaneState.LOST,
                )
                _envelope, request = self.envelope_for(*pair)
                self.assertEqual(request.command_allowed, expected)

    def test_a_geometry_refusal_pairs_only_with_a_lateral_stop(self):
        for longitudinal_reason in LongitudinalControlReason:
            if _LONGITUDINAL_REASON_MODE[longitudinal_reason] is not (
                LongitudinalControlMode.STOPPING
            ):
                continue
            expected = (
                longitudinal_reason is LongitudinalControlReason.LATERAL_STOP
            )
            with self.subTest(longitudinal=longitudinal_reason):
                pair = self.bound_pair(
                    LateralControlReason.INSUFFICIENT_GEOMETRY,
                    longitudinal_reason,
                    safety=SafetyState.DEGRADED,
                    temporal=TemporalLaneState.TRACKING,
                )
                _envelope, request = self.envelope_for(*pair)
                self.assertEqual(request.command_allowed, expected)
                if expected:
                    self.assertIs(
                        request.reason, CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP
                    )

    def test_a_permitted_lateral_record_pairs_with_every_genuine_stop(self):
        for longitudinal_reason, safety, temporal, lateral_reason in (
            (
                LongitudinalControlReason.TARGET_STOP,
                SafetyState.NOMINAL,
                TemporalLaneState.TRACKING,
                LateralControlReason.NOMINAL_TRACKING,
            ),
            (
                LongitudinalControlReason.SPEED_STALE,
                SafetyState.NOMINAL,
                TemporalLaneState.TRACKING,
                LateralControlReason.NOMINAL_TRACKING,
            ),
            (
                LongitudinalControlReason.TARGET_EXPIRED,
                SafetyState.DEGRADED,
                TemporalLaneState.TRACKING,
                LateralControlReason.DEGRADED_TRACKING,
            ),
            (
                LongitudinalControlReason.REVERSE_MOTION,
                SafetyState.DEGRADED,
                TemporalLaneState.COASTING,
                LateralControlReason.DEGRADED_COASTING,
            ),
            (
                LongitudinalControlReason.EXCESSIVE_LATERAL_MOTION,
                SafetyState.NOMINAL,
                TemporalLaneState.TRACKING,
                LateralControlReason.NOMINAL_TRACKING,
            ),
        ):
            with self.subTest(longitudinal=longitudinal_reason):
                pair = self.bound_pair(
                    lateral_reason,
                    longitudinal_reason,
                    safety=safety,
                    temporal=temporal,
                )
                _envelope, request = self.envelope_for(*pair)
                self.assertTrue(request.command_allowed)
                self.assertIs(
                    request.reason, CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP
                )

    MUTATIONS = (
        "none",
        "guidance stamp",
        "snapshot revision",
        "lateral mode",
        "evaluation time",
    )

    def mutated(self, lateral_record, longitudinal_record, mutation):
        """Break exactly one same-cut axis, leaving the rest coherent."""
        if mutation == "none":
            return lateral_record, longitudinal_record
        if mutation == "guidance stamp":
            return lateral_record, bypassed(
                longitudinal_record,
                guidance_stamp=SampleStamp(
                    frame=99, sim_time=24.75, monotonic=BASE - 0.05
                ),
            )
        if mutation == "snapshot revision":
            return lateral_record, bypassed(longitudinal_record, snapshot_revision=99)
        if mutation == "lateral mode":
            other = (
                LateralControlMode.DISABLED
                if longitudinal_record.lateral_mode is not LateralControlMode.DISABLED
                else LateralControlMode.NOMINAL
            )
            return lateral_record, bypassed(longitudinal_record, lateral_mode=other)
        return (
            bypassed(lateral_record, evaluation_monotonic=BASE - 5.0),
            longitudinal_record,
        )

    def test_the_whole_reason_cross_product_matches_the_independent_oracle(self):
        actionable = 0
        refused = 0
        violations = []
        for lateral_reason in LateralControlReason:
            for longitudinal_reason in LongitudinalControlReason:
                for safety in SafetyState:
                    for temporal in TemporalLaneState:
                        base_pair = self.bound_pair(
                            lateral_reason,
                            longitudinal_reason,
                            safety=safety,
                            temporal=temporal,
                        )
                        for mutation in self.MUTATIONS:
                            pair = self.mutated(*base_pair, mutation)
                            _envelope, request = self.envelope_for(*pair)
                            if not request.command_allowed:
                                refused += 1
                                continue
                            actionable += 1
                            # Anything actionable must be a pair the independent
                            # oracle says one real cut produces, and no mutated
                            # variant may ever be actionable at all.
                            reachable = (
                                mutation == "none"
                                and real_lateral_shape(
                                    lateral_reason, safety, temporal, True
                                )
                                and longitudinal_reason
                                in REAL_SAME_CUT_LONGITUDINAL[lateral_reason]
                            )
                            if not reachable:
                                violations.append(
                                    (
                                        lateral_reason.name,
                                        longitudinal_reason.name,
                                        safety.name,
                                        temporal.name,
                                        mutation,
                                    )
                                )
        self.assertEqual(violations, [])
        self.assertGreater(actionable, 0)
        self.assertGreater(refused, 20_000)

    def test_matching_identity_with_impossible_ancestry_is_still_refused(self):
        """Every field agrees; only the branch ancestry is impossible."""
        pair = self.bound_pair(
            LateralControlReason.IDENTITY_MISMATCH,
            LongitudinalControlReason.LATERAL_STOP,
        )
        lateral_record, longitudinal_record = pair
        self.assertEqual(lateral_record.source_stamp, longitudinal_record.guidance_stamp)
        self.assertEqual(
            lateral_record.snapshot_revision, longitudinal_record.snapshot_revision
        )
        self.assertIs(lateral_record.safety_state, longitudinal_record.safety_state)
        self.assertIs(lateral_record.temporal_state, longitudinal_record.temporal_state)
        self.assertIs(lateral_record.mode, longitudinal_record.lateral_mode)
        self.assertEqual(
            lateral_record.evaluation_monotonic,
            longitudinal_record.evaluation_monotonic,
        )
        _envelope, request = self.envelope_for(*pair)
        self.assertFalse(request.command_allowed)

    def test_a_mismatched_cut_identity_is_still_refused(self):
        lateral_record, longitudinal_record = self.bound_pair(
            LateralControlReason.INSUFFICIENT_GEOMETRY,
            LongitudinalControlReason.LATERAL_STOP,
            safety=SafetyState.DEGRADED,
        )
        moved = bypassed(
            longitudinal_record,
            guidance_stamp=SampleStamp(frame=99, sim_time=9.9, monotonic=BASE - 0.05),
        )
        _envelope, request = self.envelope_for(lateral_record, moved)
        self.assertNeutral(
            request, CommandEnvelopeReason.CROSS_RECORD_IDENTITY_MISMATCH
        )

    def test_no_bound_pair_may_carry_an_initializing_safety_state(self):
        """Mission 8 gives an INITIALIZING decision no stamp, revision, or
        temporal state, and Mission 10A refuses to bind without all three."""
        for longitudinal_reason in LongitudinalControlReason:
            for lateral_reason in LateralControlReason:
                with self.subTest(
                    lateral=lateral_reason, longitudinal=longitudinal_reason
                ):
                    pair = self.bound_pair(
                        lateral_reason,
                        longitudinal_reason,
                        safety=SafetyState.INITIALIZING,
                        temporal=TemporalLaneState.TRACKING,
                    )
                    _envelope, request = self.envelope_for(*pair)
                    self.assertFalse(request.command_allowed)

    def test_an_initializing_actionable_provenance_is_rejected(self):
        with self.assertRaises(ValueError):
            CommandEnvelopeRequest(
                mode=CommandEnvelopeMode.STOPPING,
                reason=CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP,
                command_allowed=True,
                stop_required=True,
                brake_request=0.20,
                brake_authority=BRAKE_AUTHORITY,
                controller_epoch=0,
                **stop_provenance(
                    safety_state=SafetyState.INITIALIZING,
                    lateral_mode=LateralControlMode.DISABLED,
                    lateral_reason=LateralControlReason.SAFETY_NOT_PERMITTED,
                    longitudinal_reason=LongitudinalControlReason.SAFETY_STOP,
                ),
            )

    def test_the_production_pair_table_states_the_same_relationships(self):
        import control.command as module

        self.assertEqual(
            set(module._SAME_CUT_LONGITUDINAL_REASONS),
            set(REAL_SAME_CUT_LONGITUDINAL),
        )
        for reason, allowed in REAL_SAME_CUT_LONGITUDINAL.items():
            with self.subTest(reason=reason):
                self.assertEqual(
                    set(module._SAME_CUT_LONGITUDINAL_REASONS[reason]), set(allowed)
                )


class ActionableProvenancePairTest(EnvelopeTestCase):
    """The public constructor enforces the same same-cut relationship."""

    def test_the_reported_public_stop_tuple_is_rejected(self):
        with self.assertRaises(ValueError):
            CommandEnvelopeRequest(
                mode=CommandEnvelopeMode.STOPPING,
                reason=CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP,
                command_allowed=True,
                stop_required=True,
                brake_request=0.20,
                brake_authority=BRAKE_AUTHORITY,
                controller_epoch=0,
                **stop_provenance(
                    longitudinal_reason=LongitudinalControlReason.LATERAL_STOP,
                    lateral_mode=LateralControlMode.DISABLED,
                    lateral_reason=LateralControlReason.IDENTITY_MISMATCH,
                ),
            )

    def test_every_public_stop_pair_matches_the_independent_oracle(self):
        for lateral_reason in LateralControlReason:
            for longitudinal_reason in LongitudinalControlReason:
                if _LONGITUDINAL_REASON_MODE[longitudinal_reason] is not (
                    LongitudinalControlMode.STOPPING
                ):
                    continue
                lateral_mode = _LATERAL_REASON_MODE[lateral_reason]
                permitted_lateral = lateral_mode in (
                    LateralControlMode.NOMINAL,
                    LateralControlMode.DEGRADED,
                )
                for safety in (SafetyState.NOMINAL, SafetyState.DEGRADED,
                               SafetyState.FAIL_SAFE):
                    for temporal in REAL_BOUND_TEMPORAL:
                        with self.subTest(
                            lateral=lateral_reason,
                            longitudinal=longitudinal_reason,
                            safety=safety,
                            temporal=temporal,
                        ):
                            try:
                                CommandEnvelopeRequest(
                                    mode=CommandEnvelopeMode.STOPPING,
                                    reason=(
                                        CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP
                                    ),
                                    command_allowed=True,
                                    stop_required=True,
                                    steering_request=0.0,
                                    brake_request=0.20,
                                    steering_authority=(
                                        LATERAL_AUTHORITY if permitted_lateral else 0.0
                                    ),
                                    brake_authority=BRAKE_AUTHORITY,
                                    controller_epoch=0,
                                    **stop_provenance(
                                        safety_state=safety,
                                        temporal_state=temporal,
                                        lateral_mode=lateral_mode,
                                        lateral_reason=lateral_reason,
                                        longitudinal_reason=longitudinal_reason,
                                    ),
                                )
                            except ValueError:
                                continue
                            self.assertTrue(
                                real_lateral_shape(
                                    lateral_reason, safety, temporal, True
                                )
                            )
                            self.assertIn(
                                longitudinal_reason,
                                REAL_SAME_CUT_LONGITUDINAL[lateral_reason],
                            )


# ==========================================================================
# R3.C  Positive actionable authorities
# ==========================================================================


class ActionableAuthorityTest(EnvelopeTestCase):
    def command(self, **overrides):
        base = dict(
            mode=CommandEnvelopeMode.NOMINAL,
            reason=CommandEnvelopeReason.NOMINAL_COMMAND,
            command_allowed=True,
            stop_required=False,
            steering_request=0.0,
            throttle_request=0.0,
            brake_request=0.0,
            steering_authority=LATERAL_AUTHORITY,
            throttle_authority=THROTTLE_AUTHORITY,
            brake_authority=BRAKE_AUTHORITY,
            controller_epoch=0,
            effective_target_speed_mps=2.0,
            speed_error_mps=1.0,
            speed_authority_mps=3.0,
        )
        base.update(command_provenance())
        base.update(overrides)
        return CommandEnvelopeRequest(**base)

    def stop(self, **overrides):
        base = dict(
            mode=CommandEnvelopeMode.STOPPING,
            reason=CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP,
            command_allowed=True,
            stop_required=True,
            steering_request=0.0,
            brake_request=0.20,
            steering_authority=LATERAL_AUTHORITY,
            throttle_authority=0.0,
            brake_authority=BRAKE_AUTHORITY,
            controller_epoch=0,
        )
        base.update(stop_provenance())
        base.update(overrides)
        return CommandEnvelopeRequest(**base)

    def test_a_nominal_command_requires_positive_authority_on_every_axis(self):
        self.command()  # the reachable shape still constructs
        for axis in ("steering_authority", "throttle_authority", "brake_authority"):
            with self.subTest(axis=axis):
                with self.assertRaises(ValueError):
                    self.command(**{axis: 0.0})

    def test_a_degraded_command_requires_positive_authority_on_every_axis(self):
        degraded = dict(
            mode=CommandEnvelopeMode.DEGRADED,
            reason=CommandEnvelopeReason.DEGRADED_COMMAND,
        )
        degraded.update(command_provenance(degraded=True))
        self.command(**degraded)
        for axis in ("steering_authority", "throttle_authority", "brake_authority"):
            with self.subTest(axis=axis):
                with self.assertRaises(ValueError):
                    self.command(**dict(degraded, **{axis: 0.0}))

    def test_the_reported_all_zero_actionable_record_is_rejected(self):
        with self.assertRaises(ValueError):
            self.command(
                steering_authority=0.0,
                throttle_authority=0.0,
                brake_authority=0.0,
            )

    def test_zero_axes_stay_legitimate_under_positive_authority(self):
        request = self.command(
            steering_request=0.0, throttle_request=0.0, brake_request=0.0
        )
        self.assertTrue(request.command_allowed)
        self.assertEqual(request.throttle_request, 0.0)
        self.assertGreater(request.throttle_authority, 0.0)

    def test_a_trusted_stop_requires_positive_brake_authority(self):
        self.stop()
        with self.assertRaises(ValueError):
            self.stop(brake_request=0.0, brake_authority=0.0)

    def test_a_trusted_stop_carries_no_throttle_authority(self):
        with self.assertRaises(ValueError):
            self.stop(throttle_authority=THROTTLE_AUTHORITY)

    def test_stopping_steering_authority_tracks_the_lateral_permission(self):
        with self.assertRaises(ValueError):
            self.stop(steering_authority=0.0)
        refused = self.stop(
            steering_authority=0.0,
            **stop_provenance(
                safety_state=SafetyState.DEGRADED,
                lateral_mode=LateralControlMode.DISABLED,
                lateral_reason=LateralControlReason.INSUFFICIENT_GEOMETRY,
                longitudinal_reason=LongitudinalControlReason.LATERAL_STOP,
            ),
        )
        self.assertEqual(refused.steering_authority, 0.0)
        with self.assertRaises(ValueError):
            self.stop(
                steering_authority=LATERAL_AUTHORITY,
                **stop_provenance(
                    safety_state=SafetyState.DEGRADED,
                    lateral_mode=LateralControlMode.DISABLED,
                    lateral_reason=LateralControlReason.INSUFFICIENT_GEOMETRY,
                    longitudinal_reason=LongitudinalControlReason.LATERAL_STOP,
                ),
            )

    def test_a_disabled_envelope_still_carries_no_authority(self):
        request = CommandEnvelopeRequest(
            mode=CommandEnvelopeMode.DISABLED,
            reason=CommandEnvelopeReason.UPSTREAM_REFUSAL,
            command_allowed=False,
            evaluation_monotonic=BASE,
            controller_epoch=0,
        )
        self.assertEqual(request.steering_authority, 0.0)
        self.assertEqual(request.throttle_authority, 0.0)
        self.assertEqual(request.brake_authority, 0.0)

    def test_every_real_envelope_output_still_reconstructs(self):
        rebuilt = 0
        for request in run_script(CommandEnvelope()):
            clone = CommandEnvelopeRequest(
                **{
                    field.name: getattr(request, field.name)
                    for field in fields(CommandEnvelopeRequest)
                }
            )
            self.assertEqual(clone, request)
            if request.command_allowed:
                self.assertGreater(request.brake_authority, 0.0)
                if request.mode is not CommandEnvelopeMode.STOPPING:
                    self.assertGreater(request.steering_authority, 0.0)
                    self.assertGreater(request.throttle_authority, 0.0)
            rebuilt += 1
        self.assertGreater(rebuilt, 0)


# ==========================================================================
# R3.D  Caller conversion versus internal fault boundary
# ==========================================================================


class InjectedFault(RuntimeError):
    """A fault injected into a Mission 10B internal helper."""


class FaultyMath:
    """A stand-in for the ``math`` module whose ``isfinite`` can fail."""

    def __init__(self, fail_after: int = 0, once: bool = False) -> None:
        self.calls = 0
        self.fail_after = fail_after
        self.once = once
        self.fired = False

    def __getattr__(self, name):
        return getattr(math, name)

    def isfinite(self, value):
        self.calls += 1
        if self.calls > self.fail_after and not (self.once and self.fired):
            self.fired = True
            raise InjectedFault("injected finiteness failure")
        return math.isfinite(value)


class BoundaryTestCase(EnvelopeTestCase):
    def warmed(self):
        envelope = CommandEnvelope(CONFIG)
        request = envelope.update(lateral(0), permitted(0), now=BASE)
        self.assertTrue(request.command_allowed)
        return envelope, request

    def patch(self, name, value):
        import control.command as module

        original = getattr(module, name)
        setattr(module, name, value)
        self.addCleanup(setattr, module, name, original)

    @contextlib.contextmanager
    def scoped(self, name, value):
        """Patch one module global for the duration of a ``with`` block.

        :meth:`patch` restores at *test* teardown, which would let one subTest's
        injected fault leak into the next one and mask a real answer.
        """
        import control.command as module

        original = getattr(module, name)
        setattr(module, name, value)
        try:
            yield
        finally:
            setattr(module, name, original)

    def assertInternalError(self, envelope, before, *, evaluated):
        request = envelope.latest_request
        self.assertIs(request.reason, CommandEnvelopeReason.INTERNAL_ERROR)
        self.assertIs(request.mode, CommandEnvelopeMode.DISABLED)
        self.assertFalse(request.command_allowed)
        self.assertFalse(request.stop_required)
        self.assertEqual(request.evaluation_monotonic, evaluated)
        for value in (
            request.steering_request,
            request.throttle_request,
            request.brake_request,
            request.steering_authority,
            request.throttle_authority,
            request.brake_authority,
        ):
            self.assertEqual(value, 0.0)
        self.assertFalse(request.shaping_applied)
        self.assertFalse(request.evidence_bound)
        self.assertIsNone(request.guidance_stamp)
        after = envelope.metrics
        self.assertEqual(after.internal_errors, before.internal_errors + 1)
        self.assertEqual(after.refusals, before.refusals + 1)
        self.assertEqual(after.evaluations, before.evaluations + 1)
        self.assertEqual(after.commands, before.commands)
        state = envelope._state
        self.assertEqual(state.shaping.throttle_anchor, 0.0)
        self.assertEqual(state.shaping.brake_anchor, 0.0)
        self.assertIsNone(state.shaping.shaping_monotonic)
        self.assertIs(state.shaping.transition, _Transition.IDLE)


class CallerConversionBoundaryTest(BoundaryTestCase):
    def test_a_hostile_caller_conversion_still_publishes_time_invalid(self):
        class Exploding:
            def __float__(self):
                raise RuntimeError("hostile __float__")

        numbers.Real.register(Exploding)
        envelope = CommandEnvelope(CONFIG)
        request = envelope.update(lateral(0), permitted(0), now=Exploding())
        self.assertNeutral(request, CommandEnvelopeReason.TIME_INVALID)
        self.assertIsNone(request.evaluation_monotonic)
        self.assertEqual(envelope.metrics.internal_errors, 0)

    def test_a_hostile_class_check_still_publishes_time_invalid(self):
        class HostileClass:
            @property
            def __class__(self):
                raise RuntimeError("hostile __class__")

        envelope = CommandEnvelope(CONFIG)
        request = envelope.update(lateral(0), permitted(0), now=HostileClass())
        self.assertNeutral(request, CommandEnvelopeReason.TIME_INVALID)
        self.assertEqual(envelope.metrics.internal_errors, 0)

    def test_a_bool_control_clock_is_still_rejected(self):
        envelope = CommandEnvelope(CONFIG)
        request = envelope.update(lateral(0), permitted(0), now=True)
        self.assertNeutral(request, CommandEnvelopeReason.TIME_INVALID)

    def test_a_non_finite_control_clock_is_a_caller_problem_not_a_defect(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                envelope = CommandEnvelope(CONFIG)
                request = envelope.update(lateral(0), permitted(0), now=value)
                self.assertNeutral(request, CommandEnvelopeReason.TIME_INVALID)
                self.assertIsNone(request.evaluation_monotonic)
                self.assertEqual(envelope.metrics.internal_errors, 0)

    def test_the_caller_object_is_converted_exactly_once(self):
        drift = DriftingReal(BASE, BASE + 1000.0)
        envelope = CommandEnvelope(CONFIG)
        envelope.update(lateral(0), permitted(0), now=drift)
        self.assertEqual(envelope.latest_request.evaluation_monotonic, BASE)

    def test_a_float_subclass_clock_is_normalized_to_an_exact_float(self):
        envelope = CommandEnvelope(CONFIG)
        request = envelope.update(lateral(0), permitted(0), now=SneakyFloat(BASE))
        self.assertIs(type(request.evaluation_monotonic), float)
        self.assertEqual(request.evaluation_monotonic, BASE)

    def test_negative_zero_is_folded_by_the_internal_stage(self):
        import control.command as module

        self.assertEqual(
            math.copysign(1.0, module._normalized_control_time(-0.0)), 1.0
        )

    def test_keyboard_interrupt_from_the_caller_clock_still_propagates(self):
        class Interrupting:
            def __float__(self):
                raise KeyboardInterrupt

        numbers.Real.register(Interrupting)
        envelope, warm = self.warmed()
        before = envelope.metrics
        with self.assertRaises(KeyboardInterrupt):
            envelope.update(lateral(1), permitted(1), now=Interrupting())
        self.assertIs(envelope.latest_request, warm)
        self.assertEqual(envelope.metrics, before)

    def test_system_exit_from_the_caller_clock_still_propagates(self):
        class Exiting:
            def __float__(self):
                raise SystemExit

        numbers.Real.register(Exiting)
        envelope = CommandEnvelope(CONFIG)
        with self.assertRaises(SystemExit):
            envelope.update(lateral(0), permitted(0), now=Exiting())


class InternalFaultBoundaryTest(BoundaryTestCase):
    def test_a_one_shot_finiteness_fault_is_an_internal_error_not_time_invalid(self):
        envelope, _warm = self.warmed()
        before = envelope.metrics
        self.patch("math", FaultyMath(fail_after=0, once=True))
        with self.assertRaises(InjectedFault):
            envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertInternalError(envelope, before, evaluated=None)

    def test_a_persistent_finiteness_fault_still_publishes_internal_error(self):
        envelope, _warm = self.warmed()
        before = envelope.metrics
        self.patch("math", FaultyMath(fail_after=0))
        with self.assertRaises(InjectedFault):
            envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertInternalError(envelope, before, evaluated=None)

    def test_a_fault_after_the_clock_publishes_the_trustworthy_time(self):
        envelope, _warm = self.warmed()
        before = envelope.metrics
        # Let the control-clock normalization through, then fail.
        self.patch("math", FaultyMath(fail_after=1))
        with self.assertRaises(InjectedFault):
            envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertInternalError(envelope, before, evaluated=BASE + STEP)

    def test_a_normalization_helper_fault_is_an_internal_error(self):
        envelope, _warm = self.warmed()
        before = envelope.metrics

        def boom(value):
            raise InjectedFault("injected normalization failure")

        self.patch("_normalized_control_time", boom)
        with self.assertRaises(InjectedFault):
            envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertInternalError(envelope, before, evaluated=None)

    def test_a_persistent_internal_stamp_fault_re_raises_from_lateral_validation(self):
        envelope, _warm = self.warmed()
        before = envelope.metrics

        def boom(*args, **kwargs):
            raise InjectedFault("persistent internal stamp failure")

        self.patch("_canonical_stamp", boom)
        with self.assertRaises(InjectedFault) as caught:
            envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertEqual(str(caught.exception), "persistent internal stamp failure")
        self.assertInternalError(envelope, before, evaluated=BASE + STEP)
        self.assertNotIn(id(caught.exception), retained_ids(envelope))

    def test_a_one_shot_internal_stamp_fault_is_not_malformed_evidence(self):
        import control.command as module

        envelope, _warm = self.warmed()
        before = envelope.metrics
        original = module._canonical_stamp
        calls = [0]

        def once(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 1:
                raise InjectedFault("one-shot internal stamp failure")
            return original(*args, **kwargs)

        self.patch("_canonical_stamp", once)
        with self.assertRaises(InjectedFault):
            envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertInternalError(envelope, before, evaluated=BASE + STEP)

    def test_an_internal_stamp_fault_re_raises_from_longitudinal_validation(self):
        import control.command as module

        envelope, _warm = self.warmed()
        before = envelope.metrics
        original = module._canonical_stamp
        calls = [0]

        def fail_second(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                raise InjectedFault("longitudinal stamp failure")
            return original(*args, **kwargs)

        self.patch("_canonical_stamp", fail_second)
        with self.assertRaises(InjectedFault):
            envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertInternalError(envelope, before, evaluated=BASE + STEP)

    def test_an_internal_shape_validation_fault_re_raises_as_internal_error(self):
        envelope, _warm = self.warmed()
        before = envelope.metrics

        def boom(*args, **kwargs):
            raise InjectedFault("internal shape validation failure")

        self.patch("_validate_detached_lateral", boom)
        with self.assertRaises(InjectedFault) as caught:
            envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertEqual(str(caught.exception), "internal shape validation failure")
        self.assertInternalError(envelope, before, evaluated=BASE + STEP)
        self.assertNotIn(id(caught.exception), retained_ids(envelope))

    def test_caller_numeric_conversion_failures_remain_malformed_evidence(self):
        for axis in ("lateral", "longitudinal"):
            with self.subTest(axis=axis):
                envelope, _warm = self.warmed()
                before = envelope.metrics
                if axis == "lateral":
                    lat = bypassed(
                        lateral(1),
                        steering=ExplodingReal(InjectedFault("caller conversion")),
                    )
                    lon = permitted(1)
                    expected = CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE
                else:
                    lat = lateral(1)
                    lon = bypassed(
                        permitted(1),
                        throttle_request=ExplodingReal(
                            InjectedFault("caller conversion")
                        ),
                    )
                    expected = CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE
                request = envelope.update(lat, lon, now=BASE + STEP)
                self.assertNeutral(request, expected)
                self.assertEqual(envelope.metrics.internal_errors, before.internal_errors)

    def test_caller_property_failures_remain_inside_the_acquisition_boundary(self):
        descriptor = LateralControlRequest.steering
        envelope, _warm = self.warmed()
        before = envelope.metrics
        record = lateral(1)

        def boom(_self):
            raise InjectedFault("caller property failure")

        LateralControlRequest.steering = property(boom)
        try:
            request = envelope.update(record, permitted(1), now=BASE + STEP)
        finally:
            LateralControlRequest.steering = descriptor
        self.assertNeutral(
            request, CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE
        )
        self.assertEqual(envelope.metrics.internal_errors, before.internal_errors)

    def test_recovery_after_an_internal_validation_fault_is_clean(self):
        import control.command as module

        envelope, _warm = self.warmed()
        original = module._canonical_stamp
        module._canonical_stamp = lambda *args, **kwargs: (_ for _ in ()).throw(
            InjectedFault("recoverable stamp failure")
        )
        try:
            with self.assertRaises(InjectedFault):
                envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        finally:
            module._canonical_stamp = original
        recovered = envelope.update(
            lateral(1), permitted(1), now=BASE + STEP
        )
        self.assertTrue(recovered.command_allowed)
        self.assertEqual(recovered.throttle_request, 0.0)

    def test_a_request_validation_helper_fault_is_an_internal_error(self):
        envelope, _warm = self.warmed()
        before = envelope.metrics

        def boom(*args, **kwargs):
            raise InjectedFault("injected validation failure")

        self.patch("_canonicalize_lateral_request", boom)
        with self.assertRaises(InjectedFault):
            envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertInternalError(envelope, before, evaluated=BASE + STEP)

    def test_a_metrics_helper_fault_is_an_internal_error(self):
        envelope, _warm = self.warmed()
        before = envelope.metrics

        def boom(*args, **kwargs):
            raise InjectedFault("injected metrics failure")

        self.patch("_accepted_metrics", boom)
        with self.assertRaises(InjectedFault):
            envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertInternalError(envelope, before, evaluated=BASE + STEP)

    def test_a_state_constructor_fault_is_an_internal_error(self):
        envelope, _warm = self.warmed()
        before = envelope.metrics
        import control.command as module

        real_state = module._ControllerState
        calls = [0]

        def boom(*args, **kwargs):
            calls[0] += 1
            raise InjectedFault("injected state construction failure")

        self.patch("_ControllerState", boom)
        with self.assertRaises(InjectedFault):
            envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        # The emergency path binds the real state type at definition time, so it
        # still built one even though the module global was replaced.
        self.assertIs(module._ControllerState, boom)
        self.assertGreater(calls[0], 0)
        module._ControllerState = real_state
        self.assertInternalError(envelope, before, evaluated=BASE + STEP)

    def test_an_accepted_path_constructor_fault_is_an_internal_error(self):
        envelope, _warm = self.warmed()
        before = envelope.metrics

        def boom(*args, **kwargs):
            raise InjectedFault("injected request construction failure")

        self.patch("CommandEnvelopeRequest", boom)
        with self.assertRaises(InjectedFault):
            envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertInternalError(envelope, before, evaluated=BASE + STEP)

    def test_a_replace_helper_fault_cannot_disable_the_emergency_publication(self):
        envelope, _warm = self.warmed()
        before = envelope.metrics

        def boom(*args, **kwargs):
            raise InjectedFault("injected replace failure")

        self.patch("replace", boom)
        with self.assertRaises(InjectedFault):
            envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertInternalError(envelope, before, evaluated=BASE + STEP)

    def test_no_stale_actionable_output_survives_any_injected_fault(self):
        """Whatever a fault does, the previous *command* stops being the answer.

        Some of these land inside the documented evidence-reading boundary,
        where any failure to read a caller-supplied record is by contract a
        neutral refusal rather than an internal error.  Either outcome is
        fail-closed; what must never happen is the earlier actionable command
        remaining visible.
        """

        def thrower(*args, **kwargs):
            raise InjectedFault("injected helper failure")

        faults = (
            ("math", FaultyMath(fail_after=0)),
            ("_finite_real", thrower),
            ("_bounded_real", thrower),
            ("_builtin_bool", thrower),
            ("_canonical_stamp", thrower),
            ("_normalized_control_time", thrower),
            ("_provenance", thrower),
            ("_fingerprint", thrower),
            ("_accepted_metrics", thrower),
            ("_same_cut_conflict", thrower),
            ("_lateral_producer_conflict", thrower),
        )
        raised = 0
        for name, fault in faults:
            with self.subTest(fault=name):
                envelope, warm = self.warmed()
                self.assertTrue(warm.command_allowed)
                import control.command as module

                original = getattr(module, name)
                setattr(module, name, fault)
                try:
                    try:
                        envelope.update(lateral(1), permitted(1), now=BASE + STEP)
                    except InjectedFault:
                        raised += 1
                finally:
                    setattr(module, name, original)
                visible = envelope.latest_request
                self.assertFalse(visible.command_allowed)
                self.assertIs(visible.mode, CommandEnvelopeMode.DISABLED)
                self.assertIsNot(visible, warm)
                self.assertEqual(visible.brake_request, 0.0)
                self.assertEqual(visible.throttle_request, 0.0)
                self.assertEqual(envelope._state.shaping.brake_anchor, 0.0)
        self.assertGreater(raised, 0)

    def test_a_persistent_fault_then_a_valid_pair_recovers_safely(self):
        envelope, _warm = self.warmed()
        faulty = FaultyMath(fail_after=0)
        self.patch("math", faulty)
        with self.assertRaises(InjectedFault):
            envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        import control.command as module

        module.math = math
        recovered = envelope.update(lateral(2), permitted(2), now=BASE + 2 * STEP)
        self.assertTrue(recovered.command_allowed)
        self.assertEqual(recovered.brake_request, 0.0)

    def test_alternating_caller_and_internal_failures_stay_fail_closed(self):
        class Exploding:
            def __float__(self):
                raise RuntimeError("hostile __float__")

        numbers.Real.register(Exploding)
        envelope, _warm = self.warmed()
        for index in range(1, 6):
            caller = envelope.update(
                lateral(index), permitted(index), now=Exploding()
            )
            self.assertNeutral(caller, CommandEnvelopeReason.TIME_INVALID)
            import control.command as module

            module.math = FaultyMath(fail_after=0)
            try:
                with self.assertRaises(InjectedFault):
                    envelope.update(
                        lateral(index), permitted(index), now=BASE + index * STEP
                    )
            finally:
                module.math = math
            self.assertIs(
                envelope.latest_request.reason, CommandEnvelopeReason.INTERNAL_ERROR
            )
            self.assertFalse(envelope.latest_request.command_allowed)


# ==========================================================================
# R5  Expected caller-contract failure vs unexpected internal failure
#
# A raw ``TypeError`` or ``ValueError`` means two completely different things
# depending on where it arose, so the oracle below is deliberately independent
# of the production classification.  Each case states in the test itself whether
# the fault was injected into *caller-owned* machinery or into one of Mission
# 10B's *own* internal steps, and then asserts the outcome that answer requires.
# No production predicate, private marker class, or exception type is consulted
# to decide what should happen -- only to inject the fault.
# ==========================================================================


def raising(exception: BaseException):
    """A caller property that raises ``exception`` when it is read."""

    def read(_self):
        raise exception

    return property(read)


class EvidenceClassificationBoundaryTest(BoundaryTestCase):
    """Where a raw exception arose decides how Mission 10B reports it."""

    # ------------------------------------------------------------- oracles

    def assertCallerEvidence(self, envelope, before, baseline, warm, request, reason):
        """Everything an *expected caller-contract* failure must publish."""
        self.assertNeutral(request, reason)
        after = envelope.metrics
        self.assertEqual(after.internal_errors, before.internal_errors)
        self.assertEqual(after.refusals, before.refusals + 1)
        self.assertEqual(after.evaluations, before.evaluations + 1)
        self.assertEqual(after.commands, before.commands)
        self.assertEqual(envelope._state.evidence, baseline)
        shaping = envelope._state.shaping
        self.assertEqual(shaping.throttle_anchor, 0.0)
        self.assertEqual(shaping.brake_anchor, 0.0)
        self.assertIsNone(shaping.shaping_monotonic)
        self.assertIs(shaping.transition, _Transition.IDLE)
        self.assertIsNot(request, warm)

    def assertInternalFault(self, envelope, before, baseline, warm, caught):
        """Everything an *unexpected internal* failure must publish."""
        self.assertInternalError(envelope, before, evaluated=BASE + STEP)
        visible = envelope.latest_request
        # The malformed-evidence answer is exactly what must *not* be given.
        # Mission 10B keeps no separate malformed counter, so the observable
        # statement is the published reason plus an unmoved command count.
        self.assertNotIn(
            visible.reason,
            (
                CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE,
                CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE,
            ),
        )
        self.assertIsNot(visible, warm)
        self.assertEqual(envelope._state.evidence, baseline)
        self.assertNotIn(id(caught), retained_ids(envelope))
        for value in retained_objects(envelope):
            self.assertNotIsInstance(value, BaseException)
            self.assertFalse(hasattr(value, "tb_frame"))

    # ------------------------------------------------------------- drivers

    def run_caller_property(self, owner, attribute, exception, reason):
        """The caller's own property raises while Mission 10B reads it."""
        envelope, warm = self.warmed()
        before = envelope.metrics
        baseline = envelope._state.evidence
        # Both records must exist before the accepted property is replaced.
        lat, lon = lateral(1), permitted(1)
        descriptor = getattr(owner, attribute)
        setattr(owner, attribute, raising(exception))
        try:
            request = envelope.update(lat, lon, now=BASE + STEP)
        finally:
            setattr(owner, attribute, descriptor)
        self.assertCallerEvidence(envelope, before, baseline, warm, request, reason)
        recovered = envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertTrue(recovered.command_allowed)

    def run_caller_conversion(self, lat, lon, exception, reason):
        """The caller's single numeric conversion raises."""
        envelope, warm = self.warmed()
        before = envelope.metrics
        baseline = envelope._state.evidence
        request = envelope.update(lat, lon, now=BASE + STEP)
        self.assertCallerEvidence(envelope, before, baseline, warm, request, reason)
        recovered = envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertTrue(recovered.command_allowed)

    def run_internal_fault(self, name, exception, *, fail_after=0, once=False):
        """One of Mission 10B's own internal steps raises unexpectedly.

        ``fail_after`` skips that many leading calls, which is how the
        *longitudinal* use of a helper the lateral axis reaches first is
        targeted.  ``once`` fires a single time and then behaves normally, so
        recovery can be proven with the faulty helper still installed.
        """
        import control.command as module

        envelope, warm = self.warmed()
        before = envelope.metrics
        baseline = envelope._state.evidence
        original = getattr(module, name)
        counters = {"calls": 0, "fired": False}

        def boom(*args, **kwargs):
            counters["calls"] += 1
            if counters["calls"] > fail_after and not (once and counters["fired"]):
                counters["fired"] = True
                raise exception
            return original(*args, **kwargs)

        setattr(module, name, boom)
        try:
            with self.assertRaises(type(exception)) as caught:
                envelope.update(lateral(1), permitted(1), now=BASE + STEP)
            # The identical original exception object, not a translation of it.
            self.assertIs(caught.exception, exception)
            self.assertInternalFault(
                envelope, before, baseline, warm, caught.exception
            )
            if once:
                # A one-shot fault must not poison the next update even though
                # the faulty helper is still installed.
                recovered = envelope.update(
                    lateral(1), permitted(1), now=BASE + STEP
                )
                self.assertTrue(recovered.command_allowed)
        finally:
            setattr(module, name, original)
        if not once:
            recovered = envelope.update(lateral(1), permitted(1), now=BASE + STEP)
            self.assertTrue(recovered.command_allowed)

    # --------------------------------------------------- caller-originated

    def test_a_lateral_property_raising_is_malformed_lateral_evidence(self):
        for exception in (TypeError("caller type"), ValueError("caller value")):
            with self.subTest(exception=type(exception).__name__):
                self.run_caller_property(
                    LateralControlRequest,
                    "steering",
                    exception,
                    CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE,
                )

    def test_a_longitudinal_property_raising_is_malformed_longitudinal_evidence(self):
        for exception in (TypeError("caller type"), ValueError("caller value")):
            with self.subTest(exception=type(exception).__name__):
                self.run_caller_property(
                    LongitudinalControlRequest,
                    "throttle_request",
                    exception,
                    CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE,
                )

    def test_a_lateral_conversion_raising_is_malformed_lateral_evidence(self):
        for exception in (TypeError("caller type"), ValueError("caller value")):
            with self.subTest(exception=type(exception).__name__):
                self.run_caller_conversion(
                    bypassed(lateral(1), steering=ExplodingReal(exception)),
                    permitted(1),
                    exception,
                    CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE,
                )

    def test_a_longitudinal_conversion_raising_is_malformed_longitudinal_evidence(self):
        for exception in (TypeError("caller type"), ValueError("caller value")):
            with self.subTest(exception=type(exception).__name__):
                self.run_caller_conversion(
                    lateral(1),
                    bypassed(
                        permitted(1), throttle_request=ExplodingReal(exception)
                    ),
                    exception,
                    CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE,
                )

    def test_a_detached_caller_value_that_breaks_its_contract_stays_malformed(self):
        """The contract clauses proven after acquisition are still caller-owned."""
        cases = (
            (bypassed(lateral(1), steering=float("nan")), permitted(1),
             CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE),
            (bypassed(lateral(1), clamped=1), permitted(1),
             CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE),
            (bypassed(lateral(1), snapshot_revision=-1), permitted(1),
             CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE),
            # Only the accepted Mission 9 constructor catches this one.
            (bypassed(lateral(1), steering=0.99), permitted(1),
             CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE),
            (lateral(1), bypassed(permitted(1), brake_request=float("inf")),
             CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE),
            (lateral(1), bypassed(permitted(1), brake_clamped=0),
             CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE),
            (lateral(1), bypassed(permitted(1), target_epoch=-1),
             CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE),
            # Only the accepted Mission 10A constructor catches this one.
            (lateral(1),
             bypassed(permitted(1), throttle_request=0.20, brake_request=0.20),
             CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE),
        )
        for lat, lon, reason in cases:
            with self.subTest(reason=reason.name):
                envelope, warm = self.warmed()
                before = envelope.metrics
                baseline = envelope._state.evidence
                request = envelope.update(lat, lon, now=BASE + STEP)
                self.assertCallerEvidence(
                    envelope, before, baseline, warm, request, reason
                )

    # ------------------------------------------------------ internal, both

    def test_every_internal_lateral_boundary_reports_an_internal_error(self):
        for name, fail_after in (
            ("_canonical_stamp", 0),
            ("_validate_detached_lateral", 0),
            ("_LATERAL_RECONSTRUCTOR", 0),
        ):
            for factory in (TypeError, ValueError):
                for once in (False, True):
                    with self.subTest(step=name, raises=factory.__name__, once=once):
                        self.run_internal_fault(
                            name,
                            factory(f"internal {name} defect"),
                            fail_after=fail_after,
                            once=once,
                        )

    def test_every_internal_longitudinal_boundary_reports_an_internal_error(self):
        # The lateral axis canonicalizes its stamp first, so the longitudinal
        # use of the shared helper is the second call.
        for name, fail_after in (
            ("_canonical_stamp", 1),
            ("_validate_detached_longitudinal", 0),
            ("_LONGITUDINAL_RECONSTRUCTOR", 0),
        ):
            for factory in (TypeError, ValueError):
                for once in (False, True):
                    with self.subTest(step=name, raises=factory.__name__, once=once):
                        self.run_internal_fault(
                            name,
                            factory(f"internal {name} defect"),
                            fail_after=fail_after,
                            once=once,
                        )

    def test_the_producer_shape_proofs_report_an_internal_error(self):
        for name in (
            "_prove_lateral_producer_shape",
            "_prove_longitudinal_producer_shape",
        ):
            for factory in (TypeError, ValueError):
                with self.subTest(step=name, raises=factory.__name__):
                    self.run_internal_fault(name, factory(f"internal {name} defect"))

    def test_the_detached_evidence_validators_report_an_internal_error(self):
        """A defect *inside* a contract validator is not the caller's fault."""
        for name in ("_evidence_real", "_evidence_bool", "_evidence_index"):
            for factory in (TypeError, ValueError):
                with self.subTest(step=name, raises=factory.__name__):
                    self.run_internal_fault(name, factory(f"internal {name} defect"))

    def test_an_internal_fault_does_not_advance_the_accepted_order(self):
        """The internal-error state keeps the accepted evidence baseline."""
        envelope, warm = self.warmed()
        accepted_before = envelope._state.accepted
        fingerprint_before = envelope._state.evidence.fingerprint
        self.patch("_LATERAL_RECONSTRUCTOR", self.thrower(TypeError("defect")))
        with self.assertRaises(TypeError):
            envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertIs(envelope._state.accepted, accepted_before)
        self.assertEqual(envelope._state.evidence.fingerprint, fingerprint_before)
        self.assertIsNot(envelope.latest_request, warm)

    @staticmethod
    def thrower(exception):
        def boom(*args, **kwargs):
            raise exception

        return boom

    # ------------------------------------------------------------ contrast

    def test_the_two_classifications_never_answer_the_same_way(self):
        """One update, one fault, one honest answer -- and they differ."""
        caller = CommandEnvelope(CONFIG)
        caller.update(lateral(0), permitted(0), now=BASE)
        caller_request = caller.update(
            bypassed(lateral(1), steering=ExplodingReal(TypeError("caller"))),
            permitted(1),
            now=BASE + STEP,
        )

        internal, _warm = self.warmed()
        before = internal.metrics
        self.patch("_validate_detached_lateral", self.thrower(TypeError("internal")))
        with self.assertRaises(TypeError):
            internal.update(lateral(1), permitted(1), now=BASE + STEP)

        self.assertIs(
            caller_request.reason,
            CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE,
        )
        self.assertIs(
            internal.latest_request.reason, CommandEnvelopeReason.INTERNAL_ERROR
        )
        self.assertEqual(caller.metrics.internal_errors, 0)
        self.assertEqual(internal.metrics.internal_errors, before.internal_errors + 1)
        # Both are fail-closed and neutral; only the explanation differs.
        for request in (caller_request, internal.latest_request):
            self.assertIs(request.mode, CommandEnvelopeMode.DISABLED)
            self.assertFalse(request.command_allowed)
            self.assertEqual(request.steering_authority, 0.0)
            self.assertEqual(request.throttle_authority, 0.0)
            self.assertEqual(request.brake_authority, 0.0)


class EmergencyPublicationIndependenceTest(EnvelopeTestCase):
    def test_the_emergency_record_matches_the_ordinary_constructor_field_by_field(self):
        import control.command as module

        for evaluated in (None, BASE, -0.0, BASE + STEP):
            for epoch in (0, 1, 7):
                with self.subTest(evaluated=evaluated, epoch=epoch):
                    ordinary = CommandEnvelopeRequest(
                        mode=CommandEnvelopeMode.DISABLED,
                        reason=CommandEnvelopeReason.INTERNAL_ERROR,
                        command_allowed=False,
                        evaluation_monotonic=evaluated,
                        controller_epoch=epoch,
                    )
                    emergency = module._emergency_request(evaluated, epoch)
                    self.assertIs(type(emergency), CommandEnvelopeRequest)
                    for field in fields(CommandEnvelopeRequest):
                        self.assertEqual(
                            getattr(emergency, field.name),
                            getattr(ordinary, field.name),
                            field.name,
                        )
                    self.assertEqual(emergency, ordinary)

    def test_the_emergency_record_writes_every_declared_field(self):
        import control.command as module

        self.assertEqual(
            set(name for name, _value in module._INTERNAL_ERROR_FIELDS),
            {field.name for field in fields(CommandEnvelopeRequest)},
        )
        self.assertEqual(
            module._REQUEST_FIELD_NAMES,
            tuple(field.name for field in fields(CommandEnvelopeRequest)),
        )

    def test_the_emergency_record_is_immutable(self):
        import control.command as module

        emergency = module._emergency_request(BASE, 0)
        with self.assertRaises(FrozenInstanceError):
            emergency.brake_request = 1.0

    def test_the_emergency_path_uses_no_validation_helper(self):
        """Every mutable helper the ordinary path uses is replaced at once."""
        import control.command as module

        names = (
            "_finite_real",
            "_bounded_real",
            "_builtin_bool",
            "_canonical_stamp",
            "_nonnegative_builtin_int",
            "math",
            "replace",
            "_REASON_MODE",
        )
        originals = {name: getattr(module, name) for name in names}

        def boom(*args, **kwargs):
            raise InjectedFault("no helper may be consulted")

        # A previous state built *before* the helpers are broken, so the only
        # construction happening under the fault is the emergency one.
        previous = CommandEnvelope(CONFIG)._state
        try:
            for name in names:
                setattr(module, name, boom)
            emergency = module._emergency_request(BASE, 3)
            state = module._internal_error_state(previous, BASE)
        finally:
            for name, value in originals.items():
                setattr(module, name, value)
        self.assertIs(emergency.reason, CommandEnvelopeReason.INTERNAL_ERROR)
        self.assertEqual(emergency.controller_epoch, 3)
        self.assertIs(state.request.reason, CommandEnvelopeReason.INTERNAL_ERROR)
        self.assertEqual(state.metrics.internal_errors, 1)
        self.assertEqual(state.metrics.refusals, 1)

    def test_the_emergency_state_preserves_evidence_and_clears_shaping(self):
        envelope = CommandEnvelope(CONFIG)
        envelope.update(lateral(0), permitted(0, throttle=0.0, brake=0.30), now=BASE)
        import control.command as module

        before = envelope._state
        state = module._internal_error_state(before, BASE + STEP)
        self.assertEqual(state.evidence, before.evidence)
        self.assertIs(state.accepted, before.accepted)
        self.assertEqual(state.shaping.brake_anchor, 0.0)
        self.assertIsNone(state.shaping.shaping_monotonic)
        self.assertIs(state.shaping.transition, _Transition.IDLE)

    def test_the_emergency_record_retains_no_exception_or_traceback(self):
        envelope = CommandEnvelope(CONFIG)
        envelope.update(lateral(0), permitted(0), now=BASE)
        import control.command as module

        original = module.math
        module.math = FaultyMath(fail_after=0)
        try:
            with self.assertRaises(InjectedFault):
                envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        finally:
            module.math = original
        for value in retained_objects(envelope):
            self.assertNotIsInstance(value, BaseException)
            self.assertFalse(hasattr(value, "tb_frame"))


# ==========================================================================
# R3.E  Foreign top-level type: no publication whatsoever
# ==========================================================================


class ForeignTopLevelTypeTest(EnvelopeTestCase):
    """A foreign top-level argument is programmer misuse: it raises and the
    entire previous controller state is left untouched."""

    FOREIGN = (
        ("none", None),
        ("object", object()),
        ("mapping", {"mode": None, "reason": None}),
        ("string", "lateral"),
        ("tuple", (1, 2, 3)),
        ("list", [1, 2, 3]),
        ("class object", LateralControlRequest),
        ("partial proxy", type("Proxy", (), {"mode": None, "reason": None})()),
    )

    def warmed(self):
        envelope = CommandEnvelope(CONFIG)
        warm = envelope.update(lateral(0), permitted(0), now=BASE)
        self.assertTrue(warm.command_allowed)
        return envelope, warm

    def assertUntouched(self, envelope, warm, state, metrics):
        self.assertIs(envelope.latest_request, warm)
        self.assertIs(envelope._state, state)
        self.assertEqual(envelope.metrics, metrics)
        self.assertTrue(envelope.latest_request.command_allowed)

    def test_accepted_upstream_producers_publish_exact_runtime_types(self):
        self.assertIs(type(lateral()), LateralControlRequest)
        self.assertIs(type(permitted()), LongitudinalControlRequest)

    def test_spoofed_class_objects_cannot_cross_the_exact_type_boundary(self):
        class SpoofedLateral:
            @property
            def __class__(self):
                return LateralControlRequest

        class SpoofedLongitudinal:
            @property
            def __class__(self):
                return LongitudinalControlRequest

        envelope, warm = self.warmed()
        state = envelope._state
        metrics = envelope.metrics
        with self.assertRaises(TypeError):
            envelope.update(SpoofedLateral(), permitted(1), now=BASE + STEP)
        with self.assertRaises(TypeError):
            envelope.update(lateral(1), SpoofedLongitudinal(), now=BASE + STEP)
        self.assertUntouched(envelope, warm, state, metrics)

    def test_hostile_getattribute_is_never_consulted_by_the_type_boundary(self):
        class Hostile:
            def __getattribute__(self, name):
                raise AssertionError(f"caller attribute was consulted: {name}")

        envelope, warm = self.warmed()
        state = envelope._state
        metrics = envelope.metrics
        with self.assertRaises(TypeError):
            envelope.update(Hostile(), permitted(1), now=BASE + STEP)
        with self.assertRaises(TypeError):
            envelope.update(lateral(1), Hostile(), now=BASE + STEP)
        self.assertUntouched(envelope, warm, state, metrics)

    def test_upstream_subclasses_are_outside_the_exact_public_contract(self):
        envelope, warm = self.warmed()
        state = envelope._state
        metrics = envelope.metrics
        with self.assertRaises(TypeError):
            envelope.update(unvalidated(lateral(1)), permitted(1), now=BASE + STEP)
        with self.assertRaises(TypeError):
            envelope.update(lateral(1), unvalidated(permitted(1)), now=BASE + STEP)
        self.assertUntouched(envelope, warm, state, metrics)

    def test_a_foreign_lateral_argument_publishes_nothing_from_a_warm_envelope(self):
        for label, value in self.FOREIGN:
            with self.subTest(argument=label):
                envelope, warm = self.warmed()
                state = envelope._state
                metrics = envelope.metrics
                with self.assertRaises(TypeError):
                    envelope.update(value, permitted(1), now=BASE + STEP)
                self.assertUntouched(envelope, warm, state, metrics)

    def test_a_foreign_longitudinal_argument_publishes_nothing(self):
        for label, value in self.FOREIGN:
            with self.subTest(argument=label):
                envelope, warm = self.warmed()
                state = envelope._state
                metrics = envelope.metrics
                with self.assertRaises(TypeError):
                    envelope.update(lateral(1), value, now=BASE + STEP)
                self.assertUntouched(envelope, warm, state, metrics)

    def test_swapped_arguments_publish_nothing(self):
        envelope, warm = self.warmed()
        state = envelope._state
        metrics = envelope.metrics
        with self.assertRaises(TypeError):
            envelope.update(permitted(1), lateral(1), now=BASE + STEP)
        self.assertUntouched(envelope, warm, state, metrics)

    def test_the_wrong_exact_upstream_type_publishes_nothing(self):
        envelope, warm = self.warmed()
        state = envelope._state
        metrics = envelope.metrics
        with self.assertRaises(TypeError):
            envelope.update(lateral(1), lateral(1), now=BASE + STEP)
        with self.assertRaises(TypeError):
            envelope.update(permitted(1), permitted(1), now=BASE + STEP)
        self.assertUntouched(envelope, warm, state, metrics)

    def test_a_fresh_envelope_also_publishes_nothing(self):
        envelope = CommandEnvelope(CONFIG)
        state = envelope._state
        metrics = envelope.metrics
        fresh = envelope.latest_request
        self.assertIs(fresh.reason, CommandEnvelopeReason.NOT_EVALUATED)
        for label, value in self.FOREIGN:
            with self.subTest(argument=label):
                with self.assertRaises(TypeError):
                    envelope.update(value, permitted(0), now=BASE)
        self.assertIs(envelope.latest_request, fresh)
        self.assertIs(envelope._state, state)
        self.assertEqual(envelope.metrics, metrics)

    def test_no_foreign_caller_object_is_retained(self):
        envelope, _warm = self.warmed()
        sentinel = type("Sentinel", (), {})()
        marker = weakref.ref(sentinel)
        with self.assertRaises(TypeError):
            envelope.update(sentinel, permitted(1), now=BASE + STEP)
        self.assertNotIn(id(sentinel), retained_ids(envelope))
        del sentinel
        gc.collect()
        self.assertIsNone(marker())

    def test_the_shaping_epoch_survives_the_raised_type_error(self):
        """Later valid traffic continues against the *unchanged* shaping epoch."""
        clean = CommandEnvelope(CONFIG)
        clean.update(lateral(0), permitted(0), now=BASE)
        expected = clean.update(lateral(1), permitted(1), now=BASE + STEP)

        envelope, _warm = self.warmed()
        with self.assertRaises(TypeError):
            envelope.update(None, permitted(1), now=BASE + STEP)
        with self.assertRaises(TypeError):
            envelope.update(lateral(1), object(), now=BASE + STEP)
        actual = envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertEqual(actual, expected)
        self.assertEqual(envelope.metrics, clean.metrics)


# ==========================================================================
# R6  The detached contract boundary
#
# Mission 10B proves both accepted upstream contracts *itself*, over detached
# built-ins, and only then reconstructs each record once.  Nothing below trusts
# that arrangement: the oracle for "is this evidence acceptable?" is always the
# pristine accepted constructor, and the oracle for "was this failure the
# caller's fault?" is always where the fault was injected -- never a Mission 10B
# predicate, private marker class, or exception type.
# ==========================================================================


class Foreign:
    """A type no accepted upstream field may ever hold."""


NAN = float("nan")
INF = float("inf")
GUIDANCE = SampleStamp(frame=5, sim_time=1.25, monotonic=BASE - 0.05)
SPEED_STAMP = SampleStamp(frame=5, sim_time=1.25, monotonic=BASE - 0.02)

# The target identity the accepted builders publish, and the two values exactly
# one ULP outside its issue/expiry order.  Written with ``nextafter`` because a
# hand-typed decimal at this magnitude rounds back onto the boundary itself and
# would silently stop testing the clause.
TARGET_ISSUED = BASE - 0.5
TARGET_EXPIRY = BASE + 50.0
ISSUE_AFTER_EXPIRY = math.nextafter(TARGET_EXPIRY, INF)
EXPIRY_BEFORE_ISSUE = math.nextafter(TARGET_ISSUED, -INF)


def mapping_of(record):
    """Every declared field of one accepted record, as a plain mapping."""
    return {field.name: getattr(record, field.name) for field in fields(record)}


# The upstream helpers each accepted ``__post_init__`` actually depends on.
# Discovered from the accepted sources rather than assumed, and pinned by
# ``test_the_upstream_dependency_inventory_is_complete`` below.
LATERAL_DEPENDENCIES = (
    "_finite_real",
    "_bounded_real",
    "_builtin_bool",
    "_canonical_stamp",
    "_nonnegative_builtin_int",
)
LONGITUDINAL_DEPENDENCIES = LATERAL_DEPENDENCIES + ("_nonnegative_real",)


def upstream_call_counts():
    """Install counters on every accepted upstream constructor dependency."""
    import control.lateral as lateral_module
    import control.longitudinal as longitudinal_module

    counts = {}
    restore = []
    for module, names in (
        (lateral_module, LATERAL_DEPENDENCIES),
        (longitudinal_module, LONGITUDINAL_DEPENDENCIES),
    ):
        for name in names:
            key = f"{module.__name__}.{name}"
            counts[key] = 0
            original = getattr(module, name)
            restore.append((module, name, original))

            def counting(*args, _key=key, _original=original, **kwargs):
                counts[_key] += 1
                return _original(*args, **kwargs)

            setattr(module, name, counting)
    return counts, restore


# ------------------------------------------------------- differential bases


def lateral_bases():
    """Accepted Mission 9 shapes spanning every branch of its own contract."""
    return {
        "permitted-nominal": mapping_of(lateral(0)),
        "permitted-degraded": mapping_of(
            lateral(
                0,
                mode=LateralControlMode.DEGRADED,
                reason=LateralControlReason.DEGRADED_TRACKING,
                safety=SafetyState.DEGRADED,
                steering=-0.20,
                authority=0.25,
                clamped=True,
                rate_limited=True,
            )
        ),
        "permitted-coasting": mapping_of(
            lateral(
                0,
                mode=LateralControlMode.DEGRADED,
                reason=LateralControlReason.DEGRADED_COASTING,
                safety=SafetyState.DEGRADED,
                temporal=TemporalLaneState.COASTING,
                steering=0.0,
                authority=0.25,
            )
        ),
        "refused-bound": mapping_of(
            refused_lateral(0, temporal=TemporalLaneState.LOST)
        ),
        "refused-safety": mapping_of(
            refused_lateral(
                0,
                reason=LateralControlReason.SAFETY_NOT_PERMITTED,
                temporal=TemporalLaneState.LOST,
            )
        ),
        "refused-source": mapping_of(
            refused_lateral(
                0,
                reason=LateralControlReason.SOURCE_REJECTED,
                stamp=None,
                revision=None,
                temporal=None,
            )
        ),
        "refused-unevaluated": mapping_of(
            refused_lateral(
                0,
                reason=LateralControlReason.NOT_EVALUATED,
                evaluated=None,
                stamp=None,
                revision=None,
                safety=None,
                temporal=None,
            )
        ),
    }


def longitudinal_bases():
    """Accepted Mission 10A shapes spanning every branch of its own contract."""
    return {
        "permitted-nominal": mapping_of(permitted(0)),
        "permitted-degraded": mapping_of(
            permitted(
                0,
                mode=LongitudinalControlMode.DEGRADED,
                reason=LongitudinalControlReason.DEGRADED_TRACKING,
                throttle=0.0,
                brake=0.10,
                brake_clamped=True,
                safety=SafetyState.DEGRADED,
                lateral_mode=LateralControlMode.DEGRADED,
            )
        ),
        "target-stop": mapping_of(target_stop(0)),
        "trusted-stop": mapping_of(stopping(0)),
        "bound-neutral": mapping_of(bound_neutral(0)),
        "unbound-refusal": mapping_of(unbound_refusal(0)),
        "unevaluated": mapping_of(
            unbound_refusal(
                0,
                reason=LongitudinalControlReason.NOT_EVALUATED,
                evaluated=None,
            )
        ),
    }


REAL_MUTATIONS = (
    0.0, -0.0, 0.10, -0.10, 0.25, 0.35, 0.9999999999999999, 1.0,
    1.0000000000000002, -1.0, -1.0000000000000002, 2.0, -2.0,
    NAN, INF, -INF, 1e-323, SneakyFloat(0.10), True, False, 1, 0, None,
)
AUTHORITY_MUTATIONS = (
    0.0, -0.0, 1e-323, 0.09999999999999999, 0.10, 0.25, 1.0,
    1.0000000000000002, -1e-9, NAN, INF, None,
)
INDEX_MUTATIONS = (None, 0, 5, -1, True, False, SneakyInt(5), 1.0)
FLAG_MUTATIONS = (True, False, 1, 0, None)
TIME_MUTATIONS = (None, BASE, -0.0, NAN, INF, True)
STAMP_MUTATIONS = (
    None,
    GUIDANCE,
    SampleStamp(frame=-1, sim_time=1.25, monotonic=BASE),
    SampleStamp(frame=True, sim_time=1.25, monotonic=BASE),
    SampleStamp(frame=SneakyInt(5), sim_time=1.25, monotonic=BASE),
    SampleStamp(frame=5, sim_time=NAN, monotonic=BASE),
    SampleStamp(frame=5, sim_time=1.25, monotonic=INF),
)
SPEED_MUTATIONS = (
    None, 0.0, -0.0, -1e-9, 1.0, 3.0, 29.999999999999996,
    HARD_TARGET_CEILING_MPS, 30.000000000000004, 100.0, NAN, INF, -INF, True, 1,
)
PEDAL_MUTATIONS = (
    0.0, -0.0, 1e-323, 0.10, 0.20, 0.35, 0.39999999999999997, 0.40,
    0.9999999999999999, 1.0, 1.0000000000000002, -1e-9, -0.10,
    NAN, INF, -INF, True, 1, None, SneakyFloat(0.20),
)


def lateral_mutations():
    return {
        "mode": tuple(LateralControlMode) + (None, Foreign()),
        "reason": tuple(LateralControlReason) + (None, Foreign()),
        "lateral_allowed": FLAG_MUTATIONS + (Foreign(),),
        "steering": REAL_MUTATIONS + (Foreign(),),
        "raw_steering": REAL_MUTATIONS + (Foreign(),),
        "steering_authority": AUTHORITY_MUTATIONS + (Foreign(),),
        "clamped": FLAG_MUTATIONS + (Foreign(),),
        "rate_limited": FLAG_MUTATIONS + (Foreign(),),
        "evaluation_monotonic": TIME_MUTATIONS + (Foreign(),),
        "source_stamp": STAMP_MUTATIONS + (Foreign(),),
        "snapshot_revision": INDEX_MUTATIONS + (Foreign(),),
        "lateral_error": REAL_MUTATIONS + (Foreign(),),
        "heading_error": REAL_MUTATIONS + (Foreign(),),
        "safety_state": tuple(SafetyState) + (None, Foreign()),
        "temporal_state": tuple(TemporalLaneState) + (None, Foreign()),
    }


def longitudinal_mutations():
    return {
        "mode": tuple(LongitudinalControlMode) + (None, Foreign()),
        "reason": tuple(LongitudinalControlReason) + (None, Foreign()),
        "longitudinal_allowed": FLAG_MUTATIONS + (Foreign(),),
        "stop_required": FLAG_MUTATIONS + (Foreign(),),
        "throttle_request": PEDAL_MUTATIONS + (Foreign(),),
        "brake_request": PEDAL_MUTATIONS + (Foreign(),),
        "effective_target_speed_mps": SPEED_MUTATIONS + (Foreign(),),
        "observed_forward_speed_mps": SPEED_MUTATIONS + (Foreign(),),
        "speed_error_mps": SPEED_MUTATIONS + (-3.0, Foreign()),
        "speed_authority_mps": SPEED_MUTATIONS + (Foreign(),),
        "throttle_authority": PEDAL_MUTATIONS + (Foreign(),),
        "brake_authority": PEDAL_MUTATIONS + (Foreign(),),
        "throttle_clamped": FLAG_MUTATIONS + (Foreign(),),
        "brake_clamped": FLAG_MUTATIONS + (Foreign(),),
        "evaluation_monotonic": TIME_MUTATIONS + (Foreign(),),
        "guidance_stamp": STAMP_MUTATIONS + (Foreign(),),
        "speed_stamp": STAMP_MUTATIONS + (Foreign(),),
        "snapshot_revision": INDEX_MUTATIONS + (Foreign(),),
        "target_origin": tuple(TargetSpeedOrigin) + (None, Foreign()),
        "target_epoch": INDEX_MUTATIONS + (Foreign(),),
        "target_revision": INDEX_MUTATIONS + (Foreign(),),
        "target_issued_monotonic": (
            None, TARGET_ISSUED, TARGET_EXPIRY, ISSUE_AFTER_EXPIRY,
            BASE + 60.0, NAN, INF, True, Foreign(),
        ),
        "target_valid_until_monotonic": (
            None, TARGET_EXPIRY, TARGET_ISSUED, EXPIRY_BEFORE_ISSUE,
            BASE - 10.0, NAN, -INF, True, Foreign(),
        ),
        "safety_state": tuple(SafetyState) + (None, Foreign()),
        "lateral_mode": tuple(LateralControlMode) + (None, Foreign()),
        "temporal_state": tuple(TemporalLaneState) + (None, Foreign()),
    }


def lateral_candidates():
    """Every candidate field mapping the lateral differential compares."""
    for base_name, base in lateral_bases().items():
        yield f"{base_name}:pristine", dict(base)
        for field, values in lateral_mutations().items():
            for value in values:
                candidate = dict(base)
                candidate[field] = value
                yield f"{base_name}:{field}={value!r}", candidate
        for reason, mode in _LATERAL_REASON_MODE.items():
            for allowed in (True, False):
                candidate = dict(base)
                candidate["mode"] = mode
                candidate["reason"] = reason
                candidate["lateral_allowed"] = allowed
                yield f"{base_name}:{reason.name}/allowed={allowed}", candidate


def longitudinal_candidates():
    """Every candidate field mapping the longitudinal differential compares."""
    for base_name, base in longitudinal_bases().items():
        yield f"{base_name}:pristine", dict(base)
        for field, values in longitudinal_mutations().items():
            for value in values:
                candidate = dict(base)
                candidate[field] = value
                yield f"{base_name}:{field}={value!r}", candidate
        for reason, mode in _LONGITUDINAL_REASON_MODE.items():
            for allowed in (True, False):
                for stop in (True, False):
                    candidate = dict(base)
                    candidate["mode"] = mode
                    candidate["reason"] = reason
                    candidate["longitudinal_allowed"] = allowed
                    candidate["stop_required"] = stop
                    yield (
                        f"{base_name}:{reason.name}/allowed={allowed}/stop={stop}",
                        candidate,
                    )
        for throttle in (0.0, 1e-323, 0.10, 0.35, 0.36):
            for brake in (0.0, 1e-323, 0.10, 0.40, 0.41):
                candidate = dict(base)
                candidate["throttle_request"] = throttle
                candidate["brake_request"] = brake
                yield f"{base_name}:pedals={throttle!r}/{brake!r}", candidate


class ContractParityTest(EnvelopeTestCase):
    """Mission 10B's own contract judgement equals the accepted constructors'."""

    def accepted_by_upstream(self, cls, candidate):
        """The independent oracle: does the *pristine* accepted contract hold?"""
        try:
            cls(**candidate)
        except (TypeError, ValueError):
            return False
        return True

    def accepted_by_mission_10b(self, cls, candidate, detach, validate, prove):
        """Mission 10B's own detached judgement, with no constructor involved."""
        record = bypassed(self.ANY[cls], **candidate)
        try:
            acquired = detach(record)
        except Exception:  # acquisition is part of the same caller boundary
            return False
        try:
            prove(validate(acquired))
        except _MalformedEvidence:
            return False
        return True

    def setUp(self):
        import control.command as module

        self.module = module
        self.ANY = {
            LateralControlRequest: lateral(0),
            LongitudinalControlRequest: permitted(0),
        }

    def run_differential(self, cls, candidates, detach, validate, prove):
        divergences = []
        accepted = 0
        total = 0
        for label, candidate in candidates:
            total += 1
            expected = self.accepted_by_upstream(cls, candidate)
            accepted += expected
            with self.subTest(candidate=label):
                actual = self.accepted_by_mission_10b(
                    cls, candidate, detach, validate, prove
                )
                if expected != actual:
                    divergences.append((label, expected, actual))
                self.assertEqual(
                    actual,
                    expected,
                    f"{label}: accepted constructor said {expected}, "
                    f"Mission 10B said {actual}",
                )
        # Guard the guard: a differential that accepts everything, or rejects
        # everything, would agree trivially and prove nothing.
        self.assertGreater(accepted, 100, "the matrix must contain valid records")
        self.assertGreater(
            total - accepted, 100, "the matrix must contain invalid records"
        )
        self.assertEqual(divergences, [])

    def test_the_local_lateral_contract_matches_the_accepted_constructor(self):
        self.run_differential(
            LateralControlRequest,
            lateral_candidates(),
            self.module._detach_lateral,
            self.module._validate_detached_lateral,
            self.module._prove_detached_lateral_contract,
        )

    def test_the_local_longitudinal_contract_matches_the_accepted_constructor(self):
        self.run_differential(
            LongitudinalControlRequest,
            longitudinal_candidates(),
            self.module._detach_longitudinal,
            self.module._validate_detached_longitudinal,
            self.module._prove_detached_longitudinal_contract,
        )

    def run_value_parity(self, cls, candidates, detach, validate, prove, rebuild):
        """Every accepted record is rebuilt into the pristine constructor's answer.

        The one permitted difference is Mission 10B's own documented negative-zero
        folding, which is applied during detachment and predates this boundary.
        """
        checked = 0
        folded = 0
        for label, candidate in candidates:
            if not self.accepted_by_upstream(cls, candidate):
                continue
            checked += 1
            with self.subTest(candidate=label):
                pristine = cls(**candidate)
                record = bypassed(self.ANY[cls], **candidate)
                rebuilt = rebuild(validate(detach(record)))
                self.assertIs(type(rebuilt), cls)
                for field in fields(cls):
                    mine = getattr(rebuilt, field.name)
                    theirs = getattr(pristine, field.name)
                    if (
                        type(mine) is float
                        and type(theirs) is float
                        and mine == 0.0
                        and theirs == 0.0
                        and math.copysign(1.0, mine) != math.copysign(1.0, theirs)
                    ):
                        self.assertEqual(math.copysign(1.0, mine), 1.0, field.name)
                        folded += 1
                        continue
                    if mine != mine and theirs != theirs:  # pragma: no cover
                        continue
                    self.assertEqual(mine, theirs, f"{label}: {field.name}")
                    self.assertIs(type(mine), type(theirs), f"{label}: {field.name}")
        self.assertGreater(checked, 100)
        return folded

    def test_the_rebuilt_lateral_record_matches_the_pristine_constructor(self):
        folded = self.run_value_parity(
            LateralControlRequest,
            lateral_candidates(),
            self.module._detach_lateral,
            self.module._validate_detached_lateral,
            self.module._prove_detached_lateral_contract,
            self.module._reconstruct_lateral,
        )
        # Mission 9's own canonicalizer keeps a negative zero; Mission 10B's
        # folds it.  That difference is deliberate, documented, and predates the
        # detached-contract boundary.
        self.assertGreater(folded, 0)

    def test_the_rebuilt_longitudinal_record_matches_the_pristine_constructor(self):
        folded = self.run_value_parity(
            LongitudinalControlRequest,
            longitudinal_candidates(),
            self.module._detach_longitudinal,
            self.module._validate_detached_longitudinal,
            self.module._prove_detached_longitudinal_contract,
            self.module._reconstruct_longitudinal,
        )
        # Mission 10A folds negative zero itself, so nothing can differ at all.
        self.assertEqual(folded, 0)

    def test_the_mirrored_upstream_tables_match_the_accepted_sources(self):
        import control.lateral as lateral_module
        import control.longitudinal as longitudinal_module

        self.assertEqual(
            dict(self.module._LATERAL_REASON_MODE), dict(lateral_module._REASON_MODE)
        )
        self.assertEqual(
            dict(self.module._LONGITUDINAL_REASON_MODE),
            dict(longitudinal_module._REASON_MODE),
        )
        self.assertEqual(len(self.module._LATERAL_REASON_MODE), len(LateralControlReason))
        self.assertEqual(
            len(self.module._LONGITUDINAL_REASON_MODE), len(LongitudinalControlReason)
        )
        self.assertEqual(
            self.module._PERMITTED_LATERAL_MODES, lateral_module._PERMITTED_MODES
        )
        self.assertEqual(
            self.module._PERMITTED_LONGITUDINAL_MODES,
            longitudinal_module._PERMITTED_MODES,
        )
        self.assertEqual(
            self.module._PERMITTED_LATERAL_MODES,
            longitudinal_module._PERMITTED_LATERAL_MODES,
        )
        for upstream in (lateral_module, longitudinal_module):
            self.assertEqual(
                self.module._PERMITTED_SAFETY_STATES, upstream._PERMITTED_SAFETY_STATES
            )
            self.assertEqual(self.module._GUIDANCE_STATES, upstream._GUIDANCE_STATES)
        self.assertEqual(
            self.module.HARD_TARGET_CEILING_MPS,
            longitudinal_module.HARD_TARGET_CEILING_MPS,
        )

    def test_the_upstream_dependency_inventory_is_complete(self):
        """The dependency lists are read off the accepted sources, not assumed.

        A newly added upstream helper would make the fault-injection matrix
        incomplete without failing anything, so the inventory is derived here
        and compared against the declared list.
        """
        import control.lateral as lateral_module
        import control.longitudinal as longitudinal_module

        def module_calls(function, module):
            """Module-level callables this function invokes, transitively."""
            found = set()
            pending = [function]
            seen = set()
            while pending:
                current = pending.pop()
                if current in seen:
                    continue
                seen.add(current)
                tree = ast.parse(
                    textwrap.dedent(inspect.getsource(current))
                )
                for node in ast.walk(tree):
                    if not (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                    ):
                        continue
                    target = getattr(module, node.func.id, None)
                    if (
                        callable(target)
                        and getattr(target, "__module__", None) == module.__name__
                    ):
                        found.add(node.func.id)
                        pending.append(target)
            return found

        for module, post_init, declared in (
            (
                lateral_module,
                LateralControlRequest.__post_init__,
                LATERAL_DEPENDENCIES,
            ),
            (
                longitudinal_module,
                LongitudinalControlRequest.__post_init__,
                LONGITUDINAL_DEPENDENCIES,
            ),
        ):
            with self.subTest(module=module.__name__):
                self.assertEqual(module_calls(post_init, module), set(declared))

    def test_no_upstream_constructor_judges_caller_evidence(self):
        """Structural: no accepted constructor sits inside a malformed catch."""
        with open(self.module.__file__, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        constructors = {
            "_LATERAL_RECONSTRUCTOR",
            "_LONGITUDINAL_RECONSTRUCTOR",
            "LateralControlRequest",
            "LongitudinalControlRequest",
        }
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            absorbs = False
            for handler in node.handlers:
                names = []
                if isinstance(handler.type, ast.Name):
                    names = [handler.type.id]
                elif isinstance(handler.type, ast.Tuple):
                    names = [
                        element.id
                        for element in handler.type.elts
                        if isinstance(element, ast.Name)
                    ]
                if {"TypeError", "ValueError", "Exception"} & set(names):
                    absorbs = True
            if not absorbs:
                continue
            for inner in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id in constructors
                ):
                    offenders.append((inner.func.id, inner.lineno))
        self.assertEqual(offenders, [])
        # The R5 contract names are gone entirely, not merely unused.
        self.assertFalse(hasattr(self.module, "_LATERAL_CONTRACT"))
        self.assertFalse(hasattr(self.module, "_LONGITUDINAL_CONTRACT"))
        self.assertFalse(hasattr(self.module, "_prove_lateral_contract"))
        self.assertFalse(hasattr(self.module, "_prove_longitudinal_contract"))


class DetachedMalformedMatrixTest(BoundaryTestCase):
    """Every locally proven contract violation, through the public API."""

    def assertMalformed(self, lat, lon, reason):
        """One malformed pair: refused, counted as evidence, and recoverable."""
        envelope, warm = self.warmed()
        before = envelope.metrics
        baseline = envelope._state.evidence
        request = envelope.update(lat, lon, now=BASE + STEP)
        self.assertNeutral(request, reason)
        after = envelope.metrics
        self.assertEqual(after.internal_errors, before.internal_errors)
        self.assertEqual(after.refusals, before.refusals + 1)
        self.assertEqual(after.evaluations, before.evaluations + 1)
        self.assertEqual(after.commands, before.commands)
        self.assertEqual(envelope._state.evidence, baseline)
        shaping = envelope._state.shaping
        self.assertEqual(shaping.throttle_anchor, 0.0)
        self.assertEqual(shaping.brake_anchor, 0.0)
        self.assertIsNone(shaping.shaping_monotonic)
        self.assertIs(shaping.transition, _Transition.IDLE)
        self.assertIsNot(request, warm)
        for value in retained_objects(envelope):
            self.assertNotIsInstance(value, BaseException)
            self.assertFalse(hasattr(value, "tb_frame"))
            self.assertIsNot(value, lat)
            self.assertIsNot(value, lon)
        recovered = envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertTrue(recovered.command_allowed)
        self.assertEqual(recovered.throttle_request, 0.0)

    # Each entry names the accepted Mission 9 clause it violates.  The expected
    # outcome is stated here, not read from any Mission 10B predicate.
    LATERAL_VIOLATIONS = (
        ("mode contradicts its reason", dict(mode=LateralControlMode.DEGRADED)),
        (
            "permission contradicts its mode",
            dict(lateral_allowed=False, steering=0.0, raw_steering=0.0,
                 steering_authority=0.0, lateral_error=None, heading_error=None),
        ),
        ("steering above its upper bound", dict(steering=1.0000000000000002)),
        ("steering below its lower bound", dict(steering=-1.0000000000000002)),
        ("steering beyond its own authority", dict(steering=0.99)),
        ("non-finite raw steering", dict(raw_steering=NAN)),
        ("authority above one", dict(steering_authority=1.0000000000000002)),
        ("negative authority", dict(steering_authority=-1e-9)),
        ("lane error above its bound", dict(lateral_error=1.0000000000000002)),
        ("heading error below its bound", dict(heading_error=-1.0000000000000002)),
        ("a permitted request missing its time", dict(evaluation_monotonic=None)),
        ("a permitted request missing its stamp", dict(source_stamp=None)),
        ("a permitted request missing its revision", dict(snapshot_revision=None)),
        ("a permitted request missing its lane error", dict(lateral_error=None)),
        ("a permitted request missing its heading", dict(heading_error=None)),
        (
            "a permitted request under an unpermitted safety state",
            dict(safety_state=SafetyState.FAIL_SAFE),
        ),
        (
            "a permitted request without guidance-bearing state",
            dict(temporal_state=TemporalLaneState.LOST),
        ),
        (
            "nominal mode without nominal safety",
            dict(safety_state=SafetyState.DEGRADED),
        ),
        ("an int where an exact bool is due", dict(clamped=1)),
        ("an int subclass revision", dict(snapshot_revision=SneakyInt(5))),
        ("a negative revision", dict(snapshot_revision=-1)),
        ("a negative stamp frame",
         dict(source_stamp=SampleStamp(frame=-1, sim_time=1.0, monotonic=BASE))),
        ("a non-finite stamp member",
         dict(source_stamp=SampleStamp(frame=5, sim_time=INF, monotonic=BASE))),
    )

    REFUSED_LATERAL_VIOLATIONS = (
        ("a refusal carrying steering", dict(steering=0.10)),
        ("a refusal carrying raw steering", dict(raw_steering=0.10)),
        ("a refusal claiming authority", dict(steering_authority=0.10)),
        ("a refusal reporting clamping", dict(clamped=True)),
        ("a refusal reporting rate limiting", dict(rate_limited=True)),
        ("a refusal exposing lane guidance", dict(lateral_error=0.05)),
        ("a refusal exposing heading guidance", dict(heading_error=0.05)),
    )

    LONGITUDINAL_VIOLATIONS = (
        ("mode contradicts its reason", dict(mode=LongitudinalControlMode.DEGRADED)),
        ("permission contradicts its mode", dict(longitudinal_allowed=False)),
        ("stop requirement contradicts its mode", dict(stop_required=True)),
        ("throttle above one", dict(throttle_request=1.0000000000000002,
                                    throttle_authority=1.0)),
        ("negative throttle", dict(throttle_request=-1e-9)),
        ("brake above one", dict(brake_request=1.0000000000000002,
                                 brake_authority=1.0, throttle_request=0.0)),
        ("negative brake", dict(brake_request=-1e-9)),
        ("both pedals at once", dict(throttle_request=0.20, brake_request=0.20)),
        ("throttle beyond its authority", dict(throttle_request=0.36)),
        (
            "brake beyond its authority",
            dict(throttle_request=0.0, brake_request=0.41),
        ),
        ("throttle authority above one",
         dict(throttle_authority=1.0000000000000002)),
        ("negative brake authority", dict(brake_authority=-1e-9)),
        ("a negative effective target", dict(effective_target_speed_mps=-1e-9)),
        (
            "an effective target above the hard ceiling",
            dict(effective_target_speed_mps=30.000000000000004,
                 speed_authority_mps=30.000000000000004),
        ),
        (
            "a speed authority above the hard ceiling",
            dict(speed_authority_mps=30.000000000000004),
        ),
        ("a negative speed authority", dict(speed_authority_mps=-1e-9)),
        (
            "an effective target above its speed authority",
            dict(effective_target_speed_mps=4.0, speed_authority_mps=3.0),
        ),
        ("a non-finite observed speed", dict(observed_forward_speed_mps=NAN)),
        ("a non-finite speed error", dict(speed_error_mps=INF)),
        (
            "an issue time one ULP after its expiry",
            dict(target_issued_monotonic=ISSUE_AFTER_EXPIRY),
        ),
        (
            "an expiry one ULP before its issue time",
            dict(target_valid_until_monotonic=EXPIRY_BEFORE_ISSUE),
        ),
        ("an incomplete speed law", dict(speed_error_mps=None)),
        (
            "an evaluated law under an unpermitted safety state",
            dict(safety_state=SafetyState.FAIL_SAFE),
        ),
        (
            "an evaluated law under an unpermitted lateral mode",
            dict(lateral_mode=LateralControlMode.DISABLED),
        ),
        (
            "an evaluated law without guidance-bearing state",
            dict(temporal_state=TemporalLaneState.LOST),
        ),
        (
            "nominal authority without fully nominal evidence",
            dict(safety_state=SafetyState.DEGRADED),
        ),
        ("a propelling request missing its guidance stamp",
         dict(guidance_stamp=None)),
        ("a propelling request missing its speed stamp", dict(speed_stamp=None)),
        ("a propelling request missing its target origin", dict(target_origin=None)),
        ("a propelling request missing its target epoch", dict(target_epoch=None)),
        ("a propelling request missing its target revision",
         dict(target_revision=None)),
        ("a propelling request missing its issue time",
         dict(target_issued_monotonic=None)),
        ("a propelling request missing its expiry",
         dict(target_valid_until_monotonic=None)),
        ("a propelling request missing its revision", dict(snapshot_revision=None)),
        ("a negative target epoch", dict(target_epoch=-1)),
        ("an int subclass target revision", dict(target_revision=SneakyInt(1))),
        ("an int where an exact bool is due", dict(brake_clamped=1)),
    )

    def test_every_lateral_contract_violation_is_malformed_lateral_evidence(self):
        for label, overrides in self.LATERAL_VIOLATIONS:
            with self.subTest(violation=label):
                self.assertMalformed(
                    bypassed(lateral(1), **overrides),
                    permitted(1),
                    CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE,
                )

    def test_every_refused_lateral_violation_is_malformed_lateral_evidence(self):
        base = refused_lateral(1, temporal=TemporalLaneState.LOST)
        for label, overrides in self.REFUSED_LATERAL_VIOLATIONS:
            with self.subTest(violation=label):
                self.assertMalformed(
                    bypassed(base, **overrides),
                    permitted(1),
                    CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE,
                )

    def test_every_longitudinal_contract_violation_is_malformed_longitudinal(self):
        for label, overrides in self.LONGITUDINAL_VIOLATIONS:
            with self.subTest(violation=label):
                self.assertMalformed(
                    lateral(1),
                    bypassed(permitted(1), **overrides),
                    CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE,
                )

    def test_a_refused_longitudinal_record_cannot_report_clamping(self):
        for label, overrides in (
            ("a refusal reporting throttle clamping", dict(throttle_clamped=True)),
            ("a brakeless refusal reporting brake clamping",
             dict(brake_clamped=True)),
        ):
            with self.subTest(violation=label):
                self.assertMalformed(
                    lateral(1),
                    bypassed(bound_neutral(1), **overrides),
                    CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE,
                )

    def test_an_unevaluated_record_cannot_expose_evidence(self):
        unevaluated_lateral = refused_lateral(
            1,
            reason=LateralControlReason.NOT_EVALUATED,
            evaluated=None,
            stamp=None,
            revision=None,
            safety=None,
            temporal=None,
        )
        for field, value in (
            ("evaluation_monotonic", BASE + STEP),
            ("source_stamp", GUIDANCE),
            ("snapshot_revision", 5),
            ("safety_state", SafetyState.NOMINAL),
            ("temporal_state", TemporalLaneState.TRACKING),
        ):
            with self.subTest(axis="lateral", field=field):
                self.assertMalformed(
                    bypassed(unevaluated_lateral, **{field: value}),
                    permitted(1),
                    CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE,
                )
        unevaluated_longitudinal = unbound_refusal(
            1, reason=LongitudinalControlReason.NOT_EVALUATED, evaluated=None
        )
        for field, value in (
            ("evaluation_monotonic", BASE + STEP),
            ("guidance_stamp", GUIDANCE),
            ("speed_stamp", SPEED_STAMP),
            ("snapshot_revision", 5),
            ("target_origin", TargetSpeedOrigin.TEST),
            ("target_epoch", 1),
            ("target_revision", 1),
            ("target_issued_monotonic", BASE),
            ("target_valid_until_monotonic", BASE + 50.0),
            ("safety_state", SafetyState.NOMINAL),
            ("lateral_mode", LateralControlMode.NOMINAL),
            ("temporal_state", TemporalLaneState.TRACKING),
            ("observed_forward_speed_mps", 1.0),
        ):
            with self.subTest(axis="longitudinal", field=field):
                self.assertMalformed(
                    lateral(1),
                    bypassed(unevaluated_longitudinal, **{field: value}),
                    CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE,
                )

    def test_a_stopping_record_must_actually_request_brake(self):
        self.assertMalformed(
            lateral(1),
            bypassed(stopping(1), brake_request=0.0),
            CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE,
        )

    def test_a_disabled_record_cannot_carry_pedal_authority(self):
        for field in ("throttle_authority", "brake_authority", "brake_request"):
            with self.subTest(field=field):
                self.assertMalformed(
                    lateral(1),
                    bypassed(bound_neutral(1), **{field: 0.10}),
                    CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE,
                )

    def test_a_stopping_record_cannot_carry_throttle_authority(self):
        for field in ("throttle_authority", "throttle_request"):
            with self.subTest(field=field):
                self.assertMalformed(
                    lateral(1),
                    bypassed(stopping(1), **{field: 0.10}),
                    CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE,
                )

    # ------------------------------------------------------------ boundaries

    def test_numeric_boundaries_are_accepted_inside_and_refused_outside(self):
        """Immediately inside, exactly at, and immediately outside every limit."""
        def speed_law(value):
            """Keep the Mission 10A speed law exact while the target moves."""
            return {
                "speed_authority_mps": HARD_TARGET_CEILING_MPS,
                "observed_forward_speed_mps": 1.0,
                "speed_error_mps": value - 1.0,
            }

        cases = (
            # (axis, field, inside, at, outside, companions(value))
            ("lateral", "steering", 0.34, 0.35, 0.35000000000000003,
             lambda value: {}),
            ("lateral", "lateral_error", 0.9999999999999999, 1.0,
             1.0000000000000002, lambda value: {}),
            ("lateral", "heading_error", -0.9999999999999999, -1.0,
             -1.0000000000000002, lambda value: {}),
            ("lateral", "steering_authority", 0.9999999999999999, 1.0,
             1.0000000000000002, lambda value: {"steering": 0.10}),
            ("longitudinal", "throttle_request", 0.34, 0.35,
             0.35000000000000003, lambda value: {}),
            ("longitudinal", "throttle_authority", 0.9999999999999999, 1.0,
             1.0000000000000002, lambda value: {}),
            ("longitudinal", "effective_target_speed_mps",
             29.999999999999996, HARD_TARGET_CEILING_MPS, 30.000000000000004,
             speed_law),
            ("longitudinal", "speed_authority_mps", 29.999999999999996,
             HARD_TARGET_CEILING_MPS, 30.000000000000004, lambda value: {}),
        )
        for axis, field, inside, at, outside, companions in cases:
            for position, value in (
                ("inside", inside), ("at", at), ("outside", outside)
            ):
                with self.subTest(axis=axis, field=field, position=position):
                    overrides = dict(companions(value))
                    overrides[field] = value
                    if axis == "lateral":
                        lat = bypassed(lateral(1), **overrides)
                        lon = permitted(1)
                        reason = CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE
                        pristine = LateralControlRequest
                    else:
                        lat = lateral(1)
                        lon = bypassed(permitted(1), **overrides)
                        reason = (
                            CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE
                        )
                        pristine = LongitudinalControlRequest
                    # The independent oracle decides what should happen.
                    candidate = mapping_of(lon if axis == "longitudinal" else lat)
                    try:
                        pristine(**candidate)
                    except (TypeError, ValueError):
                        acceptable = False
                    else:
                        acceptable = True
                    envelope, _warm = self.warmed()
                    before = envelope.metrics
                    request = envelope.update(lat, lon, now=BASE + STEP)
                    self.assertEqual(
                        envelope.metrics.internal_errors, before.internal_errors
                    )
                    if acceptable:
                        self.assertNotIn(
                            request.reason,
                            (
                                CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE,
                                CommandEnvelopeReason
                                .MALFORMED_LONGITUDINAL_EVIDENCE,
                            ),
                        )
                    else:
                        self.assertNeutral(request, reason)

    # --------------------------------------------- reconstruction is unreached

    def test_locally_malformed_evidence_never_reaches_reconstruction(self):
        """A bomb in the reconstructor seam must never be triggered."""

        def bomb(*args, **kwargs):
            raise AssertionError("reconstruction must not be reached")

        for axis, seam, lat, lon, reason in (
            (
                "lateral",
                "_LATERAL_RECONSTRUCTOR",
                bypassed(lateral(1), steering=0.99),
                permitted(1),
                CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE,
            ),
            (
                "longitudinal",
                "_LONGITUDINAL_RECONSTRUCTOR",
                lateral(1),
                bypassed(permitted(1), throttle_request=0.20, brake_request=0.20),
                CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE,
            ),
        ):
            with self.subTest(axis=axis):
                envelope, _warm = self.warmed()
                with self.scoped(seam, bomb):
                    request = envelope.update(lat, lon, now=BASE + STEP)
                self.assertNeutral(request, reason)

    def test_no_upstream_dependency_runs_for_locally_malformed_evidence(self):
        """Not one line of accepted upstream constructor code is executed."""
        cases = (
            (
                "lateral",
                bypassed(lateral(1), steering=0.99),
                permitted(1),
                CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE,
                ("control.lateral", "control.longitudinal"),
            ),
            (
                "longitudinal",
                lateral(1),
                bypassed(permitted(1), throttle_request=0.20, brake_request=0.20),
                CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE,
                ("control.longitudinal",),
            ),
        )
        for axis, lat, lon, reason, silent in cases:
            with self.subTest(axis=axis):
                envelope, _warm = self.warmed()
                counts, restore = upstream_call_counts()
                try:
                    request = envelope.update(lat, lon, now=BASE + STEP)
                finally:
                    for module, name, original in restore:
                        setattr(module, name, original)
                self.assertNeutral(request, reason)
                for key, value in counts.items():
                    if key.rsplit(".", 1)[0] in silent:
                        self.assertEqual(value, 0, key)

    def test_each_accepted_record_is_reconstructed_exactly_once(self):
        envelope, _warm = self.warmed()
        lat, lon = lateral(1), permitted(1)  # built before instrumentation
        counts, restore = upstream_call_counts()
        try:
            request = envelope.update(lat, lon, now=BASE + STEP)
        finally:
            for module, name, original in restore:
                setattr(module, name, original)
        self.assertTrue(request.command_allowed)
        # One Mission 9 construction canonicalizes one stamp and three bools;
        # one Mission 10A construction canonicalizes two stamps and four bools.
        self.assertEqual(counts["control.lateral._canonical_stamp"], 1)
        self.assertEqual(counts["control.lateral._builtin_bool"], 3)
        self.assertEqual(counts["control.longitudinal._canonical_stamp"], 2)
        self.assertEqual(counts["control.longitudinal._builtin_bool"], 4)


class UpstreamDependencyFaultTest(BoundaryTestCase):
    """A defect inside an accepted upstream dependency is an internal error.

    This is the exact R5 defect: with a *valid* record in hand, a fault injected
    into one of the mutable helpers the accepted constructor calls was absorbed
    by the stage-2 contract call and published as malformed caller evidence.  The
    fault is injected into accepted upstream code that Mission 10B merely
    depends on, so the honest answer is an internal error and the identical
    original exception.
    """

    def assertReRaised(self, envelope, before, baseline, warm, exception, evaluated):
        self.assertInternalError(envelope, before, evaluated=evaluated)
        visible = envelope.latest_request
        self.assertNotIn(
            visible.reason,
            (
                CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE,
                CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE,
            ),
        )
        self.assertIsNot(visible, warm)
        self.assertEqual(envelope._state.evidence, baseline)
        self.assertNotIn(id(exception), retained_ids(envelope))
        for value in retained_objects(envelope):
            self.assertNotIsInstance(value, BaseException)
            self.assertFalse(hasattr(value, "tb_frame"))

    def run_upstream_fault(self, module_name, helper, exception, *, once, cold):
        import importlib

        module = importlib.import_module(module_name)
        if cold:
            envelope = CommandEnvelope(CONFIG)
            warm = envelope.latest_request
            evaluated = BASE
            control_now = BASE
            index = 0
        else:
            envelope, warm = self.warmed()
            evaluated = BASE + STEP
            control_now = BASE + STEP
            index = 1
        before = envelope.metrics
        baseline = envelope._state.evidence
        # Every record this test hands over is valid and is built *before*
        # injection: a persistent fault would otherwise break the builders
        # themselves, and the update under test would never even be entered.
        lat, lon = lateral(index), permitted(index)
        repeat_lat, repeat_lon = lateral(index), permitted(index)
        recovery_lat, recovery_lon = lateral(index), permitted(index)

        original = getattr(module, helper)
        counters = {"calls": 0, "fired": False}

        def boom(*args, **kwargs):
            counters["calls"] += 1
            if not (once and counters["fired"]):
                counters["fired"] = True
                raise exception
            return original(*args, **kwargs)

        setattr(module, helper, boom)
        try:
            with self.assertRaises(type(exception)) as caught:
                envelope.update(lat, lon, now=control_now)
            self.assertIs(caught.exception, exception)
            self.assertGreater(counters["calls"], 0)
            self.assertReRaised(
                envelope, before, baseline, warm, caught.exception, evaluated
            )
            if not once:
                # A persistent fault repeats the same fail-closed behaviour.
                repeat_before = envelope.metrics
                with self.assertRaises(type(exception)):
                    envelope.update(repeat_lat, repeat_lon, now=control_now)
                self.assertEqual(
                    envelope.metrics.internal_errors,
                    repeat_before.internal_errors + 1,
                )
                self.assertIs(
                    envelope.latest_request.reason,
                    CommandEnvelopeReason.INTERNAL_ERROR,
                )
        finally:
            setattr(module, helper, original)
        recovered = envelope.update(recovery_lat, recovery_lon, now=control_now)
        self.assertTrue(recovered.command_allowed)
        # The first propulsion after recovery is zero and rate limited.
        self.assertEqual(recovered.throttle_request, 0.0)
        self.assertEqual(recovered.brake_request, 0.0)

    def test_every_lateral_upstream_dependency_fault_is_an_internal_error(self):
        for helper in LATERAL_DEPENDENCIES:
            for factory in (TypeError, ValueError, RuntimeError):
                for once in (False, True):
                    with self.subTest(helper=helper, raises=factory.__name__,
                                      once=once):
                        self.run_upstream_fault(
                            "control.lateral",
                            helper,
                            factory(f"upstream {helper} defect"),
                            once=once,
                            cold=False,
                        )

    def test_every_longitudinal_upstream_dependency_fault_is_an_internal_error(self):
        for helper in LONGITUDINAL_DEPENDENCIES:
            for factory in (TypeError, ValueError, RuntimeError):
                for once in (False, True):
                    with self.subTest(helper=helper, raises=factory.__name__,
                                      once=once):
                        self.run_upstream_fault(
                            "control.longitudinal",
                            helper,
                            factory(f"upstream {helper} defect"),
                            once=once,
                            cold=False,
                        )

    def test_an_upstream_dependency_fault_from_a_cold_controller_still_re_raises(self):
        for module_name, helper in (
            ("control.lateral", "_finite_real"),
            ("control.longitudinal", "_canonical_stamp"),
        ):
            for factory in (TypeError, ValueError, RuntimeError):
                with self.subTest(module=module_name, raises=factory.__name__):
                    self.run_upstream_fault(
                        module_name,
                        helper,
                        factory("cold upstream defect"),
                        once=False,
                        cold=True,
                    )

    def run_module_fault(self, name, exception):
        """One Mission 10B internal step raises, with the fault scoped tightly."""
        envelope, warm = self.warmed()
        before = envelope.metrics
        baseline = envelope._state.evidence
        lat, lon = lateral(1), permitted(1)

        def boom(*args, **kwargs):
            raise exception

        with self.scoped(name, boom):
            with self.assertRaises(type(exception)) as caught:
                envelope.update(lat, lon, now=BASE + STEP)
        self.assertIs(caught.exception, exception)
        self.assertReRaised(
            envelope, before, baseline, warm, caught.exception, BASE + STEP
        )
        recovered = envelope.update(lateral(1), permitted(1), now=BASE + STEP)
        self.assertTrue(recovered.command_allowed)

    def test_the_reconstruction_seams_report_internal_errors(self):
        for seam in ("_LATERAL_RECONSTRUCTOR", "_LONGITUDINAL_RECONSTRUCTOR"):
            for factory in (TypeError, ValueError, RuntimeError):
                with self.subTest(seam=seam, raises=factory.__name__):
                    self.run_module_fault(seam, factory(f"{seam} defect"))

    def test_the_detached_contract_proofs_report_internal_errors(self):
        for name in (
            "_prove_detached_lateral_contract",
            "_prove_detached_longitudinal_contract",
            "_evidence_bounded",
            "_evidence_real",
            "_evidence_bool",
            "_evidence_index",
            "_evidence_stamp",
        ):
            for factory in (TypeError, ValueError, RuntimeError):
                with self.subTest(step=name, raises=factory.__name__):
                    self.run_module_fault(
                        name, factory(f"internal {name} defect")
                    )

    def test_a_malformed_record_and_an_upstream_fault_answer_differently(self):
        """The two classifications must not collapse into one another."""
        import control.lateral as lateral_module

        malformed, _warm = self.warmed()
        malformed_request = malformed.update(
            bypassed(lateral(1), steering=0.99), permitted(1), now=BASE + STEP
        )

        faulty, _warm2 = self.warmed()
        before = faulty.metrics
        exception = ValueError("upstream defect")
        lat, lon = lateral(1), permitted(1)
        original = lateral_module._finite_real

        def boom(*args, **kwargs):
            raise exception

        lateral_module._finite_real = boom
        try:
            with self.assertRaises(ValueError) as caught:
                faulty.update(lat, lon, now=BASE + STEP)
        finally:
            lateral_module._finite_real = original

        self.assertIs(
            malformed_request.reason,
            CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE,
        )
        self.assertEqual(malformed.metrics.internal_errors, 0)
        self.assertIs(caught.exception, exception)
        self.assertIs(
            faulty.latest_request.reason, CommandEnvelopeReason.INTERNAL_ERROR
        )
        self.assertEqual(
            faulty.metrics.internal_errors, before.internal_errors + 1
        )
        # Both remain fail-closed and neutral; only the explanation differs.
        for request in (malformed_request, faulty.latest_request):
            self.assertIs(request.mode, CommandEnvelopeMode.DISABLED)
            self.assertFalse(request.command_allowed)
            self.assertEqual(request.steering_authority, 0.0)
            self.assertEqual(request.throttle_authority, 0.0)
            self.assertEqual(request.brake_authority, 0.0)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    unittest.main()
