"""Bounded, deterministic command *envelope* over two accepted control requests.

Mission 10B answers exactly one question: given the Mission 9 lateral request and
the Mission 10A longitudinal request that were produced for one evidence cut at
one control time, what single bounded, coherent, immutable command envelope may
be *requested* right now?

It is an envelope producer, not an actuator.  It builds no ``carla.VehicleControl``,
calls no ``apply_control``, owns no CARLA actor, contacts no simulator, and reads
no sensor, target, perception result, or Mission 8 decision.  The published record
is named :class:`CommandEnvelopeRequest` precisely because nothing here is ever
applied to a vehicle.  It creates no thread, queue, timer, socket, or clock: the
caller supplies host-monotonic time explicitly, exactly as Missions 8, 9, and 10A
require.

Scope boundary
--------------
Mission 10B combines, bounds, and shapes two already-published *requests*.  It
never re-decides safety, never re-derives lane geometry, never re-derives speed,
and never re-runs a speed law.  It implements no adaptive cruise control, no
obstacle detection, no collision avoidance, and no reverse driving.  Mission 9
already owns steering slew, so Mission 10B adds **no second steering rate
limiter**: it validates the steering intent and clamps it to the final authority
and nothing more.

Trust model
-----------
Both inputs are treated as hostile, caller-controlled evidence.  Neither record is
spot-checked on the handful of fields the envelope happens to consume: each is
read exactly once per public field, detached into built-ins and accepted enums,
proven field by field and relationship by relationship against the *whole*
accepted Mission 9 or Mission 10A contract, and only then reconstructed -- exactly
once -- through its own accepted constructor.  The producer relationships those
constructors do not encode -- which reasons carry a binding, which carry speed-law
evidence, which imply which upstream states -- are proven separately.  No caller
object, nested mutable object, exception, or traceback is ever retained.

That contract proof is deliberately Mission 10B's *own* code rather than an
upstream constructor call.  A constructor is not a classifier: it runs accepted
upstream code and mutable upstream helpers, and a genuine defect in that
machinery raises the same ``TypeError`` or ``ValueError`` a contract rejection
does.  Catching around it would answer an upstream defect as untrustworthy caller
evidence.  Restating each clause locally costs bounded duplication, which is
pinned by differential tests against the pristine accepted constructors, and buys
a boundary where every malformed answer is a deliberate signal raised at a named
invariant.

Expected failure and unexpected failure are separated by *where* a failure
happens, never by which exception class it happens to be.  Reading a caller's
record and proving it against its own accepted contract are the only stages that
make a statement about the caller, and they fail closed with a neutral malformed
refusal.  Everything after them -- the one reconstruction of the detached record,
proving the producer relationships, shaping, and publication -- is internal work,
so an unexpected exception there is a controller defect: it publishes
``INTERNAL_ERROR``, counts as one, and is re-raised unchanged rather than being
reported as bad evidence.  A raw ``TypeError`` or ``ValueError`` is therefore
never caught around internal processing, and never caught merely because it arose
during a constructor call.

Classification of a rejected pair
---------------------------------
* an intrinsic Mission 9 failure is ``MALFORMED_LATERAL_EVIDENCE``;
* an intrinsic Mission 10A failure is ``MALFORMED_LONGITUDINAL_EVIDENCE``;
* a pair whose two axes disagree about whether motion is permitted at all -- a
  permitted longitudinal request handed a refused lateral request -- is
  ``AXIS_CONFLICT``.  This is Mission 10B's step "reject permitted M10A paired
  with refused M9", and it is settled at the *head* of the binding stage rather
  than after it, because the mode equality below would otherwise mask a genuine
  permission disagreement as a mere identity mismatch.  Both classifications stay
  fail-closed and neutral; only the published explanation differs;
* any other disagreement between the current pair is
  ``CROSS_RECORD_IDENTITY_MISMATCH``;
* a replay, a regression, or a mutated duplicate is ``SOURCE_REJECTED``;
* upstream evaluation times that agree with each other but not with ``now`` is
  ``TIME_INVALID``, as is a non-finite or regressed ``now``;
* an upstream record that simply declined to act is ``UPSTREAM_REFUSAL``.

Asymmetric shaping
------------------
Propulsion is slow to arrive and instant to leave; deceleration is instant to
arrive and slow to leave.  Throttle *increases* are rate limited, throttle
decreases are immediate; ordinary brake *increases* are immediate, ordinary brake
releases are rate limited.  A trusted Mission 10A controlled stop is neither
shaped nor weakened nor strengthened: its brake value and its brake authority are
republished exactly.

Brake-to-throttle interlock
---------------------------
The envelope never transitions directly from positive brake to positive throttle.
A throttle request made while accepted brake history is still positive suppresses
throttle, releases brake at the configured rate, and the update that first reaches
zero brake also publishes zero throttle and enters a zero-pedal dwell.  Only a
later, strictly newer accepted update may begin the throttle rise.

Boundedness
-----------
Retained state is O(1) and split into two independent fixed-size groups: an
evidence/order group (one evaluation baseline, one guidance stamp, one speed
stamp, one target identity, one canonical fingerprint) that advances only after a
fully bound accepted output, and a shaping/interlock group (two pedal anchors, one
shaping time, one bounded transition state) that every neutral refusal clears.
There is no command, frame, or decision history, no queue, and no log.  The
envelope is single-consumer and lock-free; its metrics are diagnostics read by
that same consumer and are explicitly **not** claimed to be concurrently atomic.

Every default in :class:`CommandEnvelopeConfig` is a deliberately conservative
offline exhibition placeholder.  Nothing here has been validated against a real
vehicle, a real road, or a running simulator, and no claim is made about stopping
distance, deceleration, or any other physical outcome.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, fields, replace
from enum import Enum

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


# Structural bounds on the configurable envelope policy.  These bound the
# *contract*, not any measured vehicle limit.
MAX_ENVELOPE_RATE_PER_SECOND = 100.0
MAX_SHAPING_DELTA_SECONDS = 60.0


__all__ = (
    "DEFAULT_COMMAND_ENVELOPE_CONFIG",
    "MAX_ENVELOPE_RATE_PER_SECOND",
    "MAX_SHAPING_DELTA_SECONDS",
    "CommandEnvelope",
    "CommandEnvelopeConfig",
    "CommandEnvelopeMetrics",
    "CommandEnvelopeMode",
    "CommandEnvelopeReason",
    "CommandEnvelopeRequest",
)


# --------------------------------------------------------------------------
# Canonicalizers
# --------------------------------------------------------------------------


def _finite_real(value: object, name: str) -> float:
    """Canonicalize one caller number into a detached, finite built-in float.

    A ``bool`` is rejected outright even though Python calls it an integer, the
    conversion happens exactly once so a hostile ``__float__`` cannot answer two
    different values, and negative zero is folded onto ``+0.0`` so a sign that
    carries no magnitude can never reach a comparison or a published field.
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


def _acquire_real(value: object, name: str) -> float:
    """Perform only caller-controlled numeric work for evidence acquisition.

    The type checks and the one ``float`` conversion may consult caller code and
    therefore run inside the narrow malformed-evidence boundary.  Finiteness,
    normalization, bounds, and every other validation step deliberately happen
    later, after this function has returned a detached built-in float.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number")
    return float(value)


def _bounded_real(value: object, name: str, *, low: float, high: float) -> float:
    result = _finite_real(value, name)
    if not low <= result <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return result


def _nonnegative_builtin_int(value: object, name: str) -> int:
    """Require an exact built-in ``int``, so no ``__index__`` is ever consulted."""
    if isinstance(value, bool) or type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative built-in integer")
    return value


def _builtin_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a built-in bool")
    return value


_ACQUIRED_STAMP = object()


def _acquire_stamp(value: object, name: str) -> tuple[object, object, float, float]:
    """Read one caller-owned stamp exactly once without validating it internally."""
    if not isinstance(value, SampleStamp):
        raise ValueError(f"{name} must be a SampleStamp")
    frame = value.frame
    sim_time = value.sim_time
    monotonic = value.monotonic
    return (
        _ACQUIRED_STAMP,
        frame,
        _acquire_real(sim_time, f"{name}.sim_time"),
        _acquire_real(monotonic, f"{name}.monotonic"),
    )


def _canonical_stamp(value: object, name: str) -> SampleStamp:
    """Detach one stamp into a fresh, exact, built-in-valued base instance.

    Each member is read exactly once, so a mutable or hostile stamp subtype
    cannot answer one value to a gate and another to a comparison, and the
    returned object is an exact :class:`SampleStamp`, never the caller's.
    """
    if (
        type(value) is tuple
        and len(value) == 4
        and value[0] is _ACQUIRED_STAMP
    ):
        _, frame, sim_time, monotonic = value
    else:
        if not isinstance(value, SampleStamp):
            raise ValueError(f"{name} must be a SampleStamp")
        frame = value.frame
        sim_time = value.sim_time
        monotonic = value.monotonic
    return SampleStamp(
        frame=_nonnegative_builtin_int(frame, f"{name}.frame"),
        sim_time=_finite_real(sim_time, f"{name}.sim_time"),
        monotonic=_finite_real(monotonic, f"{name}.monotonic"),
    )


def _stamp_key(stamp: SampleStamp | None) -> tuple[object, ...]:
    """Expand one canonical stamp into exact scalars for the fingerprint."""
    if stamp is None:
        return (None,)
    return (stamp.frame, stamp.sim_time, stamp.monotonic)


# The sentinel a failed caller-time conversion produces.  It is a distinct object
# rather than ``None`` so the "the caller's clock could not be converted at all"
# case can never be confused with a converted value.
_CALLER_TIME_UNCONVERTIBLE = object()


def _convert_caller_time(value: object) -> float:
    """Stage 1+2 of the control clock: the *only* caller-controlled work.

    This performs exactly two things and nothing else -- it applies the public
    input contract that rejects ``bool``, and it performs the single
    caller-controlled conversion ``float(value)``.  Both consult caller-defined
    machinery: a hostile ``__class__`` can raise from the type checks and a
    registered ``numbers.Real`` can raise anything at all from ``__float__``.

    No finiteness test, no normalization, and no other controller work happens
    here, so the narrow exception boundary its caller installs around this call
    covers caller failures *only* and can never absorb an internal defect.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("now must be a finite real number")
    return float(value)


def _normalized_control_time(converted: object) -> float | None:
    """Stage 5 of the control clock: internal-only, inside the normal boundary.

    ``converted`` is whatever the one caller conversion returned, so it is a
    ``float`` or -- because ``__float__`` may still return a strict subclass --
    an instance of one.  ``float.__float__`` reads the stored double straight out
    of the object at C level without consulting any caller-defined method, so the
    normalization to an exact built-in ``float`` runs no caller code.  The
    finiteness test and the signed-zero fold then follow on that exact value.

    ``None`` means the caller's clock is genuinely not a usable control time,
    which is ``TIME_INVALID`` and not an internal error.  Anything that goes
    wrong *inside* this function is a controller defect and is deliberately left
    to propagate into the internal-error boundary.
    """
    value = float.__float__(converted)
    if not math.isfinite(value):
        return None
    return value + 0.0


# --------------------------------------------------------------------------
# Public vocabulary
# --------------------------------------------------------------------------


class CommandEnvelopeMode(Enum):
    """How much command authority this envelope carries, if any."""

    DISABLED = "disabled"
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    STOPPING = "stopping"


class CommandEnvelopeReason(Enum):
    """One bounded explanation for the current command envelope."""

    NOT_EVALUATED = "not_evaluated"
    MALFORMED_LATERAL_EVIDENCE = "malformed_lateral_evidence"
    MALFORMED_LONGITUDINAL_EVIDENCE = "malformed_longitudinal_evidence"
    CROSS_RECORD_IDENTITY_MISMATCH = "cross_record_identity_mismatch"
    TIME_INVALID = "time_invalid"
    SOURCE_REJECTED = "source_rejected"
    UPSTREAM_REFUSAL = "upstream_refusal"
    AXIS_CONFLICT = "axis_conflict"
    INTERNAL_ERROR = "internal_error"

    NOMINAL_COMMAND = "nominal_command"
    NOMINAL_SHAPED = "nominal_shaped"
    NOMINAL_TRANSITION_INTERLOCK = "nominal_transition_interlock"

    DEGRADED_COMMAND = "degraded_command"
    DEGRADED_SHAPED = "degraded_shaped"
    DEGRADED_TRANSITION_INTERLOCK = "degraded_transition_interlock"

    TRUSTED_UPSTREAM_STOP = "trusted_upstream_stop"


# Every reason belongs to exactly one mode, so an envelope can never advertise
# command authority under a fail-closed explanation, and a neutral refusal can
# never present itself as a trusted controlled stop.
_REASON_MODE: dict[CommandEnvelopeReason, CommandEnvelopeMode] = {
    CommandEnvelopeReason.NOT_EVALUATED: CommandEnvelopeMode.DISABLED,
    CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE: CommandEnvelopeMode.DISABLED,
    CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE: (
        CommandEnvelopeMode.DISABLED
    ),
    CommandEnvelopeReason.CROSS_RECORD_IDENTITY_MISMATCH: (
        CommandEnvelopeMode.DISABLED
    ),
    CommandEnvelopeReason.TIME_INVALID: CommandEnvelopeMode.DISABLED,
    CommandEnvelopeReason.SOURCE_REJECTED: CommandEnvelopeMode.DISABLED,
    CommandEnvelopeReason.UPSTREAM_REFUSAL: CommandEnvelopeMode.DISABLED,
    CommandEnvelopeReason.AXIS_CONFLICT: CommandEnvelopeMode.DISABLED,
    CommandEnvelopeReason.INTERNAL_ERROR: CommandEnvelopeMode.DISABLED,
    CommandEnvelopeReason.NOMINAL_COMMAND: CommandEnvelopeMode.NOMINAL,
    CommandEnvelopeReason.NOMINAL_SHAPED: CommandEnvelopeMode.NOMINAL,
    CommandEnvelopeReason.NOMINAL_TRANSITION_INTERLOCK: CommandEnvelopeMode.NOMINAL,
    CommandEnvelopeReason.DEGRADED_COMMAND: CommandEnvelopeMode.DEGRADED,
    CommandEnvelopeReason.DEGRADED_SHAPED: CommandEnvelopeMode.DEGRADED,
    CommandEnvelopeReason.DEGRADED_TRANSITION_INTERLOCK: (
        CommandEnvelopeMode.DEGRADED
    ),
    CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP: CommandEnvelopeMode.STOPPING,
}

# An envelope carries a command in exactly these three modes and nowhere else.
_ACTIONABLE_MODES = frozenset(
    (
        CommandEnvelopeMode.NOMINAL,
        CommandEnvelopeMode.DEGRADED,
        CommandEnvelopeMode.STOPPING,
    )
)

# The reason triples selected for a permitted pair, keyed by envelope mode.
_COMMAND_REASONS: dict[CommandEnvelopeMode, tuple[CommandEnvelopeReason, ...]] = {
    CommandEnvelopeMode.NOMINAL: (
        CommandEnvelopeReason.NOMINAL_COMMAND,
        CommandEnvelopeReason.NOMINAL_SHAPED,
        CommandEnvelopeReason.NOMINAL_TRANSITION_INTERLOCK,
    ),
    CommandEnvelopeMode.DEGRADED: (
        CommandEnvelopeReason.DEGRADED_COMMAND,
        CommandEnvelopeReason.DEGRADED_SHAPED,
        CommandEnvelopeReason.DEGRADED_TRANSITION_INTERLOCK,
    ),
}

# Every pedal-shaping flag combination the real ``_command`` state machine can
# publish, in the order (throttle clamp, brake clamp, throttle rise limit, brake
# release limit, transition interlock).  Steering clamping is independent and
# therefore doubles this finite set.  A controlled stop has no pedal shaping and
# is validated separately.
_REACHABLE_PEDAL_SHAPING_SHAPES = frozenset(
    (
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
    )
)

# Mission 8 permits autonomy in exactly these two states.
_PERMITTED_SAFETY_STATES = frozenset((SafetyState.NOMINAL, SafetyState.DEGRADED))

# Mission 9 carries lateral authority in exactly these two modes.
_PERMITTED_LATERAL_MODES = frozenset(
    (LateralControlMode.NOMINAL, LateralControlMode.DEGRADED)
)

# Mission 10A carries propulsion authority in exactly these two modes.
_PERMITTED_LONGITUDINAL_MODES = frozenset(
    (LongitudinalControlMode.NOMINAL, LongitudinalControlMode.DEGRADED)
)

# The only Mission 7 states that can carry guidance at all.
_GUIDANCE_STATES = frozenset(
    (TemporalLaneState.TRACKING, TemporalLaneState.COASTING)
)


# --------------------------------------------------------------------------
# Accepted upstream producer shapes
# --------------------------------------------------------------------------

# Mission 10A publishes its twelve-field identity binding together or not at all.
# These are exactly the reasons its refusal path reaches, all of which publish no
# binding whatsoever; every other reason is published with a complete binding.
_UNBOUND_LONGITUDINAL_REASONS = frozenset(
    (
        LongitudinalControlReason.NOT_EVALUATED,
        LongitudinalControlReason.TIME_INVALID,
        LongitudinalControlReason.SPEED_UNAVAILABLE,
        LongitudinalControlReason.IDENTITY_MISMATCH,
        LongitudinalControlReason.EVIDENCE_MALFORMED,
        LongitudinalControlReason.VEHICLE_NOT_ALIVE,
        LongitudinalControlReason.SOURCE_REJECTED,
    )
)

# The reasons whose outcome ran the Mission 10A speed law, and therefore publish
# the complete target/observed/error/authority quadruple.
_SPEED_LAW_LONGITUDINAL_REASONS = frozenset(
    (
        LongitudinalControlReason.NOMINAL_TRACKING,
        LongitudinalControlReason.DEGRADED_TRACKING,
        LongitudinalControlReason.DEGRADED_COASTING,
        LongitudinalControlReason.TARGET_STOP,
    )
)

# The reasons Mission 10A can only reach *after* both the Mission 8 permission
# gate and the Mission 9 permission gate have already been passed.
_GATED_LONGITUDINAL_REASONS = _SPEED_LAW_LONGITUDINAL_REASONS | frozenset(
    (
        LongitudinalControlReason.TARGET_EXPIRED,
        LongitudinalControlReason.SPEED_STALE,
        LongitudinalControlReason.EXCESSIVE_LATERAL_MOTION,
        LongitudinalControlReason.REVERSE_MOTION,
    )
)

# The exact presence pattern of the four optional speed-law fields, in the order
# (effective target, observed forward speed, speed error, speed authority), that
# each accepted Mission 10A publishing branch actually emits.  Mission 10A has
# exactly three such branch shapes: its refusal path publishes no speed evidence
# at all, its bound neutral and its non-target controlled stops publish the
# observed forward speed alone, and only the branches that actually ran the speed
# law publish the complete quadruple.  Any other combination of the four is a
# shape no accepted producer emits, and is therefore malformed evidence rather
# than something to be carried forward or repaired.
_NO_SPEED_EVIDENCE = (False, False, False, False)
_OBSERVED_SPEED_ONLY = (False, True, False, False)
_COMPLETE_SPEED_LAW = (True, True, True, True)

# Whether each *unbound* Mission 10A refusal states an evaluation time, as a set
# of the presence values that reason can actually publish.  Read off the accepted
# source rather than assumed:
#
# * `NOT_EVALUATED` is the record `_clear_state` publishes and its own
#   constructor forbids it from exposing any evidence at all, time included.
# * `TIME_INVALID` has **two** real branches and therefore **both** shapes: the
#   two clock gates in `LongitudinalController.update` refuse with `None`
#   because no trustworthy time exists, while the `_TimeInvalid` raised out of
#   `_read_evidence` -- a decision age that contradicts the control clock, or a
#   negative source age -- refuses with the canonical `evaluated` already in
#   hand.  Requiring `None` here would reject a legitimate Mission 10A output.
# * every other unbound refusal is reached with a canonical `evaluated` and
#   always states it.
_UNBOUND_TIME_SHAPE: dict[LongitudinalControlReason, frozenset[bool]] = {
    LongitudinalControlReason.NOT_EVALUATED: frozenset((False,)),
    LongitudinalControlReason.TIME_INVALID: frozenset((False, True)),
    LongitudinalControlReason.SPEED_UNAVAILABLE: frozenset((True,)),
    LongitudinalControlReason.IDENTITY_MISMATCH: frozenset((True,)),
    LongitudinalControlReason.EVIDENCE_MALFORMED: frozenset((True,)),
    LongitudinalControlReason.VEHICLE_NOT_ALIVE: frozenset((True,)),
    LongitudinalControlReason.SOURCE_REJECTED: frozenset((True,)),
}

_SPEED_LAW_SHAPE: dict[LongitudinalControlReason, tuple[bool, bool, bool, bool]] = {
    reason: (
        _NO_SPEED_EVIDENCE
        if reason in _UNBOUND_LONGITUDINAL_REASONS
        else _COMPLETE_SPEED_LAW
        if reason in _SPEED_LAW_LONGITUDINAL_REASONS
        else _OBSERVED_SPEED_ONLY
    )
    for reason in LongitudinalControlReason
}

# The accepted Mission 9 and Mission 10A reason/mode contracts, mirrored here so
# the *public* envelope record can prove a provenance combination is reachable
# without importing a private upstream table.  Mission 10B tests pin both tables
# against the accepted upstream maps, so a drift shows up as a test failure
# rather than as a silently weaker output invariant.
_LATERAL_REASON_MODE: dict[LateralControlReason, LateralControlMode] = {
    LateralControlReason.NOT_EVALUATED: LateralControlMode.DISABLED,
    LateralControlReason.NOMINAL_TRACKING: LateralControlMode.NOMINAL,
    LateralControlReason.DEGRADED_TRACKING: LateralControlMode.DEGRADED,
    LateralControlReason.DEGRADED_COASTING: LateralControlMode.DEGRADED,
    LateralControlReason.SAFETY_NOT_PERMITTED: LateralControlMode.DISABLED,
    LateralControlReason.INSUFFICIENT_GEOMETRY: LateralControlMode.DISABLED,
    LateralControlReason.IDENTITY_MISMATCH: LateralControlMode.DISABLED,
    LateralControlReason.EVIDENCE_MALFORMED: LateralControlMode.DISABLED,
    LateralControlReason.SOURCE_REJECTED: LateralControlMode.DISABLED,
    LateralControlReason.TIME_INVALID: LateralControlMode.DISABLED,
}

_LONGITUDINAL_REASON_MODE: dict[
    LongitudinalControlReason, LongitudinalControlMode
] = {
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

# Mission 9 refusals that discard their whole evidence baseline publish no source
# stamp, no revision, no temporal state, no safety state, and no evaluation time
# at all.
_EVIDENCE_FREE_LATERAL_REASONS = frozenset(
    (
        LateralControlReason.NOT_EVALUATED,
        LateralControlReason.TIME_INVALID,
    )
)

# The two Mission 9 refusals whose branch is reached before the Mission 8 gate
# has been read at all, and which therefore publish a safety state exactly when
# they publish bound evidence.
_EVIDENCE_REFUSAL_LATERAL_REASONS = frozenset(
    (
        LateralControlReason.IDENTITY_MISMATCH,
        LateralControlReason.EVIDENCE_MALFORMED,
    )
)

# The Mission 8 states that exist but never permit autonomy.  For any decision
# the accepted Mission 8 constructor would build, ``autonomy_allowed`` is exactly
# ``state in {NOMINAL, DEGRADED}`` and only a ``FAIL_SAFE`` decision may be
# latched, so Mission 9's permission gate fails exactly on these three.
_UNPERMITTED_SAFETY_STATES = frozenset(
    (
        SafetyState.INITIALIZING,
        SafetyState.RECOVERING,
        SafetyState.FAIL_SAFE,
    )
)

# The Mission 7 states a *bound* Mission 9 record can carry.  Mission 9 reads its
# evidence only out of a ``TemporalLaneObservation``, whose own constructor
# refuses to wrap an ``UNINITIALIZED`` estimate at all ("a temporal observation
# requires an accepted estimate"), so no bound Mission 9 record can ever expose
# ``UNINITIALIZED`` however its other fields are arranged.
_BOUND_TEMPORAL_STATES = frozenset(
    (
        TemporalLaneState.TRACKING,
        TemporalLaneState.COASTING,
        TemporalLaneState.LOST,
        TemporalLaneState.INPUT_UNUSABLE,
    )
)

# The exact ``(safety state, temporal state)`` combinations each *bound* Mission
# 9 reason can publish, reconstructed branch by branch from ``control/lateral.py``
# together with the producers that bound its inputs.  The reconstruction and the
# reasoning behind each entry are in ``_lateral_producer_conflict``.
_BOUND_LATERAL_SHAPES: dict[
    LateralControlReason, tuple[frozenset, frozenset]
] = {
    # Refused at the source-order gate, before the Mission 8 permission is read.
    LateralControlReason.SOURCE_REJECTED: (
        frozenset((None,)),
        _BOUND_TEMPORAL_STATES,
    ),
    # Published exactly when the Mission 8 gate fails.
    LateralControlReason.SAFETY_NOT_PERMITTED: (
        _UNPERMITTED_SAFETY_STATES | frozenset((None,)),
        _BOUND_TEMPORAL_STATES,
    ),
    # The late branches, past the Mission 8 gate, where the decision turns out
    # not to describe this evidence at all.
    LateralControlReason.IDENTITY_MISMATCH: (
        _PERMITTED_SAFETY_STATES,
        _BOUND_TEMPORAL_STATES,
    ),
    LateralControlReason.EVIDENCE_MALFORMED: (
        _PERMITTED_SAFETY_STATES,
        _BOUND_TEMPORAL_STATES,
    ),
    # Past the gate *and* past the identity comparison, with no complete-lane
    # geometry to steer from.
    LateralControlReason.INSUFFICIENT_GEOMETRY: (
        frozenset((SafetyState.DEGRADED,)),
        _GUIDANCE_STATES,
    ),
    LateralControlReason.NOMINAL_TRACKING: (
        frozenset((SafetyState.NOMINAL,)),
        frozenset((TemporalLaneState.TRACKING,)),
    ),
    LateralControlReason.DEGRADED_TRACKING: (
        frozenset((SafetyState.DEGRADED,)),
        frozenset((TemporalLaneState.TRACKING,)),
    ),
    LateralControlReason.DEGRADED_COASTING: (
        frozenset((SafetyState.DEGRADED,)),
        frozenset((TemporalLaneState.COASTING,)),
    ),
}


def _lateral_producer_conflict(
    reason: LateralControlReason,
    safety_state: SafetyState | None,
    temporal_state: TemporalLaneState | None,
    bound: bool,
) -> str | None:
    """Why Mission 9 could not have published this shape, or ``None``.

    One pure statement of the accepted Mission 9 publishing branches, shared by
    the Mission 9 canonicalizer and by the public envelope record so the two can
    never disagree about which combinations exist.  Every rule is read off
    ``control/lateral.py`` and the accepted producers that bound its inputs:

    * ``NOT_EVALUATED`` and ``TIME_INVALID`` discard the whole baseline and
      publish nothing -- no stamp, no revision, no temporal state, no safety
      state, no evaluation time.  Mission 9's clock gate always refuses with
      ``evaluated=None``, so both of its ``TIME_INVALID`` branches are unbound.
    * ``SOURCE_REJECTED`` is refused at the source-order gate, *before* the
      Mission 8 permission is read, so it publishes bound evidence and no safety
      state whatsoever.
    * ``IDENTITY_MISMATCH`` and ``EVIDENCE_MALFORMED`` each have two branches:
      an early one before any evidence exists, which publishes neither evidence
      nor a safety state, and a late one past the Mission 8 gate, which
      publishes both.  They are therefore bound exactly when they carry a state,
      and a bound one has already passed that gate, so its state is permitted.
    * ``SAFETY_NOT_PERMITTED`` is published exactly when the Mission 8 gate
      *fails*, which for a decision the accepted Mission 8 constructor would
      build means the state is ``INITIALIZING``, ``RECOVERING``, or
      ``FAIL_SAFE``.  Mission 9 publishes ``None`` instead when the object it
      was handed answers with something that is not a ``SafetyState`` at all.
    * ``INSUFFICIENT_GEOMETRY`` is reached past the gate *and* past the identity
      comparison, so the decision and the estimate agree on the temporal state.
      A permitted Mission 8 state only ever accompanies ``TRACKING`` or
      ``COASTING``, and ``NOMINAL`` narrows that further: Mission 8 calls a
      result nominal only for ``TRACKING`` with a ``DETECTED`` raw state, and
      Mission 7 requires *complete* temporal geometry for exactly that pair, so
      guidance is always available and the geometry branch is unreachable.  It
      is therefore a ``DEGRADED`` refusal over tracking or coasting evidence.
    * ``NOMINAL_TRACKING`` needs a nominal safety state, and Mission 8 is nominal
      only over ``TRACKING``.  ``DEGRADED_TRACKING`` cannot be nominal-safety
      because Mission 7 forbids an extrapolated ``TRACKING`` estimate, so nominal
      safety over tracking always selects ``NOMINAL_TRACKING`` instead.
      ``DEGRADED_COASTING`` needs ``COASTING``, which Mission 8 only ever calls
      degraded.
    * no bound record carries ``UNINITIALIZED``: see ``_BOUND_TEMPORAL_STATES``.
    """
    if reason in _EVIDENCE_FREE_LATERAL_REASONS:
        if bound or safety_state is not None or temporal_state is not None:
            return "a lateral refusal that discards its baseline exposes nothing"
        return None
    if not bound:
        if reason not in _EVIDENCE_REFUSAL_LATERAL_REASONS:
            return "this lateral reason is only ever published with bound evidence"
        if safety_state is not None or temporal_state is not None:
            return "an unbound lateral evidence refusal exposes no upstream state"
        return None
    states, temporals = _BOUND_LATERAL_SHAPES[reason]
    if safety_state not in states:
        return "this lateral reason cannot carry this safety state"
    if temporal_state not in temporals:
        return "this lateral reason cannot carry this temporal state"
    return None

# The Mission 10A mode each actionable envelope mode is built from.
_ENVELOPE_MODE_FOR_LONGITUDINAL: dict[
    LongitudinalControlMode, CommandEnvelopeMode
] = {
    LongitudinalControlMode.NOMINAL: CommandEnvelopeMode.NOMINAL,
    LongitudinalControlMode.DEGRADED: CommandEnvelopeMode.DEGRADED,
    LongitudinalControlMode.STOPPING: CommandEnvelopeMode.STOPPING,
}


def _mode_selection_reason(
    safety_state: SafetyState,
    lateral_mode: LateralControlMode,
    temporal_state: TemporalLaneState,
) -> LongitudinalControlReason:
    """The reason accepted Mission 10A selects for a *permitted* outcome.

    Mission 10A calls a permitted outcome nominal only under a nominal safety
    state with a nominal lateral mode and a tracking estimate, coasting exactly
    when the estimate is coasting, and degraded tracking otherwise.  Mission 10B
    recomputes it rather than trusting the published reason.
    """
    if temporal_state is TemporalLaneState.COASTING:
        return LongitudinalControlReason.DEGRADED_COASTING
    if (
        safety_state is SafetyState.NOMINAL
        and lateral_mode is LateralControlMode.NOMINAL
    ):
        return LongitudinalControlReason.NOMINAL_TRACKING
    return LongitudinalControlReason.DEGRADED_TRACKING


def _longitudinal_state_conflict(
    reason: LongitudinalControlReason,
    safety_state: SafetyState,
    lateral_mode: LateralControlMode,
    temporal_state: TemporalLaneState,
) -> str | None:
    """Why this bound Mission 10A reason cannot carry these states, or ``None``.

    One pure statement of the per-reason upstream relationships, shared by the
    Mission 10A canonicalizer and by the public envelope record, so the two can
    never disagree about which combinations are reachable.
    """
    # Mission 8 forbids an ``INITIALIZING`` decision from exposing any result
    # evidence at all -- no stamp, no revision, no temporal state -- and Mission
    # 10A refuses to bind without all three.  A bound Mission 10A record
    # therefore never carries it, whatever its reason.
    if safety_state is SafetyState.INITIALIZING:
        return "a bound longitudinal record cannot carry an initializing state"
    if safety_state not in _PERMITTED_SAFETY_STATES and (
        lateral_mode is not LateralControlMode.DISABLED
    ):
        return "an unpermitted safety state cannot carry a permitted lateral mode"
    if reason in _GATED_LONGITUDINAL_REASONS:
        if (
            safety_state not in _PERMITTED_SAFETY_STATES
            or lateral_mode not in _PERMITTED_LATERAL_MODES
            or temporal_state not in _GUIDANCE_STATES
        ):
            return "a gated longitudinal outcome requires fully permitted evidence"
    elif reason is LongitudinalControlReason.LATERAL_STOP:
        if (
            safety_state not in _PERMITTED_SAFETY_STATES
            or lateral_mode is not LateralControlMode.DISABLED
        ):
            return "a lateral stop requires permitted safety and a refused lateral axis"
    elif reason is LongitudinalControlReason.SAFETY_STOP:
        if safety_state in _PERMITTED_SAFETY_STATES:
            return "a safety stop requires an unpermitted safety state"
    elif reason is LongitudinalControlReason.FAIL_SAFE_STOP:
        if safety_state is not SafetyState.FAIL_SAFE:
            return "a fail-safe stop requires the FAIL_SAFE state"
    return None


# The Mission 10A outcomes a *permitted* Mission 9 record can accompany on one
# real evidence cut.  Once both permission gates are passed Mission 10A can only
# reach its freshness, signed-motion, and speed-law branches, plus the vertical
# anomaly that outranks every one of them.
_PERMITTED_PAIR_LONGITUDINAL_REASONS = _GATED_LONGITUDINAL_REASONS | frozenset(
    (LongitudinalControlReason.EXCESSIVE_VERTICAL_MOTION,)
)

# The exact *bound* Mission 10A reasons each Mission 9 reason can accompany when
# both records came from one real upstream cut.  Derived directly from the gate
# order in ``LongitudinalController.update`` and from what
# ``_read_evidence`` requires of the Mission 9 record it is handed:
#
# * Mission 10A binds nothing at all unless the Mission 9 record carries a source
#   stamp, a revision, an evaluation time, a safety state, *and* a temporal
#   state, and unless the decision it was given agrees with that record on the
#   stamp, the revision, the safety state, and the temporal state.  Every
#   Mission 9 reason that publishes an incomplete binding therefore has no bound
#   Mission 10A partner whatsoever.
# * ``SOURCE_REJECTED`` publishes bound evidence but no safety state, so
#   ``_read_evidence`` refuses it -- it too has no bound partner.
# * a bound ``IDENTITY_MISMATCH`` or ``EVIDENCE_MALFORMED`` is Mission 9 stating
#   that the decision does *not* describe this evidence.  It compares the stamp,
#   the revision, the temporal state, and the temporal confidence; Mission 10A
#   independently re-compares the first three, so on one cut it must reach the
#   same verdict and refuse.  Confidence is the only axis Mission 10A does not
#   re-read, and it cannot differ on a real cut: an equal snapshot revision means
#   one Mission 6 snapshot, hence one Mission 7 estimate, hence one confidence.
#   These two therefore have no bound partner either -- which is exactly why an
#   impossible pair can never become a trusted stop.
# * ``SAFETY_NOT_PERMITTED`` means the Mission 8 gate failed, so Mission 10A
#   reaches its own latched or unpermitted branch.
# * ``INSUFFICIENT_GEOMETRY`` passed the Mission 8 gate and failed the Mission 9
#   one, which is precisely Mission 10A's lateral-permission branch.
_SAME_CUT_LONGITUDINAL_REASONS: dict[
    LateralControlReason, frozenset[LongitudinalControlReason]
] = {
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
    LateralControlReason.NOMINAL_TRACKING: _PERMITTED_PAIR_LONGITUDINAL_REASONS,
    LateralControlReason.DEGRADED_TRACKING: _PERMITTED_PAIR_LONGITUDINAL_REASONS,
    LateralControlReason.DEGRADED_COASTING: _PERMITTED_PAIR_LONGITUDINAL_REASONS,
}


def _same_cut_conflict(
    lateral_reason: LateralControlReason,
    longitudinal_reason: LongitudinalControlReason,
) -> str | None:
    """Why no single real upstream cut could have produced this pair.

    Validating the two records independently is not enough.  Each one can be a
    perfectly legitimate Mission 9 or Mission 10A output and still describe a
    situation the other one contradicts, because Mission 10A is handed the very
    Mission 9 record Mission 10B is looking at and its own gate order fixes which
    outcome that record can lead to.  This is the pure, finite, source-derived
    statement of that relationship, applied to every *bound* Mission 10A record
    -- the only ones that can carry an identity forward into an actionable
    envelope.

    It proves the pair could genuinely coexist.  It is deliberately not a
    blacklist of known-bad combinations: the table above is an allow-list read
    off the accepted producers, so a combination nobody thought to enumerate is
    refused rather than admitted.
    """
    if longitudinal_reason not in _SAME_CUT_LONGITUDINAL_REASONS[lateral_reason]:
        return "no real evidence cut produces this lateral/longitudinal pair"
    return None


def _actionable_provenance_conflict(
    mode: CommandEnvelopeMode,
    safety_state: SafetyState,
    temporal_state: TemporalLaneState,
    lateral_mode: LateralControlMode,
    lateral_reason: LateralControlReason,
    longitudinal_mode: LongitudinalControlMode,
    longitudinal_reason: LongitudinalControlReason,
) -> str | None:
    """Why no Mission 10B path could publish this provenance, or ``None``.

    An actionable envelope claims that one real Mission 9 record and one real
    Mission 10A record, bound to one evidence cut, produced it.  This proves the
    claim is internally possible: each upstream mode matches its own upstream
    reason, the envelope mode is the Mission 10A mode it was built from, the
    per-reason Mission 9 branch relationships hold for a bound record, the
    per-reason Mission 10A state relationships hold, the pair is one a single
    real upstream cut could have produced, the Mission 10A mode selection
    recomputes to the published reason, and a permitted Mission 9 record agrees
    with the temporal estimate about coasting.
    """
    if _LATERAL_REASON_MODE[lateral_reason] is not lateral_mode:
        return "lateral mode does not match its own lateral reason"
    if _LONGITUDINAL_REASON_MODE[longitudinal_reason] is not longitudinal_mode:
        return "longitudinal mode does not match its own longitudinal reason"
    if _ENVELOPE_MODE_FOR_LONGITUDINAL.get(longitudinal_mode) is not mode:
        return "the envelope mode does not match its longitudinal mode"
    # An actionable envelope is bound, so the Mission 9 record behind it carried
    # a source stamp *and* the safety state the binding proved equal to Mission
    # 10A's.  Every Mission 9 branch that cannot satisfy both is excluded here.
    conflict = _lateral_producer_conflict(
        lateral_reason, safety_state, temporal_state, True
    )
    if conflict is not None:
        return conflict
    conflict = _longitudinal_state_conflict(
        longitudinal_reason, safety_state, lateral_mode, temporal_state
    )
    if conflict is not None:
        return conflict
    # Both records are individually possible; prove they could have arisen
    # together.  Without this an impossible pair -- a Mission 9 record that
    # refused *because the decision did not describe its evidence* alongside a
    # Mission 10A record that bound that same decision to it -- would still
    # publish an actionable trusted stop.
    conflict = _same_cut_conflict(lateral_reason, longitudinal_reason)
    if conflict is not None:
        return conflict
    if longitudinal_mode in _PERMITTED_LONGITUDINAL_MODES:
        expected = _mode_selection_reason(safety_state, lateral_mode, temporal_state)
        if longitudinal_reason is not expected:
            return "the longitudinal reason contradicts its own mode selection"
    if lateral_mode in _PERMITTED_LATERAL_MODES:
        coasting = lateral_reason is LateralControlReason.DEGRADED_COASTING
        if coasting is not (temporal_state is TemporalLaneState.COASTING):
            return "a permitted lateral reason contradicts the temporal estimate"
    return None


class _EvidenceError(Exception):
    """Internal marker for evidence Mission 10B must fail closed on."""


class _MalformedEvidence(_EvidenceError):
    """One upstream record contradicts its own accepted contract."""


# --------------------------------------------------------------------------
# Detached-evidence contract validators
#
# Everything below runs *after* acquisition, on detached built-ins and accepted
# enums, so none of it can reach caller code.  Each validator states one clause
# of the upstream contract and converts a violation into the private malformed
# signal deliberately, at the exact point the caller's value is proven invalid.
# The canonical normalization each one then delegates to is ordinary internal
# work: a failure *there* is a controller defect, and its raw exception is left
# alone so the ``INTERNAL_ERROR`` boundary sees it.  That is why this stage never
# catches a raw ``TypeError`` or ``ValueError``.  Catching one would make an
# internal defect indistinguishable from malformed caller evidence, which is
# exactly the misclassification this boundary exists to prevent.
#
# These validators are also the *complete* statement of both accepted upstream
# request contracts.  Nothing here asks an upstream constructor whether the
# caller's evidence is acceptable, because such a call runs accepted upstream
# code whose own defect would then be answered as bad caller evidence.  Every
# clause is instead restated locally over detached built-ins, so a rejection is
# always a deliberate ``_MalformedEvidence`` raised at a named invariant and
# never an absorbed raw exception.  The duplication this costs is bounded, and
# it is pinned: Mission 10B's tests compare this local judgement against the
# pristine accepted constructors over a differential matrix, so any upstream
# contract drift fails a test rather than silently weakening the boundary.
# --------------------------------------------------------------------------


def _evidence_bool(value: object, name: str) -> bool:
    """Prove one detached shaping flag is an exact built-in ``bool``."""
    if type(value) is not bool:
        raise _MalformedEvidence(f"{name} must be a built-in bool")
    return _builtin_bool(value, name)


def _evidence_index(value: object, name: str) -> int:
    """Prove one detached revision or epoch is a non-negative built-in ``int``."""
    if type(value) is not int or value < 0:
        raise _MalformedEvidence(f"{name} must be a non-negative built-in integer")
    return _nonnegative_builtin_int(value, name)


def _evidence_real(value: object, name: str) -> float:
    """Prove one detached number is finite, then normalize it canonically.

    Acquisition applied the caller-facing numeric contract and performed the one
    ``float`` conversion, so ``value`` is always an exact built-in ``float`` by
    the time it arrives here and the finiteness test consults nothing the caller
    owns.  The exact-type gate also keeps a huge caller integer from reaching
    ``math.isfinite`` and raising ``OverflowError``; acquisition has already
    rejected those as unreadable evidence.
    """
    if type(value) is not float or not math.isfinite(value):
        raise _MalformedEvidence(f"{name} must be a finite real number")
    return _finite_real(value, name)


def _evidence_bounded(value: object, name: str, *, low: float, high: float) -> float:
    """Prove one detached number is finite and inside its own accepted range."""
    result = _evidence_real(value, name)
    if not low <= result <= high:
        raise _MalformedEvidence(f"{name} must be in [{low}, {high}]")
    return result


def _evidence_stamp(value: object, name: str) -> SampleStamp:
    """Prove one acquired stamp's members, then rebuild it canonically."""
    if not (
        type(value) is tuple and len(value) == 4 and value[0] is _ACQUIRED_STAMP
    ):
        raise _MalformedEvidence(f"{name} must be a SampleStamp")
    _, frame, sim_time, monotonic = value
    _evidence_index(frame, f"{name}.frame")
    _evidence_real(sim_time, f"{name}.sim_time")
    _evidence_real(monotonic, f"{name}.monotonic")
    return _canonical_stamp(value, name)


# The accepted request classes are reached through exactly one module name each,
# and for exactly one purpose: building the detached record Mission 10B will
# actually use, once, from values Mission 10B has *already* proven acceptable on
# its own.
#
# They are deliberately **not** used to judge caller evidence.  A constructor
# call is not a classifier: it runs accepted upstream code, which calls mutable
# upstream helpers, and a genuine defect anywhere inside that machinery raises
# the same ``TypeError`` or ``ValueError`` a contract rejection does.  Catching
# those around a constructor would therefore report an upstream controller
# defect as untrustworthy caller evidence -- publishing a malformed refusal,
# leaving ``internal_errors`` unmoved, and swallowing the original exception
# instead of re-raising it.  Every contract clause is restated over detached
# built-ins above precisely so that no such catch is needed.
#
# By the time either name below is called, the detached values have satisfied
# the whole accepted contract locally, so a failure here cannot be a contract
# rejection.  It can only be a defect -- in the constructor, in ``__post_init__``,
# or in an upstream helper either of them depends on -- and its exception is left
# entirely alone so the ``INTERNAL_ERROR`` boundary sees it and re-raises it
# unchanged.  This call sits outside every malformed-evidence catch.
_LATERAL_RECONSTRUCTOR = LateralControlRequest
_LONGITUDINAL_RECONSTRUCTOR = LongitudinalControlRequest


def _detach_lateral(lateral: LateralControlRequest) -> dict[str, object]:
    """Acquire every Mission 9 public field exactly once.

    This is the complete caller-code boundary: public attributes and nested stamp
    members are read once, and caller-owned real numbers are converted once.  It
    performs no finiteness, range, enum, shape, or constructor validation.
    """
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

    return dict(
        mode=mode,
        reason=reason,
        lateral_allowed=lateral_allowed,
        steering=_acquire_real(steering, "lateral.steering"),
        raw_steering=_acquire_real(raw_steering, "lateral.raw_steering"),
        steering_authority=_acquire_real(
            steering_authority, "lateral.steering_authority"
        ),
        clamped=clamped,
        rate_limited=rate_limited,
        evaluation_monotonic=(
            None
            if evaluation_monotonic is None
            else _acquire_real(evaluation_monotonic, "lateral.evaluation_monotonic")
        ),
        source_stamp=(
            None
            if source_stamp is None
            else _acquire_stamp(source_stamp, "lateral.source_stamp")
        ),
        snapshot_revision=snapshot_revision,
        lateral_error=(
            None
            if lateral_error is None
            else _acquire_real(lateral_error, "lateral.lateral_error")
        ),
        heading_error=(
            None
            if heading_error is None
            else _acquire_real(heading_error, "lateral.heading_error")
        ),
        safety_state=safety_state,
        temporal_state=temporal_state,
    )


def _validate_detached_lateral(acquired: dict[str, object]) -> dict[str, object]:
    """Validate and normalize caller-free Mission 9 values internally.

    This is the per-field half of the complete accepted Mission 9 contract: the
    exact types, the reason/mode map, the permission/mode agreement, and every
    numeric range the accepted constructor enforces.  The relationships *between*
    fields are proven immediately afterwards by
    :func:`_prove_detached_lateral_contract`.

    Every clause below is a statement about *caller* evidence and fails with the
    private malformed signal.  Nothing here catches a raw exception, so a defect
    in one of the internal helpers this delegates to reaches the internal-error
    boundary instead of being reported as bad caller evidence.
    """
    mode = acquired["mode"]
    reason = acquired["reason"]
    safety_state = acquired["safety_state"]
    temporal_state = acquired["temporal_state"]
    if type(mode) is not LateralControlMode:
        raise _MalformedEvidence("lateral mode must be a LateralControlMode")
    if type(reason) is not LateralControlReason:
        raise _MalformedEvidence("lateral reason must be a LateralControlReason")
    if safety_state is not None and type(safety_state) is not SafetyState:
        raise _MalformedEvidence("lateral safety state must be a SafetyState")
    if temporal_state is not None and type(temporal_state) is not TemporalLaneState:
        raise _MalformedEvidence("lateral temporal state must be a TemporalLaneState")
    # The accepted Mission 9 reason/mode contract, and the permission that
    # follows from the mode alone.  Both are settled before any number is looked
    # at, exactly as the accepted constructor settles them.
    if _LATERAL_REASON_MODE[reason] is not mode:
        raise _MalformedEvidence("lateral mode does not match its own reason")
    allowed = _evidence_bool(acquired["lateral_allowed"], "lateral.lateral_allowed")
    if allowed is not (mode in _PERMITTED_LATERAL_MODES):
        raise _MalformedEvidence("lateral permission does not match its own mode")

    source_stamp = acquired["source_stamp"]
    snapshot_revision = acquired["snapshot_revision"]
    lateral_error = acquired["lateral_error"]
    heading_error = acquired["heading_error"]
    evaluated = acquired["evaluation_monotonic"]
    return dict(
        mode=mode,
        reason=reason,
        lateral_allowed=allowed,
        steering=_evidence_bounded(
            acquired["steering"], "lateral.steering", low=-1.0, high=1.0
        ),
        raw_steering=_evidence_real(
            acquired["raw_steering"], "lateral.raw_steering"
        ),
        steering_authority=_evidence_bounded(
            acquired["steering_authority"],
            "lateral.steering_authority",
            low=0.0,
            high=1.0,
        ),
        clamped=_evidence_bool(acquired["clamped"], "lateral.clamped"),
        rate_limited=_evidence_bool(
            acquired["rate_limited"], "lateral.rate_limited"
        ),
        evaluation_monotonic=(
            None
            if evaluated is None
            else _evidence_real(evaluated, "lateral.evaluation_monotonic")
        ),
        source_stamp=(
            None
            if source_stamp is None
            else _evidence_stamp(source_stamp, "lateral.source_stamp")
        ),
        snapshot_revision=(
            None
            if snapshot_revision is None
            else _evidence_index(
                snapshot_revision, "lateral.snapshot_revision"
            )
        ),
        lateral_error=(
            None
            if lateral_error is None
            else _evidence_bounded(
                lateral_error, "lateral.lateral_error", low=-1.0, high=1.0
            )
        ),
        heading_error=(
            None
            if heading_error is None
            else _evidence_bounded(
                heading_error, "lateral.heading_error", low=-1.0, high=1.0
            )
        ),
        safety_state=safety_state,
        temporal_state=temporal_state,
    )


def _prove_detached_lateral_contract(detached: dict[str, object]) -> None:
    """Prove the accepted Mission 9 relationships *between* detached fields.

    This is the second half of the complete accepted Mission 9 contract, stated
    over values that are already exact built-ins and accepted enums.  Together
    with :func:`_validate_detached_lateral` it covers every clause the accepted
    ``LateralControlRequest.__post_init__`` enforces, so the reconstruction that
    follows can only fail through a defect.

    Every clause here is a statement about *caller* evidence and is signalled
    deliberately.  No raw exception is caught, and no upstream constructor or
    upstream helper is consulted.
    """
    allowed = detached["lateral_allowed"]
    mode = detached["mode"]
    reason = detached["reason"]
    steering = detached["steering"]
    raw_steering = detached["raw_steering"]
    authority = detached["steering_authority"]
    evaluated = detached["evaluation_monotonic"]
    stamp = detached["source_stamp"]
    revision = detached["snapshot_revision"]
    lateral_error = detached["lateral_error"]
    heading_error = detached["heading_error"]
    safety_state = detached["safety_state"]
    temporal_state = detached["temporal_state"]

    if allowed:
        # A permitted Mission 9 request steers strictly inside the authority it
        # published for itself, and states the whole evidence cut behind it.
        if abs(steering) > authority:
            raise _MalformedEvidence("lateral steering cannot exceed its own authority")
        if (
            evaluated is None
            or stamp is None
            or revision is None
            or lateral_error is None
            or heading_error is None
        ):
            raise _MalformedEvidence("a permitted lateral request requires complete evidence")
        if safety_state not in _PERMITTED_SAFETY_STATES:
            raise _MalformedEvidence(
                "a permitted lateral request requires a permitted safety state"
            )
        if temporal_state not in _GUIDANCE_STATES:
            raise _MalformedEvidence(
                "a permitted lateral request requires a guidance-bearing state"
            )
        if (
            mode is LateralControlMode.NOMINAL
            and safety_state is not SafetyState.NOMINAL
        ):
            raise _MalformedEvidence(
                "nominal lateral mode requires a nominal safety state"
            )
    else:
        # A refused Mission 9 request is exactly neutral: it steers nothing,
        # claims no authority, reports no shaping, and exposes no lane guidance.
        if steering != 0.0 or raw_steering != 0.0 or authority != 0.0:
            raise _MalformedEvidence(
                "a refused lateral request cannot carry steering evidence"
            )
        if detached["clamped"] or detached["rate_limited"]:
            raise _MalformedEvidence(
                "a refused lateral request cannot report shaping activity"
            )
        if lateral_error is not None or heading_error is not None:
            raise _MalformedEvidence(
                "a refused lateral request cannot expose lane guidance"
            )

    if reason is LateralControlReason.NOT_EVALUATED and any(
        value is not None
        for value in (evaluated, stamp, revision, safety_state, temporal_state)
    ):
        raise _MalformedEvidence(
            "an unevaluated lateral request cannot expose any evidence"
        )


def _prove_lateral_producer_shape(lateral: LateralControlRequest) -> None:
    """Prove the Mission 9 relationships its own constructor does not encode."""
    reason = lateral.reason
    stamp = lateral.source_stamp
    revision = lateral.snapshot_revision
    temporal_state = lateral.temporal_state
    safety_state = lateral.safety_state
    evaluated = lateral.evaluation_monotonic

    # Mission 9 publishes stamp, revision, and temporal state from one accepted
    # evidence record, so a record carrying only some of them never came from it.
    bound = stamp is not None
    if (revision is None) is bound or (temporal_state is None) is bound:
        raise _MalformedEvidence("lateral source evidence must be complete or absent")
    if safety_state is not None and not bound:
        raise _MalformedEvidence("a lateral safety state requires source evidence")

    # The evaluation time is present for every branch except the two that
    # discard the whole baseline, which publish nothing at all.
    if reason in _EVIDENCE_FREE_LATERAL_REASONS:
        if evaluated is not None:
            raise _MalformedEvidence(
                "a lateral refusal that discards its baseline states no time"
            )
    elif evaluated is None:
        raise _MalformedEvidence("an evaluated lateral request requires its time")

    # Every per-reason branch relationship, from the one shared statement the
    # public envelope record uses too.  This settles the safety *and* temporal
    # state together, so the permitted reasons need no separate estimate check.
    conflict = _lateral_producer_conflict(reason, safety_state, temporal_state, bound)
    if conflict is not None:
        raise _MalformedEvidence(conflict)

    if not lateral.lateral_allowed:
        return
    # A permitted Mission 9 request always carries a strictly positive authority
    # taken from its own validated configuration.
    if not lateral.steering_authority > 0.0:
        raise _MalformedEvidence("a permitted lateral request requires authority")


def _reconstruct_lateral(detached: dict[str, object]) -> LateralControlRequest:
    """Build the detached Mission 9 record Mission 10B will use, exactly once.

    Internal work only.  Mission 10B's own local validation has already proven
    these exact values against the whole accepted Mission 9 contract, so nothing
    this raises can be a statement about caller evidence: a failure here, in
    ``__post_init__``, or in any upstream helper either depends on is a defect.
    No exception is caught, and this call sits outside every malformed-evidence
    boundary, so that defect reaches ``INTERNAL_ERROR`` and is re-raised.
    """
    return _LATERAL_RECONSTRUCTOR(**detached)


def _canonicalize_lateral_request(
    lateral: LateralControlRequest,
) -> LateralControlRequest:
    """Rebuild one exact, detached :class:`LateralControlRequest`.

    Every caller-controlled attribute is read exactly once, detached, proven
    against the whole accepted Mission 9 contract *locally*, and only then
    reconstructed once.  A malformed field invalidates the record even when
    Mission 10B performs no arithmetic on it.  The producer relationships the
    accepted constructor does not encode are proven last.

    The stages are deliberately distinct, because only the first three make any
    statement about the caller.  Acquisition and the two detached-contract
    proofs fail closed with the private malformed signal; the single
    reconstruction and the producer proof are internal, and an unexpected
    exception from either -- including one raised inside the accepted upstream
    constructor or an upstream helper it depends on -- is left to reach the
    ``INTERNAL_ERROR`` boundary rather than being disguised as bad evidence.
    """
    if not isinstance(lateral, LateralControlRequest):
        raise _MalformedEvidence("lateral must be a LateralControlRequest")
    try:
        acquired = _detach_lateral(lateral)
    except Exception as exc:  # hostile attribute, property, or numeric
        raise _MalformedEvidence("lateral evidence could not be read") from exc
    detached = _validate_detached_lateral(acquired)
    _prove_detached_lateral_contract(detached)
    canonical = _reconstruct_lateral(detached)
    _prove_lateral_producer_shape(canonical)
    return canonical


def _detach_longitudinal(
    longitudinal: LongitudinalControlRequest,
) -> dict[str, object]:
    """Acquire every Mission 10A public field and nested stamp member once."""
    mode = longitudinal.mode
    reason = longitudinal.reason
    longitudinal_allowed = longitudinal.longitudinal_allowed
    stop_required = longitudinal.stop_required
    throttle_request = longitudinal.throttle_request
    brake_request = longitudinal.brake_request
    effective_target = longitudinal.effective_target_speed_mps
    observed_forward = longitudinal.observed_forward_speed_mps
    speed_error = longitudinal.speed_error_mps
    speed_authority = longitudinal.speed_authority_mps
    throttle_authority = longitudinal.throttle_authority
    brake_authority = longitudinal.brake_authority
    throttle_clamped = longitudinal.throttle_clamped
    brake_clamped = longitudinal.brake_clamped
    evaluation_monotonic = longitudinal.evaluation_monotonic
    guidance_stamp = longitudinal.guidance_stamp
    speed_stamp = longitudinal.speed_stamp
    snapshot_revision = longitudinal.snapshot_revision
    target_origin = longitudinal.target_origin
    target_epoch = longitudinal.target_epoch
    target_revision = longitudinal.target_revision
    target_issued = longitudinal.target_issued_monotonic
    target_valid_until = longitudinal.target_valid_until_monotonic
    safety_state = longitudinal.safety_state
    lateral_mode = longitudinal.lateral_mode
    temporal_state = longitudinal.temporal_state

    return dict(
        mode=mode,
        reason=reason,
        longitudinal_allowed=longitudinal_allowed,
        stop_required=stop_required,
        throttle_request=_acquire_real(
            throttle_request, "longitudinal.throttle_request"
        ),
        brake_request=_acquire_real(brake_request, "longitudinal.brake_request"),
        effective_target_speed_mps=(
            None
            if effective_target is None
            else _acquire_real(
                effective_target, "longitudinal.effective_target_speed_mps"
            )
        ),
        observed_forward_speed_mps=(
            None
            if observed_forward is None
            else _acquire_real(
                observed_forward, "longitudinal.observed_forward_speed_mps"
            )
        ),
        speed_error_mps=(
            None
            if speed_error is None
            else _acquire_real(speed_error, "longitudinal.speed_error_mps")
        ),
        speed_authority_mps=(
            None
            if speed_authority is None
            else _acquire_real(speed_authority, "longitudinal.speed_authority_mps")
        ),
        throttle_authority=_acquire_real(
            throttle_authority, "longitudinal.throttle_authority"
        ),
        brake_authority=_acquire_real(
            brake_authority, "longitudinal.brake_authority"
        ),
        throttle_clamped=throttle_clamped,
        brake_clamped=brake_clamped,
        evaluation_monotonic=(
            None
            if evaluation_monotonic is None
            else _acquire_real(
                evaluation_monotonic, "longitudinal.evaluation_monotonic"
            )
        ),
        guidance_stamp=(
            None
            if guidance_stamp is None
            else _acquire_stamp(guidance_stamp, "longitudinal.guidance_stamp")
        ),
        speed_stamp=(
            None
            if speed_stamp is None
            else _acquire_stamp(speed_stamp, "longitudinal.speed_stamp")
        ),
        snapshot_revision=snapshot_revision,
        target_origin=target_origin,
        target_epoch=target_epoch,
        target_revision=target_revision,
        target_issued_monotonic=(
            None
            if target_issued is None
            else _acquire_real(
                target_issued, "longitudinal.target_issued_monotonic"
            )
        ),
        target_valid_until_monotonic=(
            None
            if target_valid_until is None
            else _acquire_real(
                target_valid_until, "longitudinal.target_valid_until_monotonic"
            )
        ),
        safety_state=safety_state,
        lateral_mode=lateral_mode,
        temporal_state=temporal_state,
    )


def _validate_detached_longitudinal(
    acquired: dict[str, object],
) -> dict[str, object]:
    """Validate and normalize caller-free Mission 10A values internally.

    The per-field half of the complete accepted Mission 10A contract: the exact
    types, the reason/mode map, the permission/mode and ``stop_required``/mode
    agreements, and every numeric range and ceiling the accepted constructor
    enforces.  The relationships *between* fields are proven immediately
    afterwards by :func:`_prove_detached_longitudinal_contract`.

    As with the Mission 9 record, every clause is a statement about *caller*
    evidence and fails with the private malformed signal, and nothing here
    catches a raw exception, so an internal helper defect reaches the
    internal-error boundary instead of being reported as bad caller evidence.
    """
    mode = acquired["mode"]
    reason = acquired["reason"]
    target_origin = acquired["target_origin"]
    safety_state = acquired["safety_state"]
    lateral_mode = acquired["lateral_mode"]
    temporal_state = acquired["temporal_state"]
    if type(mode) is not LongitudinalControlMode:
        raise _MalformedEvidence(
            "longitudinal mode must be a LongitudinalControlMode"
        )
    if type(reason) is not LongitudinalControlReason:
        raise _MalformedEvidence(
            "longitudinal reason must be a LongitudinalControlReason"
        )
    if target_origin is not None and type(target_origin) is not TargetSpeedOrigin:
        raise _MalformedEvidence(
            "longitudinal target origin must be a TargetSpeedOrigin"
        )
    if safety_state is not None and type(safety_state) is not SafetyState:
        raise _MalformedEvidence("longitudinal safety state must be a SafetyState")
    if lateral_mode is not None and type(lateral_mode) is not LateralControlMode:
        raise _MalformedEvidence(
            "longitudinal lateral mode must be a LateralControlMode"
        )
    if temporal_state is not None and type(temporal_state) is not TemporalLaneState:
        raise _MalformedEvidence(
            "longitudinal temporal state must be a TemporalLaneState"
        )
    # The accepted Mission 10A reason/mode contract, and the two dispositions
    # that follow from the mode alone.  All three are settled before any number
    # is looked at, exactly as the accepted constructor settles them.
    if _LONGITUDINAL_REASON_MODE[reason] is not mode:
        raise _MalformedEvidence("longitudinal mode does not match its own reason")
    allowed = _evidence_bool(
        acquired["longitudinal_allowed"], "longitudinal.longitudinal_allowed"
    )
    if allowed is not (mode in _PERMITTED_LONGITUDINAL_MODES):
        raise _MalformedEvidence(
            "longitudinal permission does not match its own mode"
        )
    stop_required = _evidence_bool(
        acquired["stop_required"], "longitudinal.stop_required"
    )
    if stop_required is not (mode is LongitudinalControlMode.STOPPING):
        raise _MalformedEvidence(
            "longitudinal stop requirement does not match its own mode"
        )

    effective_target = acquired["effective_target_speed_mps"]
    observed_forward = acquired["observed_forward_speed_mps"]
    speed_error = acquired["speed_error_mps"]
    speed_authority = acquired["speed_authority_mps"]
    evaluated = acquired["evaluation_monotonic"]
    guidance_stamp = acquired["guidance_stamp"]
    speed_stamp = acquired["speed_stamp"]
    snapshot_revision = acquired["snapshot_revision"]
    target_epoch = acquired["target_epoch"]
    target_revision = acquired["target_revision"]
    target_issued = acquired["target_issued_monotonic"]
    target_valid_until = acquired["target_valid_until_monotonic"]
    return dict(
        mode=mode,
        reason=reason,
        longitudinal_allowed=allowed,
        stop_required=stop_required,
        throttle_request=_evidence_bounded(
            acquired["throttle_request"],
            "longitudinal.throttle_request",
            low=0.0,
            high=1.0,
        ),
        brake_request=_evidence_bounded(
            acquired["brake_request"],
            "longitudinal.brake_request",
            low=0.0,
            high=1.0,
        ),
        # Non-negative *and* inside the accepted Mission 10A hard ceiling, which
        # is the pair of clauses its constructor states separately.
        effective_target_speed_mps=(
            None
            if effective_target is None
            else _evidence_bounded(
                effective_target,
                "longitudinal.effective_target_speed_mps",
                low=0.0,
                high=HARD_TARGET_CEILING_MPS,
            )
        ),
        observed_forward_speed_mps=(
            None
            if observed_forward is None
            else _evidence_real(
                observed_forward, "longitudinal.observed_forward_speed_mps"
            )
        ),
        speed_error_mps=(
            None
            if speed_error is None
            else _evidence_real(speed_error, "longitudinal.speed_error_mps")
        ),
        speed_authority_mps=(
            None
            if speed_authority is None
            else _evidence_bounded(
                speed_authority,
                "longitudinal.speed_authority_mps",
                low=0.0,
                high=HARD_TARGET_CEILING_MPS,
            )
        ),
        throttle_authority=_evidence_bounded(
            acquired["throttle_authority"],
            "longitudinal.throttle_authority",
            low=0.0,
            high=1.0,
        ),
        brake_authority=_evidence_bounded(
            acquired["brake_authority"],
            "longitudinal.brake_authority",
            low=0.0,
            high=1.0,
        ),
        throttle_clamped=_evidence_bool(
            acquired["throttle_clamped"], "longitudinal.throttle_clamped"
        ),
        brake_clamped=_evidence_bool(
            acquired["brake_clamped"], "longitudinal.brake_clamped"
        ),
        evaluation_monotonic=(
            None
            if evaluated is None
            else _evidence_real(evaluated, "longitudinal.evaluation_monotonic")
        ),
        guidance_stamp=(
            None
            if guidance_stamp is None
            else _evidence_stamp(guidance_stamp, "longitudinal.guidance_stamp")
        ),
        speed_stamp=(
            None
            if speed_stamp is None
            else _evidence_stamp(speed_stamp, "longitudinal.speed_stamp")
        ),
        snapshot_revision=(
            None
            if snapshot_revision is None
            else _evidence_index(
                snapshot_revision, "longitudinal.snapshot_revision"
            )
        ),
        target_origin=target_origin,
        target_epoch=(
            None
            if target_epoch is None
            else _evidence_index(
                target_epoch, "longitudinal.target_epoch"
            )
        ),
        target_revision=(
            None
            if target_revision is None
            else _evidence_index(
                target_revision, "longitudinal.target_revision"
            )
        ),
        target_issued_monotonic=(
            None
            if target_issued is None
            else _evidence_real(
                target_issued, "longitudinal.target_issued_monotonic"
            )
        ),
        target_valid_until_monotonic=(
            None
            if target_valid_until is None
            else _evidence_real(
                target_valid_until,
                "longitudinal.target_valid_until_monotonic",
            )
        ),
        safety_state=safety_state,
        lateral_mode=lateral_mode,
        temporal_state=temporal_state,
    )


def _prove_longitudinal_producer_shape(
    longitudinal: LongitudinalControlRequest,
) -> None:
    """Prove the Mission 10A relationships its own constructor does not encode."""
    reason = longitudinal.reason
    bound = longitudinal.guidance_stamp is not None
    if bound is (reason in _UNBOUND_LONGITUDINAL_REASONS):
        raise _MalformedEvidence(
            "longitudinal identity binding contradicts its own reason"
        )

    # The optional speed-law evidence is settled first and exactly: each accepted
    # Mission 10A branch emits one specific presence pattern, and every other
    # combination -- including every *partial* one -- is a shape no producer
    # emits.  Settling it here means a malformed record can never survive
    # canonicalization and surface later as an output-construction failure.
    speed_law_shape = (
        longitudinal.effective_target_speed_mps is not None,
        longitudinal.observed_forward_speed_mps is not None,
        longitudinal.speed_error_mps is not None,
        longitudinal.speed_authority_mps is not None,
    )
    if speed_law_shape != _SPEED_LAW_SHAPE[reason]:
        raise _MalformedEvidence(
            "longitudinal speed-law evidence is not a shape Mission 10A publishes"
        )

    binding = (
        longitudinal.speed_stamp,
        longitudinal.snapshot_revision,
        longitudinal.target_origin,
        longitudinal.target_epoch,
        longitudinal.target_revision,
        longitudinal.target_issued_monotonic,
        longitudinal.target_valid_until_monotonic,
        longitudinal.safety_state,
        longitudinal.lateral_mode,
        longitudinal.temporal_state,
    )
    evaluated = longitudinal.evaluation_monotonic
    if bound:
        if evaluated is None or any(value is None for value in binding):
            raise _MalformedEvidence(
                "a bound longitudinal request requires its complete binding"
            )
    else:
        if any(value is not None for value in binding):
            raise _MalformedEvidence(
                "an unbound longitudinal request cannot expose any binding"
            )
        # Mission 10A publishes an evaluated refusal with the control time it
        # was refused at.  The exact per-reason rule, read off the accepted
        # source, is in `_UNBOUND_TIME_SHAPE`.
        allowed_times = _UNBOUND_TIME_SHAPE[reason]
        if (evaluated is not None) not in allowed_times:
            raise _MalformedEvidence(
                "a longitudinal refusal states its time exactly when its own "
                "branch does"
            )
        return

    conflict = _longitudinal_state_conflict(
        reason,
        longitudinal.safety_state,
        longitudinal.lateral_mode,
        longitudinal.temporal_state,
    )
    if conflict is not None:
        raise _MalformedEvidence(conflict)

    # The complete-law arithmetic is proven only for the branches whose contract
    # actually requires the complete law; the branches that publish an observed
    # speed alone have nothing to reconcile and nothing is invented for them.
    if reason in _SPEED_LAW_LONGITUDINAL_REASONS:
        effective_target = longitudinal.effective_target_speed_mps
        speed_error = longitudinal.speed_error_mps
        speed_authority = longitudinal.speed_authority_mps
        observed = longitudinal.observed_forward_speed_mps
        if effective_target > speed_authority:
            raise _MalformedEvidence(
                "an effective target cannot exceed its own speed authority"
            )
        # The Mission 10A law is exactly `error = effective_target - observed`,
        # evaluated once in IEEE-754 double precision on these very values.
        if speed_error != (effective_target - observed) + 0.0:
            raise _MalformedEvidence("longitudinal speed error contradicts its law")
        if (reason is LongitudinalControlReason.TARGET_STOP) is not (
            effective_target == 0.0
        ):
            raise _MalformedEvidence(
                "a target stop is exactly a zero effective target and nothing else"
            )

    if longitudinal.longitudinal_allowed and not (
        longitudinal.throttle_authority > 0.0
        and longitudinal.brake_authority > 0.0
    ):
        raise _MalformedEvidence(
            "a permitted longitudinal request requires both pedal authorities"
        )


def _prove_detached_longitudinal_contract(detached: dict[str, object]) -> None:
    """Prove the accepted Mission 10A relationships *between* detached fields.

    The second half of the complete accepted Mission 10A contract, stated over
    values that are already exact built-ins and accepted enums: the
    never-both-pedals rule, each pedal inside its own authority, the disabled and
    stopping pedal rules, the target issue/expiry order, the identity a
    propelling or stopping request requires, the completeness an evaluated speed
    law requires, the nominal-authority rule, the clamping a refusal may report,
    and the neutrality an unevaluated record requires.  Together with
    :func:`_validate_detached_longitudinal` it covers every clause the accepted
    ``LongitudinalControlRequest.__post_init__`` enforces.

    Every clause here is a statement about *caller* evidence and is signalled
    deliberately.  No raw exception is caught, and no upstream constructor or
    upstream helper is consulted.
    """
    allowed = detached["longitudinal_allowed"]
    stop_required = detached["stop_required"]
    mode = detached["mode"]
    reason = detached["reason"]
    throttle = detached["throttle_request"]
    brake = detached["brake_request"]
    throttle_authority = detached["throttle_authority"]
    brake_authority = detached["brake_authority"]
    effective_target = detached["effective_target_speed_mps"]
    observed_forward = detached["observed_forward_speed_mps"]
    speed_error = detached["speed_error_mps"]
    speed_authority = detached["speed_authority_mps"]
    safety_state = detached["safety_state"]
    lateral_mode = detached["lateral_mode"]
    temporal_state = detached["temporal_state"]
    target_issued = detached["target_issued_monotonic"]
    target_valid_until = detached["target_valid_until_monotonic"]

    # Pedal coherence.  Mission 10A never asks for both at once, never asks for
    # more of either than it granted itself, and never propels without
    # permission.
    if throttle > 0.0 and brake > 0.0:
        raise _MalformedEvidence(
            "longitudinal throttle and brake can never both be requested"
        )
    if throttle > throttle_authority:
        raise _MalformedEvidence("longitudinal throttle cannot exceed its own authority")
    if brake > brake_authority:
        raise _MalformedEvidence("longitudinal brake cannot exceed its own authority")
    if not allowed and throttle != 0.0:
        raise _MalformedEvidence(
            "a longitudinal request without propulsion permission cannot throttle"
        )
    if mode is LongitudinalControlMode.DISABLED and (
        throttle != 0.0
        or brake != 0.0
        or throttle_authority != 0.0
        or brake_authority != 0.0
    ):
        raise _MalformedEvidence(
            "a disabled longitudinal request cannot carry any pedal authority"
        )
    if mode is LongitudinalControlMode.STOPPING and (
        throttle != 0.0 or throttle_authority != 0.0
    ):
        raise _MalformedEvidence(
            "a stopping longitudinal request cannot carry throttle authority"
        )

    if (
        target_issued is not None
        and target_valid_until is not None
        and target_issued > target_valid_until
    ):
        raise _MalformedEvidence(
            "a longitudinal target issue time cannot follow its expiry"
        )

    # The twelve-field identity binding, exactly as the accepted constructor
    # groups it.
    identity = (
        detached["evaluation_monotonic"],
        detached["guidance_stamp"],
        detached["speed_stamp"],
        detached["snapshot_revision"],
        detached["target_origin"],
        detached["target_epoch"],
        detached["target_revision"],
        target_issued,
        target_valid_until,
        safety_state,
        lateral_mode,
        temporal_state,
    )
    if stop_required and brake <= 0.0:
        raise _MalformedEvidence(
            "a stopping longitudinal request must request positive brake"
        )
    if (allowed or brake > 0.0 or stop_required) and any(
        value is None for value in identity
    ):
        raise _MalformedEvidence(
            "a propelling or stopping longitudinal request requires complete identity"
        )

    if allowed or reason is LongitudinalControlReason.TARGET_STOP:
        if any(
            value is None
            for value in (
                effective_target,
                observed_forward,
                speed_error,
                speed_authority,
            )
        ):
            raise _MalformedEvidence(
                "an evaluated longitudinal control law requires complete evidence"
            )
        if safety_state not in _PERMITTED_SAFETY_STATES:
            raise _MalformedEvidence(
                "an evaluated longitudinal law requires a permitted safety state"
            )
        if lateral_mode not in _PERMITTED_LATERAL_MODES:
            raise _MalformedEvidence(
                "an evaluated longitudinal law requires a permitted lateral mode"
            )
        if temporal_state not in _GUIDANCE_STATES:
            raise _MalformedEvidence(
                "an evaluated longitudinal law requires a guidance-bearing state"
            )

    if allowed:
        if mode is LongitudinalControlMode.NOMINAL and (
            safety_state is not SafetyState.NOMINAL
            or lateral_mode is not LateralControlMode.NOMINAL
            or temporal_state is not TemporalLaneState.TRACKING
        ):
            raise _MalformedEvidence(
                "nominal longitudinal authority requires fully nominal evidence"
            )
        if (
            effective_target is not None
            and speed_authority is not None
            and effective_target > speed_authority
        ):
            raise _MalformedEvidence(
                "a longitudinal effective target cannot exceed its speed authority"
            )
    else:
        if detached["throttle_clamped"]:
            raise _MalformedEvidence(
                "a refused longitudinal request cannot report throttle clamping"
            )
        if brake == 0.0 and detached["brake_clamped"]:
            raise _MalformedEvidence(
                "a longitudinal request with no brake cannot report brake clamping"
            )

    if reason is LongitudinalControlReason.NOT_EVALUATED and any(
        value is not None
        for value in (
            *identity,
            effective_target,
            observed_forward,
            speed_error,
            speed_authority,
        )
    ):
        raise _MalformedEvidence(
            "an unevaluated longitudinal request cannot expose any evidence"
        )


def _reconstruct_longitudinal(
    detached: dict[str, object],
) -> LongitudinalControlRequest:
    """Build the detached Mission 10A record Mission 10B will use, exactly once.

    Internal work only, on values Mission 10B's own local validation has already
    proven against the whole accepted Mission 10A contract.  A failure here, in
    ``__post_init__``, or in any upstream helper either depends on is a defect,
    so no exception is caught and this call sits outside every
    malformed-evidence boundary.
    """
    return _LONGITUDINAL_RECONSTRUCTOR(**detached)


def _canonicalize_longitudinal_request(
    longitudinal: LongitudinalControlRequest,
) -> LongitudinalControlRequest:
    """Rebuild one exact, detached :class:`LongitudinalControlRequest`.

    The same stages as the Mission 9 record, with the same boundary: acquisition
    and the two detached-contract proofs are the only ones that speak about the
    caller and fail closed with the private malformed signal, while the single
    reconstruction and the producer proof are internal and let an unexpected
    exception -- including one from the accepted upstream constructor or a
    helper it depends on -- reach the ``INTERNAL_ERROR`` boundary.
    """
    if not isinstance(longitudinal, LongitudinalControlRequest):
        raise _MalformedEvidence(
            "longitudinal must be a LongitudinalControlRequest"
        )
    try:
        acquired = _detach_longitudinal(longitudinal)
    except Exception as exc:  # hostile attribute, property, or numeric
        raise _MalformedEvidence("longitudinal evidence could not be read") from exc
    detached = _validate_detached_longitudinal(acquired)
    _prove_detached_longitudinal_contract(detached)
    canonical = _reconstruct_longitudinal(detached)
    _prove_longitudinal_producer_shape(canonical)
    return canonical


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CommandEnvelopeConfig:
    """One small, exact command-envelope policy for one controller epoch.

    Every value is a deliberately conservative offline exhibition placeholder,
    not a tuned or measured vehicle parameter, and nothing here has been
    validated against a real vehicle, a real road, or a running simulator.

    * ``maximum_steering``, ``maximum_throttle``, and ``maximum_ordinary_brake``
      are final normalized ceilings.  Each is combined with the matching upstream
      authority by taking the smaller of the two, so the envelope can only ever
      subtract authority.
    * ``maximum_ordinary_brake`` bounds *ordinary* nominal and degraded braking
      only.  It never weakens a trusted Mission 10A controlled stop, whose brake
      value and brake authority are republished exactly.
    * ``throttle_rise_rate_per_second`` bounds how fast requested throttle may
      *increase* per real second; decreases are always immediate.
    * ``brake_release_rate_per_second`` bounds how fast ordinary requested brake
      may *decrease* per real second; increases are always immediate.
    * ``maximum_shaping_delta_seconds`` caps the shaping interval, so a long gap
      between accepted updates cannot grant an unbounded single-step change.
    """

    maximum_steering: float = 0.35
    maximum_throttle: float = 0.35
    maximum_ordinary_brake: float = 0.40
    throttle_rise_rate_per_second: float = 0.50
    brake_release_rate_per_second: float = 1.00
    maximum_shaping_delta_seconds: float = 0.25

    def __post_init__(self) -> None:
        ceilings = {}
        for name in (
            "maximum_steering",
            "maximum_throttle",
            "maximum_ordinary_brake",
        ):
            value = _finite_real(getattr(self, name), name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0.0, 1.0]")
            ceilings[name] = value
        rates = {}
        for name in (
            "throttle_rise_rate_per_second",
            "brake_release_rate_per_second",
        ):
            value = _finite_real(getattr(self, name), name)
            if not 0.0 < value <= MAX_ENVELOPE_RATE_PER_SECOND:
                raise ValueError(
                    f"{name} must be in (0.0, {MAX_ENVELOPE_RATE_PER_SECOND}]"
                )
            rates[name] = value
        delta = _finite_real(
            self.maximum_shaping_delta_seconds, "maximum_shaping_delta_seconds"
        )
        if not 0.0 < delta <= MAX_SHAPING_DELTA_SECONDS:
            raise ValueError(
                "maximum_shaping_delta_seconds must be in "
                f"(0.0, {MAX_SHAPING_DELTA_SECONDS}]"
            )
        for name, value in ceilings.items():
            object.__setattr__(self, name, value)
        for name, value in rates.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "maximum_shaping_delta_seconds", delta)


DEFAULT_COMMAND_ENVELOPE_CONFIG = CommandEnvelopeConfig()


def _detached_config(config: CommandEnvelopeConfig) -> CommandEnvelopeConfig:
    """Rebuild one config from its scalars so no caller record is retained."""
    return CommandEnvelopeConfig(
        maximum_steering=config.maximum_steering,
        maximum_throttle=config.maximum_throttle,
        maximum_ordinary_brake=config.maximum_ordinary_brake,
        throttle_rise_rate_per_second=config.throttle_rise_rate_per_second,
        brake_release_rate_per_second=config.brake_release_rate_per_second,
        maximum_shaping_delta_seconds=config.maximum_shaping_delta_seconds,
    )


# --------------------------------------------------------------------------
# Published record
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CommandEnvelopeRequest:
    """One immutable command envelope and the detached evidence behind it.

    ``command_allowed`` is the single question any later actuation layer must
    obey before this becomes a vehicle command, and ``stop_required``
    distinguishes a *trusted controlled-stop request* from a neutral refusal that
    asks for nothing at all.  Mission 10B applies neither: the three axes are
    normalized desired values, never pedal or steering commands.

    A refusal exposes nothing.  Every ``DISABLED`` envelope is exactly zero on
    all three axes and all three authorities, reports no shaping of any kind, and
    publishes no upstream provenance whatsoever, so a neutral refusal can never
    be mistaken for evidence about the world or for a trusted stop.
    """

    mode: CommandEnvelopeMode
    reason: CommandEnvelopeReason
    command_allowed: bool
    stop_required: bool = False

    steering_request: float = 0.0
    throttle_request: float = 0.0
    brake_request: float = 0.0

    steering_authority: float = 0.0
    throttle_authority: float = 0.0
    brake_authority: float = 0.0

    steering_clamped: bool = False
    throttle_clamped: bool = False
    brake_clamped: bool = False
    throttle_rate_limited: bool = False
    brake_release_rate_limited: bool = False
    transition_interlocked: bool = False
    shaping_applied: bool = False

    evaluation_monotonic: float | None = None
    controller_epoch: int = 0
    evidence_bound: bool = False

    guidance_stamp: SampleStamp | None = None
    speed_stamp: SampleStamp | None = None
    snapshot_revision: int | None = None
    target_origin: TargetSpeedOrigin | None = None
    target_epoch: int | None = None
    target_revision: int | None = None
    target_issued_monotonic: float | None = None
    target_valid_until_monotonic: float | None = None
    safety_state: SafetyState | None = None
    temporal_state: TemporalLaneState | None = None
    lateral_mode: LateralControlMode | None = None
    lateral_reason: LateralControlReason | None = None
    longitudinal_mode: LongitudinalControlMode | None = None
    longitudinal_reason: LongitudinalControlReason | None = None

    effective_target_speed_mps: float | None = None
    observed_forward_speed_mps: float | None = None
    speed_error_mps: float | None = None
    speed_authority_mps: float | None = None

    def __post_init__(self) -> None:
        if type(self.reason) is not CommandEnvelopeReason:
            raise ValueError("reason must be a CommandEnvelopeReason")
        if type(self.mode) is not CommandEnvelopeMode:
            raise ValueError("mode must be a CommandEnvelopeMode")
        if self.mode is not _REASON_MODE[self.reason]:
            raise ValueError("mode does not match its command envelope reason")
        allowed = _builtin_bool(self.command_allowed, "command_allowed")
        if allowed is not (self.mode in _ACTIONABLE_MODES):
            raise ValueError("command_allowed does not match the envelope mode")
        stop_required = _builtin_bool(self.stop_required, "stop_required")
        if stop_required is not (self.mode is CommandEnvelopeMode.STOPPING):
            raise ValueError("stop_required does not match the envelope mode")

        steering = _bounded_real(
            self.steering_request, "steering_request", low=-1.0, high=1.0
        )
        throttle = _bounded_real(
            self.throttle_request, "throttle_request", low=0.0, high=1.0
        )
        brake = _bounded_real(self.brake_request, "brake_request", low=0.0, high=1.0)
        steering_authority = _bounded_real(
            self.steering_authority, "steering_authority", low=0.0, high=1.0
        )
        throttle_authority = _bounded_real(
            self.throttle_authority, "throttle_authority", low=0.0, high=1.0
        )
        brake_authority = _bounded_real(
            self.brake_authority, "brake_authority", low=0.0, high=1.0
        )

        flags = {}
        for name in (
            "steering_clamped",
            "throttle_clamped",
            "brake_clamped",
            "throttle_rate_limited",
            "brake_release_rate_limited",
            "transition_interlocked",
        ):
            flags[name] = _builtin_bool(getattr(self, name), name)
        shaping_applied = _builtin_bool(self.shaping_applied, "shaping_applied")
        if shaping_applied is not any(flags.values()):
            raise ValueError("shaping_applied must be the exact disjunction of flags")

        if abs(steering) > steering_authority:
            raise ValueError("steering cannot exceed its own authority")
        if throttle > throttle_authority:
            raise ValueError("throttle cannot exceed its own authority")
        if brake > brake_authority:
            raise ValueError("brake cannot exceed its own authority")
        if throttle > 0.0 and brake > 0.0:
            raise ValueError("throttle and brake can never both be requested")

        pedal_shape = (
            flags["throttle_clamped"],
            flags["brake_clamped"],
            flags["throttle_rate_limited"],
            flags["brake_release_rate_limited"],
            flags["transition_interlocked"],
        )
        if self.mode is CommandEnvelopeMode.STOPPING:
            if any(pedal_shape):
                raise ValueError("a trusted stop cannot report pedal shaping")
        elif pedal_shape not in _REACHABLE_PEDAL_SHAPING_SHAPES:
            raise ValueError(
                "pedal shaping flags are not a reachable envelope output shape"
            )
        if flags["steering_clamped"] and abs(steering) != steering_authority:
            raise ValueError(
                "a steering clamp must publish exactly at steering authority"
            )
        if flags["brake_clamped"] and (
            brake <= 0.0 or brake != brake_authority
        ):
            raise ValueError("a brake clamp must publish positive brake authority")
        if flags["brake_release_rate_limited"] and brake <= 0.0:
            raise ValueError("a rate-limited brake release must remain positive")
        if flags["throttle_rate_limited"] and (
            brake != 0.0 or throttle >= throttle_authority
        ):
            raise ValueError(
                "a rate-limited throttle rise must remain below throttle authority"
            )
        if flags["transition_interlocked"]:
            if throttle != 0.0:
                raise ValueError("an interlocked envelope cannot carry throttle")
            if flags["brake_release_rate_limited"] is not (brake > 0.0):
                raise ValueError(
                    "an interlock carries positive brake exactly while its release "
                    "is rate limited"
                )
        if (
            flags["throttle_clamped"]
            and not flags["throttle_rate_limited"]
            and not flags["transition_interlocked"]
            and (throttle <= 0.0 or throttle != throttle_authority)
        ):
            raise ValueError(
                "an unopposed throttle clamp must publish throttle authority"
            )

        evaluated = self.evaluation_monotonic
        if evaluated is not None:
            evaluated = _finite_real(evaluated, "evaluation_monotonic")
        epoch = _nonnegative_builtin_int(self.controller_epoch, "controller_epoch")
        bound = _builtin_bool(self.evidence_bound, "evidence_bound")
        if bound is not (self.mode in _ACTIONABLE_MODES):
            raise ValueError("evidence_bound does not match the envelope mode")

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
        if self.target_origin is not None and (
            type(self.target_origin) is not TargetSpeedOrigin
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

        for name, expected in (
            ("safety_state", SafetyState),
            ("temporal_state", TemporalLaneState),
            ("lateral_mode", LateralControlMode),
            ("lateral_reason", LateralControlReason),
            ("longitudinal_mode", LongitudinalControlMode),
            ("longitudinal_reason", LongitudinalControlReason),
        ):
            value = getattr(self, name)
            if value is not None and type(value) is not expected:
                raise ValueError(f"{name} must be a {expected.__name__} or None")

        effective_target = self.effective_target_speed_mps
        if effective_target is not None:
            effective_target = _bounded_real(
                effective_target,
                "effective_target_speed_mps",
                low=0.0,
                high=HARD_TARGET_CEILING_MPS,
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
            speed_authority = _bounded_real(
                speed_authority,
                "speed_authority_mps",
                low=0.0,
                high=HARD_TARGET_CEILING_MPS,
            )

        identity = (
            guidance_stamp,
            speed_stamp,
            revision,
            self.target_origin,
            target_epoch,
            target_revision,
            target_issued,
            target_valid_until,
            self.safety_state,
            self.temporal_state,
            self.lateral_mode,
            self.lateral_reason,
            self.longitudinal_mode,
            self.longitudinal_reason,
        )
        speed_law = (effective_target, speed_error, speed_authority)
        if bound:
            if evaluated is None or any(value is None for value in identity):
                raise ValueError(
                    "an actionable envelope requires complete identity binding"
                )
            if observed_forward is None:
                raise ValueError(
                    "an actionable envelope requires the observed forward speed"
                )
        else:
            if (
                steering != 0.0
                or throttle != 0.0
                or brake != 0.0
                or steering_authority != 0.0
                or throttle_authority != 0.0
                or brake_authority != 0.0
            ):
                raise ValueError("a refused envelope cannot carry any axis authority")
            if shaping_applied:
                raise ValueError("a refused envelope cannot report shaping activity")
            if observed_forward is not None or any(
                value is not None for value in (*identity, *speed_law)
            ):
                raise ValueError("a refused envelope cannot expose any provenance")

        if any(value is None for value in speed_law) and any(
            value is not None for value in speed_law
        ):
            raise ValueError("speed-law evidence must be complete or absent")
        if self.mode is CommandEnvelopeMode.STOPPING:
            if throttle != 0.0 or throttle_authority != 0.0:
                raise ValueError("a stopping envelope cannot carry throttle authority")
            if brake <= 0.0:
                raise ValueError("a stopping envelope must request positive brake")
            if any(
                flags[name]
                for name in (
                    "throttle_clamped",
                    "brake_clamped",
                    "throttle_rate_limited",
                    "brake_release_rate_limited",
                    "transition_interlocked",
                )
            ):
                raise ValueError("a trusted stop is republished exactly, never shaped")
        if bound:
            # An actionable envelope claims a real Mission 9 record and a real
            # Mission 10A record produced it.  Prove the claim is possible: no
            # caller may hand-build a command the envelope could never emit.
            conflict = _actionable_provenance_conflict(
                self.mode,
                self.safety_state,
                self.temporal_state,
                self.lateral_mode,
                self.lateral_reason,
                self.longitudinal_mode,
                self.longitudinal_reason,
            )
            if conflict is not None:
                raise ValueError(conflict)
            expected_shape = _SPEED_LAW_SHAPE[self.longitudinal_reason]
            if (
                effective_target is not None,
                observed_forward is not None,
                speed_error is not None,
                speed_authority is not None,
            ) != expected_shape:
                raise ValueError(
                    "speed-law evidence does not match its longitudinal branch"
                )
            if expected_shape is _COMPLETE_SPEED_LAW:
                if effective_target > speed_authority:
                    raise ValueError(
                        "effective target cannot exceed its speed authority"
                    )
                if speed_error != (effective_target - observed_forward) + 0.0:
                    raise ValueError("speed error contradicts its own law")
                if (
                    self.longitudinal_reason
                    is LongitudinalControlReason.TARGET_STOP
                ) is not (effective_target == 0.0):
                    raise ValueError(
                        "a target stop is exactly a zero effective target"
                    )
                # A *permitted* Mission 10A outcome always has a strictly
                # positive effective target; a trusted stop is actionable but
                # not permitted, and its target is exactly zero.
                if (
                    self.longitudinal_mode in _PERMITTED_LONGITUDINAL_MODES
                    and effective_target <= 0.0
                ):
                    raise ValueError(
                        "a permitted envelope requires a positive effective target"
                    )
            if self.lateral_mode is LateralControlMode.DISABLED and (
                steering != 0.0
                or steering_authority != 0.0
                or flags["steering_clamped"]
            ):
                raise ValueError(
                    "a refused lateral axis cannot carry steering in an envelope"
                )
            # An actionable envelope always takes each authority as the smaller
            # of a configured ceiling and the matching upstream authority, and
            # both of those are strictly positive on every axis the pair is
            # permitted to move: Mission 9 publishes a positive steering
            # authority for a permitted request, Mission 10A publishes positive
            # throttle *and* brake authorities for a permitted request and a
            # positive service-brake authority for every controlled stop, and
            # every envelope ceiling is validated into ``(0.0, 1.0]``.  An
            # actionable record whose authority is zero is therefore a shape no
            # Mission 10B path can emit -- it claims command authority while
            # admitting it has none.  A zero *axis* stays perfectly legitimate:
            # a permitted pair may genuinely request nothing.
            if self.mode is CommandEnvelopeMode.STOPPING:
                if brake_authority <= 0.0:
                    raise ValueError(
                        "a trusted stop requires positive brake authority"
                    )
                if (self.lateral_mode in _PERMITTED_LATERAL_MODES) is not (
                    steering_authority > 0.0
                ):
                    raise ValueError(
                        "stopping steering authority is positive exactly when "
                        "its Mission 9 record is permitted"
                    )
            elif (
                steering_authority <= 0.0
                or throttle_authority <= 0.0
                or brake_authority <= 0.0
            ):
                raise ValueError(
                    "a permitted envelope requires positive authority on every axis"
                )
            if self.mode is not CommandEnvelopeMode.STOPPING:
                plain, altered, interlocked = _COMMAND_REASONS[self.mode]
                if flags["transition_interlocked"]:
                    expected_reason = interlocked
                elif shaping_applied:
                    expected_reason = altered
                else:
                    expected_reason = plain
                if self.reason is not expected_reason:
                    raise ValueError(
                        "the envelope reason does not match its own shaping flags"
                    )

        object.__setattr__(self, "command_allowed", allowed)
        object.__setattr__(self, "stop_required", stop_required)
        object.__setattr__(self, "steering_request", steering)
        object.__setattr__(self, "throttle_request", throttle)
        object.__setattr__(self, "brake_request", brake)
        object.__setattr__(self, "steering_authority", steering_authority)
        object.__setattr__(self, "throttle_authority", throttle_authority)
        object.__setattr__(self, "brake_authority", brake_authority)
        for name, value in flags.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "shaping_applied", shaping_applied)
        object.__setattr__(self, "evaluation_monotonic", evaluated)
        object.__setattr__(self, "controller_epoch", epoch)
        object.__setattr__(self, "evidence_bound", bound)
        object.__setattr__(self, "guidance_stamp", guidance_stamp)
        object.__setattr__(self, "speed_stamp", speed_stamp)
        object.__setattr__(self, "snapshot_revision", revision)
        object.__setattr__(self, "target_epoch", target_epoch)
        object.__setattr__(self, "target_revision", target_revision)
        object.__setattr__(self, "target_issued_monotonic", target_issued)
        object.__setattr__(self, "target_valid_until_monotonic", target_valid_until)
        object.__setattr__(self, "effective_target_speed_mps", effective_target)
        object.__setattr__(self, "observed_forward_speed_mps", observed_forward)
        object.__setattr__(self, "speed_error_mps", speed_error)
        object.__setattr__(self, "speed_authority_mps", speed_authority)

    @property
    def guidance_frame(self) -> int | None:
        """The CARLA camera frame this envelope was bound to, if any."""
        stamp = self.guidance_stamp
        return None if stamp is None else stamp.frame

    @property
    def speed_frame(self) -> int | None:
        """The CARLA ego frame this envelope's speed evidence came from, if any."""
        stamp = self.speed_stamp
        return None if stamp is None else stamp.frame


@dataclass(frozen=True, slots=True)
class CommandEnvelopeMetrics:
    """One immutable view of the bounded scalar envelope counters.

    These are lifetime diagnostics.  No envelope behavior is derived from them,
    and they are read from the single owning consumer, not concurrently.
    """

    evaluations: int
    commands: int
    nominal_commands: int
    degraded_commands: int
    stop_commands: int
    refusals: int
    idempotent_replays: int
    steering_clamps: int
    throttle_clamps: int
    brake_clamps: int
    throttle_rate_limits: int
    brake_release_rate_limits: int
    transition_interlocks: int
    internal_errors: int
    resets: int
    controller_epoch: int


# --------------------------------------------------------------------------
# Private state
# --------------------------------------------------------------------------


class _Transition(Enum):
    """The bounded brake-to-throttle interlock state."""

    IDLE = "idle"
    RELEASING_BRAKE = "releasing_brake"
    ZERO_DWELL = "zero_dwell"


@dataclass(frozen=True, slots=True)
class _EvidenceState:
    """Accepted evidence, order, and fingerprint baselines.

    This group advances only after a fully bound accepted output.  Malformed,
    mismatched, rejected, conflicting, and time-invalid inputs leave it exactly
    as it was.
    """

    evaluation_monotonic: float | None = None
    guidance_stamp: SampleStamp | None = None
    speed_stamp: SampleStamp | None = None
    snapshot_revision: int | None = None
    target_origin: TargetSpeedOrigin | None = None
    target_epoch: int | None = None
    target_revision: int | None = None
    target_issued_monotonic: float | None = None
    target_valid_until_monotonic: float | None = None
    fingerprint: tuple[object, ...] | None = None


@dataclass(frozen=True, slots=True)
class _ShapingState:
    """Accepted pedal anchors, shaping time, and interlock position.

    This group is entirely separate from the evidence group and is cleared by
    every neutral refusal, so no stale throttle or brake can survive a malformed
    or neutral interval and reappear afterwards.
    """

    throttle_anchor: float = 0.0
    brake_anchor: float = 0.0
    shaping_monotonic: float | None = None
    transition: _Transition = _Transition.IDLE


@dataclass(frozen=True, slots=True)
class _ControllerState:
    """The whole envelope state, replaced by exactly one reference assignment.

    Keeping the visible request, the idempotency anchor, both baseline groups,
    and the lifetime counters inside one frozen record is what makes settlement
    genuinely atomic: every fallible step -- validating, arbitrating, shaping,
    constructing the record, and accounting for it -- happens while building the
    *next* state, and a failure anywhere in that construction leaves the previous
    state untouched because it was never partially overwritten.
    """

    request: CommandEnvelopeRequest
    accepted: CommandEnvelopeRequest | None
    evidence: _EvidenceState
    shaping: _ShapingState
    metrics: CommandEnvelopeMetrics


_CLEARED_EVIDENCE = _EvidenceState()
_CLEARED_SHAPING = _ShapingState()

_INITIAL_METRICS = CommandEnvelopeMetrics(
    evaluations=0,
    commands=0,
    nominal_commands=0,
    degraded_commands=0,
    stop_commands=0,
    refusals=0,
    idempotent_replays=0,
    steering_clamps=0,
    throttle_clamps=0,
    brake_clamps=0,
    throttle_rate_limits=0,
    brake_release_rate_limits=0,
    transition_interlocks=0,
    internal_errors=0,
    resets=0,
    controller_epoch=0,
)


def _cleared_state(metrics: CommandEnvelopeMetrics) -> _ControllerState:
    """One complete neutral state for a fresh epoch, with truthful counters."""
    return _ControllerState(
        request=CommandEnvelopeRequest(
            mode=CommandEnvelopeMode.DISABLED,
            reason=CommandEnvelopeReason.NOT_EVALUATED,
            command_allowed=False,
            controller_epoch=metrics.controller_epoch,
        ),
        accepted=None,
        evidence=_CLEARED_EVIDENCE,
        shaping=_CLEARED_SHAPING,
        metrics=metrics,
    )


def _accepted_metrics(
    metrics: CommandEnvelopeMetrics, request: CommandEnvelopeRequest
) -> CommandEnvelopeMetrics:
    """Pure: the lifetime counters after one accepted publication.

    This mutates nothing.  It is evaluated *before* the next state exists, so a
    failure here can never leave an advanced baseline behind a stale counter.
    """
    mode = request.mode
    return replace(
        metrics,
        evaluations=metrics.evaluations + 1,
        commands=metrics.commands + 1,
        nominal_commands=(
            metrics.nominal_commands + (1 if mode is CommandEnvelopeMode.NOMINAL else 0)
        ),
        degraded_commands=(
            metrics.degraded_commands
            + (1 if mode is CommandEnvelopeMode.DEGRADED else 0)
        ),
        stop_commands=(
            metrics.stop_commands + (1 if mode is CommandEnvelopeMode.STOPPING else 0)
        ),
        steering_clamps=metrics.steering_clamps + (1 if request.steering_clamped else 0),
        throttle_clamps=metrics.throttle_clamps + (1 if request.throttle_clamped else 0),
        brake_clamps=metrics.brake_clamps + (1 if request.brake_clamped else 0),
        throttle_rate_limits=(
            metrics.throttle_rate_limits + (1 if request.throttle_rate_limited else 0)
        ),
        brake_release_rate_limits=(
            metrics.brake_release_rate_limits
            + (1 if request.brake_release_rate_limited else 0)
        ),
        transition_interlocks=(
            metrics.transition_interlocks + (1 if request.transition_interlocked else 0)
        ),
    )


# The neutral ``INTERNAL_ERROR`` answer must not depend on anything whose failure
# could have caused it.  Binding the constructor *name* at definition time is not
# enough: the bound ``CommandEnvelopeRequest`` still runs ``__post_init__``, which
# calls the mutable module globals ``_bounded_real``, ``_builtin_bool``,
# ``_finite_real``, ``math``, and ``_REASON_MODE``.  A fault in any one of those
# is exactly the kind of fault that reaches this path, so re-entering them here
# would leave the previous -- possibly actionable -- publication standing as the
# apparent current answer.
#
# The emergency record is therefore built once, at import, through the ordinary
# validated constructor, and later reproduced field by field with
# ``object.__setattr__``.  This is a deliberately specialized construction, not
# an uncontrolled mutation: the template is a fully validated instance, *every*
# declared field is written from it, and the only two values that vary are ones
# this module produced itself and already knows to be exact -- a canonical
# control time (or ``None``) and a non-negative integer epoch.  The Mission 10B
# tests pin every published field of an emergency record against one the ordinary
# constructor built, so the two can never drift apart.
_INTERNAL_ERROR_TEMPLATE = CommandEnvelopeRequest(
    mode=CommandEnvelopeMode.DISABLED,
    reason=CommandEnvelopeReason.INTERNAL_ERROR,
    command_allowed=False,
)

_REQUEST_FIELD_NAMES: tuple[str, ...] = tuple(
    field.name for field in fields(CommandEnvelopeRequest)
)
_INTERNAL_ERROR_FIELDS: tuple[tuple[str, object], ...] = tuple(
    (name, getattr(_INTERNAL_ERROR_TEMPLATE, name)) for name in _REQUEST_FIELD_NAMES
)


def _emergency_request(
    evaluated: float | None,
    controller_epoch: int,
    *,
    _request_type: type = CommandEnvelopeRequest,
    _template_fields: tuple[tuple[str, object], ...] = _INTERNAL_ERROR_FIELDS,
    _new=object.__new__,
    _set=object.__setattr__,
) -> CommandEnvelopeRequest:
    """Build one neutral ``INTERNAL_ERROR`` record without re-entering validation.

    Every declared field is written from the validated import-time template and
    the two this module owns are then overwritten.  Nothing here consults a
    module global, a caller object, or any validation helper, so no ordinary
    mutable-helper fault can prevent the record from being built.
    """
    request = _new(_request_type)
    for name, value in _template_fields:
        _set(request, name, value)
    _set(request, "evaluation_monotonic", evaluated)
    _set(request, "controller_epoch", controller_epoch)
    return request


def _internal_error_state(
    state: _ControllerState,
    evaluated: float | None,
    *,
    _state_type: type = _ControllerState,
    _metrics_type: type = CommandEnvelopeMetrics,
    _emergency=_emergency_request,
    _cleared_shaping: _ShapingState = _CLEARED_SHAPING,
) -> _ControllerState:
    """One complete neutral ``INTERNAL_ERROR`` state replacement.

    The whole shaping and interlock history is destroyed, the accepted evidence,
    order, and fingerprint baselines are carried over exactly as they were, and
    the canonical control time is published when one had already been
    established.  No exception, traceback, or caller object is retained.

    Everything this fail-closed path needs is bound at definition time *and* none
    of it re-enters the validation helpers, so a failure anywhere in the ordinary
    evaluation -- including inside those helpers themselves -- cannot also
    disable the neutral publication that answers it.  The counters are rebuilt
    field by field rather than through ``dataclasses.replace`` for the same
    reason: ``replace`` is a module global a fault could have replaced.

    The only remaining way this can fail is if the Python runtime primitives it
    is built from -- ``object.__new__``, ``object.__setattr__``, and plain
    dataclass slot assignment -- are themselves unavailable.  That limit is real
    and is why :meth:`CommandEnvelope.update` still guards this call, but no
    ordinary mutable-module-helper failure reaches it.
    """
    previous = state.metrics
    metrics = _metrics_type(
        evaluations=previous.evaluations + 1,
        commands=previous.commands,
        nominal_commands=previous.nominal_commands,
        degraded_commands=previous.degraded_commands,
        stop_commands=previous.stop_commands,
        refusals=previous.refusals + 1,
        idempotent_replays=previous.idempotent_replays,
        steering_clamps=previous.steering_clamps,
        throttle_clamps=previous.throttle_clamps,
        brake_clamps=previous.brake_clamps,
        throttle_rate_limits=previous.throttle_rate_limits,
        brake_release_rate_limits=previous.brake_release_rate_limits,
        transition_interlocks=previous.transition_interlocks,
        internal_errors=previous.internal_errors + 1,
        resets=previous.resets,
        controller_epoch=previous.controller_epoch,
    )
    return _state_type(
        request=_emergency(evaluated, metrics.controller_epoch),
        accepted=state.accepted,
        evidence=state.evidence,
        shaping=_cleared_shaping,
        metrics=metrics,
    )


def _rise_limited(
    candidate: float, anchor: float, rate: float, dt: float
) -> tuple[float, bool]:
    """Bound one increase; decreases and equalities pass through untouched."""
    if candidate <= anchor:
        return candidate, False
    ceiling = anchor + rate * dt
    if ceiling < candidate:
        return ceiling, True
    return candidate, False


def _release_limited(
    candidate: float, anchor: float, rate: float, dt: float
) -> tuple[float, bool]:
    """Bound one decrease; increases and equalities pass through untouched."""
    if candidate >= anchor:
        return candidate, False
    floor = anchor - rate * dt
    if floor > candidate:
        return floor, True
    return candidate, False


def _clamp_symmetric(value: float, authority: float) -> tuple[float, bool]:
    """Clamp one signed axis into its authority and report whether it moved."""
    if value > authority:
        return authority, True
    if value < -authority:
        return -authority, True
    return value, False


def _ordered_source(
    stamp: SampleStamp,
    baseline: SampleStamp | None,
    revision: int,
    baseline_revision: int | None,
) -> bool:
    """Return ``True`` when one stamped source is a legitimate next observation.

    Mission 6 duplicate semantics are preserved exactly: a duplicate is the same
    frame *and* the same simulation time, a higher frame at an equal simulation
    time is legitimate, and anything older or reordered is not.  A duplicate
    whose monotonic identity or bound revision has changed is an identity-mutated
    replay and is rejected rather than re-evaluated.
    """
    if baseline is None:
        return True
    if stamp.frame == baseline.frame and stamp.sim_time == baseline.sim_time:
        return (
            stamp.monotonic == baseline.monotonic and revision == baseline_revision
        )
    return stamp.frame > baseline.frame and stamp.sim_time >= baseline.sim_time


# --------------------------------------------------------------------------
# The envelope
# --------------------------------------------------------------------------


class CommandEnvelope:
    """Single-consumer, lock-free command envelope for one controller epoch.

    ``update`` consumes the Mission 9 lateral request and the Mission 10A
    longitudinal request produced for one evidence cut, together with the one
    canonical host-monotonic ``now`` all three layers were given, and returns the
    current immutable :class:`CommandEnvelopeRequest`.  It creates no thread,
    queue, timer, or clock, and it contacts nothing.

    Missions 8, 9, and 10A remain the sole authorities on whether motion is
    permitted.  This class never re-decides that, never overrides it, and can
    only subtract authority: it additionally refuses whenever the two records do
    not describe one coherent cut, whenever either is stale or replayed, and
    whenever the two axes disagree about whether motion is permitted at all.

    All mutable state lives in exactly one frozen :class:`_ControllerState`
    reference.  Every outcome computes a complete replacement first and assigns
    it once, so no observer can ever see a partially advanced combination of
    visible request, idempotency anchor, evidence baseline, shaping baseline, and
    lifetime counters.
    """

    __slots__ = ("_config", "_state")

    def __init__(
        self,
        config: CommandEnvelopeConfig = DEFAULT_COMMAND_ENVELOPE_CONFIG,
    ) -> None:
        if not isinstance(config, CommandEnvelopeConfig):
            raise TypeError("config must be a CommandEnvelopeConfig")
        # Detach every caller-owned record, including an exact frozen base
        # instance, before it can influence later envelope output.
        self._config = _detached_config(config)
        self._state = _cleared_state(_INITIAL_METRICS)

    # ------------------------------------------------------------- accessors

    @property
    def config(self) -> CommandEnvelopeConfig:
        return _detached_config(self._config)

    @property
    def latest_request(self) -> CommandEnvelopeRequest:
        return self._state.request

    @property
    def command_allowed(self) -> bool:
        return self._state.request.command_allowed

    @property
    def metrics(self) -> CommandEnvelopeMetrics:
        return self._state.metrics

    # ------------------------------------------------------------- lifecycle

    def reset(self) -> CommandEnvelopeRequest:
        """Begin a fresh envelope epoch; lifetime counters remain truthful.

        Mission 10B owns ``controller_epoch``, so the epoch advances here and
        nowhere else.  No previous command, evidence baseline, order baseline,
        fingerprint, pedal anchor, shaping time, interlock position, or identity
        association crosses the boundary, and the published envelope returns to a
        neutral ``NOT_EVALUATED`` refusal with all three axes and all three
        authorities at exactly ``0.0``.  Old upstream identities may legitimately
        be accepted again in the new local epoch, because every local order
        baseline is gone.  Repeated calls are idempotent apart from the epoch and
        the reset counter.
        """
        metrics = self._state.metrics
        next_state = _cleared_state(
            replace(
                metrics,
                resets=metrics.resets + 1,
                controller_epoch=metrics.controller_epoch + 1,
            )
        )
        self._state = next_state
        return next_state.request

    # ---------------------------------------------------------------- update

    def update(
        self,
        lateral: LateralControlRequest,
        longitudinal: LongitudinalControlRequest,
        *,
        now: float,
    ) -> CommandEnvelopeRequest:
        """Publish one bounded command envelope at control time ``now``.

        ``lateral`` and ``longitudinal`` must be the requests produced for one
        evidence cut at exactly this ``now``.  Both are read but never retained.
        Sample ``now`` from the same host-monotonic clock Missions 8, 9, and 10A
        were given, and never let it regress.

        A ``lateral`` or ``longitudinal`` argument of the wrong top-level type is
        programmer misuse rather than untrustworthy evidence, and it raises
        :class:`TypeError` **without publishing anything at all**.  Nothing about
        the controller changes: ``latest_request`` is still the *previous*
        publication, no metric moves, no shaping or evidence baseline is touched,
        and no caller object is retained.  A caller that sees this exception must
        treat the update as not having happened -- reading ``latest_request``
        afterwards yields the earlier result, never a newly published one, so it
        must never be interpreted as this call's answer.  Later valid traffic
        continues against that unchanged shaping epoch.

        An unexpected internal failure is never hidden: one complete neutral
        ``INTERNAL_ERROR`` state replaces the whole previous state, so the
        published envelope becomes that refusal, the shaping and interlock
        history is cleared, and the accepted evidence, order, and fingerprint
        baselines and the previously accepted output are left exactly as they
        were.  When ``now`` had already been converted, that trustworthy control
        time is published with the refusal.  The original exception is re-raised
        immediately with nothing about it retained.
        """
        if type(lateral) is not LateralControlRequest:
            raise TypeError("lateral must be a LateralControlRequest")
        if type(longitudinal) is not LongitudinalControlRequest:
            raise TypeError("longitudinal must be a LongitudinalControlRequest")
        state = self._state
        # Stage 1+2 of the control clock, and the *only* caller-controlled work
        # inside ``update``.  Applying the public input contract and performing
        # the single ``float(now)`` conversion are the one place a caller-owned
        # object runs arbitrary code: a hostile ``__class__`` may raise from the
        # type checks and a registered ``numbers.Real`` may raise anything at all
        # from ``__float__``.  That is a caller failure, not a controller defect,
        # and it means no trustworthy control time exists -- exactly
        # ``TIME_INVALID``.
        #
        # The boundary covers that conversion and nothing else.  Finiteness,
        # normalization, and every other check happen afterwards inside the
        # ordinary internal boundary below, so a fault in one of *those* is
        # reported truthfully as an internal error instead of being disguised as
        # a bad caller clock.  ``BaseException`` is deliberately not caught, so
        # ``KeyboardInterrupt`` and ``SystemExit`` still propagate.
        try:
            converted: object = _convert_caller_time(now)
        except Exception:
            converted = _CALLER_TIME_UNCONVERTIBLE
        # Stage 5 onwards runs inside the ordinary internal boundary.  The
        # canonical control time is established first so that an unexpected
        # failure further in can still publish the trustworthy time it was
        # evaluated at; a failure of the normalization itself leaves it unset,
        # which is truthful because no trustworthy time was ever established.
        evaluated: float | None = None
        try:
            if converted is not _CALLER_TIME_UNCONVERTIBLE:
                evaluated = _normalized_control_time(converted)
            next_state = self._evaluate(state, lateral, longitudinal, evaluated)
        except BaseException:
            try:
                self._state = _internal_error_state(state, evaluated)
            except BaseException:  # only reachable if the runtime itself fails
                pass
            raise
        self._state = next_state
        return next_state.request

    def _evaluate(
        self,
        state: _ControllerState,
        lateral: LateralControlRequest,
        longitudinal: LongitudinalControlRequest,
        evaluated: float | None,
    ) -> _ControllerState:
        """Run the whole safety precedence for one pair, in strict order.

        This is a pure function of the supplied state: it mutates nothing and
        returns the complete next state its caller then installs in one step.
        """
        # 1/2. Control clock.  ``evaluated`` is ``None`` when the caller's object
        #      could not be converted at all, and when the value it converted to
        #      is not a usable control time.  Both are caller problems, and
        #      without a trustworthy clock no shaping statement can be made at
        #      all, so the refusal publishes no time either.
        if evaluated is None:
            return self._refuse(state, CommandEnvelopeReason.TIME_INVALID, None)
        baseline_evaluation = state.evidence.evaluation_monotonic
        if baseline_evaluation is not None and evaluated < baseline_evaluation:
            return self._refuse(state, CommandEnvelopeReason.TIME_INVALID, None)

        # 3. Mission 9, completely reconstructed and proven against its own
        #    accepted contract before a single field of it is used.
        try:
            lateral = _canonicalize_lateral_request(lateral)
        except _EvidenceError:
            return self._refuse(
                state, CommandEnvelopeReason.MALFORMED_LATERAL_EVIDENCE, evaluated
            )

        # 4. Mission 10A, likewise.
        try:
            longitudinal = _canonicalize_longitudinal_request(longitudinal)
        except _EvidenceError:
            return self._refuse(
                state,
                CommandEnvelopeReason.MALFORMED_LONGITUDINAL_EVIDENCE,
                evaluated,
            )

        # 5/6. Upstream refusals that carry no bindable identity at all,
        #      including the exact evidence-free initial pair that construction
        #      and reset publish.  There is nothing here to bind, order, or
        #      arbitrate: the honest answer is a neutral refusal.
        if longitudinal.guidance_stamp is None:
            return self._refuse(
                state, CommandEnvelopeReason.UPSTREAM_REFUSAL, evaluated
            )

        # 7a. Axis permission.  A pair whose two axes disagree about whether
        #     motion is permitted at all is an axis conflict, not an identity
        #     mismatch, and is settled before the mode equality below could mask
        #     it.  This is the "permitted M10A paired with refused M9" rejection.
        if longitudinal.longitudinal_allowed and not lateral.lateral_allowed:
            return self._refuse(
                state, CommandEnvelopeReason.AXIS_CONFLICT, evaluated
            )

        # 7b. One control instant shared by the caller and both records.  When
        #     the two records agree with each other but not with the caller, the
        #     clock is what is wrong, not the pair.
        lateral_evaluated = lateral.evaluation_monotonic
        longitudinal_evaluated = longitudinal.evaluation_monotonic
        if lateral_evaluated != evaluated or longitudinal_evaluated != evaluated:
            if lateral_evaluated == longitudinal_evaluated:
                return self._refuse(
                    state, CommandEnvelopeReason.TIME_INVALID, evaluated
                )
            return self._refuse(
                state, CommandEnvelopeReason.CROSS_RECORD_IDENTITY_MISMATCH, evaluated
            )

        # 7c. One evidence cut.  Mission 10A already bound its own four sources;
        #     Mission 10B is handed the two results separately and proves for
        #     itself that they describe one camera frame, one snapshot, and one
        #     agreed upstream disposition.
        if (
            lateral.source_stamp != longitudinal.guidance_stamp
            or lateral.snapshot_revision != longitudinal.snapshot_revision
            or lateral.mode is not longitudinal.lateral_mode
            or lateral.safety_state is not longitudinal.safety_state
            or lateral.temporal_state is not longitudinal.temporal_state
        ):
            return self._refuse(
                state, CommandEnvelopeReason.CROSS_RECORD_IDENTITY_MISMATCH, evaluated
            )

        # 7d. The Mission 10A mode-selection relationship, recomputed from the
        #     bound pair rather than trusted.  Mission 10A calls a permitted
        #     outcome nominal only under fully nominal evidence, coasting exactly
        #     when the temporal estimate is coasting, and degraded tracking
        #     otherwise.
        if longitudinal.mode in _PERMITTED_LONGITUDINAL_MODES:
            expected = _mode_selection_reason(
                longitudinal.safety_state,
                lateral.mode,
                longitudinal.temporal_state,
            )
            if longitudinal.reason is not expected:
                return self._refuse(
                    state,
                    CommandEnvelopeReason.CROSS_RECORD_IDENTITY_MISMATCH,
                    evaluated,
                )

        # 7e. One *reachable* cut.  Every field above can agree and the pair can
        #     still be one no real upstream cut ever produces, because Mission
        #     10A was handed this very Mission 9 record and its gate order fixes
        #     which outcome that record can lead to.  A Mission 9 record that
        #     refused precisely because the Mission 8 decision did not describe
        #     its evidence cannot accompany a Mission 10A outcome that bound that
        #     same decision to it, however equal the individual fields are made.
        conflict = _same_cut_conflict(lateral.reason, longitudinal.reason)
        if conflict is not None:
            return self._refuse(
                state, CommandEnvelopeReason.CROSS_RECORD_IDENTITY_MISMATCH, evaluated
            )

        # 8. Local order, then the exact canonical fingerprint.  An unchanged
        #    control instant may only ever re-present the exact accepted pair
        #    whose output is still the visible one; anything else is a replay.
        fingerprint = _fingerprint(lateral, longitudinal)
        evidence = state.evidence
        if (
            baseline_evaluation is not None
            and evaluated == baseline_evaluation
        ):
            accepted = state.accepted
            if (
                accepted is not None
                and state.request is accepted
                and evidence.fingerprint == fingerprint
            ):
                metrics = replace(
                    state.metrics,
                    evaluations=state.metrics.evaluations + 1,
                    idempotent_replays=state.metrics.idempotent_replays + 1,
                )
                return _ControllerState(
                    request=accepted,
                    accepted=accepted,
                    evidence=evidence,
                    shaping=state.shaping,
                    metrics=metrics,
                )
            return self._refuse(
                state, CommandEnvelopeReason.SOURCE_REJECTED, evaluated
            )
        if not self._ordered(longitudinal, evidence):
            return self._refuse(
                state, CommandEnvelopeReason.SOURCE_REJECTED, evaluated
            )

        # 9. Trusted Mission 10A controlled stop.
        if longitudinal.mode is LongitudinalControlMode.STOPPING:
            return self._stop(state, lateral, longitudinal, evaluated, fingerprint)

        # 10. Trusted, bound Mission 10A neutral refusal.  It is coherent and it
        #     asks for nothing, so the envelope asks for nothing either.
        if longitudinal.mode is LongitudinalControlMode.DISABLED:
            return self._refuse(
                state, CommandEnvelopeReason.UPSTREAM_REFUSAL, evaluated
            )

        # 11/12. A permitted pair.
        return self._command(state, lateral, longitudinal, evaluated, fingerprint)

    # ----------------------------------------------------------------- order

    def _ordered(
        self, longitudinal: LongitudinalControlRequest, state: _EvidenceState
    ) -> bool:
        """Return ``True`` when every bound source is a legitimate next one."""
        if not _ordered_source(
            longitudinal.guidance_stamp,
            state.guidance_stamp,
            longitudinal.snapshot_revision,
            state.snapshot_revision,
        ):
            return False
        if not _ordered_source(
            longitudinal.speed_stamp,
            state.speed_stamp,
            longitudinal.snapshot_revision,
            state.snapshot_revision,
        ):
            return False
        if state.target_origin is None:
            return True
        # A different issuer or a different epoch is a new controller contract,
        # and a new contract requires an explicit reset.
        if (
            longitudinal.target_origin is not state.target_origin
            or longitudinal.target_epoch != state.target_epoch
        ):
            return False
        if (
            longitudinal.target_revision == state.target_revision
            and longitudinal.target_issued_monotonic == state.target_issued_monotonic
        ):
            return (
                longitudinal.target_valid_until_monotonic
                == state.target_valid_until_monotonic
            )
        return (
            longitudinal.target_revision > state.target_revision
            and longitudinal.target_issued_monotonic
            >= state.target_issued_monotonic
        )

    # -------------------------------------------------------------- outcomes

    def _stop(
        self,
        state: _ControllerState,
        lateral: LateralControlRequest,
        longitudinal: LongitudinalControlRequest,
        evaluated: float,
        fingerprint: tuple[object, ...],
    ) -> _ControllerState:
        """Republish one trusted Mission 10A controlled stop, exactly.

        The stop brake is neither shaped, nor weakened by the ordinary brake
        ceiling, nor strengthened: Mission 10A already bounded it against its own
        service-brake limit and Mission 10B is not entitled to second-guess a
        trusted deceleration demand.  It is immediate, including as the very
        first accepted update of an epoch.  Steering survives only while the
        bound Mission 9 request is itself permitted.
        """
        steering = 0.0
        steering_authority = 0.0
        steering_clamped = False
        if lateral.lateral_allowed:
            steering_authority = min(
                self._config.maximum_steering, lateral.steering_authority
            )
            steering, steering_clamped = _clamp_symmetric(
                lateral.steering, steering_authority
            )
        brake = longitudinal.brake_request
        request = CommandEnvelopeRequest(
            mode=CommandEnvelopeMode.STOPPING,
            reason=CommandEnvelopeReason.TRUSTED_UPSTREAM_STOP,
            command_allowed=True,
            stop_required=True,
            steering_request=steering,
            throttle_request=0.0,
            brake_request=brake,
            steering_authority=steering_authority,
            throttle_authority=0.0,
            brake_authority=longitudinal.brake_authority,
            steering_clamped=steering_clamped,
            shaping_applied=steering_clamped,
            evaluation_monotonic=evaluated,
            controller_epoch=state.metrics.controller_epoch,
            evidence_bound=True,
            **_provenance(lateral, longitudinal),
        )
        # A trusted stop cancels any propulsion transition outright and becomes
        # the new brake anchor, so resuming propulsion afterwards must release
        # that brake and dwell at zero first.
        shaping = _ShapingState(
            throttle_anchor=0.0,
            brake_anchor=brake,
            shaping_monotonic=evaluated,
            transition=_Transition.IDLE,
        )
        return self._settle(
            state, request, longitudinal, evaluated, fingerprint, shaping
        )

    def _command(
        self,
        state: _ControllerState,
        lateral: LateralControlRequest,
        longitudinal: LongitudinalControlRequest,
        evaluated: float,
        fingerprint: tuple[object, ...],
    ) -> _ControllerState:
        """Arbitrate, clamp, shape, and interlock one permitted pair."""
        config = self._config
        shaping = state.shaping

        steering_authority = min(config.maximum_steering, lateral.steering_authority)
        throttle_authority = min(
            config.maximum_throttle, longitudinal.throttle_authority
        )
        brake_authority = min(
            config.maximum_ordinary_brake, longitudinal.brake_authority
        )

        steering, steering_clamped = _clamp_symmetric(
            lateral.steering, steering_authority
        )

        throttle_candidate = longitudinal.throttle_request
        throttle_clamped = throttle_candidate > throttle_authority
        if throttle_clamped:
            throttle_candidate = throttle_authority
        brake_candidate = longitudinal.brake_request
        brake_clamped = brake_candidate > brake_authority
        if brake_clamped:
            brake_candidate = brake_authority

        # The shaping interval.  There is no artificial minimum: the first
        # accepted update of an epoch, and the first after any neutralization,
        # has no anchor time and therefore no allowance at all.
        if shaping.shaping_monotonic is None:
            dt = 0.0
        else:
            dt = evaluated - shaping.shaping_monotonic
            if dt < 0.0:
                dt = 0.0
            elif dt > config.maximum_shaping_delta_seconds:
                dt = config.maximum_shaping_delta_seconds

        # Both anchors are first brought inside the *current* authority, so an
        # authority reduction takes effect immediately instead of being slewed.
        throttle_anchor = min(shaping.throttle_anchor, throttle_authority)
        brake_anchor = min(shaping.brake_anchor, brake_authority)

        throttle_rate_limited = False
        brake_release_rate_limited = False
        transition_interlocked = False

        if brake_candidate > 0.0:
            # Deceleration demand.  Throttle leaves immediately, brake arrives
            # immediately, and any propulsion transition is cancelled.
            throttle = 0.0
            brake, brake_release_rate_limited = _release_limited(
                brake_candidate,
                brake_anchor,
                config.brake_release_rate_per_second,
                dt,
            )
            transition = _Transition.IDLE
        elif throttle_candidate > 0.0:
            if shaping.transition is _Transition.ZERO_DWELL:
                # A strictly newer accepted update has arrived after the
                # zero-pedal dwell, so the rise may finally begin, from zero.
                throttle, throttle_rate_limited = _rise_limited(
                    throttle_candidate,
                    throttle_anchor,
                    config.throttle_rise_rate_per_second,
                    dt,
                )
                brake, brake_release_rate_limited = _release_limited(
                    0.0, brake_anchor, config.brake_release_rate_per_second, dt
                )
                transition = _Transition.IDLE
            elif brake_anchor > 0.0:
                # Never straight from positive brake to positive throttle.
                transition_interlocked = True
                throttle = 0.0
                brake, brake_release_rate_limited = _release_limited(
                    0.0, brake_anchor, config.brake_release_rate_per_second, dt
                )
                transition = (
                    _Transition.ZERO_DWELL
                    if brake == 0.0
                    else _Transition.RELEASING_BRAKE
                )
            else:
                throttle, throttle_rate_limited = _rise_limited(
                    throttle_candidate,
                    throttle_anchor,
                    config.throttle_rise_rate_per_second,
                    dt,
                )
                brake = 0.0
                transition = _Transition.IDLE
        else:
            # Neither pedal is asked for: throttle leaves immediately and any
            # remaining brake keeps releasing at the configured rate.
            throttle = 0.0
            brake, brake_release_rate_limited = _release_limited(
                0.0, brake_anchor, config.brake_release_rate_per_second, dt
            )
            transition = _Transition.IDLE

        mode = (
            CommandEnvelopeMode.NOMINAL
            if longitudinal.mode is LongitudinalControlMode.NOMINAL
            else CommandEnvelopeMode.DEGRADED
        )
        shaped = (
            steering_clamped
            or throttle_clamped
            or brake_clamped
            or throttle_rate_limited
            or brake_release_rate_limited
            or transition_interlocked
        )
        plain, altered, interlocked = _COMMAND_REASONS[mode]
        if transition_interlocked:
            reason = interlocked
        elif shaped:
            reason = altered
        else:
            reason = plain

        request = CommandEnvelopeRequest(
            mode=mode,
            reason=reason,
            command_allowed=True,
            stop_required=False,
            steering_request=steering,
            throttle_request=throttle,
            brake_request=brake,
            steering_authority=steering_authority,
            throttle_authority=throttle_authority,
            brake_authority=brake_authority,
            steering_clamped=steering_clamped,
            throttle_clamped=throttle_clamped,
            brake_clamped=brake_clamped,
            throttle_rate_limited=throttle_rate_limited,
            brake_release_rate_limited=brake_release_rate_limited,
            transition_interlocked=transition_interlocked,
            shaping_applied=shaped,
            evaluation_monotonic=evaluated,
            controller_epoch=state.metrics.controller_epoch,
            evidence_bound=True,
            **_provenance(lateral, longitudinal),
        )
        next_shaping = _ShapingState(
            throttle_anchor=throttle,
            brake_anchor=brake,
            shaping_monotonic=evaluated,
            transition=transition,
        )
        return self._settle(
            state, request, longitudinal, evaluated, fingerprint, next_shaping
        )

    def _refuse(
        self,
        state: _ControllerState,
        reason: CommandEnvelopeReason,
        evaluated: float | None,
    ) -> _ControllerState:
        """Build one neutral refusal that advances no accepted baseline at all.

        Every axis and every authority becomes exactly ``+0.0`` immediately, no
        shaping is claimed, no provenance is exposed, any previously visible
        command stops being the answer, and the whole shaping and interlock
        history is destroyed so no stale throttle or brake can reappear on the
        other side of the refusal.  The evidence, order, and fingerprint
        baselines and the idempotency anchor are carried over exactly as the last
        accepted output left them.
        """
        metrics = replace(
            state.metrics,
            evaluations=state.metrics.evaluations + 1,
            refusals=state.metrics.refusals + 1,
        )
        request = CommandEnvelopeRequest(
            mode=CommandEnvelopeMode.DISABLED,
            reason=reason,
            command_allowed=False,
            evaluation_monotonic=evaluated,
            controller_epoch=metrics.controller_epoch,
        )
        return _ControllerState(
            request=request,
            accepted=state.accepted,
            evidence=state.evidence,
            shaping=_CLEARED_SHAPING,
            metrics=metrics,
        )

    def _settle(
        self,
        state: _ControllerState,
        request: CommandEnvelopeRequest,
        longitudinal: LongitudinalControlRequest,
        evaluated: float,
        fingerprint: tuple[object, ...],
        shaping: _ShapingState,
    ) -> _ControllerState:
        """Build the complete next state for one fully constructed envelope.

        Every fallible step -- constructing the request, accounting for it, and
        building both baseline groups -- happens here, before any of it becomes
        visible.  The caller installs the result with a single assignment, so a
        failure anywhere leaves the previous state entirely intact rather than
        half-advanced.
        """
        metrics = _accepted_metrics(state.metrics, request)
        evidence = _EvidenceState(
            evaluation_monotonic=evaluated,
            guidance_stamp=longitudinal.guidance_stamp,
            speed_stamp=longitudinal.speed_stamp,
            snapshot_revision=longitudinal.snapshot_revision,
            target_origin=longitudinal.target_origin,
            target_epoch=longitudinal.target_epoch,
            target_revision=longitudinal.target_revision,
            target_issued_monotonic=longitudinal.target_issued_monotonic,
            target_valid_until_monotonic=(
                longitudinal.target_valid_until_monotonic
            ),
            fingerprint=fingerprint,
        )
        return _ControllerState(
            request=request,
            accepted=request,
            evidence=evidence,
            shaping=shaping,
            metrics=metrics,
        )


def _provenance(
    lateral: LateralControlRequest, longitudinal: LongitudinalControlRequest
) -> dict[str, object]:
    """The exact detached provenance every actionable envelope must publish.

    Nothing is invented.  Mission 10A publishes no target mode, no raw target
    speed, no speed capture time, and no Mission 8 reason, so none of those
    appear here, and the optional Mission 10A speed-law evidence is carried
    forward with exactly the shape Mission 10A gave it.
    """
    return dict(
        guidance_stamp=longitudinal.guidance_stamp,
        speed_stamp=longitudinal.speed_stamp,
        snapshot_revision=longitudinal.snapshot_revision,
        target_origin=longitudinal.target_origin,
        target_epoch=longitudinal.target_epoch,
        target_revision=longitudinal.target_revision,
        target_issued_monotonic=longitudinal.target_issued_monotonic,
        target_valid_until_monotonic=longitudinal.target_valid_until_monotonic,
        safety_state=longitudinal.safety_state,
        temporal_state=longitudinal.temporal_state,
        lateral_mode=lateral.mode,
        lateral_reason=lateral.reason,
        longitudinal_mode=longitudinal.mode,
        longitudinal_reason=longitudinal.reason,
        effective_target_speed_mps=longitudinal.effective_target_speed_mps,
        observed_forward_speed_mps=longitudinal.observed_forward_speed_mps,
        speed_error_mps=longitudinal.speed_error_mps,
        speed_authority_mps=longitudinal.speed_authority_mps,
    )


def _fingerprint(
    lateral: LateralControlRequest, longitudinal: LongitudinalControlRequest
) -> tuple[object, ...]:
    """One exact canonical fingerprint of the whole reconstructed pair.

    Every published field of both records participates, with stamps expanded
    into their exact scalars, so any change of canonical content at an unchanged
    evaluation instant is visible as an inequality rather than being replayed.
    """
    return (
        lateral.mode,
        lateral.reason,
        lateral.lateral_allowed,
        lateral.steering,
        lateral.raw_steering,
        lateral.steering_authority,
        lateral.clamped,
        lateral.rate_limited,
        lateral.evaluation_monotonic,
        *_stamp_key(lateral.source_stamp),
        lateral.snapshot_revision,
        lateral.lateral_error,
        lateral.heading_error,
        lateral.safety_state,
        lateral.temporal_state,
        longitudinal.mode,
        longitudinal.reason,
        longitudinal.longitudinal_allowed,
        longitudinal.stop_required,
        longitudinal.throttle_request,
        longitudinal.brake_request,
        longitudinal.effective_target_speed_mps,
        longitudinal.observed_forward_speed_mps,
        longitudinal.speed_error_mps,
        longitudinal.speed_authority_mps,
        longitudinal.throttle_authority,
        longitudinal.brake_authority,
        longitudinal.throttle_clamped,
        longitudinal.brake_clamped,
        longitudinal.evaluation_monotonic,
        *_stamp_key(longitudinal.guidance_stamp),
        *_stamp_key(longitudinal.speed_stamp),
        longitudinal.snapshot_revision,
        longitudinal.target_origin,
        longitudinal.target_epoch,
        longitudinal.target_revision,
        longitudinal.target_issued_monotonic,
        longitudinal.target_valid_until_monotonic,
        longitudinal.safety_state,
        longitudinal.lateral_mode,
        longitudinal.temporal_state,
    )
