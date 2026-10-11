"""Bounded, deterministic lateral steering *intent* from published evidence.

Mission 9 answers exactly one question: given the Mission 7 temporal lane
estimate that Mission 6 published, and the Mission 8 permission that was issued
for that very same evidence, what bounded lateral steering intent may be
requested right now?

It is an intent producer, not an actuator.  It builds no ``carla.VehicleControl``,
calls no ``apply_control``, owns no CARLA actor, and emits no throttle, brake,
target speed, or longitudinal command.  It creates no thread, queue, timer,
socket, or clock: the caller supplies host-monotonic time explicitly, exactly as
Mission 8 requires.

Sign convention
---------------
Positive requested steering means "steer toward increasing normalized image x",
that is, toward the right-hand side of the camera image.  Negative means left.
Zero means straight ahead.  The unit is a normalized, dimensionless steering
intent bounded to ``[-1.0, +1.0]``; it is not a road-wheel angle.

Mission 5 publishes two already-normalized image-space scalars, both bounded to
``[-1.0, 1.0]`` by ``LaneObservation`` and carried forward unchanged in scale by
Mission 7:

* ``center_offset`` is ``lane_center_x - 0.5``, evaluated at normalized image
  row ``center_evaluation_y`` (``0.90`` by default, near the bottom of the
  frame, which shows the road closest to the ego).  Positive means the lane
  centre lies to the *right* of the image centre.
* ``heading_proxy`` is the image-space centreline slope
  ``(center_x_at_roi_bottom - center_x_at_roi_top) / (roi_bottom - roi_top)``.
  Normalized y increases downward, so positive means the centreline moves
  rightward as rows move *down* the image, equivalently leftward as rows move
  *up*; rows further up show road further ahead of the ego.

Mission 5 fits each boundary as a straight line, so the centreline is straight
too and its column at any row ``y`` is exactly

    x_c(y) = (0.5 + center_offset) + heading_proxy * (y - center_evaluation_y)

Evaluating it a normalized lookahead ``L`` further ahead, i.e. ``L`` rows *above*
the evaluation row, gives that row's lane-centre displacement from the image
centre:

    x_c(center_evaluation_y - L) - 0.5 = center_offset - L * heading_proxy

That quantity is positive exactly when the lane centre at the lookahead row lies
to the right of the image centre, which is exactly when the ego must steer right
to move toward the lane centre.  Mission 9 therefore requests

    raw_steering = offset_gain * center_offset - heading_gain * heading_proxy

so ``heading_gain / offset_gain`` *is* the effective normalized lookahead ``L``.
The default gains give ``L = 0.40 / 1.20 = 0.333...`` and an effective lookahead
row of ``0.90 - 0.333... = 0.566...``, which lies inside the default Mission 5
ROI ``[0.50, 1.00]`` and therefore inside rows Mission 5 actually fitted.

``L = 0`` degenerates to the near cross-track term alone and larger ``L`` looks
further ahead; both signs above are fixed by the published Mission 5 definitions
and are proven by the sign tests, not assumed.

No image-width or pixel normalization constant appears anywhere here.  Both
consumed scalars are already normalized and bounded by the accepted Mission 5
contract, so there is nothing to rescale and no image size to hardcode.

Retained state is O(1): one immutable request, one previous steering value and
its evaluation time, one canonical accepted source stamp, and bounded lifetime
counters.  There is no command, frame, or decision history.  The controller is
single-consumer and lock-free; its metrics are diagnostics read by that same
consumer, not concurrently atomic.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from enum import Enum

from perception.runtime import PerceptionRuntimeResult
from perception.state import LaneDetectionState
from perception.temporal import (
    TemporalLaneEstimate,
    TemporalLaneObservation,
    TemporalLaneState,
)
from safety.supervisor import SafetyDecision, SafetyState
from telemetry.state import SampleStamp


MAX_LATERAL_GAIN = 100.0
MAX_STEERING_RATE_PER_SECOND = 100.0


__all__ = (
    "DEFAULT_LATERAL_CONTROLLER_CONFIG",
    "MAX_LATERAL_GAIN",
    "MAX_STEERING_RATE_PER_SECOND",
    "LateralControlMode",
    "LateralControlReason",
    "LateralControlRequest",
    "LateralController",
    "LateralControllerConfig",
    "LateralControllerMetrics",
)


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _bounded_real(value: object, name: str, *, low: float, high: float) -> float:
    result = _finite_real(value, name)
    if not low <= result <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
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


class LateralControlMode(Enum):
    """How much lateral authority this request carries, if any."""

    DISABLED = "disabled"
    NOMINAL = "nominal"
    DEGRADED = "degraded"


class LateralControlReason(Enum):
    """One bounded explanation for the current lateral control request."""

    NOT_EVALUATED = "not_evaluated"
    NOMINAL_TRACKING = "nominal_tracking"
    DEGRADED_TRACKING = "degraded_tracking"
    DEGRADED_COASTING = "degraded_coasting"
    SAFETY_NOT_PERMITTED = "safety_not_permitted"
    INSUFFICIENT_GEOMETRY = "insufficient_geometry"
    IDENTITY_MISMATCH = "identity_mismatch"
    EVIDENCE_MALFORMED = "evidence_malformed"
    SOURCE_REJECTED = "source_rejected"
    TIME_INVALID = "time_invalid"


# Every reason belongs to exactly one mode, so a request can never advertise
# steering authority under a fail-closed explanation.
_REASON_MODE: dict[LateralControlReason, LateralControlMode] = {
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

# Lateral control is permitted in exactly two modes and nowhere else.
_PERMITTED_MODES = frozenset(
    (LateralControlMode.NOMINAL, LateralControlMode.DEGRADED)
)

# Mission 8 permits autonomy in exactly these two states.
_PERMITTED_SAFETY_STATES = frozenset((SafetyState.NOMINAL, SafetyState.DEGRADED))

# The only Mission 7 states that can carry lateral guidance at all.
_GUIDANCE_STATES = frozenset(
    (TemporalLaneState.TRACKING, TemporalLaneState.COASTING)
)


@dataclass(frozen=True, slots=True)
class LateralControllerConfig:
    """One small, exact lateral policy for a controller epoch.

    Defaults are deliberately conservative exhibition values, not tuned or
    measured vehicle parameters:

    * ``offset_gain`` and ``heading_gain`` are dimensionless multipliers on the
      already-normalized Mission 5 scalars.  Their ratio is the effective
      normalized image lookahead described in the module docstring.
    * ``nominal_max_steering`` bounds a fully healthy request; the default keeps
      requested intent well inside the normalized range.
    * ``degraded_max_steering`` must be no larger, so degraded evidence can
      never obtain nominal authority.
    * ``max_steering_rate_per_second`` bounds how fast requested steering may
      change per real second of caller-supplied monotonic time.
    * ``steering_deadband`` suppresses tiny requests around centre; the default
      of ``0.0`` leaves it inert.
    """

    offset_gain: float = 1.20
    heading_gain: float = 0.40
    nominal_max_steering: float = 0.35
    degraded_max_steering: float = 0.15
    max_steering_rate_per_second: float = 1.50
    steering_deadband: float = 0.0

    def __post_init__(self) -> None:
        offset_gain = _finite_real(self.offset_gain, "offset_gain")
        if not 0.0 < offset_gain <= MAX_LATERAL_GAIN:
            raise ValueError(f"offset_gain must be in (0.0, {MAX_LATERAL_GAIN}]")
        heading_gain = _finite_real(self.heading_gain, "heading_gain")
        if not 0.0 <= heading_gain <= MAX_LATERAL_GAIN:
            raise ValueError(f"heading_gain must be in [0.0, {MAX_LATERAL_GAIN}]")
        nominal = _finite_real(self.nominal_max_steering, "nominal_max_steering")
        if not 0.0 < nominal <= 1.0:
            raise ValueError("nominal_max_steering must be in (0.0, 1.0]")
        degraded = _finite_real(self.degraded_max_steering, "degraded_max_steering")
        if not 0.0 < degraded <= nominal:
            raise ValueError(
                "degraded_max_steering must be in (0.0, nominal_max_steering]"
            )
        rate = _finite_real(
            self.max_steering_rate_per_second, "max_steering_rate_per_second"
        )
        if not 0.0 < rate <= MAX_STEERING_RATE_PER_SECOND:
            raise ValueError(
                "max_steering_rate_per_second must be in "
                f"(0.0, {MAX_STEERING_RATE_PER_SECOND}]"
            )
        deadband = _finite_real(self.steering_deadband, "steering_deadband")
        if not 0.0 <= deadband < degraded:
            raise ValueError(
                "steering_deadband must be in [0.0, degraded_max_steering)"
            )
        object.__setattr__(self, "offset_gain", offset_gain)
        object.__setattr__(self, "heading_gain", heading_gain)
        object.__setattr__(self, "nominal_max_steering", nominal)
        object.__setattr__(self, "degraded_max_steering", degraded)
        object.__setattr__(self, "max_steering_rate_per_second", rate)
        object.__setattr__(self, "steering_deadband", deadband)


DEFAULT_LATERAL_CONTROLLER_CONFIG = LateralControllerConfig()


@dataclass(frozen=True, slots=True)
class LateralControlRequest:
    """One immutable lateral steering intent and the evidence behind it.

    ``lateral_allowed`` is the single question a downstream Mission 10 command
    layer must obey before turning this into any vehicle command.  Mission 9
    itself issues none.  The mode/reason/permission triple is validated here, so
    no code path can publish nonzero steering under a fail-closed explanation.
    """

    mode: LateralControlMode
    reason: LateralControlReason
    lateral_allowed: bool
    steering: float = 0.0
    raw_steering: float = 0.0
    steering_authority: float = 0.0
    clamped: bool = False
    rate_limited: bool = False
    evaluation_monotonic: float | None = None
    source_stamp: SampleStamp | None = None
    snapshot_revision: int | None = None
    lateral_error: float | None = None
    heading_error: float | None = None
    safety_state: SafetyState | None = None
    temporal_state: TemporalLaneState | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, LateralControlReason):
            raise ValueError("reason must be a LateralControlReason")
        if not isinstance(self.mode, LateralControlMode):
            raise ValueError("mode must be a LateralControlMode")
        if self.mode is not _REASON_MODE[self.reason]:
            raise ValueError("mode does not match its lateral control reason")
        allowed = _builtin_bool(self.lateral_allowed, "lateral_allowed")
        if allowed is not (self.mode in _PERMITTED_MODES):
            raise ValueError("lateral_allowed does not match the control mode")
        steering = _bounded_real(self.steering, "steering", low=-1.0, high=1.0)
        raw_steering = _finite_real(self.raw_steering, "raw_steering")
        authority = _bounded_real(
            self.steering_authority, "steering_authority", low=0.0, high=1.0
        )
        clamped = _builtin_bool(self.clamped, "clamped")
        rate_limited = _builtin_bool(self.rate_limited, "rate_limited")
        evaluated = self.evaluation_monotonic
        if evaluated is not None:
            evaluated = _finite_real(evaluated, "evaluation_monotonic")
        stamp = (
            None if self.source_stamp is None else _canonical_stamp(self.source_stamp)
        )
        revision = self.snapshot_revision
        if revision is not None:
            revision = _nonnegative_builtin_int(revision, "snapshot_revision")
        lateral_error = self.lateral_error
        if lateral_error is not None:
            lateral_error = _bounded_real(
                lateral_error, "lateral_error", low=-1.0, high=1.0
            )
        heading_error = self.heading_error
        if heading_error is not None:
            heading_error = _bounded_real(
                heading_error, "heading_error", low=-1.0, high=1.0
            )
        if self.safety_state is not None and not isinstance(
            self.safety_state, SafetyState
        ):
            raise ValueError("safety_state must be a SafetyState or None")
        if self.temporal_state is not None and not isinstance(
            self.temporal_state, TemporalLaneState
        ):
            raise ValueError("temporal_state must be a TemporalLaneState or None")

        if allowed:
            if abs(steering) > authority:
                raise ValueError("steering cannot exceed its own authority")
            if (
                evaluated is None
                or stamp is None
                or revision is None
                or lateral_error is None
                or heading_error is None
            ):
                raise ValueError("a permitted request requires complete evidence")
            if self.safety_state not in _PERMITTED_SAFETY_STATES:
                raise ValueError("a permitted request requires permitted safety state")
            if self.temporal_state not in _GUIDANCE_STATES:
                raise ValueError("a permitted request requires guidance-bearing state")
            if (
                self.mode is LateralControlMode.NOMINAL
                and self.safety_state is not SafetyState.NOMINAL
            ):
                raise ValueError("nominal lateral mode requires nominal safety state")
        else:
            if steering != 0.0 or raw_steering != 0.0 or authority != 0.0:
                raise ValueError("a refused request cannot carry steering evidence")
            if clamped or rate_limited:
                raise ValueError("a refused request cannot report shaping activity")
            if lateral_error is not None or heading_error is not None:
                raise ValueError("a refused request cannot expose lane guidance")
        if self.reason is LateralControlReason.NOT_EVALUATED and any(
            value is not None
            for value in (evaluated, stamp, revision, self.safety_state, self.temporal_state)
        ):
            raise ValueError("an unevaluated request cannot expose any evidence")

        object.__setattr__(self, "lateral_allowed", allowed)
        object.__setattr__(self, "steering", steering)
        object.__setattr__(self, "raw_steering", raw_steering)
        object.__setattr__(self, "steering_authority", authority)
        object.__setattr__(self, "clamped", clamped)
        object.__setattr__(self, "rate_limited", rate_limited)
        object.__setattr__(self, "evaluation_monotonic", evaluated)
        object.__setattr__(self, "source_stamp", stamp)
        object.__setattr__(self, "snapshot_revision", revision)
        object.__setattr__(self, "lateral_error", lateral_error)
        object.__setattr__(self, "heading_error", heading_error)

    @property
    def source_frame(self) -> int | None:
        """The CARLA source frame this request was computed from, if any."""
        stamp = self.source_stamp
        return None if stamp is None else stamp.frame

    @property
    def source_simulation_timestamp(self) -> float | None:
        """The source simulation time this request was computed from, if any."""
        stamp = self.source_stamp
        return None if stamp is None else stamp.sim_time


@dataclass(frozen=True, slots=True)
class LateralControllerMetrics:
    """One immutable view of the bounded scalar controller counters.

    These are diagnostics.  No control behavior is derived from them, and they
    are read from the single owning consumer, not concurrently.
    """

    evaluations: int
    enabled_requests: int
    disabled_requests: int
    nominal_requests: int
    degraded_requests: int
    safety_rejects: int
    identity_rejects: int
    geometry_rejects: int
    malformed_rejects: int
    source_rejects: int
    time_rejects: int
    clamp_activations: int
    rate_limit_activations: int
    resets: int
    last_accepted_source_stamp: SampleStamp | None


class _EvidenceError(Exception):
    """Internal marker for evidence Mission 9 must fail closed on."""


class _MalformedEvidence(_EvidenceError):
    pass


class _IdentityMismatch(_EvidenceError):
    pass


@dataclass(frozen=True, slots=True)
class _AcceptedEvidence:
    """The only per-result evidence the controller reads, all canonical."""

    source_stamp: SampleStamp
    snapshot_revision: int
    temporal_state: TemporalLaneState
    extrapolated: bool
    confidence: float
    center_offset: float | None
    heading_proxy: float | None


def _temporal_guidance(
    evidence: _AcceptedEvidence,
) -> tuple[float, float] | None:
    """Return the only lane geometry Mission 9 will ever steer from.

    This is the single narrow extraction point.  It answers with Mission 7's
    *temporal* complete-lane scalars, never with the raw Mission 5 observation,
    so a ``COASTING`` estimate whose current raw frame detected nothing still
    yields bounded extrapolated guidance while an unusable raw frame never
    reaches the control law.  ``None`` means "there is not enough accepted
    geometry to steer", which is the answer for ``LOST``, ``INPUT_UNUSABLE``,
    ``UNINITIALIZED``, and for any partial track: Mission 7 exposes complete
    ``center_offset``/``heading_proxy`` only when *both* boundaries are present,
    and Mission 9 never invents the missing one.
    """
    if evidence.temporal_state not in _GUIDANCE_STATES:
        return None
    center_offset = evidence.center_offset
    heading_proxy = evidence.heading_proxy
    if center_offset is None or heading_proxy is None:
        return None
    return center_offset, heading_proxy


def _result_evidence(result: PerceptionRuntimeResult) -> _AcceptedEvidence:
    """Canonicalize one published result and prove its identity is coherent.

    Every caller-visible attribute is read exactly once, so a subclass property
    cannot answer one value to a gate and another to the control law.  This
    repeats the accepted Mission 8 outer/carrier/estimate binding rather than
    trusting it, because Mission 9 is handed the result and the decision
    separately and must prove for itself that they describe one camera frame.
    """
    observation = result.observation
    if not isinstance(observation, TemporalLaneObservation):
        raise _MalformedEvidence("result must carry a TemporalLaneObservation")
    estimate = observation.temporal_estimate
    if not isinstance(estimate, TemporalLaneEstimate):
        raise _MalformedEvidence("carrier must expose a TemporalLaneEstimate")

    try:
        outer_stamp = _canonical_stamp(result.source_stamp, "result.source_stamp")
        outer_revision = _nonnegative_builtin_int(
            result.snapshot_revision, "result.snapshot_revision"
        )
        carrier_state = observation.state
        if not isinstance(carrier_state, LaneDetectionState):
            raise _MalformedEvidence("carrier state must be a LaneDetectionState")
        raw_revision = _nonnegative_builtin_int(
            observation.snapshot_revision, "observation.snapshot_revision"
        )
        raw_frame = observation.source_frame
        raw_sim_time = observation.source_simulation_timestamp
        raw_monotonic = observation.source_receive_monotonic
        presence = (
            raw_frame is not None,
            raw_sim_time is not None,
            raw_monotonic is not None,
        )
        if any(presence) and not all(presence):
            raise _MalformedEvidence("raw source evidence must be complete or absent")
        if all(presence):
            raw_stamp = _canonical_stamp(
                SampleStamp(
                    frame=raw_frame,
                    sim_time=raw_sim_time,
                    monotonic=raw_monotonic,
                ),
                "observation source stamp",
            )
        else:
            raw_stamp = None
            if carrier_state is not LaneDetectionState.INPUT_UNUSABLE:
                raise _MalformedEvidence(
                    "only raw INPUT_UNUSABLE may omit its source stamp"
                )
        estimate_revision = estimate.snapshot_revision
        if estimate_revision is not None:
            estimate_revision = _nonnegative_builtin_int(
                estimate_revision, "estimate.snapshot_revision"
            )
        raw_estimate_stamp = estimate.source_stamp
        estimate_stamp = (
            None
            if raw_estimate_stamp is None
            else _canonical_stamp(raw_estimate_stamp, "estimate.source_stamp")
        )
        temporal_state = estimate.state
        if not isinstance(temporal_state, TemporalLaneState):
            raise _MalformedEvidence("estimate state must be a TemporalLaneState")
        raw_state = estimate.raw_state
        if raw_state is not None and not isinstance(raw_state, LaneDetectionState):
            raise _MalformedEvidence("estimate raw state must be a LaneDetectionState")
        extrapolated = estimate.extrapolated
        if type(extrapolated) is not bool:
            raise _MalformedEvidence("estimate extrapolated must be a built-in bool")
        confidence = _bounded_real(
            estimate.confidence, "estimate.confidence", low=0.0, high=1.0
        )
        center_offset = estimate.center_offset
        if center_offset is not None:
            center_offset = _bounded_real(
                center_offset, "estimate.center_offset", low=-1.0, high=1.0
            )
        heading_proxy = estimate.heading_proxy
        if heading_proxy is not None:
            heading_proxy = _bounded_real(
                heading_proxy, "estimate.heading_proxy", low=-1.0, high=1.0
            )
        # Mission 7 publishes both complete-lane scalars together or not at all.
        # Half a pair is malformed evidence, never a licence to assume the rest.
        if (center_offset is None) is not (heading_proxy is None):
            raise _MalformedEvidence("complete-lane scalars must both be present")
    except _EvidenceError:
        raise
    except ValueError as exc:
        raise _MalformedEvidence(str(exc)) from exc

    if raw_revision != outer_revision:
        raise _IdentityMismatch("carrier snapshot revision differs from the result")
    if raw_stamp is not None and not _same_stamp(raw_stamp, outer_stamp):
        raise _IdentityMismatch("carrier source stamp differs from the result")
    if raw_state is not carrier_state:
        raise _IdentityMismatch("carrier and estimate disagree on the raw state")
    if estimate_revision != outer_revision:
        raise _IdentityMismatch("estimate snapshot revision differs from the result")
    if estimate_stamp is None:
        if temporal_state is not TemporalLaneState.INPUT_UNUSABLE:
            raise _IdentityMismatch("only an unusable estimate may omit its stamp")
    elif not _same_stamp(estimate_stamp, outer_stamp):
        raise _IdentityMismatch("estimate source stamp differs from the result")

    return _AcceptedEvidence(
        source_stamp=outer_stamp,
        snapshot_revision=outer_revision,
        temporal_state=temporal_state,
        extrapolated=extrapolated,
        confidence=confidence,
        center_offset=center_offset,
        heading_proxy=heading_proxy,
    )


class LateralController:
    """Single-consumer, lock-free lateral steering intent for one epoch.

    ``update`` consumes one published Mission 6 result together with the
    Mission 8 decision issued for that same evidence and returns the current
    immutable :class:`LateralControlRequest`.  It creates no thread, queue,
    timer, or clock: the caller supplies host-monotonic ``now`` explicitly, from
    the same clock it gave Mission 8.

    Mission 8 remains the sole authority on whether autonomy is permitted.  This
    class never re-decides that question, never overrides it, and can only
    subtract authority: it additionally refuses to steer when the decision does
    not describe the supplied evidence, when the evidence is stale or replayed,
    or when Mission 7 carries no complete-lane geometry to steer from.
    """

    __slots__ = (
        "_clamp_activations",
        "_config",
        "_degraded_requests",
        "_disabled_requests",
        "_enabled_requests",
        "_evaluations",
        "_geometry_rejects",
        "_identity_rejects",
        "_last_accepted_stamp",
        "_malformed_rejects",
        "_nominal_requests",
        "_previous_steering",
        "_previous_time",
        "_rate_limit_activations",
        "_request",
        "_resets",
        "_safety_rejects",
        "_source_rejects",
        "_time_rejects",
    )

    def __init__(
        self,
        config: LateralControllerConfig = DEFAULT_LATERAL_CONTROLLER_CONFIG,
    ) -> None:
        if not isinstance(config, LateralControllerConfig):
            raise TypeError("config must be a LateralControllerConfig")
        # Detach every caller-owned record, including an exact frozen base
        # instance, before it can influence later control output.
        self._config = LateralControllerConfig(
            offset_gain=config.offset_gain,
            heading_gain=config.heading_gain,
            nominal_max_steering=config.nominal_max_steering,
            degraded_max_steering=config.degraded_max_steering,
            max_steering_rate_per_second=config.max_steering_rate_per_second,
            steering_deadband=config.steering_deadband,
        )
        self._evaluations = 0
        self._enabled_requests = 0
        self._disabled_requests = 0
        self._nominal_requests = 0
        self._degraded_requests = 0
        self._safety_rejects = 0
        self._identity_rejects = 0
        self._geometry_rejects = 0
        self._malformed_rejects = 0
        self._source_rejects = 0
        self._time_rejects = 0
        self._clamp_activations = 0
        self._rate_limit_activations = 0
        self._resets = 0
        self._clear_state()

    @property
    def config(self) -> LateralControllerConfig:
        return LateralControllerConfig(
            offset_gain=self._config.offset_gain,
            heading_gain=self._config.heading_gain,
            nominal_max_steering=self._config.nominal_max_steering,
            degraded_max_steering=self._config.degraded_max_steering,
            max_steering_rate_per_second=self._config.max_steering_rate_per_second,
            steering_deadband=self._config.steering_deadband,
        )

    @property
    def latest_request(self) -> LateralControlRequest:
        return self._request

    @property
    def lateral_allowed(self) -> bool:
        return self._request.lateral_allowed

    @property
    def metrics(self) -> LateralControllerMetrics:
        stamp = self._last_accepted_stamp
        return LateralControllerMetrics(
            evaluations=self._evaluations,
            enabled_requests=self._enabled_requests,
            disabled_requests=self._disabled_requests,
            nominal_requests=self._nominal_requests,
            degraded_requests=self._degraded_requests,
            safety_rejects=self._safety_rejects,
            identity_rejects=self._identity_rejects,
            geometry_rejects=self._geometry_rejects,
            malformed_rejects=self._malformed_rejects,
            source_rejects=self._source_rejects,
            time_rejects=self._time_rejects,
            clamp_activations=self._clamp_activations,
            rate_limit_activations=self._rate_limit_activations,
            resets=self._resets,
            last_accepted_source_stamp=(
                None if stamp is None else _canonical_stamp(stamp)
            ),
        )

    def reset(self) -> LateralControlRequest:
        """Begin a fresh controller epoch; lifetime counters remain truthful.

        No previous command, rate-limit time, or source baseline crosses the
        boundary, and the published request returns to neutral and refused.
        Repeated calls are idempotent apart from the reset counter.
        """
        self._resets += 1
        self._clear_state()
        return self._request

    def _clear_state(self) -> None:
        self._previous_steering: float | None = None
        self._previous_time: float | None = None
        self._last_accepted_stamp: SampleStamp | None = None
        self._request = LateralControlRequest(
            mode=LateralControlMode.DISABLED,
            reason=LateralControlReason.NOT_EVALUATED,
            lateral_allowed=False,
        )

    def update(
        self,
        result: PerceptionRuntimeResult,
        decision: SafetyDecision,
        *,
        now: float,
    ) -> LateralControlRequest:
        """Produce one bounded lateral steering intent at control time ``now``.

        ``result`` must be the published Mission 6 result and ``decision`` the
        Mission 8 decision issued for that same result.  Both are read but never
        retained.  Sample ``now`` from the same host-monotonic clock Mission 8
        was given, and never let it regress: a regressed or non-finite control
        clock fails closed and clears the rate-limit baseline.
        """
        if not isinstance(result, PerceptionRuntimeResult):
            raise TypeError("result must be a PerceptionRuntimeResult")
        if not isinstance(decision, SafetyDecision):
            raise TypeError("decision must be a SafetyDecision")
        self._evaluations += 1

        # 1. Control time.  Without a trustworthy clock no slew statement can be
        #    made at all, so the rate-limit baseline is discarded rather than
        #    carried across the discontinuity.
        try:
            evaluated = _finite_real(now, "now")
        except ValueError:
            return self._refuse(LateralControlReason.TIME_INVALID, None)
        previous_time = self._previous_time
        if previous_time is not None and evaluated < previous_time:
            return self._refuse(LateralControlReason.TIME_INVALID, None)

        # 2/3. Structure, then internal outer/carrier/estimate identity.
        #      Reading published evidence is pure attribute access and
        #      validation, so any exception at all means the evidence could not
        #      be read and must fail closed rather than escape to the caller.
        try:
            evidence = _result_evidence(result)
        except _IdentityMismatch:
            return self._refuse(LateralControlReason.IDENTITY_MISMATCH, evaluated)
        except Exception:
            return self._refuse(LateralControlReason.EVIDENCE_MALFORMED, evaluated)

        # 4. Mission 6 source order over the bound stamp.  A duplicate is a
        #    legitimate re-evaluation of the newest evidence and is accepted; an
        #    older or reordered replay is refused and must not move the baseline.
        stamp = evidence.source_stamp
        baseline = self._last_accepted_stamp
        if baseline is not None:
            duplicate = (
                stamp.frame == baseline.frame and stamp.sim_time == baseline.sim_time
            )
            if not duplicate and (
                stamp.frame <= baseline.frame or stamp.sim_time < baseline.sim_time
            ):
                return self._refuse(
                    LateralControlReason.SOURCE_REJECTED,
                    evaluated,
                    evidence=evidence,
                )
        self._last_accepted_stamp = stamp

        # 5. The Mission 8 gate, before any steering value is computed at all.
        #    Each attribute is read exactly once so a hostile subclass cannot
        #    answer differently here and below.
        safety_allowed = decision.autonomy_allowed
        safety_state = decision.state
        safety_latched = decision.latched
        if (
            safety_allowed is not True
            or safety_state not in _PERMITTED_SAFETY_STATES
            or safety_latched is not False
        ):
            return self._refuse(
                LateralControlReason.SAFETY_NOT_PERMITTED,
                evaluated,
                evidence=evidence,
                safety_state=(
                    safety_state if isinstance(safety_state, SafetyState) else None
                ),
            )

        # 6. The decision must describe *this* evidence, not another frame.
        try:
            decision_stamp = decision.source_stamp
            if decision_stamp is None:
                raise _IdentityMismatch("a permitting decision requires a source stamp")
            decision_stamp = _canonical_stamp(decision_stamp, "decision.source_stamp")
            decision_revision = decision.snapshot_revision
            if decision_revision is None:
                raise _IdentityMismatch("a permitting decision requires a revision")
            decision_revision = _nonnegative_builtin_int(
                decision_revision, "decision.snapshot_revision"
            )
            decision_temporal_state = decision.temporal_state
            decision_confidence = decision.temporal_confidence
            if decision_confidence is None:
                raise _IdentityMismatch("a permitting decision requires confidence")
            decision_confidence = _bounded_real(
                decision_confidence, "decision.temporal_confidence", low=0.0, high=1.0
            )
        except _IdentityMismatch:
            return self._refuse(
                LateralControlReason.IDENTITY_MISMATCH,
                evaluated,
                evidence=evidence,
                safety_state=safety_state,
            )
        except Exception:
            return self._refuse(
                LateralControlReason.EVIDENCE_MALFORMED,
                evaluated,
                evidence=evidence,
                safety_state=safety_state,
            )
        if (
            not _same_stamp(decision_stamp, stamp)
            or decision_revision != evidence.snapshot_revision
            or decision_temporal_state is not evidence.temporal_state
            or decision_confidence != evidence.confidence
        ):
            return self._refuse(
                LateralControlReason.IDENTITY_MISMATCH,
                evaluated,
                evidence=evidence,
                safety_state=safety_state,
            )

        # 7. Geometry.  A degraded permission is never a licence to invent one.
        guidance = _temporal_guidance(evidence)
        if guidance is None:
            return self._refuse(
                LateralControlReason.INSUFFICIENT_GEOMETRY,
                evaluated,
                evidence=evidence,
                safety_state=safety_state,
            )
        lateral_error, heading_error = guidance

        # 8. Mode and authority.  Nominal authority additionally requires live,
        #    non-extrapolated tracking, so coasting can only ever be degraded
        #    even if Mission 8 were to call it nominal.
        if (
            safety_state is SafetyState.NOMINAL
            and evidence.temporal_state is TemporalLaneState.TRACKING
            and not evidence.extrapolated
        ):
            reason = LateralControlReason.NOMINAL_TRACKING
            authority = self._config.nominal_max_steering
        else:
            reason = (
                LateralControlReason.DEGRADED_COASTING
                if evidence.temporal_state is TemporalLaneState.COASTING
                else LateralControlReason.DEGRADED_TRACKING
            )
            authority = self._config.degraded_max_steering

        raw_steering = (
            self._config.offset_gain * lateral_error
            - self._config.heading_gain * heading_error
        )
        if not math.isfinite(raw_steering):  # pragma: no cover - bounded inputs
            return self._refuse(
                LateralControlReason.EVIDENCE_MALFORMED,
                evaluated,
                evidence=evidence,
                safety_state=safety_state,
            )

        # The deadband is a separate shaping step from the authority clamp, so
        # suppressing a tiny request must not be counted as a clamp activation.
        deadbanded = (
            0.0
            if abs(raw_steering) < self._config.steering_deadband
            else raw_steering
        )
        target = deadbanded
        if target > authority:
            target = authority
        elif target < -authority:
            target = -authority
        clamped = target != deadbanded

        # The slew baseline is first brought inside the *current* authority, so
        # a nominal-to-degraded downgrade reduces magnitude immediately instead
        # of slewing while still exceeding the lower bound.  Both endpoints then
        # lie within the authority, so the rate-limited result does too.
        previous_steering = self._previous_steering
        rate_limited = False
        if previous_steering is None or previous_time is None:
            steering = target
        else:
            anchor = previous_steering
            if anchor > authority:
                anchor = authority
            elif anchor < -authority:
                anchor = -authority
            allowance = self._config.max_steering_rate_per_second * (
                evaluated - previous_time
            )
            if not math.isfinite(allowance) or allowance < 0.0:
                allowance = 0.0
            if target > anchor + allowance:
                steering = anchor + allowance
                rate_limited = True
            elif target < anchor - allowance:
                steering = anchor - allowance
                rate_limited = True
            else:
                steering = target
            # Floating-point addition can leave the slewed value a few ulps
            # outside the authority even though both endpoints are inside it.
            if steering > authority:
                steering = authority
            elif steering < -authority:
                steering = -authority

        if clamped:
            self._clamp_activations += 1
        if rate_limited:
            self._rate_limit_activations += 1
        self._previous_steering = steering
        self._previous_time = evaluated
        request = LateralControlRequest(
            mode=_REASON_MODE[reason],
            reason=reason,
            lateral_allowed=True,
            steering=steering,
            raw_steering=raw_steering,
            steering_authority=authority,
            clamped=clamped,
            rate_limited=rate_limited,
            evaluation_monotonic=evaluated,
            source_stamp=stamp,
            snapshot_revision=evidence.snapshot_revision,
            lateral_error=lateral_error,
            heading_error=heading_error,
            safety_state=safety_state,
            temporal_state=evidence.temporal_state,
        )
        self._enabled_requests += 1
        if request.mode is LateralControlMode.NOMINAL:
            self._nominal_requests += 1
        else:
            self._degraded_requests += 1
        self._request = request
        return request

    def _refuse(
        self,
        reason: LateralControlReason,
        evaluated: float | None,
        *,
        evidence: _AcceptedEvidence | None = None,
        safety_state: SafetyState | None = None,
    ) -> LateralControlRequest:
        """Publish one neutral, refused request and record why.

        A refused request is exactly zero steering.  That zero is deliberately
        not rate limited: reducing intent to neutral is always permitted, and a
        safety refusal must not be slewed into over several control steps.  The
        slew baseline therefore becomes that published zero, so re-enabling
        ramps from neutral rather than from a stale command.  A refusal caused
        by an untrustworthy clock instead discards the baseline entirely.
        """
        if reason is LateralControlReason.TIME_INVALID:
            self._time_rejects += 1
        elif reason is LateralControlReason.EVIDENCE_MALFORMED:
            self._malformed_rejects += 1
        elif reason is LateralControlReason.IDENTITY_MISMATCH:
            self._identity_rejects += 1
        elif reason is LateralControlReason.SOURCE_REJECTED:
            self._source_rejects += 1
        elif reason is LateralControlReason.SAFETY_NOT_PERMITTED:
            self._safety_rejects += 1
        else:
            assert reason is LateralControlReason.INSUFFICIENT_GEOMETRY
            self._geometry_rejects += 1

        if evaluated is None:
            self._previous_steering = None
            self._previous_time = None
        else:
            self._previous_steering = 0.0
            self._previous_time = evaluated
        request = LateralControlRequest(
            mode=LateralControlMode.DISABLED,
            reason=reason,
            lateral_allowed=False,
            evaluation_monotonic=evaluated,
            source_stamp=None if evidence is None else evidence.source_stamp,
            snapshot_revision=None if evidence is None else evidence.snapshot_revision,
            safety_state=safety_state,
            temporal_state=None if evidence is None else evidence.temporal_state,
        )
        self._disabled_requests += 1
        self._request = request
        return request
