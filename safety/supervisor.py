"""Bounded, deterministic safety supervision over published perception results.

Mission 8 answers exactly one question: may autonomous control be permitted
right now?  It issues no steering, throttle, brake, or CARLA command; it owns
no actor, worker, queue, timer, or lock; and it never inspects Mission 7
tracker internals or the deliberately non-atomic Mission 7 metrics.

The supervisor consumes only immutable published evidence:

* one ``PerceptionRuntimeResult`` whose observation is a Mission 7
  ``TemporalLaneObservation`` and therefore carries its atomic
  ``TemporalLaneEstimate``,
* one bounded ``ComponentHealth`` record copied out of the Mission 6A runtime
  and Mission 6B bridge; the argument is optional per call, but a supervisor
  epoch that has never received one holds component health *unknown* and
  refuses autonomy, because unknown is not healthy,
* one caller-supplied host-monotonic safety time, sampled after the result is
  read so that ``now >= result.source_stamp.monotonic``.

It publishes one immutable ``SafetyDecision``.  Mission 7 does not age its own
track when no observation arrives, so the supervisor owns result freshness:
:meth:`SafetySupervisor.evaluate` advances safety time with no new perception
input at all and expires previously usable evidence deterministically.

The supervisor is single-consumer and lock-free.  Its retained state is O(1):
one decision, one canonical accepted source identity, a handful of scalars, and
bounded lifetime counters.  There is no result, frame, or decision history.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from enum import Enum

from perception.live_bridge import PerceptionLiveBridgeState
from perception.runtime import PerceptionRuntimeResult, PerceptionRuntimeState
from perception.state import LaneDetectionState
from perception.temporal import (
    TemporalLaneEstimate,
    TemporalLaneObservation,
    TemporalLaneState,
)
from telemetry.state import SampleStamp


MAX_SAFETY_RESULT_AGE_SECONDS = 60.0
MAX_REQUIRED_HEALTHY_RESULTS = 1_000_000


__all__ = (
    "DEFAULT_SAFETY_SUPERVISOR_CONFIG",
    "MAX_REQUIRED_HEALTHY_RESULTS",
    "MAX_SAFETY_RESULT_AGE_SECONDS",
    "ComponentHealth",
    "SafetyDecision",
    "SafetyReason",
    "SafetyState",
    "SafetySupervisor",
    "SafetySupervisorConfig",
    "SafetySupervisorMetrics",
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


def _unit_real(value: object, name: str) -> float:
    result = _finite_real(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0]")
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


class SafetyState(Enum):
    """The five bounded safety-permission states."""

    INITIALIZING = "initializing"
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    FAIL_SAFE = "fail_safe"


class SafetyReason(Enum):
    """One bounded explanation for the current safety state."""

    STARTUP_NO_RESULT = "startup_no_result"
    NOMINAL_TRACKING = "nominal_tracking"
    DEGRADED_PARTIAL_TRACKING = "degraded_partial_tracking"
    DEGRADED_COASTING = "degraded_coasting"
    DEGRADED_LOW_CONFIDENCE = "degraded_low_confidence"
    RECOVERY_PENDING = "recovery_pending"
    CONFIDENCE_BELOW_THRESHOLD = "confidence_below_threshold"
    RESULT_STALE = "result_stale"
    TEMPORAL_LOST = "temporal_lost"
    TEMPORAL_INPUT_UNUSABLE = "temporal_input_unusable"
    TEMPORAL_UNINITIALIZED = "temporal_uninitialized"
    COMPONENT_NOT_READY = "component_not_ready"
    RUNTIME_FAULT = "runtime_fault"
    BRIDGE_FAULT = "bridge_fault"
    IDENTITY_MISMATCH = "identity_mismatch"
    EVIDENCE_MALFORMED = "evidence_malformed"
    CLOCK_REGRESSED = "clock_regressed"


# Every reason belongs to exactly one state, so a decision can never advertise a
# permitted state under an unsafe explanation.
_REASON_STATE: dict[SafetyReason, SafetyState] = {
    SafetyReason.STARTUP_NO_RESULT: SafetyState.INITIALIZING,
    SafetyReason.NOMINAL_TRACKING: SafetyState.NOMINAL,
    SafetyReason.DEGRADED_PARTIAL_TRACKING: SafetyState.DEGRADED,
    SafetyReason.DEGRADED_COASTING: SafetyState.DEGRADED,
    SafetyReason.DEGRADED_LOW_CONFIDENCE: SafetyState.DEGRADED,
    SafetyReason.RECOVERY_PENDING: SafetyState.RECOVERING,
    SafetyReason.CONFIDENCE_BELOW_THRESHOLD: SafetyState.FAIL_SAFE,
    SafetyReason.RESULT_STALE: SafetyState.FAIL_SAFE,
    SafetyReason.TEMPORAL_LOST: SafetyState.FAIL_SAFE,
    SafetyReason.TEMPORAL_INPUT_UNUSABLE: SafetyState.FAIL_SAFE,
    SafetyReason.TEMPORAL_UNINITIALIZED: SafetyState.FAIL_SAFE,
    SafetyReason.COMPONENT_NOT_READY: SafetyState.FAIL_SAFE,
    SafetyReason.RUNTIME_FAULT: SafetyState.FAIL_SAFE,
    SafetyReason.BRIDGE_FAULT: SafetyState.FAIL_SAFE,
    SafetyReason.IDENTITY_MISMATCH: SafetyState.FAIL_SAFE,
    SafetyReason.EVIDENCE_MALFORMED: SafetyState.FAIL_SAFE,
    SafetyReason.CLOCK_REGRESSED: SafetyState.FAIL_SAFE,
}

# Autonomy is permitted in exactly two states and nowhere else.
_PERMITTED_STATES = frozenset((SafetyState.NOMINAL, SafetyState.DEGRADED))

# Hard failures latch until an explicit supervisor reset / new epoch.  Every
# other unsafe reason is a soft loss handled by recovery hysteresis.
_HARD_REASONS = frozenset(
    (
        SafetyReason.RUNTIME_FAULT,
        SafetyReason.BRIDGE_FAULT,
        SafetyReason.IDENTITY_MISMATCH,
        SafetyReason.EVIDENCE_MALFORMED,
        SafetyReason.CLOCK_REGRESSED,
    )
)


@dataclass(frozen=True, slots=True)
class SafetySupervisorConfig:
    """One small, exact safety policy for a supervisor epoch.

    Defaults are derived from the accepted Mission 5/6 contracts rather than
    copied from Mission 7's miss/coast policy:

    * ``max_result_age_seconds`` allows one accepted 10 Hz runtime period
      (``0.10 s``) plus the entire accepted Mission 6A pending budget
      (``0.25 s``).  Three consecutively missing camera frames therefore expire
      autonomy.  The bound is inclusive, matching the accepted Mission 6A
      pending-age and Mission 7 coast-age conventions.
    * ``min_nominal_confidence`` is the accepted Mission 5
      ``detection_confidence`` gate, so autonomy needs smoothed evidence that
      still meets the bar Mission 5 requires to call a lane ``DETECTED``.
    * ``min_degraded_confidence`` is the accepted Mission 5
      ``min_boundary_confidence`` gate, the weakest boundary Mission 5 will
      publish at all.
    * ``required_healthy_results`` is three strong results, about ``0.30 s`` of
      uninterrupted 10 Hz tracking, before autonomy resumes after a soft loss.
    """

    max_result_age_seconds: float = 0.35
    min_nominal_confidence: float = 0.65
    min_degraded_confidence: float = 0.40
    required_healthy_results: int = 3

    def __post_init__(self) -> None:
        max_age = _finite_real(self.max_result_age_seconds, "max_result_age_seconds")
        if not 0.0 < max_age <= MAX_SAFETY_RESULT_AGE_SECONDS:
            raise ValueError(
                "max_result_age_seconds must be in "
                f"(0.0, {MAX_SAFETY_RESULT_AGE_SECONDS}]"
            )
        nominal = _unit_real(self.min_nominal_confidence, "min_nominal_confidence")
        degraded = _unit_real(self.min_degraded_confidence, "min_degraded_confidence")
        if degraded > nominal:
            raise ValueError(
                "min_degraded_confidence cannot exceed min_nominal_confidence"
            )
        required = _nonnegative_builtin_int(
            self.required_healthy_results, "required_healthy_results"
        )
        if not 1 <= required <= MAX_REQUIRED_HEALTHY_RESULTS:
            raise ValueError(
                "required_healthy_results must be in "
                f"[1, {MAX_REQUIRED_HEALTHY_RESULTS}]"
            )
        object.__setattr__(self, "max_result_age_seconds", max_age)
        object.__setattr__(self, "min_nominal_confidence", nominal)
        object.__setattr__(self, "min_degraded_confidence", degraded)
        object.__setattr__(self, "required_healthy_results", required)


DEFAULT_SAFETY_SUPERVISOR_CONFIG = SafetySupervisorConfig()


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    """Bounded immutable copy of Mission 6 runtime and bridge health evidence.

    The supervisor never retains a runtime or bridge object.  Callers copy the
    public state and fault presence once per control step; a torn read across
    those two independent properties can only add fault evidence, so it fails
    closed.

    Supplying this record is how a caller proves component health.  Omitting it
    means "no new sample", never "the components are fine": a supervisor epoch
    that has never seen one treats component health as unknown and refuses
    autonomy.
    """

    runtime_state: PerceptionRuntimeState
    runtime_fault_present: bool = False
    bridge_state: PerceptionLiveBridgeState | None = None
    bridge_fault_present: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_state, PerceptionRuntimeState):
            raise ValueError("runtime_state must be a PerceptionRuntimeState")
        if self.bridge_state is not None and not isinstance(
            self.bridge_state, PerceptionLiveBridgeState
        ):
            raise ValueError("bridge_state must be a PerceptionLiveBridgeState or None")
        runtime_fault = _builtin_bool(
            self.runtime_fault_present, "runtime_fault_present"
        )
        bridge_fault = _builtin_bool(self.bridge_fault_present, "bridge_fault_present")
        if bridge_fault and self.bridge_state is None:
            raise ValueError("bridge fault evidence requires a bridge state")
        object.__setattr__(self, "runtime_fault_present", runtime_fault)
        object.__setattr__(self, "bridge_fault_present", bridge_fault)

    @classmethod
    def from_components(cls, runtime: object, bridge: object = None) -> "ComponentHealth":
        """Copy scalar health out of a live runtime and optional bridge."""
        runtime_state = runtime.state
        runtime_fault_present = runtime.fault is not None
        if bridge is None:
            bridge_state = None
            bridge_fault_present = False
        else:
            bridge_state = bridge.state
            bridge_fault_present = bridge.fault is not None
        return cls(
            runtime_state=runtime_state,
            runtime_fault_present=runtime_fault_present,
            bridge_state=bridge_state,
            bridge_fault_present=bridge_fault_present,
        )

    @property
    def runtime_faulted(self) -> bool:
        return (
            self.runtime_state is PerceptionRuntimeState.FAULTED
            or self.runtime_fault_present
        )

    @property
    def bridge_faulted(self) -> bool:
        return (
            self.bridge_state is PerceptionLiveBridgeState.FAULTED
            or self.bridge_fault_present
        )

    @property
    def ready(self) -> bool:
        """True only while the runtime runs and any bridge still admits input."""
        if self.runtime_state is not PerceptionRuntimeState.RUNNING:
            return False
        return (
            self.bridge_state is None
            or self.bridge_state is PerceptionLiveBridgeState.OPEN
        )


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    """One immutable safety permission and the bounded evidence behind it.

    ``autonomy_allowed`` is the single question downstream controllers must
    obey.  The state/reason/permission triple is validated here, so no code
    path can publish a permitted decision under an unsafe explanation.
    """

    state: SafetyState
    reason: SafetyReason
    autonomy_allowed: bool
    evaluation_monotonic: float | None = None
    source_stamp: SampleStamp | None = None
    snapshot_revision: int | None = None
    result_age_seconds: float | None = None
    temporal_state: TemporalLaneState | None = None
    temporal_confidence: float | None = None
    recovery_streak: int = 0
    latched: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.reason, SafetyReason):
            raise ValueError("reason must be a SafetyReason")
        if not isinstance(self.state, SafetyState):
            raise ValueError("state must be a SafetyState")
        if self.state is not _REASON_STATE[self.reason]:
            raise ValueError("state does not match its safety reason")
        allowed = _builtin_bool(self.autonomy_allowed, "autonomy_allowed")
        if allowed is not (self.state in _PERMITTED_STATES):
            raise ValueError("autonomy_allowed does not match the safety state")
        latched = _builtin_bool(self.latched, "latched")
        if latched and self.state is not SafetyState.FAIL_SAFE:
            raise ValueError("only a FAIL_SAFE decision may be latched")
        evaluated = self.evaluation_monotonic
        if evaluated is None:
            if self.state is not SafetyState.INITIALIZING:
                raise ValueError("only an initializing decision may omit its time")
        else:
            evaluated = _finite_real(evaluated, "evaluation_monotonic")
        stamp = None if self.source_stamp is None else _canonical_stamp(self.source_stamp)
        revision = self.snapshot_revision
        if revision is not None:
            revision = _nonnegative_builtin_int(revision, "snapshot_revision")
        age = self.result_age_seconds
        if age is not None:
            age = _finite_real(age, "result_age_seconds")
            if age < 0.0:
                raise ValueError("result_age_seconds must be non-negative")
        if self.temporal_state is not None and not isinstance(
            self.temporal_state, TemporalLaneState
        ):
            raise ValueError("temporal_state must be a TemporalLaneState or None")
        confidence = self.temporal_confidence
        if confidence is not None:
            confidence = _unit_real(confidence, "temporal_confidence")
        streak = _nonnegative_builtin_int(self.recovery_streak, "recovery_streak")
        if self.state is SafetyState.INITIALIZING and any(
            value is not None
            for value in (stamp, revision, age, self.temporal_state, confidence)
        ):
            raise ValueError("an initializing decision cannot expose result evidence")
        object.__setattr__(self, "autonomy_allowed", allowed)
        object.__setattr__(self, "latched", latched)
        object.__setattr__(self, "evaluation_monotonic", evaluated)
        object.__setattr__(self, "source_stamp", stamp)
        object.__setattr__(self, "snapshot_revision", revision)
        object.__setattr__(self, "result_age_seconds", age)
        object.__setattr__(self, "temporal_confidence", confidence)
        object.__setattr__(self, "recovery_streak", streak)


@dataclass(frozen=True, slots=True)
class SafetySupervisorMetrics:
    """One immutable view of the bounded scalar supervisor counters.

    These are diagnostics.  No safety behavior is derived from them, and they
    are read from the single owning consumer, not concurrently.
    """

    evaluations: int
    updates_received: int
    new_results_accepted: int
    duplicates_rejected: int
    out_of_order_rejected: int
    initializing_decisions: int
    nominal_decisions: int
    degraded_decisions: int
    recovering_decisions: int
    fail_safe_decisions: int
    stale_expirations: int
    component_fault_trips: int
    identity_mismatch_trips: int
    malformed_evidence_trips: int
    clock_regression_trips: int
    recovery_sequences: int
    resets: int
    last_accepted_source_stamp: SampleStamp | None


class _EvidenceError(Exception):
    """Internal marker for evidence the supervisor must fail closed on."""


class _MalformedEvidence(_EvidenceError):
    pass


class _IdentityMismatch(_EvidenceError):
    pass


@dataclass(frozen=True, slots=True)
class _AcceptedEvidence:
    """The only per-result evidence the supervisor retains, all canonical."""

    source_stamp: SampleStamp
    snapshot_revision: int
    reference_monotonic: float
    temporal_state: TemporalLaneState
    raw_state: LaneDetectionState | None
    confidence: float


def _temporal_disposition(
    state: TemporalLaneState,
    raw_state: LaneDetectionState | None,
    confidence: float,
    config: SafetySupervisorConfig,
) -> SafetyReason:
    """Map fresh, coherent Mission 7 evidence onto one bounded safety reason."""
    if state is TemporalLaneState.TRACKING:
        if raw_state is LaneDetectionState.DETECTED:
            if confidence >= config.min_nominal_confidence:
                return SafetyReason.NOMINAL_TRACKING
            if confidence >= config.min_degraded_confidence:
                return SafetyReason.DEGRADED_LOW_CONFIDENCE
            return SafetyReason.CONFIDENCE_BELOW_THRESHOLD
        if raw_state is LaneDetectionState.PARTIAL:
            if confidence >= config.min_degraded_confidence:
                return SafetyReason.DEGRADED_PARTIAL_TRACKING
            return SafetyReason.CONFIDENCE_BELOW_THRESHOLD
        # Mission 7 forbids any other raw state under TRACKING.  Reaching here
        # means the published evidence violates its own invariant.
        return SafetyReason.EVIDENCE_MALFORMED
    if state is TemporalLaneState.COASTING:
        if confidence >= config.min_degraded_confidence:
            return SafetyReason.DEGRADED_COASTING
        return SafetyReason.CONFIDENCE_BELOW_THRESHOLD
    if state is TemporalLaneState.LOST:
        return SafetyReason.TEMPORAL_LOST
    if state is TemporalLaneState.INPUT_UNUSABLE:
        return SafetyReason.TEMPORAL_INPUT_UNUSABLE
    # Mission 7 refuses to build a carrier around an UNINITIALIZED estimate, so
    # this is defence in depth against evidence assembled outside that path.
    return SafetyReason.TEMPORAL_UNINITIALIZED


def _validated_health(health: object) -> None:
    if health is not None and not isinstance(health, ComponentHealth):
        raise TypeError("health must be a ComponentHealth or None")


def _result_evidence(result: PerceptionRuntimeResult) -> _AcceptedEvidence:
    """Canonicalize one published result and prove its identity is coherent.

    The outer Mission 6 result, the inner raw Mission 5 carrier, and the
    Mission 7 estimate must all describe the same camera frame and snapshot.
    Mission 7 already enforces raw-to-estimate coherence inside the carrier;
    this gate adds the outer-to-carrier comparison that a temporal pipeline
    reused across runtime epochs without reset can violate.
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
        # Every caller-visible attribute is read exactly once, so a subclass
        # property cannot answer one value to the gate and another to policy.
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
        confidence = _unit_real(estimate.confidence, "estimate.confidence")
    except _EvidenceError:
        raise
    except ValueError as exc:
        raise _MalformedEvidence(str(exc)) from exc

    if raw_revision != outer_revision:
        raise _IdentityMismatch("carrier snapshot revision differs from the result")
    if raw_stamp is not None and not _same_stamp(raw_stamp, outer_stamp):
        raise _IdentityMismatch("carrier source stamp differs from the result")
    if raw_state is not carrier_state:
        # Mission 7's constructor already pairs these.  Verifying it here means
        # the safety policy never reads a detection state the raw carrier
        # disagrees with, whatever assembled the pair.
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
        reference_monotonic=outer_stamp.monotonic,
        temporal_state=temporal_state,
        raw_state=raw_state,
        confidence=confidence,
    )


class SafetySupervisor:
    """Single-consumer, lock-free safety-permission gate for one epoch.

    ``update`` consumes one newly published runtime result; ``evaluate``
    advances safety time with no new result at all.  Both return the current
    immutable :class:`SafetyDecision`.  Neither creates a thread, queue, timer,
    or clock: the caller supplies host-monotonic time explicitly.

    Downstream Mission 9/10 controllers must require
    ``decision.autonomy_allowed is True`` before issuing any vehicle command.
    Mission 8 itself issues none.
    """

    __slots__ = (
        "_accepted",
        "_clock_regression_trips",
        "_component_fault_trips",
        "_components_ready",
        "_config",
        "_decision",
        "_degraded_decisions",
        "_duplicates_rejected",
        "_evaluations",
        "_fail_safe_decisions",
        "_health_observed",
        "_healthy_streak",
        "_identity_mismatch_trips",
        "_initializing_decisions",
        "_last_evaluation_monotonic",
        "_latched_reason",
        "_malformed_evidence_trips",
        "_new_results_accepted",
        "_nominal_decisions",
        "_out_of_order_rejected",
        "_recovering_decisions",
        "_recovery_armed",
        "_recovery_sequences",
        "_resets",
        "_stale_expirations",
        "_stale_tripped",
        "_updates_received",
    )

    def __init__(
        self,
        config: SafetySupervisorConfig = DEFAULT_SAFETY_SUPERVISOR_CONFIG,
    ) -> None:
        if not isinstance(config, SafetySupervisorConfig):
            raise TypeError("config must be a SafetySupervisorConfig")
        # Detach every caller-owned record, including an exact frozen base
        # instance, before it can influence later safety decisions.
        self._config = SafetySupervisorConfig(
            max_result_age_seconds=config.max_result_age_seconds,
            min_nominal_confidence=config.min_nominal_confidence,
            min_degraded_confidence=config.min_degraded_confidence,
            required_healthy_results=config.required_healthy_results,
        )
        self._evaluations = 0
        self._updates_received = 0
        self._new_results_accepted = 0
        self._duplicates_rejected = 0
        self._out_of_order_rejected = 0
        self._initializing_decisions = 0
        self._nominal_decisions = 0
        self._degraded_decisions = 0
        self._recovering_decisions = 0
        self._fail_safe_decisions = 0
        self._stale_expirations = 0
        self._component_fault_trips = 0
        self._identity_mismatch_trips = 0
        self._malformed_evidence_trips = 0
        self._clock_regression_trips = 0
        self._recovery_sequences = 0
        self._resets = 0
        self._clear_state()

    @property
    def config(self) -> SafetySupervisorConfig:
        return SafetySupervisorConfig(
            max_result_age_seconds=self._config.max_result_age_seconds,
            min_nominal_confidence=self._config.min_nominal_confidence,
            min_degraded_confidence=self._config.min_degraded_confidence,
            required_healthy_results=self._config.required_healthy_results,
        )

    @property
    def latest_decision(self) -> SafetyDecision:
        return self._decision

    @property
    def autonomy_allowed(self) -> bool:
        return self._decision.autonomy_allowed

    @property
    def metrics(self) -> SafetySupervisorMetrics:
        accepted = self._accepted
        return SafetySupervisorMetrics(
            evaluations=self._evaluations,
            updates_received=self._updates_received,
            new_results_accepted=self._new_results_accepted,
            duplicates_rejected=self._duplicates_rejected,
            out_of_order_rejected=self._out_of_order_rejected,
            initializing_decisions=self._initializing_decisions,
            nominal_decisions=self._nominal_decisions,
            degraded_decisions=self._degraded_decisions,
            recovering_decisions=self._recovering_decisions,
            fail_safe_decisions=self._fail_safe_decisions,
            stale_expirations=self._stale_expirations,
            component_fault_trips=self._component_fault_trips,
            identity_mismatch_trips=self._identity_mismatch_trips,
            malformed_evidence_trips=self._malformed_evidence_trips,
            clock_regression_trips=self._clock_regression_trips,
            recovery_sequences=self._recovery_sequences,
            resets=self._resets,
            last_accepted_source_stamp=(
                None if accepted is None else _canonical_stamp(accepted.source_stamp)
            ),
        )

    def reset(self) -> SafetyDecision:
        """Begin a fresh supervisor epoch; lifetime counters remain truthful."""
        self._resets += 1
        self._clear_state()
        return self._decision

    def _clear_state(self) -> None:
        self._accepted: _AcceptedEvidence | None = None
        self._latched_reason: SafetyReason | None = None
        self._last_evaluation_monotonic: float | None = None
        # Component health starts UNKNOWN for every epoch, and unknown is not
        # healthy.  ``_components_ready`` only carries meaning once
        # ``_health_observed`` is true; it is initialized closed as well so a
        # single missed assignment cannot re-open the gate.
        self._health_observed = False
        self._components_ready = False
        self._healthy_streak = 0
        self._recovery_armed = False
        self._stale_tripped = False
        self._decision = SafetyDecision(
            state=SafetyState.INITIALIZING,
            reason=SafetyReason.STARTUP_NO_RESULT,
            autonomy_allowed=False,
        )

    def update(
        self,
        result: PerceptionRuntimeResult,
        *,
        now: float,
        health: ComponentHealth | None = None,
    ) -> SafetyDecision:
        """Consume one published runtime result at safety time ``now``.

        ``result`` must be a ``PerceptionRuntimeResult`` carrying a Mission 7
        ``TemporalLaneObservation``.  Use :meth:`evaluate` when the runtime has
        published nothing yet or nothing new; passing ``None`` here is a caller
        error rather than a silent no-op.

        ``health`` omitted means "no new component sample".  Until this epoch
        has seen one, component health is unknown and autonomy stays denied.

        Sample ``now`` *after* reading the result, from the same host-monotonic
        clock the Mission 6 runtime and bridge use, so that
        ``now >= result.source_stamp.monotonic`` always holds.
        """
        if not isinstance(result, PerceptionRuntimeResult):
            raise TypeError(
                "result must be a PerceptionRuntimeResult; use evaluate() when "
                "no new runtime result exists"
            )
        # Structurally invalid calls raise before any accounting changes, as in
        # the accepted Mission 6A submission path.
        evaluated = _finite_real(now, "now")
        _validated_health(health)
        self._updates_received += 1
        self._advance(evaluated, health)
        accepted_new = False
        if self._latched_reason is None:
            accepted_new = self._ingest(result)
        return self._decide(evaluated, accepted_new)

    def evaluate(
        self,
        *,
        now: float,
        health: ComponentHealth | None = None,
    ) -> SafetyDecision:
        """Advance safety time without any new perception result.

        This is how Mission 8 closes the accepted Mission 7 gap: Mission 7 only
        ages its track when a newer observation arrives, so freshness must be
        enforced here even while perception publishes nothing at all.

        ``health`` follows the same contract as :meth:`update`: omitting it
        supplies no new component evidence and never asserts that components
        are healthy.
        """
        evaluated = _finite_real(now, "now")
        _validated_health(health)
        self._advance(evaluated, health)
        return self._decide(evaluated, False)

    def _advance(self, evaluated: float, health: ComponentHealth | None) -> None:
        """Fold in optional component evidence, then check clock coherence."""
        if health is not None:
            self._apply_health(health)
        previous = self._last_evaluation_monotonic
        if previous is not None and evaluated < previous:
            self._latch(SafetyReason.CLOCK_REGRESSED)
        else:
            self._last_evaluation_monotonic = evaluated

    def _apply_health(self, health: ComponentHealth) -> None:
        # Detach a possibly hostile subclass into an exact validated record
        # before any safety decision reads it twice.
        record = ComponentHealth(
            runtime_state=health.runtime_state,
            runtime_fault_present=health.runtime_fault_present,
            bridge_state=health.bridge_state,
            bridge_fault_present=health.bridge_fault_present,
        )
        if record.runtime_faulted:
            self._latch(SafetyReason.RUNTIME_FAULT, component=True)
        elif record.bridge_faulted:
            self._latch(SafetyReason.BRIDGE_FAULT, component=True)
        # This epoch has now seen real component evidence.  Later calls that
        # omit ``health`` mean "no new sample", so this readiness sticks until
        # it is replaced or the epoch is reset.
        self._health_observed = True
        self._components_ready = record.ready

    def _latch(self, reason: SafetyReason, *, component: bool = False) -> None:
        """Record the first hard failure; later ones cannot rewrite it."""
        if self._latched_reason is not None:
            return
        self._latched_reason = reason
        self._healthy_streak = 0
        if component:
            self._component_fault_trips += 1
        elif reason is SafetyReason.IDENTITY_MISMATCH:
            self._identity_mismatch_trips += 1
        elif reason is SafetyReason.EVIDENCE_MALFORMED:
            self._malformed_evidence_trips += 1
        elif reason is SafetyReason.CLOCK_REGRESSED:
            self._clock_regression_trips += 1

    def _ingest(self, result: PerceptionRuntimeResult) -> bool:
        """Gate one result on identity first, then on Mission 6 source order."""
        try:
            evidence = _result_evidence(result)
        except _IdentityMismatch:
            self._latch(SafetyReason.IDENTITY_MISMATCH)
            return False
        except _MalformedEvidence:
            self._latch(SafetyReason.EVIDENCE_MALFORMED)
            return False

        previous = self._accepted
        if previous is not None:
            stamp = evidence.source_stamp
            baseline = previous.source_stamp
            if stamp.frame == baseline.frame and stamp.sim_time == baseline.sim_time:
                self._duplicates_rejected += 1
                return False
            if stamp.frame <= baseline.frame or stamp.sim_time < baseline.sim_time:
                self._out_of_order_rejected += 1
                return False

        self._accepted = evidence
        self._new_results_accepted += 1
        self._stale_tripped = False
        return True

    def _decide(self, evaluated: float, accepted_new: bool) -> SafetyDecision:
        self._evaluations += 1
        latched = self._latched_reason
        if latched is not None:
            return self._publish(latched, evaluated, latched=True)

        accepted = self._accepted
        if accepted is None:
            return self._publish(SafetyReason.STARTUP_NO_RESULT, evaluated)

        age = evaluated - accepted.reference_monotonic
        if not math.isfinite(age):
            self._latch(SafetyReason.EVIDENCE_MALFORMED)
            return self._publish(SafetyReason.EVIDENCE_MALFORMED, evaluated, latched=True)
        if age < 0.0:
            self._latch(SafetyReason.CLOCK_REGRESSED)
            return self._publish(SafetyReason.CLOCK_REGRESSED, evaluated, latched=True)

        # Freshness first: it is the contract Mission 7 cannot enforce, and it
        # must be observable regardless of any other component evidence.
        if age > self._config.max_result_age_seconds:
            if not self._stale_tripped:
                self._stale_tripped = True
                self._stale_expirations += 1
            return self._soft_loss(SafetyReason.RESULT_STALE, evaluated, age)
        # Unknown component health and observed-not-ready component health are
        # the same answer to the only question that matters here: autonomy is
        # not permitted.  Requiring both flags keeps "never told" from ever
        # reading as "told it was fine".
        if not (self._health_observed and self._components_ready):
            return self._soft_loss(SafetyReason.COMPONENT_NOT_READY, evaluated, age)

        reason = _temporal_disposition(
            accepted.temporal_state,
            accepted.raw_state,
            accepted.confidence,
            self._config,
        )
        if reason in _HARD_REASONS:
            self._latch(reason)
            return self._publish(reason, evaluated, latched=True, age=age)
        if _REASON_STATE[reason] is SafetyState.FAIL_SAFE:
            return self._soft_loss(reason, evaluated, age)

        if self._recovery_armed:
            strong = reason is SafetyReason.NOMINAL_TRACKING
            if accepted_new:
                # Only a strictly newer, fresh, coherent, strong result extends
                # the streak.  Duplicates, out-of-order input, plain ticks, and
                # partial/coasting evidence never do.
                self._healthy_streak = self._healthy_streak + 1 if strong else 0
            if self._healthy_streak < self._config.required_healthy_results:
                return self._publish(
                    SafetyReason.RECOVERY_PENDING, evaluated, age=age
                )
            self._recovery_armed = False
            self._healthy_streak = 0
        return self._publish(reason, evaluated, age=age)

    def _soft_loss(
        self, reason: SafetyReason, evaluated: float, age: float | None
    ) -> SafetyDecision:
        """Fail closed without latching and require a fresh healthy streak."""
        if not self._recovery_armed:
            self._recovery_armed = True
            self._recovery_sequences += 1
        self._healthy_streak = 0
        return self._publish(reason, evaluated, age=age)

    def _publish(
        self,
        reason: SafetyReason,
        evaluated: float,
        *,
        latched: bool = False,
        age: float | None = None,
    ) -> SafetyDecision:
        state = _REASON_STATE[reason]
        accepted = self._accepted
        if state is SafetyState.INITIALIZING or accepted is None:
            stamp = revision = temporal_state = confidence = None
            reported_age = None
        else:
            stamp = accepted.source_stamp
            revision = accepted.snapshot_revision
            temporal_state = accepted.temporal_state
            confidence = accepted.confidence
            if age is None:
                candidate = evaluated - accepted.reference_monotonic
                reported_age = (
                    candidate
                    if math.isfinite(candidate) and candidate >= 0.0
                    else None
                )
            else:
                reported_age = age
        decision = SafetyDecision(
            state=state,
            reason=reason,
            autonomy_allowed=state in _PERMITTED_STATES,
            evaluation_monotonic=evaluated,
            source_stamp=stamp,
            snapshot_revision=revision,
            result_age_seconds=reported_age,
            temporal_state=temporal_state,
            temporal_confidence=confidence,
            recovery_streak=self._healthy_streak,
            latched=latched,
        )
        if state is SafetyState.INITIALIZING:
            self._initializing_decisions += 1
        elif state is SafetyState.NOMINAL:
            self._nominal_decisions += 1
        elif state is SafetyState.DEGRADED:
            self._degraded_decisions += 1
        elif state is SafetyState.RECOVERING:
            self._recovering_decisions += 1
        else:
            self._fail_safe_decisions += 1
        self._decision = decision
        return decision
