"""Immutable sensor freshness policy and pure status classification.

This module deliberately knows nothing about CARLA, callbacks, locks, clocks,
or mutable suite state.  Callers provide one bounded evidence record and one
already-sampled host-monotonic evaluation time.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from enum import Enum


STATUS_TEXT_LIMIT = 192
MAX_STARTUP_GRACE_SECONDS = 10.0


class SensorLifecycle(Enum):
    NOT_AVAILABLE = "not_available"
    NOT_ATTACHED = "not_attached"
    ATTACHMENT_FAILED = "attachment_failed"
    ATTACHED = "attached"
    DESTROYED = "destroyed"


class SensorReadiness(Enum):
    NOT_APPLICABLE = "not_applicable"
    WAITING_FOR_FIRST_SAMPLE = "waiting_for_first_sample"
    READY = "ready"


class SensorFreshness(Enum):
    NOT_APPLICABLE = "not_applicable"
    WAITING_FOR_FIRST_SAMPLE = "waiting_for_first_sample"
    FRESH = "fresh"
    STALE = "stale"


class OperationalHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


class SensorKind(Enum):
    CONTINUOUS = "continuous"
    EVENT = "event"


def bounded_text(value: str | None, limit: int = STATUS_TEXT_LIMIT) -> str | None:
    """Normalize and bound diagnostic text without retaining an exception."""
    if value is None:
        return None
    text = " ".join(str(value).split()) or "<no detail>"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _positive_finite_real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return result


def finite_monotonic(value, name: str = "evaluated_monotonic") -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


@dataclass(frozen=True, slots=True)
class ContinuousFreshnessPolicy:
    expected_interval: float
    stale_after: float
    startup_grace: float

    def __post_init__(self) -> None:
        expected = _positive_finite_real(self.expected_interval, "expected_interval")
        stale = _positive_finite_real(self.stale_after, "stale_after")
        grace = _positive_finite_real(self.startup_grace, "startup_grace")
        if not expected <= stale <= grace <= MAX_STARTUP_GRACE_SECONDS:
            raise ValueError(
                "freshness policy must satisfy expected_interval <= stale_after "
                "<= startup_grace <= 10.0"
            )
        object.__setattr__(self, "expected_interval", expected)
        object.__setattr__(self, "stale_after", stale)
        object.__setattr__(self, "startup_grace", grace)


GNSS_FRESHNESS = ContinuousFreshnessPolicy(0.10, 0.50, 1.00)
IMU_FRESHNESS = ContinuousFreshnessPolicy(0.05, 0.25, 0.50)
RGB_CAMERA_FRESHNESS = ContinuousFreshnessPolicy(0.10, 0.50, 1.00)


@dataclass(frozen=True, slots=True)
class SensorFreshnessConfig:
    """Policies for all five suite sensors; event sensors must remain ``None``."""

    gnss: ContinuousFreshnessPolicy = GNSS_FRESHNESS
    imu: ContinuousFreshnessPolicy = IMU_FRESHNESS
    rgb_camera: ContinuousFreshnessPolicy = RGB_CAMERA_FRESHNESS
    collision: ContinuousFreshnessPolicy | None = None
    lane_invasion: ContinuousFreshnessPolicy | None = None

    def __post_init__(self) -> None:
        for name in ("gnss", "imu", "rgb_camera"):
            if not isinstance(getattr(self, name), ContinuousFreshnessPolicy):
                raise ValueError(f"continuous sensor {name!r} requires a freshness policy")
        for name in ("collision", "lane_invasion"):
            if getattr(self, name) is not None:
                raise ValueError(f"event sensor {name!r} rejects a freshness policy")

    def policy_for(self, name: str) -> ContinuousFreshnessPolicy | None:
        if name not in ("gnss", "imu", "rgb_camera", "collision", "lane_invasion"):
            raise KeyError(name)
        return getattr(self, name)


@dataclass(frozen=True, slots=True)
class SensorEvidence:
    """Primitive-only raw evidence consumed by the pure evaluator."""

    name: str
    kind: SensorKind
    enabled: bool
    attach_attempted: bool
    attached: bool
    attachment_failed: bool
    destroyed: bool
    attachment_monotonic: float | None
    accepted_count: int
    last_receive_monotonic: float | None
    active_error: bool
    error_count: int
    active_error_reason: str | None = None
    event_count: int | None = None


@dataclass(frozen=True, slots=True)
class SensorStatus:
    """Immutable composite status; no live object or raw exception text."""

    name: str
    kind: SensorKind
    lifecycle: SensorLifecycle
    readiness: SensorReadiness
    freshness: SensorFreshness
    operational_health: OperationalHealth
    evaluated_monotonic: float
    age_seconds: float | None
    expected_interval: float | None
    stale_after: float | None
    accepted_count: int
    event_count: int | None
    error_count: int
    reason: str | None = None


def _status(
    evidence: SensorEvidence,
    evaluated: float,
    lifecycle: SensorLifecycle,
    readiness: SensorReadiness,
    freshness: SensorFreshness,
    health: OperationalHealth,
    *,
    policy: ContinuousFreshnessPolicy | None,
    age: float | None = None,
    reason: str | None = None,
) -> SensorStatus:
    return SensorStatus(
        name=evidence.name,
        kind=evidence.kind,
        lifecycle=lifecycle,
        readiness=readiness,
        freshness=freshness,
        operational_health=health,
        evaluated_monotonic=evaluated,
        age_seconds=age,
        expected_interval=None if policy is None else policy.expected_interval,
        stale_after=None if policy is None else policy.stale_after,
        accepted_count=evidence.accepted_count,
        event_count=evidence.event_count,
        error_count=evidence.error_count,
        reason=bounded_text(reason),
    )


def _degradation_reason(evidence: SensorEvidence, clock_rollback: bool) -> str | None:
    reasons = []
    if clock_rollback:
        reasons.append("host monotonic clock rollback")
    if evidence.active_error:
        reasons.append(evidence.active_error_reason or "callback error")
    return bounded_text("; ".join(reasons)) if reasons else None


def evaluate_sensor_status(
    evidence: SensorEvidence,
    policy: ContinuousFreshnessPolicy | None,
    evaluated_monotonic: float,
    *,
    clock_rollback: bool = False,
) -> SensorStatus:
    """Classify one sensor at one caller-supplied coherent monotonic time."""
    evaluated = finite_monotonic(evaluated_monotonic)
    if isinstance(evidence.accepted_count, bool) or type(evidence.accepted_count) is not int:
        raise ValueError("accepted_count must be a built-in integer")
    if evidence.accepted_count < 0:
        raise ValueError("accepted_count must be non-negative")
    if isinstance(evidence.error_count, bool) or type(evidence.error_count) is not int:
        raise ValueError("error_count must be a built-in integer")
    if evidence.error_count < 0:
        raise ValueError("error_count must be non-negative")
    if evidence.kind is SensorKind.CONTINUOUS and not isinstance(
        policy, ContinuousFreshnessPolicy
    ):
        raise ValueError("continuous sensors require a freshness policy")
    if evidence.kind is SensorKind.EVENT and policy is not None:
        raise ValueError("event sensors reject continuous freshness policies")

    not_applicable = SensorReadiness.NOT_APPLICABLE
    freshness_na = SensorFreshness.NOT_APPLICABLE
    event_count = evidence.event_count
    if evidence.kind is SensorKind.EVENT:
        if event_count is None:
            event_count = evidence.accepted_count
        if isinstance(event_count, bool) or type(event_count) is not int or event_count < 0:
            raise ValueError("event_count must be a non-negative built-in integer")
        evidence = SensorEvidence(
            name=evidence.name,
            kind=evidence.kind,
            enabled=evidence.enabled,
            attach_attempted=evidence.attach_attempted,
            attached=evidence.attached,
            attachment_failed=evidence.attachment_failed,
            destroyed=evidence.destroyed,
            attachment_monotonic=evidence.attachment_monotonic,
            accepted_count=evidence.accepted_count,
            last_receive_monotonic=evidence.last_receive_monotonic,
            active_error=evidence.active_error,
            error_count=evidence.error_count,
            active_error_reason=evidence.active_error_reason,
            event_count=event_count,
        )

    if not evidence.enabled:
        return _status(
            evidence, evaluated, SensorLifecycle.NOT_AVAILABLE,
            not_applicable, freshness_na, OperationalHealth.HEALTHY, policy=policy,
        )
    if evidence.destroyed:
        destroyed_health = (
            OperationalHealth.FAILED
            if evidence.attachment_failed
            else (
                OperationalHealth.DEGRADED
                if evidence.active_error or clock_rollback
                else OperationalHealth.HEALTHY
            )
        )
        return _status(
            evidence, evaluated, SensorLifecycle.DESTROYED,
            not_applicable, freshness_na, destroyed_health, policy=policy,
            reason=_degradation_reason(evidence, clock_rollback),
        )
    if evidence.attachment_failed:
        return _status(
            evidence, evaluated, SensorLifecycle.ATTACHMENT_FAILED,
            not_applicable, freshness_na, OperationalHealth.FAILED, policy=policy,
            reason="sensor attachment failed",
        )
    if not evidence.attached:
        return _status(
            evidence, evaluated, SensorLifecycle.NOT_ATTACHED,
            not_applicable, freshness_na, OperationalHealth.HEALTHY, policy=policy,
        )

    degradation = _degradation_reason(evidence, clock_rollback)
    health = OperationalHealth.DEGRADED if degradation else OperationalHealth.HEALTHY
    if evidence.kind is SensorKind.EVENT:
        return _status(
            evidence, evaluated, SensorLifecycle.ATTACHED,
            not_applicable, freshness_na, health, policy=None, reason=degradation,
        )

    assert policy is not None
    if evidence.accepted_count == 0:
        if evidence.attachment_monotonic is None:
            age = None
            freshness = SensorFreshness.WAITING_FOR_FIRST_SAMPLE
            health = OperationalHealth.DEGRADED
            degradation = degradation or "attachment timestamp unavailable"
        else:
            attached_at = finite_monotonic(
                evidence.attachment_monotonic, "attachment_monotonic"
            )
            age = max(0.0, evaluated - attached_at)
            freshness = (
                SensorFreshness.WAITING_FOR_FIRST_SAMPLE
                if age <= policy.startup_grace
                else SensorFreshness.STALE
            )
        return _status(
            evidence, evaluated, SensorLifecycle.ATTACHED,
            SensorReadiness.WAITING_FOR_FIRST_SAMPLE, freshness, health,
            policy=policy, age=age, reason=degradation,
        )

    if evidence.last_receive_monotonic is None:
        raise ValueError("accepted continuous evidence requires a receive timestamp")
    received = finite_monotonic(
        evidence.last_receive_monotonic, "last_receive_monotonic"
    )
    age = max(0.0, evaluated - received)
    freshness = (
        SensorFreshness.FRESH
        if age <= policy.stale_after
        else SensorFreshness.STALE
    )
    return _status(
        evidence, evaluated, SensorLifecycle.ATTACHED,
        SensorReadiness.READY, freshness, health,
        policy=policy, age=age, reason=degradation,
    )
