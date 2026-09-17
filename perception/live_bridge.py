"""CARLA-free bounded admission bridge for live perception snapshots.

The bridge is deliberately smaller than a runtime: it owns no worker, queue,
pending slot, or result history.  It only gates callback submissions into an
already-owned runtime, accounts for calls which have crossed that gate, and
provides a bounded drain point for deterministic teardown.

Neither construction nor import reads the injected clock.  Runtime submission,
clock sampling, snapshot attribute access, and exception formatting all happen
without the bridge lock held.
"""

from __future__ import annotations

import math
import numbers
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from telemetry.freshness import STATUS_TEXT_LIMIT
from telemetry.state import (
    CameraFrame,
    CameraFrameMetadata,
    SampleStamp,
    TelemetrySnapshot,
)


MAX_BRIDGE_DRAIN_TIMEOUT_SECONDS = 60.0


__all__ = (
    "MAX_BRIDGE_DRAIN_TIMEOUT_SECONDS",
    "PerceptionBridge",
    "PerceptionBridgeFault",
    "PerceptionBridgeMetrics",
    "PerceptionBridgeState",
    "PerceptionLiveBridge",
    "PerceptionLiveBridgeFault",
    "PerceptionLiveBridgeMetrics",
    "PerceptionLiveBridgeState",
)


class PerceptionLiveBridgeState(Enum):
    """One-shot admission and drain state for a bridge instance."""

    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    FAULTED = "faulted"


@dataclass(frozen=True, slots=True)
class PerceptionLiveBridgeFault:
    """Bounded fault evidence which retains no exception or traceback."""

    stage: str
    exception_type: str
    message: str
    source_stamp: SampleStamp | None
    snapshot_revision: int | None
    occurred_monotonic: float | None


@dataclass(frozen=True, slots=True)
class PerceptionLiveBridgeMetrics:
    """One immutable scalar view of all bounded bridge accounting.

    Admitted calls partition exactly into runtime acceptances, runtime
    rejections, and failures before a definitive runtime result.  A clock or
    instrumentation fault after a valid runtime bool remains orthogonal fault
    evidence and never reclassifies or double-counts that terminal result.
    """

    submissions_received: int
    submissions_admitted: int
    runtime_submissions_accepted: int
    runtime_submissions_rejected: int
    submissions_rejected_closed: int
    submission_failures: int
    current_in_flight: int
    maximum_in_flight: int
    last_callback_to_submit_latency_seconds: float | None
    maximum_callback_to_submit_latency_seconds: float | None
    last_submitted_source_stamp: SampleStamp | None


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _drain_timeout(value: object) -> float:
    timeout = _finite_real(value, "timeout_seconds")
    if not 0.0 <= timeout <= MAX_BRIDGE_DRAIN_TIMEOUT_SECONDS:
        raise ValueError(
            "timeout_seconds must be in "
            f"[0.0, {MAX_BRIDGE_DRAIN_TIMEOUT_SECONDS}]"
        )
    return timeout


def _nonnegative_builtin_int(value: object, name: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative built-in integer")
    return value


def _canonical_stamp(value: object) -> SampleStamp:
    if not isinstance(value, SampleStamp):
        raise ValueError("snapshot must contain a canonical camera SampleStamp")
    frame = _nonnegative_builtin_int(value.frame, "camera source frame")
    sim_time = _finite_real(value.sim_time, "camera simulation timestamp")
    received = _finite_real(value.monotonic, "camera receive monotonic timestamp")
    if (
        type(value) is SampleStamp
        and type(value.sim_time) is float
        and type(value.monotonic) is float
    ):
        return value
    return SampleStamp(frame=frame, sim_time=sim_time, monotonic=received)


def _snapshot_evidence(snapshot: object) -> tuple[SampleStamp, int]:
    """Copy only trusted immutable scalar identity from one snapshot.

    Attribute access is intentionally performed by the caller before any
    bridge lock acquisition, so a hostile subclass cannot deadlock the bridge.
    The runtime remains responsible for its complete submission validation.
    """

    if not isinstance(snapshot, TelemetrySnapshot):
        raise TypeError("snapshot must be a TelemetrySnapshot")
    revision = _nonnegative_builtin_int(snapshot.revision, "snapshot revision")
    frame = snapshot.camera
    if not isinstance(frame, CameraFrame) or not isinstance(
        frame.metadata, CameraFrameMetadata
    ):
        raise ValueError("snapshot must contain canonical camera metadata")
    return _canonical_stamp(frame.metadata.stamp), revision


def _duration(later: float, earlier: float) -> float:
    result = later - earlier
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError(
            "monotonic clock regressed while measuring callback-to-submit latency"
        )
    return result


def _bounded_plain_text(value: str, fallback: str) -> str:
    if type(value) is not str:
        return fallback
    normalized = " ".join(value.split()) or fallback
    if len(normalized) <= STATUS_TEXT_LIMIT:
        return normalized
    return normalized[: STATUS_TEXT_LIMIT - 3] + "..."


def _exception_type(exc: Exception) -> str:
    exception_class = type(exc)
    # Bypass a hostile metaclass override.  Descriptor behavior, if any, still
    # executes outside all bridge locks and an escaping BaseException remains
    # deliberately uncaught by the ordinary-Exception containment path.
    try:
        module = type.__getattribute__(exception_class, "__module__")
        qualname = type.__getattribute__(exception_class, "__qualname__")
    except Exception:
        module = qualname = None
    if type(module) is str and type(qualname) is str and module and qualname:
        value = module + "." + qualname
    else:
        value = "<unavailable exception type>"
    return _bounded_plain_text(value, "<unavailable exception type>")


def _ordinary_fault(
    *,
    stage: str,
    exc: Exception,
    source_stamp: SampleStamp | None,
    snapshot_revision: int | None,
    occurred_monotonic: float | None,
) -> PerceptionLiveBridgeFault:
    try:
        raw_message = str(exc)
    except Exception:
        raw_message = "<unprintable exception>"
    return PerceptionLiveBridgeFault(
        stage=stage,
        exception_type=_exception_type(exc),
        message=_bounded_plain_text(raw_message, "<no detail>"),
        source_stamp=source_stamp,
        snapshot_revision=snapshot_revision,
        occurred_monotonic=occurred_monotonic,
    )


def _uncaught_base_exception_fault(
    *,
    stage: str,
    source_stamp: SampleStamp | None,
    snapshot_revision: int | None,
    occurred_monotonic: float | None,
) -> PerceptionLiveBridgeFault:
    return PerceptionLiveBridgeFault(
        stage=stage,
        exception_type="<uncaught BaseException>",
        message="bridge submission exited through BaseException",
        source_stamp=source_stamp,
        snapshot_revision=snapshot_revision,
        occurred_monotonic=occurred_monotonic,
    )


class PerceptionLiveBridge:
    """Bounded, thread-safe callback-to-runtime admission bridge.

    ``submit`` never waits for another bridge submission.  Calls admitted while
    the state is :attr:`~PerceptionLiveBridgeState.OPEN` are counted in flight
    until the runtime call and completion-clock sample finish.  A runtime
    rejection is a normal ``False`` result; an ordinary exception is contained
    and fail-closes the bridge.  An escaping ``BaseException`` also fail-closes
    and finalizes accounting, then continues propagating unchanged.

    The bridge is one-shot.  ``close_admission`` and ``drain`` expose the two
    cleanup stages independently; ``close`` is their combined convenience.
    A non-faulted drain timeout leaves
    :attr:`~PerceptionLiveBridgeState.CLOSING`, so a later call can retry after
    already-admitted submissions finish.  Once a fault occurs, public state
    remains :attr:`~PerceptionLiveBridgeState.FAULTED` through every cleanup
    attempt while the same bounded drain and retry behavior remains available.
    """

    __slots__ = (
        "_clock",
        "_condition",
        "_current_in_flight",
        "_fault",
        "_last_callback_to_submit_latency_seconds",
        "_last_submitted_source_stamp",
        "_lock",
        "_maximum_callback_to_submit_latency_seconds",
        "_maximum_in_flight",
        "_runtime_submissions_accepted",
        "_runtime_submissions_rejected",
        "_state",
        "_submission_failures",
        "_submissions_admitted",
        "_submissions_received",
        "_submissions_rejected_closed",
        "_submit_callable",
    )

    def __init__(
        self,
        runtime: object,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        try:
            submit_callable = runtime.submit
        except AttributeError as exc:
            raise TypeError("runtime must provide a callable submit method") from exc
        if not callable(submit_callable):
            raise TypeError("runtime must provide a callable submit method")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self._submit_callable: Callable[[TelemetrySnapshot], bool] = submit_callable
        self._clock = clock
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._state = PerceptionLiveBridgeState.OPEN
        self._fault: PerceptionLiveBridgeFault | None = None
        self._submissions_received = 0
        self._submissions_admitted = 0
        self._runtime_submissions_accepted = 0
        self._runtime_submissions_rejected = 0
        self._submissions_rejected_closed = 0
        self._submission_failures = 0
        self._current_in_flight = 0
        self._maximum_in_flight = 0
        self._last_callback_to_submit_latency_seconds: float | None = None
        self._maximum_callback_to_submit_latency_seconds: float | None = None
        self._last_submitted_source_stamp: SampleStamp | None = None

    @property
    def state(self) -> PerceptionLiveBridgeState:
        with self._lock:
            return self._state

    @property
    def fault(self) -> PerceptionLiveBridgeFault | None:
        with self._lock:
            return self._fault

    @property
    def metrics(self) -> PerceptionLiveBridgeMetrics:
        with self._lock:
            return PerceptionLiveBridgeMetrics(
                submissions_received=self._submissions_received,
                submissions_admitted=self._submissions_admitted,
                runtime_submissions_accepted=(
                    self._runtime_submissions_accepted
                ),
                runtime_submissions_rejected=(
                    self._runtime_submissions_rejected
                ),
                submissions_rejected_closed=(
                    self._submissions_rejected_closed
                ),
                submission_failures=self._submission_failures,
                current_in_flight=self._current_in_flight,
                maximum_in_flight=self._maximum_in_flight,
                last_callback_to_submit_latency_seconds=(
                    self._last_callback_to_submit_latency_seconds
                ),
                maximum_callback_to_submit_latency_seconds=(
                    self._maximum_callback_to_submit_latency_seconds
                ),
                last_submitted_source_stamp=self._last_submitted_source_stamp,
            )

    def submit(self, snapshot: TelemetrySnapshot) -> bool:
        """Submit one snapshot without retaining it after this call returns."""

        with self._condition:
            self._submissions_received += 1
            if self._state is not PerceptionLiveBridgeState.OPEN:
                self._submissions_rejected_closed += 1
                return False
            self._submissions_admitted += 1
            self._current_in_flight += 1
            if self._current_in_flight > self._maximum_in_flight:
                self._maximum_in_flight = self._current_in_flight

        source_stamp: SampleStamp | None = None
        snapshot_revision: int | None = None
        occurred_monotonic: float | None = None
        runtime_invoked = False
        runtime_outcome: bool | None = None
        latency: float | None = None
        fault: PerceptionLiveBridgeFault | None = None
        disposition: str | None = None
        stage = "snapshot"
        escaped_base_exception = True

        try:
            try:
                source_stamp, snapshot_revision = _snapshot_evidence(snapshot)
                stage = "runtime_submit"
                runtime_invoked = True
                result = self._submit_callable(snapshot)
                stage = "runtime_result"
                if type(result) is not bool:
                    raise TypeError("runtime submit must return a built-in bool")
                runtime_outcome = result
                # A definitive runtime bool is this admission's mutually
                # exclusive terminal classification.  Later bridge-only
                # instrumentation may fault the bridge, but cannot turn the
                # same call into a second terminal failure.
                disposition = "accepted" if result else "rejected"
                stage = "clock"
                occurred_monotonic = _finite_real(
                    self._clock(), "bridge completion clock"
                )
                latency = _duration(occurred_monotonic, source_stamp.monotonic)
            except Exception as exc:
                # Close the gate before diagnostic formatting, which may be
                # hostile or reentrant.  Formatting itself remains outside the
                # bridge lock.
                self._fail_close_admission()
                fault = _ordinary_fault(
                    stage=stage,
                    exc=exc,
                    source_stamp=source_stamp,
                    snapshot_revision=snapshot_revision,
                    occurred_monotonic=occurred_monotonic,
                )
                if disposition is None:
                    disposition = "failure"
            escaped_base_exception = False
        finally:
            # BaseException may escape before or after a definitive runtime
            # result.  Do not inspect or catch it; publish constant bounded
            # evidence, release waiters, and let it propagate unchanged.  A
            # result already classified accepted/rejected stays in that one
            # terminal bucket.
            if escaped_base_exception:
                fault = _uncaught_base_exception_fault(
                    stage=stage,
                    source_stamp=source_stamp,
                    snapshot_revision=snapshot_revision,
                    occurred_monotonic=occurred_monotonic,
                )
                if disposition is None:
                    disposition = "failure"
            assert disposition is not None
            self._finalize_submission(
                disposition=disposition,
                runtime_invoked=runtime_invoked,
                runtime_outcome=runtime_outcome,
                source_stamp=source_stamp,
                latency=latency,
                fault=fault,
            )

        return disposition == "accepted" and fault is None

    def close_admission(self) -> None:
        """Close the one-shot admission gate without waiting for in-flight calls."""

        with self._condition:
            if self._state in (
                PerceptionLiveBridgeState.CLOSED,
                PerceptionLiveBridgeState.FAULTED,
            ):
                return
            self._state = PerceptionLiveBridgeState.CLOSING
            self._condition.notify_all()

    def drain(self, timeout_seconds: float) -> None:
        """Drain admitted calls within ``timeout_seconds`` or remain retryable."""

        timeout = _drain_timeout(timeout_seconds)
        self._drain_validated(timeout)

    def close(self, timeout_seconds: float) -> None:
        """Close admission and boundedly drain already-admitted submissions."""

        # Validate caller-controlled policy before making the one-shot gate
        # transition.  A malformed close request must not make later cleanup
        # impossible.
        timeout = _drain_timeout(timeout_seconds)
        self.close_admission()
        self._drain_validated(timeout)

    def _drain_validated(self, timeout: float) -> None:
        timed_out = False
        with self._condition:
            if self._state is PerceptionLiveBridgeState.CLOSED:
                return
            if self._state is not PerceptionLiveBridgeState.FAULTED:
                self._state = PerceptionLiveBridgeState.CLOSING
            if self._current_in_flight == 0:
                if self._state is not PerceptionLiveBridgeState.FAULTED:
                    self._state = PerceptionLiveBridgeState.CLOSED
                self._condition.notify_all()
                return
            drained = self._condition.wait_for(
                lambda: self._current_in_flight == 0,
                timeout,
            )
            if drained:
                if self._state is not PerceptionLiveBridgeState.FAULTED:
                    self._state = PerceptionLiveBridgeState.CLOSED
                self._condition.notify_all()
            else:
                # Preserve the applicable terminal history.  Completion of an
                # outstanding call only notifies; a later drain/close retries
                # the wait without rewriting FAULTED or auto-finalizing CLOSED.
                if self._state is not PerceptionLiveBridgeState.FAULTED:
                    self._state = PerceptionLiveBridgeState.CLOSING
                timed_out = True
        if timed_out:
            raise TimeoutError(
                "perception bridge submissions did not drain within "
                f"{timeout} seconds"
            )

    def _fail_close_admission(self) -> None:
        with self._condition:
            if self._state is not PerceptionLiveBridgeState.CLOSED:
                self._state = PerceptionLiveBridgeState.FAULTED
            self._condition.notify_all()

    def _finalize_submission(
        self,
        *,
        disposition: str,
        runtime_invoked: bool,
        runtime_outcome: bool | None,
        source_stamp: SampleStamp | None,
        latency: float | None,
        fault: PerceptionLiveBridgeFault | None,
    ) -> None:
        with self._condition:
            if runtime_outcome is True:
                self._runtime_submissions_accepted += 1
            elif runtime_outcome is False:
                self._runtime_submissions_rejected += 1
            if runtime_invoked and source_stamp is not None:
                self._last_submitted_source_stamp = source_stamp
            if latency is not None:
                self._last_callback_to_submit_latency_seconds = latency
                maximum = self._maximum_callback_to_submit_latency_seconds
                if maximum is None or latency > maximum:
                    self._maximum_callback_to_submit_latency_seconds = latency
            if disposition == "failure":
                self._submission_failures += 1
            if fault is not None:
                if self._fault is None:
                    self._fault = fault
                if self._state is not PerceptionLiveBridgeState.CLOSED:
                    self._state = PerceptionLiveBridgeState.FAULTED

            # Decrement and notify in the same critical section, including the
            # BaseException path, so drain waiters cannot miss completion.
            self._current_in_flight -= 1
            self._condition.notify_all()


# Short aliases keep the module ergonomic while the longer names make the live
# ownership boundary explicit in harness code.
PerceptionBridgeState = PerceptionLiveBridgeState
PerceptionBridgeFault = PerceptionLiveBridgeFault
PerceptionBridgeMetrics = PerceptionLiveBridgeMetrics
PerceptionBridge = PerceptionLiveBridge
