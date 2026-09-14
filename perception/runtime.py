"""Bounded, CARLA-independent execution for single-frame lane perception.

The runtime accepts coherent telemetry snapshots through an O(1) submission
path, retains exactly one pending snapshot, and publishes only the latest
immutable perception result.  Ordering follows the existing camera
``SampleStamp`` contract; whole-snapshot revisions are never frame identity.

No worker is created until :meth:`PerceptionRuntime.start` is called.  The
module performs no clock reads at import or construction time and deliberately
imports neither sensor adapters nor NumPy.  NumPy remains a lazy implementation
detail of ``perceive_lanes`` when valid pixel processing actually begins.
"""

from __future__ import annotations

import math
import numbers
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from perception.lane import perceive_lanes
from perception.state import LaneObservation
from telemetry.freshness import SensorKind, SensorStatus, bounded_text
from telemetry.state import (
    CameraFrame,
    CameraFrameMetadata,
    SampleStamp,
    TelemetrySnapshot,
)


MIN_TARGET_HZ = 1.0 / 60.0
MAX_TARGET_HZ = 1_000.0
MAX_RUNTIME_INTERVAL_SECONDS = 60.0
PENDING_CAPACITY = 1


__all__ = (
    "PENDING_CAPACITY",
    "PerceptionRuntime",
    "PerceptionRuntimeConfig",
    "PerceptionRuntimeFault",
    "PerceptionRuntimeMetrics",
    "PerceptionRuntimeResult",
    "PerceptionRuntimeState",
)


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


def _bounded_positive_real(value: object, name: str, upper: float) -> float:
    result = _finite_real(value, name)
    if not 0.0 < result <= upper:
        raise ValueError(f"{name} must be in (0.0, {upper}]")
    return result


def _nonnegative_builtin_int(value: object, name: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative built-in integer")
    return value


def _validated_stamp(stamp: object) -> SampleStamp:
    if not isinstance(stamp, SampleStamp):
        raise ValueError("snapshot must contain a canonical camera SampleStamp")
    frame = _nonnegative_builtin_int(stamp.frame, "camera source frame")
    sim_time = _finite_real(stamp.sim_time, "camera simulation timestamp")
    received = _finite_real(stamp.monotonic, "camera receive monotonic timestamp")
    if (
        type(stamp) is SampleStamp
        and type(stamp.sim_time) is float
        and type(stamp.monotonic) is float
    ):
        return stamp
    return SampleStamp(frame=frame, sim_time=sim_time, monotonic=received)


def _duration(later: float, earlier: float, name: str) -> float:
    result = later - earlier
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError(f"monotonic clock regressed while measuring {name}")
    return result


@dataclass(frozen=True, slots=True)
class PerceptionRuntimeConfig:
    """Validated timing policy for one runtime execution epoch.

    ``target_hz`` defines the real-time processing budget.  The corresponding
    period is used for deadline accounting; submissions themselves wake the
    worker, so the runtime does not synthesize camera ticks or delay fresh work.
    """

    target_hz: float = 10.0
    max_pending_age_seconds: float = 0.25
    shutdown_timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        target = _finite_real(self.target_hz, "target_hz")
        if not MIN_TARGET_HZ <= target <= MAX_TARGET_HZ:
            raise ValueError(
                f"target_hz must be in [{MIN_TARGET_HZ}, {MAX_TARGET_HZ}]"
            )
        period = 1.0 / target
        if not math.isfinite(period) or period <= 0.0:
            raise ValueError("target_hz must produce a finite positive period")
        pending_age = _bounded_positive_real(
            self.max_pending_age_seconds,
            "max_pending_age_seconds",
            MAX_RUNTIME_INTERVAL_SECONDS,
        )
        shutdown = _bounded_positive_real(
            self.shutdown_timeout_seconds,
            "shutdown_timeout_seconds",
            MAX_RUNTIME_INTERVAL_SECONDS,
        )
        object.__setattr__(self, "target_hz", target)
        object.__setattr__(self, "max_pending_age_seconds", pending_age)
        object.__setattr__(self, "shutdown_timeout_seconds", shutdown)

    @property
    def processing_period_seconds(self) -> float:
        return 1.0 / self.target_hz


class PerceptionRuntimeState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    STOPPING = "stopping"
    FAULTED = "faulted"


@dataclass(frozen=True, slots=True)
class PerceptionRuntimeResult:
    """Latest payload-free result and the evidence needed to interpret it."""

    observation: LaneObservation
    source_stamp: SampleStamp
    source_sequence: int | None
    snapshot_revision: int
    ingestion_monotonic: float
    processing_start_monotonic: float
    processing_finish_monotonic: float
    pending_age_seconds: float
    processing_latency_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.observation, LaneObservation):
            raise ValueError("observation must be a LaneObservation")
        source_stamp = _validated_stamp(self.source_stamp)
        if self.source_sequence is not None:
            _nonnegative_builtin_int(self.source_sequence, "source_sequence")
        _nonnegative_builtin_int(self.snapshot_revision, "snapshot_revision")
        ingested = _finite_real(self.ingestion_monotonic, "ingestion_monotonic")
        started = _finite_real(
            self.processing_start_monotonic, "processing_start_monotonic"
        )
        finished = _finite_real(
            self.processing_finish_monotonic, "processing_finish_monotonic"
        )
        pending_age = _finite_real(self.pending_age_seconds, "pending_age_seconds")
        latency = _finite_real(
            self.processing_latency_seconds, "processing_latency_seconds"
        )
        if pending_age < 0.0 or latency < 0.0:
            raise ValueError("runtime durations must be non-negative")
        if pending_age != started - ingested:
            raise ValueError("pending_age_seconds is inconsistent with timestamps")
        if latency != finished - started:
            raise ValueError(
                "processing_latency_seconds is inconsistent with timestamps"
            )
        object.__setattr__(self, "source_stamp", source_stamp)
        object.__setattr__(self, "ingestion_monotonic", ingested)
        object.__setattr__(self, "processing_start_monotonic", started)
        object.__setattr__(self, "processing_finish_monotonic", finished)
        object.__setattr__(self, "pending_age_seconds", pending_age)
        object.__setattr__(self, "processing_latency_seconds", latency)


@dataclass(frozen=True, slots=True)
class PerceptionRuntimeFault:
    """Bounded fault evidence that never retains an exception or traceback."""

    stage: str
    exception_type: str
    message: str
    source_stamp: SampleStamp | None
    source_sequence: int | None
    snapshot_revision: int | None
    occurred_monotonic: float | None


@dataclass(frozen=True, slots=True)
class PerceptionRuntimeMetrics:
    """One immutable, atomic view of the bounded runtime counters.

    At quiescence, accepted inputs partition into processed, stale, coalesced,
    pending-discarded, and abandoned terminal dispositions.  Processed inputs
    partition into perception successes and perception failures.
    """

    inputs_received: int
    inputs_accepted: int
    inputs_processed: int
    duplicate_inputs_rejected: int
    out_of_order_inputs_rejected: int
    stale_inputs_rejected: int
    inputs_abandoned: int
    source_frames_inferred_dropped: int
    pending_frames_coalesced: int
    pending_inputs_discarded: int
    perception_successes: int
    perception_failures: int
    deadline_misses: int
    last_processing_latency_seconds: float | None
    maximum_processing_latency_seconds: float | None
    last_pending_age_seconds: float | None
    last_processed_source_stamp: SampleStamp | None


@dataclass(frozen=True, slots=True)
class _PendingInput:
    snapshot: TelemetrySnapshot
    source_stamp: SampleStamp
    source_sequence: int | None
    snapshot_revision: int
    ingestion_monotonic: float


class PerceptionRuntime:
    """Single-worker latest-frame-wins lane-perception runtime.

    A clean ``STOPPED -> RUNNING`` transition begins a new execution epoch and
    resets metrics, ordering baselines, result, and fault evidence.  Repeated
    ``start`` while running and repeated ``stop`` while stopped are idempotent.
    A faulted runtime must be stopped before it can be started again.
    """

    pending_capacity = PENDING_CAPACITY

    __slots__ = (
        "_clock",
        "_condition",
        "_config",
        "_deadline_misses",
        "_duplicate_inputs_rejected",
        "_fault",
        "_inputs_accepted",
        "_inputs_abandoned",
        "_inputs_processed",
        "_inputs_received",
        "_last_accepted_source_sequence",
        "_last_accepted_stamp",
        "_last_pending_age_seconds",
        "_last_processed_source_stamp",
        "_last_processing_latency_seconds",
        "_latest_result",
        "_lock",
        "_maximum_processing_latency_seconds",
        "_out_of_order_inputs_rejected",
        "_pending",
        "_pending_frames_coalesced",
        "_pending_inputs_discarded",
        "_perception_callable",
        "_perception_failures",
        "_perception_successes",
        "_source_frames_inferred_dropped",
        "_stale_inputs_rejected",
        "_state",
        "_stop_requested",
        "_worker",
    )

    def __init__(
        self,
        config: PerceptionRuntimeConfig = PerceptionRuntimeConfig(),
        *,
        clock: Callable[[], float] = time.monotonic,
        perception_callable: Callable[[TelemetrySnapshot], LaneObservation] = perceive_lanes,
    ) -> None:
        if not isinstance(config, PerceptionRuntimeConfig):
            raise TypeError("config must be a PerceptionRuntimeConfig")
        if type(config) is not PerceptionRuntimeConfig:
            config = PerceptionRuntimeConfig(
                target_hz=config.target_hz,
                max_pending_age_seconds=config.max_pending_age_seconds,
                shutdown_timeout_seconds=config.shutdown_timeout_seconds,
            )
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(perception_callable):
            raise TypeError("perception_callable must be callable")
        self._config = config
        self._clock = clock
        self._perception_callable = perception_callable
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._state = PerceptionRuntimeState.STOPPED
        self._worker: threading.Thread | None = None
        self._stop_requested = False
        self._pending: _PendingInput | None = None
        self._latest_result: PerceptionRuntimeResult | None = None
        self._fault: PerceptionRuntimeFault | None = None
        self._reset_epoch_locked()

    @property
    def config(self) -> PerceptionRuntimeConfig:
        return self._config

    @property
    def state(self) -> PerceptionRuntimeState:
        with self._lock:
            return self._state

    @property
    def latest_result(self) -> PerceptionRuntimeResult | None:
        with self._lock:
            return self._latest_result

    @property
    def fault(self) -> PerceptionRuntimeFault | None:
        with self._lock:
            return self._fault

    @property
    def has_pending_input(self) -> bool:
        with self._lock:
            return self._pending is not None

    @property
    def worker_alive(self) -> bool:
        with self._lock:
            worker = self._worker
        return worker is not None and worker.is_alive()

    @property
    def metrics(self) -> PerceptionRuntimeMetrics:
        with self._lock:
            return PerceptionRuntimeMetrics(
                inputs_received=self._inputs_received,
                inputs_accepted=self._inputs_accepted,
                inputs_processed=self._inputs_processed,
                duplicate_inputs_rejected=self._duplicate_inputs_rejected,
                out_of_order_inputs_rejected=self._out_of_order_inputs_rejected,
                stale_inputs_rejected=self._stale_inputs_rejected,
                inputs_abandoned=self._inputs_abandoned,
                source_frames_inferred_dropped=(
                    self._source_frames_inferred_dropped
                ),
                pending_frames_coalesced=self._pending_frames_coalesced,
                pending_inputs_discarded=self._pending_inputs_discarded,
                perception_successes=self._perception_successes,
                perception_failures=self._perception_failures,
                deadline_misses=self._deadline_misses,
                last_processing_latency_seconds=(
                    self._last_processing_latency_seconds
                ),
                maximum_processing_latency_seconds=(
                    self._maximum_processing_latency_seconds
                ),
                last_pending_age_seconds=self._last_pending_age_seconds,
                last_processed_source_stamp=self._last_processed_source_stamp,
            )

    def start(self) -> None:
        """Start one explicit non-daemon worker, or do nothing if running."""
        with self._condition:
            if self._state is PerceptionRuntimeState.RUNNING:
                return
            if self._state is PerceptionRuntimeState.STOPPING:
                raise RuntimeError("cannot start while runtime shutdown is incomplete")
            if self._state is PerceptionRuntimeState.FAULTED:
                raise RuntimeError("stop the faulted runtime before restarting it")
            self._reset_epoch_locked()
            self._stop_requested = False
            worker = threading.Thread(
                target=self._worker_main,
                name="PerceptionRuntime",
                daemon=False,
            )
            self._worker = worker
            self._state = PerceptionRuntimeState.RUNNING
            try:
                worker.start()
            except Exception:
                self._worker = None
                self._state = PerceptionRuntimeState.STOPPED
                raise

    def stop(self) -> None:
        """Request shutdown and join the worker within the configured timeout.

        A claimed perception call is allowed to finish.  The one unclaimed
        pending input, if any, is discarded and counted so it cannot leak into
        a later execution epoch.  A timeout leaves the runtime ``STOPPING``;
        calling ``stop`` again safely retries the join.
        """
        with self._condition:
            if self._state is PerceptionRuntimeState.STOPPED:
                return
            worker = self._worker
            if worker is threading.current_thread():
                raise RuntimeError("the perception worker cannot stop itself")
            if self._state is not PerceptionRuntimeState.STOPPING:
                self._state = PerceptionRuntimeState.STOPPING
                self._stop_requested = True
                if self._pending is not None:
                    self._pending = None
                    self._pending_inputs_discarded += 1
                self._condition.notify_all()

        if worker is not None:
            worker.join(self._config.shutdown_timeout_seconds)
            if worker.is_alive():
                raise TimeoutError(
                    "perception worker did not stop within "
                    f"{self._config.shutdown_timeout_seconds} seconds"
                )

        with self._condition:
            # Another stop caller may have joined the same old worker.  If it
            # already finalized this epoch, a subsequent start can install a
            # replacement before this caller reaches the lock.  A late
            # finalizer must never clobber that replacement's RUNNING state.
            if self._worker is not worker:
                return
            self._worker = None
            self._stop_requested = False
            self._state = PerceptionRuntimeState.STOPPED
            self._condition.notify_all()

    def submit(self, snapshot: TelemetrySnapshot) -> bool:
        """Submit one canonical camera snapshot without waiting for perception.

        Returns ``True`` only when the source evidence is newer and the runtime
        is running.  Valid attempts made during one running epoch are accounted
        as accepted, duplicate, or out-of-order.  Structurally invalid camera
        identity raises before metrics change; submissions outside ``RUNNING``
        return ``False`` and do not read the clock.  An ingestion-clock error
        propagates before the shared state or pending slot is changed.
        """
        source_stamp, source_sequence, revision = _submission_evidence(snapshot)
        with self._lock:
            if self._state is not PerceptionRuntimeState.RUNNING:
                return False
        ingested = _finite_real(self._clock(), "runtime ingestion clock")

        with self._condition:
            if self._state is not PerceptionRuntimeState.RUNNING:
                return False
            self._inputs_received += 1
            previous = self._last_accepted_stamp
            if previous is not None:
                duplicate = (
                    source_stamp.frame == previous.frame
                    and source_stamp.sim_time == previous.sim_time
                )
                regressed = (
                    source_stamp.frame <= previous.frame
                    or source_stamp.sim_time < previous.sim_time
                )
                if duplicate:
                    self._duplicate_inputs_rejected += 1
                    return False
                if regressed:
                    self._out_of_order_inputs_rejected += 1
                    return False

            previous_sequence = self._last_accepted_source_sequence
            if previous_sequence is not None and source_sequence is not None:
                if source_sequence > previous_sequence:
                    self._source_frames_inferred_dropped += max(
                        0, source_sequence - previous_sequence - 1
                    )
                    next_sequence = source_sequence
                else:
                    # No reset/wrap arithmetic exists in telemetry.  Break the
                    # inference chain rather than fabricate a large later gap.
                    next_sequence = None
            else:
                next_sequence = source_sequence

            self._last_accepted_stamp = source_stamp
            self._last_accepted_source_sequence = next_sequence
            self._inputs_accepted += 1
            if self._pending is not None:
                self._pending_frames_coalesced += 1
            self._pending = _PendingInput(
                snapshot=snapshot,
                source_stamp=source_stamp,
                source_sequence=source_sequence,
                snapshot_revision=revision,
                ingestion_monotonic=ingested,
            )
            self._condition.notify()
            return True

    def _reset_epoch_locked(self) -> None:
        self._pending = None
        self._latest_result = None
        self._fault = None
        self._last_accepted_stamp: SampleStamp | None = None
        self._last_accepted_source_sequence: int | None = None
        self._inputs_received = 0
        self._inputs_accepted = 0
        self._inputs_abandoned = 0
        self._inputs_processed = 0
        self._duplicate_inputs_rejected = 0
        self._out_of_order_inputs_rejected = 0
        self._stale_inputs_rejected = 0
        self._source_frames_inferred_dropped = 0
        self._pending_frames_coalesced = 0
        self._pending_inputs_discarded = 0
        self._perception_successes = 0
        self._perception_failures = 0
        self._deadline_misses = 0
        self._last_processing_latency_seconds: float | None = None
        self._maximum_processing_latency_seconds: float | None = None
        self._last_pending_age_seconds: float | None = None
        self._last_processed_source_stamp: SampleStamp | None = None

    def _worker_main(self) -> None:
        worker = threading.current_thread()
        item: _PendingInput | None = None
        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._stop_requested:
                        self._condition.wait()
                    if self._stop_requested:
                        return
                    item = self._pending
                    self._pending = None
                assert item is not None
                should_continue = self._process_item(item)
                # The long-lived worker frame must not retain the last snapshot
                # and its RGB payload while waiting for future work.
                item = None
                if not should_continue:
                    return
        finally:
            # This is lifecycle finalization, not BaseException handling.  An
            # escaping BaseException remains uncaught and reaches threading's
            # excepthook after ownership and accounting become truthful.
            self._finalize_worker_exit(worker, item)

    def _process_item(self, item: _PendingInput) -> bool:
        try:
            started = _finite_real(self._clock(), "runtime processing-start clock")
            pending_age = _duration(
                started, item.ingestion_monotonic, "pending age"
            )
        except Exception as exc:
            self._publish_fault(item, "clock", exc, None, None, processed=False)
            return False

        with self._lock:
            self._last_pending_age_seconds = pending_age
            if pending_age > self._config.max_pending_age_seconds:
                self._stale_inputs_rejected += 1
                return True

        try:
            observation = self._perception_callable(item.snapshot)
            if not isinstance(observation, LaneObservation):
                raise TypeError("perception callable must return a LaneObservation")
        except Exception as exc:
            finished: float | None = None
            latency: float | None = None
            try:
                finished = _finite_real(self._clock(), "runtime failure clock")
                latency = _duration(finished, started, "processing latency")
            except Exception:
                finished = None
                latency = None
            self._publish_fault(
                item,
                "perception",
                exc,
                finished,
                latency,
                processed=True,
            )
            return False

        try:
            finished = _finite_real(self._clock(), "runtime processing-finish clock")
            latency = _duration(finished, started, "processing latency")
        except Exception as exc:
            # The perception callable completed successfully.  A missing or
            # regressed finish timestamp is a runtime-clock fault, not a
            # perception failure, and the faulting clock is not retried.
            self._publish_fault(
                item,
                "clock",
                exc,
                None,
                None,
                processed=False,
            )
            return False

        try:
            result = PerceptionRuntimeResult(
                observation=observation,
                source_stamp=item.source_stamp,
                source_sequence=item.source_sequence,
                snapshot_revision=item.snapshot_revision,
                ingestion_monotonic=item.ingestion_monotonic,
                processing_start_monotonic=started,
                processing_finish_monotonic=finished,
                pending_age_seconds=pending_age,
                processing_latency_seconds=latency,
            )
        except Exception as exc:
            self._publish_fault(
                item,
                "result",
                exc,
                finished,
                latency,
                processed=False,
            )
            return False

        with self._condition:
            self._record_completed_invocation_locked(
                item.source_stamp, latency, success=True
            )
            self._latest_result = result
            self._condition.notify_all()
        return True

    def _record_completed_invocation_locked(
        self,
        source_stamp: SampleStamp,
        latency: float | None,
        *,
        success: bool,
    ) -> None:
        self._inputs_processed += 1
        self._last_processed_source_stamp = source_stamp
        if success:
            self._perception_successes += 1
        else:
            self._perception_failures += 1
        self._last_processing_latency_seconds = latency
        if latency is None:
            return
        if (
            self._maximum_processing_latency_seconds is None
            or latency > self._maximum_processing_latency_seconds
        ):
            self._maximum_processing_latency_seconds = latency
        if latency > self._config.processing_period_seconds:
            self._deadline_misses += 1

    def _record_abandoned_input_locked(self) -> None:
        self._inputs_abandoned += 1
        self._last_processing_latency_seconds = None

    def _publish_fault(
        self,
        item: _PendingInput,
        stage: str,
        exc: Exception,
        occurred: float | None,
        latency: float | None,
        *,
        processed: bool,
    ) -> None:
        fault = _fault_evidence(item, stage, exc, occurred)
        with self._condition:
            if processed:
                self._record_completed_invocation_locked(
                    item.source_stamp, latency, success=False
                )
            else:
                self._record_abandoned_input_locked()
            if self._pending is not None:
                self._pending = None
                self._pending_inputs_discarded += 1
            self._fault = fault
            self._state = PerceptionRuntimeState.FAULTED
            self._condition.notify_all()

    def _finalize_worker_exit(
        self, worker: threading.Thread, item: _PendingInput | None
    ) -> None:
        with self._condition:
            if self._worker is not worker:
                return
            unexpected = item is not None or self._state is PerceptionRuntimeState.RUNNING
            if not unexpected:
                self._condition.notify_all()
                return
            if item is not None:
                self._record_abandoned_input_locked()
            if self._pending is not None:
                self._pending = None
                self._pending_inputs_discarded += 1
            if self._fault is None:
                self._fault = _worker_exit_fault(item)
            self._stop_requested = True
            self._state = PerceptionRuntimeState.FAULTED
            self._condition.notify_all()


def _submission_evidence(
    snapshot: object,
) -> tuple[SampleStamp, int | None, int]:
    if not isinstance(snapshot, TelemetrySnapshot):
        raise TypeError("snapshot must be a TelemetrySnapshot")
    revision = _nonnegative_builtin_int(snapshot.revision, "snapshot revision")
    frame = snapshot.camera
    if not isinstance(frame, CameraFrame) or not isinstance(
        frame.metadata, CameraFrameMetadata
    ):
        raise ValueError("snapshot must contain canonical camera metadata")
    stamp = _validated_stamp(frame.metadata.stamp)
    sequence = _camera_source_sequence(snapshot)
    return stamp, sequence, revision


def _camera_source_sequence(snapshot: TelemetrySnapshot) -> int | None:
    if type(snapshot.statuses) is not tuple:
        return None
    statuses = tuple(
        status
        for status in snapshot.statuses
        if isinstance(status, SensorStatus) and status.name == "rgb_camera"
    )
    if len(statuses) != 1:
        return None
    status = statuses[0]
    count = status.accepted_count
    if (
        status.kind is not SensorKind.CONTINUOUS
        or isinstance(count, bool)
        or type(count) is not int
        or count < 1
    ):
        return None
    return count


def _fault_evidence(
    item: _PendingInput,
    stage: str,
    exc: Exception,
    occurred: float | None,
) -> PerceptionRuntimeFault:
    try:
        text = str(exc)
    except Exception:
        text = "<unprintable exception>"
    message = bounded_text(text) or "<no detail>"
    exception_class = type(exc)
    # Read class metadata through ``type`` itself so a hostile metaclass or a
    # non-string ``__module__`` value cannot execute formatting code while an
    # ordinary perception Exception is being contained.
    try:
        module = type.__getattribute__(exception_class, "__module__")
        qualname = type.__getattribute__(exception_class, "__qualname__")
    except Exception:
        module = qualname = None
    if type(module) is str and type(qualname) is str and module and qualname:
        raw_exception_type = module + "." + qualname
    else:
        raw_exception_type = "<unavailable exception type>"
    exception_type = (
        bounded_text(raw_exception_type) or "<unavailable exception type>"
    )
    return PerceptionRuntimeFault(
        stage=stage,
        exception_type=exception_type,
        message=message,
        source_stamp=item.source_stamp,
        source_sequence=item.source_sequence,
        snapshot_revision=item.snapshot_revision,
        occurred_monotonic=occurred,
    )


def _worker_exit_fault(item: _PendingInput | None) -> PerceptionRuntimeFault:
    return PerceptionRuntimeFault(
        stage="worker",
        exception_type="<uncaught BaseException>",
        message="perception worker exited unexpectedly",
        source_stamp=None if item is None else item.source_stamp,
        source_sequence=None if item is None else item.source_sequence,
        snapshot_revision=None if item is None else item.snapshot_revision,
        occurred_monotonic=None,
    )
