"""Mission 6B bounded live-perception observation harness.

This module is deliberately inert when imported.  It uses only the Python
standard library until :func:`run_live_perception_integration` has passed the
exact ``CARLA_INTEGRATION=1`` gate.  In particular, normal import and test
collection do not import CARLA, NumPy, the perception runtime, the live bridge,
or the sensor suite; they do not connect, spawn actors, start workers, or
mutate the environment.

The live entry only observes an already-running CARLA server.  It never
launches CARLA and never applies vehicle control.  The lifecycle itself is
dependency-injected so all orchestration and cleanup behavior can be tested
offline with fakes.
"""

from __future__ import annotations

import importlib
import math
import os
import time
import unittest
import unicodedata
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from numbers import Real


CARLA_INTEGRATION = "CARLA_INTEGRATION"
CARLA_HOST = "CARLA_HOST"
CARLA_PORT = "CARLA_PORT"
CARLA_TIMEOUT_SECONDS = "CARLA_TIMEOUT_SECONDS"
CARLA_INTEGRATION_CYCLES = "CARLA_INTEGRATION_CYCLES"
CARLA_INTEGRATION_TICK_OWNER = "CARLA_INTEGRATION_TICK_OWNER"

CARLA_PERCEPTION_FRAME_TARGET = "CARLA_PERCEPTION_FRAME_TARGET"
CARLA_PERCEPTION_DURATION_SECONDS = "CARLA_PERCEPTION_DURATION_SECONDS"
CARLA_PERCEPTION_TARGET_HZ = "CARLA_PERCEPTION_TARGET_HZ"
CARLA_PERCEPTION_MAX_PENDING_AGE_SECONDS = (
    "CARLA_PERCEPTION_MAX_PENDING_AGE_SECONDS"
)
CARLA_PERCEPTION_SHUTDOWN_TIMEOUT_SECONDS = (
    "CARLA_PERCEPTION_SHUTDOWN_TIMEOUT_SECONDS"
)

DISABLED_MESSAGE = (
    "live perception integration disabled; set CARLA_INTEGRATION=1"
)

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 2000
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_CYCLES = 1
DEFAULT_TICK_OWNER = False
DEFAULT_FRAME_TARGET = 10
DEFAULT_TARGET_HZ = 10.0
DEFAULT_MAX_PENDING_AGE_SECONDS = 0.25
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 2.0

MAX_CYCLES = 20
MAX_FRAME_TARGET = 10_000
MAX_DURATION_SECONDS = 3_600.0
MAX_PROGRESSION_STEPS = 100_000
MAX_OWNED_ACTOR_IDS = 16
EXPECTED_SENSOR_NAMES = (
    "gnss",
    "imu",
    "collision",
    "lane_invasion",
    "rgb_camera",
)
MAX_COMPONENT_FAULTS = 12
MAX_FAULT_COMPONENT_LENGTH = 48
MAX_FAULT_STAGE_LENGTH = 64
MAX_FAULT_TYPE_LENGTH = 64
MAX_FAULT_MESSAGE_LENGTH = 256
MAX_CLEANUP_DETAIL_LENGTH = 192
FIRST_RETAINED_SAMPLES = 8
FINAL_RETAINED_SAMPLES = 16
FIRST_RETAINED_CYCLES = 4
FINAL_RETAINED_CYCLES = 8


class ObservationMode(Enum):
    FRAME_TARGET = "FRAME_TARGET"
    DURATION = "DURATION"


class CleanupStage(Enum):
    CLOSE_BRIDGE_ADMISSION = "close_bridge_admission"
    DRAIN_BRIDGE = "drain_bridge"
    DESTROY_SENSOR_SUITE = "destroy_sensor_suite"
    STOP_PERCEPTION_RUNTIME = "stop_perception_runtime"
    DESTROY_EGO = "destroy_ego"
    VERIFY_ACTOR_DISAPPEARANCE = "verify_actor_disappearance"
    RELEASE_REFERENCES = "release_references"
    VERIFY_RUNTIME_WORKER = "verify_runtime_worker"


CLEANUP_ORDER = tuple(CleanupStage)


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def integration_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return true only for the exact opt-in string ``"1"``."""
    return _environment(environ).get(CARLA_INTEGRATION) == "1"


def _require_text(value: object, variable: str, accepted: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{variable} must be {accepted}")
    return value


def _parse_integer(
    value: object, variable: str, lower: int, upper: int
) -> int:
    accepted = f"a base-10 integer in {lower}..{upper}"
    text = _require_text(value, variable, accepted)
    if not text or not text.isascii() or not text.isdecimal():
        raise ValueError(f"{variable} must be {accepted}")
    parsed = int(text, 10)
    if not lower <= parsed <= upper:
        raise ValueError(f"{variable} must be {accepted}")
    return parsed


def _parse_float(
    value: object,
    variable: str,
    lower: float,
    upper: float,
    *,
    allow_zero: bool = False,
) -> float:
    accepted = (
        f"numeric zero or a finite number in {lower}..{upper}"
        if allow_zero
        else f"a finite number in {lower}..{upper}"
    )
    text = _require_text(value, variable, accepted)
    try:
        parsed = float(text)
    except ValueError:
        raise ValueError(f"{variable} must be {accepted}") from None
    in_range = lower <= parsed <= upper
    if not math.isfinite(parsed) or not (in_range or (allow_zero and parsed == 0.0)):
        raise ValueError(f"{variable} must be {accepted}")
    return parsed


@dataclass(frozen=True, slots=True)
class PerceptionIntegrationConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    observation_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    cycles: int = DEFAULT_CYCLES
    tick_owner: bool = DEFAULT_TICK_OWNER
    frame_target: int | None = DEFAULT_FRAME_TARGET
    duration_seconds: float | None = None
    target_hz: float = DEFAULT_TARGET_HZ
    max_pending_age_seconds: float = DEFAULT_MAX_PENDING_AGE_SECONDS
    shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if type(self.host) is not str or not 1 <= len(self.host) <= 255:
            raise ValueError("host must contain 1..255 characters")
        if type(self.port) is not int or not 1 <= self.port <= 65_535:
            raise ValueError("port must be a built-in integer in 1..65535")
        _finite_range(
            self.observation_timeout_seconds,
            "observation_timeout_seconds",
            0.001,
            120.0,
        )
        if type(self.cycles) is not int or not 1 <= self.cycles <= MAX_CYCLES:
            raise ValueError(f"cycles must be a built-in integer in 1..{MAX_CYCLES}")
        if type(self.tick_owner) is not bool:
            raise ValueError("tick_owner must be bool")
        if self.frame_target is None:
            if self.duration_seconds is None:
                raise ValueError("one observation bound must be configured")
        elif type(self.frame_target) is not int or not (
            1 <= self.frame_target <= MAX_FRAME_TARGET
        ):
            raise ValueError(
                f"frame_target must be None or a built-in integer in 1..{MAX_FRAME_TARGET}"
            )
        if self.duration_seconds is not None:
            _finite_range(
                self.duration_seconds,
                "duration_seconds",
                0.001,
                MAX_DURATION_SECONDS,
            )
            if self.frame_target is not None:
                raise ValueError("frame_target and duration_seconds are mutually exclusive")
        _finite_range(self.target_hz, "target_hz", 1.0 / 60.0, 1_000.0)
        _finite_range(
            self.max_pending_age_seconds,
            "max_pending_age_seconds",
            0.001,
            60.0,
        )
        _finite_range(
            self.shutdown_timeout_seconds,
            "shutdown_timeout_seconds",
            0.001,
            60.0,
        )

    @property
    def observation_mode(self) -> ObservationMode:
        if self.duration_seconds is not None:
            return ObservationMode.DURATION
        return ObservationMode.FRAME_TARGET


def _finite_range(value: object, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    parsed = float(value)
    if not math.isfinite(parsed) or not lower <= parsed <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    return parsed


def parse_integration_config(
    environ: Mapping[str, str] | None = None,
) -> PerceptionIntegrationConfig:
    """Parse the bounded Mission 6B environment contract without live work."""
    source = _environment(environ)
    host_text = _require_text(
        source.get(CARLA_HOST, DEFAULT_HOST),
        CARLA_HOST,
        "text containing 1..255 characters after stripping whitespace",
    )
    host = host_text.strip()
    if not 1 <= len(host) <= 255:
        raise ValueError(
            f"{CARLA_HOST} must contain 1..255 characters after stripping whitespace"
        )

    port = _parse_integer(
        source.get(CARLA_PORT, str(DEFAULT_PORT)), CARLA_PORT, 1, 65_535
    )
    observation_timeout = _parse_float(
        source.get(CARLA_TIMEOUT_SECONDS, str(DEFAULT_TIMEOUT_SECONDS)),
        CARLA_TIMEOUT_SECONDS,
        0.001,
        120.0,
    )
    cycles = _parse_integer(
        source.get(CARLA_INTEGRATION_CYCLES, str(DEFAULT_CYCLES)),
        CARLA_INTEGRATION_CYCLES,
        1,
        MAX_CYCLES,
    )
    tick_text = source.get(CARLA_INTEGRATION_TICK_OWNER, "0")
    if tick_text not in ("0", "1"):
        raise ValueError(f"{CARLA_INTEGRATION_TICK_OWNER} must be one of: 0, 1")

    duration_explicit = CARLA_PERCEPTION_DURATION_SECONDS in source
    duration_raw = source.get(CARLA_PERCEPTION_DURATION_SECONDS, "0")
    duration = _parse_float(
        duration_raw,
        CARLA_PERCEPTION_DURATION_SECONDS,
        0.001,
        MAX_DURATION_SECONDS,
        allow_zero=True,
    )
    frame_explicit = CARLA_PERCEPTION_FRAME_TARGET in source
    if duration > 0.0:
        if frame_explicit:
            raise ValueError(
                f"positive {CARLA_PERCEPTION_DURATION_SECONDS} cannot be combined "
                f"with explicit {CARLA_PERCEPTION_FRAME_TARGET}"
            )
        frame_target = None
        duration_seconds: float | None = duration
    else:
        # An explicit numeric zero selects the normal frame-target mode.
        del duration_explicit
        frame_target = _parse_integer(
            source.get(CARLA_PERCEPTION_FRAME_TARGET, str(DEFAULT_FRAME_TARGET)),
            CARLA_PERCEPTION_FRAME_TARGET,
            1,
            MAX_FRAME_TARGET,
        )
        duration_seconds = None

    return PerceptionIntegrationConfig(
        host=host,
        port=port,
        observation_timeout_seconds=observation_timeout,
        cycles=cycles,
        tick_owner=tick_text == "1",
        frame_target=frame_target,
        duration_seconds=duration_seconds,
        target_hz=_parse_float(
            source.get(CARLA_PERCEPTION_TARGET_HZ, str(DEFAULT_TARGET_HZ)),
            CARLA_PERCEPTION_TARGET_HZ,
            1.0 / 60.0,
            1_000.0,
        ),
        max_pending_age_seconds=_parse_float(
            source.get(
                CARLA_PERCEPTION_MAX_PENDING_AGE_SECONDS,
                str(DEFAULT_MAX_PENDING_AGE_SECONDS),
            ),
            CARLA_PERCEPTION_MAX_PENDING_AGE_SECONDS,
            0.001,
            60.0,
        ),
        shutdown_timeout_seconds=_parse_float(
            source.get(
                CARLA_PERCEPTION_SHUTDOWN_TIMEOUT_SECONDS,
                str(DEFAULT_SHUTDOWN_TIMEOUT_SECONDS),
            ),
            CARLA_PERCEPTION_SHUTDOWN_TIMEOUT_SECONDS,
            0.001,
            60.0,
        ),
    )


def require_integration_opt_in(
    environ: Mapping[str, str] | None = None,
) -> PerceptionIntegrationConfig:
    """Gate before parsing so disabled runs ignore all secondary values."""
    if not integration_enabled(environ):
        raise unittest.SkipTest(DISABLED_MESSAGE)
    return parse_integration_config(environ)


class PhaseTimeout(TimeoutError):
    def __init__(self, phase: str, timeout_seconds: float) -> None:
        self.phase = phase
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"phase {phase!r} exceeded {float(timeout_seconds):.3f} seconds"
        )


class Deadline:
    """One clamped host-monotonic deadline for observation or cleanup."""

    __slots__ = (
        "_clock",
        "_expired",
        "_last",
        "expiry",
        "start",
        "timeout_seconds",
    )

    def __init__(self, timeout_seconds: float, clock: Callable[[], float]) -> None:
        self.timeout_seconds = _finite_range(
            timeout_seconds, "timeout_seconds", 0.001, MAX_DURATION_SECONDS
        )
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self.start = self._read()
        self.expiry = self.start + self.timeout_seconds
        if not math.isfinite(self.expiry) or self.expiry <= self.start:
            raise ValueError("deadline expiry must advance beyond its start")
        self._last = self.start
        self._expired = False

    def _read(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("monotonic clock must return a finite real")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("monotonic clock must return a finite real")
        return parsed

    def _now(self) -> float:
        value = self._read()
        if value < self._last:
            return self._last
        self._last = value
        return value

    def remaining(self) -> float:
        now = self._now()
        if now >= self.expiry:
            self._expired = True
        return 0.0 if self._expired else max(0.0, self.expiry - now)

    def expired(self) -> bool:
        self.remaining()
        return self._expired

    def elapsed(self) -> float:
        return max(0.0, self._now() - self.start)

    def raise_if_expired(self, phase: str) -> None:
        if self.expired():
            raise PhaseTimeout(phase, self.timeout_seconds)


def _bounded_text(value: object, limit: int) -> str:
    try:
        raw = str(value)
    except BaseException as exc:  # diagnostics must not block cleanup
        raw = f"<unprintable: {type(exc).__name__}>"
    without_controls = "".join(
        " " if unicodedata.category(character) == "Cc" else character
        for character in raw
    )
    text = " ".join(without_controls.split()) or "<no message>"
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


@dataclass(frozen=True, slots=True)
class ComponentFault:
    component: str
    stage: str
    exception_type: str
    message: str


class _FaultLedger:
    """Retain the first bounded immutable fault for each bounded component."""

    __slots__ = ("_faults", "_seen", "omitted")

    def __init__(self) -> None:
        self._faults: list[ComponentFault] = []
        self._seen: set[str] = set()
        self.omitted = 0

    def add(self, component: str, stage: str, value: object) -> None:
        bounded_component = _bounded_text(component, MAX_FAULT_COMPONENT_LENGTH)
        if bounded_component in self._seen:
            return
        self._seen.add(bounded_component)
        if len(self._faults) >= MAX_COMPONENT_FAULTS:
            self.omitted += 1
            return
        if isinstance(value, BaseException):
            type_name = type(value).__name__
            message = _bounded_text(value, MAX_FAULT_MESSAGE_LENGTH)
        else:
            type_name = type(value).__name__
            message = _bounded_text(value, MAX_FAULT_MESSAGE_LENGTH)
        self._faults.append(
            ComponentFault(
                component=bounded_component,
                stage=_bounded_text(stage, MAX_FAULT_STAGE_LENGTH),
                exception_type=_bounded_text(type_name, MAX_FAULT_TYPE_LENGTH),
                message=message,
            )
        )

    @property
    def values(self) -> tuple[ComponentFault, ...]:
        return tuple(self._faults)


@dataclass(frozen=True, slots=True)
class SourceStampEvidence:
    frame: int
    sim_time: float
    monotonic: float


@dataclass(frozen=True, slots=True)
class BridgeMetricsEvidence:
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
    last_submitted_source_stamp: SourceStampEvidence | None


@dataclass(frozen=True, slots=True)
class RuntimeMetricsEvidence:
    inputs_received: int
    inputs_accepted: int
    inputs_processed: int
    duplicate_inputs_rejected: int
    out_of_order_inputs_rejected: int
    stale_inputs_rejected: int
    source_frames_inferred_dropped: int
    pending_frames_coalesced: int
    pending_inputs_discarded: int
    inputs_abandoned: int
    perception_successes: int
    perception_failures: int
    deadline_misses: int
    last_processed_source_stamp: SourceStampEvidence | None


@dataclass(frozen=True, slots=True)
class ObservationSample:
    accepted_camera_snapshots: int
    runtime_submissions_accepted: int
    runtime_inputs_processed: int
    source_stamp: SourceStampEvidence | None


@dataclass(frozen=True, slots=True)
class CleanupStepResult:
    stage: CleanupStage
    attempted: bool
    succeeded: bool
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class DisappearanceEvidence:
    verified: bool
    remaining_actor_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PerceptionCycleReport:
    cycle_number: int
    success: bool
    observation_completed: bool
    observation_mode: ObservationMode
    accepted_camera_snapshots: int
    elapsed_seconds: float
    samples: tuple[ObservationSample, ...]
    omitted_sample_count: int
    bridge_metrics: BridgeMetricsEvidence | None
    runtime_metrics: RuntimeMetricsEvidence | None
    bridge_state: str | None
    runtime_state: str | None
    faults: tuple[ComponentFault, ...]
    omitted_fault_count: int
    owned_actor_ids: tuple[int, ...]
    remaining_actor_ids: tuple[int, ...]
    disappearance_verified: bool
    cleanup_steps: tuple[CleanupStepResult, ...]
    runtime_worker_alive_after_cleanup: bool | None


@dataclass(frozen=True, slots=True)
class PerceptionIntegrationReport:
    config: PerceptionIntegrationConfig
    success: bool
    requested_cycle_count: int
    attempted_cycle_count: int
    passed_cycle_count: int
    failed_cycle_number: int | None
    retained_cycle_reports: tuple[PerceptionCycleReport, ...]
    omitted_cycle_report_count: int


class _BoundedRetention:
    __slots__ = ("_final", "_first", "_first_limit", "total")

    def __init__(self, first_limit: int, final_limit: int) -> None:
        self._first_limit = first_limit
        self._first: list[object] = []
        self._final: deque[object] = deque(maxlen=final_limit)
        self.total = 0

    def add(self, value: object) -> None:
        self.total += 1
        if len(self._first) < self._first_limit:
            self._first.append(value)
        else:
            self._final.append(value)

    @property
    def values(self) -> tuple:
        return tuple(self._first) + tuple(self._final)

    @property
    def omitted(self) -> int:
        return max(0, self.total - len(self.values))


def _builtin_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be a non-negative built-in integer")
    return value


def _optional_finite_nonnegative(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be None or a finite non-negative real")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise TypeError(f"{name} must be None or a finite non-negative real")
    return parsed


def _copy_stamp(value: object | None) -> SourceStampEvidence | None:
    if value is None:
        return None
    frame = getattr(value, "frame")
    sim_time = getattr(value, "sim_time")
    monotonic = getattr(value, "monotonic")
    if type(frame) is not int or frame < 0:
        raise TypeError("source stamp frame must be a non-negative built-in integer")
    return SourceStampEvidence(
        frame=frame,
        sim_time=_finite_range(sim_time, "source stamp sim_time", 0.0, 1.0e15),
        monotonic=_finite_range(
            monotonic, "source stamp monotonic", -1.0e15, 1.0e15
        ),
    )


def _copy_bridge_metrics(bridge: object) -> BridgeMetricsEvidence:
    metrics = getattr(bridge, "metrics")
    return BridgeMetricsEvidence(
        submissions_received=_builtin_nonnegative_int(
            getattr(metrics, "submissions_received"), "submissions_received"
        ),
        submissions_admitted=_builtin_nonnegative_int(
            getattr(metrics, "submissions_admitted"), "submissions_admitted"
        ),
        runtime_submissions_accepted=_builtin_nonnegative_int(
            getattr(metrics, "runtime_submissions_accepted"),
            "runtime_submissions_accepted",
        ),
        runtime_submissions_rejected=_builtin_nonnegative_int(
            getattr(metrics, "runtime_submissions_rejected"),
            "runtime_submissions_rejected",
        ),
        submissions_rejected_closed=_builtin_nonnegative_int(
            getattr(metrics, "submissions_rejected_closed"),
            "submissions_rejected_closed",
        ),
        submission_failures=_builtin_nonnegative_int(
            getattr(metrics, "submission_failures"), "submission_failures"
        ),
        current_in_flight=_builtin_nonnegative_int(
            getattr(metrics, "current_in_flight"), "current_in_flight"
        ),
        maximum_in_flight=_builtin_nonnegative_int(
            getattr(metrics, "maximum_in_flight"), "maximum_in_flight"
        ),
        last_callback_to_submit_latency_seconds=_optional_finite_nonnegative(
            getattr(metrics, "last_callback_to_submit_latency_seconds"),
            "last_callback_to_submit_latency_seconds",
        ),
        maximum_callback_to_submit_latency_seconds=_optional_finite_nonnegative(
            getattr(metrics, "maximum_callback_to_submit_latency_seconds"),
            "maximum_callback_to_submit_latency_seconds",
        ),
        last_submitted_source_stamp=_copy_stamp(
            getattr(metrics, "last_submitted_source_stamp")
        ),
    )


def _copy_runtime_metrics(runtime: object) -> RuntimeMetricsEvidence:
    metrics = getattr(runtime, "metrics")
    return RuntimeMetricsEvidence(
        inputs_received=_builtin_nonnegative_int(
            getattr(metrics, "inputs_received"), "inputs_received"
        ),
        inputs_accepted=_builtin_nonnegative_int(
            getattr(metrics, "inputs_accepted"), "inputs_accepted"
        ),
        inputs_processed=_builtin_nonnegative_int(
            getattr(metrics, "inputs_processed"), "inputs_processed"
        ),
        duplicate_inputs_rejected=_builtin_nonnegative_int(
            getattr(metrics, "duplicate_inputs_rejected"),
            "duplicate_inputs_rejected",
        ),
        out_of_order_inputs_rejected=_builtin_nonnegative_int(
            getattr(metrics, "out_of_order_inputs_rejected"),
            "out_of_order_inputs_rejected",
        ),
        stale_inputs_rejected=_builtin_nonnegative_int(
            getattr(metrics, "stale_inputs_rejected"), "stale_inputs_rejected"
        ),
        source_frames_inferred_dropped=_builtin_nonnegative_int(
            getattr(metrics, "source_frames_inferred_dropped"),
            "source_frames_inferred_dropped",
        ),
        pending_frames_coalesced=_builtin_nonnegative_int(
            getattr(metrics, "pending_frames_coalesced"),
            "pending_frames_coalesced",
        ),
        pending_inputs_discarded=_builtin_nonnegative_int(
            getattr(metrics, "pending_inputs_discarded"),
            "pending_inputs_discarded",
        ),
        inputs_abandoned=_builtin_nonnegative_int(
            getattr(metrics, "inputs_abandoned"), "inputs_abandoned"
        ),
        perception_successes=_builtin_nonnegative_int(
            getattr(metrics, "perception_successes"), "perception_successes"
        ),
        perception_failures=_builtin_nonnegative_int(
            getattr(metrics, "perception_failures"), "perception_failures"
        ),
        deadline_misses=_builtin_nonnegative_int(
            getattr(metrics, "deadline_misses"), "deadline_misses"
        ),
        last_processed_source_stamp=_copy_stamp(
            getattr(metrics, "last_processed_source_stamp")
        ),
    )


def _active_integration_evidence(
    bridge: BridgeMetricsEvidence, runtime: RuntimeMetricsEvidence
) -> bool:
    """Require proof that callbacks crossed both bridge and runtime seams."""
    return (
        bridge.runtime_submissions_accepted > 0
        and bridge.runtime_submissions_accepted == runtime.inputs_accepted
        and runtime.inputs_processed > 0
    )


def _terminal_metrics_coherent(
    bridge: BridgeMetricsEvidence, runtime: RuntimeMetricsEvidence
) -> bool:
    """Check fresh-cycle bridge/runtime accounting after both are quiescent."""
    bridge_partition = (
        bridge.submissions_admitted
        == bridge.runtime_submissions_accepted
        + bridge.runtime_submissions_rejected
        + bridge.submission_failures
    )
    runtime_received_partition = (
        runtime.inputs_received
        == runtime.inputs_accepted
        + runtime.duplicate_inputs_rejected
        + runtime.out_of_order_inputs_rejected
    )
    runtime_accepted_partition = (
        runtime.inputs_accepted
        == runtime.inputs_processed
        + runtime.stale_inputs_rejected
        + runtime.pending_frames_coalesced
        + runtime.pending_inputs_discarded
        + runtime.inputs_abandoned
    )
    runtime_processed_partition = (
        runtime.inputs_processed
        == runtime.perception_successes + runtime.perception_failures
    )
    return (
        bridge.current_in_flight == 0
        and _active_integration_evidence(bridge, runtime)
        and bridge.runtime_submissions_accepted == runtime.inputs_accepted
        and bridge_partition
        and runtime_received_partition
        and runtime_accepted_partition
        and runtime_processed_partition
    )


def _state_name(owner: object) -> str:
    state = getattr(owner, "state")
    value = getattr(state, "value", state)
    if type(value) is not str:
        raise TypeError("state value must be text")
    return _bounded_text(value, 32)


def _copy_component_fault(
    ledger: _FaultLedger, component: str, owner: object
) -> None:
    value = getattr(owner, "fault")
    if value is None:
        return
    stage = getattr(value, "stage", "reported_fault")
    exception_type = getattr(value, "exception_type", type(value).__name__)
    message = getattr(value, "message", value)
    bounded = ComponentFault(
        component=_bounded_text(component, MAX_FAULT_COMPONENT_LENGTH),
        stage=_bounded_text(stage, MAX_FAULT_STAGE_LENGTH),
        exception_type=_bounded_text(exception_type, MAX_FAULT_TYPE_LENGTH),
        message=_bounded_text(message, MAX_FAULT_MESSAGE_LENGTH),
    )
    if bounded.component in ledger._seen:
        return
    ledger._seen.add(bounded.component)
    if len(ledger._faults) >= MAX_COMPONENT_FAULTS:
        ledger.omitted += 1
    else:
        ledger._faults.append(bounded)


def _positive_actor_ids(values: Iterable[object]) -> tuple[int, ...]:
    result: list[int] = []
    seen: set[int] = set()
    iterator = iter(values)
    for value in islice(iterator, MAX_OWNED_ACTOR_IDS + 1):
        if len(result) >= MAX_OWNED_ACTOR_IDS:
            raise ValueError(
                f"owned actor identity count exceeds {MAX_OWNED_ACTOR_IDS}"
            )
        if type(value) is not int or value <= 0:
            raise ValueError("owned actor ids must be positive built-in integers")
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _merge_actor_ids(existing: tuple[int, ...], added: Iterable[object]) -> tuple[int, ...]:
    return _positive_actor_ids((*existing, *tuple(islice(iter(added), MAX_OWNED_ACTOR_IDS + 1))))


def _required_operation(operations: object, name: str) -> Callable:
    value = getattr(operations, name)
    if not callable(value):
        raise TypeError(f"operations.{name} must be callable")
    return value


def _cleanup_result(
    stage: CleanupStage,
    attempted: bool,
    succeeded: bool,
    detail: object | None = None,
) -> CleanupStepResult:
    return CleanupStepResult(
        stage=stage,
        attempted=attempted,
        succeeded=succeeded,
        detail=(
            None
            if detail is None
            else _bounded_text(detail, MAX_CLEANUP_DETAIL_LENGTH)
        ),
    )


def _normalise_disappearance(value: object) -> DisappearanceEvidence:
    if isinstance(value, DisappearanceEvidence):
        return value
    remaining = getattr(value, "remaining_actor_ids", None)
    verified = getattr(value, "verified", None)
    if remaining is None and type(value) is tuple:
        remaining = value
    if remaining is None:
        raise TypeError("disappearance result must expose remaining_actor_ids")
    ids = _positive_actor_ids(remaining)
    if verified is None:
        verification = getattr(value, "verification", None)
        state = getattr(verification, "value", verification)
        verified = state == "CONFIRMED"
    if type(verified) is not bool:
        raise TypeError("disappearance verified flag must be bool")
    return DisappearanceEvidence(verified=verified, remaining_actor_ids=ids)


def run_perception_cycle(
    config: PerceptionIntegrationConfig,
    cycle_number: int,
    operations: object,
    *,
    monotonic: Callable[[], float] = time.monotonic,
) -> PerceptionCycleReport:
    """Run one fresh injected cycle and deterministically reach all cleanup stages."""
    if not isinstance(config, PerceptionIntegrationConfig):
        raise TypeError("config must be PerceptionIntegrationConfig")
    if type(cycle_number) is not int or cycle_number <= 0:
        raise ValueError("cycle_number must be a positive built-in integer")
    if not callable(monotonic):
        raise TypeError("monotonic must be callable")

    faults = _FaultLedger()
    samples = _BoundedRetention(FIRST_RETAINED_SAMPLES, FINAL_RETAINED_SAMPLES)
    cleanup_steps: list[CleanupStepResult] = []
    owned_actor_ids: tuple[int, ...] = ()
    remaining_actor_ids: tuple[int, ...] = ()
    disappearance_verified = False
    runtime_worker_alive_after_cleanup: bool | None = None
    bridge_metrics: BridgeMetricsEvidence | None = None
    runtime_metrics: RuntimeMetricsEvidence | None = None
    bridge_state: str | None = None
    runtime_state: str | None = None
    perception_input_usable = False
    accepted_count = 0
    observation_completed = False
    elapsed_seconds = 0.0
    last_sample_count: int | None = None
    deferred: tuple[BaseException, object] | None = None
    bridge_drain_needs_retry = False
    runtime_stop_needs_retry = False

    context = None
    ego = None
    runtime = None
    bridge = None
    suite = None
    deadline: Deadline | None = None

    def remember_base_exception(exc: BaseException) -> None:
        nonlocal deferred
        if deferred is None and not isinstance(exc, Exception):
            deferred = (exc, exc.__traceback__)

    def record_exception(component: str, stage: str, exc: BaseException) -> None:
        faults.add(component, stage, exc)
        remember_base_exception(exc)

    try:
        try:
            context = _required_operation(operations, "prepare_cycle")(config)
            ego = _required_operation(operations, "spawn_ego")(context, config)
            ego_id = _required_operation(operations, "ego_actor_id")(ego)
            owned_actor_ids = _merge_actor_ids(owned_actor_ids, (ego_id,))
            runtime = _required_operation(operations, "create_runtime")(config)
            bridge = _required_operation(operations, "create_bridge")(
                runtime, config, monotonic
            )
            suite = _required_operation(operations, "create_suite")(
                bridge, config, monotonic
            )
            _required_operation(operations, "attach_suite")(
                suite, context, ego, config
            )
            _required_operation(operations, "start_runtime")(runtime)

            observation_bound = (
                config.duration_seconds
                if config.duration_seconds is not None
                else config.observation_timeout_seconds
            )
            assert observation_bound is not None
            deadline = Deadline(observation_bound, monotonic)

            for _step_number in range(MAX_PROGRESSION_STEPS):
                try:
                    bridge_metrics = _copy_bridge_metrics(bridge)
                    runtime_metrics = _copy_runtime_metrics(runtime)
                    perception_input_usable = _required_operation(
                        operations, "perception_input_usable"
                    )(runtime)
                    if type(perception_input_usable) is not bool:
                        raise TypeError(
                            "perception_input_usable must return a built-in bool"
                        )
                    bridge_state = _state_name(bridge)
                    runtime_state = _state_name(runtime)
                    _copy_component_fault(faults, "bridge", bridge)
                    _copy_component_fault(faults, "runtime", runtime)
                    accepted_count = bridge_metrics.submissions_admitted
                    if last_sample_count != accepted_count:
                        samples.add(
                            ObservationSample(
                                accepted_camera_snapshots=accepted_count,
                                runtime_submissions_accepted=(
                                    bridge_metrics.runtime_submissions_accepted
                                ),
                                runtime_inputs_processed=runtime_metrics.inputs_processed,
                                source_stamp=bridge_metrics.last_submitted_source_stamp,
                            )
                        )
                        last_sample_count = accepted_count
                except BaseException as exc:
                    record_exception("reporting", "observation_metrics", exc)
                    break

                if faults.values:
                    break
                try:
                    worker_alive = _required_operation(
                        operations, "worker_alive"
                    )(runtime)
                    if type(worker_alive) is not bool:
                        raise TypeError("worker_alive must return bool")
                    if not worker_alive:
                        faults.add(
                            "runtime_worker",
                            "observation",
                            "runtime worker exited before observation completed",
                        )
                        break
                except BaseException as exc:
                    record_exception("runtime_worker", "observation", exc)
                    break

                if (
                    config.observation_mode is ObservationMode.FRAME_TARGET
                    and accepted_count >= config.frame_target
                    and _active_integration_evidence(
                        bridge_metrics, runtime_metrics
                    )
                    and perception_input_usable
                ):
                    observation_completed = True
                    break
                if deadline.expired():
                    if config.observation_mode is ObservationMode.DURATION:
                        active_evidence = _active_integration_evidence(
                            bridge_metrics, runtime_metrics
                        )
                        observation_completed = (
                            active_evidence and perception_input_usable
                        )
                        if not observation_completed:
                            if active_evidence and not perception_input_usable:
                                faults.add(
                                    "sensor_health",
                                    "required_rgb_status",
                                    "no processed snapshot satisfied required RGB sensor health",
                                )
                            else:
                                faults.add(
                                    "observation",
                                    "duration",
                                    "duration ended without an accepted camera snapshot",
                                )
                    else:
                        active_evidence = _active_integration_evidence(
                            bridge_metrics, runtime_metrics
                        )
                        if (
                            accepted_count >= config.frame_target
                            and active_evidence
                            and not perception_input_usable
                        ):
                            faults.add(
                                "sensor_health",
                                "required_rgb_status",
                                "no processed snapshot satisfied required RGB sensor health",
                            )
                        else:
                            faults.add(
                                "observation",
                                "frame_target_timeout",
                                PhaseTimeout(
                                    "accepted_camera_frame_target",
                                    deadline.timeout_seconds,
                                ),
                            )
                    break

                try:
                    _required_operation(operations, "progress_once")(
                        context, deadline
                    )
                except TimeoutError as exc:
                    if (
                        config.observation_mode is ObservationMode.DURATION
                        and deadline.expired()
                    ):
                        observation_completed = (
                            bridge_metrics is not None
                            and runtime_metrics is not None
                            and _active_integration_evidence(
                                bridge_metrics, runtime_metrics
                            )
                            and perception_input_usable
                        )
                        if not observation_completed:
                            active_evidence = (
                                bridge_metrics is not None
                                and runtime_metrics is not None
                                and _active_integration_evidence(
                                    bridge_metrics, runtime_metrics
                                )
                            )
                            if active_evidence and not perception_input_usable:
                                faults.add(
                                    "sensor_health",
                                    "required_rgb_status",
                                    "no processed snapshot satisfied required RGB sensor health",
                                )
                            else:
                                faults.add("observation", "duration", exc)
                    else:
                        record_exception("observation", "world_progression", exc)
                    break
                except BaseException as exc:
                    record_exception("observation", "world_progression", exc)
                    break
            else:
                faults.add(
                    "observation",
                    "progression_limit",
                    f"progression exceeded {MAX_PROGRESSION_STEPS} bounded steps",
                )
        except BaseException as exc:
            record_exception("setup", "cycle_setup", exc)
    finally:
        # 1. Close bridge admission without waiting.
        if bridge is None:
            cleanup_steps.append(
                _cleanup_result(CleanupStage.CLOSE_BRIDGE_ADMISSION, False, True)
            )
        else:
            try:
                _required_operation(operations, "close_bridge_admission")(bridge)
            except BaseException as exc:
                record_exception("bridge", "close_admission", exc)
                cleanup_steps.append(
                    _cleanup_result(
                        CleanupStage.CLOSE_BRIDGE_ADMISSION, True, False, exc
                    )
                )
            else:
                cleanup_steps.append(
                    _cleanup_result(
                        CleanupStage.CLOSE_BRIDGE_ADMISSION, True, True
                    )
                )

        # Admission is now closed.  Sampling observation elapsed time cannot
        # permit another callback to cross the bridge gate, even if the
        # injected clock is hostile or slow.
        if deadline is not None:
            try:
                elapsed_seconds = deadline.elapsed()
            except BaseException as exc:
                record_exception("clock", "elapsed", exc)

        # 2. Boundedly drain calls admitted before the gate closed.
        if bridge is None:
            cleanup_steps.append(
                _cleanup_result(CleanupStage.DRAIN_BRIDGE, False, True)
            )
        else:
            try:
                _required_operation(operations, "drain_bridge")(
                    bridge, config.shutdown_timeout_seconds
                )
            except BaseException as exc:
                bridge_drain_needs_retry = True
                record_exception("bridge_drain", "drain", exc)
                cleanup_steps.append(
                    _cleanup_result(CleanupStage.DRAIN_BRIDGE, True, False, exc)
                )
            else:
                cleanup_steps.append(
                    _cleanup_result(CleanupStage.DRAIN_BRIDGE, True, True)
                )

        # Recover bounded ownership evidence only after bridge admission is
        # closed and the first drain attempt has completed.  Attribute lookup
        # itself is guarded so a hostile descriptor cannot skip cleanup.
        if ego is None and context is not None:
            try:
                recover = getattr(operations, "recover_owned_ego", None)
                if callable(recover):
                    ego = recover(context)
                    if ego is not None:
                        ego_id = _required_operation(operations, "ego_actor_id")(ego)
                        owned_actor_ids = _merge_actor_ids(
                            owned_actor_ids, (ego_id,)
                        )
            except BaseException as exc:
                record_exception("ego", "ownership_recovery", exc)

        if suite is not None:
            try:
                sensor_ids = _required_operation(
                    operations, "sensor_actor_ids"
                )(suite)
                owned_actor_ids = _merge_actor_ids(owned_actor_ids, sensor_ids)
            except BaseException as exc:
                record_exception("suite", "sensor_actor_id_capture", exc)

        # 3. SensorSuite owns callback shutdown, tick removal, and sensors.
        if suite is None:
            cleanup_steps.append(
                _cleanup_result(CleanupStage.DESTROY_SENSOR_SUITE, False, True)
            )
        else:
            try:
                _required_operation(operations, "destroy_suite")(suite)
            except BaseException as exc:
                record_exception("suite_cleanup", "destroy", exc)
                cleanup_steps.append(
                    _cleanup_result(
                        CleanupStage.DESTROY_SENSOR_SUITE, True, False, exc
                    )
                )
            else:
                cleanup_steps.append(
                    _cleanup_result(CleanupStage.DESTROY_SENSOR_SUITE, True, True)
                )

        # A timed-out drain stays retryable.  Once listeners are stopped there
        # can be no new callback admission, so make one more bounded attempt
        # while preserving the first timeout as the stage's failed verdict.
        if bridge is not None and bridge_drain_needs_retry:
            try:
                _required_operation(operations, "drain_bridge")(
                    bridge, config.shutdown_timeout_seconds
                )
            except BaseException as exc:
                record_exception("bridge_drain_retry", "drain", exc)

        # 4. Stop (and boundedly join) the one perception worker.
        if runtime is None:
            cleanup_steps.append(
                _cleanup_result(CleanupStage.STOP_PERCEPTION_RUNTIME, False, True)
            )
        else:
            try:
                _required_operation(operations, "stop_runtime")(runtime)
            except BaseException as exc:
                runtime_stop_needs_retry = True
                record_exception("runtime_cleanup", "stop", exc)
                cleanup_steps.append(
                    _cleanup_result(
                        CleanupStage.STOP_PERCEPTION_RUNTIME, True, False, exc
                    )
                )
            else:
                cleanup_steps.append(
                    _cleanup_result(
                        CleanupStage.STOP_PERCEPTION_RUNTIME, True, True
                    )
                )

        # 5. Only now may the ego be destroyed: sensors are already gone.
        if ego is None:
            cleanup_steps.append(_cleanup_result(CleanupStage.DESTROY_EGO, False, True))
        else:
            try:
                _required_operation(operations, "destroy_ego")(ego)
            except BaseException as exc:
                record_exception("ego_cleanup", "destroy", exc)
                cleanup_steps.append(
                    _cleanup_result(CleanupStage.DESTROY_EGO, True, False, exc)
                )
            else:
                cleanup_steps.append(
                    _cleanup_result(CleanupStage.DESTROY_EGO, True, True)
                )

        # 6. Verify every copied owned actor id against fresh world snapshots.
        if context is None or not owned_actor_ids:
            cleanup_steps.append(
                _cleanup_result(
                    CleanupStage.VERIFY_ACTOR_DISAPPEARANCE, False, False,
                    "owned actor ids or world context unavailable",
                )
            )
        else:
            try:
                disappearance = _normalise_disappearance(
                    _required_operation(operations, "verify_disappearance")(
                        context,
                        owned_actor_ids,
                        config.shutdown_timeout_seconds,
                        monotonic,
                    )
                )
                disappearance_verified = disappearance.verified
                remaining_actor_ids = disappearance.remaining_actor_ids
                if not disappearance_verified or remaining_actor_ids:
                    raise RuntimeError(
                        "owned actor disappearance was not confirmed; "
                        f"remaining={remaining_actor_ids}"
                    )
            except BaseException as exc:
                record_exception("actor_disappearance", "verification", exc)
                cleanup_steps.append(
                    _cleanup_result(
                        CleanupStage.VERIFY_ACTOR_DISAPPEARANCE,
                        True,
                        False,
                        exc,
                    )
                )
            else:
                cleanup_steps.append(
                    _cleanup_result(
                        CleanupStage.VERIFY_ACTOR_DISAPPEARANCE, True, True
                    )
                )

        # A runtime stop timeout also remains retryable.  Ego and sensor
        # ownership cleanup is already complete, so one final bounded stop can
        # release a worker which finished after the original timeout without
        # erasing that timeout from the cleanup verdict.
        if runtime is not None and runtime_stop_needs_retry:
            try:
                _required_operation(operations, "stop_runtime")(runtime)
            except BaseException as exc:
                record_exception("runtime_cleanup_retry", "stop", exc)

        # Copy final scalar evidence only after both retry opportunities.  No
        # snapshot, actor, callback, exception, or traceback is retained.
        if bridge is not None:
            try:
                bridge_metrics = _copy_bridge_metrics(bridge)
                bridge_state = _state_name(bridge)
                accepted_count = bridge_metrics.submissions_admitted
                _copy_component_fault(faults, "bridge", bridge)
            except BaseException as exc:
                record_exception("reporting", "final_bridge_metrics", exc)
        if runtime is not None:
            try:
                runtime_metrics = _copy_runtime_metrics(runtime)
                runtime_state = _state_name(runtime)
                _copy_component_fault(faults, "runtime", runtime)
            except BaseException as exc:
                record_exception("reporting", "final_runtime_metrics", exc)
        if bridge_metrics is not None and runtime_metrics is not None:
            final_sample = ObservationSample(
                accepted_camera_snapshots=bridge_metrics.submissions_admitted,
                runtime_submissions_accepted=(
                    bridge_metrics.runtime_submissions_accepted
                ),
                runtime_inputs_processed=runtime_metrics.inputs_processed,
                source_stamp=bridge_metrics.last_submitted_source_stamp,
            )
            if last_sample_count != final_sample.accepted_camera_snapshots:
                samples.add(final_sample)
                last_sample_count = final_sample.accepted_camera_snapshots

        # 7. Clear callback, actor, world, and client references.  Runtime is
        # retained only long enough for the authoritative worker check below.
        try:
            release = getattr(operations, "release_references", None)
            if release is not None:
                if not callable(release):
                    raise TypeError("operations.release_references must be callable")
                release(context, suite, bridge, ego)
        except BaseException as exc:
            record_exception("references", "release", exc)
            cleanup_steps.append(
                _cleanup_result(CleanupStage.RELEASE_REFERENCES, True, False, exc)
            )
        else:
            cleanup_steps.append(
                _cleanup_result(CleanupStage.RELEASE_REFERENCES, True, True)
            )
        context = None
        suite = None
        ego = None

        # 8. A stop timeout never suppresses this final truth check.
        if runtime is None:
            cleanup_steps.append(
                _cleanup_result(CleanupStage.VERIFY_RUNTIME_WORKER, False, True)
            )
        else:
            try:
                runtime_worker_alive_after_cleanup = _required_operation(
                    operations, "worker_alive"
                )(runtime)
                if type(runtime_worker_alive_after_cleanup) is not bool:
                    raise TypeError("worker_alive must return bool")
                if runtime_worker_alive_after_cleanup:
                    raise RuntimeError("perception runtime worker remains alive")
            except BaseException as exc:
                record_exception("runtime_worker", "final_verification", exc)
                cleanup_steps.append(
                    _cleanup_result(
                        CleanupStage.VERIFY_RUNTIME_WORKER, True, False, exc
                    )
                )
            else:
                cleanup_steps.append(
                    _cleanup_result(CleanupStage.VERIFY_RUNTIME_WORKER, True, True)
                )
        runtime = None
        bridge = None

    if deferred is not None:
        exc, traceback = deferred
        deferred = None
        raise exc.with_traceback(traceback)

    cleanup_complete = (
        tuple(step.stage for step in cleanup_steps) == CLEANUP_ORDER
        and all(step.succeeded for step in cleanup_steps)
    )
    success = (
        observation_completed
        and not faults.values
        and cleanup_complete
        and disappearance_verified
        and not remaining_actor_ids
        and runtime_worker_alive_after_cleanup is False
        and bridge_metrics is not None
        and runtime_metrics is not None
        and _terminal_metrics_coherent(bridge_metrics, runtime_metrics)
    )
    return PerceptionCycleReport(
        cycle_number=cycle_number,
        success=success,
        observation_completed=observation_completed,
        observation_mode=config.observation_mode,
        accepted_camera_snapshots=accepted_count,
        elapsed_seconds=elapsed_seconds,
        samples=samples.values,
        omitted_sample_count=samples.omitted,
        bridge_metrics=bridge_metrics,
        runtime_metrics=runtime_metrics,
        bridge_state=bridge_state,
        runtime_state=runtime_state,
        faults=faults.values,
        omitted_fault_count=faults.omitted,
        owned_actor_ids=owned_actor_ids,
        remaining_actor_ids=remaining_actor_ids,
        disappearance_verified=disappearance_verified,
        cleanup_steps=tuple(cleanup_steps),
        runtime_worker_alive_after_cleanup=runtime_worker_alive_after_cleanup,
    )


def run_perception_cycles(
    config: PerceptionIntegrationConfig,
    operations: object,
    *,
    monotonic: Callable[[], float] = time.monotonic,
) -> PerceptionIntegrationReport:
    """Run bounded fresh cycles, failing fast after one fully-cleaned failure."""
    if not isinstance(config, PerceptionIntegrationConfig):
        raise TypeError("config must be PerceptionIntegrationConfig")
    retained = _BoundedRetention(FIRST_RETAINED_CYCLES, FINAL_RETAINED_CYCLES)
    attempted = 0
    passed = 0
    failed_cycle = None
    for cycle_number in range(1, config.cycles + 1):
        report = run_perception_cycle(
            config, cycle_number, operations, monotonic=monotonic
        )
        retained.add(report)
        attempted += 1
        if report.success:
            passed += 1
            continue
        failed_cycle = cycle_number
        break
    success = attempted == config.cycles and passed == attempted
    return PerceptionIntegrationReport(
        config=config,
        success=success,
        requested_cycle_count=config.cycles,
        attempted_cycle_count=attempted,
        passed_cycle_count=passed,
        failed_cycle_number=failed_cycle,
        retained_cycle_reports=retained.values,
        omitted_cycle_report_count=retained.omitted,
    )


@dataclass(slots=True)
class _DefaultCycleContext:
    client: object
    world: object
    policy: object
    blueprint: object
    role_name_required: bool
    retained_actor: object | None = None
    owned_ego: object | None = None


class _DefaultLiveOperations:
    """Lazy adapter over committed live APIs, constructed only after opt-in."""

    __slots__ = (
        "_bridge_module",
        "_environ",
        "_harness",
        "_monotonic",
        "_runtime_module",
        "_suite_module",
    )

    def __init__(
        self,
        harness_module: object,
        runtime_module: object,
        bridge_module: object,
        suite_module: object,
        environ: Mapping[str, str],
        monotonic: Callable[[], float],
    ) -> None:
        self._harness = harness_module
        self._runtime_module = runtime_module
        self._bridge_module = bridge_module
        self._suite_module = suite_module
        self._environ = environ
        self._monotonic = monotonic

    def _sensor_config(self, config: PerceptionIntegrationConfig):
        return self._harness.IntegrationConfig(
            host=config.host,
            port=config.port,
            timeout_seconds=config.observation_timeout_seconds,
            cycles=config.cycles,
            tick_owner=config.tick_owner,
            cycles_explicit=True,
        )

    def prepare_cycle(self, config: PerceptionIntegrationConfig):
        dependencies = self._harness.load_live_dependencies(self._environ)
        client = self._harness.connect_client(
            dependencies.carla, self._sensor_config(config)
        )
        self._harness.verify_versions(client)
        world = self._harness.acquire_world(client)
        mode = self._harness.inspect_world_mode(world)
        policy = self._harness.require_progression_policy(mode, config.tick_owner)
        blueprint = dependencies.pick_vehicle_blueprint(
            world.get_blueprint_library()
        )
        role_name_required = self._harness.configure_ego_blueprint(blueprint)
        return _DefaultCycleContext(
            client=client,
            world=world,
            policy=policy,
            blueprint=blueprint,
            role_name_required=role_name_required,
        )

    def spawn_ego(self, context: _DefaultCycleContext, config):
        deadline = self._harness.Deadline(
            config.observation_timeout_seconds, self._monotonic
        )

        def retain(actor: object) -> None:
            context.retained_actor = actor

        owned = self._harness.spawn_owned_ego(
            context.world,
            context.blueprint,
            deadline,
            role_name_required=context.role_name_required,
            ownership_sink=retain,
        )
        # Ownership must be durable before any post-spawn validation can
        # raise.  The committed helper uses this field to preserve an
        # asynchronous BaseException which landed after actor creation.
        context.owned_ego = owned
        pending = getattr(owned, "pending_base_exception", None)
        if pending is not None:
            if not isinstance(pending, BaseException):
                raise TypeError("pending_base_exception must be BaseException or None")
            raise pending
        failure = self._harness.verify_ego_identity(owned)
        if failure is not None:
            raise RuntimeError(f"{failure.phase}: {failure.message}")
        return owned

    def recover_owned_ego(self, context: _DefaultCycleContext):
        if context.owned_ego is not None:
            return context.owned_ego
        if context.retained_actor is None:
            return None
        context.owned_ego = self._harness.capture_owned_ego(
            context.retained_actor, context.role_name_required
        )
        return context.owned_ego

    @staticmethod
    def ego_actor_id(owned) -> int:
        actor_id = owned.actor_id
        if type(actor_id) is not int or actor_id <= 0:
            raise ValueError("owned ego id must be a positive built-in integer")
        return actor_id

    def create_runtime(self, config: PerceptionIntegrationConfig):
        runtime_config = self._runtime_module.PerceptionRuntimeConfig(
            target_hz=config.target_hz,
            max_pending_age_seconds=config.max_pending_age_seconds,
            shutdown_timeout_seconds=config.shutdown_timeout_seconds,
        )
        return self._runtime_module.PerceptionRuntime(
            runtime_config, clock=self._monotonic
        )

    def create_bridge(self, runtime, _config, monotonic):
        return self._bridge_module.PerceptionLiveBridge(
            runtime, clock=monotonic
        )

    def create_suite(self, bridge, _config, monotonic):
        return self._suite_module.SensorSuite(
            monotonic_clock=monotonic,
            camera_snapshot_consumer=bridge,
        )

    def attach_suite(self, suite, context, ego, config) -> None:
        deadline = Deadline(config.observation_timeout_seconds, self._monotonic)
        report = suite.attach(
            context.world,
            ego.actor,
            pre_operation_guard=deadline.raise_if_expired,
        )
        if report.all_attached is not True:
            raise RuntimeError("not all configured sensors attached")
        if report.tick_registered is not True:
            raise RuntimeError("world tick callback was not registered")

    @staticmethod
    def start_runtime(runtime) -> None:
        runtime.start()

    def progress_once(self, context, deadline) -> object:
        return self._harness.progress_world_once(
            context.world, context.policy, deadline
        )

    @staticmethod
    def sensor_actor_ids(suite) -> tuple[int, ...]:
        topology = suite.topology_snapshot()
        return tuple(identity.actor_id for identity in topology.identities)

    @staticmethod
    def perception_input_usable(runtime) -> bool:
        """Reflect the accepted Mission 5 RGB status gate without importing it."""
        result = runtime.latest_result
        if result is None:
            return False
        observation = result.observation
        state = observation.state
        value = state.value
        if type(value) is not str:
            raise TypeError("lane observation state value must be text")
        return value in ("not_detected", "partial", "detected")

    @staticmethod
    def close_bridge_admission(bridge) -> None:
        bridge.close_admission()

    @staticmethod
    def drain_bridge(bridge, timeout_seconds: float) -> None:
        bridge.drain(timeout_seconds)

    @staticmethod
    def destroy_suite(suite) -> None:
        report = suite.destroy()
        errors = list(report.errors)
        results = tuple(report.results)
        if report.tick_unregistered is not True:
            errors.append("world tick callback was not unregistered")
        if tuple(result.name for result in results) != EXPECTED_SENSOR_NAMES:
            errors.append("sensor cleanup results were incomplete or out of order")
        for result in results:
            errors.extend(result.errors)
            if result.stopped is not True or result.destroyed is not True:
                errors.append(f"{result.name} cleanup incomplete")
        if errors:
            raise RuntimeError("; ".join(_bounded_text(value, 96) for value in errors))

    @staticmethod
    def stop_runtime(runtime) -> None:
        runtime.stop()

    def destroy_ego(self, ego) -> None:
        self._harness.destroy_owned_ego(ego)

    def verify_disappearance(
        self, context, actor_ids, timeout_seconds, monotonic
    ) -> DisappearanceEvidence:
        result = self._harness.verify_owned_actors_disappearance(
            context.world,
            actor_ids,
            context.policy,
            self._harness.Deadline(timeout_seconds, monotonic),
        )
        return DisappearanceEvidence(
            verified=(result.verification.value == "CONFIRMED"),
            remaining_actor_ids=tuple(result.remaining_actor_ids),
        )

    @staticmethod
    def release_references(context, _suite, _bridge, _ego) -> None:
        if context is not None:
            context.retained_actor = None
            context.owned_ego = None
            context.blueprint = None
            context.world = None
            context.client = None

    @staticmethod
    def worker_alive(runtime) -> bool:
        return runtime.worker_alive


def _load_default_operations(
    environ: Mapping[str, str],
    importer: Callable[[str], object],
    monotonic: Callable[[], float],
) -> _DefaultLiveOperations:
    """Load every project/live dependency only after the exact opt-in gate."""
    if not integration_enabled(environ):
        raise unittest.SkipTest(DISABLED_MESSAGE)
    harness_module = importer("tests.integration_sensor_suite")
    runtime_module = importer("perception.runtime")
    bridge_module = importer("perception.live_bridge")
    suite_module = importer("sensors.suite")
    return _DefaultLiveOperations(
        harness_module,
        runtime_module,
        bridge_module,
        suite_module,
        environ,
        monotonic,
    )


def run_live_perception_integration(
    environ: Mapping[str, str] | None = None,
    *,
    operations: object | None = None,
    importer: Callable[[str], object] = importlib.import_module,
    monotonic: Callable[[], float] = time.monotonic,
) -> PerceptionIntegrationReport:
    """Exact-gated live entry; injected operations keep offline tests fake-only."""
    config = require_integration_opt_in(environ)
    source = _environment(environ)
    if operations is None:
        operations = _load_default_operations(source, importer, monotonic)
    return run_perception_cycles(config, operations, monotonic=monotonic)


class PerceptionRuntimeLiveTest(unittest.TestCase):
    """Explicit Mission 6B live entry; this file is not normal test discovery."""

    def test_bounded_live_perception_observation(self) -> None:
        report = run_live_perception_integration()
        if report.success:
            return
        failed = report.failed_cycle_number
        detail = "no retained failed cycle"
        for cycle in report.retained_cycle_reports:
            if cycle.cycle_number == failed:
                detail = (
                    f"faults={cycle.faults}; remaining={cycle.remaining_actor_ids}; "
                    f"cleanup={cycle.cleanup_steps}"
                )
                break
        self.fail(
            _bounded_text(
                f"Mission 6B live perception failed at cycle {failed}: {detail}",
                2_048,
            )
        )


if __name__ == "__main__":
    unittest.main()
