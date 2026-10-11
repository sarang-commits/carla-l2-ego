"""Mission 4G stdlib-only harness foundation and exact-opt-in live path.

Ordinary ``test_*.py`` discovery does not execute this module as a test file.
Importing it is deliberately inert: no CARLA, project, display, network, or
filesystem activity occurs. The live path runs only behind the exact opt-in
gate, imports live dependencies only after that gate, and repeats the complete
Mission 4G.3 SensorSuite lifecycle only through the Mission 4G.5 outer runner.
"""

from __future__ import annotations

import importlib
import math
import ntpath
import os
import re
import threading
import time
import unittest
import unicodedata
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from fractions import Fraction
from numbers import Real

CARLA_INTEGRATION = "CARLA_INTEGRATION"
CARLA_HOST = "CARLA_HOST"
CARLA_PORT = "CARLA_PORT"
CARLA_TIMEOUT_SECONDS = "CARLA_TIMEOUT_SECONDS"
CARLA_INTEGRATION_CYCLES = "CARLA_INTEGRATION_CYCLES"
CARLA_INTEGRATION_SOAK_SECONDS = "CARLA_INTEGRATION_SOAK_SECONDS"
CARLA_INTEGRATION_TICK_OWNER = "CARLA_INTEGRATION_TICK_OWNER"
CARLA_INTEGRATION_MEMORY_STABILITY = "CARLA_INTEGRATION_MEMORY_STABILITY"
CARLA_MEMORY_WORKING_SET_MAX_PEAK_GROWTH_BYTES = (
    "CARLA_MEMORY_WORKING_SET_MAX_PEAK_GROWTH_BYTES"
)
CARLA_MEMORY_WORKING_SET_MAX_FINAL_GROWTH_BYTES = (
    "CARLA_MEMORY_WORKING_SET_MAX_FINAL_GROWTH_BYTES"
)
CARLA_MEMORY_WORKING_SET_MAX_FINAL_GROWTH_PERCENT = (
    "CARLA_MEMORY_WORKING_SET_MAX_FINAL_GROWTH_PERCENT"
)
CARLA_MEMORY_PRIVATE_COMMIT_MAX_PEAK_GROWTH_BYTES = (
    "CARLA_MEMORY_PRIVATE_COMMIT_MAX_PEAK_GROWTH_BYTES"
)
CARLA_MEMORY_PRIVATE_COMMIT_MAX_FINAL_GROWTH_BYTES = (
    "CARLA_MEMORY_PRIVATE_COMMIT_MAX_FINAL_GROWTH_BYTES"
)
CARLA_MEMORY_PRIVATE_COMMIT_MAX_FINAL_GROWTH_PERCENT = (
    "CARLA_MEMORY_PRIVATE_COMMIT_MAX_FINAL_GROWTH_PERCENT"
)
CARLA_MEMORY_WORKING_SET_MAX_SLOPE_BYTES_PER_MINUTE = (
    "CARLA_MEMORY_WORKING_SET_MAX_SLOPE_BYTES_PER_MINUTE"
)
CARLA_MEMORY_PRIVATE_COMMIT_MAX_SLOPE_BYTES_PER_MINUTE = (
    "CARLA_MEMORY_PRIVATE_COMMIT_MAX_SLOPE_BYTES_PER_MINUTE"
)
CARLA_SERVER_PID = "CARLA_SERVER_PID"

DISABLED_MESSAGE = "live CARLA integration disabled; set CARLA_INTEGRATION=1"

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 2000
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_CYCLES = 1
DEFAULT_SOAK_SECONDS = 0.0
DEFAULT_TICK_OWNER = False

EXPECTED_SENSOR_NAMES = (
    "gnss",
    "imu",
    "collision",
    "lane_invasion",
    "rgb_camera",
)

MAX_PRIMARY_PHASE_LENGTH = 64
MAX_PRIMARY_MESSAGE_LENGTH = 256
MAX_CLEANUP_ERROR_LENGTH = 256
MAX_DISPLAYED_CLEANUP_ERRORS = 8
MAX_COMPLETE_DIAGNOSTIC_LENGTH = 2048
FIRST_RETAINED_CYCLE_SUMMARIES = 8
FINAL_RETAINED_CYCLE_SUMMARIES = 24
MAX_RETAINED_CYCLE_SUMMARIES = (
    FIRST_RETAINED_CYCLE_SUMMARIES + FINAL_RETAINED_CYCLE_SUMMARIES
)
MAX_RETAINED_SUPPLEMENTAL_CLEANUP_ERRORS = 8


class ExecutionMode(Enum):
    CYCLE_COUNT = "CYCLE_COUNT"
    TIMED_SOAK = "TIMED_SOAK"


class TerminationReason(Enum):
    CYCLE_LIMIT_REACHED = "CYCLE_LIMIT_REACHED"
    SOAK_DURATION_REACHED = "SOAK_DURATION_REACHED"
    FUNCTIONAL_FAILURE = "FUNCTIONAL_FAILURE"
    CLEANUP_FAILURE = "CLEANUP_FAILURE"
    DISAPPEARANCE_FAILURE = "DISAPPEARANCE_FAILURE"
    MEMORY_PROCESS_DISAPPEARED = "MEMORY_PROCESS_DISAPPEARED"
    MEMORY_SAMPLING_FAILURE = "MEMORY_SAMPLING_FAILURE"
    MEMORY_STABILITY_FAILURE = "MEMORY_STABILITY_FAILURE"


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def integration_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return True only for the exact opt-in string ``"1"``."""
    return _environment(environ).get(CARLA_INTEGRATION) == "1"


@dataclass(frozen=True, slots=True)
class IntegrationConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    cycles: int = DEFAULT_CYCLES
    soak_seconds: float = DEFAULT_SOAK_SECONDS
    tick_owner: bool = DEFAULT_TICK_OWNER
    cycles_explicit: bool = False


def _require_text(value, variable: str, accepted: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{variable} must be {accepted}")
    return value


def _parse_base10_integer(value, variable: str, lower: int, upper: int) -> int:
    accepted = f"a base-10 integer in {lower}..{upper}"
    text = _require_text(value, variable, accepted)
    if not text or not text.isascii() or not text.isdecimal():
        raise ValueError(f"{variable} must be {accepted}")
    parsed = int(text, 10)
    if not lower <= parsed <= upper:
        raise ValueError(f"{variable} must be {accepted}")
    return parsed


def _parse_finite_float(value, variable: str, lower: float, upper: float) -> float:
    accepted = f"a finite number in {lower:.1f}..{upper:.1f} inclusive"
    text = _require_text(value, variable, accepted)
    try:
        parsed = float(text)
    except ValueError:
        raise ValueError(f"{variable} must be {accepted}") from None
    if not math.isfinite(parsed) or not lower <= parsed <= upper:
        raise ValueError(f"{variable} must be {accepted}")
    return parsed


def parse_integration_config(
    environ: Mapping[str, str] | None = None,
) -> IntegrationConfig:
    """Parse and validate the complete offline environment contract."""
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

    port = _parse_base10_integer(
        source.get(CARLA_PORT, str(DEFAULT_PORT)), CARLA_PORT, 1, 65535
    )
    timeout_seconds = _parse_finite_float(
        source.get(CARLA_TIMEOUT_SECONDS, str(DEFAULT_TIMEOUT_SECONDS)),
        CARLA_TIMEOUT_SECONDS,
        2.0,
        120.0,
    )
    cycles = _parse_base10_integer(
        source.get(CARLA_INTEGRATION_CYCLES, str(DEFAULT_CYCLES)),
        CARLA_INTEGRATION_CYCLES,
        1,
        20,
    )
    cycles_explicit = CARLA_INTEGRATION_CYCLES in source

    soak_accepted = "numeric zero or a finite number in 120.0..3600.0 inclusive"
    soak_text = _require_text(
        source.get(CARLA_INTEGRATION_SOAK_SECONDS, str(DEFAULT_SOAK_SECONDS)),
        CARLA_INTEGRATION_SOAK_SECONDS,
        soak_accepted,
    )
    try:
        soak_seconds = float(soak_text)
    except ValueError:
        raise ValueError(
            f"{CARLA_INTEGRATION_SOAK_SECONDS} must be {soak_accepted}"
        ) from None
    if not math.isfinite(soak_seconds) or not (
        soak_seconds == 0.0 or 120.0 <= soak_seconds <= 3600.0
    ):
        raise ValueError(
            f"{CARLA_INTEGRATION_SOAK_SECONDS} must be {soak_accepted}"
        )

    tick_text = source.get(CARLA_INTEGRATION_TICK_OWNER)
    if tick_text is None or tick_text == "0":
        tick_owner = False
    elif tick_text == "1":
        tick_owner = True
    else:
        raise ValueError(f"{CARLA_INTEGRATION_TICK_OWNER} must be one of: 0, 1")

    return IntegrationConfig(
        host=host,
        port=port,
        timeout_seconds=timeout_seconds,
        cycles=cycles,
        soak_seconds=soak_seconds,
        tick_owner=tick_owner,
        cycles_explicit=cycles_explicit,
    )


def require_integration_opt_in(
    environ: Mapping[str, str] | None = None,
) -> IntegrationConfig:
    """Gate first, then parse; disabled runs ignore every secondary value."""
    if not integration_enabled(environ):
        raise unittest.SkipTest(DISABLED_MESSAGE)
    return parse_integration_config(environ)


def select_execution_mode(config: IntegrationConfig) -> ExecutionMode:
    """Select one unambiguous outer-runner mode without performing live work."""
    if not isinstance(config, IntegrationConfig):
        raise TypeError("config must be IntegrationConfig")
    if config.soak_seconds > 0.0:
        if config.cycles_explicit:
            raise ValueError(
                f"positive {CARLA_INTEGRATION_SOAK_SECONDS} cannot be combined "
                f"with explicit {CARLA_INTEGRATION_CYCLES}"
            )
        return ExecutionMode.TIMED_SOAK
    return ExecutionMode.CYCLE_COUNT


class PhaseTimeout(TimeoutError):
    def __init__(self, phase: str, timeout_seconds: float) -> None:
        self.phase = phase
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"phase {phase!r} exceeded {float(timeout_seconds):.3f} seconds"
        )


class Deadline:
    """Fixed deadline for single-threaded integration-harness use.

    The host-monotonic clock is injectable for deterministic offline tests.
    """

    def __init__(
        self,
        timeout_seconds: float,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, Real):
            raise ValueError("timeout_seconds must be finite and strictly positive")
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_seconds must be finite and strictly positive")
        if not callable(monotonic):
            raise TypeError("monotonic must be callable")
        self.timeout_seconds = timeout
        self._monotonic = monotonic
        started_at = self._read_clock()
        expires_at = started_at + timeout
        if not math.isfinite(expires_at) or expires_at <= started_at:
            raise ValueError("deadline expiry must be finite and advance beyond start")
        self.start = started_at
        self.expiry = expires_at
        self._last_effective_time = started_at
        self._has_expired = False

    def _read_clock(self) -> float:
        raw_value = self._monotonic()
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise ValueError("monotonic clock must return a finite value")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("monotonic clock must return a finite value")
        return value

    def _effective_time(self) -> float:
        current = self._read_clock()
        if current < self._last_effective_time:
            return self._last_effective_time
        self._last_effective_time = current
        return current

    def _sample_expiration(self) -> tuple[float, bool]:
        current = self._effective_time()
        if current >= self.expiry:
            self._has_expired = True
        return current, self._has_expired

    def expired(self) -> bool:
        _current, has_expired = self._sample_expiration()
        return has_expired

    def remaining(self) -> float:
        current, has_expired = self._sample_expiration()
        if has_expired:
            return 0.0
        return max(0.0, self.expiry - current)

    def raise_if_expired(self, phase: str) -> None:
        if self.expired():
            raise PhaseTimeout(phase, self.timeout_seconds)


class WorldMode(Enum):
    ASYNCHRONOUS = "ASYNCHRONOUS"
    SYNCHRONOUS = "SYNCHRONOUS"


class ProgressionPolicy(Enum):
    PASSIVE_WAIT = "PASSIVE_WAIT"
    OWNED_TICK = "OWNED_TICK"
    SKIP = "SKIP"


def decide_world_progression(
    synchronous_mode: bool | WorldMode, tick_owner: bool
) -> ProgressionPolicy:
    if isinstance(synchronous_mode, WorldMode):
        synchronous = synchronous_mode is WorldMode.SYNCHRONOUS
    elif isinstance(synchronous_mode, bool):
        synchronous = synchronous_mode
    else:
        raise TypeError("synchronous_mode must be bool or WorldMode")
    if not isinstance(tick_owner, bool):
        raise TypeError("tick_owner must be bool")
    if not synchronous:
        return ProgressionPolicy.PASSIVE_WAIT
    if tick_owner:
        return ProgressionPolicy.OWNED_TICK
    return ProgressionPolicy.SKIP


@dataclass(frozen=True, slots=True)
class ValidationFailure:
    phase: str
    message: str
    cycle: int | None = None


class CleanupVerification(Enum):
    CONFIRMED = "CONFIRMED"
    FAILED_ACTORS_REMAIN = "FAILED_ACTORS_REMAIN"
    UNVERIFIED_DISCONNECTED = "UNVERIFIED_DISCONNECTED"


@dataclass(frozen=True, slots=True)
class CleanupOutcome:
    primary_failure: ValidationFailure | None
    cleanup_errors: tuple[str, ...]
    verification: CleanupVerification


def _bounded_text(value, limit: int) -> str:
    try:
        raw_text = str(value)
    except BaseException as exc:  # diagnostic conversion must never skip cleanup
        raw_text = f"<unprintable: {type(exc).__name__}>"
    without_controls = "".join(
        " " if unicodedata.category(character) == "Cc" else character
        for character in raw_text
    )
    text = " ".join(without_controls.split()) or "<no message>"
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _cleanup_error(step: str, exc: Exception) -> str:
    message = _bounded_text(exc, MAX_CLEANUP_ERROR_LENGTH)
    return _bounded_text(
        f"{step} {type(exc).__name__}: {message}",
        MAX_CLEANUP_ERROR_LENGTH,
    )


def coordinate_cleanup(
    primary_failure: ValidationFailure | None,
    suite_cleanup: Callable[[], object] | None = None,
    ego_cleanup: Callable[[], object] | None = None,
    final_cleanup: Callable[[], CleanupVerification] | None = None,
) -> CleanupOutcome:
    """Attempt suite, ego, and final cleanup in order without short-circuiting."""
    if primary_failure is not None and not isinstance(
        primary_failure, ValidationFailure
    ):
        raise TypeError("primary_failure must be ValidationFailure or None")

    errors: list[str] = []
    verification = CleanupVerification.UNVERIFIED_DISCONNECTED
    for step, cleanup in (
        ("suite", suite_cleanup),
        ("ego", ego_cleanup),
        ("final", final_cleanup),
    ):
        if cleanup is None:
            continue
        try:
            result = cleanup()
            if step == "final":
                if not isinstance(result, CleanupVerification):
                    raise TypeError("must return CleanupVerification")
                verification = result
        except Exception as exc:  # noqa: BLE001 - later cleanup must still run
            errors.append(_cleanup_error(step, exc))
            if step == "final":
                verification = CleanupVerification.UNVERIFIED_DISCONNECTED

    return CleanupOutcome(primary_failure, tuple(errors), verification)


def format_cleanup_outcome(outcome: CleanupOutcome) -> str:
    """Return one bounded, deterministic cleanup diagnostic."""
    failure = outcome.primary_failure
    if failure is None:
        primary = "primary=none"
    else:
        cycle = "" if failure.cycle is None else f"cycle {failure.cycle} "
        primary = (
            f"primary={cycle}"
            f"{_bounded_text(failure.phase, MAX_PRIMARY_PHASE_LENGTH)}: "
            f"{_bounded_text(failure.message, MAX_PRIMARY_MESSAGE_LENGTH)}"
        )

    displayed_errors = outcome.cleanup_errors[:MAX_DISPLAYED_CLEANUP_ERRORS]
    errors = "none" if not displayed_errors else " | ".join(
        _bounded_text(error, MAX_CLEANUP_ERROR_LENGTH)
        for error in displayed_errors
    )
    diagnostic_without_omission = (
        f"{primary}; verification={outcome.verification.value}; "
        f"cleanup_errors={errors}"
    )
    omitted_count = len(outcome.cleanup_errors) - len(displayed_errors)
    if omitted_count:
        omission_marker = f" | (+{omitted_count} more)"
        available = MAX_COMPLETE_DIAGNOSTIC_LENGTH - len(omission_marker)
        return (
            _bounded_text(diagnostic_without_omission, available)
            + omission_marker
        )
    return _bounded_text(
        diagnostic_without_omission,
        MAX_COMPLETE_DIAGNOSTIC_LENGTH,
    )


@dataclass(slots=True)
class CycleState:
    cycle_number: int
    ego_id: int | None = None
    sensor_ids: tuple[int, ...] = ()
    snapshot_reference: object | None = None
    camera_reference: object | None = None
    primary_failure: ValidationFailure | None = None
    cleanup_errors: tuple[str, ...] = ()


def new_cycle_state(cycle_number: int) -> CycleState:
    return CycleState(cycle_number=cycle_number)


@dataclass(frozen=True, slots=True)
class MemorySample:
    elapsed_seconds: float
    current_bytes: int


@dataclass(frozen=True, slots=True)
class MemoryGrowthResult:
    passed: bool
    final_delta_bytes: int
    maximum_delta_bytes: int
    slope_bytes_per_minute: float
    reason: str


def _non_negative_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def evaluate_memory_growth(
    baseline_bytes: int,
    samples: Iterable[MemorySample],
    fixed_growth_allowance_bytes: int,
    slope_allowance_bytes_per_minute: float,
) -> MemoryGrowthResult:
    """Evaluate fixed and OLS-slope memory growth without collecting memory."""
    baseline = _non_negative_integer(baseline_bytes, "baseline_bytes")
    fixed_allowance = _non_negative_integer(
        fixed_growth_allowance_bytes, "fixed_growth_allowance_bytes"
    )
    if isinstance(slope_allowance_bytes_per_minute, bool) or not isinstance(
        slope_allowance_bytes_per_minute, Real
    ):
        raise ValueError(
            "slope_allowance_bytes_per_minute must be finite and non-negative"
        )
    slope_allowance = float(slope_allowance_bytes_per_minute)
    if not math.isfinite(slope_allowance) or slope_allowance < 0.0:
        raise ValueError(
            "slope_allowance_bytes_per_minute must be finite and non-negative"
        )

    sample_values = tuple(samples)
    if len(sample_values) < 2:
        raise ValueError("samples must contain at least two MemorySample values")

    times: list[float] = []
    byte_values: list[int] = []
    previous_time: float | None = None
    for sample in sample_values:
        if not isinstance(sample, MemorySample):
            raise ValueError("samples must contain MemorySample values")
        if isinstance(sample.elapsed_seconds, bool) or not isinstance(
            sample.elapsed_seconds, Real
        ):
            raise ValueError("sample timestamps must be finite and strictly increasing")
        elapsed = float(sample.elapsed_seconds)
        if not math.isfinite(elapsed) or (
            previous_time is not None and elapsed <= previous_time
        ):
            raise ValueError("sample timestamps must be finite and strictly increasing")
        current = _non_negative_integer(sample.current_bytes, "sample current_bytes")
        times.append(elapsed)
        byte_values.append(current)
        previous_time = elapsed

    # Translate before calculating OLS and retain exact binary-float values as
    # Fractions.  This prevents large absolute counters from erasing small
    # growth and prevents valid, extremely close timestamps from underflowing
    # the denominator to zero.
    time_origin = Fraction.from_float(times[0])
    byte_origin = byte_values[0]
    translated_times = tuple(
        Fraction.from_float(value) - time_origin for value in times
    )
    translated_bytes = tuple(value - byte_origin for value in byte_values)
    count = len(translated_times)
    sum_time = sum(translated_times, Fraction())
    sum_bytes = sum(translated_bytes)
    denominator = (
        count * sum(value * value for value in translated_times)
        - sum_time * sum_time
    )
    numerator = (
        count
        * sum(
            elapsed * current
            for elapsed, current in zip(translated_times, translated_bytes)
        )
        - sum_time * sum_bytes
    )
    slope_per_second = numerator / denominator
    slope_per_minute_exact = slope_per_second * 60
    try:
        slope_per_minute = float(slope_per_minute_exact)
    except OverflowError:
        slope_per_minute = (
            math.inf if slope_per_minute_exact > 0 else -math.inf
        )

    final_delta = byte_values[-1] - baseline
    maximum_delta = max(byte_values) - baseline
    fixed_failed = maximum_delta > fixed_allowance
    slope_failed = slope_per_minute_exact > Fraction.from_float(slope_allowance)
    if fixed_failed and slope_failed:
        reason = "fixed growth allowance and slope allowance exceeded"
    elif fixed_failed:
        reason = "fixed growth allowance exceeded"
    elif slope_failed:
        reason = "slope allowance exceeded"
    else:
        reason = "pass"

    return MemoryGrowthResult(
        passed=not fixed_failed and not slope_failed,
        final_delta_bytes=final_delta,
        maximum_delta_bytes=maximum_delta,
        slope_bytes_per_minute=slope_per_minute,
        reason=reason,
    )


# -- Mission 4G.6: bounded memory-stability contract and models --

MEMORY_SAMPLING_INTERVAL_SECONDS = 5.0
# Every external/native sampling request must complete within this bound as
# observed by the controlling thread; a call that never returns is abandoned
# on its daemon worker and classified as a mandatory sampling failure.
MEMORY_NATIVE_CALL_TIMEOUT_SECONDS = 5.0
# The sampling loop itself never blocks longer than one cadence wait plus two
# bounded native calls, so this join bound is generous yet still finite.
MEMORY_MONITOR_JOIN_TIMEOUT_SECONDS = 20.0
FIRST_RETAINED_MEMORY_SAMPLES = 16
FINAL_RETAINED_MEMORY_SAMPLES = 240
MAX_RETAINED_MEMORY_SAMPLES = (
    FIRST_RETAINED_MEMORY_SAMPLES + FINAL_RETAINED_MEMORY_SAMPLES
)
MAX_RETAINED_MEMORY_DIAGNOSTICS = 8
MINIMUM_MEMORY_VERDICT_SAMPLES = 5
MINIMUM_MEMORY_VERDICT_SPAN_SECONDS = 60.0
MAX_MEMORY_BYTE_LIMIT = 2**63 - 1
MAX_MEMORY_PID = 4294967295
MAX_MEMORY_PERCENT_LIMIT = 10000.0
MAX_PROCESS_CREATION_TIME_100NS = 2**64 - 1
MAX_MEMORY_SAMPLE_BYTES = 2**64 - 1
MAX_MEMORY_EXECUTABLE_PATH_CHARACTERS = 32767
EXPECTED_MEMORY_EXECUTABLE_BASENAME = "CarlaUE4-Win64-Shipping.exe"
MEMORY_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

REQUIRED_MEMORY_BYTE_THRESHOLD_VARIABLES = (
    CARLA_MEMORY_WORKING_SET_MAX_PEAK_GROWTH_BYTES,
    CARLA_MEMORY_WORKING_SET_MAX_FINAL_GROWTH_BYTES,
    CARLA_MEMORY_PRIVATE_COMMIT_MAX_PEAK_GROWTH_BYTES,
    CARLA_MEMORY_PRIVATE_COMMIT_MAX_FINAL_GROWTH_BYTES,
)
REQUIRED_MEMORY_PERCENT_THRESHOLD_VARIABLES = (
    CARLA_MEMORY_WORKING_SET_MAX_FINAL_GROWTH_PERCENT,
    CARLA_MEMORY_PRIVATE_COMMIT_MAX_FINAL_GROWTH_PERCENT,
)
OPTIONAL_MEMORY_SLOPE_THRESHOLD_VARIABLES = (
    CARLA_MEMORY_WORKING_SET_MAX_SLOPE_BYTES_PER_MINUTE,
    CARLA_MEMORY_PRIVATE_COMMIT_MAX_SLOPE_BYTES_PER_MINUTE,
)


class MemoryStabilityVerdict(Enum):
    NOT_REQUESTED = "NOT_REQUESTED"
    STABLE = "STABLE"
    UNSTABLE = "UNSTABLE"
    INDETERMINATE = "INDETERMINATE"


class MemoryCounterAvailability(Enum):
    NOT_SAMPLED = "NOT_SAMPLED"
    AVAILABLE = "AVAILABLE"
    UNSUPPORTED = "UNSUPPORTED"
    NO_ACTIVITY = "NO_ACTIVITY"
    ZERO_OR_UNIFIED_MEMORY = "ZERO_OR_UNIFIED_MEMORY"
    UNRELIABLE = "UNRELIABLE"


class ProcessIdentityStatus(Enum):
    NOT_TRACKED = "NOT_TRACKED"
    UNAVAILABLE = "UNAVAILABLE"
    PRESERVED = "PRESERVED"
    EXITED = "EXITED"


class MemoryMonitoredProcessExited(RuntimeError):
    """The monitored process object exited; raised by sampler adapters."""


class MemoryNativeCallTimeout(RuntimeError):
    """A native sampling operation did not complete within its bound."""


def _run_native_call_bounded(
    description: str, operation: Callable[[], object], timeout_seconds: float
):
    """Run one native operation on a disposable daemon worker, bounded.

    The calling thread always regains control within ``timeout_seconds``.  A
    call that never returns cannot be cancelled, so its worker is
    deliberately abandoned as a daemon thread (it never prevents interpreter
    exit), its eventual result is discarded, and MemoryNativeCallTimeout is
    raised; callers must then treat the underlying native handles as unsafe
    to touch.  A completed operation's result or exception is delivered
    unchanged, and the operation is never retried.
    """
    completion = threading.Event()
    outcome: list = []

    def run() -> None:
        try:
            result = operation()
        except BaseException as exc:
            outcome.append((None, exc))
        else:
            outcome.append((result, None))
        completion.set()

    worker = threading.Thread(
        target=run, name="memory-native-call", daemon=True
    )
    worker.start()
    if not completion.wait(timeout_seconds):
        raise MemoryNativeCallTimeout(
            f"{description} did not complete within "
            f"{float(timeout_seconds):.3f} seconds"
        )
    result, error = outcome[0]
    if error is not None:
        raise error
    return result


@dataclass(frozen=True, slots=True)
class ProcessMemorySample:
    """One bounded process-memory observation copied to plain values."""

    elapsed_seconds: float
    working_set_bytes: int
    private_commit_bytes: int
    gpu_dedicated_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ProcessCounterReading:
    """One raw counter reading delivered by a process sampler adapter."""

    working_set_bytes: int
    private_commit_bytes: int
    pagefile_usage_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class GpuEvidenceReading:
    """One advisory GPU reading delivered by a GPU sampler adapter."""

    status: MemoryCounterAvailability
    dedicated_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class MemoryThresholds:
    """Strictly parsed upper limits; equality with a limit always passes."""

    working_set_max_peak_growth_bytes: int
    working_set_max_final_growth_bytes: int
    working_set_max_final_growth_percent: float
    private_commit_max_peak_growth_bytes: int
    private_commit_max_final_growth_bytes: int
    private_commit_max_final_growth_percent: float
    working_set_max_slope_bytes_per_minute: float | None = None
    private_commit_max_slope_bytes_per_minute: float | None = None


@dataclass(frozen=True, slots=True)
class MemoryStabilityConfig:
    """Validated CARLA_INTEGRATION_MEMORY_STABILITY configuration."""

    enabled: bool
    thresholds: MemoryThresholds | None = None
    server_pid: int | None = None


@dataclass(frozen=True, slots=True)
class MemoryMetricReport:
    """Complete evaluated metrics for one memory counter."""

    counter: str
    sample_count: int
    span_seconds: float
    start_bytes: int
    peak_bytes: int
    peak_elapsed_seconds: float
    final_bytes: int
    growth_bytes: int
    peak_growth_bytes: int
    growth_percent: float | None
    peak_growth_percent: float | None
    slope_bytes_per_minute: float | None
    failed_thresholds: tuple[str, ...]
    unevaluated_thresholds: tuple[str, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class MemoryStabilityReport:
    """Frozen CARLA-free memory-stability evidence for one soak run."""

    verdict: MemoryStabilityVerdict
    monitoring_enabled: bool
    identity: ProcessIdentityStatus
    mandatory_sampling_failure: bool
    mandatory_shutdown_failure: bool
    pid: int | None
    process_creation_time_100ns: int | None
    executable_path: str | None
    executable_basename: str | None
    total_sample_count: int
    retained_sample_count: int
    omitted_sample_count: int
    missed_interval_count: int
    samples: tuple[ProcessMemorySample, ...]
    working_set: MemoryMetricReport | None
    private_commit: MemoryMetricReport | None
    thresholds: MemoryThresholds | None
    gpu_status: MemoryCounterAvailability
    gpu_final_dedicated_bytes: int | None
    gpu_peak_dedicated_bytes: int | None
    diagnostics: tuple[str, ...]
    omitted_diagnostic_count: int


NOT_REQUESTED_MEMORY_REPORT = MemoryStabilityReport(
    verdict=MemoryStabilityVerdict.NOT_REQUESTED,
    monitoring_enabled=False,
    identity=ProcessIdentityStatus.NOT_TRACKED,
    mandatory_sampling_failure=False,
    mandatory_shutdown_failure=False,
    pid=None,
    process_creation_time_100ns=None,
    executable_path=None,
    executable_basename=None,
    total_sample_count=0,
    retained_sample_count=0,
    omitted_sample_count=0,
    missed_interval_count=0,
    samples=(),
    working_set=None,
    private_commit=None,
    thresholds=None,
    gpu_status=MemoryCounterAvailability.NOT_SAMPLED,
    gpu_final_dedicated_bytes=None,
    gpu_peak_dedicated_bytes=None,
    diagnostics=(),
    omitted_diagnostic_count=0,
)


def _parse_memory_byte_limit(value, variable: str) -> int:
    return _parse_base10_integer(value, variable, 0, MAX_MEMORY_BYTE_LIMIT)


def _parse_memory_pid(value, variable: str) -> int:
    return _parse_base10_integer(value, variable, 1, MAX_MEMORY_PID)


def _parse_memory_unsigned_float(
    value, variable: str, lower: float, upper: float
) -> float:
    """Parse one strict, unsigned, finite decimal without normalization."""
    accepted = (
        f"an unsigned finite decimal number in {lower}..{upper} inclusive"
    )
    text = _require_text(value, variable, accepted)
    if (
        not text
        or not text.isascii()
        or text != text.strip()
        or text[0] in "+-"
        or "_" in text
    ):
        raise ValueError(f"{variable} must be {accepted}")
    try:
        parsed = float(text)
    except ValueError:
        raise ValueError(f"{variable} must be {accepted}") from None
    if not math.isfinite(parsed) or not lower <= parsed <= upper:
        raise ValueError(f"{variable} must be {accepted}")
    return parsed


def parse_memory_stability_config(
    environ: Mapping[str, str] | None,
    config: IntegrationConfig,
    execution_mode: ExecutionMode,
) -> MemoryStabilityConfig:
    """Validate the complete Mission 4G.6 contract before any live work."""
    if not isinstance(config, IntegrationConfig):
        raise TypeError("config must be IntegrationConfig")
    if not isinstance(execution_mode, ExecutionMode):
        raise TypeError("execution_mode must be ExecutionMode")
    source = _environment(environ)

    gate = source.get(CARLA_INTEGRATION_MEMORY_STABILITY)
    if gate is None or gate == "0":
        return MemoryStabilityConfig(enabled=False)
    if gate != "1":
        raise ValueError(
            f"{CARLA_INTEGRATION_MEMORY_STABILITY} must be one of: 0, 1"
        )

    if execution_mode is not ExecutionMode.TIMED_SOAK:
        raise ValueError(
            f"{CARLA_INTEGRATION_MEMORY_STABILITY}=1 requires timed-soak "
            f"mode: a positive {CARLA_INTEGRATION_SOAK_SECONDS} and no "
            f"explicit {CARLA_INTEGRATION_CYCLES}"
        )
    if not config.soak_seconds > 0.0 or config.cycles_explicit:
        raise ValueError(
            f"{CARLA_INTEGRATION_MEMORY_STABILITY}=1 requires timed-soak "
            f"mode: a positive {CARLA_INTEGRATION_SOAK_SECONDS} and no "
            f"explicit {CARLA_INTEGRATION_CYCLES}"
        )
    # The memory-stability admission contract is strict raw-input: the exact
    # unmodified CARLA_HOST value must match, with no whitespace stripping
    # and no case folding.  Legacy host parsing elsewhere is unchanged.
    raw_host = source.get(CARLA_HOST, DEFAULT_HOST)
    if raw_host not in MEMORY_LOCAL_HOSTS:
        raise ValueError(
            f"{CARLA_INTEGRATION_MEMORY_STABILITY}=1 requires the raw "
            f"{CARLA_HOST} value to be exactly one of: "
            + ", ".join(MEMORY_LOCAL_HOSTS)
        )

    required = (
        REQUIRED_MEMORY_BYTE_THRESHOLD_VARIABLES
        + REQUIRED_MEMORY_PERCENT_THRESHOLD_VARIABLES
    )
    missing = tuple(name for name in required if name not in source)
    if missing:
        raise ValueError(
            "memory stability requires every threshold; missing: "
            + ", ".join(missing)
        )

    byte_limits = {
        name: _parse_memory_byte_limit(source[name], name)
        for name in REQUIRED_MEMORY_BYTE_THRESHOLD_VARIABLES
    }
    percent_limits = {
        name: _parse_memory_unsigned_float(
            source[name], name, 0.0, MAX_MEMORY_PERCENT_LIMIT
        )
        for name in REQUIRED_MEMORY_PERCENT_THRESHOLD_VARIABLES
    }
    slope_limits: dict[str, float | None] = {}
    for name in OPTIONAL_MEMORY_SLOPE_THRESHOLD_VARIABLES:
        if name in source:
            slope_limits[name] = _parse_memory_unsigned_float(
                source[name], name, 0.0, math.inf
            )
        else:
            slope_limits[name] = None

    server_pid = None
    if CARLA_SERVER_PID in source:
        server_pid = _parse_memory_pid(
            source[CARLA_SERVER_PID], CARLA_SERVER_PID
        )

    thresholds = MemoryThresholds(
        working_set_max_peak_growth_bytes=byte_limits[
            CARLA_MEMORY_WORKING_SET_MAX_PEAK_GROWTH_BYTES
        ],
        working_set_max_final_growth_bytes=byte_limits[
            CARLA_MEMORY_WORKING_SET_MAX_FINAL_GROWTH_BYTES
        ],
        working_set_max_final_growth_percent=percent_limits[
            CARLA_MEMORY_WORKING_SET_MAX_FINAL_GROWTH_PERCENT
        ],
        private_commit_max_peak_growth_bytes=byte_limits[
            CARLA_MEMORY_PRIVATE_COMMIT_MAX_PEAK_GROWTH_BYTES
        ],
        private_commit_max_final_growth_bytes=byte_limits[
            CARLA_MEMORY_PRIVATE_COMMIT_MAX_FINAL_GROWTH_BYTES
        ],
        private_commit_max_final_growth_percent=percent_limits[
            CARLA_MEMORY_PRIVATE_COMMIT_MAX_FINAL_GROWTH_PERCENT
        ],
        working_set_max_slope_bytes_per_minute=slope_limits[
            CARLA_MEMORY_WORKING_SET_MAX_SLOPE_BYTES_PER_MINUTE
        ],
        private_commit_max_slope_bytes_per_minute=slope_limits[
            CARLA_MEMORY_PRIVATE_COMMIT_MAX_SLOPE_BYTES_PER_MINUTE
        ],
    )
    return MemoryStabilityConfig(
        enabled=True, thresholds=thresholds, server_pid=server_pid
    )


# -- Missions 4G.2 + 4G.3: gated live ego and SensorSuite validation --

REQUIRED_CARLA_VERSION = "0.9.16"
REQUIRED_CARLA_COMPONENTS = (0, 9, 16)
EGO_ROLE_NAME = "hero"
VERSION_TEXT_LIMIT = 64
EGO_DISAPPEARANCE_TIMEOUT_SECONDS = 5.0
PROGRESSION_STEP_LIMIT_SECONDS = 1.0
SYNCHRONOUS_UNOWNED_MESSAGE = (
    "synchronous world without CARLA_INTEGRATION_TICK_OWNER=1; "
    "set CARLA_INTEGRATION_TICK_OWNER=1 only when this harness owns ticking"
)
LIVE_DEPENDENCY_MESSAGE = "live_dependency_loading failed"
SENSOR_SUITE_DEPENDENCY_MESSAGE = "sensor_suite_dependency_loading failed"
EXPECTED_SENSOR_TYPES = (
    "sensor.other.gnss",
    "sensor.other.imu",
    "sensor.other.collision",
    "sensor.other.lane_invasion",
    "sensor.camera.rgb",
)
EXPECTED_CAMERA_WIDTH = 640
EXPECTED_CAMERA_HEIGHT = 360
EXPECTED_CAMERA_FOV = 90.0
EXPECTED_CAMERA_PIXEL_FORMAT = "RGB8"
EXPECTED_CAMERA_ROW_STRIDE = EXPECTED_CAMERA_WIDTH * 3

_VERSION_IDENTIFIER = r"[0-9A-Za-z]+(?:-[0-9A-Za-z]+)*"
_VERSION_PATTERN = re.compile(
    r"(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)"
    rf"(?P<suffix>[-+]{_VERSION_IDENTIFIER}(?:\.{_VERSION_IDENTIFIER})*)?",
    re.ASCII,
)
_LIVE_RPC_EXCEPTIONS = (RuntimeError, OSError)


class ProgressionRPCError(RuntimeError):
    """A live exception raised specifically by tick or wait_for_tick."""


def _phase_error_message(phase: str, exc: BaseException) -> str:
    return _bounded_text(
        f"{phase} failed: {type(exc).__name__}: "
        f"{_bounded_text(exc, MAX_PRIMARY_MESSAGE_LENGTH)}",
        MAX_PRIMARY_MESSAGE_LENGTH,
    )


@dataclass(frozen=True, slots=True)
class LiveDependencies:
    """Live-only objects loaded strictly after the exact opt-in."""

    carla: object
    pick_vehicle_blueprint: Callable[[object], object]


def _import_live_dependency(
    name: str, importer: Callable[[str], object]
) -> object:
    try:
        return importer(name)
    except Exception as exc:
        raise RuntimeError(_phase_error_message(f"{LIVE_DEPENDENCY_MESSAGE} {name}", exc)) from exc


def load_live_dependencies(
    environ: Mapping[str, str] | None = None,
    importer: Callable[[str], object] = importlib.import_module,
) -> LiveDependencies:
    """Gate first, then import carla and the committed blueprint helper."""
    if not integration_enabled(environ):
        raise unittest.SkipTest(DISABLED_MESSAGE)

    carla_module = _import_live_dependency("carla", importer)
    spawn_module = _import_live_dependency("spawn_ego_vehicle", importer)
    try:
        picker = getattr(spawn_module, "pick_vehicle_blueprint")
        if not callable(picker):
            raise TypeError("pick_vehicle_blueprint must be callable")
    except Exception as exc:
        raise RuntimeError(
            _phase_error_message(
                f"{LIVE_DEPENDENCY_MESSAGE} spawn_ego_vehicle.pick_vehicle_blueprint",
                exc,
            )
        ) from exc
    return LiveDependencies(carla=carla_module, pick_vehicle_blueprint=picker)


@dataclass(frozen=True, slots=True)
class SensorSuiteDependencies:
    """Exact committed SensorSuite API, resolved only after world policy."""

    SensorSuite: Callable[[], object]


def load_sensor_suite_dependencies(
    environ: Mapping[str, str] | None = None,
    importer: Callable[[str], object] = importlib.import_module,
) -> SensorSuiteDependencies:
    """Gate first, then resolve (but never construct) the SensorSuite API."""
    if not integration_enabled(environ):
        raise unittest.SkipTest(DISABLED_MESSAGE)
    module = _import_live_dependency("sensors.suite", importer)
    try:
        constructor = getattr(module, "SensorSuite")
        if not callable(constructor):
            raise TypeError("SensorSuite must be callable")
        for method_name in ("attach", "snapshot", "topology_snapshot", "destroy"):
            if not callable(getattr(constructor, method_name, None)):
                raise TypeError(
                    f"SensorSuite.{method_name} must be a public callable"
                )
    except Exception as exc:
        raise RuntimeError(
            _phase_error_message(SENSOR_SUITE_DEPENDENCY_MESSAGE, exc)
        ) from exc
    return SensorSuiteDependencies(SensorSuite=constructor)


@dataclass(frozen=True, slots=True)
class CarlaVersion:
    major: int
    minor: int
    patch: int
    suffix: str | None = None


def parse_carla_version(value: object) -> CarlaVersion:
    """Parse one anchored ASCII semantic version without coercion or trimming."""
    if not isinstance(value, str):
        raise ValueError("CARLA version must be a string")
    match = _VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("CARLA version must be major.minor.patch with an optional suffix")
    return CarlaVersion(
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
        match.group("suffix"),
    )


def _required_version(value: object) -> tuple[CarlaVersion | None, ValueError | None]:
    try:
        parsed = parse_carla_version(value)
    except ValueError as exc:
        return None, exc
    if (parsed.major, parsed.minor, parsed.patch) != REQUIRED_CARLA_COMPONENTS:
        return parsed, ValueError(f"semantic base must be {REQUIRED_CARLA_VERSION}")
    return parsed, None


def versions_compatible(client_version: object, server_version: object) -> bool:
    """Validate client and server independently against CARLA 0.9.16."""
    _client, client_error = _required_version(client_version)
    _server, server_error = _required_version(server_version)
    return client_error is None and server_error is None


@dataclass(frozen=True, slots=True)
class VersionReport:
    client_version: str
    server_version: str
    compatible: bool


def _version_diagnostic(value: object) -> str:
    if isinstance(value, str):
        return _bounded_text(value, VERSION_TEXT_LIMIT)
    return _bounded_text(
        f"<{type(value).__name__}> {_bounded_text(value, VERSION_TEXT_LIMIT)}",
        VERSION_TEXT_LIMIT,
    )


def verify_versions(client) -> VersionReport:
    """Query and independently validate both versions before world access."""
    try:
        raw_client_version = client.get_client_version()
    except Exception as exc:
        raise RuntimeError(_phase_error_message("client_version_query", exc)) from exc
    try:
        raw_server_version = client.get_server_version()
    except Exception as exc:
        raise RuntimeError(_phase_error_message("server_version_query", exc)) from exc

    _client, client_error = _required_version(raw_client_version)
    _server, server_error = _required_version(raw_server_version)
    if client_error is not None or server_error is not None:
        incompatible = []
        causes = []
        if client_error is not None:
            incompatible.append("client version is incompatible")
            causes.append(client_error)
        if server_error is not None:
            incompatible.append("server version is incompatible")
            causes.append(server_error)
        message = _bounded_text(
            "version_validation failed: "
            + "; ".join(incompatible)
            + f"; required={REQUIRED_CARLA_VERSION!r}"
            + f"; client={_version_diagnostic(raw_client_version)!r}"
            + f"; server={_version_diagnostic(raw_server_version)!r}",
            MAX_PRIMARY_MESSAGE_LENGTH,
        )
        raise RuntimeError(message) from causes[0]

    return VersionReport(
        client_version=raw_client_version,
        server_version=raw_server_version,
        compatible=True,
    )


def connect_client(carla_module, config: IntegrationConfig):
    """Construct and configure a client with bounded phase diagnostics."""
    try:
        client = carla_module.Client(config.host, config.port)
    except Exception as exc:
        raise RuntimeError(_phase_error_message("client_construction", exc)) from exc
    try:
        client.set_timeout(config.timeout_seconds)
    except Exception as exc:
        raise RuntimeError(_phase_error_message("timeout_configuration", exc)) from exc
    return client


def acquire_world(client):
    try:
        return client.get_world()
    except Exception as exc:
        raise RuntimeError(_phase_error_message("world_acquisition", exc)) from exc


def inspect_world_mode(world) -> WorldMode:
    """Inspect current settings without changing any world state."""
    try:
        settings = world.get_settings()
        synchronous = settings.synchronous_mode
        if type(synchronous) is not bool:
            raise TypeError("synchronous_mode must be bool")
    except Exception as exc:
        raise RuntimeError(_phase_error_message("settings_inspection", exc)) from exc
    return WorldMode.SYNCHRONOUS if synchronous else WorldMode.ASYNCHRONOUS


def require_progression_policy(mode: WorldMode, tick_owner: bool) -> ProgressionPolicy:
    """Apply the safe ownership policy and skip an unowned synchronous world."""
    try:
        policy = decide_world_progression(mode, tick_owner)
    except Exception as exc:
        raise RuntimeError(
            _phase_error_message("progression_policy_selection", exc)
        ) from exc
    if policy is ProgressionPolicy.SKIP:
        raise unittest.SkipTest(SYNCHRONOUS_UNOWNED_MESSAGE)
    return policy


def progress_world_once(
    world, policy: ProgressionPolicy, deadline: Deadline
) -> object:
    """Perform zero or one deadline-capped CARLA progression RPC."""
    if not isinstance(policy, ProgressionPolicy):
        raise TypeError("policy must be ProgressionPolicy")
    if policy is ProgressionPolicy.SKIP:
        raise unittest.SkipTest(SYNCHRONOUS_UNOWNED_MESSAGE)

    phase = "world_progression"
    deadline.raise_if_expired(phase)
    remaining = deadline.remaining()
    timeout = min(PROGRESSION_STEP_LIMIT_SECONDS, remaining)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise PhaseTimeout(phase, deadline.timeout_seconds)

    try:
        if policy is ProgressionPolicy.PASSIVE_WAIT:
            return world.wait_for_tick(timeout)
        if policy is ProgressionPolicy.OWNED_TICK:
            return world.tick(timeout)
    except _LIVE_RPC_EXCEPTIONS as exc:
        raise ProgressionRPCError(
            _phase_error_message("world_progression_rpc", exc)
        ) from exc
    raise TypeError("unsupported progression policy")


def configure_ego_blueprint(blueprint) -> bool:
    """Set literal role_name='hero' when the selected blueprint supports it."""
    role_name_supported = bool(blueprint.has_attribute("role_name"))
    if role_name_supported:
        blueprint.set_attribute("role_name", "hero")
    return role_name_supported


@dataclass(frozen=True, slots=True)
class OwnedEgo:
    actor: object
    actor_id: object
    type_id: object
    role_name: object
    role_name_required: bool
    identity_errors: tuple[str, ...] = ()
    pending_base_exception: BaseException | None = None


def _identity_observation_error(field: str, exc: BaseException) -> str:
    return _bounded_text(
        f"{field} unavailable: {type(exc).__name__}: "
        f"{_bounded_text(exc, VERSION_TEXT_LIMIT)}",
        MAX_PRIMARY_MESSAGE_LENGTH,
    )


def capture_owned_ego(actor, role_name_required: bool) -> OwnedEgo:
    """Retain the exact proxy and copy bounded identity data immediately."""
    if type(role_name_required) is not bool:
        raise TypeError("role_name_required must be bool")

    errors = []
    pending: BaseException | None = None
    actor_id = None
    type_id = None
    role_name = None

    try:
        actor_id = actor.id
    except BaseException as exc:  # retain proxy even across KeyboardInterrupt
        errors.append(_identity_observation_error("actor id", exc))
        if not isinstance(exc, Exception):
            pending = exc

    if pending is None:
        try:
            type_id = actor.type_id
        except BaseException as exc:
            errors.append(_identity_observation_error("type_id", exc))
            if not isinstance(exc, Exception):
                pending = exc

    if pending is None and role_name_required:
        try:
            attributes = actor.attributes
            if not isinstance(attributes, Mapping):
                raise TypeError("actor attributes must be a mapping")
            role_name = attributes.get("role_name")
        except BaseException as exc:
            errors.append(_identity_observation_error("role_name", exc))
            if not isinstance(exc, Exception):
                pending = exc

    return OwnedEgo(
        actor=actor,
        actor_id=actor_id,
        type_id=type_id,
        role_name=role_name,
        role_name_required=role_name_required,
        identity_errors=tuple(errors),
        pending_base_exception=pending,
    )


def spawn_owned_ego(
    world,
    blueprint,
    deadline: Deadline,
    *,
    role_name_required: bool = False,
    ownership_sink: Callable[[object], None] | None = None,
) -> OwnedEgo:
    """Attempt spawn points in order under one fixed host deadline."""
    try:
        spawn_points = tuple(world.get_map().get_spawn_points())
    except Exception as exc:
        raise RuntimeError(_phase_error_message("ego_spawn", exc)) from exc
    if not spawn_points:
        raise RuntimeError("ego_spawn failed: current map has no spawn points")

    for spawn_point in spawn_points:
        deadline.raise_if_expired("ego_spawn")
        actor = None
        try:
            actor = world.try_spawn_actor(blueprint, spawn_point)
            if actor is not None:
                if ownership_sink is not None:
                    ownership_sink(actor)
                return capture_owned_ego(actor, role_name_required)
        except BaseException as exc:
            if actor is None:
                if isinstance(exc, Exception):
                    raise RuntimeError(
                        _phase_error_message("ego_spawn", exc)
                    ) from exc
                raise

            # The RPC has succeeded.  Retain its exact proxy and copy its
            # identity even when an asynchronous exception lands between the
            # RPC return and the normal retention/capture instructions.
            if ownership_sink is not None:
                ownership_sink(actor)
            recovered = capture_owned_ego(actor, role_name_required)
            return OwnedEgo(
                actor=recovered.actor,
                actor_id=recovered.actor_id,
                type_id=recovered.type_id,
                role_name=recovered.role_name,
                role_name_required=recovered.role_name_required,
                identity_errors=recovered.identity_errors,
                pending_base_exception=exc,
            )

    raise RuntimeError(
        f"ego_spawn failed: all {len(spawn_points)} spawn points are occupied"
    )


def _valid_owned_actor_id(actor_id: object) -> bool:
    return type(actor_id) is int and actor_id > 0


def verify_ego_identity(owned: OwnedEgo) -> ValidationFailure | None:
    """Validate copied identity; actor.is_alive is deliberately irrelevant."""
    if not isinstance(owned, OwnedEgo):
        raise TypeError("owned must be OwnedEgo")
    problems = list(owned.identity_errors)
    if not _valid_owned_actor_id(owned.actor_id):
        problems.append("actor id is not a positive built-in integer")
    if type(owned.type_id) is not str or not owned.type_id.startswith("vehicle."):
        problems.append(
            f"type_id {_bounded_text(owned.type_id, VERSION_TEXT_LIMIT)!r} "
            "is not a vehicle string"
        )
    if owned.role_name_required and (
        type(owned.role_name) is not str or owned.role_name != "hero"
    ):
        problems.append(
            f"role_name {_bounded_text(owned.role_name, VERSION_TEXT_LIMIT)!r} "
            "is not 'hero'"
        )
    if not problems:
        return None
    return ValidationFailure(
        "ego_identity",
        _bounded_text("; ".join(problems), MAX_PRIMARY_MESSAGE_LENGTH),
        1,
    )


def destroy_owned_ego(owned: OwnedEgo) -> None:
    """Submit destruction once and require the literal success result True."""
    result = owned.actor.destroy()
    if result is not True:
        raise RuntimeError(
            f"destroy() returned {_bounded_text(result, VERSION_TEXT_LIMIT)}; expected True"
        )


def _observe_present_owned_actors(world, actor_ids) -> tuple[int, ...]:
    """Return the ids still present in one fresh per-frame world snapshot.

    CARLA's client caches every actor this client spawned, so
    ``world.get_actor``/``get_actors`` keep returning destroyed actors for the
    rest of the episode; only per-frame snapshot membership can prove absence.
    An id is treated as absent only when ``has_actor`` returns literal False.
    """
    snapshot = world.get_snapshot()
    return tuple(
        actor_id
        for actor_id in actor_ids
        if snapshot.has_actor(actor_id) is not False
    )


def verify_ego_disappearance(
    world, ego_id: int, policy: ProgressionPolicy, deadline: Deadline
) -> CleanupVerification:
    """Confirm absence only through per-frame snapshots of the copied id."""
    if not _valid_owned_actor_id(ego_id):
        return CleanupVerification.UNVERIFIED_DISCONNECTED
    if not isinstance(policy, ProgressionPolicy):
        raise TypeError("policy must be ProgressionPolicy")

    while True:
        try:
            present = _observe_present_owned_actors(world, (ego_id,))
        except _LIVE_RPC_EXCEPTIONS:
            return CleanupVerification.UNVERIFIED_DISCONNECTED
        if not present:
            return CleanupVerification.CONFIRMED
        if deadline.expired():
            return CleanupVerification.FAILED_ACTORS_REMAIN
        try:
            progress_world_once(world, policy, deadline)
        except PhaseTimeout:
            return CleanupVerification.FAILED_ACTORS_REMAIN
        except ProgressionRPCError:
            return CleanupVerification.UNVERIFIED_DISCONNECTED


@dataclass(frozen=True, slots=True)
class ConnectionSpawnReport:
    """Compact, CARLA-free record of one live connection/spawn cycle."""

    versions: VersionReport
    world_mode: WorldMode
    policy: ProgressionPolicy
    ego_id: int | None
    ego_type_id: str | None
    outcome: CleanupOutcome


@dataclass(frozen=True, slots=True)
class SensorIdentityRecord:
    name: str
    actor_id: int
    type_id: str


@dataclass(frozen=True, slots=True)
class SensorHealthReport:
    name: str
    attached: bool
    messages_received: int
    error_count: int
    callback_error_present: bool


@dataclass(frozen=True, slots=True)
class SensorReadinessEvaluation:
    ready: bool
    failure: ValidationFailure | None
    health: tuple[SensorHealthReport, ...]
    ego_present: bool
    gnss_present: bool
    imu_present: bool
    camera_present: bool


@dataclass(frozen=True, slots=True)
class CameraValidationReport:
    width_px: int
    height_px: int
    horizontal_fov_deg: float
    pixel_format: str
    row_stride_bytes: int
    payload_length: int
    transform_present: bool


@dataclass(frozen=True, slots=True)
class OwnedActorsVerification:
    verification: CleanupVerification
    remaining_actor_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SensorSuiteValidationReport:
    """Frozen CARLA-free evidence from one Mission 4G.3 live cycle."""

    versions: VersionReport
    world_mode: WorldMode
    policy: ProgressionPolicy
    ego_id: int | None
    ego_type_id: str | None
    ego_role_valid: bool
    sensor_identities: tuple[SensorIdentityRecord, ...]
    attachment_success: bool
    tick_registration_success: bool
    readiness_success: bool
    sensor_health: tuple[SensorHealthReport, ...]
    ego_telemetry_present: bool
    gnss_present: bool
    imu_present: bool
    camera_present: bool
    camera: CameraValidationReport | None
    primary_error: ValidationFailure | None
    cleanup_errors: tuple[str, ...]
    final_disappearance: CleanupVerification
    remaining_owned_actor_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ActorKey:
    """Cycle-local identity for one actor owned by a repeated run."""

    cycle_index: int
    actor_id: int


@dataclass(frozen=True, slots=True)
class RepeatedCycleSummary:
    """Compact CARLA-free evidence retained for one completed live cycle."""

    cycle_index: int
    success: bool
    elapsed_seconds: float
    ego_id: int | None
    sensor_actor_ids: tuple[int, ...]
    readiness_success: bool
    primary_error: ValidationFailure | None
    cleanup_errors: tuple[str, ...]
    disappearance: CleanupVerification
    remaining_actor_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RepeatedSensorSuiteValidationReport:
    """Bounded aggregate result from repeated complete Mission 4G.3 cycles."""

    execution_mode: ExecutionMode
    requested_cycle_count: int | None
    requested_soak_seconds: float
    attempted_cycle_count: int
    completed_cycle_count: int
    passed_cycle_count: int
    elapsed_seconds: float
    termination_reason: TerminationReason
    cycles: tuple[RepeatedCycleSummary, ...]
    omitted_cycle_summary_count: int
    first_primary_error: ValidationFailure | None
    failed_cycle_index: int | None
    supplemental_cleanup_errors: tuple[str, ...]
    omitted_cleanup_error_count: int
    final_disappearance: CleanupVerification | None
    remaining_owned_actors: tuple[ActorKey, ...]
    success: bool
    memory: MemoryStabilityReport = NOT_REQUESTED_MEMORY_REPORT


def _topology_identities(topology) -> tuple:
    try:
        identities = topology.identities
    except Exception as exc:
        raise ValueError("topology must expose identities") from exc
    if type(identities) is not tuple:
        raise ValueError("topology identities must be a tuple")
    return identities


def _copy_topology_evidence(topology) -> tuple[SensorIdentityRecord, ...]:
    """Copy only exact primitives for diagnostics, even from invalid topology."""
    copied = []
    try:
        identities = _topology_identities(topology)
    except ValueError:
        return ()
    for identity in identities:
        try:
            name = identity.name
            actor_id = identity.actor_id
            type_id = identity.type_id
        except Exception:
            continue
        if (
            type(name) is str
            and type(actor_id) is int
            and actor_id > 0
            and type(type_id) is str
        ):
            copied.append(SensorIdentityRecord(name, actor_id, type_id))
    return tuple(copied)


def validate_sensor_topology(topology, ego_id: int) -> tuple[SensorIdentityRecord, ...]:
    """Strictly validate the exact ordered five-sensor copied topology."""
    if not _valid_owned_actor_id(ego_id):
        raise ValueError("ego id must be a positive built-in integer")
    identities = _topology_identities(topology)
    problems = []
    if len(identities) != len(EXPECTED_SENSOR_NAMES):
        problems.append(
            f"expected {len(EXPECTED_SENSOR_NAMES)} identities, got {len(identities)}"
        )

    copied = []
    seen_names = set()
    seen_ids = set()
    for index, identity in enumerate(identities):
        try:
            name = identity.name
            actor_id = identity.actor_id
            type_id = identity.type_id
            identity_errors = identity.identity_errors
        except Exception as exc:
            problems.append(
                f"identity {index} unreadable: {type(exc).__name__}"
            )
            continue

        expected_name = (
            EXPECTED_SENSOR_NAMES[index]
            if index < len(EXPECTED_SENSOR_NAMES)
            else None
        )
        expected_type = (
            EXPECTED_SENSOR_TYPES[index]
            if index < len(EXPECTED_SENSOR_TYPES)
            else None
        )
        if type(name) is not str or name != expected_name:
            problems.append(
                f"identity {index} name {_bounded_text(name, VERSION_TEXT_LIMIT)!r} "
                f"does not match {expected_name!r}"
            )
        if type(identity_errors) is not tuple or identity_errors:
            problems.append(f"identity {index} contains identity observation errors")
        if not _valid_owned_actor_id(actor_id):
            problems.append(
                f"identity {index} actor id is not a positive built-in integer"
            )
        if type(type_id) is not str or type_id != expected_type:
            problems.append(
                f"identity {index} type_id {_bounded_text(type_id, VERSION_TEXT_LIMIT)!r} "
                f"does not match {expected_type!r}"
            )
        if type(name) is str:
            if name in seen_names:
                problems.append(f"duplicate sensor name {name!r}")
            seen_names.add(name)
        if _valid_owned_actor_id(actor_id):
            if actor_id in seen_ids:
                problems.append(f"duplicate sensor actor id {actor_id}")
            if actor_id == ego_id:
                problems.append(f"sensor actor id {actor_id} equals ego id")
            seen_ids.add(actor_id)
        if (
            type(name) is str
            and _valid_owned_actor_id(actor_id)
            and type(type_id) is str
        ):
            copied.append(SensorIdentityRecord(name, actor_id, type_id))

    if problems:
        raise ValueError(
            _bounded_text("; ".join(problems), MAX_PRIMARY_MESSAGE_LENGTH)
        )
    return tuple(copied)


def evaluate_sensor_readiness(snapshot) -> SensorReadinessEvaluation:
    """Classify one immutable snapshot without making a CARLA call."""
    try:
        health_values = snapshot.sensors
        if type(health_values) is not tuple:
            raise TypeError("sensor health must be a tuple")
    except Exception as exc:
        failure = _primary_failure("sensor_health", exc)
        return SensorReadinessEvaluation(
            False, failure, (), False, False, False, False
        )

    problems = []
    health_by_name = {}
    copied_health = []
    for index, health in enumerate(health_values):
        try:
            name = health.name
            attached = health.attached
            messages_received = health.messages_received
            error_count = health.error_count
            last_error = health.last_error
        except Exception as exc:
            problems.append(f"health {index} unreadable: {type(exc).__name__}")
            continue
        if type(name) is not str or name in health_by_name:
            problems.append(f"health {index} has invalid or duplicate name")
            continue
        health_by_name[name] = health
        if attached is not True:
            problems.append(f"sensor {name!r} is not attached")
        if type(messages_received) is not int or messages_received < 0:
            problems.append(f"sensor {name!r} has invalid message count")
            messages_received = 0
        if type(error_count) is not int or error_count < 0:
            problems.append(f"sensor {name!r} has invalid error count")
            error_count = 1
        callback_error = error_count != 0 or last_error is not None
        if callback_error:
            problems.append(f"sensor {name!r} retained a callback error")
        copied_health.append(
            SensorHealthReport(
                name=name,
                attached=attached is True,
                messages_received=messages_received,
                error_count=error_count,
                callback_error_present=callback_error,
            )
        )

    if set(health_by_name) != set(EXPECTED_SENSOR_NAMES) or len(health_by_name) != 5:
        problems.append("sensor health names do not match the exact five sensors")

    try:
        ego = snapshot.ego
        gnss = snapshot.gnss
        imu = snapshot.imu
        camera = snapshot.camera
    except Exception as exc:
        problems.append(f"telemetry fields unreadable: {type(exc).__name__}")
        ego = gnss = imu = camera = None

    ego_present = ego is not None
    if ego_present:
        try:
            if ego.vehicle_alive is not True:
                problems.append("ego vehicle_alive is not literal True")
        except Exception as exc:
            problems.append(f"ego health unreadable: {type(exc).__name__}")

    gnss_present = gnss is not None
    imu_present = imu is not None
    camera_present = camera is not None
    required_counts_ready = all(
        name in health_by_name
        and type(health_by_name[name].messages_received) is int
        and health_by_name[name].messages_received >= 1
        for name in ("gnss", "imu", "rgb_camera")
    )
    ordered_health = tuple(
        next((item for item in copied_health if item.name == name), None)
        for name in EXPECTED_SENSOR_NAMES
    )
    copied = tuple(item for item in ordered_health if item is not None)
    failure = (
        ValidationFailure(
            "sensor_health",
            _bounded_text("; ".join(problems), MAX_PRIMARY_MESSAGE_LENGTH),
            1,
        )
        if problems
        else None
    )
    ready = (
        failure is None
        and ego_present
        and gnss_present
        and imu_present
        and camera_present
        and required_counts_ready
    )
    return SensorReadinessEvaluation(
        ready,
        failure,
        copied,
        ego_present,
        gnss_present,
        imu_present,
        camera_present,
    )


def validate_camera_snapshot(snapshot, state: CycleState) -> CameraValidationReport:
    """Validate exact default RGB metadata and copy only compact evidence."""
    if not isinstance(state, CycleState):
        raise TypeError("state must be CycleState")
    camera = snapshot.camera
    if camera is None or state.camera_reference is not camera:
        raise ValueError("CycleState camera_reference is not snapshot.camera")
    metadata = camera.metadata
    checks = (
        (type(metadata.width_px) is int and metadata.width_px == EXPECTED_CAMERA_WIDTH, "width"),
        (type(metadata.height_px) is int and metadata.height_px == EXPECTED_CAMERA_HEIGHT, "height"),
        (
            type(metadata.horizontal_fov_deg) is float
            and metadata.horizontal_fov_deg == EXPECTED_CAMERA_FOV,
            "horizontal_fov",
        ),
        (
            type(metadata.pixel_format) is str
            and metadata.pixel_format == EXPECTED_CAMERA_PIXEL_FORMAT,
            "pixel_format",
        ),
        (
            type(metadata.row_stride_bytes) is int
            and metadata.row_stride_bytes == EXPECTED_CAMERA_ROW_STRIDE,
            "row_stride",
        ),
        (metadata.transform is not None, "transform"),
        (
            type(camera.rgb_bytes) is bytes
            and len(camera.rgb_bytes)
            == EXPECTED_CAMERA_WIDTH * EXPECTED_CAMERA_HEIGHT * 3,
            "payload_length",
        ),
    )
    failed = tuple(name for passed, name in checks if not passed)
    if failed:
        raise ValueError("invalid camera fields: " + ", ".join(failed))
    return CameraValidationReport(
        width_px=metadata.width_px,
        height_px=metadata.height_px,
        horizontal_fov_deg=metadata.horizontal_fov_deg,
        pixel_format=metadata.pixel_format,
        row_stride_bytes=metadata.row_stride_bytes,
        payload_length=len(camera.rgb_bytes),
        transform_present=True,
    )


def verify_owned_actors_disappearance(
    world,
    owned_actor_ids: Iterable[int],
    policy: ProgressionPolicy,
    deadline: Deadline,
) -> OwnedActorsVerification:
    """Verify all copied owned IDs with one shared fixed deadline."""
    if not isinstance(policy, ProgressionPolicy):
        raise TypeError("policy must be ProgressionPolicy")
    remaining = []
    seen = set()
    for actor_id in owned_actor_ids:
        if not _valid_owned_actor_id(actor_id):
            raise ValueError("owned actor ids must be positive built-in integers")
        if actor_id not in seen:
            seen.add(actor_id)
            remaining.append(actor_id)
    if not remaining:
        return OwnedActorsVerification(
            CleanupVerification.UNVERIFIED_DISCONNECTED, ()
        )

    while True:
        try:
            remaining = list(_observe_present_owned_actors(world, remaining))
        except _LIVE_RPC_EXCEPTIONS:
            return OwnedActorsVerification(
                CleanupVerification.UNVERIFIED_DISCONNECTED,
                tuple(remaining),
            )
        if not remaining:
            return OwnedActorsVerification(CleanupVerification.CONFIRMED, ())
        if deadline.expired():
            return OwnedActorsVerification(
                CleanupVerification.FAILED_ACTORS_REMAIN,
                tuple(remaining),
            )
        try:
            progress_world_once(world, policy, deadline)
        except PhaseTimeout:
            return OwnedActorsVerification(
                CleanupVerification.FAILED_ACTORS_REMAIN,
                tuple(remaining),
            )
        except ProgressionRPCError:
            return OwnedActorsVerification(
                CleanupVerification.UNVERIFIED_DISCONNECTED,
                tuple(remaining),
            )


def _primary_failure(phase: str, exc: BaseException) -> ValidationFailure:
    return ValidationFailure(
        phase,
        _bounded_text(
            f"{type(exc).__name__}: {_bounded_text(exc, MAX_PRIMARY_MESSAGE_LENGTH)}",
            MAX_PRIMARY_MESSAGE_LENGTH,
        ),
        1,
    )


def _report_without_owned_ego(
    versions: VersionReport,
    world_mode: WorldMode,
    policy: ProgressionPolicy,
    primary: ValidationFailure,
) -> ConnectionSpawnReport:
    return ConnectionSpawnReport(
        versions=versions,
        world_mode=world_mode,
        policy=policy,
        ego_id=None,
        ego_type_id=None,
        outcome=coordinate_cleanup(primary),
    )


def run_connection_spawn_cycle(
    environ: Mapping[str, str] | None = None,
    *,
    importer: Callable[[str], object] = importlib.import_module,
    monotonic: Callable[[], float] = time.monotonic,
    disappearance_timeout_seconds: float = EGO_DISAPPEARANCE_TIMEOUT_SECONDS,
) -> ConnectionSpawnReport:
    """Connect, inspect, spawn one owned ego, and always clean it up."""
    config = require_integration_opt_in(environ)
    dependencies = load_live_dependencies(environ, importer=importer)
    client = connect_client(dependencies.carla, config)
    versions = verify_versions(client)
    world = acquire_world(client)
    world_mode = inspect_world_mode(world)
    policy = require_progression_policy(world_mode, config.tick_owner)

    try:
        blueprint_library = world.get_blueprint_library()
        blueprint = dependencies.pick_vehicle_blueprint(blueprint_library)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        return _report_without_owned_ego(
            versions, world_mode, policy, _primary_failure("blueprint_selection", exc)
        )

    try:
        role_name_required = configure_ego_blueprint(blueprint)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        return _report_without_owned_ego(
            versions,
            world_mode,
            policy,
            _primary_failure("blueprint_configuration", exc),
        )

    retained_actor = None
    owned: OwnedEgo | None = None
    spawn_failure: ValidationFailure | None = None
    primary: ValidationFailure | None = None
    outcome: CleanupOutcome | None = None
    interrupted: KeyboardInterrupt | None = None
    cleanup_base_exception: BaseException | None = None
    destroy_completed = False
    destroy_error: Exception | None = None
    destroy_result: object = None
    verification_completed = False
    verification_error: Exception | None = None
    verification_result = CleanupVerification.UNVERIFIED_DISCONNECTED
    verification_deadline: Deadline | None = None
    verification_attempts = 0

    def retain_actor(actor) -> None:
        nonlocal retained_actor
        retained_actor = actor

    def defer_cleanup_base_exception(exc: BaseException) -> None:
        nonlocal cleanup_base_exception
        if cleanup_base_exception is None:
            cleanup_base_exception = exc

    def cleanup_owned_record() -> OwnedEgo:
        nonlocal owned
        if (
            not isinstance(owned, OwnedEgo)
            or owned.actor is not retained_actor
        ):
            owned = OwnedEgo(
                actor=retained_actor,
                actor_id=None,
                type_id=None,
                role_name=None,
                role_name_required=role_name_required,
                identity_errors=(
                    "owned actor record was unavailable or inconsistent",
                ),
            )
        return owned

    def destroy_for_cleanup() -> None:
        nonlocal destroy_completed, destroy_error, destroy_result
        if destroy_completed:
            if destroy_error is not None:
                raise destroy_error
            if destroy_result is not True:
                destroy_error = RuntimeError(
                    "destroy() returned "
                    f"{_bounded_text(destroy_result, VERSION_TEXT_LIMIT)}; "
                    "expected True"
                )
                raise destroy_error
            return

        cleanup_owned = cleanup_owned_record()
        invocation_started = False
        try:
            # Keep the state changes and call on one traced source line: a
            # boundary interrupt is retried before submission, never after it.
            invocation_started = True; destroy_result = cleanup_owned.actor.destroy(); destroy_completed = True
        except BaseException as exc:
            if not invocation_started:
                defer_cleanup_base_exception(exc)
                return destroy_for_cleanup()
            destroy_completed = True
            if isinstance(exc, Exception):
                destroy_error = exc
                raise
            defer_cleanup_base_exception(exc)
            destroy_error = RuntimeError(
                f"destroy() interrupted by {type(exc).__name__}"
            )
            raise destroy_error from exc
        return destroy_for_cleanup()

    def confirm_disappearance() -> CleanupVerification:
        nonlocal verification_attempts, verification_completed
        nonlocal verification_deadline, verification_error, verification_result
        if verification_completed:
            if verification_error is not None:
                raise verification_error
            return verification_result

        cleanup_owned = cleanup_owned_record()
        if not _valid_owned_actor_id(cleanup_owned.actor_id):
            verification_result = CleanupVerification.UNVERIFIED_DISCONNECTED
            verification_completed = True
            return verification_result

        if verification_deadline is None:
            verification_deadline = Deadline(
                disappearance_timeout_seconds,
                monotonic,
            )
        try:
            # The same fixed deadline is reused if one nonordinary boundary
            # interrupt occurs before the authoritative lookup can begin.
            verification_attempts += 1; verification_result = verify_ego_disappearance(world, cleanup_owned.actor_id, policy, verification_deadline); verification_completed = True
        except BaseException as exc:
            if isinstance(exc, Exception):
                verification_completed = True
                verification_result = CleanupVerification.UNVERIFIED_DISCONNECTED
                verification_error = exc
                raise
            defer_cleanup_base_exception(exc)
            if verification_attempts < 2:
                return confirm_disappearance()
            verification_completed = True
            verification_result = CleanupVerification.UNVERIFIED_DISCONNECTED
            verification_error = RuntimeError(
                f"disappearance verification interrupted by {type(exc).__name__}"
            )
            raise verification_error from exc
        return verification_result

    def coordinate_owned_cleanup() -> CleanupOutcome:
        cleanup_owned_record()
        while True:
            try:
                return coordinate_cleanup(
                    primary,
                    ego_cleanup=destroy_for_cleanup,
                    final_cleanup=confirm_disappearance,
                )
            except BaseException as exc:
                if isinstance(exc, Exception):
                    raise
                defer_cleanup_base_exception(exc)

    try:
        try:
            capture_failure: BaseException | None = None
            try:
                spawn_deadline = Deadline(config.timeout_seconds, monotonic)
                owned = spawn_owned_ego(
                    world,
                    blueprint,
                    spawn_deadline,
                    role_name_required=role_name_required,
                    ownership_sink=retain_actor,
                )
            except BaseException as exc:
                if retained_actor is None:
                    if isinstance(exc, Exception):
                        spawn_failure = _primary_failure("ego_spawn", exc)
                    else:
                        raise
                else:
                    capture_failure = exc
                    owned = OwnedEgo(
                        actor=retained_actor,
                        actor_id=None,
                        type_id=None,
                        role_name=None,
                        role_name_required=role_name_required,
                        identity_errors=(
                            _identity_observation_error("identity capture", exc),
                        ),
                    )

            if retained_actor is not None:
                if not isinstance(owned, OwnedEgo):
                    raise TypeError("owned ego capture must return OwnedEgo")
                if owned.actor is not retained_actor:
                    raise RuntimeError(
                        "owned ego capture returned a different actor proxy"
                    )
                if capture_failure is not None:
                    raise capture_failure
                if owned.pending_base_exception is not None:
                    raise owned.pending_base_exception
                result = verify_ego_identity(owned)
                if result is not None and not isinstance(result, ValidationFailure):
                    raise TypeError(
                        "verify_ego_identity must return ValidationFailure or None"
                    )
                primary = result
        except KeyboardInterrupt as exc:
            interrupted = exc
        except Exception as exc:
            if retained_actor is None:
                spawn_failure = _primary_failure("ego_spawn", exc)
            else:
                primary = _primary_failure("post_spawn_validation", exc)
        finally:
            if retained_actor is not None:
                outcome = coordinate_owned_cleanup()
    except BaseException as exc:
        if not isinstance(exc, Exception):
            defer_cleanup_base_exception(exc)
        try:
            if retained_actor is not None and outcome is None:
                outcome = coordinate_owned_cleanup()
        except BaseException:
            if interrupted is not None:
                raise interrupted.with_traceback(interrupted.__traceback__)
            if cleanup_base_exception is not None:
                raise cleanup_base_exception.with_traceback(
                    cleanup_base_exception.__traceback__
                )
            raise
        if interrupted is not None:
            raise interrupted.with_traceback(interrupted.__traceback__)
        if cleanup_base_exception is not None:
            raise cleanup_base_exception.with_traceback(
                cleanup_base_exception.__traceback__
            )
        raise

    if interrupted is not None:
        raise interrupted.with_traceback(interrupted.__traceback__)
    if cleanup_base_exception is not None:
        raise cleanup_base_exception.with_traceback(
            cleanup_base_exception.__traceback__
        )
    if retained_actor is None:
        if spawn_failure is None:
            raise RuntimeError("ego spawn ended without an actor or failure report")
        return _report_without_owned_ego(
            versions,
            world_mode,
            policy,
            spawn_failure,
        )
    if owned is None or outcome is None:
        raise RuntimeError("owned ego cleanup ended without a report")

    ego_id = owned.actor_id if _valid_owned_actor_id(owned.actor_id) else None
    ego_type_id = owned.type_id if type(owned.type_id) is str else None
    return ConnectionSpawnReport(
        versions=versions,
        world_mode=world_mode,
        policy=policy,
        ego_id=ego_id,
        ego_type_id=ego_type_id,
        outcome=outcome,
    )


def _empty_sensor_suite_report(
    versions: VersionReport,
    world_mode: WorldMode,
    policy: ProgressionPolicy,
    primary: ValidationFailure,
) -> SensorSuiteValidationReport:
    return SensorSuiteValidationReport(
        versions=versions,
        world_mode=world_mode,
        policy=policy,
        ego_id=None,
        ego_type_id=None,
        ego_role_valid=False,
        sensor_identities=(),
        attachment_success=False,
        tick_registration_success=False,
        readiness_success=False,
        sensor_health=(),
        ego_telemetry_present=False,
        gnss_present=False,
        imu_present=False,
        camera_present=False,
        camera=None,
        primary_error=primary,
        cleanup_errors=(),
        final_disappearance=CleanupVerification.UNVERIFIED_DISCONNECTED,
        remaining_owned_actor_ids=(),
    )


def _validate_attachment_report(report) -> None:
    try:
        results = report.results
        all_attached = report.all_attached
    except Exception as exc:
        raise ValueError("attachment report is unreadable") from exc
    if type(results) is not tuple or len(results) != len(EXPECTED_SENSOR_NAMES):
        raise ValueError("attachment report must contain exactly five results")
    for index, result in enumerate(results):
        try:
            name = result.name
            attached = result.attached
            error = result.error
        except Exception as exc:
            raise ValueError(f"attachment result {index} is unreadable") from exc
        if name != EXPECTED_SENSOR_NAMES[index]:
            raise ValueError(f"attachment result {index} has the wrong sensor name")
        if attached is not True or error is not None:
            raise ValueError(f"sensor {name!r} did not attach successfully")
    if all_attached is not True:
        raise ValueError("AttachReport.all_attached is not literal True")


def _cleanup_diagnostic(step: str, value) -> str:
    if isinstance(value, BaseException):
        detail = (
            f"{type(value).__name__}: "
            f"{_bounded_text(value, MAX_CLEANUP_ERROR_LENGTH)}"
        )
    else:
        detail = _bounded_text(value, MAX_CLEANUP_ERROR_LENGTH)
    return _bounded_text(f"{step} {detail}", MAX_CLEANUP_ERROR_LENGTH)


def run_sensor_suite_validation_cycle(
    environ: Mapping[str, str] | None = None,
    *,
    importer: Callable[[str], object] = importlib.import_module,
    monotonic: Callable[[], float] = time.monotonic,
    disappearance_timeout_seconds: float = EGO_DISAPPEARANCE_TIMEOUT_SECONDS,
) -> SensorSuiteValidationReport:
    """Run one exact-opt-in ego + default SensorSuite validation cycle."""
    config = require_integration_opt_in(environ)
    dependencies = load_live_dependencies(environ, importer=importer)
    client = connect_client(dependencies.carla, config)
    versions = verify_versions(client)
    world = acquire_world(client)
    world_mode = inspect_world_mode(world)
    policy = require_progression_policy(world_mode, config.tick_owner)

    try:
        blueprint_library = world.get_blueprint_library()
        blueprint = dependencies.pick_vehicle_blueprint(blueprint_library)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        return _empty_sensor_suite_report(
            versions,
            world_mode,
            policy,
            _primary_failure("blueprint_selection", exc),
        )
    try:
        role_name_required = configure_ego_blueprint(blueprint)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        return _empty_sensor_suite_report(
            versions,
            world_mode,
            policy,
            _primary_failure("blueprint_configuration", exc),
        )

    state = new_cycle_state(1)
    retained_actor = None
    owned: OwnedEgo | None = None
    suite = None
    topology = None
    attachment_report = None
    primary: ValidationFailure | None = None
    deferred: tuple[BaseException, object] | None = None
    cleanup_errors: list[str] = []
    sensor_identities: tuple[SensorIdentityRecord, ...] = ()
    attachment_success = False
    tick_registration_success = False
    readiness_success = False
    health: tuple[SensorHealthReport, ...] = ()
    ego_present = False
    gnss_present = False
    imu_present = False
    camera_present = False
    camera_report: CameraValidationReport | None = None
    final_verification = OwnedActorsVerification(
        CleanupVerification.UNVERIFIED_DISCONNECTED, ()
    )
    snapshot = None

    def retain_actor(actor) -> None:
        nonlocal retained_actor
        retained_actor = actor

    def set_primary(phase: str, exc: BaseException) -> None:
        nonlocal primary
        if primary is None:
            primary = _primary_failure(phase, exc)

    def set_validation_failure(failure: ValidationFailure) -> None:
        nonlocal primary
        if primary is None:
            primary = failure

    def defer(exc: BaseException) -> None:
        nonlocal deferred
        if deferred is None and not isinstance(exc, Exception):
            deferred = (exc, exc.__traceback__)

    def add_cleanup_error(step: str, value) -> None:
        text = _cleanup_diagnostic(step, value)
        if text not in cleanup_errors:
            cleanup_errors.append(text)

    try:
        try:
            spawn_deadline = Deadline(config.timeout_seconds, monotonic)
            owned = spawn_owned_ego(
                world,
                blueprint,
                spawn_deadline,
                role_name_required=role_name_required,
                ownership_sink=retain_actor,
            )
        except BaseException as exc:
            if retained_actor is None:
                if isinstance(exc, Exception):
                    return _empty_sensor_suite_report(
                        versions,
                        world_mode,
                        policy,
                        _primary_failure("ego_spawn", exc),
                    )
                raise
            if isinstance(exc, Exception):
                set_primary("ego_spawn", exc)
            else:
                defer(exc)

        if retained_actor is not None:
            if not isinstance(owned, OwnedEgo) or owned.actor is not retained_actor:
                set_primary(
                    "post_spawn_validation",
                    RuntimeError("owned ego capture was unavailable or inconsistent"),
                )
                owned = capture_owned_ego(retained_actor, role_name_required)
            if owned.pending_base_exception is not None:
                defer(owned.pending_base_exception)
            identity_failure = verify_ego_identity(owned)
            if identity_failure is not None:
                set_validation_failure(identity_failure)

        if retained_actor is not None and primary is None and deferred is None:
            try:
                sensor_dependencies = load_sensor_suite_dependencies(
                    environ, importer=importer
                )
            except Exception as exc:
                set_primary("sensor_suite_dependency_loading", exc)
            else:
                sensor_deadline = Deadline(config.timeout_seconds, monotonic)
                try:
                    sensor_deadline.raise_if_expired("sensor_suite_construction")
                    suite = sensor_dependencies.SensorSuite()
                except BaseException as exc:
                    if isinstance(exc, Exception):
                        set_primary("sensor_suite_construction", exc)
                    else:
                        defer(exc)

                if suite is not None and deferred is None:
                    try:
                        attachment_report = suite.attach(
                            world,
                            retained_actor,
                            pre_operation_guard=sensor_deadline.raise_if_expired,
                        )
                    except BaseException as exc:
                        if isinstance(exc, Exception):
                            set_primary("sensor_attachment", exc)
                        else:
                            defer(exc)

                if suite is not None:
                    try:
                        topology = suite.topology_snapshot()
                    except BaseException as exc:
                        if isinstance(exc, Exception):
                            set_primary("sensor_topology", exc)
                        else:
                            defer(exc)

                if topology is not None:
                    sensor_identities = _copy_topology_evidence(topology)
                    state.sensor_ids = tuple(
                        record.actor_id for record in sensor_identities
                    )

                if attachment_report is not None:
                    try:
                        _validate_attachment_report(attachment_report)
                    except Exception as exc:
                        set_primary("sensor_attachment", exc)
                    else:
                        attachment_success = True

                    try:
                        if attachment_report.tick_registered is not True:
                            raise ValueError("world tick registration did not succeed")
                    except Exception as exc:
                        set_primary("tick_registration", exc)
                    else:
                        tick_registration_success = True

                if topology is not None and owned is not None:
                    try:
                        validated = validate_sensor_topology(topology, owned.actor_id)
                    except Exception as exc:
                        set_primary("sensor_topology", exc)
                    else:
                        sensor_identities = validated
                        state.sensor_ids = tuple(
                            record.actor_id for record in validated
                        )

                if (
                    suite is not None
                    and sensor_deadline is not None
                    and primary is None
                    and deferred is None
                ):
                    while True:
                        try:
                            snapshot = suite.snapshot()
                            state.snapshot_reference = snapshot
                            state.camera_reference = snapshot.camera
                            evaluation = evaluate_sensor_readiness(snapshot)
                        except BaseException as exc:
                            if isinstance(exc, Exception):
                                set_primary("sensor_readiness", exc)
                            else:
                                defer(exc)
                            break

                        health = evaluation.health
                        ego_present = evaluation.ego_present
                        gnss_present = evaluation.gnss_present
                        imu_present = evaluation.imu_present
                        camera_present = evaluation.camera_present
                        if evaluation.failure is not None:
                            set_validation_failure(evaluation.failure)
                            break
                        if evaluation.ready:
                            readiness_success = True
                            try:
                                camera_report = validate_camera_snapshot(
                                    snapshot, state
                                )
                            except BaseException as exc:
                                readiness_success = False
                                if isinstance(exc, Exception):
                                    set_primary("camera_validation", exc)
                                else:
                                    defer(exc)
                            break
                        if sensor_deadline.expired():
                            set_primary(
                                "sensor_readiness",
                                PhaseTimeout(
                                    "sensor_readiness",
                                    sensor_deadline.timeout_seconds,
                                ),
                            )
                            break
                        try:
                            progress_world_once(world, policy, sensor_deadline)
                        except PhaseTimeout:
                            set_primary(
                                "sensor_readiness",
                                PhaseTimeout(
                                    "sensor_readiness",
                                    sensor_deadline.timeout_seconds,
                                ),
                            )
                            break
                        except BaseException as exc:
                            if isinstance(exc, Exception):
                                set_primary("sensor_readiness", exc)
                            else:
                                defer(exc)
                            break
    except BaseException as exc:
        if isinstance(exc, Exception):
            set_primary("sensor_cycle", exc)
        else:
            defer(exc)
    finally:
        # Full snapshots and image references are validation-only temporaries.
        state.snapshot_reference = None
        state.camera_reference = None

        if suite is not None:
            suite_cleanup = None
            for _attempt in range(len(EXPECTED_SENSOR_NAMES) * 3 + 3):
                try:
                    suite_cleanup = suite.destroy()
                except BaseException as exc:
                    add_cleanup_error("suite", exc)
                    defer(exc)
                    continue
                break
            if suite_cleanup is None:
                add_cleanup_error("suite", "cleanup did not return a report")
            else:
                try:
                    cleanup_results = suite_cleanup.results
                    if type(cleanup_results) is not tuple:
                        raise TypeError("cleanup results must be a tuple")
                    result_names = tuple(result.name for result in cleanup_results)
                    if result_names != EXPECTED_SENSOR_NAMES:
                        add_cleanup_error(
                            "suite", "cleanup results do not match exact sensor order"
                        )
                    created_names = {item.name for item in sensor_identities}
                    attached_names = set()
                    if attachment_report is not None:
                        attached_names = {
                            result.name
                            for result in attachment_report.results
                            if result.attached is True
                        }
                    for error in suite_cleanup.errors:
                        add_cleanup_error("suite", error)
                    for result in cleanup_results:
                        for error in result.errors:
                            add_cleanup_error(f"sensor {result.name}", error)
                        if (
                            result.name in created_names
                            and result.destroyed is not True
                        ):
                            add_cleanup_error(
                                f"sensor {result.name}",
                                "owned actor destruction was not successful",
                            )
                        if (
                            result.name in attached_names
                            and result.stopped is not True
                        ):
                            add_cleanup_error(
                                f"sensor {result.name}",
                                "owned listener stop was not successful",
                            )
                    if (
                        tick_registration_success
                        and suite_cleanup.tick_unregistered is not True
                    ):
                        add_cleanup_error(
                            "suite", "world tick registration was not unregistered"
                        )
                except BaseException as exc:
                    add_cleanup_error("suite report", exc)
                    defer(exc)

        if retained_actor is not None:
            ego_cleanup_finished = False
            for _attempt in range(2):
                invocation_started = False
                try:
                    # Keep the state change and call on one traced source
                    # line: a boundary interrupt is retried before submission,
                    # never after it.
                    invocation_started = True; ego_destroy_result = retained_actor.destroy()
                except BaseException as exc:
                    defer(exc)
                    if not invocation_started:
                        continue
                    add_cleanup_error("ego destroy", exc)
                    ego_cleanup_finished = True
                    break
                else:
                    ego_cleanup_finished = True
                    if ego_destroy_result is not True:
                        add_cleanup_error(
                            "ego destroy",
                            f"destroy() returned {_bounded_text(ego_destroy_result, VERSION_TEXT_LIMIT)}; expected True",
                        )
                    break
            if not ego_cleanup_finished:
                add_cleanup_error("ego destroy", "submission could not be completed")

        owned_ids = []
        for actor_id in state.sensor_ids:
            if _valid_owned_actor_id(actor_id) and actor_id not in owned_ids:
                owned_ids.append(actor_id)
        ego_id_for_cleanup = owned.actor_id if isinstance(owned, OwnedEgo) else None
        if (
            _valid_owned_actor_id(ego_id_for_cleanup)
            and ego_id_for_cleanup not in owned_ids
        ):
            owned_ids.append(ego_id_for_cleanup)

        if owned_ids:
            try:
                cleanup_deadline = Deadline(
                    disappearance_timeout_seconds, monotonic
                )
            except BaseException as exc:
                add_cleanup_error("final verification", exc)
                defer(exc)
            else:
                for _attempt in range(2):
                    try:
                        final_verification = verify_owned_actors_disappearance(
                            world, tuple(owned_ids), policy, cleanup_deadline
                        )
                    except BaseException as exc:
                        add_cleanup_error("final verification", exc)
                        defer(exc)
                        continue
                    break
                else:
                    final_verification = OwnedActorsVerification(
                        CleanupVerification.UNVERIFIED_DISCONNECTED,
                        tuple(owned_ids),
                    )

        state.snapshot_reference = None
        state.camera_reference = None
        snapshot = None

    if deferred is not None:
        exc, traceback = deferred
        raise exc.with_traceback(traceback)

    if retained_actor is None or owned is None:
        if primary is None:
            primary = ValidationFailure(
                "ego_spawn", "ego spawn ended without retained ownership", 1
            )
        return _empty_sensor_suite_report(versions, world_mode, policy, primary)

    ego_id = owned.actor_id if _valid_owned_actor_id(owned.actor_id) else None
    ego_type_id = owned.type_id if type(owned.type_id) is str else None
    ego_role_valid = (
        not owned.role_name_required
        or (type(owned.role_name) is str and owned.role_name == EGO_ROLE_NAME)
    )
    report = SensorSuiteValidationReport(
        versions=versions,
        world_mode=world_mode,
        policy=policy,
        ego_id=ego_id,
        ego_type_id=ego_type_id,
        ego_role_valid=ego_role_valid,
        sensor_identities=sensor_identities,
        attachment_success=attachment_success,
        tick_registration_success=tick_registration_success,
        readiness_success=readiness_success,
        sensor_health=health,
        ego_telemetry_present=ego_present,
        gnss_present=gnss_present,
        imu_present=imu_present,
        camera_present=camera_present,
        camera=camera_report,
        primary_error=primary,
        cleanup_errors=tuple(cleanup_errors),
        final_disappearance=final_verification.verification,
        remaining_owned_actor_ids=final_verification.remaining_actor_ids,
    )
    # Nothing live or payload-bearing remains needed once immutable evidence
    # and disappearance status have been copied into ``report``.
    state.snapshot_reference = None
    state.camera_reference = None
    snapshot = None
    topology = None
    attachment_report = None
    suite = None
    retained_actor = None
    owned = None
    client = None
    world = None
    dependencies = None
    return report


def _sensor_suite_validation_report_succeeded(
    report: SensorSuiteValidationReport,
) -> bool:
    """Apply the complete Mission 4G.3 live verdict without retaining payloads."""
    if not isinstance(report, SensorSuiteValidationReport):
        return False
    try:
        camera = report.camera
        health = {item.name: item for item in report.sensor_health}
        identities_valid = (
            len(report.sensor_identities) == len(EXPECTED_SENSOR_NAMES)
            and tuple(item.name for item in report.sensor_identities)
            == EXPECTED_SENSOR_NAMES
            and tuple(item.type_id for item in report.sensor_identities)
            == EXPECTED_SENSOR_TYPES
            and len({item.actor_id for item in report.sensor_identities}) == 5
            and all(
                _valid_owned_actor_id(item.actor_id)
                and item.actor_id != report.ego_id
                for item in report.sensor_identities
            )
        )
        health_valid = (
            set(health) == set(EXPECTED_SENSOR_NAMES)
            and len(health) == 5
            and all(
                item.attached is True
                and type(item.error_count) is int
                and item.error_count == 0
                and item.callback_error_present is False
                for item in health.values()
            )
            and all(
                type(health[name].messages_received) is int
                and health[name].messages_received >= 1
                for name in ("gnss", "imu", "rgb_camera")
            )
        )
        camera_valid = (
            camera is not None
            and camera.width_px == EXPECTED_CAMERA_WIDTH
            and camera.height_px == EXPECTED_CAMERA_HEIGHT
            and camera.horizontal_fov_deg == EXPECTED_CAMERA_FOV
            and camera.pixel_format == EXPECTED_CAMERA_PIXEL_FORMAT
            and camera.row_stride_bytes == EXPECTED_CAMERA_ROW_STRIDE
            and camera.payload_length
            == EXPECTED_CAMERA_WIDTH * EXPECTED_CAMERA_HEIGHT * 3
            and camera.transform_present is True
        )
        policy_valid = (
            report.world_mode is WorldMode.ASYNCHRONOUS
            and report.policy is ProgressionPolicy.PASSIVE_WAIT
        ) or (
            report.world_mode is WorldMode.SYNCHRONOUS
            and report.policy is ProgressionPolicy.OWNED_TICK
        )
        return (
            type(report.ego_id) is int
            and report.ego_id > 0
            and type(report.ego_type_id) is str
            and report.ego_type_id.startswith("vehicle.")
            and report.ego_role_valid is True
            and report.versions.compatible is True
            and versions_compatible(
                report.versions.client_version,
                report.versions.server_version,
            )
            and policy_valid
            and identities_valid
            and report.attachment_success is True
            and report.tick_registration_success is True
            and report.readiness_success is True
            and report.ego_telemetry_present is True
            and report.gnss_present is True
            and report.imu_present is True
            and report.camera_present is True
            and health_valid
            and camera_valid
            and report.primary_error is None
            and not report.cleanup_errors
            and report.final_disappearance is CleanupVerification.CONFIRMED
            and not report.remaining_owned_actor_ids
        )
    except Exception:
        return False


def _repeated_clock_value(monotonic: Callable[[], float]) -> float:
    if not callable(monotonic):
        raise TypeError("monotonic must be callable")
    raw_value = monotonic()
    if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
        raise ValueError("monotonic clock must return a finite value")
    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError("monotonic clock must return a finite value")
    return value


class _ClampedMonotonic:
    """Process-local host clock view whose effective time never moves backward."""

    __slots__ = ("_last", "_monotonic")

    def __init__(self, monotonic: Callable[[], float]) -> None:
        if not callable(monotonic):
            raise TypeError("monotonic must be callable")
        self._monotonic = monotonic
        self._last: float | None = None

    def __call__(self) -> float:
        current = _repeated_clock_value(self._monotonic)
        if self._last is not None and current < self._last:
            return self._last
        self._last = current
        return current


def _repeated_elapsed(start: float, end: float) -> float:
    return max(0.0, end - start)


def _cycle_failure_copy(
    failure: ValidationFailure | None, cycle_index: int
) -> ValidationFailure | None:
    if failure is None:
        return None
    if not isinstance(failure, ValidationFailure):
        return ValidationFailure(
            "cycle_report",
            "primary_error was not ValidationFailure or None",
            cycle_index,
        )
    return ValidationFailure(
        _bounded_text(failure.phase, MAX_PRIMARY_PHASE_LENGTH),
        _bounded_text(failure.message, MAX_PRIMARY_MESSAGE_LENGTH),
        cycle_index,
    )


def _compact_actor_ids(values: Iterable[object]) -> tuple[int, ...]:
    copied = []
    try:
        iterator = iter(values)
    except Exception:
        return ()
    for value in iterator:
        if _valid_owned_actor_id(value):
            copied.append(value)
    return tuple(copied)


def _copy_bounded_cleanup_errors(
    values: Iterable[object],
) -> tuple[tuple[str, ...], int]:
    retained: list[str] = []
    omitted = 0
    for value in values:
        if len(retained) < MAX_RETAINED_SUPPLEMENTAL_CLEANUP_ERRORS:
            retained.append(_bounded_text(value, MAX_CLEANUP_ERROR_LENGTH))
        else:
            omitted += 1
    return tuple(retained), omitted


def _summarize_repeated_cycle(
    report: SensorSuiteValidationReport,
    cycle_index: int,
    elapsed_seconds: float,
) -> tuple[RepeatedCycleSummary, int]:
    if not isinstance(report, SensorSuiteValidationReport):
        raise TypeError("single-cycle runner must return SensorSuiteValidationReport")
    try:
        cleanup_errors, omitted_cleanup_errors = _copy_bounded_cleanup_errors(
            report.cleanup_errors
        )
        disappearance = (
            report.final_disappearance
            if isinstance(report.final_disappearance, CleanupVerification)
            else CleanupVerification.UNVERIFIED_DISCONNECTED
        )
        return (
            RepeatedCycleSummary(
                cycle_index=cycle_index,
                success=_sensor_suite_validation_report_succeeded(report),
                elapsed_seconds=elapsed_seconds,
                ego_id=(
                    report.ego_id if _valid_owned_actor_id(report.ego_id) else None
                ),
                sensor_actor_ids=_compact_actor_ids(
                    identity.actor_id for identity in report.sensor_identities
                ),
                readiness_success=report.readiness_success is True,
                primary_error=_cycle_failure_copy(report.primary_error, cycle_index),
                cleanup_errors=cleanup_errors,
                disappearance=disappearance,
                remaining_actor_ids=_compact_actor_ids(
                    report.remaining_owned_actor_ids
                ),
            ),
            omitted_cleanup_errors,
        )
    finally:
        report = None


def _failure_termination_reason(summary: RepeatedCycleSummary) -> TerminationReason:
    if summary.primary_error is not None or summary.readiness_success is not True:
        return TerminationReason.FUNCTIONAL_FAILURE
    if summary.cleanup_errors:
        return TerminationReason.CLEANUP_FAILURE
    if (
        summary.disappearance is not CleanupVerification.CONFIRMED
        or summary.remaining_actor_ids
    ):
        return TerminationReason.DISAPPEARANCE_FAILURE
    return TerminationReason.FUNCTIONAL_FAILURE


def _retain_cycle_summary(
    summary: RepeatedCycleSummary,
    first: list[RepeatedCycleSummary],
    final: list[RepeatedCycleSummary],
) -> None:
    if summary.cycle_index <= FIRST_RETAINED_CYCLE_SUMMARIES:
        first.append(summary)
        return
    if len(final) == FINAL_RETAINED_CYCLE_SUMMARIES:
        del final[0]
    final.append(summary)


def run_repeated_sensor_suite_validation_cycles(
    environ: Mapping[str, str] | None = None,
    *,
    importer: Callable[[str], object] = importlib.import_module,
    monotonic: Callable[[], float] = time.monotonic,
    disappearance_timeout_seconds: float = EGO_DISAPPEARANCE_TIMEOUT_SECONDS,
    admission_guard: Callable[[], TerminationReason | None] | None = None,
) -> RepeatedSensorSuiteValidationReport:
    """Run count-limited or host-monotonic complete cycles with strict fail-fast."""
    config = require_integration_opt_in(environ)
    execution_mode = select_execution_mode(config)
    if admission_guard is not None and not callable(admission_guard):
        raise TypeError("admission_guard must be callable or None")
    total_soak_deadline: Deadline | None = None
    if execution_mode is ExecutionMode.TIMED_SOAK:
        clock: Callable[[], float] = _ClampedMonotonic(monotonic)
        total_soak_deadline = Deadline(config.soak_seconds, clock)
        run_started = total_soak_deadline.start
    else:
        clock = monotonic
        run_started = _repeated_clock_value(clock)

    attempted = 0
    completed = 0
    passed = 0
    first_summaries: list[RepeatedCycleSummary] = []
    final_summaries: list[RepeatedCycleSummary] = []
    summarized = 0
    first_primary: ValidationFailure | None = None
    failed_cycle_index: int | None = None
    supplemental_cleanup_errors: list[str] = []
    omitted_cleanup_errors = 0
    remaining_owned_actors: list[ActorKey] = []
    final_disappearance: CleanupVerification | None = None
    termination_reason: TerminationReason
    cycle_index = 1

    while True:
        if admission_guard is not None:
            admission_veto = admission_guard()
            if admission_veto is not None:
                if not isinstance(admission_veto, TerminationReason):
                    raise TypeError(
                        "admission_guard must return TerminationReason or None"
                    )
                termination_reason = admission_veto
                break
        if execution_mode is ExecutionMode.CYCLE_COUNT:
            if cycle_index > config.cycles:
                termination_reason = TerminationReason.CYCLE_LIMIT_REACHED
                break
        else:
            if total_soak_deadline is None:
                raise RuntimeError("timed soak requires a total deadline")
            cycle_started, soak_expired = total_soak_deadline._sample_expiration()
            if soak_expired:
                termination_reason = TerminationReason.SOAK_DURATION_REACHED
                break

        if execution_mode is ExecutionMode.CYCLE_COUNT:
            cycle_started = _repeated_clock_value(clock)
        cycle_report = None
        try:
            attempted += 1
            cycle_report = run_sensor_suite_validation_cycle(
                environ,
                importer=importer,
                monotonic=clock,
                disappearance_timeout_seconds=disappearance_timeout_seconds,
            )
            completed += 1
            cycle_finished = _repeated_clock_value(clock)
            summary, summary_omitted_cleanup_errors = _summarize_repeated_cycle(
                cycle_report,
                cycle_index,
                _repeated_elapsed(cycle_started, cycle_finished),
            )
        finally:
            # Also clear on a post-cycle clock/summary exception so the outer
            # traceback frame cannot retain the complete inner report.
            cycle_report = None
        _retain_cycle_summary(summary, first_summaries, final_summaries)
        summarized += 1
        final_disappearance = summary.disappearance
        if summary.success:
            passed += 1
        else:
            failed_cycle_index = cycle_index
        if first_primary is None and summary.primary_error is not None:
            first_primary = summary.primary_error
        for error in summary.cleanup_errors:
            if (
                len(supplemental_cleanup_errors)
                < MAX_RETAINED_SUPPLEMENTAL_CLEANUP_ERRORS
            ):
                supplemental_cleanup_errors.append(
                    _bounded_text(
                        f"cycle {cycle_index}: {error}",
                        MAX_CLEANUP_ERROR_LENGTH,
                    )
                )
            else:
                omitted_cleanup_errors += 1
        omitted_cleanup_errors += summary_omitted_cleanup_errors
        remaining_owned_actors.extend(
            ActorKey(cycle_index, actor_id)
            for actor_id in summary.remaining_actor_ids
        )

        if not summary.success:
            termination_reason = _failure_termination_reason(summary)
            break
        cycle_index += 1

    run_finished = _repeated_clock_value(clock)
    retained_summaries = tuple(first_summaries + final_summaries)
    omitted_cycle_summaries = summarized - len(retained_summaries)
    retained_cleanup_errors = tuple(supplemental_cleanup_errors)
    retained_remaining = tuple(remaining_owned_actors)
    common_success = (
        completed == passed
        and all(summary.success for summary in retained_summaries)
        and not retained_cleanup_errors
        and omitted_cleanup_errors == 0
        and (
            final_disappearance is CleanupVerification.CONFIRMED
            if completed
            else final_disappearance is None
        )
        and not retained_remaining
    )
    if execution_mode is ExecutionMode.CYCLE_COUNT:
        overall_success = (
            termination_reason is TerminationReason.CYCLE_LIMIT_REACHED
            and attempted == config.cycles
            and completed == config.cycles
            and passed == config.cycles
            and common_success
        )
        requested_cycle_count: int | None = config.cycles
        requested_soak_seconds = 0.0
    else:
        overall_success = (
            termination_reason is TerminationReason.SOAK_DURATION_REACHED
            and completed >= 1
            and passed >= 1
            and common_success
        )
        requested_cycle_count = None
        requested_soak_seconds = config.soak_seconds

    return RepeatedSensorSuiteValidationReport(
        execution_mode=execution_mode,
        requested_cycle_count=requested_cycle_count,
        requested_soak_seconds=requested_soak_seconds,
        attempted_cycle_count=attempted,
        completed_cycle_count=completed,
        passed_cycle_count=passed,
        elapsed_seconds=_repeated_elapsed(run_started, run_finished),
        termination_reason=termination_reason,
        cycles=retained_summaries,
        omitted_cycle_summary_count=omitted_cycle_summaries,
        first_primary_error=first_primary,
        failed_cycle_index=failed_cycle_index,
        supplemental_cleanup_errors=retained_cleanup_errors,
        omitted_cleanup_error_count=omitted_cleanup_errors,
        final_disappearance=final_disappearance,
        remaining_owned_actors=retained_remaining,
        success=overall_success,
    )


def _valid_memory_thresholds(thresholds) -> bool:
    """Validate every threshold field without trusting parser provenance."""
    if type(thresholds) is not MemoryThresholds:
        return False
    byte_limits = (
        thresholds.working_set_max_peak_growth_bytes,
        thresholds.working_set_max_final_growth_bytes,
        thresholds.private_commit_max_peak_growth_bytes,
        thresholds.private_commit_max_final_growth_bytes,
    )
    if not all(
        type(value) is int and 0 <= value <= MAX_MEMORY_BYTE_LIMIT
        for value in byte_limits
    ):
        return False
    percent_limits = (
        thresholds.working_set_max_final_growth_percent,
        thresholds.private_commit_max_final_growth_percent,
    )
    if not all(
        type(value) is float
        and math.isfinite(value)
        and 0.0 <= value <= MAX_MEMORY_PERCENT_LIMIT
        for value in percent_limits
    ):
        return False
    slope_limits = (
        thresholds.working_set_max_slope_bytes_per_minute,
        thresholds.private_commit_max_slope_bytes_per_minute,
    )
    return all(
        value is None
        or (
            type(value) is float
            and math.isfinite(value)
            and value >= 0.0
        )
        for value in slope_limits
    )


def _memory_percentage_matches(
    observed: float | None, growth_bytes: int, start_bytes: int
) -> bool:
    """Compare stored ratio evidence with deterministic float tolerance.

    Reports are produced with one binary-float division and multiplication.
    ``isclose`` at 1e-12 relative/absolute tolerance accepts only insignificant
    representation drift while still rejecting materially altered evidence.
    A zero baseline has no defined percentage and must be represented by None.
    """
    if start_bytes == 0:
        return observed is None
    if type(observed) is not float or not math.isfinite(observed):
        return False
    expected = growth_bytes / start_bytes * 100.0
    return math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)


def _memory_percentage_within_limit(
    growth_bytes: int, start_bytes: int, configured_limit: float
) -> bool:
    """Compare exact byte growth with the exact supplied float limit.

    Stored percentage fields are display evidence and may be checked with a
    small representation tolerance.  The verdict boundary is different: it
    cross-multiplies the byte ratio with the configured float's exact binary
    rational value, so no tolerance can turn a true breach into equality.
    """
    if start_bytes <= 0:
        raise ValueError("percentage thresholds require a positive baseline")
    limit_numerator, limit_denominator = configured_limit.as_integer_ratio()
    return (
        growth_bytes * 100 * limit_denominator
        <= start_bytes * limit_numerator
    )


def _valid_memory_executable_identity(path, basename) -> bool:
    """Validate immutable absolute Windows executable-path evidence."""
    if (
        type(path) is not str
        or not 1 <= len(path) <= MAX_MEMORY_EXECUTABLE_PATH_CHARACTERS
        or type(basename) is not str
        or basename != EXPECTED_MEMORY_EXECUTABLE_BASENAME
        or any(unicodedata.category(character) == "Cc" for character in path)
    ):
        return False
    normalized_separators = path.replace("/", "\\")
    drive, tail = ntpath.splitdrive(normalized_separators)
    if not drive or not tail.startswith("\\"):
        return False
    if drive.startswith("\\"):
        unc_parts = tuple(part for part in drive.split("\\") if part)
        if len(unc_parts) != 2:
            return False
        drive_parts = unc_parts
    else:
        if len(drive) != 2 or drive[1] != ":" or not drive[0].isalpha():
            return False
        drive_parts = ()
    invalid_component_characters = frozenset('<>:"|?*')
    parts = drive_parts + tuple(part for part in tail.split("\\") if part)
    if not parts:
        return False
    for part in parts:
        if (
            part in (".", "..")
            or part.endswith((" ", "."))
            or any(character in invalid_component_characters for character in part)
        ):
            return False
    return (
        normalized_separators.rsplit("\\", 1)[-1] == basename
        and ntpath.basename(ntpath.normpath(normalized_separators)) == basename
    )


def _valid_retained_memory_diagnostic(text) -> bool:
    """Accept only the monitor's bounded, normalized diagnostic form."""
    return (
        type(text) is str
        and 1 <= len(text) <= MAX_CLEANUP_ERROR_LENGTH
        and _bounded_text(text, MAX_CLEANUP_ERROR_LENGTH) == text
    )


def _recomputed_memory_slope(
    samples: tuple[ProcessMemorySample, ...], counter: str
) -> float | None:
    """Recompute the exact-regression slope when every sample is retained."""
    if (
        len(samples) < MINIMUM_MEMORY_VERDICT_SAMPLES
        or samples[-1].elapsed_seconds - samples[0].elapsed_seconds
        < MINIMUM_MEMORY_VERDICT_SPAN_SECONDS
    ):
        return None
    time_origin = Fraction(samples[0].elapsed_seconds)
    byte_origin = getattr(samples[0], f"{counter}_bytes")
    sum_time = Fraction(0)
    sum_time_sq = Fraction(0)
    sum_bytes = 0
    sum_time_bytes = Fraction(0)
    for sample in samples:
        time_delta = Fraction(sample.elapsed_seconds) - time_origin
        byte_delta = getattr(sample, f"{counter}_bytes") - byte_origin
        sum_time += time_delta
        sum_time_sq += time_delta * time_delta
        sum_bytes += byte_delta
        sum_time_bytes += time_delta * byte_delta
    count = len(samples)
    denominator = count * sum_time_sq - sum_time * sum_time
    if denominator <= 0:
        return None
    exact = (count * sum_time_bytes - sum_time * sum_bytes) / denominator * 60
    try:
        return float(exact)
    except OverflowError:
        return math.inf if exact > 0 else -math.inf


def _valid_stable_memory_metric(
    metric,
    counter: str,
    samples: tuple[ProcessMemorySample, ...],
    total_sample_count: int,
    omitted_sample_count: int,
    max_peak_growth_bytes: int,
    max_final_growth_bytes: int,
    max_final_growth_percent: float,
    max_slope_bytes_per_minute: float | None,
) -> bool:
    """Recompute one metric and every threshold outcome for STABLE."""
    if type(metric) is not MemoryMetricReport or metric.counter != counter:
        return False
    if type(metric.sample_count) is not int:
        return False
    if metric.sample_count != total_sample_count:
        return False
    if type(metric.span_seconds) is not float or not math.isfinite(
        metric.span_seconds
    ):
        return False
    retained_span = samples[-1].elapsed_seconds - samples[0].elapsed_seconds
    if (
        metric.span_seconds != retained_span
        or metric.span_seconds < MINIMUM_MEMORY_VERDICT_SPAN_SECONDS
    ):
        return False
    if (
        type(metric.peak_elapsed_seconds) is not float
        or not math.isfinite(metric.peak_elapsed_seconds)
        or not samples[0].elapsed_seconds
        <= metric.peak_elapsed_seconds
        <= samples[-1].elapsed_seconds
    ):
        return False
    integer_values = (
        metric.start_bytes,
        metric.peak_bytes,
        metric.final_bytes,
        metric.growth_bytes,
        metric.peak_growth_bytes,
    )
    if not all(type(value) is int for value in integer_values):
        return False
    if not all(
        0 <= value <= MAX_MEMORY_SAMPLE_BYTES
        for value in (
            metric.start_bytes,
            metric.peak_bytes,
            metric.final_bytes,
        )
    ):
        return False
    retained_values = tuple(
        getattr(sample, f"{counter}_bytes") for sample in samples
    )
    if (
        metric.start_bytes != retained_values[0]
        or metric.final_bytes != retained_values[-1]
        or metric.peak_bytes < max(retained_values)
        or metric.peak_bytes < metric.start_bytes
        or metric.peak_bytes < metric.final_bytes
    ):
        return False
    if omitted_sample_count == 0:
        observed_peak = max(retained_values)
        peak_index = retained_values.index(observed_peak)
        if (
            metric.peak_bytes != observed_peak
            or metric.peak_elapsed_seconds
            != samples[peak_index].elapsed_seconds
        ):
            return False
    growth_bytes = metric.final_bytes - metric.start_bytes
    peak_growth_bytes = metric.peak_bytes - metric.start_bytes
    if (
        metric.growth_bytes != growth_bytes
        or metric.peak_growth_bytes != peak_growth_bytes
    ):
        return False
    if not _memory_percentage_matches(
        metric.growth_percent, growth_bytes, metric.start_bytes
    ) or not _memory_percentage_matches(
        metric.peak_growth_percent, peak_growth_bytes, metric.start_bytes
    ):
        return False

    slope = metric.slope_bytes_per_minute
    if slope is not None and (
        type(slope) is not float or not math.isfinite(slope)
    ):
        return False
    if omitted_sample_count == 0:
        recomputed_slope = _recomputed_memory_slope(samples, counter)
        if slope != recomputed_slope:
            return False

    failed = []
    unevaluated = []
    # Byte outcomes use exact integer comparisons.
    if peak_growth_bytes > max_peak_growth_bytes:
        failed.append("max_peak_growth_bytes")
    if growth_bytes > max_final_growth_bytes:
        failed.append("max_final_growth_bytes")
    if metric.start_bytes == 0:
        unevaluated.append("max_final_growth_percent")
    elif not _memory_percentage_within_limit(
        growth_bytes, metric.start_bytes, max_final_growth_percent
    ):
        failed.append("max_final_growth_percent")
    if max_slope_bytes_per_minute is not None:
        if slope is None:
            unevaluated.append("max_slope_bytes_per_minute")
        elif slope > max_slope_bytes_per_minute:
            failed.append("max_slope_bytes_per_minute")
    expected_failed = tuple(failed)
    expected_unevaluated = tuple(unevaluated)
    expected_passed = not expected_failed and not expected_unevaluated
    return (
        type(metric.failed_thresholds) is tuple
        and metric.failed_thresholds == expected_failed
        and type(metric.unevaluated_thresholds) is tuple
        and metric.unevaluated_thresholds == expected_unevaluated
        and metric.passed is expected_passed
        and expected_passed
    )


def _stable_memory_report_coherent(memory) -> bool:
    """Reject a STABLE memory report whose internal evidence is incoherent.

    The live acceptance boundary never trusts ``verdict`` alone: monitoring
    must have produced full threshold configuration, preserved process
    identity, both mandatory metrics passing over at least five samples
    spanning at least sixty seconds, no mandatory sampling, close, or
    shutdown failure evidence, internally coherent retention counts, and
    well-formed advisory-only GPU evidence.
    """
    if not isinstance(memory, MemoryStabilityReport):
        return False
    if memory.verdict is not MemoryStabilityVerdict.STABLE:
        return False
    if memory.monitoring_enabled is not True:
        return False
    if memory.identity is not ProcessIdentityStatus.PRESERVED:
        return False
    if memory.mandatory_sampling_failure is not False:
        return False
    if memory.mandatory_shutdown_failure is not False:
        return False
    if isinstance(memory.pid, bool) or type(memory.pid) is not int:
        return False
    if not 1 <= memory.pid <= MAX_MEMORY_PID:
        return False
    if (
        type(memory.process_creation_time_100ns) is not int
        or not 0
        <= memory.process_creation_time_100ns
        <= MAX_PROCESS_CREATION_TIME_100NS
    ):
        return False
    if not _valid_memory_executable_identity(
        memory.executable_path, memory.executable_basename
    ):
        return False
    if not _valid_memory_thresholds(memory.thresholds):
        return False
    counts = (
        memory.total_sample_count,
        memory.retained_sample_count,
        memory.omitted_sample_count,
        memory.missed_interval_count,
        memory.omitted_diagnostic_count,
    )
    if not all(
        not isinstance(value, bool) and type(value) is int and value >= 0
        for value in counts
    ):
        return False
    if type(memory.samples) is not tuple:
        return False
    if memory.retained_sample_count != len(memory.samples):
        return False
    if memory.retained_sample_count > MAX_RETAINED_MEMORY_SAMPLES:
        return False
    if memory.omitted_sample_count != (
        memory.total_sample_count - memory.retained_sample_count
    ):
        return False
    if memory.total_sample_count <= MAX_RETAINED_MEMORY_SAMPLES:
        if memory.omitted_sample_count != 0:
            return False
    elif memory.retained_sample_count != MAX_RETAINED_MEMORY_SAMPLES:
        return False
    if memory.total_sample_count < MINIMUM_MEMORY_VERDICT_SAMPLES:
        return False
    previous_elapsed = None
    for sample in memory.samples:
        if not isinstance(sample, ProcessMemorySample):
            return False
        elapsed = sample.elapsed_seconds
        if (
            type(elapsed) is not float
            or not math.isfinite(elapsed)
            or elapsed < 0.0
        ):
            return False
        if previous_elapsed is not None and elapsed <= previous_elapsed:
            return False
        previous_elapsed = elapsed
        for value in (sample.working_set_bytes, sample.private_commit_bytes):
            if (
                type(value) is not int
                or not 0 <= value <= MAX_MEMORY_SAMPLE_BYTES
            ):
                return False
        gpu_value = sample.gpu_dedicated_bytes
        if gpu_value is not None and (
            isinstance(gpu_value, bool)
            or type(gpu_value) is not int
            or gpu_value < 0
        ):
            return False
    if not memory.samples or memory.samples[0].elapsed_seconds != 0.0:
        return False
    limits = memory.thresholds
    if not _valid_stable_memory_metric(
        memory.working_set,
        "working_set",
        memory.samples,
        memory.total_sample_count,
        memory.omitted_sample_count,
        limits.working_set_max_peak_growth_bytes,
        limits.working_set_max_final_growth_bytes,
        limits.working_set_max_final_growth_percent,
        limits.working_set_max_slope_bytes_per_minute,
    ):
        return False
    if not _valid_stable_memory_metric(
        memory.private_commit,
        "private_commit",
        memory.samples,
        memory.total_sample_count,
        memory.omitted_sample_count,
        limits.private_commit_max_peak_growth_bytes,
        limits.private_commit_max_final_growth_bytes,
        limits.private_commit_max_final_growth_percent,
        limits.private_commit_max_slope_bytes_per_minute,
    ):
        return False
    # GPU evidence stays advisory: any availability state is acceptable,
    # but present values must be well-formed.
    if not isinstance(memory.gpu_status, MemoryCounterAvailability):
        return False
    for value in (
        memory.gpu_final_dedicated_bytes,
        memory.gpu_peak_dedicated_bytes,
    ):
        if value is not None and (
            isinstance(value, bool) or type(value) is not int or value < 0
        ):
            return False
    if memory.gpu_final_dedicated_bytes is not None and (
        memory.gpu_peak_dedicated_bytes is None
        or memory.gpu_peak_dedicated_bytes < memory.gpu_final_dedicated_bytes
    ):
        return False
    if type(memory.diagnostics) is not tuple:
        return False
    retained_diagnostic_count = len(memory.diagnostics)
    if (
        retained_diagnostic_count > MAX_RETAINED_MEMORY_DIAGNOSTICS
        or not all(
            _valid_retained_memory_diagnostic(text)
            for text in memory.diagnostics
        )
        or (
            memory.omitted_diagnostic_count > 0
            and retained_diagnostic_count != MAX_RETAINED_MEMORY_DIAGNOSTICS
        )
    ):
        return False
    return True


def _repeated_sensor_suite_validation_report_succeeded(
    report: RepeatedSensorSuiteValidationReport,
) -> bool:
    """Validate aggregate invariants at the live unittest boundary."""
    if not isinstance(report, RepeatedSensorSuiteValidationReport):
        return False
    counts = (
        report.attempted_cycle_count,
        report.completed_cycle_count,
        report.passed_cycle_count,
        report.omitted_cycle_summary_count,
        report.omitted_cleanup_error_count,
    )
    if not all(type(value) is int for value in counts):
        return False
    if any(value < 0 for value in counts):
        return False
    if not isinstance(report.memory, MemoryStabilityReport):
        return False
    if report.memory.verdict is MemoryStabilityVerdict.STABLE:
        # Never trust the verdict alone: a malformed internally constructed
        # STABLE report is rejected at this acceptance boundary.
        if not _stable_memory_report_coherent(report.memory):
            return False
    elif report.memory.verdict is MemoryStabilityVerdict.NOT_REQUESTED:
        if (
            report.memory.monitoring_enabled is not False
            or report.memory.identity is not ProcessIdentityStatus.NOT_TRACKED
            or report.memory.mandatory_sampling_failure
            or report.memory.mandatory_shutdown_failure
            or report.memory.pid is not None
            or report.memory.process_creation_time_100ns is not None
            or report.memory.executable_path is not None
            or report.memory.executable_basename is not None
            or report.memory.total_sample_count != 0
            or report.memory.retained_sample_count != 0
            or report.memory.omitted_sample_count != 0
            or report.memory.samples != ()
            or report.memory.working_set is not None
            or report.memory.private_commit is not None
            or report.memory.thresholds is not None
        ):
            return False
    else:
        return False
    if not isinstance(report.execution_mode, ExecutionMode) or not isinstance(
        report.termination_reason, TerminationReason
    ):
        return False
    if (
        type(report.cycles) is not tuple
        or len(report.cycles) > MAX_RETAINED_CYCLE_SUMMARIES
    ):
        return False
    for value in (report.requested_soak_seconds, report.elapsed_seconds):
        if type(value) is not float:
            return False
        if not math.isfinite(float(value)) or value < 0:
            return False
    if report.omitted_cycle_summary_count != (
        report.completed_cycle_count - len(report.cycles)
    ):
        return False
    if report.completed_cycle_count != report.passed_cycle_count:
        return False

    if not all(
        isinstance(summary, RepeatedCycleSummary) for summary in report.cycles
    ):
        return False
    if report.completed_cycle_count <= MAX_RETAINED_CYCLE_SUMMARIES:
        expected_indexes = tuple(range(1, report.completed_cycle_count + 1))
    else:
        expected_indexes = tuple(range(1, FIRST_RETAINED_CYCLE_SUMMARIES + 1)) + tuple(
            range(
                report.completed_cycle_count - FINAL_RETAINED_CYCLE_SUMMARIES + 1,
                report.completed_cycle_count + 1,
            )
        )
    if tuple(summary.cycle_index for summary in report.cycles) != expected_indexes:
        return False

    for summary in report.cycles:
        if (
            type(summary.cycle_index) is not int
            or summary.success is not True
            or isinstance(summary.elapsed_seconds, bool)
            or not isinstance(summary.elapsed_seconds, Real)
            or not math.isfinite(float(summary.elapsed_seconds))
            or summary.elapsed_seconds < 0
            or not _valid_owned_actor_id(summary.ego_id)
            or type(summary.sensor_actor_ids) is not tuple
            or len(summary.sensor_actor_ids) != len(EXPECTED_SENSOR_NAMES)
            or len(set(summary.sensor_actor_ids)) != len(EXPECTED_SENSOR_NAMES)
            or not all(
                _valid_owned_actor_id(actor_id)
                and actor_id != summary.ego_id
                for actor_id in summary.sensor_actor_ids
            )
            or summary.readiness_success is not True
            or summary.primary_error is not None
            or type(summary.cleanup_errors) is not tuple
            or summary.cleanup_errors
            or summary.disappearance is not CleanupVerification.CONFIRMED
            or type(summary.remaining_actor_ids) is not tuple
            or summary.remaining_actor_ids
        ):
            return False

    common_success = (
        report.success is True
        and report.completed_cycle_count >= 1
        and report.attempted_cycle_count == report.completed_cycle_count
        and report.first_primary_error is None
        and report.failed_cycle_index is None
        and type(report.supplemental_cleanup_errors) is tuple
        and not report.supplemental_cleanup_errors
        and report.omitted_cleanup_error_count == 0
        and report.final_disappearance is CleanupVerification.CONFIRMED
        and type(report.remaining_owned_actors) is tuple
        and not report.remaining_owned_actors
    )
    if not common_success:
        return False
    if report.execution_mode is ExecutionMode.CYCLE_COUNT:
        return (
            type(report.requested_cycle_count) is int
            and 1 <= report.requested_cycle_count <= 20
            and report.requested_soak_seconds == 0.0
            and report.attempted_cycle_count == report.requested_cycle_count
            and report.termination_reason is TerminationReason.CYCLE_LIMIT_REACHED
            # Memory monitoring exists only around timed soak, so any other
            # verdict in cycle-count mode is internally inconsistent.
            and report.memory.verdict is MemoryStabilityVerdict.NOT_REQUESTED
        )
    return (
        report.requested_cycle_count is None
        and 120.0 <= report.requested_soak_seconds <= 3600.0
        and report.elapsed_seconds >= report.requested_soak_seconds
        and report.termination_reason is TerminationReason.SOAK_DURATION_REACHED
    )


# -- Mission 4G.6: bounded monitor around the timed-soak outer runner --


class _MemoryCounterAccumulator:
    """Constant-space exact online metrics over every valid sample.

    Slope accumulates OLS sums over exact centered deltas: integer byte
    deltas against the first sample's integer byte origin, and exact
    binary-float time offsets (as Fractions) against the first timestamp.
    Large absolute counters therefore never erase small growth — values far
    beyond 2**53 keep byte-exact slopes — and no unbounded sample history is
    materialized; state size is bounded by value magnitudes, never by the
    number of samples.
    """

    __slots__ = (
        "_byte_origin",
        "_sum_bytes",
        "_sum_time",
        "_sum_time_bytes",
        "_sum_time_sq",
        "_time_origin",
        "count",
        "final_bytes",
        "final_elapsed_seconds",
        "first_elapsed_seconds",
        "peak_bytes",
        "peak_elapsed_seconds",
        "start_bytes",
    )

    def __init__(self) -> None:
        self.count = 0
        self.start_bytes = 0
        self.peak_bytes = 0
        self.peak_elapsed_seconds = 0.0
        self.final_bytes = 0
        self.first_elapsed_seconds = 0.0
        self.final_elapsed_seconds = 0.0
        self._byte_origin = 0
        self._time_origin = Fraction(0)
        self._sum_time = Fraction(0)
        self._sum_time_sq = Fraction(0)
        self._sum_bytes = 0
        self._sum_time_bytes = Fraction(0)

    def add(self, elapsed_seconds: float, value_bytes: int) -> None:
        if self.count == 0:
            self.start_bytes = value_bytes
            self.peak_bytes = value_bytes
            self.peak_elapsed_seconds = elapsed_seconds
            self.first_elapsed_seconds = elapsed_seconds
            self._byte_origin = value_bytes
            self._time_origin = Fraction(elapsed_seconds)
        elif value_bytes > self.peak_bytes:
            # A strictly-greater update keeps the earliest peak timestamp.
            self.peak_bytes = value_bytes
            self.peak_elapsed_seconds = elapsed_seconds
        self.final_bytes = value_bytes
        self.final_elapsed_seconds = elapsed_seconds
        self.count += 1
        time_delta = Fraction(elapsed_seconds) - self._time_origin
        byte_delta = value_bytes - self._byte_origin
        self._sum_time += time_delta
        self._sum_time_sq += time_delta * time_delta
        self._sum_bytes += byte_delta
        self._sum_time_bytes += time_delta * byte_delta

    @property
    def span_seconds(self) -> float:
        if self.count == 0:
            return 0.0
        return self.final_elapsed_seconds - self.first_elapsed_seconds

    def slope_bytes_per_minute(self) -> float | None:
        if (
            self.count < MINIMUM_MEMORY_VERDICT_SAMPLES
            or self.span_seconds < MINIMUM_MEMORY_VERDICT_SPAN_SECONDS
        ):
            return None
        denominator = (
            self.count * self._sum_time_sq - self._sum_time * self._sum_time
        )
        if denominator <= 0:
            # Only duplicated timestamps carry no slope information.
            return None
        numerator = (
            self.count * self._sum_time_bytes - self._sum_time * self._sum_bytes
        )
        exact = numerator / denominator * 60
        try:
            return float(exact)
        except OverflowError:
            return math.inf if exact > 0 else -math.inf


class _BoundedMemorySampleRetention:
    """Retain the first 16 samples plus the rolling final 240 samples."""

    __slots__ = ("_final", "_first", "total_count")

    def __init__(self) -> None:
        self._first: list[ProcessMemorySample] = []
        self._final: deque[ProcessMemorySample] = deque(
            maxlen=FINAL_RETAINED_MEMORY_SAMPLES
        )
        self.total_count = 0

    def add(self, sample: ProcessMemorySample) -> None:
        self.total_count += 1
        if len(self._first) < FIRST_RETAINED_MEMORY_SAMPLES:
            self._first.append(sample)
        else:
            self._final.append(sample)

    @property
    def last_elapsed_seconds(self) -> float | None:
        if self._final:
            return self._final[-1].elapsed_seconds
        if self._first:
            return self._first[-1].elapsed_seconds
        return None

    def snapshot(self) -> tuple[ProcessMemorySample, ...]:
        return tuple(self._first) + tuple(self._final)


def _memory_metric_report(
    counter: str,
    accumulator: _MemoryCounterAccumulator,
    max_peak_growth_bytes: int,
    max_final_growth_bytes: int,
    max_final_growth_percent: float,
    max_slope_bytes_per_minute: float | None,
) -> MemoryMetricReport | None:
    if accumulator.count == 0:
        return None
    growth_bytes = accumulator.final_bytes - accumulator.start_bytes
    peak_growth_bytes = accumulator.peak_bytes - accumulator.start_bytes
    if accumulator.start_bytes > 0:
        growth_percent = growth_bytes / accumulator.start_bytes * 100.0
        peak_growth_percent = (
            peak_growth_bytes / accumulator.start_bytes * 100.0
        )
    else:
        growth_percent = None
        peak_growth_percent = None
    slope = accumulator.slope_bytes_per_minute()

    failed: list[str] = []
    unevaluated: list[str] = []
    if peak_growth_bytes > max_peak_growth_bytes:
        failed.append("max_peak_growth_bytes")
    if growth_bytes > max_final_growth_bytes:
        failed.append("max_final_growth_bytes")
    if growth_percent is None:
        unevaluated.append("max_final_growth_percent")
    elif not _memory_percentage_within_limit(
        growth_bytes, accumulator.start_bytes, max_final_growth_percent
    ):
        failed.append("max_final_growth_percent")
    if max_slope_bytes_per_minute is not None:
        if slope is None:
            unevaluated.append("max_slope_bytes_per_minute")
        elif slope > max_slope_bytes_per_minute:
            failed.append("max_slope_bytes_per_minute")

    return MemoryMetricReport(
        counter=counter,
        sample_count=accumulator.count,
        span_seconds=accumulator.span_seconds,
        start_bytes=accumulator.start_bytes,
        peak_bytes=accumulator.peak_bytes,
        peak_elapsed_seconds=accumulator.peak_elapsed_seconds,
        final_bytes=accumulator.final_bytes,
        growth_bytes=growth_bytes,
        peak_growth_bytes=peak_growth_bytes,
        growth_percent=growth_percent,
        peak_growth_percent=peak_growth_percent,
        slope_bytes_per_minute=slope,
        failed_thresholds=tuple(failed),
        unevaluated_thresholds=tuple(unevaluated),
        passed=not failed and not unevaluated,
    )


def _advance_cadence_deadline(
    previous_deadline: float, now: float, interval: float
) -> tuple[float, int]:
    """Choose the next strictly-future absolute deadline in constant time.

    ``missed`` counts cadence instants after ``previous_deadline`` that have
    been reached by ``now``; exact equality is therefore counted.  The count
    comes from the exact rational values of the three input floats using a
    fixed number of integer operations.  Only conversion of the conceptual
    next deadline back to float may round; if it rounds down to ``now``, one
    ``nextafter`` supplies the least greater representable float.
    """
    if not math.isfinite(previous_deadline) or not math.isfinite(now):
        raise ValueError("cadence advancement requires finite clock values")
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError("cadence interval must be positive and finite")

    def representable_fallback() -> float:
        fallback = math.nextafter(now, math.inf)
        if not math.isfinite(fallback):
            raise OverflowError(
                "no finite cadence deadline is representable after now"
            )
        return fallback

    previous_numerator, previous_denominator = (
        previous_deadline.as_integer_ratio()
    )
    now_numerator, now_denominator = now.as_integer_ratio()
    interval_numerator, interval_denominator = interval.as_integer_ratio()

    delta_numerator = (
        now_numerator * previous_denominator
        - previous_numerator * now_denominator
    )
    delta_denominator = now_denominator * previous_denominator
    if delta_numerator <= 0:
        missed = 0
    else:
        quotient_numerator = delta_numerator * interval_denominator
        quotient_denominator = delta_denominator * interval_numerator
        missed = quotient_numerator // quotient_denominator

    next_step = missed + 1
    candidate_numerator = (
        previous_numerator * interval_denominator
        + next_step * interval_numerator * previous_denominator
    )
    candidate_denominator = previous_denominator * interval_denominator
    try:
        candidate = candidate_numerator / candidate_denominator
    except OverflowError:
        candidate = math.inf
    if math.isfinite(candidate) and candidate > now:
        return candidate, missed
    return representable_fallback(), missed


def _default_memory_thread_factory(target: Callable[[], None]):
    return threading.Thread(
        target=target, name="memory-stability-monitor", daemon=True
    )


class MemoryStabilityMonitor:
    """Bounded 5-second-cadence process-memory monitor for one soak run.

    The monitor samples on absolute monotonic deadlines through an
    interruptible event wait, records bounded evidence under one lock, and
    never retries a sampling operation, reacquires another PID, or bursts
    late samples.  It exists only around the timed-soak outer runner; the
    single-cycle runner remains memory-unaware.
    """

    __slots__ = (
        "_baseline_time",
        "_closed",
        "_diagnostics",
        "_executable_basename",
        "_executable_path",
        "_final_taken",
        "_gpu",
        "_gpu_close_finalized",
        "_gpu_final_bytes",
        "_gpu_native_abandoned",
        "_gpu_peak_bytes",
        "_gpu_status",
        "_join_timed_out",
        "_join_timeout_seconds",
        "_lock",
        "_mandatory_close_failure",
        "_missed_intervals",
        "_monotonic",
        "_native_call_executor",
        "_native_call_timeout_seconds",
        "_omitted_diagnostics",
        "_pagefile_alias_noted",
        "_peak_breached",
        "_pid",
        "_private_commit",
        "_process_creation_time_100ns",
        "_process_close_finalized",
        "_process_exited",
        "_process_native_abandoned",
        "_retention",
        "_sampler",
        "_sampling_failed",
        "_shutdown_failed",
        "_stop_event",
        "_thread",
        "_thread_factory",
        "_thread_started",
        "_thresholds",
        "_working_set",
    )

    def __init__(
        self,
        *,
        sampler,
        thresholds: MemoryThresholds,
        gpu_sampler=None,
        pid: int | None = None,
        process_creation_time_100ns: int | None = None,
        executable_path: str | None = None,
        executable_basename: str | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        stop_event=None,
        thread_factory=None,
        native_call_executor=None,
        native_call_timeout_seconds: float = MEMORY_NATIVE_CALL_TIMEOUT_SECONDS,
        join_timeout_seconds: float = MEMORY_MONITOR_JOIN_TIMEOUT_SECONDS,
        gpu_startup_status: MemoryCounterAvailability = (
            MemoryCounterAvailability.NOT_SAMPLED
        ),
        startup_diagnostics: tuple[str, ...] = (),
        omitted_startup_diagnostic_count: int = 0,
    ) -> None:
        if not isinstance(thresholds, MemoryThresholds):
            raise TypeError("thresholds must be MemoryThresholds")
        if not callable(monotonic):
            raise TypeError("monotonic must be callable")
        if native_call_executor is not None and not callable(
            native_call_executor
        ):
            raise TypeError("native_call_executor must be callable")
        if not isinstance(gpu_startup_status, MemoryCounterAvailability):
            raise TypeError(
                "gpu_startup_status must be MemoryCounterAvailability"
            )
        if type(startup_diagnostics) is not tuple:
            raise TypeError("startup_diagnostics must be a tuple")
        if (
            type(omitted_startup_diagnostic_count) is not int
            or omitted_startup_diagnostic_count < 0
        ):
            raise ValueError(
                "omitted_startup_diagnostic_count must be a non-negative integer"
            )
        for name, value in (
            ("native_call_timeout_seconds", native_call_timeout_seconds),
            ("join_timeout_seconds", join_timeout_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be a positive finite number")
        self._sampler = sampler
        self._gpu = gpu_sampler
        self._thresholds = thresholds
        self._monotonic = monotonic
        self._stop_event = (
            stop_event if stop_event is not None else threading.Event()
        )
        self._thread_factory = (
            thread_factory
            if thread_factory is not None
            else _default_memory_thread_factory
        )
        self._native_call_executor = (
            native_call_executor
            if native_call_executor is not None
            else _run_native_call_bounded
        )
        self._native_call_timeout_seconds = float(native_call_timeout_seconds)
        self._join_timeout_seconds = float(join_timeout_seconds)
        self._pid = (
            pid
            if type(pid) is int and not isinstance(pid, bool) and pid > 0
            else None
        )
        self._process_creation_time_100ns = (
            process_creation_time_100ns
            if type(process_creation_time_100ns) is int
            and not isinstance(process_creation_time_100ns, bool)
            and 0
            <= process_creation_time_100ns
            <= MAX_PROCESS_CREATION_TIME_100NS
            else None
        )
        self._executable_path = (
            executable_path
            if isinstance(executable_path, str)
            and 1
            <= len(executable_path)
            <= MAX_MEMORY_EXECUTABLE_PATH_CHARACTERS
            else None
        )
        self._executable_basename = (
            executable_basename
            if isinstance(executable_basename, str)
            and 1 <= len(executable_basename) <= 260
            else None
        )
        self._lock = threading.Lock()
        self._thread = None
        self._thread_started = False
        self._join_timed_out = False
        self._process_native_abandoned = False
        self._gpu_native_abandoned = False
        self._process_close_finalized = False
        self._gpu_close_finalized = False
        self._mandatory_close_failure = False
        self._shutdown_failed = False
        self._baseline_time: float | None = None
        self._working_set = _MemoryCounterAccumulator()
        self._private_commit = _MemoryCounterAccumulator()
        self._retention = _BoundedMemorySampleRetention()
        self._missed_intervals = 0
        self._diagnostics: list[str] = []
        self._omitted_diagnostics = omitted_startup_diagnostic_count
        for diagnostic in startup_diagnostics:
            self._note_diagnostic_locked(diagnostic)
        self._process_exited = False
        self._sampling_failed = False
        self._peak_breached = False
        self._pagefile_alias_noted = False
        self._gpu_status = gpu_startup_status
        self._gpu_final_bytes: int | None = None
        self._gpu_peak_bytes: int | None = None
        self._final_taken = False
        self._closed = False

    def _read_clock(self) -> float:
        return _repeated_clock_value(self._monotonic)

    def _note_diagnostic_locked(self, text: str) -> None:
        if len(self._diagnostics) < MAX_RETAINED_MEMORY_DIAGNOSTICS:
            self._diagnostics.append(
                _bounded_text(text, MAX_CLEANUP_ERROR_LENGTH)
            )
        else:
            self._omitted_diagnostics += 1

    def _record_sampling_failure(self, context: str, exc: BaseException) -> None:
        text = (
            f"{context}: {type(exc).__name__}: "
            f"{_bounded_text(exc, MAX_CLEANUP_ERROR_LENGTH)}"
        )
        with self._lock:
            if isinstance(exc, MemoryMonitoredProcessExited):
                self._process_exited = True
            else:
                self._sampling_failed = True
            self._note_diagnostic_locked(text)

    def _native_resource_abandoned(self, resource: str) -> bool:
        with self._lock:
            if resource == "process":
                return self._process_native_abandoned
            return self._gpu_native_abandoned

    def _mark_native_abandoned(self, resource: str, description: str) -> None:
        with self._lock:
            if resource == "process":
                self._process_native_abandoned = True
            else:
                self._gpu_native_abandoned = True
            self._note_diagnostic_locked(
                f"{description}: {resource} native call abandoned; the "
                f"{resource} handle is no longer safe to close"
            )

    def _bounded_native(self, resource: str, description: str, operation):
        """Run one native operation with a deterministic bounded wait.

        A timed-out or interrupted wait means the native call may still be
        in flight on its abandoned worker, so the resource is permanently
        marked unsafe: no further native call runs against it and its handle
        is never closed from another execution context.  The failed call is
        never retried and no replacement PID is ever acquired.
        """
        if self._native_resource_abandoned(resource):
            raise MemoryNativeCallTimeout(
                f"{description} refused: an earlier {resource} native call "
                "never completed"
            )
        try:
            return self._native_call_executor(
                description, operation, self._native_call_timeout_seconds
            )
        except MemoryNativeCallTimeout:
            self._mark_native_abandoned(resource, description)
            raise
        except Exception:
            # The native call completed and reported a failure; the handle
            # itself is still safely owned.
            raise
        except BaseException:
            # The bounded wait itself was interrupted (never masked); the
            # in-flight state is unknown, so abandon conservatively.
            self._mark_native_abandoned(resource, description)
            raise

    def _observe_gpu(self) -> int | None:
        if self._gpu is None:
            return None
        try:
            reading = self._bounded_native(
                "gpu", "gpu sampling", self._gpu.sample
            )
            if self._native_resource_abandoned("gpu"):
                # A concurrent join timeout can abandon the resource while
                # this worker is returning.  Discard the late result before it
                # can mutate retained evidence.
                return None
            status = reading.status
            value = reading.dedicated_bytes
            if not isinstance(status, MemoryCounterAvailability):
                raise TypeError(
                    "gpu status must be MemoryCounterAvailability"
                )
        except Exception:
            # Preserve previously valid GPU evidence; never fabricate zeros.
            with self._lock:
                self._gpu_status = MemoryCounterAvailability.UNRELIABLE
            return None
        observed: int | None = None
        if (
            status is MemoryCounterAvailability.AVAILABLE
            and not isinstance(value, bool)
            and type(value) is int
            and value >= 0
        ):
            observed = value
        elif (
            status is MemoryCounterAvailability.ZERO_OR_UNIFIED_MEMORY
            and value == 0
        ):
            observed = 0
        with self._lock:
            self._gpu_status = status
            if observed is not None:
                self._gpu_final_bytes = observed
                if (
                    self._gpu_peak_bytes is None
                    or observed > self._gpu_peak_bytes
                ):
                    self._gpu_peak_bytes = observed
        return observed

    def _take_sample(self, context: str, elapsed_seconds: float) -> bool:
        try:
            reading = self._bounded_native(
                "process", context, self._sampler.sample
            )
        except Exception as exc:
            self._record_sampling_failure(context, exc)
            return False
        if self._native_resource_abandoned("process"):
            # A join timeout may have abandoned this handle while the bounded
            # call was finishing.  Late completion never mutates monitor state.
            return False
        working_set = getattr(reading, "working_set_bytes", None)
        private_commit = getattr(reading, "private_commit_bytes", None)
        pagefile = getattr(reading, "pagefile_usage_bytes", None)
        if (
            isinstance(working_set, bool)
            or type(working_set) is not int
            or working_set <= 0
            or isinstance(private_commit, bool)
            or type(private_commit) is not int
            or private_commit <= 0
        ):
            self._record_sampling_failure(
                context,
                ValueError(
                    "active process counters must be positive integers"
                ),
            )
            return False
        if (
            not isinstance(pagefile, bool)
            and type(pagefile) is int
            and pagefile != private_commit
        ):
            with self._lock:
                if not self._pagefile_alias_noted:
                    self._pagefile_alias_noted = True
                    self._note_diagnostic_locked(
                        f"PagefileUsage {pagefile} is inconsistent with "
                        f"PrivateUsage {private_commit}"
                    )
        gpu_bytes = self._observe_gpu()
        with self._lock:
            self._retention.add(
                ProcessMemorySample(
                    elapsed_seconds=elapsed_seconds,
                    working_set_bytes=working_set,
                    private_commit_bytes=private_commit,
                    gpu_dedicated_bytes=gpu_bytes,
                )
            )
            self._working_set.add(elapsed_seconds, working_set)
            self._private_commit.add(elapsed_seconds, private_commit)
            limits = self._thresholds
            if (
                self._working_set.peak_bytes - self._working_set.start_bytes
                > limits.working_set_max_peak_growth_bytes
                or self._private_commit.peak_bytes
                - self._private_commit.start_bytes
                > limits.private_commit_max_peak_growth_bytes
            ):
                # A peak breach can never shrink again, so it irreversibly
                # blocks admission of the next cycle.
                self._peak_breached = True
        return True

    def _elapsed_at(self, now: float) -> float:
        elapsed = now - self._baseline_time
        with self._lock:
            last = self._retention.last_elapsed_seconds
        if last is not None and elapsed < last:
            return last
        return max(0.0, elapsed)

    def take_baseline(self) -> None:
        """Take the one baseline sample before the first cycle admission."""
        if self._baseline_time is not None:
            raise RuntimeError("baseline sample was already taken")
        self._baseline_time = self._read_clock()
        if not self._take_sample("baseline sampling", 0.0):
            raise RuntimeError("memory baseline sampling failed")

    def start(self) -> None:
        """Construct, then start, the sampling thread as separate stages."""
        if self._baseline_time is None:
            raise RuntimeError("take_baseline must run before start")
        if self._thread is not None:
            raise RuntimeError("monitor was already started")
        try:
            thread = self._thread_factory(self._run_sampling_loop)
        except Exception as exc:
            self._record_sampling_failure("monitor thread construction", exc)
            raise
        # The constructed-but-unstarted thread is tracked separately so that
        # cleanup never joins a thread whose start() failed or never ran.
        self._thread = thread
        try:
            thread.start()
        except Exception as exc:
            self._record_sampling_failure("monitor thread start", exc)
            raise
        self._thread_started = True

    def _run_sampling_loop(self) -> None:
        try:
            interval = MEMORY_SAMPLING_INTERVAL_SECONDS
            deadline = self._baseline_time + interval
            while True:
                if self._stop_event.is_set():
                    return
                now = self._read_clock()
                if self._stop_event.is_set():
                    return
                remaining = deadline - now
                if remaining > 0.0:
                    if self._stop_event.wait(remaining):
                        return
                    if self._stop_event.is_set():
                        return
                    now = self._read_clock()
                    if self._stop_event.is_set():
                        # Stop arrived during the wake (including a very
                        # late one): exit without taking another sample.
                        return
                deadline, missed = _advance_cadence_deadline(
                    deadline, now, interval
                )
                if self._stop_event.is_set():
                    return
                elapsed = self._elapsed_at(now)
                # This check is intentionally adjacent to the native sample
                # boundary; every earlier observation can become stale.
                if self._stop_event.is_set():
                    return
                # When late, take exactly one sample and move to the already
                # computed strictly-future deadline; never burst or backfill.
                if not self._take_sample("cadence sampling", elapsed):
                    return
                if missed:
                    with self._lock:
                        self._missed_intervals += missed
        except BaseException as exc:
            self._record_sampling_failure("sampling loop", exc)

    def request_stop(self) -> None:
        """Signal the sampling loop to stop; a failure is recorded evidence."""
        try:
            self._stop_event.set()
        except BaseException as exc:
            with self._lock:
                self._shutdown_failed = True
                self._note_diagnostic_locked(
                    "stop request failed: "
                    f"{type(exc).__name__}: "
                    f"{_bounded_text(exc, MAX_CLEANUP_ERROR_LENGTH)}"
                )
            if not isinstance(exc, Exception):
                raise

    def join(self) -> None:
        """Join the started sampling thread within the configured bound.

        A thread that was never successfully started is never joined.  A
        join that does not complete within the bound permanently abandons
        both native resources (the loop thread may still be using them) and
        records a mandatory shutdown failure; it is not retried.
        """
        thread = self._thread
        if thread is None or not self._thread_started or self._join_timed_out:
            return
        try:
            thread.join(self._join_timeout_seconds)
            still_alive = bool(thread.is_alive())
        except BaseException as exc:
            self._join_timed_out = True
            with self._lock:
                self._shutdown_failed = True
                self._process_native_abandoned = True
                self._gpu_native_abandoned = True
                self._note_diagnostic_locked(
                    "monitor thread join failed: "
                    f"{type(exc).__name__}: "
                    f"{_bounded_text(exc, MAX_CLEANUP_ERROR_LENGTH)}"
                )
            if not isinstance(exc, Exception):
                raise
            return
        if still_alive:
            self._join_timed_out = True
            with self._lock:
                self._shutdown_failed = True
                self._process_native_abandoned = True
                self._gpu_native_abandoned = True
                self._note_diagnostic_locked(
                    "monitor thread join timed out after "
                    f"{self._join_timeout_seconds:.3f} seconds; sampler "
                    "handles are no longer safe to close"
                )

    def take_final_sample(self) -> None:
        """Take one final sample, coalescing an identical timestamp.

        Final sampling is intentionally impossible before a baseline, after a
        prior final-stage attempt, after resource finalization, after process
        exit, after a native-call or join abandonment, or while a started
        worker is still alive.  Stop-stage and ordinary earlier sampling
        failures do not themselves make the retained process handle unsafe.
        """
        if self._baseline_time is None or self._final_taken:
            return
        self._final_taken = True
        with self._lock:
            blocked = (
                self._process_exited
                or self._process_native_abandoned
                or self._process_close_finalized
                or self._closed
            )
        if blocked:
            return
        thread = self._thread
        if thread is not None and self._thread_started:
            try:
                worker_alive = bool(thread.is_alive())
            except BaseException as exc:
                with self._lock:
                    self._shutdown_failed = True
                    self._process_native_abandoned = True
                    self._gpu_native_abandoned = True
                    self._note_diagnostic_locked(
                        "final sampling could not verify worker shutdown: "
                        f"{type(exc).__name__}"
                    )
                if not isinstance(exc, Exception):
                    raise
                return
            if worker_alive:
                with self._lock:
                    self._shutdown_failed = True
                    self._process_native_abandoned = True
                    self._gpu_native_abandoned = True
                    self._note_diagnostic_locked(
                        "final sampling skipped while the monitor worker "
                        "remained alive"
                    )
                return
        try:
            now = self._read_clock()
        except BaseException as exc:
            self._record_sampling_failure("final sampling", exc)
            if not isinstance(exc, Exception):
                raise
            return
        elapsed = self._elapsed_at(now)
        with self._lock:
            last = self._retention.last_elapsed_seconds
        if last is not None and elapsed == last:
            # Coalesced: the previous sample already stands as the final
            # observation for this exact timestamp.
            return
        try:
            self._take_sample("final sampling", elapsed)
        except BaseException as exc:
            self._record_sampling_failure("final sampling", exc)
            raise

    def _record_close_failure(
        self, resource: str, mandatory: bool, text: str
    ) -> None:
        with self._lock:
            if mandatory:
                self._mandatory_close_failure = True
            self._note_diagnostic_locked(text)

    def _resource_close_finalized(self, resource: str) -> bool:
        if resource == "process":
            return self._process_close_finalized
        return self._gpu_close_finalized

    def _finalize_resource_close(self, resource: str) -> None:
        if resource == "process":
            self._process_close_finalized = True
        else:
            self._gpu_close_finalized = True

    def _close_resource(
        self, resource: str, target, *, mandatory: bool
    ) -> BaseException | None:
        """Attempt one resource stage and defer any BaseException."""
        if self._resource_close_finalized(resource):
            return None
        deferred = None
        try:
            if target is None:
                return None
            closer = getattr(target, "close", None)
            if not callable(closer):
                return None
            if self._native_resource_abandoned(resource):
                self._record_close_failure(
                    resource,
                    mandatory,
                    f"{resource} sampler close refused: a native call may "
                    "still be in flight",
                )
                return None
            self._bounded_native(
                resource, f"{resource} sampler close", closer
            )
        except Exception as exc:
            self._record_close_failure(
                resource,
                mandatory,
                f"{resource} sampler close failed: "
                f"{type(exc).__name__}: "
                f"{_bounded_text(exc, MAX_CLEANUP_ERROR_LENGTH)}",
            )
        except BaseException as exc:
            self._record_close_failure(
                resource,
                mandatory,
                f"{resource} sampler close interrupted: {type(exc).__name__}",
            )
            deferred = exc
        finally:
            # A completed, failed, interrupted, missing, or deliberately
            # abandoned close stage is final.  Native cleanup is never
            # retried, and later monitor operations cannot reuse it.
            self._finalize_resource_close(resource)
        return deferred

    def close(self) -> None:
        """Release sampler-owned native resources deterministically.

        Every resource is closed at most once, through the bounded native
        executor, and never while an execution context may still be using
        it.  A process-sampler close failure or refusal is mandatory
        evidence that prevents a STABLE verdict; a GPU close failure stays
        advisory.  Each resource's closure proceeds independently of the
        other's outcome.
        """
        if self._closed:
            return
        deferred: list[BaseException] = []
        if self._thread is not None and self._thread_started:
            try:
                loop_alive = bool(self._thread.is_alive())
            except BaseException as exc:
                loop_alive = True
                deferred.append(exc)
                with self._lock:
                    self._shutdown_failed = True
                    self._note_diagnostic_locked(
                        "sampling thread state inspection interrupted: "
                        f"{type(exc).__name__}"
                    )
            if loop_alive and not self._join_timed_out:
                # The sampling thread may still be mid-call; treat both
                # resources as unsafe rather than closing under it.
                with self._lock:
                    self._process_native_abandoned = True
                    self._gpu_native_abandoned = True
                    self._note_diagnostic_locked(
                        "close requested while the sampling thread is "
                        "still running; sampler handles left untouched"
                    )
        gpu_failure = self._close_resource(
            "gpu", self._gpu, mandatory=False
        )
        if gpu_failure is not None:
            deferred.append(gpu_failure)
        process_failure = self._close_resource(
            "process", self._sampler, mandatory=True
        )
        if process_failure is not None:
            deferred.append(process_failure)
        self._closed = (
            self._gpu_close_finalized and self._process_close_finalized
        )
        if deferred:
            raise deferred[0]

    def admission_block_reason(self) -> TerminationReason | None:
        """Report the highest-precedence reason to stop admitting cycles."""
        with self._lock:
            if self._process_exited:
                return TerminationReason.MEMORY_PROCESS_DISAPPEARED
            if self._sampling_failed:
                return TerminationReason.MEMORY_SAMPLING_FAILURE
            if self._peak_breached:
                return TerminationReason.MEMORY_STABILITY_FAILURE
        return None

    def build_report(
        self,
    ) -> tuple[MemoryStabilityReport, TerminationReason | None]:
        """Build the frozen report and the termination-reason override."""
        limits = self._thresholds
        with self._lock:
            samples = self._retention.snapshot()
            total = self._retention.total_count
            missed = self._missed_intervals
            diagnostics = tuple(self._diagnostics)
            omitted_diagnostics = self._omitted_diagnostics
            exited = self._process_exited
            sampling_failed = self._sampling_failed
            shutdown_failed = (
                self._shutdown_failed or self._mandatory_close_failure
            )
            gpu_status = self._gpu_status
            gpu_final = self._gpu_final_bytes
            gpu_peak = self._gpu_peak_bytes
            working_set = _memory_metric_report(
                "working_set",
                self._working_set,
                limits.working_set_max_peak_growth_bytes,
                limits.working_set_max_final_growth_bytes,
                limits.working_set_max_final_growth_percent,
                limits.working_set_max_slope_bytes_per_minute,
            )
            private_commit = _memory_metric_report(
                "private_commit",
                self._private_commit,
                limits.private_commit_max_peak_growth_bytes,
                limits.private_commit_max_final_growth_bytes,
                limits.private_commit_max_final_growth_percent,
                limits.private_commit_max_slope_bytes_per_minute,
            )

        identity = (
            ProcessIdentityStatus.EXITED
            if exited
            else ProcessIdentityStatus.PRESERVED
        )
        override: TerminationReason | None = None
        if exited:
            verdict = MemoryStabilityVerdict.INDETERMINATE
            override = TerminationReason.MEMORY_PROCESS_DISAPPEARED
        elif sampling_failed:
            verdict = MemoryStabilityVerdict.INDETERMINATE
            override = TerminationReason.MEMORY_SAMPLING_FAILURE
        elif (
            working_set is not None
            and private_commit is not None
            and (
                working_set.failed_thresholds
                or private_commit.failed_thresholds
            )
        ):
            verdict = MemoryStabilityVerdict.UNSTABLE
            override = TerminationReason.MEMORY_STABILITY_FAILURE
        elif (
            shutdown_failed
            or working_set is None
            or private_commit is None
            or working_set.sample_count < MINIMUM_MEMORY_VERDICT_SAMPLES
            or working_set.span_seconds < MINIMUM_MEMORY_VERDICT_SPAN_SECONDS
            or working_set.unevaluated_thresholds
            or private_commit.unevaluated_thresholds
        ):
            # A mandatory close or shutdown failure compromises evidence
            # integrity and can never yield a STABLE verdict.
            verdict = MemoryStabilityVerdict.INDETERMINATE
        else:
            verdict = MemoryStabilityVerdict.STABLE

        report = MemoryStabilityReport(
            verdict=verdict,
            monitoring_enabled=True,
            identity=identity,
            mandatory_sampling_failure=sampling_failed,
            mandatory_shutdown_failure=shutdown_failed,
            pid=self._pid,
            process_creation_time_100ns=self._process_creation_time_100ns,
            executable_path=self._executable_path,
            executable_basename=self._executable_basename,
            total_sample_count=total,
            retained_sample_count=len(samples),
            omitted_sample_count=total - len(samples),
            missed_interval_count=missed,
            samples=samples,
            working_set=working_set,
            private_commit=private_commit,
            thresholds=limits,
            gpu_status=gpu_status,
            gpu_final_dedicated_bytes=gpu_final,
            gpu_peak_dedicated_bytes=gpu_peak,
            diagnostics=diagnostics,
            omitted_diagnostic_count=omitted_diagnostics,
        )
        if (
            report.verdict is MemoryStabilityVerdict.STABLE
            and not _stable_memory_report_coherent(report)
        ):
            # Defense in depth: malformed or contradictory construction-time
            # evidence is never emitted with a STABLE verdict, even before the
            # independent live acceptance boundary checks it again.
            report = replace(
                report, verdict=MemoryStabilityVerdict.INDETERMINATE
            )
        return report, override


class _WindowsProcessSamplerAdapter:
    """Translate the isolated Windows sampler into the monitor protocol."""

    __slots__ = ("_module", "_sampler")

    def __init__(self, module, sampler) -> None:
        self._module = module
        self._sampler = sampler

    def sample(self) -> ProcessCounterReading:
        try:
            counters = self._sampler.sample()
        except Exception as exc:
            if isinstance(exc, self._module.ProcessExitedError):
                raise MemoryMonitoredProcessExited(
                    _bounded_text(exc, MAX_CLEANUP_ERROR_LENGTH)
                ) from exc
            raise
        return ProcessCounterReading(
            working_set_bytes=counters.working_set_bytes,
            private_commit_bytes=counters.private_commit_bytes,
            pagefile_usage_bytes=counters.pagefile_usage_bytes,
        )

    def close(self) -> None:
        self._sampler.close()


class _WindowsGpuSamplerAdapter:
    """Translate the isolated PDH GPU sampler into the monitor protocol."""

    __slots__ = ("_sampler",)

    def __init__(self, sampler) -> None:
        self._sampler = sampler

    def sample(self) -> GpuEvidenceReading:
        observed = self._sampler.sample()
        status_name = getattr(getattr(observed, "status", None), "name", None)
        try:
            status = MemoryCounterAvailability[status_name]
        except (KeyError, TypeError):
            status = MemoryCounterAvailability.UNRELIABLE
        value = getattr(observed, "dedicated_bytes", None)
        if isinstance(value, bool) or (
            value is not None and type(value) is not int
        ):
            value = None
        return GpuEvidenceReading(status=status, dedicated_bytes=value)

    def close(self) -> None:
        self._sampler.close()


def _frozen_exception_diagnostics(
    context: str, exc: BaseException
) -> tuple[tuple[str, ...], int]:
    """Freeze one exception and its notes as bounded plain strings."""
    detail = _bounded_text(exc, MAX_CLEANUP_ERROR_LENGTH)
    primary = _bounded_text(
        f"{context}: {type(exc).__name__}: {detail}",
        MAX_CLEANUP_ERROR_LENGTH,
    )
    notes = getattr(exc, "__notes__", ())
    if not isinstance(notes, (tuple, list)):
        notes = ()
    retained_notes = notes[: MAX_RETAINED_MEMORY_DIAGNOSTICS - 1]
    secondary = tuple(
        _bounded_text(
            f"{context} secondary evidence: "
            f"{_bounded_text(note, MAX_CLEANUP_ERROR_LENGTH)}",
            MAX_CLEANUP_ERROR_LENGTH,
        )
        for note in retained_notes
    )
    retained = (primary,) + secondary
    omitted = max(0, 1 + len(notes) - len(retained))
    return retained, omitted


def _default_memory_monitor_factory(
    importer: Callable[[str], object],
) -> Callable[[MemoryStabilityConfig], MemoryStabilityMonitor]:
    def build(memory_config: MemoryStabilityConfig) -> MemoryStabilityMonitor:
        module = _import_live_dependency(
            "tests.windows_memory_sampler", importer
        )
        # Process resolution and opening are native work too, so they run
        # through the same bounded contract as every later sampling call; a
        # hung open is abandoned on its daemon worker and classified as a
        # memory sampling failure by the caller.
        process_sampler = _run_native_call_bounded(
            "process sampler open",
            lambda: module.WindowsProcessMemorySampler.open(
                pid=memory_config.server_pid
            ),
            MEMORY_NATIVE_CALL_TIMEOUT_SECONDS,
        )
        gpu_adapter = None
        gpu_startup_status = MemoryCounterAvailability.NOT_SAMPLED
        startup_diagnostics: tuple[str, ...] = ()
        omitted_startup_diagnostics = 0
        try:
            try:
                gpu_sampler = module.GpuDedicatedMemorySampler(
                    process_sampler.identity.pid
                )
                opened_status = _run_native_call_bounded(
                    "gpu sampler open",
                    gpu_sampler.open,
                    MEMORY_NATIVE_CALL_TIMEOUT_SECONDS,
                )
                if getattr(opened_status, "name", None) != "AVAILABLE":
                    raise RuntimeError(
                        "gpu sampler startup returned unavailable status "
                        f"{getattr(opened_status, 'name', opened_status)!r}"
                    )
                gpu_adapter = _WindowsGpuSamplerAdapter(gpu_sampler)
            except Exception as gpu_failure:
                gpu_adapter = None
                gpu_startup_status = MemoryCounterAvailability.UNSUPPORTED
                (
                    startup_diagnostics,
                    omitted_startup_diagnostics,
                ) = _frozen_exception_diagnostics(
                    "gpu sampler startup failed", gpu_failure
                )
            identity = process_sampler.identity
            return MemoryStabilityMonitor(
                sampler=_WindowsProcessSamplerAdapter(module, process_sampler),
                thresholds=memory_config.thresholds,
                gpu_sampler=gpu_adapter,
                pid=identity.pid,
                process_creation_time_100ns=identity.creation_time_100ns,
                executable_path=identity.image_path,
                executable_basename=module.image_base_name(
                    identity.image_path
                ),
                gpu_startup_status=gpu_startup_status,
                startup_diagnostics=startup_diagnostics,
                omitted_startup_diagnostic_count=(
                    omitted_startup_diagnostics
                ),
            )
        except BaseException as original_failure:
            deferred_cleanup_failure = None
            if gpu_adapter is not None:
                try:
                    _run_native_call_bounded(
                        "gpu sampler close",
                        gpu_adapter.close,
                        MEMORY_NATIVE_CALL_TIMEOUT_SECONDS,
                    )
                except BaseException as cleanup_exc:
                    deferred_cleanup_failure = cleanup_exc
            try:
                _run_native_call_bounded(
                    "process sampler close",
                    process_sampler.close,
                    MEMORY_NATIVE_CALL_TIMEOUT_SECONDS,
                )
            except BaseException as cleanup_exc:
                if deferred_cleanup_failure is None:
                    deferred_cleanup_failure = cleanup_exc
            # Construction failure remains primary and retains exact identity;
            # every safe rollback stage has already run independently.
            if deferred_cleanup_failure is not None:
                cleanup_detail = _bounded_text(
                    deferred_cleanup_failure, MAX_CLEANUP_ERROR_LENGTH
                )
                original_failure.add_note(
                    _bounded_text(
                        "monitor construction cleanup also failed: "
                        f"{type(deferred_cleanup_failure).__name__}: "
                        f"{cleanup_detail}",
                        MAX_CLEANUP_ERROR_LENGTH,
                    )
                )
            raise

    return build


def _memory_unavailable_report(
    memory_config: MemoryStabilityConfig, exc: Exception
) -> MemoryStabilityReport:
    diagnostics, omitted_diagnostics = _frozen_exception_diagnostics(
        "memory monitor startup failed", exc
    )
    return MemoryStabilityReport(
        verdict=MemoryStabilityVerdict.INDETERMINATE,
        monitoring_enabled=True,
        identity=ProcessIdentityStatus.UNAVAILABLE,
        mandatory_sampling_failure=True,
        mandatory_shutdown_failure=False,
        pid=memory_config.server_pid,
        process_creation_time_100ns=None,
        executable_path=None,
        executable_basename=None,
        total_sample_count=0,
        retained_sample_count=0,
        omitted_sample_count=0,
        missed_interval_count=0,
        samples=(),
        working_set=None,
        private_commit=None,
        thresholds=memory_config.thresholds,
        gpu_status=MemoryCounterAvailability.NOT_SAMPLED,
        gpu_final_dedicated_bytes=None,
        gpu_peak_dedicated_bytes=None,
        diagnostics=diagnostics,
        omitted_diagnostic_count=omitted_diagnostics,
    )


def _memory_failure_aggregate(
    config: IntegrationConfig,
    elapsed_seconds: float,
    termination_reason: TerminationReason,
    memory_report: MemoryStabilityReport,
) -> RepeatedSensorSuiteValidationReport:
    """Aggregate for a memory failure raised before any cycle admission."""
    return RepeatedSensorSuiteValidationReport(
        execution_mode=ExecutionMode.TIMED_SOAK,
        requested_cycle_count=None,
        requested_soak_seconds=config.soak_seconds,
        attempted_cycle_count=0,
        completed_cycle_count=0,
        passed_cycle_count=0,
        elapsed_seconds=elapsed_seconds,
        termination_reason=termination_reason,
        cycles=(),
        omitted_cycle_summary_count=0,
        first_primary_error=None,
        failed_cycle_index=None,
        supplemental_cleanup_errors=(),
        omitted_cleanup_error_count=0,
        final_disappearance=None,
        remaining_owned_actors=(),
        success=False,
        memory=memory_report,
    )


def run_repeated_sensor_suite_validation_with_memory_stability(
    environ: Mapping[str, str] | None = None,
    *,
    importer: Callable[[str], object] = importlib.import_module,
    monotonic: Callable[[], float] = time.monotonic,
    disappearance_timeout_seconds: float = EGO_DISAPPEARANCE_TIMEOUT_SECONDS,
    monitor_factory: Callable[
        [MemoryStabilityConfig], MemoryStabilityMonitor
    ] | None = None,
) -> RepeatedSensorSuiteValidationReport:
    """Run the Mission 4G.5 outer runner with optional memory stability.

    Disabled mode delegates unchanged and reports NOT_REQUESTED.  Enabled
    mode validates the complete configuration contract before any process
    enumeration or live work, samples around the unchanged outer runner,
    blocks only next-cycle admission on memory evidence, and always stops,
    joins, and closes the monitor in the outer ``finally`` while preserving
    the exact identity of any escaping BaseException.
    """
    config = require_integration_opt_in(environ)
    execution_mode = select_execution_mode(config)
    memory_config = parse_memory_stability_config(
        environ, config, execution_mode
    )
    if not memory_config.enabled:
        return run_repeated_sensor_suite_validation_cycles(
            environ,
            importer=importer,
            monotonic=monotonic,
            disappearance_timeout_seconds=disappearance_timeout_seconds,
        )

    factory = (
        monitor_factory
        if monitor_factory is not None
        else _default_memory_monitor_factory(importer)
    )
    run_started = _repeated_clock_value(monotonic)
    monitor = None
    aggregate: RepeatedSensorSuiteValidationReport | None = None
    startup_error: Exception | None = None
    memory_report: MemoryStabilityReport | None = None
    override: TerminationReason | None = None
    escaping: BaseException | None = None
    escaping_traceback = None
    cleanup_failures: list[BaseException] = []
    try:
        try:
            monitor = factory(memory_config)
            monitor.take_baseline()
            monitor.start()
        except Exception as exc:
            startup_error = exc
        if startup_error is None:
            aggregate = run_repeated_sensor_suite_validation_cycles(
                environ,
                importer=importer,
                monotonic=monotonic,
                disappearance_timeout_seconds=disappearance_timeout_seconds,
                admission_guard=monitor.admission_block_reason,
            )
    except BaseException as exc:
        escaping = exc
        escaping_traceback = exc.__traceback__
    finally:
        if monitor is not None:
            # One ordered finalization tree handles every exit path.  Each
            # stage is independent, final sampling internally proves that the
            # worker is stopped, and report construction happens only after
            # both native-resource close stages have been finalized.
            for cleanup_stage in (
                monitor.request_stop,
                monitor.join,
                monitor.take_final_sample,
                monitor.close,
            ):
                try:
                    cleanup_stage()
                except BaseException as cleanup_exc:
                    cleanup_failures.append(cleanup_exc)
            try:
                memory_report, override = monitor.build_report()
            except BaseException as report_exc:
                cleanup_failures.append(report_exc)

    if escaping is not None:
        raise escaping.with_traceback(escaping_traceback)
    if cleanup_failures:
        raise cleanup_failures[0]
    if startup_error is not None:
        failure_elapsed = _repeated_elapsed(
            run_started, _repeated_clock_value(monotonic)
        )
        if memory_report is None:
            memory_report = _memory_unavailable_report(
                memory_config, startup_error
            )
            override = None
        termination = (
            override
            if override is not None
            else TerminationReason.MEMORY_SAMPLING_FAILURE
        )
        return _memory_failure_aggregate(
            config, failure_elapsed, termination, memory_report
        )

    if aggregate is None or memory_report is None:
        raise RuntimeError("memory stability finalization produced no report")
    termination = aggregate.termination_reason
    if (
        override is not None
        and termination is TerminationReason.SOAK_DURATION_REACHED
    ):
        termination = override
    success = (
        aggregate.success
        and termination is TerminationReason.SOAK_DURATION_REACHED
        and memory_report.verdict is MemoryStabilityVerdict.STABLE
    )
    return replace(
        aggregate,
        termination_reason=termination,
        success=success,
        memory=memory_report,
    )


class ConnectionSpawnLiveTest(unittest.TestCase):
    """The single Mission 4G.5 live entry, gated before all live work."""

    def test_connection_and_single_owned_ego_cycle(self) -> None:
        report = run_repeated_sensor_suite_validation_with_memory_stability()
        if not _repeated_sensor_suite_validation_report_succeeded(report):
            primary = "none"
            if report.first_primary_error is not None:
                primary = (
                    f"cycle {report.first_primary_error.cycle} "
                    f"{report.first_primary_error.phase}: "
                    f"{report.first_primary_error.message}"
                )
            self.fail(
                _bounded_text(
                    f"mode={report.execution_mode.value}; "
                    f"termination={report.termination_reason.value}; "
                    f"requested_cycles={report.requested_cycle_count}; "
                    f"requested_soak={report.requested_soak_seconds}; "
                    f"attempted={report.attempted_cycle_count}; "
                    f"completed={report.completed_cycle_count}; "
                    f"passed={report.passed_cycle_count}; "
                    f"failed_cycle={report.failed_cycle_index}; "
                    f"primary={primary}; cleanup_errors="
                    f"{report.supplemental_cleanup_errors or 'none'}; "
                    f"omitted_summaries={report.omitted_cycle_summary_count}; "
                    f"omitted_cleanup={report.omitted_cleanup_error_count}; "
                    f"remaining={report.remaining_owned_actors}; "
                    f"memory={report.memory.verdict.value}",
                    MAX_COMPLETE_DIAGNOSTIC_LENGTH,
                )
            )


if __name__ == "__main__":
    unittest.main()
