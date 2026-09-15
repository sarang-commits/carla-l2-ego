"""Bounded, deterministic temporal tracking for image-space lane evidence.

The tracker is deliberately a lock-free, single-consumer state machine.  It
retains one canonical geometry estimate and scalar accounting only; it owns no
clock, worker, queue, frame, or observation history.  Source simulation time
drives coast age, while source frame/simulation-time ordering matches the
accepted perception runtime contract.

``TemporalLanePipeline`` is an opt-in adapter for ``PerceptionRuntime``.  Its
return type remains a ``LaneObservation`` so the Mission 6 runtime and bridge
contracts stay unchanged, while the immutable temporal estimate is published
atomically as an additional field on that observation subtype.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from perception.lane import perceive_lanes
from perception.state import (
    LaneBoundaryObservation,
    LaneBoundarySide,
    LaneDetectionState,
    LaneObservation,
    LaneObservationReason,
    NormalizedImagePoint,
)
from telemetry.state import (
    CameraFrame,
    CameraFrameMetadata,
    SampleStamp,
    TelemetrySnapshot,
)


MAX_TEMPORAL_CONSECUTIVE_MISSES = 1_000_000
MAX_TEMPORAL_COAST_AGE_SECONDS = 86_400.0


__all__ = (
    "DEFAULT_TEMPORAL_LANE_TRACKER_CONFIG",
    "MAX_TEMPORAL_COAST_AGE_SECONDS",
    "MAX_TEMPORAL_CONSECUTIVE_MISSES",
    "TemporalLaneBoundary",
    "TemporalLaneEstimate",
    "TemporalLaneObservation",
    "TemporalLanePipeline",
    "TemporalLaneState",
    "TemporalLaneTracker",
    "TemporalLaneTrackerConfig",
    "TemporalLaneTrackerMetrics",
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


def _canonical_stamp(value: object, name: str = "source_stamp") -> SampleStamp:
    if not isinstance(value, SampleStamp):
        raise ValueError(f"{name} must be a SampleStamp")
    raw_frame = value.frame
    raw_sim_time = value.sim_time
    raw_monotonic = value.monotonic
    frame = _nonnegative_builtin_int(raw_frame, f"{name}.frame")
    sim_time = _finite_real(raw_sim_time, f"{name}.sim_time")
    monotonic = _finite_real(raw_monotonic, f"{name}.monotonic")
    # Always detach even an exact frozen input. ``object.__setattr__`` can
    # otherwise let its caller rewrite the retained ordering baseline.
    return SampleStamp(frame=frame, sim_time=sim_time, monotonic=monotonic)


def _canonical_point(value: object, name: str) -> NormalizedImagePoint:
    if not isinstance(value, NormalizedImagePoint):
        raise ValueError(f"{name} must be a NormalizedImagePoint")
    return NormalizedImagePoint(
        x=_unit_real(value.x, f"{name}.x"),
        y=_unit_real(value.y, f"{name}.y"),
    )


def _canonical_boundary(
    value: object, side: LaneBoundarySide, name: str
) -> LaneBoundaryObservation | None:
    if value is None:
        return None
    if not isinstance(value, LaneBoundaryObservation):
        raise ValueError(f"{name} must be a LaneBoundaryObservation or None")
    if value.side is not side:
        raise ValueError(f"{name} must have side {side.value!r}")
    endpoints = value.endpoints
    if type(endpoints) is not tuple or len(endpoints) != 2:
        raise ValueError(f"{name}.endpoints must be a two-item tuple")
    return LaneBoundaryObservation(
        side=side,
        endpoints=(
            _canonical_point(endpoints[0], f"{name}.endpoints[0]"),
            _canonical_point(endpoints[1], f"{name}.endpoints[1]"),
        ),
        x_intercept=_finite_real(value.x_intercept, f"{name}.x_intercept"),
        x_slope=_finite_real(value.x_slope, f"{name}.x_slope"),
        support_count=_nonnegative_builtin_int(
            value.support_count, f"{name}.support_count"
        ),
        fit_residual=_finite_real(value.fit_residual, f"{name}.fit_residual"),
        confidence=_unit_real(value.confidence, f"{name}.confidence"),
    )


def _canonical_centerline(
    value: object,
) -> tuple[NormalizedImagePoint, NormalizedImagePoint] | None:
    if value is None:
        return None
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("centerline_endpoints must be a two-item tuple or None")
    return (
        _canonical_point(value[0], "centerline_endpoints[0]"),
        _canonical_point(value[1], "centerline_endpoints[1]"),
    )


def _canonical_lane_observation(value: object) -> LaneObservation:
    """Copy a LaneObservation and every nested retained value to exact types."""
    if not isinstance(value, LaneObservation):
        raise TypeError("observation must be a LaneObservation")
    revision = _nonnegative_builtin_int(value.snapshot_revision, "snapshot_revision")
    source_frame = value.source_frame
    if source_frame is not None:
        source_frame = _nonnegative_builtin_int(source_frame, "source_frame")
    source_simulation_timestamp = value.source_simulation_timestamp
    if source_simulation_timestamp is not None:
        source_simulation_timestamp = _finite_real(
            source_simulation_timestamp, "source_simulation_timestamp"
        )
    source_receive_monotonic = value.source_receive_monotonic
    if source_receive_monotonic is not None:
        source_receive_monotonic = _finite_real(
            source_receive_monotonic, "source_receive_monotonic"
        )
    if not isinstance(value.state, LaneDetectionState):
        raise ValueError("state must be a LaneDetectionState")
    if not isinstance(value.reason, LaneObservationReason):
        raise ValueError("reason must be a LaneObservationReason")
    left = _canonical_boundary(
        value.left_boundary, LaneBoundarySide.LEFT, "left_boundary"
    )
    right = _canonical_boundary(
        value.right_boundary, LaneBoundarySide.RIGHT, "right_boundary"
    )
    centerline = _canonical_centerline(value.centerline_endpoints)
    center_offset = value.center_offset
    if center_offset is not None:
        center_offset = _finite_real(center_offset, "center_offset")
    heading_proxy = value.heading_proxy
    if heading_proxy is not None:
        heading_proxy = _finite_real(heading_proxy, "heading_proxy")
    canonical = LaneObservation(
        snapshot_revision=revision,
        source_frame=source_frame,
        source_simulation_timestamp=source_simulation_timestamp,
        source_receive_monotonic=source_receive_monotonic,
        state=value.state,
        reason=value.reason,
        left_boundary=left,
        right_boundary=right,
        centerline_endpoints=centerline,
        center_offset=center_offset,
        heading_proxy=heading_proxy,
        confidence=_unit_real(value.confidence, "confidence"),
    )
    presence = (
        canonical.source_frame is not None,
        canonical.source_simulation_timestamp is not None,
        canonical.source_receive_monotonic is not None,
    )
    if any(presence) and not all(presence):
        raise ValueError("observation source evidence must be complete or absent")
    if canonical.state is not LaneDetectionState.INPUT_UNUSABLE and not all(presence):
        raise ValueError("usable and not-detected observations require a source stamp")
    return canonical


def _observation_stamp(observation: LaneObservation) -> SampleStamp | None:
    if observation.source_frame is None:
        return None
    assert observation.source_simulation_timestamp is not None
    assert observation.source_receive_monotonic is not None
    return SampleStamp(
        frame=observation.source_frame,
        sim_time=observation.source_simulation_timestamp,
        monotonic=observation.source_receive_monotonic,
    )


def _resolve_source_stamp(
    observation: LaneObservation, supplied: object | None
) -> SampleStamp | None:
    derived = _observation_stamp(observation)
    if supplied is None:
        if derived is None and observation.state is not LaneDetectionState.INPUT_UNUSABLE:
            raise ValueError("observation requires finite source stamp evidence")
        return derived
    explicit = _canonical_stamp(supplied)
    if derived is not None and (
        derived.frame != explicit.frame
        or derived.sim_time != explicit.sim_time
        or derived.monotonic != explicit.monotonic
    ):
        raise ValueError("supplied source_stamp does not match observation evidence")
    return explicit


def _ema(previous: float, current: float, alpha: float, name: str) -> float:
    try:
        result = math.fsum((alpha * current, (1.0 - alpha) * previous))
    except OverflowError as exc:
        raise ValueError(f"{name} smoothing produced a non-finite result") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} smoothing produced a non-finite result")
    return float(result)


def _source_age(current_sim_time: float, last_usable_sim_time: float | None) -> float | None:
    if last_usable_sim_time is None:
        return None
    age = current_sim_time - last_usable_sim_time
    if not math.isfinite(age) or age < 0.0:
        return None
    return float(age)


class TemporalLaneState(Enum):
    UNINITIALIZED = "uninitialized"
    TRACKING = "tracking"
    COASTING = "coasting"
    LOST = "lost"
    INPUT_UNUSABLE = "input_unusable"


@dataclass(frozen=True, slots=True)
class TemporalLaneTrackerConfig:
    """One small, exact policy record for a tracker lifecycle."""

    smoothing_alpha: float = 0.35
    max_consecutive_misses: int = 2
    max_coast_age_seconds: float = 0.30
    coast_confidence_decay: float = 0.75
    minimum_coast_confidence: float = 0.10

    def __post_init__(self) -> None:
        alpha = _finite_real(self.smoothing_alpha, "smoothing_alpha")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0.0, 1.0]")
        misses = _nonnegative_builtin_int(
            self.max_consecutive_misses, "max_consecutive_misses"
        )
        if misses > MAX_TEMPORAL_CONSECUTIVE_MISSES:
            raise ValueError(
                "max_consecutive_misses cannot exceed "
                f"{MAX_TEMPORAL_CONSECUTIVE_MISSES}"
            )
        coast_age = _finite_real(
            self.max_coast_age_seconds, "max_coast_age_seconds"
        )
        if not 0.0 < coast_age <= MAX_TEMPORAL_COAST_AGE_SECONDS:
            raise ValueError(
                "max_coast_age_seconds must be in "
                f"(0.0, {MAX_TEMPORAL_COAST_AGE_SECONDS}]"
            )
        decay = _finite_real(
            self.coast_confidence_decay, "coast_confidence_decay"
        )
        if not 0.0 < decay < 1.0:
            raise ValueError("coast_confidence_decay must be in (0.0, 1.0)")
        minimum = _finite_real(
            self.minimum_coast_confidence, "minimum_coast_confidence"
        )
        if not 0.0 < minimum <= 1.0:
            raise ValueError("minimum_coast_confidence must be in (0.0, 1.0]")
        object.__setattr__(self, "smoothing_alpha", alpha)
        object.__setattr__(self, "max_consecutive_misses", misses)
        object.__setattr__(self, "max_coast_age_seconds", coast_age)
        object.__setattr__(self, "coast_confidence_decay", decay)
        object.__setattr__(self, "minimum_coast_confidence", minimum)


DEFAULT_TEMPORAL_LANE_TRACKER_CONFIG = TemporalLaneTrackerConfig()


@dataclass(frozen=True, slots=True)
class TemporalLaneBoundary:
    """Smoothed line coefficients plus conservative boundary confidence."""

    side: LaneBoundarySide
    x_intercept: float
    x_slope: float
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.side, LaneBoundarySide):
            raise ValueError("side must be a LaneBoundarySide")
        object.__setattr__(
            self, "x_intercept", _finite_real(self.x_intercept, "x_intercept")
        )
        object.__setattr__(self, "x_slope", _finite_real(self.x_slope, "x_slope"))
        object.__setattr__(
            self, "confidence", _unit_real(self.confidence, "confidence")
        )


def _canonical_temporal_boundary(
    value: object, side: LaneBoundarySide, name: str
) -> TemporalLaneBoundary | None:
    if value is None:
        return None
    if not isinstance(value, TemporalLaneBoundary) or value.side is not side:
        raise ValueError(f"{name} must be a matching TemporalLaneBoundary or None")
    return TemporalLaneBoundary(
        side=side,
        x_intercept=value.x_intercept,
        x_slope=value.x_slope,
        confidence=value.confidence,
    )


@dataclass(frozen=True, slots=True)
class TemporalLaneEstimate:
    """One immutable, payload-free view of current temporal lane evidence."""

    source_stamp: SampleStamp | None
    snapshot_revision: int | None
    state: TemporalLaneState
    raw_state: LaneDetectionState | None
    raw_reason: LaneObservationReason | None
    left_boundary: TemporalLaneBoundary | None = None
    right_boundary: TemporalLaneBoundary | None = None
    center_offset: float | None = None
    heading_proxy: float | None = None
    confidence: float = 0.0
    consecutive_misses: int = 0
    track_age_seconds: float | None = None
    extrapolated: bool = False

    def __post_init__(self) -> None:
        stamp = None if self.source_stamp is None else _canonical_stamp(self.source_stamp)
        if self.snapshot_revision is not None:
            _nonnegative_builtin_int(self.snapshot_revision, "snapshot_revision")
        if not isinstance(self.state, TemporalLaneState):
            raise ValueError("state must be a TemporalLaneState")
        if self.raw_state is not None and not isinstance(
            self.raw_state, LaneDetectionState
        ):
            raise ValueError("raw_state must be a LaneDetectionState or None")
        if self.raw_reason is not None and not isinstance(
            self.raw_reason, LaneObservationReason
        ):
            raise ValueError("raw_reason must be a LaneObservationReason or None")
        left = _canonical_temporal_boundary(
            self.left_boundary, LaneBoundarySide.LEFT, "left_boundary"
        )
        right = _canonical_temporal_boundary(
            self.right_boundary, LaneBoundarySide.RIGHT, "right_boundary"
        )
        center_offset = self.center_offset
        if center_offset is not None:
            center_offset = _finite_real(center_offset, "center_offset")
            if not -1.0 <= center_offset <= 1.0:
                raise ValueError("center_offset must be in [-1.0, 1.0]")
        heading_proxy = self.heading_proxy
        if heading_proxy is not None:
            heading_proxy = _finite_real(heading_proxy, "heading_proxy")
            if not -1.0 <= heading_proxy <= 1.0:
                raise ValueError("heading_proxy must be in [-1.0, 1.0]")
        confidence = _unit_real(self.confidence, "confidence")
        misses = _nonnegative_builtin_int(
            self.consecutive_misses, "consecutive_misses"
        )
        track_age = self.track_age_seconds
        if track_age is not None:
            track_age = _finite_real(track_age, "track_age_seconds")
            if track_age < 0.0:
                raise ValueError("track_age_seconds must be non-negative")
        if type(self.extrapolated) is not bool:
            raise ValueError("extrapolated must be a built-in bool")

        geometry_present = left is not None or right is not None
        center_present = center_offset is not None or heading_proxy is not None
        if geometry_present:
            boundary_confidence = min(
                boundary.confidence
                for boundary in (left, right)
                if boundary is not None
            )
            if confidence > boundary_confidence:
                raise ValueError(
                    "temporal confidence cannot exceed present boundary confidence"
                )
        if center_present and (
            center_offset is None
            or heading_proxy is None
            or left is None
            or right is None
        ):
            raise ValueError("center geometry requires both boundaries and both scalars")

        if self.state is TemporalLaneState.UNINITIALIZED:
            if any(
                value is not None
                for value in (
                    stamp,
                    self.snapshot_revision,
                    self.raw_state,
                    self.raw_reason,
                    left,
                    right,
                    center_offset,
                    heading_proxy,
                    track_age,
                )
            ) or confidence != 0.0 or misses != 0 or self.extrapolated:
                raise ValueError("UNINITIALIZED cannot expose observation or track evidence")
        else:
            if self.snapshot_revision is None or self.raw_state is None or self.raw_reason is None:
                raise ValueError("non-initial estimates require raw observation evidence")

        if self.state is TemporalLaneState.TRACKING:
            if stamp is None:
                raise ValueError("TRACKING requires a source stamp")
            if self.raw_state not in (
                LaneDetectionState.DETECTED,
                LaneDetectionState.PARTIAL,
            ):
                raise ValueError("TRACKING requires detected or partial raw evidence")
            if not geometry_present or misses != 0 or track_age != 0.0 or self.extrapolated:
                raise ValueError("TRACKING requires current, non-extrapolated geometry")
            if self.raw_state is LaneDetectionState.DETECTED and (
                left is None or right is None or not center_present
            ):
                raise ValueError("detected TRACKING requires complete temporal geometry")
            if self.raw_state is LaneDetectionState.PARTIAL and center_present:
                raise ValueError("partial TRACKING cannot expose complete-lane geometry")
        elif self.state is TemporalLaneState.COASTING:
            if (
                stamp is None
                or self.raw_state is not LaneDetectionState.NOT_DETECTED
                or not geometry_present
                or misses == 0
                or track_age is None
                or confidence <= 0.0
                or not self.extrapolated
            ):
                raise ValueError("COASTING requires bounded retained track evidence")
        elif self.state is TemporalLaneState.LOST:
            if (
                stamp is None
                or self.raw_state is not LaneDetectionState.NOT_DETECTED
                or geometry_present
                or center_present
                or confidence != 0.0
                or misses == 0
                or self.extrapolated
            ):
                raise ValueError("LOST cannot expose usable temporal geometry")
        elif self.state is TemporalLaneState.INPUT_UNUSABLE:
            if (
                self.raw_state is not LaneDetectionState.INPUT_UNUSABLE
                or geometry_present
                or center_present
                or confidence != 0.0
                or misses != 0
                or self.extrapolated
            ):
                raise ValueError("INPUT_UNUSABLE cannot expose usable temporal geometry")

        object.__setattr__(self, "source_stamp", stamp)
        object.__setattr__(self, "left_boundary", left)
        object.__setattr__(self, "right_boundary", right)
        object.__setattr__(self, "center_offset", center_offset)
        object.__setattr__(self, "heading_proxy", heading_proxy)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "consecutive_misses", misses)
        object.__setattr__(self, "track_age_seconds", track_age)


@dataclass(frozen=True, slots=True, kw_only=True)
class TemporalLaneObservation(LaneObservation):
    """Raw M5-compatible observation carrying its atomic M7 estimate."""

    temporal_estimate: TemporalLaneEstimate

    def __post_init__(self) -> None:
        LaneObservation.__post_init__(self)
        if type(self.temporal_estimate) is not TemporalLaneEstimate:
            raise ValueError("temporal_estimate must be an exact TemporalLaneEstimate")
        estimate = self.temporal_estimate
        if estimate.state is TemporalLaneState.UNINITIALIZED:
            raise ValueError("a temporal observation requires an accepted estimate")
        if estimate.snapshot_revision != self.snapshot_revision:
            raise ValueError("raw and temporal snapshot revisions must match")
        if estimate.raw_state is not self.state or estimate.raw_reason is not self.reason:
            raise ValueError("raw and temporal state evidence must match")
        stamp_presence = (
            self.source_frame is not None,
            self.source_simulation_timestamp is not None,
            self.source_receive_monotonic is not None,
        )
        if any(stamp_presence) and not all(stamp_presence):
            raise ValueError("raw source evidence must be complete or absent")
        intrinsic_stamp = _observation_stamp(self)
        if intrinsic_stamp is not None:
            temporal_stamp = estimate.source_stamp
            if temporal_stamp is None or (
                intrinsic_stamp.frame != temporal_stamp.frame
                or intrinsic_stamp.sim_time != temporal_stamp.sim_time
                or intrinsic_stamp.monotonic != temporal_stamp.monotonic
            ):
                raise ValueError("raw and temporal source stamps must match")
        elif self.state is not LaneDetectionState.INPUT_UNUSABLE:
            raise ValueError("only raw INPUT_UNUSABLE may omit its source stamp")


@dataclass(frozen=True, slots=True)
class TemporalLaneTrackerMetrics:
    observations_received: int
    observations_accepted: int
    duplicates_rejected: int
    out_of_order_rejected: int
    invalid_observations_rejected: int
    usable_updates: int
    coast_updates: int
    lost_updates: int
    input_unusable_updates: int
    unstamped_input_unusable_updates: int
    recoveries: int
    resets: int
    maximum_consecutive_misses: int
    last_accepted_source_stamp: SampleStamp | None


@dataclass(frozen=True, slots=True)
class _TrackGeometry:
    left_boundary: TemporalLaneBoundary | None
    right_boundary: TemporalLaneBoundary | None
    center_offset: float | None
    heading_proxy: float | None


def _initial_estimate() -> TemporalLaneEstimate:
    return TemporalLaneEstimate(
        source_stamp=None,
        snapshot_revision=None,
        state=TemporalLaneState.UNINITIALIZED,
        raw_state=None,
        raw_reason=None,
    )


def _boundary_from_raw(
    raw: LaneBoundaryObservation,
    previous: TemporalLaneBoundary | None,
    alpha: float,
    *,
    allow_confidence_increase: bool,
) -> TemporalLaneBoundary:
    if previous is None:
        intercept = raw.x_intercept
        slope = raw.x_slope
        confidence = raw.confidence
    else:
        intercept = _ema(previous.x_intercept, raw.x_intercept, alpha, "x_intercept")
        slope = _ema(previous.x_slope, raw.x_slope, alpha, "x_slope")
        if allow_confidence_increase:
            confidence = min(
                raw.confidence,
                _ema(previous.confidence, raw.confidence, alpha, "boundary confidence"),
            )
        else:
            confidence = min(previous.confidence, raw.confidence)
    return TemporalLaneBoundary(
        side=raw.side,
        x_intercept=intercept,
        x_slope=slope,
        confidence=confidence,
    )


def _tracking_geometry(
    observation: LaneObservation,
    previous: _TrackGeometry | None,
    alpha: float,
) -> _TrackGeometry:
    allow_increase = observation.state is LaneDetectionState.DETECTED
    previous_left = None if previous is None else previous.left_boundary
    previous_right = None if previous is None else previous.right_boundary
    left = (
        None
        if observation.left_boundary is None
        else _boundary_from_raw(
            observation.left_boundary,
            previous_left,
            alpha,
            allow_confidence_increase=allow_increase,
        )
    )
    right = (
        None
        if observation.right_boundary is None
        else _boundary_from_raw(
            observation.right_boundary,
            previous_right,
            alpha,
            allow_confidence_increase=allow_increase,
        )
    )
    if left is None and right is None:
        raise ValueError("detected or partial observation contains no lane geometry")
    if observation.state is LaneDetectionState.DETECTED:
        assert observation.center_offset is not None
        assert observation.heading_proxy is not None
        if (
            previous is not None
            and previous.center_offset is not None
            and previous.heading_proxy is not None
        ):
            center_offset = _ema(
                previous.center_offset,
                observation.center_offset,
                alpha,
                "center_offset",
            )
            heading_proxy = _ema(
                previous.heading_proxy,
                observation.heading_proxy,
                alpha,
                "heading_proxy",
            )
        else:
            center_offset = observation.center_offset
            heading_proxy = observation.heading_proxy
    else:
        center_offset = None
        heading_proxy = None
    return _TrackGeometry(left, right, center_offset, heading_proxy)


def _decayed_boundary(
    boundary: TemporalLaneBoundary | None, factor: float
) -> TemporalLaneBoundary | None:
    if boundary is None:
        return None
    return TemporalLaneBoundary(
        side=boundary.side,
        x_intercept=boundary.x_intercept,
        x_slope=boundary.x_slope,
        confidence=boundary.confidence * factor,
    )


class TemporalLaneTracker:
    """O(1), lock-free lane tracker for one serialized observation stream.

    ``update`` and ``reset`` must be called by one consumer at a time.  Runtime
    integration satisfies that contract by invoking the pipeline on its single
    worker.  Cross-thread consumers should use the immutable estimate embedded
    in ``PerceptionRuntime.latest_result`` rather than racing this object.

    The upstream lane configuration is expected to remain fixed for one
    tracker lifecycle; M5 observations do not carry their ROI/config identity.
    """

    __slots__ = (
        "_coast_updates",
        "_config",
        "_consecutive_misses",
        "_duplicates_rejected",
        "_estimate",
        "_input_unusable_updates",
        "_invalid_observations_rejected",
        "_last_accepted_source_stamp",
        "_last_usable_sim_time",
        "_lost_updates",
        "_maximum_consecutive_misses",
        "_observations_accepted",
        "_observations_received",
        "_out_of_order_rejected",
        "_recoveries",
        "_resets",
        "_track_confidence",
        "_track_geometry",
        "_unstamped_input_unusable_updates",
        "_usable_updates",
    )

    def __init__(
        self,
        config: TemporalLaneTrackerConfig = DEFAULT_TEMPORAL_LANE_TRACKER_CONFIG,
    ) -> None:
        if not isinstance(config, TemporalLaneTrackerConfig):
            raise TypeError("config must be a TemporalLaneTrackerConfig")
        # Detach every caller-owned record, including an exact frozen base
        # instance, before it can influence later transition decisions.
        self._config = TemporalLaneTrackerConfig(
            smoothing_alpha=config.smoothing_alpha,
            max_consecutive_misses=config.max_consecutive_misses,
            max_coast_age_seconds=config.max_coast_age_seconds,
            coast_confidence_decay=config.coast_confidence_decay,
            minimum_coast_confidence=config.minimum_coast_confidence,
        )
        self._observations_received = 0
        self._observations_accepted = 0
        self._duplicates_rejected = 0
        self._out_of_order_rejected = 0
        self._invalid_observations_rejected = 0
        self._usable_updates = 0
        self._coast_updates = 0
        self._lost_updates = 0
        self._input_unusable_updates = 0
        self._unstamped_input_unusable_updates = 0
        self._recoveries = 0
        self._resets = 0
        self._maximum_consecutive_misses = 0
        self._clear_state()

    @property
    def config(self) -> TemporalLaneTrackerConfig:
        return TemporalLaneTrackerConfig(
            smoothing_alpha=self._config.smoothing_alpha,
            max_consecutive_misses=self._config.max_consecutive_misses,
            max_coast_age_seconds=self._config.max_coast_age_seconds,
            coast_confidence_decay=self._config.coast_confidence_decay,
            minimum_coast_confidence=self._config.minimum_coast_confidence,
        )

    @property
    def latest_estimate(self) -> TemporalLaneEstimate:
        return self._estimate

    @property
    def metrics(self) -> TemporalLaneTrackerMetrics:
        return TemporalLaneTrackerMetrics(
            observations_received=self._observations_received,
            observations_accepted=self._observations_accepted,
            duplicates_rejected=self._duplicates_rejected,
            out_of_order_rejected=self._out_of_order_rejected,
            invalid_observations_rejected=self._invalid_observations_rejected,
            usable_updates=self._usable_updates,
            coast_updates=self._coast_updates,
            lost_updates=self._lost_updates,
            input_unusable_updates=self._input_unusable_updates,
            unstamped_input_unusable_updates=(
                self._unstamped_input_unusable_updates
            ),
            recoveries=self._recoveries,
            resets=self._resets,
            maximum_consecutive_misses=self._maximum_consecutive_misses,
            last_accepted_source_stamp=(
                None
                if self._last_accepted_source_stamp is None
                else _canonical_stamp(self._last_accepted_source_stamp)
            ),
        )

    def reset(self) -> TemporalLaneEstimate:
        """Forget source order and all geometry; retain lifetime counters."""
        self._resets += 1
        self._clear_state()
        return self._estimate

    def _clear_state(self) -> None:
        self._estimate = _initial_estimate()
        self._last_accepted_source_stamp: SampleStamp | None = None
        self._last_usable_sim_time: float | None = None
        self._track_geometry: _TrackGeometry | None = None
        self._track_confidence = 0.0
        self._consecutive_misses = 0

    def update(
        self,
        observation: LaneObservation,
        *,
        source_stamp: SampleStamp | None = None,
    ) -> TemporalLaneEstimate:
        """Accept one newer observation and return the current temporal view.

        Normal M5 observations carry a complete stamp.  M5 deliberately erases
        stamps from ``INPUT_UNUSABLE`` observations, so that state alone may be
        submitted without a stamp: it is surfaced immediately, clears guidance,
        and does not advance the source-order baseline.  Supplying malformed or
        mismatched evidence is an error, never an unstamped fallback.
        """
        self._observations_received += 1
        try:
            canonical = _canonical_lane_observation(observation)
            stamp = _resolve_source_stamp(canonical, source_stamp)
        except Exception:
            self._invalid_observations_rejected += 1
            raise

        previous_stamp = self._last_accepted_source_stamp
        if stamp is not None and previous_stamp is not None:
            duplicate = (
                stamp.frame == previous_stamp.frame
                and stamp.sim_time == previous_stamp.sim_time
            )
            if duplicate:
                self._duplicates_rejected += 1
                return self._estimate
            if stamp.frame <= previous_stamp.frame or stamp.sim_time < previous_stamp.sim_time:
                self._out_of_order_rejected += 1
                return self._estimate

        try:
            transition = self._transition(canonical, stamp)
        except Exception:
            self._invalid_observations_rejected += 1
            raise
        (
            estimate,
            geometry,
            confidence,
            misses,
            last_usable_sim_time,
            disposition,
        ) = transition

        prior_state = self._estimate.state
        self._estimate = estimate
        self._track_geometry = geometry
        self._track_confidence = confidence
        self._consecutive_misses = misses
        self._last_usable_sim_time = last_usable_sim_time
        if stamp is not None:
            self._last_accepted_source_stamp = stamp
        self._observations_accepted += 1
        if disposition == "usable":
            self._usable_updates += 1
            if prior_state in (
                TemporalLaneState.COASTING,
                TemporalLaneState.LOST,
                TemporalLaneState.INPUT_UNUSABLE,
            ):
                self._recoveries += 1
        elif disposition == "coast":
            self._coast_updates += 1
        elif disposition == "lost":
            self._lost_updates += 1
        else:
            assert disposition == "input_unusable"
            self._input_unusable_updates += 1
            if stamp is None:
                self._unstamped_input_unusable_updates += 1
        if misses > self._maximum_consecutive_misses:
            self._maximum_consecutive_misses = misses
        return estimate

    def _transition(
        self, observation: LaneObservation, stamp: SampleStamp | None
    ) -> tuple[
        TemporalLaneEstimate,
        _TrackGeometry | None,
        float,
        int,
        float | None,
        str,
    ]:
        if observation.state is LaneDetectionState.INPUT_UNUSABLE:
            age = (
                None
                if stamp is None
                else _source_age(stamp.sim_time, self._last_usable_sim_time)
            )
            estimate = TemporalLaneEstimate(
                source_stamp=stamp,
                snapshot_revision=observation.snapshot_revision,
                state=TemporalLaneState.INPUT_UNUSABLE,
                raw_state=observation.state,
                raw_reason=observation.reason,
                confidence=0.0,
                consecutive_misses=0,
                track_age_seconds=age,
                extrapolated=False,
            )
            return (
                estimate,
                None,
                0.0,
                0,
                self._last_usable_sim_time,
                "input_unusable",
            )

        assert stamp is not None
        if observation.state is LaneDetectionState.NOT_DETECTED:
            misses = self._consecutive_misses + 1
            age = _source_age(stamp.sim_time, self._last_usable_sim_time)
            decayed_confidence = self._track_confidence * self._config.coast_confidence_decay
            can_coast = (
                self._track_geometry is not None
                and misses <= self._config.max_consecutive_misses
                and age is not None
                and age <= self._config.max_coast_age_seconds
                and decayed_confidence >= self._config.minimum_coast_confidence
            )
            if can_coast:
                assert self._track_geometry is not None
                geometry = _TrackGeometry(
                    left_boundary=_decayed_boundary(
                        self._track_geometry.left_boundary,
                        self._config.coast_confidence_decay,
                    ),
                    right_boundary=_decayed_boundary(
                        self._track_geometry.right_boundary,
                        self._config.coast_confidence_decay,
                    ),
                    center_offset=self._track_geometry.center_offset,
                    heading_proxy=self._track_geometry.heading_proxy,
                )
                estimate = TemporalLaneEstimate(
                    source_stamp=stamp,
                    snapshot_revision=observation.snapshot_revision,
                    state=TemporalLaneState.COASTING,
                    raw_state=observation.state,
                    raw_reason=observation.reason,
                    left_boundary=geometry.left_boundary,
                    right_boundary=geometry.right_boundary,
                    center_offset=geometry.center_offset,
                    heading_proxy=geometry.heading_proxy,
                    confidence=decayed_confidence,
                    consecutive_misses=misses,
                    track_age_seconds=age,
                    extrapolated=True,
                )
                return (
                    estimate,
                    geometry,
                    decayed_confidence,
                    misses,
                    self._last_usable_sim_time,
                    "coast",
                )
            estimate = TemporalLaneEstimate(
                source_stamp=stamp,
                snapshot_revision=observation.snapshot_revision,
                state=TemporalLaneState.LOST,
                raw_state=observation.state,
                raw_reason=observation.reason,
                confidence=0.0,
                consecutive_misses=misses,
                track_age_seconds=age,
                extrapolated=False,
            )
            return (
                estimate,
                None,
                0.0,
                misses,
                self._last_usable_sim_time,
                "lost",
            )

        if observation.state not in (
            LaneDetectionState.DETECTED,
            LaneDetectionState.PARTIAL,
        ):
            raise ValueError("unsupported lane detection state")
        prior_geometry = self._track_geometry
        age = _source_age(stamp.sim_time, self._last_usable_sim_time)
        if age is None or age > self._config.max_coast_age_seconds:
            prior_geometry = None
        if observation.state is LaneDetectionState.DETECTED and (
            prior_geometry is None
            or prior_geometry.left_boundary is None
            or prior_geometry.right_boundary is None
            or prior_geometry.center_offset is None
            or prior_geometry.heading_proxy is None
        ):
            # Complete-lane quantities are mutually coherent under one fixed
            # upstream geometry convention.  After partial evidence, initialize
            # the whole complete geometry together rather than mixing a
            # one-sided EMA with current complete-lane scalars.
            prior_geometry = None
        geometry = _tracking_geometry(
            observation, prior_geometry, self._config.smoothing_alpha
        )
        current_confidence = min(
            observation.confidence,
            *(
                boundary.confidence
                for boundary in (geometry.left_boundary, geometry.right_boundary)
                if boundary is not None
            ),
        )
        if prior_geometry is None:
            confidence = current_confidence
        elif observation.state is LaneDetectionState.PARTIAL:
            confidence = min(self._track_confidence, current_confidence)
        else:
            confidence = min(
                current_confidence,
                _ema(
                    self._track_confidence,
                    current_confidence,
                    self._config.smoothing_alpha,
                    "track confidence",
                ),
            )
        estimate = TemporalLaneEstimate(
            source_stamp=stamp,
            snapshot_revision=observation.snapshot_revision,
            state=TemporalLaneState.TRACKING,
            raw_state=observation.state,
            raw_reason=observation.reason,
            left_boundary=geometry.left_boundary,
            right_boundary=geometry.right_boundary,
            center_offset=geometry.center_offset,
            heading_proxy=geometry.heading_proxy,
            confidence=confidence,
            consecutive_misses=0,
            track_age_seconds=0.0,
            extrapolated=False,
        )
        return estimate, geometry, confidence, 0, stamp.sim_time, "usable"


def _snapshot_source_stamp(snapshot: object) -> SampleStamp | None:
    if not isinstance(snapshot, TelemetrySnapshot):
        return None
    frame = snapshot.camera
    if not isinstance(frame, CameraFrame) or not isinstance(
        frame.metadata, CameraFrameMetadata
    ):
        return None
    try:
        return _canonical_stamp(frame.metadata.stamp, "snapshot camera stamp")
    except (TypeError, ValueError):
        return None


def _temporal_observation(
    raw: LaneObservation, estimate: TemporalLaneEstimate
) -> TemporalLaneObservation:
    return TemporalLaneObservation(
        snapshot_revision=raw.snapshot_revision,
        source_frame=raw.source_frame,
        source_simulation_timestamp=raw.source_simulation_timestamp,
        source_receive_monotonic=raw.source_receive_monotonic,
        state=raw.state,
        reason=raw.reason,
        left_boundary=raw.left_boundary,
        right_boundary=raw.right_boundary,
        centerline_endpoints=raw.centerline_endpoints,
        center_offset=raw.center_offset,
        heading_proxy=raw.heading_proxy,
        confidence=raw.confidence,
        temporal_estimate=estimate,
    )


class TemporalLanePipeline:
    """Opt-in M5 -> M7 callable compatible with ``PerceptionRuntime``.

    Create one pipeline per runtime epoch, or call ``reset`` while that runtime
    is stopped before starting a new source epoch.  The pipeline creates no
    thread and retains no snapshot or RGB payload.
    """

    __slots__ = ("_latest_observation", "_perception_callable", "_tracker")

    def __init__(
        self,
        config: TemporalLaneTrackerConfig = DEFAULT_TEMPORAL_LANE_TRACKER_CONFIG,
        *,
        perception_callable: Callable[[TelemetrySnapshot], LaneObservation] = perceive_lanes,
    ) -> None:
        if not callable(perception_callable):
            raise TypeError("perception_callable must be callable")
        self._tracker = TemporalLaneTracker(config)
        self._perception_callable = perception_callable
        self._latest_observation: TemporalLaneObservation | None = None

    @property
    def config(self) -> TemporalLaneTrackerConfig:
        return self._tracker.config

    @property
    def tracker(self) -> TemporalLaneTracker:
        return self._tracker

    @property
    def latest_estimate(self) -> TemporalLaneEstimate:
        return self._tracker.latest_estimate

    @property
    def metrics(self) -> TemporalLaneTrackerMetrics:
        return self._tracker.metrics

    def reset(self) -> TemporalLaneEstimate:
        estimate = self._tracker.reset()
        self._latest_observation = None
        return estimate

    def __call__(self, snapshot: TelemetrySnapshot) -> TemporalLaneObservation:
        raw = self._perception_callable(snapshot)
        canonical = _canonical_lane_observation(raw)
        previous_estimate = self._tracker.latest_estimate
        estimate = self._tracker.update(
            canonical, source_stamp=_snapshot_source_stamp(snapshot)
        )
        if estimate is previous_estimate:
            # The tracker rejected duplicate/out-of-order source evidence.  Do
            # not pair that rejected raw frame with the prior accepted estimate.
            if self._latest_observation is None:
                raise RuntimeError("tracker rejected input without an accepted carrier")
            return self._latest_observation
        result = _temporal_observation(canonical, estimate)
        self._latest_observation = result
        return result
