"""Deterministic, bounded, single-frame image-space lane perception.

The implementation consumes only the immutable RGB8 frame and evaluated
camera status already present in one ``TelemetrySnapshot``. Disconnected
candidate clusters remain separate, line hypotheses are fitted before side
classification, and all retained candidates, hypotheses, and pair comparisons
have explicit finite bounds. No clock, history, random source, filesystem,
socket, display, or CARLA API is used.
"""

from __future__ import annotations

import bisect
import math
import numbers
from dataclasses import dataclass

from perception.state import (
    DEFAULT_LANE_CONFIG,
    MAX_VERTICAL_BINS,
    LaneBoundaryObservation,
    LaneBoundarySide,
    LaneDetectionState,
    LaneObservation,
    LaneObservationReason,
    LanePerceptionConfig,
    NormalizedImagePoint,
)
from telemetry.freshness import (
    OperationalHealth,
    SensorFreshness,
    SensorKind,
    SensorLifecycle,
    SensorReadiness,
    SensorStatus,
)
from telemetry.state import (
    PIXEL_FORMAT_RGB8,
    CameraFrame,
    CameraFrameMetadata,
    SampleStamp,
    TelemetrySnapshot,
)


CAMERA_SENSOR_NAME = "rgb_camera"

# These implementation bounds are deliberately independent of image size.
MAX_REPRESENTATIVES_PER_BIN = 8
MAX_BOUNDARY_SEEDS = 512
MAX_BOUNDARY_HYPOTHESES = 16
MAX_TRACK_SKIPPED_BINS = 2
MAX_TRACK_LINKS_PER_BIN_GAP = 4
MAX_OUTGOING_LINKS_PER_REPRESENTATIVE = (
    (MAX_TRACK_SKIPPED_BINS + 1) * MAX_TRACK_LINKS_PER_BIN_GAP
)
MAX_PARTIAL_TRACKS_PER_REPRESENTATIVE = 4
MAX_PARTIAL_TRACKS_PER_BIN = (
    MAX_REPRESENTATIVES_PER_BIN * MAX_PARTIAL_TRACKS_PER_REPRESENTATIVE
)
MAX_COMPLETED_TRACKS = 64
MAX_PAIR_COMBINATIONS = (
    MAX_BOUNDARY_HYPOTHESES * (MAX_BOUNDARY_HYPOTHESES - 1) // 2
)
AMBIGUITY_QUALITY_MARGIN = 0.04
AMBIGUITY_CONFIDENCE_MARGIN = 0.08
AMBIGUITY_POSITION_DELTA = 0.04


@dataclass(frozen=True, slots=True)
class _CandidateRepresentative:
    """One actual candidate pixel plus bounded support from its cluster."""

    bin_index: int
    cluster_index: int
    x: float
    y: float
    pixel_count: int
    row_count: int
    bin_row_count: int
    minimum_x: float
    maximum_x: float
    raster_tolerance: float

    @property
    def row_coverage(self) -> float:
        return self.row_count / float(self.bin_row_count)


def _scalar_line(
    count: int,
    sum_y: float,
    sum_x: float,
    sum_yy: float,
    sum_xy: float,
) -> tuple[float | None, float | None]:
    """Return ``(intercept, slope)`` from ordinary-least-squares accumulators.

    ``(None, None)`` is returned in exactly the cases where
    :func:`_closed_form_fit` returns ``None`` for the same points: fewer than
    two points, a non-finite or non-positive vertical dispersion, or a
    non-finite coefficient. Only Python scalar arithmetic is used, so this
    stays available before NumPy is imported and allocates nothing.
    """
    if count < 2:
        return None, None
    divisor = float(count)
    mean_y = sum_y / divisor
    mean_x = sum_x / divisor
    denominator = sum_yy - sum_y * mean_y
    if not math.isfinite(denominator) or denominator <= 0.0:
        return None, None
    slope = (sum_xy - sum_x * mean_y) / denominator
    intercept = mean_x - slope * mean_y
    if not math.isfinite(slope) or not math.isfinite(intercept):
        return None, None
    return intercept, slope


_BINARY64_UNIT_ROUNDOFF = 2.0**-53


@dataclass(frozen=True, slots=True)
class _TrackFit:
    """Immutable sufficient statistics for one partial or completed track.

    Appending one representative is O(1): the five accumulators, the occupied
    bin set, and the raster-uncertainty maximum are all updated without
    revisiting existing members, constructing a NumPy array, calling
    :func:`_closed_form_fit`, or touching module-level state. Because every
    track in this module is built by appending representatives in increasing
    bin order, folding from scratch and extending incrementally traverse the
    same values in the same order and therefore agree bit for bit.
    """

    count: int
    sum_y: float
    sum_x: float
    sum_yy: float
    sum_xy: float
    intercept: float | None
    slope: float | None
    # Legal bin occupancy as a bit set; invalid private inputs set no bit.
    bin_mask: int
    maximum_raster_tolerance: float
    normalized_coordinates: bool

    def extended(
        self,
        representative: _CandidateRepresentative,
        vertical_bin_count: int = DEFAULT_LANE_CONFIG.vertical_bin_count,
    ) -> "_TrackFit":
        """Append one representative, failing closed for an invalid bin index.

        ``vertical_bin_count`` is the active configuration bound, not merely
        the wider implementation maximum.  The default preserves the ordinary
        private-helper behavior for tracks constructed outside a perception
        call; configured searches pass their validated bound explicitly.
        """
        x = representative.x
        y = representative.y
        count = self.count + 1
        sum_y = self.sum_y + y
        sum_x = self.sum_x + x
        sum_yy = self.sum_yy + y * y
        sum_xy = self.sum_xy + x * y
        intercept, slope = _scalar_line(count, sum_y, sum_x, sum_yy, sum_xy)
        bin_index = representative.bin_index
        bin_mask = self.bin_mask
        if (
            type(vertical_bin_count) is int
            and 0 < vertical_bin_count <= MAX_VERTICAL_BINS
            and type(bin_index) is int
            and 0 <= bin_index < vertical_bin_count
        ):
            bin_mask |= 1 << bin_index
        return _TrackFit(
            count=count,
            sum_y=sum_y,
            sum_x=sum_x,
            sum_yy=sum_yy,
            sum_xy=sum_xy,
            intercept=intercept,
            slope=slope,
            bin_mask=bin_mask,
            maximum_raster_tolerance=max(
                self.maximum_raster_tolerance,
                representative.raster_tolerance,
            ),
            normalized_coordinates=(
                self.normalized_coordinates
                and math.isfinite(x)
                and math.isfinite(y)
                and 0.0 <= x <= 1.0
                and 0.0 <= y <= 1.0
            ),
        )


_EMPTY_TRACK_FIT = _TrackFit(
    count=0,
    sum_y=0.0,
    sum_x=0.0,
    sum_yy=0.0,
    sum_xy=0.0,
    intercept=None,
    slope=None,
    bin_mask=0,
    maximum_raster_tolerance=0.0,
    normalized_coordinates=True,
)


@dataclass(frozen=True, slots=True)
class _RepresentativeTrack:
    """One deterministically linked cross-bin candidate structure.

    ``fit`` and ``identity`` are deterministic functions of
    ``representatives``. :func:`_extended_track` supplies both directly so one
    extension costs O(1) beyond the representative-tuple copy; constructing a
    track from representatives alone derives them once here instead.
    """

    representatives: tuple[_CandidateRepresentative, ...]
    fit: _TrackFit | None = None
    identity: tuple[tuple[object, ...], ...] | None = None
    vertical_bin_count: int = DEFAULT_LANE_CONFIG.vertical_bin_count

    def __post_init__(self) -> None:
        if self.fit is None:
            fit = _EMPTY_TRACK_FIT
            for representative in self.representatives:
                fit = fit.extended(representative, self.vertical_bin_count)
            object.__setattr__(self, "fit", fit)
        if self.identity is None:
            object.__setattr__(
                self,
                "identity",
                tuple(
                    _representative_identity(representative)
                    for representative in self.representatives
                ),
            )


@dataclass(frozen=True, slots=True)
class _BoundaryHypothesis:
    intercept: float
    slope: float
    support_count: int
    vertical_span: float
    residual: float
    maximum_error: float
    observed_support: float
    confidence: float
    track: _RepresentativeTrack


@dataclass(frozen=True, slots=True)
class _PairCandidate:
    left: LaneBoundaryObservation
    right: LaneBoundaryObservation
    left_hypothesis: _BoundaryHypothesis
    right_hypothesis: _BoundaryHypothesis
    confidence: float
    center_offset: float
    quality: float


def _is_finite_real(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, numbers.Real)
        and math.isfinite(float(value))
    )


def _is_nonnegative_builtin_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _unusable(
    snapshot: TelemetrySnapshot, reason: LaneObservationReason
) -> LaneObservation:
    return LaneObservation(
        snapshot_revision=snapshot.revision,
        source_frame=None,
        source_simulation_timestamp=None,
        source_receive_monotonic=None,
        state=LaneDetectionState.INPUT_UNUSABLE,
        reason=reason,
        confidence=0.0,
    )


def _status_is_structurally_coherent(status: SensorStatus) -> bool:
    if type(status.name) is not str:
        return False
    if not isinstance(status.kind, SensorKind):
        return False
    if not isinstance(status.lifecycle, SensorLifecycle):
        return False
    if not isinstance(status.readiness, SensorReadiness):
        return False
    if not isinstance(status.freshness, SensorFreshness):
        return False
    if not isinstance(status.operational_health, OperationalHealth):
        return False
    if not _is_finite_real(status.evaluated_monotonic):
        return False
    if not _is_nonnegative_builtin_int(status.accepted_count):
        return False
    if not _is_nonnegative_builtin_int(status.error_count):
        return False
    if status.event_count is not None:
        return False
    for value in (status.age_seconds, status.expected_interval, status.stale_after):
        if value is not None and not _is_finite_real(value):
            return False
    if status.age_seconds is not None and status.age_seconds < 0.0:
        return False
    for value in (status.expected_interval, status.stale_after):
        if value is not None and value <= 0.0:
            return False
    if (
        status.expected_interval is not None
        and status.stale_after is not None
        and status.expected_interval > status.stale_after
    ):
        return False
    return status.reason is None or type(status.reason) is str


def _status_is_usable(status: SensorStatus) -> bool:
    if not (
        status.kind is SensorKind.CONTINUOUS
        and status.lifecycle is SensorLifecycle.ATTACHED
        and status.readiness is SensorReadiness.READY
        and status.freshness is SensorFreshness.FRESH
        and status.operational_health is OperationalHealth.HEALTHY
    ):
        return False
    if status.accepted_count < 1:
        return False
    if (
        status.age_seconds is None
        or status.expected_interval is None
        or status.stale_after is None
    ):
        return False
    return status.age_seconds <= status.stale_after


def _camera_gate(
    snapshot: TelemetrySnapshot, config: LanePerceptionConfig
) -> tuple[CameraFrame | None, LaneObservationReason | None]:
    if type(snapshot.statuses) is not tuple:
        return None, LaneObservationReason.STATUS_INCOHERENT
    if any(not isinstance(status, SensorStatus) for status in snapshot.statuses):
        return None, LaneObservationReason.STATUS_INCOHERENT
    camera_statuses = tuple(
        status for status in snapshot.statuses if status.name == CAMERA_SENSOR_NAME
    )
    if not camera_statuses:
        return None, LaneObservationReason.STATUS_MISSING
    if len(camera_statuses) != 1:
        return None, LaneObservationReason.STATUS_DUPLICATED
    status = camera_statuses[0]
    if not _status_is_structurally_coherent(status):
        return None, LaneObservationReason.STATUS_INCOHERENT
    if not _status_is_usable(status):
        return None, LaneObservationReason.STATUS_UNUSABLE

    frame = snapshot.camera
    if frame is None:
        return None, LaneObservationReason.CAMERA_MISSING
    if not isinstance(frame, CameraFrame) or not isinstance(
        frame.metadata, CameraFrameMetadata
    ):
        return None, LaneObservationReason.CAMERA_METADATA_INVALID
    metadata = frame.metadata
    if not isinstance(metadata.stamp, SampleStamp):
        return None, LaneObservationReason.CAMERA_METADATA_INVALID
    if metadata.pixel_format != PIXEL_FORMAT_RGB8:
        return None, LaneObservationReason.CAMERA_METADATA_INVALID
    if not (
        type(metadata.width_px) is int
        and metadata.width_px > 0
        and type(metadata.height_px) is int
        and metadata.height_px > 0
        and type(metadata.row_stride_bytes) is int
        and metadata.row_stride_bytes == metadata.width_px * 3
    ):
        return None, LaneObservationReason.CAMERA_METADATA_INVALID
    if not _is_finite_real(metadata.horizontal_fov_deg) or not (
        0.0 < float(metadata.horizontal_fov_deg) <= 180.0
    ):
        return None, LaneObservationReason.CAMERA_METADATA_INVALID
    stamp = metadata.stamp
    if not _is_nonnegative_builtin_int(stamp.frame):
        return None, LaneObservationReason.CAMERA_METADATA_INVALID
    if not _is_finite_real(stamp.sim_time) or not _is_finite_real(stamp.monotonic):
        return None, LaneObservationReason.CAMERA_METADATA_INVALID
    pixel_count = metadata.width_px * metadata.height_px
    if pixel_count > config.max_image_pixels:
        return None, LaneObservationReason.IMAGE_TOO_LARGE
    if type(frame.rgb_bytes) is not bytes:
        return None, LaneObservationReason.CAMERA_PAYLOAD_INVALID
    if len(frame.rgb_bytes) != pixel_count * 3:
        return None, LaneObservationReason.CAMERA_PAYLOAD_INVALID
    return frame, None


def _roi_pixel_bounds(
    width: int, height: int, config: LanePerceptionConfig
) -> tuple[int, int, int, int]:
    left, top, right, bottom = config.roi
    x0 = max(0, min(width - 1, int(math.floor(left * width))))
    y0 = max(0, min(height - 1, int(math.floor(top * height))))
    x1 = max(x0 + 1, min(width, int(math.ceil(right * width))))
    y1 = max(y0 + 1, min(height, int(math.ceil(bottom * height))))
    return x0, y0, x1, y1


def _candidate_representatives(
    frame: CameraFrame, config: LanePerceptionConfig
) -> tuple[_CandidateRepresentative, ...]:
    """Return bounded representatives for disconnected per-bin row tracks.

    Candidate pixels are first split into contiguous runs on every raster row.
    Runs on consecutive rows are matched one-to-one by center displacement,
    bounded by the configured maximum slope. This preserves two parallel
    markings even when their x projections overlap across a tall bin. Every
    component receives a deterministic local index, x extent, and an actual
    observed pixel nearest its centroid. Only the strongest eight components
    per bin survive, ordered without relying on image-center position.
    """
    import numpy

    metadata = frame.metadata
    width = metadata.width_px
    height = metadata.height_px
    image = numpy.frombuffer(frame.rgb_bytes, dtype=numpy.uint8).reshape(
        (height, width, 3)
    )
    x0, y0, x1, y1 = _roi_pixel_bounds(width, height, config)
    pixels = image[y0:y1, x0:x1]
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]

    white_min = config.white_min_rgb
    channel_max = numpy.maximum(numpy.maximum(red, green), blue)
    channel_min = numpy.minimum(numpy.minimum(red, green), blue)
    white = (
        (red >= white_min[0])
        & (green >= white_min[1])
        & (blue >= white_min[2])
        & ((channel_max.astype(numpy.int16) - channel_min.astype(numpy.int16))
           <= config.white_max_channel_spread)
    )
    yellow_min = config.yellow_min_rgb
    yellow_max = config.yellow_max_rgb
    yellow = (
        (red >= yellow_min[0])
        & (red <= yellow_max[0])
        & (green >= yellow_min[1])
        & (green <= yellow_max[1])
        & (blue >= yellow_min[2])
        & (blue <= yellow_max[2])
        & (
            numpy.abs(red.astype(numpy.int16) - green.astype(numpy.int16))
            <= config.yellow_max_red_green_delta
        )
    )
    candidate_mask = white | yellow
    if not bool(numpy.any(candidate_mask)):
        return ()

    x_denominator = float(max(1, width - 1))
    y_denominator = float(max(1, height - 1))
    roi_height_px, _ = candidate_mask.shape
    maximum_column_step = max(
        1,
        int(
            math.ceil(
                config.max_abs_slope
                * x_denominator
                / float(max(1, height - 1))
            )
        ),
    )
    raster_tolerance = (
        0.5 / x_denominator
        + 0.5 * config.max_abs_slope / y_denominator
    )
    representatives: list[_CandidateRepresentative] = []
    for bin_index in range(config.vertical_bin_count):
        row_start = (bin_index * roi_height_px) // config.vertical_bin_count
        row_end = (
            (bin_index + 1) * roi_height_px
        ) // config.vertical_bin_count
        if row_end <= row_start:
            continue
        components: dict[int, dict[str, object]] = {}
        active_component_ids: tuple[int, ...] = ()
        next_component_id = 0
        for local_row in range(row_start, row_end):
            occupied = numpy.flatnonzero(candidate_mask[local_row, :])
            if occupied.size == 0:
                active_component_ids = ()
                continue
            splits = numpy.flatnonzero(numpy.diff(occupied) > 1) + 1
            row_groups = numpy.split(occupied, splits)
            runs = tuple(
                (
                    int(group[0]),
                    int(group[-1]),
                    (float(group[0]) + float(group[-1])) * 0.5,
                    int(group.size),
                )
                for group in row_groups
                if group.size
            )

            # Run centers ascend with x, so only a bounded window of runs can
            # satisfy the one-row displacement bound. Narrowing to that window
            # keeps the identical acceptance test below and therefore builds
            # the identical match list, without scanning unreachable runs.
            run_centers = [run[2] for run in runs]
            possible_matches: list[tuple[tuple[float, ...], int, int]] = []
            for component_id in active_component_ids:
                component = components[component_id]
                previous_center = float(component["last_center"])
                previous_width = int(component["last_width"])
                first_index = bisect.bisect_left(
                    run_centers, previous_center - maximum_column_step
                )
                last_index = bisect.bisect_right(
                    run_centers, previous_center + maximum_column_step
                )
                for run_index in range(first_index, last_index):
                    minimum_x, maximum_x, center_x, run_width = runs[run_index]
                    center_delta = abs(center_x - previous_center)
                    if center_delta > maximum_column_step:
                        continue
                    key = (
                        center_delta,
                        float(abs(run_width - previous_width)),
                        float(component_id),
                        float(minimum_x),
                        float(maximum_x),
                    )
                    possible_matches.append((key, component_id, run_index))
            possible_matches.sort(key=lambda match: match[0])
            matched_components: set[int] = set()
            matched_runs: set[int] = set()
            run_components: dict[int, int] = {}
            for _, component_id, run_index in possible_matches:
                if component_id in matched_components or run_index in matched_runs:
                    continue
                matched_components.add(component_id)
                matched_runs.add(run_index)
                run_components[run_index] = component_id

            next_active: list[int] = []
            for run_index, (minimum_x, maximum_x, center_x, run_width) in enumerate(runs):
                component_id = run_components.get(run_index)
                if component_id is None:
                    component_id = next_component_id
                    next_component_id += 1
                    components[component_id] = {
                        "runs": [],
                        "pixel_count": 0,
                        "sum_x": 0.0,
                        "sum_y": 0.0,
                        "minimum_x": minimum_x,
                        "maximum_x": maximum_x,
                        "last_center": center_x,
                        "last_width": run_width,
                    }
                component = components[component_id]
                component_runs = component["runs"]
                assert isinstance(component_runs, list)
                component_runs.append(
                    (local_row, minimum_x, maximum_x)
                )
                component["pixel_count"] = int(component["pixel_count"]) + run_width
                component["sum_x"] = float(component["sum_x"]) + (
                    (minimum_x + maximum_x) * run_width * 0.5
                )
                component["sum_y"] = float(component["sum_y"]) + (
                    local_row * run_width
                )
                component["minimum_x"] = min(
                    int(component["minimum_x"]), minimum_x
                )
                component["maximum_x"] = max(
                    int(component["maximum_x"]), maximum_x
                )
                component["last_center"] = center_x
                component["last_width"] = run_width
                next_active.append(component_id)
            active_component_ids = tuple(next_active)

        unindexed: list[_CandidateRepresentative] = []
        for component in components.values():
            pixel_count = int(component["pixel_count"])
            component_runs = component["runs"]
            assert isinstance(component_runs, list)
            if pixel_count <= 0 or not component_runs:
                continue
            mean_x = float(component["sum_x"]) / pixel_count
            mean_y = float(component["sum_y"]) / pixel_count
            best_pixel: tuple[float, int, int] | None = None
            for local_row, minimum_x, maximum_x in component_runs:
                nearest_x = max(minimum_x, min(maximum_x, int(round(mean_x))))
                distance = (
                    ((nearest_x - mean_x) / x_denominator) ** 2
                    + ((local_row - mean_y) / y_denominator) ** 2
                )
                candidate = (distance, nearest_x, local_row)
                if best_pixel is None or candidate < best_pixel:
                    best_pixel = candidate
            assert best_pixel is not None
            _, observed_local_x, observed_local_y = best_pixel
            observed_x = observed_local_x + x0
            observed_y = observed_local_y + y0
            unindexed.append(
                _CandidateRepresentative(
                    bin_index=bin_index,
                    cluster_index=-1,
                    x=float(observed_x / x_denominator),
                    y=float(observed_y / y_denominator),
                    pixel_count=pixel_count,
                    row_count=len(component_runs),
                    bin_row_count=row_end - row_start,
                    minimum_x=float(
                        (int(component["minimum_x"]) + x0) / x_denominator
                    ),
                    maximum_x=float(
                        (int(component["maximum_x"]) + x0) / x_denominator
                    ),
                    raster_tolerance=float(raster_tolerance),
                )
            )

        unindexed.sort(
            key=lambda representative: (
                representative.x,
                representative.y,
                representative.minimum_x,
                representative.maximum_x,
                -representative.row_count,
                -representative.pixel_count,
            )
        )
        retained: list[tuple[tuple[float, ...], _CandidateRepresentative]] = []
        for cluster_index, representative in enumerate(unindexed):
            indexed = _CandidateRepresentative(
                bin_index=representative.bin_index,
                cluster_index=cluster_index,
                x=representative.x,
                y=representative.y,
                pixel_count=representative.pixel_count,
                row_count=representative.row_count,
                bin_row_count=representative.bin_row_count,
                minimum_x=representative.minimum_x,
                maximum_x=representative.maximum_x,
                raster_tolerance=representative.raster_tolerance,
            )
            strength_key = (
                -indexed.row_coverage,
                -float(indexed.pixel_count),
                indexed.x,
                indexed.y,
                float(indexed.cluster_index),
            )
            retained.append((strength_key, indexed))
        retained.sort(key=lambda item: item[0])
        representatives.extend(
            representative
            for _, representative in retained[:MAX_REPRESENTATIVES_PER_BIN]
        )
    return tuple(
        sorted(
            representatives,
            key=lambda representative: (
                representative.bin_index,
                representative.cluster_index,
                representative.x,
                representative.y,
                -representative.row_count,
                -representative.pixel_count,
            ),
        )
    )


def _line_is_oriented(
    slope: float, side: LaneBoundarySide, config: LanePerceptionConfig
) -> bool:
    magnitude = abs(slope)
    if not config.min_abs_slope <= magnitude <= config.max_abs_slope:
        return False
    if side is LaneBoundarySide.LEFT:
        return slope < 0.0
    return slope > 0.0


def _closed_form_fit(
    points: tuple[tuple[float, float], ...]
) -> tuple[float, float] | None:
    import numpy

    ys = numpy.asarray(tuple(point[1] for point in points), dtype=numpy.float64)
    xs = numpy.asarray(tuple(point[0] for point in points), dtype=numpy.float64)
    mean_y = numpy.mean(ys, dtype=numpy.float64)
    mean_x = numpy.mean(xs, dtype=numpy.float64)
    centered_y = ys - mean_y
    denominator = numpy.sum(centered_y * centered_y, dtype=numpy.float64)
    if not bool(numpy.isfinite(denominator)) or float(denominator) <= 0.0:
        return None
    slope = numpy.sum(
        centered_y * (xs - mean_x), dtype=numpy.float64
    ) / denominator
    intercept = mean_x - slope * mean_y
    if not bool(numpy.isfinite(slope)) or not bool(numpy.isfinite(intercept)):
        return None
    return float(intercept), float(slope)


def _projected_endpoints(
    intercept: float, slope: float, config: LanePerceptionConfig
) -> tuple[NormalizedImagePoint, NormalizedImagePoint] | None:
    left, top, right, bottom = config.roi
    top_x = intercept + slope * top
    bottom_x = intercept + slope * bottom
    if not all(math.isfinite(value) for value in (top_x, bottom_x)):
        return None
    if not (left <= top_x <= right and left <= bottom_x <= right):
        return None
    if not (0.0 <= top_x <= 1.0 and 0.0 <= bottom_x <= 1.0):
        return None
    return (
        NormalizedImagePoint(float(top_x), float(top)),
        NormalizedImagePoint(float(bottom_x), float(bottom)),
    )


def _boundary_confidence(
    support: int,
    span: float,
    residual: float,
    slope: float,
    observed_support: float,
    config: LanePerceptionConfig,
) -> float:
    support_score = min(1.0, support / float(config.vertical_bin_count))
    coverage_score = min(1.0, span / (config.roi[3] - config.roi[1]))
    residual_score = max(0.0, 1.0 - residual / config.max_fit_residual)
    midpoint = (config.min_abs_slope + config.max_abs_slope) * 0.5
    half_range = (config.max_abs_slope - config.min_abs_slope) * 0.5
    if half_range == 0.0:
        slope_score = 1.0
    else:
        slope_score = max(0.0, 1.0 - abs(abs(slope) - midpoint) / half_range)
    return float(
        min(
            1.0,
            max(
                0.0,
                0.35 * support_score
                + 0.25 * coverage_score
                + 0.25 * residual_score
                + 0.05 * slope_score
                + 0.10 * max(0.0, min(1.0, observed_support)),
            ),
        )
    )


def _residual_is_acceptable(
    residual: float, config: LanePerceptionConfig
) -> bool:
    return math.isfinite(residual) and 0.0 <= residual <= config.max_fit_residual


def _slope_is_plausible(slope: float, config: LanePerceptionConfig) -> bool:
    return (
        math.isfinite(slope)
        and config.min_abs_slope <= abs(slope) <= config.max_abs_slope
    )


def _coerce_representatives(
    points: tuple[object, ...], config: LanePerceptionConfig
) -> tuple[_CandidateRepresentative, ...]:
    """Canonicalize internal representatives and legacy private-helper points."""
    canonical: dict[tuple[object, ...], _CandidateRepresentative] = {}
    roi_top = config.roi[1]
    roi_height = config.roi[3] - roi_top
    for point in points:
        if isinstance(point, _CandidateRepresentative):
            representative = point
        else:
            if type(point) is not tuple or len(point) != 2:
                continue
            x, y = point
            if not _is_finite_real(x) or not _is_finite_real(y):
                continue
            x = float(x)
            y = float(y)
            bin_index = int(
                math.floor(
                    ((y - roi_top) / roi_height) * config.vertical_bin_count
                )
            )
            bin_index = max(0, min(config.vertical_bin_count - 1, bin_index))
            representative = _CandidateRepresentative(
                bin_index=bin_index,
                cluster_index=-1,
                x=x,
                y=y,
                pixel_count=1,
                row_count=1,
                bin_row_count=1,
                minimum_x=x,
                maximum_x=x,
                raster_tolerance=0.0,
            )
        if not (
            type(representative.bin_index) is int
            and 0 <= representative.bin_index < config.vertical_bin_count
        ):
            continue
        key = (
            representative.bin_index,
            round(representative.x, 15),
            round(representative.y, 15),
            representative.pixel_count,
            representative.row_count,
            representative.bin_row_count,
            round(representative.minimum_x, 15),
            round(representative.maximum_x, 15),
            round(representative.raster_tolerance, 15),
        )
        canonical[key] = representative
    ordered = sorted(
        canonical.values(),
        key=lambda representative: (
            representative.bin_index,
            representative.x,
            representative.y,
            -representative.row_count,
            -representative.pixel_count,
            representative.minimum_x,
            representative.maximum_x,
        ),
    )
    next_cluster_by_bin: dict[int, int] = {}
    reindexed = []
    for representative in ordered:
        cluster_index = next_cluster_by_bin.get(representative.bin_index, 0)
        next_cluster_by_bin[representative.bin_index] = cluster_index + 1
        reindexed.append(
            _CandidateRepresentative(
                bin_index=representative.bin_index,
                cluster_index=cluster_index,
                x=representative.x,
                y=representative.y,
                pixel_count=representative.pixel_count,
                row_count=representative.row_count,
                bin_row_count=representative.bin_row_count,
                minimum_x=representative.minimum_x,
                maximum_x=representative.maximum_x,
                raster_tolerance=representative.raster_tolerance,
            )
        )
    return tuple(reindexed)


def _select_one_per_bin(
    representatives: tuple[_CandidateRepresentative, ...],
    intercept: float,
    slope: float,
    config: LanePerceptionConfig,
) -> tuple[_CandidateRepresentative, ...]:
    selected: dict[int, tuple[tuple[float, ...], _CandidateRepresentative]] = {}
    for representative in representatives:
        error = abs(
            representative.x - (intercept + slope * representative.y)
        )
        if error > config.line_fit_tolerance:
            continue
        key = (
            error,
            -representative.row_coverage,
            -float(representative.pixel_count),
            representative.x,
            representative.y,
        )
        previous = selected.get(representative.bin_index)
        if previous is None or key < previous[0]:
            selected[representative.bin_index] = (key, representative)
    return tuple(selected[index][1] for index in sorted(selected))


def _representative_identity(
    representative: _CandidateRepresentative,
) -> tuple[object, ...]:
    return (
        representative.bin_index,
        representative.cluster_index,
        round(representative.x, 15),
        round(representative.y, 15),
        representative.pixel_count,
        representative.row_count,
    )


def _representatives_can_link(
    first: _CandidateRepresentative,
    second: _CandidateRepresentative,
    config: LanePerceptionConfig,
) -> bool:
    bin_gap = second.bin_index - first.bin_index
    if not 1 <= bin_gap <= MAX_TRACK_SKIPPED_BINS + 1:
        return False
    delta_y = second.y - first.y
    if not math.isfinite(delta_y) or delta_y <= 0.0:
        return False
    displacement_allowance = (
        config.max_abs_slope * delta_y
        + first.raster_tolerance
        + second.raster_tolerance
    )
    return abs(second.x - first.x) <= displacement_allowance


def _bounded_outgoing_links(
    representatives: tuple[_CandidateRepresentative, ...],
    config: LanePerceptionConfig,
) -> dict[_CandidateRepresentative, tuple[_CandidateRepresentative, ...]]:
    """Build a fixed-degree forward graph over nearby vertical bins."""
    by_bin: dict[int, list[_CandidateRepresentative]] = {}
    for representative in representatives:
        by_bin.setdefault(representative.bin_index, []).append(representative)
    links: dict[
        _CandidateRepresentative, tuple[_CandidateRepresentative, ...]
    ] = {}
    for first in representatives:
        retained: list[
            tuple[tuple[float, ...], _CandidateRepresentative]
        ] = []
        for bin_gap in range(1, MAX_TRACK_SKIPPED_BINS + 2):
            candidates = []
            for second in by_bin.get(first.bin_index + bin_gap, ()):
                if not _representatives_can_link(first, second, config):
                    continue
                delta_y = second.y - first.y
                implied_slope = abs(second.x - first.x) / delta_y
                key = (
                    float(bin_gap),
                    abs(second.x - first.x),
                    implied_slope,
                    -second.row_coverage,
                    -float(second.pixel_count),
                    float(second.cluster_index),
                    second.x,
                    second.y,
                )
                candidates.append((key, second))
            candidates.sort(key=lambda candidate: candidate[0])
            retained.extend(candidates[:MAX_TRACK_LINKS_PER_BIN_GAP])
        retained.sort(key=lambda candidate: candidate[0])
        links[first] = tuple(
            second
            for _, second in retained[:MAX_OUTGOING_LINKS_PER_REPRESENTATIVE]
        )
    return links


def _extended_track(
    track: _RepresentativeTrack,
    representative: _CandidateRepresentative,
    identity: tuple[object, ...],
) -> _RepresentativeTrack:
    """Append one representative using O(1) scalar fit and identity updates."""
    return _RepresentativeTrack(
        track.representatives + (representative,),
        track.fit.extended(representative, track.vertical_bin_count),
        track.identity + (identity,),
        track.vertical_bin_count,
    )


def _track_prediction_tolerance(
    track: _RepresentativeTrack,
    next_representative: _CandidateRepresentative,
    config: LanePerceptionConfig,
) -> float:
    raster_tolerance = max(
        track.fit.maximum_raster_tolerance,
        next_representative.raster_tolerance,
    )
    return min(
        config.line_fit_tolerance * 0.5,
        max(1.0e-12, 2.0 * raster_tolerance),
    )


def _extension_fit_error_bounds(
    fit: _TrackFit,
    next_y: float,
) -> tuple[float, float] | None:
    """Bound scalar/accepted slope and prediction disagreement.

    Both fits consume at most ``MAX_VERTICAL_BINS`` normalized points in
    binary64.  For ``k`` rounded operations, the standard forward-error factor
    is ``gamma(k) = k*u / (1 - k*u)``, where ``u = 2**-53``.  The accepted fit
    forms centered products and the scalar fit forms raw moments; eight times
    ``count * gamma(8*count + 32)`` conservatively encloses the absolute
    difference between their numerator and denominator reductions, including
    means, products, and final subtraction.  The quotient perturbation bound
    then gives the slope envelope.  The prediction envelope additionally
    covers both mean-derived intercepts and their multiply/add evaluations.

    ``None`` means the denominator is too close to zero for that envelope to
    prove agreement.  The caller must use the accepted fit in that case.
    This helper is O(1), scalar-only, and intentionally conservative: its job
    is to prove when a fallback is unnecessary, not to estimate typical error.
    """
    count = fit.count
    if (
        count < 2
        or not fit.normalized_coordinates
        or not math.isfinite(next_y)
        or not 0.0 <= next_y <= 1.0
    ):
        return None

    operation_count = 8 * count + 32
    scaled_roundoff = operation_count * _BINARY64_UNIT_ROUNDOFF
    gamma = scaled_roundoff / (1.0 - scaled_roundoff)
    moment_delta = 8.0 * count * gamma

    divisor = float(count)
    mean_y = fit.sum_y / divisor
    denominator = fit.sum_yy - fit.sum_y * mean_y
    numerator = fit.sum_xy - fit.sum_x * mean_y
    accepted_denominator_minimum = denominator - moment_delta
    if (
        not math.isfinite(denominator)
        or not math.isfinite(numerator)
        or accepted_denominator_minimum <= 0.0
    ):
        return None

    slope_delta = (
        moment_delta / accepted_denominator_minimum
        + abs(numerator)
        * moment_delta
        / (denominator * accepted_denominator_minimum)
    )
    slope = fit.slope
    intercept = fit.intercept
    if slope is None or intercept is None or not math.isfinite(slope_delta):
        return None

    # The same gamma envelope bounds the difference between the two means.
    # Extra factors cover the intercept multiply/subtract and both prediction
    # multiply/add evaluations without assuming fused operations.
    mean_delta = 4.0 * gamma
    intercept_delta = 4.0 * (
        mean_delta * (1.0 + abs(slope))
        + slope_delta * (1.0 + mean_delta)
        + gamma * max(1.0, abs(intercept), abs(slope))
    )
    prediction_delta = 4.0 * (
        intercept_delta
        + abs(next_y) * slope_delta
        + gamma
        * max(
            1.0,
            abs(intercept),
            abs(slope),
            abs(intercept + slope * next_y),
        )
    )
    if not math.isfinite(prediction_delta):
        return None
    return slope_delta, prediction_delta


def _authoritative_extension_fit(
    track: _RepresentativeTrack,
    cache: dict[
        tuple[tuple[object, ...], ...], tuple[float, float] | None
    ]
    | None,
) -> tuple[float, float] | None:
    """Return and optionally cache the accepted fit for an ambiguous gate."""
    identity = track.identity
    assert identity is not None
    if cache is not None and identity in cache:
        return cache[identity]
    fitted = _closed_form_fit(
        tuple(
            (representative.x, representative.y)
            for representative in track.representatives
        )
    )
    if cache is not None:
        cache[identity] = fitted
    return fitted


def _authoritative_extension_decision(
    track: _RepresentativeTrack,
    next_representative: _CandidateRepresentative,
    config: LanePerceptionConfig,
    cache: dict[
        tuple[tuple[object, ...], ...], tuple[float, float] | None
    ]
    | None,
) -> bool:
    """Apply the accepted fit and unchanged comparisons after ambiguity."""
    fitted = _authoritative_extension_fit(track, cache)
    if fitted is None:
        return False
    intercept, slope = fitted
    if abs(slope) > config.max_abs_slope:
        return False
    predicted_x = intercept + slope * next_representative.y
    return abs(next_representative.x - predicted_x) <= _track_prediction_tolerance(
        track, next_representative, config
    )


def _track_can_extend(
    track: _RepresentativeTrack,
    next_representative: _CandidateRepresentative,
    config: LanePerceptionConfig,
    authoritative_fit_cache: dict[
        tuple[tuple[object, ...], ...], tuple[float, float] | None
    ]
    | None = None,
) -> bool:
    """Reproduce the accepted extension decision with a scalar fast path.

    Clearly separated fits use only O(1) incremental state.  Undefined,
    ill-conditioned, or numerically boundary-adjacent decisions use the
    accepted centered fit.  A caller-owned cache prevents repeated fallback
    fits for one immutable track during a perception call.
    """
    fit = track.fit
    if not (
        type(next_representative.bin_index) is int
        and 0 <= next_representative.bin_index < config.vertical_bin_count
    ):
        return False
    if not _representatives_can_link(
        track.representatives[-1], next_representative, config
    ):
        return False
    if fit.bin_mask & (1 << next_representative.bin_index):
        return False
    if fit.count < 2:
        return True
    slope = fit.slope
    if slope is None:
        return _authoritative_extension_decision(
            track, next_representative, config, authoritative_fit_cache
        )

    error_bounds = _extension_fit_error_bounds(fit, next_representative.y)
    if error_bounds is None:
        return _authoritative_extension_decision(
            track, next_representative, config, authoritative_fit_cache
        )
    slope_delta, prediction_delta = error_bounds
    slope_magnitude = abs(slope)
    slope_guard = slope_delta + 4.0 * max(
        math.ulp(slope_magnitude), math.ulp(config.max_abs_slope)
    )
    if abs(slope_magnitude - config.max_abs_slope) <= slope_guard:
        return _authoritative_extension_decision(
            track, next_representative, config, authoritative_fit_cache
        )
    if slope_magnitude > config.max_abs_slope:
        return False

    predicted_x = fit.intercept + slope * next_representative.y
    prediction_error = abs(next_representative.x - predicted_x)
    tolerance = _track_prediction_tolerance(track, next_representative, config)
    prediction_guard = prediction_delta + 4.0 * max(
        math.ulp(prediction_error),
        math.ulp(tolerance),
        math.ulp(abs(predicted_x)),
    )
    if abs(prediction_error - tolerance) <= prediction_guard:
        return _authoritative_extension_decision(
            track, next_representative, config, authoritative_fit_cache
        )
    return prediction_error <= tolerance


def _track_quality_key(
    track: _RepresentativeTrack,
) -> tuple[object, ...]:
    """Rank one track without refitting it.

    Coefficients and the member identities are already carried by the track,
    so repeated sort-key evaluation never triggers repeated fitting. The
    residual and observed-support aggregates stay exact ``math.fsum`` passes
    over the track's own members.
    """
    representatives = track.representatives
    fit = track.fit
    count = len(representatives)
    first = representatives[0]
    last = representatives[-1]
    span = last.y - first.y
    missing_bins = last.bin_index - first.bin_index + 1 - count
    slope = fit.slope
    if slope is None:
        residual = math.inf if count >= 2 else 0.0
    else:
        intercept = fit.intercept
        residual = math.sqrt(
            math.fsum(
                (
                    representative.x
                    - (intercept + slope * representative.y)
                )
                ** 2
                for representative in representatives
            )
            / count
        )
    observed_support = math.fsum(
        representative.row_coverage for representative in representatives
    ) / count
    return (
        -count,
        missing_bins,
        -span,
        residual,
        -observed_support,
        track.identity,
    )


def _accepted_track_quality_key(
    track: _RepresentativeTrack,
    cheap_key: tuple[object, ...] | None = None,
    accepted_key_cache: dict[
        tuple[tuple[object, ...], ...], tuple[object, ...]
    ]
    | None = None,
) -> tuple[object, ...]:
    """Return the exact accepted Mission 5 ranking key for one track.

    The hot-path key deliberately uses the carried scalar coefficients. Even a
    last-bit residual difference can change membership or returned order. The
    caller supplies the already-computed cheap key so invariant
    elements are reused.  When a per-perception-call cache is supplied, the
    authoritative key is computed at most once for one immutable track
    identity, even if that track participates in several cap selections.
    """
    if cheap_key is None:
        cheap_key = _track_quality_key(track)
    identity = track.identity
    assert identity is not None
    if accepted_key_cache is not None:
        cached = accepted_key_cache.get(identity)
        if cached is not None:
            return cached
    representatives = track.representatives
    if len(representatives) < 2:
        residual = 0.0
    else:
        fitted = _closed_form_fit(
            tuple(
                (representative.x, representative.y)
                for representative in representatives
            )
        )
        if fitted is None:
            residual = math.inf
        else:
            intercept, slope = fitted
            residual = math.sqrt(
                math.fsum(
                    (
                        representative.x
                        - (intercept + slope * representative.y)
                    )
                    ** 2
                    for representative in representatives
                )
                / len(representatives)
            )
    accepted_key = cheap_key[:3] + (residual,) + cheap_key[4:]
    if accepted_key_cache is not None:
        accepted_key_cache[identity] = accepted_key
    return accepted_key


def _select_tracks_by_accepted_quality(
    tracks: tuple[_RepresentativeTrack, ...],
    limit: int,
    accepted_key_cache: dict[
        tuple[tuple[object, ...], ...], tuple[object, ...]
    ]
    | None = None,
) -> tuple[_RepresentativeTrack, ...]:
    """Return the exactly ordered accepted hard-cap prefix.

    The first three key elements are identical for the incremental and
    authoritative fits, so a residual can change the retained set only when
    the exact-prefix group at the cutoff extends to both sides of it. Cheap
    keys therefore determine membership except for that crossing group, which
    is authoritatively reranked before the cap is applied. Every selected group
    sharing the fit-independent prefix is then sorted by complete accepted
    keys; singleton groups already have an invariant position. This reproduces
    the accepted order for non-crossing groups and non-binding selections too.

    A caller-owned cache permits reuse across the three cap sites in one
    perception call. Standalone private-helper calls receive an ephemeral
    selection-local cache.
    """
    if limit <= 0 or not tracks:
        return ()
    if accepted_key_cache is None:
        accepted_key_cache = {}
    ranked = [(_track_quality_key(track), track) for track in tracks]
    ranked.sort(key=lambda item: item[0])
    if len(ranked) <= limit:
        selected = ranked
    else:
        cutoff_prefix = ranked[limit - 1][0][:3]
        if cutoff_prefix == ranked[limit][0][:3]:
            first = limit - 1
            while first > 0 and ranked[first - 1][0][:3] == cutoff_prefix:
                first -= 1
            last = limit + 1
            while last < len(ranked) and ranked[last][0][:3] == cutoff_prefix:
                last += 1
            exact_group = [
                (
                    _accepted_track_quality_key(
                        track, cheap_key, accepted_key_cache
                    ),
                    track,
                )
                for cheap_key, track in ranked[first:last]
            ]
            exact_group.sort(key=lambda item: item[0])
            ranked[first:last] = exact_group
        selected = ranked[:limit]

    exactly_ordered: list[_RepresentativeTrack] = []
    first = 0
    while first < len(selected):
        prefix = selected[first][0][:3]
        last = first + 1
        while last < len(selected) and selected[last][0][:3] == prefix:
            last += 1
        group = selected[first:last]
        if len(group) > 1:
            exact_group = [
                (
                    _accepted_track_quality_key(
                        track, cheap_key, accepted_key_cache
                    ),
                    track,
                )
                for cheap_key, track in group
            ]
            exact_group.sort(key=lambda item: item[0])
            exactly_ordered.extend(track for _, track in exact_group)
        else:
            exactly_ordered.append(group[0][1])
        first = last
    return tuple(exactly_ordered)


def _continuous_representative_tracks(
    representatives: tuple[_CandidateRepresentative, ...],
    config: LanePerceptionConfig,
) -> tuple[_RepresentativeTrack, ...]:
    """Construct bounded prediction-consistent tracks by vertical-bin DP.

    Ordinary extension decisions and every cheap ranking key are answered from
    the track's own O(1) incremental fit state. Numerically ambiguous extension
    gates use a per-call cached accepted fit. Each local cap selection decorates
    cheap quality once per candidate. Exact ranking keys are fitted lazily only
    for crossing groups and selected equal-prefix groups, then reused by
    immutable identity for the rest of this call so every returned cap prefix
    has accepted ordering.
    """
    representatives = tuple(
        representative
        for representative in representatives
        if type(representative.bin_index) is int
        and 0 <= representative.bin_index < config.vertical_bin_count
    )
    accepted_key_cache: dict[
        tuple[tuple[object, ...], ...], tuple[object, ...]
    ] = {}
    authoritative_extension_fit_cache: dict[
        tuple[tuple[object, ...], ...], tuple[float, float] | None
    ] = {}
    links = _bounded_outgoing_links(representatives, config)
    incoming: dict[
        _CandidateRepresentative, list[_CandidateRepresentative]
    ] = {representative: [] for representative in representatives}
    for first, next_values in links.items():
        for second in next_values:
            incoming[second].append(first)
    for values in incoming.values():
        values.sort(key=_representative_identity)

    by_bin: dict[int, list[_CandidateRepresentative]] = {}
    for representative in representatives:
        by_bin.setdefault(representative.bin_index, []).append(representative)
    tracks_by_end: dict[
        _CandidateRepresentative, tuple[_RepresentativeTrack, ...]
    ] = {}
    for bin_index in sorted(by_bin):
        bin_tracks: list[_RepresentativeTrack] = []
        for representative in sorted(
            by_bin[bin_index], key=_representative_identity
        ):
            identity = _representative_identity(representative)
            candidates = [
                _RepresentativeTrack(
                    (representative,),
                    vertical_bin_count=config.vertical_bin_count,
                )
            ]
            for previous in incoming[representative]:
                for track in tracks_by_end.get(previous, ()):
                    if _track_can_extend(
                        track,
                        representative,
                        config,
                        authoritative_extension_fit_cache,
                    ):
                        candidates.append(
                            _extended_track(track, representative, identity)
                        )
            unique: dict[tuple[object, ...], _RepresentativeTrack] = {}
            for track in candidates:
                unique[track.identity] = track
            bin_tracks.extend(
                _select_tracks_by_accepted_quality(
                    tuple(unique.values()),
                    MAX_PARTIAL_TRACKS_PER_REPRESENTATIVE,
                    accepted_key_cache,
                )
            )
        retained_by_end: dict[
            _CandidateRepresentative, list[_RepresentativeTrack]
        ] = {}
        for track in _select_tracks_by_accepted_quality(
            tuple(bin_tracks),
            MAX_PARTIAL_TRACKS_PER_BIN,
            accepted_key_cache,
        ):
            representative = track.representatives[-1]
            retained_by_end.setdefault(representative, []).append(track)
        for representative, retained in retained_by_end.items():
            tracks_by_end[representative] = tuple(retained)

    completed: dict[tuple[object, ...], _RepresentativeTrack] = {}
    for tracks in tracks_by_end.values():
        for track in tracks:
            if len(track.representatives) < config.min_support_bins:
                continue
            span = track.representatives[-1].y - track.representatives[0].y
            if not math.isfinite(span) or span < config.min_vertical_span:
                continue
            completed[track.identity] = track
    return _select_tracks_by_accepted_quality(
        tuple(completed.values()), MAX_COMPLETED_TRACKS, accepted_key_cache
    )


def _maximum_individual_error_tolerance(
    track: _RepresentativeTrack, config: LanePerceptionConfig
) -> float:
    return min(
        config.line_fit_tolerance,
        max(1.0e-12, 3.0 * track.fit.maximum_raster_tolerance),
    )


def _track_support_is_valid(
    track: _RepresentativeTrack,
    intercept: float,
    slope: float,
    config: LanePerceptionConfig,
) -> tuple[bool, float]:
    """Re-validate one retained track exactly, in bounded O(n) work.

    Distinct ordered bins, the skipped-bin bound, the full continuity walk,
    per-member error against the authoritative coefficients, and complete
    observed-row support are all still checked here. Only the per-step cost
    changed: each prefix extension is now O(1).
    """
    representatives = track.representatives
    bins = tuple(representative.bin_index for representative in representatives)
    if len(set(bins)) != len(bins) or bins != tuple(sorted(bins)):
        return False, math.inf
    if any(
        second - first - 1 > MAX_TRACK_SKIPPED_BINS
        for first, second in zip(bins, bins[1:])
    ):
        return False, math.inf
    prefix = _RepresentativeTrack(
        (representatives[0],),
        vertical_bin_count=config.vertical_bin_count,
    )
    authoritative_extension_fit_cache: dict[
        tuple[tuple[object, ...], ...], tuple[float, float] | None
    ] = {}
    for representative in representatives[1:]:
        if not _track_can_extend(
            prefix,
            representative,
            config,
            authoritative_extension_fit_cache,
        ):
            return False, math.inf
        prefix = _extended_track(
            prefix, representative, _representative_identity(representative)
        )
    errors = tuple(
        abs(representative.x - (intercept + slope * representative.y))
        for representative in representatives
    )
    maximum_error = max(errors, default=math.inf)
    tolerance = _maximum_individual_error_tolerance(track, config)
    if not math.isfinite(maximum_error) or maximum_error > tolerance:
        return False, maximum_error
    supported_rows = sum(error <= tolerance for error in errors)
    if supported_rows != len(representatives):
        return False, maximum_error
    return True, float(maximum_error)


def _hypotheses_equivalent(
    first: _BoundaryHypothesis,
    second: _BoundaryHypothesis,
    config: LanePerceptionConfig,
) -> bool:
    tolerance = min(0.01, config.line_fit_tolerance * 0.25)
    for y in (config.roi[1], config.center_evaluation_y, config.roi[3]):
        first_x = first.intercept + first.slope * y
        second_x = second.intercept + second.slope * y
        if abs(first_x - second_x) > tolerance:
            return False
    return True


def _fit_boundary_hypotheses(
    points: tuple[object, ...], config: LanePerceptionConfig
) -> tuple[_BoundaryHypothesis, ...]:
    """Fit orientation-neutral lines only from continuity-valid tracks.

    The cross-bin graph has bounded degree and skipped-bin distance. Dynamic
    programming retains a fixed number of prediction-consistent partial and
    completed tracks. A completed track is one hypothesis seed; unlike the old
    global inlier selection, fitting never substitutes a nearby representative
    from another track. Every contributing bin supplies exactly one observed
    representative, and both RMS and maximum individual error are gated.
    """
    canonical = _coerce_representatives(points, config)
    by_bin: dict[int, list[_CandidateRepresentative]] = {}
    for representative in canonical:
        by_bin.setdefault(representative.bin_index, []).append(representative)
    bounded = []
    for bin_index in sorted(by_bin):
        values = sorted(
            by_bin[bin_index],
            key=lambda representative: (
                -representative.row_coverage,
                -representative.pixel_count,
                representative.x,
                representative.y,
                representative.cluster_index,
            ),
        )[:MAX_REPRESENTATIVES_PER_BIN]
        bounded.extend(values)
    representatives = tuple(
        sorted(
            bounded,
            key=lambda representative: (
                representative.bin_index,
                representative.x,
                representative.y,
                representative.cluster_index,
            ),
        )
    )
    occupied_bins = {
        representative.bin_index for representative in representatives
    }
    if len(occupied_bins) < config.min_support_bins:
        return ()

    tracks = _continuous_representative_tracks(representatives, config)
    fitted_candidates: list[
        tuple[tuple[float, ...], _BoundaryHypothesis]
    ] = []
    for track in tracks[:MAX_BOUNDARY_SEEDS]:
        inliers = track.representatives
        fitted = _closed_form_fit(
            tuple((representative.x, representative.y) for representative in inliers)
        )
        if fitted is None:
            continue
        intercept, slope = fitted
        if not _slope_is_plausible(slope, config):
            continue
        first_y = inliers[0].y
        last_y = inliers[-1].y
        span = last_y - first_y
        if not math.isfinite(span) or span < config.min_vertical_span:
            continue
        squared_error = math.fsum(
            (
                representative.x
                - (intercept + slope * representative.y)
            )
            ** 2
            for representative in inliers
        )
        residual = math.sqrt(squared_error / len(inliers))
        if not _residual_is_acceptable(residual, config):
            continue
        support_valid, maximum_error = _track_support_is_valid(
            track, intercept, slope, config
        )
        if not support_valid:
            continue
        if _projected_endpoints(intercept, slope, config) is None:
            continue
        observed_support = math.fsum(
            representative.row_coverage for representative in inliers
        ) / len(inliers)
        confidence = _boundary_confidence(
            len(inliers),
            span,
            residual,
            slope,
            observed_support,
            config,
        )
        if confidence < config.min_boundary_confidence:
            continue
        hypothesis = _BoundaryHypothesis(
            intercept=float(intercept),
            slope=float(slope),
            support_count=int(len(inliers)),
            vertical_span=float(span),
            residual=float(residual),
            maximum_error=float(maximum_error),
            observed_support=float(observed_support),
            confidence=float(confidence),
            track=track,
        )
        bins = tuple(
            representative.bin_index for representative in inliers
        )
        missing_bins = bins[-1] - bins[0] + 1 - len(bins)
        quality_key = (
            -float(hypothesis.support_count),
            float(missing_bins),
            hypothesis.residual,
            hypothesis.maximum_error,
            -hypothesis.vertical_span,
            -hypothesis.observed_support,
            -hypothesis.confidence,
            hypothesis.intercept,
            hypothesis.slope,
        )
        fitted_candidates.append((quality_key, hypothesis))

    fitted_candidates.sort(key=lambda candidate: candidate[0])
    deduplicated: list[_BoundaryHypothesis] = []
    for _, hypothesis in fitted_candidates:
        if any(
            _hypotheses_equivalent(hypothesis, retained, config)
            for retained in deduplicated
        ):
            continue
        deduplicated.append(hypothesis)
        if len(deduplicated) >= MAX_BOUNDARY_HYPOTHESES:
            break
    return tuple(deduplicated)


def _boundary_from_hypothesis(
    hypothesis: _BoundaryHypothesis,
    side: LaneBoundarySide,
    config: LanePerceptionConfig,
) -> LaneBoundaryObservation:
    endpoints = _projected_endpoints(
        hypothesis.intercept, hypothesis.slope, config
    )
    assert endpoints is not None
    return LaneBoundaryObservation(
        side=side,
        endpoints=endpoints,
        x_intercept=hypothesis.intercept,
        x_slope=hypothesis.slope,
        support_count=hypothesis.support_count,
        fit_residual=hypothesis.residual,
        confidence=hypothesis.confidence,
    )


def _fit_boundary(
    points: tuple[tuple[float, float], ...],
    side: LaneBoundarySide,
    config: LanePerceptionConfig,
) -> LaneBoundaryObservation | None:
    """Compatibility wrapper for fitting one requested oriented boundary."""
    for hypothesis in _fit_boundary_hypotheses(points, config):
        if _line_is_oriented(hypothesis.slope, side, config):
            return _boundary_from_hypothesis(hypothesis, side, config)
    return None


def _boundary_x(boundary: LaneBoundaryObservation, y: float) -> float:
    return boundary.x_intercept + boundary.x_slope * y


def _pair_geometry(
    left: LaneBoundaryObservation,
    right: LaneBoundaryObservation,
    config: LanePerceptionConfig,
) -> tuple[
    bool,
    float,
    tuple[NormalizedImagePoint, NormalizedImagePoint] | None,
    float | None,
    float | None,
]:
    if not (
        _line_is_oriented(left.x_slope, LaneBoundarySide.LEFT, config)
        and _line_is_oriented(right.x_slope, LaneBoundarySide.RIGHT, config)
    ):
        return False, 0.0, None, None, None
    roi_top = config.roi[1]
    roi_bottom = config.roi[3]
    sample_ys = tuple(
        sorted(
            {
                roi_top,
                (roi_top + roi_bottom) * 0.5,
                config.center_evaluation_y,
                roi_bottom,
            }
        )
    )
    widths = []
    for y in sample_ys:
        left_x = _boundary_x(left, y)
        right_x = _boundary_x(right, y)
        width = right_x - left_x
        center_x = (left_x + right_x) * 0.5
        if not all(
            math.isfinite(value)
            for value in (left_x, right_x, width, center_x)
        ):
            return False, 0.0, None, None, None
        if not left_x < right_x:
            return False, 0.0, None, None, None
        if not config.min_lane_width <= width <= config.max_lane_width:
            return False, 0.0, None, None, None
        if not 0.0 <= center_x <= 1.0:
            return False, 0.0, None, None, None
        widths.append(width)

    top_center_x = (_boundary_x(left, roi_top) + _boundary_x(right, roi_top)) * 0.5
    bottom_center_x = (
        _boundary_x(left, roi_bottom) + _boundary_x(right, roi_bottom)
    ) * 0.5
    evaluation_center_x = (
        _boundary_x(left, config.center_evaluation_y)
        + _boundary_x(right, config.center_evaluation_y)
    ) * 0.5
    center_offset = evaluation_center_x - 0.5
    heading_proxy = (bottom_center_x - top_center_x) / (roi_bottom - roi_top)
    if not all(
        math.isfinite(value)
        for value in (
            top_center_x,
            bottom_center_x,
            center_offset,
            heading_proxy,
        )
    ):
        return False, 0.0, None, None, None
    if not (
        0.0 <= top_center_x <= 1.0
        and 0.0 <= bottom_center_x <= 1.0
        and -1.0 <= center_offset <= 1.0
        and -1.0 <= heading_proxy <= 1.0
    ):
        return False, 0.0, None, None, None

    width_consistency = 1.0 - min(
        1.0, (max(widths) - min(widths)) / config.max_lane_width
    )
    slope_consistency = 1.0 - min(
        1.0,
        abs(abs(left.x_slope) - abs(right.x_slope))
        / (2.0 * config.max_abs_slope),
    )
    pair_confidence = 0.5 * width_consistency + 0.5 * slope_consistency
    aggregate = float(
        min(left.confidence, right.confidence, max(0.0, min(1.0, pair_confidence)))
    )
    centerline = (
        NormalizedImagePoint(float(top_center_x), float(roi_top)),
        NormalizedImagePoint(float(bottom_center_x), float(roi_bottom)),
    )
    return (
        True,
        aggregate,
        centerline,
        float(center_offset),
        float(heading_proxy),
    )


def _pair_quality(
    left: _BoundaryHypothesis,
    right: _BoundaryHypothesis,
    confidence: float,
    center_offset: float,
    config: LanePerceptionConfig,
) -> float:
    support_score = min(
        1.0,
        min(left.support_count, right.support_count)
        / float(config.vertical_bin_count),
    )
    roi_height = config.roi[3] - config.roi[1]
    span_score = min(
        1.0,
        min(left.vertical_span, right.vertical_span) / roi_height,
    )
    residual_score = max(
        0.0,
        1.0
        - ((left.residual + right.residual) * 0.5)
        / config.max_fit_residual,
    )
    center_score = 1.0 - min(
        1.0,
        abs(center_offset) / max(0.001, config.max_lane_width * 0.5),
    )
    evaluation_y = config.center_evaluation_y
    width = (
        right.intercept
        + right.slope * evaluation_y
        - left.intercept
        - left.slope * evaluation_y
    )
    width_midpoint = (config.min_lane_width + config.max_lane_width) * 0.5
    width_half_range = (config.max_lane_width - config.min_lane_width) * 0.5
    width_score = (
        1.0
        if width_half_range == 0.0
        else max(0.0, 1.0 - abs(width - width_midpoint) / width_half_range)
    )
    return float(
        0.55 * confidence
        + 0.15 * support_score
        + 0.10 * span_score
        + 0.10 * residual_score
        + 0.05 * center_score
        + 0.05 * width_score
    )


def _pair_is_materially_different(
    first: _PairCandidate,
    second: _PairCandidate,
    config: LanePerceptionConfig,
) -> bool:
    evaluation_y = config.center_evaluation_y
    positions = (
        abs(first.center_offset - second.center_offset),
        abs(
            _boundary_x(first.left, evaluation_y)
            - _boundary_x(second.left, evaluation_y)
        ),
        abs(
            _boundary_x(first.right, evaluation_y)
            - _boundary_x(second.right, evaluation_y)
        ),
    )
    return max(positions) >= AMBIGUITY_POSITION_DELTA


def _select_boundaries(
    hypotheses: tuple[_BoundaryHypothesis, ...],
    config: LanePerceptionConfig,
) -> tuple[
    LaneBoundaryObservation | None,
    LaneBoundaryObservation | None,
    bool,
]:
    """Order completed lines at the evaluation row, then select one pair."""
    if not hypotheses:
        return None, None, False
    if len(hypotheses) == 1:
        hypothesis = hypotheses[0]
        if hypothesis.slope < 0.0:
            return (
                _boundary_from_hypothesis(
                    hypothesis, LaneBoundarySide.LEFT, config
                ),
                None,
                False,
            )
        return (
            None,
            _boundary_from_hypothesis(
                hypothesis, LaneBoundarySide.RIGHT, config
            ),
            False,
        )

    valid_candidates: list[
        tuple[tuple[float, ...], _PairCandidate]
    ] = []
    invalid_candidates: list[
        tuple[
            tuple[float, ...],
            LaneBoundaryObservation,
            LaneBoundaryObservation,
        ]
    ] = []
    evaluation_y = config.center_evaluation_y
    pair_combinations = 0
    pair_limit_reached = False
    for first_index, first_hypothesis in enumerate(hypotheses):
        for second_hypothesis in hypotheses[first_index + 1 :]:
            if pair_combinations >= MAX_PAIR_COMBINATIONS:
                pair_limit_reached = True
                break
            pair_combinations += 1
            first_x = (
                first_hypothesis.intercept
                + first_hypothesis.slope * evaluation_y
            )
            second_x = (
                second_hypothesis.intercept
                + second_hypothesis.slope * evaluation_y
            )
            if (first_x, first_hypothesis.intercept, first_hypothesis.slope) <= (
                second_x,
                second_hypothesis.intercept,
                second_hypothesis.slope,
            ):
                left_hypothesis = first_hypothesis
                right_hypothesis = second_hypothesis
            else:
                left_hypothesis = second_hypothesis
                right_hypothesis = first_hypothesis
            left = _boundary_from_hypothesis(
                left_hypothesis, LaneBoundarySide.LEFT, config
            )
            right = _boundary_from_hypothesis(
                right_hypothesis, LaneBoundarySide.RIGHT, config
            )
            invalid_key = (
                -float(
                    min(
                        left_hypothesis.support_count,
                        right_hypothesis.support_count,
                    )
                ),
                left_hypothesis.residual + right_hypothesis.residual,
                -left_hypothesis.vertical_span - right_hypothesis.vertical_span,
                left_hypothesis.intercept,
                left_hypothesis.slope,
                right_hypothesis.intercept,
                right_hypothesis.slope,
            )
            invalid_candidates.append((invalid_key, left, right))
            valid, confidence, _, center_offset, _ = _pair_geometry(
                left, right, config
            )
            if not valid:
                continue
            assert center_offset is not None
            quality = _pair_quality(
                left_hypothesis,
                right_hypothesis,
                confidence,
                center_offset,
                config,
            )
            candidate = _PairCandidate(
                left=left,
                right=right,
                left_hypothesis=left_hypothesis,
                right_hypothesis=right_hypothesis,
                confidence=confidence,
                center_offset=center_offset,
                quality=quality,
            )
            candidate_key = (
                -quality,
                -confidence,
                -float(
                    min(
                        left_hypothesis.support_count,
                        right_hypothesis.support_count,
                    )
                ),
                -float(
                    left_hypothesis.support_count
                    + right_hypothesis.support_count
                ),
                left_hypothesis.residual + right_hypothesis.residual,
                abs(center_offset),
                left_hypothesis.intercept,
                left_hypothesis.slope,
                right_hypothesis.intercept,
                right_hypothesis.slope,
            )
            valid_candidates.append((candidate_key, candidate))
        if pair_limit_reached:
            break

    if not valid_candidates:
        invalid_candidates.sort(key=lambda candidate: candidate[0])
        _, left, right = invalid_candidates[0]
        return left, right, False

    valid_candidates.sort(key=lambda candidate: candidate[0])
    best = valid_candidates[0][1]
    best_minimum_support = min(
        best.left_hypothesis.support_count,
        best.right_hypothesis.support_count,
    )
    ambiguous = False
    for _, competitor in valid_candidates[1:]:
        competitor_minimum_support = min(
            competitor.left_hypothesis.support_count,
            competitor.right_hypothesis.support_count,
        )
        if competitor.quality < best.quality - AMBIGUITY_QUALITY_MARGIN:
            continue
        if competitor.confidence < best.confidence - AMBIGUITY_CONFIDENCE_MARGIN:
            continue
        if competitor_minimum_support < best_minimum_support - 1:
            continue
        if _pair_is_materially_different(best, competitor, config):
            ambiguous = True
            break
    return best.left, best.right, ambiguous


def _observation_from_boundaries(
    snapshot: TelemetrySnapshot,
    frame: CameraFrame,
    left: LaneBoundaryObservation | None,
    right: LaneBoundaryObservation | None,
    config: LanePerceptionConfig,
    *,
    ambiguous: bool = False,
) -> LaneObservation:
    stamp = frame.metadata.stamp
    common = dict(
        snapshot_revision=snapshot.revision,
        source_frame=stamp.frame,
        source_simulation_timestamp=stamp.sim_time,
        source_receive_monotonic=stamp.monotonic,
        left_boundary=left,
        right_boundary=right,
    )
    boundary_count = int(left is not None) + int(right is not None)
    if boundary_count == 0:
        return LaneObservation(
            **common,
            state=LaneDetectionState.NOT_DETECTED,
            reason=LaneObservationReason.NO_PLAUSIBLE_BOUNDARY,
            confidence=0.0,
        )
    if boundary_count == 1:
        boundary = left if left is not None else right
        assert boundary is not None
        return LaneObservation(
            **common,
            state=LaneDetectionState.PARTIAL,
            reason=LaneObservationReason.SINGLE_PLAUSIBLE_BOUNDARY,
            confidence=boundary.confidence,
        )
    assert left is not None and right is not None
    valid_pair, confidence, centerline, offset, heading = _pair_geometry(
        left, right, config
    )
    if not valid_pair:
        return LaneObservation(
            **common,
            state=LaneDetectionState.PARTIAL,
            reason=LaneObservationReason.PAIR_GEOMETRY_INVALID,
            confidence=0.0,
        )
    if ambiguous:
        ambiguity_cap = (
            math.nextafter(config.detection_confidence, 0.0)
            if config.detection_confidence > 0.0
            else 0.0
        )
        return LaneObservation(
            **common,
            state=LaneDetectionState.PARTIAL,
            reason=LaneObservationReason.CONFIDENCE_BELOW_THRESHOLD,
            confidence=min(confidence, ambiguity_cap),
        )
    if confidence < config.detection_confidence:
        return LaneObservation(
            **common,
            state=LaneDetectionState.PARTIAL,
            reason=LaneObservationReason.CONFIDENCE_BELOW_THRESHOLD,
            confidence=confidence,
        )
    assert centerline is not None and offset is not None and heading is not None
    return LaneObservation(
        **common,
        state=LaneDetectionState.DETECTED,
        reason=LaneObservationReason.DETECTED,
        centerline_endpoints=centerline,
        center_offset=offset,
        heading_proxy=heading,
        confidence=confidence,
    )


def perceive_lanes(
    snapshot: TelemetrySnapshot,
    config: LanePerceptionConfig = DEFAULT_LANE_CONFIG,
) -> LaneObservation:
    """Return a bounded immutable observation from one coherent snapshot.

    Input evidence fails closed unless exactly one canonical RGB-camera status
    is attached, ready, fresh, healthy, and continuous, and its immutable RGB8
    payload is exactly coherent with finite source metadata. Equal inputs and
    configuration values return equal observations.
    """
    if not isinstance(snapshot, TelemetrySnapshot):
        raise TypeError("snapshot must be a TelemetrySnapshot")
    if not isinstance(config, LanePerceptionConfig):
        raise TypeError("config must be a LanePerceptionConfig")
    if not _is_nonnegative_builtin_int(snapshot.revision):
        raise ValueError("snapshot revision must be a non-negative built-in integer")
    frame, gate_reason = _camera_gate(snapshot, config)
    if gate_reason is not None:
        return _unusable(snapshot, gate_reason)
    assert frame is not None
    representatives = _candidate_representatives(frame, config)
    hypotheses = _fit_boundary_hypotheses(representatives, config)
    left, right, ambiguous = _select_boundaries(hypotheses, config)
    return _observation_from_boundaries(
        snapshot,
        frame,
        left,
        right,
        config,
        ambiguous=ambiguous,
    )
