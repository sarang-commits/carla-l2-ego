"""Frozen public records for deterministic image-space lane observations.

Coordinates are normalized to ``[0.0, 1.0]`` with the origin at the image's
top-left, x increasing rightward, and y increasing downward.  A boundary is
the line ``x(y) = x_intercept + x_slope * y``.  This is image evidence only;
it is not world geometry, vehicle pose, or a steering command.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from enum import Enum


MAX_CONFIGURED_IMAGE_PIXELS = 16_000_000
MAX_VERTICAL_BINS = 64


class LaneDetectionState(Enum):
    INPUT_UNUSABLE = "input_unusable"
    NOT_DETECTED = "not_detected"
    PARTIAL = "partial"
    DETECTED = "detected"


class LaneBoundarySide(Enum):
    LEFT = "left"
    RIGHT = "right"


class LaneObservationReason(Enum):
    DETECTED = "detected"
    STATUS_MISSING = "status_missing"
    STATUS_DUPLICATED = "status_duplicated"
    STATUS_INCOHERENT = "status_incoherent"
    STATUS_UNUSABLE = "status_unusable"
    CAMERA_MISSING = "camera_missing"
    CAMERA_METADATA_INVALID = "camera_metadata_invalid"
    CAMERA_PAYLOAD_INVALID = "camera_payload_invalid"
    IMAGE_TOO_LARGE = "image_too_large"
    NO_PLAUSIBLE_BOUNDARY = "no_plausible_boundary"
    SINGLE_PLAUSIBLE_BOUNDARY = "single_plausible_boundary"
    PAIR_GEOMETRY_INVALID = "pair_geometry_invalid"
    CONFIDENCE_BELOW_THRESHOLD = "confidence_below_threshold"


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _unit_real(value: object, name: str) -> float:
    result = _finite_real(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0]")
    return result


def _positive_real(value: object, name: str, *, upper: float) -> float:
    result = _finite_real(value, name)
    if not 0.0 < result <= upper:
        raise ValueError(f"{name} must be in (0.0, {upper}]")
    return result


def _builtin_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive built-in integer")
    return value


def _rgb_triplet(value: object, name: str) -> tuple[int, int, int]:
    if type(value) is not tuple or len(value) != 3:
        raise ValueError(f"{name} must be a three-item tuple")
    for channel in value:
        if isinstance(channel, bool) or type(channel) is not int:
            raise ValueError(f"{name} channels must be built-in integers")
        if not 0 <= channel <= 255:
            raise ValueError(f"{name} channels must be in [0, 255]")
    return value


@dataclass(frozen=True, slots=True)
class NormalizedImagePoint:
    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _unit_real(self.x, "x"))
        object.__setattr__(self, "y", _unit_real(self.y, "y"))


@dataclass(frozen=True, slots=True)
class LaneBoundaryObservation:
    """One fitted boundary represented by two ROI intersections.

    ``fit_residual`` is normalized-x root-mean-square residual over the fixed
    bin representatives. ``confidence`` is a bounded heuristic score, not a
    calibrated probability.
    """

    side: LaneBoundarySide
    endpoints: tuple[NormalizedImagePoint, NormalizedImagePoint]
    x_intercept: float
    x_slope: float
    support_count: int
    fit_residual: float
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.side, LaneBoundarySide):
            raise ValueError("side must be a LaneBoundarySide")
        if type(self.endpoints) is not tuple or len(self.endpoints) != 2:
            raise ValueError("endpoints must be a two-item tuple")
        if not all(isinstance(point, NormalizedImagePoint) for point in self.endpoints):
            raise ValueError("endpoints must contain NormalizedImagePoint values")
        intercept = _finite_real(self.x_intercept, "x_intercept")
        slope = _finite_real(self.x_slope, "x_slope")
        support = _builtin_positive_int(self.support_count, "support_count")
        residual = _finite_real(self.fit_residual, "fit_residual")
        if residual < 0.0:
            raise ValueError("fit_residual must be non-negative")
        confidence = _unit_real(self.confidence, "confidence")
        object.__setattr__(self, "x_intercept", intercept)
        object.__setattr__(self, "x_slope", slope)
        object.__setattr__(self, "support_count", support)
        object.__setattr__(self, "fit_residual", residual)
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class LaneObservation:
    snapshot_revision: int
    source_frame: int | None
    source_simulation_timestamp: float | None
    source_receive_monotonic: float | None
    state: LaneDetectionState
    reason: LaneObservationReason
    left_boundary: LaneBoundaryObservation | None = None
    right_boundary: LaneBoundaryObservation | None = None
    centerline_endpoints: tuple[NormalizedImagePoint, NormalizedImagePoint] | None = None
    center_offset: float | None = None
    heading_proxy: float | None = None
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.snapshot_revision, bool) or type(self.snapshot_revision) is not int:
            raise ValueError("snapshot_revision must be a built-in integer")
        if self.snapshot_revision < 0:
            raise ValueError("snapshot_revision must be non-negative")
        if self.source_frame is not None:
            if isinstance(self.source_frame, bool) or type(self.source_frame) is not int:
                raise ValueError("source_frame must be a built-in integer or None")
            if self.source_frame < 0:
                raise ValueError("source_frame must be non-negative")
        for name in ("source_simulation_timestamp", "source_receive_monotonic"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_real(value, name))
        if not isinstance(self.state, LaneDetectionState):
            raise ValueError("state must be a LaneDetectionState")
        if not isinstance(self.reason, LaneObservationReason):
            raise ValueError("reason must be a LaneObservationReason")
        for boundary, side in (
            (self.left_boundary, LaneBoundarySide.LEFT),
            (self.right_boundary, LaneBoundarySide.RIGHT),
        ):
            if boundary is not None and (
                not isinstance(boundary, LaneBoundaryObservation)
                or boundary.side is not side
            ):
                raise ValueError("boundary is absent or must match its field side")
        if self.centerline_endpoints is not None:
            if type(self.centerline_endpoints) is not tuple or len(self.centerline_endpoints) != 2:
                raise ValueError("centerline_endpoints must be a two-item tuple or None")
            if not all(
                isinstance(point, NormalizedImagePoint)
                for point in self.centerline_endpoints
            ):
                raise ValueError("centerline_endpoints must contain normalized points")
        for name in ("center_offset", "heading_proxy"):
            value = getattr(self, name)
            if value is not None:
                result = _finite_real(value, name)
                if not -1.0 <= result <= 1.0:
                    raise ValueError(f"{name} must be in [-1.0, 1.0]")
                object.__setattr__(self, name, result)
        object.__setattr__(self, "confidence", _unit_real(self.confidence, "confidence"))

        complete_geometry = (
            self.centerline_endpoints,
            self.center_offset,
            self.heading_proxy,
        )
        if self.state is LaneDetectionState.DETECTED:
            if self.left_boundary is None or self.right_boundary is None:
                raise ValueError("DETECTED requires both boundaries")
            if any(value is None for value in complete_geometry):
                raise ValueError("DETECTED requires complete-lane geometry")
        elif any(value is not None for value in complete_geometry):
            raise ValueError("complete-lane geometry is only valid for DETECTED")
        if self.state is LaneDetectionState.INPUT_UNUSABLE and (
            self.left_boundary is not None or self.right_boundary is not None
        ):
            raise ValueError("INPUT_UNUSABLE cannot contain lane boundaries")


@dataclass(frozen=True, slots=True)
class LanePerceptionConfig:
    """Validated deterministic heuristic settings.

    The thresholds are normalized where possible and are intentionally
    conservative defaults for the repository's 640x360 RGB interface. They
    are not measured accuracy guarantees.
    """

    roi: tuple[float, float, float, float] = (0.0, 0.50, 1.0, 1.0)
    white_min_rgb: tuple[int, int, int] = (200, 200, 200)
    white_max_channel_spread: int = 55
    yellow_min_rgb: tuple[int, int, int] = (160, 140, 0)
    yellow_max_rgb: tuple[int, int, int] = (255, 255, 140)
    yellow_max_red_green_delta: int = 100
    vertical_bin_count: int = 18
    min_support_bins: int = 8
    line_fit_tolerance: float = 0.035
    max_fit_residual: float = 0.025
    min_vertical_span: float = 0.30
    min_abs_slope: float = 0.05
    max_abs_slope: float = 0.80
    min_lane_width: float = 0.15
    max_lane_width: float = 0.80
    min_boundary_confidence: float = 0.40
    detection_confidence: float = 0.65
    center_evaluation_y: float = 0.90
    max_image_pixels: int = 1_000_000

    def __post_init__(self) -> None:
        if type(self.roi) is not tuple or len(self.roi) != 4:
            raise ValueError("roi must be a four-item tuple")
        left, top, right, bottom = (
            _unit_real(value, f"roi[{index}]")
            for index, value in enumerate(self.roi)
        )
        if not left < right or not top < bottom:
            raise ValueError("roi must have non-empty, increasing coordinates")
        object.__setattr__(self, "roi", (left, top, right, bottom))

        white_min = _rgb_triplet(self.white_min_rgb, "white_min_rgb")
        yellow_min = _rgb_triplet(self.yellow_min_rgb, "yellow_min_rgb")
        yellow_max = _rgb_triplet(self.yellow_max_rgb, "yellow_max_rgb")
        if any(low > high for low, high in zip(yellow_min, yellow_max)):
            raise ValueError("yellow RGB minimums must not exceed maximums")
        object.__setattr__(self, "white_min_rgb", white_min)
        object.__setattr__(self, "yellow_min_rgb", yellow_min)
        object.__setattr__(self, "yellow_max_rgb", yellow_max)

        for name in ("white_max_channel_spread", "yellow_max_red_green_delta"):
            value = getattr(self, name)
            if isinstance(value, bool) or type(value) is not int or not 0 <= value <= 255:
                raise ValueError(f"{name} must be a built-in integer in [0, 255]")

        bins = _builtin_positive_int(self.vertical_bin_count, "vertical_bin_count")
        if bins > MAX_VERTICAL_BINS:
            raise ValueError(f"vertical_bin_count cannot exceed {MAX_VERTICAL_BINS}")
        support = _builtin_positive_int(self.min_support_bins, "min_support_bins")
        if support > bins:
            raise ValueError("min_support_bins cannot exceed vertical_bin_count")

        tolerance = _positive_real(self.line_fit_tolerance, "line_fit_tolerance", upper=1.0)
        residual = _positive_real(self.max_fit_residual, "max_fit_residual", upper=1.0)
        if residual > tolerance:
            raise ValueError("max_fit_residual cannot exceed line_fit_tolerance")
        span = _positive_real(self.min_vertical_span, "min_vertical_span", upper=1.0)
        if span > bottom - top:
            raise ValueError("min_vertical_span cannot exceed ROI height")
        min_slope = _positive_real(self.min_abs_slope, "min_abs_slope", upper=4.0)
        max_slope = _positive_real(self.max_abs_slope, "max_abs_slope", upper=4.0)
        if min_slope > max_slope:
            raise ValueError("min_abs_slope cannot exceed max_abs_slope")
        min_width = _positive_real(self.min_lane_width, "min_lane_width", upper=1.0)
        max_width = _positive_real(self.max_lane_width, "max_lane_width", upper=1.0)
        if min_width > max_width:
            raise ValueError("min_lane_width cannot exceed max_lane_width")
        if max_width > right - left:
            raise ValueError("max_lane_width cannot exceed ROI width")
        boundary_confidence = _unit_real(
            self.min_boundary_confidence, "min_boundary_confidence"
        )
        detection_confidence = _unit_real(
            self.detection_confidence, "detection_confidence"
        )
        if boundary_confidence > detection_confidence:
            raise ValueError(
                "min_boundary_confidence cannot exceed detection_confidence"
            )
        evaluation_y = _unit_real(self.center_evaluation_y, "center_evaluation_y")
        if not top <= evaluation_y <= bottom:
            raise ValueError("center_evaluation_y must lie within the ROI")
        pixels = _builtin_positive_int(self.max_image_pixels, "max_image_pixels")
        if pixels > MAX_CONFIGURED_IMAGE_PIXELS:
            raise ValueError(
                f"max_image_pixels cannot exceed {MAX_CONFIGURED_IMAGE_PIXELS}"
            )

        object.__setattr__(self, "line_fit_tolerance", tolerance)
        object.__setattr__(self, "max_fit_residual", residual)
        object.__setattr__(self, "min_vertical_span", span)
        object.__setattr__(self, "min_abs_slope", min_slope)
        object.__setattr__(self, "max_abs_slope", max_slope)
        object.__setattr__(self, "min_lane_width", min_width)
        object.__setattr__(self, "max_lane_width", max_width)
        object.__setattr__(self, "min_boundary_confidence", boundary_confidence)
        object.__setattr__(self, "detection_confidence", detection_confidence)
        object.__setattr__(self, "center_evaluation_y", evaluation_y)
        object.__setattr__(self, "max_image_pixels", pixels)


DEFAULT_LANE_CONFIG = LanePerceptionConfig()
