"""Deterministic offline acceptance tests for Mission 7 temporal lane tracking.

The suite uses only immutable synthetic Mission 5 observations and telemetry
records.  It deliberately imports no NumPy, CARLA, Pygame, OpenCV, or SciPy and
never opens a socket or starts a worker.
"""

from __future__ import annotations

import collections
import dataclasses
import gc
import math
import os
import pathlib
import subprocess
import sys
import textwrap
import unittest
import weakref
from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction
from unittest import mock

from perception.state import (
    LaneBoundaryObservation,
    LaneBoundarySide,
    LaneDetectionState,
    LaneObservation,
    LaneObservationReason,
    NormalizedImagePoint,
)
from perception.temporal import (
    TemporalLaneBoundary,
    TemporalLaneEstimate,
    TemporalLaneObservation,
    TemporalLanePipeline,
    TemporalLaneState,
    TemporalLaneTracker,
    TemporalLaneTrackerConfig,
    TemporalLaneTrackerMetrics,
)
from telemetry.state import (
    PIXEL_FORMAT_RGB8,
    CameraFrame,
    CameraFrameMetadata,
    SampleStamp,
    TelemetrySnapshot,
)


_DEFAULT = object()


def source_stamp(
    frame: int,
    sim_time: float | None = None,
    monotonic: float | None = None,
) -> SampleStamp:
    selected_sim_time = frame / 10.0 if sim_time is None else sim_time
    selected_monotonic = (
        selected_sim_time if monotonic is None else monotonic
    )
    return SampleStamp(
        frame=frame,
        sim_time=selected_sim_time,
        monotonic=selected_monotonic,
    )


def raw_boundary(
    side: LaneBoundarySide,
    x_intercept: float,
    x_slope: float,
    confidence: float = 0.8,
) -> LaneBoundaryObservation:
    top = 0.5
    bottom = 1.0
    return LaneBoundaryObservation(
        side=side,
        endpoints=(
            NormalizedImagePoint(x_intercept + x_slope * top, top),
            NormalizedImagePoint(x_intercept + x_slope * bottom, bottom),
        ),
        x_intercept=x_intercept,
        x_slope=x_slope,
        support_count=12,
        fit_residual=0.01,
        confidence=confidence,
    )


def left_boundary(
    x_intercept: float = 0.55,
    x_slope: float = -0.25,
    confidence: float = 0.8,
) -> LaneBoundaryObservation:
    return raw_boundary(
        LaneBoundarySide.LEFT,
        x_intercept,
        x_slope,
        confidence,
    )


def right_boundary(
    x_intercept: float = 0.45,
    x_slope: float = 0.25,
    confidence: float = 0.8,
) -> LaneBoundaryObservation:
    return raw_boundary(
        LaneBoundarySide.RIGHT,
        x_intercept,
        x_slope,
        confidence,
    )


def raw_observation(
    state: LaneDetectionState = LaneDetectionState.DETECTED,
    *,
    stamp: SampleStamp | None | object = _DEFAULT,
    revision: int = 1,
    left: LaneBoundaryObservation | None | object = _DEFAULT,
    right: LaneBoundaryObservation | None | object = _DEFAULT,
    center_offset: float | None | object = _DEFAULT,
    heading_proxy: float | None | object = _DEFAULT,
    confidence: float | object = _DEFAULT,
    reason: LaneObservationReason | object = _DEFAULT,
) -> LaneObservation:
    if stamp is _DEFAULT:
        stamp = None if state is LaneDetectionState.INPUT_UNUSABLE else source_stamp(1)

    if state is LaneDetectionState.DETECTED:
        selected_left = left_boundary() if left is _DEFAULT else left
        selected_right = right_boundary() if right is _DEFAULT else right
        selected_offset = 0.0 if center_offset is _DEFAULT else center_offset
        selected_heading = 0.0 if heading_proxy is _DEFAULT else heading_proxy
        selected_confidence = 0.8 if confidence is _DEFAULT else confidence
        selected_reason = (
            LaneObservationReason.DETECTED if reason is _DEFAULT else reason
        )
        centerline = (
            NormalizedImagePoint(0.5 + float(selected_offset), 0.5),
            NormalizedImagePoint(
                0.5 + float(selected_offset) + 0.5 * float(selected_heading),
                1.0,
            ),
        )
    elif state is LaneDetectionState.PARTIAL:
        selected_left = left_boundary() if left is _DEFAULT else left
        selected_right = None if right is _DEFAULT else right
        selected_offset = None
        selected_heading = None
        selected_confidence = (
            selected_left.confidence
            if confidence is _DEFAULT and selected_left is not None
            else 0.0 if confidence is _DEFAULT else confidence
        )
        selected_reason = (
            LaneObservationReason.SINGLE_PLAUSIBLE_BOUNDARY
            if reason is _DEFAULT
            else reason
        )
        centerline = None
    elif state is LaneDetectionState.NOT_DETECTED:
        selected_left = None if left is _DEFAULT else left
        selected_right = None if right is _DEFAULT else right
        selected_offset = None
        selected_heading = None
        selected_confidence = 0.0 if confidence is _DEFAULT else confidence
        selected_reason = (
            LaneObservationReason.NO_PLAUSIBLE_BOUNDARY
            if reason is _DEFAULT
            else reason
        )
        centerline = None
    else:
        selected_left = None
        selected_right = None
        selected_offset = None
        selected_heading = None
        selected_confidence = 0.0 if confidence is _DEFAULT else confidence
        selected_reason = (
            LaneObservationReason.STATUS_UNUSABLE
            if reason is _DEFAULT
            else reason
        )
        centerline = None

    assert stamp is None or isinstance(stamp, SampleStamp)
    return LaneObservation(
        snapshot_revision=revision,
        source_frame=None if stamp is None else stamp.frame,
        source_simulation_timestamp=None if stamp is None else stamp.sim_time,
        source_receive_monotonic=None if stamp is None else stamp.monotonic,
        state=state,
        reason=selected_reason,
        left_boundary=selected_left,
        right_boundary=selected_right,
        centerline_endpoints=centerline,
        center_offset=selected_offset,
        heading_proxy=selected_heading,
        confidence=selected_confidence,
    )


def snapshot_with_stamp(stamp: SampleStamp, revision: int = 1) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        revision=revision,
        captured_monotonic=stamp.monotonic,
        camera=CameraFrame(
            metadata=CameraFrameMetadata(
                stamp=stamp,
                width_px=1,
                height_px=1,
                horizontal_fov_deg=90.0,
                pixel_format=PIXEL_FORMAT_RGB8,
                row_stride_bytes=3,
            ),
            rgb_bytes=b"\0\0\0",
        ),
    )


def ema(previous: float, current: float, alpha: float) -> float:
    return alpha * current + (1.0 - alpha) * previous


class TemporalAssertions(unittest.TestCase):
    def assert_no_geometry(self, estimate: TemporalLaneEstimate) -> None:
        self.assertIsNone(estimate.left_boundary)
        self.assertIsNone(estimate.right_boundary)
        self.assertIsNone(estimate.center_offset)
        self.assertIsNone(estimate.heading_proxy)

    def assert_tracking(self, estimate: TemporalLaneEstimate) -> None:
        self.assertIs(estimate.state, TemporalLaneState.TRACKING)
        self.assertFalse(estimate.extrapolated)
        self.assertEqual(estimate.consecutive_misses, 0)


class PublicModelAndConfigurationTest(TemporalAssertions):
    def test_temporal_states_are_exact_and_distinct_from_raw_states(self):
        self.assertEqual(
            {name: value.value for name, value in TemporalLaneState.__members__.items()},
            {
                "UNINITIALIZED": "uninitialized",
                "TRACKING": "tracking",
                "COASTING": "coasting",
                "LOST": "lost",
                "INPUT_UNUSABLE": "input_unusable",
            },
        )
        self.assertFalse(issubclass(TemporalLaneState, LaneDetectionState))

    def test_default_configuration_values_and_field_order_are_exact(self):
        config = TemporalLaneTrackerConfig()
        self.assertEqual(
            tuple(field.name for field in fields(config)),
            (
                "smoothing_alpha",
                "max_consecutive_misses",
                "max_coast_age_seconds",
                "coast_confidence_decay",
                "minimum_coast_confidence",
            ),
        )
        self.assertEqual(config.smoothing_alpha, 0.35)
        self.assertEqual(config.max_consecutive_misses, 2)
        self.assertEqual(config.max_coast_age_seconds, 0.30)
        self.assertEqual(config.coast_confidence_decay, 0.75)
        self.assertEqual(config.minimum_coast_confidence, 0.10)
        self.assertIs(type(config.smoothing_alpha), float)
        self.assertIs(type(config.max_consecutive_misses), int)
        self.assertFalse(hasattr(config, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            config.smoothing_alpha = 0.5

    def test_configuration_rejects_invalid_finite_ranges_and_types(self):
        bad_changes = (
            {"smoothing_alpha": 0.0},
            {"smoothing_alpha": -0.1},
            {"smoothing_alpha": 1.01},
            {"smoothing_alpha": True},
            {"smoothing_alpha": "0.5"},
            {"max_consecutive_misses": -1},
            {"max_consecutive_misses": True},
            {"max_consecutive_misses": 2.0},
            {"max_coast_age_seconds": 0.0},
            {"max_coast_age_seconds": -0.01},
            {"max_coast_age_seconds": True},
            {"coast_confidence_decay": 0.0},
            {"coast_confidence_decay": -0.01},
            {"coast_confidence_decay": 1.0},
            {"coast_confidence_decay": 1.01},
            {"minimum_coast_confidence": 0.0},
            {"minimum_coast_confidence": -0.01},
            {"minimum_coast_confidence": 1.01},
        )
        base = TemporalLaneTrackerConfig()
        for changes in bad_changes:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(base, **changes)

        for field_name in (
            "smoothing_alpha",
            "max_coast_age_seconds",
            "coast_confidence_decay",
            "minimum_coast_confidence",
        ):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=field_name, value=value), self.assertRaises(
                    ValueError
                ):
                    replace(base, **{field_name: value})

    def test_useful_exact_configuration_boundaries_are_accepted(self):
        config = TemporalLaneTrackerConfig(
            smoothing_alpha=1.0,
            max_consecutive_misses=0,
            max_coast_age_seconds=0.001,
            coast_confidence_decay=0.5,
            minimum_coast_confidence=1.0,
        )
        self.assertEqual(config.smoothing_alpha, 1.0)
        self.assertEqual(config.max_consecutive_misses, 0)
        self.assertEqual(config.max_coast_age_seconds, 0.001)
        self.assertEqual(config.coast_confidence_decay, 0.5)
        self.assertEqual(config.minimum_coast_confidence, 1.0)

    def test_config_subclass_is_canonicalized_and_not_retained(self):
        class ConfigSubclass(TemporalLaneTrackerConfig):
            __slots__ = ("__weakref__",)

        hostile = ConfigSubclass(
            smoothing_alpha=Fraction(1, 2),
            max_consecutive_misses=3,
            max_coast_age_seconds=Fraction(1, 4),
            coast_confidence_decay=Fraction(3, 4),
            minimum_coast_confidence=Fraction(1, 10),
        )
        reference = weakref.ref(hostile)
        tracker = TemporalLaneTracker(hostile)
        self.assertIs(type(tracker.config), TemporalLaneTrackerConfig)
        self.assertIsNot(tracker.config, hostile)
        self.assertEqual(tracker.config.smoothing_alpha, 0.5)
        self.assertIs(type(tracker.config.smoothing_alpha), float)
        hostile = None
        gc.collect()
        self.assertIsNone(reference())

    def test_exact_base_config_is_detached_before_retention(self):
        config = TemporalLaneTrackerConfig()
        tracker = TemporalLaneTracker(config)
        self.assertIsNot(tracker.config, config)
        self.assertEqual(tracker.config, config)
        object.__setattr__(config, "smoothing_alpha", 1.0)
        self.assertEqual(tracker.config.smoothing_alpha, 0.35)
        published = tracker.config
        object.__setattr__(published, "max_consecutive_misses", 999)
        self.assertEqual(tracker.config.max_consecutive_misses, 2)

    def test_public_record_field_order_is_exact(self):
        self.assertEqual(
            tuple(field.name for field in fields(TemporalLaneBoundary)),
            ("side", "x_intercept", "x_slope", "confidence"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(TemporalLaneEstimate)),
            (
                "source_stamp",
                "snapshot_revision",
                "state",
                "raw_state",
                "raw_reason",
                "left_boundary",
                "right_boundary",
                "center_offset",
                "heading_proxy",
                "confidence",
                "consecutive_misses",
                "track_age_seconds",
                "extrapolated",
            ),
        )

    def test_temporal_boundary_validation_and_canonicalization(self):
        boundary = TemporalLaneBoundary(
            LaneBoundarySide.LEFT,
            Fraction(1, 2),
            Fraction(-1, 4),
            Fraction(3, 4),
        )
        self.assertIs(type(boundary.x_intercept), float)
        self.assertIs(type(boundary.x_slope), float)
        self.assertIs(type(boundary.confidence), float)
        self.assertFalse(hasattr(boundary, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            boundary.confidence = 0.0
        for changes in (
            {"side": "left"},
            {"x_intercept": float("nan")},
            {"x_slope": float("inf")},
            {"confidence": -0.01},
            {"confidence": 1.01},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(boundary, **changes)

    def test_initial_estimate_is_frozen_slotted_and_empty(self):
        tracker = TemporalLaneTracker()
        estimate = tracker.latest_estimate
        self.assertIsInstance(estimate, TemporalLaneEstimate)
        self.assertIs(estimate.state, TemporalLaneState.UNINITIALIZED)
        self.assertIsNone(estimate.source_stamp)
        self.assertIsNone(estimate.snapshot_revision)
        self.assertIsNone(estimate.raw_state)
        self.assertIsNone(estimate.raw_reason)
        self.assert_no_geometry(estimate)
        self.assertEqual(estimate.confidence, 0.0)
        self.assertEqual(estimate.consecutive_misses, 0)
        self.assertIsNone(estimate.track_age_seconds)
        self.assertFalse(estimate.extrapolated)
        self.assertFalse(hasattr(estimate, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            estimate.confidence = 1.0

    def test_direct_estimate_construction_rejects_incoherent_state_evidence(self):
        stamp = source_stamp(1, 0.0)
        left = TemporalLaneBoundary(LaneBoundarySide.LEFT, 0.55, -0.25, 0.8)
        right = TemporalLaneBoundary(LaneBoundarySide.RIGHT, 0.45, 0.25, 0.8)
        tracking = TemporalLaneEstimate(
            source_stamp=stamp,
            snapshot_revision=1,
            state=TemporalLaneState.TRACKING,
            raw_state=LaneDetectionState.DETECTED,
            raw_reason=LaneObservationReason.DETECTED,
            left_boundary=left,
            right_boundary=right,
            center_offset=0.0,
            heading_proxy=0.0,
            confidence=0.8,
            consecutive_misses=0,
            track_age_seconds=0.0,
            extrapolated=False,
        )
        coasting = TemporalLaneEstimate(
            source_stamp=source_stamp(2, 0.1),
            snapshot_revision=2,
            state=TemporalLaneState.COASTING,
            raw_state=LaneDetectionState.NOT_DETECTED,
            raw_reason=LaneObservationReason.NO_PLAUSIBLE_BOUNDARY,
            left_boundary=left,
            right_boundary=right,
            center_offset=0.0,
            heading_proxy=0.0,
            confidence=0.6,
            consecutive_misses=1,
            track_age_seconds=0.1,
            extrapolated=True,
        )
        lost = TemporalLaneEstimate(
            source_stamp=source_stamp(3, 0.4),
            snapshot_revision=3,
            state=TemporalLaneState.LOST,
            raw_state=LaneDetectionState.NOT_DETECTED,
            raw_reason=LaneObservationReason.NO_PLAUSIBLE_BOUNDARY,
            confidence=0.0,
            consecutive_misses=3,
            track_age_seconds=0.4,
            extrapolated=False,
        )
        unusable = TemporalLaneEstimate(
            source_stamp=None,
            snapshot_revision=4,
            state=TemporalLaneState.INPUT_UNUSABLE,
            raw_state=LaneDetectionState.INPUT_UNUSABLE,
            raw_reason=LaneObservationReason.STATUS_UNUSABLE,
            confidence=0.0,
            consecutive_misses=0,
            track_age_seconds=None,
            extrapolated=False,
        )
        initial = TemporalLaneTracker().latest_estimate

        invalid_cases = (
            (tracking, {"state": "tracking"}),
            (tracking, {"raw_state": "detected"}),
            (tracking, {"raw_state": LaneDetectionState.NOT_DETECTED}),
            (tracking, {"source_stamp": None}),
            (
                tracking,
                {
                    "left_boundary": None,
                    "right_boundary": None,
                    "center_offset": None,
                    "heading_proxy": None,
                },
            ),
            (tracking, {"right_boundary": None}),
            (tracking, {"center_offset": None}),
            (tracking, {"raw_state": LaneDetectionState.PARTIAL}),
            (tracking, {"consecutive_misses": 1}),
            (tracking, {"extrapolated": True}),
            (tracking, {"confidence": -0.01}),
            (tracking, {"confidence": 0.9}),
            (tracking, {"confidence": 1.01}),
            (tracking, {"confidence": float("nan")}),
            (coasting, {"raw_state": LaneDetectionState.DETECTED}),
            (coasting, {"left_boundary": None, "right_boundary": None,
                        "center_offset": None, "heading_proxy": None}),
            (coasting, {"consecutive_misses": 0}),
            (coasting, {"confidence": 0.0}),
            (coasting, {"extrapolated": False}),
            (lost, {"left_boundary": left}),
            (lost, {"confidence": 0.1}),
            (lost, {"consecutive_misses": 0}),
            (lost, {"extrapolated": True}),
            (unusable, {"left_boundary": left}),
            (unusable, {"confidence": 0.1}),
            (unusable, {"consecutive_misses": 1}),
            (unusable, {"extrapolated": True}),
            (initial, {"raw_state": LaneDetectionState.NOT_DETECTED}),
            (initial, {"confidence": 0.1}),
            (initial, {"extrapolated": True}),
        )
        for base, changes in invalid_cases:
            with self.subTest(state=base.state, changes=changes), self.assertRaises(
                ValueError
            ):
                replace(base, **changes)

    def test_metrics_schema_is_exact_frozen_and_initially_zero(self):
        tracker = TemporalLaneTracker()
        metrics = tracker.metrics
        self.assertIsInstance(metrics, TemporalLaneTrackerMetrics)
        self.assertEqual(
            tuple(field.name for field in fields(metrics)),
            (
                "observations_received",
                "observations_accepted",
                "duplicates_rejected",
                "out_of_order_rejected",
                "invalid_observations_rejected",
                "usable_updates",
                "coast_updates",
                "lost_updates",
                "input_unusable_updates",
                "unstamped_input_unusable_updates",
                "recoveries",
                "resets",
                "maximum_consecutive_misses",
                "last_accepted_source_stamp",
            ),
        )
        for field in fields(metrics):
            value = getattr(metrics, field.name)
            if field.name == "last_accepted_source_stamp":
                self.assertIsNone(value)
            else:
                self.assertEqual(value, 0, field.name)
                self.assertIs(type(value), int)
        with self.assertRaises(FrozenInstanceError):
            metrics.resets = 1


class InitializationAndSmoothingTest(TemporalAssertions):
    def test_first_detected_sample_initializes_every_supported_field_exactly(self):
        stamp = source_stamp(7, 1.25, 9.5)
        left = left_boundary(0.57, -0.27, 0.82)
        right = right_boundary(0.43, 0.31, 0.76)
        raw = raw_observation(
            stamp=stamp,
            revision=19,
            left=left,
            right=right,
            center_offset=0.08,
            heading_proxy=-0.06,
            confidence=0.74,
        )
        tracker = TemporalLaneTracker()

        estimate = tracker.update(raw)

        self.assert_tracking(estimate)
        self.assertEqual(estimate.source_stamp, stamp)
        self.assertIs(type(estimate.source_stamp), SampleStamp)
        self.assertEqual(estimate.snapshot_revision, 19)
        self.assertIs(estimate.raw_state, LaneDetectionState.DETECTED)
        self.assertIs(estimate.raw_reason, LaneObservationReason.DETECTED)
        self.assertEqual(
            estimate.left_boundary,
            TemporalLaneBoundary(
                LaneBoundarySide.LEFT,
                left.x_intercept,
                left.x_slope,
                left.confidence,
            ),
        )
        self.assertEqual(
            estimate.right_boundary,
            TemporalLaneBoundary(
                LaneBoundarySide.RIGHT,
                right.x_intercept,
                right.x_slope,
                right.confidence,
            ),
        )
        self.assertEqual(estimate.center_offset, 0.08)
        self.assertEqual(estimate.heading_proxy, -0.06)
        self.assertEqual(estimate.confidence, 0.74)
        self.assertEqual(estimate.track_age_seconds, 0.0)
        self.assertIs(tracker.latest_estimate, estimate)

    def test_first_single_boundary_partial_is_tracking_without_invented_geometry(self):
        left = left_boundary(0.61, -0.28, 0.67)
        estimate = TemporalLaneTracker().update(
            raw_observation(
                LaneDetectionState.PARTIAL,
                stamp=source_stamp(1, 0.0),
                left=left,
                confidence=0.67,
            )
        )
        self.assert_tracking(estimate)
        self.assertEqual(
            estimate.left_boundary,
            TemporalLaneBoundary(
                LaneBoundarySide.LEFT,
                left.x_intercept,
                left.x_slope,
                left.confidence,
            ),
        )
        self.assertIsNone(estimate.right_boundary)
        self.assertIsNone(estimate.center_offset)
        self.assertIsNone(estimate.heading_proxy)
        self.assertEqual(estimate.confidence, 0.67)

    def test_first_not_detected_is_lost_without_geometry(self):
        estimate = TemporalLaneTracker().update(
            raw_observation(
                LaneDetectionState.NOT_DETECTED,
                stamp=source_stamp(1, 0.0),
            )
        )
        self.assertIs(estimate.state, TemporalLaneState.LOST)
        self.assertIs(estimate.raw_state, LaneDetectionState.NOT_DETECTED)
        self.assert_no_geometry(estimate)
        self.assertEqual(estimate.confidence, 0.0)
        self.assertEqual(estimate.consecutive_misses, 1)
        self.assertFalse(estimate.extrapolated)

    def test_second_detected_sample_uses_exact_ema(self):
        alpha = 0.25
        tracker = TemporalLaneTracker(
            TemporalLaneTrackerConfig(smoothing_alpha=alpha)
        )
        first_left = left_boundary(0.55, -0.25, 0.80)
        first_right = right_boundary(0.45, 0.25, 0.70)
        second_left = left_boundary(0.75, -0.45, 0.60)
        second_right = right_boundary(0.61, 0.35, 0.90)
        tracker.update(
            raw_observation(
                stamp=source_stamp(1, 0.0),
                left=first_left,
                right=first_right,
                center_offset=-0.20,
                heading_proxy=-0.10,
                confidence=0.70,
            )
        )

        estimate = tracker.update(
            raw_observation(
                stamp=source_stamp(2, 0.1),
                revision=2,
                left=second_left,
                right=second_right,
                center_offset=0.20,
                heading_proxy=0.10,
                confidence=0.90,
            )
        )

        self.assert_tracking(estimate)
        assert estimate.left_boundary is not None
        assert estimate.right_boundary is not None
        self.assertEqual(
            estimate.left_boundary.x_intercept,
            ema(first_left.x_intercept, second_left.x_intercept, alpha),
        )
        self.assertEqual(
            estimate.left_boundary.x_slope,
            ema(first_left.x_slope, second_left.x_slope, alpha),
        )
        self.assertEqual(
            estimate.left_boundary.confidence,
            min(
                second_left.confidence,
                ema(first_left.confidence, second_left.confidence, alpha),
            ),
        )
        self.assertEqual(
            estimate.right_boundary.x_intercept,
            ema(first_right.x_intercept, second_right.x_intercept, alpha),
        )
        self.assertEqual(
            estimate.right_boundary.x_slope,
            ema(first_right.x_slope, second_right.x_slope, alpha),
        )
        self.assertEqual(
            estimate.right_boundary.confidence,
            min(
                second_right.confidence,
                ema(first_right.confidence, second_right.confidence, alpha),
            ),
        )
        self.assertEqual(estimate.center_offset, ema(-0.20, 0.20, alpha))
        self.assertEqual(estimate.heading_proxy, ema(-0.10, 0.10, alpha))
        self.assertEqual(estimate.confidence, 0.60)

    def test_partial_updates_only_present_side_and_clears_complete_geometry(self):
        alpha = 0.5
        tracker = TemporalLaneTracker(
            TemporalLaneTrackerConfig(smoothing_alpha=alpha)
        )
        first = tracker.update(
            raw_observation(
                stamp=source_stamp(1, 0.0),
                center_offset=0.10,
                heading_proxy=0.08,
            )
        )
        updated_left = left_boundary(0.75, -0.35, 0.60)

        partial = tracker.update(
            raw_observation(
                LaneDetectionState.PARTIAL,
                stamp=source_stamp(2, 0.1),
                revision=2,
                left=updated_left,
                confidence=0.60,
            )
        )

        self.assert_tracking(partial)
        assert first.left_boundary is not None
        assert first.right_boundary is not None
        assert partial.left_boundary is not None
        self.assertEqual(
            partial.left_boundary.x_intercept,
            ema(first.left_boundary.x_intercept, updated_left.x_intercept, alpha),
        )
        self.assertEqual(
            partial.left_boundary.x_slope,
            ema(first.left_boundary.x_slope, updated_left.x_slope, alpha),
        )
        self.assertIsNone(partial.right_boundary)
        self.assertIsNone(partial.center_offset)
        self.assertIsNone(partial.heading_proxy)
        self.assertEqual(partial.confidence, min(first.confidence, 0.60))
        self.assertFalse(partial.extrapolated)

    def test_detected_after_one_side_partial_reinitializes_all_complete_geometry(self):
        tracker = TemporalLaneTracker(
            TemporalLaneTrackerConfig(smoothing_alpha=0.05)
        )
        tracker.update(
            raw_observation(
                stamp=source_stamp(1, 0.0),
                left=left_boundary(0.55, -0.25, 0.90),
                right=right_boundary(0.45, 0.25, 0.88),
                center_offset=-0.25,
                heading_proxy=-0.20,
                confidence=0.86,
            )
        )
        partial = tracker.update(
            raw_observation(
                LaneDetectionState.PARTIAL,
                stamp=source_stamp(2, 0.1),
                revision=2,
                left=left_boundary(0.70, -0.35, 0.55),
                confidence=0.55,
            )
        )
        self.assertIsNotNone(partial.left_boundary)
        self.assertIsNone(partial.right_boundary)
        self.assertIsNone(partial.center_offset)
        self.assertIsNone(partial.heading_proxy)

        current_left = left_boundary(0.63, -0.29, 0.67)
        current_right = right_boundary(0.51, 0.33, 0.71)
        detected = tracker.update(
            raw_observation(
                stamp=source_stamp(3, 0.2),
                revision=3,
                left=current_left,
                right=current_right,
                center_offset=0.19,
                heading_proxy=0.14,
                confidence=0.61,
            )
        )

        self.assert_tracking(detected)
        self.assertEqual(
            detected.left_boundary,
            TemporalLaneBoundary(
                LaneBoundarySide.LEFT,
                current_left.x_intercept,
                current_left.x_slope,
                current_left.confidence,
            ),
        )
        self.assertEqual(
            detected.right_boundary,
            TemporalLaneBoundary(
                LaneBoundarySide.RIGHT,
                current_right.x_intercept,
                current_right.x_slope,
                current_right.confidence,
            ),
        )
        self.assertEqual(detected.center_offset, 0.19)
        self.assertEqual(detected.heading_proxy, 0.14)
        self.assertEqual(detected.confidence, 0.61)

    def test_lower_confidence_detected_caps_aggregate_and_both_boundaries(self):
        alpha = 0.2
        tracker = TemporalLaneTracker(
            TemporalLaneTrackerConfig(smoothing_alpha=alpha)
        )
        first_left = left_boundary(0.55, -0.25, 0.92)
        first_right = right_boundary(0.45, 0.25, 0.88)
        tracker.update(
            raw_observation(
                stamp=source_stamp(1, 0.0),
                left=first_left,
                right=first_right,
                confidence=0.84,
            )
        )
        current_left = left_boundary(0.65, -0.30, 0.31)
        current_right = right_boundary(0.55, 0.30, 0.27)

        estimate = tracker.update(
            raw_observation(
                stamp=source_stamp(2, 0.1),
                revision=2,
                left=current_left,
                right=current_right,
                center_offset=0.10,
                heading_proxy=0.05,
                confidence=0.23,
            )
        )

        assert estimate.left_boundary is not None
        assert estimate.right_boundary is not None
        self.assertEqual(estimate.left_boundary.confidence, current_left.confidence)
        self.assertEqual(estimate.right_boundary.confidence, current_right.confidence)
        self.assertEqual(estimate.confidence, 0.23)
        self.assertLess(
            estimate.left_boundary.confidence,
            ema(first_left.confidence, current_left.confidence, alpha),
        )
        self.assertLess(
            estimate.right_boundary.confidence,
            ema(first_right.confidence, current_right.confidence, alpha),
        )
        self.assertLess(estimate.confidence, ema(0.84, 0.23, alpha))

    def test_newly_appearing_fields_initialize_directly_instead_of_from_zero(self):
        tracker = TemporalLaneTracker(
            TemporalLaneTrackerConfig(smoothing_alpha=0.1)
        )
        tracker.update(
            raw_observation(
                LaneDetectionState.PARTIAL,
                stamp=source_stamp(1, 0.0),
                left=left_boundary(),
            )
        )
        detected = tracker.update(
            raw_observation(
                stamp=source_stamp(2, 0.1),
                center_offset=0.22,
                heading_proxy=-0.12,
            )
        )
        self.assertEqual(detected.center_offset, 0.22)
        self.assertEqual(detected.heading_proxy, -0.12)
        self.assertIsNotNone(detected.right_boundary)

    def test_repeated_steady_input_converges_and_stays_bounded(self):
        tracker = TemporalLaneTracker(
            TemporalLaneTrackerConfig(smoothing_alpha=0.2)
        )
        tracker.update(
            raw_observation(
                stamp=source_stamp(1, 0.0),
                center_offset=-0.3,
                heading_proxy=-0.2,
            )
        )
        latest = None
        for frame in range(2, 102):
            latest = tracker.update(
                raw_observation(
                    stamp=source_stamp(frame, frame / 100.0),
                    revision=frame,
                    center_offset=0.2,
                    heading_proxy=0.1,
                )
            )
        assert latest is not None
        self.assertAlmostEqual(latest.center_offset, 0.2, places=9)
        self.assertAlmostEqual(latest.heading_proxy, 0.1, places=9)
        self.assertTrue(-1.0 <= latest.center_offset <= 1.0)
        self.assertTrue(-1.0 <= latest.heading_proxy <= 1.0)


class CoastingAndExpirationTest(TemporalAssertions):
    def test_one_and_two_misses_coast_with_multiplicative_confidence_decay(self):
        tracker = TemporalLaneTracker()
        detected = tracker.update(
            raw_observation(stamp=source_stamp(1, 0.0), confidence=0.8)
        )
        first = tracker.update(
            raw_observation(
                LaneDetectionState.NOT_DETECTED,
                stamp=source_stamp(2, 0.1),
                revision=2,
            )
        )
        second = tracker.update(
            raw_observation(
                LaneDetectionState.NOT_DETECTED,
                stamp=source_stamp(3, 0.2),
                revision=3,
            )
        )

        self.assertIs(first.state, TemporalLaneState.COASTING)
        self.assertIs(second.state, TemporalLaneState.COASTING)
        self.assertTrue(first.extrapolated)
        self.assertTrue(second.extrapolated)
        self.assertEqual(first.consecutive_misses, 1)
        self.assertEqual(second.consecutive_misses, 2)
        self.assertEqual(first.track_age_seconds, 0.1)
        self.assertEqual(second.track_age_seconds, 0.2)
        self.assertEqual(first.confidence, detected.confidence * 0.75)
        self.assertEqual(second.confidence, first.confidence * 0.75)
        self.assertEqual(first.left_boundary.side, LaneBoundarySide.LEFT)
        self.assertEqual(second.right_boundary.side, LaneBoundarySide.RIGHT)
        self.assertEqual(first.left_boundary.x_intercept, detected.left_boundary.x_intercept)
        self.assertEqual(second.right_boundary.x_slope, detected.right_boundary.x_slope)
        self.assertEqual(
            first.left_boundary.confidence,
            detected.left_boundary.confidence * 0.75,
        )

    def test_miss_limit_is_inclusive_and_next_miss_loses(self):
        tracker = TemporalLaneTracker(
            TemporalLaneTrackerConfig(
                max_consecutive_misses=2,
                max_coast_age_seconds=10.0,
                coast_confidence_decay=0.99,
                minimum_coast_confidence=0.001,
            )
        )
        tracker.update(raw_observation(stamp=source_stamp(1, 0.0)))
        for frame in (2, 3):
            estimate = tracker.update(
                raw_observation(
                    LaneDetectionState.NOT_DETECTED,
                    stamp=source_stamp(frame, 0.0),
                    revision=frame,
                )
            )
            self.assertIs(estimate.state, TemporalLaneState.COASTING)
        lost = tracker.update(
            raw_observation(
                LaneDetectionState.NOT_DETECTED,
                stamp=source_stamp(4, 0.0),
                revision=4,
            )
        )
        self.assertIs(lost.state, TemporalLaneState.LOST)
        self.assertEqual(lost.consecutive_misses, 3)
        self.assert_no_geometry(lost)
        self.assertEqual(lost.confidence, 0.0)
        self.assertFalse(lost.extrapolated)

    def test_coast_age_limit_is_inclusive_and_next_float_loses(self):
        tracker = TemporalLaneTracker(
            TemporalLaneTrackerConfig(
                max_consecutive_misses=100,
                max_coast_age_seconds=0.30,
                coast_confidence_decay=0.99,
                minimum_coast_confidence=0.001,
            )
        )
        tracker.update(raw_observation(stamp=source_stamp(1, 0.0)))
        equality = tracker.update(
            raw_observation(
                LaneDetectionState.NOT_DETECTED,
                stamp=source_stamp(2, 0.30),
            )
        )
        self.assertIs(equality.state, TemporalLaneState.COASTING)
        self.assertEqual(equality.track_age_seconds, 0.30)
        above = tracker.update(
            raw_observation(
                LaneDetectionState.NOT_DETECTED,
                stamp=source_stamp(3, math.nextafter(0.30, math.inf)),
            )
        )
        self.assertIs(above.state, TemporalLaneState.LOST)
        self.assert_no_geometry(above)

    def test_extreme_finite_sim_time_overflow_loses_then_reinitializes_exactly(self):
        extreme = sys.float_info.max
        tracker = TemporalLaneTracker(
            TemporalLaneTrackerConfig(
                smoothing_alpha=0.01,
                max_consecutive_misses=10,
                max_coast_age_seconds=10.0,
            )
        )
        tracker.update(
            raw_observation(
                stamp=source_stamp(1, -extreme, -1.0),
                left=left_boundary(0.55, -0.25, 0.9),
                right=right_boundary(0.45, 0.25, 0.9),
                center_offset=-0.25,
                heading_proxy=-0.20,
                confidence=0.9,
            )
        )

        lost = tracker.update(
            raw_observation(
                LaneDetectionState.NOT_DETECTED,
                stamp=source_stamp(2, extreme, 0.0),
                revision=2,
            )
        )

        self.assertIs(lost.state, TemporalLaneState.LOST)
        self.assertIsNone(lost.track_age_seconds)
        self.assert_no_geometry(lost)
        self.assertEqual(lost.confidence, 0.0)

        current_left = left_boundary(0.65, -0.30, 0.58)
        current_right = right_boundary(0.55, 0.35, 0.62)
        recovered = tracker.update(
            raw_observation(
                stamp=source_stamp(3, extreme, 1.0),
                revision=3,
                left=current_left,
                right=current_right,
                center_offset=0.20,
                heading_proxy=-0.10,
                confidence=0.54,
            )
        )

        self.assert_tracking(recovered)
        self.assertEqual(
            recovered.left_boundary,
            TemporalLaneBoundary(
                LaneBoundarySide.LEFT,
                current_left.x_intercept,
                current_left.x_slope,
                current_left.confidence,
            ),
        )
        self.assertEqual(
            recovered.right_boundary,
            TemporalLaneBoundary(
                LaneBoundarySide.RIGHT,
                current_right.x_intercept,
                current_right.x_slope,
                current_right.confidence,
            ),
        )
        self.assertEqual(recovered.center_offset, 0.20)
        self.assertEqual(recovered.heading_proxy, -0.10)
        self.assertEqual(recovered.confidence, 0.54)
        self.assertEqual(recovered.track_age_seconds, 0.0)
        self.assertEqual(tracker.metrics.lost_updates, 1)
        self.assertEqual(tracker.metrics.recoveries, 1)

        seen = set()

        def assert_all_floats_finite(value):
            marker = id(value)
            if marker in seen:
                return
            seen.add(marker)
            if type(value) is float:
                self.assertTrue(math.isfinite(value), value)
            elif dataclasses.is_dataclass(value) and not isinstance(value, type):
                for field in fields(value):
                    assert_all_floats_finite(getattr(value, field.name))
            elif type(value) is tuple:
                for item in value:
                    assert_all_floats_finite(item)

        assert_all_floats_finite(lost)
        assert_all_floats_finite(recovered)

    def test_minimum_coast_confidence_equality_is_retained(self):
        tracker = TemporalLaneTracker(
            TemporalLaneTrackerConfig(
                max_consecutive_misses=10,
                max_coast_age_seconds=10.0,
                coast_confidence_decay=0.5,
                minimum_coast_confidence=0.2,
            )
        )
        tracker.update(
            raw_observation(stamp=source_stamp(1, 0.0), confidence=0.8)
        )
        first = tracker.update(
            raw_observation(
                LaneDetectionState.NOT_DETECTED,
                stamp=source_stamp(2, 0.1),
            )
        )
        equality = tracker.update(
            raw_observation(
                LaneDetectionState.NOT_DETECTED,
                stamp=source_stamp(3, 0.2),
            )
        )
        expired = tracker.update(
            raw_observation(
                LaneDetectionState.NOT_DETECTED,
                stamp=source_stamp(4, 0.3),
            )
        )
        self.assertEqual(first.confidence, 0.4)
        self.assertEqual(equality.confidence, 0.2)
        self.assertIs(equality.state, TemporalLaneState.COASTING)
        self.assertIs(expired.state, TemporalLaneState.LOST)
        self.assertEqual(expired.confidence, 0.0)

    def test_lost_stays_lost_and_never_reexposes_stale_geometry(self):
        tracker = TemporalLaneTracker(
            TemporalLaneTrackerConfig(max_consecutive_misses=1)
        )
        tracker.update(raw_observation(stamp=source_stamp(1, 0.0)))
        tracker.update(
            raw_observation(
                LaneDetectionState.NOT_DETECTED,
                stamp=source_stamp(2, 0.1),
            )
        )
        for frame in range(3, 25):
            estimate = tracker.update(
                raw_observation(
                    LaneDetectionState.NOT_DETECTED,
                    stamp=source_stamp(frame, frame / 10.0),
                    revision=frame,
                )
            )
            self.assertIs(estimate.state, TemporalLaneState.LOST)
            self.assert_no_geometry(estimate)
            self.assertEqual(estimate.confidence, 0.0)
            self.assertFalse(estimate.extrapolated)


class UnusableInputAndRecoveryTest(TemporalAssertions):
    def test_stamped_input_unusable_is_immediate_and_hides_track(self):
        tracker = TemporalLaneTracker()
        tracker.update(raw_observation(stamp=source_stamp(1, 0.0)))
        unusable_stamp = source_stamp(2, 0.1, 4.0)

        estimate = tracker.update(
            raw_observation(
                LaneDetectionState.INPUT_UNUSABLE,
                stamp=None,
                revision=2,
            ),
            source_stamp=unusable_stamp,
        )

        self.assertIs(estimate.state, TemporalLaneState.INPUT_UNUSABLE)
        self.assertIs(estimate.raw_state, LaneDetectionState.INPUT_UNUSABLE)
        self.assertEqual(estimate.source_stamp, unusable_stamp)
        self.assert_no_geometry(estimate)
        self.assertEqual(estimate.confidence, 0.0)
        self.assertFalse(estimate.extrapolated)
        self.assertEqual(estimate.consecutive_misses, 0)
        self.assertEqual(tracker.metrics.input_unusable_updates, 1)
        self.assertEqual(tracker.metrics.last_accepted_source_stamp, unusable_stamp)

    def test_unstamped_input_unusable_is_allowed_without_advancing_order(self):
        tracker = TemporalLaneTracker()
        first_stamp = source_stamp(10, 1.0)
        tracker.update(raw_observation(stamp=first_stamp))

        estimate = tracker.update(
            raw_observation(
                LaneDetectionState.INPUT_UNUSABLE,
                stamp=None,
                revision=11,
            )
        )

        self.assertIs(estimate.state, TemporalLaneState.INPUT_UNUSABLE)
        self.assertIsNone(estimate.source_stamp)
        self.assert_no_geometry(estimate)
        self.assertEqual(tracker.metrics.last_accepted_source_stamp, first_stamp)
        self.assertEqual(tracker.metrics.input_unusable_updates, 1)
        self.assertEqual(tracker.metrics.unstamped_input_unusable_updates, 1)

        duplicate = tracker.update(
            raw_observation(stamp=source_stamp(10, 1.0, 99.0), revision=99)
        )
        self.assertEqual(duplicate, estimate)
        self.assertEqual(tracker.metrics.duplicates_rejected, 1)

    def test_first_unstamped_input_unusable_then_valid_input_recovers(self):
        tracker = TemporalLaneTracker()
        first = tracker.update(
            raw_observation(LaneDetectionState.INPUT_UNUSABLE, stamp=None)
        )
        self.assertIs(first.state, TemporalLaneState.INPUT_UNUSABLE)
        recovered = tracker.update(
            raw_observation(stamp=source_stamp(1, 0.0), revision=2)
        )
        self.assert_tracking(recovered)
        self.assertEqual(tracker.metrics.recoveries, 1)

    def test_valid_detection_recovers_from_coasting_lost_and_unusable(self):
        factories = (
            "coasting",
            "lost",
            "unusable",
        )
        for target in factories:
            with self.subTest(target=target):
                tracker = TemporalLaneTracker(
                    TemporalLaneTrackerConfig(max_consecutive_misses=1)
                )
                tracker.update(raw_observation(stamp=source_stamp(1, 0.0)))
                tracker.update(
                    raw_observation(
                        LaneDetectionState.NOT_DETECTED,
                        stamp=source_stamp(2, 0.1),
                    )
                )
                next_frame = 3
                if target == "lost":
                    tracker.update(
                        raw_observation(
                            LaneDetectionState.NOT_DETECTED,
                            stamp=source_stamp(3, 0.2),
                        )
                    )
                    next_frame = 4
                elif target == "unusable":
                    tracker.update(
                        raw_observation(
                            LaneDetectionState.INPUT_UNUSABLE,
                            stamp=None,
                            revision=3,
                        ),
                        source_stamp=source_stamp(3, 0.2),
                    )
                    next_frame = 4
                recovered = tracker.update(
                    raw_observation(
                        stamp=source_stamp(next_frame, 0.3),
                        revision=next_frame,
                        center_offset=0.12,
                    )
                )
                self.assert_tracking(recovered)
                self.assertIsNotNone(recovered.left_boundary)
                self.assertIsNotNone(recovered.right_boundary)
                self.assertIsNotNone(recovered.center_offset)
                self.assertIsNotNone(recovered.heading_proxy)
                if target == "lost":
                    self.assertEqual(recovered.center_offset, 0.12)
                self.assertEqual(tracker.metrics.recoveries, 1)

    def test_unusable_never_increases_confidence_or_refreshes_last_good_age(self):
        tracker = TemporalLaneTracker()
        tracker.update(
            raw_observation(stamp=source_stamp(1, 0.0), confidence=0.9)
        )
        unusable = tracker.update(
            raw_observation(LaneDetectionState.INPUT_UNUSABLE, stamp=None),
            source_stamp=source_stamp(2, 0.2),
        )
        self.assertEqual(unusable.confidence, 0.0)
        self.assertEqual(unusable.track_age_seconds, 0.2)
        second = tracker.update(
            raw_observation(
                LaneDetectionState.INPUT_UNUSABLE,
                stamp=None,
                revision=3,
            ),
            source_stamp=source_stamp(3, 0.25),
        )
        self.assertEqual(second.confidence, 0.0)
        self.assertEqual(second.track_age_seconds, 0.25)
        self.assert_no_geometry(second)


class SourceOrderingAndValidationTest(TemporalAssertions):
    def test_duplicate_and_all_order_regressions_match_m6_exactly(self):
        tracker = TemporalLaneTracker()
        baseline = tracker.update(
            raw_observation(stamp=source_stamp(10, 10.0, 10.0), revision=10)
        )
        attempts = (
            raw_observation(
                stamp=source_stamp(10, 10.0, 99.0),
                revision=500,
                center_offset=0.4,
            ),
            raw_observation(
                stamp=source_stamp(10, 11.0),
                revision=501,
                center_offset=0.4,
            ),
            raw_observation(
                stamp=source_stamp(9, 9.0),
                revision=502,
                center_offset=0.4,
            ),
            raw_observation(
                stamp=source_stamp(11, 9.5),
                revision=503,
                center_offset=0.4,
            ),
        )
        for attempt in attempts:
            self.assertIs(tracker.update(attempt), baseline)
            self.assertIs(tracker.latest_estimate, baseline)

        metrics = tracker.metrics
        self.assertEqual(metrics.observations_received, 5)
        self.assertEqual(metrics.observations_accepted, 1)
        self.assertEqual(metrics.duplicates_rejected, 1)
        self.assertEqual(metrics.out_of_order_rejected, 3)
        self.assertEqual(metrics.last_accepted_source_stamp, source_stamp(10, 10.0, 10.0))
        self.assertEqual(baseline.center_offset, 0.0)

    def test_higher_frame_with_equal_simulation_time_is_accepted(self):
        tracker = TemporalLaneTracker()
        tracker.update(raw_observation(stamp=source_stamp(10, 1.0)))
        estimate = tracker.update(
            raw_observation(
                stamp=source_stamp(11, 1.0),
                revision=2,
                center_offset=0.2,
            )
        )
        self.assert_tracking(estimate)
        self.assertEqual(estimate.source_stamp.frame, 11)
        self.assertEqual(estimate.source_stamp.sim_time, 1.0)
        self.assertEqual(tracker.metrics.observations_accepted, 2)

    def test_newer_input_after_rejections_updates_normally(self):
        tracker = TemporalLaneTracker(
            TemporalLaneTrackerConfig(smoothing_alpha=1.0)
        )
        tracker.update(raw_observation(stamp=source_stamp(10, 10.0)))
        tracker.update(raw_observation(stamp=source_stamp(10, 10.0, 99.0)))
        tracker.update(raw_observation(stamp=source_stamp(11, 9.0)))
        accepted = tracker.update(
            raw_observation(
                stamp=source_stamp(11, 11.0),
                revision=12,
                center_offset=0.3,
            )
        )
        self.assertEqual(accepted.center_offset, 0.3)
        self.assertEqual(accepted.source_stamp.frame, 11)
        self.assertEqual(tracker.metrics.observations_accepted, 2)

    def test_explicit_stamp_must_match_embedded_source_evidence(self):
        tracker = TemporalLaneTracker()
        raw = raw_observation(stamp=source_stamp(1, 0.1, 0.1))
        before = tracker.latest_estimate
        with self.assertRaises(ValueError):
            tracker.update(raw, source_stamp=source_stamp(2, 0.2, 0.2))
        self.assertIs(tracker.latest_estimate, before)
        self.assertEqual(tracker.metrics.invalid_observations_rejected, 1)
        self.assertEqual(tracker.metrics.observations_accepted, 0)

    def test_unstamped_non_unusable_observation_is_rejected_without_state_mutation(self):
        tracker = TemporalLaneTracker()
        before = tracker.latest_estimate
        unstamped = raw_observation(LaneDetectionState.NOT_DETECTED, stamp=None)
        with self.assertRaises(ValueError):
            tracker.update(unstamped)
        self.assertIs(tracker.latest_estimate, before)
        self.assertEqual(tracker.metrics.observations_received, 1)
        self.assertEqual(tracker.metrics.invalid_observations_rejected, 1)
        self.assertEqual(tracker.metrics.observations_accepted, 0)

    def test_partial_embedded_stamp_is_invalid(self):
        raw = LaneObservation(
            snapshot_revision=1,
            source_frame=1,
            source_simulation_timestamp=None,
            source_receive_monotonic=0.0,
            state=LaneDetectionState.NOT_DETECTED,
            reason=LaneObservationReason.NO_PLAUSIBLE_BOUNDARY,
        )
        tracker = TemporalLaneTracker()
        with self.assertRaises(ValueError):
            tracker.update(raw)
        self.assertEqual(tracker.metrics.invalid_observations_rejected, 1)

    def test_partial_without_geometry_is_rejected_before_order_or_track_commit(self):
        tracker = TemporalLaneTracker()
        first_stamp = source_stamp(1, 0.0)
        baseline = tracker.update(raw_observation(stamp=first_stamp))
        malformed = raw_observation(
            LaneDetectionState.PARTIAL,
            stamp=source_stamp(2, 0.1),
            revision=2,
            left=None,
            right=None,
            confidence=0.5,
        )

        with self.assertRaises(ValueError):
            tracker.update(malformed)

        self.assertIs(tracker.latest_estimate, baseline)
        self.assertEqual(tracker.metrics.last_accepted_source_stamp, first_stamp)
        self.assertEqual(tracker.metrics.observations_accepted, 1)
        self.assertEqual(tracker.metrics.invalid_observations_rejected, 1)
        accepted = tracker.update(
            raw_observation(stamp=source_stamp(2, 0.1), revision=3)
        )
        self.assert_tracking(accepted)
        self.assertEqual(tracker.metrics.observations_accepted, 2)

    def test_invalid_input_type_is_rejected(self):
        tracker = TemporalLaneTracker()
        with self.assertRaises(TypeError):
            tracker.update(object())
        self.assertIs(tracker.latest_estimate.state, TemporalLaneState.UNINITIALIZED)
        self.assertEqual(tracker.metrics.observations_received, 1)
        self.assertEqual(tracker.metrics.invalid_observations_rejected, 1)

    def test_source_stamp_subclass_and_numeric_values_are_canonicalized(self):
        class StampSubclass(SampleStamp):
            __slots__ = ("__weakref__",)

        hostile = StampSubclass(
            frame=7,
            sim_time=Fraction(5, 4),
            monotonic=Fraction(19, 2),
        )
        reference = weakref.ref(hostile)
        tracker = TemporalLaneTracker()
        estimate = tracker.update(
            raw_observation(LaneDetectionState.INPUT_UNUSABLE, stamp=None),
            source_stamp=hostile,
        )
        self.assertIs(type(estimate.source_stamp), SampleStamp)
        self.assertIs(type(estimate.source_stamp.frame), int)
        self.assertIs(type(estimate.source_stamp.sim_time), float)
        self.assertIs(type(estimate.source_stamp.monotonic), float)
        self.assertIs(type(tracker.metrics.last_accepted_source_stamp), SampleStamp)
        self.assertIsNot(estimate.source_stamp, hostile)
        hostile = None
        gc.collect()
        self.assertIsNone(reference())

    def test_exact_caller_and_metrics_stamps_cannot_rewrite_order_baseline(self):
        caller_stamp = source_stamp(1, 0.0)
        tracker = TemporalLaneTracker()
        tracker.update(
            raw_observation(LaneDetectionState.INPUT_UNUSABLE, stamp=None),
            source_stamp=caller_stamp,
        )
        metrics_stamp = tracker.metrics.last_accepted_source_stamp
        self.assertIsNotNone(metrics_stamp)
        self.assertIsNot(metrics_stamp, caller_stamp)

        object.__setattr__(caller_stamp, "frame", 100)
        object.__setattr__(metrics_stamp, "frame", 200)
        self.assertEqual(tracker.metrics.last_accepted_source_stamp.frame, 1)
        accepted = tracker.update(
            raw_observation(stamp=source_stamp(2, 0.1), revision=2)
        )
        self.assert_tracking(accepted)
        self.assertEqual(accepted.source_stamp.frame, 2)

    def test_malformed_explicit_stamp_on_unusable_input_is_rejected(self):
        bad_stamps = (
            SampleStamp(True, 0.0, 0.0),
            SampleStamp(-1, 0.0, 0.0),
            SampleStamp(1, float("nan"), 0.0),
            SampleStamp(1, float("inf"), 0.0),
            SampleStamp(1, 0.0, float("-inf")),
        )
        for bad in bad_stamps:
            with self.subTest(stamp=bad):
                tracker = TemporalLaneTracker()
                before = tracker.latest_estimate
                with self.assertRaises(ValueError):
                    tracker.update(
                        raw_observation(
                            LaneDetectionState.INPUT_UNUSABLE,
                            stamp=None,
                        ),
                        source_stamp=bad,
                    )
                self.assertIs(tracker.latest_estimate, before)
                self.assertIsNone(tracker.metrics.last_accepted_source_stamp)
                self.assertEqual(tracker.metrics.observations_received, 1)
                self.assertEqual(tracker.metrics.observations_accepted, 0)
                self.assertEqual(tracker.metrics.invalid_observations_rejected, 1)
                self.assertEqual(tracker.metrics.input_unusable_updates, 0)
                self.assertEqual(tracker.metrics.unstamped_input_unusable_updates, 0)

    def test_raw_observation_subclass_is_not_retained(self):
        class WeakObservation(LaneObservation):
            pass

        base = raw_observation(stamp=source_stamp(1, 0.0))
        values = {field.name: getattr(base, field.name) for field in fields(LaneObservation)}
        hostile = WeakObservation(**values)
        reference = weakref.ref(hostile)
        tracker = TemporalLaneTracker()
        tracker.update(hostile)
        hostile = None
        values = None
        base = None
        gc.collect()
        self.assertIsNone(reference())
        self.assertIs(type(tracker.latest_estimate), TemporalLaneEstimate)
        self.assertIs(type(tracker.latest_estimate.left_boundary), TemporalLaneBoundary)


class ResetMetricsAndBoundednessTest(TemporalAssertions):
    def test_reset_returns_initial_state_and_resets_order_but_keeps_lifetime_metrics(self):
        tracker = TemporalLaneTracker()
        first = tracker.update(raw_observation(stamp=source_stamp(10, 1.0)))
        self.assert_tracking(first)

        reset = tracker.reset()

        self.assertIs(reset, tracker.latest_estimate)
        self.assertIs(reset.state, TemporalLaneState.UNINITIALIZED)
        self.assert_no_geometry(reset)
        self.assertIsNone(reset.source_stamp)
        self.assertIsNone(tracker.metrics.last_accepted_source_stamp)
        self.assertEqual(tracker.metrics.observations_received, 1)
        self.assertEqual(tracker.metrics.observations_accepted, 1)
        self.assertEqual(tracker.metrics.usable_updates, 1)
        self.assertEqual(tracker.metrics.resets, 1)

        same_old_source = tracker.update(
            raw_observation(stamp=source_stamp(10, 1.0), revision=2)
        )
        self.assert_tracking(same_old_source)
        self.assertEqual(tracker.metrics.observations_accepted, 2)
        self.assertEqual(tracker.metrics.duplicates_rejected, 0)

    def test_reset_is_safe_from_every_noninitial_state_and_repeated(self):
        for target in (
            TemporalLaneState.TRACKING,
            TemporalLaneState.COASTING,
            TemporalLaneState.LOST,
            TemporalLaneState.INPUT_UNUSABLE,
        ):
            with self.subTest(target=target):
                tracker = TemporalLaneTracker(
                    TemporalLaneTrackerConfig(max_consecutive_misses=0)
                )
                tracker.update(raw_observation(stamp=source_stamp(1, 0.0)))
                if target is TemporalLaneState.COASTING:
                    tracker = TemporalLaneTracker()
                    tracker.update(raw_observation(stamp=source_stamp(1, 0.0)))
                    tracker.update(
                        raw_observation(
                            LaneDetectionState.NOT_DETECTED,
                            stamp=source_stamp(2, 0.1),
                        )
                    )
                elif target is TemporalLaneState.LOST:
                    tracker.update(
                        raw_observation(
                            LaneDetectionState.NOT_DETECTED,
                            stamp=source_stamp(2, 0.1),
                        )
                    )
                elif target is TemporalLaneState.INPUT_UNUSABLE:
                    tracker.update(
                        raw_observation(LaneDetectionState.INPUT_UNUSABLE, stamp=None),
                        source_stamp=source_stamp(2, 0.1),
                    )
                self.assertIs(tracker.latest_estimate.state, target)
                first_reset = tracker.reset()
                second_reset = tracker.reset()
                self.assertIs(first_reset.state, TemporalLaneState.UNINITIALIZED)
                self.assertIs(second_reset.state, TemporalLaneState.UNINITIALIZED)
                self.assert_no_geometry(second_reset)
                self.assertEqual(tracker.metrics.resets, 2)

    def test_metrics_accounting_identity_and_atomic_snapshots(self):
        tracker = TemporalLaneTracker()
        before = tracker.metrics
        tracker.update(raw_observation(stamp=source_stamp(10, 1.0)))
        tracker.update(raw_observation(stamp=source_stamp(10, 1.0, 99.0)))
        tracker.update(raw_observation(stamp=source_stamp(9, 0.9)))
        with self.assertRaises(ValueError):
            tracker.update(raw_observation(LaneDetectionState.NOT_DETECTED, stamp=None))
        tracker.update(
            raw_observation(LaneDetectionState.INPUT_UNUSABLE, stamp=None)
        )
        metrics = tracker.metrics

        self.assertEqual(before.observations_received, 0)
        self.assertEqual(
            metrics.observations_received,
            metrics.observations_accepted
            + metrics.duplicates_rejected
            + metrics.out_of_order_rejected
            + metrics.invalid_observations_rejected,
        )
        self.assertEqual(
            metrics.observations_accepted,
            metrics.usable_updates
            + metrics.coast_updates
            + metrics.lost_updates
            + metrics.input_unusable_updates,
        )
        self.assertLessEqual(
            metrics.unstamped_input_unusable_updates,
            metrics.input_unusable_updates,
        )
        self.assertEqual(metrics.observations_received, 5)
        self.assertEqual(metrics.observations_accepted, 2)
        self.assertEqual(metrics.duplicates_rejected, 1)
        self.assertEqual(metrics.out_of_order_rejected, 1)
        self.assertEqual(metrics.invalid_observations_rejected, 1)

    def test_long_miss_stream_retains_no_history_and_accounting_is_exact(self):
        miss_count = 100_000
        tracker = TemporalLaneTracker(
            TemporalLaneTrackerConfig(
                max_coast_age_seconds=10.0,
                coast_confidence_decay=0.99,
                minimum_coast_confidence=0.001,
            )
        )
        tracker.update(raw_observation(stamp=source_stamp(1, 0.0)))
        for index in range(miss_count):
            frame = index + 2
            tracker.update(
                raw_observation(
                    LaneDetectionState.NOT_DETECTED,
                    stamp=source_stamp(frame, 0.0),
                    revision=frame,
                )
            )

        estimate = tracker.latest_estimate
        metrics = tracker.metrics
        self.assertIs(estimate.state, TemporalLaneState.LOST)
        self.assert_no_geometry(estimate)
        self.assertEqual(estimate.consecutive_misses, miss_count)
        self.assertEqual(metrics.observations_received, miss_count + 1)
        self.assertEqual(metrics.observations_accepted, miss_count + 1)
        self.assertEqual(metrics.usable_updates, 1)
        self.assertEqual(metrics.coast_updates, 2)
        self.assertEqual(metrics.lost_updates, miss_count - 2)
        self.assertEqual(metrics.maximum_consecutive_misses, miss_count)
        self.assertFalse(hasattr(tracker, "__dict__"))
        slot_names = set(TemporalLaneTracker.__slots__)
        self.assertFalse(
            any(
                token in name.lower()
                for name in slot_names
                for token in ("history", "queue", "deque", "buffer", "frames")
            )
        )
        for name in slot_names:
            value = getattr(tracker, name)
            self.assertNotIsInstance(
                value,
                (list, dict, set, collections.deque),
                name,
            )

    def test_duplicate_flood_has_zero_track_mutation_and_bounded_state(self):
        tracker = TemporalLaneTracker()
        raw = raw_observation(stamp=source_stamp(1, 0.0))
        baseline = tracker.update(raw)
        for _ in range(10_000):
            self.assertIs(tracker.update(raw), baseline)
        self.assertIs(tracker.latest_estimate, baseline)
        self.assertEqual(tracker.metrics.duplicates_rejected, 10_000)
        self.assertEqual(tracker.metrics.observations_accepted, 1)

    def test_alternating_detection_and_dropout_recovers_deterministically(self):
        cycles = 1_000
        tracker = TemporalLaneTracker()
        tracker.update(raw_observation(stamp=source_stamp(1, 0.0)))
        frame = 1
        prior_tracking_confidence = tracker.latest_estimate.confidence
        for cycle in range(cycles):
            frame += 1
            coast = tracker.update(
                raw_observation(
                    LaneDetectionState.NOT_DETECTED,
                    stamp=source_stamp(frame, frame / 100.0),
                    revision=frame,
                )
            )
            self.assertIs(coast.state, TemporalLaneState.COASTING)
            self.assertLessEqual(coast.confidence, prior_tracking_confidence)
            frame += 1
            tracked = tracker.update(
                raw_observation(
                    stamp=source_stamp(frame, frame / 100.0),
                    revision=frame,
                )
            )
            self.assert_tracking(tracked)
            prior_tracking_confidence = tracked.confidence
        self.assertEqual(tracker.metrics.coast_updates, cycles)
        self.assertEqual(tracker.metrics.recoveries, cycles)

    def test_reset_stress_retains_no_prior_order_or_geometry(self):
        tracker = TemporalLaneTracker()
        for _ in range(1_000):
            tracker.update(raw_observation(stamp=source_stamp(1, 0.0)))
            reset = tracker.reset()
            self.assertIs(reset.state, TemporalLaneState.UNINITIALIZED)
            self.assertIsNone(tracker.metrics.last_accepted_source_stamp)
            self.assert_no_geometry(reset)
        self.assertEqual(tracker.metrics.resets, 1_000)
        self.assertEqual(tracker.metrics.observations_accepted, 1_000)


class DeterminismRetentionAndImportSafetyTest(TemporalAssertions):
    def test_two_independent_trackers_produce_equality_identical_streams(self):
        streams = (
            (raw_observation(stamp=source_stamp(1, 0.0)), None),
            (
                raw_observation(
                    LaneDetectionState.PARTIAL,
                    stamp=source_stamp(2, 0.1),
                    revision=2,
                    left=left_boundary(0.60, -0.30, 0.7),
                ),
                None,
            ),
            (
                raw_observation(
                    LaneDetectionState.NOT_DETECTED,
                    stamp=source_stamp(3, 0.2),
                    revision=3,
                ),
                None,
            ),
            (
                raw_observation(
                    LaneDetectionState.INPUT_UNUSABLE,
                    stamp=None,
                    revision=4,
                ),
                source_stamp(4, 0.25),
            ),
            (
                raw_observation(
                    stamp=source_stamp(5, 0.3),
                    revision=5,
                    center_offset=0.1,
                ),
                None,
            ),
        )
        first = TemporalLaneTracker()
        second = TemporalLaneTracker()
        first_results = []
        second_results = []
        for observation, explicit_stamp in streams:
            first_results.append(
                first.update(observation, source_stamp=explicit_stamp)
            )
            second_results.append(
                second.update(observation, source_stamp=explicit_stamp)
            )
        self.assertEqual(first_results, second_results)
        self.assertEqual(first.metrics, second.metrics)

    def test_tracker_does_not_retain_replaced_raw_observation(self):
        class WeakObservation(LaneObservation):
            pass

        base = raw_observation(stamp=source_stamp(1, 0.0))
        raw_values = {
            field.name: getattr(base, field.name) for field in fields(LaneObservation)
        }
        raw = WeakObservation(**raw_values)
        raw_reference = weakref.ref(raw)
        tracker = TemporalLaneTracker()
        first_estimate = tracker.update(raw)
        raw = None
        base = None
        raw_values = None
        first_estimate = None
        tracker.update(raw_observation(stamp=source_stamp(2, 0.1), revision=2))
        gc.collect()
        self.assertIsNone(raw_reference())
        self.assertIsNot(tracker.latest_estimate, first_estimate)

    def test_temporal_import_and_construction_are_dependency_and_thread_safe(self):
        repository = pathlib.Path(__file__).resolve().parents[1]
        code = textwrap.dedent(
            """
            import socket
            import sys
            import threading

            def forbidden_activity(*_args, **_kwargs):
                raise AssertionError('temporal import attempted socket/thread activity')

            before = tuple(threading.enumerate())
            socket.socket = forbidden_activity
            socket.create_connection = forbidden_activity
            threading.Thread.start = forbidden_activity
            from perception.temporal import TemporalLanePipeline, TemporalLaneTracker
            from perception import (
                TemporalLanePipeline as PackageTemporalLanePipeline,
                TemporalLaneTracker as PackageTemporalLaneTracker,
            )
            after_import = tuple(threading.enumerate())
            tracker = TemporalLaneTracker()
            pipeline = TemporalLanePipeline(perception_callable=lambda value: None)
            after_construction = tuple(threading.enumerate())
            forbidden = ('numpy', 'carla', 'pygame', 'cv2', 'scipy')
            assert not [name for name in forbidden if name in sys.modules]
            assert after_import == before
            assert after_construction == before
            assert PackageTemporalLanePipeline is TemporalLanePipeline
            assert PackageTemporalLaneTracker is TemporalLaneTracker
            assert tracker.latest_estimate.state.value == 'uninitialized'
            assert pipeline.latest_estimate.state.value == 'uninitialized'
            print('TEMPORAL_IMPORT_SAFE')
            """
        )
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "TEMPORAL_IMPORT_SAFE")

    def test_updates_do_not_read_host_clock(self):
        tracker = TemporalLaneTracker()
        with mock.patch(
            "time.monotonic",
            side_effect=AssertionError("temporal tracker read host clock"),
        ):
            tracker.update(raw_observation(stamp=source_stamp(1, 0.0)))
            tracker.update(
                raw_observation(
                    LaneDetectionState.NOT_DETECTED,
                    stamp=source_stamp(2, 0.1),
                )
            )


class PipelineTest(TemporalAssertions):
    def test_pipeline_is_runtime_compatible_and_attaches_temporal_estimate(self):
        stamp = source_stamp(7, 1.25, 9.5)
        snapshot = snapshot_with_stamp(stamp, revision=19)
        expected_raw = raw_observation(
            stamp=stamp,
            revision=19,
            center_offset=0.08,
            heading_proxy=-0.06,
        )
        calls = []

        def perceive(value):
            calls.append(value)
            return expected_raw

        pipeline = TemporalLanePipeline(perception_callable=perceive)
        result = pipeline(snapshot)

        self.assertEqual(calls, [snapshot])
        self.assertIsInstance(result, TemporalLaneObservation)
        self.assertIsInstance(result, LaneObservation)
        for field in fields(LaneObservation):
            self.assertEqual(
                getattr(result, field.name),
                getattr(expected_raw, field.name),
                field.name,
            )
        self.assertIs(result.temporal_estimate, pipeline.latest_estimate)
        self.assertIs(result.temporal_estimate.state, TemporalLaneState.TRACKING)
        self.assertEqual(result.temporal_estimate.source_stamp, stamp)
        self.assertIs(pipeline.tracker.latest_estimate, pipeline.latest_estimate)
        self.assertEqual(pipeline.metrics, pipeline.tracker.metrics)

    def test_pipeline_supplies_snapshot_stamp_for_raw_input_unusable(self):
        stamp = source_stamp(3, 0.3, 2.0)
        snapshot = snapshot_with_stamp(stamp, revision=5)
        pipeline = TemporalLanePipeline(
            perception_callable=lambda _value: raw_observation(
                LaneDetectionState.INPUT_UNUSABLE,
                stamp=None,
                revision=5,
            )
        )
        result = pipeline(snapshot)
        self.assertIs(result.state, LaneDetectionState.INPUT_UNUSABLE)
        self.assertIs(
            result.temporal_estimate.state,
            TemporalLaneState.INPUT_UNUSABLE,
        )
        self.assertEqual(result.temporal_estimate.source_stamp, stamp)
        self.assertEqual(pipeline.metrics.unstamped_input_unusable_updates, 0)

    def test_pipeline_config_properties_and_reset_delegate_to_tracker(self):
        config = TemporalLaneTrackerConfig(smoothing_alpha=0.5)
        stamp = source_stamp(1, 0.0)
        pipeline = TemporalLanePipeline(
            config,
            perception_callable=lambda _value: raw_observation(stamp=stamp),
        )
        self.assertIsNot(pipeline.config, config)
        self.assertEqual(pipeline.config, config)
        self.assertIsNot(pipeline.config, pipeline.tracker.config)
        self.assertEqual(pipeline.config, pipeline.tracker.config)
        pipeline(snapshot_with_stamp(stamp))
        reset = pipeline.reset()
        self.assertIs(reset, pipeline.latest_estimate)
        self.assertIs(reset.state, TemporalLaneState.UNINITIALIZED)
        self.assertEqual(pipeline.metrics.resets, 1)

    def test_pipeline_rejection_returns_last_accepted_atomic_carrier(self):
        def perceive(value):
            stamp = value.camera.metadata.stamp
            return raw_observation(stamp=stamp, revision=value.revision)

        pipeline = TemporalLanePipeline(perception_callable=perceive)
        accepted = pipeline(
            snapshot_with_stamp(source_stamp(10, 1.0), revision=10)
        )
        rejected = pipeline(
            snapshot_with_stamp(source_stamp(9, 0.9), revision=9)
        )
        self.assertIs(rejected, accepted)
        self.assertEqual(rejected.source_frame, 10)
        self.assertEqual(rejected.temporal_estimate.source_stamp.frame, 10)
        self.assertEqual(pipeline.metrics.out_of_order_rejected, 1)

        values = {
            field.name: getattr(accepted, field.name)
            for field in fields(LaneObservation)
        }
        mismatched = replace(
            accepted.temporal_estimate,
            snapshot_revision=accepted.snapshot_revision + 1,
        )
        with self.assertRaises(ValueError):
            TemporalLaneObservation(**values, temporal_estimate=mismatched)

    def test_pipeline_rejects_invalid_callable_and_invalid_result(self):
        with self.assertRaises(TypeError):
            TemporalLanePipeline(perception_callable=None)

        pipeline = TemporalLanePipeline(perception_callable=lambda _value: object())
        with self.assertRaises(TypeError):
            pipeline(snapshot_with_stamp(source_stamp(1, 0.0)))
        self.assertIs(
            pipeline.latest_estimate.state,
            TemporalLaneState.UNINITIALIZED,
        )
        self.assertEqual(pipeline.metrics.observations_received, 0)

    def test_temporal_lane_observation_is_frozen_slotted_and_requires_estimate(self):
        stamp = source_stamp(1, 0.0)
        pipeline = TemporalLanePipeline(
            perception_callable=lambda _value: raw_observation(stamp=stamp)
        )
        result = pipeline(snapshot_with_stamp(stamp))
        self.assertFalse(hasattr(result, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            result.temporal_estimate = pipeline.latest_estimate

        raw = raw_observation(stamp=stamp)
        values = {field.name: getattr(raw, field.name) for field in fields(LaneObservation)}
        with self.assertRaises(TypeError):
            TemporalLaneObservation(**values)


if __name__ == "__main__":
    unittest.main()
