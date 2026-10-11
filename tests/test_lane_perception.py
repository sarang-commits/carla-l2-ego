"""Offline synthetic tests for Mission 5 image-space lane perception."""

from __future__ import annotations

import ast
import collections
import contextlib
import dataclasses
import enum
import functools
import math
import os
import pathlib
import subprocess
import sys
import textwrap
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from unittest import mock

import numpy

import perception
from perception.lane import (
    MAX_BOUNDARY_HYPOTHESES,
    MAX_BOUNDARY_SEEDS,
    MAX_COMPLETED_TRACKS,
    MAX_OUTGOING_LINKS_PER_REPRESENTATIVE,
    MAX_PAIR_COMBINATIONS,
    MAX_PARTIAL_TRACKS_PER_BIN,
    MAX_PARTIAL_TRACKS_PER_REPRESENTATIVE,
    MAX_REPRESENTATIVES_PER_BIN,
    MAX_TRACK_LINKS_PER_BIN_GAP,
    MAX_TRACK_SKIPPED_BINS,
    _EMPTY_TRACK_FIT,
    _CandidateRepresentative,
    _RepresentativeTrack,
    _TrackFit,
    _accepted_track_quality_key,
    _bounded_outgoing_links,
    _boundary_x,
    _camera_gate,
    _candidate_representatives,
    _closed_form_fit,
    _coerce_representatives,
    _continuous_representative_tracks,
    _extended_track,
    _fit_boundary,
    _fit_boundary_hypotheses,
    _line_is_oriented,
    _maximum_individual_error_tolerance,
    _observation_from_boundaries,
    _pair_geometry,
    _representative_identity,
    _residual_is_acceptable,
    _roi_pixel_bounds,
    _scalar_line,
    _select_boundaries,
    _select_tracks_by_accepted_quality,
    _track_can_extend,
    _track_prediction_tolerance,
    _track_quality_key,
    _track_support_is_valid,
    _representatives_can_link,
)
from perception.state import MAX_CONFIGURED_IMAGE_PIXELS, MAX_VERTICAL_BINS
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


WHITE = (245, 245, 245)
YELLOW = (235, 205, 40)
WIDTH = 160
HEIGHT = 100
BASE_LEFT = (0.60, -0.40)
BASE_RIGHT = (0.40, 0.40)
_DEFAULT = object()


def run_isolated_python(code: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def import_safety_probe_code(import_statement: str) -> str:
    return textwrap.dedent(
        f"""
        import _thread
        import builtins
        import io
        import os
        import sys
        import threading
        import time

        forbidden_modules = (
            "numpy",
            "carla",
            "pygame",
            "manual_drive",
            "socket",
        )
        assert not [name for name in forbidden_modules if name in sys.modules]

        environment_before = dict(os.environ)
        environment_identity = id(os.environ)
        putenv_calls = []
        unsetenv_calls = []
        filesystem_calls = []
        clock_calls = []
        thread_calls = []
        threads_before = tuple(
            sorted(thread.ident for thread in threading.enumerate())
        )

        original_putenv = os.putenv
        original_unsetenv = os.unsetenv

        def recording_putenv(name, value):
            putenv_calls.append((repr(name), repr(value)))
            return original_putenv(name, value)

        def recording_unsetenv(name):
            unsetenv_calls.append(repr(name))
            return original_unsetenv(name)

        os.putenv = recording_putenv
        os.unsetenv = recording_unsetenv

        def recording_wrapper(calls, label, original):
            def wrapped(*args, **kwargs):
                calls.append(label)
                return original(*args, **kwargs)
            return wrapped

        for owner, name in (
            (builtins, "open"),
            (io, "open"),
            (os, "open"),
            (os, "listdir"),
            (os, "scandir"),
            (os, "stat"),
            (os, "lstat"),
            (os, "access"),
        ):
            setattr(
                owner,
                name,
                recording_wrapper(
                    filesystem_calls,
                    f"{{owner.__name__}}.{{name}}",
                    getattr(owner, name),
                ),
            )

        for name in (
            "time",
            "time_ns",
            "monotonic",
            "monotonic_ns",
            "perf_counter",
            "perf_counter_ns",
            "process_time",
            "process_time_ns",
            "thread_time",
            "thread_time_ns",
        ):
            if hasattr(time, name):
                setattr(
                    time,
                    name,
                    recording_wrapper(clock_calls, f"time.{{name}}", getattr(time, name)),
                )

        threading.Thread.start = recording_wrapper(
            thread_calls, "threading.Thread.start", threading.Thread.start
        )
        _thread.start_new_thread = recording_wrapper(
            thread_calls, "_thread.start_new_thread", _thread.start_new_thread
        )

        {import_statement}

        threads_after = tuple(
            sorted(thread.ident for thread in threading.enumerate())
        )
        assert "numpy" not in sys.modules
        assert not [name for name in forbidden_modules[1:] if name in sys.modules]
        assert not putenv_calls, putenv_calls
        assert not unsetenv_calls, unsetenv_calls
        assert id(os.environ) == environment_identity
        assert dict(os.environ) == environment_before
        assert not filesystem_calls, filesystem_calls
        assert not clock_calls, clock_calls
        assert not thread_calls, thread_calls
        assert threads_after == threads_before, (threads_before, threads_after)
        print("IMPORT_SAFE")
        """
    )


def healthy_camera_status(**changes) -> SensorStatus:
    status = SensorStatus(
        name="rgb_camera",
        kind=SensorKind.CONTINUOUS,
        lifecycle=SensorLifecycle.ATTACHED,
        readiness=SensorReadiness.READY,
        freshness=SensorFreshness.FRESH,
        operational_health=OperationalHealth.HEALTHY,
        evaluated_monotonic=100.1,
        age_seconds=0.1,
        expected_interval=0.1,
        stale_after=0.5,
        accepted_count=1,
        event_count=None,
        error_count=0,
        reason=None,
    )
    return replace(status, **changes)


def rgb_frame(
    lines=(),
    *,
    width=WIDTH,
    height=HEIGHT,
    frame=42,
    sim_time=12.5,
    monotonic=100.0,
    horizontal=False,
    fill_roi=False,
) -> CameraFrame:
    payload = bytearray(width * height * 3)
    if fill_roi:
        for y in range(height // 2, height):
            for x in range(width):
                offset = (y * width + x) * 3
                payload[offset : offset + 3] = bytes(WHITE)
    for intercept, slope, color in lines:
        for y in range(height // 2, height):
            normalized_y = y / float(max(1, height - 1))
            center_x = int(round((intercept + slope * normalized_y) * (width - 1)))
            for x in range(center_x - 2, center_x + 3):
                if 0 <= x < width:
                    offset = (y * width + x) * 3
                    payload[offset : offset + 3] = bytes(color)
    if horizontal:
        y = int(round(0.75 * (height - 1)))
        for x in range(width):
            offset = (y * width + x) * 3
            payload[offset : offset + 3] = bytes(WHITE)
    metadata = CameraFrameMetadata(
        stamp=SampleStamp(frame=frame, sim_time=sim_time, monotonic=monotonic),
        width_px=width,
        height_px=height,
        horizontal_fov_deg=90.0,
        pixel_format=PIXEL_FORMAT_RGB8,
        row_stride_bytes=width * 3,
        transform=None,
    )
    return CameraFrame(metadata=metadata, rgb_bytes=bytes(payload))


def scene_frame(
    markings=(),
    *,
    width=WIDTH,
    height=HEIGHT,
    bright_pixels=(),
) -> CameraFrame:
    """Author a synthetic RGB8 scene without using detector internals.

    Markings are ``(intercept, slope, color, pixel_width)`` with optional
    normalized ``start_y`` and ``end_y`` items.
    """
    payload = bytearray(width * height * 3)
    for marking in markings:
        if len(marking) == 4:
            intercept, slope, color, pixel_width = marking
            start_y, end_y = 0.5, 1.0
        else:
            intercept, slope, color, pixel_width, start_y, end_y = marking
        half_width = pixel_width // 2
        for y in range(height):
            normalized_y = y / float(max(1, height - 1))
            if not start_y <= normalized_y <= end_y:
                continue
            center_x = int(
                round((intercept + slope * normalized_y) * (width - 1))
            )
            first_x = center_x - half_width
            for x in range(first_x, first_x + pixel_width):
                if 0 <= x < width:
                    offset = (y * width + x) * 3
                    payload[offset : offset + 3] = bytes(color)
    for x, y, color in bright_pixels:
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            payload[offset : offset + 3] = bytes(color)
    metadata = CameraFrameMetadata(
        stamp=SampleStamp(frame=42, sim_time=12.5, monotonic=100.0),
        width_px=width,
        height_px=height,
        horizontal_fov_deg=90.0,
        pixel_format=PIXEL_FORMAT_RGB8,
        row_stride_bytes=width * 3,
        transform=None,
    )
    return CameraFrame(metadata=metadata, rgb_bytes=bytes(payload))


def erase_vertical_bins(
    frame: CameraFrame,
    bins: tuple[int, ...],
    config=perception.DEFAULT_LANE_CONFIG,
) -> CameraFrame:
    payload = bytearray(frame.rgb_bytes)
    width = frame.metadata.width_px
    _, y0, _, y1 = _roi_pixel_bounds(
        width,
        frame.metadata.height_px,
        config,
    )
    roi_height = y1 - y0
    for bin_index in bins:
        row_start = y0 + (
            bin_index * roi_height
        ) // config.vertical_bin_count
        row_end = y0 + (
            (bin_index + 1) * roi_height
        ) // config.vertical_bin_count
        for row in range(row_start, row_end):
            offset = row * width * 3
            payload[offset : offset + width * 3] = bytes(width * 3)
    return replace(frame, rgb_bytes=bytes(payload))


def observed_support_distances(
    frame: CameraFrame,
    boundary_value: perception.LaneBoundaryObservation,
    normalized_rows: tuple[float, ...],
) -> tuple[float, ...]:
    width = frame.metadata.width_px
    height = frame.metadata.height_px
    distances = []
    for requested_y in normalized_rows:
        row = int(round(requested_y * (height - 1)))
        actual_y = row / float(max(1, height - 1))
        predicted_x = _boundary_x(boundary_value, actual_y)
        observed_xs = []
        for column in range(width):
            offset = (row * width + column) * 3
            if frame.rgb_bytes[offset : offset + 3] != b"\0\0\0":
                observed_xs.append(column / float(max(1, width - 1)))
        distances.append(
            min(
                (abs(observed_x - predicted_x) for observed_x in observed_xs),
                default=math.inf,
            )
        )
    return tuple(distances)


def nearest_marking_sources(
    boundary_value: perception.LaneBoundaryObservation,
    markings,
    normalized_rows: tuple[float, ...],
) -> tuple[int, ...]:
    sources = []
    for normalized_y in normalized_rows:
        predicted_x = _boundary_x(boundary_value, normalized_y)
        sources.append(
            min(
                range(len(markings)),
                key=lambda index: (
                    abs(
                        predicted_x
                        - (
                            markings[index][0]
                            + markings[index][1] * normalized_y
                        )
                    ),
                    index,
                ),
            )
        )
    return tuple(sources)


def snapshot(*, camera=_DEFAULT, statuses=_DEFAULT, revision=7) -> TelemetrySnapshot:
    selected_camera = rgb_frame() if camera is _DEFAULT else camera
    selected_statuses = (
        (healthy_camera_status(),) if statuses is _DEFAULT else statuses
    )
    return TelemetrySnapshot(
        revision=revision,
        captured_monotonic=100.1,
        camera=selected_camera,
        statuses=selected_statuses,
    )


def detected_frame(left=BASE_LEFT, right=BASE_RIGHT, left_color=WHITE, right_color=WHITE):
    return rgb_frame(
        lines=(
            (left[0], left[1], left_color),
            (right[0], right[1], right_color),
        )
    )


def boundary(
    side: perception.LaneBoundarySide,
    intercept: float,
    slope: float,
    *,
    confidence: float = 0.9,
    config: perception.LanePerceptionConfig = perception.DEFAULT_LANE_CONFIG,
) -> perception.LaneBoundaryObservation:
    top = config.roi[1]
    bottom = config.roi[3]
    return perception.LaneBoundaryObservation(
        side=side,
        endpoints=(
            perception.NormalizedImagePoint(intercept + slope * top, top),
            perception.NormalizedImagePoint(intercept + slope * bottom, bottom),
        ),
        x_intercept=intercept,
        x_slope=slope,
        support_count=config.vertical_bin_count,
        fit_residual=0.0,
        confidence=confidence,
    )


def assert_no_complete_geometry(testcase: unittest.TestCase, observation) -> None:
    testcase.assertIsNone(observation.centerline_endpoints)
    testcase.assertIsNone(observation.center_offset)
    testcase.assertIsNone(observation.heading_proxy)


class ConfigurationTest(unittest.TestCase):
    def test_default_configuration_is_valid_and_frozen(self):
        config = perception.LanePerceptionConfig()
        self.assertEqual(config, perception.DEFAULT_LANE_CONFIG)
        with self.assertRaises(FrozenInstanceError):
            config.vertical_bin_count = 20

    def test_bool_rejected_for_integer_and_number_fields(self):
        for changes in (
            {"vertical_bin_count": True},
            {"max_image_pixels": False},
            {"line_fit_tolerance": True},
            {"center_evaluation_y": False},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(perception.DEFAULT_LANE_CONFIG, **changes)

    def test_wrong_numeric_types_are_rejected(self):
        for bad in ("0.2", None, object(), complex(1.0, 0.0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(perception.DEFAULT_LANE_CONFIG, line_fit_tolerance=bad)

    def test_nan_and_infinities_are_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(perception.DEFAULT_LANE_CONFIG, max_fit_residual=bad)

    def test_inverted_and_empty_roi_are_rejected(self):
        for roi in ((0.8, 0.5, 0.2, 1.0), (0.2, 0.5, 0.2, 1.0), (0.0, 0.8, 1.0, 0.7), (0.0, 0.5, 1.0, 0.5)):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                replace(perception.DEFAULT_LANE_CONFIG, roi=roi)

    def test_roi_outside_unit_interval_is_rejected(self):
        for roi in ((-0.01, 0.5, 1.0, 1.0), (0.0, 0.5, 1.01, 1.0)):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                replace(perception.DEFAULT_LANE_CONFIG, roi=roi)

    def test_invalid_rgb_threshold_collections_are_rejected(self):
        for changes in (
            {"white_min_rgb": [200, 200, 200]},
            {"white_min_rgb": (200, 200)},
            {"white_min_rgb": (200, True, 200)},
            {"yellow_max_rgb": (255, 255, 256)},
            {"white_max_channel_spread": -1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(perception.DEFAULT_LANE_CONFIG, **changes)

    def test_contradictory_rgb_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            replace(
                perception.DEFAULT_LANE_CONFIG,
                yellow_min_rgb=(200, 140, 0),
                yellow_max_rgb=(199, 255, 140),
            )

    def test_invalid_bin_and_support_counts_are_rejected(self):
        for changes in (
            {"vertical_bin_count": 0},
            {"vertical_bin_count": MAX_VERTICAL_BINS + 1},
            {"min_support_bins": 0},
            {"min_support_bins": 19},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(perception.DEFAULT_LANE_CONFIG, **changes)

    def test_nonpositive_tolerances_and_span_are_rejected(self):
        for changes in (
            {"line_fit_tolerance": 0.0},
            {"max_fit_residual": -0.1},
            {"min_vertical_span": 0.0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(perception.DEFAULT_LANE_CONFIG, **changes)

    def test_contradictory_fit_geometry_is_rejected(self):
        for changes in (
            {"max_fit_residual": 0.04},
            {"min_vertical_span": 0.51},
            {"min_abs_slope": 0.9},
            {"min_lane_width": 0.9},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(perception.DEFAULT_LANE_CONFIG, **changes)

    def test_confidence_bounds_and_order_are_validated(self):
        for changes in (
            {"min_boundary_confidence": -0.01},
            {"detection_confidence": 1.01},
            {"min_boundary_confidence": 0.8, "detection_confidence": 0.7},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(perception.DEFAULT_LANE_CONFIG, **changes)

    def test_center_evaluation_must_be_inside_roi(self):
        with self.assertRaises(ValueError):
            replace(perception.DEFAULT_LANE_CONFIG, center_evaluation_y=0.49)

    def test_maximum_image_size_is_explicitly_bounded(self):
        accepted = replace(
            perception.DEFAULT_LANE_CONFIG,
            max_image_pixels=MAX_CONFIGURED_IMAGE_PIXELS,
        )
        self.assertEqual(accepted.max_image_pixels, MAX_CONFIGURED_IMAGE_PIXELS)
        with self.assertRaises(ValueError):
            replace(
                perception.DEFAULT_LANE_CONFIG,
                max_image_pixels=MAX_CONFIGURED_IMAGE_PIXELS + 1,
            )


class InputGatingTest(unittest.TestCase):
    def assert_unusable(self, value, reason=None):
        observation = perception.perceive_lanes(value)
        self.assertIs(observation.state, perception.LaneDetectionState.INPUT_UNUSABLE)
        if reason is not None:
            self.assertIs(observation.reason, reason)
        self.assertIsNone(observation.left_boundary)
        self.assertIsNone(observation.right_boundary)
        assert_no_complete_geometry(self, observation)
        return observation

    def test_non_snapshot_raises_type_error(self):
        for value in (None, object(), "snapshot"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                perception.perceive_lanes(value)

    def test_wrong_config_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            perception.perceive_lanes(snapshot(), None)

    def test_missing_status_is_unusable(self):
        self.assert_unusable(snapshot(statuses=()), perception.LaneObservationReason.STATUS_MISSING)

    def test_duplicate_camera_statuses_are_unusable(self):
        status = healthy_camera_status()
        self.assert_unusable(snapshot(statuses=(status, status)), perception.LaneObservationReason.STATUS_DUPLICATED)

    def test_unavailable_status_is_unusable(self):
        status = healthy_camera_status(
            lifecycle=SensorLifecycle.NOT_AVAILABLE,
            readiness=SensorReadiness.NOT_APPLICABLE,
            freshness=SensorFreshness.NOT_APPLICABLE,
            age_seconds=None,
            accepted_count=0,
        )
        self.assert_unusable(snapshot(statuses=(status,)))

    def test_not_attached_status_is_unusable(self):
        status = healthy_camera_status(
            lifecycle=SensorLifecycle.NOT_ATTACHED,
            readiness=SensorReadiness.NOT_APPLICABLE,
            freshness=SensorFreshness.NOT_APPLICABLE,
            age_seconds=None,
            accepted_count=0,
        )
        self.assert_unusable(snapshot(statuses=(status,)))

    def test_attachment_failed_status_is_unusable(self):
        status = healthy_camera_status(
            lifecycle=SensorLifecycle.ATTACHMENT_FAILED,
            readiness=SensorReadiness.NOT_APPLICABLE,
            freshness=SensorFreshness.NOT_APPLICABLE,
            operational_health=OperationalHealth.FAILED,
            age_seconds=None,
            accepted_count=0,
        )
        self.assert_unusable(snapshot(statuses=(status,)))

    def test_waiting_status_is_unusable(self):
        status = healthy_camera_status(
            readiness=SensorReadiness.WAITING_FOR_FIRST_SAMPLE,
            freshness=SensorFreshness.WAITING_FOR_FIRST_SAMPLE,
            accepted_count=0,
        )
        self.assert_unusable(snapshot(statuses=(status,)))

    def test_stale_status_with_payload_is_unusable(self):
        status = healthy_camera_status(freshness=SensorFreshness.STALE, age_seconds=0.6)
        self.assert_unusable(snapshot(camera=detected_frame(), statuses=(status,)))

    def test_degraded_status_with_recent_payload_is_unusable(self):
        status = healthy_camera_status(operational_health=OperationalHealth.DEGRADED)
        self.assert_unusable(snapshot(camera=detected_frame(), statuses=(status,)))

    def test_failed_health_is_unusable(self):
        self.assert_unusable(snapshot(statuses=(healthy_camera_status(operational_health=OperationalHealth.FAILED),)))

    def test_destroyed_status_is_unusable(self):
        status = healthy_camera_status(
            lifecycle=SensorLifecycle.DESTROYED,
            readiness=SensorReadiness.NOT_APPLICABLE,
            freshness=SensorFreshness.NOT_APPLICABLE,
            age_seconds=None,
        )
        self.assert_unusable(snapshot(statuses=(status,)))

    def test_noncontinuous_camera_status_is_unusable(self):
        self.assert_unusable(snapshot(statuses=(healthy_camera_status(kind=SensorKind.EVENT),)))

    def test_incoherent_status_record_is_unusable(self):
        status = healthy_camera_status(accepted_count=True)
        self.assert_unusable(snapshot(statuses=(status,)), perception.LaneObservationReason.STATUS_INCOHERENT)

    def test_continuous_camera_rejects_event_count_evidence(self):
        status = healthy_camera_status(event_count=1)
        self.assert_unusable(
            snapshot(statuses=(status,)),
            perception.LaneObservationReason.STATUS_INCOHERENT,
        )

    def test_raw_status_free_snapshot_fails_closed(self):
        self.assert_unusable(
            snapshot(statuses=()),
            perception.LaneObservationReason.STATUS_MISSING,
        )

    def test_non_tuple_status_collection_is_unusable(self):
        self.assert_unusable(snapshot(statuses=[healthy_camera_status()]), perception.LaneObservationReason.STATUS_INCOHERENT)

    def test_missing_camera_payload_is_unusable(self):
        self.assert_unusable(snapshot(camera=None), perception.LaneObservationReason.CAMERA_MISSING)

    def test_wrong_pixel_format_is_unusable(self):
        frame = rgb_frame()
        malformed = replace(frame, metadata=replace(frame.metadata, pixel_format="BGR8"))
        self.assert_unusable(snapshot(camera=malformed))

    def test_zero_and_negative_dimensions_are_unusable(self):
        frame = rgb_frame()
        for width, height in ((0, HEIGHT), (-1, HEIGHT), (WIDTH, 0), (WIDTH, -1)):
            malformed = replace(frame, metadata=replace(frame.metadata, width_px=width, height_px=height))
            with self.subTest(width=width, height=height):
                self.assert_unusable(snapshot(camera=malformed))

    def test_malformed_stride_is_unusable(self):
        frame = rgb_frame()
        malformed = replace(frame, metadata=replace(frame.metadata, row_stride_bytes=WIDTH * 3 + 1))
        self.assert_unusable(snapshot(camera=malformed))

    def test_short_and_long_payloads_are_unusable(self):
        frame = rgb_frame()
        for payload in (frame.rgb_bytes[:-1], frame.rgb_bytes + b"\0"):
            with self.subTest(length=len(payload)):
                self.assert_unusable(snapshot(camera=replace(frame, rgb_bytes=payload)))

    def test_non_bytes_payload_is_unusable(self):
        frame = rgb_frame()
        self.assert_unusable(snapshot(camera=replace(frame, rgb_bytes=memoryview(frame.rgb_bytes))))

    def test_nonfinite_source_timestamps_are_unusable(self):
        frame = rgb_frame()
        for field_name in ("sim_time", "monotonic"):
            for bad in (float("nan"), float("inf"), float("-inf")):
                stamp = replace(frame.metadata.stamp, **{field_name: bad})
                malformed = replace(frame, metadata=replace(frame.metadata, stamp=stamp))
                with self.subTest(field=field_name, bad=bad):
                    self.assert_unusable(snapshot(camera=malformed))

    def test_invalid_source_frame_is_unusable(self):
        frame = rgb_frame()
        for bad in (-1, True, 1.0):
            stamp = replace(frame.metadata.stamp, frame=bad)
            malformed = replace(frame, metadata=replace(frame.metadata, stamp=stamp))
            with self.subTest(bad=bad):
                self.assert_unusable(snapshot(camera=malformed))

    def test_image_over_configured_bound_is_unusable(self):
        frame = rgb_frame()
        metadata = replace(
            frame.metadata,
            width_px=1001,
            height_px=1000,
            row_stride_bytes=3003,
        )
        observation = self.assert_unusable(
            snapshot(camera=replace(frame, metadata=metadata, rgb_bytes=b""))
        )
        self.assertIs(observation.reason, perception.LaneObservationReason.IMAGE_TOO_LARGE)

    def test_gate_failures_do_not_enter_pixel_processing(self):
        stale = snapshot(statuses=(healthy_camera_status(freshness=SensorFreshness.STALE, age_seconds=1.0),))
        with mock.patch("perception.lane._candidate_representatives", side_effect=AssertionError("pixels touched")):
            self.assert_unusable(stale)

    def test_valid_fresh_healthy_blank_input_reaches_bounded_result(self):
        observation = perception.perceive_lanes(snapshot())
        self.assertIs(observation.state, perception.LaneDetectionState.NOT_DETECTED)


class BasicDetectionTest(unittest.TestCase):
    def assert_detected(self, frame):
        observation = perception.perceive_lanes(snapshot(camera=frame))
        self.assertIs(observation.state, perception.LaneDetectionState.DETECTED)
        self.assertIsNotNone(observation.left_boundary)
        self.assertIsNotNone(observation.right_boundary)
        self.assertIsNotNone(observation.centerline_endpoints)
        self.assertTrue(math.isfinite(observation.confidence))
        return observation

    def test_symmetric_white_boundaries_are_detected(self):
        observation = self.assert_detected(detected_frame())
        self.assertAlmostEqual(observation.center_offset, 0.0, delta=0.015)

    def test_symmetric_yellow_boundaries_are_detected(self):
        self.assert_detected(detected_frame(left_color=YELLOW, right_color=YELLOW))

    def test_mixed_white_and_yellow_boundaries_are_detected(self):
        frame = detected_frame(left_color=YELLOW, right_color=WHITE)
        observation = self.assert_detected(frame)
        self.assertEqual(
            perception.perceive_lanes(snapshot(camera=frame)),
            observation,
        )

    def test_centered_lane_has_near_zero_offset_and_heading(self):
        observation = self.assert_detected(detected_frame())
        self.assertAlmostEqual(observation.center_offset, 0.0, delta=0.015)
        self.assertAlmostEqual(observation.heading_proxy, 0.0, delta=0.02)

    def test_lane_shifted_left_has_negative_offset(self):
        observation = self.assert_detected(
            detected_frame(left=(0.52, -0.40), right=(0.32, 0.40))
        )
        self.assertLess(observation.center_offset, 0.0)

    def test_lane_shifted_right_has_positive_offset(self):
        observation = self.assert_detected(
            detected_frame(left=(0.68, -0.40), right=(0.48, 0.40))
        )
        self.assertGreater(observation.center_offset, 0.0)

    def test_positive_heading_proxy_means_center_moves_right_toward_bottom(self):
        heading = 0.12
        left = (BASE_LEFT[0] - 0.75 * heading, BASE_LEFT[1] + heading)
        right = (BASE_RIGHT[0] - 0.75 * heading, BASE_RIGHT[1] + heading)
        observation = self.assert_detected(detected_frame(left=left, right=right))
        self.assertGreater(observation.heading_proxy, 0.0)

    def test_negative_heading_proxy_means_center_moves_left_toward_bottom(self):
        heading = -0.12
        left = (BASE_LEFT[0] - 0.75 * heading, BASE_LEFT[1] + heading)
        right = (BASE_RIGHT[0] - 0.75 * heading, BASE_RIGHT[1] + heading)
        observation = self.assert_detected(detected_frame(left=left, right=right))
        self.assertLess(observation.heading_proxy, 0.0)


class AdversarialRobustnessTest(unittest.TestCase):
    SUPPORT_ROWS = (0.52, 0.625, 0.75, 0.875, 0.90, 0.98)

    def observe(self, markings, *, width=WIDTH, height=HEIGHT):
        return perception.perceive_lanes(
            snapshot(
                camera=scene_frame(
                    markings,
                    width=width,
                    height=height,
                )
            )
        )

    def assert_supported_pair_or_honest_ambiguity(
        self,
        observation,
        expected_evaluation_positions,
        *,
        clean_offset=0.0,
    ):
        if observation.state is perception.LaneDetectionState.DETECTED:
            self.assertAlmostEqual(
                observation.center_offset,
                clean_offset,
                delta=0.03,
            )
        else:
            self.assertIn(
                observation.state,
                (
                    perception.LaneDetectionState.PARTIAL,
                    perception.LaneDetectionState.NOT_DETECTED,
                ),
            )
            self.assertLess(
                observation.confidence,
                perception.DEFAULT_LANE_CONFIG.detection_confidence,
            )
        for boundary_value in (
            observation.left_boundary,
            observation.right_boundary,
        ):
            if boundary_value is None:
                continue
            evaluation_x = _boundary_x(
                boundary_value,
                perception.DEFAULT_LANE_CONFIG.center_evaluation_y,
            )
            self.assertLessEqual(
                min(
                    abs(evaluation_x - expected)
                    for expected in expected_evaluation_positions
                ),
                0.025,
            )

    def assert_boundary_has_continuous_observed_support(
        self,
        frame,
        boundary_value,
        markings,
        pixel_width,
    ):
        distances = observed_support_distances(
            frame,
            boundary_value,
            self.SUPPORT_ROWS,
        )
        raster_support_tolerance = (
            (pixel_width * 0.5 + 2.0)
            / (frame.metadata.width_px - 1)
        )
        self.assertTrue(
            all(distance <= raster_support_tolerance for distance in distances),
            distances,
        )
        sources = nearest_marking_sources(
            boundary_value,
            markings,
            self.SUPPORT_ROWS,
        )
        self.assertEqual(len(set(sources)), 1, sources)
        source = markings[sources[0]]
        for normalized_y in self.SUPPORT_ROWS:
            self.assertAlmostEqual(
                _boundary_x(boundary_value, normalized_y),
                source[0] + source[1] * normalized_y,
                delta=0.006,
            )

    def test_exact_four_marking_scene_has_no_cross_cluster_stitch(self):
        markings = (
            (0.60, -0.40, WHITE, 5),
            (0.65, -0.40, YELLOW, 5),
            (0.40, 0.40, WHITE, 5),
            (0.45, 0.40, YELLOW, 5),
        )
        frame = scene_frame(markings, width=640, height=360)
        observation = perception.perceive_lanes(snapshot(camera=frame))
        clean = perception.perceive_lanes(
            snapshot(
                camera=scene_frame(
                    (markings[0], markings[2]),
                    width=640,
                    height=360,
                )
            )
        )
        self.assertIn(
            observation.state,
            (
                perception.LaneDetectionState.DETECTED,
                perception.LaneDetectionState.PARTIAL,
            ),
        )
        self.assertLessEqual(observation.confidence, clean.confidence)
        for boundary_value in (
            observation.left_boundary,
            observation.right_boundary,
        ):
            self.assertIsNotNone(boundary_value)
            self.assertAlmostEqual(abs(boundary_value.x_slope), 0.40, delta=0.01)
            self.assert_boundary_has_continuous_observed_support(
                frame,
                boundary_value,
                markings,
                5,
            )

        representatives = _candidate_representatives(
            frame,
            perception.DEFAULT_LANE_CONFIG,
        )
        hypotheses = _fit_boundary_hypotheses(
            representatives,
            perception.DEFAULT_LANE_CONFIG,
        )
        self.assertEqual(len(hypotheses), 4)
        for hypothesis in hypotheses:
            track = hypothesis.track.representatives
            self.assertEqual(len({value.bin_index for value in track}), len(track))
            self.assertEqual(len({value.cluster_index for value in track}), 1)
            self.assertLess(hypothesis.maximum_error, 0.002)

    def test_parallel_marking_separation_and_width_sweep_never_stitches(self):
        for separation in (0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050):
            for pixel_width in (1, 5, 11):
                markings = (
                    (0.60, -0.40, WHITE, pixel_width),
                    (0.60 + separation, -0.40, YELLOW, pixel_width),
                    (0.40, 0.40, WHITE, pixel_width),
                    (0.40 + separation, 0.40, YELLOW, pixel_width),
                )
                frame = scene_frame(markings, width=640, height=360)
                value = snapshot(camera=frame)
                with self.subTest(
                    separation=separation,
                    pixel_width=pixel_width,
                ):
                    observation = perception.perceive_lanes(value)
                    self.assertEqual(perception.perceive_lanes(value), observation)
                    clean = perception.perceive_lanes(
                        snapshot(
                            camera=scene_frame(
                                (markings[0], markings[2]),
                                width=640,
                                height=360,
                            )
                        )
                    )
                    self.assertIn(
                        observation.state,
                        (
                            perception.LaneDetectionState.DETECTED,
                            perception.LaneDetectionState.PARTIAL,
                        ),
                    )
                    self.assertLessEqual(
                        observation.confidence,
                        clean.confidence + 1.0e-12,
                    )
                    for boundary_value in (
                        observation.left_boundary,
                        observation.right_boundary,
                    ):
                        self.assertIsNotNone(boundary_value)
                        self.assertAlmostEqual(
                            abs(boundary_value.x_slope),
                            0.40,
                            delta=0.01,
                        )
                        self.assert_boundary_has_continuous_observed_support(
                            frame,
                            boundary_value,
                            markings,
                            pixel_width,
                        )

    def test_equal_strength_disconnected_halves_do_not_form_boundary(self):
        markings = (
            (0.60, -0.40, WHITE, 5, 0.50, 0.72),
            (0.625, -0.40, YELLOW, 5, 0.78, 1.00),
        )
        value = snapshot(
            camera=scene_frame(markings, width=640, height=360)
        )
        observation = perception.perceive_lanes(value)
        self.assertEqual(perception.perceive_lanes(value), observation)
        self.assertIs(
            observation.state,
            perception.LaneDetectionState.NOT_DETECTED,
        )
        self.assertIsNone(observation.left_boundary)
        self.assertIsNone(observation.right_boundary)

    def test_unequal_strength_disconnected_halves_do_not_form_boundary(self):
        markings = (
            (0.60, -0.40, WHITE, 11, 0.50, 0.72),
            (0.625, -0.40, YELLOW, 1, 0.78, 1.00),
        )
        value = snapshot(
            camera=scene_frame(markings, width=640, height=360)
        )
        observation = perception.perceive_lanes(value)
        self.assertEqual(perception.perceive_lanes(value), observation)
        self.assertIs(
            observation.state,
            perception.LaneDetectionState.NOT_DETECTED,
        )
        self.assertIsNone(observation.left_boundary)
        self.assertIsNone(observation.right_boundary)

    def test_second_marking_widths_do_not_create_midpoint_boundary(self):
        clean_markings = (
            (BASE_LEFT[0], BASE_LEFT[1], WHITE, 5),
            (BASE_RIGHT[0], BASE_RIGHT[1], WHITE, 5),
        )
        clean = self.observe(clean_markings, width=640, height=360)
        self.assertIs(clean.state, perception.LaneDetectionState.DETECTED)
        expected_positions = (0.24, 0.61, 0.76)
        for interferer_width in (1, 5, 11):
            markings = clean_markings + ((0.25, 0.40, WHITE, interferer_width),)
            with self.subTest(interferer_width=interferer_width):
                observation = self.observe(markings, width=640, height=360)
                self.assert_supported_pair_or_honest_ambiguity(
                    observation,
                    expected_positions,
                )
                self.assertLessEqual(observation.confidence, clean.confidence)

    def test_adjacent_lane_marking_does_not_form_phantom_ego_boundary(self):
        observation = self.observe(
            (
                (BASE_LEFT[0], BASE_LEFT[1], WHITE, 5),
                (BASE_RIGHT[0], BASE_RIGHT[1], WHITE, 5),
                (0.55, 0.40, YELLOW, 5),
            ),
            width=640,
            height=360,
        )
        self.assert_supported_pair_or_honest_ambiguity(
            observation,
            (0.24, 0.76, 0.91),
        )

    def test_curb_like_marking_cannot_report_high_confidence_shift(self):
        observation = self.observe(
            (
                (BASE_LEFT[0], BASE_LEFT[1], WHITE, 5),
                (BASE_RIGHT[0], BASE_RIGHT[1], WHITE, 5),
                (0.58, 0.25, WHITE, 5),
            ),
            width=640,
            height=360,
        )
        self.assert_supported_pair_or_honest_ambiguity(
            observation,
            (0.24, 0.76, 0.805),
        )

    def test_equal_support_competing_lanes_are_deterministically_ambiguous(self):
        markings = (
            (0.48, -0.40, WHITE, 5),
            (0.28, 0.40, WHITE, 5),
            (0.72, -0.40, YELLOW, 5),
            (0.52, 0.40, YELLOW, 5),
        )
        value = snapshot(camera=scene_frame(markings, width=640, height=360))
        first = perception.perceive_lanes(value)
        self.assertIs(first.state, perception.LaneDetectionState.PARTIAL)
        self.assertIs(
            first.reason,
            perception.LaneObservationReason.CONFIDENCE_BELOW_THRESHOLD,
        )
        self.assertLess(
            first.confidence,
            perception.DEFAULT_LANE_CONFIG.detection_confidence,
        )
        for _ in range(5):
            self.assertEqual(perception.perceive_lanes(value), first)

    def test_boundary_crossing_image_center_remains_detectable(self):
        observation = self.observe(
            (
                (0.72, -0.35, WHITE, 5),
                (0.60, 0.20, YELLOW, 5),
            )
        )
        self.assertIs(observation.state, perception.LaneDetectionState.DETECTED)
        self.assertGreater(observation.left_boundary.endpoints[0].x, 0.5)
        self.assertLess(observation.left_boundary.endpoints[1].x, 0.5)

    def test_rigid_lateral_translation_sequence_is_monotonic(self):
        offsets = []
        for shift in (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15):
            observation = self.observe(
                (
                    (BASE_LEFT[0] + shift, BASE_LEFT[1], WHITE, 5),
                    (BASE_RIGHT[0] + shift, BASE_RIGHT[1], YELLOW, 5),
                )
            )
            with self.subTest(shift=shift):
                self.assertIs(
                    observation.state,
                    perception.LaneDetectionState.DETECTED,
                )
                self.assertAlmostEqual(
                    observation.center_offset,
                    shift,
                    delta=0.015,
                )
            offsets.append(observation.center_offset)
        self.assertEqual(offsets, sorted(offsets))
        self.assertLess(offsets[0], 0.0)
        self.assertGreater(offsets[-1], 0.0)

    def test_both_boundaries_can_remain_in_same_image_half(self):
        observation = self.observe(
            (
                (0.29, -0.27, WHITE, 5),
                (0.18, 0.27, YELLOW, 5),
            ),
            width=640,
            height=360,
        )
        self.assertIs(observation.state, perception.LaneDetectionState.DETECTED)
        for y in (
            perception.DEFAULT_LANE_CONFIG.roi[1],
            perception.DEFAULT_LANE_CONFIG.center_evaluation_y,
            perception.DEFAULT_LANE_CONFIG.roi[3],
        ):
            self.assertLess(_boundary_x(observation.left_boundary, y), 0.5)
            self.assertLess(_boundary_x(observation.right_boundary, y), 0.5)

    def test_both_boundaries_can_remain_in_right_image_half(self):
        observation = self.observe(
            (
                (0.82, -0.27, WHITE, 5),
                (0.71, 0.27, YELLOW, 5),
            ),
            width=640,
            height=360,
        )
        self.assertIs(observation.state, perception.LaneDetectionState.DETECTED)
        for y in (
            perception.DEFAULT_LANE_CONFIG.roi[1],
            perception.DEFAULT_LANE_CONFIG.center_evaluation_y,
            perception.DEFAULT_LANE_CONFIG.roi[3],
        ):
            self.assertGreater(_boundary_x(observation.left_boundary, y), 0.5)
            self.assertGreater(_boundary_x(observation.right_boundary, y), 0.5)

    def test_crossing_fitted_boundaries_are_not_detected_end_to_end(self):
        observation = self.observe(
            (
                (0.80, -0.30, WHITE, 5),
                (0.20, 0.30, YELLOW, 5),
            )
        )
        self.assertIsNot(
            observation.state,
            perception.LaneDetectionState.DETECTED,
        )
        assert_no_complete_geometry(self, observation)

    def test_fixed_bright_outliers_remain_bounded(self):
        pixels = tuple(
            (
                (index * 37) % WIDTH,
                HEIGHT // 2 + ((index * 19) % (HEIGHT // 2)),
                WHITE,
            )
            for index in range(120)
        )
        frame = scene_frame(bright_pixels=pixels)
        first = perception.perceive_lanes(snapshot(camera=frame))
        second = perception.perceive_lanes(snapshot(camera=frame))
        self.assertEqual(first, second)
        self.assertIn(
            first.state,
            (
                perception.LaneDetectionState.NOT_DETECTED,
                perception.LaneDetectionState.PARTIAL,
            ),
        )


class ContinuityOcclusionAndToleranceTest(unittest.TestCase):
    def setUp(self):
        self.frame = scene_frame(
            (
                (BASE_LEFT[0], BASE_LEFT[1], WHITE, 5),
                (BASE_RIGHT[0], BASE_RIGHT[1], YELLOW, 5),
            ),
            width=640,
            height=360,
        )

    def test_one_missing_bin_remains_detected(self):
        frame = erase_vertical_bins(self.frame, (8,))
        value = snapshot(camera=frame)
        observation = perception.perceive_lanes(value)
        self.assertEqual(perception.perceive_lanes(value), observation)
        self.assertIs(
            observation.state,
            perception.LaneDetectionState.DETECTED,
        )
        self.assertEqual(observation.left_boundary.support_count, 17)
        self.assertEqual(observation.right_boundary.support_count, 17)

    def test_maximum_missing_bins_remain_detected(self):
        bins = tuple(range(8, 8 + MAX_TRACK_SKIPPED_BINS))
        frame = erase_vertical_bins(self.frame, bins)
        value = snapshot(camera=frame)
        observation = perception.perceive_lanes(value)
        self.assertEqual(perception.perceive_lanes(value), observation)
        self.assertIs(
            observation.state,
            perception.LaneDetectionState.DETECTED,
        )
        expected_support = (
            perception.DEFAULT_LANE_CONFIG.vertical_bin_count
            - MAX_TRACK_SKIPPED_BINS
        )
        self.assertEqual(
            observation.left_boundary.support_count,
            expected_support,
        )
        self.assertEqual(
            observation.right_boundary.support_count,
            expected_support,
        )

    def test_one_more_than_maximum_missing_bins_breaks_tracks(self):
        bins = tuple(range(8, 9 + MAX_TRACK_SKIPPED_BINS))
        frame = erase_vertical_bins(self.frame, bins)
        value = snapshot(camera=frame)
        observation = perception.perceive_lanes(value)
        self.assertEqual(perception.perceive_lanes(value), observation)
        self.assertIs(
            observation.state,
            perception.LaneDetectionState.NOT_DETECTED,
        )
        self.assertIsNone(observation.left_boundary)
        self.assertIsNone(observation.right_boundary)

    def test_one_five_and_eleven_pixel_markings_remain_detectable(self):
        for pixel_width in (1, 5, 11):
            with self.subTest(pixel_width=pixel_width):
                observation = perception.perceive_lanes(
                    snapshot(
                        camera=scene_frame(
                            (
                                (BASE_LEFT[0], BASE_LEFT[1], WHITE, pixel_width),
                                (BASE_RIGHT[0], BASE_RIGHT[1], YELLOW, pixel_width),
                            ),
                            width=640,
                            height=360,
                        )
                    )
                )
                self.assertIs(
                    observation.state,
                    perception.LaneDetectionState.DETECTED,
                )

    def test_steepest_configured_slopes_remain_detectable(self):
        config = replace(
            perception.DEFAULT_LANE_CONFIG,
            min_lane_width=0.13,
            max_lane_width=0.95,
            detection_confidence=0.55,
        )
        observation = perception.perceive_lanes(
            snapshot(
                camera=scene_frame(
                    (
                        (0.81, -config.max_abs_slope, WHITE, 5),
                        (0.15, config.max_abs_slope, YELLOW, 5),
                    ),
                    width=640,
                    height=360,
                )
            ),
            config,
        )
        self.assertIs(
            observation.state,
            perception.LaneDetectionState.DETECTED,
        )
        self.assertAlmostEqual(
            observation.left_boundary.x_slope,
            -config.max_abs_slope,
            delta=0.003,
        )
        self.assertAlmostEqual(
            observation.right_boundary.x_slope,
            config.max_abs_slope,
            delta=0.003,
        )

    def test_near_zero_valid_slopes_remain_detectable(self):
        observation = perception.perceive_lanes(
            snapshot(
                camera=scene_frame(
                    (
                        (0.35, -0.06, WHITE, 5),
                        (0.65, 0.06, YELLOW, 5),
                    ),
                    width=640,
                    height=360,
                )
            )
        )
        self.assertIs(
            observation.state,
            perception.LaneDetectionState.DETECTED,
        )
        self.assertAlmostEqual(
            observation.left_boundary.x_slope,
            -0.06,
            delta=0.003,
        )
        self.assertAlmostEqual(
            observation.right_boundary.x_slope,
            0.06,
            delta=0.003,
        )

    def test_rasterized_diagonals_across_resolutions_remain_detectable(self):
        for width, height, pixel_width in (
            (160, 100, 1),
            (320, 180, 5),
            (640, 360, 11),
        ):
            with self.subTest(
                width=width,
                height=height,
                pixel_width=pixel_width,
            ):
                observation = perception.perceive_lanes(
                    snapshot(
                        camera=scene_frame(
                            (
                                (BASE_LEFT[0], BASE_LEFT[1], WHITE, pixel_width),
                                (BASE_RIGHT[0], BASE_RIGHT[1], YELLOW, pixel_width),
                            ),
                            width=width,
                            height=height,
                        )
                    )
                )
                self.assertIs(
                    observation.state,
                    perception.LaneDetectionState.DETECTED,
                )


class RoiAndResolutionTest(unittest.TestCase):
    def test_nondefault_vertical_roi_end_to_end(self):
        config = replace(
            perception.DEFAULT_LANE_CONFIG,
            roi=(0.0, 0.25, 1.0, 0.85),
            center_evaluation_y=0.75,
        )
        frame = scene_frame(
            (
                (0.45, -0.30, WHITE, 5, 0.25, 0.85),
                (0.55, 0.30, YELLOW, 5, 0.25, 0.85),
            ),
            width=320,
            height=180,
        )
        observation = perception.perceive_lanes(snapshot(camera=frame), config)
        self.assertIs(observation.state, perception.LaneDetectionState.DETECTED)
        self.assertAlmostEqual(observation.center_offset, 0.0, delta=0.015)

    def test_nondefault_horizontal_roi_end_to_end(self):
        config = replace(
            perception.DEFAULT_LANE_CONFIG,
            roi=(0.10, 0.50, 0.90, 1.0),
        )
        observation = perception.perceive_lanes(
            snapshot(camera=detected_frame()),
            config,
        )
        self.assertIs(observation.state, perception.LaneDetectionState.DETECTED)

    def test_pixels_immediately_outside_roi_are_ignored(self):
        config = replace(
            perception.DEFAULT_LANE_CONFIG,
            roi=(0.10, 0.50, 0.90, 1.0),
        )
        width, height = 320, 180
        clean_frame = scene_frame(
            (
                (BASE_LEFT[0], BASE_LEFT[1], WHITE, 5),
                (BASE_RIGHT[0], BASE_RIGHT[1], YELLOW, 5),
            ),
            width=width,
            height=height,
        )
        x0, y0, x1, _ = _roi_pixel_bounds(width, height, config)
        outside = tuple((x0 - 1, y, WHITE) for y in range(y0, height))
        outside += tuple((x1, y, WHITE) for y in range(y0, height))
        outside += tuple((x, y0 - 1, WHITE) for x in range(x0, x1))
        noisy_frame = scene_frame(
            (
                (BASE_LEFT[0], BASE_LEFT[1], WHITE, 5),
                (BASE_RIGHT[0], BASE_RIGHT[1], YELLOW, 5),
            ),
            width=width,
            height=height,
            bright_pixels=outside,
        )
        clean = perception.perceive_lanes(snapshot(camera=clean_frame), config)
        noisy = perception.perceive_lanes(snapshot(camera=noisy_frame), config)
        self.assertEqual(noisy, clean)
        self.assertIs(noisy.state, perception.LaneDetectionState.DETECTED)

    def test_pixels_immediately_inside_roi_are_processed(self):
        config = replace(
            perception.DEFAULT_LANE_CONFIG,
            roi=(0.10, 0.50, 0.90, 1.0),
        )
        width, height = 320, 180
        x0, y0, x1, y1 = _roi_pixel_bounds(width, height, config)
        self.assertEqual((x0, y0, x1, y1), (32, 90, 288, 180))
        frame = scene_frame(
            (
                (0.495, -0.39, WHITE, 1),
                (0.505, 0.39, YELLOW, 1),
            ),
            width=width,
            height=height,
        )
        observation = perception.perceive_lanes(snapshot(camera=frame), config)
        self.assertIs(observation.state, perception.LaneDetectionState.DETECTED)
        self.assertLessEqual(observation.left_boundary.endpoints[1].x, 0.11)
        self.assertGreaterEqual(observation.right_boundary.endpoints[1].x, 0.89)

    def test_different_resolution_is_detected(self):
        frame = scene_frame(
            (
                (BASE_LEFT[0], BASE_LEFT[1], WHITE, 3),
                (BASE_RIGHT[0], BASE_RIGHT[1], YELLOW, 3),
            ),
            width=320,
            height=180,
        )
        observation = perception.perceive_lanes(snapshot(camera=frame))
        self.assertIs(observation.state, perception.LaneDetectionState.DETECTED)

    def test_production_resolution_640_by_360_is_detected(self):
        frame = scene_frame(
            (
                (BASE_LEFT[0], BASE_LEFT[1], WHITE, 5),
                (BASE_RIGHT[0], BASE_RIGHT[1], YELLOW, 5),
            ),
            width=640,
            height=360,
        )
        observation = perception.perceive_lanes(snapshot(camera=frame))
        self.assertIs(observation.state, perception.LaneDetectionState.DETECTED)
        self.assertAlmostEqual(observation.center_offset, 0.0, delta=0.01)


class PartialAndAbsentEvidenceTest(unittest.TestCase):
    def test_blank_image_is_not_detected(self):
        observation = perception.perceive_lanes(snapshot())
        self.assertIs(observation.state, perception.LaneDetectionState.NOT_DETECTED)
        assert_no_complete_geometry(self, observation)

    def test_left_boundary_only_is_partial(self):
        frame = rgb_frame(lines=((BASE_LEFT[0], BASE_LEFT[1], WHITE),))
        observation = perception.perceive_lanes(snapshot(camera=frame))
        self.assertIs(observation.state, perception.LaneDetectionState.PARTIAL)
        self.assertIsNotNone(observation.left_boundary)
        self.assertIsNone(observation.right_boundary)
        assert_no_complete_geometry(self, observation)

    def test_right_boundary_only_is_partial(self):
        frame = rgb_frame(lines=((BASE_RIGHT[0], BASE_RIGHT[1], WHITE),))
        observation = perception.perceive_lanes(snapshot(camera=frame))
        self.assertIs(observation.state, perception.LaneDetectionState.PARTIAL)
        self.assertIsNone(observation.left_boundary)
        self.assertIsNotNone(observation.right_boundary)
        assert_no_complete_geometry(self, observation)

    def test_sparse_candidates_are_not_detected(self):
        config = perception.DEFAULT_LANE_CONFIG
        points = tuple((0.6 - 0.4 * y, y) for y in (0.52, 0.62, 0.72, 0.82))
        self.assertIsNone(_fit_boundary(points, perception.LaneBoundarySide.LEFT, config))

    def test_horizontal_line_is_not_detected(self):
        observation = perception.perceive_lanes(snapshot(camera=rgb_frame(horizontal=True)))
        self.assertIs(observation.state, perception.LaneDetectionState.NOT_DETECTED)

    def test_near_zero_slope_boundaries_are_not_detected(self):
        frame = detected_frame(left=(0.40, -0.01), right=(0.60, 0.01))
        observation = perception.perceive_lanes(snapshot(camera=frame))
        self.assertIsNot(observation.state, perception.LaneDetectionState.DETECTED)

    def test_excessive_candidate_outliers_do_not_escape_bounds(self):
        observation = perception.perceive_lanes(snapshot(camera=rgb_frame(fill_roi=True)))
        self.assertIs(observation.state, perception.LaneDetectionState.NOT_DETECTED)
        self.assertEqual(observation.confidence, 0.0)

    def test_crossing_boundaries_are_never_detected(self):
        config = perception.DEFAULT_LANE_CONFIG
        left = boundary(perception.LaneBoundarySide.LEFT, 0.80, -0.20)
        right = boundary(perception.LaneBoundarySide.RIGHT, 0.20, 0.20)
        observation = _observation_from_boundaries(snapshot(), rgb_frame(), left, right, config)
        self.assertIs(observation.state, perception.LaneDetectionState.PARTIAL)
        self.assertIs(
            observation.reason,
            perception.LaneObservationReason.PAIR_GEOMETRY_INVALID,
        )
        self.assertEqual(observation.confidence, 0.0)
        assert_no_complete_geometry(self, observation)

    def test_implausibly_narrow_pair_is_partial(self):
        left = boundary(perception.LaneBoundarySide.LEFT, 0.49, -0.10)
        right = boundary(perception.LaneBoundarySide.RIGHT, 0.51, 0.10)
        observation = _observation_from_boundaries(
            snapshot(), rgb_frame(), left, right, perception.DEFAULT_LANE_CONFIG
        )
        self.assertIs(observation.state, perception.LaneDetectionState.PARTIAL)
        assert_no_complete_geometry(self, observation)

    def test_implausibly_wide_pair_is_partial(self):
        left = boundary(perception.LaneBoundarySide.LEFT, 0.48, -0.40)
        right = boundary(perception.LaneBoundarySide.RIGHT, 0.52, 0.40)
        observation = _observation_from_boundaries(
            snapshot(), rgb_frame(), left, right, perception.DEFAULT_LANE_CONFIG
        )
        self.assertIs(observation.state, perception.LaneDetectionState.PARTIAL)
        assert_no_complete_geometry(self, observation)

    def test_two_valid_but_weak_boundaries_are_partial(self):
        config = replace(perception.DEFAULT_LANE_CONFIG, detection_confidence=1.0)
        observation = perception.perceive_lanes(snapshot(camera=detected_frame()), config)
        self.assertIs(observation.state, perception.LaneDetectionState.PARTIAL)
        self.assertIs(observation.reason, perception.LaneObservationReason.CONFIDENCE_BELOW_THRESHOLD)
        assert_no_complete_geometry(self, observation)

    def test_deterministic_subthreshold_noise_is_bounded(self):
        frame = rgb_frame()
        payload = bytearray(frame.rgb_bytes)
        for index in range(0, len(payload), 97):
            payload[index] = 180
        observation = perception.perceive_lanes(snapshot(camera=replace(frame, rgb_bytes=bytes(payload))))
        self.assertIs(observation.state, perception.LaneDetectionState.NOT_DETECTED)


class ExactBoundaryTest(unittest.TestCase):
    def test_minimum_support_equality_passes_and_one_less_fails(self):
        config = perception.DEFAULT_LANE_CONFIG
        ys = tuple(0.51 + index * 0.06 for index in range(config.min_support_bins))
        points = tuple((0.60 - 0.40 * y, y) for y in ys)
        self.assertIsNotNone(_fit_boundary(points, perception.LaneBoundarySide.LEFT, config))
        self.assertIsNone(_fit_boundary(points[:-1], perception.LaneBoundarySide.LEFT, config))

    def test_residual_limit_equality_passes_and_next_value_fails(self):
        config = perception.DEFAULT_LANE_CONFIG
        exact = config.max_fit_residual
        self.assertTrue(_residual_is_acceptable(exact, config))
        self.assertFalse(_residual_is_acceptable(math.nextafter(exact, math.inf), config))

    def test_maximum_individual_error_gate_rejects_unsupported_fit(self):
        config = perception.DEFAULT_LANE_CONFIG
        representatives = tuple(
            _CandidateRepresentative(
                bin_index=index,
                cluster_index=0,
                x=0.60 - 0.40 * y,
                y=y,
                pixel_count=5,
                row_count=10,
                bin_row_count=10,
                minimum_x=0.60 - 0.40 * y - 0.003,
                maximum_x=0.60 - 0.40 * y + 0.003,
                raster_tolerance=0.002,
            )
            for index in range(config.min_support_bins)
            for y in (
                config.roi[1]
                + (index + 0.5)
                * (config.roi[3] - config.roi[1])
                / config.vertical_bin_count,
            )
        )
        track = _RepresentativeTrack(representatives)
        valid, maximum_error = _track_support_is_valid(
            track,
            0.60,
            -0.40,
            config,
        )
        self.assertTrue(valid)
        self.assertLess(maximum_error, 1.0e-12)
        tolerance = _maximum_individual_error_tolerance(track, config)
        valid, maximum_error = _track_support_is_valid(
            track,
            0.60 + tolerance * 1.1,
            -0.40,
            config,
        )
        self.assertFalse(valid)
        self.assertGreater(maximum_error, tolerance)

    def test_minimum_slope_equality_passes_and_next_lower_fails(self):
        config = perception.DEFAULT_LANE_CONFIG
        exact = -config.min_abs_slope
        self.assertTrue(_line_is_oriented(exact, perception.LaneBoundarySide.LEFT, config))
        self.assertFalse(_line_is_oriented(math.nextafter(exact, 0.0), perception.LaneBoundarySide.LEFT, config))

    def test_maximum_slope_equality_passes_and_next_higher_fails(self):
        config = perception.DEFAULT_LANE_CONFIG
        exact = config.max_abs_slope
        self.assertTrue(_line_is_oriented(exact, perception.LaneBoundarySide.RIGHT, config))
        self.assertFalse(_line_is_oriented(math.nextafter(exact, math.inf), perception.LaneBoundarySide.RIGHT, config))

    def test_lane_width_equal_limits_passes_and_one_value_outside_fails(self):
        base = perception.DEFAULT_LANE_CONFIG
        left = boundary(perception.LaneBoundarySide.LEFT, *BASE_LEFT)
        right = boundary(perception.LaneBoundarySide.RIGHT, *BASE_RIGHT)
        ys = (base.roi[1], (base.roi[1] + base.roi[3]) * 0.5, base.center_evaluation_y, base.roi[3])
        widths = tuple(_boundary_x(right, y) - _boundary_x(left, y) for y in ys)
        equality = replace(base, min_lane_width=min(widths), max_lane_width=max(widths))
        self.assertTrue(_pair_geometry(left, right, equality)[0])
        too_narrow = replace(equality, min_lane_width=math.nextafter(min(widths), math.inf))
        too_wide = replace(equality, max_lane_width=math.nextafter(max(widths), 0.0))
        self.assertFalse(_pair_geometry(left, right, too_narrow)[0])
        self.assertFalse(_pair_geometry(left, right, too_wide)[0])

    def test_confidence_threshold_equality_detects(self):
        threshold = 0.65
        config = replace(perception.DEFAULT_LANE_CONFIG, detection_confidence=threshold)
        left = boundary(perception.LaneBoundarySide.LEFT, *BASE_LEFT, confidence=threshold, config=config)
        right = boundary(perception.LaneBoundarySide.RIGHT, *BASE_RIGHT, confidence=threshold, config=config)
        observation = _observation_from_boundaries(snapshot(), rgb_frame(), left, right, config)
        self.assertEqual(observation.confidence, threshold)
        self.assertIs(observation.state, perception.LaneDetectionState.DETECTED)

    def test_one_representable_confidence_below_threshold_is_partial(self):
        threshold = 0.65
        below = math.nextafter(threshold, 0.0)
        config = replace(perception.DEFAULT_LANE_CONFIG, detection_confidence=threshold)
        left = boundary(perception.LaneBoundarySide.LEFT, *BASE_LEFT, confidence=below, config=config)
        right = boundary(perception.LaneBoundarySide.RIGHT, *BASE_RIGHT, confidence=below, config=config)
        observation = _observation_from_boundaries(snapshot(), rgb_frame(), left, right, config)
        self.assertEqual(observation.confidence, below)
        self.assertIs(observation.state, perception.LaneDetectionState.PARTIAL)

    def test_exact_normalized_roi_limits_are_accepted(self):
        config = replace(
            perception.DEFAULT_LANE_CONFIG,
            roi=(0.0, 0.0, 1.0, 1.0),
            center_evaluation_y=1.0,
            min_vertical_span=1.0,
        )
        self.assertEqual(config.roi, (0.0, 0.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            replace(config, roi=(math.nextafter(0.0, -math.inf), 0.0, 1.0, 1.0))

    def test_geometry_unit_bounds_are_inclusive(self):
        point = perception.NormalizedImagePoint(0.0, 1.0)
        self.assertEqual((point.x, point.y), (0.0, 1.0))
        with self.assertRaises(ValueError):
            perception.NormalizedImagePoint(math.nextafter(0.0, -math.inf), 1.0)


class DeterminismTest(unittest.TestCase):
    def test_repeated_identical_calls_return_equal_observations(self):
        value = snapshot(camera=detected_frame())
        first = perception.perceive_lanes(value)
        for _ in range(20):
            self.assertEqual(perception.perceive_lanes(value), first)

    def test_reversed_internal_candidate_order_does_not_change_fit(self):
        config = perception.DEFAULT_LANE_CONFIG
        points = tuple((0.60 - 0.40 * y, y) for y in (0.51, 0.56, 0.62, 0.68, 0.74, 0.80, 0.87, 0.94, 0.99))
        forward = _fit_boundary(points, perception.LaneBoundarySide.LEFT, config)
        reverse = _fit_boundary(tuple(reversed(points)), perception.LaneBoundarySide.LEFT, config)
        self.assertEqual(forward, reverse)

    def test_reversed_and_shuffled_representatives_preserve_end_to_end_result(self):
        frame = scene_frame(
            (
                (BASE_LEFT[0], BASE_LEFT[1], WHITE, 5),
                (BASE_RIGHT[0], BASE_RIGHT[1], YELLOW, 5),
                (0.25, 0.40, WHITE, 1),
            ),
            width=320,
            height=180,
        )
        value = snapshot(camera=frame)
        expected = perception.perceive_lanes(value)
        representatives = _candidate_representatives(
            frame,
            perception.DEFAULT_LANE_CONFIG,
        )
        orders = (
            tuple(reversed(representatives)),
            representatives[::2] + representatives[1::2],
        )
        for order in orders:
            with self.subTest(order_length=len(order)), mock.patch(
                "perception.lane._candidate_representatives",
                return_value=order,
            ):
                self.assertEqual(perception.perceive_lanes(value), expected)

    def test_candidate_and_hypothesis_bounds_are_explicitly_enforced(self):
        markings = tuple(
            (0.05 + index * 0.075, (-0.20 if index % 2 else 0.20), WHITE, 1)
            for index in range(12)
        )
        frame = scene_frame(markings, width=640, height=360)
        representatives = _candidate_representatives(
            frame,
            perception.DEFAULT_LANE_CONFIG,
        )
        counts_by_bin = {}
        for representative in representatives:
            counts_by_bin[representative.bin_index] = (
                counts_by_bin.get(representative.bin_index, 0) + 1
            )
            x = int(round(representative.x * (frame.metadata.width_px - 1)))
            y = int(round(representative.y * (frame.metadata.height_px - 1)))
            offset = (y * frame.metadata.width_px + x) * 3
            self.assertNotEqual(frame.rgb_bytes[offset : offset + 3], b"\0\0\0")
        self.assertTrue(counts_by_bin)
        self.assertLessEqual(max(counts_by_bin.values()), MAX_REPRESENTATIVES_PER_BIN)
        links = _bounded_outgoing_links(
            representatives,
            perception.DEFAULT_LANE_CONFIG,
        )
        self.assertTrue(links)
        self.assertLessEqual(
            max(len(values) for values in links.values()),
            MAX_OUTGOING_LINKS_PER_REPRESENTATIVE,
        )
        tracks = _continuous_representative_tracks(
            representatives,
            perception.DEFAULT_LANE_CONFIG,
        )
        self.assertLessEqual(len(tracks), MAX_COMPLETED_TRACKS)
        for track in tracks:
            bins = tuple(
                representative.bin_index
                for representative in track.representatives
            )
            self.assertEqual(len(bins), len(set(bins)))
            self.assertTrue(
                all(
                    second - first - 1 <= MAX_TRACK_SKIPPED_BINS
                    for first, second in zip(bins, bins[1:])
                )
            )
        hypotheses = _fit_boundary_hypotheses(
            tuple(reversed(representatives)),
            perception.DEFAULT_LANE_CONFIG,
        )
        self.assertLessEqual(len(hypotheses), MAX_BOUNDARY_HYPOTHESES)
        self.assertEqual(MAX_BOUNDARY_SEEDS, 512)
        self.assertEqual(MAX_PAIR_COMBINATIONS, 120)
        self.assertEqual(MAX_TRACK_SKIPPED_BINS, 2)
        self.assertEqual(MAX_TRACK_LINKS_PER_BIN_GAP, 4)
        self.assertEqual(MAX_OUTGOING_LINKS_PER_REPRESENTATIVE, 12)
        self.assertEqual(MAX_PARTIAL_TRACKS_PER_REPRESENTATIVE, 4)
        self.assertEqual(MAX_PARTIAL_TRACKS_PER_BIN, 32)
        self.assertEqual(MAX_COMPLETED_TRACKS, 64)
        left, right, ambiguous = _select_boundaries(
            hypotheses,
            perception.DEFAULT_LANE_CONFIG,
        )
        self.assertIn(ambiguous, (True, False))
        self.assertTrue(left is None or isinstance(left, perception.LaneBoundaryObservation))
        self.assertTrue(right is None or isinstance(right, perception.LaneBoundaryObservation))

    def test_named_pair_combination_cap_is_enforced_during_enumeration(self):
        frame = scene_frame(
            (
                (0.60, -0.40, WHITE, 5),
                (0.65, -0.40, YELLOW, 5),
                (0.40, 0.40, WHITE, 5),
                (0.45, 0.40, YELLOW, 5),
            ),
            width=640,
            height=360,
        )
        hypotheses = _fit_boundary_hypotheses(
            _candidate_representatives(
                frame,
                perception.DEFAULT_LANE_CONFIG,
            ),
            perception.DEFAULT_LANE_CONFIG,
        )
        self.assertGreaterEqual(len(hypotheses), 3)
        with mock.patch(
            "perception.lane.MAX_PAIR_COMBINATIONS",
            1,
        ), mock.patch(
            "perception.lane._pair_geometry",
            wraps=_pair_geometry,
        ) as geometry:
            _select_boundaries(
                hypotheses,
                perception.DEFAULT_LANE_CONFIG,
            )
        self.assertEqual(geometry.call_count, 1)

    def test_competing_representatives_in_one_bin_do_not_inflate_support(self):
        config = perception.DEFAULT_LANE_CONFIG
        ys = (0.51, 0.57, 0.63, 0.69, 0.75, 0.81, 0.87)
        points = tuple((0.60 - 0.40 * y, y) for y in ys)
        points += tuple((0.60 - 0.40 * 0.88 + index * 0.001, 0.88) for index in range(12))
        self.assertEqual(_fit_boundary_hypotheses(points, config), ())

    def test_hypothesis_tie_breaking_is_stable(self):
        config = perception.DEFAULT_LANE_CONFIG
        points = tuple((0.60 - 0.40 * y, y) for y in (0.51, 0.56, 0.62, 0.68, 0.74, 0.80, 0.87, 0.94, 0.99))
        results = {_fit_boundary(points, perception.LaneBoundarySide.LEFT, config) for _ in range(10)}
        self.assertEqual(len(results), 1)

    def test_module_global_state_does_not_grow(self):
        import perception.lane as lane_module

        before = tuple(sorted(lane_module.__dict__))
        value = snapshot(camera=detected_frame())
        for _ in range(20):
            perception.perceive_lanes(value)
        self.assertEqual(tuple(sorted(lane_module.__dict__)), before)

    def test_input_snapshot_frame_and_bytes_remain_unchanged(self):
        frame = detected_frame()
        value = snapshot(camera=frame)
        original_snapshot = value
        original_frame = frame
        original_bytes = frame.rgb_bytes
        perception.perceive_lanes(value)
        self.assertEqual(value, original_snapshot)
        self.assertEqual(frame, original_frame)
        self.assertIs(frame.rgb_bytes, original_bytes)

    def test_source_evidence_is_copied_exactly_without_clock_calls(self):
        frame = detected_frame()
        value = snapshot(camera=frame, revision=123)
        with mock.patch("time.monotonic", side_effect=AssertionError("clock called")):
            observation = perception.perceive_lanes(value)
        self.assertEqual(observation.snapshot_revision, 123)
        self.assertEqual(observation.source_frame, frame.metadata.stamp.frame)
        self.assertEqual(observation.source_simulation_timestamp, frame.metadata.stamp.sim_time)
        self.assertEqual(observation.source_receive_monotonic, frame.metadata.stamp.monotonic)


class ImmutabilityAndRetentionTest(unittest.TestCase):
    def setUp(self):
        self.observation = perception.perceive_lanes(snapshot(camera=detected_frame()))
        self.assertIs(self.observation.state, perception.LaneDetectionState.DETECTED)

    def test_public_records_are_frozen_and_slotted(self):
        records = (
            perception.DEFAULT_LANE_CONFIG,
            self.observation,
            self.observation.left_boundary,
            self.observation.centerline_endpoints[0],
        )
        for record in records:
            with self.subTest(record=type(record).__name__):
                self.assertFalse(hasattr(record, "__dict__"))
                with self.assertRaises((FrozenInstanceError, AttributeError, TypeError)):
                    record.extra = 1

    def test_output_collections_are_tuples_only(self):
        self.assertIs(type(self.observation.centerline_endpoints), tuple)
        self.assertIs(type(self.observation.left_boundary.endpoints), tuple)
        self.assertIs(type(self.observation.right_boundary.endpoints), tuple)

    def test_output_scalars_are_builtin_python_values(self):
        for value in (
            self.observation.snapshot_revision,
            self.observation.source_frame,
            self.observation.confidence,
            self.observation.center_offset,
            self.observation.heading_proxy,
            self.observation.left_boundary.x_intercept,
            self.observation.left_boundary.x_slope,
            self.observation.left_boundary.fit_residual,
        ):
            self.assertIn(type(value), (int, float))
            self.assertNotIsInstance(value, numpy.generic)

    def test_recursive_output_graph_has_no_retained_or_mutable_objects(self):
        forbidden = (
            TelemetrySnapshot,
            CameraFrame,
            bytes,
            bytearray,
            memoryview,
            numpy.ndarray,
            numpy.generic,
            dict,
            list,
            set,
            BaseException,
        )
        seen = set()

        def walk(value):
            marker = id(value)
            if marker in seen:
                return
            seen.add(marker)
            self.assertNotIsInstance(value, forbidden)
            self.assertFalse(callable(value))
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                for field in fields(value):
                    walk(getattr(value, field.name))
            elif type(value) is tuple:
                for item in value:
                    walk(item)
            else:
                self.assertTrue(
                    value is None
                    or isinstance(value, enum.Enum)
                    or type(value) in (int, float, str)
                )

        walk(self.observation)

    def test_every_public_geometric_value_is_finite_and_bounded(self):
        values = [
            self.observation.confidence,
            self.observation.center_offset,
            self.observation.heading_proxy,
        ]
        for boundary_value in (self.observation.left_boundary, self.observation.right_boundary):
            values.extend((boundary_value.x_intercept, boundary_value.x_slope, boundary_value.fit_residual, boundary_value.confidence))
            for point in boundary_value.endpoints:
                self.assertTrue(0.0 <= point.x <= 1.0)
                self.assertTrue(0.0 <= point.y <= 1.0)
        for value in values:
            self.assertTrue(math.isfinite(value))
        self.assertTrue(0.0 <= self.observation.confidence <= 1.0)


class ImportSafetyAndPreservationTest(unittest.TestCase):
    def test_canonical_camera_name_matches_sensor_source_without_importing_it(self):
        repository = pathlib.Path(__file__).resolve().parents[1]

        def assigned_string(path, name):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id == name:
                    self.assertIsInstance(node.value, ast.Constant)
                    self.assertIs(type(node.value.value), str)
                    return node.value.value
            self.fail(f"{name} not found in {path}")

        perception_name = assigned_string(
            repository / "perception" / "lane.py",
            "CAMERA_SENSOR_NAME",
        )
        producer_name = assigned_string(
            repository / "sensors" / "rgb_camera.py",
            "RGB_CAMERA_NAME",
        )
        self.assertEqual(perception_name, "rgb_camera")
        self.assertEqual(producer_name, "rgb_camera")
        self.assertEqual(perception_name, producer_name)

    def test_fresh_package_import_has_no_dependency_or_side_effects(self):
        result = run_isolated_python(import_safety_probe_code("import perception"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "IMPORT_SAFE")

    def test_fresh_direct_module_import_has_no_dependency_or_side_effects(self):
        result = run_isolated_python(
            import_safety_probe_code("import perception.lane")
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "IMPORT_SAFE")

    def test_camera_gate_failures_do_not_import_numpy(self):
        code = textwrap.dedent(
            """
            import sys
            from dataclasses import replace

            assert "numpy" not in sys.modules
            import perception
            import perception.lane as lane_module
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
            assert "numpy" not in sys.modules

            width = 160
            height = 100
            status = SensorStatus(
                name="rgb_camera",
                kind=SensorKind.CONTINUOUS,
                lifecycle=SensorLifecycle.ATTACHED,
                readiness=SensorReadiness.READY,
                freshness=SensorFreshness.FRESH,
                operational_health=OperationalHealth.HEALTHY,
                evaluated_monotonic=100.1,
                age_seconds=0.1,
                expected_interval=0.1,
                stale_after=0.5,
                accepted_count=1,
                event_count=None,
                error_count=0,
                reason=None,
            )
            metadata = CameraFrameMetadata(
                stamp=SampleStamp(frame=42, sim_time=12.5, monotonic=100.0),
                width_px=width,
                height_px=height,
                horizontal_fov_deg=90.0,
                pixel_format=PIXEL_FORMAT_RGB8,
                row_stride_bytes=width * 3,
                transform=None,
            )
            frame = CameraFrame(
                metadata=metadata,
                rgb_bytes=bytes(width * height * 3),
            )

            def make_snapshot(camera, statuses):
                return TelemetrySnapshot(
                    revision=7,
                    captured_monotonic=100.1,
                    camera=camera,
                    statuses=statuses,
                )

            stale_status = replace(
                status,
                freshness=SensorFreshness.STALE,
                age_seconds=1.0,
            )
            degraded_status = replace(
                status,
                operational_health=OperationalHealth.DEGRADED,
            )
            malformed_stride = replace(
                frame,
                metadata=replace(metadata, row_stride_bytes=width * 3 + 1),
            )
            malformed_payload = replace(frame, rgb_bytes=frame.rgb_bytes[:-1])
            oversized = replace(
                frame,
                metadata=replace(
                    metadata,
                    width_px=1001,
                    height_px=1000,
                    row_stride_bytes=3003,
                ),
                rgb_bytes=b"",
            )
            cases = (
                ("stale", make_snapshot(frame, (stale_status,))),
                ("degraded", make_snapshot(frame, (degraded_status,))),
                ("duplicate", make_snapshot(frame, (status, status))),
                ("malformed_stride", make_snapshot(malformed_stride, (status,))),
                ("malformed_payload", make_snapshot(malformed_payload, (status,))),
                ("oversized", make_snapshot(oversized, (status,))),
            )

            def pixels_touched(*args, **kwargs):
                raise AssertionError("pixel processing entered")

            lane_module._candidate_representatives = pixels_touched
            for name, value in cases:
                result = perception.perceive_lanes(value)
                assert result.state is perception.LaneDetectionState.INPUT_UNUSABLE, name
                assert "numpy" not in sys.modules, name
            print("FAIL_CLOSED_SAFE")
            """
        )
        result = run_isolated_python(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "FAIL_CLOSED_SAFE")

    def test_valid_input_imports_numpy_only_after_gate_and_is_deterministic(self):
        code = textwrap.dedent(
            """
            import sys

            assert "numpy" not in sys.modules
            import perception
            import perception.lane as lane_module
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
            assert "numpy" not in sys.modules

            width = 160
            height = 100
            payload = bytearray(width * height * 3)
            for intercept, slope in ((0.60, -0.40), (0.40, 0.40)):
                for y in range(height // 2, height):
                    normalized_y = y / float(height - 1)
                    center_x = int(
                        round((intercept + slope * normalized_y) * (width - 1))
                    )
                    for x in range(center_x - 2, center_x + 3):
                        if 0 <= x < width:
                            offset = (y * width + x) * 3
                            payload[offset:offset + 3] = bytes((245, 245, 245))
            frame = CameraFrame(
                metadata=CameraFrameMetadata(
                    stamp=SampleStamp(frame=42, sim_time=12.5, monotonic=100.0),
                    width_px=width,
                    height_px=height,
                    horizontal_fov_deg=90.0,
                    pixel_format=PIXEL_FORMAT_RGB8,
                    row_stride_bytes=width * 3,
                    transform=None,
                ),
                rgb_bytes=bytes(payload),
            )
            status = SensorStatus(
                name="rgb_camera",
                kind=SensorKind.CONTINUOUS,
                lifecycle=SensorLifecycle.ATTACHED,
                readiness=SensorReadiness.READY,
                freshness=SensorFreshness.FRESH,
                operational_health=OperationalHealth.HEALTHY,
                evaluated_monotonic=100.1,
                age_seconds=0.1,
                expected_interval=0.1,
                stale_after=0.5,
                accepted_count=1,
                event_count=None,
                error_count=0,
                reason=None,
            )
            snapshot = TelemetrySnapshot(
                revision=7,
                captured_monotonic=100.1,
                camera=frame,
                statuses=(status,),
            )

            events = []
            original_gate = lane_module._camera_gate
            original_candidates = lane_module._candidate_representatives

            def checked_gate(value, config):
                assert "numpy" not in sys.modules
                gated_frame, reason = original_gate(value, config)
                assert gated_frame is frame
                assert reason is None
                assert "numpy" not in sys.modules
                events.append("gate_passed")
                return gated_frame, reason

            def checked_candidates(value, config):
                assert events == ["gate_passed"]
                assert "numpy" not in sys.modules
                candidates = original_candidates(value, config)
                assert "numpy" in sys.modules
                events.append("pixels_processed")
                return candidates

            lane_module._camera_gate = checked_gate
            lane_module._candidate_representatives = checked_candidates
            first = perception.perceive_lanes(snapshot)
            assert events == ["gate_passed", "pixels_processed"]
            assert "numpy" in sys.modules
            assert first.state is perception.LaneDetectionState.DETECTED
            assert first.reason is perception.LaneObservationReason.DETECTED
            assert first.left_boundary is not None
            assert first.right_boundary is not None
            assert abs(first.center_offset) <= 0.015
            assert abs(first.heading_proxy) <= 0.02
            assert first.source_frame == 42
            assert first.source_simulation_timestamp == 12.5
            assert first.source_receive_monotonic == 100.0

            lane_module._camera_gate = original_gate
            lane_module._candidate_representatives = original_candidates
            second = perception.perceive_lanes(snapshot)
            assert second == first
            print("VALID_INPUT_SAFE")
            """
        )
        result = run_isolated_python(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "VALID_INPUT_SAFE")

    def test_fresh_import_does_not_load_forbidden_runtime_modules(self):
        code = (
            "import sys; import perception; "
            "bad=[name for name in ('carla','pygame','manual_drive','socket') if name in sys.modules]; "
            "assert not bad, bad; print('IMPORT_SAFE')"
        )
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "IMPORT_SAFE")

    def test_production_module_names_exclude_forbidden_dependencies(self):
        import perception.lane as lane_module

        names = set(lane_module.__dict__)
        for forbidden in (
            "numpy",
            "carla",
            "pygame",
            "manual_drive",
            "socket",
            "cv2",
            "scipy",
        ):
            self.assertNotIn(forbidden, names)

    def test_existing_telemetry_record_shapes_are_unchanged(self):
        self.assertEqual(
            tuple(field.name for field in fields(CameraFrame)),
            ("metadata", "rgb_bytes"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(TelemetrySnapshot)),
            (
                "revision",
                "captured_monotonic",
                "ego",
                "gnss",
                "imu",
                "sensors",
                "collisions",
                "lane_invasions",
                "camera",
                "collision_sequence",
                "lane_invasion_sequence",
                "statuses",
            ),
        )

    def test_no_control_object_or_actor_is_needed(self):
        value = snapshot(camera=detected_frame())
        observation = perception.perceive_lanes(value)
        self.assertIs(observation.state, perception.LaneDetectionState.DETECTED)
        self.assertFalse(any("control" in type(item).__name__.lower() for item in dataclasses.astuple(observation)))

    def test_no_live_environment_variable_is_required(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            observation = perception.perceive_lanes(snapshot(camera=detected_frame()))
        self.assertIs(observation.state, perception.LaneDetectionState.DETECTED)


# Strict coefficient-agreement bound between the incremental scalar fit and
# the authoritative ``_closed_form_fit``.  This is sufficient for prediction
# gates on raster-derived representatives, but it is not a ranking-equivalence
# claim: an RMS residual is also an unthresholded lexicographic key element, so
# hard-cap selection restores that element from the authoritative fit exactly.
INCREMENTAL_FIT_TOLERANCE = 1.0e-11
# The test-local accepted helpers below are transcribed from this immutable
# Mission 5 source.  These literals deliberately avoid a Git dependency in the
# ordinary offline suite while making source drift reviewable.
ACCEPTED_MISSION_5_COMMIT = "ede9826bb1842244010229c2dad018ab8877a14f"
ACCEPTED_MISSION_5_LANE_BLOB = "867be9d112bc6d8ce8857b9f832202d9a11fcab7"

DENSE_FOUR_MARKINGS = (
    (0.60, -0.40, WHITE, 5),
    (0.65, -0.40, YELLOW, 5),
    (0.40, 0.40, WHITE, 5),
    (0.45, 0.40, YELLOW, 5),
)
DENSE_THIRTY_MARKINGS = tuple(
    (0.030 + index * 0.031, (-0.30 if index % 2 else 0.30), WHITE, 5)
    for index in range(30)
)
DENSE_SIXTY_TWO_THIN_MARKINGS = tuple(
    (0.015 + index * 0.0157, (-0.24 if index % 2 else 0.24), WHITE, 1)
    for index in range(62)
)


def accepted_track_quality_key(track, counter=None):
    """Copy of ``_track_quality_key`` from the pinned Mission 5 lane blob."""
    representatives = track.representatives
    bins = tuple(value.bin_index for value in representatives)
    span = representatives[-1].y - representatives[0].y
    missing_bins = bins[-1] - bins[0] + 1 - len(bins)
    if len(representatives) < 2:
        residual = 0.0
    else:
        if counter is not None:
            counter["_track_quality_key"] += 1
        fitted = _closed_form_fit(
            tuple((value.x, value.y) for value in representatives)
        )
        if fitted is None:
            residual = math.inf
        else:
            intercept, slope = fitted
            residual = math.sqrt(
                math.fsum(
                    (value.x - (intercept + slope * value.y)) ** 2
                    for value in representatives
                )
                / len(representatives)
            )
    observed_support = math.fsum(
        value.row_coverage for value in representatives
    ) / len(representatives)
    return (
        -len(representatives),
        missing_bins,
        -span,
        residual,
        -observed_support,
        tuple(_representative_identity(value) for value in representatives),
    )


def accepted_track_can_extend(track, next_representative, config, counter):
    """Exercise the accepted Mission 5 extension workload on the same input."""
    last = track.representatives[-1]
    if not _representatives_can_link(last, next_representative, config):
        return False
    if any(
        value.bin_index == next_representative.bin_index
        for value in track.representatives
    ):
        return False
    if len(track.representatives) < 2:
        return True
    counter["_track_can_extend"] += 1
    fitted = _closed_form_fit(
        tuple((value.x, value.y) for value in track.representatives)
    )
    if fitted is None:
        return False
    intercept, slope = fitted
    if abs(slope) > config.max_abs_slope:
        return False
    predicted_x = intercept + slope * next_representative.y
    return abs(next_representative.x - predicted_x) <= _track_prediction_tolerance(
        track, next_representative, config
    )


def scalar_track_can_extend(track, next_representative, config):
    """The pre-R3 scalar-only decision, retained to prove each regression."""
    if not _representatives_can_link(
        track.representatives[-1], next_representative, config
    ):
        return False
    if track.fit.bin_mask & (1 << next_representative.bin_index):
        return False
    if track.fit.count < 2:
        return True
    if track.fit.slope is None:
        return False
    if abs(track.fit.slope) > config.max_abs_slope:
        return False
    predicted_x = track.fit.intercept + track.fit.slope * next_representative.y
    return abs(next_representative.x - predicted_x) <= _track_prediction_tolerance(
        track, next_representative, config
    )


def accepted_continuous_tracks(representatives, config):
    """Run the accepted bounded DP and derive its actual fit-call workload."""
    counter = collections.Counter()
    links = _bounded_outgoing_links(representatives, config)
    incoming = {representative: [] for representative in representatives}
    for first, next_values in links.items():
        for second in next_values:
            incoming[second].append(first)
    for values in incoming.values():
        values.sort(key=_representative_identity)

    by_bin = {}
    for representative in representatives:
        by_bin.setdefault(representative.bin_index, []).append(representative)
    tracks_by_end = {}
    for bin_index in sorted(by_bin):
        bin_tracks = []
        for representative in sorted(
            by_bin[bin_index], key=_representative_identity
        ):
            candidates = [_RepresentativeTrack((representative,))]
            for previous in incoming[representative]:
                for track in tracks_by_end.get(previous, ()):
                    if accepted_track_can_extend(
                        track, representative, config, counter
                    ):
                        candidates.append(
                            _RepresentativeTrack(
                                track.representatives + (representative,)
                            )
                        )
            unique = {}
            for track in candidates:
                identity = tuple(
                    _representative_identity(value)
                    for value in track.representatives
                )
                unique[identity] = track
            ordered = sorted(
                unique.values(),
                key=lambda track: accepted_track_quality_key(track, counter),
            )
            for track in ordered[:MAX_PARTIAL_TRACKS_PER_REPRESENTATIVE]:
                bin_tracks.append(
                    (
                        accepted_track_quality_key(track, counter),
                        representative,
                        track,
                    )
                )
        bin_tracks.sort(key=lambda value: value[0])
        retained_by_end = {}
        for _, representative, track in bin_tracks[:MAX_PARTIAL_TRACKS_PER_BIN]:
            retained_by_end.setdefault(representative, []).append(track)
        for representative, retained in retained_by_end.items():
            tracks_by_end[representative] = tuple(retained)

    completed = {}
    for tracks in tracks_by_end.values():
        for track in tracks:
            if len(track.representatives) < config.min_support_bins:
                continue
            span = track.representatives[-1].y - track.representatives[0].y
            if not math.isfinite(span) or span < config.min_vertical_span:
                continue
            identity = tuple(
                _representative_identity(value)
                for value in track.representatives
            )
            completed[identity] = track
    ordered = tuple(
        sorted(
            completed.values(),
            key=lambda track: accepted_track_quality_key(track, counter),
        )[:MAX_COMPLETED_TRACKS]
    )
    counter["_fit_boundary_hypotheses"] += len(ordered[:MAX_BOUNDARY_SEEDS])
    return ordered, counter


def near_tie_ranking_tracks():
    """Five equal-prefix tracks whose scalar and accepted residual orders differ."""
    ys = tuple(0.52 + index * 0.026 for index in range(10))
    values = (
        ("C1", tuple(0.10 for _ in ys)),
        ("C2", tuple(0.20 for _ in ys)),
        ("C3", tuple(0.30 for _ in ys)),
        (
            "A",
            (
                0.41309289545847167,
                0.417617039339835,
                0.42214118322119837,
                0.4266653271025617,
                0.43118947098392507,
                0.4357136148652884,
                0.4402377587466518,
                0.4447619026280151,
                0.4492860465093785,
                0.45381019039074183,
            ),
        ),
        (
            "B",
            (
                0.4837788021244378,
                0.467339811315602,
                0.45090082050676616,
                0.4344618296979304,
                0.4180228388890946,
                0.40158384808025877,
                0.38514485727142295,
                0.3687058664625872,
                0.35226687565375137,
                0.33582788484491555,
            ),
        ),
    )
    named = []
    for cluster_index, (name, xs) in enumerate(values):
        representatives = tuple(
            _CandidateRepresentative(
                bin_index=index,
                cluster_index=cluster_index,
                x=x,
                y=y,
                pixel_count=5,
                row_count=10,
                bin_row_count=10,
                minimum_x=x - 0.002,
                maximum_x=x + 0.002,
                raster_tolerance=0.002,
            )
            for index, (x, y) in enumerate(zip(xs, ys))
        )
        named.append((name, _RepresentativeTrack(representatives)))
    return tuple(named)


def counting_fit_calls():
    """Return (patcher, counter) attributing _closed_form_fit to its caller."""
    counter = collections.Counter()
    original = _closed_form_fit

    def counting(points):
        counter[sys._getframe(1).f_code.co_name] += 1
        return original(points)

    return mock.patch("perception.lane._closed_form_fit", counting), counter


def linear_representatives(
    count,
    intercept,
    slope,
    *,
    first_y=0.52,
    step=0.026,
    raster_tolerance=0.002,
):
    return tuple(
        _CandidateRepresentative(
            bin_index=index,
            cluster_index=0,
            x=intercept + slope * (first_y + index * step),
            y=first_y + index * step,
            pixel_count=5,
            row_count=10,
            bin_row_count=10,
            minimum_x=intercept + slope * (first_y + index * step) - 0.003,
            maximum_x=intercept + slope * (first_y + index * step) + 0.003,
            raster_tolerance=raster_tolerance,
        )
        for index in range(count)
    )


def pixel_representatives(
    width,
    height,
    coordinates,
    *,
    raster_tolerance=0.01,
):
    """Build normalized representatives from exact integer pixel centers."""
    return tuple(
        _CandidateRepresentative(
            bin_index=index,
            cluster_index=0,
            x=x_px / float(width - 1),
            y=y_px / float(height - 1),
            pixel_count=1,
            row_count=1,
            bin_row_count=1,
            minimum_x=x_px / float(width - 1),
            maximum_x=x_px / float(width - 1),
            raster_tolerance=raster_tolerance,
        )
        for index, (x_px, y_px) in enumerate(coordinates)
    )


def next_representative_on_accepted_line(
    track,
    y,
    *,
    raster_tolerance=0.01,
):
    fitted = _closed_form_fit(
        tuple((value.x, value.y) for value in track.representatives)
    )
    if fitted is None:
        x = track.representatives[-1].x
    else:
        intercept, slope = fitted
        x = intercept + slope * y
    return _CandidateRepresentative(
        bin_index=track.representatives[-1].bin_index + 1,
        cluster_index=0,
        x=x,
        y=y,
        pixel_count=1,
        row_count=1,
        bin_row_count=1,
        minimum_x=x,
        maximum_x=x,
        raster_tolerance=raster_tolerance,
    )


class UntouchableRepresentative:
    """Stand-in that fails loudly if an existing track member is revisited."""

    __slots__ = ()

    def __getattr__(self, name):
        raise AssertionError(f"existing track member was read: {name}")


class IncrementalFitStateTest(unittest.TestCase):
    def assert_matches_closed_form(self, representatives):
        points = tuple((value.x, value.y) for value in representatives)
        expected = _closed_form_fit(points)
        track = _RepresentativeTrack(representatives)
        if expected is None:
            self.assertIsNone(track.fit.slope)
            self.assertIsNone(track.fit.intercept)
            return
        intercept, slope = expected
        self.assertIsNotNone(track.fit.slope)
        for actual, reference, name in (
            (track.fit.slope, slope, "slope"),
            (track.fit.intercept, intercept, "intercept"),
        ):
            with self.subTest(coefficient=name):
                self.assertLessEqual(
                    abs(actual - reference),
                    INCREMENTAL_FIT_TOLERANCE * max(1.0, abs(reference)),
                    f"{name}: {actual!r} vs {reference!r}",
                )

    def test_two_and_three_point_fits_match_closed_form(self):
        for count in (2, 3):
            with self.subTest(count=count):
                self.assert_matches_closed_form(
                    linear_representatives(count, 0.60, -0.40)
                )

    def test_full_eighteen_bin_track_matches_closed_form(self):
        self.assert_matches_closed_form(
            linear_representatives(
                perception.DEFAULT_LANE_CONFIG.vertical_bin_count, 0.60, -0.40
            )
        )

    def test_near_zero_and_maximum_slopes_match_closed_form(self):
        config = perception.DEFAULT_LANE_CONFIG
        for slope in (
            config.min_abs_slope,
            -config.min_abs_slope,
            config.max_abs_slope,
            -config.max_abs_slope,
        ):
            with self.subTest(slope=slope):
                self.assert_matches_closed_form(
                    linear_representatives(18, 0.50, slope)
                )

    def test_translated_coordinates_match_closed_form(self):
        for offset in (0.0, 0.15, 0.30, -0.20):
            with self.subTest(offset=offset):
                self.assert_matches_closed_form(
                    linear_representatives(12, 0.50 + offset, -0.30)
                )

    def test_large_finite_scalar_inputs_match_closed_form(self):
        for magnitude in (1.0e3, 1.0e6, 1.0e9):
            with self.subTest(magnitude=magnitude):
                count = 6
                accumulated = _EMPTY_TRACK_FIT
                xs = []
                ys = []
                for index in range(count):
                    y = magnitude * (1.0 + index)
                    x = magnitude * (2.0 + 0.5 * index)
                    xs.append(x)
                    ys.append(y)
                    accumulated = accumulated.extended(
                        _CandidateRepresentative(
                            bin_index=index,
                            cluster_index=0,
                            x=x,
                            y=y,
                            pixel_count=1,
                            row_count=1,
                            bin_row_count=1,
                            minimum_x=x,
                            maximum_x=x,
                            raster_tolerance=0.0,
                        )
                    )
                expected = _closed_form_fit(tuple(zip(xs, ys)))
                self.assertIsNotNone(expected)
                intercept, slope = expected
                self.assertLessEqual(
                    abs(accumulated.slope - slope),
                    INCREMENTAL_FIT_TOLERANCE * max(1.0, abs(slope)),
                )
                self.assertLessEqual(
                    abs(accumulated.intercept - intercept),
                    INCREMENTAL_FIT_TOLERANCE * max(1.0, abs(intercept)),
                )

    def test_degenerate_inputs_are_undefined_exactly_like_closed_form(self):
        self.assertEqual(_scalar_line(0, 0.0, 0.0, 0.0, 0.0), (None, None))
        self.assertEqual(_scalar_line(1, 0.5, 0.5, 0.25, 0.25), (None, None))
        # Identical y values leave zero vertical dispersion, exactly the case
        # where _closed_form_fit reports no fit.
        repeated = 0.75
        self.assertEqual(
            _scalar_line(2, 2.0 * repeated, 1.2, 2.0 * repeated * repeated, 1.2 * repeated),
            (None, None),
        )
        self.assertIsNone(
            _closed_form_fit(((0.6, repeated), (0.6, repeated)))
        )
        self.assertEqual(
            _scalar_line(2, math.inf, 1.0, math.inf, math.inf), (None, None)
        )

    def test_incremental_and_from_scratch_folds_agree_bit_for_bit(self):
        representatives = linear_representatives(18, 0.60, -0.40)
        incremental = _RepresentativeTrack((representatives[0],))
        for representative in representatives[1:]:
            incremental = _extended_track(
                incremental,
                representative,
                _representative_identity(representative),
            )
        from_scratch = _RepresentativeTrack(representatives)
        self.assertEqual(incremental.fit, from_scratch.fit)
        self.assertEqual(incremental.identity, from_scratch.identity)
        self.assertEqual(incremental, from_scratch)

    def test_empty_fit_state_is_neutral_and_frozen(self):
        self.assertEqual(_EMPTY_TRACK_FIT.count, 0)
        self.assertEqual(_EMPTY_TRACK_FIT.bin_mask, 0)
        self.assertEqual(_EMPTY_TRACK_FIT.maximum_raster_tolerance, 0.0)
        self.assertIsNone(_EMPTY_TRACK_FIT.slope)
        self.assertFalse(hasattr(_EMPTY_TRACK_FIT, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            _EMPTY_TRACK_FIT.count = 1


class InternalBinIndexValidationTest(unittest.TestCase):
    config = perception.DEFAULT_LANE_CONFIG

    def representative(self, bin_index):
        return _CandidateRepresentative(
            bin_index=bin_index,
            cluster_index=0,
            x=0.50,
            y=0.75,
            pixel_count=1,
            row_count=1,
            bin_row_count=1,
            minimum_x=0.50,
            maximum_x=0.50,
            raster_tolerance=0.0,
        )

    def test_invalid_bin_indices_fail_closed_before_track_search(self):
        for bin_index in (
            -2,
            -1,
            self.config.vertical_bin_count,
            self.config.vertical_bin_count + 1,
            MAX_VERTICAL_BINS - 1,
            MAX_VERTICAL_BINS,
            10**9,
            0.0,
            1.5,
            False,
            True,
        ):
            representative = self.representative(bin_index)
            with self.subTest(bin_index=bin_index):
                self.assertEqual(
                    _coerce_representatives((representative,), self.config), ()
                )
                self.assertEqual(
                    _continuous_representative_tracks(
                        (representative,), self.config
                    ),
                    (),
                )
                self.assertEqual(
                    _fit_boundary_hypotheses((representative,), self.config), ()
                )
                fit = _EMPTY_TRACK_FIT.extended(
                    representative, self.config.vertical_bin_count
                )
                self.assertEqual(fit.bin_mask, 0)

    def test_invalid_index_never_shifts_or_mutates_an_existing_mask(self):
        occupied = _EMPTY_TRACK_FIT.extended(
            self.representative(0), self.config.vertical_bin_count
        )
        for bin_index in (
            -2,
            -1,
            self.config.vertical_bin_count,
            self.config.vertical_bin_count + 1,
            MAX_VERTICAL_BINS - 1,
            MAX_VERTICAL_BINS,
            10**9,
            1.0,
            False,
            True,
        ):
            with self.subTest(bin_index=bin_index):
                fit = occupied.extended(
                    self.representative(bin_index),
                    self.config.vertical_bin_count,
                )
                self.assertEqual(fit.bin_mask, occupied.bin_mask)

    def test_boundary_bin_indices_remain_valid(self):
        for bin_index in (0, self.config.vertical_bin_count - 1):
            representative = self.representative(bin_index)
            with self.subTest(bin_index=bin_index):
                self.assertEqual(
                    _coerce_representatives((representative,), self.config),
                    (replace(representative, cluster_index=0),),
                )
                self.assertEqual(
                    _EMPTY_TRACK_FIT.extended(
                        representative, self.config.vertical_bin_count
                    ).bin_mask,
                    1 << bin_index,
                )

    def test_maximum_config_accepts_only_its_valid_upper_endpoint(self):
        valid = self.representative(MAX_VERTICAL_BINS - 1)
        invalid = self.representative(MAX_VERTICAL_BINS)
        self.assertEqual(
            _EMPTY_TRACK_FIT.extended(valid, MAX_VERTICAL_BINS).bin_mask,
            1 << (MAX_VERTICAL_BINS - 1),
        )
        self.assertEqual(
            _EMPTY_TRACK_FIT.extended(invalid, MAX_VERTICAL_BINS).bin_mask,
            0,
        )


class HotPathStructureTest(unittest.TestCase):
    def setUp(self):
        self.config = perception.DEFAULT_LANE_CONFIG
        self.representatives = linear_representatives(10, 0.60, -0.40)
        self.track = _RepresentativeTrack(self.representatives[:-1])
        self.next_representative = self.representatives[-1]

    def test_track_extension_does_not_revisit_existing_members(self):
        """The O(1) update must not iterate the track it extends."""
        poisoned = (
            self.representatives[0],
            UntouchableRepresentative(),
            UntouchableRepresentative(),
            self.representatives[3],
        )
        real = _RepresentativeTrack(self.representatives[:4])
        opaque = _RepresentativeTrack(poisoned, real.fit, real.identity)
        self.assertTrue(
            _track_can_extend(opaque, self.representatives[4], self.config)
        )
        _track_prediction_tolerance(
            opaque, self.representatives[4], self.config
        )
        _maximum_individual_error_tolerance(opaque, self.config)
        extended = _extended_track(
            opaque,
            self.representatives[4],
            _representative_identity(self.representatives[4]),
        )
        self.assertEqual(extended.fit.count, 5)

    def test_track_can_extend_performs_zero_closed_form_fits(self):
        with mock.patch(
            "perception.lane._closed_form_fit",
            side_effect=AssertionError("_track_can_extend refitted"),
        ):
            for length in range(1, len(self.representatives)):
                prefix = _RepresentativeTrack(self.representatives[:length])
                _track_can_extend(
                    prefix, self.next_representative, self.config
                )

    def test_track_can_extend_constructs_no_numpy_array(self):
        constructors = ("asarray", "array", "frombuffer", "mean", "sum", "empty")
        with contextlib.ExitStack() as stack:
            for name in constructors:
                stack.enter_context(
                    mock.patch.object(
                        numpy,
                        name,
                        side_effect=AssertionError(f"numpy.{name} used"),
                    )
                )
            for length in range(1, len(self.representatives)):
                prefix = _RepresentativeTrack(self.representatives[:length])
                _track_can_extend(
                    prefix, self.next_representative, self.config
                )
                _track_quality_key(prefix)

    def test_track_quality_key_performs_zero_closed_form_fits(self):
        with mock.patch(
            "perception.lane._closed_form_fit",
            side_effect=AssertionError("_track_quality_key refitted"),
        ):
            for length in range(1, len(self.representatives) + 1):
                _track_quality_key(
                    _RepresentativeTrack(self.representatives[:length])
                )

    def test_exact_point_eight_boundary_matches_accepted_at_square_resolutions(self):
        disagreements = collections.Counter()
        disagreement_signs = set()
        for size in (96, 128, 160, 200, 256):
            denominator = size - 1
            for sign, coordinates in (
                (1, ((7, 20), (11, 25))),
                (-1, ((denominator - 7, 20), (denominator - 11, 25))),
            ):
                representatives = pixel_representatives(
                    size, size, coordinates
                )
                track = _RepresentativeTrack(representatives)
                next_representative = next_representative_on_accepted_line(
                    track, 30.0 / denominator
                )
                accepted = accepted_track_can_extend(
                    track,
                    next_representative,
                    self.config,
                    collections.Counter(),
                )
                scalar = scalar_track_can_extend(
                    track, next_representative, self.config
                )
                corrected = _track_can_extend(
                    track, next_representative, self.config
                )
                with self.subTest(size=size, sign=sign):
                    self.assertEqual(corrected, accepted)
                    self.assertAlmostEqual(abs(track.fit.slope), 0.8, places=12)
                if scalar != accepted:
                    disagreements[(scalar, accepted)] += 1
                    disagreement_signs.add(sign)

        self.assertEqual(
            set(disagreements), {(True, False), (False, True)}
        )
        self.assertEqual(disagreement_signs, {-1, 1})

        known = _RepresentativeTrack(
            pixel_representatives(96, 96, ((7, 20), (11, 25)))
        )
        accepted = _closed_form_fit(
            tuple((value.x, value.y) for value in known.representatives)
        )
        self.assertEqual(known.fit.slope.hex(), "0x1.999999999999ap-1")
        self.assertEqual(accepted[1].hex(), "0x1.999999999999bp-1")

    def test_configurable_slope_limits_and_ulp_neighbors_match_accepted(self):
        fixtures = (
            (96, ((7, 20), (11, 25))),
            (64, ((1, 1), (3, 2))),
            (64, ((1, 1), (0, 2))),
        )
        raw_directions = set()
        exercised_limits = set()
        for size, coordinates in fixtures:
            track = _RepresentativeTrack(
                pixel_representatives(size, size, coordinates, raster_tolerance=0.02)
            )
            fitted = _closed_form_fit(
                tuple((value.x, value.y) for value in track.representatives)
            )
            self.assertIsNotNone(fitted)
            accepted_slope = fitted[1]
            next_representative = next_representative_on_accepted_line(
                track,
                (coordinates[-1][1] + 1.0) / (size - 1),
                raster_tolerance=0.02,
            )
            for limit in (
                math.nextafter(abs(accepted_slope), 0.0),
                abs(accepted_slope),
                math.nextafter(abs(accepted_slope), math.inf),
            ):
                config = replace(self.config, max_abs_slope=limit)
                accepted = accepted_track_can_extend(
                    track,
                    next_representative,
                    config,
                    collections.Counter(),
                )
                scalar = scalar_track_can_extend(
                    track, next_representative, config
                )
                with self.subTest(
                    size=size, coordinates=coordinates, limit=limit.hex()
                ):
                    self.assertEqual(
                        _track_can_extend(track, next_representative, config),
                        accepted,
                    )
                if scalar != accepted:
                    raw_directions.add((scalar, accepted))
                exercised_limits.add(limit)
        self.assertEqual(raw_directions, {(True, False), (False, True)})
        self.assertGreaterEqual(len(exercised_limits), 6)

    def test_prediction_tolerance_below_equal_above_and_both_disagreements(self):
        size = 96
        track = _RepresentativeTrack(
            pixel_representatives(
                size,
                size,
                ((7, 20), (9, 25)),
                raster_tolerance=0.01,
            )
        )
        config = replace(self.config, max_abs_slope=1.0)
        fitted = _closed_form_fit(
            tuple((value.x, value.y) for value in track.representatives)
        )
        self.assertIsNotNone(fitted)
        next_y = 30.0 / (size - 1)
        prototype = next_representative_on_accepted_line(
            track, next_y, raster_tolerance=0.01
        )
        accepted_prediction = fitted[0] + fitted[1] * next_y
        tolerance = _track_prediction_tolerance(track, prototype, config)

        exact_x = accepted_prediction - tolerance
        below_x = math.nextafter(exact_x, accepted_prediction)
        above_x = math.nextafter(exact_x, -math.inf)
        boundary_cases = (
            ("below", below_x, True),
            ("equal", exact_x, True),
            ("above", above_x, False),
        )
        self.assertEqual(abs(exact_x - accepted_prediction), tolerance)
        for name, x, expected in boundary_cases:
            next_representative = replace(
                prototype, x=x, minimum_x=x, maximum_x=x
            )
            accepted = accepted_track_can_extend(
                track,
                next_representative,
                config,
                collections.Counter(),
            )
            with self.subTest(boundary=name):
                self.assertEqual(accepted, expected)
                self.assertEqual(
                    _track_can_extend(track, next_representative, config),
                    accepted,
                )

        scalar_accepts = replace(
            prototype, x=above_x, minimum_x=above_x, maximum_x=above_x
        )
        self.assertTrue(scalar_track_can_extend(track, scalar_accepts, config))
        self.assertFalse(
            accepted_track_can_extend(
                track, scalar_accepts, config, collections.Counter()
            )
        )

        accepted_accepts_x = accepted_prediction + tolerance
        accepted_accepts = replace(
            prototype,
            x=accepted_accepts_x,
            minimum_x=accepted_accepts_x,
            maximum_x=accepted_accepts_x,
        )
        self.assertFalse(scalar_track_can_extend(track, accepted_accepts, config))
        self.assertTrue(
            accepted_track_can_extend(
                track, accepted_accepts, config, collections.Counter()
            )
        )
        self.assertTrue(_track_can_extend(track, accepted_accepts, config))

    def test_short_long_square_and_rectangular_boundary_sweep(self):
        attempts = 0
        for width, height in (
            (96, 96),
            (128, 128),
            (256, 256),
            (160, 90),
            (320, 180),
            (640, 360),
            (800, 600),
        ):
            for count in (2, 3, 8, 18):
                for sign in (-1, 1):
                    start_x = width - 20 if sign < 0 else 20
                    coordinates = tuple(
                        (start_x + sign * 2 * index, 10 + 3 * index)
                        for index in range(count)
                    )
                    track = _RepresentativeTrack(
                        pixel_representatives(
                            width,
                            height,
                            coordinates,
                            raster_tolerance=0.02,
                        ),
                        vertical_bin_count=32,
                    )
                    fitted = _closed_form_fit(
                        tuple(
                            (value.x, value.y)
                            for value in track.representatives
                        )
                    )
                    self.assertIsNotNone(fitted)
                    next_y = (10 + 3 * count) / float(height - 1)
                    next_representative = next_representative_on_accepted_line(
                        track, next_y, raster_tolerance=0.02
                    )
                    accepted_magnitude = abs(fitted[1])
                    for limit in (
                        math.nextafter(accepted_magnitude, 0.0),
                        accepted_magnitude,
                        math.nextafter(accepted_magnitude, math.inf),
                    ):
                        config = replace(
                            self.config,
                            vertical_bin_count=32,
                            max_abs_slope=limit,
                        )
                        expected = accepted_track_can_extend(
                            track,
                            next_representative,
                            config,
                            collections.Counter(),
                        )
                        with self.subTest(
                            width=width,
                            height=height,
                            count=count,
                            sign=sign,
                            limit=limit.hex(),
                        ):
                            self.assertEqual(
                                _track_can_extend(
                                    track, next_representative, config
                                ),
                                expected,
                            )
                        attempts += 1
        self.assertEqual(attempts, 168)

    def test_defined_degenerate_and_far_decisions_choose_expected_path(self):
        far = _RepresentativeTrack(
            linear_representatives(8, 0.60, -0.40)
        )
        far_next = linear_representatives(9, 0.60, -0.40)[-1]
        with mock.patch(
            "perception.lane._closed_form_fit",
            side_effect=AssertionError("far decision used authoritative fit"),
        ):
            self.assertTrue(_track_can_extend(far, far_next, self.config))

        repeated = tuple(
            replace(
                self.representatives[index],
                bin_index=index,
                x=0.40 + 0.01 * index,
                y=0.70,
            )
            for index in range(2)
        )
        degenerate = _RepresentativeTrack(repeated)
        next_representative = replace(
            self.representatives[2],
            bin_index=2,
            x=0.42,
            y=0.72,
            raster_tolerance=0.05,
        )
        with mock.patch(
            "perception.lane._closed_form_fit", wraps=_closed_form_fit
        ) as fitted:
            self.assertFalse(
                _track_can_extend(
                    degenerate, next_representative, self.config
                )
            )
        self.assertEqual(fitted.call_count, 1)

        singleton = _RepresentativeTrack((self.representatives[0],))
        self.assertTrue(
            _track_can_extend(singleton, self.representatives[1], self.config)
        )

    def test_ambiguous_fit_is_cached_but_far_fit_is_never_requested(self):
        track = _RepresentativeTrack(
            pixel_representatives(96, 96, ((7, 20), (11, 25)))
        )
        next_representative = next_representative_on_accepted_line(
            track, 30.0 / 95.0
        )
        cache = {}
        with mock.patch(
            "perception.lane._closed_form_fit", wraps=_closed_form_fit
        ) as fitted:
            first = _track_can_extend(
                track, next_representative, self.config, cache
            )
            second = _track_can_extend(
                track, next_representative, self.config, cache
            )
        self.assertEqual(first, second)
        self.assertEqual(fitted.call_count, 1)
        self.assertEqual(len(cache), 1)

    def test_repeated_sort_key_evaluation_never_refits(self):
        track = _RepresentativeTrack(self.representatives)
        first = _track_quality_key(track)
        with mock.patch(
            "perception.lane._closed_form_fit",
            side_effect=AssertionError("repeated ranking refitted"),
        ):
            for _ in range(50):
                self.assertEqual(_track_quality_key(track), first)
            self.assertEqual(
                sorted(
                    [track, _RepresentativeTrack(self.representatives[:6])],
                    key=_track_quality_key,
                )[0],
                track,
            )

    def test_complete_selection_matches_accepted_mission_5_definition_exactly(self):
        named = near_tie_ranking_tracks()
        tracks = tuple(track for _, track in named)
        cheap = tuple(sorted(tracks, key=_track_quality_key)[:4])
        accepted = tuple(sorted(tracks, key=accepted_track_quality_key)[:4])
        corrected = _select_tracks_by_accepted_quality(tracks, 4)

        self.assertNotEqual(
            tuple(track.identity for track in cheap),
            tuple(track.identity for track in accepted),
        )
        self.assertNotEqual(
            {track.identity for track in cheap},
            {track.identity for track in accepted},
        )
        self.assertEqual(
            tuple(track.identity for track in corrected),
            tuple(track.identity for track in accepted),
        )
        for track in tracks:
            self.assertEqual(
                _track_quality_key(track)[:3],
                accepted_track_quality_key(track)[:3],
            )

    def test_empty_singleton_and_nonpositive_limits_match_contract(self):
        singleton = _RepresentativeTrack((self.representatives[0],))
        with mock.patch(
            "perception.lane._closed_form_fit",
            side_effect=AssertionError("singleton selection fitted"),
        ):
            self.assertEqual(_select_tracks_by_accepted_quality((), 4), ())
            self.assertEqual(
                _select_tracks_by_accepted_quality((singleton,), 1),
                (singleton,),
            )
            self.assertEqual(
                _select_tracks_by_accepted_quality((singleton,), 10),
                (singleton,),
            )
            for limit in (0, -1, -10):
                self.assertEqual(
                    _select_tracks_by_accepted_quality((singleton,), limit), ()
                )

    def test_nonbinding_selection_restores_complete_accepted_order(self):
        tracks = tuple(track for _, track in near_tie_ranking_tracks())
        cheap = tuple(sorted(tracks, key=_track_quality_key))
        accepted = tuple(sorted(tracks, key=accepted_track_quality_key))
        self.assertNotEqual(
            tuple(track.identity for track in cheap),
            tuple(track.identity for track in accepted),
        )
        for limit in (len(tracks), len(tracks) + 20):
            with self.subTest(limit=limit):
                selected = _select_tracks_by_accepted_quality(tracks, limit)
                self.assertEqual(
                    tuple(track.identity for track in selected),
                    tuple(track.identity for track in accepted),
                )

    def test_binding_cutoff_between_prefix_groups_reorders_retained_group(self):
        named = near_tie_ranking_tracks()
        longer = tuple(track for _, track in named)
        shorter = tuple(
            _RepresentativeTrack(track.representatives[:-1])
            for _, track in named
        )
        tracks = longer + shorter
        limit = len(longer)
        cheap = tuple(sorted(tracks, key=_track_quality_key)[:limit])
        accepted = tuple(
            sorted(tracks, key=accepted_track_quality_key)[:limit]
        )
        self.assertEqual(
            {track.identity for track in cheap},
            {track.identity for track in accepted},
        )
        self.assertNotEqual(
            tuple(track.identity for track in cheap),
            tuple(track.identity for track in accepted),
        )
        selected = _select_tracks_by_accepted_quality(tracks, limit)
        self.assertEqual(
            tuple(track.identity for track in selected),
            tuple(track.identity for track in accepted),
        )

    def test_binding_cutoff_inside_equal_prefix_group_is_exact(self):
        tracks = tuple(track for _, track in near_tie_ranking_tracks())
        accepted = tuple(sorted(tracks, key=accepted_track_quality_key)[:4])
        selected = _select_tracks_by_accepted_quality(tracks, 4)
        self.assertEqual(
            {track.identity for track in selected},
            {track.identity for track in accepted},
        )
        self.assertEqual(
            tuple(track.identity for track in selected),
            tuple(track.identity for track in accepted),
        )

    def test_crossing_group_and_earlier_retained_group_are_both_exact(self):
        named = near_tie_ranking_tracks()
        earlier = tuple(track for _, track in named)
        crossing = tuple(
            _RepresentativeTrack(track.representatives[:-1])
            for _, track in named
        )
        tracks = tuple(reversed(earlier + crossing))
        limit = len(earlier) + 2
        accepted = tuple(
            sorted(tracks, key=accepted_track_quality_key)[:limit]
        )
        selected = _select_tracks_by_accepted_quality(tracks, limit)
        self.assertNotEqual(
            tuple(
                track.identity
                for track in sorted(earlier, key=_track_quality_key)
            ),
            tuple(
                track.identity
                for track in sorted(earlier, key=accepted_track_quality_key)
            ),
        )
        self.assertEqual(
            tuple(track.identity for track in selected),
            tuple(track.identity for track in accepted),
        )

    def test_reversed_input_preserves_accepted_returned_order(self):
        tracks = tuple(track for _, track in near_tie_ranking_tracks())
        expected = tuple(
            track.identity
            for track in sorted(tracks, key=accepted_track_quality_key)[:4]
        )
        for values in (tracks, tuple(reversed(tracks))):
            with self.subTest(reversed=values is not tracks):
                selected = _select_tracks_by_accepted_quality(values, 4)
                self.assertEqual(
                    tuple(track.identity for track in selected), expected
                )

    def test_cap_order_is_identical_across_python_hash_seeds(self):
        code = textwrap.dedent(
            """
            from perception.lane import _select_tracks_by_accepted_quality
            from tests.test_lane_perception import near_tie_ranking_tracks

            tracks = tuple(track for _, track in near_tie_ranking_tracks())
            selected = _select_tracks_by_accepted_quality(
                tuple(reversed(tracks)), 4
            )
            print(repr(tuple(track.identity for track in selected)))
            """
        )
        outputs = []
        for seed in ("0", "1", "17", "2147483647"):
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["PYTHONHASHSEED"] = seed
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=os.path.dirname(os.path.dirname(__file__)),
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            outputs.append(completed.stdout)
        self.assertEqual(len(set(outputs)), 1)

    def test_local_cap_selection_computes_each_quality_key_once_per_track(self):
        tracks = tuple(track for _, track in near_tie_ranking_tracks())
        seen = collections.Counter()
        original = _track_quality_key

        def recording(track):
            seen[id(track)] += 1
            return original(track)

        with mock.patch("perception.lane._track_quality_key", recording):
            _select_tracks_by_accepted_quality(tracks, 4)
        self.assertEqual(len(seen), len(tracks))
        self.assertEqual(set(seen.values()), {1})

    def test_exact_boundary_resolver_fits_crossing_group_once(self):
        tracks = tuple(track for _, track in near_tie_ranking_tracks())
        involved = collections.Counter()
        cache_misses = collections.Counter()
        cache_hits = collections.Counter()
        original = _accepted_track_quality_key

        def recording(track, cheap_key=None, accepted_key_cache=None):
            if (
                accepted_key_cache is not None
                and track.identity in accepted_key_cache
            ):
                cache_hits[track.identity] += 1
            else:
                cache_misses[track.identity] += 1
            involved[track.identity] += 1
            return original(track, cheap_key, accepted_key_cache)

        with mock.patch(
            "perception.lane._accepted_track_quality_key", recording
        ), mock.patch(
            "perception.lane._closed_form_fit", wraps=_closed_form_fit
        ) as fitted:
            selected = _select_tracks_by_accepted_quality(tracks, 4)
        expected = tuple(sorted(tracks, key=accepted_track_quality_key)[:4])
        self.assertEqual(
            tuple(track.identity for track in selected),
            tuple(track.identity for track in expected),
        )
        self.assertEqual(fitted.call_count, len(tracks))
        self.assertEqual(set(cache_misses.values()), {1})
        self.assertGreater(sum(cache_hits.values()), 0)
        self.assertTrue(all(value >= 1 for value in involved.values()))

    def test_exact_boundary_resolver_does_not_fit_without_crossing_prefix(self):
        tracks = tuple(
            _RepresentativeTrack(self.representatives[:length])
            for length in range(10, 5, -1)
        )
        with mock.patch(
            "perception.lane._closed_form_fit",
            side_effect=AssertionError("non-crossing cap fitted"),
        ):
            selected = _select_tracks_by_accepted_quality(tracks, 4)
        self.assertEqual(tuple(len(track.representatives) for track in selected), (10, 9, 8, 7))

    def test_all_three_hard_cap_sites_match_exact_accepted_order(self):
        frame = scene_frame(DENSE_FOUR_MARKINGS, width=640, height=360)
        representatives = _candidate_representatives(frame, self.config)
        original = _select_tracks_by_accepted_quality
        observed_limits = collections.Counter()
        binding_limits = collections.Counter()
        nonbinding_limits = collections.Counter()

        def checking(tracks, limit, accepted_key_cache=None):
            selected = original(tracks, limit, accepted_key_cache)
            accepted = tuple(
                sorted(tracks, key=accepted_track_quality_key)[:limit]
            )
            self.assertEqual(
                tuple(track.identity for track in selected),
                tuple(track.identity for track in accepted),
            )
            observed_limits[limit] += 1
            if len(tracks) > limit:
                binding_limits[limit] += 1
            else:
                nonbinding_limits[limit] += 1
            return selected

        exercised_bin_cap = 5
        exercised_completed_cap = 3
        with mock.patch(
            "perception.lane.MAX_PARTIAL_TRACKS_PER_BIN", exercised_bin_cap
        ), mock.patch(
            "perception.lane.MAX_COMPLETED_TRACKS", exercised_completed_cap
        ), mock.patch(
            "perception.lane._select_tracks_by_accepted_quality", checking
        ):
            _continuous_representative_tracks(representatives, self.config)
        self.assertGreater(
            observed_limits[MAX_PARTIAL_TRACKS_PER_REPRESENTATIVE], 0
        )
        self.assertGreater(observed_limits[exercised_bin_cap], 0)
        self.assertEqual(observed_limits[exercised_completed_cap], 1)
        self.assertGreater(
            binding_limits[MAX_PARTIAL_TRACKS_PER_REPRESENTATIVE], 0
        )
        self.assertGreater(
            nonbinding_limits[MAX_PARTIAL_TRACKS_PER_REPRESENTATIVE], 0
        )
        self.assertGreater(binding_limits[exercised_bin_cap], 0)
        self.assertEqual(binding_limits[exercised_completed_cap], 1)

    def test_per_call_cache_reuses_keys_and_does_not_cross_calls(self):
        frame = scene_frame(DENSE_FOUR_MARKINGS, width=640, height=360)
        representatives = _candidate_representatives(frame, self.config)
        original = _accepted_track_quality_key
        caches = []
        misses = []
        hits = []

        def recording(track, cheap_key=None, accepted_key_cache=None):
            self.assertIsNotNone(accepted_key_cache)
            if not caches or caches[-1] is not accepted_key_cache:
                caches.append(accepted_key_cache)
                misses.append(collections.Counter())
                hits.append(collections.Counter())
            if track.identity in accepted_key_cache:
                hits[-1][track.identity] += 1
            else:
                misses[-1][track.identity] += 1
            return original(track, cheap_key, accepted_key_cache)

        with mock.patch(
            "perception.lane._accepted_track_quality_key", recording
        ), mock.patch(
            "perception.lane.MAX_PARTIAL_TRACKS_PER_BIN", 5
        ), mock.patch(
            "perception.lane.MAX_COMPLETED_TRACKS", 3
        ):
            first = _continuous_representative_tracks(
                representatives, self.config
            )
            second = _continuous_representative_tracks(
                representatives, self.config
            )

        self.assertEqual(
            tuple(track.identity for track in first),
            tuple(track.identity for track in second),
        )
        self.assertEqual(len(caches), 2)
        self.assertIsNot(caches[0], caches[1])
        for index in range(2):
            self.assertEqual(set(misses[index].values()), {1})
            self.assertGreater(sum(hits[index].values()), 0)
            self.assertTrue(caches[index])
            for identity, key in caches[index].items():
                self.assertIsInstance(identity, tuple)
                self.assertIsInstance(key, tuple)
                flattened = repr((identity, key))
                self.assertNotIn("CameraFrame", flattened)
                self.assertNotIn("rgb_bytes", flattened)
                self.assertNotIn("array(", flattened)

    def test_accepted_reference_provenance_is_pinned(self):
        self.assertEqual(
            ACCEPTED_MISSION_5_COMMIT,
            "ede9826bb1842244010229c2dad018ab8877a14f",
        )
        self.assertEqual(
            ACCEPTED_MISSION_5_LANE_BLOB,
            "867be9d112bc6d8ce8857b9f832202d9a11fcab7",
        )

    def test_no_module_level_mutable_cache_exists(self):
        import perception.lane as lane_module

        for name, value in vars(lane_module).items():
            if name.startswith("__"):
                continue
            with self.subTest(name=name):
                self.assertNotIsInstance(value, (dict, list, set, bytearray))
                self.assertFalse(
                    hasattr(value, "cache_info"),
                    f"{name} looks like a functools cache",
                )
                self.assertFalse(isinstance(value, functools._lru_cache_wrapper))

    def test_search_fits_only_selected_or_cutoff_crossing_groups(self):
        frame = scene_frame(DENSE_FOUR_MARKINGS, width=640, height=360)
        representatives = _candidate_representatives(frame, self.config)
        callers = collections.Counter()

        def recording(points):
            callers[sys._getframe(1).f_code.co_name] += 1
            return _closed_form_fit(points)

        with mock.patch("perception.lane._closed_form_fit", recording):
            tracks = _continuous_representative_tracks(
                representatives, self.config
            )
        self.assertTrue(tracks)
        self.assertEqual(set(callers), {"_accepted_track_quality_key"})


class FitCallBudgetTest(unittest.TestCase):
    config = perception.DEFAULT_LANE_CONFIG

    def measure(self, markings):
        frame = scene_frame(markings, width=640, height=360)
        value = snapshot(camera=frame)
        perception.perceive_lanes(value)  # ensure NumPy is already imported
        patcher, counter = counting_fit_calls()
        with patcher:
            perception.perceive_lanes(value)
        return counter, frame

    def accepted_workload(self, frame):
        representatives = _candidate_representatives(frame, self.config)
        return accepted_continuous_tracks(representatives, self.config)

    def test_dense_four_marking_scene_eliminates_inner_loop_fitting(self):
        counter, frame = self.measure(DENSE_FOUR_MARKINGS)
        accepted_tracks, baseline = self.accepted_workload(frame)
        self.assertEqual(counter["_track_can_extend"], 0)
        self.assertEqual(counter["_track_quality_key"], 0)
        total = sum(counter.values())
        self.assertLessEqual(
            total,
            0.20 * sum(baseline.values()),
            f"expected at least an 80% reduction from {sum(baseline.values())}, got {total}",
        )
        tracks = _continuous_representative_tracks(
            _candidate_representatives(frame, self.config), self.config
        )
        self.assertEqual(
            tuple(track.identity for track in tracks),
            tuple(track.identity for track in accepted_tracks),
        )
        self.assertEqual(counter["_fit_boundary_hypotheses"], len(tracks))
        self.assertLessEqual(
            counter["_accepted_track_quality_key"],
            baseline["_track_quality_key"],
        )

    def test_dense_thirty_marking_scene_eliminates_inner_loop_fitting(self):
        counter, frame = self.measure(DENSE_THIRTY_MARKINGS)
        accepted_tracks, baseline = self.accepted_workload(frame)
        self.assertEqual(counter["_track_can_extend"], 0)
        self.assertEqual(counter["_track_quality_key"], 0)
        total = sum(counter.values())
        self.assertLessEqual(total, 0.20 * sum(baseline.values()))
        corrected_tracks = _continuous_representative_tracks(
            _candidate_representatives(frame, self.config), self.config
        )
        self.assertEqual(
            tuple(track.identity for track in corrected_tracks),
            tuple(track.identity for track in accepted_tracks),
        )

    def test_dense_sixty_two_marking_scene_reduces_exercised_baseline_fits(self):
        counter, frame = self.measure(DENSE_SIXTY_TWO_THIN_MARKINGS)
        _, baseline = self.accepted_workload(frame)
        self.assertEqual(counter["_track_can_extend"], 0)
        self.assertEqual(counter["_track_quality_key"], 0)
        self.assertLessEqual(
            sum(counter.values()), 0.20 * sum(baseline.values())
        )

    def test_clean_scene_fit_calls_track_retained_hypothesis_seeds(self):
        counter, frame = self.measure(
            (
                (BASE_LEFT[0], BASE_LEFT[1], WHITE, 5),
                (BASE_RIGHT[0], BASE_RIGHT[1], YELLOW, 5),
            )
        )
        _, baseline = self.accepted_workload(frame)
        self.assertEqual(counter["_track_can_extend"], 0)
        self.assertEqual(counter["_track_quality_key"], 0)
        tracks = _continuous_representative_tracks(
            _candidate_representatives(frame, self.config), self.config
        )
        self.assertEqual(counter["_fit_boundary_hypotheses"], len(tracks))
        self.assertLess(sum(counter.values()), sum(baseline.values()))


class FinalValidationPreservedTest(unittest.TestCase):
    config = perception.DEFAULT_LANE_CONFIG

    def test_every_seed_is_authoritatively_fitted_and_revalidated(self):
        frame = scene_frame(DENSE_FOUR_MARKINGS, width=640, height=360)
        representatives = _candidate_representatives(frame, self.config)
        tracks = _continuous_representative_tracks(representatives, self.config)
        callers = collections.Counter()

        def recording(points):
            callers[sys._getframe(1).f_code.co_name] += 1
            return _closed_form_fit(points)

        with mock.patch(
            "perception.lane._closed_form_fit", recording
        ), mock.patch(
            "perception.lane._track_support_is_valid",
            wraps=_track_support_is_valid,
        ) as support:
            hypotheses = _fit_boundary_hypotheses(representatives, self.config)
        self.assertTrue(hypotheses)
        self.assertEqual(callers["_fit_boundary_hypotheses"], len(tracks))
        self.assertGreater(callers["_accepted_track_quality_key"], 0)
        self.assertGreaterEqual(support.call_count, len(hypotheses))
        for hypothesis in hypotheses:
            self.assertLessEqual(
                hypothesis.residual, self.config.max_fit_residual
            )
            self.assertLessEqual(
                hypothesis.maximum_error,
                _maximum_individual_error_tolerance(
                    hypothesis.track, self.config
                ),
            )

    def test_public_coefficients_come_from_the_authoritative_fit(self):
        frame = scene_frame(
            (
                (BASE_LEFT[0], BASE_LEFT[1], WHITE, 5),
                (BASE_RIGHT[0], BASE_RIGHT[1], YELLOW, 5),
            ),
            width=640,
            height=360,
        )
        representatives = _candidate_representatives(frame, self.config)
        hypotheses = _fit_boundary_hypotheses(representatives, self.config)
        self.assertTrue(hypotheses)
        for hypothesis in hypotheses:
            authoritative = _closed_form_fit(
                tuple(
                    (value.x, value.y)
                    for value in hypothesis.track.representatives
                )
            )
            self.assertIsNotNone(authoritative)
            intercept, slope = authoritative
            self.assertEqual(hypothesis.intercept, intercept)
            self.assertEqual(hypothesis.slope, slope)

    def test_support_validation_still_rejects_an_unsupported_track(self):
        representatives = linear_representatives(
            self.config.min_support_bins, 0.60, -0.40
        )
        track = _RepresentativeTrack(representatives)
        tolerance = _maximum_individual_error_tolerance(track, self.config)
        valid, _ = _track_support_is_valid(track, 0.60, -0.40, self.config)
        self.assertTrue(valid)
        rejected, error = _track_support_is_valid(
            track, 0.60 + tolerance * 1.5, -0.40, self.config
        )
        self.assertFalse(rejected)
        self.assertGreater(error, tolerance)

    def test_support_validation_still_walks_the_full_continuity_chain(self):
        representatives = linear_representatives(6, 0.60, -0.40)

        # An interior member that violates the track's own fitted prediction
        # must still be caught by the re-walk, not only by the error gate.
        deviating = representatives[:3] + (
            replace(representatives[3], x=representatives[3].x + 0.010),
        ) + representatives[4:]
        track = _RepresentativeTrack(deviating)
        self.assertGreater(
            0.010,
            _track_prediction_tolerance(
                _RepresentativeTrack(representatives[:3]),
                deviating[3],
                self.config,
            ),
        )
        valid, error = _track_support_is_valid(track, 0.60, -0.40, self.config)
        self.assertFalse(valid)
        self.assertEqual(error, math.inf)

        # A skipped-bin run beyond the configured bound is still refused.
        broken = representatives[:3] + tuple(
            replace(value, bin_index=value.bin_index + 6)
            for value in representatives[3:]
        )
        valid, error = _track_support_is_valid(
            _RepresentativeTrack(broken), 0.60, -0.40, self.config
        )
        self.assertFalse(valid)
        self.assertEqual(error, math.inf)


class HotPathRegressionTest(unittest.TestCase):
    """Behaviour that the optimization must not disturb."""

    config = perception.DEFAULT_LANE_CONFIG

    def test_four_marking_phantom_boundary_regression_remains_fixed(self):
        markings = DENSE_FOUR_MARKINGS
        frame = scene_frame(markings, width=640, height=360)
        representatives = _candidate_representatives(frame, self.config)
        hypotheses = _fit_boundary_hypotheses(representatives, self.config)
        self.assertEqual(len(hypotheses), 4)
        for hypothesis in hypotheses:
            track = hypothesis.track.representatives
            self.assertEqual(
                len({value.bin_index for value in track}), len(track)
            )
            self.assertEqual(len({value.cluster_index for value in track}), 1)
            self.assertLess(hypothesis.maximum_error, 0.002)
            self.assertAlmostEqual(abs(hypothesis.slope), 0.40, delta=0.01)

    def test_missing_bin_behaviour_is_exact_after_optimization(self):
        base = scene_frame(
            (
                (BASE_LEFT[0], BASE_LEFT[1], WHITE, 5),
                (BASE_RIGHT[0], BASE_RIGHT[1], YELLOW, 5),
            ),
            width=640,
            height=360,
        )
        expected = {
            1: (perception.LaneDetectionState.DETECTED, 17),
            MAX_TRACK_SKIPPED_BINS: (
                perception.LaneDetectionState.DETECTED,
                self.config.vertical_bin_count - MAX_TRACK_SKIPPED_BINS,
            ),
            MAX_TRACK_SKIPPED_BINS + 1: (
                perception.LaneDetectionState.NOT_DETECTED,
                None,
            ),
        }
        for gap, (state, support) in expected.items():
            frame = erase_vertical_bins(base, tuple(range(8, 8 + gap)))
            with self.subTest(gap=gap):
                observation = perception.perceive_lanes(snapshot(camera=frame))
                self.assertIs(observation.state, state)
                if support is None:
                    self.assertIsNone(observation.left_boundary)
                    self.assertIsNone(observation.right_boundary)
                else:
                    self.assertEqual(
                        observation.left_boundary.support_count, support
                    )
                    self.assertEqual(
                        observation.right_boundary.support_count, support
                    )

    def test_track_switching_between_parallel_markings_is_still_refused(self):
        for separation in (0.020, 0.035, 0.050):
            markings = (
                (0.60, -0.40, WHITE, 5),
                (0.60 + separation, -0.40, YELLOW, 5),
                (0.40, 0.40, WHITE, 5),
                (0.40 + separation, 0.40, YELLOW, 5),
            )
            frame = scene_frame(markings, width=640, height=360)
            with self.subTest(separation=separation):
                hypotheses = _fit_boundary_hypotheses(
                    _candidate_representatives(frame, self.config), self.config
                )
                self.assertTrue(hypotheses)
                for hypothesis in hypotheses:
                    clusters = {
                        value.cluster_index
                        for value in hypothesis.track.representatives
                    }
                    self.assertEqual(len(clusters), 1)

    def test_center_crossing_and_same_half_boundaries_remain_fixed(self):
        crossing = perception.perceive_lanes(
            snapshot(
                camera=scene_frame(
                    ((0.72, -0.35, WHITE, 5), (0.60, 0.20, YELLOW, 5))
                )
            )
        )
        self.assertIs(crossing.state, perception.LaneDetectionState.DETECTED)
        self.assertGreater(crossing.left_boundary.endpoints[0].x, 0.5)
        self.assertLess(crossing.left_boundary.endpoints[1].x, 0.5)
        for markings, comparison in (
            (((0.29, -0.27, WHITE, 5), (0.18, 0.27, YELLOW, 5)), self.assertLess),
            (((0.82, -0.27, WHITE, 5), (0.71, 0.27, YELLOW, 5)), self.assertGreater),
        ):
            observation = perception.perceive_lanes(
                snapshot(camera=scene_frame(markings, width=640, height=360))
            )
            with self.subTest(markings=markings):
                self.assertIs(
                    observation.state, perception.LaneDetectionState.DETECTED
                )
                for y in (
                    self.config.roi[1],
                    self.config.center_evaluation_y,
                    self.config.roi[3],
                ):
                    comparison(_boundary_x(observation.left_boundary, y), 0.5)
                    comparison(_boundary_x(observation.right_boundary, y), 0.5)

    def test_reordered_representatives_and_links_give_identical_tracks(self):
        frame = scene_frame(
            (
                (BASE_LEFT[0], BASE_LEFT[1], WHITE, 5),
                (BASE_RIGHT[0], BASE_RIGHT[1], YELLOW, 5),
                (0.25, 0.40, WHITE, 1),
            ),
            width=320,
            height=180,
        )
        value = snapshot(camera=frame)
        expected_observation = perception.perceive_lanes(value)
        representatives = _candidate_representatives(frame, self.config)
        expected_tracks = tuple(
            track.identity
            for track in _continuous_representative_tracks(
                representatives, self.config
            )
        )

        for order in (
            tuple(reversed(representatives)),
            representatives[::2] + representatives[1::2],
        ):
            with self.subTest(order="representatives"):
                self.assertEqual(
                    tuple(
                        track.identity
                        for track in _continuous_representative_tracks(
                            order, self.config
                        )
                    ),
                    expected_tracks,
                )
                with mock.patch(
                    "perception.lane._candidate_representatives",
                    return_value=order,
                ):
                    self.assertEqual(
                        perception.perceive_lanes(value), expected_observation
                    )

        original_links = _bounded_outgoing_links

        def reversed_links(values, config):
            links = original_links(values, config)
            return {
                key: tuple(reversed(links[key]))
                for key in reversed(list(links))
            }

        with mock.patch(
            "perception.lane._bounded_outgoing_links", reversed_links
        ):
            self.assertEqual(
                tuple(
                    track.identity
                    for track in _continuous_representative_tracks(
                        representatives, self.config
                    )
                ),
                expected_tracks,
            )
            self.assertEqual(
                perception.perceive_lanes(value), expected_observation
            )

    def test_repeated_identical_calls_remain_exactly_equal(self):
        for markings, width, height in (
            (DENSE_FOUR_MARKINGS, 640, 360),
            (DENSE_THIRTY_MARKINGS, 640, 360),
            (
                (
                    (BASE_LEFT[0], BASE_LEFT[1], WHITE, 5),
                    (BASE_RIGHT[0], BASE_RIGHT[1], YELLOW, 5),
                ),
                640,
                360,
            ),
        ):
            value = snapshot(
                camera=scene_frame(markings, width=width, height=height)
            )
            first = perception.perceive_lanes(value)
            with self.subTest(markings=len(markings)):
                for _ in range(5):
                    repeated = perception.perceive_lanes(value)
                    self.assertEqual(repeated, first)
                    for name in (
                        "confidence",
                        "center_offset",
                        "heading_proxy",
                    ):
                        self.assertEqual(
                            getattr(repeated, name), getattr(first, name)
                        )

    def test_public_observation_is_identical_across_python_hash_seeds(self):
        code = textwrap.dedent(
            """
            import perception
            from tests.test_lane_perception import (
                DENSE_FOUR_MARKINGS,
                scene_frame,
                snapshot,
            )

            value = snapshot(
                camera=scene_frame(
                    DENSE_FOUR_MARKINGS, width=640, height=360
                )
            )
            print(repr(perception.perceive_lanes(value)))
            """
        )
        outputs = []
        for seed in ("0", "1", "17", "2147483647"):
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["PYTHONHASHSEED"] = seed
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=os.path.dirname(os.path.dirname(__file__)),
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            outputs.append(completed.stdout)
        self.assertEqual(len(set(outputs)), 1)

    def test_pair_combination_cap_remains_directly_enforced(self):
        hypotheses = _fit_boundary_hypotheses(
            _candidate_representatives(
                scene_frame(DENSE_FOUR_MARKINGS, width=640, height=360),
                self.config,
            ),
            self.config,
        )
        self.assertGreaterEqual(len(hypotheses), 3)
        for cap in (1, 2):
            with self.subTest(cap=cap), mock.patch(
                "perception.lane.MAX_PAIR_COMBINATIONS", cap
            ), mock.patch(
                "perception.lane._pair_geometry", wraps=_pair_geometry
            ) as geometry:
                _select_boundaries(hypotheses, self.config)
            self.assertEqual(geometry.call_count, cap)

    def test_output_retains_no_internal_optimization_state(self):
        observation = perception.perceive_lanes(
            snapshot(camera=detected_frame())
        )
        self.assertIs(observation.state, perception.LaneDetectionState.DETECTED)
        forbidden = (
            _TrackFit,
            _RepresentativeTrack,
            _CandidateRepresentative,
            TelemetrySnapshot,
            CameraFrame,
            bytes,
            bytearray,
            memoryview,
            numpy.ndarray,
            numpy.generic,
            dict,
            list,
            set,
            frozenset,
            BaseException,
        )
        seen = set()

        def walk(value):
            marker = id(value)
            if marker in seen:
                return
            seen.add(marker)
            self.assertNotIsInstance(value, forbidden)
            self.assertFalse(callable(value))
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                for field in fields(value):
                    walk(getattr(value, field.name))
            elif type(value) is tuple:
                for item in value:
                    walk(item)
            else:
                self.assertTrue(
                    value is None
                    or isinstance(value, enum.Enum)
                    or type(value) in (int, float, str)
                )

        walk(observation)

    def test_hot_path_helpers_stay_scalar_only_at_import(self):
        result = run_isolated_python(
            textwrap.dedent(
                """
                import builtins
                import sys

                assert "numpy" not in sys.modules
                from perception.lane import (
                    _EMPTY_TRACK_FIT,
                    _CandidateRepresentative,
                    _RepresentativeTrack,
                    _scalar_line,
                    _select_tracks_by_accepted_quality,
                    _track_can_extend,
                    _track_quality_key,
                )
                from perception.state import DEFAULT_LANE_CONFIG

                assert "numpy" not in sys.modules
                original_import = builtins.__import__

                def blocked_import(name, *args, **kwargs):
                    if name == "numpy" or name.startswith("numpy."):
                        raise AssertionError("ordinary scalar path imported NumPy")
                    return original_import(name, *args, **kwargs)

                builtins.__import__ = blocked_import
                representatives = tuple(
                    _CandidateRepresentative(
                        bin_index=index,
                        cluster_index=0,
                        x=0.60 - 0.40 * (0.52 + index * 0.026),
                        y=0.52 + index * 0.026,
                        pixel_count=5,
                        row_count=10,
                        bin_row_count=10,
                        minimum_x=0.0,
                        maximum_x=1.0,
                        raster_tolerance=0.002,
                    )
                    for index in range(6)
                )
                track = _RepresentativeTrack(representatives[:-1])
                assert _track_can_extend(
                    track, representatives[-1], DEFAULT_LANE_CONFIG
                )
                assert _track_quality_key(track)[0] == -5
                singleton = _RepresentativeTrack((representatives[0],))
                assert _select_tracks_by_accepted_quality(
                    (singleton,), 4
                ) == (singleton,)
                assert _scalar_line(1, 0.0, 0.0, 0.0, 0.0) == (None, None)
                assert "numpy" not in sys.modules
                builtins.__import__ = original_import
                print("SCALAR_ONLY")
                """
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "SCALAR_ONLY")

    def test_ambiguous_extension_loads_numpy_only_for_authoritative_fallback(self):
        result = run_isolated_python(
            textwrap.dedent(
                """
                import sys

                from perception.lane import (
                    _CandidateRepresentative,
                    _RepresentativeTrack,
                    _track_can_extend,
                )
                from perception.state import DEFAULT_LANE_CONFIG

                assert "numpy" not in sys.modules
                denominator = 95.0
                representatives = tuple(
                    _CandidateRepresentative(
                        bin_index=index,
                        cluster_index=0,
                        x=x_px / denominator,
                        y=y_px / denominator,
                        pixel_count=1,
                        row_count=1,
                        bin_row_count=1,
                        minimum_x=x_px / denominator,
                        maximum_x=x_px / denominator,
                        raster_tolerance=0.01,
                    )
                    for index, (x_px, y_px) in enumerate(
                        ((7, 20), (11, 25), (15, 30))
                    )
                )
                track = _RepresentativeTrack(representatives[:2])
                assert not _track_can_extend(
                    track, representatives[2], DEFAULT_LANE_CONFIG
                )
                assert "numpy" in sys.modules
                print("BOUNDARY_FALLBACK_ONLY")
                """
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "BOUNDARY_FALLBACK_ONLY")

    def test_nineteen_square_boundary_public_regressions_match_accepted(self):
        cases = []
        for size in (96, 128, 160, 200, 256):
            pixel = 1.0 / (size - 1)
            for phase in range(4):
                if len(cases) == 19:
                    break
                shift = (phase - 1.5) * pixel
                markings = (
                    (0.61 + shift, -0.40, WHITE, 3),
                    (0.39 + shift, 0.40, YELLOW, 3),
                    (0.90 - shift, -0.80, WHITE, 1),
                    (0.10 - shift, 0.80, YELLOW, 1),
                )
                cases.append((size, phase, markings))

        self.assertEqual(len(cases), 19)
        for size, phase, markings in cases:
            frame = scene_frame(markings, width=size, height=size)
            value = snapshot(camera=frame)
            with mock.patch(
                "perception.lane._continuous_representative_tracks",
                side_effect=lambda representatives, config: (
                    accepted_continuous_tracks(representatives, config)[0]
                ),
            ):
                accepted = perception.perceive_lanes(value)
            corrected = perception.perceive_lanes(value)
            with self.subTest(size=size, phase=phase):
                self.assertEqual(corrected, accepted)


if __name__ == "__main__":
    unittest.main()
