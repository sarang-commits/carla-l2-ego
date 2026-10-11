"""Deterministic unit tests for the Mission 9 lateral controller.

Every input here is an immutable synthetic record built from the accepted
Mission 5/6/7/8 public types.  No CARLA package, server, socket, worker, or wall
clock is involved: control time is always supplied explicitly.
"""

from __future__ import annotations

import math
import numbers
import unittest
from dataclasses import FrozenInstanceError, fields

from control.lateral import (
    DEFAULT_LATERAL_CONTROLLER_CONFIG,
    MAX_LATERAL_GAIN,
    MAX_STEERING_RATE_PER_SECOND,
    LateralControlMode,
    LateralControlReason,
    LateralControlRequest,
    LateralController,
    LateralControllerConfig,
    _REASON_MODE,
)
from perception.runtime import PerceptionRuntimeResult
from perception.state import (
    LaneBoundaryObservation,
    LaneBoundarySide,
    LaneDetectionState,
    LaneObservationReason,
    NormalizedImagePoint,
)
from perception.temporal import (
    TemporalLaneBoundary,
    TemporalLaneEstimate,
    TemporalLaneObservation,
    TemporalLaneState,
)
from safety.supervisor import SafetyDecision, SafetyReason, SafetyState
from telemetry.state import SampleStamp


DETECTED = "detected"
PARTIAL = "partial"
COASTING = "coasting"
LOST = "lost"
UNUSABLE = "unusable"


@numbers.Real.register
class DriftingReal:
    """A hostile ``numbers.Real`` whose float value changes between reads."""

    def __init__(self, *values: float) -> None:
        self._values = list(values)

    def __float__(self) -> float:
        if len(self._values) > 1:
            return float(self._values.pop(0))
        return float(self._values[0])


class ConfigSubclass(LateralControllerConfig):
    """A caller-owned config subtype that must never be retained as-is."""


def unvalidated(base, **overrides):
    """Return a validation-bypassing clone of one accepted frozen record.

    Mission 9 must survive evidence that no accepted constructor would ever
    build.  Subclassing and assigning the slots directly is the only way to
    hand it such a record without weakening the upstream validators.
    """
    subclass = type(f"Unvalidated{type(base).__name__}", (type(base),), {})
    clone = object.__new__(subclass)
    for field in fields(base):
        object.__setattr__(
            clone,
            field.name,
            overrides[field.name]
            if field.name in overrides
            else getattr(base, field.name),
        )
    return clone


def exploding(base, name: str, exception: BaseException):
    """Return a clone whose ``name`` attribute raises when it is read."""
    clone = unvalidated(base)

    def raiser(self):
        raise exception

    setattr(type(clone), name, property(raiser))
    return clone


def raw_boundary(
    side: LaneBoundarySide, confidence: float = 0.80
) -> LaneBoundaryObservation:
    if side is LaneBoundarySide.LEFT:
        upper_x, lower_x, slope, intercept = 0.34, 0.24, -0.20, 0.44
    else:
        upper_x, lower_x, slope, intercept = 0.66, 0.76, 0.20, 0.56
    return LaneBoundaryObservation(
        side=side,
        endpoints=(
            NormalizedImagePoint(upper_x, 0.50),
            NormalizedImagePoint(lower_x, 1.00),
        ),
        x_intercept=intercept,
        x_slope=slope,
        support_count=12,
        fit_residual=0.01,
        confidence=confidence,
    )


def temporal_boundary(
    side: LaneBoundarySide, confidence: float
) -> TemporalLaneBoundary:
    intercept, slope = (
        (0.44, -0.20) if side is LaneBoundarySide.LEFT else (0.56, 0.20)
    )
    return TemporalLaneBoundary(
        side=side,
        x_intercept=intercept,
        x_slope=slope,
        confidence=confidence,
    )


def build_carrier(
    frame: int,
    kind: str = DETECTED,
    *,
    offset: float = 0.0,
    heading: float = 0.0,
    raw_offset: float | None = None,
    raw_heading: float | None = None,
    sim_time: float | None = None,
    monotonic: float | None = None,
    revision: int | None = None,
    confidence: float = 0.80,
) -> TemporalLaneObservation:
    """Build one exact Mission 7 carrier for the requested temporal state."""
    selected_sim = frame / 10.0 if sim_time is None else sim_time
    selected_monotonic = frame / 10.0 if monotonic is None else monotonic
    selected_revision = frame if revision is None else revision
    stamp = SampleStamp(
        frame=frame, sim_time=selected_sim, monotonic=selected_monotonic
    )
    source = dict(
        source_frame=frame,
        source_simulation_timestamp=selected_sim,
        source_receive_monotonic=selected_monotonic,
    )
    left = temporal_boundary(LaneBoundarySide.LEFT, confidence)
    right = temporal_boundary(LaneBoundarySide.RIGHT, confidence)

    if kind == DETECTED:
        raw_state = LaneDetectionState.DETECTED
        raw_reason = LaneObservationReason.DETECTED
        estimate_fields = dict(
            state=TemporalLaneState.TRACKING,
            left_boundary=left,
            right_boundary=right,
            center_offset=offset,
            heading_proxy=heading,
            confidence=confidence,
            consecutive_misses=0,
            track_age_seconds=0.0,
            extrapolated=False,
        )
        raw_fields = dict(
            left_boundary=raw_boundary(LaneBoundarySide.LEFT),
            right_boundary=raw_boundary(LaneBoundarySide.RIGHT),
            centerline_endpoints=(
                NormalizedImagePoint(0.50, 0.50),
                NormalizedImagePoint(0.50, 1.00),
            ),
            center_offset=offset if raw_offset is None else raw_offset,
            heading_proxy=heading if raw_heading is None else raw_heading,
            confidence=confidence,
        )
    elif kind == PARTIAL:
        raw_state = LaneDetectionState.PARTIAL
        raw_reason = LaneObservationReason.SINGLE_PLAUSIBLE_BOUNDARY
        estimate_fields = dict(
            state=TemporalLaneState.TRACKING,
            left_boundary=left,
            confidence=confidence,
            consecutive_misses=0,
            track_age_seconds=0.0,
            extrapolated=False,
        )
        raw_fields = dict(
            left_boundary=raw_boundary(LaneBoundarySide.LEFT),
            confidence=confidence,
        )
    elif kind == COASTING:
        raw_state = LaneDetectionState.NOT_DETECTED
        raw_reason = LaneObservationReason.NO_PLAUSIBLE_BOUNDARY
        estimate_fields = dict(
            state=TemporalLaneState.COASTING,
            left_boundary=left,
            right_boundary=right,
            center_offset=offset,
            heading_proxy=heading,
            confidence=confidence,
            consecutive_misses=1,
            track_age_seconds=0.10,
            extrapolated=True,
        )
        raw_fields = dict(confidence=0.0)
    elif kind == LOST:
        raw_state = LaneDetectionState.NOT_DETECTED
        raw_reason = LaneObservationReason.NO_PLAUSIBLE_BOUNDARY
        estimate_fields = dict(
            state=TemporalLaneState.LOST,
            confidence=0.0,
            consecutive_misses=3,
            track_age_seconds=0.30,
            extrapolated=False,
        )
        raw_fields = dict(confidence=0.0)
    elif kind == UNUSABLE:
        raw_state = LaneDetectionState.INPUT_UNUSABLE
        raw_reason = LaneObservationReason.STATUS_UNUSABLE
        estimate_fields = dict(
            state=TemporalLaneState.INPUT_UNUSABLE,
            confidence=0.0,
            consecutive_misses=0,
            extrapolated=False,
        )
        raw_fields = dict(confidence=0.0)
    else:  # pragma: no cover - guards the test builder itself
        raise AssertionError(f"unsupported synthetic temporal kind: {kind!r}")

    estimate = TemporalLaneEstimate(
        source_stamp=stamp,
        snapshot_revision=selected_revision,
        raw_state=raw_state,
        raw_reason=raw_reason,
        **estimate_fields,
    )
    return TemporalLaneObservation(
        snapshot_revision=selected_revision,
        state=raw_state,
        reason=raw_reason,
        temporal_estimate=estimate,
        **source,
        **raw_fields,
    )


def wrap(
    carrier,
    *,
    frame: int | None = None,
    sim_time: float | None = None,
    monotonic: float | None = None,
    revision: int | None = None,
) -> PerceptionRuntimeResult:
    """Wrap one carrier in a runtime result, optionally with a foreign identity."""
    estimate_stamp = carrier.temporal_estimate.source_stamp
    selected_frame = estimate_stamp.frame if frame is None else frame
    selected_sim = estimate_stamp.sim_time if sim_time is None else sim_time
    selected_monotonic = (
        estimate_stamp.monotonic if monotonic is None else monotonic
    )
    selected_revision = (
        carrier.snapshot_revision if revision is None else revision
    )
    return PerceptionRuntimeResult(
        observation=carrier,
        source_stamp=SampleStamp(
            frame=selected_frame,
            sim_time=selected_sim,
            monotonic=selected_monotonic,
        ),
        source_sequence=None,
        snapshot_revision=selected_revision,
        ingestion_monotonic=selected_monotonic,
        processing_start_monotonic=selected_monotonic,
        processing_finish_monotonic=selected_monotonic,
        pending_age_seconds=0.0,
        processing_latency_seconds=0.0,
    )


def result(frame: int = 1, kind: str = DETECTED, **kwargs) -> PerceptionRuntimeResult:
    return wrap(build_carrier(frame, kind, **kwargs))


_PERMITTING_REASON = {
    SafetyState.NOMINAL: SafetyReason.NOMINAL_TRACKING,
    SafetyState.DEGRADED: SafetyReason.DEGRADED_COASTING,
}
_REFUSING = {
    SafetyState.INITIALIZING: SafetyReason.STARTUP_NO_RESULT,
    SafetyState.RECOVERING: SafetyReason.RECOVERY_PENDING,
    SafetyState.FAIL_SAFE: SafetyReason.TEMPORAL_LOST,
}


def decision(
    value: PerceptionRuntimeResult,
    *,
    state: SafetyState = SafetyState.NOMINAL,
    reason: SafetyReason | None = None,
    now: float = 100.0,
    stamp: SampleStamp | None = None,
    revision: int | None = None,
    temporal_state: TemporalLaneState | None = None,
    confidence: float | None = None,
) -> SafetyDecision:
    """Build the Mission 8 decision a healthy supervisor would issue here."""
    estimate = value.observation.temporal_estimate
    if state in _PERMITTING_REASON:
        selected_reason = _PERMITTING_REASON[state] if reason is None else reason
        return SafetyDecision(
            state=state,
            reason=selected_reason,
            autonomy_allowed=True,
            evaluation_monotonic=now,
            source_stamp=value.source_stamp if stamp is None else stamp,
            snapshot_revision=(
                value.snapshot_revision if revision is None else revision
            ),
            result_age_seconds=0.0,
            temporal_state=(
                estimate.state if temporal_state is None else temporal_state
            ),
            temporal_confidence=(
                estimate.confidence if confidence is None else confidence
            ),
        )
    selected_reason = _REFUSING[state] if reason is None else reason
    if state is SafetyState.INITIALIZING:
        return SafetyDecision(
            state=state,
            reason=selected_reason,
            autonomy_allowed=False,
        )
    return SafetyDecision(
        state=state,
        reason=selected_reason,
        autonomy_allowed=False,
        evaluation_monotonic=now,
        source_stamp=value.source_stamp if stamp is None else stamp,
        snapshot_revision=value.snapshot_revision if revision is None else revision,
        result_age_seconds=0.0,
        temporal_state=estimate.state,
        temporal_confidence=estimate.confidence,
    )


def pair(frame: int = 1, kind: str = DETECTED, *, state=SafetyState.NOMINAL, **kwargs):
    """Return one coherent (result, decision) pair for ``frame``."""
    value = result(frame, kind, **kwargs)
    return value, decision(value, state=state)


class LateralTestCase(unittest.TestCase):
    def assert_refused(self, request, reason: LateralControlReason) -> None:
        self.assertIs(request.reason, reason)
        self.assertIs(request.mode, LateralControlMode.DISABLED)
        self.assertFalse(request.lateral_allowed)
        self.assertEqual(request.steering, 0.0)
        self.assertEqual(request.raw_steering, 0.0)
        self.assertEqual(request.steering_authority, 0.0)
        self.assertIsNone(request.lateral_error)
        self.assertIsNone(request.heading_error)

    def drive(self, controller, frame, kind=DETECTED, *, now, state=SafetyState.NOMINAL, **kwargs):
        value, verdict = pair(frame, kind, state=state, **kwargs)
        return controller.update(value, verdict, now=now)


class ConfigurationTest(LateralTestCase):
    def test_defaults_are_bounded_and_ordered(self):
        config = DEFAULT_LATERAL_CONTROLLER_CONFIG
        self.assertEqual(config.offset_gain, 1.20)
        self.assertEqual(config.heading_gain, 0.40)
        self.assertEqual(config.nominal_max_steering, 0.35)
        self.assertEqual(config.degraded_max_steering, 0.15)
        self.assertEqual(config.max_steering_rate_per_second, 1.50)
        self.assertEqual(config.steering_deadband, 0.0)
        self.assertLessEqual(config.degraded_max_steering, config.nominal_max_steering)
        self.assertLessEqual(config.nominal_max_steering, 1.0)
        self.assertGreater(config.degraded_max_steering, 0.0)

    def test_default_gain_ratio_is_a_lookahead_inside_the_mission_5_roi(self):
        # heading_gain / offset_gain is the effective normalized image lookahead
        # measured upward from the Mission 5 evaluation row 0.90.  It must land
        # inside the default ROI [0.50, 1.00] that Mission 5 actually fitted.
        config = DEFAULT_LATERAL_CONTROLLER_CONFIG
        lookahead = config.heading_gain / config.offset_gain
        self.assertGreater(0.90 - lookahead, 0.50)
        self.assertLess(0.90 - lookahead, 0.90)

    def test_hostile_real_values_are_canonicalized_to_exact_floats(self):
        config = LateralControllerConfig(
            offset_gain=DriftingReal(1.0, 9.0),
            heading_gain=DriftingReal(0.5, 9.0),
            nominal_max_steering=DriftingReal(0.40, 9.0),
            degraded_max_steering=DriftingReal(0.20, 9.0),
            max_steering_rate_per_second=DriftingReal(2.0, 9.0),
            steering_deadband=DriftingReal(0.01, 9.0),
        )
        for name, expected in (
            ("offset_gain", 1.0),
            ("heading_gain", 0.5),
            ("nominal_max_steering", 0.40),
            ("degraded_max_steering", 0.20),
            ("max_steering_rate_per_second", 2.0),
            ("steering_deadband", 0.01),
        ):
            value = getattr(config, name)
            self.assertIs(type(value), float, name)
            self.assertEqual(value, expected, name)

    def test_non_finite_values_are_rejected(self):
        for name in (
            "offset_gain",
            "heading_gain",
            "nominal_max_steering",
            "degraded_max_steering",
            "max_steering_rate_per_second",
            "steering_deadband",
        ):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        LateralControllerConfig(**{name: value})

    def test_bool_is_not_a_number(self):
        for name in (
            "offset_gain",
            "heading_gain",
            "nominal_max_steering",
            "degraded_max_steering",
            "max_steering_rate_per_second",
            "steering_deadband",
        ):
            for value in (True, False):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        LateralControllerConfig(**{name: value})

    def test_non_real_values_are_rejected(self):
        with self.assertRaises(ValueError):
            LateralControllerConfig(offset_gain="1.0")

    def test_invalid_authority_ordering_is_rejected(self):
        with self.assertRaises(ValueError):
            LateralControllerConfig(
                nominal_max_steering=0.20, degraded_max_steering=0.30
            )

    def test_equal_authorities_are_permitted(self):
        config = LateralControllerConfig(
            nominal_max_steering=0.30, degraded_max_steering=0.30
        )
        self.assertEqual(config.degraded_max_steering, config.nominal_max_steering)

    def test_out_of_range_authorities_are_rejected(self):
        for kwargs in (
            dict(nominal_max_steering=0.0),
            dict(nominal_max_steering=-0.10),
            dict(nominal_max_steering=1.01),
            dict(degraded_max_steering=0.0),
            dict(degraded_max_steering=-0.10),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    LateralControllerConfig(**kwargs)

    def test_invalid_rate_is_rejected(self):
        for value in (0.0, -1.0, MAX_STEERING_RATE_PER_SECOND + 1.0):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    LateralControllerConfig(max_steering_rate_per_second=value)

    def test_invalid_gains_are_rejected(self):
        for kwargs in (
            dict(offset_gain=0.0),
            dict(offset_gain=-1.0),
            dict(offset_gain=MAX_LATERAL_GAIN + 1.0),
            dict(heading_gain=-1.0),
            dict(heading_gain=MAX_LATERAL_GAIN + 1.0),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    LateralControllerConfig(**kwargs)

    def test_zero_heading_gain_is_permitted(self):
        self.assertEqual(LateralControllerConfig(heading_gain=0.0).heading_gain, 0.0)

    def test_deadband_must_stay_below_the_smallest_authority(self):
        with self.assertRaises(ValueError):
            LateralControllerConfig(
                degraded_max_steering=0.10, steering_deadband=0.10
            )
        with self.assertRaises(ValueError):
            LateralControllerConfig(steering_deadband=-0.01)
        self.assertEqual(
            LateralControllerConfig(
                degraded_max_steering=0.10, steering_deadband=0.09
            ).steering_deadband,
            0.09,
        )

    def test_config_is_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            DEFAULT_LATERAL_CONTROLLER_CONFIG.offset_gain = 2.0

    def test_controller_detaches_a_caller_owned_config_subclass(self):
        supplied = ConfigSubclass(offset_gain=2.0)
        controller = LateralController(supplied)
        self.assertIs(type(controller.config), LateralControllerConfig)
        self.assertIsNot(controller.config, supplied)
        self.assertEqual(controller.config.offset_gain, 2.0)

    def test_controller_rejects_a_foreign_config(self):
        with self.assertRaises(TypeError):
            LateralController(config="fast")


class ReasonModeContractTest(LateralTestCase):
    def test_every_reason_maps_to_exactly_one_mode(self):
        self.assertEqual(set(_REASON_MODE), set(LateralControlReason))

    def test_only_tracking_reasons_carry_authority(self):
        permitted = {
            reason
            for reason, mode in _REASON_MODE.items()
            if mode is not LateralControlMode.DISABLED
        }
        self.assertEqual(
            permitted,
            {
                LateralControlReason.NOMINAL_TRACKING,
                LateralControlReason.DEGRADED_TRACKING,
                LateralControlReason.DEGRADED_COASTING,
            },
        )

    def test_a_refused_request_cannot_carry_steering(self):
        with self.assertRaises(ValueError):
            LateralControlRequest(
                mode=LateralControlMode.DISABLED,
                reason=LateralControlReason.SAFETY_NOT_PERMITTED,
                lateral_allowed=False,
                steering=0.20,
            )

    def test_mode_must_match_its_reason(self):
        with self.assertRaises(ValueError):
            LateralControlRequest(
                mode=LateralControlMode.NOMINAL,
                reason=LateralControlReason.SAFETY_NOT_PERMITTED,
                lateral_allowed=True,
            )

    def test_steering_cannot_exceed_its_own_authority(self):
        with self.assertRaises(ValueError):
            LateralControlRequest(
                mode=LateralControlMode.NOMINAL,
                reason=LateralControlReason.NOMINAL_TRACKING,
                lateral_allowed=True,
                steering=0.40,
                steering_authority=0.35,
                evaluation_monotonic=1.0,
                source_stamp=SampleStamp(frame=1, sim_time=0.1, monotonic=0.1),
                snapshot_revision=1,
                lateral_error=0.0,
                heading_error=0.0,
                safety_state=SafetyState.NOMINAL,
                temporal_state=TemporalLaneState.TRACKING,
            )


class StartupTest(LateralTestCase):
    def test_construction_publishes_a_neutral_unevaluated_request(self):
        controller = LateralController()
        request = controller.latest_request
        self.assert_refused(request, LateralControlReason.NOT_EVALUATED)
        self.assertIsNone(request.evaluation_monotonic)
        self.assertIsNone(request.source_stamp)
        self.assertIsNone(request.snapshot_revision)
        self.assertIsNone(request.source_frame)
        self.assertIsNone(request.source_simulation_timestamp)
        self.assertFalse(controller.lateral_allowed)

    def test_update_requires_the_accepted_argument_types(self):
        controller = LateralController()
        value, verdict = pair()
        with self.assertRaises(TypeError):
            controller.update(None, verdict, now=1.0)
        with self.assertRaises(TypeError):
            controller.update(value, None, now=1.0)


class SafetyGateTest(LateralTestCase):
    def test_every_refusing_safety_state_yields_exactly_zero(self):
        for state in (
            SafetyState.INITIALIZING,
            SafetyState.RECOVERING,
            SafetyState.FAIL_SAFE,
        ):
            with self.subTest(state=state):
                controller = LateralController()
                request = self.drive(
                    controller, 1, now=1.0, state=state, offset=0.40, heading=-0.40
                )
                self.assert_refused(
                    request, LateralControlReason.SAFETY_NOT_PERMITTED
                )

    def test_perfect_geometry_cannot_bypass_the_safety_gate(self):
        controller = LateralController()
        value = result(1, DETECTED, offset=0.30, heading=-0.30, confidence=1.0)
        refused = SafetyDecision(
            state=SafetyState.FAIL_SAFE,
            reason=SafetyReason.RESULT_STALE,
            autonomy_allowed=False,
            evaluation_monotonic=1.0,
            source_stamp=value.source_stamp,
            snapshot_revision=value.snapshot_revision,
            result_age_seconds=9.0,
            temporal_state=TemporalLaneState.TRACKING,
            temporal_confidence=1.0,
        )
        request = controller.update(value, refused, now=1.0)
        self.assert_refused(request, LateralControlReason.SAFETY_NOT_PERMITTED)
        self.assertEqual(controller.metrics.safety_rejects, 1)

    def test_a_latched_decision_is_refused_even_if_it_claims_permission(self):
        controller = LateralController()
        value, verdict = pair(1, offset=0.30)
        hostile = unvalidated(verdict, latched=True)
        self.assert_refused(
            controller.update(value, hostile, now=1.0),
            LateralControlReason.SAFETY_NOT_PERMITTED,
        )

    def test_a_non_bool_permission_claim_is_refused(self):
        controller = LateralController()
        value, verdict = pair(1, offset=0.30)
        hostile = unvalidated(verdict, autonomy_allowed=1)
        self.assert_refused(
            controller.update(value, hostile, now=1.0),
            LateralControlReason.SAFETY_NOT_PERMITTED,
        )

    def test_permission_under_an_unpermitted_state_is_refused(self):
        controller = LateralController()
        value, verdict = pair(1, offset=0.30)
        hostile = unvalidated(verdict, state=SafetyState.FAIL_SAFE)
        self.assert_refused(
            controller.update(value, hostile, now=1.0),
            LateralControlReason.SAFETY_NOT_PERMITTED,
        )


class SteeringSignTest(LateralTestCase):
    """The Mission 9 acceptance invariant: steering points at the lane centre.

    Positive steering means "toward increasing normalized image x", i.e. right.
    Mission 5 defines ``center_offset`` as ``lane_center_x - 0.5`` at row 0.90
    and ``heading_proxy`` as the image centreline slope with y increasing
    downward, so the lane centre at a row ``L`` further ahead sits at
    ``center_offset - L * heading_proxy`` relative to the image centre.
    """

    def steer(self, *, offset=0.0, heading=0.0, config=None):
        controller = LateralController(
            DEFAULT_LATERAL_CONTROLLER_CONFIG if config is None else config
        )
        return self.drive(controller, 1, now=1.0, offset=offset, heading=heading)

    def test_centered_lane_requests_no_steering(self):
        request = self.steer(offset=0.0, heading=0.0)
        self.assertTrue(request.lateral_allowed)
        self.assertEqual(request.steering, 0.0)
        self.assertEqual(request.raw_steering, 0.0)

    def test_lane_centre_to_the_right_steers_right(self):
        # center_offset > 0 means the lane centre lies right of the image
        # centre, so the ego is left of the lane and must steer right.
        request = self.steer(offset=0.10)
        self.assertGreater(request.steering, 0.0)

    def test_lane_centre_to_the_left_steers_left(self):
        request = self.steer(offset=-0.10)
        self.assertLess(request.steering, 0.0)

    def test_opposite_offsets_produce_opposite_steering(self):
        right = self.steer(offset=0.10)
        left = self.steer(offset=-0.10)
        self.assertEqual(right.steering, -left.steering)
        self.assertGreater(right.steering, 0.0)

    def test_lane_receding_to_the_right_steers_right(self):
        # heading_proxy < 0 means the centreline moves leftward toward the
        # bottom of the image, i.e. rightward as rows move up toward road that
        # is further ahead: the lane ahead lies right, so steer right.
        request = self.steer(heading=-0.20)
        self.assertGreater(request.steering, 0.0)

    def test_lane_receding_to_the_left_steers_left(self):
        request = self.steer(heading=0.20)
        self.assertLess(request.steering, 0.0)

    def test_opposite_headings_produce_opposite_steering(self):
        right = self.steer(heading=-0.20)
        left = self.steer(heading=0.20)
        self.assertEqual(right.steering, -left.steering)
        self.assertGreater(right.steering, 0.0)

    def test_agreeing_offset_and_heading_reinforce(self):
        offset_only = self.steer(offset=0.10)
        heading_only = self.steer(heading=-0.20)
        combined = self.steer(offset=0.10, heading=-0.20)
        self.assertGreater(combined.steering, offset_only.steering)
        self.assertGreater(combined.steering, heading_only.steering)
        self.assertAlmostEqual(
            combined.raw_steering,
            offset_only.raw_steering + heading_only.raw_steering,
            places=12,
        )

    def test_mirrored_geometry_produces_exactly_mirrored_steering(self):
        right = self.steer(offset=0.12, heading=-0.18)
        left = self.steer(offset=-0.12, heading=0.18)
        self.assertEqual(right.steering, -left.steering)

    def test_raw_steering_is_the_lookahead_lane_centre_displacement(self):
        config = LateralControllerConfig(offset_gain=1.0, heading_gain=0.25)
        offset, heading = 0.12, -0.18
        request = self.steer(offset=offset, heading=heading, config=config)
        lookahead = config.heading_gain / config.offset_gain
        # The Mission 5 centreline is a straight image-space line, so its
        # displacement from the image centre one lookahead further ahead is
        # exactly the requested raw steering divided by offset_gain.
        self.assertAlmostEqual(
            request.raw_steering / config.offset_gain,
            offset - lookahead * heading,
            places=12,
        )

    def test_reported_errors_are_the_temporal_scalars(self):
        request = self.steer(offset=0.09, heading=-0.07)
        self.assertEqual(request.lateral_error, 0.09)
        self.assertEqual(request.heading_error, -0.07)

    def test_deadband_suppresses_tiny_requests_symmetrically(self):
        config = LateralControllerConfig(offset_gain=1.0, steering_deadband=0.05)
        self.assertEqual(self.steer(offset=0.04, config=config).steering, 0.0)
        self.assertEqual(self.steer(offset=-0.04, config=config).steering, 0.0)
        self.assertGreater(self.steer(offset=0.06, config=config).steering, 0.0)
        self.assertLess(self.steer(offset=-0.06, config=config).steering, 0.0)

    def test_the_deadband_is_not_reported_as_an_authority_clamp(self):
        config = LateralControllerConfig(offset_gain=1.0, steering_deadband=0.05)
        controller = LateralController(config)
        suppressed = self.drive(controller, 1, now=1.0, offset=0.04)
        self.assertEqual(suppressed.steering, 0.0)
        self.assertEqual(suppressed.raw_steering, 0.04)
        self.assertFalse(suppressed.clamped)
        self.assertEqual(controller.metrics.clamp_activations, 0)


class AuthorityTest(LateralTestCase):
    def test_nominal_authority_is_the_configured_nominal_bound(self):
        controller = LateralController()
        request = self.drive(controller, 1, now=1.0, offset=1.0)
        self.assertIs(request.mode, LateralControlMode.NOMINAL)
        self.assertIs(request.reason, LateralControlReason.NOMINAL_TRACKING)
        self.assertEqual(
            request.steering_authority,
            DEFAULT_LATERAL_CONTROLLER_CONFIG.nominal_max_steering,
        )

    def test_positive_and_negative_clamps_are_symmetric(self):
        config = LateralControllerConfig(
            offset_gain=1.0, heading_gain=0.0, max_steering_rate_per_second=100.0
        )
        high = LateralController(config)
        low = LateralController(config)
        positive = self.drive(high, 1, now=1.0, offset=1.0)
        negative = self.drive(low, 1, now=1.0, offset=-1.0)
        self.assertEqual(positive.steering, config.nominal_max_steering)
        self.assertEqual(negative.steering, -config.nominal_max_steering)
        self.assertTrue(positive.clamped)
        self.assertTrue(negative.clamped)

    def test_the_exact_authority_boundary_is_not_a_clamp(self):
        config = LateralControllerConfig(
            offset_gain=1.0, heading_gain=0.0, nominal_max_steering=0.35
        )
        controller = LateralController(config)
        request = self.drive(controller, 1, now=1.0, offset=0.35)
        self.assertEqual(request.steering, 0.35)
        self.assertFalse(request.clamped)
        self.assertEqual(controller.metrics.clamp_activations, 0)

    def test_degraded_authority_is_strictly_lower_for_the_same_error(self):
        config = LateralControllerConfig(
            offset_gain=1.0,
            heading_gain=0.0,
            nominal_max_steering=0.40,
            degraded_max_steering=0.10,
            max_steering_rate_per_second=100.0,
        )
        nominal = LateralController(config)
        degraded = LateralController(config)
        strong = self.drive(nominal, 1, now=1.0, offset=0.90)
        weak = self.drive(
            degraded, 1, COASTING, now=1.0, state=SafetyState.DEGRADED, offset=0.90
        )
        self.assertIs(strong.mode, LateralControlMode.NOMINAL)
        self.assertIs(weak.mode, LateralControlMode.DEGRADED)
        self.assertLess(abs(weak.steering), abs(strong.steering))
        self.assertEqual(weak.steering_authority, config.degraded_max_steering)

    def test_degraded_never_exceeds_nominal_over_a_sweep_of_errors(self):
        config = LateralControllerConfig(max_steering_rate_per_second=100.0)
        for index in range(-10, 11):
            offset = index / 10.0
            with self.subTest(offset=offset):
                nominal = self.drive(
                    LateralController(config), 1, now=1.0, offset=offset
                )
                degraded = self.drive(
                    LateralController(config),
                    1,
                    COASTING,
                    now=1.0,
                    state=SafetyState.DEGRADED,
                    offset=offset,
                )
                self.assertLessEqual(
                    abs(degraded.steering), abs(nominal.steering) + 1e-12
                )

    def test_coasting_can_never_claim_nominal_authority(self):
        # Even if Mission 8 called extrapolated evidence nominal, Mission 9 may
        # only subtract authority, never add it.
        config = LateralControllerConfig(max_steering_rate_per_second=100.0)
        controller = LateralController(config)
        value = result(1, COASTING, offset=0.90)
        verdict = decision(value, state=SafetyState.NOMINAL)
        request = controller.update(value, verdict, now=1.0)
        self.assertIs(request.mode, LateralControlMode.DEGRADED)
        self.assertIs(request.reason, LateralControlReason.DEGRADED_COASTING)
        self.assertEqual(request.steering_authority, config.degraded_max_steering)

    def test_degraded_tracking_is_reported_separately_from_coasting(self):
        controller = LateralController()
        request = self.drive(
            controller, 1, DETECTED, now=1.0, state=SafetyState.DEGRADED, offset=0.05
        )
        self.assertIs(request.reason, LateralControlReason.DEGRADED_TRACKING)
        self.assertIs(request.temporal_state, TemporalLaneState.TRACKING)


class TemporalGeometryTest(LateralTestCase):
    def test_tracking_with_complete_geometry_steers(self):
        controller = LateralController()
        request = self.drive(controller, 1, DETECTED, now=1.0, offset=0.08)
        self.assertTrue(request.lateral_allowed)
        self.assertIs(request.temporal_state, TemporalLaneState.TRACKING)

    def test_partial_tracking_never_fabricates_the_missing_boundary(self):
        controller = LateralController()
        value = result(1, PARTIAL)
        verdict = decision(
            value,
            state=SafetyState.DEGRADED,
            reason=SafetyReason.DEGRADED_PARTIAL_TRACKING,
        )
        request = controller.update(value, verdict, now=1.0)
        self.assert_refused(request, LateralControlReason.INSUFFICIENT_GEOMETRY)
        self.assertEqual(controller.metrics.geometry_rejects, 1)

    def test_coasting_steers_from_temporal_rather_than_unusable_raw_geometry(self):
        controller = LateralController()
        value = result(1, COASTING, offset=0.20, heading=-0.10)
        # The raw Mission 5 frame detected nothing at all this tick.
        self.assertIs(value.observation.state, LaneDetectionState.NOT_DETECTED)
        self.assertIsNone(value.observation.center_offset)
        verdict = decision(value, state=SafetyState.DEGRADED)
        request = controller.update(value, verdict, now=1.0)
        self.assertTrue(request.lateral_allowed)
        self.assertEqual(request.lateral_error, 0.20)
        self.assertEqual(request.heading_error, -0.10)
        self.assertGreater(request.steering, 0.0)

    def test_temporal_geometry_wins_over_a_disagreeing_raw_frame(self):
        controller = LateralController()
        value = result(
            1, DETECTED, offset=0.20, heading=0.0, raw_offset=-0.20, raw_heading=0.0
        )
        self.assertEqual(value.observation.center_offset, -0.20)
        request = controller.update(value, decision(value), now=1.0)
        self.assertEqual(request.lateral_error, 0.20)
        self.assertGreater(request.steering, 0.0)

    def test_lost_and_unusable_states_cannot_steer(self):
        for kind in (LOST, UNUSABLE):
            with self.subTest(kind=kind):
                controller = LateralController()
                value = result(1, kind)
                # Force a permitting decision so only Mission 9's own geometry
                # gate can be responsible for the refusal.
                verdict = decision(value, state=SafetyState.DEGRADED)
                self.assert_refused(
                    controller.update(value, verdict, now=1.0),
                    LateralControlReason.INSUFFICIENT_GEOMETRY,
                )

    def test_uninitialized_temporal_evidence_cannot_steer(self):
        controller = LateralController()
        value = result(1, DETECTED, offset=0.20)
        estimate = unvalidated(
            value.observation.temporal_estimate,
            state=TemporalLaneState.UNINITIALIZED,
        )
        carrier = unvalidated(value.observation, temporal_estimate=estimate)
        hostile = wrap(value.observation, frame=1)
        hostile = unvalidated(hostile, observation=carrier)
        verdict = decision(
            value, state=SafetyState.DEGRADED, temporal_state=TemporalLaneState.UNINITIALIZED
        )
        self.assert_refused(
            controller.update(hostile, verdict, now=1.0),
            LateralControlReason.INSUFFICIENT_GEOMETRY,
        )

    def test_non_finite_temporal_geometry_is_rejected(self):
        for name in ("center_offset", "heading_proxy"):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(name=name, value=value):
                    controller = LateralController()
                    published = result(1, DETECTED, offset=0.10, heading=0.10)
                    estimate = unvalidated(
                        published.observation.temporal_estimate, **{name: value}
                    )
                    carrier = unvalidated(
                        published.observation, temporal_estimate=estimate
                    )
                    hostile = unvalidated(published, observation=carrier)
                    self.assert_refused(
                        controller.update(hostile, decision(published), now=1.0),
                        LateralControlReason.EVIDENCE_MALFORMED,
                    )

    def test_out_of_range_temporal_geometry_is_rejected(self):
        controller = LateralController()
        published = result(1, DETECTED, offset=0.10)
        estimate = unvalidated(
            published.observation.temporal_estimate, center_offset=4.0
        )
        carrier = unvalidated(published.observation, temporal_estimate=estimate)
        hostile = unvalidated(published, observation=carrier)
        self.assert_refused(
            controller.update(hostile, decision(published), now=1.0),
            LateralControlReason.EVIDENCE_MALFORMED,
        )

    def test_half_a_geometry_pair_is_malformed_not_an_assumption(self):
        controller = LateralController()
        published = result(1, DETECTED, offset=0.10, heading=0.10)
        estimate = unvalidated(
            published.observation.temporal_estimate, heading_proxy=None
        )
        carrier = unvalidated(published.observation, temporal_estimate=estimate)
        hostile = unvalidated(published, observation=carrier)
        self.assert_refused(
            controller.update(hostile, decision(published), now=1.0),
            LateralControlReason.EVIDENCE_MALFORMED,
        )

    def test_evidence_that_raises_while_being_read_fails_closed(self):
        for name, exception in (
            ("observation", AttributeError("no observation")),
            ("source_stamp", TypeError("no stamp")),
            ("snapshot_revision", RuntimeError("no revision")),
        ):
            with self.subTest(field=name):
                controller = LateralController()
                published = result(1, DETECTED, offset=0.30)
                hostile = exploding(published, name, exception)
                self.assert_refused(
                    controller.update(hostile, decision(published), now=1.0),
                    LateralControlReason.EVIDENCE_MALFORMED,
                )

    def test_a_decision_that_raises_while_being_read_fails_closed(self):
        for name in ("snapshot_revision", "temporal_confidence", "temporal_state"):
            with self.subTest(field=name):
                controller = LateralController()
                value, verdict = pair(1, offset=0.30)
                hostile = exploding(verdict, name, RuntimeError("no evidence"))
                self.assert_refused(
                    controller.update(value, hostile, now=1.0),
                    LateralControlReason.EVIDENCE_MALFORMED,
                )

    def test_a_plain_mission_5_observation_is_malformed_evidence(self):
        controller = LateralController()
        published = result(1, DETECTED, offset=0.10)
        raw = published.observation
        plain = unvalidated(raw)
        # Strip the Mission 7 carrier identity by presenting a bare observation.
        object.__setattr__(plain, "temporal_estimate", None)
        hostile = unvalidated(published, observation=plain)
        self.assert_refused(
            controller.update(hostile, decision(published), now=1.0),
            LateralControlReason.EVIDENCE_MALFORMED,
        )


class IdentityBindingTest(LateralTestCase):
    def test_an_exactly_matching_pair_is_accepted(self):
        controller = LateralController()
        value, verdict = pair(7, offset=0.05)
        request = controller.update(value, verdict, now=1.0)
        self.assertTrue(request.lateral_allowed)
        self.assertEqual(request.source_frame, 7)
        self.assertEqual(request.snapshot_revision, 7)

    def test_a_decision_for_another_frame_cannot_authorize_this_one(self):
        controller = LateralController()
        value = result(9, DETECTED, offset=0.30)
        other = result(8, DETECTED, offset=0.30)
        request = controller.update(value, decision(other), now=1.0)
        self.assert_refused(request, LateralControlReason.IDENTITY_MISMATCH)
        self.assertEqual(controller.metrics.identity_rejects, 1)

    def test_each_decision_identity_field_is_bound(self):
        base = result(9, DETECTED, offset=0.30)
        stamp = base.source_stamp
        cases = {
            "frame": dict(
                stamp=SampleStamp(
                    frame=stamp.frame + 1,
                    sim_time=stamp.sim_time,
                    monotonic=stamp.monotonic,
                )
            ),
            "sim_time": dict(
                stamp=SampleStamp(
                    frame=stamp.frame,
                    sim_time=stamp.sim_time + 1.0,
                    monotonic=stamp.monotonic,
                )
            ),
            "monotonic": dict(
                stamp=SampleStamp(
                    frame=stamp.frame,
                    sim_time=stamp.sim_time,
                    monotonic=stamp.monotonic + 1.0,
                )
            ),
            "revision": dict(revision=base.snapshot_revision + 1),
            "temporal_state": dict(temporal_state=TemporalLaneState.COASTING),
            "confidence": dict(confidence=0.5),
        }
        for name, kwargs in cases.items():
            with self.subTest(field=name):
                controller = LateralController()
                request = controller.update(
                    base, decision(base, **kwargs), now=1.0
                )
                self.assert_refused(request, LateralControlReason.IDENTITY_MISMATCH)

    def test_a_permitting_decision_without_evidence_is_refused(self):
        controller = LateralController()
        value, verdict = pair(3, offset=0.20)
        for name in ("source_stamp", "snapshot_revision", "temporal_confidence"):
            with self.subTest(field=name):
                hostile = unvalidated(verdict, **{name: None})
                self.assert_refused(
                    LateralController().update(value, hostile, now=1.0),
                    LateralControlReason.IDENTITY_MISMATCH,
                )
        self.assertTrue(controller.update(value, verdict, now=1.0).lateral_allowed)

    def test_carrier_revision_must_match_the_published_result(self):
        controller = LateralController()
        carrier = build_carrier(4, DETECTED, offset=0.20)
        value = wrap(carrier, revision=carrier.snapshot_revision + 1)
        self.assert_refused(
            controller.update(value, decision(value), now=1.0),
            LateralControlReason.IDENTITY_MISMATCH,
        )

    def test_carrier_stamp_must_match_the_published_result(self):
        controller = LateralController()
        carrier = build_carrier(4, DETECTED, offset=0.20)
        value = wrap(carrier, frame=99)
        self.assert_refused(
            controller.update(value, decision(value), now=1.0),
            LateralControlReason.IDENTITY_MISMATCH,
        )

    def test_estimate_and_carrier_must_agree_on_the_raw_state(self):
        controller = LateralController()
        published = result(4, DETECTED, offset=0.20)
        estimate = unvalidated(
            published.observation.temporal_estimate,
            raw_state=LaneDetectionState.PARTIAL,
        )
        carrier = unvalidated(published.observation, temporal_estimate=estimate)
        hostile = unvalidated(published, observation=carrier)
        self.assert_refused(
            controller.update(hostile, decision(published), now=1.0),
            LateralControlReason.IDENTITY_MISMATCH,
        )

    def test_the_published_stamp_is_a_detached_exact_sample_stamp(self):
        controller = LateralController()
        value, verdict = pair(5, offset=0.10)
        request = controller.update(value, verdict, now=1.0)
        self.assertIs(type(request.source_stamp), SampleStamp)
        self.assertIsNot(request.source_stamp, value.source_stamp)
        self.assertIsNot(
            controller.metrics.last_accepted_source_stamp,
            controller.metrics.last_accepted_source_stamp,
        )


class SourceOrderTest(LateralTestCase):
    def test_a_duplicate_pair_is_a_legitimate_re_evaluation(self):
        controller = LateralController()
        value, verdict = pair(4, offset=0.02)
        first = controller.update(value, verdict, now=1.0)
        second = controller.update(value, verdict, now=1.5)
        self.assertTrue(second.lateral_allowed)
        self.assertEqual(second.steering, first.steering)
        self.assertEqual(controller.metrics.source_rejects, 0)

    def test_an_older_frame_is_refused_without_moving_the_baseline(self):
        controller = LateralController()
        controller.update(*pair(5, offset=0.02), now=1.0)
        older_value, older_verdict = pair(4, offset=0.30)
        self.assert_refused(
            controller.update(older_value, older_verdict, now=1.1),
            LateralControlReason.SOURCE_REJECTED,
        )
        self.assertEqual(controller.metrics.last_accepted_source_stamp.frame, 5)
        # The retained baseline is still frame 5, so frame 6 remains acceptable.
        self.assertTrue(controller.update(*pair(6, offset=0.02), now=1.2).lateral_allowed)

    def test_an_equal_frame_with_a_different_sim_time_is_refused(self):
        controller = LateralController()
        controller.update(*pair(5, offset=0.02), now=1.0)
        value = result(5, DETECTED, offset=0.02, sim_time=9.0)
        self.assert_refused(
            controller.update(value, decision(value), now=1.1),
            LateralControlReason.SOURCE_REJECTED,
        )

    def test_a_higher_frame_with_an_equal_sim_time_is_accepted(self):
        controller = LateralController()
        first = result(5, DETECTED, offset=0.02, sim_time=0.5)
        controller.update(first, decision(first), now=1.0)
        second = result(6, DETECTED, offset=0.02, sim_time=0.5)
        self.assertTrue(
            controller.update(second, decision(second), now=1.1).lateral_allowed
        )

    def test_a_regressed_sim_time_is_refused(self):
        controller = LateralController()
        first = result(5, DETECTED, offset=0.02, sim_time=0.5)
        controller.update(first, decision(first), now=1.0)
        second = result(6, DETECTED, offset=0.02, sim_time=0.4)
        self.assert_refused(
            controller.update(second, decision(second), now=1.1),
            LateralControlReason.SOURCE_REJECTED,
        )

    def test_a_refused_replay_publishes_neutral_steering(self):
        controller = LateralController()
        controller.update(*pair(5, offset=0.30), now=1.0)
        request = controller.update(*pair(1, offset=0.30), now=1.1)
        self.assertEqual(request.steering, 0.0)
        self.assertEqual(controller.metrics.source_rejects, 1)


class RateLimitTest(LateralTestCase):
    CONFIG = LateralControllerConfig(
        offset_gain=1.0,
        heading_gain=0.0,
        nominal_max_steering=1.0,
        degraded_max_steering=0.50,
        max_steering_rate_per_second=1.0,
    )

    def test_the_first_command_is_bounded_by_authority_only(self):
        controller = LateralController(self.CONFIG)
        request = self.drive(controller, 1, now=100.0, offset=0.80)
        self.assertEqual(request.steering, 0.80)
        self.assertFalse(request.rate_limited)

    def test_an_unchanged_target_is_never_rate_limited(self):
        controller = LateralController(self.CONFIG)
        self.drive(controller, 1, now=0.0, offset=0.25)
        request = self.drive(controller, 2, now=0.25, offset=0.25)
        self.assertEqual(request.steering, 0.25)
        self.assertFalse(request.rate_limited)

    def test_positive_slew_is_bounded(self):
        controller = LateralController(self.CONFIG)
        self.drive(controller, 1, now=0.0, offset=0.0)
        request = self.drive(controller, 2, now=0.25, offset=0.90)
        self.assertEqual(request.steering, 0.25)
        self.assertTrue(request.rate_limited)
        self.assertEqual(controller.metrics.rate_limit_activations, 1)

    def test_negative_slew_is_bounded(self):
        controller = LateralController(self.CONFIG)
        self.drive(controller, 1, now=0.0, offset=0.0)
        request = self.drive(controller, 2, now=0.25, offset=-0.90)
        self.assertEqual(request.steering, -0.25)
        self.assertTrue(request.rate_limited)

    def test_the_exact_rate_boundary_is_not_a_limit(self):
        controller = LateralController(self.CONFIG)
        self.drive(controller, 1, now=0.0, offset=0.0)
        request = self.drive(controller, 2, now=0.25, offset=0.25)
        self.assertEqual(request.steering, 0.25)
        self.assertFalse(request.rate_limited)
        self.assertEqual(controller.metrics.rate_limit_activations, 0)

    def test_zero_dt_permits_no_change_at_all(self):
        controller = LateralController(self.CONFIG)
        self.drive(controller, 1, now=5.0, offset=0.10)
        request = self.drive(controller, 2, now=5.0, offset=0.90)
        self.assertEqual(request.steering, 0.10)
        self.assertTrue(request.rate_limited)

    def test_a_tiny_dt_permits_only_a_tiny_change(self):
        controller = LateralController(self.CONFIG)
        self.drive(controller, 1, now=0.0, offset=0.0)
        request = self.drive(controller, 2, now=0.001, offset=1.0)
        self.assertAlmostEqual(request.steering, 0.001, places=12)
        self.assertTrue(request.rate_limited)

    def test_a_long_dt_still_reaches_but_never_exceeds_the_target(self):
        controller = LateralController(self.CONFIG)
        self.drive(controller, 1, now=0.0, offset=0.0)
        request = self.drive(controller, 2, now=60.0, offset=0.90)
        self.assertEqual(request.steering, 0.90)
        self.assertFalse(request.rate_limited)

    def test_alternating_extremes_never_escape_the_envelope(self):
        controller = LateralController(self.CONFIG)
        previous = self.drive(controller, 1, now=0.1, offset=0.0).steering
        self.assertEqual(previous, 0.0)
        for index in range(2, 62):
            now = index / 10.0
            offset = 1.0 if index % 2 else -1.0
            request = self.drive(controller, index, now=now, offset=offset)
            self.assertLessEqual(
                abs(request.steering), request.steering_authority + 1e-12
            )
            self.assertLessEqual(
                abs(request.steering - previous),
                self.CONFIG.max_steering_rate_per_second * 0.1 + 1e-12,
            )
            previous = request.steering

    def test_reset_clears_the_command_and_slew_history(self):
        controller = LateralController(self.CONFIG)
        self.drive(controller, 1, now=0.0, offset=0.30)
        self.assertEqual(controller.latest_request.steering, 0.30)
        cleared = controller.reset()
        self.assert_refused(cleared, LateralControlReason.NOT_EVALUATED)
        self.assertIsNone(controller.metrics.last_accepted_source_stamp)
        # Frame 1 is acceptable again and behaves as a first command.
        restarted = self.drive(controller, 1, now=100.0, offset=0.90)
        self.assertEqual(restarted.steering, 0.90)
        self.assertFalse(restarted.rate_limited)

    def test_reset_is_idempotent(self):
        controller = LateralController(self.CONFIG)
        self.drive(controller, 1, now=0.0, offset=0.30)
        first = controller.reset()
        second = controller.reset()
        self.assertEqual(first, second)
        self.assertEqual(controller.metrics.resets, 2)

    def test_regressed_control_time_fails_closed_and_drops_the_baseline(self):
        controller = LateralController(self.CONFIG)
        self.drive(controller, 1, now=10.0, offset=0.30)
        request = self.drive(controller, 2, now=9.0, offset=0.30)
        self.assert_refused(request, LateralControlReason.TIME_INVALID)
        self.assertIsNone(request.evaluation_monotonic)
        self.assertEqual(controller.metrics.time_rejects, 1)
        # The clock is untrusted, so the next command starts a fresh epoch.
        recovered = self.drive(controller, 3, now=20.0, offset=0.90)
        self.assertEqual(recovered.steering, 0.90)
        self.assertFalse(recovered.rate_limited)

    def test_non_finite_control_time_fails_closed(self):
        for value in (math.nan, math.inf, -math.inf, "soon", True):
            with self.subTest(value=value):
                controller = LateralController(self.CONFIG)
                self.assert_refused(
                    self.drive(controller, 1, now=value, offset=0.30),
                    LateralControlReason.TIME_INVALID,
                )

    def test_a_refusal_zeros_immediately_and_re_enables_from_neutral(self):
        controller = LateralController(self.CONFIG)
        self.drive(controller, 1, now=0.0, offset=0.90)
        self.assertEqual(controller.latest_request.steering, 0.90)
        refused = self.drive(
            controller, 2, now=0.1, state=SafetyState.FAIL_SAFE, offset=0.90
        )
        self.assertEqual(refused.steering, 0.0)
        # Re-enabling ramps from the published zero, not from the stale 0.90.
        resumed = self.drive(controller, 3, now=0.2, offset=0.90)
        self.assertAlmostEqual(resumed.steering, 0.10, places=12)
        self.assertTrue(resumed.rate_limited)

    def test_an_authority_downgrade_reduces_magnitude_immediately(self):
        controller = LateralController(self.CONFIG)
        self.drive(controller, 1, now=0.0, offset=0.90)
        self.assertEqual(controller.latest_request.steering, 0.90)
        request = self.drive(
            controller,
            2,
            COASTING,
            now=0.001,
            state=SafetyState.DEGRADED,
            offset=0.90,
        )
        self.assertEqual(request.steering_authority, self.CONFIG.degraded_max_steering)
        self.assertLessEqual(
            abs(request.steering), self.CONFIG.degraded_max_steering
        )
        self.assertLess(abs(request.steering), 0.90)

    def test_steering_stays_inside_authority_across_a_long_mixed_stream(self):
        controller = LateralController(self.CONFIG)
        states = (SafetyState.NOMINAL, SafetyState.DEGRADED, SafetyState.FAIL_SAFE)
        for index in range(1, 300):
            state = states[index % len(states)]
            kind = COASTING if state is SafetyState.DEGRADED else DETECTED
            request = self.drive(
                controller,
                index,
                kind,
                now=index / 20.0,
                state=state,
                offset=1.0 if index % 5 else -1.0,
            )
            self.assertLessEqual(abs(request.steering), 1.0)
            self.assertLessEqual(
                abs(request.steering), request.steering_authority + 1e-12
            )


class DeterminismAndBoundednessTest(LateralTestCase):
    SCRIPT = (
        (1, DETECTED, SafetyState.NOMINAL, 0.30, -0.10),
        (2, DETECTED, SafetyState.NOMINAL, -0.40, 0.20),
        (3, COASTING, SafetyState.DEGRADED, 0.90, -0.90),
        (4, DETECTED, SafetyState.FAIL_SAFE, 0.10, 0.10),
        (5, PARTIAL, SafetyState.DEGRADED, 0.0, 0.0),
        (6, DETECTED, SafetyState.NOMINAL, 0.05, -0.05),
    )

    def run_script(self, controller):
        requests = []
        for frame, kind, state, offset, heading in self.SCRIPT:
            value = result(frame, kind, offset=offset, heading=heading)
            verdict = decision(
                value,
                state=state,
                reason=(
                    SafetyReason.DEGRADED_PARTIAL_TRACKING
                    if kind == PARTIAL
                    else None
                ),
            )
            requests.append(controller.update(value, verdict, now=frame / 10.0))
        return tuple(requests)

    def test_identical_streams_produce_identical_requests(self):
        first = self.run_script(LateralController())
        second = self.run_script(LateralController())
        self.assertEqual(first, second)

    def test_independent_controllers_stay_equivalent(self):
        left = LateralController()
        right = LateralController()
        for frame, kind, state, offset, heading in self.SCRIPT:
            value = result(frame, kind, offset=offset, heading=heading)
            verdict = decision(
                value,
                state=state,
                reason=(
                    SafetyReason.DEGRADED_PARTIAL_TRACKING
                    if kind == PARTIAL
                    else None
                ),
            )
            self.assertEqual(
                left.update(value, verdict, now=frame / 10.0),
                right.update(value, verdict, now=frame / 10.0),
            )
        self.assertEqual(left.metrics, right.metrics)

    def test_retained_state_is_o1_over_a_long_stream(self):
        controller = LateralController()
        self.assertFalse(hasattr(controller, "__dict__"))

        def footprint():
            return tuple(
                (name, type(getattr(controller, name)).__name__)
                for name in LateralController.__slots__
            )

        self.drive(controller, 1, now=0.1, offset=0.10)
        before = footprint()
        for index in range(2, 2002):
            self.drive(controller, index, now=index / 10.0, offset=0.10)
        self.assertEqual(footprint(), before)
        self.assertEqual(controller.metrics.evaluations, 2001)
        self.assertEqual(controller.metrics.enabled_requests, 2001)

    def test_caller_evidence_is_not_retained(self):
        controller = LateralController()
        value, verdict = pair(1, offset=0.10)
        request = controller.update(value, verdict, now=1.0)
        retained = tuple(
            getattr(controller, name) for name in LateralController.__slots__
        )
        for item in retained:
            self.assertIsNot(item, value)
            self.assertIsNot(item, verdict)
            self.assertIsNot(item, value.observation)
            self.assertIsNot(item, value.observation.temporal_estimate)
            self.assertIsNot(item, value.source_stamp)
        self.assertIsNot(request.source_stamp, value.source_stamp)

    def test_the_published_request_is_frozen(self):
        controller = LateralController()
        request = self.drive(controller, 1, now=1.0, offset=0.10)
        with self.assertRaises(FrozenInstanceError):
            request.steering = 0.9

    def test_metrics_account_for_every_disposition(self):
        controller = LateralController()
        self.drive(controller, 1, now=0.1, offset=0.10)
        self.drive(controller, 2, COASTING, now=0.2, state=SafetyState.DEGRADED, offset=0.10)
        self.drive(controller, 3, now=0.3, state=SafetyState.FAIL_SAFE)
        self.drive(controller, 1, now=0.4, offset=0.10)
        self.drive(controller, 4, now=0.0, offset=0.10)
        metrics = controller.metrics
        self.assertEqual(metrics.evaluations, 5)
        self.assertEqual(metrics.enabled_requests, 2)
        self.assertEqual(metrics.disabled_requests, 3)
        self.assertEqual(metrics.nominal_requests, 1)
        self.assertEqual(metrics.degraded_requests, 1)
        self.assertEqual(metrics.safety_rejects, 1)
        self.assertEqual(metrics.source_rejects, 1)
        self.assertEqual(metrics.time_rejects, 1)
        self.assertEqual(metrics.resets, 0)


class ControlScopeTest(LateralTestCase):
    def test_the_module_declares_no_actuator_or_worker_surface(self):
        """No identifier in the module names an actuator, worker, or CARLA API.

        Only identifiers are inspected, never prose, so the module docstring may
        state truthfully what Mission 9 does *not* do without tripping this.
        """
        import ast
        import control.lateral as module

        with open(module.__file__, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        identifiers: set[str] = set()
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                identifiers.add(node.name)
            elif isinstance(node, ast.arg):
                identifiers.add(node.arg)
            elif isinstance(node, ast.keyword) and node.arg is not None:
                identifiers.add(node.arg)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    imported.add(node.module.split(".")[0])
        # Nothing outside this allowlist is imported at all, so no worker,
        # queue, timer, socket, subprocess, or CARLA API can exist here.
        self.assertEqual(
            imported,
            {
                "__future__",
                "dataclasses",
                "enum",
                "math",
                "numbers",
                "perception",
                "safety",
                "telemetry",
            },
        )
        lowered = {name.lower() for name in identifiers}
        for forbidden in (
            "carla",
            "vehiclecontrol",
            "apply_control",
            "throttle",
            "brake",
            "autopilot",
            "numpy",
            "pygame",
            "scipy",
        ):
            matches = sorted(name for name in lowered if forbidden in name)
            self.assertEqual(matches, [], f"{forbidden}: {matches}")

    def test_no_public_name_exposes_a_longitudinal_command(self):
        import control

        for name in control.__all__:
            lowered = name.lower()
            for forbidden in ("throttle", "brake", "speed", "accel"):
                self.assertNotIn(forbidden, lowered, name)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    unittest.main()
