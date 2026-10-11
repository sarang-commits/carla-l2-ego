"""Deterministic unit tests for the Mission 8 safety supervisor.

Every input here is an immutable synthetic record built from the accepted
Mission 5/6/7 public types.  No CARLA package, server, socket, worker, or wall
clock is involved: safety time is always supplied explicitly.
"""

from __future__ import annotations

import math
import numbers
import threading
import unittest
from dataclasses import FrozenInstanceError

from perception.live_bridge import PerceptionLiveBridgeState
from perception.runtime import PerceptionRuntimeResult, PerceptionRuntimeState
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
    TemporalLaneState,
)
from safety.supervisor import (
    DEFAULT_SAFETY_SUPERVISOR_CONFIG,
    MAX_REQUIRED_HEALTHY_RESULTS,
    MAX_SAFETY_RESULT_AGE_SECONDS,
    ComponentHealth,
    SafetyDecision,
    SafetyReason,
    SafetyState,
    SafetySupervisor,
    SafetySupervisorConfig,
    _temporal_disposition,
)
from telemetry.state import SampleStamp


DETECTED = "detected"
PARTIAL = "partial"
COASTING = "coasting"
LOST = "lost"
UNUSABLE = "unusable"

RUNNING = ComponentHealth(runtime_state=PerceptionRuntimeState.RUNNING)


@numbers.Real.register
class DriftingReal:
    """A hostile ``numbers.Real`` whose float value changes between reads."""

    def __init__(self, *values: float) -> None:
        self._values = list(values)

    def __float__(self) -> float:
        if len(self._values) > 1:
            return float(self._values.pop(0))
        return float(self._values[0])


class StampSubclass(SampleStamp):
    """A caller-owned stamp subtype that must never be retained as-is."""


def raw_boundary(
    side: LaneBoundarySide, confidence: float = 0.8
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
    kind: str,
    *,
    sim_time: float | None = None,
    monotonic: float | None = None,
    revision: int | None = None,
    confidence: float = 0.8,
    stamped: bool = True,
) -> TemporalLaneObservation:
    """Build one exact Mission 7 carrier for the requested temporal state."""
    selected_sim = frame / 10.0 if sim_time is None else sim_time
    selected_monotonic = frame / 10.0 if monotonic is None else monotonic
    selected_revision = frame if revision is None else revision
    stamp = SampleStamp(
        frame=frame, sim_time=selected_sim, monotonic=selected_monotonic
    )
    source = dict(
        source_frame=frame if stamped else None,
        source_simulation_timestamp=selected_sim if stamped else None,
        source_receive_monotonic=selected_monotonic if stamped else None,
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
            center_offset=0.0,
            heading_proxy=0.0,
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
            center_offset=0.0,
            heading_proxy=0.0,
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
            center_offset=0.0,
            heading_proxy=0.0,
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
        source_stamp=None if (kind == UNUSABLE and not stamped) else stamp,
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
    carrier: LaneObservation,
    *,
    frame: int | None = None,
    sim_time: float | None = None,
    monotonic: float | None = None,
    revision: int | None = None,
) -> PerceptionRuntimeResult:
    """Wrap one carrier in a runtime result, optionally with a foreign identity."""
    selected_frame = carrier.source_frame if frame is None else frame
    selected_sim = (
        carrier.source_simulation_timestamp if sim_time is None else sim_time
    )
    selected_monotonic = (
        carrier.source_receive_monotonic if monotonic is None else monotonic
    )
    selected_revision = carrier.snapshot_revision if revision is None else revision
    stamp = SampleStamp(
        frame=selected_frame,
        sim_time=selected_sim,
        monotonic=selected_monotonic,
    )
    return PerceptionRuntimeResult(
        observation=carrier,
        source_stamp=stamp,
        source_sequence=None,
        snapshot_revision=selected_revision,
        ingestion_monotonic=selected_monotonic,
        processing_start_monotonic=selected_monotonic,
        processing_finish_monotonic=selected_monotonic,
        pending_age_seconds=0.0,
        processing_latency_seconds=0.0,
    )


def result(
    frame: int = 1,
    kind: str = DETECTED,
    *,
    sim_time: float | None = None,
    monotonic: float | None = None,
    revision: int | None = None,
    **kwargs,
) -> PerceptionRuntimeResult:
    """Build one fully coherent runtime result for the requested state.

    The outer runtime identity is always supplied explicitly so that a
    deliberately unstamped ``INPUT_UNUSABLE`` carrier still gets the canonical
    camera stamp the Mission 7 pipeline would have recovered from its snapshot.
    """
    carrier = build_carrier(
        frame,
        kind,
        sim_time=sim_time,
        monotonic=monotonic,
        revision=revision,
        **kwargs,
    )
    return wrap(
        carrier,
        frame=frame,
        sim_time=frame / 10.0 if sim_time is None else sim_time,
        monotonic=frame / 10.0 if monotonic is None else monotonic,
        revision=frame if revision is None else revision,
    )


class SafetyStartupTest(unittest.TestCase):
    def test_construction_starts_initializing_without_autonomy(self):
        supervisor = SafetySupervisor()
        decision = supervisor.latest_decision
        self.assertIs(decision.state, SafetyState.INITIALIZING)
        self.assertIs(decision.reason, SafetyReason.STARTUP_NO_RESULT)
        self.assertFalse(decision.autonomy_allowed)
        self.assertFalse(decision.latched)
        self.assertIsNone(decision.evaluation_monotonic)
        self.assertIsNone(decision.source_stamp)
        self.assertIsNone(decision.result_age_seconds)
        self.assertIsNone(decision.temporal_state)
        self.assertEqual(supervisor.metrics.evaluations, 0)
        self.assertEqual(supervisor.config, DEFAULT_SAFETY_SUPERVISOR_CONFIG)

    def test_tick_without_result_remains_initializing(self):
        supervisor = SafetySupervisor()
        for index, now in enumerate((0.0, 1.0, 900.0), 1):
            decision = supervisor.evaluate(now=now, health=RUNNING)
            self.assertIs(decision.state, SafetyState.INITIALIZING)
            self.assertFalse(decision.autonomy_allowed)
            self.assertEqual(decision.evaluation_monotonic, now)
            self.assertEqual(supervisor.metrics.evaluations, index)
        metrics = supervisor.metrics
        self.assertEqual(metrics.initializing_decisions, 3)
        self.assertEqual(metrics.updates_received, 0)
        self.assertIsNone(metrics.last_accepted_source_stamp)

    def test_reset_is_idempotent_and_restores_startup(self):
        supervisor = SafetySupervisor()
        supervisor.update(result(1), now=0.1, health=RUNNING)
        self.assertTrue(supervisor.autonomy_allowed)
        first = supervisor.reset()
        second = supervisor.reset()
        self.assertEqual(first, second)
        self.assertIs(first.state, SafetyState.INITIALIZING)
        self.assertFalse(supervisor.autonomy_allowed)
        metrics = supervisor.metrics
        self.assertEqual(metrics.resets, 2)
        self.assertIsNone(metrics.last_accepted_source_stamp)
        # Lifetime accounting stays truthful across a reset.
        self.assertEqual(metrics.new_results_accepted, 1)
        # A new epoch has no ordering baseline, so the same frame is accepted.
        again = supervisor.update(result(1), now=0.1, health=RUNNING)
        self.assertIs(again.state, SafetyState.NOMINAL)
        self.assertEqual(supervisor.metrics.duplicates_rejected, 0)


class SafetyNominalTest(unittest.TestCase):
    def test_fresh_detected_tracking_permits_autonomy(self):
        supervisor = SafetySupervisor()
        decision = supervisor.update(result(1, DETECTED), now=0.15, health=RUNNING)
        self.assertIs(decision.state, SafetyState.NOMINAL)
        self.assertIs(decision.reason, SafetyReason.NOMINAL_TRACKING)
        self.assertTrue(decision.autonomy_allowed)
        self.assertFalse(decision.latched)
        self.assertIs(decision.temporal_state, TemporalLaneState.TRACKING)
        self.assertEqual(decision.temporal_confidence, 0.8)
        self.assertEqual(decision.source_stamp, SampleStamp(1, 0.1, 0.1))
        self.assertEqual(decision.snapshot_revision, 1)
        self.assertAlmostEqual(decision.result_age_seconds, 0.05)
        self.assertEqual(decision.recovery_streak, 0)
        metrics = supervisor.metrics
        self.assertEqual(metrics.new_results_accepted, 1)
        self.assertEqual(metrics.nominal_decisions, 1)
        self.assertEqual(metrics.last_accepted_source_stamp, SampleStamp(1, 0.1, 0.1))

    def test_exact_nominal_confidence_threshold_permits_autonomy(self):
        threshold = DEFAULT_SAFETY_SUPERVISOR_CONFIG.min_nominal_confidence
        supervisor = SafetySupervisor()
        decision = supervisor.update(
            result(1, DETECTED, confidence=threshold), now=0.1, health=RUNNING
        )
        self.assertIs(decision.state, SafetyState.NOMINAL)
        self.assertTrue(decision.autonomy_allowed)
        self.assertEqual(decision.temporal_confidence, threshold)

    def test_confidence_below_nominal_threshold_degrades_and_above_permits(self):
        threshold = DEFAULT_SAFETY_SUPERVISOR_CONFIG.min_nominal_confidence
        below = math.nextafter(threshold, 0.0)
        above = math.nextafter(threshold, 1.0)

        lowered = SafetySupervisor()
        low = lowered.update(
            result(1, DETECTED, confidence=below), now=0.1, health=RUNNING
        )
        self.assertIs(low.state, SafetyState.DEGRADED)
        self.assertIs(low.reason, SafetyReason.DEGRADED_LOW_CONFIDENCE)
        self.assertTrue(low.autonomy_allowed)

        raised = SafetySupervisor()
        high = raised.update(
            result(1, DETECTED, confidence=above), now=0.1, health=RUNNING
        )
        self.assertIs(high.state, SafetyState.NOMINAL)
        self.assertTrue(high.autonomy_allowed)


class SafetyDegradedTest(unittest.TestCase):
    def test_fresh_partial_tracking_is_degraded(self):
        supervisor = SafetySupervisor()
        decision = supervisor.update(result(1, PARTIAL), now=0.1, health=RUNNING)
        self.assertIs(decision.state, SafetyState.DEGRADED)
        self.assertIs(decision.reason, SafetyReason.DEGRADED_PARTIAL_TRACKING)
        self.assertTrue(decision.autonomy_allowed)
        self.assertIs(decision.temporal_state, TemporalLaneState.TRACKING)
        self.assertEqual(supervisor.metrics.degraded_decisions, 1)

    def test_fresh_coasting_is_degraded(self):
        supervisor = SafetySupervisor()
        decision = supervisor.update(result(1, COASTING), now=0.1, health=RUNNING)
        self.assertIs(decision.state, SafetyState.DEGRADED)
        self.assertIs(decision.reason, SafetyReason.DEGRADED_COASTING)
        self.assertTrue(decision.autonomy_allowed)
        self.assertIs(decision.temporal_state, TemporalLaneState.COASTING)

    def test_exact_degraded_confidence_threshold_is_degraded(self):
        threshold = DEFAULT_SAFETY_SUPERVISOR_CONFIG.min_degraded_confidence
        for kind, reason in (
            (PARTIAL, SafetyReason.DEGRADED_PARTIAL_TRACKING),
            (COASTING, SafetyReason.DEGRADED_COASTING),
        ):
            with self.subTest(kind=kind):
                supervisor = SafetySupervisor()
                decision = supervisor.update(
                    result(1, kind, confidence=threshold), now=0.1, health=RUNNING
                )
                self.assertIs(decision.reason, reason)
                self.assertTrue(decision.autonomy_allowed)

    def test_partial_and_coasting_below_degraded_threshold_fail_closed(self):
        threshold = DEFAULT_SAFETY_SUPERVISOR_CONFIG.min_degraded_confidence
        below = math.nextafter(threshold, 0.0)
        for kind in (DETECTED, PARTIAL, COASTING):
            with self.subTest(kind=kind):
                supervisor = SafetySupervisor()
                decision = supervisor.update(
                    result(1, kind, confidence=below), now=0.1, health=RUNNING
                )
                self.assertIs(decision.state, SafetyState.FAIL_SAFE)
                self.assertIs(
                    decision.reason, SafetyReason.CONFIDENCE_BELOW_THRESHOLD
                )
                self.assertFalse(decision.autonomy_allowed)
                self.assertFalse(decision.latched)


class SafetyTemporalPolicyTest(unittest.TestCase):
    def test_temporal_lost_fails_closed(self):
        supervisor = SafetySupervisor()
        supervisor.update(result(1, DETECTED), now=0.1, health=RUNNING)
        decision = supervisor.update(result(2, LOST), now=0.2, health=RUNNING)
        self.assertIs(decision.state, SafetyState.FAIL_SAFE)
        self.assertIs(decision.reason, SafetyReason.TEMPORAL_LOST)
        self.assertFalse(decision.autonomy_allowed)
        self.assertFalse(decision.latched)
        self.assertIs(decision.temporal_state, TemporalLaneState.LOST)

    def test_temporal_input_unusable_fails_closed(self):
        supervisor = SafetySupervisor()
        supervisor.update(result(1, DETECTED), now=0.1, health=RUNNING)
        decision = supervisor.update(result(2, UNUSABLE), now=0.2, health=RUNNING)
        self.assertIs(decision.state, SafetyState.FAIL_SAFE)
        self.assertIs(decision.reason, SafetyReason.TEMPORAL_INPUT_UNUSABLE)
        self.assertFalse(decision.autonomy_allowed)
        self.assertFalse(decision.latched)

    def test_unstamped_unusable_carrier_fails_closed_without_latching(self):
        supervisor = SafetySupervisor()
        decision = supervisor.update(
            result(2, UNUSABLE, stamped=False), now=0.2, health=RUNNING
        )
        self.assertIs(decision.reason, SafetyReason.TEMPORAL_INPUT_UNUSABLE)
        self.assertFalse(decision.autonomy_allowed)
        self.assertEqual(supervisor.metrics.identity_mismatch_trips, 0)

    def test_uninitialized_temporal_state_is_not_permitted(self):
        # Mission 7 structurally refuses to publish a carrier around an
        # UNINITIALIZED estimate, so the supervisor branch is defence in depth.
        uninitialized = TemporalLaneEstimate(
            source_stamp=None,
            snapshot_revision=None,
            state=TemporalLaneState.UNINITIALIZED,
            raw_state=None,
            raw_reason=None,
        )
        with self.assertRaises(ValueError):
            TemporalLaneObservation(
                snapshot_revision=1,
                source_frame=1,
                source_simulation_timestamp=0.1,
                source_receive_monotonic=0.1,
                state=LaneDetectionState.NOT_DETECTED,
                reason=LaneObservationReason.NO_PLAUSIBLE_BOUNDARY,
                temporal_estimate=uninitialized,
            )
        reason = _temporal_disposition(
            TemporalLaneState.UNINITIALIZED,
            None,
            0.0,
            DEFAULT_SAFETY_SUPERVISOR_CONFIG,
        )
        self.assertIs(reason, SafetyReason.TEMPORAL_UNINITIALIZED)
        defensive = SafetyDecision(
            state=SafetyState.FAIL_SAFE,
            reason=reason,
            autonomy_allowed=False,
            evaluation_monotonic=1.0,
        )
        self.assertIs(defensive.state, SafetyState.FAIL_SAFE)
        self.assertFalse(defensive.autonomy_allowed)

    def test_unusable_evidence_never_advances_recovery_streak(self):
        supervisor = SafetySupervisor()
        supervisor.update(result(1, LOST), now=0.1, health=RUNNING)
        for frame, kind in ((2, UNUSABLE), (3, LOST), (4, UNUSABLE)):
            decision = supervisor.update(
                result(frame, kind), now=frame / 10.0, health=RUNNING
            )
            self.assertFalse(decision.autonomy_allowed)
            self.assertEqual(decision.recovery_streak, 0)
        self.assertEqual(supervisor.metrics.recovery_sequences, 1)


class SafetyFreshnessTest(unittest.TestCase):
    def setUp(self):
        self.max_age = DEFAULT_SAFETY_SUPERVISOR_CONFIG.max_result_age_seconds
        # A zero reference keeps ``now - reference`` exact, so the tests below
        # examine the real one-ULP boundary rather than a rounded neighbourhood.
        self.reference = 0.0

    def fresh_supervisor(self) -> SafetySupervisor:
        supervisor = SafetySupervisor()
        supervisor.update(
            result(1, DETECTED, monotonic=self.reference),
            now=self.reference,
            health=RUNNING,
        )
        return supervisor

    def test_age_just_below_maximum_remains_permitted(self):
        supervisor = self.fresh_supervisor()
        under = math.nextafter(self.max_age, 0.0)
        decision = supervisor.evaluate(now=self.reference + under, health=RUNNING)
        self.assertIs(decision.state, SafetyState.NOMINAL)
        self.assertTrue(decision.autonomy_allowed)
        self.assertEqual(decision.result_age_seconds, under)
        self.assertLess(decision.result_age_seconds, self.max_age)

    def test_exact_maximum_age_remains_permitted(self):
        supervisor = self.fresh_supervisor()
        decision = supervisor.evaluate(
            now=self.reference + self.max_age, health=RUNNING
        )
        self.assertIs(decision.state, SafetyState.NOMINAL)
        self.assertTrue(decision.autonomy_allowed)
        self.assertEqual(decision.result_age_seconds, self.max_age)
        self.assertEqual(supervisor.metrics.stale_expirations, 0)

    def test_one_ulp_above_maximum_age_fails_closed(self):
        supervisor = self.fresh_supervisor()
        over = math.nextafter(self.max_age, math.inf)
        decision = supervisor.evaluate(now=self.reference + over, health=RUNNING)
        self.assertIs(decision.state, SafetyState.FAIL_SAFE)
        self.assertIs(decision.reason, SafetyReason.RESULT_STALE)
        self.assertFalse(decision.autonomy_allowed)
        self.assertFalse(decision.latched)
        self.assertEqual(decision.result_age_seconds, over)
        self.assertGreater(decision.result_age_seconds, self.max_age)
        self.assertEqual(supervisor.metrics.stale_expirations, 1)

    def test_boundary_holds_away_from_a_zero_reference(self):
        # The same inclusive rule must hold for an arbitrary monotonic origin.
        reference = 1234.5
        supervisor = SafetySupervisor()
        supervisor.update(
            result(1, DETECTED, monotonic=reference), now=reference, health=RUNNING
        )
        fresh = supervisor.evaluate(now=reference + 0.30, health=RUNNING)
        self.assertIs(fresh.state, SafetyState.NOMINAL)
        stale = supervisor.evaluate(now=reference + 0.40, health=RUNNING)
        self.assertIs(stale.reason, SafetyReason.RESULT_STALE)
        self.assertGreater(stale.result_age_seconds, self.max_age)

    def test_stale_transition_requires_no_new_perception_result(self):
        supervisor = self.fresh_supervisor()
        # Not one further runtime result is submitted; only safety time moves.
        states = [
            supervisor.evaluate(now=self.reference + step, health=RUNNING).state
            for step in (0.10, 0.20, 0.30, 0.40, 0.50)
        ]
        self.assertEqual(
            states,
            [
                SafetyState.NOMINAL,
                SafetyState.NOMINAL,
                SafetyState.NOMINAL,
                SafetyState.FAIL_SAFE,
                SafetyState.FAIL_SAFE,
            ],
        )
        self.assertEqual(supervisor.metrics.updates_received, 1)
        self.assertEqual(supervisor.metrics.stale_expirations, 1)

    def test_repeated_ticks_cannot_restore_safety(self):
        supervisor = self.fresh_supervisor()
        supervisor.evaluate(now=self.reference + 1.0, health=RUNNING)
        for step in range(2, 60):
            decision = supervisor.evaluate(
                now=self.reference + float(step), health=RUNNING
            )
            self.assertIs(decision.reason, SafetyReason.RESULT_STALE)
            self.assertFalse(decision.autonomy_allowed)
        # Exactly one expiry edge is counted no matter how long the tick runs.
        self.assertEqual(supervisor.metrics.stale_expirations, 1)
        self.assertEqual(supervisor.metrics.recovery_sequences, 1)

    def test_out_of_order_result_cannot_refresh_age(self):
        supervisor = SafetySupervisor()
        supervisor.update(
            result(5, DETECTED, monotonic=0.5), now=0.5, health=RUNNING
        )
        stale_now = 0.5 + self.max_age + 0.1
        # An older frame carrying a very recent host timestamp must not be
        # able to make the retained safety evidence look fresh again.
        decision = supervisor.update(
            result(3, DETECTED, sim_time=0.3, monotonic=stale_now),
            now=stale_now,
            health=RUNNING,
        )
        self.assertIs(decision.reason, SafetyReason.RESULT_STALE)
        self.assertFalse(decision.autonomy_allowed)
        self.assertEqual(decision.source_stamp.frame, 5)
        self.assertGreater(decision.result_age_seconds, self.max_age)
        self.assertEqual(supervisor.metrics.out_of_order_rejected, 1)
        self.assertEqual(supervisor.metrics.new_results_accepted, 1)


class SafetyIdentityTest(unittest.TestCase):
    def assert_latched_mismatch(self, supervisor: SafetySupervisor, decision):
        self.assertIs(decision.state, SafetyState.FAIL_SAFE)
        self.assertIs(decision.reason, SafetyReason.IDENTITY_MISMATCH)
        self.assertFalse(decision.autonomy_allowed)
        self.assertTrue(decision.latched)
        self.assertEqual(supervisor.metrics.identity_mismatch_trips, 1)
        self.assertEqual(supervisor.metrics.new_results_accepted, 0)

    def test_outer_and_carrier_frame_mismatch_latches(self):
        supervisor = SafetySupervisor()
        mismatched = wrap(build_carrier(4, DETECTED), frame=5)
        decision = supervisor.update(mismatched, now=0.5, health=RUNNING)
        self.assert_latched_mismatch(supervisor, decision)

    def test_simulation_time_mismatch_latches(self):
        supervisor = SafetySupervisor()
        mismatched = wrap(build_carrier(4, DETECTED), sim_time=0.9)
        decision = supervisor.update(mismatched, now=0.5, health=RUNNING)
        self.assert_latched_mismatch(supervisor, decision)

    def test_receive_monotonic_mismatch_latches(self):
        supervisor = SafetySupervisor()
        mismatched = wrap(build_carrier(4, DETECTED), monotonic=0.9)
        decision = supervisor.update(mismatched, now=1.0, health=RUNNING)
        self.assert_latched_mismatch(supervisor, decision)

    def test_snapshot_revision_mismatch_latches(self):
        supervisor = SafetySupervisor()
        mismatched = wrap(build_carrier(4, DETECTED), revision=77)
        decision = supervisor.update(mismatched, now=0.5, health=RUNNING)
        self.assert_latched_mismatch(supervisor, decision)

    def test_desynchronized_temporal_estimate_latches(self):
        # Mission 7 pairs raw and temporal evidence atomically in its
        # constructor.  This probe bypasses that constructor to prove Mission 8
        # verifies the pair itself rather than trusting the producer.
        carrier = build_carrier(4, DETECTED)
        foreign = build_carrier(9, DETECTED).temporal_estimate
        object.__setattr__(carrier, "temporal_estimate", foreign)
        supervisor = SafetySupervisor()
        decision = supervisor.update(wrap(carrier, frame=4), now=0.5, health=RUNNING)
        self.assert_latched_mismatch(supervisor, decision)

    def test_raw_state_disagreement_between_carrier_and_estimate_latches(self):
        # Same frame, same revision, contradictory detection state: the safety
        # policy must never read a state the raw carrier disagrees with.
        carrier = build_carrier(4, DETECTED)
        foreign = build_carrier(4, COASTING).temporal_estimate
        object.__setattr__(carrier, "temporal_estimate", foreign)
        self.assertEqual(
            foreign.source_stamp, SampleStamp(4, 0.4, 0.4), "probe must share identity"
        )
        supervisor = SafetySupervisor()
        decision = supervisor.update(wrap(carrier), now=0.5, health=RUNNING)
        self.assert_latched_mismatch(supervisor, decision)

    def test_exactly_consistent_evidence_is_accepted(self):
        supervisor = SafetySupervisor()
        coherent = result(4, DETECTED)
        decision = supervisor.update(coherent, now=0.5, health=RUNNING)
        self.assertIs(decision.state, SafetyState.NOMINAL)
        self.assertEqual(supervisor.metrics.identity_mismatch_trips, 0)
        self.assertEqual(supervisor.metrics.malformed_evidence_trips, 0)
        self.assertEqual(decision.source_stamp, coherent.source_stamp)
        self.assertIs(type(decision.source_stamp), SampleStamp)

    def test_cross_epoch_cached_carrier_is_detected(self):
        # Reproduces the accepted Mission 7 reuse limitation in miniature: a
        # prior-epoch carrier republished under a new epoch's runtime identity.
        supervisor = SafetySupervisor()
        supervisor.update(result(1, DETECTED), now=0.1, health=RUNNING)
        cached = build_carrier(40, DETECTED, sim_time=4.0, monotonic=4.0)
        republished = wrap(cached, frame=2, sim_time=0.2, monotonic=0.2, revision=2)
        decision = supervisor.update(republished, now=0.2, health=RUNNING)
        self.assertIs(decision.reason, SafetyReason.IDENTITY_MISMATCH)
        self.assertTrue(decision.latched)
        self.assertFalse(decision.autonomy_allowed)
        self.assertEqual(supervisor.metrics.identity_mismatch_trips, 1)

    def test_non_temporal_observation_is_malformed_and_latches(self):
        plain = LaneObservation(
            snapshot_revision=1,
            source_frame=1,
            source_simulation_timestamp=0.1,
            source_receive_monotonic=0.1,
            state=LaneDetectionState.NOT_DETECTED,
            reason=LaneObservationReason.NO_PLAUSIBLE_BOUNDARY,
        )
        supervisor = SafetySupervisor()
        decision = supervisor.update(wrap(plain), now=0.1, health=RUNNING)
        self.assertIs(decision.reason, SafetyReason.EVIDENCE_MALFORMED)
        self.assertTrue(decision.latched)
        self.assertFalse(decision.autonomy_allowed)
        self.assertEqual(supervisor.metrics.malformed_evidence_trips, 1)

    def test_identity_gate_precedes_the_ordering_gate(self):
        # A stale-ordering carrier must not mask an invariant violation.
        supervisor = SafetySupervisor()
        supervisor.update(result(10, DETECTED, sim_time=1.0, monotonic=1.0), now=1.0)
        mismatched = wrap(build_carrier(3, DETECTED), frame=4)
        decision = supervisor.update(mismatched, now=1.0)
        self.assertIs(decision.reason, SafetyReason.IDENTITY_MISMATCH)
        self.assertEqual(supervisor.metrics.out_of_order_rejected, 0)
        self.assertEqual(supervisor.metrics.identity_mismatch_trips, 1)


class SafetyComponentFaultTest(unittest.TestCase):
    FAULTED_RUNTIME = ComponentHealth(
        runtime_state=PerceptionRuntimeState.FAULTED,
        runtime_fault_present=True,
    )
    FAULTED_BRIDGE = ComponentHealth(
        runtime_state=PerceptionRuntimeState.RUNNING,
        bridge_state=PerceptionLiveBridgeState.FAULTED,
        bridge_fault_present=True,
    )

    def test_runtime_fault_latches_immediately(self):
        supervisor = SafetySupervisor()
        supervisor.update(result(1, DETECTED), now=0.1, health=RUNNING)
        decision = supervisor.evaluate(now=0.15, health=self.FAULTED_RUNTIME)
        self.assertIs(decision.state, SafetyState.FAIL_SAFE)
        self.assertIs(decision.reason, SafetyReason.RUNTIME_FAULT)
        self.assertTrue(decision.latched)
        self.assertFalse(decision.autonomy_allowed)
        self.assertEqual(supervisor.metrics.component_fault_trips, 1)

    def test_runtime_fault_without_faulted_state_still_latches(self):
        supervisor = SafetySupervisor()
        torn = ComponentHealth(
            runtime_state=PerceptionRuntimeState.STOPPED,
            runtime_fault_present=True,
        )
        decision = supervisor.update(result(1, DETECTED), now=0.1, health=torn)
        self.assertIs(decision.reason, SafetyReason.RUNTIME_FAULT)
        self.assertTrue(decision.latched)

    def test_bridge_fault_latches_immediately(self):
        supervisor = SafetySupervisor()
        supervisor.update(result(1, DETECTED), now=0.1, health=RUNNING)
        decision = supervisor.update(
            result(2, DETECTED), now=0.2, health=self.FAULTED_BRIDGE
        )
        self.assertIs(decision.reason, SafetyReason.BRIDGE_FAULT)
        self.assertTrue(decision.latched)
        self.assertFalse(decision.autonomy_allowed)
        self.assertEqual(supervisor.metrics.component_fault_trips, 1)
        # A latched supervisor stops consuming perception evidence entirely.
        self.assertEqual(supervisor.metrics.new_results_accepted, 1)

    def test_both_components_faulted_report_runtime_fault_once(self):
        both = ComponentHealth(
            runtime_state=PerceptionRuntimeState.FAULTED,
            runtime_fault_present=True,
            bridge_state=PerceptionLiveBridgeState.FAULTED,
            bridge_fault_present=True,
        )
        supervisor = SafetySupervisor()
        first = supervisor.evaluate(now=0.1, health=both)
        second = supervisor.evaluate(now=0.2, health=both)
        self.assertIs(first.reason, SafetyReason.RUNTIME_FAULT)
        self.assertIs(second.reason, SafetyReason.RUNTIME_FAULT)
        self.assertTrue(second.latched)
        self.assertEqual(supervisor.metrics.component_fault_trips, 1)

    def test_healthy_result_cannot_clear_hard_latch(self):
        supervisor = SafetySupervisor()
        supervisor.evaluate(now=0.1, health=self.FAULTED_RUNTIME)
        for frame in (1, 2, 3, 4, 5, 6):
            decision = supervisor.update(
                result(frame, DETECTED), now=frame / 10.0, health=RUNNING
            )
            self.assertIs(decision.reason, SafetyReason.RUNTIME_FAULT)
            self.assertTrue(decision.latched)
            self.assertFalse(decision.autonomy_allowed)
        self.assertEqual(supervisor.metrics.new_results_accepted, 0)

    def test_reset_clears_hard_latch(self):
        supervisor = SafetySupervisor()
        supervisor.evaluate(now=0.1, health=self.FAULTED_RUNTIME)
        cleared = supervisor.reset()
        self.assertIs(cleared.state, SafetyState.INITIALIZING)
        self.assertFalse(cleared.latched)
        decision = supervisor.update(result(1, DETECTED), now=0.1, health=RUNNING)
        self.assertIs(decision.state, SafetyState.NOMINAL)
        self.assertTrue(decision.autonomy_allowed)
        self.assertEqual(supervisor.metrics.component_fault_trips, 1)

    def test_component_not_ready_fails_closed_without_latching(self):
        supervisor = SafetySupervisor()
        supervisor.update(result(1, DETECTED), now=0.1, health=RUNNING)
        stopped = ComponentHealth(runtime_state=PerceptionRuntimeState.STOPPED)
        decision = supervisor.evaluate(now=0.15, health=stopped)
        self.assertIs(decision.reason, SafetyReason.COMPONENT_NOT_READY)
        self.assertFalse(decision.autonomy_allowed)
        self.assertFalse(decision.latched)
        self.assertEqual(supervisor.metrics.component_fault_trips, 0)

    def test_closed_bridge_is_not_ready(self):
        supervisor = SafetySupervisor()
        closed = ComponentHealth(
            runtime_state=PerceptionRuntimeState.RUNNING,
            bridge_state=PerceptionLiveBridgeState.CLOSED,
        )
        decision = supervisor.update(result(1, DETECTED), now=0.1, health=closed)
        self.assertIs(decision.reason, SafetyReason.COMPONENT_NOT_READY)
        self.assertFalse(decision.autonomy_allowed)

    def test_component_health_validates_its_evidence(self):
        with self.assertRaises(ValueError):
            ComponentHealth(runtime_state="running")
        with self.assertRaises(ValueError):
            ComponentHealth(
                runtime_state=PerceptionRuntimeState.RUNNING,
                bridge_state="open",
            )
        with self.assertRaises(ValueError):
            ComponentHealth(
                runtime_state=PerceptionRuntimeState.RUNNING,
                runtime_fault_present=1,
            )
        with self.assertRaises(ValueError):
            # Bridge fault evidence without a bridge is contradictory.
            ComponentHealth(
                runtime_state=PerceptionRuntimeState.RUNNING,
                bridge_fault_present=True,
            )
        with self.assertRaises(TypeError):
            SafetySupervisor().evaluate(now=0.1, health=object())


class SafetyUnknownHealthTest(unittest.TestCase):
    """Omitting ``health`` supplies no new evidence; it never asserts health.

    A supervisor epoch that has never observed a ``ComponentHealth`` record
    holds component health unknown, and unknown is not healthy.
    """

    NOT_READY = ComponentHealth(runtime_state=PerceptionRuntimeState.STOPPED)
    FAULTED_RUNTIME = ComponentHealth(
        runtime_state=PerceptionRuntimeState.FAULTED,
        runtime_fault_present=True,
    )
    FAULTED_BRIDGE = ComponentHealth(
        runtime_state=PerceptionRuntimeState.RUNNING,
        bridge_state=PerceptionLiveBridgeState.FAULTED,
        bridge_fault_present=True,
    )

    def test_valid_result_without_any_health_is_not_permitted(self):
        supervisor = SafetySupervisor()
        decision = supervisor.update(result(1, DETECTED), now=0.1)
        self.assertIs(decision.state, SafetyState.FAIL_SAFE)
        self.assertIs(decision.reason, SafetyReason.COMPONENT_NOT_READY)
        self.assertFalse(decision.autonomy_allowed)
        self.assertFalse(decision.latched)

    def test_evaluate_without_any_health_is_not_permitted(self):
        supervisor = SafetySupervisor()
        supervisor.update(result(1, DETECTED), now=0.1)
        decision = supervisor.evaluate(now=0.15)
        self.assertIs(decision.reason, SafetyReason.COMPONENT_NOT_READY)
        self.assertFalse(decision.autonomy_allowed)

    def test_no_temporal_evidence_is_permitted_while_health_is_unknown(self):
        # Every state that could otherwise permit autonomy must stay denied.
        for kind in (DETECTED, PARTIAL, COASTING):
            for confidence in (1.0, 0.8, 0.65, 0.5, 0.4):
                with self.subTest(kind=kind, confidence=confidence):
                    supervisor = SafetySupervisor()
                    decision = supervisor.update(
                        result(1, kind, confidence=confidence), now=0.1
                    )
                    self.assertFalse(decision.autonomy_allowed)
                    self.assertIs(
                        decision.reason, SafetyReason.COMPONENT_NOT_READY
                    )

    def test_startup_without_a_result_still_reports_initializing(self):
        # Unknown health must not pre-empt the startup explanation.
        supervisor = SafetySupervisor()
        decision = supervisor.evaluate(now=0.1)
        self.assertIs(decision.state, SafetyState.INITIALIZING)
        self.assertIs(decision.reason, SafetyReason.STARTUP_NO_RESULT)

    def test_first_supplied_healthy_sample_permits_autonomy(self):
        supervisor = SafetySupervisor()
        decision = supervisor.update(result(1, DETECTED), now=0.1, health=RUNNING)
        self.assertIs(decision.state, SafetyState.NOMINAL)
        self.assertTrue(decision.autonomy_allowed)

    def test_observed_ready_persists_through_omission(self):
        supervisor = SafetySupervisor()
        supervisor.update(result(1, DETECTED), now=0.1, health=RUNNING)
        decision = supervisor.update(result(2, DETECTED), now=0.2)
        self.assertIs(decision.state, SafetyState.NOMINAL)
        self.assertTrue(decision.autonomy_allowed)

    def test_observed_not_ready_persists_through_omission(self):
        supervisor = SafetySupervisor()
        supervisor.update(result(1, DETECTED), now=0.1, health=self.NOT_READY)
        decision = supervisor.update(result(2, DETECTED), now=0.2)
        self.assertIs(decision.reason, SafetyReason.COMPONENT_NOT_READY)
        self.assertFalse(decision.autonomy_allowed)
        self.assertFalse(decision.latched)

    def test_runtime_fault_latch_persists_through_omission(self):
        supervisor = SafetySupervisor()
        supervisor.update(result(1, DETECTED), now=0.1, health=self.FAULTED_RUNTIME)
        decision = supervisor.update(result(2, DETECTED), now=0.2)
        self.assertIs(decision.reason, SafetyReason.RUNTIME_FAULT)
        self.assertTrue(decision.latched)
        self.assertFalse(decision.autonomy_allowed)

    def test_bridge_fault_latch_persists_through_omission(self):
        supervisor = SafetySupervisor()
        supervisor.update(result(1, DETECTED), now=0.1, health=self.FAULTED_BRIDGE)
        decision = supervisor.update(result(2, DETECTED), now=0.2)
        self.assertIs(decision.reason, SafetyReason.BRIDGE_FAULT)
        self.assertTrue(decision.latched)

    def test_reset_after_healthy_returns_health_to_unknown(self):
        supervisor = SafetySupervisor()
        self.assertTrue(
            supervisor.update(
                result(1, DETECTED), now=0.1, health=RUNNING
            ).autonomy_allowed
        )
        supervisor.reset()
        decision = supervisor.update(result(9, DETECTED), now=0.9)
        self.assertIs(decision.reason, SafetyReason.COMPONENT_NOT_READY)
        self.assertFalse(decision.autonomy_allowed)

    def test_reset_after_hard_fault_clears_latch_but_not_permission(self):
        supervisor = SafetySupervisor()
        supervisor.update(result(1, DETECTED), now=0.1, health=self.FAULTED_RUNTIME)
        self.assertTrue(supervisor.latest_decision.latched)
        supervisor.reset()
        decision = supervisor.update(result(9, DETECTED), now=0.9)
        self.assertFalse(decision.latched)
        self.assertIs(decision.reason, SafetyReason.COMPONENT_NOT_READY)
        self.assertFalse(decision.autonomy_allowed)

    def test_rising_frame_stream_without_health_never_permits_autonomy(self):
        # A faulted runtime restarted inside one CARLA session keeps publishing
        # strictly newer frames.  Without health evidence the supervisor must
        # still refuse rather than silently resume.
        supervisor = SafetySupervisor()
        for frame in list(range(100, 103)) + list(range(111, 121)):
            update = supervisor.update(
                result(frame, DETECTED, monotonic=frame / 1000, sim_time=frame / 10),
                now=frame / 1000,
            )
            tick = supervisor.evaluate(now=frame / 1000 + 0.001)
            self.assertFalse(update.autonomy_allowed, frame)
            self.assertFalse(tick.autonomy_allowed, frame)

    def test_unknown_health_never_completes_a_recovery_streak(self):
        supervisor = SafetySupervisor()
        supervisor.update(result(1, LOST), now=0.1)
        for frame in range(2, 12):
            decision = supervisor.update(result(frame, DETECTED), now=frame / 10)
            # Asserting the reason keeps this from passing for an unrelated
            # fail-closed cause such as a regressed clock.
            self.assertIs(decision.reason, SafetyReason.COMPONENT_NOT_READY, frame)
            self.assertFalse(decision.autonomy_allowed, frame)
            self.assertEqual(decision.recovery_streak, 0, frame)

    def test_duplicate_and_out_of_order_cannot_permit_while_unknown(self):
        supervisor = SafetySupervisor()
        supervisor.update(result(5, DETECTED), now=0.5)
        duplicate = supervisor.update(result(5, DETECTED), now=0.51)
        regressed = supervisor.update(result(2, DETECTED), now=0.52)
        self.assertFalse(duplicate.autonomy_allowed)
        self.assertFalse(regressed.autonomy_allowed)
        metrics = supervisor.metrics
        self.assertEqual(metrics.duplicates_rejected, 1)
        self.assertEqual(metrics.out_of_order_rejected, 1)
        self.assertEqual(metrics.new_results_accepted, 1)

    def test_unknown_health_is_an_ordinary_soft_loss(self):
        # Unknown health takes the existing COMPONENT_NOT_READY soft path, so
        # the first real health sample resumes through the normal N=3
        # hysteresis rather than jumping straight back to NOMINAL.
        supervisor = SafetySupervisor()
        supervisor.update(result(1, DETECTED), now=0.1)
        states = [
            supervisor.update(result(frame, DETECTED), now=frame / 10, health=RUNNING)
            for frame in (2, 3, 4)
        ]
        self.assertIs(states[0].state, SafetyState.RECOVERING)
        self.assertIs(states[1].state, SafetyState.RECOVERING)
        self.assertIs(states[2].state, SafetyState.NOMINAL)
        self.assertTrue(states[2].autonomy_allowed)

    def test_omitted_health_does_not_refresh_result_freshness(self):
        supervisor = SafetySupervisor()
        supervisor.update(result(1, DETECTED, monotonic=0.0), now=0.0, health=RUNNING)
        for now in (0.10, 0.20, 0.30):
            supervisor.evaluate(now=now)
        decision = supervisor.evaluate(now=0.36)
        self.assertIs(decision.reason, SafetyReason.RESULT_STALE)
        self.assertFalse(decision.autonomy_allowed)

    def test_unknown_health_does_not_change_source_order_accounting(self):
        # A higher frame at equal simulation time stays valid, exactly as in
        # the accepted Mission 6 contract.
        supervisor = SafetySupervisor()
        supervisor.update(result(10, DETECTED, sim_time=1.0), now=1.0)
        supervisor.update(result(10, DETECTED, sim_time=1.0), now=1.01)
        supervisor.update(
            result(11, DETECTED, sim_time=1.0, monotonic=1.0), now=1.02, health=RUNNING
        )
        metrics = supervisor.metrics
        self.assertEqual(metrics.duplicates_rejected, 1)
        self.assertEqual(metrics.out_of_order_rejected, 0)
        self.assertEqual(metrics.new_results_accepted, 2)

    def test_identity_mismatch_still_latches_while_health_is_unknown(self):
        supervisor = SafetySupervisor()
        carrier = build_carrier(3, DETECTED)
        decision = supervisor.update(
            wrap(carrier, frame=9, sim_time=0.9, revision=9), now=1.0
        )
        self.assertIs(decision.reason, SafetyReason.IDENTITY_MISMATCH)
        self.assertTrue(decision.latched)
        self.assertEqual(supervisor.metrics.out_of_order_rejected, 0)

    def test_unknown_health_state_stays_bounded_across_a_long_stream(self):
        supervisor = SafetySupervisor()
        for frame in range(1, 501):
            supervisor.update(
                result(frame, DETECTED, monotonic=frame / 1000, sim_time=frame / 10),
                now=frame / 1000,
            )
            supervisor.evaluate(now=frame / 1000 + 0.0001)
        self.assertFalse(supervisor.autonomy_allowed)
        self.assertEqual(supervisor.metrics.evaluations, 1000)
        # One armed recovery sequence for the whole stream, not one per sample.
        self.assertEqual(supervisor.metrics.recovery_sequences, 1)


class SafetyRecoveryTest(unittest.TestCase):
    def armed_supervisor(self, config=None) -> SafetySupervisor:
        supervisor = SafetySupervisor(
            DEFAULT_SAFETY_SUPERVISOR_CONFIG if config is None else config
        )
        supervisor.update(result(1, DETECTED), now=0.1, health=RUNNING)
        loss = supervisor.update(result(2, LOST), now=0.2, health=RUNNING)
        self.assertIs(loss.state, SafetyState.FAIL_SAFE)
        return supervisor

    def test_soft_loss_arms_recovery(self):
        supervisor = self.armed_supervisor()
        decision = supervisor.update(result(3, DETECTED), now=0.3, health=RUNNING)
        self.assertIs(decision.state, SafetyState.RECOVERING)
        self.assertIs(decision.reason, SafetyReason.RECOVERY_PENDING)
        self.assertFalse(decision.autonomy_allowed)
        self.assertEqual(supervisor.metrics.recovery_sequences, 1)

    def test_first_healthy_sample_remains_recovering(self):
        supervisor = self.armed_supervisor()
        decision = supervisor.update(result(3, DETECTED), now=0.3, health=RUNNING)
        self.assertEqual(decision.recovery_streak, 1)
        self.assertFalse(decision.autonomy_allowed)

    def test_one_below_required_streak_remains_recovering(self):
        supervisor = self.armed_supervisor()
        required = supervisor.config.required_healthy_results
        for offset in range(required - 1):
            frame = 3 + offset
            decision = supervisor.update(
                result(frame, DETECTED), now=frame / 10.0, health=RUNNING
            )
            self.assertIs(decision.state, SafetyState.RECOVERING)
            self.assertEqual(decision.recovery_streak, offset + 1)
            self.assertFalse(decision.autonomy_allowed)

    def test_exact_required_streak_re_enables_autonomy(self):
        supervisor = self.armed_supervisor()
        required = supervisor.config.required_healthy_results
        decisions = [
            supervisor.update(
                result(3 + offset, DETECTED),
                now=(3 + offset) / 10.0,
                health=RUNNING,
            )
            for offset in range(required)
        ]
        self.assertEqual(
            [decision.autonomy_allowed for decision in decisions],
            [False] * (required - 1) + [True],
        )
        self.assertIs(decisions[-1].state, SafetyState.NOMINAL)
        self.assertEqual(decisions[-1].recovery_streak, 0)
        # A later result no longer needs a streak.
        after = supervisor.update(result(9, DETECTED), now=0.9, health=RUNNING)
        self.assertIs(after.state, SafetyState.NOMINAL)
        self.assertEqual(supervisor.metrics.recovery_sequences, 1)

    def test_unsafe_sample_resets_streak(self):
        supervisor = self.armed_supervisor()
        supervisor.update(result(3, DETECTED), now=0.3, health=RUNNING)
        broken = supervisor.update(result(4, LOST), now=0.4, health=RUNNING)
        self.assertEqual(broken.recovery_streak, 0)
        resumed = supervisor.update(result(5, DETECTED), now=0.5, health=RUNNING)
        self.assertIs(resumed.state, SafetyState.RECOVERING)
        self.assertEqual(resumed.recovery_streak, 1)
        self.assertEqual(supervisor.metrics.recovery_sequences, 1)

    def test_duplicate_does_not_advance_streak(self):
        supervisor = self.armed_supervisor()
        supervisor.update(result(3, DETECTED), now=0.3, health=RUNNING)
        for _repeat in range(5):
            decision = supervisor.update(result(3, DETECTED), now=0.3, health=RUNNING)
            self.assertEqual(decision.recovery_streak, 1)
            self.assertFalse(decision.autonomy_allowed)
        self.assertEqual(supervisor.metrics.duplicates_rejected, 5)

    def test_out_of_order_does_not_advance_streak(self):
        supervisor = self.armed_supervisor()
        supervisor.update(result(5, DETECTED), now=0.5, health=RUNNING)
        for frame in (3, 4, 5):
            decision = supervisor.update(
                result(frame, DETECTED), now=0.5, health=RUNNING
            )
            self.assertEqual(decision.recovery_streak, 1)
            self.assertFalse(decision.autonomy_allowed)
        self.assertEqual(supervisor.metrics.out_of_order_rejected, 2)
        self.assertEqual(supervisor.metrics.duplicates_rejected, 1)

    def test_partial_and_coasting_do_not_advance_streak(self):
        for kind in (PARTIAL, COASTING):
            with self.subTest(kind=kind):
                supervisor = self.armed_supervisor()
                supervisor.update(result(3, DETECTED), now=0.3, health=RUNNING)
                weak = supervisor.update(result(4, kind), now=0.4, health=RUNNING)
                self.assertIs(weak.state, SafetyState.RECOVERING)
                self.assertFalse(weak.autonomy_allowed)
                self.assertEqual(weak.recovery_streak, 0)

    def test_low_confidence_detected_sample_does_not_advance_streak(self):
        supervisor = self.armed_supervisor()
        below = math.nextafter(
            DEFAULT_SAFETY_SUPERVISOR_CONFIG.min_nominal_confidence, 0.0
        )
        weak = supervisor.update(
            result(3, DETECTED, confidence=below), now=0.3, health=RUNNING
        )
        self.assertIs(weak.state, SafetyState.RECOVERING)
        self.assertEqual(weak.recovery_streak, 0)

    def test_reset_during_recovery_restores_startup(self):
        supervisor = self.armed_supervisor()
        supervisor.update(result(3, DETECTED), now=0.3, health=RUNNING)
        cleared = supervisor.reset()
        self.assertIs(cleared.state, SafetyState.INITIALIZING)
        self.assertEqual(cleared.recovery_streak, 0)
        decision = supervisor.update(result(1, DETECTED), now=0.1, health=RUNNING)
        self.assertIs(decision.state, SafetyState.NOMINAL)
        self.assertTrue(decision.autonomy_allowed)

    def test_stale_recovery_requires_a_fresh_streak(self):
        supervisor = SafetySupervisor(
            SafetySupervisorConfig(required_healthy_results=2)
        )
        supervisor.update(result(1, DETECTED, monotonic=1.0), now=1.0, health=RUNNING)
        stale = supervisor.evaluate(now=5.0, health=RUNNING)
        self.assertIs(stale.reason, SafetyReason.RESULT_STALE)
        first = supervisor.update(
            result(2, DETECTED, monotonic=5.0), now=5.0, health=RUNNING
        )
        self.assertIs(first.state, SafetyState.RECOVERING)
        second = supervisor.update(
            result(3, DETECTED, sim_time=0.3, monotonic=5.1), now=5.1, health=RUNNING
        )
        self.assertIs(second.state, SafetyState.NOMINAL)
        self.assertTrue(second.autonomy_allowed)


class SafetyOrderingTest(unittest.TestCase):
    def test_duplicate_result_is_rejected_idempotently(self):
        supervisor = SafetySupervisor()
        first = supervisor.update(result(4, DETECTED), now=0.4, health=RUNNING)
        repeat = supervisor.update(result(4, DETECTED), now=0.4, health=RUNNING)
        self.assertIs(repeat.state, SafetyState.NOMINAL)
        self.assertEqual(repeat.source_stamp, first.source_stamp)
        self.assertEqual(supervisor.metrics.duplicates_rejected, 1)
        self.assertEqual(supervisor.metrics.new_results_accepted, 1)

    def test_older_frame_is_rejected(self):
        supervisor = SafetySupervisor()
        supervisor.update(result(4, DETECTED), now=0.4, health=RUNNING)
        decision = supervisor.update(result(3, DETECTED), now=0.4, health=RUNNING)
        self.assertEqual(decision.source_stamp.frame, 4)
        self.assertEqual(supervisor.metrics.out_of_order_rejected, 1)

    def test_regressed_simulation_time_is_rejected(self):
        supervisor = SafetySupervisor()
        supervisor.update(
            result(4, DETECTED, sim_time=0.40, monotonic=0.40), now=0.40, health=RUNNING
        )
        decision = supervisor.update(
            result(5, DETECTED, sim_time=0.30, monotonic=0.45), now=0.45, health=RUNNING
        )
        self.assertEqual(decision.source_stamp.frame, 4)
        self.assertEqual(decision.source_stamp.sim_time, 0.40)
        self.assertEqual(supervisor.metrics.out_of_order_rejected, 1)

    def test_newer_frame_at_equal_simulation_time_is_accepted(self):
        supervisor = SafetySupervisor()
        supervisor.update(
            result(4, DETECTED, sim_time=0.40, monotonic=0.40), now=0.40, health=RUNNING
        )
        decision = supervisor.update(
            result(5, DETECTED, sim_time=0.40, monotonic=0.42), now=0.42, health=RUNNING
        )
        self.assertEqual(decision.source_stamp.frame, 5)
        self.assertEqual(supervisor.metrics.new_results_accepted, 2)
        self.assertEqual(supervisor.metrics.out_of_order_rejected, 0)

    def test_rejected_result_leaves_accepted_evidence_unchanged(self):
        supervisor = SafetySupervisor()
        accepted = supervisor.update(
            result(4, DETECTED, confidence=0.90), now=0.40, health=RUNNING
        )
        for frame, kind in ((4, LOST), (3, UNUSABLE), (2, LOST)):
            decision = supervisor.update(
                result(frame, kind), now=0.40, health=RUNNING
            )
            self.assertIs(decision.state, SafetyState.NOMINAL)
            self.assertEqual(decision.temporal_confidence, 0.90)
            self.assertEqual(decision.source_stamp, accepted.source_stamp)
            self.assertIs(decision.temporal_state, TemporalLaneState.TRACKING)
        metrics = supervisor.metrics
        self.assertEqual(metrics.duplicates_rejected, 1)
        self.assertEqual(metrics.out_of_order_rejected, 2)
        self.assertEqual(metrics.new_results_accepted, 1)


class SafetyValidationTest(unittest.TestCase):
    def test_nan_configuration_rejected(self):
        for field in (
            "max_result_age_seconds",
            "min_nominal_confidence",
            "min_degraded_confidence",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    SafetySupervisorConfig(**{field: math.nan})
        with self.assertRaises(ValueError):
            SafetySupervisor().evaluate(now=math.nan)

    def test_positive_infinity_configuration_rejected(self):
        for field in (
            "max_result_age_seconds",
            "min_nominal_confidence",
            "min_degraded_confidence",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    SafetySupervisorConfig(**{field: math.inf})
        with self.assertRaises(ValueError):
            SafetySupervisor().evaluate(now=math.inf)

    def test_negative_infinity_configuration_rejected(self):
        for field in (
            "max_result_age_seconds",
            "min_nominal_confidence",
            "min_degraded_confidence",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    SafetySupervisorConfig(**{field: -math.inf})
        with self.assertRaises(ValueError):
            SafetySupervisor().evaluate(now=-math.inf)

    def test_invalid_confidence_thresholds_rejected(self):
        for value in (-0.01, 1.01, "0.5", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    SafetySupervisorConfig(min_nominal_confidence=value)
                with self.assertRaises(ValueError):
                    SafetySupervisorConfig(min_degraded_confidence=value)
        with self.assertRaises(ValueError):
            SafetySupervisorConfig(
                min_nominal_confidence=0.40,
                min_degraded_confidence=math.nextafter(0.40, 1.0),
            )
        equal = SafetySupervisorConfig(
            min_nominal_confidence=0.40, min_degraded_confidence=0.40
        )
        self.assertEqual(equal.min_degraded_confidence, 0.40)

    def test_invalid_freshness_age_rejected(self):
        for value in (0.0, -0.1, MAX_SAFETY_RESULT_AGE_SECONDS + 0.1, "0.3"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    SafetySupervisorConfig(max_result_age_seconds=value)
        edge = SafetySupervisorConfig(
            max_result_age_seconds=MAX_SAFETY_RESULT_AGE_SECONDS
        )
        self.assertEqual(
            edge.max_result_age_seconds, MAX_SAFETY_RESULT_AGE_SECONDS
        )

    def test_invalid_recovery_sample_count_rejected(self):
        for value in (0, -1, 2.0, "3", MAX_REQUIRED_HEALTHY_RESULTS + 1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    SafetySupervisorConfig(required_healthy_results=value)
        edge = SafetySupervisorConfig(
            required_healthy_results=MAX_REQUIRED_HEALTHY_RESULTS
        )
        self.assertEqual(
            edge.required_healthy_results, MAX_REQUIRED_HEALTHY_RESULTS
        )

    def test_bool_values_rejected_where_numbers_are_required(self):
        for field in (
            "max_result_age_seconds",
            "min_nominal_confidence",
            "min_degraded_confidence",
            "required_healthy_results",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    SafetySupervisorConfig(**{field: True})
        with self.assertRaises(ValueError):
            SafetySupervisor().evaluate(now=True)

    def test_hostile_real_and_config_subclasses_are_canonicalized(self):
        class HostileConfig(SafetySupervisorConfig):
            pass

        hostile = HostileConfig(
            max_result_age_seconds=DriftingReal(0.50, 99.0),
            min_nominal_confidence=DriftingReal(0.70, 0.0),
            min_degraded_confidence=DriftingReal(0.30, 0.0),
            required_healthy_results=2,
        )
        supervisor = SafetySupervisor(hostile)
        retained = supervisor.config
        self.assertIs(type(retained), SafetySupervisorConfig)
        self.assertEqual(retained.max_result_age_seconds, 0.50)
        self.assertEqual(retained.min_nominal_confidence, 0.70)
        self.assertEqual(retained.min_degraded_confidence, 0.30)
        for value in (
            retained.max_result_age_seconds,
            retained.min_nominal_confidence,
            retained.min_degraded_confidence,
        ):
            self.assertIs(type(value), float)

        drifting_now = DriftingReal(0.10, 900.0)
        decision = supervisor.evaluate(now=drifting_now, health=RUNNING)
        self.assertIs(type(decision.evaluation_monotonic), float)
        self.assertEqual(decision.evaluation_monotonic, 0.10)

        # A caller-owned stamp subtype is copied, never retained.
        carrier = build_carrier(1, DETECTED)
        foreign = PerceptionRuntimeResult(
            observation=carrier,
            source_stamp=StampSubclass(frame=1, sim_time=0.1, monotonic=0.1),
            source_sequence=None,
            snapshot_revision=1,
            ingestion_monotonic=0.1,
            processing_start_monotonic=0.1,
            processing_finish_monotonic=0.1,
            pending_age_seconds=0.0,
            processing_latency_seconds=0.0,
        )
        accepted = supervisor.update(foreign, now=0.15, health=RUNNING)
        self.assertIs(type(accepted.source_stamp), SampleStamp)
        self.assertIs(
            type(supervisor.metrics.last_accepted_source_stamp), SampleStamp
        )

    def test_update_requires_a_runtime_result(self):
        supervisor = SafetySupervisor()
        for value in (None, object(), build_carrier(1, DETECTED)):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(TypeError):
                    supervisor.update(value, now=0.1)
        self.assertEqual(supervisor.metrics.updates_received, 0)
        self.assertEqual(supervisor.metrics.evaluations, 0)

    def test_invalid_safety_time_leaves_accounting_untouched(self):
        supervisor = SafetySupervisor()
        with self.assertRaises(ValueError):
            supervisor.update(result(1, DETECTED), now=math.nan)
        self.assertEqual(supervisor.metrics.updates_received, 0)
        self.assertEqual(supervisor.metrics.evaluations, 0)
        self.assertIs(supervisor.latest_decision.state, SafetyState.INITIALIZING)

    def test_supervisor_requires_a_config_instance(self):
        with self.assertRaises(TypeError):
            SafetySupervisor(config="default")

    def test_regressed_safety_time_latches(self):
        supervisor = SafetySupervisor()
        supervisor.update(result(1, DETECTED), now=1.0, health=RUNNING)
        decision = supervisor.evaluate(now=0.5, health=RUNNING)
        self.assertIs(decision.reason, SafetyReason.CLOCK_REGRESSED)
        self.assertTrue(decision.latched)
        self.assertFalse(decision.autonomy_allowed)
        self.assertEqual(supervisor.metrics.clock_regression_trips, 1)
        recovered = supervisor.update(result(2, DETECTED), now=2.0, health=RUNNING)
        self.assertIs(recovered.reason, SafetyReason.CLOCK_REGRESSED)
        self.assertTrue(recovered.latched)

    def test_safety_time_before_the_evidence_reference_latches(self):
        supervisor = SafetySupervisor()
        decision = supervisor.update(
            result(1, DETECTED, monotonic=9.0), now=1.0, health=RUNNING
        )
        self.assertIs(decision.reason, SafetyReason.CLOCK_REGRESSED)
        self.assertTrue(decision.latched)
        self.assertIsNone(decision.result_age_seconds)


class SafetyDecisionInvariantTest(unittest.TestCase):
    def test_permitted_state_cannot_carry_an_unsafe_reason(self):
        with self.assertRaises(ValueError):
            SafetyDecision(
                state=SafetyState.NOMINAL,
                reason=SafetyReason.TEMPORAL_LOST,
                autonomy_allowed=True,
            )

    def test_autonomy_flag_must_match_the_state(self):
        with self.assertRaises(ValueError):
            SafetyDecision(
                state=SafetyState.FAIL_SAFE,
                reason=SafetyReason.RESULT_STALE,
                autonomy_allowed=True,
            )
        with self.assertRaises(ValueError):
            SafetyDecision(
                state=SafetyState.NOMINAL,
                reason=SafetyReason.NOMINAL_TRACKING,
                autonomy_allowed=False,
            )
        with self.assertRaises(ValueError):
            SafetyDecision(
                state=SafetyState.NOMINAL,
                reason=SafetyReason.NOMINAL_TRACKING,
                autonomy_allowed=1,
            )

    def test_only_fail_safe_may_latch(self):
        with self.assertRaises(ValueError):
            SafetyDecision(
                state=SafetyState.RECOVERING,
                reason=SafetyReason.RECOVERY_PENDING,
                autonomy_allowed=False,
                evaluation_monotonic=1.0,
                latched=True,
            )

    def test_initializing_decision_cannot_expose_result_evidence(self):
        with self.assertRaises(ValueError):
            SafetyDecision(
                state=SafetyState.INITIALIZING,
                reason=SafetyReason.STARTUP_NO_RESULT,
                autonomy_allowed=False,
                snapshot_revision=1,
            )

    def test_non_initializing_decision_requires_an_evaluation_time(self):
        with self.assertRaises(ValueError):
            SafetyDecision(
                state=SafetyState.FAIL_SAFE,
                reason=SafetyReason.RESULT_STALE,
                autonomy_allowed=False,
            )

    def test_decision_is_frozen_and_slotted(self):
        supervisor = SafetySupervisor()
        decision = supervisor.update(result(1, DETECTED), now=0.1, health=RUNNING)
        with self.assertRaises(FrozenInstanceError):
            decision.autonomy_allowed = False
        self.assertFalse(hasattr(decision, "__dict__"))
        self.assertFalse(hasattr(supervisor, "__dict__"))
        self.assertFalse(hasattr(supervisor.metrics, "__dict__"))

    def test_negative_result_age_is_rejected(self):
        with self.assertRaises(ValueError):
            SafetyDecision(
                state=SafetyState.FAIL_SAFE,
                reason=SafetyReason.RESULT_STALE,
                autonomy_allowed=False,
                evaluation_monotonic=1.0,
                result_age_seconds=-0.1,
            )


class SafetyDeterminismTest(unittest.TestCase):
    STREAM = (
        (1, DETECTED, 0.10),
        (2, DETECTED, 0.20),
        (3, PARTIAL, 0.30),
        (4, COASTING, 0.40),
        (5, LOST, 0.50),
        (5, LOST, 0.50),
        (4, DETECTED, 0.50),
        (6, DETECTED, 0.60),
        (7, DETECTED, 0.70),
        (8, DETECTED, 0.80),
        (9, UNUSABLE, 0.90),
        (10, DETECTED, 1.00),
    )

    def run_stream(self, supervisor: SafetySupervisor) -> list[SafetyDecision]:
        decisions = []
        for frame, kind, now in self.STREAM:
            decisions.append(
                supervisor.update(
                    result(frame, kind, monotonic=now), now=now, health=RUNNING
                )
            )
            decisions.append(
                supervisor.evaluate(now=now + 0.01, health=RUNNING)
            )
        return decisions

    def test_identical_streams_yield_equal_decisions(self):
        supervisor = SafetySupervisor()
        first = self.run_stream(supervisor)
        supervisor.reset()
        second = self.run_stream(supervisor)
        self.assertEqual(first, second)
        self.assertTrue(any(d.autonomy_allowed for d in first))
        self.assertTrue(any(not d.autonomy_allowed for d in first))

    def test_independent_supervisors_remain_equivalent(self):
        left = SafetySupervisor()
        right = SafetySupervisor()
        self.assertEqual(self.run_stream(left), self.run_stream(right))
        self.assertEqual(left.metrics, right.metrics)
        self.assertEqual(left.latest_decision, right.latest_decision)


class SafetyBoundednessTest(unittest.TestCase):
    def test_supervisor_creates_no_thread_and_keeps_o1_state(self):
        threads_before = set(threading.enumerate())
        supervisor = SafetySupervisor()
        for frame in range(1, 501):
            now = frame / 10.0
            supervisor.update(
                result(frame, DETECTED, monotonic=now), now=now, health=RUNNING
            )
            supervisor.evaluate(now=now, health=RUNNING)
        self.assertEqual(set(threading.enumerate()), threads_before)
        slots = set(SafetySupervisor.__slots__)
        self.assertEqual(len(slots), len(SafetySupervisor.__slots__))
        for name in slots:
            value = getattr(supervisor, name)
            self.assertNotIsInstance(value, (list, dict, set, tuple, bytearray))
        self.assertEqual(supervisor.metrics.new_results_accepted, 500)
        self.assertEqual(supervisor.metrics.evaluations, 1000)

    def test_no_control_surface_is_exposed(self):
        forbidden = (
            "apply_control",
            "steer",
            "throttle",
            "brake",
            "control",
            "actor",
            "vehicle",
        )
        names = tuple(dir(SafetySupervisor)) + tuple(dir(SafetyDecision))
        for name in names:
            for token in forbidden:
                self.assertNotIn(token, name.lower(), name)


if __name__ == "__main__":
    unittest.main()
