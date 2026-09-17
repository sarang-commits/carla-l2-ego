"""Thread-safe aggregation of telemetry from sensor callbacks and world ticks.

Design rules honored here:
  * Exactly one RLock guards all shared state.
  * Callers parse CARLA data *outside* the lock and hand in finished records.
  * The lock is held only while replacing state or building a snapshot; no
    CARLA API or user callback is ever invoked under the lock.
  * A callback gate (`_accepting`) drops every update once shutdown begins.
  * Only the latest value per source is kept -- no unbounded history/queues.
  * `revision` increases monotonically on every accepted update.
"""

from __future__ import annotations

import collections
import math
import numbers
import threading
import time
from dataclasses import replace

from telemetry.state import (
    CameraFrame,
    CameraFrameMetadata,
    CollisionEvent,
    EgoKinematics,
    GnssState,
    ImuState,
    LaneInvasionEvent,
    SampleStamp,
    SensorHealth,
    TelemetrySnapshot,
)

# Bounded telemetry: no history or cache is ever allowed to grow without limit.
COLLISION_HISTORY = 128
LANE_HISTORY = 128
LANE_DEBOUNCE_CACHE = 32
LANE_DEBOUNCE_SECONDS = 0.25
ERROR_TEXT_LIMIT = 192


def _initial_health(name: str) -> SensorHealth:
    return SensorHealth(
        name=name,
        attached=False,
        messages_received=0,
        error_count=0,
    )


def _collision_signature(event: CollisionEvent):
    """Content identity of a collision, excluding host receive time.

    Two callbacks for the same physical contact (same frame, actors, impulse,
    transform) share a signature and are treated as exact duplicates; anything
    that differs -- other actor, impulse, or frame -- is a distinct event.
    """
    return (
        event.stamp.frame,
        event.self_actor,
        event.other_actor,
        event.normal_impulse,
        event.transform,
    )


class TelemetryAggregator:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._accepting = True
        self._revision = 0
        self._ego: EgoKinematics | None = None
        self._gnss: GnssState | None = None
        self._imu: ImuState | None = None
        self._health: dict[str, SensorHealth] = {}
        # Bounded event histories (deque drops the oldest past maxlen).
        self._collisions: collections.deque = collections.deque(maxlen=COLLISION_HISTORY)
        self._lane_invasions: collections.deque = collections.deque(maxlen=LANE_HISTORY)
        # Debounce state for lane invasions: signature -> last accepted sim time.
        self._lane_debounce: "collections.OrderedDict" = collections.OrderedDict()
        self._lane_last_sim_time: float | None = None
        # Latest camera frame only -- no queue, history, or backlog.
        self._camera: CameraFrame | None = None
        # Primitive-only ordering evidence is kept separately so comparisons
        # under the lock never dispatch through caller-provided scalar hooks.
        self._camera_order: tuple[int, float] | None = None
        # Monotonic accepted-event counters (Mission 4F preparation).
        self._collision_sequence = 0
        self._lane_invasion_sequence = 0

    # -- registration / lifecycle ------------------------------------------

    def register(self, name: str) -> None:
        with self._lock:
            self._health.setdefault(name, _initial_health(name))

    def mark_attached(
        self, name: str, attached: bool, attachment_monotonic: float | None = None
    ) -> None:
        if attachment_monotonic is None:
            attachment_monotonic = time.monotonic()
        attachment_monotonic = _finite_timestamp(
            attachment_monotonic, "attachment monotonic timestamp"
        )
        with self._lock:
            current = self._health.get(name) or _initial_health(name)
            self._health[name] = replace(
                current,
                attached=bool(attached),
                attach_attempted=True,
                attachment_failed=not bool(attached),
                attachment_monotonic=attachment_monotonic,
            )

    def begin_shutdown(self) -> None:
        """Close the callback gate: subsequent updates are ignored."""
        with self._lock:
            self._accepting = False

    @property
    def accepting(self) -> bool:
        with self._lock:
            return self._accepting

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    # -- accepted updates (records are already parsed by the caller) -------

    def update_ego(self, ego: EgoKinematics) -> None:
        with self._lock:
            if not self._accepting:
                return
            self._ego = ego
            self._revision += 1

    def mark_vehicle_missing(self, stamp: SampleStamp) -> None:
        """Flag the ego as no longer alive without inventing kinematics."""
        with self._lock:
            if not self._accepting:
                return
            if self._ego is not None and self._ego.vehicle_alive:
                self._ego = replace(self._ego, stamp=stamp, vehicle_alive=False)
                self._revision += 1

    def submit_gnss(self, state: GnssState) -> bool:
        with self._lock:
            if not self._accepting:
                return False
            if not self._accept_continuous_stamp_locked("gnss", state.stamp):
                return False
            self._gnss = state
            self._record_message_locked("gnss", state.stamp)
            self._revision += 1
            return True

    def submit_imu(self, state: ImuState) -> bool:
        with self._lock:
            if not self._accepting:
                return False
            if not self._accept_continuous_stamp_locked("imu", state.stamp):
                return False
            self._imu = state
            self._record_message_locked("imu", state.stamp)
            self._revision += 1
            return True

    def submit_collision(self, event: CollisionEvent) -> bool:
        """Publish a collision, suppressing exact consecutive duplicates.

        Parsing happens outside this lock; only the append/dedup/revision is
        atomic here. Returns whether the event was stored.
        """
        with self._lock:
            if not self._accepting:                  # recheck gate at publish
                return False
            if not self._accept_valid_stamp_locked("collision", event.stamp):
                return False
            if self._collisions and _collision_signature(
                self._collisions[-1]
            ) == _collision_signature(event):
                return False                         # exact duplicate suppressed
            self._collisions.append(event)
            self._collision_sequence += 1
            self._record_message_locked("collision", event.stamp)
            self._revision += 1
            return True

    def submit_lane_invasion(self, event: LaneInvasionEvent) -> bool:
        """Publish a lane invasion, debouncing identical signatures.

        Identical crossed-marking signatures within LANE_DEBOUNCE_SECONDS of
        simulation time are dropped. If simulation time moves backwards the
        debounce state is cleared. Returns whether the event was stored.
        """
        with self._lock:
            if not self._accepting:                  # recheck gate at publish
                return False
            if not self._accept_valid_stamp_locked("lane_invasion", event.stamp):
                return False

            sim_time = event.stamp.sim_time
            if self._lane_last_sim_time is not None and sim_time < self._lane_last_sim_time:
                self._lane_debounce.clear()          # time rolled back -> reset
            self._lane_last_sim_time = sim_time

            signature = event.crossed_markings
            last_seen = self._lane_debounce.get(signature)
            if last_seen is not None and (sim_time - last_seen) < LANE_DEBOUNCE_SECONDS:
                return False                         # debounced

            self._lane_debounce[signature] = sim_time
            self._lane_debounce.move_to_end(signature)
            while len(self._lane_debounce) > LANE_DEBOUNCE_CACHE:
                self._lane_debounce.popitem(last=False)  # evict oldest

            self._lane_invasions.append(event)
            self._lane_invasion_sequence += 1
            self._record_message_locked("lane_invasion", event.stamp)
            self._revision += 1
            return True

    def submit_camera_frame(self, frame: CameraFrame) -> bool:
        """Publish the latest camera frame; only strictly newer frames win.

        Conversion happens outside this lock. The gate recheck, frame-number
        comparison, replacement, health update, and revision bump are one
        atomic step here. The first valid frame is accepted; afterwards a
        frame is accepted only if its source frame number is strictly higher,
        so duplicates and older frames never displace the last valid frame.
        Returns whether the frame was stored.
        """
        if not self.accepting:
            return False
        stamp, problem, error_monotonic = _prepare_camera_submission(frame)
        with self._lock:
            return self._submit_camera_frame_locked(
                frame, stamp, problem, error_monotonic
            )

    def submit_camera_frame_and_snapshot(
        self, frame: CameraFrame
    ) -> TelemetrySnapshot | None:
        """Atomically publish ``frame`` and capture its exact snapshot.

        ``None`` means the frame was rejected by the existing shutdown,
        validity, duplicate, or source-order rules.  On acceptance, camera
        publication, health accounting, the revision increment, and snapshot
        construction all occur under the same lock.  A concurrent newer
        publication therefore cannot replace this callback's camera before
        its snapshot is built.

        The returned record is immutable and this method invokes no consumer;
        callers may safely hand it to external code only after this method has
        returned and the aggregator lock has been released.
        """
        if not self.accepting:
            return None
        stamp, problem, error_monotonic = _prepare_camera_submission(frame)
        with self._lock:
            if not self._submit_camera_frame_locked(
                frame, stamp, problem, error_monotonic
            ):
                return None
            # The camera receive timestamp is already a canonical finite
            # monotonic value in the same clock domain used by live wiring.
            # Reusing it avoids another clock read while holding the lock.
            assert stamp is not None
            captured_monotonic = stamp.monotonic
            return self._snapshot_locked(captured_monotonic)

    def record_error(
        self,
        name: str,
        message: str,
        error_monotonic: float | None = None,
        reason: str = "callback error",
    ) -> None:
        if error_monotonic is None or not _is_finite_timestamp(error_monotonic):
            error_monotonic = time.monotonic()
        with self._lock:
            if not self._accepting:
                return
            self._record_error_locked(name, message, error_monotonic, reason)

    # -- snapshot ----------------------------------------------------------

    def snapshot(self) -> TelemetrySnapshot:
        """Build an immutable snapshot. Makes no CARLA calls."""
        with self._lock:
            return self._snapshot_locked(time.monotonic())

    def raw_snapshot(self) -> TelemetrySnapshot:
        """Build raw immutable evidence without reading any clock."""
        with self._lock:
            return self._snapshot_locked(0.0)

    # -- internals (caller already holds the lock) -------------------------

    def _submit_camera_frame_locked(
        self,
        frame: CameraFrame,
        stamp: SampleStamp | None,
        problem: str | None,
        error_monotonic: float | None,
    ) -> bool:
        if not self._accepting:                      # recheck gate at publish
            return False
        if problem is not None:
            assert error_monotonic is not None
            self._record_error_locked(
                "rgb_camera",
                problem,
                error_monotonic,
                "invalid sample timestamp",
            )
            return False
        assert stamp is not None
        incoming_order = (stamp.frame, stamp.sim_time)
        if self._camera_order is not None:
            previous_frame, previous_sim_time = self._camera_order
            duplicate = (
                stamp.frame == previous_frame
                and stamp.sim_time == previous_sim_time
            )
            regressed = (
                stamp.frame <= previous_frame
                or stamp.sim_time < previous_sim_time
            )
            if duplicate:
                return False
            if regressed:
                self._record_error_locked(
                    "rgb_camera",
                    "camera source frame or timestamp regressed",
                    stamp.monotonic,
                    "source order regression",
                )
                return False
        self._camera = frame                         # latest frame only
        self._camera_order = incoming_order
        self._record_message_locked("rgb_camera", stamp)
        self._revision += 1
        return True

    def _record_message_locked(self, name: str, stamp: SampleStamp) -> None:
        current = self._health.get(name) or _initial_health(name)
        self._health[name] = replace(
            current,
            messages_received=current.messages_received + 1,
            last_frame=stamp.frame,
            last_sim_time=stamp.sim_time,
            last_receive_monotonic=stamp.monotonic,
            active_error=False,
            active_error_reason=None,
        )

    def _accept_valid_stamp_locked(self, name: str, stamp: SampleStamp) -> bool:
        problem = _stamp_problem(stamp)
        if problem is None:
            return True
        self._record_error_locked(
            name,
            problem,
            time.monotonic(),
            "invalid sample timestamp",
        )
        return False

    def _accept_continuous_stamp_locked(self, name: str, stamp: SampleStamp) -> bool:
        if not self._accept_valid_stamp_locked(name, stamp):
            return False
        current = self._health.get(name) or _initial_health(name)
        if current.messages_received == 0:
            return True
        assert current.last_frame is not None and current.last_sim_time is not None
        if stamp.frame > current.last_frame and stamp.sim_time >= current.last_sim_time:
            return True
        if stamp.frame == current.last_frame and stamp.sim_time == current.last_sim_time:
            return False
        self._record_error_locked(
            name,
            "continuous sensor source frame or timestamp regressed",
            stamp.monotonic,
            "source order regression",
        )
        return False

    def _record_error_locked(
        self, name: str, message: str, error_monotonic: float, reason: str
    ) -> None:
        current = self._health.get(name) or _initial_health(name)
        self._health[name] = replace(
            current,
            error_count=current.error_count + 1,
            last_error=_bounded_text(message),
            last_error_monotonic=_finite_timestamp(
                error_monotonic, "callback error monotonic timestamp"
            ),
            active_error=True,
            active_error_reason=_bounded_text(reason),
        )

    def _snapshot_locked(self, captured_monotonic: float) -> TelemetrySnapshot:
        sensors = tuple(self._health[name] for name in sorted(self._health))
        return TelemetrySnapshot(
            revision=self._revision,
            captured_monotonic=captured_monotonic,
            ego=self._ego,
            gnss=self._gnss,
            imu=self._imu,
            sensors=sensors,
            collisions=tuple(self._collisions),
            lane_invasions=tuple(self._lane_invasions),
            camera=self._camera,
            collision_sequence=self._collision_sequence,
            lane_invasion_sequence=self._lane_invasion_sequence,
        )


def _bounded_text(value) -> str:
    text = " ".join(str(value).split()) or "<no detail>"
    return text if len(text) <= ERROR_TEXT_LIMIT else text[: ERROR_TEXT_LIMIT - 3] + "..."


def _is_finite_timestamp(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, numbers.Real)
        and math.isfinite(float(value))
    )


def _finite_timestamp(value, name: str) -> float:
    if not _is_finite_timestamp(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _stamp_problem(stamp: SampleStamp) -> str | None:
    if not _is_finite_timestamp(stamp.sim_time):
        return "sample simulation timestamp is not finite"
    if not _is_finite_timestamp(stamp.monotonic):
        return "sample receive monotonic timestamp is not finite"
    return None


def _prepare_camera_submission(
    frame: CameraFrame,
) -> tuple[SampleStamp | None, str | None, float | None]:
    """Copy primitive camera ordering evidence before taking the state lock.

    The accepted snapshot still retains the exact input ``CameraFrame`` and
    its exact source stamp.  This canonical copy exists only for locked
    validation, health accounting, ordering, and capture time, so hostile
    attribute or numeric hooks can never execute while the aggregator lock is
    held.
    """
    if not isinstance(frame, CameraFrame):
        raise TypeError("frame must be a canonical CameraFrame")
    metadata = frame.metadata
    if not isinstance(metadata, CameraFrameMetadata):
        raise TypeError("frame must contain canonical CameraFrameMetadata")
    source = metadata.stamp
    if not isinstance(source, SampleStamp):
        raise TypeError("camera metadata must contain a canonical SampleStamp")

    source_frame = source.frame
    raw_sim_time = source.sim_time
    raw_monotonic = source.monotonic
    if (
        isinstance(source_frame, bool)
        or type(source_frame) is not int
        or source_frame < 0
    ):
        return (
            None,
            "camera source frame must be a non-negative built-in integer",
            _camera_error_monotonic(raw_monotonic),
        )

    sim_time = _canonical_finite_timestamp(raw_sim_time)
    if sim_time is None:
        return (
            None,
            "sample simulation timestamp is not finite",
            _camera_error_monotonic(raw_monotonic),
        )
    monotonic = _canonical_finite_timestamp(raw_monotonic)
    if monotonic is None:
        return (
            None,
            "sample receive monotonic timestamp is not finite",
            _finite_timestamp(
                time.monotonic(), "callback error monotonic timestamp"
            ),
        )
    return (
        SampleStamp(
            frame=source_frame,
            sim_time=sim_time,
            monotonic=monotonic,
        ),
        None,
        None,
    )


def _canonical_finite_timestamp(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _camera_error_monotonic(value) -> float:
    canonical = _canonical_finite_timestamp(value)
    if canonical is not None:
        return canonical
    return _finite_timestamp(time.monotonic(), "callback error monotonic timestamp")
