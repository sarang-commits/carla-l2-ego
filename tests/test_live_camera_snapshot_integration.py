"""Focused offline tests for Mission 6B accepted-camera integration.

These tests use only CARLA-shaped fakes.  They never create a client, connect
to a server, start a perception worker, or retain an unbounded frame history.
"""

from __future__ import annotations

import fractions
import gc
import threading
import time
import unittest
import weakref
from unittest import mock

import sensors.suite as suite_module
from perception.live_bridge import PerceptionLiveBridge
from perception.lane import perceive_lanes
from perception.state import LaneObservationReason
from sensors.gnss import GNSS_BLUEPRINT
from sensors.rgb_camera import RGB_CAMERA_BLUEPRINT, RgbCameraSensor
from sensors.suite import SensorSuite, SensorSuiteConfig
from telemetry.aggregator import TelemetryAggregator
from telemetry.freshness import (
    ContinuousFreshnessPolicy,
    OperationalHealth,
    SensorFreshnessConfig,
    SensorFreshness,
    SensorKind,
    SensorLifecycle,
    SensorReadiness,
)
from telemetry.state import (
    CameraFrame,
    CameraFrameMetadata,
    SampleStamp,
    TelemetrySnapshot,
)
from tests.fakes import FakeGnssMeasurement, FakeTransform, FakeVehicle, FakeWorld
from tests.test_rgb_camera import camera_image


WAIT_SECONDS = 2.0
ENABLED_SENSOR_NAMES = (
    "gnss",
    "imu",
    "collision",
    "lane_invasion",
    "rgb_camera",
)


def mark_suite_attached(suite: SensorSuite) -> None:
    """Publish trusted attached evidence without constructing CARLA actors."""
    for name in ENABLED_SENSOR_NAMES:
        suite._aggregator.register(name)
        suite._aggregator.mark_attached(name, True, 0.0)


def rollback_reason_count(snapshot: TelemetrySnapshot) -> int:
    return sum(
        status.reason == "host monotonic clock rollback"
        for status in snapshot.statuses
    )


class ReverseCompletionClock:
    """Assign increasing samples while forcing the first call to finish last."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = 0
        self._armed = False
        self.first_assigned = threading.Event()
        self.release_first = threading.Event()
        self.assignments: list[tuple[str, float]] = []

    def arm(self) -> None:
        with self._lock:
            self._armed = True
            self.assignments.clear()

    def __call__(self) -> float:
        with self._lock:
            self._next += 1
            value = float(self._next)
            should_block = self._armed and not self.assignments
            if self._armed:
                self.assignments.append((threading.current_thread().name, value))
        if should_block:
            self.first_assigned.set()
            if not self.release_first.wait(WAIT_SECONDS):
                raise RuntimeError("reverse-completion clock was not released")
        return value


def make_frame(
    frame_number: int,
    *,
    sim_time=None,
    receive_monotonic=None,
    stamp_type=SampleStamp,
) -> CameraFrame:
    sim_time = float(frame_number) if sim_time is None else sim_time
    receive_monotonic = (
        float(frame_number)
        if receive_monotonic is None
        else receive_monotonic
    )
    stamp = stamp_type(
        frame=frame_number,
        sim_time=sim_time,
        monotonic=receive_monotonic,
    )
    return CameraFrame(
        metadata=CameraFrameMetadata(
            stamp=stamp,
            width_px=1,
            height_px=1,
            horizontal_fov_deg=90.0,
            pixel_format="RGB8",
            row_stride_bytes=3,
        ),
        rgb_bytes=b"\x01\x02\x03",
    )


def attach_suite(consumer, **world_options):
    world = FakeWorld(**world_options)
    suite = SensorSuite(camera_snapshot_consumer=consumer)
    report = suite.attach(world, FakeVehicle())
    if not report.all_attached:
        suite.destroy()
        raise AssertionError(f"fake suite did not attach: {report!r}")
    return world, suite


class AtomicAcceptedCameraSnapshotTest(unittest.TestCase):
    def test_acceptance_returns_exact_frame_and_source_stamp(self):
        aggregator = TelemetryAggregator()
        aggregator.register("rgb_camera")
        frame = make_frame(41, sim_time=12.75, receive_monotonic=99.25)
        source_stamp = frame.metadata.stamp

        snapshot = aggregator.submit_camera_frame_and_snapshot(frame)

        self.assertIsInstance(snapshot, TelemetrySnapshot)
        assert snapshot is not None
        self.assertIs(snapshot.camera, frame)
        self.assertIs(snapshot.camera.metadata.stamp, source_stamp)
        self.assertEqual(snapshot.camera.metadata.stamp.frame, 41)
        self.assertEqual(snapshot.camera.metadata.stamp.sim_time, 12.75)
        self.assertEqual(snapshot.captured_monotonic, 99.25)
        health = {value.name: value for value in snapshot.sensors}["rgb_camera"]
        self.assertEqual(health.messages_received, 1)
        self.assertEqual(health.last_frame, 41)
        self.assertEqual(health.last_sim_time, 12.75)

    def test_rejections_return_no_snapshot_and_keep_prior_camera(self):
        aggregator = TelemetryAggregator()
        first = make_frame(10, sim_time=1.0)
        accepted = aggregator.submit_camera_frame_and_snapshot(first)
        assert accepted is not None

        self.assertIsNone(
            aggregator.submit_camera_frame_and_snapshot(
                make_frame(10, sim_time=1.0, receive_monotonic=100.0)
            )
        )
        self.assertIsNone(
            aggregator.submit_camera_frame_and_snapshot(
                make_frame(9, sim_time=0.9)
            )
        )
        self.assertIs(aggregator.raw_snapshot().camera, first)

        aggregator.begin_shutdown()
        self.assertIsNone(
            aggregator.submit_camera_frame_and_snapshot(make_frame(11))
        )
        self.assertIs(aggregator.raw_snapshot().camera, first)

    def test_later_publication_cannot_replace_camera_in_prior_snapshot(self):
        aggregator = TelemetryAggregator()
        first = make_frame(10, sim_time=0.1)
        second = make_frame(11, sim_time=0.2)

        first_snapshot = aggregator.submit_camera_frame_and_snapshot(first)
        second_snapshot = aggregator.submit_camera_frame_and_snapshot(second)

        assert first_snapshot is not None and second_snapshot is not None
        self.assertIs(first_snapshot.camera, first)
        self.assertIs(second_snapshot.camera, second)
        self.assertIs(aggregator.raw_snapshot().camera, second)
        # Whole-snapshot revision is deliberately not camera identity.
        self.assertEqual(first_snapshot.revision, 1)
        self.assertNotEqual(
            first_snapshot.revision, first.metadata.stamp.frame
        )

    def test_concurrent_returns_are_always_associated_with_their_input(self):
        for _ in range(25):
            aggregator = TelemetryAggregator()
            barrier = threading.Barrier(2)
            outcomes = []
            outcomes_lock = threading.Lock()

            def publish(frame):
                barrier.wait()
                result = aggregator.submit_camera_frame_and_snapshot(frame)
                with outcomes_lock:
                    outcomes.append((frame, result))

            inputs = (make_frame(100), make_frame(101))
            threads = [
                threading.Thread(target=publish, args=(value,))
                for value in inputs
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(WAIT_SECONDS)
                self.assertFalse(thread.is_alive())

            accepted = 0
            for frame, snapshot in outcomes:
                if snapshot is not None:
                    accepted += 1
                    self.assertIs(snapshot.camera, frame)
                    self.assertIs(
                        snapshot.camera.metadata.stamp,
                        frame.metadata.stamp,
                    )
            self.assertGreaterEqual(accepted, 1)
            self.assertEqual(
                aggregator.raw_snapshot().camera.metadata.stamp.frame, 101
            )

    def test_hostile_stamp_and_numeric_hooks_never_run_under_lock(self):
        aggregator = TelemetryAggregator()
        under_lock = threading.Event()

        def record_context():
            owned = getattr(aggregator._lock, "_is_owned", lambda: False)()
            if owned:
                under_lock.set()

        class HookStamp(SampleStamp):
            __slots__ = ()

            def __getattribute__(self, name):
                if name in {"frame", "sim_time", "monotonic"}:
                    record_context()
                return object.__getattribute__(self, name)

        class HookReal(fractions.Fraction):
            def __float__(self):
                record_context()
                return super().__float__()

            def __eq__(self, other):
                record_context()
                return super().__eq__(other)

            def __lt__(self, other):
                record_context()
                return super().__lt__(other)

            def __le__(self, other):
                record_context()
                return super().__le__(other)

        first = make_frame(
            1,
            sim_time=HookReal(1, 2),
            receive_monotonic=HookReal(3, 2),
            stamp_type=HookStamp,
        )
        first_stamp = first.metadata.stamp
        second = make_frame(
            2,
            sim_time=HookReal(3, 2),
            receive_monotonic=HookReal(5, 2),
            stamp_type=HookStamp,
        )

        first_snapshot = aggregator.submit_camera_frame_and_snapshot(first)
        second_snapshot = aggregator.submit_camera_frame_and_snapshot(second)

        assert first_snapshot is not None and second_snapshot is not None
        self.assertIs(first_snapshot.camera, first)
        self.assertIs(first_snapshot.camera.metadata.stamp, first_stamp)
        self.assertFalse(under_lock.is_set())
        health = {item.name: item for item in second_snapshot.sensors}["rgb_camera"]
        self.assertIs(type(health.last_sim_time), float)
        self.assertIs(type(health.last_receive_monotonic), float)

    def test_reentrant_numeric_hook_cannot_deadlock_or_overwrite_newer_frame(self):
        aggregator = TelemetryAggregator()
        nested = make_frame(2, sim_time=2.0)

        class ReentrantReal(fractions.Fraction):
            fired = False

            def __float__(self):
                if not self.fired:
                    type(self).fired = True
                    nested_snapshot = aggregator.submit_camera_frame_and_snapshot(
                        nested
                    )
                    self_test.assertIsNotNone(nested_snapshot)
                return super().__float__()

        self_test = self
        outer = make_frame(1, sim_time=ReentrantReal(1, 1))
        self.assertIsNone(aggregator.submit_camera_frame_and_snapshot(outer))
        self.assertIs(aggregator.raw_snapshot().camera, nested)


class CameraSnapshotConsumerWiringTest(unittest.TestCase):
    def test_post_attach_consumer_snapshot_has_one_usable_camera_status(self):
        received = []
        world, suite = attach_suite(lambda snapshot: received.append(snapshot))
        try:
            world.spawned[RGB_CAMERA_BLUEPRINT].emit(
                camera_image(frame=23, timestamp=7.5)
            )
            self.assertEqual(len(received), 1)
            snapshot = received[0]
            camera_statuses = tuple(
                status
                for status in snapshot.statuses
                if status.name == "rgb_camera"
            )
            self.assertEqual(len(camera_statuses), 1)
            status = camera_statuses[0]
            self.assertIs(status.kind, SensorKind.CONTINUOUS)
            self.assertIs(status.lifecycle, SensorLifecycle.ATTACHED)
            self.assertIs(status.readiness, SensorReadiness.READY)
            self.assertIs(status.freshness, SensorFreshness.FRESH)
            self.assertIs(
                status.operational_health, OperationalHealth.HEALTHY
            )
            self.assertEqual(status.accepted_count, 1)
            self.assertEqual(status.error_count, 0)
            self.assertIsNone(status.event_count)
            self.assertIsNotNone(status.age_seconds)
            self.assertLessEqual(status.age_seconds, status.stale_after)
            self.assertEqual(snapshot.camera.metadata.stamp.frame, 23)
            self.assertEqual(snapshot.camera.metadata.stamp.sim_time, 7.5)
            # The real Mission 5 gate must advance beyond status rejection.
            observation = perceive_lanes(snapshot)
            self.assertIs(
                observation.reason,
                LaneObservationReason.NO_PLAUSIBLE_BOUNDARY,
            )
        finally:
            suite.destroy()

    def test_real_bridge_receives_callback_snapshot_on_shared_clock_basis(self):
        class SharedClock:
            def __init__(self):
                self._lock = threading.Lock()
                self._value = 10.0

            def __call__(self):
                with self._lock:
                    self._value += 0.01
                    return self._value

        class Runtime:
            def __init__(self):
                self.received = []

            def submit(self, snapshot):
                self.received.append(snapshot)
                return True

        clock = SharedClock()
        runtime = Runtime()
        bridge = PerceptionLiveBridge(runtime, clock=clock)
        world = FakeWorld()
        suite = SensorSuite(
            monotonic_clock=clock,
            camera_snapshot_consumer=bridge,
        )
        report = suite.attach(world, FakeVehicle())
        self.assertTrue(report.all_attached)
        try:
            world.spawned[RGB_CAMERA_BLUEPRINT].emit(
                camera_image(frame=17, timestamp=4.25)
            )
            bridge.close(0.1)
        finally:
            suite.destroy()

        self.assertEqual(len(runtime.received), 1)
        submitted = runtime.received[0]
        self.assertEqual(submitted.camera.metadata.stamp.frame, 17)
        self.assertEqual(submitted.camera.metadata.stamp.sim_time, 4.25)
        self.assertEqual(
            bridge.metrics.last_submitted_source_stamp,
            submitted.camera.metadata.stamp,
        )
        self.assertAlmostEqual(
            bridge.metrics.last_callback_to_submit_latency_seconds,
            0.02,
        )

    def test_no_consumer_uses_original_publication_path(self):
        class AggregatorSpy:
            def __init__(self):
                self.original_calls = 0

            def submit_camera_frame(self, _frame):
                self.original_calls += 1
                return True

            def submit_camera_frame_and_snapshot(self, _frame):
                raise AssertionError("atomic snapshot path should not be used")

        aggregator = AggregatorSpy()
        sensor = RgbCameraSensor(aggregator)
        self.assertTrue(sensor.submit(make_frame(1)))
        self.assertEqual(aggregator.original_calls, 1)

    def test_consumer_receives_only_accepted_exact_source_identity(self):
        class Bridge:
            def __init__(self):
                self.received = []

            def submit(self, snapshot):
                stamp = snapshot.camera.metadata.stamp
                self.received.append((snapshot, stamp.frame, stamp.sim_time))
                return True

        bridge = Bridge()
        world, suite = attach_suite(bridge)
        actor = world.spawned[RGB_CAMERA_BLUEPRINT]
        retained_callback = actor._callback
        try:
            actor.emit(camera_image(frame=5, timestamp=1.25))
            actor.emit(camera_image(frame=5, timestamp=1.25))
            actor.emit(camera_image(frame=4, timestamp=1.0))
            self.assertEqual(
                [(frame, stamp) for _snapshot, frame, stamp in bridge.received],
                [(5, 1.25)],
            )
            delivered = bridge.received[0][0]
            self.assertEqual(delivered.camera.metadata.stamp.frame, 5)
            self.assertEqual(delivered.camera.metadata.stamp.sim_time, 1.25)
        finally:
            suite.destroy()

        retained_callback(camera_image(frame=99, timestamp=99.0))
        self.assertEqual(len(bridge.received), 1)

    def test_blocked_consumer_does_not_block_concurrent_newer_publication(self):
        first_entered = threading.Event()
        second_entered = threading.Event()
        release_first = threading.Event()
        received = {}
        received_lock = threading.Lock()

        def consume(snapshot):
            frame = snapshot.camera.metadata.stamp.frame
            with received_lock:
                received[frame] = snapshot
            if frame == 1:
                first_entered.set()
                if not release_first.wait(WAIT_SECONDS):
                    raise RuntimeError("test did not release first consumer")
            elif frame == 2:
                second_entered.set()
            return True

        world, suite = attach_suite(consume)
        actor = world.spawned[RGB_CAMERA_BLUEPRINT]
        first_thread = threading.Thread(
            target=actor.emit,
            args=(camera_image(frame=1, timestamp=1.0),),
        )
        second_thread = threading.Thread(
            target=actor.emit,
            args=(camera_image(frame=2, timestamp=2.0),),
        )
        try:
            first_thread.start()
            self.assertTrue(first_entered.wait(WAIT_SECONDS))
            second_thread.start()
            self.assertTrue(
                second_entered.wait(WAIT_SECONDS),
                "newer callback was blocked by a lock held across consumer code",
            )
        finally:
            release_first.set()
            first_thread.join(WAIT_SECONDS)
            second_thread.join(WAIT_SECONDS)
            suite.destroy()
        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())
        self.assertEqual(set(received), {1, 2})
        self.assertEqual(received[1].camera.metadata.stamp.frame, 1)
        self.assertEqual(received[2].camera.metadata.stamp.frame, 2)

    def test_reentrant_consumer_preserves_each_callback_association(self):
        received = []

        class ReentrantBridge:
            actor = None

            def submit(self, snapshot):
                frame = snapshot.camera.metadata.stamp.frame
                received.append((frame, snapshot))
                if frame == 1:
                    self.actor.emit(camera_image(frame=2, timestamp=2.0))
                return True

        bridge = ReentrantBridge()
        world, suite = attach_suite(bridge)
        bridge.actor = world.spawned[RGB_CAMERA_BLUEPRINT]
        try:
            bridge.actor.emit(camera_image(frame=1, timestamp=1.0))
            self.assertEqual([frame for frame, _snapshot in received], [1, 2])
            self.assertEqual(received[0][1].camera.metadata.stamp.frame, 1)
            self.assertEqual(received[1][1].camera.metadata.stamp.frame, 2)
            self.assertEqual(suite.snapshot().camera.metadata.stamp.frame, 2)
        finally:
            suite.destroy()

    def test_consumer_runs_outside_sensor_and_aggregator_locks(self):
        suite_holder = []
        lock_observations = []

        def consume(_snapshot):
            suite = suite_holder[0]
            sensor = suite._camera_sensor
            self.assertIsNotNone(sensor)
            acquired = sensor._snapshot_consumer_lock.acquire(blocking=False)
            lock_observations.append(acquired)
            if acquired:
                sensor._snapshot_consumer_lock.release()
            acquired = suite._camera_snapshot_consumer_lock.acquire(
                blocking=False
            )
            lock_observations.append(acquired)
            if acquired:
                suite._camera_snapshot_consumer_lock.release()
            acquired = suite._snapshot_lock.acquire(blocking=False)
            lock_observations.append(acquired)
            if acquired:
                suite._snapshot_lock.release()
            aggregator_owned = getattr(
                suite._aggregator._lock, "_is_owned", lambda: False
            )()
            lock_observations.append(not aggregator_owned)
            return object()  # hostile truth/equality hooks are never needed

        world, suite = attach_suite(consume)
        suite_holder.append(suite)
        try:
            world.spawned[RGB_CAMERA_BLUEPRINT].emit(camera_image())
            self.assertEqual(lock_observations, [True, True, True, True])
        finally:
            suite.destroy()

    def test_ordinary_hostile_consumer_failure_is_not_camera_parse_error(self):
        class HostileError(RuntimeError):
            def __str__(self):
                raise AssertionError("exception text must not be formatted here")

            def __repr__(self):
                raise AssertionError("exception repr must not be formatted here")

        def fail(_snapshot):
            raise HostileError()

        world, suite = attach_suite(fail)
        try:
            world.spawned[RGB_CAMERA_BLUEPRINT].emit(
                camera_image(frame=7, timestamp=0.7)
            )
            world.spawned[GNSS_BLUEPRINT].emit(
                FakeGnssMeasurement(
                    frame=1,
                    timestamp=0.1,
                    latitude=1.0,
                    longitude=2.0,
                    altitude=3.0,
                    transform=FakeTransform(),
                )
            )
            snapshot = suite.snapshot()
            health = {item.name: item for item in snapshot.sensors}
            self.assertEqual(snapshot.camera.metadata.stamp.frame, 7)
            self.assertEqual(health["rgb_camera"].messages_received, 1)
            self.assertEqual(health["rgb_camera"].error_count, 0)
            self.assertIsNotNone(snapshot.gnss)
        finally:
            suite.destroy()

    def test_immediate_listen_callback_reaches_consumer(self):
        received = []
        world, suite = attach_suite(
            lambda snapshot: received.append(snapshot),
            immediate_emit={
                RGB_CAMERA_BLUEPRINT: camera_image(frame=3, timestamp=0.3)
            },
        )
        try:
            self.assertEqual(
                [snapshot.camera.metadata.stamp.frame for snapshot in received],
                [3],
            )
            immediate_statuses = tuple(
                status
                for status in received[0].statuses
                if status.name == "rgb_camera"
            )
            self.assertEqual(len(immediate_statuses), 1)
            self.assertEqual(immediate_statuses[0].accepted_count, 1)
            self.assertIs(
                immediate_statuses[0].lifecycle,
                SensorLifecycle.NOT_ATTACHED,
            )

            world.spawned[RGB_CAMERA_BLUEPRINT].emit(
                camera_image(frame=4, timestamp=0.4)
            )
            attached_status = next(
                status
                for status in received[1].statuses
                if status.name == "rgb_camera"
            )
            self.assertEqual(attached_status.accepted_count, 2)
            self.assertIs(attached_status.lifecycle, SensorLifecycle.ATTACHED)
            self.assertIs(attached_status.readiness, SensorReadiness.READY)
            self.assertIs(attached_status.freshness, SensorFreshness.FRESH)
        finally:
            suite.destroy()

    def test_suite_owned_decorator_keeps_only_a_weak_suite_reference(self):
        class Bridge:
            def submit(self, _snapshot):
                return True

        bridge = Bridge()
        world = FakeWorld()
        suite = SensorSuite(camera_snapshot_consumer=bridge)
        suite.attach(world, FakeVehicle())
        reference = weakref.ref(suite)

        del suite
        gc.collect()

        self.assertIsNone(reference())

    def test_suite_destruction_releases_bridge_reference(self):
        class Bridge:
            def submit(self, _snapshot):
                return True

        bridge = Bridge()
        reference = weakref.ref(bridge)
        _world, suite = attach_suite(bridge)
        del bridge
        gc.collect()
        self.assertIsNotNone(reference())

        suite.destroy()
        gc.collect()
        self.assertIsNone(reference())

    def test_source_image_and_mutable_buffer_are_not_retained(self):
        received = []
        world, suite = attach_suite(lambda snapshot: received.append(snapshot))
        raw = bytearray([11, 22, 33, 44])
        image = camera_image(frame=1, timestamp=0.1, raw_data=raw)
        image_reference = weakref.ref(image)
        try:
            world.spawned[RGB_CAMERA_BLUEPRINT].emit(image)
            del image
            gc.collect()
            self.assertIsNone(image_reference())
            raw[:] = b"\x00\x00\x00\x00"
            self.assertEqual(received[0].camera.rgb_bytes, bytes([33, 22, 11]))
        finally:
            suite.destroy()


class CameraSnapshotConsumerConfigurationTest(unittest.TestCase):
    def test_disabled_rgb_consumer_fails_before_sensor_or_consumer_resolution(self):
        class Consumer:
            def __init__(self):
                self.submit_lookups = 0

            @property
            def submit(self):
                self.submit_lookups += 1
                return lambda _snapshot: True

        consumer = Consumer()
        consumer_reference = weakref.ref(consumer)
        config = SensorSuiteConfig(enable_rgb_camera=False)

        with mock.patch.object(suite_module, "GnssSensor") as gnss_constructor:
            with self.assertRaisesRegex(
                ValueError, "camera snapshot consumer requires RGB camera"
            ):
                SensorSuite(
                    config=config,
                    camera_snapshot_consumer=consumer,
                )

        gnss_constructor.assert_not_called()
        self.assertEqual(consumer.submit_lookups, 0)
        del consumer
        gc.collect()
        self.assertIsNone(consumer_reference())

    def test_disabled_rgb_without_consumer_remains_valid(self):
        config = SensorSuiteConfig(
            enable_gnss=False,
            enable_imu=False,
            enable_collision=False,
            enable_lane_invasion=False,
            enable_rgb_camera=False,
        )
        suite = SensorSuite(config=config)
        self.assertEqual(suite._sensors, [])
        self.assertIsNone(suite._camera_sensor)
        self.assertIsNone(suite._camera_snapshot_consumer)
        suite.destroy()

    def test_enabled_rgb_with_consumer_remains_valid_and_releases_reference(self):
        class Consumer:
            def submit(self, _snapshot):
                return True

        config = SensorSuiteConfig(
            enable_gnss=False,
            enable_imu=False,
            enable_collision=False,
            enable_lane_invasion=False,
            enable_rgb_camera=True,
        )
        consumer = Consumer()
        reference = weakref.ref(consumer)
        suite = SensorSuite(config=config, camera_snapshot_consumer=consumer)
        self.assertIsNotNone(suite._camera_sensor)
        self.assertEqual(len(suite._sensors), 1)

        del consumer
        gc.collect()
        self.assertIsNotNone(reference())
        suite.destroy()
        gc.collect()
        self.assertIsNone(reference())

    def test_repeated_invalid_construction_retains_no_consumer(self):
        class Consumer:
            def submit(self, _snapshot):
                return True

        config = SensorSuiteConfig(
            enable_gnss=False,
            enable_imu=False,
            enable_collision=False,
            enable_lane_invasion=False,
            enable_rgb_camera=False,
        )
        references = []
        for _ in range(100):
            consumer = Consumer()
            references.append(weakref.ref(consumer))
            with self.assertRaises(ValueError):
                SensorSuite(
                    config=config,
                    camera_snapshot_consumer=consumer,
                )
            del consumer

        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))


class SuiteClockLockSafetyTest(unittest.TestCase):
    def test_injected_clock_runs_before_snapshot_lock(self):
        suite_holder = []
        acquired_observations = []
        values = iter((2.0, 1.0))

        def hostile_clock():
            suite = suite_holder[0]
            acquired = suite._snapshot_lock.acquire(blocking=False)
            acquired_observations.append(acquired)
            if acquired:
                # Reentrant access to trusted aggregator evidence is safe and
                # demonstrates that no suite lock encloses this hook.
                suite._aggregator.raw_snapshot()
                suite._snapshot_lock.release()
            return next(values)

        config = SensorSuiteConfig(
            enable_gnss=False,
            enable_imu=False,
            enable_collision=False,
            enable_lane_invasion=False,
            enable_rgb_camera=False,
        )
        suite = SensorSuite(config=config, monotonic_clock=hostile_clock)
        suite_holder.append(suite)

        first = suite.snapshot()
        second = suite.snapshot()

        self.assertEqual(acquired_observations, [True, True])
        self.assertEqual(first.captured_monotonic, 2.0)
        self.assertEqual(second.captured_monotonic, 2.0)

    def test_forced_reverse_completion_is_stale_overlap_not_rollback(self):
        clock = ReverseCompletionClock()
        suite = SensorSuite(monotonic_clock=clock)
        mark_suite_attached(suite)
        clock.arm()
        snapshots = {}
        errors = []

        def take(label):
            try:
                snapshots[label] = suite.snapshot()
            except BaseException as exc:
                errors.append((label, type(exc).__name__))

        first = threading.Thread(target=take, args=("first",), name="first")
        second = threading.Thread(target=take, args=("second",), name="second")
        first.start()
        self.assertTrue(clock.first_assigned.wait(WAIT_SECONDS))
        second.start()
        second.join(WAIT_SECONDS)
        self.assertFalse(second.is_alive())
        clock.release_first.set()
        first.join(WAIT_SECONDS)

        self.assertFalse(first.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(
            [name for name, _value in clock.assignments],
            ["first", "second"],
        )
        self.assertEqual(rollback_reason_count(snapshots["first"]), 0)
        self.assertEqual(rollback_reason_count(snapshots["second"]), 0)
        self.assertTrue(
            all(
                status.operational_health is OperationalHealth.HEALTHY
                for snapshot in snapshots.values()
                for status in snapshot.statuses
            )
        )

    def test_mirrored_overlap_inverts_reservation_sample_and_commit_order(self):
        class MirroredClock:
            def __init__(self):
                self.older_reserved = threading.Event()
                self.newer_sampled = threading.Event()
                self.release_newer = threading.Event()
                self.reservation_order = []
                self.sample_order = []

            def __call__(self):
                name = threading.current_thread().name
                self.reservation_order.append(name)
                if name == "older-g0":
                    self.older_reserved.set()
                    if not self.newer_sampled.wait(WAIT_SECONDS):
                        raise RuntimeError("newer sample was not captured")
                    self.sample_order.append((name, 6.0))
                    return 6.0
                if name == "newer-g1":
                    self.sample_order.append((name, 1.0))
                    self.newer_sampled.set()
                    if not self.release_newer.wait(WAIT_SECONDS):
                        raise RuntimeError("newer sample was not released")
                    return 1.0
                raise AssertionError(f"unexpected clock caller {name!r}")

        clock = MirroredClock()
        suite = SensorSuite(monotonic_clock=clock)
        mark_suite_attached(suite)
        snapshots = {}
        commit_order = []
        errors = []

        def take(label):
            try:
                snapshots[label] = suite.snapshot()
                commit_order.append(label)
            except BaseException as exc:
                errors.append((label, type(exc).__name__))

        older = threading.Thread(target=take, args=("older-g0",), name="older-g0")
        newer = threading.Thread(target=take, args=("newer-g1",), name="newer-g1")
        older.start()
        self.assertTrue(clock.older_reserved.wait(WAIT_SECONDS))
        newer.start()
        self.assertTrue(clock.newer_sampled.wait(WAIT_SECONDS))
        older.join(WAIT_SECONDS)
        self.assertFalse(older.is_alive())
        clock.release_newer.set()
        newer.join(WAIT_SECONDS)

        self.assertFalse(newer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(clock.reservation_order, ["older-g0", "newer-g1"])
        self.assertEqual(
            clock.sample_order,
            [("newer-g1", 1.0), ("older-g0", 6.0)],
        )
        self.assertEqual(commit_order, ["older-g0", "newer-g1"])
        self.assertEqual(snapshots["older-g0"].captured_monotonic, 6.0)
        self.assertEqual(snapshots["newer-g1"].captured_monotonic, 6.0)
        self.assertEqual(rollback_reason_count(snapshots["older-g0"]), 0)
        self.assertEqual(rollback_reason_count(snapshots["newer-g1"]), 0)
        self.assertTrue(
            all(
                status.operational_health is OperationalHealth.HEALTHY
                for snapshot in snapshots.values()
                for status in snapshot.statuses
            )
        )

    def test_stale_late_sample_advances_floor_for_already_reserved_generation(self):
        class StaleAdvanceClock:
            def __init__(self):
                self.oldest_reserved = threading.Event()
                self.release_oldest = threading.Event()
                self.already_reserved_sampled = threading.Event()
                self.release_already_reserved = threading.Event()
                self.reservation_order = []
                self.sample_order = []

            def __call__(self):
                name = threading.current_thread().name
                self.reservation_order.append(name)
                if name == "oldest-g0":
                    self.oldest_reserved.set()
                    if not self.release_oldest.wait(WAIT_SECONDS):
                        raise RuntimeError("oldest observation was not released")
                    self.sample_order.append((name, 3.0))
                    return 3.0
                if name == "first-commit-g1":
                    self.sample_order.append((name, 2.0))
                    return 2.0
                if name == "reserved-g2":
                    self.sample_order.append((name, 1.0))
                    self.already_reserved_sampled.set()
                    if not self.release_already_reserved.wait(WAIT_SECONDS):
                        raise RuntimeError("reserved observation was not released")
                    return 1.0
                self.sample_order.append((name, 2.0))
                return 2.0

        clock = StaleAdvanceClock()
        suite = SensorSuite(monotonic_clock=clock)
        mark_suite_attached(suite)
        snapshots = {}
        commit_order = []
        errors = []

        def take(label):
            try:
                snapshots[label] = suite.snapshot()
                commit_order.append(label)
            except BaseException as exc:
                errors.append((label, type(exc).__name__))

        oldest = threading.Thread(target=take, args=("oldest-g0",), name="oldest-g0")
        first_commit = threading.Thread(
            target=take,
            args=("first-commit-g1",),
            name="first-commit-g1",
        )
        already_reserved = threading.Thread(
            target=take,
            args=("reserved-g2",),
            name="reserved-g2",
        )
        oldest.start()
        self.assertTrue(clock.oldest_reserved.wait(WAIT_SECONDS))
        first_commit.start()
        first_commit.join(WAIT_SECONDS)
        self.assertFalse(first_commit.is_alive())
        already_reserved.start()
        self.assertTrue(clock.already_reserved_sampled.wait(WAIT_SECONDS))
        clock.release_oldest.set()
        oldest.join(WAIT_SECONDS)
        self.assertFalse(oldest.is_alive())
        clock.release_already_reserved.set()
        already_reserved.join(WAIT_SECONDS)

        self.assertFalse(already_reserved.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(
            clock.reservation_order[:3],
            ["oldest-g0", "first-commit-g1", "reserved-g2"],
        )
        self.assertEqual(
            clock.sample_order[:3],
            [
                ("first-commit-g1", 2.0),
                ("reserved-g2", 1.0),
                ("oldest-g0", 3.0),
            ],
        )
        self.assertEqual(
            commit_order,
            ["first-commit-g1", "oldest-g0", "reserved-g2"],
        )
        self.assertEqual(snapshots["oldest-g0"].captured_monotonic, 3.0)
        self.assertEqual(snapshots["reserved-g2"].captured_monotonic, 3.0)
        self.assertEqual(rollback_reason_count(snapshots["reserved-g2"]), 0)

        genuinely_fresh = suite.snapshot()
        self.assertEqual(genuinely_fresh.captured_monotonic, 3.0)
        self.assertEqual(rollback_reason_count(genuinely_fresh), 5)

    def test_three_generation_permutation_never_infers_sample_order(self):
        class ThreeGenerationClock:
            def __init__(self):
                self.oldest_reserved = threading.Event()
                self.middle_sampled = threading.Event()
                self.newest_sampled = threading.Event()
                self.release_middle = threading.Event()
                self.release_newest = threading.Event()
                self.reservation_order = []
                self.sample_order = []

            def __call__(self):
                name = threading.current_thread().name
                self.reservation_order.append(name)
                if name == "g0":
                    self.oldest_reserved.set()
                    if not self.newest_sampled.wait(WAIT_SECONDS):
                        raise RuntimeError("newest sample was not captured")
                    self.sample_order.append((name, 6.0))
                    return 6.0
                if name == "g1":
                    self.sample_order.append((name, 4.0))
                    self.middle_sampled.set()
                    if not self.release_middle.wait(WAIT_SECONDS):
                        raise RuntimeError("middle sample was not released")
                    return 4.0
                if name == "g2":
                    self.sample_order.append((name, 1.0))
                    self.newest_sampled.set()
                    if not self.release_newest.wait(WAIT_SECONDS):
                        raise RuntimeError("newest sample was not released")
                    return 1.0
                raise AssertionError(f"unexpected clock caller {name!r}")

        clock = ThreeGenerationClock()
        suite = SensorSuite(monotonic_clock=clock)
        mark_suite_attached(suite)
        snapshots = {}
        commit_order = []
        errors = []

        def take(label):
            try:
                snapshots[label] = suite.snapshot()
                commit_order.append(label)
            except BaseException as exc:
                errors.append((label, type(exc).__name__))

        callers = {
            name: threading.Thread(target=take, args=(name,), name=name)
            for name in ("g0", "g1", "g2")
        }
        callers["g0"].start()
        self.assertTrue(clock.oldest_reserved.wait(WAIT_SECONDS))
        callers["g1"].start()
        self.assertTrue(clock.middle_sampled.wait(WAIT_SECONDS))
        callers["g2"].start()
        self.assertTrue(clock.newest_sampled.wait(WAIT_SECONDS))
        callers["g0"].join(WAIT_SECONDS)
        self.assertFalse(callers["g0"].is_alive())
        clock.release_newest.set()
        callers["g2"].join(WAIT_SECONDS)
        self.assertFalse(callers["g2"].is_alive())
        clock.release_middle.set()
        callers["g1"].join(WAIT_SECONDS)

        self.assertFalse(callers["g1"].is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(clock.reservation_order, ["g0", "g1", "g2"])
        self.assertEqual(
            clock.sample_order,
            [("g1", 4.0), ("g2", 1.0), ("g0", 6.0)],
        )
        self.assertEqual(commit_order, ["g0", "g2", "g1"])
        self.assertTrue(
            all(snapshot.captured_monotonic == 6.0 for snapshot in snapshots.values())
        )
        self.assertEqual(
            sum(rollback_reason_count(snapshot) for snapshot in snapshots.values()),
            0,
        )

    def test_real_monotonic_with_bounded_preemption_has_no_false_rollback(self):
        caller_count = 4
        snapshots_per_caller = 1_000
        start = threading.Barrier(caller_count)
        counter_lock = threading.Lock()
        calls = 0

        def clock():
            nonlocal calls
            value = time.monotonic()
            with counter_lock:
                calls += 1
                should_yield = calls % 7 == 0
            if should_yield:
                time.sleep(0)
            return value

        suite = SensorSuite(monotonic_clock=clock)
        mark_suite_attached(suite)
        rollback_counts = [0] * caller_count
        degraded_counts = [0] * caller_count
        errors = []

        def exercise(index):
            try:
                start.wait(WAIT_SECONDS)
                for _ in range(snapshots_per_caller):
                    snapshot = suite.snapshot()
                    rollback_counts[index] += rollback_reason_count(snapshot)
                    degraded_counts[index] += sum(
                        status.operational_health is OperationalHealth.DEGRADED
                        for status in snapshot.statuses
                    )
            except BaseException as exc:
                errors.append((index, type(exc).__name__))

        callers = tuple(
            threading.Thread(target=exercise, args=(index,))
            for index in range(caller_count)
        )
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join(WAIT_SECONDS * 10.0)

        self.assertTrue(all(not caller.is_alive() for caller in callers))
        self.assertEqual(errors, [])
        self.assertEqual(calls, caller_count * snapshots_per_caller)
        self.assertEqual(sum(rollback_counts), 0)
        self.assertEqual(sum(degraded_counts), 0)


class SuiteTrustedConfigurationTest(unittest.TestCase):
    _ENABLED_FIELDS = {
        "enable_gnss",
        "enable_imu",
        "enable_collision",
        "enable_lane_invasion",
        "enable_rgb_camera",
    }

    def test_external_config_attribute_hooks_never_run_during_snapshot(self):
        observations = []
        suite_holder = []

        class HookConfig(SensorSuiteConfig):
            def __getattribute__(self, name):
                if name in SuiteTrustedConfigurationTest._ENABLED_FIELDS:
                    suite = suite_holder[0] if suite_holder else None
                    if suite is not None:
                        acquired = suite._snapshot_lock.acquire(blocking=False)
                        observations.append((name, acquired))
                        if acquired:
                            suite._snapshot_lock.release()
                return super().__getattribute__(name)

        config = HookConfig()
        suite = SensorSuite(config=config)
        suite_holder.append(suite)
        suite.snapshot()
        suite.snapshot()

        self.assertEqual(observations, [])

    def test_external_freshness_policy_hook_never_runs_during_snapshot(self):
        observations = []
        suite_holder = []

        class HookFreshness(SensorFreshnessConfig):
            def policy_for(self, name):
                suite = suite_holder[0] if suite_holder else None
                if suite is not None:
                    acquired = suite._snapshot_lock.acquire(blocking=False)
                    observations.append((name, acquired))
                    if acquired:
                        suite._snapshot_lock.release()
                return super().policy_for(name)

        freshness = HookFreshness()
        suite = SensorSuite(config=SensorSuiteConfig(freshness=freshness))
        suite_holder.append(suite)
        suite.snapshot()
        suite.snapshot()

        self.assertEqual(observations, [])

    def test_reentrant_config_hook_runs_only_outside_snapshot_lock(self):
        suite_holder = []
        armed = threading.Event()
        hook_finished = threading.Event()
        lock_observations = []

        class ReentrantConfig(SensorSuiteConfig):
            def __getattribute__(self, name):
                if (
                    name == "enable_gnss"
                    and armed.is_set()
                    and suite_holder
                ):
                    suite = suite_holder[0]
                    acquired = suite._snapshot_lock.acquire(blocking=False)
                    lock_observations.append(acquired)
                    if acquired:
                        suite._snapshot_lock.release()
                    suite.snapshot()
                    hook_finished.set()
                return super().__getattribute__(name)

        config = ReentrantConfig()
        suite = SensorSuite(config=config)
        suite_holder.append(suite)
        armed.set()
        caller = threading.Thread(
            target=lambda: bool(config.enable_gnss),
            name="reentrant-config-hook",
            daemon=True,
        )
        caller.start()
        caller.join(WAIT_SECONDS)

        self.assertFalse(caller.is_alive())
        self.assertTrue(hook_finished.is_set())
        self.assertEqual(lock_observations, [True])
        lock_observations.clear()
        suite.snapshot()
        self.assertEqual(lock_observations, [])

    def test_base_config_enablement_and_freshness_semantics_are_preserved(self):
        freshness = SensorFreshnessConfig(
            gnss=ContinuousFreshnessPolicy(0.2, 0.6, 1.2),
            imu=ContinuousFreshnessPolicy(0.1, 0.4, 0.8),
            rgb_camera=ContinuousFreshnessPolicy(0.25, 0.75, 1.5),
        )
        config = SensorSuiteConfig(
            enable_gnss=False,
            enable_imu=True,
            enable_collision=False,
            enable_lane_invasion=True,
            enable_rgb_camera=False,
            freshness=freshness,
        )
        suite = SensorSuite(config=config, monotonic_clock=lambda: 0.5)
        snapshot = suite.snapshot()
        statuses = {status.name: status for status in snapshot.statuses}

        self.assertIs(statuses["gnss"].lifecycle, SensorLifecycle.NOT_AVAILABLE)
        self.assertEqual(statuses["gnss"].expected_interval, 0.2)
        self.assertIs(statuses["imu"].lifecycle, SensorLifecycle.NOT_ATTACHED)
        self.assertEqual(statuses["imu"].expected_interval, 0.1)
        self.assertIs(
            statuses["collision"].lifecycle,
            SensorLifecycle.NOT_AVAILABLE,
        )
        self.assertIs(
            statuses["lane_invasion"].lifecycle,
            SensorLifecycle.NOT_ATTACHED,
        )
        self.assertIs(
            statuses["rgb_camera"].lifecycle,
            SensorLifecycle.NOT_AVAILABLE,
        )
        self.assertEqual(statuses["rgb_camera"].expected_interval, 0.25)

    def test_config_subclasses_are_copied_to_exact_snapshot_values(self):
        class PolicySubclass(ContinuousFreshnessPolicy):
            pass

        class FreshnessSubclass(SensorFreshnessConfig):
            pass

        class ConfigSubclass(SensorSuiteConfig):
            pass

        freshness = FreshnessSubclass(
            gnss=PolicySubclass(0.2, 0.6, 1.2),
            imu=PolicySubclass(0.1, 0.4, 0.8),
            rgb_camera=PolicySubclass(0.25, 0.75, 1.5),
        )
        suite = SensorSuite(config=ConfigSubclass(freshness=freshness))

        self.assertEqual(len(suite._snapshot_status_specs), 5)
        for _name, _kind, enabled, policy in suite._snapshot_status_specs:
            self.assertIs(type(enabled), bool)
            if policy is not None:
                self.assertIs(type(policy), ContinuousFreshnessPolicy)


class SuiteClockRecoveryAndStressTest(unittest.TestCase):
    def test_camera_callback_reverse_completion_preserves_identity_and_health(self):
        received = []
        clock = ReverseCompletionClock()
        world = FakeWorld()
        suite = SensorSuite(
            monotonic_clock=clock,
            camera_snapshot_consumer=lambda snapshot: received.append(snapshot),
        )
        report = suite.attach(world, FakeVehicle())
        self.assertTrue(report.all_attached)
        clock.arm()
        ordinary = []
        errors = []

        def take_ordinary_snapshot():
            try:
                ordinary.append(suite.snapshot())
            except BaseException as exc:
                errors.append(type(exc).__name__)

        caller = threading.Thread(
            target=take_ordinary_snapshot,
            name="ordinary-snapshot",
        )
        caller.start()
        self.assertTrue(clock.first_assigned.wait(WAIT_SECONDS))
        world.spawned[RGB_CAMERA_BLUEPRINT].emit(
            camera_image(
                frame=73,
                timestamp=7.3,
                raw_data=bytearray([11, 22, 33, 44]),
            )
        )
        clock.release_first.set()
        caller.join(WAIT_SECONDS)

        self.assertFalse(caller.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(received), 1)
        self.assertEqual(len(ordinary), 1)
        accepted = suite._aggregator.raw_snapshot().camera
        self.assertIs(received[0].camera, accepted)
        self.assertEqual(received[0].camera.metadata.stamp.frame, 73)
        self.assertEqual(received[0].camera.metadata.stamp.sim_time, 7.3)
        self.assertEqual(received[0].camera.rgb_bytes, bytes([33, 22, 11]))
        self.assertEqual(rollback_reason_count(received[0]), 0)
        self.assertEqual(rollback_reason_count(ordinary[0]), 0)
        suite.destroy()

    def test_sequential_clock_regression_is_detected_clamped_and_recovers(self):
        values = iter((10.0, 9.0, 11.0))
        suite = SensorSuite(monotonic_clock=lambda: next(values))
        mark_suite_attached(suite)

        initial = suite.snapshot()
        rolled_back = suite.snapshot()
        recovered = suite.snapshot()

        self.assertEqual(initial.captured_monotonic, 10.0)
        self.assertEqual(rolled_back.captured_monotonic, 10.0)
        self.assertEqual(rollback_reason_count(rolled_back), 5)
        self.assertTrue(
            all(
                status.operational_health is OperationalHealth.DEGRADED
                for status in rolled_back.statuses
            )
        )
        self.assertEqual(recovered.captured_monotonic, 11.0)
        self.assertEqual(rollback_reason_count(recovered), 0)

    def test_reentrant_snapshot_clock_cannot_deadlock_or_poison_ordering(self):
        suite_holder = []
        nested = []
        lock_observations = []
        calls = 0

        def reentrant_clock():
            nonlocal calls
            calls += 1
            if calls == 1:
                suite = suite_holder[0]
                acquired = suite._snapshot_lock.acquire(blocking=False)
                lock_observations.append(acquired)
                if acquired:
                    suite._snapshot_lock.release()
                nested.append(suite.snapshot())
                return 2.0
            if calls == 2:
                return 1.0
            return 3.0

        config = SensorSuiteConfig(
            enable_gnss=False,
            enable_imu=False,
            enable_collision=False,
            enable_lane_invasion=False,
            enable_rgb_camera=False,
        )
        suite = SensorSuite(config=config, monotonic_clock=reentrant_clock)
        suite_holder.append(suite)
        mark_suite_attached(suite)

        outer = suite.snapshot()
        later = suite.snapshot()

        self.assertEqual(lock_observations, [True])
        self.assertEqual(nested[0].captured_monotonic, 1.0)
        self.assertEqual(outer.captured_monotonic, 2.0)
        self.assertEqual(later.captured_monotonic, 3.0)
        self.assertEqual(rollback_reason_count(nested[0]), 0)
        self.assertEqual(rollback_reason_count(outer), 0)
        self.assertEqual(rollback_reason_count(later), 0)

    def test_clock_exception_leaves_no_stranded_ordering_state(self):
        calls = 0

        def clock():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("clock failure")
            return 5.0

        suite = SensorSuite(monotonic_clock=clock)
        with self.assertRaisesRegex(RuntimeError, "clock failure"):
            suite.snapshot()
        self.assertEqual(suite.snapshot().captured_monotonic, 5.0)

    def test_clock_baseexception_propagates_without_stranding_later_call(self):
        class Fatal(BaseException):
            pass

        calls = 0

        def clock():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise Fatal("clock exit")
            return 6.0

        suite = SensorSuite(monotonic_clock=clock)
        with self.assertRaises(Fatal):
            suite.snapshot()
        self.assertEqual(suite.snapshot().captured_monotonic, 6.0)

    def test_skipped_exception_generations_preserve_fresh_rollback_detection(self):
        class Fatal(BaseException):
            pass

        outcomes = iter((10.0, RuntimeError("clock failure"), Fatal("clock exit"), 9.0))

        def clock():
            outcome = next(outcomes)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        suite = SensorSuite(monotonic_clock=clock)
        mark_suite_attached(suite)
        initial = suite.snapshot()
        with self.assertRaisesRegex(RuntimeError, "clock failure"):
            suite.snapshot()
        with self.assertRaises(Fatal):
            suite.snapshot()
        rolled_back = suite.snapshot()

        self.assertEqual(initial.captured_monotonic, 10.0)
        self.assertEqual(rolled_back.captured_monotonic, 10.0)
        self.assertEqual(rollback_reason_count(rolled_back), 5)

    def test_twenty_thousand_concurrent_snapshots_have_no_false_rollback(self):
        caller_count = 4
        snapshots_per_caller = 5_000
        barrier = threading.Barrier(caller_count)
        value_lock = threading.Lock()
        next_value = 0

        def clock():
            nonlocal next_value
            with value_lock:
                next_value += 1
                value = float(next_value)
            barrier.wait(WAIT_SECONDS * 10.0)
            return value

        suite = SensorSuite(monotonic_clock=clock)
        mark_suite_attached(suite)
        rollback_counts = [0] * caller_count
        degraded_counts = [0] * caller_count
        errors = []

        def exercise(index):
            try:
                for _ in range(snapshots_per_caller):
                    value = suite.snapshot()
                    rollback_counts[index] += rollback_reason_count(value)
                    degraded_counts[index] += sum(
                        status.operational_health is OperationalHealth.DEGRADED
                        for status in value.statuses
                    )
            except BaseException as exc:
                errors.append((index, type(exc).__name__))

        callers = tuple(
            threading.Thread(target=exercise, args=(index,))
            for index in range(caller_count)
        )
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join(WAIT_SECONDS * 15.0)

        self.assertTrue(all(not caller.is_alive() for caller in callers))
        self.assertEqual(errors, [])
        self.assertEqual(next_value, caller_count * snapshots_per_caller)
        self.assertEqual(sum(rollback_counts), 0)
        self.assertEqual(sum(degraded_counts), 0)


if __name__ == "__main__":
    unittest.main()
