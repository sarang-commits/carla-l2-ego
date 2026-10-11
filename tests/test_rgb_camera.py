"""Unit tests for Mission 4E: RGB camera conversion, publication, lifecycle.

No CARLA server is required. The camera image fake is local to this module;
generic world/vehicle fakes come from tests.fakes, and the collision/lane
measurement builders are reused from tests.test_events for the cross-sensor
isolation checks.
"""

import gc
import threading
import unittest
import weakref
from dataclasses import FrozenInstanceError

from sensors.collision import COLLISION_BLUEPRINT
from sensors.gnss import GNSS_BLUEPRINT
from sensors.imu import IMU_BLUEPRINT
from sensors.lane_invasion import LANE_INVASION_BLUEPRINT
from sensors.rgb_camera import (
    RGB_CAMERA_BLUEPRINT,
    CameraMount,
    RgbCameraConfig,
    parse_camera_frame,
    validate_camera_config,
)
from sensors.suite import SensorSuite, SensorSuiteConfig
from telemetry.aggregator import TelemetryAggregator
from telemetry.state import CameraFrame, CameraFrameMetadata, SampleStamp, TransformState
from tests.fakes import (
    FakeActorSnapshot,
    FakeGnssMeasurement,
    FakeImuMeasurement,
    FakeTransform,
    FakeVector,
    FakeVehicle,
    FakeWorld,
    FakeWorldSnapshot,
)
from tests.test_events import collision_measurement, lane_measurement


class FakeCameraImage:
    """CARLA-shaped sensor.camera.rgb output (BGRA byte buffer)."""

    def __init__(self, frame, timestamp, width, height, fov, raw_data, transform=None):
        self.frame = frame
        self.timestamp = timestamp
        self.width = width
        self.height = height
        self.fov = fov
        self.raw_data = raw_data
        self.transform = transform if transform is not None else FakeTransform()


def camera_image(frame=1, timestamp=0.5, width=1, height=1, fov=90.0, raw_data=None):
    if raw_data is None:
        raw_data = bytes([11, 22, 33, 44]) * (width * height)
    return FakeCameraImage(frame, timestamp, width, height, fov, raw_data)


def make_frame(frame_no, payload=b"\x01\x02\x03"):
    """Build a 1x1 CameraFrame record directly (no parsing involved)."""
    metadata = CameraFrameMetadata(
        stamp=SampleStamp(frame=frame_no, sim_time=float(frame_no), monotonic=0.0),
        width_px=1,
        height_px=1,
        horizontal_fov_deg=90.0,
        pixel_format="RGB8",
        row_stride_bytes=3,
        transform=None,
    )
    return CameraFrame(metadata=metadata, rgb_bytes=payload)


def attach_suite(events=None, warning_sink=None, config=None, **world_kwargs):
    world = FakeWorld(events=events, **world_kwargs)
    suite = SensorSuite(config=config, warning_sink=warning_sink)
    report = suite.attach(world, FakeVehicle())
    return world, suite, report


class ConversionTest(unittest.TestCase):
    def test_1x1_bgra_becomes_rgb(self):
        frame = parse_camera_frame(camera_image(raw_data=bytes([11, 22, 33, 44])))
        self.assertEqual(frame.rgb_bytes, bytes([33, 22, 11]))
        self.assertEqual(frame.metadata.width_px, 1)
        self.assertEqual(frame.metadata.height_px, 1)
        self.assertEqual(frame.metadata.pixel_format, "RGB8")
        self.assertEqual(frame.metadata.row_stride_bytes, 3)
        self.assertEqual(frame.metadata.horizontal_fov_deg, 90.0)
        self.assertEqual(frame.metadata.stamp.frame, 1)
        self.assertEqual(frame.metadata.stamp.sim_time, 0.5)

    def test_2x2_preserves_row_and_pixel_order(self):
        raw = bytes(
            [
                10, 11, 12, 255,   # row 0, pixel 0 (B, G, R, A)
                20, 21, 22, 255,   # row 0, pixel 1
                30, 31, 32, 255,   # row 1, pixel 0
                40, 41, 42, 255,   # row 1, pixel 1
            ]
        )
        frame = parse_camera_frame(camera_image(width=2, height=2, raw_data=raw))
        self.assertEqual(
            frame.rgb_bytes,
            bytes([12, 11, 10, 22, 21, 20, 32, 31, 30, 42, 41, 40]),
        )
        self.assertEqual(frame.metadata.row_stride_bytes, 6)

    def test_alpha_change_does_not_affect_rgb(self):
        opaque = parse_camera_frame(camera_image(raw_data=bytes([11, 22, 33, 255])))
        clear = parse_camera_frame(camera_image(raw_data=bytes([11, 22, 33, 0])))
        self.assertEqual(opaque.rgb_bytes, clear.rgb_bytes)

    def test_short_and_long_buffers_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_camera_frame(camera_image(raw_data=bytes(3)))    # short
        with self.assertRaises(ValueError):
            parse_camera_frame(camera_image(raw_data=bytes(5)))    # long

    def test_zero_and_negative_dimensions_are_rejected(self):
        for width, height in ((0, 1), (-1, 1), (1, 0), (1, -1)):
            with self.assertRaises(ValueError):
                parse_camera_frame(
                    camera_image(width=width, height=height, raw_data=bytes(4))
                )

    def test_bool_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_camera_frame(camera_image(width=True, raw_data=bytes(4)))
        with self.assertRaises(ValueError):
            parse_camera_frame(camera_image(height=True, raw_data=bytes(4)))

    def test_output_type_is_exactly_bytes(self):
        frame = parse_camera_frame(camera_image())
        self.assertIs(type(frame.rgb_bytes), bytes)

    def test_records_are_frozen(self):
        frame = parse_camera_frame(camera_image())
        with self.assertRaises(FrozenInstanceError):
            frame.rgb_bytes = b""
        with self.assertRaises(FrozenInstanceError):
            frame.metadata.width_px = 5

    def test_payload_is_excluded_from_repr(self):
        frame = parse_camera_frame(camera_image())
        self.assertNotIn("rgb_bytes", repr(frame))

    def test_source_mutation_does_not_change_output(self):
        raw = bytearray([11, 22, 33, 44])
        frame = parse_camera_frame(camera_image(raw_data=raw))
        raw[0] = 250
        raw[2] = 251
        self.assertEqual(frame.rgb_bytes, bytes([33, 22, 11]))

    def test_source_image_is_not_retained(self):
        image = camera_image()
        frame = parse_camera_frame(image)
        ref = weakref.ref(image)
        del image
        gc.collect()
        self.assertIsNone(ref())
        self.assertEqual(len(frame.rgb_bytes), 3)   # result still usable

    def test_stored_values_are_primitive(self):
        frame = parse_camera_frame(camera_image())
        metadata = frame.metadata
        self.assertIs(type(frame.rgb_bytes), bytes)
        self.assertIs(type(metadata.width_px), int)
        self.assertIs(type(metadata.height_px), int)
        self.assertIs(type(metadata.horizontal_fov_deg), float)
        self.assertIs(type(metadata.row_stride_bytes), int)
        self.assertIsInstance(metadata.transform, TransformState)


class ConfigValidationTest(unittest.TestCase):
    def test_default_config_is_valid(self):
        validate_camera_config(RgbCameraConfig())   # must not raise

    def test_invalid_configs_are_rejected(self):
        bad_configs = [
            RgbCameraConfig(width_px=0),
            RgbCameraConfig(width_px=-640),
            RgbCameraConfig(width_px=True),
            RgbCameraConfig(height_px=0),
            RgbCameraConfig(height_px=360.0),
            RgbCameraConfig(horizontal_fov_deg=0.0),
            RgbCameraConfig(horizontal_fov_deg=181.0),
            RgbCameraConfig(horizontal_fov_deg=float("inf")),
            RgbCameraConfig(sensor_tick=-0.1),
            RgbCameraConfig(sensor_tick=float("nan")),
            RgbCameraConfig(mount=CameraMount(z=float("nan"))),
        ]
        for config in bad_configs:
            with self.assertRaises(ValueError):
                validate_camera_config(config)


class CameraAggregatorTest(unittest.TestCase):
    def test_first_valid_frame_is_accepted(self):
        agg = TelemetryAggregator()
        self.assertTrue(agg.submit_camera_frame(make_frame(10)))
        snap = agg.snapshot()
        self.assertIsNotNone(snap.camera)
        self.assertEqual(snap.camera.metadata.stamp.frame, 10)

    def test_duplicate_frame_is_rejected_and_prior_kept(self):
        agg = TelemetryAggregator()
        first = make_frame(10, payload=b"\x01\x01\x01")
        agg.submit_camera_frame(first)
        revision = agg.snapshot().revision
        self.assertFalse(
            agg.submit_camera_frame(make_frame(10, payload=b"\x02\x02\x02"))
        )
        snap = agg.snapshot()
        self.assertIs(snap.camera, first)            # prior frame unchanged
        self.assertEqual(snap.revision, revision)    # revision untouched

    def test_older_frame_is_rejected(self):
        agg = TelemetryAggregator()
        newer = make_frame(20)
        agg.submit_camera_frame(newer)
        self.assertFalse(agg.submit_camera_frame(make_frame(19)))
        self.assertIs(agg.snapshot().camera, newer)

    def test_newer_frame_replaces_older(self):
        agg = TelemetryAggregator()
        agg.submit_camera_frame(make_frame(10))
        self.assertTrue(agg.submit_camera_frame(make_frame(11)))
        self.assertEqual(agg.snapshot().camera.metadata.stamp.frame, 11)

    def test_ten_thousand_publications_keep_only_latest(self):
        agg = TelemetryAggregator()
        for frame_no in range(1, 10001):
            self.assertTrue(agg.submit_camera_frame(make_frame(frame_no)))
        snap = agg.snapshot()
        self.assertEqual(snap.camera.metadata.stamp.frame, 10000)
        self.assertEqual(snap.revision, 10000)       # one bump per acceptance

    def test_snapshot_shares_frame_reference(self):
        agg = TelemetryAggregator()
        frame = make_frame(5)
        agg.submit_camera_frame(frame)
        snap_a = agg.snapshot()
        snap_b = agg.snapshot()
        self.assertIs(snap_a.camera, frame)          # no payload copying
        self.assertIs(snap_b.camera, frame)
        self.assertIs(snap_a.camera.rgb_bytes, snap_b.camera.rgb_bytes)

    def test_shutdown_between_conversion_and_publication_drops_frame(self):
        agg = TelemetryAggregator()
        frame = parse_camera_frame(camera_image())   # converted first...
        agg.begin_shutdown()                         # ...then the gate closes
        self.assertFalse(agg.submit_camera_frame(frame))
        self.assertIsNone(agg.snapshot().camera)

    def test_concurrent_100_and_101_always_end_at_101(self):
        for _ in range(50):
            agg = TelemetryAggregator()
            barrier = threading.Barrier(2)

            def publish(frame, agg=agg, barrier=barrier):
                barrier.wait()
                agg.submit_camera_frame(frame)

            threads = [
                threading.Thread(target=publish, args=(make_frame(100),)),
                threading.Thread(target=publish, args=(make_frame(101),)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(agg.snapshot().camera.metadata.stamp.frame, 101)


class CameraSuiteTest(unittest.TestCase):
    def test_camera_is_fifth_default_sensor(self):
        _world, suite, report = attach_suite()
        names = [r.name for r in report.results]
        self.assertEqual(
            names, ["gnss", "imu", "collision", "lane_invasion", "rgb_camera"]
        )
        self.assertTrue(report.all_attached)
        suite.destroy()

    def test_blueprint_attributes_are_configured(self):
        world, suite, _report = attach_suite()
        blueprint = world.blueprints[RGB_CAMERA_BLUEPRINT]
        self.assertEqual(blueprint.attributes["image_size_x"], "640")
        self.assertEqual(blueprint.attributes["image_size_y"], "360")
        self.assertEqual(blueprint.attributes["fov"], "90.0")
        self.assertEqual(blueprint.attributes["sensor_tick"], "0.1")
        self.assertEqual(blueprint.attributes["enable_postprocess_effects"], "false")
        suite.destroy()

    def test_rigid_mount_transform_is_used(self):
        world, suite, _report = attach_suite()
        mount = world.spawn_transforms[RGB_CAMERA_BLUEPRINT]
        self.assertAlmostEqual(mount.location.x, 1.50, places=4)
        self.assertAlmostEqual(mount.location.y, 0.00, places=4)
        self.assertAlmostEqual(mount.location.z, 1.70, places=4)
        self.assertAlmostEqual(mount.rotation.pitch, -5.0, places=4)
        self.assertAlmostEqual(mount.rotation.yaw, 0.0, places=4)
        self.assertAlmostEqual(mount.rotation.roll, 0.0, places=4)
        # Other sensors keep attaching at the vehicle origin.
        origin = world.spawn_transforms[GNSS_BLUEPRINT]
        self.assertEqual(origin.location.x, 0.0)
        suite.destroy()

    def test_custom_mount_and_size_are_honored(self):
        config = SensorSuiteConfig(
            rgb_camera=RgbCameraConfig(
                width_px=320,
                height_px=180,
                mount=CameraMount(x=2.0, y=-0.5, z=1.2, pitch=-10.0, yaw=5.0, roll=1.0),
            )
        )
        world, suite, _report = attach_suite(config=config)
        blueprint = world.blueprints[RGB_CAMERA_BLUEPRINT]
        self.assertEqual(blueprint.attributes["image_size_x"], "320")
        self.assertEqual(blueprint.attributes["image_size_y"], "180")
        mount = world.spawn_transforms[RGB_CAMERA_BLUEPRINT]
        self.assertAlmostEqual(mount.location.x, 2.0, places=4)
        self.assertAlmostEqual(mount.location.y, -0.5, places=4)
        self.assertAlmostEqual(mount.rotation.pitch, -10.0, places=4)
        suite.destroy()

    def test_invalid_camera_config_fails_only_the_camera(self):
        config = SensorSuiteConfig(rgb_camera=RgbCameraConfig(width_px=0))
        world, suite, report = attach_suite(config=config)
        by_name = {r.name: r for r in report.results}
        self.assertFalse(by_name["rgb_camera"].attached)
        self.assertNotIn(RGB_CAMERA_BLUEPRINT, world.spawned)  # never spawned
        for name in ("gnss", "imu", "collision", "lane_invasion"):
            self.assertTrue(by_name[name].attached)
        suite.destroy()

    def test_camera_spawn_failure_does_not_block_others(self):
        def exploding_sink(_message):
            raise RuntimeError("sink boom")           # must stay isolated too

        _world, suite, report = attach_suite(
            warning_sink=exploding_sink,
            spawn_fail=(RGB_CAMERA_BLUEPRINT,),
        )
        by_name = {r.name: r for r in report.results}
        self.assertFalse(by_name["rgb_camera"].attached)
        for name in ("gnss", "imu", "collision", "lane_invasion"):
            self.assertTrue(by_name[name].attached)
        suite.destroy()

    def test_listen_failure_stops_then_destroys_actor(self):
        # The fake registers/activates the callback and then has listen()
        # raise, so the actor is potentially listening: cleanup must stop() it
        # before destroying it, and the other sensors stay healthy.
        events = []
        world, suite, report = attach_suite(
            events=events, fail_listen=(RGB_CAMERA_BLUEPRINT,)
        )
        by_name = {r.name: r for r in report.results}
        self.assertFalse(by_name["rgb_camera"].attached)     # listen() raised
        for name in ("gnss", "imu", "collision", "lane_invasion"):
            self.assertTrue(by_name[name].attached)          # others healthy

        camera_actor = world.spawned[RGB_CAMERA_BLUEPRINT]   # spawn succeeded
        self.assertIsNotNone(camera_actor._callback)         # callback registered

        suite.destroy()

        # Potentially-listening actor is both stopped and destroyed...
        self.assertTrue(camera_actor.stopped)
        self.assertTrue(camera_actor.destroyed)
        stopped = [target for kind, target in events if kind == "stop"]
        destroyed = [target for kind, target in events if kind == "destroy"]
        self.assertEqual(stopped[0], RGB_CAMERA_BLUEPRINT)   # stopped first
        self.assertEqual(destroyed[0], RGB_CAMERA_BLUEPRINT)  # destroyed first
        # ...with every stop happening before any destroy.
        kinds = [kind for kind, _ in events if kind in ("stop", "destroy")]
        self.assertLess(
            max(i for i, k in enumerate(kinds) if k == "stop"),
            min(i for i, k in enumerate(kinds) if k == "destroy"),
        )

    def test_spawn_failed_sensor_is_never_stopped(self):
        # listen() was never invoked (no actor), so cleanup must not stop it.
        events = []
        _world, suite, _report = attach_suite(
            events=events, spawn_fail=(RGB_CAMERA_BLUEPRINT,)
        )
        suite.destroy()
        stopped = [target for kind, target in events if kind == "stop"]
        destroyed = [target for kind, target in events if kind == "destroy"]
        self.assertNotIn(RGB_CAMERA_BLUEPRINT, stopped)
        self.assertNotIn(RGB_CAMERA_BLUEPRINT, destroyed)

    def test_immediate_callback_during_listen_is_safe(self):
        _world, suite, report = attach_suite(
            immediate_emit={RGB_CAMERA_BLUEPRINT: camera_image(frame=3)}
        )
        by_name = {r.name: r for r in report.results}
        self.assertTrue(by_name["rgb_camera"].attached)
        snap = suite.snapshot()
        self.assertIsNotNone(snap.camera)
        self.assertEqual(snap.camera.metadata.stamp.frame, 3)
        suite.destroy()

    def test_malformed_frame_preserves_prior_and_is_isolated(self):
        world, suite, _report = attach_suite()
        camera_actor = world.spawned[RGB_CAMERA_BLUEPRINT]

        camera_actor.emit(camera_image(frame=5))
        self.assertEqual(suite.snapshot().camera.metadata.stamp.frame, 5)

        health_before = {h.name: h for h in suite.snapshot().sensors}
        malformed = camera_image(frame=6, raw_data=bytes(3))
        self.assertIsNone(camera_actor.emit(malformed))  # failure stays isolated
        snap = suite.snapshot()
        self.assertEqual(snap.camera.metadata.stamp.frame, 5)  # prior kept
        health = {h.name: h for h in snap.sensors}
        self.assertEqual(
            health["rgb_camera"].error_count,
            health_before["rgb_camera"].error_count + 1,
        )
        self.assertEqual(
            health["rgb_camera"].last_error,
            "ValueError('BGRA buffer is 3 bytes, expected exactly 4 for 1x1')",
        )

        # The other four sensors keep delivering.
        world.spawned[GNSS_BLUEPRINT].emit(
            FakeGnssMeasurement(
                frame=1, timestamp=0.1, latitude=1.0, longitude=2.0, altitude=3.0
            )
        )
        world.spawned[IMU_BLUEPRINT].emit(
            FakeImuMeasurement(
                frame=1,
                timestamp=0.1,
                accelerometer=FakeVector(0.0, 0.0, 9.81),
                gyroscope=FakeVector(0.0, 0.0, 0.0),
                compass=0.0,
            )
        )
        world.spawned[COLLISION_BLUEPRINT].emit(collision_measurement())
        world.spawned[LANE_INVASION_BLUEPRINT].emit(lane_measurement())

        snap = suite.snapshot()
        self.assertIsNotNone(snap.gnss)
        self.assertIsNotNone(snap.imu)
        self.assertEqual(len(snap.collisions), 1)
        self.assertEqual(len(snap.lane_invasions), 1)

        # A later valid frame still succeeds; an older one is still rejected.
        camera_actor.emit(camera_image(frame=9))
        self.assertEqual(suite.snapshot().camera.metadata.stamp.frame, 9)
        camera_actor.emit(camera_image(frame=8))
        self.assertEqual(suite.snapshot().camera.metadata.stamp.frame, 9)
        suite.destroy()

    def test_late_camera_callbacks_after_shutdown_leave_all_state_unchanged(self):
        warnings = []
        world, suite, _report = attach_suite(warning_sink=warnings.append)
        camera_actor = world.spawned[RGB_CAMERA_BLUEPRINT]
        retained_callback = camera_actor._callback
        self.assertIsNotNone(retained_callback)

        # Establish every snapshot field that a late callback must preserve.
        camera_actor.emit(camera_image(frame=5))
        world.spawned[GNSS_BLUEPRINT].emit(
            FakeGnssMeasurement(
                frame=2, timestamp=0.2, latitude=1.0, longitude=2.0, altitude=3.0
            )
        )
        world.spawned[IMU_BLUEPRINT].emit(
            FakeImuMeasurement(
                frame=2,
                timestamp=0.2,
                accelerometer=FakeVector(0.0, 0.0, 9.81),
                gyroscope=FakeVector(0.0, 0.0, 0.0),
                compass=0.0,
            )
        )
        world.spawned[COLLISION_BLUEPRINT].emit(collision_measurement(frame=3))
        world.spawned[LANE_INVASION_BLUEPRINT].emit(lane_measurement(frame=3))
        world.tick(
            FakeWorldSnapshot(
                frame=4,
                elapsed_seconds=0.4,
                actors={
                    10: FakeActorSnapshot(
                        FakeTransform(), FakeVector(3.0, 4.0, 0.0)
                    )
                },
            )
        )

        before = suite.snapshot()
        camera_health_before = {h.name: h for h in before.sensors}["rgb_camera"]
        self.assertIsNotNone(before.camera)
        self.assertIsNotNone(before.gnss)
        self.assertIsNotNone(before.imu)
        self.assertIsNotNone(before.ego)
        self.assertEqual(len(before.collisions), 1)
        self.assertEqual(len(before.lane_invasions), 1)
        self.assertEqual(camera_health_before.error_count, 0)
        self.assertIsNone(camera_health_before.last_error)
        self.assertEqual(before.collision_sequence, 1)
        self.assertEqual(before.lane_invasion_sequence, 1)
        self.assertEqual(warnings, [])

        suite.destroy()

        def assert_telemetry_unchanged(after):
            camera_health_after = {h.name: h for h in after.sensors}["rgb_camera"]
            self.assertEqual(
                camera_health_after.error_count, camera_health_before.error_count
            )
            self.assertEqual(
                camera_health_after.last_error, camera_health_before.last_error
            )
            self.assertEqual(after.sensors, before.sensors)
            self.assertEqual(after.revision, before.revision)
            self.assertIs(after.camera, before.camera)
            self.assertEqual(after.collision_sequence, before.collision_sequence)
            self.assertEqual(
                after.lane_invasion_sequence, before.lane_invasion_sequence
            )
            self.assertEqual(after.collisions, before.collisions)
            self.assertEqual(after.lane_invasions, before.lane_invasions)
            self.assertEqual(after.gnss, before.gnss)
            self.assertEqual(after.imu, before.imu)
            self.assertEqual(after.ego, before.ego)
            self.assertEqual(warnings, [])

        malformed = camera_image(frame=6, raw_data=bytes(3))
        self.assertIsNone(retained_callback(malformed))
        assert_telemetry_unchanged(suite.snapshot())

        self.assertIsNone(retained_callback(camera_image(frame=99)))
        assert_telemetry_unchanged(suite.snapshot())


if __name__ == "__main__":
    unittest.main()
