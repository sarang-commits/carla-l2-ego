"""RGB camera sensor: sensor.camera.rgb, latest-valid-frame telemetry only.

Each CARLA image is converted -- outside the aggregator lock -- from BGRA to a
C-contiguous immutable RGB byte payload and published as the single latest
camera frame. There is no frame history, queue, recording, or display, and
the published records retain no CARLA object, NumPy array, view, memoryview,
or source buffer.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy

from sensors.base import BaseSensor, set_blueprint_attribute
from telemetry.state import (
    PIXEL_FORMAT_RGB8,
    CameraFrame,
    CameraFrameMetadata,
    SampleStamp,
    transform_state,
)

RGB_CAMERA_BLUEPRINT = "sensor.camera.rgb"
RGB_CAMERA_NAME = "rgb_camera"


@dataclass(frozen=True, slots=True)
class CameraMount:
    """Rigid mount relative to the ego vehicle origin (meters / degrees)."""

    x: float = 1.50
    y: float = 0.00
    z: float = 1.70
    pitch: float = -5.0
    yaw: float = 0.0
    roll: float = 0.0


@dataclass(frozen=True, slots=True)
class RgbCameraConfig:
    width_px: int = 640
    height_px: int = 360
    horizontal_fov_deg: float = 90.0
    sensor_tick: float = 0.10
    mount: CameraMount = CameraMount()


def _require_genuine_positive_int(value, name: str) -> int:
    """A real positive int -- bools and other numeric types are rejected."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a genuine int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def validate_camera_config(config: RgbCameraConfig) -> None:
    """Reject an invalid camera configuration before any actor is created."""
    _require_genuine_positive_int(config.width_px, "width_px")
    _require_genuine_positive_int(config.height_px, "height_px")
    fov = float(config.horizontal_fov_deg)
    if not math.isfinite(fov) or not 0.0 < fov <= 180.0:
        raise ValueError(
            f"horizontal_fov_deg must be finite and in (0, 180], "
            f"got {config.horizontal_fov_deg!r}"
        )
    tick = float(config.sensor_tick)
    if not math.isfinite(tick) or tick < 0.0:
        raise ValueError(
            f"sensor_tick must be finite and non-negative, got {config.sensor_tick!r}"
        )
    mount = config.mount
    for axis in ("x", "y", "z", "pitch", "yaw", "roll"):
        if not math.isfinite(float(getattr(mount, axis))):
            raise ValueError(f"mount.{axis} must be finite")


def parse_camera_frame(measurement, receive_monotonic=None) -> CameraFrame:
    """Convert one CARLA BGRA image into an immutable RGB CameraFrame.

    Validates dimensions and the exact BGRA byte length, copies the source
    buffer once into private bytes, reshapes via NumPy, drops alpha, reorders
    BGR to RGB in the same pixel order, and materializes contiguous immutable
    bytes. Nothing from the source measurement survives in the result.
    """
    width = _require_genuine_positive_int(measurement.width, "width")
    height = _require_genuine_positive_int(measurement.height, "height")
    stamp = SampleStamp(
        frame=int(measurement.frame),
        sim_time=float(measurement.timestamp),
        monotonic=(
            time.monotonic() if receive_monotonic is None else receive_monotonic
        ),
    )
    fov = float(measurement.fov)
    transform = transform_state(measurement.transform)

    expected_bgra_length = width * height * 4
    raw = bytes(measurement.raw_data)                # stable private copy
    if len(raw) != expected_bgra_length:
        raise ValueError(
            f"BGRA buffer is {len(raw)} bytes, expected exactly "
            f"{expected_bgra_length} for {width}x{height}"
        )

    bgra = numpy.frombuffer(raw, dtype=numpy.uint8).reshape(height, width, 4)
    # Channels 2,1,0 = R,G,B: drops alpha and reorders in one step.
    rgb_bytes = numpy.ascontiguousarray(bgra[:, :, 2::-1]).tobytes()
    if len(rgb_bytes) != width * height * 3:
        raise ValueError(
            f"RGB payload is {len(rgb_bytes)} bytes, expected {width * height * 3}"
        )

    metadata = CameraFrameMetadata(
        stamp=stamp,
        width_px=width,
        height_px=height,
        horizontal_fov_deg=fov,
        pixel_format=PIXEL_FORMAT_RGB8,
        row_stride_bytes=width * 3,
        transform=transform,
    )
    return CameraFrame(metadata=metadata, rgb_bytes=rgb_bytes)


class RgbCameraSensor(BaseSensor):
    name = RGB_CAMERA_NAME
    blueprint_id = RGB_CAMERA_BLUEPRINT

    def __init__(
        self,
        aggregator,
        config: RgbCameraConfig | None = None,
        monotonic_clock=None,
        snapshot_consumer=None,
    ) -> None:
        config = config if config is not None else RgbCameraConfig()
        super().__init__(aggregator, config.sensor_tick, monotonic_clock)
        self.config = config
        self._snapshot_consumer_lock = threading.Lock()
        self._snapshot_consumer = _resolve_snapshot_consumer(snapshot_consumer)

    def configure_blueprint(self, blueprint) -> None:
        # Validation happens here, before the actor is spawned.
        validate_camera_config(self.config)
        set_blueprint_attribute(blueprint, "image_size_x", self.config.width_px)
        set_blueprint_attribute(blueprint, "image_size_y", self.config.height_px)
        set_blueprint_attribute(blueprint, "fov", self.config.horizontal_fov_deg)
        set_blueprint_attribute(blueprint, "enable_postprocess_effects", "false")
        # sensor_tick is set by BaseSensor.attach from self.tick.

    def make_spawn_transform(self, carla_module):
        mount = self.config.mount
        return carla_module.Transform(
            carla_module.Location(x=mount.x, y=mount.y, z=mount.z),
            carla_module.Rotation(pitch=mount.pitch, yaw=mount.yaw, roll=mount.roll),
        )

    def parse(self, measurement, receive_monotonic=None) -> CameraFrame:
        return parse_camera_frame(measurement, receive_monotonic)

    def submit(self, frame: CameraFrame) -> bool:
        """Publish one frame and optionally pass its exact snapshot onward.

        Consumer lookup and invocation happen only after the aggregator has
        returned its atomic accepted-frame snapshot.  Ordinary consumer
        failures are isolated here because a frame that was already parsed
        and published successfully must not be reclassified as a malformed
        camera callback.  A bridge is responsible for its own bounded fault
        evidence; ``BaseException`` retains the base callback policy and is
        not swallowed.
        """
        with self._snapshot_consumer_lock:
            consumer = self._snapshot_consumer
        if consumer is None:
            # Preserve the pre-integration path exactly when no consumer is
            # configured (including avoiding unnecessary snapshot assembly).
            return self._aggregator.submit_camera_frame(frame)

        snapshot = self._aggregator.submit_camera_frame_and_snapshot(frame)
        if snapshot is None:
            return False

        try:
            consumer(snapshot)
        except Exception:                            # noqa: BLE001 - isolate hook
            pass
        return True

    def release_snapshot_consumer(self) -> None:
        """Close this sensor's integration seam and release its strong ref."""
        consumer = None
        with self._snapshot_consumer_lock:
            consumer = self._snapshot_consumer
            self._snapshot_consumer = None
        # Keep the old reference alive until after lock release so a hostile
        # finalizer or weakref callback can never execute under our lock.
        del consumer

    def destroy(self) -> bool:
        try:
            return super().destroy()
        finally:
            self.release_snapshot_consumer()


def _resolve_snapshot_consumer(consumer):
    """Normalize a callable or bridge-shaped object without retaining both."""
    if consumer is None:
        return None
    try:
        submit = consumer.submit
    except AttributeError:
        if callable(consumer):
            return consumer
        raise TypeError(
            "snapshot_consumer must be callable or expose callable submit()"
        ) from None
    if not callable(submit):
        raise TypeError(
            "snapshot_consumer must be callable or expose callable submit()"
        )
    return submit
