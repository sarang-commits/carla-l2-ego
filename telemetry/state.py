"""Immutable, CARLA-independent telemetry records and pure helpers.

These dataclasses never retain CARLA actor, sensor, or measurement objects:
callback data is copied into plain Python floats/ints at the boundary. That
keeps the records safe to pass across threads and trivial to unit-test with
no CARLA package or server involved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from telemetry.freshness import SensorStatus


@dataclass(frozen=True, slots=True)
class SampleStamp:
    """When a sample belongs to (sim) and when we received it (host)."""

    frame: int              # CARLA frame the sample belongs to (source frame)
    sim_time: float         # simulation timestamp, seconds
    monotonic: float        # host receive time, time.monotonic() seconds


@dataclass(frozen=True, slots=True)
class Vector3:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class EulerDegrees:
    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True, slots=True)
class TransformState:
    location: Vector3
    rotation: EulerDegrees


@dataclass(frozen=True, slots=True)
class EgoKinematics:
    stamp: SampleStamp
    transform: TransformState
    velocity: Vector3
    speed_mps: float
    speed_kph: float
    vehicle_alive: bool


@dataclass(frozen=True, slots=True)
class GnssState:
    stamp: SampleStamp
    latitude: float
    longitude: float
    altitude: float
    transform: TransformState | None = None


@dataclass(frozen=True, slots=True)
class ImuState:
    stamp: SampleStamp
    accelerometer: Vector3   # m/s^2
    gyroscope: Vector3       # rad/s
    compass_rad: float       # radians
    heading_deg: float       # degrees, normalized to [0, 360)
    transform: TransformState | None = None


@dataclass(frozen=True, slots=True)
class ActorRef:
    """Identity of a CARLA actor, copied out (the actor is never retained)."""

    id: int
    type_id: str


@dataclass(frozen=True, slots=True)
class CollisionEvent:
    stamp: SampleStamp
    self_actor: ActorRef        # the ego the collision sensor is attached to
    other_actor: ActorRef       # what the ego hit
    normal_impulse: Vector3
    impulse_magnitude: float
    transform: TransformState | None = None


@dataclass(frozen=True, slots=True)
class LaneMarkingInfo:
    type: str                   # stable enum name, e.g. "Solid", "Broken"
    color: str                  # stable enum name, e.g. "White", "Yellow"
    lane_change: str            # stable enum name, e.g. "NONE", "Right"
    width: float


@dataclass(frozen=True, slots=True)
class LaneInvasionEvent:
    stamp: SampleStamp
    # Deterministically sorted; duplicates preserved.
    crossed_markings: tuple[LaneMarkingInfo, ...]
    transform: TransformState | None = None


PIXEL_FORMAT_RGB8 = "RGB8"


@dataclass(frozen=True, slots=True)
class CameraFrameMetadata:
    """Shape/geometry of one converted RGB camera frame."""

    stamp: SampleStamp
    width_px: int
    height_px: int
    horizontal_fov_deg: float
    pixel_format: str           # always PIXEL_FORMAT_RGB8
    row_stride_bytes: int       # width_px * 3
    transform: TransformState | None = None


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """Latest RGB camera frame; the payload is immutable bytes.

    ``rgb_bytes`` is exactly width_px * height_px * 3 bytes and is excluded
    from repr so printing a frame never dumps the pixel payload.
    """

    metadata: CameraFrameMetadata
    rgb_bytes: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class SensorHealth:
    name: str
    attached: bool
    messages_received: int
    error_count: int
    last_frame: int | None = None
    last_sim_time: float | None = None
    last_receive_monotonic: float | None = None
    last_error: str | None = None
    attach_attempted: bool = False
    attachment_failed: bool = False
    attachment_monotonic: float | None = None
    last_error_monotonic: float | None = None
    active_error: bool = False
    active_error_reason: str | None = None


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """A coherent, immutable view of the latest telemetry at one revision."""

    revision: int
    captured_monotonic: float
    ego: EgoKinematics | None = None
    gnss: GnssState | None = None
    imu: ImuState | None = None
    sensors: tuple[SensorHealth, ...] = ()
    collisions: tuple[CollisionEvent, ...] = ()
    lane_invasions: tuple[LaneInvasionEvent, ...] = ()
    camera: CameraFrame | None = None
    # Monotonic counts of *accepted* events (consumed by Mission 4F).
    collision_sequence: int = 0
    lane_invasion_sequence: int = 0
    # SensorSuite adds one immutable evaluated status per configured slot.
    statuses: tuple[SensorStatus, ...] = ()


def vector_magnitude(x: float, y: float, z: float) -> float:
    """Euclidean magnitude of a 3D vector."""
    return math.sqrt(x * x + y * y + z * z)


def speed_from_velocity(vx: float, vy: float, vz: float) -> tuple[float, float]:
    """Return (speed_mps, speed_kph) from a velocity vector."""
    speed_mps = vector_magnitude(vx, vy, vz)
    return speed_mps, speed_mps * 3.6


def normalize_heading(compass_rad: float) -> float:
    """heading_deg = degrees(compass_rad) % 360, always in [0, 360)."""
    return math.degrees(compass_rad) % 360.0


def transform_state(carla_transform) -> TransformState:
    """Copy a CARLA (or CARLA-shaped) transform into an immutable record.

    Duck-typed on purpose: reads ``.location`` and ``.rotation`` and copies
    the numeric fields out, so the CARLA object is never retained.
    """
    location = carla_transform.location
    rotation = carla_transform.rotation
    return TransformState(
        location=Vector3(float(location.x), float(location.y), float(location.z)),
        rotation=EulerDegrees(
            float(rotation.roll), float(rotation.pitch), float(rotation.yaw)
        ),
    )
