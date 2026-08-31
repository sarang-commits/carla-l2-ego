"""Mission 2: spawn one ego vehicle in CARLA, hold it for 15 seconds, clean up.

Connects to a running CARLA server at localhost:2000, spawns a single ego
vehicle (preferring vehicle.tesla.model3) at a valid map spawn point, moves
the spectator camera above and behind it, waits 15 seconds, and always
destroys the vehicle on exit via a finally block.
"""

import sys
import time

import carla

HOST = "localhost"
PORT = 2000
TIMEOUT_SECONDS = 10.0
PREFERRED_BLUEPRINT = "vehicle.tesla.model3"
ALIVE_SECONDS = 15.0

CARLA_LAUNCH_PATH = r"C:\Users\kksre\Downloads\CARLA_Latest\CarlaUE4.exe"


def pick_vehicle_blueprint(blueprint_library):
    """Return the preferred vehicle blueprint, or a deterministic fallback."""
    preferred = blueprint_library.filter(PREFERRED_BLUEPRINT)
    if len(preferred) > 0:
        return preferred[0]

    print(f"Blueprint {PREFERRED_BLUEPRINT} not available, using fallback.")
    vehicles = blueprint_library.filter("vehicle.*")
    if len(vehicles) == 0:
        raise RuntimeError("no vehicle blueprints available on this server")
    return sorted(vehicles, key=lambda bp: bp.id)[0]


def spawn_at_free_spawn_point(world, blueprint):
    """Spawn the vehicle at the first unoccupied map spawn point."""
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("current map has no spawn points")

    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(blueprint, spawn_point)
        if vehicle is not None:
            return vehicle, spawn_point

    raise RuntimeError(
        f"could not spawn vehicle: all {len(spawn_points)} spawn points are occupied"
    )


def move_spectator_behind(world, vehicle):
    """Place the spectator camera above and behind the vehicle, looking at it."""
    transform = vehicle.get_transform()
    forward = transform.get_forward_vector()
    location = carla.Location(
        x=transform.location.x - 8.0 * forward.x,
        y=transform.location.y - 8.0 * forward.y,
        z=transform.location.z + 5.0,
    )
    rotation = carla.Rotation(pitch=-20.0, yaw=transform.rotation.yaw)
    world.get_spectator().set_transform(carla.Transform(location, rotation))


def main() -> int:
    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT_SECONDS)

    try:
        world = client.get_world()
    except RuntimeError as exc:
        print(f"ERROR: could not connect to the CARLA server: {exc}", file=sys.stderr)
        print("Is the simulator running? Launch it manually with:", file=sys.stderr)
        print(f"  {CARLA_LAUNCH_PATH}", file=sys.stderr)
        return 1

    blueprint = pick_vehicle_blueprint(world.get_blueprint_library())
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")

    vehicle = None
    try:
        vehicle, spawn_point = spawn_at_free_spawn_point(world, blueprint)
        loc = spawn_point.location
        print(f"Vehicle ID:     {vehicle.id}")
        print(f"Blueprint:      {vehicle.type_id}")
        print(f"Spawn location: x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}")

        move_spectator_behind(world, vehicle)
        print("Spectator moved above and behind the vehicle.")

        print(f"Keeping vehicle alive for {ALIVE_SECONDS:.0f} seconds...")
        time.sleep(ALIVE_SECONDS)
        return 0
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if vehicle is not None:
            destroyed = vehicle.destroy()
            if destroyed:
                print(f"Cleanup: vehicle {vehicle.id} destroyed.")
            else:
                print(
                    f"Cleanup WARNING: vehicle {vehicle.id} could not be destroyed.",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    sys.exit(main())
