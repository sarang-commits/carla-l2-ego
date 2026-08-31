"""Smoke test: verify the Python client can connect to a running CARLA server.

Connects to localhost:2000 and prints client/server versions, the loaded
map name, and the current actor count. Exits with a clear error message
if the simulator is not running.
"""

import sys

import carla

HOST = "localhost"
PORT = 2000
TIMEOUT_SECONDS = 10.0

CARLA_LAUNCH_PATH = r"C:\Users\kksre\Downloads\CARLA_Latest\CarlaUE4.exe"


def main() -> int:
    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT_SECONDS)

    print(f"CARLA client version: {client.get_client_version()}")
    print(f"Connecting to {HOST}:{PORT} (timeout: {TIMEOUT_SECONDS:.0f}s)...")

    try:
        server_version = client.get_server_version()
        world = client.get_world()
        map_name = world.get_map().name
        actor_count = len(world.get_actors())
    except RuntimeError as exc:
        print(f"ERROR: could not connect to the CARLA server: {exc}", file=sys.stderr)
        print("Is the simulator running? Launch it manually with:", file=sys.stderr)
        print(f"  {CARLA_LAUNCH_PATH}", file=sys.stderr)
        return 1

    print(f"CARLA server version: {server_version}")
    print(f"Map name:             {map_name}")
    print(f"Actor count:          {actor_count}")
    print("Smoke test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
