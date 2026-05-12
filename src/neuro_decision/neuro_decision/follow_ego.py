import argparse
import glob
import math
import os
import sys
import time

egg = glob.glob(os.path.expanduser('~/carla_sim/PythonAPI/carla/dist/carla-*.egg'))
if egg:
    sys.path.append(egg[0])

import carla

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--role-name', default='ego_vehicle')
    parser.add_argument('--spawn-index', type=int, default=30)
    parser.add_argument('--follow-distance', type=float, default=10.0)
    parser.add_argument('--height', type=float, default=5.0)
    parser.add_argument('--pitch', type=float, default=-12.0)
    parser.add_argument('--interval', type=float, default=0.05)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)

    world = client.get_world()
    carla_map = world.get_map()
    spectator = world.get_spectator()

    ego = None
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.attributes.get('role_name') == args.role_name:
            ego = actor
            break

    if ego is None:
        print(f'role_name={args.role_name} 인 차량을 찾지 못했습니다.')
        sys.exit(1)

    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        print('spawn point를 찾지 못했습니다.')
        sys.exit(1)

    idx = args.spawn_index % len(spawn_points)
    sp = spawn_points[idx]

    ego.set_autopilot(False)
    ego.set_transform(sp)
    ego.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    ego.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))

    print(f'ego_vehicle를 spawn point #{idx} 로 이동했습니다.')
    print(f'x={sp.location.x:.2f}, y={sp.location.y:.2f}, z={sp.location.z:.2f}, yaw={sp.rotation.yaw:.2f}')
    print('Ctrl+C 로 종료')

    while True:
        t = ego.get_transform()
        yaw_rad = math.radians(t.rotation.yaw)

        cam_x = t.location.x - args.follow_distance * math.cos(yaw_rad)
        cam_y = t.location.y - args.follow_distance * math.sin(yaw_rad)
        cam_z = t.location.z + args.height

        spectator.set_transform(carla.Transform(
            carla.Location(x=cam_x, y=cam_y, z=cam_z),
            carla.Rotation(pitch=args.pitch, yaw=t.rotation.yaw, roll=0.0)
        ))

        time.sleep(args.interval)

if __name__ == '__main__':
    main()
