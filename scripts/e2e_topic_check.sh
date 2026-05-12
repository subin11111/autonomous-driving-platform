#!/usr/bin/env bash
set -u

topics=(
  /perception/real_world_lane_points
  /perception/real_world_drivable_points
  /perception/closest_obstacle
  /traffic_light_state
  /behavior_state
  /desired_speed
  /desired_steering_normalized
  /cmd_vel
  /vehicle/mcu_tx
)

echo "== topic info =="
for topic in "${topics[@]}"; do
  echo
  echo "-- ${topic}"
  ros2 topic info "${topic}" || true
done

echo
echo "== one-shot echo =="
for topic in /behavior_state /cmd_vel /vehicle/mcu_tx /traffic_light_state; do
  echo
  echo "-- ${topic}"
  timeout 5 ros2 topic echo "${topic}" --once || true
done
