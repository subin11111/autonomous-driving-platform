#!/usr/bin/env bash
set -u

fail_count=0
warn_count=0

ok() {
  echo "[OK] $*"
}

warn() {
  echo "[WARN] $*"
  warn_count=$((warn_count + 1))
}

fail() {
  echo "[FAIL] $*"
  fail_count=$((fail_count + 1))
}

need_cmd() {
  if command -v "$1" >/dev/null 2>&1; then
    ok "command available: $1"
  else
    fail "command missing: $1"
  fi
}

publisher_count() {
  local topic="$1"
  ros2 topic info "$topic" 2>/dev/null | awk -F': ' '/Publisher count/ {print $2}'
}

topic_info_check() {
  local topic="$1"
  local required="$2"
  local count

  echo
  echo "== $topic =="
  if ! ros2 topic info "$topic"; then
    if [[ "$required" == "fail" ]]; then
      fail "$topic is not available"
    else
      warn "$topic is not available"
    fi
    return
  fi

  count="$(publisher_count "$topic")"
  if [[ -z "$count" || "$count" == "0" ]]; then
    if [[ "$required" == "fail" ]]; then
      fail "$topic has no publisher"
    else
      warn "$topic has no publisher"
    fi
  else
    ok "$topic publisher count: $count"
  fi
}

echo "== environment =="
if [[ -n "${ROS_DISTRO:-}" ]]; then
  ok "ROS_DISTRO=$ROS_DISTRO"
else
  warn "ROS_DISTRO is not set. Source /opt/ros/humble/setup.bash and install/setup.bash first."
fi

need_cmd ros2
need_cmd timeout
need_cmd awk

if ! command -v ros2 >/dev/null 2>&1; then
  fail "ros2 is required; stopping checks"
  echo "Preflight result: FAIL"
  exit 1
fi

echo
echo "== packages =="
for pkg in autonomous_bringup yolopv2_ros neuro_decision vehicle_serial_bridge; do
  if ros2 pkg prefix "$pkg" >/dev/null 2>&1; then
    ok "package found: $pkg"
  else
    fail "package missing: $pkg"
  fi
done

topic_info_check /perception/real_world_lane_points fail
topic_info_check /perception/real_world_drivable_points warn
topic_info_check /perception/closest_obstacle warn
topic_info_check /traffic_light_state warn
topic_info_check /behavior_state warn
topic_info_check /desired_speed warn
topic_info_check /desired_steering_normalized warn
topic_info_check /cmd_vel fail
topic_info_check /vehicle/mcu_tx warn

echo
echo "== one-shot echo =="
behavior_msg="$(timeout 5 ros2 topic echo /behavior_state --once 2>/dev/null || true)"
if [[ -z "$behavior_msg" ]]; then
  warn "/behavior_state did not produce a sample"
else
  echo "$behavior_msg"
  if grep -q "STOP" <<<"$behavior_msg"; then
    warn "/behavior_state sample is STOP"
  else
    ok "/behavior_state produced a non-STOP sample"
  fi
fi

cmd_vel_msg="$(timeout 5 ros2 topic echo /cmd_vel --once 2>/dev/null || true)"
if [[ -z "$cmd_vel_msg" ]]; then
  fail "/cmd_vel did not produce a sample"
else
  echo "$cmd_vel_msg"
  if grep -Eq "x: 0(\\.0+)?$" <<<"$cmd_vel_msg" && grep -Eq "z: 0(\\.0+)?$" <<<"$cmd_vel_msg"; then
    warn "/cmd_vel sample appears to be zero"
  else
    ok "/cmd_vel produced a non-zero-looking sample"
  fi
fi

mcu_tx_msg="$(timeout 5 ros2 topic echo /vehicle/mcu_tx --once 2>/dev/null || true)"
if [[ -z "$mcu_tx_msg" ]]; then
  warn "/vehicle/mcu_tx did not produce a sample"
else
  echo "$mcu_tx_msg"
  ok "/vehicle/mcu_tx produced a sample"
fi

traffic_light_msg="$(timeout 5 ros2 topic echo /traffic_light_state --once 2>/dev/null || true)"
if [[ -z "$traffic_light_msg" ]]; then
  warn "/traffic_light_state did not produce a sample"
else
  echo "$traffic_light_msg"
  ok "/traffic_light_state produced a sample"
fi

echo
echo "== topic hz =="
if timeout 8 ros2 topic hz /cmd_vel; then
  ok "/cmd_vel hz check completed"
else
  fail "/cmd_vel hz check failed or timed out"
fi

if timeout 12 ros2 topic hz /perception/real_world_lane_points; then
  ok "/perception/real_world_lane_points hz check completed"
else
  fail "/perception/real_world_lane_points hz check failed or timed out"
fi

echo
if [[ "$fail_count" -gt 0 ]]; then
  echo "Preflight result: FAIL ($fail_count fail, $warn_count warn)"
  exit 1
fi

if [[ "$warn_count" -gt 0 ]]; then
  echo "Preflight result: WARN ($warn_count warn)"
  exit 0
fi

echo "Preflight result: OK"
