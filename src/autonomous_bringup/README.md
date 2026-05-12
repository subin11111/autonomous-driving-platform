autonomous_bringup
==================

Mock-safe launch files for checking the perception -> planning -> cmd_vel -> MCU bridge path.

Mock E2E launch
---------------

Run the E2E mock launch:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch autonomous_bringup e2e_mock.launch.py \
  mock_serial:=true \
  lane_timeout_s:=2.0 \
  enable_traffic_light:=false \
  use_roi_color_fallback:=false
```

`mock_serial` defaults to `true` and should stay true until hardware safety checks are complete. The launch still exposes `mcu_port` as `/dev/ttyACM0`, but the bridge must not open it while `mock_serial:=true`.

Hardware-safe launch
--------------------

`e2e_hardware_safe.launch.py` is the only bringup launch intended for real MCU serial connection. It is still safe by default:

- `enable_vehicle_bridge` defaults to `false`.
- `mock_serial` defaults to `true`.
- `mock_serial:=false` is refused unless all confirmation arguments are present.

Dry-run the full stack without opening serial:

```bash
ros2 launch autonomous_bringup e2e_hardware_safe.launch.py
```

Real serial is gated behind an explicit confirmation string:

```bash
ros2 launch autonomous_bringup e2e_hardware_safe.launch.py \
  enable_vehicle_bridge:=true \
  mock_serial:=false \
  require_hardware_confirm:=true \
  hardware_confirm_text:=I_UNDERSTAND_THIS_CAN_MOVE_THE_VEHICLE
```

Do not run that command until `scripts/e2e_preflight_check.sh` passes, the vehicle is physically secured, and `docs/hardware_preflight_checklist.md` has been followed.

Why `lane_timeout_s` defaults to 2.0 here
-----------------------------------------

`behavior_node` defaults `lane_timeout_s` to 0.5 seconds. On CPU, YOLOPv2 inference can publish `/perception/real_world_lane_points` slower than that, which makes `behavior_node` alternate between `STOP`, `lane_reacquire_wait`, and `LANE_KEEPING`. This mock launch uses `lane_timeout_s:=2.0` as a test-time override so the end-to-end chain is easier to inspect. With GPU inference or a faster model path, reduce it back toward the behavior default.

On CPU-only runs with long inference gaps, `lane_timeout_s:=3.0` or `4.0` may be needed for bench inspection. Treat that as a test-time override, not a replacement for improving perception rate before real driving.

Quick topic checks:

```bash
ros2 topic info /perception/real_world_lane_points
ros2 topic info /perception/real_world_drivable_points
ros2 topic info /perception/closest_obstacle
ros2 topic info /traffic_light_state
ros2 topic info /behavior_state
ros2 topic info /desired_speed
ros2 topic info /desired_steering_normalized
ros2 topic info /cmd_vel
ros2 topic info /vehicle/mcu_tx
```

Watch for mock MCU commands:

```bash
ros2 topic echo /vehicle/mcu_tx
```

Expected values include `W`, `A`, `D`, `C`, and Space depending on `/cmd_vel` and stop state.

Preflight helper:

```bash
scripts/e2e_preflight_check.sh
```

The script only reads ROS package/topic state. It does not open the serial port.

If `behavior_state` repeats STOP/LANE_KEEPING:

- Check `/perception/real_world_lane_points` rate with `ros2 topic hz /perception/real_world_lane_points`.
- Check `/yolopv2/lane_mask` rate.
- Confirm `lane_timeout_s` in the launch args.
- Watch YOLOPv2 inference logs for processed frame rate.

Traffic light state
-------------------

The current `traffic_light_detector_node` emits a single COCO `traffic light` bbox class, not red/yellow/green classes. Because of that, `/traffic_light_state` often remains `UNKNOWN`. Actual color state requires either validated ROI color fallback (`use_roi_color_fallback:=true`) or a detector/model that emits red/yellow/green classes directly.
