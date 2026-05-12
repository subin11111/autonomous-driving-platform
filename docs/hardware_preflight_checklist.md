# Hardware Preflight Checklist

Use this checklist before any real MCU or vehicle connection. Keep the vehicle bridge in mock mode until every software and bench check below is complete.

## 1. Software Checks

- `colcon build` succeeds for `autonomous_bringup`, `yolopv2_ros`, `neuro_decision`, and `vehicle_serial_bridge`.
- `e2e_mock.launch.py` runs with `mock_serial:=true`.
- `/cmd_vel` is published and does not jump unexpectedly.
- `/vehicle/mcu_tx` shows mock `W`, `S`, `A`, `D`, `C`, or Space commands as expected.
- `/behavior_state` transitions are understood, for example `STOP` to `LANE_KEEPING`.
- `/perception/real_world_lane_points` publishes non-empty `PointCloud2` messages with a non-empty `frame_id`.
- Run `scripts/e2e_preflight_check.sh` and resolve every `FAIL` before hardware tests.

## 2. Traffic Light Checks

- The current traffic light detector uses the COCO `traffic light` bbox class.
- `/traffic_light_state` may remain `UNKNOWN` because the detector does not directly emit red/yellow/green classes.
- Do not trust RED/GREEN/YELLOW behavior until ROI color fallback is validated in the real camera environment or a color-class detector is used.
- Keep `enable_traffic_light:=false` unless the traffic light state source has been validated.

## 3. Hardware Checks

- Confirm the vehicle wheels are off the ground before any powered test.
- Start with motor power disabled, a relay disconnected, or the drive path physically isolated.
- Prepare an E-stop or immediate power cutoff.
- Check MCU and DAC wiring before connecting the vehicle harness.
- Measure MCP4728 output voltage with a multimeter before connecting throttle or brake lines.
- Confirm relay HIGH/LOW direction before commanding motion.
- Check that throttle and brake lines cannot be driven at the same time by wiring or firmware faults.
- Run low-voltage or no-load tests before any ground-driving test.

## 4. Real Serial Connection Procedure

1. Run the mock E2E launch first:

   ```bash
   ros2 launch autonomous_bringup e2e_mock.launch.py mock_serial:=true
   ```

2. Confirm `/vehicle/mcu_tx` shows only expected `W`, `S`, `A`, `D`, `C`, and Space commands.
3. Run the hardware-safe launch with the bridge disabled:

   ```bash
   ros2 launch autonomous_bringup e2e_hardware_safe.launch.py enable_vehicle_bridge:=false
   ```

4. With motors or drive hardware isolated, start the bridge only after explicit confirmation:

   ```bash
   ros2 launch autonomous_bringup e2e_hardware_safe.launch.py \
     enable_vehicle_bridge:=true \
     mock_serial:=false \
     require_hardware_confirm:=true \
     hardware_confirm_text:=I_UNDERSTAND_THIS_CAN_MOVE_THE_VEHICLE
   ```

5. Keep `/cmd_vel` at zero for the first serial connection.
6. Confirm Space stop behavior before any `W` or `S` command.
7. Test steering and drive commands with the wheels lifted or the drive path disconnected.

## 5. Stop Conditions

- Do not connect real hardware if `/behavior_state` is unstable.
- Do not connect real hardware if `/cmd_vel` jumps or oscillates unexpectedly.
- Keep `enable_traffic_light:=false` if `/traffic_light_state` differs from the scene.
- Do not drive if `/perception/real_world_lane_points` rate is too low for the configured timeout.
- Stop immediately if `mcu_serial_bridge` reports serial errors.
