# vehicle_serial_bridge

ROS2 상위 제어 토픽을 MCU 단일 문자 시리얼 명령으로 변환하는 브리지 패키지입니다.

## Topics

- Subscribe: /cmd_vel (geometry_msgs/msg/Twist)
- Subscribe: /vehicle/estop (std_msgs/msg/Bool)
- Publish: /vehicle/mcu_status (std_msgs/msg/String)
- Publish: /vehicle/mcu_tx (std_msgs/msg/String)

## Command Mapping

- linear.x > linear_deadband -> W
- linear.x < -linear_deadband -> S
- abs(linear.x) <= linear_deadband -> Space
- angular.z > angular_deadband -> A
- angular.z < -angular_deadband -> D
- abs(angular.z) <= angular_deadband -> C (send_center_command=true 일 때)

Drive 명령을 먼저, Steer 명령을 나중에 전송합니다.

## Safety Behavior

- E-stop true 수신 시 즉시 Space 전송
- E-stop 상태에서는 주행 명령 무시
- watchdog_timeout 동안 cmd_vel 미수신 시 Space 전송
- 노드 시작/종료 시 Space 전송 시도
- 상태 변경 시에만 명령 전송 (serial spam 감소)

## Build

colcon build --packages-select vehicle_serial_bridge
source install/setup.bash

## Run

ros2 run vehicle_serial_bridge mcu_serial_bridge --ros-args -p port:=/dev/ttyACM0 -p baudrate:=115200

## Run (mock serial)

ros2 run vehicle_serial_bridge mcu_serial_bridge --ros-args -p mock_serial:=true

## Launch

ros2 launch vehicle_serial_bridge mcu_serial_bridge.launch.py

## Test Commands

전진 테스트:
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

전진 좌회전 테스트:
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}"

전진 우회전 테스트:
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: -0.5}}"

후진 테스트:
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: -0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

정지 테스트:
ros2 topic pub -1 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

E-stop 테스트:
ros2 topic pub -1 /vehicle/estop std_msgs/msg/Bool "{data: true}"

디버그 확인:
ros2 topic echo /vehicle/mcu_tx
ros2 topic echo /vehicle/mcu_status