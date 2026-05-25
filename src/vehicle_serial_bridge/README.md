# vehicle_serial_bridge

ROS2 상위 제어 토픽을 Arduino/MCU 시리얼 명령으로 변환하는 브리지 패키지입니다.

이 브리지는 `/desired_speed`, `/desired_steering_angle_deg`, `/behavior_state`를 직접 구독해서 실제 속도값과 조향각을 Arduino에 전송합니다.

## Interface

새 Arduino 펌웨어용 `CMD` line protocol만 지원합니다.

- Subscribe: `/desired_speed` (`std_msgs/msg/Float64`, m/s)
- Subscribe: `/desired_steering_angle_deg` (`std_msgs/msg/Float64`, degree)
- Subscribe: `/behavior_state` (`std_msgs/msg/String`)
- Subscribe: `/vehicle/estop` (`std_msgs/msg/Bool`)
- Publish: `/vehicle/mcu_status` (`std_msgs/msg/String`)
- Publish: `/vehicle/mcu_tx` (`std_msgs/msg/String`)

`/cmd_vel`은 사용하지 않습니다. `mcu_serial_bridge`가 판단/조향 노드의 공식 토픽을 직접 구독하므로 실제 조향각이 Arduino까지 보존됩니다.

## Serial Protocol

Arduino 입력은 줄 단위 newline 종료 문자열입니다. Arduino는 `\n` 기준으로 한 줄씩 읽어 CSV 형태로 파싱해야 합니다.

일반 주행 명령:

```text
CMD,<speed_mps>,<steering_deg>,<behavior_state>\n
```

예:

```text
CMD,0.500,10.00,LANE_KEEPING\n
CMD,0.000,-3.00,LANE_KEEPING\n
CMD,0.300,0.00,FOLLOW_VEHICLE\n
```

정지:

```text
STOP\n
```

비상 정지:

```text
ESTOP\n
```

형식 규칙:

- `speed_mps`: 소수점 3자리
- `steering_deg`: 소수점 2자리
- `behavior_state`: 공백 없는 uppercase 문자열
- 기본 조향 범위: `-20.0` ~ `+20.0` degree
- 기본 속도 범위: `-1.40` ~ `1.40` m/s (`allow_reverse=true`)

## Safety Behavior

- `behavior_state`가 `stop_states` 안에 있으면 `STOP\n`을 주기적으로 전송합니다.
- STOP은 Arduino에서 모터 정지 + 조향 중립으로 처리해야 합니다.
- `/vehicle/estop=true` 수신 시 즉시 `ESTOP\n`을 전송합니다.
- E-stop 활성 중에는 일반 `CMD` 명령을 절대 전송하지 않습니다.
- `/vehicle/estop=false`가 되어도 즉시 주행 명령을 보내지 않고, 다음 valid command cycle에서만 전송합니다.
- speed, steering, state 입력 중 하나라도 timeout이면 `STOP\n`을 전송합니다.
- 노드 종료 시 `STOP\n`을 전송하고 serial을 닫습니다.
- `speed_mps`는 `max_abs_speed_mps`로 clamp합니다.
- `allow_reverse=true`가 기본값이므로 음수 speed를 후진 명령으로 전송합니다.
- `steering_deg`는 `max_abs_steering_deg`로 clamp합니다.

## Parameters

- `port`: serial port (`/dev/ttyACM0`)
- `baudrate`: serial baudrate (`115200`)
- `mock_serial`: mock backend 사용 여부 (`false`)
- `serial_timeout`: serial read timeout (`0.01`)
- `write_timeout`: serial write timeout (`0.01`)
- `command_publish_period_s`: numeric command publish period (`0.05`)
- `desired_speed_topic`: `/desired_speed`
- `desired_steering_angle_deg_topic`: `/desired_steering_angle_deg`
- `behavior_state_topic`: `/behavior_state`
- `max_abs_speed_mps`: `1.40`
- `max_abs_steering_deg`: `20.0`
- `allow_reverse`: `true`
- `stop_states`: `['STOP', 'EMERGENCY_STOP', 'ESTOP', 'RED_LIGHT', 'OBSTACLE_STOP']`
- `speed_input_timeout_s`: `0.5`
- `steering_input_timeout_s`: `0.5`
- `state_input_timeout_s`: `0.5`

## Build

```bash
cd ~/dream_ws/neuro_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select vehicle_serial_bridge
source install/setup.bash
```

실행 파일 확인:

```bash
ros2 pkg executables vehicle_serial_bridge
```

기대:

```text
vehicle_serial_bridge mcu_serial_bridge
```

## Run

Mock numeric test:

```bash
ros2 run vehicle_serial_bridge mcu_serial_bridge --ros-args \
  -p mock_serial:=true
```

실제 Arduino:

```bash
ros2 run vehicle_serial_bridge mcu_serial_bridge --ros-args \
  -p mock_serial:=false \
  -p port:=/dev/ttyACM0 \
  -p baudrate:=115200 \
  -p max_abs_steering_deg:=20.0
```

Launch:

```bash
ros2 launch vehicle_serial_bridge mcu_serial_bridge.launch.py
```

## Manual Test

터미널 1:

```bash
ros2 run vehicle_serial_bridge mcu_serial_bridge --ros-args \
  -p mock_serial:=true \
  -p max_abs_steering_deg:=20.0 \
  -p max_abs_speed_mps:=1.40
```

터미널 2:

```bash
ros2 topic echo /vehicle/mcu_tx std_msgs/msg/String
```

터미널 3:

```bash
ros2 topic pub -r 10 /behavior_state std_msgs/msg/String '{data: "LANE_KEEPING"}'
```

터미널 4:

```bash
ros2 topic pub -r 10 /desired_speed std_msgs/msg/Float64 '{data: 0.5}'
```

터미널 5:

```bash
ros2 topic pub -r 10 /desired_steering_angle_deg std_msgs/msg/Float64 '{data: 3.0}'
```

기대 `/vehicle/mcu_tx`:

```text
CMD,0.500,3.00,LANE_KEEPING
```

조향 변경:

```bash
ros2 topic pub -r 10 /desired_steering_angle_deg std_msgs/msg/Float64 '{data: 20.0}'
```

기대:

```text
CMD,0.500,20.00,LANE_KEEPING
```

STOP:

```bash
ros2 topic pub -r 10 /behavior_state std_msgs/msg/String '{data: "STOP"}'
```

기대:

```text
STOP
```

E-stop:

```bash
ros2 topic pub --once /vehicle/estop std_msgs/msg/Bool '{data: true}'
```

기대:

```text
ESTOP
```

추가 검증 항목:

- `desired_steering_angle_deg=30.0`, `max_abs_steering_deg=20.0` -> `20.00`으로 clamp
- `desired_speed=2.0`, `max_abs_speed_mps=1.40` -> `1.400`으로 clamp
- `allow_reverse=true`, `desired_speed=-0.5` -> `-0.500` 전송
- 입력 토픽 중 하나가 끊기면 `STOP` 전송

## Arduino Firmware Requirements

이 저장소에는 Arduino 펌웨어 파일이 포함되어 있지 않습니다. 실제 차량에서 사용하려면 Arduino 펌웨어가 아래 규칙을 구현해야 합니다.

- Serial line을 `\n` 기준으로 읽습니다.
- `CMD,<speed_mps>,<steering_deg>,<behavior_state>` 라인을 CSV로 파싱합니다.
- `speed_mps`를 실제 모터 제어값으로 변환합니다.
- `steering_deg`를 서보 목표각으로 변환합니다.
- `steering_deg`는 기본 `-20.0` ~ `+20.0` 범위라고 가정합니다.
- `STOP`은 모터 정지 + 조향 중립으로 처리합니다.
- `ESTOP`은 즉시 모터 정지 후 E-stop latch 또는 안전 모드에 진입합니다.
- 일정 시간 serial command가 없으면 Arduino 자체 watchdog으로 모터를 정지해야 합니다.

## Safety Checklist

- 먼저 `mock_serial:=true`로 `/vehicle/mcu_tx`를 확인합니다.
- 실제 Arduino 연결 전 바퀴를 띄웁니다.
- E-stop을 준비한 상태에서 테스트합니다.
- `STOP` 동작 확인 전 전진 명령을 보내지 않습니다.
