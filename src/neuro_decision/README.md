# neuro_decision

자율주행 자동차의 판단 및 제어 모듈

## 3가지 노드

1. behavior_node - 신호등/장애물/차선 기반으로 목표 속도/조향 직접 생성
2. speed_control_node - 목표 속도(m/s)와 실제 속도로 PID 스로틀/브레이크 생성
3. pure_pursuit_node - 목표 조향과 스로틀/브레이크를 차량 제어 명령으로 변환

## 필요한 토픽 (인지 모듈에서 발행)

- /perception/real_world_lane_points (차선)
- /perception/closest_obstacle (장애물)
- /traffic_light_state (신호등)

## 발행하는 토픽

- /desired_speed (목표 속도, m/s)
- /desired_steer (목표 조향, -1.0 ~ 1.0)
- /carla/ego_vehicle/vehicle_control_cmd (제어 명령)

## 실행

```bash
colcon build
source install/setup.bash
ros2 launch neuro_decision neuro_decision.launch.py
```

## 팀 정보

- 판단/제어: 당신
- 인지: 팀원
