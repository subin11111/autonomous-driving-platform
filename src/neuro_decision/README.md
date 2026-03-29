# neuro_decision

자율주행 자동차의 판단 및 제어 모듈

## 3가지 노드

1. behavior_node - 신호등, 장애물 기반 의사결정
2. pure_pursuit_node - 조향각 계산 및 제어
3. speed_control_node - PID 기반 속도 제어

## 필요한 토픽 (인지 모듈에서 발행)

- /perception/real_world_lane_points (차선)
- /perception/closest_obstacle (장애물)
- /traffic_light_state (신호등)

## 발행하는 토픽

- /goal_pose (목표점)
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
