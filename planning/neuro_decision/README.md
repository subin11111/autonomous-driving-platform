# neuro_decision

Perception 기반 자율주행 자동차의 판단 및 제어 모듈 (v4+)

## 📋 개요

3개 노드로 구성된 제어 스택:

| 노드 | 역할 | 입력 | 출력 |
|------|------|------|------|
| **behavior_node** | Perception 데이터 기반 의사결정 | 차선점, 장애물, 신호등 | 목표속도, 목표점 |
| **speed_control_node** | 목표속도 추종 제어 | 목표속도, 현재속도, 상태 | 스로틀, 브레이크 |
| **pure_pursuit_node** | 목표점 기반 조향 제어 | 목표점, 스로틀, 브레이크 | 차량제어명령 |

---

## 🛰️ 필요한 Perception 토픽 (입력)

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/perception/real_world_lane_points` | `sensor_msgs/PointCloud2` | 차선 중심 포인트 (local 좌표계) |
| `/perception/closest_obstacle` | `sensor_msgs/PointCloud2` | 가장 가까운 장애물 위치 |
| `/perception/drivable_area` | `sensor_msgs/PointCloud2` | 주행 가능 영역 |
| `/traffic_light_state` | `std_msgs/String` | 신호등 상태 (RED/YELLOW/GREEN) |
| `/carla/ego_vehicle/speedometer` | `std_msgs/Float32` | 현재 속도 (m/s) |

---

## 📤 발행하는 토픽 (출력)

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/desired_speed` | `std_msgs/Float64` | 목표 속도 (m/s) |
| `/target_point` | `geometry_msgs/Point` | 차량 기준 목표점 (m) |
| `/behavior_state` | `std_msgs/String` | 현재 행동 상태 |
| `/behavior_debug_text` | `std_msgs/String` | 디버그 메시지 |
| `/speed_command` | `std_msgs/Float64` | 스로틀 명령 (0.0~1.0) |
| `/brake_command` | `std_msgs/Float64` | 브레이크 명령 (0.0~1.0) |
| `/speed_control_debug_text` | `std_msgs/String` | 속도제어 디버그 메시지 |
| `/carla/ego_vehicle/vehicle_control_cmd` | `carla_msgs/CarlaEgoVehicleControl` | 최종 제어 명령 |
| `/pure_pursuit_status` | `std_msgs/String` | 조향 상태 메시지 |

---

## 🎛️ 파라미터 상세 설명

### 1️⃣ behavior_node.py (Perception 기반 의사결정)

#### 기본 제어 파라미터
```yaml
control_period_s: 0.1                      # 제어 주기 (100ms)
```

#### 동적 속도 제어 (직선/곡선 구간별)
```yaml
desired_speed_straight_mps: 2.20           # 직선 구간 목표 속도 (m/s)
  → 증가: 직선에서 더 빠르게 주행
  → 감소: 직선에서 더 천천히 주행
  
desired_speed_gentle_turn_mps: 1.25        # 완만한 곡선 목표 속도 (m/s)
  → 증가: 완만한 코너에서 더 빠르게
  → 감소: 완만한 코너에서 더 천천히
  
desired_speed_sharp_turn_mps: 0.95         # 급한 곡선 목표 속도 (m/s)
  → 증가: 급한 코너에서 더 빠르게 (안정성 저하 주의)
  → 감소: 급한 코너에서 더 천천히 (지나친 감속 주의)

# 곡선 판정 임계값 (차량 기준 측방 거리)
turn_threshold_abs_local_y_small: 0.35     # 직선/완만한곡선 경계 (m)
  → 증가: 더 큰 오차까지 "직선"으로 간주 (코너 감지 늦음)
  → 감소: 더 작은 오차에서 "코너"로 전환 (지나친 감속)
  
turn_threshold_abs_local_y_large: 0.85     # 완만한곡선/급한곡선 경계 (m)
  → 증가: 더 큰 오차까지 "완만한곡선"으로 간주
  → 감소: 더 작은 오차에서 "급한곡선"으로 전환
```

#### 목표점 생성 (Lookahead 방식)
```yaml
lookahead_straight_m: 8.0                  # 직선 구간 미리보기 거리 (m)
  → 증가: 더 앞을 보고 목표점 선택 (완만한 주행, 반응 느림)
  → 감소: 가까운 곳만 보고 목표점 선택 (민감한 주행, 흔들림)
  
lookahead_turn_m: 4.2                      # 곡선 구간 미리보기 거리 (m)
  → 증가: 코너에서 더 큰 호를 그리며 회전
  → 감소: 코너에서 더 작은 호를 그리며 회전

# 목표점 평균화 (노이즈 필터링)
averaging_window_straight: 8                # 직선: 몇 개 포인트 평균할지
  → 증가: 더 부드러움 (반응 느림)
  → 감소: 더 예민함 (흔들림)
  
averaging_window_turn: 3                    # 곡선: 몇 개 포인트 평균할지
  → 증가: 코너에서도 부드러움
  → 감소: 코너에서 더 예민하게 반응
```

#### 목표점 안정화
```yaml
target_y_clamp_m: 1.8                      # 측방 오차 최대값 (m)
  → 증가: 더 큰 조향 가능 (급격한 회전)
  → 감소: 제한된 조향만 가능 (안전하지만 회전력 부족)
  
center_offset_m: -0.10                     # 곡선 구간 오프셋 (m, 음수=좌측)
  → 양수: 차선의 우측 중심 추종
  → 음수: 차선의 좌측 중심 추종
  
target_smoothing_alpha: 0.60                # 목표점 필터 강도 (0~1)
  → 높을수록: 새로운 값을 빠르게 반영 (흔들림 가능)
  → 낮을수록: 이전 값을 많이 유지 (반응 느림)
  수식: target_filtered = 0.60 * new + 0.40 * prev
```

#### Perception 타임아웃
```yaml
lane_timeout_s: 0.5                        # 차선 데이터 timeout (초)
  → 0.5초 이상 데이터 없으면 정지 (보수적)
  
obstacle_timeout_s: 0.5                    # 장애물 데이터 timeout (초)
  
traffic_light_timeout_s: 2.0               # 신호등 데이터 timeout (초)
  → 더 길게: 신호등 신뢰도 낮음
  → 더 짧게: 신호등 신뢰도 높음
```

#### 안전 거리 임계값
```yaml
caution_distance_m: 20.0                   # 주의 거리 (감속 시작)
  → 증가: 더 멀리서 감속 시작 (여유 있음)
  → 감소: 더 가까워서야 감속 (급격한 감속)
  
emergency_stop_distance_m: 3.5             # 긴급정지 거리 (m)
  → 증가: 더 멀리서 긴급정지 (안전하지만 불편)
  → 감소: 더 가까워서야 긴급정지 (위험)
```

#### Perception 데이터 필터
```yaml
lane_y_limit_m: 8.0                        # 차선 측방 범위 제한 (m)
  → 증가: 더 넓은 범위의 차선 포인트 허용
  → 감소: 더 좁은 범위만 사용 (노이즈 감소)
  
obstacle_corridor_half_width_m: 2.5        # 장애물 감지 폭 (m)
  → 증가: 더 넓은 범위의 장애물 감지
  → 감소: 더 좁은 범위만 감지 (정면만)
```

---

### 2️⃣ speed_control_node.py (부드러운 속도 제어)

#### Smoothing 파라미터 (Perception 기반 대응)
```yaml
throttle_filter_alpha: 0.08                # 스로틀 필터 강도 (0~1)
  → 높을수록: 빠르게 반응 (흔들림)
  → 낮을수록: 천천히 반응 (부드러움, 반응 지연)
  수식: throttle = 0.08 * new + 0.92 * prev
  
brake_filter_alpha: 0.10                   # 브레이크 필터 강도
  → 높을수록: 빠르게 반응
  → 낮을수록: 천천히 반응 (급격한 감속 방지)
```

#### 스로틀 단계 제어 (오차 기반)
```yaml
throttle_fast: 0.40                        # 오차 > 0.30m/s 일 때
  → 증가: 빠른 가속 (0~1.0 범위)
  → 감소: 느린 가속
  
throttle_medium: 0.32                      # 0.15 < 오차 <= 0.30m/s
  → 증가: 중간 속도 가속
  → 감소: 완만한 가속
  
throttle_hold: 0.22                        # -0.05 <= 오차 <= 0.15m/s
  → 증가: 속도 유지 수준 상향
  → 감소: 속도 유지 수준 하향
  
throttle_trim: 0.10                        # -0.20 < 오차 < -0.05m/s
  → 증가: 최소 주행 스로틀 상향
  → 감소: 최소 주행 스로틀 하향
  
throttle_min: 0.04                         # 오차 <= -0.20m/s
  → 증가: 감속 중 최소값 상향
  → 감소: 감속 중 최소값 하향
```

#### 정지 브레이크 제어
```yaml
stop_brake_high_speed: 0.35                # 현재속도 > 0.25m/s 일 때
  → 증가: 강한 브레이킹 (빨리 정지)
  → 감소: 약한 브레이킹 (천천히 정지)
  
stop_brake_low_speed: 0.75                 # 현재속도 <= 0.25m/s 일 때
  → 증가: 강한 브레이킹
  → 감소: 약한 브레이킹
```

#### Soft-start (부드러운 출발)
```yaml
launch_speed_threshold: 0.15               # 출발 판정 속도 (m/s)
  → 증가: 더 큰 속도까지 soft-start 적용
  → 감소: 빠르게 일반 제어로 전환
  
launch_throttle: 0.08                      # 출발 초기 스로틀
  → 증가: 강한 출발 (쏠림 방지)
  → 감소: 약한 출발 (느린 시작)
  
launch_throttle_max: 0.12                  # 출발 최대 스로틀
  
throttle_filter_alpha_launch: 0.06         # 출발시 필터 강도 (더 부드러움)
  
launch_duration_s: 1.0                     # 출발 지속 시간 (초)
  → 증가: 더 오래 soft-start 적용
  → 감소: 빠르게 일반 제어로 전환
```

---

### 3️⃣ pure_pursuit_node.py (안정적인 조향 제어)

#### 기본 파라미터
```yaml
wheelbase: 2.9                             # 차량 축거 (m)
  → 증가: 더 큰 회전 반경 (안정적, 조향 완만)
  → 감소: 더 작은 회전 반경 (민감, 흔들림)
  
max_steering_angle_rad: 0.75               # 최대 조향각 (라디안)
  → 증가: 더 큰 회전 가능 (날카로운 회전)
  → 감소: 제한된 회전만 가능 (부드러운 회전)
```

#### Smoothing (Perception 노이즈 필터)
```yaml
steer_ema_alpha: 0.18                      # 조향 필터 강도 (0~1)
  → 높을수록: 빠르게 반응 (perception 노이즈 직반영)
  → 낮을수록: 천천히 반응 (부드러움, 반응 지연)
  수식: steer_filtered = 0.18 * new + 0.82 * prev
  
control_period_s: 0.05                     # 제어 주기 (50ms)
```

#### 조향 Gain (감도 조절)
```yaml
steering_gain: 0.58                        # 기본 조향 게인
  → 증가: 같은 오차에 더 크게 회전 (민감)
  → 감소: 같은 오차에 더 약하게 회전 (무디림)
  
left_turn_gain: 1.0                        # 좌회전 게인 배수
  → 1.0 초과: 좌회전 강화
  → 1.0 미만: 좌회전 약화
  
right_turn_gain: 1.0                       # 우회전 게인 배수
  → 1.0 초과: 우회전 강화
  → 1.0 미만: 우회전 약화
```

#### 조향 변화율 제한 (급격한 변화 방지)
```yaml
max_steer_delta_per_cycle: 0.040           # 한 주기당 최대 변화량 (라디안)
  → 증가: 빠른 조향 변화 가능 (흔들림)
  → 감소: 천천히 조향 변화 (부드러움, 반응 지연)
```

#### 횡방향 오차 제한
```yaml
target_y_clamp_m: 1.40                     # 목표점 측방 제한 (m)
  → 증가: 더 큰 측방 오차 허용 (급격한 회전)
  → 감소: 작은 측방 오차만 허용 (부드러운 회전)
```

#### Timeout 안전
```yaml
command_timeout_s: 1.0                     # 신호 손실 timeout (초)
  → 증가: 더 오래 신호 대기 (신뢰도 낮음)
  → 감소: 빨리 안전정지 (민감함)
```

---

## 🚀 실행 방법

### 1. 빌드
```bash
cd ~/dream_ws/neuro_ws
colcon build --packages-select neuro_decision
source install/setup.bash
```

### 2. 실행
```bash
# 전체 노드 실행 (launch 파일)
ros2 launch neuro_decision neuro_decision.launch.py

# 개별 노드 실행
ros2 run neuro_decision behavior_node
ros2 run neuro_decision speed_control_node
ros2 run neuro_decision pure_pursuit_node
```

### 3. 모니터링
```bash
# 디버그 메시지 확인
ros2 topic echo /behavior_debug_text
ros2 topic echo /speed_control_debug_text
ros2 topic echo /pure_pursuit_status

# 목표값 모니터링
ros2 topic echo /desired_speed
ros2 topic echo /target_point
```

---

## 🔧 파라미터 조정 가이드

### 상황: 직선에서 쓸데없이 자주 흔들림
→ `target_smoothing_alpha` 감소 (0.60 → 0.50)
→ `steer_ema_alpha` 증가 (0.18 → 0.25)
→ `averaging_window_straight` 증가 (8 → 10)

### 상황: 코너를 돌때 너무 천천히 감속
→ `desired_speed_gentle_turn_mps` 증가 (1.25 → 1.50)
→ `desired_speed_sharp_turn_mps` 증가 (0.95 → 1.10)

### 상황: 출발이 너무 끊김
→ `throttle_filter_alpha_launch` 감소 (0.06 → 0.04)
→ `launch_throttle` 증가 (0.08 → 0.10)

### 상황: 장애물 회피가 너무 늦음
→ `caution_distance_m` 증가 (20.0 → 25.0)
→ `astar_trigger_distance_m` 증가 (15.0 → 20.0)

---

## 📊 시스템 아키텍처

```
Perception Module
    ↓
behavior_node (perception 기반 판단)
    ├─ 신호등 상태 판단
    ├─ 장애물 거리 계산
    ├─ 차선점 기반 목표점 생성
    └─→ /desired_speed, /target_point
        ↓
speed_control_node (부드러운 속도 제어)
    ├─ 목표속도 추종 (exponential filter)
    ├─ 오차 기반 단계 제어
    └─→ /speed_command, /brake_command
        ↓
pure_pursuit_node (안정적인 조향)
    ├─ Pure pursuit 알고리즘
    ├─ 조향 필터링 (EMA + rate limit)
    └─→ /carla/ego_vehicle/vehicle_control_cmd
        ↓
CARLA Vehicle
```

---

## 🛠️ 주요 개선 사항 (v4+)

✅ **Perception 기반**: CARLA waypoint 대신 perception 데이터 사용
✅ **강화된 필터링**: Exponential smoothing으로 노이즈 억제
✅ **부드러운 제어**: 모든 단계에서 필터 적용으로 끊김 없음
✅ **동적 속도**: 직선/곡선별 차등 속도 제어
✅ **안정적인 조향**: EMA 필터 + rate limit + 게인 조절
✅ **다중 안전장치**: timeout 감시 + 신호 손실 감지

---

## 📝 팀 정보

- **판단/제어 모듈**: 당신 (behavior_node, speed_control_node, pure_pursuit_node)
- **Perception 모듈**: 팀원 (차선 인식, 장애물 감지)

