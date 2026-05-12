# neuro_decision

Perception 기반 실차 판단/조향 모듈.

이 패키지는 유아용 전동차 실차 자율주행용 판단과 조향만 담당한다.
실제 모터, 서보, PWM, PID, failsafe는 Arduino 제어부에서 처리한다.
CARLA 전용 제어 메시지는 사용하지 않는다.

## 🔌 현재 토픽 계약

### 제어팀이 받는 토픽
- `/desired_speed` (`std_msgs/Float64`): 목표 속도 [m/s]
- `/desired_steering_angle_rad` (`std_msgs/Float64`): 목표 조향각 [rad]
- `/desired_steering_normalized` (`std_msgs/Float64`): 정규화 조향값 [-1, 1]
- 조향/속도 외 throttle, brake, steer 직접 계산은 하지 않는다.

### 인지팀이 넘겨주는 토픽
- `/perception/real_world_lane_points` (`sensor_msgs/PointCloud2`)
- `/perception/closest_obstacle` (`sensor_msgs/PointCloud2`)
- `drivable_area_topic` 파라미터 기본값 `/perception/real_world_drivable_points` (`sensor_msgs/PointCloud2`)
- `/traffic_light_state` (`std_msgs/String`)
- `speed_topic` 파라미터 기본값 `/carla/ego_vehicle/speedometer` (`std_msgs/Float32`)
- `/yolopv2/detections` (`vision_msgs/Detection2DArray`)

## 📋 개요

현재 실사용 제어 스택:

| 노드 | 역할 | 입력 | 출력 |
|------|------|------|------|
| **behavior_node** | Perception 데이터 기반 판단 | 차선점, 장애물, 주행 가능 영역, 신호등, 속도 | 목표속도, 목표점, 행동 상태 |
| **steering_command_node** | 목표점 기반 조향 명령 생성 | 목표점, 행동 상태 | 조향각, 정규화 조향값 |

핵심 정책:
- `enable_obstacle_avoidance`, `enable_lane_change`, `enable_cutin_detection`, `enable_traffic_light`, `enable_stopline`는 기본값이 모두 `False`다.
- Local Lattice Avoidance는 drivable area 안에서 local target point를 고르고, A*는 사용하지 않는다.
- 차선 변경은 큰 lateral offset을 quintic smoothstep으로 보간한다.
- 가까운 장애물은 회피보다 STOP 또는 EMERGENCY_STOP이 우선이다.
- `ros2 launch neuro_decision neuro_decision.launch.py`는 behavior_node와 steering_command_node만 띄운다.

보관 중인 CARLA용 백업 노드:
- `speed_control_node.py`
- `pure_pursuit_node.py`
- `waypoint_behavior_node.py`

---

## 🛰️ 필요한 Perception 토픽 (입력)

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/perception/real_world_lane_points` | `sensor_msgs/PointCloud2` | 차선 중심 포인트 (local 좌표계) |
| `/perception/closest_obstacle` | `sensor_msgs/PointCloud2` | 가장 가까운 장애물 위치 |
| `drivable_area_topic` 기본값 `/perception/real_world_drivable_points` | `sensor_msgs/PointCloud2` | 주행 가능 영역 |
| `/traffic_light_state` | `std_msgs/String` | 신호등 상태 (RED/YELLOW/GREEN) |
| `/carla/ego_vehicle/speedometer` | `std_msgs/Float32` | 현재 속도 (m/s) |
| `/yolopv2/detections` | `vision_msgs/Detection2DArray` | 차량/보행자 검출 결과 |

---

## 📤 발행하는 토픽 (출력)

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/desired_speed` | `std_msgs/Float64` | 목표 속도 (m/s) |
| `/target_point` | `geometry_msgs/Point` | 차량 기준 목표점 (m) |
| `/behavior_state` | `std_msgs/String` | 현재 행동 상태 |
| `/behavior_debug_text` | `std_msgs/String` | 디버그 메시지 |
| `/desired_steering_angle_rad` | `std_msgs/Float64` | 목표 조향각 (rad) |
| `/desired_steering_normalized` | `std_msgs/Float64` | 정규화 조향값 (-1.0~1.0) |
| `/steering_command_debug_text` | `std_msgs/String` | 조향 계산 디버그 메시지 |

---

## 🎛️ 파라미터 상세 설명

### 1️⃣ behavior_node.py (Perception 기반 의사결정)

#### 기본 제어 파라미터
```yaml
control_period_s: 0.1                      # 제어 주기 (100ms)
```

#### 동적 속도 제어 (직선/곡선 구간별)
```yaml
desired_speed_straight_mps: 1.10           # 직선 구간 목표 속도 (m/s)
  → 증가: 직선에서 더 빠르게 주행
  → 감소: 직선에서 더 천천히 주행
  
desired_speed_gentle_turn_mps: 0.75        # 완만한 곡선 목표 속도 (m/s)
  → 증가: 완만한 코너에서 더 빠르게
  → 감소: 완만한 코너에서 더 천천히
  
desired_speed_sharp_turn_mps: 0.45         # 급한 곡선 목표 속도 (m/s)
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
lookahead_straight_m: 3.5                  # 직선 구간 미리보기 거리 (m)
  → 증가: 더 앞을 보고 목표점 선택 (완만한 주행, 반응 느림)
  → 감소: 가까운 곳만 보고 목표점 선택 (민감한 주행, 흔들림)
  
lookahead_turn_m: 2.0                      # 곡선 구간 미리보기 거리 (m)
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
target_y_clamp_m: 1.2                      # 측방 오차 최대값 (m)
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

#### 회피/차선변경 안정화
```yaml
avoidance_target_smoothing_alpha: 0.25     # AVOID_OBSTACLE target_y EMA 강도
avoidance_target_y_rate_limit_m: 0.15      # AVOID_OBSTACLE target_y 1회 변화량 제한

lane_change_ramp_duration_s: 2.0           # 차선 변경 보간 시간
  → quintic smoothstep으로 시작/끝을 부드럽게 만듦

avoidance_commit_count: 3                  # lattice 후보가 몇 번 연속 유효해야 회피 시작할지
avoidance_release_count: 5                 # 후보가 사라진 뒤 몇 번 더 유지할지
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

#### 신호등/정지선 구조
```yaml
enable_traffic_light: false                # 기본 OFF
enable_stopline: false                     # 기본 OFF
unknown_light_forced_go_s: 7.0             # UNKNOWN 장기 지속 시 stuck 방지 옵션
unknown_light_stop_duration_s: 3.0         # UNKNOWN + stopline 감지 시 정지 latch
intersection_ignore_duration_s: 5.0        # 교차로 직후 stopline 재검출 무시용 timestamp
```

#### 안전 거리 임계값
```yaml
caution_distance_m: 20.0                   # 주의 거리 (감속 시작)
  → 증가: 더 멀리서 감속 시작 (여유 있음)
  → 감소: 더 가까워서야 감속 (급격한 감속)
  
emergency_stop_distance_m: 0.8             # 긴급정지 거리 (m)
  → 증가: 더 멀리서 긴급정지 (안전하지만 불편)
  → 감소: 더 가까워서야 긴급정지 (위험)

near_obstacle_stop_distance_m: 1.8         # 근접 장애물 정지 거리 (m)
```

#### Perception 데이터 필터
```yaml
lane_y_limit_m: 8.0                        # 차선 측방 범위 제한 (m)
  → 증가: 더 넓은 범위의 차선 포인트 허용
  → 감소: 더 좁은 범위만 사용 (노이즈 감소)
  
obstacle_corridor_half_width_m: 1.0        # 장애물 감지 폭 (m)
  → 증가: 더 넓은 범위의 장애물 감지
  → 감소: 더 좁은 범위만 감지 (정면만)
```

#### Local Lattice Avoidance
```yaml
enable_obstacle_avoidance: false           # 기본 OFF
avoidance_method: LATTICE                  # 현재는 Local Lattice Avoidance만 사용
avoidance_target_x_m: 3.0                  # 후보 target 전방 거리
avoidance_lateral_candidates_m: [-1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8]
```

---

### 2️⃣ speed_control_node.py (CARLA 백업용 속도 제어)

이 노드는 현재 실차 제어 경로에서는 사용하지 않고, CARLA 시뮬레이션 백업으로 보관 중입니다.

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

### 3️⃣ steering_command_node.py (실차 조향 명령)

이 노드는 behavior_node가 발행한 `/target_point`와 `/behavior_state`만 받아 조향각을 만든다.
실차에서는 throttle, brake, steer를 직접 계산하지 않고 조향각과 정규화 조향값만 발행한다.

#### 기본 파라미터
```yaml
wheelbase: 0.95                            # 차량 축거 (m)
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
command_timeout_s: 0.5                     # target_point / behavior_state timeout
```

#### 상태별 조향 profile
```yaml
LANE_KEEPING:
  gain: 0.55
  delta_per_cycle: 0.035
  ema_alpha: 0.15

FOLLOW_VEHICLE:
  gain: 0.56
  delta_per_cycle: 0.035
  ema_alpha: 0.15

PREPARE_LANE_CHANGE_LEFT / PREPARE_LANE_CHANGE_RIGHT:
  gain: 0.60
  delta_per_cycle: 0.045
  ema_alpha: 0.20

LANE_CHANGE_LEFT / LANE_CHANGE_RIGHT:
  gain: 0.65
  delta_per_cycle: 0.055
  ema_alpha: 0.22

RETURN_TO_LANE:
  gain: 0.52
  delta_per_cycle: 0.040
  ema_alpha: 0.18

AVOID_OBSTACLE:
  gain: 0.58
  delta_per_cycle: 0.040
  ema_alpha: 0.18
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

#### 디버그 메시지
```yaml
profile_state: 현재 적용된 조향 프로파일 이름
target_y_clamped: target_y 제한 적용 여부
safe_zero_reason: timeout / unknown_state / stop_state / emergency_stop_state
```
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
ros2 run neuro_decision steering_command_node
```

### 3. 모니터링
```bash
# 디버그 메시지 확인
ros2 topic echo /behavior_debug_text
ros2 topic echo /steering_command_debug_text

# 목표값 모니터링
ros2 topic echo /desired_speed
ros2 topic echo /target_point
ros2 topic echo /desired_steering_angle_rad
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
→ `enable_obstacle_avoidance` 를 `True`로 켠 뒤 `avoidance_commit_count` 또는 `avoidance_target_y_rate_limit_m` 조정

---

## 📊 시스템 아키텍처

```
Perception Module
    ↓
behavior_node (perception 기반 판단)
    ├─ 신호등/정지선은 기본 OFF 구조
    ├─ 장애물 거리 계산 및 Local Lattice Avoidance
    ├─ 차선점 기반 목표점 생성
    └─→ /desired_speed, /target_point, /behavior_state
        ↓
steering_command_node (목표 조향각 생성)
  ├─ target_point + behavior_state 기반 조향 계산
  ├─ 상태별 gain / EMA / rate limit
  └─→ /desired_steering_angle_rad, /desired_steering_normalized
        ↓
제어팀 / 아두이노
  ├─ /desired_speed
  └─ /desired_steering_angle_rad
    ↓
실차 모터 / 서보
```

---

## 🛠️ 주요 개선 사항 (v4+)

✅ **Perception 기반**: CARLA waypoint 대신 perception 데이터 사용
✅ **강화된 필터링**: Exponential smoothing과 rate limit으로 노이즈 억제
✅ **부드러운 제어**: lattice target smoothing과 quintic lane-change 보간 적용
✅ **동적 속도**: 직선/곡선별 차등 속도 제어
✅ **안정적인 조향**: profile 기반 EMA 필터 + rate limit + timeout safe-zero
✅ **다중 안전장치**: timeout 감시 + 신호 손실 감지 + STOP 우선 정책

---

## 📝 팀 정보

- **판단/제어 모듈**: 당신 (behavior_node, steering_command_node)
- **Perception 모듈**: 팀원 (차선 인식, 장애물 감지)

## behavior_node 내부 상태 변수 설명

아래는 `behavior_node`에서 유지하는 주요 내부 상태 변수 그룹과 간단한 설명이다. 변수명, 업데이트 지점, 사용 용도를 참고하라.

1) Perception 입력 저장 변수
- **lane_points**: `/perception/real_world_lane_points` 콜백에서 업데이트, 차선 기반 목표점 계산에 사용
- **obstacle_points**: `/perception/closest_obstacle` 콜백에서 업데이트, 장애물 판정 및 avoidance 계산에 사용
- **drivable_area_points**: `drivable_area_topic` 콜백에서 업데이트, 로컬 Lattice 및 목표점 드라이버블 검사에 사용
- **latest_detections**: `/yolopv2/detections` 콜백에서 업데이트, 객체 분류 요약에 사용
- **detection_class_summary**: `get_detection_class_summary()`로 계산, detection 기반 보조 판단에 사용

2) 장애물 / 선행차 상태
- **obstacle_distance** / **obstacle_x** / **obstacle_y**: `compute_closest_relevant_obstacle()`에서 계산, 근접 장애물 결정에 사용
- **lead_vehicle_distance** / **lead_vehicle_x** / **lead_vehicle_y**: `select_forward_vehicle_on_path()`에서 계산, Follow Vehicle 상태 판단 및 속도 제어에 사용
- **lead_vehicle_lost_count**: 선행차 지속성 추적에 사용

3) 차선 변경 상태
- **lane_change_start_time**: 차선 변경 시작 시간, 보간(progress) 계산에 사용
- **left_lane_possible** / **right_lane_possible**: `evaluate_lane_change_options()` 결과, 차선 변경 가용성 판단에 사용
- **left_target_y_debug** / **right_target_y_debug**: 차선 변경 타겟 debug 정보
- **return_finish_count**: 복귀 완료 조건 유지 카운트

4) Local Lattice Avoidance 상태
- **avoidance_candidate_count**: 후보가 연속 유효했는지 카운트 (commit hysteresis)
- **avoidance_release_count_current**: 후보 소실 후 유지 카운트 (release hysteresis)
- **avoidance_active**: 현재 회피 모드 활성 여부
- **avoidance_target_x** / **avoidance_target_y**: 선택된 회피 목표점
- **avoidance_score**: 선택된 후보 점수
- **smoothed_avoidance_target_y** / **avoidance_smoothing_initialized**: EMA 기반 회피 목표 Y 평활 상태

5) Traffic / Stopline 상태
- **traffic_light_state**: `/traffic_light_state` 콜백으로 업데이트 (RED/YELLOW/GREEN/UNKNOWN)
- **stopline_mode**: enable_stopline 기반 상태
- **stopline_hold_active** / **stopline_hold_until** / **stopline_hold_started_at**: stopline 정지 latch 상태
- **unknown_light_forced_go_until** / **intersection_ignore_until**: UNKNOWN 관련 임시 허용/무시 타이머

6) Pedestrian / Cut-in 상태
- **pedestrian_intrusion_detected** / **pedestrian_intrusion_distance** / **pedestrian_intrusion_severity**: 보행자 침입 감지 및 심각도
- **pedestrian_detection_warning**: 감지 토픽만으로 위험 신호가 있을 때 경고
- **cutin_intrusion_detected** / **cutin_intrusion_distance**: cut-in 휴리스틱 결과

7) Target / Speed smoothing 상태
- **target_x** / **target_y**: 현재 목표점
- **filtered_target_x** / **filtered_target_y**: smoothing 적용된 목표점
- **desired_speed** / **filtered_desired_speed**: 목표 속도 및 smoothing

8) Valid / Timeout 상태
- **lane_valid**, **obstacle_valid**, **drivable_valid**, **tl_valid**, **speed_valid**, **detection_valid**: 각 입력의 liveliness/타임아웃 유효성
- **received_lane**, **received_obstacle**, **received_drivable**, **received_speed**, **received_detection**, **received_traffic_light**: 콜백 수신 여부

## 현재 구현되어 있지만 실차 검증 또는 보완이 필요한 기능

아래 기능들은 코드상으로는 구현되어 있으나, 실차 테스트 전까지는 보수적으로 사용하거나 비활성화해두는 것을 권장한다.

1. Local Lattice Avoidance
- 코드상으로 구현되어 있음.
- `enable_obstacle_avoidance = False` 기본 OFF.
- RViz에서 avoidance target 검증 필요.
- drivable area 품질에 크게 의존하므로 실차 전 검증 필수.

2. Lane Change
- 상태와 target 생성 구조는 구현되어 있음.
- `enable_lane_change = False` 기본 OFF.
- 유아용 전동차에서는 차선 변경 시험은 마지막 단계로 권장.

3. Traffic Light
- `/traffic_light_state` 입력 구조가 존재함.
- `enable_traffic_light = False` 기본 OFF.
- RED/YELLOW/GREEN/UNKNOWN 처리를 내부적으로 지원하지만 실차 전 검증 필요.

4. Stopline
- stopline hold 구조는 있으나 실제 stopline detection 입력 토픽이 확정되지 않음.
- `enable_stopline = False` 기본 OFF.

5. Pedestrian Detection
- YOLO detection과 장애물 포인트를 함께 사용해 판단함.
- pedestrian label은 내부적으로 `pedestrian`으로 통일됨.
- false positive/negative 검증 필요.

6. Cut-in Detection
- 휴리스틱만 구현되어 있음 (`enable_cutin_detection = False` 기본 OFF).

7. Obstacle Classification
- z값과 detection summary를 사용함; `/perception/closest_obstacle`의 z 의미(centroid/raw point) 확인 필요.

8. Speed Feedback
- `speed_topic` 기본값은 `/carla/ego_vehicle/speedometer` 일 수 있음. 실차에서는 제어파트와 토픽 합의 필요.

9. QoS
- PointCloud2 토픽의 QoS 일치 여부 확인 필요 (인지 파트와 QoS mismatch 시 callback 미수신 가능).

10. Behavior Node Size
- 현재 `behavior_node.py`에 기능이 집중되어 있음. 장기적으로는 `detection_utils.py`, `obstacle_reasoner.py`, `lattice_avoidance.py`, `speed_policy.py` 등으로 분리 권장.

## 안전 기본값 (요약)
- `enable_obstacle_avoidance = False`
- `enable_lane_change = False`
- `enable_cutin_detection = False`
- `enable_traffic_light = False`
- `enable_stopline = False`

기본 동작: lane 미수신 -> STOP, 가까운 장애물 -> STOP/EMERGENCY_STOP 우선, STOP 상태일 때 desired_speed 즉시 0.

## 제어파트 인터페이스 유지사항
- `/desired_speed` : `std_msgs/msg/Float64` (m/s)
- `/desired_steering_angle_rad` : `std_msgs/msg/Float64` (rad)
- `/behavior_state` : `std_msgs/msg/String` (참고용)

제어파트에서 구현해야 할 failsafe:
- `/desired_speed` timeout 시 모터 정지
- `/desired_steering_angle_rad` timeout 시 조향 중립
- STOP/EMERGENCY_STOP 상태 시 모터 정지
- 물리 E-stop

---

## 변경 및 주의사항
- `ObstacleType.VEHICLE`는 코드 상에서 "car-like obstacle"을 의미한다. YOLO label은 내부적으로 `car`로 통일한다.
- `lane_blocked_distance_m` 파라미터를 도입하여 lane blocked 판정 기준을 명확히 함.
- `require_drivable_for_lane_keeping` 파라미터로 drivable area가 끊겼을 때의 동작을 제어할 수 있음.


