# neuro_decision 패키지

자율주행 유아용 전동차를 위한 **판단(Decision) 및 조향(Steering) 모듈**입니다.

| 노드 | 기능 | 입력 | 출력 |
|------|------|------|------|
| **behavior_node** | 상태 결정 + 속도/목표점 생성 | LiDAR, 카메라, 신호등, 속도 | 상태, 속도, 목표점, 디버그 정보 |
| **steering_command_node** | 목표점 → 조향각 변환 | 목표점, 현재 상태 | 조향각 (도) |

---

## 📋 기능 개요

### ✅ **활성화된 기능** (실차 배포용)

#### 1. **Lane Keeping** (차선 유지)
- **입력**: LiDAR 차선점 (`/perception/lane_points`)
- **처리**: 
  - Lookahead 기반 목표점 생성 (3.5m/직선, 2.0m/곡선)
  - 이동 평균 필터 (8개/직선, 3개/곡선)
  - EMA 스무싱
- **출력**: 상태 `LANE_KEEPING`, 속도, 목표점
- **안전**: 
  - 차선 타임아웃 0.5s → STOP
  - 첫 데이터 수신 전까지 대기

#### 2. **Dynamic Speed Control** (곡률 기반 속도)
```
직선        : 1.10 m/s
완만한 곡선 : 0.75 m/s
급한 곡선   : 0.45 m/s
상한선 (안전): 1.40 m/s (유아 보호)
```
- 목표점의 측방 거리(Y)에 따라 동적 결정
- STOP/EMERGENCY_STOP 시 즉시 0 m/s

#### 3. **Follow Vehicle** (선행차 추종)
- **입력**: YOLO v2 감지 (`/perception/detections`)
- **처리**:
  - 거리 기반 추종 (1.6~5.0m)
  - 속도 감소 인수 0.8배
  - Time headway 1.2s
  - 최소 속도 0.25 m/s
- **상태**: `FOLLOW_VEHICLE`
- **타임아웃**: 0.7s → fallback to lane keeping
- **비활성화 조건**: 차선 변경 비활성화 시 사용 불가

#### 4. **Obstacle Detection & Emergency Stop** (다층 안전)

**Layer 1: Pedestrian Intrusion**
- 감지 거리: 5.0m 전방, ±1.5m 횡방
- 자신감 역치: 0.30
- **Emergency Stop** (0.6m): 즉시 정지
- **Stop** (1.5m): 속도 0
- 예외 처리: 파싱 에러 시 안전한 fallback

**Layer 2: Near Obstacle**
- 거리 0.8m 미만 → `EMERGENCY_STOP` (즉시)
- 거리 1.8m 미만 → `STOP` (hysteresis 있음)
- 타임아웃 0.5s → STOP

**Layer 3: Lane Blocked Detection**
- 조건: `obstacle_valid=true` **AND** `거리 < 3.0m`
- **주의**: `drivable_valid` 끊김 무시 (안정성)
- 차선 변경 비활성화 시 즉시 STOP

#### 5. **Steering Control** (목표점 → 조향각)
- **입력**: 목표점 (x, y) + 현재 상태
- **계산**:
  - Ackermann 기구학
  - Pure Pursuit 알고리즘
  - 상태별 제어 gain (LANE_KEEPING: 0.55, LANE_CHANGE: 0.65 등)
- **필터**: EMA 스무싱 (진동 감소)
- **출력**: `/desired_steering_angle_deg` (-30~30도)

---

### 🔴 **비활성화된 기능** (현재 OFF)

| 기능 | 상태 | 파라미터 | 설명 |
|------|------|---------|------|
| **Lane Change** | ❌ OFF | `enable_lane_change=False` | 차선 변경 완전 비활성화 (실차 안정성) |
| **Traffic Light** | ❌ OFF | `enable_traffic_light=False` | 신호등 미지원 (센서 미배치) |
| **Stopline** | ❌ OFF | `enable_stopline=False` | 정지선 미지원 (센서 미배치) |
| **Obstacle Avoidance** | ❌ OFF | `enable_obstacle_avoidance=False` | Local Lattice 회피 비활성 |
| **Cut-in Detection** | ❌ OFF | `enable_cutin_detection=False` | 급진입 감지 미지원 |

**주의**: 차선 변경 활성화 시에도 선행차 추종이 제한될 수 있습니다.

---

## 📡 Topic 인터페이스

### **구독 (Inputs)**

| 토픽 | 메시지 타입 | 설명 | 타임아웃 |
|------|-----------|------|---------|
| `/perception/lane_points` | `PointCloud2` | LiDAR 차선 포인트 클라우드 | 0.5s |
| `/perception/obstacle_points` | `PointCloud2` | LiDAR 장애물 포인트 클라우드 | 0.5s |
| `/perception/detections` | `Detection2DArray` | YOLO v2 감지 결과 (선행차, 보행자) | 0.7s |
| `/perception/real_world_drivable_points` | `PointCloud2` | Drivable area 포인트 | 0.5s |
| `/vehicle/current_speed_mps` | `Float32` | 현재 속도 (m/s) | 1.0s |
| `/traffic_light_state` | `String` | 신호등 상태 (RED/YELLOW/GREEN) | 2.0s |

### **발행 (Outputs)**

| 토픽 | 메시지 타입 | 설명 | 빈도 |
|------|-----------|------|------|
| `/planning/desired_speed_mps` | `Float32` | 목표 속도 (m/s) | 10 Hz |
| `/planning/target_point` | `Point` | 목표점 (x, y) | 10 Hz |
| `/planning/current_state` | `String` | 현재 상태 (상태명) | 10 Hz |
| `/planning/desired_steering_angle_deg` | `Float32` | 목표 조향각 (도) | 10 Hz |
| `/behavior_debug_text` | `String` | 디버그 정보 (상태, 거리, 이유) | 10 Hz |

---

## 🎯 상태 머신 (State Machine)

```
LANE_KEEPING (기본)
    ↓
FOLLOW_VEHICLE (선행차 감지)
    ↓
STOP (장애물 <1.8m or 타임아웃)
    ↓
EMERGENCY_STOP (장애물 <0.8m or 보행자 <0.6m)

Lane Change 비활성화 → 위 4개 상태만 사용
```

### **상태별 동작**

| 상태 | 속도 | 목표점 | 조향 | 종료 조건 |
|------|------|--------grep -R "update_obstacle_stop_hysteresis" -n planning/neuro_decision/neuro_decision/behavior_node.py|------|---------|
| `LANE_KEEPING` | 곡률 기반 (0.45~1.10) | 차선 중심 | Pure Pursuit | 선행차 감지 or 장애물 |
| `FOLLOW_VEHICLE` | 0.8×선행차 속도 | 선행차 추적 | Pure Pursuit | 선행차 손실 or 장애물 |
| `STOP` | 0.0 | 현재위치 | 0도 | 장애물 제거 (1.8m 이상) |
| `EMERGENCY_STOP` | 0.0 | 현재위치 | 0도 | 장애물 제거 (0.8m 이상) |

---

## ⚙️ 파라미터 상세 설명 & 튜닝 가이드

### **1️⃣ 속도 제어 (Speed Control)**

| 파라미터 | 기본값 | ↑ 증가 시 | ↓ 감소 시 | 권장 범위 | 주의사항 |
|---------|--------|---------|---------|---------|---------|
| `desired_speed_straight_mps` | 1.10 | 직선 주행 속도 ↑ | 주행이 느려짐 | 0.8~1.3 | 곡선에서 과속 방지 필요 |
| `desired_speed_gentle_turn_mps` | 0.75 | 부드러운 곡선 빠름 | 곡선에서 너무 느림 | 0.6~0.9 | sharp와 차이 최소 0.2 |
| `desired_speed_sharp_turn_mps` | 0.45 | 급곡선 주행 가능 | 급곡선 안정성 높음 | 0.3~0.6 | **유아 안전 최우선** |
| `max_desired_speed_mps` | 1.40 | 최대 속도 상향 ⚠️ | 속도 제한됨 | 1.2~1.5 | **변경 금지** (유아 안전) |
| `turn_threshold_abs_local_y_small` | 0.35 | 완만함 판정 늘어남 | 급함 판정 늘어남 | 0.2~0.5 | gentle↔sharp 전환점 |
| `turn_threshold_abs_local_y_large` | 0.85 | 급함 판정 범위 ↑ | 급함 판정 범위 ↓ | 0.7~1.0 | small보다 반드시 커야 함 |

**💡 튜닝 팁**:
- 직선에서 빠르고 싶으면: `desired_speed_straight_mps` ↑ (상한 1.3)
- 곡선에서 너무 느리면: `desired_speed_gentle_turn_mps` ↑ 
- 급곡선에서 흔들림 많음: `desired_speed_sharp_turn_mps` ↓

---

### **2️⃣ 목표점 생성 (Target Point Generation)**

| 파라미터 | 기본값 | ↑ 증가 시 | ↓ 감소 시 | 권장 범위 | 효과 |
|---------|--------|---------|---------|---------|------|
| `lookahead_straight_m` | 3.5 | 먼 곳 목표 → 부드러운 주행 | 가까운 곳 목표 → 반응 빠름 | 2.5~4.5 | **차선 안정성** |
| `lookahead_turn_m` | 2.0 | 곡선에서 미리 전환 | 곡선에서 따라가기 | 1.5~2.5 | **곡선 추종성** |
| `averaging_window_straight` | 8 | 평균화 ↑ → 노이즈 감소 | 평균화 ↓ → 빠른 반응 | 6~10 | **차선 흔들림 제거** |
| `averaging_window_turn` | 3 | 곡선 부드러움 ↑ | 곡선 반응 빠름 | 2~4 | straight와 다르게 설정 |
| `target_smoothing_alpha` | 0.60 | 부드러운 전환 (새 데이터 0.6 반영) | 빠른 전환 (새 데이터 덜 반영) | 0.4~0.8 | **EMA 필터 강도** |
| `center_offset_m` | -0.10 | 차선 오른쪽 | 차선 왼쪽 | -0.2~+0.2 | 차선 중심 보정 |

**💡 튜닝 팁**:
- 차선이 자꾸 흔들림: `averaging_window_straight` ↑ 또는 `target_smoothing_alpha` ↑
- 곡선에서 반응이 느림: `lookahead_turn_m` ↓ 또는 `averaging_window_turn` ↓
- 가파른 곡선 모서리 모서리 따라감: `target_smoothing_alpha` ↓

---

### **3️⃣ 안전 거리 (Safety Thresholds) ⚠️**

| 파라미터 | 기본값 | ↑ 증가 시 | ↓ 감소 시 | 권장 범위 | **주의** |
|---------|--------|---------|---------|---------|--------|
| `emergency_stop_distance_m` | 0.8 | 먼저 EMERGENCY_STOP | 너무 가까움 ⚠️ | 0.6~1.0 | **변경 금지** (충돌 리스크) |
| `near_obstacle_stop_distance_m` | 1.8 | 더 여유 있게 정지 | 장애물 가까이 정지 | 1.5~2.2 | EMERGENCY보다 커야 함 |
| `lane_blocked_distance_m` | 3.0 | 더 멀리서 차선 막힘 판정 | 더 가까이서 판정 | 2.5~3.5 | **추종/회피 결정 기준** |
| `pedestrian_emergency_distance_m` | 0.6 | 먼저 보행자 급정지 | 너무 가까움 ⚠️ | 0.5~0.8 | 보행자 보호 최우선 |
| `pedestrian_stop_distance_m` | 1.5 | 더 여유 있게 감속 | 보행자 가까이 정지 | 1.2~1.8 | EMERGENCY보다 커야 함 |

**💡 필드 테스트**:
```
권장 조정 순서:
1. emergency_stop_distance_m: 실차에서 최소 충돌 거리 실측 후 +0.2m
2. near_obstacle_stop_distance_m: 실제 제동 거리 측정
3. pedestrian_emergency_distance_m: 보행자 센서 캘리브레이션 후 설정
```

---

### **4️⃣ 타임아웃 (Sensor Staleness) ⏱️**

| 파라미터 | 기본값 | ↑ 증가 시 | ↓ 감소 시 | 권장 범위 | 권장사항 |
|---------|--------|---------|---------|---------|--------|
| `lane_timeout_s` | 0.5 | 차선 끊김 허용 범위 ↑ | 차선 끊김 민감함 | 0.3~0.7 | 10Hz 주기 = 0.1s |
| `obstacle_timeout_s` | 0.5 | 장애물 끊김 허용 범위 ↑ | 장애물 끊김 민감함 | 0.3~0.7 | lane과 동일 권장 |
| `detection_timeout_s` | 0.7 | YOLO 감지 끊김 허용 | YOLO 끊김 민감함 | 0.5~1.0 | YOLO가 더 자주 끊김 |
| `drivable_timeout_s` | 0.5 | drivable 끊김 허용 | drivable 민감함 | 0.3~0.7 | lane과 동일 권장 |
| `speed_timeout_s` | 1.0 | 속도 센서 끊김 허용 | 속도 끊김 민감함 | 0.8~1.5 | 속도계 신뢰도 낮음 |
| `traffic_light_timeout_s` | 2.0 | 신호등 미수신 허용 | 신호등 끊김 민감함 | 1.0~3.0 | 신호등 OFF 상태라 무관 |

**💡 최적화**:
- 센서 끊기는 현상이 자주 있음 → timeout 0.1~0.2초 정도 **여유** 가지기
- 부하 심할 때 LiDAR 끊김 → `lane_timeout_s` 0.7 정도로 상향 고려
- **안전 우선**: 타임아웃 ↓ = 빨리 STOP (안전하지만 느림)

---

### **5️⃣ 선행차 추종 (Follow Vehicle)**

| 파라미터 | 기본값 | ↑ 증가 시 | ↓ 감소 시 | 권장 범위 | 영향 |
|---------|--------|---------|---------|---------|------|
| `follow_vehicle_min_distance_m` | 1.6 | 가까이 따라감 | 멀리서 시작 | 1.2~2.0 | **최소 안전 거리** |
| `follow_vehicle_max_distance_m` | 5.0 | 멀리까지 추종 | 가까이만 추종 | 3.5~6.0 | 데이터 검증 거리 |
| `follow_vehicle_speed_reduction_factor` | 0.8 | 선행차보다 더 느림 (0.8배) | 선행차 속도 더 가깝게 | 0.7~0.9 | **충돌 방지** |
| `follow_vehicle_time_headway_s` | 1.2 | 거리 멀어짐 (시간 여유 ↑) | 거리 가까워짐 | 0.8~1.5 | 동적 거리 계산 |
| `follow_vehicle_min_speed_mps` | 0.25 | 낮은 속도에서 추종 | 빠른 속도에서만 추종 | 0.1~0.4 | 정체 상황 |
| `follow_vehicle_detection_score_threshold` | 0.35 | 낮은 신뢰도도 추종 | 높은 신뢰도만 추종 | 0.2~0.5 | YOLO 오검출 제어 |

**💡 튜닝**:
- 자꾸 선행차 충돌: `follow_vehicle_min_distance_m` ↑ 또는 `follow_vehicle_speed_reduction_factor` ↓
- 선행차를 못 따라감: `follow_vehicle_max_distance_m` ↑ 또는 `follow_vehicle_detection_score_threshold` ↓
- 오검출 너무 많음: `follow_vehicle_detection_score_threshold` ↑

---

### **6️⃣ 보행자 감지 (Pedestrian Detection)**

| 파라미터 | 기본값 | ↑ 증가 시 | ↓ 감소 시 | 권장 범위 | 효과 |
|---------|--------|---------|---------|---------|------|
| `pedestrian_intrusion_distance_m` | 5.0 | 더 멀리서 감지 | 더 가까이서 감지 | 3.0~7.0 | **감지 범위 (전방)** |
| `pedestrian_intrusion_lateral_threshold_m` | 1.5 | 옆에서도 감지 | 정면만 감지 | 1.0~2.0 | **감지 범위 (횡방)** |
| `pedestrian_detection_score_threshold` | 0.30 | 낮은 신뢰도도 감지 | 높은 신뢰도만 감지 | 0.2~0.5 | **YOLO 민감도** |

**💡 필드 테스트**:
- 보행자 오인지 많음 → `pedestrian_detection_score_threshold` ↑ (0.40~0.50)
- 보행자 감지 못함 → `pedestrian_intrusion_distance_m` ↑ 또는 `pedestrian_detection_score_threshold` ↓

---

### **7️⃣ 조향 제어 (Steering)**

| 파라미터 | 기본값 | ↑ 증가 시 | ↓ 감소 시 | 권장 범위 | 유의 |
|---------|--------|---------|---------|---------|------|
| `wheelbase` | 0.95 | 조향 민감도 ↑ | 조향 둔감해짐 | 0.90~1.00 | **측정값 필수** |
| `max_steering_angle_deg` | 30.0 | 급곡선 회전 가능 | 회전 반경 커짐 | 25~35 | 하드웨어 제약 |

**💡 조정**:
- 축거(wheelbase) 잘못됨 → 차선 추종이 진동함 (실측 필수!)
- 급곡선 너무 천천히 → `max_steering_angle_deg` ↑ (하드웨어 한계 확인)

---

### **8️⃣ 빨리 확인할 수 있는 튜닝 체크리스트 ✅**

```
[ ] 1. 속도가 너무 빠름 → desired_speed_straight_mps ↓ (1.0으로)
[ ] 2. 속도가 너무 느림 → desired_speed_straight_mps ↑ (1.2로)
[ ] 3. 차선 흔들림 많음 → averaging_window_straight ↑ (10으로)
[ ] 4. 곡선 반응 느림 → lookahead_turn_m ↓ (1.5로)
[ ] 5. 선행차 충돌 위험 → follow_vehicle_min_distance_m ↑ (2.0으로)
[ ] 6. 보행자 오인지 → pedestrian_detection_score_threshold ↑ (0.40으로)
[ ] 7. 타임아웃으로 자꾸 정지 → lane_timeout_s ↑ (0.7로)
[ ] 8. 조향각이 크게 흔들림 → wheelbase 값 확인 (실측 필수)
```

### **제어 주기**
```yaml
control_period_s: 0.1                     # 10 Hz (0.1초)
```

**파라미터 증감 효과**:
- ↑ 증가 (0.2로): 제어 주기 2배 → 반응 느림, CPU 부하 ↓
- ↓ 감소 (0.05로): 제어 주기 절반 → 반응 빠름, CPU 부하 ↑ (권장 X)

---

### **🔴 비활성화 기능 파라미터** (현재 OFF, 미래 사용 예약)

#### **Lane Change (차선 변경) - ❌ OFF**
```yaml
enable_lane_change: False                 # 🔴 활성화 금지 (실차 안정성)
lane_change_min_safe_distance_m: 8.0      # 최소 안전 거리
lane_change_max_lateral_distance_m: 3.5   # 최대 횡방 거리
lane_change_preparation_distance_m: 10.0  # 준비 거리
lane_change_lateral_offset_m: 1.8         # 횡방 오프셋
lane_change_speed_mps: 0.45               # 차선 변경 속도
lane_change_front_safety_distance_m: 10.0 # 전방 안전 거리
lane_change_rear_safety_distance_m: 8.0   # 후방 안전 거리
stop_while_avoidance_not_committed: True  # 회피 미진행 시 정지
```
**효과**:
- `enable_lane_change: True` → PREPARE_LANE_CHANGE_* 상태 활성화 (안정성 저하)
- `lane_change_speed_mps` ↑ → 차선 변경 빨라짐 (위험)
- `lane_change_min_safe_distance_m` ↓ → 안전 거리 감소 (위험)

---

#### **Obstacle Avoidance (장애물 회피) - ❌ OFF**
```yaml
enable_obstacle_avoidance: False          # 🔴 Local Lattice 비활성화
avoidance_method: LATTICE                 # 회피 방법 (격자 구조)
avoidance_trigger_distance_m: 4.0         # 회피 시작 거리 (m)
avoidance_min_clearance_m: 0.8            # 최소 안전 거격 (m)
avoidance_vehicle_half_width_m: 0.45      # 차량 반폭 (m)
avoidance_safety_margin_m: 0.35           # 추가 안전 마진 (m)
avoidance_target_x_m: 3.0                 # 회피 목표 거리 (m)
avoidance_lateral_candidates_m: [-1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8]  # 격자 후보
avoidance_speed_mps: 0.35                 # 회피 중 속도 (m/s)
```
**효과**:
- `enable_obstacle_avoidance: True` → AVOID_OBSTACLE 상태 활성화 (미테스트)
- `avoidance_trigger_distance_m` ↑ → 더 멀리서 회피 (안전)
- `avoidance_min_clearance_m` ↓ → 안전 거리 감소 (위험 ⚠️)
- `avoidance_speed_mps` ↓ → 회피 시 더 느려짐 (안전)

---

#### **Traffic Light & Stopline (신호등/정지선) - ❌ OFF**
```yaml
enable_traffic_light: False               # 🔴 신호등 미지원
enable_stopline: False                    # 🔴 정지선 미지원
red_light_queue_lookahead_m: 15.0         # 신호등 감지 범위
stopline_hold_duration_s: 3.0             # 정지선 대기 시간
red_light_ignore_window_s: 2.0            # 빨강 무시 시간
unknown_light_forced_go_s: 7.0            # 미식별 신호등 진행 시간
unknown_light_stop_duration_s: 3.0        # 미식별 신호등 대기 시간
intersection_ignore_duration_s: 5.0       # 교차로 무시 시간
```
**효과**:
- `enable_traffic_light: True` → 신호등 감지 활성화 (센서 필요)
- 현재는 모두 OFF 상태 (무시됨)

---

#### **Cut-in Detection (급진입 감지) - ❌ OFF**
```yaml
enable_cutin_detection: False             # 🔴 급진입 감지 비활성화
cutin_lateral_velocity_threshold_mps: 0.3 # 측방 속도 역치
cutin_detection_distance_m: 15.0          # 감지 거리
```
**효과**:
- `enable_cutin_detection: True` → EMERGENCY_STOP 가능
- 현재는 OFF 상태 (미감지)

---

## 🚀 실행 방법

### **기본 실행**
```bash
# 전체 패키지 빌드
cd /home/junghun/dream_ws/neuro_ws
colcon build --packages-select neuro_decision

# 런칭
source install/setup.bash
ros2 launch neuro_decision neuro_decision.launch.py
```

### **커스텀 파라미터로 실행**
```bash
ros2 launch neuro_decision neuro_decision.launch.py \
    max_desired_speed_mps:=1.0 \
    desired_speed_straight_mps:=0.9 \
    enable_lane_change:=false
```

### **특정 노드만 실행**
```bash
# behavior_node만
ros2 run neuro_decision behavior_node

# steering_command_node만
ros2 run neuro_decision steering_command_node
```

---

## 🔧 디버깅 & 모니터링

### **디버그 메시지 확인**
```bash
ros2 topic echo /behavior_debug_text
```
**출력 예**:
```
state=LANE_KEEPING | speed_mps=1.1 | target_y=0.25 | reason=lane_keeping
obstacle_distance=2.5 | near_obstacle_stop_active=false | main_lane_blocked=false
```

### **상태 확인**
```bash
ros2 topic echo /planning/current_state
ros2 topic echo /planning/desired_speed_mps
ros2 topic echo /planning/target_point
ros2 topic echo /planning/desired_steering_angle_deg
```

### **센서 입력 확인**
```bash
ros2 topic echo /perception/lane_points --field data | head -20
ros2 topic echo /perception/detections
```

---

## ⚠️ 안전 정책

### **1. Multi-Layer Safety**
- Layer 1: 보행자 급정지 (0.6m)
- Layer 2: 근거리 장애물 (0.8m)
- Layer 3: 추종 장애물 (1.8m)
- Layer 4: 차선 블록 판정 (3.0m)

### **2. First-Message Safety**
차선/장애물 데이터 첫 수신 전까지:
```
received_lane = false
received_obstacle = false
→ 초기 상태 STOP (안전)
```

### **3. Sensor Staleness Detection**
타임아웃 시 즉시 STOP:
- 차선: 0.5s
- 장애물: 0.5s
- 속도: 1.0s

### **4. Lane Blocked 안전 분리**
```python
# ✅ 올바른 판정 (현재)
if obstacle_valid:
    main_lane_blocked = check_blocked()
else:
    main_lane_blocked = false

# ❌ 문제 있는 판정 (이전)
if drivable_valid:  # drivable 끊김 시 오판!
    main_lane_blocked = check_blocked()
```

### **5. STOP에서 즉시 속도 0**
```python
if state in (STOP, EMERGENCY_STOP):
    filtered_desired_speed = 0.0  # 필터 bypass
```

---

## 📊 성능 사양

| 항목 | 사양 |
|------|------|
| **제어 주기** | 10 Hz (0.1s) |
| **목표 속도 범위** | 0.0~1.4 m/s |
| **조향각 범위** | -30~30도 |
| **응답 시간** (STOP) | 2-3 frame (~0.3s) |
| **메모리** | ~20MB (ROS 포함) |
| **CPU 사용률** | ~5-10% (단일 코어) |
| **파라미터 개수** | 113개 (behavior:89, steering:24) |

---

## 📝 파일 구조

```
neuro_decision/
├── neuro_decision/
│   ├── behavior_node.py          # 핵심: 상태 결정 엔진 (1717줄)
│   ├── steering_command_node.py  # 조향 각도 계산 (312줄)
│   ├── ego_vehicle.json          # 차량 설정 파일
│   └── follow_ego.py             # 보조 추종 노드
├── launch/
│   └── neuro_decision.launch.py  # 런칭 설정
├── package.xml                   # ROS 2 메타데이터
├── setup.py                      # Python 설정
└── README.md                      # 본 문서

archive/                           # 이전 버전 백업
├── behavior_node_backup.py
├── speed_control_node.py
└── pure_pursuit_node.py
```

---

## 🛠️ 개발 가이드

### **상태 추가 방법**
1. `BehaviorState` Enum에 추가
2. `decide_next_state()` 메서드에 로직 추가
3. `compute_dynamic_speed()` 메서드에 속도 결정 로직 추가
4. launch 파일에 파라미터 추가 (필요시)

### **파라미터 추가 방법**
1. `__init__` 메서드에서 `declare_parameter()` 호출
2. `self.parameter_name` 멤버변수에 로드
3. launch 파일의 `DeclareLaunchArgument`에 추가

### **새로운 센서 토픽 추가**
1. 구독자(Subscriber) 생성
2. 콜백 함수 작성
3. Timeout 로직 추가
4. 타임아웃 시 STOP 동작 보장

---


**최종 업데이트**: 2026-05-12
