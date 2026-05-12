import math
import heapq
from enum import Enum
from time import monotonic

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64, String
import sensor_msgs_py.point_cloud2 as pc2


class BehaviorState(Enum):
    """확장된 행동 상태 정의"""
    LANE_KEEPING = 'LANE_KEEPING'
    FOLLOW_VEHICLE = 'FOLLOW_VEHICLE'
    PREPARE_LANE_CHANGE_LEFT = 'PREPARE_LANE_CHANGE_LEFT'
    PREPARE_LANE_CHANGE_RIGHT = 'PREPARE_LANE_CHANGE_RIGHT'
    LANE_CHANGE_LEFT = 'LANE_CHANGE_LEFT'
    LANE_CHANGE_RIGHT = 'LANE_CHANGE_RIGHT'
    STOP = 'STOP'
    EMERGENCY_STOP = 'EMERGENCY_STOP'


class ObstacleType(Enum):
    """장애물 타입 분류"""
    UNKNOWN = 'UNKNOWN'
    VEHICLE = 'VEHICLE'
    PEDESTRIAN = 'PEDESTRIAN'
    STATIC_OBSTACLE = 'STATIC_OBSTACLE'


class BehaviorNode(Node):
    def __init__(self):
        super().__init__('behavior_node')

        # ===== 기본 제어 파라미터 =====
        self.declare_parameter('control_period_s', 0.1)

        # ===== 동적 속도 제어 파라미터 (waypoint_behavior_node 스타일) =====
        # 직선, 완만한 곡선, 급한 곡선에 따른 목표 속도
        self.declare_parameter('desired_speed_straight_mps', 2.20)
        self.declare_parameter('desired_speed_gentle_turn_mps', 1.25)
        self.declare_parameter('desired_speed_sharp_turn_mps', 0.95)

        # 곡선 판정을 위한 측방 거리 임계값 (local_y 기준)
        self.declare_parameter('turn_threshold_abs_local_y_small', 0.35)
        self.declare_parameter('turn_threshold_abs_local_y_large', 0.85)

        # ===== 목표점 생성 파라미터 =====
        # 직선/곡선 구간 판정 기준
        self.declare_parameter('lookahead_straight_m', 8.0)
        self.declare_parameter('lookahead_turn_m', 4.2)
        
        # 차선점 샘플링 및 평균화 파라미터
        self.declare_parameter('averaging_window_straight', 8)
        self.declare_parameter('averaging_window_turn', 3)

        # 목표점 생성 관련
        self.declare_parameter('target_y_clamp_m', 1.8)
        self.declare_parameter('center_offset_m', -0.10)
        self.declare_parameter('target_smoothing_alpha', 0.60)

        # ===== Perception 타임아웃 =====
        self.declare_parameter('lane_timeout_s', 0.5)
        self.declare_parameter('obstacle_timeout_s', 0.5)
        self.declare_parameter('traffic_light_timeout_s', 2.0)

        # ===== 안전 거리 임계값 =====
        self.declare_parameter('caution_distance_m', 20.0)
        self.declare_parameter('emergency_stop_distance_m', 3.5)

        # ===== Perception 데이터 필터 =====
        self.declare_parameter('lane_y_limit_m', 8.0)
        self.declare_parameter('obstacle_corridor_half_width_m', 2.5)

        # ===== A* 회피 경로 계획 (필요시) =====
        self.declare_parameter('astar_trigger_distance_m', 15.0)
        self.declare_parameter('astar_resolution_m', 0.5)
        self.declare_parameter('astar_x_max_m', 20.0)
        self.declare_parameter('astar_y_half_width_m', 6.0)
        self.declare_parameter('astar_inflation_radius_m', 1.2)
        self.declare_parameter('astar_target_step_m', 4.0)

        # ===== Vehicle 추종 파라미터 =====
        self.declare_parameter('follow_vehicle_min_distance_m', 5.0)
        self.declare_parameter('follow_vehicle_max_distance_m', 15.0)
        self.declare_parameter('follow_vehicle_speed_reduction_factor', 0.8)
        self.declare_parameter('follow_vehicle_lane_threshold_m', 0.5)

        # ===== Lane change 파라미터 =====
        self.declare_parameter('lane_change_min_safe_distance_m', 8.0)
        self.declare_parameter('lane_change_max_lateral_distance_m', 3.5)
        self.declare_parameter('lane_change_preparation_distance_m', 10.0)
        self.declare_parameter('lane_change_lateral_offset_m', 1.8)
        self.declare_parameter('lane_change_ramp_duration_s', 2.0)
        self.declare_parameter('lane_change_speed_mps', 1.2)

        # ===== 파라미터 로드 =====
        self.control_period_s = float(self.get_parameter('control_period_s').value)

        self.desired_speed_straight_mps = float(self.get_parameter('desired_speed_straight_mps').value)
        self.desired_speed_gentle_turn_mps = float(self.get_parameter('desired_speed_gentle_turn_mps').value)
        self.desired_speed_sharp_turn_mps = float(self.get_parameter('desired_speed_sharp_turn_mps').value)

        self.turn_threshold_small = float(self.get_parameter('turn_threshold_abs_local_y_small').value)
        self.turn_threshold_large = float(self.get_parameter('turn_threshold_abs_local_y_large').value)

        self.lookahead_straight_m = float(self.get_parameter('lookahead_straight_m').value)
        self.lookahead_turn_m = float(self.get_parameter('lookahead_turn_m').value)

        self.averaging_window_straight = max(1, int(self.get_parameter('averaging_window_straight').value))
        self.averaging_window_turn = max(1, int(self.get_parameter('averaging_window_turn').value))

        self.target_y_clamp_m = float(self.get_parameter('target_y_clamp_m').value)
        self.center_offset_m = float(self.get_parameter('center_offset_m').value)
        self.target_smoothing_alpha = float(self.get_parameter('target_smoothing_alpha').value)

        self.lane_timeout_s = float(self.get_parameter('lane_timeout_s').value)
        self.obstacle_timeout_s = float(self.get_parameter('obstacle_timeout_s').value)
        self.traffic_light_timeout_s = float(self.get_parameter('traffic_light_timeout_s').value)

        self.caution_distance_m = float(self.get_parameter('caution_distance_m').value)
        self.emergency_stop_distance_m = float(self.get_parameter('emergency_stop_distance_m').value)

        self.lane_y_limit_m = float(self.get_parameter('lane_y_limit_m').value)
        self.obstacle_corridor_half_width_m = float(
            self.get_parameter('obstacle_corridor_half_width_m').value
        )

        self.astar_trigger_distance_m = float(self.get_parameter('astar_trigger_distance_m').value)
        self.astar_resolution_m = float(self.get_parameter('astar_resolution_m').value)
        self.astar_x_max_m = float(self.get_parameter('astar_x_max_m').value)
        self.astar_y_half_width_m = float(self.get_parameter('astar_y_half_width_m').value)
        self.astar_inflation_radius_m = float(self.get_parameter('astar_inflation_radius_m').value)
        self.astar_target_step_m = float(self.get_parameter('astar_target_step_m').value)

        self.follow_vehicle_min_distance_m = float(self.get_parameter('follow_vehicle_min_distance_m').value)
        self.follow_vehicle_max_distance_m = float(self.get_parameter('follow_vehicle_max_distance_m').value)
        self.follow_vehicle_speed_reduction_factor = float(
            self.get_parameter('follow_vehicle_speed_reduction_factor').value
        )
        self.follow_vehicle_lane_threshold_m = float(
            self.get_parameter('follow_vehicle_lane_threshold_m').value
        )

        self.lane_change_min_safe_distance_m = float(
            self.get_parameter('lane_change_min_safe_distance_m').value
        )
        self.lane_change_max_lateral_distance_m = float(
            self.get_parameter('lane_change_max_lateral_distance_m').value
        )
        self.lane_change_preparation_distance_m = float(
            self.get_parameter('lane_change_preparation_distance_m').value
        )
        self.lane_change_lateral_offset_m = float(self.get_parameter('lane_change_lateral_offset_m').value)
        self.lane_change_ramp_duration_s = float(self.get_parameter('lane_change_ramp_duration_s').value)
        self.lane_change_speed_mps = float(self.get_parameter('lane_change_speed_mps').value)

        # ===== 입력 구독 =====
        self.lane_sub = self.create_subscription(
            PointCloud2,
            '/perception/real_world_lane_points',
            self.lane_callback,
            10
        )
        self.obs_sub = self.create_subscription(
            PointCloud2,
            '/perception/closest_obstacle',
            self.obstacle_callback,
            10
        )
        self.drivable_area_sub = self.create_subscription(
            PointCloud2,
            '/perception/drivable_area',
            self.drivable_area_callback,
            10
        )
        self.tl_sub = self.create_subscription(
            String,
            '/traffic_light_state',
            self.traffic_light_callback,
            10
        )

        # ===== 출력 발행 =====
        self.desired_speed_pub = self.create_publisher(Float64, '/desired_speed', 10)
        self.target_point_pub = self.create_publisher(Point, '/target_point', 10)
        self.state_pub = self.create_publisher(String, '/behavior_state', 10)
        self.debug_pub = self.create_publisher(String, '/behavior_debug_text', 10)

        # ===== 내부 상태 =====
        self.current_state = BehaviorState.STOP
        self.debug_reason = 'initializing'

        # Perception 입력
        self.lane_points = []              # [(x, y), ...]
        self.obstacle_points = []          # [(x, y, z), ...]
        self.drivable_area_points = []     # [(x, y), ...]
        self.traffic_light_state = 'GREEN'

        # Obstacle 요약
        self.obstacle_distance = 99.0
        self.obstacle_x = 99.0
        self.obstacle_y = 0.0
        self.obstacle_type = ObstacleType.UNKNOWN

        # Vehicle 추종 정보
        self.lead_vehicle_distance = 99.0
        self.lead_vehicle_x = 99.0
        self.lead_vehicle_y = 0.0
        self.lead_vehicle_speed_estimated = 0.0

        # Lane change 상태
        self.lane_change_target_direction = None  # 'LEFT' or 'RIGHT'
        self.lane_change_start_time = None
        self.lane_change_lateral_offset_progress = 0.0

        # 목표점 (local 좌표)
        self.target_x = 0.0
        self.target_y = 0.0

        # Smoothing 필터 상태
        self.filtered_target_x = None
        self.filtered_target_y = None
        self.filtered_desired_speed = 1.0

        # 최종 출력
        self.desired_speed = 0.0

        # 상태 추적
        self.avoidance_active = False
        self.path_blocked = False
        self.input_stale = True

        # Lane change 판단 정보
        self.left_lane_blocked = False
        self.right_lane_blocked = False
        self.left_lane_possible = False
        self.right_lane_possible = False

        # 타임스탬프
        now = monotonic()
        self.last_lane_update = now
        self.last_obstacle_update = now
        self.last_tl_update = now

        # 로깅
        self.last_state_log = None
        self.last_reason_log = None
        self.last_timeout_log = ''

        self.timer = self.create_timer(self.control_period_s, self.periodic_update)

        self.get_logger().info('🧠 behavior_node v2: vehicle following + lane change 지원')


    # =========================
    # 콜백
    # =========================
    def lane_callback(self, msg: PointCloud2):
        self.last_lane_update = monotonic()

        raw_points = list(pc2.read_points(msg, skip_nans=True))
        if not raw_points:
            self.lane_points = []
            return

        lane_points = []
        for p in raw_points:
            x = float(p[0])
            y = float(p[1])

            # 후방 점 제거 + 옆으로 너무 튄 점 제거
            if x <= 0.0:
                continue
            if abs(y) > self.lane_y_limit_m:
                continue

            lane_points.append((x, y))

        # x 기준 정렬
        lane_points.sort(key=lambda pt: pt[0])
        self.lane_points = lane_points

    def obstacle_callback(self, msg: PointCloud2):
        self.last_obstacle_update = monotonic()

        raw_points = list(pc2.read_points(msg, skip_nans=True))
        if not raw_points:
            self.obstacle_points = []
            return

        points = []
        for p in raw_points:
            x = float(p[0])
            y = float(p[1])
            z = float(p[2]) if len(p) > 2 else 0.0
            points.append((x, y, z))

        self.obstacle_points = points

    def drivable_area_callback(self, msg: PointCloud2):
        raw_points = list(pc2.read_points(msg, skip_nans=True))
        if not raw_points:
            self.drivable_area_points = []
            return

        self.drivable_area_points = [(float(p[0]), float(p[1])) for p in raw_points]

    def traffic_light_callback(self, msg: String):
        self.last_tl_update = monotonic()

        sign = msg.data.strip().upper()
        if sign not in ['RED', 'YELLOW', 'GREEN']:
            self.get_logger().warning(f'🚦 알 수 없는 신호등 상태: {msg.data}')
            return

        self.traffic_light_state = sign

    # =========================
    # Helper 함수 - 장애물 분류
    # =========================
    def classify_obstacle(self, x, y, z):
        """
        장애물 타입 분류 (현재는 z 높이 기반 휴리스틱)
        나중에 perception에서 class_id를 받으면 업데이트
        """
        # z > 0.5m이면 vehicle 의심 (높이가 있는 물체)
        if z > 0.5:
            return ObstacleType.VEHICLE
        # z < 0.2m이면 static obstacle (낮은 물체)
        elif z < 0.2:
            return ObstacleType.STATIC_OBSTACLE
        else:
            return ObstacleType.UNKNOWN

    def detect_lead_vehicle(self):
        """
        현재 차선 전방의 vehicle 감지
        같은 차선 상에 있고 충분히 먼 vehicle을 lead vehicle로 판정
        """
        lead_dist = 99.0
        lead_x = 99.0
        lead_y = 0.0

        for obs_x, obs_y, obs_z in self.obstacle_points:
            if obs_x <= 0.5:  # 후방 제거
                continue
            if abs(obs_y) > self.obstacle_corridor_half_width_m:  # 차선 밖
                continue

            obs_type = self.classify_obstacle(obs_x, obs_y, obs_z)
            if obs_type != ObstacleType.VEHICLE:
                continue

            dist = math.sqrt(obs_x * obs_x + obs_y * obs_y + obs_z * obs_z)
            if dist < lead_dist:
                lead_dist = dist
                lead_x = obs_x
                lead_y = obs_y

        return lead_dist, lead_x, lead_y

    def is_lane_blocked(self, check_distance_m=15.0):
        """
        현재 차선이 장애물/vehicle로 막혀있는지 판정
        """
        for obs_x, obs_y, obs_z in self.obstacle_points:
            if obs_x <= 0.0:
                continue
            if obs_x > check_distance_m:
                continue
            if abs(obs_y) > self.obstacle_corridor_half_width_m:
                continue

            # 거리 < emergency_stop_distance면 차선 완전 차단
            dist = math.sqrt(obs_x * obs_x + obs_y * obs_y + obs_z * obs_z)
            if dist < (self.emergency_stop_distance_m + 2.0):
                return True

        return False

    def evaluate_lane_change_options(self):
        """
        좌/우 차선 변경 가능 여부 판정
        drivable_area와 장애물 정보를 활용
        """
        left_possible = True
        right_possible = True
        left_blocked = False
        right_blocked = False

        if not self.drivable_area_points:
            # drivable area 없으면 보수적으로 판정 불가
            left_possible = False
            right_possible = False
            return left_possible, right_possible, left_blocked, right_blocked

        # drivable area의 y 범위 파악
        y_values = [y for _, y in self.drivable_area_points]
        if not y_values:
            left_possible = False
            right_possible = False
            return left_possible, right_possible, left_blocked, right_blocked

        drivable_min_y = min(y_values)
        drivable_max_y = max(y_values)

        # 현재 차선이 대략 0 근처이므로
        # 좌측: drivable_min_y 쪽, 우측: drivable_max_y 쪽으로 판정

        # 좌측 차선 변경 가능성 확인
        left_target_y = drivable_min_y
        if abs(left_target_y) < self.lane_change_max_lateral_distance_m:
            # 좌측 drivable area 있음 + 충분히 가까움
            # 해당 방향에 장애물 확인
            left_has_obstacle = self._has_obstacle_in_region(
                target_y=left_target_y,
                lookahead_dist=self.lane_change_preparation_distance_m,
                safe_distance=self.lane_change_min_safe_distance_m
            )
            left_possible = not left_has_obstacle
            left_blocked = left_has_obstacle
        else:
            left_possible = False

        # 우측 차선 변경 가능성 확인
        right_target_y = drivable_max_y
        if abs(right_target_y) < self.lane_change_max_lateral_distance_m:
            # 우측 drivable area 있음 + 충분히 가까움
            right_has_obstacle = self._has_obstacle_in_region(
                target_y=right_target_y,
                lookahead_dist=self.lane_change_preparation_distance_m,
                safe_distance=self.lane_change_min_safe_distance_m
            )
            right_possible = not right_has_obstacle
            right_blocked = right_has_obstacle
        else:
            right_possible = False

        return left_possible, right_possible, left_blocked, right_blocked

    def _has_obstacle_in_region(self, target_y, lookahead_dist, safe_distance):
        """
        특정 y 위치 근처에 장애물이 있는지 확인
        """
        for obs_x, obs_y, obs_z in self.obstacle_points:
            if obs_x <= 0.0 or obs_x > lookahead_dist:
                continue
            if abs(obs_y - target_y) > 1.5:  # target_y 주변 ±1.5m
                continue

            dist = math.sqrt(obs_x * obs_x + obs_y * obs_y + obs_z * obs_z)
            if dist < safe_distance:
                return True

        return False

    def compute_lane_change_target(self, direction):
        """
        Lane change 목표 y 좌표 계산
        direction: 'LEFT' 또는 'RIGHT'
        """
        if not self.drivable_area_points:
            return self.target_y

        y_values = [y for _, y in self.drivable_area_points]
        if direction == 'LEFT':
            target_y = min(y_values)
        elif direction == 'RIGHT':
            target_y = max(y_values)
        else:
            return self.target_y

        # 너무 극단적이지 않도록 제한
        target_y = max(-self.lane_change_lateral_offset_m, min(self.lane_change_lateral_offset_m, target_y))

        return float(target_y)

    # =========================
    # 속도 계산
    # =========================
    def compute_follow_speed(self, lead_vehicle_distance):
        """
        선행 차량 거리 기반 속도 결정
        거리가 가까울수록 더 느린 속도
        """
        if lead_vehicle_distance >= self.follow_vehicle_max_distance_m:
            # 선행 차량이 충분히 멀면 정상 속도
            return self.compute_nominal_speed(self.target_y)

        if lead_vehicle_distance <= self.follow_vehicle_min_distance_m:
            # 선행 차량이 너무 가까우면 정지 준비
            return 0.5

        # 선형 보간: max_distance에서 nominal speed, min_distance에서 0.5 m/s
        t = (lead_vehicle_distance - self.follow_vehicle_min_distance_m) / (
            self.follow_vehicle_max_distance_m - self.follow_vehicle_min_distance_m
        )
        t = max(0.0, min(1.0, t))

        nominal = self.compute_nominal_speed(self.target_y)
        reduced = nominal * self.follow_vehicle_speed_reduction_factor
        return reduced + t * (nominal - reduced)

    def compute_nominal_speed(self, local_y):
        """
        Lateral offset (local_y)에 따라 직선/완만한곡선/급한곡선 속도 결정
        waypoint_behavior_node의 로직 참고
        """
        abs_y = abs(float(local_y))
        if abs_y < self.turn_threshold_small:
            return self.desired_speed_straight_mps
        elif abs_y < self.turn_threshold_large:
            return self.desired_speed_gentle_turn_mps
        else:
            return self.desired_speed_sharp_turn_mps

    # =========================
    # 상태 결정
    # =========================
    def decide_next_state(self, now):
        """
        현재 센서 정보와 상태를 기반으로 다음 상태 결정
        """
        # 기본 안전 체크
        if self.input_stale:
            return BehaviorState.STOP, 'input_stale'

        if not self.lane_points:
            return BehaviorState.STOP, 'no_lane_points'

        if self.traffic_light_state in ('RED', 'YELLOW'):
            return BehaviorState.STOP, f'traffic_light_{self.traffic_light_state.lower()}'

        # 긴급 정지
        if self.obstacle_distance < self.emergency_stop_distance_m:
            return BehaviorState.EMERGENCY_STOP, 'obstacle_too_close'

        # Vehicle 추종 판정
        lead_dist, lead_x, lead_y = self.detect_lead_vehicle()
        if self.follow_vehicle_min_distance_m <= lead_dist <= self.follow_vehicle_max_distance_m:
            if abs(lead_y) <= self.follow_vehicle_lane_threshold_m:
                # 같은 차선에서 추종 거리 범위 내
                return BehaviorState.FOLLOW_VEHICLE, 'lead_vehicle_detected'

        # Lane change 판정
        main_lane_blocked = self.is_lane_blocked(self.lane_change_preparation_distance_m)
        left_possible, right_possible, left_blocked, right_blocked = self.evaluate_lane_change_options()

        if main_lane_blocked:
            # 현재 차선이 막힘 -> lane change 고려
            if left_possible:
                if self.current_state in (BehaviorState.LANE_KEEPING, BehaviorState.STOP):
                    return BehaviorState.PREPARE_LANE_CHANGE_LEFT, 'main_blocked_left_available'
                elif self.current_state == BehaviorState.PREPARE_LANE_CHANGE_LEFT:
                    return BehaviorState.LANE_CHANGE_LEFT, 'prepare_lane_change_left_done'
                elif self.current_state == BehaviorState.LANE_CHANGE_LEFT:
                    # Lane change 진행 중
                    if self._is_lane_change_complete():
                        return BehaviorState.LANE_KEEPING, 'lane_change_left_complete'
                    return BehaviorState.LANE_CHANGE_LEFT, 'lane_changing_left'

            if right_possible:
                if self.current_state in (BehaviorState.LANE_KEEPING, BehaviorState.STOP):
                    return BehaviorState.PREPARE_LANE_CHANGE_RIGHT, 'main_blocked_right_available'
                elif self.current_state == BehaviorState.PREPARE_LANE_CHANGE_RIGHT:
                    return BehaviorState.LANE_CHANGE_RIGHT, 'prepare_lane_change_right_done'
                elif self.current_state == BehaviorState.LANE_CHANGE_RIGHT:
                    if self._is_lane_change_complete():
                        return BehaviorState.LANE_KEEPING, 'lane_change_right_complete'
                    return BehaviorState.LANE_CHANGE_RIGHT, 'lane_changing_right'

            # Lane change 불가능하면 정지
            return BehaviorState.STOP, 'no_lane_change_option'

        # 정상 차선 유지
        return BehaviorState.LANE_KEEPING, 'cruise'

    def _is_lane_change_complete(self):
        """Lane change 완료 판정"""
        if self.lane_change_start_time is None:
            return False

        elapsed = monotonic() - self.lane_change_start_time
        return elapsed >= self.lane_change_ramp_duration_s

    # =========================
    # Target point 생성
    # =========================
    def choose_target_from_route(self):
        """
        Perception lane_points에서 목표점 선택
        상태에 따라 다르게 처리
        """
        if not self.lane_points:
            return None

        # 차량 전방 포인트만 수집
        forward_points = []
        for x, y in self.lane_points:
            if x > 0.5:
                forward_points.append((x, y))

        if not forward_points:
            return None

        # 근처 곡률로 코너 판단
        nearby_abs_y = [abs(p[1]) for p in forward_points[:3]]
        is_in_corner = len(nearby_abs_y) > 0 and max(nearby_abs_y) > self.turn_threshold_small

        # 코너/직선에 따라 lookahead 거리 결정
        target_lookahead = self.lookahead_turn_m if is_in_corner else self.lookahead_straight_m

        # lookahead 거리 이상인 첫 포인트 찾기
        candidate_idx = None
        for i, (x, y) in enumerate(forward_points):
            if x >= target_lookahead:
                candidate_idx = i
                break

        if candidate_idx is None:
            candidate_idx = max(0, len(forward_points) - 1)

        # 평균화 윈도우 크기 결정
        avg_window = self.averaging_window_turn if is_in_corner else self.averaging_window_straight

        # 선택된 포인트 주변을 평균화
        start_idx = max(0, candidate_idx - 1)
        end_idx = min(len(forward_points), candidate_idx + avg_window)
        selected = forward_points[start_idx:end_idx]

        if len(selected) == 1:
            avg_x = selected[0][0]
            avg_y = selected[0][1]
        else:
            # 거리에 따른 가중 평균
            weighted_sum_x = 0.0
            weighted_sum_y = 0.0
            weight_sum = 0.0
            for i, (x, y) in enumerate(selected):
                weight = 1.0 + (i * 0.25)
                weighted_sum_x += weight * x
                weighted_sum_y += weight * y
                weight_sum += weight

            avg_x = weighted_sum_x / weight_sum
            avg_y = weighted_sum_y / weight_sum

        # 직선: center offset 최소화, 곡선: center_offset 적용
        if not is_in_corner:
            corrected_y = avg_y * 0.95  # 95% 중앙 추종
        else:
            corrected_y = avg_y - self.center_offset_m

        # Y 제한
        corrected_y = max(-self.target_y_clamp_m, min(self.target_y_clamp_m, corrected_y))

        return is_in_corner, float(avg_x), float(corrected_y)

    def compute_target_point_for_state(self, state, now):
        """
        상태별 target point 계산
        """
        if state in (BehaviorState.STOP, BehaviorState.EMERGENCY_STOP):
            return 0.0, 0.0, False

        if state in (BehaviorState.LANE_KEEPING, BehaviorState.FOLLOW_VEHICLE):
            # 현재 차선 유지
            target = self.choose_target_from_route()
            if target is None:
                return 0.0, 0.0, False
            is_corner, target_x, target_y = target
            return target_x, target_y, is_corner

        if state == BehaviorState.PREPARE_LANE_CHANGE_LEFT:
            # Preparation: 현재 차선 유지하되 약간 왼쪽 준비
            target = self.choose_target_from_route()
            if target is None:
                return 0.0, 0.0, False
            _, target_x, target_y = target
            # 약간 왼쪽으로 offset
            target_y = target_y - 0.3
            return target_x, target_y, False

        if state == BehaviorState.PREPARE_LANE_CHANGE_RIGHT:
            # Preparation: 현재 차선 유지하되 약간 오른쪽 준비
            target = self.choose_target_from_route()
            if target is None:
                return 0.0, 0.0, False
            _, target_x, target_y = target
            # 약간 오른쪽으로 offset
            target_y = target_y + 0.3
            return target_x, target_y, False

        if state == BehaviorState.LANE_CHANGE_LEFT:
            # Lane change: 목표 차선(왼쪽)으로 이동
            if self.lane_change_start_time is None:
                self.lane_change_start_time = now
                self.lane_change_target_direction = 'LEFT'

            target_y_final = self.compute_lane_change_target('LEFT')
            target = self.choose_target_from_route()
            if target is None:
                return 0.0, target_y_final, False

            _, target_x, target_y_base = target

            # Ramp interpolation: 현재 target_y에서 final로 서서히 이동
            elapsed = now - self.lane_change_start_time
            progress = min(1.0, elapsed / self.lane_change_ramp_duration_s)
            target_y = target_y_base + progress * (target_y_final - target_y_base)

            return target_x, target_y, False

        if state == BehaviorState.LANE_CHANGE_RIGHT:
            # Lane change: 목표 차선(오른쪽)으로 이동
            if self.lane_change_start_time is None:
                self.lane_change_start_time = now
                self.lane_change_target_direction = 'RIGHT'

            target_y_final = self.compute_lane_change_target('RIGHT')
            target = self.choose_target_from_route()
            if target is None:
                return 0.0, target_y_final, False

            _, target_x, target_y_base = target

            # Ramp interpolation
            elapsed = now - self.lane_change_start_time
            progress = min(1.0, elapsed / self.lane_change_ramp_duration_s)
            target_y = target_y_base + progress * (target_y_final - target_y_base)

            return target_x, target_y, False

        # Default
        return 0.0, 0.0, False

    # =========================
    # 주기 업데이트
    # =========================
    def periodic_update(self):
        now = monotonic()

        lane_stale = (now - self.last_lane_update) > self.lane_timeout_s
        obstacle_stale = (now - self.last_obstacle_update) > self.obstacle_timeout_s
        tl_stale = (now - self.last_tl_update) > self.traffic_light_timeout_s

        self.input_stale = lane_stale or obstacle_stale

        timeout_text = f'lane_stale={lane_stale}, obs_stale={obstacle_stale}, tl_stale={tl_stale}'
        if timeout_text != self.last_timeout_log:
            self.last_timeout_log = timeout_text
            self.get_logger().debug(f'⌛ timeout: {timeout_text}')

        # 장애물 요약 (일반)
        self.obstacle_distance, self.obstacle_x, self.obstacle_y = self.compute_closest_relevant_obstacle()

        # Vehicle 추종 정보
        self.lead_vehicle_distance, self.lead_vehicle_x, self.lead_vehicle_y = self.detect_lead_vehicle()

        # Lane change 가능성 판단
        left_poss, right_poss, left_blocked, right_blocked = self.evaluate_lane_change_options()
        self.left_lane_possible = left_poss
        self.right_lane_possible = right_poss
        self.left_lane_blocked = left_blocked
        self.right_lane_blocked = right_blocked

        # ===== 상태 결정 (안전 우선) =====
        next_state, reason = self.decide_next_state(now)

        # Lane change 시작/완료 추적
        if next_state not in (BehaviorState.LANE_CHANGE_LEFT, BehaviorState.LANE_CHANGE_RIGHT):
            self.lane_change_start_time = None
            self.lane_change_target_direction = None

        self.current_state = next_state
        self.debug_reason = reason

        # ===== 목표점 생성 =====
        if next_state in (BehaviorState.STOP, BehaviorState.EMERGENCY_STOP):
            target_x = 0.0
            target_y = 0.0
            is_in_corner = False
        else:
            target_x, target_y, is_in_corner = self.compute_target_point_for_state(next_state, now)
            if target_x == 0.0 and target_y == 0.0 and next_state not in (BehaviorState.STOP, BehaviorState.EMERGENCY_STOP):
                # 유효한 target 없으면 정지
                next_state = BehaviorState.STOP
                reason = 'no_valid_target'
                target_x = 0.0
                target_y = 0.0

        # ===== Smoothing 적용 =====
        alpha = max(0.0, min(1.0, self.target_smoothing_alpha))
        if self.filtered_target_x is None or self.filtered_target_y is None:
            self.filtered_target_x = target_x
            self.filtered_target_y = target_y
        else:
            self.filtered_target_x = ((1.0 - alpha) * self.filtered_target_x) + (alpha * target_x)
            self.filtered_target_y = ((1.0 - alpha) * self.filtered_target_y) + (alpha * target_y)

        local_x = float(self.filtered_target_x)
        local_y = float(self.filtered_target_y)

        # ===== 동적 속도 결정 =====
        if next_state == BehaviorState.STOP:
            raw_desired_speed = 0.0
        elif next_state == BehaviorState.EMERGENCY_STOP:
            raw_desired_speed = 0.0
        elif next_state == BehaviorState.FOLLOW_VEHICLE:
            raw_desired_speed = self.compute_follow_speed(self.lead_vehicle_distance)
        elif next_state in (BehaviorState.PREPARE_LANE_CHANGE_LEFT, BehaviorState.PREPARE_LANE_CHANGE_RIGHT):
            # 차선 변경 준비: 약간 감속
            raw_desired_speed = self.compute_nominal_speed(local_y) * 0.9
        elif next_state in (BehaviorState.LANE_CHANGE_LEFT, BehaviorState.LANE_CHANGE_RIGHT):
            # 차선 변경 중: 더 안전한 속도
            raw_desired_speed = self.lane_change_speed_mps
        else:  # LANE_KEEPING
            raw_desired_speed = self.compute_nominal_speed(local_y)

        # 속도도 smoothing 적용
        speed_alpha = 0.10
        self.filtered_desired_speed = ((1.0 - speed_alpha) * self.filtered_desired_speed) + (
            speed_alpha * raw_desired_speed
        )

        # ===== 내부 상태 업데이트 =====
        self.target_x = local_x
        self.target_y = local_y
        self.desired_speed = float(self.filtered_desired_speed)

        # 상태 변화 로그
        if self.current_state.value != self.last_state_log or self.debug_reason != self.last_reason_log:
            self.last_state_log = self.current_state.value
            self.last_reason_log = self.debug_reason
            self.get_logger().info(
                f'state={self.current_state.value}, reason={self.debug_reason}'
            )

        self.publish_all()

    # =========================
    # 판단 보조 함수
    # =========================
    def compute_closest_relevant_obstacle(self):
        """
        판단용 장애물은 차량 전방 corridor 내부만 사용
        """
        if not self.obstacle_points:
            return 99.0, 99.0, 0.0

        relevant = []
        for x, y, z in self.obstacle_points:
            if x <= 0.0:
                continue
            if abs(y) > self.obstacle_corridor_half_width_m:
                continue
            relevant.append((x, y, z))

        if not relevant:
            return 99.0, 99.0, 0.0

        best_dist = float('inf')
        best_x = 99.0
        best_y = 0.0

        for x, y, z in relevant:
            dist = math.sqrt(x * x + y * y + z * z)
            if dist < best_dist:
                best_dist = dist
                best_x = x
                best_y = y

        return float(best_dist), float(best_x), float(best_y)

    # =========================
    # A* 로컬 회피
    # =========================
    def world_to_grid(self, x, y, resolution, y_half_width, x_cells, y_cells):
        gx = int(round(float(x) / resolution))
        gy = int(round((float(y) + y_half_width) / resolution))
        if 0 <= gx < x_cells and 0 <= gy < y_cells:
            return gx, gy
        return None

    def grid_to_world(self, gx, gy, resolution, y_half_width):
        wx = float(gx) * resolution
        wy = float(gy) * resolution - y_half_width
        return wx, wy

    def plan_astar_path(self, goal_x, goal_y):
        resolution = self.astar_resolution_m
        y_half = self.astar_y_half_width_m
        x_cells = int(round(self.astar_x_max_m / resolution)) + 1
        y_cells = int(round((2.0 * y_half) / resolution)) + 1

        occupancy = [[False for _ in range(y_cells)] for _ in range(x_cells)]

        # 주행가능영역이 있으면 y 범위 기반으로 제한
        if self.drivable_area_points:
            min_y = min(y for _, y in self.drivable_area_points)
            max_y = max(y for _, y in self.drivable_area_points)

            for gx in range(x_cells):
                for gy in range(y_cells):
                    wx, wy = self.grid_to_world(gx, gy, resolution, y_half)
                    if wy < (min_y - 0.5) or wy > (max_y + 0.5):
                        occupancy[gx][gy] = True

        # 장애물 팽창
        inflate_cells = int(math.ceil(self.astar_inflation_radius_m / resolution))
        for obs_x, obs_y, _ in self.obstacle_points:
            if obs_x < 0.0:
                continue

            obstacle_cell = self.world_to_grid(
                obs_x, obs_y, resolution, y_half, x_cells, y_cells
            )
            if obstacle_cell is None:
                continue

            obstacle_gx, obstacle_gy = obstacle_cell

            for dx in range(-inflate_cells, inflate_cells + 1):
                for dy in range(-inflate_cells, inflate_cells + 1):
                    nx = obstacle_gx + dx
                    ny = obstacle_gy + dy
                    if 0 <= nx < x_cells and 0 <= ny < y_cells:
                        cell_dist = math.sqrt((dx * resolution) ** 2 + (dy * resolution) ** 2)
                        if cell_dist <= self.astar_inflation_radius_m:
                            occupancy[nx][ny] = True

        start = self.world_to_grid(0.0, 0.0, resolution, y_half, x_cells, y_cells)
        goal = self.world_to_grid(goal_x, goal_y, resolution, y_half, x_cells, y_cells)
        if start is None or goal is None:
            return None

        sx, sy = start
        gx, gy = goal

        occupancy[sx][sy] = False
        occupancy[gx][gy] = False

        neighbors = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (-1, -1, math.sqrt(2.0)),
        ]

        def heuristic(ax, ay, bx, by):
            return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

        open_heap = []
        heapq.heappush(open_heap, (heuristic(sx, sy, gx, gy), 0.0, (sx, sy)))
        came_from = {}
        g_score = {(sx, sy): 0.0}
        visited = set()

        while open_heap:
            _, current_g, (cx, cy) = heapq.heappop(open_heap)

            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))

            if (cx, cy) == (gx, gy):
                path_cells = [(cx, cy)]
                while (cx, cy) in came_from:
                    cx, cy = came_from[(cx, cy)]
                    path_cells.append((cx, cy))
                path_cells.reverse()
                return [self.grid_to_world(px, py, resolution, y_half) for px, py in path_cells]

            for dx, dy, move_cost in neighbors:
                nx = cx + dx
                ny = cy + dy

                if not (0 <= nx < x_cells and 0 <= ny < y_cells):
                    continue
                if occupancy[nx][ny]:
                    continue

                tentative_g = current_g + move_cost
                if tentative_g < g_score.get((nx, ny), float('inf')):
                    g_score[(nx, ny)] = tentative_g
                    came_from[(nx, ny)] = (cx, cy)
                    f_score = tentative_g + heuristic(nx, ny, gx, gy)
                    heapq.heappush(open_heap, (f_score, tentative_g, (nx, ny)))

        return None

    def pick_path_target(self, path_world):
        for px, py in path_world:
            dist = math.sqrt(px * px + py * py)
            if dist >= self.astar_target_step_m:
                return float(px), float(py)

        if path_world:
            px, py = path_world[-1]
            return float(px), float(py)

        return 0.0, 0.0

    # =========================
    # 발행
    # =========================
    def publish_all(self):
        # 목표 속도 발행
        speed_msg = Float64()
        speed_msg.data = float(self.desired_speed)
        self.desired_speed_pub.publish(speed_msg)

        # 목표점 발행
        target_msg = Point()
        target_msg.x = float(self.target_x)
        target_msg.y = float(self.target_y)
        target_msg.z = 0.0
        self.target_point_pub.publish(target_msg)

        # 상태 발행
        state_msg = String()
        state_msg.data = self.current_state.value
        self.state_pub.publish(state_msg)

        # 디버그 메시지 발행
        debug_msg = String()
        debug_msg.data = (
            f'state={self.current_state.value} | '
            f'reason={self.debug_reason} | '
            f'obs={self.obstacle_distance:.2f}m | '
            f'lead_vehicle={self.lead_vehicle_distance:.2f}m | '
            f'light={self.traffic_light_state} | '
            f'input_stale={self.input_stale} | '
            f'lane_blocked={self.is_lane_blocked()} | '
            f'left_lc_possible={self.left_lane_possible} | '
            f'right_lc_possible={self.right_lane_possible} | '
            f'target=({self.target_x:.2f},{self.target_y:.2f}) | '
            f'desired_speed={self.desired_speed:.2f}mps'
        )
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
