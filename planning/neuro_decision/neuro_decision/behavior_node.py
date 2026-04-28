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
    LANE_KEEPING = 'LANE_KEEPING'
    STOP = 'STOP'
    EMERGENCY_STOP = 'EMERGENCY_STOP'


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

        self.get_logger().info('🧠 behavior_node 시작: perception 기반 + waypoint 스타일 tuning')


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

        # 장애물 요약
        self.obstacle_distance, self.obstacle_x, self.obstacle_y = self.compute_closest_relevant_obstacle()

        # ===== 상태 결정 (안전 우선) =====
        next_state = BehaviorState.LANE_KEEPING
        reason = 'cruise'

        if self.input_stale:
            next_state = BehaviorState.STOP
            reason = 'input_stale'
        elif not self.lane_points:
            next_state = BehaviorState.STOP
            reason = 'no_lane_points'
        elif self.traffic_light_state in ('RED', 'YELLOW'):
            next_state = BehaviorState.STOP
            reason = f'traffic_light_{self.traffic_light_state.lower()}'
        elif self.obstacle_distance < self.emergency_stop_distance_m:
            next_state = BehaviorState.EMERGENCY_STOP
            reason = 'obstacle_too_close'
        elif self.obstacle_distance < self.caution_distance_m:
            # 정지 강제는 아니지만 감속
            next_state = BehaviorState.LANE_KEEPING
            reason = 'obstacle_in_caution_zone'
        else:
            next_state = BehaviorState.LANE_KEEPING
            reason = 'cruise'

        self.current_state = next_state
        self.debug_reason = reason

        # ===== 목표점 생성 (perception 기반) =====
        if next_state in (BehaviorState.STOP, BehaviorState.EMERGENCY_STOP):
            # 정지 상태: 목표점 0,0
            target_x = 0.0
            target_y = 0.0
            is_in_corner = False
        else:
            # 정상 주행: lane_points에서 목표점 선택
            target = self.choose_target_from_route()
            if target is None:
                next_state = BehaviorState.STOP
                reason = 'no_forward_target'
                target_x = 0.0
                target_y = 0.0
                is_in_corner = False
            else:
                is_in_corner, target_x, target_y = target

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
        if next_state in (BehaviorState.STOP, BehaviorState.EMERGENCY_STOP):
            raw_desired_speed = 0.0
        else:
            # 필터링된 local_y 기반으로 속도 결정
            raw_desired_speed = self.calculate_dynamic_speed(local_y)

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

    def calculate_dynamic_speed(self, local_y):
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

    def choose_target_from_route(self):
        """
        Perception lane_points에서 목표점 선택
        waypoint_behavior_node의 choose_target_from_route 로직 적용
        
        Returns:
            (is_in_corner, avg_x, corrected_y) or None
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
            f'light={self.traffic_light_state} | '
            f'input_stale={self.input_stale} | '
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