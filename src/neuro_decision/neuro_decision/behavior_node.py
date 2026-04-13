import rclpy
import math
import heapq
from enum import Enum
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64, String
import sensor_msgs_py.point_cloud2 as pc2

# 주행 판단 상태: 차선 유지 또는 정지
class BehaviorState(Enum):
    LANE_KEEPING = 'LANE_KEEPING'
    STOP = 'STOP'


# 판단 노드(상위 의사결정)
# 입력: 차선/장애물/신호등
# 출력: 목표 속도(desired_speed), 목표점(target_point), 상태(behavior_state)
class BehaviorNode(Node):
    def __init__(self):
        super().__init__('behavior_node')

        # ===== [입력 구독] 인지 파트에서 전달되는 데이터 =====
        self.lane_sub = self.create_subscription(
            PointCloud2, '/perception/real_world_lane_points', self.lane_callback, 10)
        self.obs_sub = self.create_subscription(
            PointCloud2, '/perception/closest_obstacle', self.obstacle_callback, 10)
        self.drivable_area_sub = self.create_subscription(
            PointCloud2, '/perception/drivable_area', self.drivable_area_callback, 10)
        self.tl_sub = self.create_subscription(
            String, '/traffic_light_state', self.traffic_light_callback, 10)

        # ===== [출력 발행] 제어 파트로 넘길 명령 =====
        self.desired_speed_pub = self.create_publisher(Float64, '/desired_speed', 10)
        self.target_point_pub = self.create_publisher(Point, '/target_point', 10)
        self.obstacle_pub = self.create_publisher(Float64, '/obstacle_distance', 10)
        self.state_pub = self.create_publisher(String, '/behavior_state', 10)

        # ===== [내부 상태] 최신 센서/판단 결과 저장 =====
        self.current_state = BehaviorState.LANE_KEEPING
        self.obstacle_distance = 99.0
        self.obstacle_x = 99.0  # 장애물 x 좌표
        self.obstacle_y = 0.0   # 장애물 y 좌표
        self.obstacle_points = []  # 모든 장애물 포인트 (x, y) 리스트
        self.drivable_area_points = []  # 주행 가능 영역 경계 포인트 리스트
        self.traffic_light_state = 'GREEN'
        self.lookahead_distance = 5.0  # 5m 선회 목표
        self.avoidance_active = False
        self.planner_blocked = False

        # 속도 정책 (m/s)
        self.cruise_speed_mps = 8.0
        self.caution_speed_mps = 3.0
        self.stop_distance_m = 10.0
        self.caution_distance_m = 20.0
        self.emergency_stop_distance_m = 3.5

        # A* 로컬 회피 파라미터
        self.astar_trigger_distance_m = 15.0
        self.astar_resolution_m = 0.5
        self.astar_x_max_m = 20.0
        self.astar_y_half_width_m = 6.0
        self.astar_inflation_radius_m = 1.2
        self.astar_target_step_m = 4.0

        self.get_logger().info('🧠 행동 판단 노드 가동 (초기 상태: LANE_KEEPING)')

    def find_lookahead_point(self, lane_points_list):
        # 차선 포인트 중 lookahead 거리 이상인 첫 지점을 목표점으로 선택
        for x, y in lane_points_list:
            distance = math.sqrt(float(x)**2 + float(y)**2)
            if distance >= self.lookahead_distance:
                return float(x), float(y)

        if lane_points_list:
            x, y = lane_points_list[-1]
            return float(x), float(y)

        return 0.0, 0.0

    def update_state_from_obstacle(self, distance):
        # 신호등/장애물 조건으로 행동 상태(STOP/LANE_KEEPING) 갱신
        self.obstacle_distance = distance

        next_state = BehaviorState.LANE_KEEPING
        if self.traffic_light_state.upper() in ('RED', 'YELLOW'):
            next_state = BehaviorState.STOP
        elif self.obstacle_distance < self.emergency_stop_distance_m:
            next_state = BehaviorState.STOP
        elif self.planner_blocked:
            next_state = BehaviorState.STOP

        if self.current_state != next_state:
            self.current_state = next_state
            self.publish_state()
            self.get_logger().info(f'상태 변경 -> {self.current_state.value}')

    def get_desired_speed(self):
        # 현재 상황에서 목표 속도(m/s) 정책 결정
        if self.traffic_light_state.upper() in ('RED', 'YELLOW'):
            return 0.0

        if self.obstacle_distance < self.emergency_stop_distance_m:
            return 0.0

        if self.planner_blocked:
            return 0.0

        if self.avoidance_active:
            return min(self.caution_speed_mps, 2.5)

        if self.obstacle_distance < self.caution_distance_m:
            return self.caution_speed_mps

        return self.cruise_speed_mps

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

        # ===== [주행 불가 영역 반영] 주행가능 영역 경계 밖을 점유로 표시 =====
        if self.drivable_area_points:
            # 주행 가능 영역의 경계를 포함하는 bounding box 계산
            min_y = min(y for _, y in self.drivable_area_points)
            max_y = max(y for _, y in self.drivable_area_points)
            # 경계 밖의 셀을 점유 표시
            for gx in range(x_cells):
                for gy in range(y_cells):
                    wx, wy = self.grid_to_world(gx, gy, resolution, y_half)
                    # 주행 영역 경계 체크 (간단한 Y축 범위 기반)
                    if wy < min_y - 0.5 or wy > max_y + 0.5:
                        occupancy[gx][gy] = True

        # ===== [모든 장애물 포인트 반영] 팽창(inflation)을 적용한 점유 격자 =====
        if self.obstacle_points:
            inflate_cells = int(math.ceil(self.astar_inflation_radius_m / resolution))
            for obs_x, obs_y in self.obstacle_points:
                if obs_x < 0.0:  # 후방 장애물은 무시
                    continue
                obstacle_rc = self.world_to_grid(
                    obs_x,
                    obs_y,
                    resolution,
                    y_half,
                    x_cells,
                    y_cells,
                )
                if obstacle_rc is not None:
                    obstacle_gx, obstacle_gy = obstacle_rc
                    # 원형 팽창
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

        # 8방향 A* 탐색
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
            dist = math.sqrt(px ** 2 + py ** 2)
            if dist >= self.astar_target_step_m:
                return float(px), float(py)

        if path_world:
            px, py = path_world[-1]
            return float(px), float(py)

        return 0.0, 0.0

    def publish_decision_commands(self):
        # 판단 결과(목표속도)를 제어 파트 토픽으로 발행
        desired_speed = self.get_desired_speed()

        speed_msg = Float64()
        speed_msg.data = desired_speed
        self.desired_speed_pub.publish(speed_msg)

        self.get_logger().debug(
            f'명령 발행: speed={desired_speed:.2f} m/s, '
            f'obstacle={self.obstacle_distance:.2f} m, light={self.traffic_light_state}'
        )

    def publish_target_point(self, target_x, target_y):
        msg = Point()
        msg.x = float(target_x)
        msg.y = float(target_y)
        msg.z = 0.0
        self.target_point_pub.publish(msg)

    def publish_obstacle_distance(self):
        # 디버깅/모니터링용 장애물 거리 발행
        msg = Float64()
        msg.data = self.obstacle_distance
        self.obstacle_pub.publish(msg)

    def publish_state(self):
        # 현재 행동 상태 발행
        msg = String()
        msg.data = self.current_state.value
        self.state_pub.publish(msg)

    def obstacle_callback(self, msg):
        # ===== [장애물 입력 처리] 모든 장애물 포인트 저장 + 최근접 점 계산 =====
        points = list(pc2.read_points(msg, skip_nans=True))

        if not points:
            self.obstacle_distance = 99.0
            self.obstacle_x = 99.0
            self.obstacle_y = 0.0
            self.obstacle_points = []
            self.update_state_from_obstacle(99.0)
            self.publish_obstacle_distance()
            self.publish_decision_commands()
            return

        # 모든 장애물 포인트 저장 (A* 경로 계획에서 사용)
        self.obstacle_points = [(p[0], p[1]) for p in points]

        # 가장 가까운 장애물 찾기 (거리 & 위치 저장)
        min_distance = 99.0
        closest_x = 99.0
        closest_y = 0.0
        
        for p in points:
            dist = math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
            if dist < min_distance:
                min_distance = dist
                closest_x = p[0]
                closest_y = p[1]
        
        self.obstacle_distance = min_distance
        self.obstacle_x = closest_x
        self.obstacle_y = closest_y
        
        self.update_state_from_obstacle(min_distance)
        self.publish_obstacle_distance()
        self.publish_decision_commands()

    def drivable_area_callback(self, msg):
        # ===== [주행가능 영역 입력 처리] 경계 포인트 저장 =====
        points = list(pc2.read_points(msg, skip_nans=True))

        if not points:
            self.drivable_area_points = []
            self.get_logger().warning('⚠️ 주행 가능 영역 포인트 클라우드가 비어있습니다.')
            return

        # 주행 가능 영역의 경계 포인트 저장 (x, y)
        self.drivable_area_points = [(p[0], p[1]) for p in points]
        self.get_logger().debug(f'📍 주행 가능 영역 업데이트: {len(self.drivable_area_points)} 포인트')

    def traffic_light_callback(self, msg):
        # ===== [신호등 입력 처리] STOP/LANE_KEEPING 상태 전환 =====
        sign = msg.data.strip().upper()
        if sign not in ['RED', 'YELLOW', 'GREEN']:
            self.get_logger().warning(f'🚦 알 수 없는 신호등 상태 수신: {msg.data} (무시)')
            return

        self.traffic_light_state = sign
        self.get_logger().info(f'🚦 신호등 상태 수신: {self.traffic_light_state}')

        if self.traffic_light_state == 'RED':
            self.current_state = BehaviorState.STOP
            self.get_logger().warning('🚦 RED -> 정지')
        elif self.traffic_light_state == 'GREEN':
            self.update_state_from_obstacle(self.obstacle_distance)
        elif self.traffic_light_state == 'YELLOW':
            if self.current_state != BehaviorState.STOP:
                self.current_state = BehaviorState.STOP
                self.get_logger().info('🚦 YELLOW -> 정지 준비')

        self.publish_state()
        self.publish_decision_commands()


    def lane_callback(self, msg):
        # ===== [차선 입력 처리] 목표점 추출 + 목표점 발행 =====
        points = list(pc2.read_points(msg, skip_nans=True))

        if not points:
            self.get_logger().warning('⚠️ 차선 포인트 클라우드가 비어있습니다.')
            return

        lane_points_list = [(p[0], p[1]) for p in points]
        target_x, target_y = self.find_lookahead_point(lane_points_list)

        use_astar = (
            self.traffic_light_state.upper() == 'GREEN'
            and self.obstacle_distance < self.astar_trigger_distance_m
            and self.obstacle_x > 0.0
            and abs(self.obstacle_y) < self.astar_y_half_width_m
        )

        planned_x = float(target_x)
        planned_y = float(target_y)
        self.avoidance_active = False
        self.planner_blocked = False

        if use_astar:
            path_world = self.plan_astar_path(target_x, target_y)
            if path_world and len(path_world) >= 2:
                planned_x, planned_y = self.pick_path_target(path_world)
                self.avoidance_active = True
            else:
                # 회피 경로를 못 찾으면 감속 주행 대신 정지로 안전 우선
                self.planner_blocked = True
                self.get_logger().warning('⚠️ A* 회피 경로를 찾지 못해 정지 상태로 전환합니다.')

        self.update_state_from_obstacle(self.obstacle_distance)
        self.publish_target_point(planned_x, planned_y)
        self.publish_decision_commands()

def main(args=None):
    rclpy.init(args=args)
    node = BehaviorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()