import math
from enum import Enum
from time import monotonic

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64, Float32, String
from vision_msgs.msg import Detection2DArray
import sensor_msgs_py.point_cloud2 as pc2


class BehaviorState(Enum):
    """확장된 행동 상태 정의"""
    LANE_KEEPING = 'LANE_KEEPING'
    FOLLOW_VEHICLE = 'FOLLOW_VEHICLE'
    PREPARE_LANE_CHANGE_LEFT = 'PREPARE_LANE_CHANGE_LEFT'
    PREPARE_LANE_CHANGE_RIGHT = 'PREPARE_LANE_CHANGE_RIGHT'
    LANE_CHANGE_LEFT = 'LANE_CHANGE_LEFT'
    LANE_CHANGE_RIGHT = 'LANE_CHANGE_RIGHT'
    RETURN_TO_LANE = 'RETURN_TO_LANE'
    AVOID_OBSTACLE = 'AVOID_OBSTACLE'
    STOP = 'STOP'
    EMERGENCY_STOP = 'EMERGENCY_STOP'


class ObstacleType(Enum):
    """장애물 타입 분류"""
    UNKNOWN = 'UNKNOWN'
    VEHICLE = 'VEHICLE'  # car-like obstacle
    PEDESTRIAN = 'PEDESTRIAN'
    STATIC_OBSTACLE = 'STATIC_OBSTACLE'


class BehaviorNode(Node):
    def __init__(self):
        super().__init__('behavior_node')

        # ===== 기본 제어 파라미터 =====
        self.declare_parameter('control_period_s', 0.1)

        # ===== 동적 속도 제어 파라미터 =====
        self.declare_parameter('desired_speed_straight_mps', 1.10)
        self.declare_parameter('desired_speed_gentle_turn_mps', 0.75)
        self.declare_parameter('desired_speed_sharp_turn_mps', 0.45)
        self.declare_parameter('max_desired_speed_mps', 1.40)

        self.declare_parameter('turn_threshold_abs_local_y_small', 0.35)
        self.declare_parameter('turn_threshold_abs_local_y_large', 0.85)

        # ===== 목표점 생성 파라미터 =====
        self.declare_parameter('lookahead_straight_m', 3.5)
        self.declare_parameter('lookahead_turn_m', 2.0)
        self.declare_parameter('averaging_window_straight', 8)
        self.declare_parameter('averaging_window_turn', 3)
        self.declare_parameter('target_y_clamp_m', 1.2)
        self.declare_parameter('center_offset_m', -0.10)
        self.declare_parameter('target_smoothing_alpha', 0.60)

        # ===== Perception 타임아웃 =====
        self.declare_parameter('lane_timeout_s', 0.5)
        self.declare_parameter('obstacle_timeout_s', 0.5)
        self.declare_parameter('traffic_light_timeout_s', 2.0)
        self.declare_parameter('speed_timeout_s', 1.0)
        self.declare_parameter('detection_timeout_s', 0.7)
        self.declare_parameter('drivable_timeout_s', 0.5)
        self.declare_parameter('drivable_area_topic', '/perception/real_world_drivable_points')

        # ===== 안전 거리 임계값 =====
        self.declare_parameter('caution_distance_m', 20.0)
        self.declare_parameter('emergency_stop_distance_m', 0.8)
        self.declare_parameter('near_obstacle_stop_distance_m', 1.8)
        # lane blocked 판단 거리 (내부 기준으로 사용)
        self.declare_parameter('lane_blocked_distance_m', 3.0)
        # drivable area가 끊겨도 lane keeping을 허용할지 여부 (실차 테스트용 옵션)
        self.declare_parameter('require_drivable_for_lane_keeping', False)

        # ===== Perception 데이터 필터 =====
        self.declare_parameter('lane_y_limit_m', 8.0)
        self.declare_parameter('obstacle_corridor_half_width_m', 1.0)

        # ===== Vehicle 추종 파라미터 =====
        self.declare_parameter('follow_vehicle_min_distance_m', 1.6)
        self.declare_parameter('follow_vehicle_max_distance_m', 5.0)
        self.declare_parameter('follow_vehicle_speed_reduction_factor', 0.8)
        self.declare_parameter('follow_vehicle_lane_threshold_m', 0.5)
        self.declare_parameter('follow_vehicle_time_headway_s', 1.2)
        self.declare_parameter('follow_vehicle_min_gap_m', 1.0)
        self.declare_parameter('follow_vehicle_min_speed_mps', 0.25)
        self.declare_parameter('follow_vehicle_lost_count_max', 5)
        self.declare_parameter('follow_vehicle_detection_score_threshold', 0.35)
        self.declare_parameter('speed_topic', '/carla/ego_vehicle/speedometer')

        # ===== Lane change 파라미터 =====
        self.declare_parameter('lane_change_min_safe_distance_m', 8.0)
        self.declare_parameter('lane_change_max_lateral_distance_m', 3.5)
        self.declare_parameter('lane_change_preparation_distance_m', 10.0)
        self.declare_parameter('lane_change_lateral_offset_m', 1.8)
        self.declare_parameter('lane_change_ramp_duration_s', 2.0)
        self.declare_parameter('lane_change_speed_mps', 0.45)
        self.declare_parameter('lane_change_front_safety_distance_m', 10.0)
        self.declare_parameter('lane_change_rear_safety_distance_m', 8.0)
        self.declare_parameter('enable_lane_change', False)
        self.declare_parameter('stop_while_avoidance_not_committed', True)
        self.declare_parameter('creep_speed_mps', 0.20)

        # ===== Local Lattice Avoidance 파라미터 =====
        self.declare_parameter('enable_obstacle_avoidance', False)
        self.declare_parameter('avoidance_method', 'LATTICE')
        self.declare_parameter('avoidance_trigger_distance_m', 4.0)
        self.declare_parameter('avoidance_min_clearance_m', 0.8)
        self.declare_parameter('avoidance_vehicle_half_width_m', 0.45)
        self.declare_parameter('avoidance_safety_margin_m', 0.35)
        self.declare_parameter('avoidance_target_x_m', 3.0)
        self.declare_parameter('avoidance_lateral_candidates_m', [-1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8])
        self.declare_parameter('avoidance_score_clearance_weight', 3.0)
        self.declare_parameter('avoidance_score_center_weight', 0.8)
        self.declare_parameter('avoidance_score_smoothness_weight', 1.5)
        self.declare_parameter('avoidance_score_progress_weight', 0.3)
        self.declare_parameter('avoidance_commit_count', 3)
        self.declare_parameter('avoidance_release_count', 5)
        self.declare_parameter('avoidance_speed_mps', 0.35)
        self.declare_parameter('avoidance_target_smoothing_alpha', 0.25)
        self.declare_parameter('avoidance_target_y_rate_limit_m', 0.15)

        # ===== Traffic light & stopline 파라미터 =====
        self.declare_parameter('enable_traffic_light', False)
        self.declare_parameter('enable_stopline', False)
        self.declare_parameter('red_light_queue_lookahead_m', 15.0)
        self.declare_parameter('stopline_hold_duration_s', 3.0)
        self.declare_parameter('red_light_ignore_window_s', 2.0)
        self.declare_parameter('unknown_light_forced_go_s', 7.0)
        self.declare_parameter('unknown_light_stop_duration_s', 3.0)
        self.declare_parameter('intersection_ignore_duration_s', 5.0)

        # ===== 정적 장애물 회피 파라미터 =====
        self.declare_parameter('static_obstacle_commit_count', 3)
        self.declare_parameter('static_obstacle_return_finish_count', 5)
        self.declare_parameter('static_obstacle_track_distance_m', 25.0)

        # ===== Pedestrian/Cut-in 감지 파라미터 =====
        self.declare_parameter('pedestrian_intrusion_distance_m', 5.0)
        self.declare_parameter('pedestrian_intrusion_lateral_threshold_m', 1.5)
        self.declare_parameter('pedestrian_detection_score_threshold', 0.30)
        self.declare_parameter('pedestrian_emergency_distance_m', 0.6)
        self.declare_parameter('pedestrian_stop_distance_m', 1.5)
        self.declare_parameter('enable_cutin_detection', False)
        self.declare_parameter('cutin_lateral_velocity_threshold_mps', 0.3)
        self.declare_parameter('cutin_detection_distance_m', 15.0)

        # ===== 파라미터 로드 =====
        self.control_period_s = float(self.get_parameter('control_period_s').value)

        self.desired_speed_straight_mps = float(self.get_parameter('desired_speed_straight_mps').value)
        self.desired_speed_gentle_turn_mps = float(self.get_parameter('desired_speed_gentle_turn_mps').value)
        self.desired_speed_sharp_turn_mps = float(self.get_parameter('desired_speed_sharp_turn_mps').value)
        self.max_desired_speed_mps = float(self.get_parameter('max_desired_speed_mps').value)

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
        self.speed_timeout_s = float(self.get_parameter('speed_timeout_s').value)
        self.detection_timeout_s = float(self.get_parameter('detection_timeout_s').value)
        self.drivable_timeout_s = float(self.get_parameter('drivable_timeout_s').value)
        self.drivable_area_topic = str(self.get_parameter('drivable_area_topic').value)

        self.caution_distance_m = float(self.get_parameter('caution_distance_m').value)
        self.emergency_stop_distance_m = float(self.get_parameter('emergency_stop_distance_m').value)
        self.near_obstacle_stop_distance_m = float(self.get_parameter('near_obstacle_stop_distance_m').value)
        self.lane_blocked_distance_m = float(self.get_parameter('lane_blocked_distance_m').value)
        self.require_drivable_for_lane_keeping = bool(self.get_parameter('require_drivable_for_lane_keeping').value)

        self.lane_y_limit_m = float(self.get_parameter('lane_y_limit_m').value)
        self.obstacle_corridor_half_width_m = float(self.get_parameter('obstacle_corridor_half_width_m').value)

        self.follow_vehicle_min_distance_m = float(self.get_parameter('follow_vehicle_min_distance_m').value)
        self.follow_vehicle_max_distance_m = float(self.get_parameter('follow_vehicle_max_distance_m').value)
        self.follow_vehicle_speed_reduction_factor = float(self.get_parameter('follow_vehicle_speed_reduction_factor').value)
        self.follow_vehicle_lane_threshold_m = float(self.get_parameter('follow_vehicle_lane_threshold_m').value)
        self.follow_vehicle_time_headway_s = float(self.get_parameter('follow_vehicle_time_headway_s').value)
        self.follow_vehicle_min_gap_m = float(self.get_parameter('follow_vehicle_min_gap_m').value)
        self.follow_vehicle_min_speed_mps = float(self.get_parameter('follow_vehicle_min_speed_mps').value)
        self.follow_vehicle_lost_count_max = int(self.get_parameter('follow_vehicle_lost_count_max').value)
        self.follow_vehicle_detection_score_threshold = float(self.get_parameter('follow_vehicle_detection_score_threshold').value)
        self.speed_topic = str(self.get_parameter('speed_topic').value)

        self.lane_change_min_safe_distance_m = float(self.get_parameter('lane_change_min_safe_distance_m').value)
        self.lane_change_max_lateral_distance_m = float(self.get_parameter('lane_change_max_lateral_distance_m').value)
        self.lane_change_preparation_distance_m = float(self.get_parameter('lane_change_preparation_distance_m').value)
        self.lane_change_lateral_offset_m = float(self.get_parameter('lane_change_lateral_offset_m').value)
        self.lane_change_ramp_duration_s = float(self.get_parameter('lane_change_ramp_duration_s').value)
        self.lane_change_speed_mps = float(self.get_parameter('lane_change_speed_mps').value)
        self.lane_change_front_safety_distance_m = float(self.get_parameter('lane_change_front_safety_distance_m').value)
        self.lane_change_rear_safety_distance_m = float(self.get_parameter('lane_change_rear_safety_distance_m').value)
        self.enable_lane_change = bool(self.get_parameter('enable_lane_change').value)
        self.stop_while_avoidance_not_committed = bool(self.get_parameter('stop_while_avoidance_not_committed').value)
        self.creep_speed_mps = float(self.get_parameter('creep_speed_mps').value)

        self.enable_obstacle_avoidance = bool(self.get_parameter('enable_obstacle_avoidance').value)
        self.avoidance_method = str(self.get_parameter('avoidance_method').value).upper()
        self.avoidance_trigger_distance_m = float(self.get_parameter('avoidance_trigger_distance_m').value)
        self.avoidance_min_clearance_m = float(self.get_parameter('avoidance_min_clearance_m').value)
        self.avoidance_vehicle_half_width_m = float(self.get_parameter('avoidance_vehicle_half_width_m').value)
        self.avoidance_safety_margin_m = float(self.get_parameter('avoidance_safety_margin_m').value)
        self.avoidance_target_x_m = float(self.get_parameter('avoidance_target_x_m').value)
        self.avoidance_lateral_candidates_m = [float(v) for v in self.get_parameter('avoidance_lateral_candidates_m').value]
        self.avoidance_score_clearance_weight = float(self.get_parameter('avoidance_score_clearance_weight').value)
        self.avoidance_score_center_weight = float(self.get_parameter('avoidance_score_center_weight').value)
        self.avoidance_score_smoothness_weight = float(self.get_parameter('avoidance_score_smoothness_weight').value)
        self.avoidance_score_progress_weight = float(self.get_parameter('avoidance_score_progress_weight').value)
        self.avoidance_commit_count = max(1, int(self.get_parameter('avoidance_commit_count').value))
        self.avoidance_release_count = max(1, int(self.get_parameter('avoidance_release_count').value))
        self.avoidance_speed_mps = float(self.get_parameter('avoidance_speed_mps').value)
        self.avoidance_target_smoothing_alpha = float(self.get_parameter('avoidance_target_smoothing_alpha').value)
        self.avoidance_target_y_rate_limit_m = float(self.get_parameter('avoidance_target_y_rate_limit_m').value)

        self.enable_traffic_light = bool(self.get_parameter('enable_traffic_light').value)
        self.enable_stopline = bool(self.get_parameter('enable_stopline').value)
        self.red_light_queue_lookahead_m = float(self.get_parameter('red_light_queue_lookahead_m').value)
        self.stopline_hold_duration_s = float(self.get_parameter('stopline_hold_duration_s').value)
        self.red_light_ignore_window_s = float(self.get_parameter('red_light_ignore_window_s').value)
        self.unknown_light_forced_go_s = float(self.get_parameter('unknown_light_forced_go_s').value)
        self.unknown_light_stop_duration_s = float(self.get_parameter('unknown_light_stop_duration_s').value)
        self.intersection_ignore_duration_s = float(self.get_parameter('intersection_ignore_duration_s').value)

        self.static_obstacle_commit_count = int(self.get_parameter('static_obstacle_commit_count').value)
        self.static_obstacle_return_finish_count = int(self.get_parameter('static_obstacle_return_finish_count').value)
        self.static_obstacle_track_distance_m = float(self.get_parameter('static_obstacle_track_distance_m').value)

        self.pedestrian_intrusion_distance_m = float(self.get_parameter('pedestrian_intrusion_distance_m').value)
        self.pedestrian_intrusion_lateral_threshold_m = float(self.get_parameter('pedestrian_intrusion_lateral_threshold_m').value)
        self.pedestrian_detection_score_threshold = float(self.get_parameter('pedestrian_detection_score_threshold').value)
        self.pedestrian_emergency_distance_m = float(self.get_parameter('pedestrian_emergency_distance_m').value)
        self.pedestrian_stop_distance_m = float(self.get_parameter('pedestrian_stop_distance_m').value)
        self.enable_cutin_detection = bool(self.get_parameter('enable_cutin_detection').value)
        self.cutin_lateral_velocity_threshold_mps = float(self.get_parameter('cutin_lateral_velocity_threshold_mps').value)
        self.cutin_detection_distance_m = float(self.get_parameter('cutin_detection_distance_m').value)

        # ===== 입력 구독 =====
        self.lane_sub = self.create_subscription(
            PointCloud2,
            '/perception/real_world_lane_points',
            self.lane_callback,
            10,
        )
        self.obs_sub = self.create_subscription(
            PointCloud2,
            '/perception/closest_obstacle',
            self.obstacle_callback,
            10,
        )
        self.drivable_area_sub = self.create_subscription(
            PointCloud2,
            self.drivable_area_topic,
            self.drivable_area_callback,
            10,
        )
        self.tl_sub = self.create_subscription(
            String,
            '/traffic_light_state',
            self.traffic_light_callback,
            10,
        )
        self.speed_sub = self.create_subscription(
            Float32,
            self.speed_topic,
            self.speed_callback,
            10,
        )
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/yolopv2/detections',
            self.detection_callback,
            10,
        )

        # ===== 출력 발행 =====
        self.desired_speed_pub = self.create_publisher(Float64, '/desired_speed', 10)
        self.target_point_pub = self.create_publisher(Point, '/target_point', 10)
        self.state_pub = self.create_publisher(String, '/behavior_state', 10)
        self.debug_pub = self.create_publisher(String, '/behavior_debug_text', 10)

        # ===== 내부 상태 =====
        self.current_state = BehaviorState.STOP
        self.debug_reason = 'initializing'

        self.lane_points = []
        self.obstacle_points = []
        self.drivable_area_points = []
        self.traffic_light_state = 'GREEN'

        self.current_speed_mps = 0.0

        self.latest_detections = []
        self.detection_class_summary = {
            'vehicle_count': 0,
            'pedestrian_count': 0,
            'vehicle_score_max': 0.0,
            'pedestrian_score_max': 0.0,
        }

        self.obstacle_distance = 99.0
        self.obstacle_x = 99.0
        self.obstacle_y = 0.0

        self.lead_vehicle_distance = 99.0
        self.lead_vehicle_x = 99.0
        self.lead_vehicle_y = 0.0
        self.lead_vehicle_lost_count = 0

        # detection topic liveliness vs presence
        self.detection_topic_alive = False
        self.detections_present = False

        self.lane_change_start_time = None

        self.last_static_obstacle_x = None
        self.last_static_obstacle_y = None
        self.static_seen_count = 0
        self.avoidance_committed = False
        self.return_finish_count = 0
        self.static_track_match_distance_m = 1.2

        self.stopline_mode = 'DISABLED'
        self.stopline_hold_active = False
        self.stopline_hold_until = None
        self.stopline_hold_started_at = None
        self.unknown_light_forced_go_until = None
        self.intersection_ignore_until = None

        self.pedestrian_intrusion_detected = False
        self.pedestrian_intrusion_distance = 99.0
        self.pedestrian_intrusion_severity = 'NONE'
        self.pedestrian_detection_warning = False
        self.cutin_intrusion_detected = False
        self.cutin_intrusion_distance = 99.0

        self.target_x = 0.0
        self.target_y = 0.0

        self.filtered_target_x = None
        self.filtered_target_y = None
        self.filtered_desired_speed = 0.0

        self.desired_speed = 0.0

        self.input_stale = True
        self.main_lane_blocked = False
        self.left_lane_possible = False
        self.right_lane_possible = False
        self.left_target_y_debug = 0.0
        self.right_target_y_debug = 0.0

        self.avoidance_candidate_count = 0
        self.avoidance_release_count_current = 0
        self.avoidance_active = False
        self.avoidance_target_x = 0.0
        self.avoidance_target_y = 0.0
        self.avoidance_score = 0.0
        self.smoothed_avoidance_target_y = 0.0
        self.avoidance_smoothing_initialized = False

        self.fusion_mode = 'NO_CLASS_INFO'
        self.degraded_mode = 'FAILSAFE'
        self.lane_valid = False
        self.obstacle_valid = False
        self.drivable_valid = False
        self.tl_valid = False
        self.speed_valid = False
        self.detection_valid = False

        now = monotonic()
        self.last_lane_update = now
        self.last_obstacle_update = now
        self.last_tl_update = now
        self.last_speed_update = now
        self.last_detection_update = now
        self.last_drivable_update = now

        self.received_lane = False
        self.received_obstacle = False
        self.received_drivable = False
        self.received_speed = False
        self.received_detection = False
        self.received_traffic_light = False

        self.last_state_log = None
        self.last_reason_log = None

        self.timer = self.create_timer(self.control_period_s, self.periodic_update)

        self.get_logger().info('behavior_node v4: camera+lidar fusion aware behavior planning')

    # =========================
    # 콜백
    # =========================
    def lane_callback(self, msg: PointCloud2):
        self.received_lane = True
        self.last_lane_update = monotonic()
        raw_points = list(pc2.read_points(msg, skip_nans=True))
        if not raw_points:
            self.lane_points = []
            return

        lane_points = []
        for p in raw_points:
            x = float(p[0])
            y = float(p[1])
            if x <= 0.0 or abs(y) > self.lane_y_limit_m:
                continue
            lane_points.append((x, y))

        lane_points.sort(key=lambda pt: pt[0])
        self.lane_points = lane_points

    def obstacle_callback(self, msg: PointCloud2):
        self.received_obstacle = True
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
        self.received_drivable = True
        self.last_drivable_update = monotonic()
        raw_points = list(pc2.read_points(msg, skip_nans=True))
        if not raw_points:
            self.drivable_area_points = []
            return
        self.drivable_area_points = [(float(p[0]), float(p[1])) for p in raw_points]

    def traffic_light_callback(self, msg: String):
        self.received_traffic_light = True
        self.last_tl_update = monotonic()
        sign = msg.data.strip().upper()
        # Accept explicit UNKNOWN as well; any other input is treated as UNKNOWN and logged
        if sign in ['RED', 'YELLOW', 'GREEN', 'UNKNOWN']:
            self.traffic_light_state = sign
        else:
            self.traffic_light_state = 'UNKNOWN'
            # record a lightweight debug reason for invalid input
            self.debug_reason = 'invalid_traffic_light_input'

    def speed_callback(self, msg: Float32):
        self.current_speed_mps = max(0.0, float(msg.data))
        self.received_speed = True
        self.last_speed_update = monotonic()

    def detection_callback(self, msg: Detection2DArray):
        self.received_detection = True
        self.last_detection_update = monotonic()
        self.latest_detections = self.parse_yolopv2_detections(msg)
        self.detection_class_summary = self.get_detection_class_summary()

    def quintic_blend(self, t):
        t = max(0.0, min(1.0, float(t)))
        return 6.0 * (t ** 5) - 15.0 * (t ** 4) + 10.0 * (t ** 3)

    def reset_avoidance_smoothing(self):
        self.smoothed_avoidance_target_y = 0.0
        self.avoidance_smoothing_initialized = False

    def update_avoidance_target_smoothing(self, raw_y):
        raw_y = float(raw_y)
        alpha = max(0.0, min(1.0, float(self.avoidance_target_smoothing_alpha)))
        rate_limit = max(0.0, float(self.avoidance_target_y_rate_limit_m))

        if not self.avoidance_smoothing_initialized:
            self.smoothed_avoidance_target_y = raw_y
            self.avoidance_smoothing_initialized = True
        else:
            ema_y = (alpha * raw_y) + ((1.0 - alpha) * self.smoothed_avoidance_target_y)
            delta_y = ema_y - self.smoothed_avoidance_target_y
            delta_y = max(-rate_limit, min(rate_limit, delta_y))
            self.smoothed_avoidance_target_y += delta_y

        return self.smoothed_avoidance_target_y

    def should_ignore_vehicle_for_signal_queue(self, obs_x, obs_y, obs_z):
        if not (self.enable_traffic_light and self.tl_valid and self.traffic_light_state in ('RED', 'YELLOW', 'GREEN')):
            return False

        if obs_x <= 0.0 or obs_x > self.red_light_queue_lookahead_m:
            return False
        if abs(obs_y) > self.obstacle_corridor_half_width_m:
            return False

        obs_type = self.classify_obstacle_from_fusion(obs_x, obs_y, obs_z)
        return obs_type == ObstacleType.VEHICLE

    def reset_stopline_latches(self):
        self.stopline_hold_active = False
        self.stopline_hold_until = None
        self.stopline_hold_started_at = None
        self.unknown_light_forced_go_until = None

    # =========================
    # Detection 파싱/요약
    # =========================
    def parse_yolopv2_detections(self, msg: Detection2DArray):
        parsed = []
        try:
            detections = getattr(msg, 'detections', [])
            for det in detections:
                results = getattr(det, 'results', [])
                best_label = ''
                best_score = 0.0

                for result in results:
                    label, score = self._extract_label_score_from_result(result)
                    if score >= best_score:
                        best_label = label
                        best_score = score

                if not best_label and results:
                    # 결과가 있지만 id/label이 비어있는 경우
                    best_label = 'unknown'

                parsed.append({
                    'label': best_label,
                    'score': float(best_score),
                })
        except Exception:
            return []
        return parsed

    def _extract_label_score_from_result(self, result):
        label = ''
        score = 0.0
        try:
            hypothesis = getattr(result, 'hypothesis', None)
            if hypothesis is not None:
                raw_id = getattr(hypothesis, 'class_id', None)
                if raw_id is None:
                    raw_id = getattr(hypothesis, 'id', None)
                score = float(getattr(hypothesis, 'score', 0.0))
                label = self._normalize_detection_label(raw_id)
            else:
                raw_id = getattr(result, 'class_id', None)
                if raw_id is None:
                    raw_id = getattr(result, 'id', None)
                score = float(getattr(result, 'score', 0.0))
                label = self._normalize_detection_label(raw_id)
        except Exception:
            return '', 0.0
        return label, score

    def _normalize_detection_label(self, raw_id):
        if raw_id is None:
            return ''

        raw = str(raw_id).strip().lower()
        if raw == '':
            return ''

        # 권장 매핑: 내부에서 'car' / 'pedestrian' / 'unknown' 으로 일관화
        # 숫자 class id 매핑 (YOLO/COCO 계열 예상)
        id_to_label = {
            '0': 'pedestrian',
            '1': 'unknown',    # bicycle -> unknown for conservative handling
            '2': 'car',
            '3': 'unknown',    # motorcycle -> unknown
            '5': 'car',
            '7': 'car',
        }
        if raw in id_to_label:
            return id_to_label[raw]

        # 문자열 라벨 매핑: 다양한 표현을 하나로 통일
        token = raw.lower()
        car_tokens = {'car', 'truck', 'bus', 'vehicle', 'van'}
        pedestrian_tokens = {'person', 'pedestrian'}
        bike_tokens = {'bicycle', 'motorcycle', 'bike', 'motorbike'}

        if token in car_tokens:
            return 'car'
        if token in pedestrian_tokens:
            return 'pedestrian'
        if token in bike_tokens:
            return 'unknown'

        return 'unknown'

    def get_detection_class_summary(self):
        # 통합된 summary: car / pedestrian 중심
        summary = {
            'car_count': 0,
            'pedestrian_count': 0,
            'car_score_max': 0.0,
            'pedestrian_score_max': 0.0,
            # backward compatibility
            'vehicle_count': 0,
            'vehicle_score_max': 0.0,
        }

        for det in self.latest_detections:
            label = det.get('label', '')
            score = float(det.get('score', 0.0))
            if not label:
                continue

            if label == 'car':
                summary['car_count'] += 1
                summary['car_score_max'] = max(summary['car_score_max'], score)
            elif label == 'pedestrian':
                summary['pedestrian_count'] += 1
                summary['pedestrian_score_max'] = max(summary['pedestrian_score_max'], score)
            else:
                # unknown or other labels are ignored for counts
                continue

        # keep legacy vehicle_* keys equal to car_* for backward compatibility
        summary['vehicle_count'] = summary['car_count']
        summary['vehicle_score_max'] = summary['car_score_max']
        return summary

    def has_vehicle_detection(self):
        # 호환성 유지: car-like detection 확인용 함수
        return (
            self.detection_class_summary.get('car_count', 0) > 0
            and self.detection_class_summary.get('car_score_max', 0.0) >= self.follow_vehicle_detection_score_threshold
        )

    def has_pedestrian_detection(self):
        return (
            self.detection_class_summary['pedestrian_count'] > 0
            and self.detection_class_summary['pedestrian_score_max'] >= self.pedestrian_detection_score_threshold
        )

    # =========================
    # Fusion 기반 분류/판단
    # =========================
    def compute_fusion_mode(self):
        has_lidar = self.obstacle_valid
        has_camera = self.detection_valid

        if has_lidar and has_camera:
            return 'CAMERA_LIDAR'
        if has_lidar:
            return 'LIDAR_ONLY'
        if has_camera:
            return 'CAMERA_ONLY'
        return 'NO_CLASS_INFO'

    def classify_obstacle_from_fusion(self, x, y, z):
        # Geometry-first conservative classification
        # Base classification by height (z)
        if z < 0.2:
            base_type = ObstacleType.STATIC_OBSTACLE
        elif z > 0.5:
            base_type = ObstacleType.VEHICLE
        else:
            base_type = ObstacleType.UNKNOWN

        # Detection summary is auxiliary only. Do not directly match image-pixel
        # detections with lidar metric coordinates here.
        if base_type == ObstacleType.UNKNOWN and self.detection_valid and self.has_vehicle_detection() and z >= 0.4:
            return ObstacleType.VEHICLE

        if base_type == ObstacleType.UNKNOWN and self.detection_valid and self.has_pedestrian_detection():
            if x <= self.pedestrian_intrusion_distance_m and abs(y) <= self.pedestrian_intrusion_lateral_threshold_m and 0.2 <= z <= 2.0:
                return ObstacleType.PEDESTRIAN

        return base_type

    def select_forward_vehicle_on_path(self):
        lead_dist = 99.0
        lead_x = 99.0
        lead_y = 0.0

        for obs_x, obs_y, obs_z in self.obstacle_points:
            if obs_x <= 0.5 or obs_x > self.follow_vehicle_max_distance_m:
                continue
            if abs(obs_y) > self.obstacle_corridor_half_width_m:
                continue

            obs_type = self.classify_obstacle_from_fusion(obs_x, obs_y, obs_z)
            if obs_type != ObstacleType.VEHICLE:
                continue

            dist = math.sqrt(obs_x * obs_x + obs_y * obs_y)
            if dist < lead_dist:
                lead_dist = dist
                lead_x = obs_x
                lead_y = obs_y

        return lead_dist, lead_x, lead_y

    def compute_gap_to_lead_vehicle(self, lead_x, current_speed):
        ego_front_offset = 0.5
        desired_gap = max(self.follow_vehicle_min_gap_m, self.follow_vehicle_time_headway_s * current_speed)
        return lead_x - ego_front_offset - desired_gap

    def detect_pedestrian_intrusion(self):
        self.pedestrian_detection_warning = False

        if not self.obstacle_valid:
            if self.detection_valid and self.has_pedestrian_detection():
                self.pedestrian_detection_warning = True
            return False, 99.0, 'NONE'

        closest_dist = 99.0
        search_distance = max(self.pedestrian_intrusion_distance_m, self.pedestrian_stop_distance_m)
        for obs_x, obs_y, obs_z in self.obstacle_points:
            if obs_x <= 0.0 or obs_x > search_distance:
                continue
            if abs(obs_y) > self.pedestrian_intrusion_lateral_threshold_m:
                continue
            dist = math.sqrt(obs_x * obs_x + obs_y * obs_y)
            if dist < closest_dist:
                closest_dist = dist

        if not (self.detection_valid and self.has_pedestrian_detection()):
            return False, 99.0, 'NONE'

        if closest_dist == 99.0:
            self.pedestrian_detection_warning = True
            return False, 99.0, 'WARNING_ONLY'

        if closest_dist <= self.pedestrian_emergency_distance_m:
            return True, closest_dist, 'EMERGENCY_STOP'
        if closest_dist <= self.pedestrian_stop_distance_m:
            return True, closest_dist, 'STOP'

        return False, 99.0, 'NONE'

    def detect_cutin_intrusion(self):
        threshold = self.cutin_lateral_velocity_threshold_mps
        if self.detection_valid and self.has_vehicle_detection():
            threshold *= 0.8  # 차량 검출이 있으면 조금 더 민감

        for obs_x, obs_y, obs_z in self.obstacle_points:
            if obs_x <= 0.0 or obs_x > self.cutin_detection_distance_m:
                continue
            if abs(obs_y) < 0.5 or abs(obs_y) > 3.5:
                continue
            if self.should_ignore_vehicle_for_signal_queue(obs_x, obs_y, obs_z):
                continue

            obs_type = self.classify_obstacle_from_fusion(obs_x, obs_y, obs_z)
            if obs_type != ObstacleType.VEHICLE:
                continue

            lateral_rate = abs(obs_y) / max(obs_x, 0.1)
            if lateral_rate > threshold:
                dist = math.sqrt(obs_x * obs_x + obs_y * obs_y)
                return True, dist

        return False, 99.0

    def is_lane_blocked(self, check_distance_m=15.0):
        for obs_x, obs_y, obs_z in self.obstacle_points:
            if obs_x <= 0.0 or obs_x > check_distance_m:
                continue
            if abs(obs_y) > self.obstacle_corridor_half_width_m:
                continue
            if self.should_ignore_vehicle_for_signal_queue(obs_x, obs_y, obs_z):
                continue
            dist = math.sqrt(obs_x * obs_x + obs_y * obs_y)
            # lane_blocked_distance_m is the configured threshold for "blocked" status
            if dist < float(self.lane_blocked_distance_m):
                return True
        return False

    def select_blocking_static_obstacle(self):
        best_dist = 99.0
        best_x = 99.0
        best_y = 0.0

        for obs_x, obs_y, obs_z in self.obstacle_points:
            if obs_x <= 0.0 or obs_x > self.static_obstacle_track_distance_m:
                continue
            if abs(obs_y) > self.obstacle_corridor_half_width_m:
                continue
            if self.should_ignore_vehicle_for_signal_queue(obs_x, obs_y, obs_z):
                continue

            obs_type = self.classify_obstacle_from_fusion(obs_x, obs_y, obs_z)
            if obs_type != ObstacleType.STATIC_OBSTACLE:
                continue

            dist = math.sqrt(obs_x * obs_x + obs_y * obs_y)
            if dist < best_dist:
                best_dist = dist
                best_x = obs_x
                best_y = obs_y

        return best_dist, best_x, best_y

    def get_drivable_y_range_at_x(self, target_x, x_tolerance=0.7):
        if not self.drivable_valid or not self.drivable_area_points:
            return None

        target_x = float(target_x)
        x_tolerance = float(x_tolerance)
        local_y_values = [y for x, y in self.drivable_area_points if abs(x - target_x) <= x_tolerance]
        if local_y_values:
            return min(local_y_values), max(local_y_values)

        all_y_values = [y for _, y in self.drivable_area_points]
        if not all_y_values:
            return None
        return min(all_y_values), max(all_y_values)

    def is_target_in_drivable_area(self, target_x, target_y):
        if not self.drivable_valid or not self.drivable_area_points:
            return False

        drivable_range = self.get_drivable_y_range_at_x(target_x)
        if drivable_range is None:
            return False

        min_y, max_y = drivable_range
        target_y = float(target_y)
        return min_y <= target_y <= max_y

    def distance_point_to_segment(self, px, py, x1, y1, x2, y2):
        px = float(px)
        py = float(py)
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)

        dx = x2 - x1
        dy = y2 - y1
        denom = dx * dx + dy * dy
        if denom <= 1e-9:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        t = ((px - x1) * dx + (py - y1) * dy) / denom
        t = max(0.0, min(1.0, t))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    def check_candidate_collision_free(self, target_x, target_y):
        required_clearance = max(
            self.avoidance_vehicle_half_width_m + self.avoidance_safety_margin_m,
            self.avoidance_min_clearance_m,
        )
        target_x = float(target_x)
        target_y = float(target_y)

        if not self.obstacle_valid:
            return True

        for obs_x, obs_y, obs_z in self.obstacle_points:
            if obs_x <= 0.0 or obs_x > (target_x + 1.0):
                continue
            if self.should_ignore_vehicle_for_signal_queue(obs_x, obs_y, obs_z):
                continue

            obs_type = self.classify_obstacle_from_fusion(obs_x, obs_y, obs_z)
            if obs_type == ObstacleType.PEDESTRIAN:
                return False

            planar_dist = math.sqrt(obs_x * obs_x + obs_y * obs_y)
            if planar_dist <= self.emergency_stop_distance_m:
                return False

            if obs_type == ObstacleType.UNKNOWN and planar_dist <= self.avoidance_min_clearance_m:
                return False

            segment_dist = self.distance_point_to_segment(obs_x, obs_y, 0.0, 0.0, target_x, target_y)
            if segment_dist < required_clearance:
                return False

        return True

    def score_avoidance_candidate(self, target_x, target_y):
        if not self.check_candidate_collision_free(target_x, target_y):
            return None

        min_clearance = float('inf')
        for obs_x, obs_y, obs_z in self.obstacle_points:
            if obs_x <= 0.0 or obs_x > (float(target_x) + 1.0):
                continue
            if self.should_ignore_vehicle_for_signal_queue(obs_x, obs_y, obs_z):
                continue
            obs_type = self.classify_obstacle_from_fusion(obs_x, obs_y, obs_z)
            if obs_type == ObstacleType.PEDESTRIAN:
                continue
            segment_dist = self.distance_point_to_segment(obs_x, obs_y, 0.0, 0.0, target_x, target_y)
            min_clearance = min(min_clearance, segment_dist)

        if not math.isfinite(min_clearance):
            min_clearance = self.avoidance_vehicle_half_width_m + self.avoidance_safety_margin_m + 1.0

        previous_target_y = self.filtered_target_y if self.filtered_target_y is not None else self.target_y
        score = (
            self.avoidance_score_clearance_weight * min_clearance
            - self.avoidance_score_center_weight * abs(float(target_y))
            - self.avoidance_score_smoothness_weight * abs(float(target_y) - float(previous_target_y))
            + self.avoidance_score_progress_weight * float(target_x)
        )
        return float(score)

    def compute_lattice_avoidance_target(self):
        if not self.drivable_valid or not self.obstacle_valid:
            return None

        target_x = float(self.avoidance_target_x_m)
        best_candidate = None

        for target_y in self.avoidance_lateral_candidates_m:
            candidate_y = max(-self.target_y_clamp_m, min(self.target_y_clamp_m, float(target_y)))
            if not self.is_target_in_drivable_area(target_x, candidate_y):
                continue
            if not self.check_candidate_collision_free(target_x, candidate_y):
                continue
            score = self.score_avoidance_candidate(target_x, candidate_y)
            if score is None:
                continue
            if best_candidate is None or score > best_candidate[2]:
                best_candidate = (float(target_x), float(candidate_y), float(score))

        return best_candidate

    def update_static_obstacle_tracking(self, static_dist, static_x, static_y):
        if static_dist >= self.static_obstacle_track_distance_m:
            if not self.avoidance_committed:
                self.static_seen_count = 0
                self.last_static_obstacle_x = None
                self.last_static_obstacle_y = None
            return

        if self.last_static_obstacle_x is None or self.last_static_obstacle_y is None:
            self.static_seen_count = 1
        else:
            delta = math.sqrt((static_x - self.last_static_obstacle_x) ** 2 + (static_y - self.last_static_obstacle_y) ** 2)
            if delta <= self.static_track_match_distance_m:
                self.static_seen_count += 1
            else:
                self.static_seen_count = 1

        self.last_static_obstacle_x = static_x
        self.last_static_obstacle_y = static_y
        self.avoidance_committed = self.static_seen_count >= self.static_obstacle_commit_count

    def should_finish_return(self):
        self.return_finish_count += 1
        return self.return_finish_count >= self.static_obstacle_return_finish_count

    def evaluate_lane_change_options(self):
        left_possible = True
        right_possible = True
        left_blocked = False
        right_blocked = False

        if not self.drivable_valid:
            return False, False, False, False

        y_values = [y for _, y in self.drivable_area_points]
        if not y_values:
            return False, False, False, False

        drivable_min_y = min(y_values)
        drivable_max_y = max(y_values)

        route_target = self.choose_target_from_route()
        base_target_y = route_target[2] if route_target is not None else self.target_y

        left_target_y = float(base_target_y + self.lane_change_lateral_offset_m)
        right_target_y = float(base_target_y - self.lane_change_lateral_offset_m)
        self.left_target_y_debug = left_target_y
        self.right_target_y_debug = right_target_y

        if left_target_y < drivable_min_y or left_target_y > drivable_max_y or abs(left_target_y) > self.lane_change_max_lateral_distance_m:
            left_possible = False
        elif not self._is_merge_target_lane_safe(left_target_y):
            left_possible = False
            left_blocked = True

        if right_target_y < drivable_min_y or right_target_y > drivable_max_y or abs(right_target_y) > self.lane_change_max_lateral_distance_m:
            right_possible = False
        elif not self._is_merge_target_lane_safe(right_target_y):
            right_possible = False
            right_blocked = True

        return left_possible, right_possible, left_blocked, right_blocked

    def _is_merge_target_lane_safe(self, target_y):
        for obs_x, obs_y, obs_z in self.obstacle_points:
            if abs(obs_y - target_y) > 1.5:
                continue

            dist = math.sqrt(obs_x * obs_x + obs_y * obs_y)

            if 0.0 < obs_x <= self.lane_change_front_safety_distance_m and dist < self.lane_change_min_safe_distance_m:
                return False

            if -5.0 < obs_x <= 0.0 and dist < self.lane_change_rear_safety_distance_m:
                return False

        return True

    def compute_lane_change_target(self, direction):
        base_target = self.choose_target_from_route()
        base_y = base_target[2] if base_target is not None else self.target_y

        if direction == 'LEFT':
            target_y = base_y + self.lane_change_lateral_offset_m
        elif direction == 'RIGHT':
            target_y = base_y - self.lane_change_lateral_offset_m
        else:
            return float(self.target_y)

        if self.drivable_area_points:
            y_values = [y for _, y in self.drivable_area_points]
            drivable_min_y = min(y_values)
            drivable_max_y = max(y_values)
            target_y = max(drivable_min_y, min(drivable_max_y, target_y))

        target_y = max(-self.lane_change_max_lateral_distance_m, min(self.lane_change_max_lateral_distance_m, target_y))
        return float(target_y)

    # =========================
    # 속도 계산
    # =========================
    def compute_follow_speed(self, lead_vehicle_x, lead_vehicle_distance, current_speed):
        # lead_vehicle_x: forward x position of the lead vehicle (used for gap calc)
        # lead_vehicle_distance: Euclidean distance (used for thresholds/debug)
        if lead_vehicle_distance >= self.follow_vehicle_max_distance_m:
            return self.compute_nominal_speed(self.target_y)

        if lead_vehicle_distance <= self.follow_vehicle_min_distance_m:
            return self.follow_vehicle_min_speed_mps

        gap = self.compute_gap_to_lead_vehicle(lead_vehicle_x, current_speed)
        if gap > 2.0:
            return self.compute_nominal_speed(self.target_y)
        if gap < -1.0:
            return self.follow_vehicle_min_speed_mps

        t = max(0.0, min(1.0, (gap + 1.0) / 3.0))
        nominal = self.compute_nominal_speed(self.target_y)
        reduced = nominal * self.follow_vehicle_speed_reduction_factor
        return reduced + t * (nominal - reduced)

    def compute_nominal_speed(self, local_y):
        abs_y = abs(float(local_y))
        if abs_y < self.turn_threshold_small:
            return self.desired_speed_straight_mps
        if abs_y < self.turn_threshold_large:
            return self.desired_speed_gentle_turn_mps
        return self.desired_speed_sharp_turn_mps

    # =========================
    # 상태 결정
    # =========================
    def decide_next_state(self, now, current_speed):
        self.pedestrian_intrusion_detected, self.pedestrian_intrusion_distance, self.pedestrian_intrusion_severity = self.detect_pedestrian_intrusion()
        if self.enable_cutin_detection:
            self.cutin_intrusion_detected, self.cutin_intrusion_distance = self.detect_cutin_intrusion()
        else:
            self.cutin_intrusion_detected, self.cutin_intrusion_distance = False, 99.0

        # 1) pedestrian emergency
        if self.pedestrian_intrusion_detected and self.pedestrian_intrusion_severity == 'EMERGENCY_STOP':
            return BehaviorState.EMERGENCY_STOP, 'pedestrian_emergency_stop'

        # 2) near obstacle emergency/stop always wins before lane decisions
        if self.obstacle_valid and self.obstacle_distance < self.emergency_stop_distance_m:
            return BehaviorState.EMERGENCY_STOP, 'obstacle_too_close'

        if self.obstacle_valid and self.obstacle_distance < self.near_obstacle_stop_distance_m:
            return BehaviorState.STOP, 'near_obstacle_stop_band'

        # 3) lane and drivable validity
        if not self.lane_valid:
            return BehaviorState.STOP, 'lane_unavailable'

        # drivable_valid 정책: require_drivable_for_lane_keeping에 따라 동작
        if not self.drivable_valid:
            if self.require_drivable_for_lane_keeping:
                return BehaviorState.STOP, 'drivable_unavailable'
            else:
                # drivable이 잠깐 끊겨도 lane keeping을 허용한다. 단, 회피/차선 변경은 금지.
                self.debug_reason = 'drivable_missing_allowed_by_config'

        # 4) pedestrian stop
        if self.pedestrian_intrusion_detected and self.pedestrian_intrusion_severity == 'STOP':
            return BehaviorState.STOP, 'pedestrian_stop'

        # 5) traffic light / stopline (default OFF; structural guard only)
        if self.enable_traffic_light and self.tl_valid and self.traffic_light_state in ('RED', 'YELLOW'):
            queue_lead_dist, _, _ = self.detect_queue_lead_vehicle()
            if queue_lead_dist < self.red_light_queue_lookahead_m:
                return BehaviorState.STOP, 'red_light_queue_vehicle'
            return BehaviorState.STOP, f'traffic_light_{self.traffic_light_state.lower()}'

        if self.enable_stopline and self.stopline_hold_active and self.stopline_hold_until and now < self.stopline_hold_until:
            return BehaviorState.STOP, 'stopline_hold_active'

        # 6) cut-in (optional)
        if self.enable_cutin_detection and self.cutin_intrusion_detected:
            return BehaviorState.EMERGENCY_STOP, 'cutin_intrusion_detected'

        # 7) follow vehicle
        lead_dist, _, lead_y = self.select_forward_vehicle_on_path()
        if self.near_obstacle_stop_distance_m <= lead_dist <= self.follow_vehicle_max_distance_m and abs(lead_y) <= self.follow_vehicle_lane_threshold_m:
            if self.obstacle_valid and self.speed_valid:
                self.lead_vehicle_lost_count = 0
                if self.detection_valid and self.has_vehicle_detection():
                    return BehaviorState.FOLLOW_VEHICLE, 'lead_vehicle_detected_camera_boost'
                return BehaviorState.FOLLOW_VEHICLE, 'lead_vehicle_detected_lidar_only'

        self.lead_vehicle_lost_count += 1
        if self.lead_vehicle_lost_count <= self.follow_vehicle_lost_count_max and self.current_state == BehaviorState.FOLLOW_VEHICLE:
            if self.speed_valid:
                return BehaviorState.FOLLOW_VEHICLE, 'lead_vehicle_persist'

        self.lead_vehicle_lost_count = min(self.lead_vehicle_lost_count, self.follow_vehicle_lost_count_max + 1)

        # 8) return-to-lane keep
        if self.current_state == BehaviorState.RETURN_TO_LANE:
            if not self.enable_lane_change:
                return BehaviorState.STOP, 'lane_change_disabled'
            if self.should_finish_return():
                self.avoidance_committed = False
                self.static_seen_count = 0
                self.last_static_obstacle_x = None
                self.last_static_obstacle_y = None
                self.return_finish_count = 0
                return BehaviorState.LANE_KEEPING, 'return_to_lane_complete'
            return BehaviorState.RETURN_TO_LANE, 'returning_to_lane'

        # 9) lane blocked is obstacle-only (independent from drivable validity)
        main_lane_blocked = False
        if self.obstacle_valid:
            main_lane_blocked = self.is_lane_blocked(self.lane_change_preparation_distance_m)
        self.main_lane_blocked = main_lane_blocked

        # 10) Local Lattice Avoidance: local target selection inside drivable area
        if self.enable_obstacle_avoidance and self.avoidance_method == 'LATTICE' and self.drivable_valid and self.obstacle_valid and self.obstacle_distance <= self.avoidance_trigger_distance_m:
            lattice_candidate = self.compute_lattice_avoidance_target()
            if lattice_candidate is not None:
                self.avoidance_candidate_count += 1
                self.avoidance_release_count_current = 0
                self.avoidance_target_x, self.avoidance_target_y, self.avoidance_score = lattice_candidate
                if self.avoidance_candidate_count >= self.avoidance_commit_count:
                    self.avoidance_active = True
                    return BehaviorState.AVOID_OBSTACLE, 'lattice_avoidance_committed'
                return BehaviorState.STOP, 'lattice_avoidance_candidate_waiting'

            self.avoidance_candidate_count = 0
            self.avoidance_release_count_current += 1
            if self.avoidance_active:
                if self.avoidance_release_count_current >= self.avoidance_release_count:
                    self.avoidance_active = False
                    self.avoidance_release_count_current = 0
                    self.reset_avoidance_smoothing()
                    return BehaviorState.LANE_KEEPING, 'lattice_avoidance_released'
                else:
                    return BehaviorState.AVOID_OBSTACLE, 'lattice_avoidance_release_pending'

            if self.enable_lane_change and main_lane_blocked:
                # lattice candidate가 없을 때만 lane change fallback
                pass
            else:
                return BehaviorState.STOP, 'lattice_avoidance_no_safe_target'

        if self.avoidance_active:
            if not main_lane_blocked:
                self.avoidance_release_count_current += 1
                if self.avoidance_release_count_current >= self.avoidance_release_count:
                    self.avoidance_active = False
                    self.avoidance_candidate_count = 0
                    self.avoidance_release_count_current = 0
                    self.reset_avoidance_smoothing()
                else:
                    return BehaviorState.AVOID_OBSTACLE, 'lattice_avoidance_release_pending'
            else:
                self.avoidance_release_count_current = 0
                return BehaviorState.AVOID_OBSTACLE, 'lattice_avoidance_committed'

        # 11) lane blocked and lane change disabled -> immediate STOP
        if main_lane_blocked and not self.enable_lane_change:
            return BehaviorState.STOP, 'lane_blocked_lane_change_disabled'

        # 12) lane blocked and lane change enabled
        if main_lane_blocked and self.enable_lane_change:
            if not self.drivable_valid:
                return BehaviorState.STOP, 'lane_blocked_drivable_unavailable'

            left_possible, right_possible, _, _ = self.evaluate_lane_change_options()

            static_dist, static_x, static_y = self.select_blocking_static_obstacle()
            self.update_static_obstacle_tracking(static_dist, static_x, static_y)

            if not self.avoidance_committed:
                if self.stop_while_avoidance_not_committed:
                    return BehaviorState.STOP, 'lane_blocked_waiting_avoidance_commit'
                return BehaviorState.LANE_KEEPING, 'lane_blocked_creep'

            if left_possible:
                if self.current_state in (BehaviorState.LANE_KEEPING, BehaviorState.STOP):
                    return BehaviorState.PREPARE_LANE_CHANGE_LEFT, 'static_blocked_left_available'
                if self.current_state == BehaviorState.PREPARE_LANE_CHANGE_LEFT:
                    return BehaviorState.LANE_CHANGE_LEFT, 'prepare_left_done'
                if self.current_state == BehaviorState.LANE_CHANGE_LEFT:
                    if self._is_lane_change_complete():
                        return BehaviorState.RETURN_TO_LANE, 'lane_change_left_complete'
                    return BehaviorState.LANE_CHANGE_LEFT, 'lane_changing_left'

            if right_possible:
                if self.current_state in (BehaviorState.LANE_KEEPING, BehaviorState.STOP):
                    return BehaviorState.PREPARE_LANE_CHANGE_RIGHT, 'static_blocked_right_available'
                if self.current_state == BehaviorState.PREPARE_LANE_CHANGE_RIGHT:
                    return BehaviorState.LANE_CHANGE_RIGHT, 'prepare_right_done'
                if self.current_state == BehaviorState.LANE_CHANGE_RIGHT:
                    if self._is_lane_change_complete():
                        return BehaviorState.RETURN_TO_LANE, 'lane_change_right_complete'
                    return BehaviorState.LANE_CHANGE_RIGHT, 'lane_changing_right'

            return BehaviorState.STOP, 'avoidance_no_option'

        # 13) default cruise
        return BehaviorState.LANE_KEEPING, 'cruise'

    def detect_queue_lead_vehicle(self):
        # detection vehicle가 있으면 queue 차량 신뢰도 상승
        require_vehicle = self.detection_valid and self.has_vehicle_detection()

        for obs_x, obs_y, obs_z in self.obstacle_points:
            if obs_x <= 0.0 or obs_x > self.red_light_queue_lookahead_m:
                continue
            if abs(obs_y) > self.obstacle_corridor_half_width_m:
                continue

            obs_type = self.classify_obstacle_from_fusion(obs_x, obs_y, obs_z)
            if require_vehicle and obs_type != ObstacleType.VEHICLE:
                continue
            if not require_vehicle and obs_type not in (ObstacleType.VEHICLE, ObstacleType.UNKNOWN):
                continue

            dist = math.sqrt(obs_x * obs_x + obs_y * obs_y)
            return dist, obs_x, obs_y

        return 99.0, 99.0, 0.0

    def _is_lane_change_complete(self):
        if self.lane_change_start_time is None:
            return False
        return (monotonic() - self.lane_change_start_time) >= self.lane_change_ramp_duration_s

    # =========================
    # Target point 생성
    # =========================
    def choose_target_from_route(self):
        if not self.lane_points:
            return None

        forward_points = [(x, y) for x, y in self.lane_points if x > 0.5]
        if not forward_points:
            return None

        nearby_abs_y = [abs(p[1]) for p in forward_points[:3]]
        is_in_corner = len(nearby_abs_y) > 0 and max(nearby_abs_y) > self.turn_threshold_small

        target_lookahead = self.lookahead_turn_m if is_in_corner else self.lookahead_straight_m

        candidate_idx = None
        for i, (x, _) in enumerate(forward_points):
            if x >= target_lookahead:
                candidate_idx = i
                break
        if candidate_idx is None:
            candidate_idx = max(0, len(forward_points) - 1)

        avg_window = self.averaging_window_turn if is_in_corner else self.averaging_window_straight
        start_idx = max(0, candidate_idx - 1)
        end_idx = min(len(forward_points), candidate_idx + avg_window)
        selected = forward_points[start_idx:end_idx]

        if len(selected) == 1:
            avg_x, avg_y = selected[0]
        else:
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

        corrected_y = avg_y * 0.95 if not is_in_corner else (avg_y - self.center_offset_m)
        corrected_y = max(-self.target_y_clamp_m, min(self.target_y_clamp_m, corrected_y))

        return is_in_corner, float(avg_x), float(corrected_y)

    def compute_target_point_for_state(self, state, now):
        if state in (BehaviorState.STOP, BehaviorState.EMERGENCY_STOP):
            return 0.0, 0.0, False

        if state == BehaviorState.AVOID_OBSTACLE:
            target_x = float(self.avoidance_target_x)
            target_y_source = self.smoothed_avoidance_target_y if self.avoidance_smoothing_initialized else self.avoidance_target_y
            target_y = max(-self.target_y_clamp_m, min(self.target_y_clamp_m, float(target_y_source)))
            if not self.is_target_in_drivable_area(target_x, target_y):
                return 0.0, 0.0, False
            return target_x, target_y, False

        if state in (BehaviorState.LANE_KEEPING, BehaviorState.FOLLOW_VEHICLE):
            target = self.choose_target_from_route()
            if target is None:
                return 0.0, 0.0, False
            is_corner, target_x, target_y = target
            return target_x, target_y, is_corner

        if state == BehaviorState.PREPARE_LANE_CHANGE_LEFT:
            target = self.choose_target_from_route()
            if target is None:
                return 0.0, 0.0, False
            _, target_x, target_y = target
            return target_x, target_y + 0.3, False

        if state == BehaviorState.PREPARE_LANE_CHANGE_RIGHT:
            target = self.choose_target_from_route()
            if target is None:
                return 0.0, 0.0, False
            _, target_x, target_y = target
            return target_x, target_y - 0.3, False

        if state == BehaviorState.LANE_CHANGE_LEFT:
            if self.lane_change_start_time is None:
                self.lane_change_start_time = now
            target_y_final = self.compute_lane_change_target('LEFT')
            target = self.choose_target_from_route()
            if target is None:
                return 0.0, target_y_final, False
            _, target_x, target_y_base = target
            progress = min(1.0, (now - self.lane_change_start_time) / self.lane_change_ramp_duration_s)
            smooth_progress = self.quintic_blend(progress)
            target_y = target_y_base + smooth_progress * (target_y_final - target_y_base)
            return target_x, target_y, False

        if state == BehaviorState.LANE_CHANGE_RIGHT:
            if self.lane_change_start_time is None:
                self.lane_change_start_time = now
            target_y_final = self.compute_lane_change_target('RIGHT')
            target = self.choose_target_from_route()
            if target is None:
                return 0.0, target_y_final, False
            _, target_x, target_y_base = target
            progress = min(1.0, (now - self.lane_change_start_time) / self.lane_change_ramp_duration_s)
            smooth_progress = self.quintic_blend(progress)
            target_y = target_y_base + smooth_progress * (target_y_final - target_y_base)
            return target_x, target_y, False

        if state == BehaviorState.RETURN_TO_LANE:
            target_y_final = 0.0
            target = self.choose_target_from_route()
            if target is None:
                return 0.0, target_y_final, False
            _, target_x, target_y_base = target
            progress = min(1.0, self.return_finish_count / max(1, self.static_obstacle_return_finish_count))
            target_y = target_y_base + progress * (target_y_final - target_y_base)
            return target_x, target_y, False

        return 0.0, 0.0, False

    # =========================
    # 주기 업데이트
    # =========================
    def periodic_update(self):
        now = monotonic()
        self.stopline_mode = 'ENABLED' if self.enable_stopline else 'DISABLED'

        # 1) stale 검사
        lane_stale = (now - self.last_lane_update) > self.lane_timeout_s
        obstacle_stale = (now - self.last_obstacle_update) > self.obstacle_timeout_s
        tl_stale = (now - self.last_tl_update) > self.traffic_light_timeout_s
        speed_stale = (now - self.last_speed_update) > self.speed_timeout_s
        detection_stale = (now - self.last_detection_update) > self.detection_timeout_s
        drivable_stale = (now - self.last_drivable_update) > self.drivable_timeout_s

        # 2) 입력별 유효성 요약
        self.lane_valid = self.received_lane and (not lane_stale) and bool(self.lane_points)
        self.obstacle_valid = self.received_obstacle and (not obstacle_stale) and bool(self.obstacle_points)
        self.drivable_valid = self.received_drivable and (not drivable_stale) and bool(self.drivable_area_points)
        self.tl_valid = self.received_traffic_light and (not tl_stale)
        self.speed_valid = self.received_speed and (not speed_stale)
        # split detection liveliness vs presence
        self.detection_topic_alive = self.received_detection and (not detection_stale)
        self.detections_present = len(self.latest_detections) > 0
        self.detection_valid = self.detection_topic_alive

        # lane 없으면 안전 정지 우선
        self.input_stale = not self.lane_valid

        # 3) fusion summary 계산
        self.fusion_mode = self.compute_fusion_mode()
        if self.lane_valid and not self.obstacle_valid:
            self.degraded_mode = 'LANE_ONLY'
        elif self.lane_valid and self.drivable_valid and not (self.obstacle_valid and self.speed_valid):
            self.degraded_mode = 'LANE_DRIVABLE'
        elif self.lane_valid and self.obstacle_valid and self.speed_valid and self.detection_valid and not self.detections_present:
            self.degraded_mode = 'CAMERA_ALIVE_NO_OBJECTS'
        elif self.lane_valid and self.obstacle_valid and self.speed_valid and not self.detection_valid:
            self.degraded_mode = 'LIDAR_TRACKING'
        elif self.lane_valid and self.obstacle_valid and self.speed_valid and self.detection_valid and self.detections_present:
            self.degraded_mode = 'FULL_FUSION'
        else:
            self.degraded_mode = 'FAILSAFE'

        # 4) lead/intrusion/lane_blocked/lc options 계산
        if self.obstacle_valid:
            self.obstacle_distance, self.obstacle_x, self.obstacle_y = self.compute_closest_relevant_obstacle()
            self.lead_vehicle_distance, self.lead_vehicle_x, self.lead_vehicle_y = self.select_forward_vehicle_on_path()
        else:
            self.obstacle_distance, self.obstacle_x, self.obstacle_y = 99.0, 99.0, 0.0
            self.lead_vehicle_distance, self.lead_vehicle_x, self.lead_vehicle_y = 99.0, 99.0, 0.0

        if self.enable_lane_change and self.drivable_valid:
            left_poss, right_poss, _, _ = self.evaluate_lane_change_options()
            self.left_lane_possible = left_poss
            self.right_lane_possible = right_poss
        else:
            self.left_lane_possible = False
            self.right_lane_possible = False

        # 5) next state 결정
        prev_state = self.current_state
        next_state, reason = self.decide_next_state(now, self.current_speed_mps)

        if next_state not in (BehaviorState.LANE_CHANGE_LEFT, BehaviorState.LANE_CHANGE_RIGHT):
            self.lane_change_start_time = None
        if next_state != BehaviorState.RETURN_TO_LANE:
            self.return_finish_count = 0

        if prev_state == BehaviorState.AVOID_OBSTACLE and next_state != BehaviorState.AVOID_OBSTACLE:
            self.reset_avoidance_smoothing()

        if next_state == BehaviorState.AVOID_OBSTACLE:
            smoothed_avoidance_y = self.update_avoidance_target_smoothing(self.avoidance_target_y)
            if not self.is_target_in_drivable_area(self.avoidance_target_x, smoothed_avoidance_y):
                self.reset_avoidance_smoothing()
                self.avoidance_active = False
                self.avoidance_candidate_count = 0
                self.avoidance_release_count_current = 0
                next_state = BehaviorState.STOP
                reason = 'lattice_avoidance_no_safe_target'

        self.current_state = next_state
        self.debug_reason = reason

        # 6) target point 계산
        if next_state in (BehaviorState.STOP, BehaviorState.EMERGENCY_STOP):
            target_x, target_y = 0.0, 0.0
        else:
            target_x, target_y, _ = self.compute_target_point_for_state(next_state, now)

        # 7) target smoothing
        alpha = max(0.0, min(1.0, self.target_smoothing_alpha))
        was_stop_state = prev_state in (BehaviorState.STOP, BehaviorState.EMERGENCY_STOP)
        is_stop_state = next_state in (BehaviorState.STOP, BehaviorState.EMERGENCY_STOP)

        if is_stop_state:
            self.filtered_target_x = 0.0
            self.filtered_target_y = 0.0
        elif was_stop_state:
            # Reinitialize target filter when leaving STOP to avoid sudden jump from stale 0 target.
            self.filtered_target_x = target_x
            self.filtered_target_y = target_y
        elif self.filtered_target_x is None or self.filtered_target_y is None:
            self.filtered_target_x = target_x
            self.filtered_target_y = target_y
        else:
            self.filtered_target_x = ((1.0 - alpha) * self.filtered_target_x) + (alpha * target_x)
            self.filtered_target_y = ((1.0 - alpha) * self.filtered_target_y) + (alpha * target_y)

        local_x = float(self.filtered_target_x)
        local_y = float(self.filtered_target_y)

        # 8) desired speed 계산
        if next_state in (BehaviorState.STOP, BehaviorState.EMERGENCY_STOP):
            raw_desired_speed = 0.0
            self.filtered_desired_speed = 0.0
            self.desired_speed = 0.0
            self.target_x = 0.0
            self.target_y = 0.0
        elif next_state == BehaviorState.AVOID_OBSTACLE:
            raw_desired_speed = self.avoidance_speed_mps
        elif next_state == BehaviorState.FOLLOW_VEHICLE:
            if self.speed_valid:
                raw_desired_speed = self.compute_follow_speed(self.lead_vehicle_x, self.lead_vehicle_distance, self.current_speed_mps)
            else:
                # speed 신뢰도 없으면 보수적으로 감속
                raw_desired_speed = min(1.0, self.compute_nominal_speed(local_y) * 0.7)
        elif next_state in (BehaviorState.PREPARE_LANE_CHANGE_LEFT, BehaviorState.PREPARE_LANE_CHANGE_RIGHT):
            raw_desired_speed = self.compute_nominal_speed(local_y) * 0.9
        elif next_state in (BehaviorState.LANE_CHANGE_LEFT, BehaviorState.LANE_CHANGE_RIGHT):
            raw_desired_speed = self.lane_change_speed_mps
        elif next_state == BehaviorState.RETURN_TO_LANE:
            raw_desired_speed = self.lane_change_speed_mps * 0.9
        else:
            raw_desired_speed = self.compute_nominal_speed(local_y)

        if reason == 'lane_blocked_creep':
            raw_desired_speed = min(raw_desired_speed, self.creep_speed_mps)

        raw_desired_speed = max(0.0, min(self.max_desired_speed_mps, raw_desired_speed))

        # 9) speed smoothing
        if next_state not in (BehaviorState.STOP, BehaviorState.EMERGENCY_STOP):
            speed_alpha = 0.10
            self.filtered_desired_speed = ((1.0 - speed_alpha) * self.filtered_desired_speed) + (speed_alpha * raw_desired_speed)
            self.filtered_desired_speed = max(0.0, min(self.max_desired_speed_mps, self.filtered_desired_speed))
            self.desired_speed = float(self.filtered_desired_speed)
            self.target_x = local_x
            self.target_y = local_y

        self.desired_speed = max(0.0, min(self.max_desired_speed_mps, self.desired_speed))

        if self.current_state.value != self.last_state_log or self.debug_reason != self.last_reason_log:
            self.last_state_log = self.current_state.value
            self.last_reason_log = self.debug_reason
            self.get_logger().info(f'state={self.current_state.value}, reason={self.debug_reason}')

        # 10) publish
        self.publish_all()

    # =========================
    # 판단 보조 함수
    # =========================
    def compute_closest_relevant_obstacle(self):
        if not self.obstacle_points:
            return 99.0, 99.0, 0.0

        relevant = []
        for x, y, z in self.obstacle_points:
            if x <= 0.0 or abs(y) > self.obstacle_corridor_half_width_m:
                continue
            if self.should_ignore_vehicle_for_signal_queue(x, y, z):
                continue
            relevant.append((x, y, z))

        if not relevant:
            return 99.0, 99.0, 0.0

        best_dist = float('inf')
        best_x = 99.0
        best_y = 0.0
        for x, y, z in relevant:
            dist = math.sqrt(x * x + y * y)
            if dist < best_dist:
                best_dist = dist
                best_x = x
                best_y = y

        return float(best_dist), float(best_x), float(best_y)

    # =========================
    # 발행
    # =========================
    def publish_all(self):
        speed_msg = Float64()
        speed_msg.data = float(self.desired_speed)
        self.desired_speed_pub.publish(speed_msg)

        target_msg = Point()
        target_msg.x = float(self.target_x)
        target_msg.y = float(self.target_y)
        target_msg.z = 0.0
        self.target_point_pub.publish(target_msg)

        state_msg = String()
        state_msg.data = self.current_state.value
        self.state_pub.publish(state_msg)

        debug_msg = String()
        debug_msg.data = (
            f'state={self.current_state.value} | '
            f'reason={self.debug_reason} | '
            f'desired_speed={self.desired_speed:.2f}mps | '
            f'target=({self.target_x:.2f},{self.target_y:.2f}) | '
            f'lane_valid={self.lane_valid} | '
            f'obstacle_valid={self.obstacle_valid} | '
            f'drivable_valid={self.drivable_valid} | '
            f'car_count={self.detection_class_summary.get("car_count",0)} | '
            f'pedestrian_count={self.detection_class_summary.get("pedestrian_count",0)} | '
            f'car_score_max={self.detection_class_summary.get("car_score_max",0.0):.2f} | '
            f'pedestrian_score_max={self.detection_class_summary.get("pedestrian_score_max",0.0):.2f} | '
            f'avoidance_enabled={self.enable_obstacle_avoidance} | '
            f'avoidance_method={self.avoidance_method} | '
            f'avoidance_active={self.avoidance_active} | '
            f'avoidance_target=({self.avoidance_target_x:.2f},{self.avoidance_target_y:.2f}) | '
            f'avoidance_score={self.avoidance_score:.3f} | '
            f'avoidance_candidate_count={self.avoidance_candidate_count} | '
            f'avoidance_release_count={self.avoidance_release_count_current} | '
            f'main_lane_blocked={self.main_lane_blocked} | '
            f'left_lane_possible={self.left_lane_possible} | '
            f'right_lane_possible={self.right_lane_possible} | '
            f'obstacle_distance={self.obstacle_distance:.2f}m | '
            f'speed_valid={self.speed_valid} | '
            f'detection_topic_alive={self.detection_topic_alive} | '
            f'detections_present={self.detections_present} | '
            f'degraded_mode={self.degraded_mode} | '
            f'fusion_mode={self.fusion_mode} | '
            f'near_obstacle_stop_distance_m={self.near_obstacle_stop_distance_m:.2f} | '
            f'emergency_stop_distance_m={self.emergency_stop_distance_m:.2f} | '
            f'lane_blocked_distance_m={float(self.lane_blocked_distance_m):.2f} | '
            f'require_drivable_for_lane_keeping={self.require_drivable_for_lane_keeping} | '
            f'lane_change_enabled={self.enable_lane_change} | '
            f'cutin_enabled={self.enable_cutin_detection} | '
            f'cutin_intrusion_detected={self.cutin_intrusion_detected} | '
            f'traffic_light_enabled={self.enable_traffic_light} | '
            f'tl_valid={self.tl_valid} | '
            f'traffic_light_state={self.traffic_light_state} | '
            f'stopline_mode={self.stopline_mode} | '
            f'max_desired_speed_mps={self.max_desired_speed_mps:.2f} | '
            f'speed_topic={self.speed_topic} | '
            f'left_target_y={self.left_target_y_debug:.2f} | '
            f'right_target_y={self.right_target_y_debug:.2f} | '
            f'lead_vehicle_distance={self.lead_vehicle_distance:.2f}m | '
            f'ped_intrusion={self.pedestrian_intrusion_severity} | '
            f'ped_warn={self.pedestrian_detection_warning}'
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
