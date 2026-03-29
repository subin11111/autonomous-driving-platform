import rclpy
import math
from enum import Enum
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64, String
import sensor_msgs_py.point_cloud2 as pc2

class BehaviorState(Enum):
    LANE_KEEPING = 'LANE_KEEPING'
    STOP = 'STOP'

class BehaviorNode(Node):
    def __init__(self):
        super().__init__('behavior_node')

        # 구독: perception
        self.lane_sub = self.create_subscription(
            PointCloud2, '/perception/real_world_lane_points', self.lane_callback, 10)
        self.obs_sub = self.create_subscription(
            PointCloud2, '/perception/closest_obstacle', self.obstacle_callback, 10)
        self.tl_sub = self.create_subscription(
            String, '/traffic_light_state', self.traffic_light_callback, 10)

        # 발행: 제어계
        self.pose_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.obstacle_pub = self.create_publisher(Float64, '/obstacle_distance', 10)
        self.state_pub = self.create_publisher(String, '/behavior_state', 10)

        # 상태관리
        self.current_state = BehaviorState.LANE_KEEPING
        self.obstacle_distance = 99.0
        self.traffic_light_state = 'GREEN'
        self.lookahead_distance = 5.0  # 5m 선회 목표

        self.get_logger().info('🧠 행동 판단 노드 가동 (초기 상태: LANE_KEEPING)')

    def find_lookahead_point(self, lane_points_list):
        for x, y in lane_points_list:
            distance = math.sqrt(float(x)**2 + float(y)**2)
            if distance >= self.lookahead_distance:
                return float(x), float(y)

        if lane_points_list:
            x, y = lane_points_list[-1]
            return float(x), float(y)

        return 0.0, 0.0

    def update_state_from_obstacle(self, distance):
        self.obstacle_distance = distance

        if self.traffic_light_state.upper() == 'RED':
            if self.current_state != BehaviorState.STOP:
                self.current_state = BehaviorState.STOP
                self.get_logger().warning('🚦 신호등 RED 감지! 상태 변경 ➡️ [STOP]')
                self.publish_state()
            return

        if distance < 10.0:
            if self.current_state != BehaviorState.STOP:
                self.current_state = BehaviorState.STOP
                self.get_logger().warning('⚠️ 전방 10m 이내 장애물 감지! 상태 변경 ➡️ [STOP]')
                self.publish_state()
        else:
            if self.current_state != BehaviorState.LANE_KEEPING:
                self.current_state = BehaviorState.LANE_KEEPING
                self.get_logger().info('✅ 전방 안전 확보. 상태 변경 ➡️ [LANE_KEEPING]')
                self.publish_state()

    def publish_obstacle_distance(self):
        msg = Float64()
        msg.data = self.obstacle_distance
        self.obstacle_pub.publish(msg)

    def publish_state(self):
        msg = String()
        msg.data = self.current_state.value
        self.state_pub.publish(msg)

    def obstacle_callback(self, msg):
        points = list(pc2.read_points(msg, skip_nans=True))

        if not points:
            self.update_state_from_obstacle(99.0)
            self.publish_obstacle_distance()
            return

        min_distance = min(
            (math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in points),
            default=99.0,
        )

        self.update_state_from_obstacle(min_distance)
        self.publish_obstacle_distance()

    def traffic_light_callback(self, msg):
        sign = msg.data.strip().upper()
        if sign not in ['RED', 'YELLOW', 'GREEN']:
            self.get_logger().warning(f'🚦 알 수 없는 신호등 상태 수신: {msg.data} (무시)')
            return

        self.traffic_light_state = sign
        self.get_logger().info(f'🚦 신호등 상태 수신: {self.traffic_light_state}')

        if self.traffic_light_state == 'RED':
            self.current_state = BehaviorState.STOP
            self.get_logger().warning('🚦 RED -> 즉시 STOP')
        elif self.traffic_light_state == 'GREEN':
            # 녹색일 때는 장애물 상태에 따라 재평가
            self.update_state_from_obstacle(self.obstacle_distance)
        elif self.traffic_light_state == 'YELLOW':
            # 노란불은 감속을 위해 정지 결정 영역에 가깝게 로직 적용
            if self.current_state != BehaviorState.STOP:
                self.current_state = BehaviorState.STOP
                self.get_logger().info('🚦 YELLOW -> 우선 STOP 대기')


    def lane_callback(self, msg):
        points = list(pc2.read_points(msg, skip_nans=True))

        if not points:
            self.get_logger().warning('⚠️ 차선 포인트 클라우드가 비어있습니다.')
            return

        lane_points_list = [(p[0], p[1]) for p in points]
        target_x, target_y = self.find_lookahead_point(lane_points_list)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'ego_vehicle'

        if self.current_state == BehaviorState.LANE_KEEPING:
            pose_msg.pose.position.x = float(target_x)
            pose_msg.pose.position.y = float(target_y)
            self.get_logger().info(f'🎯 목표점 낙점 ➡️ X: {target_x:.1f}, Y: {target_y:.1f}')
        else:
            pose_msg.pose.position.x = 0.0
            pose_msg.pose.position.y = 0.0
            self.get_logger().info('🛑 현재 상태 STOP, 목표점 (0,0) 발행')

        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BehaviorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()