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
        self.obstacle_x = 99.0  # 장애물 x 좌표
        self.obstacle_y = 0.0   # 장애물 y 좌표
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

        # 장애물 회피 로직
        if distance < 10.0:
            if self.current_state != BehaviorState.LANE_KEEPING:
                self.current_state = BehaviorState.LANE_KEEPING
                self.get_logger().warning(f'🚧 전방 {distance:.1f}m 이내 장애물 감지! 회피 주행 시작')
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
            self.obstacle_distance = 99.0
            self.obstacle_x = 99.0
            self.obstacle_y = 0.0
            self.update_state_from_obstacle(99.0)
            self.publish_obstacle_distance()
            return

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

        # 신호등이 RED면 정지
        if self.traffic_light_state.upper() == 'RED':
            pose_msg.pose.position.x = 0.0
            pose_msg.pose.position.y = 0.0
            self.get_logger().info('🛑 신호등 RED, 목표점 (0,0) 발행')
        # 장애물이 10m 이내면 회피 주행
        elif self.obstacle_distance < 10.0:
            # 장애물 위치에 따라 반대쪽으로 회피
            avoidance_offset = 1.5 if self.obstacle_y > 0 else -1.5  # 장애물 반대쪽으로 1.5m
            adjusted_y = target_y + avoidance_offset
            
            pose_msg.pose.position.x = float(target_x)
            pose_msg.pose.position.y = float(adjusted_y)
            self.get_logger().info(f'🚧 회피 주행 ➡️ X: {target_x:.1f}, Y: {adjusted_y:.1f} (장애물: x={self.obstacle_x:.1f}, y={self.obstacle_y:.1f})')
        # 정상 주행 (신호 GREEN & 장애물 없음)
        else:
            pose_msg.pose.position.x = float(target_x)
            pose_msg.pose.position.y = float(target_y)
            self.get_logger().info(f'🎯 목표점 낙점 ➡️ X: {target_x:.1f}, Y: {target_y:.1f}')

        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BehaviorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()