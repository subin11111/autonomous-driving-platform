#!/usr/bin/env python3
import rclpy, struct, time, math
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def main():
    rclpy.init()
    node = Node('fake_lane_pub')
    pub = node.create_publisher(PointCloud2, '/perception/real_world_lane_points', 10)

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    
    point_step = 16  # 4 bytes per float * 3 floats + padding
    
    try:
        while rclpy.ok():
            header = Header()
            header.frame_id = 'ego_vehicle'
            header.stamp = node.get_clock().now().to_msg()
            
            # 곡선 차선 포인트 생성 (10개 점, 앞쪽 10m까지)
            lane_points = []
            for i in range(1, 11):
                x = float(i)
                # 부드러운 우측 곡선 (음수 부호로 방향 반대)
                y = -math.sin(x * 0.3) * 0.5
                z = 0.0
                lane_points.append((x, y, z))
            
            # 여러 점을 바이너리로 변환
            point_bytes = b''
            for x, y, z in lane_points:
                point_bytes += struct.pack('<fffxxxx', x, y, z)
            
            num_points = len(lane_points)
            
            msg = PointCloud2(
                header=header,
                height=1,
                width=num_points,  # 점의 개수
                fields=fields,
                is_bigendian=False,
                point_step=point_step,
                row_step=point_step * num_points,  # 전체 데이터 크기
                data=point_bytes,
                is_dense=True,
            )
            pub.publish(msg)
            node.get_logger().info(f'🛣️ 곡선 차선 발행: {num_points}개 점')
            
            rclpy.spin_once(node, timeout_sec=0.01)
            time.sleep(0.1)  # 10Hz
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
