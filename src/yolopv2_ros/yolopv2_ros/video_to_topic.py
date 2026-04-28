import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time


class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')

        self.declare_parameter('video_path', '/home/subin/test.mp4')
        self.declare_parameter('topic_name', '/camera/image_raw')
        self.declare_parameter('fps', 20.0)
        self.declare_parameter('loop', True)
        self.declare_parameter('display', False)
        self.declare_parameter('log_every_n_frames', 60)

        video_path = self.get_parameter('video_path').get_parameter_value().string_value
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        requested_fps = self.get_parameter('fps').get_parameter_value().double_value
        self.loop = self.get_parameter('loop').get_parameter_value().bool_value
        self.display = self.get_parameter('display').get_parameter_value().bool_value
        self.log_every_n_frames = int(self.get_parameter('log_every_n_frames').get_parameter_value().integer_value)

        self.publisher_ = self.create_publisher(Image, topic_name, 10)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f'비디오 열기 실패: {video_path}')
            raise RuntimeError(f'비디오 열기 실패: {video_path}')

        source_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if requested_fps > 0.0:
            self.fps = requested_fps
        elif source_fps > 0.0:
            self.fps = source_fps
        else:
            self.fps = 20.0

        self.frame_count = 0
        self.start_time = time.time()
        self.window_name = 'video_to_topic_preview'

        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(f'비디오 퍼블리시 시작')
        self.get_logger().info(f'video_path: {video_path}')
        self.get_logger().info(f'topic_name: {topic_name}')
        self.get_logger().info(f'fps(requested): {requested_fps}')
        self.get_logger().info(f'fps(source): {source_fps:.3f}')
        self.get_logger().info(f'fps(publish): {self.fps:.3f}')
        self.get_logger().info(f'loop: {self.loop}, display: {self.display}')

    def timer_callback(self):
        ret, frame = self.cap.read()

        if not ret:
            if not self.loop:
                self.get_logger().info('영상 끝. 퍼블리시를 중단합니다.')
                self.timer.cancel()
                return

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warning('영상 루프 재시작 실패. 다음 주기에서 재시도합니다.')
                return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'

        self.publisher_.publish(msg)

        if self.display:
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)

        self.frame_count += 1
        if self.log_every_n_frames > 0 and self.frame_count % self.log_every_n_frames == 0:
            elapsed = max(time.time() - self.start_time, 1e-6)
            self.get_logger().info(
                f'published frames: {self.frame_count}, avg_fps: {self.frame_count / elapsed:.2f}'
            )

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if getattr(self, 'display', False):
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()