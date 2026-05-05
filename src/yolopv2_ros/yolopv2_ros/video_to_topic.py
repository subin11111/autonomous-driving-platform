import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time


class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_to_topic_node')

        # declare parameters
        self.declare_parameter('video_path', '/home/subin/test.mp4')
        self.declare_parameter('output_topic', '/camera/image_raw')
        self.declare_parameter('loop', True)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('frame_id', 'camera')
        self.declare_parameter('publish_width', 0)
        self.declare_parameter('publish_height', 0)
        self.declare_parameter('log_every_n_frames', 30)

        video_path = self.get_parameter('video_path').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        requested_fps = float(self.get_parameter('fps').get_parameter_value().double_value)
        self.loop = self.get_parameter('loop').get_parameter_value().bool_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.publish_width = int(self.get_parameter('publish_width').get_parameter_value().integer_value)
        self.publish_height = int(self.get_parameter('publish_height').get_parameter_value().integer_value)
        self.log_every_n_frames = int(self.get_parameter('log_every_n_frames').get_parameter_value().integer_value)

        self.publisher_ = self.create_publisher(Image, output_topic, 10)
        self.bridge = CvBridge()

        # open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f'Video open failed: {video_path}')
            raise RuntimeError(f'Video open failed: {video_path}')

        source_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps and requested_fps <= 0.0:
            self.fps = source_fps
        elif requested_fps > 0.0:
            self.fps = requested_fps
        else:
            self.fps = 30.0

        self.frame_count = 0
        self.start_time = time.time()

        timer_period = 1.0 / max(1.0, float(self.fps))
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(f'Video publisher initialized. video_path={video_path} output_topic={output_topic} fps={self.fps} loop={self.loop}')

    def timer_callback(self):
        try:
            ret, frame = self.cap.read()

            if not ret:
                if not self.loop:
                    self.get_logger().info('Video ended; stopping publish (loop=false).')
                    self.timer.cancel()
                    return

                # rewind
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    self.get_logger().warning('Failed to restart video loop; will retry next tick.')
                    return

            # optional resize
            in_h, in_w = frame.shape[:2]
            out_w, out_h = self.publish_width, self.publish_height
            if out_w > 0 and out_h > 0 and (out_w != in_w or out_h != in_h):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id

            self.publisher_.publish(msg)

            self.frame_count += 1
            if self.log_every_n_frames > 0 and self.frame_count % self.log_every_n_frames == 0:
                elapsed = max(time.time() - self.start_time, 1e-6)
                self.get_logger().info(
                    f'published frames: {self.frame_count}, avg_fps: {self.frame_count / elapsed:.2f}, output_topic={self.publisher_.topic_name}'
                )
        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {e}')

    def destroy_node(self):
        try:
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            super().destroy_node()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()