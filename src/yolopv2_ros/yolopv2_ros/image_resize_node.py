"""
ImageResizeNode

Subscribes to /camera/image_raw and resizes to specified dimensions (default 1280x720).
Publishes resized image to /camera/image_1280x720 (or custom topic).

Purpose:
- Ensure all perception nodes (YOLOPv2, pedestrian_detector, traffic_light_detector, fusion_visualizer)
  operate on the same input resolution for proper bbox/mask alignment in fusion.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from time import time


class ImageResizeNode(Node):
    def __init__(self):
        super().__init__('image_resize_node')

        # Declare parameters
        self.declare_parameter('input_image_topic', '/camera/image_raw')
        self.declare_parameter('output_image_topic', '/camera/image_1280x720')
        self.declare_parameter('output_width', 1280)
        self.declare_parameter('output_height', 720)
        self.declare_parameter('queue_size', 10)
        self.declare_parameter('publish_even_if_same_size', True)

        # Get parameters
        self.input_image_topic = self.get_parameter('input_image_topic').value
        self.output_image_topic = self.get_parameter('output_image_topic').value
        self.output_width = self.get_parameter('output_width').value
        self.output_height = self.get_parameter('output_height').value
        queue_size = self.get_parameter('queue_size').value
        self.publish_even_if_same_size = self.get_parameter('publish_even_if_same_size').value

        # CvBridge
        self.bridge = CvBridge()

        # Statistics
        self._frame_count = 0

        # Publisher
        self.image_pub = self.create_publisher(Image, self.output_image_topic, 10)

        # Subscriber
        self.image_sub = self.create_subscription(
            Image, self.input_image_topic, self._on_image, queue_size
        )

        self.get_logger().info(
            f'ImageResizeNode initialized. '
            f'input={self.input_image_topic} output={self.output_image_topic} '
            f'target_size={self.output_width}x{self.output_height}'
        )

    def _on_image(self, msg: Image):
        try:
            self._frame_count += 1

            # Convert to cv2 image
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except CvBridgeError:
                # Try passthrough
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                if frame is None or len(frame.shape) == 0:
                    raise
                # Handle grayscale
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                # Handle BGRA
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            h, w = frame.shape[:2]

            # Check if resize is needed
            if h != self.output_height or w != self.output_width:
                frame = cv2.resize(frame, (self.output_width, self.output_height), interpolation=cv2.INTER_LINEAR)
            elif not self.publish_even_if_same_size:
                # Skip publishing if same size and publish_even_if_same_size is false
                return

            # Convert back to Image message
            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            out_msg.header = msg.header
            self.image_pub.publish(out_msg)

            # Log stats every 30 frames
            if self._frame_count % 30 == 0:
                self.get_logger().info(
                    f'ImageResizeNode: processed={self._frame_count} frames, '
                    f'input_size={w}x{h}, output_size={self.output_width}x{self.output_height}'
                )

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ImageResizeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
