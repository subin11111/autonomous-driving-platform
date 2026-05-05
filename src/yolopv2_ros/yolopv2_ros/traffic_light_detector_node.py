"""
TrafficLightDetectorNode

This node detects only the traffic light object bounding box. It does not classify red/yellow/green signal state.
Traffic light state classification (red/yellow/green) can be added later by cropping detections and running HSV or a classifier node.
"""

import rclpy
from .yolo_detector_base import YoloDetectorBase


class TrafficLightDetectorNode(YoloDetectorBase):
    def __init__(self):
        super().__init__(
            node_name='traffic_light_detector_node',
            target_class_ids=[9],
            target_class_names={9: 'traffic light'},
            default_detection_topic='/yolo/traffic_light/detections',
            default_result_image_topic='/yolo/traffic_light/result_image',
        )


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetectorNode()
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
