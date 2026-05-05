import rclpy
from .yolo_detector_base import YoloDetectorBase


class PedestrianDetectorNode(YoloDetectorBase):
    def __init__(self):
        super().__init__(
            node_name='pedestrian_detector_node',
            target_class_ids=[0],
            target_class_names={0: 'person'},
            default_detection_topic='/yolo/person/detections',
            default_result_image_topic='/yolo/person/result_image',
        )


def main(args=None):
    rclpy.init(args=args)
    node = PedestrianDetectorNode()
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
