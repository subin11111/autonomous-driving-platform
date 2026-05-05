import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None


class YoloDetectorBase(Node):
    def __init__(
        self,
        node_name: str,
        target_class_ids,
        target_class_names,
        default_detection_topic: str,
        default_result_image_topic: str,
    ):
        super().__init__(node_name)

        self.target_class_ids = target_class_ids
        self.target_class_names = target_class_names
        self.default_detection_topic = default_detection_topic
        self.default_result_image_topic = default_result_image_topic

        # declare parameters
        self.declare_parameter('input_image_topic', '/camera/image_raw')
        self.declare_parameter('detection_topic', self.default_detection_topic)
        self.declare_parameter('result_image_topic', self.default_result_image_topic)
        self.declare_parameter('model_name', 'yolov8n.pt')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('queue_size', 10)
        self.declare_parameter('publish_result_image', True)

        # read parameters
        self.input_image_topic = self.get_parameter('input_image_topic').get_parameter_value().string_value
        self.detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        self.result_image_topic = self.get_parameter('result_image_topic').get_parameter_value().string_value
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.iou_threshold = self.get_parameter('iou_threshold').get_parameter_value().double_value
        self.queue_size = self.get_parameter('queue_size').get_parameter_value().integer_value
        self.publish_result_image = self.get_parameter('publish_result_image').get_parameter_value().bool_value

        # cv bridge
        self.bridge = CvBridge()

        # frame counters
        self._frame_count = 0
        self._detections_count = 0

        # load model
        if YOLO is None:
            self.get_logger().error(
                'Failed to import ultralytics YOLO. Install with: python3 -m pip install ultralytics'
            )
            raise RuntimeError('ultralytics package required')

        try:
            self.model = YOLO(self.model_name)
        except Exception as e:
            self.get_logger().error(f'Failed to load model {self.model_name}: {e}')
            raise

        # publishers and subscribers
        self.detection_pub = self.create_publisher(Detection2DArray, self.detection_topic, 10)
        self.image_pub = self.create_publisher(Image, self.result_image_topic, 10)

        self.sub = self.create_subscription(
            Image,
            self.input_image_topic,
            self._image_callback,
            self.queue_size,
        )

        self.get_logger().info(
            f'Node {self.get_name()} started. model={self.model_name} device={self.device} '
            f'input_topic={self.input_image_topic} detection_topic={self.detection_topic} '
            f'result_image_topic={self.result_image_topic} target_class_ids={self.target_class_ids} '
            f'target_class_names={self.target_class_names} conf={self.confidence_threshold} iou={self.iou_threshold}'
        )

    def _set_bbox_center(self, bbox: BoundingBox2D, cx: float, cy: float, size_x: float, size_y: float):
        # handle different vision_msgs versions for BoundingBox2D.center
        # try bbox.center.x / bbox.center.y
        try:
            setattr(bbox.center, 'x', float(cx))
            setattr(bbox.center, 'y', float(cy))
            bbox.size_x = float(size_x)
            bbox.size_y = float(size_y)
            return
        except Exception:
            pass

        # try bbox.center.position.x / position.y
        try:
            setattr(bbox.center.position, 'x', float(cx))
            setattr(bbox.center.position, 'y', float(cy))
            bbox.size_x = float(size_x)
            bbox.size_y = float(size_y)
            return
        except Exception:
            pass

        raise AttributeError('Unsupported BoundingBox2D.center schema')

    def _image_callback(self, img_msg: Image):
        try:
            self._frame_count += 1

            # convert to cv image
            try:
                frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            except CvBridgeError:
                # try passthrough
                frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
                # handle possible gray or bgra
                if frame is None:
                    raise
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # run inference
            results = self.model.predict(
                frame,
                classes=self.target_class_ids,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

            # prepare detection message
            det_array = Detection2DArray()
            det_array.header = img_msg.header

            detections = []

            # ultralytics returns iterable results
            if results is not None and len(results) > 0:
                r = results[0]
                boxes = getattr(r, 'boxes', None)
                if boxes is not None:
                    xyxy = getattr(boxes, 'xyxy', None)
                    confs = getattr(boxes, 'conf', None)
                    cls = getattr(boxes, 'cls', None)

                    if xyxy is not None:
                        for i in range(len(xyxy)):
                            x1, y1, x2, y2 = [float(x) for x in xyxy[i]]
                            cx = (x1 + x2) / 2.0
                            cy = (y1 + y2) / 2.0
                            sx = x2 - x1
                            sy = y2 - y1
                            score = float(confs[i]) if confs is not None else 0.0
                            class_idx = int(cls[i]) if cls is not None else -1
                            class_name = self.target_class_names.get(class_idx, str(class_idx))

                            det = Detection2D()
                            det.header = img_msg.header
                            bbox = BoundingBox2D()
                            self._set_bbox_center(bbox, cx, cy, sx, sy)
                            det.bbox = bbox

                            hyp = ObjectHypothesisWithPose()
                            # hypothesis has .hypothesis.class_id and .hypothesis.score in some versions
                            try:
                                hyp.hypothesis.class_id = class_name
                                hyp.hypothesis.score = float(score)
                            except Exception:
                                # fallback: try direct attributes
                                try:
                                    hyp.class_id = class_name
                                    hyp.score = float(score)
                                except Exception:
                                    pass

                            det.results = [hyp]
                            detections.append((det, (int(x1), int(y1), int(x2), int(y2)), class_name, score))

            # package detections
            det_array.detections = [d[0] for d in detections]
            self.detection_pub.publish(det_array)

            # draw and publish result image
            if self.publish_result_image:
                vis = frame.copy()
                for det, (x1, y1, x2, y2), class_name, score in detections:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {score:.2f}"
                    cv2.putText(vis, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                try:
                    out_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
                    out_msg.header = img_msg.header
                    self.image_pub.publish(out_msg)
                except CvBridgeError as e:
                    self.get_logger().error(f'Failed to convert/publish result image: {e}')

            # logging
            self._detections_count += len(detections)
            if self._frame_count % 30 == 0:
                self.get_logger().info(f'Processed frames={self._frame_count} total_detections={self._detections_count}')

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')
