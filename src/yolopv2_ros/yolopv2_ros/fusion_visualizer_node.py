"""
FusionVisualizerNode

Subscribes to perception results from:
- YOLOPv2 (vehicle detection, drivable area mask, lane line mask)
- pedestrian_detector (YOLO person detection)
- traffic_light_detector (YOLO traffic light detection)

Fuses them into a single debug image and publishes fused detections.
No inference; only visualization and detection fusion.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import json
from time import time
from typing import Optional, Dict, List, Tuple


class FusionVisualizerNode(Node):
    def __init__(self):
        super().__init__('fusion_visualizer_node')

        # declare parameters
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('drivable_mask_topic', '/yolopv2/drivable_mask')
        self.declare_parameter('lane_mask_topic', '/yolopv2/lane_mask')
        self.declare_parameter('vehicle_detections_topic', '/yolopv2/vehicle_detections')
        self.declare_parameter('person_detections_topic', '/yolo/person/detections')
        self.declare_parameter('traffic_light_detections_topic', '/yolo/traffic_light/detections')
        self.declare_parameter('fused_image_topic', '/perception/fused_debug_image')
        self.declare_parameter('fused_detections_topic', '/perception/fused_detections')
        self.declare_parameter('queue_size', 10)
        self.declare_parameter('publish_fused_image', True)
        self.declare_parameter('publish_fused_detections', True)
        self.declare_parameter('drivable_alpha', 0.35)
        self.declare_parameter('lane_alpha', 0.75)
        self.declare_parameter('max_stale_time_sec', 0.5)
        self.declare_parameter('min_vehicle_confidence', 0.25)
        self.declare_parameter('min_person_confidence', 0.25)
        self.declare_parameter('min_traffic_light_confidence', 0.25)
        self.declare_parameter('vehicle_class_ids', [])  # empty = all classes
        self.declare_parameter('yolopv2_class_id_map_json', '{}')  # optional mapping as JSON string

        # get parameters
        image_topic = self.get_parameter('image_topic').value
        drivable_mask_topic = self.get_parameter('drivable_mask_topic').value
        lane_mask_topic = self.get_parameter('lane_mask_topic').value
        vehicle_detections_topic = self.get_parameter('vehicle_detections_topic').value
        person_detections_topic = self.get_parameter('person_detections_topic').value
        traffic_light_detections_topic = self.get_parameter('traffic_light_detections_topic').value
        fused_image_topic = self.get_parameter('fused_image_topic').value
        fused_detections_topic = self.get_parameter('fused_detections_topic').value
        queue_size = self.get_parameter('queue_size').value
        self.publish_fused_image = self.get_parameter('publish_fused_image').value
        self.publish_fused_detections = self.get_parameter('publish_fused_detections').value
        self.drivable_alpha = self.get_parameter('drivable_alpha').value
        self.lane_alpha = self.get_parameter('lane_alpha').value
        self.max_stale_time_sec = self.get_parameter('max_stale_time_sec').value
        self.min_vehicle_confidence = self.get_parameter('min_vehicle_confidence').value
        self.min_person_confidence = self.get_parameter('min_person_confidence').value
        self.min_traffic_light_confidence = self.get_parameter('min_traffic_light_confidence').value
        self.vehicle_class_ids = self.get_parameter('vehicle_class_ids').value
        yolopv2_class_id_map_json = self.get_parameter('yolopv2_class_id_map_json').value
        self.yolopv2_class_id_map = self._parse_class_id_map(yolopv2_class_id_map_json)

        # convert empty vehicle_class_ids to None (meaning all classes)
        if not self.vehicle_class_ids:
            self.vehicle_class_ids = None

        # CvBridge
        self.bridge = CvBridge()

        # latest messages cache with timestamps
        self.image_msg = None
        self.image_time = None
        self.drivable_mask_msg = None
        self.drivable_mask_time = None
        self.lane_mask_msg = None
        self.lane_mask_time = None
        self.vehicle_detections_msg = None
        self.vehicle_detections_time = None
        self.person_detections_msg = None
        self.person_detections_time = None
        self.traffic_light_detections_msg = None
        self.traffic_light_detections_time = None

        # statistics
        self._frame_count = 0

        # publishers
        self.fused_image_pub = self.create_publisher(Image, fused_image_topic, 10)
        self.fused_detections_pub = self.create_publisher(Detection2DArray, fused_detections_topic, 10)

        # subscribers
        self.image_sub = self.create_subscription(
            Image, image_topic, self._on_image, queue_size
        )
        self.drivable_mask_sub = self.create_subscription(
            Image, drivable_mask_topic, self._on_drivable_mask, queue_size
        )
        self.lane_mask_sub = self.create_subscription(
            Image, lane_mask_topic, self._on_lane_mask, queue_size
        )
        self.vehicle_detections_sub = self.create_subscription(
            Detection2DArray, vehicle_detections_topic, self._on_vehicle_detections, queue_size
        )
        self.person_detections_sub = self.create_subscription(
            Detection2DArray, person_detections_topic, self._on_person_detections, queue_size
        )
        self.traffic_light_detections_sub = self.create_subscription(
            Detection2DArray, traffic_light_detections_topic, self._on_traffic_light_detections, queue_size
        )

        self.get_logger().info(
            f'FusionVisualizerNode started. '
            f'image_topic={image_topic} '
            f'drivable_mask={drivable_mask_topic} lane_mask={lane_mask_topic} '
            f'vehicle_det={vehicle_detections_topic} '
            f'person_det={person_detections_topic} '
            f'traffic_light_det={traffic_light_detections_topic}'
        )

    def _parse_class_id_map(self, mapping_json: str) -> Dict[str, str]:
        """Parse optional class-id mapping from a JSON string parameter."""
        if not mapping_json:
            return {}
        try:
            parsed = json.loads(mapping_json)
            if isinstance(parsed, dict):
                return parsed
            self.get_logger().warn('yolopv2_class_id_map_json is not a JSON object. Ignoring it.')
            return {}
        except Exception as e:
            self.get_logger().warn(f'Failed to parse yolopv2_class_id_map_json: {e}. Using empty mapping.')
            return {}

    def _on_image(self, msg: Image):
        self.image_msg = msg
        self.image_time = time()
        try:
            self._process_frame()
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')

    def _on_drivable_mask(self, msg: Image):
        self.drivable_mask_msg = msg
        self.drivable_mask_time = time()

    def _on_lane_mask(self, msg: Image):
        self.lane_mask_msg = msg
        self.lane_mask_time = time()

    def _on_vehicle_detections(self, msg: Detection2DArray):
        self.vehicle_detections_msg = msg
        self.vehicle_detections_time = time()

    def _on_person_detections(self, msg: Detection2DArray):
        self.person_detections_msg = msg
        self.person_detections_time = time()

    def _on_traffic_light_detections(self, msg: Detection2DArray):
        self.traffic_light_detections_msg = msg
        self.traffic_light_detections_time = time()

    def _is_stale(self, msg_time: Optional[float]) -> bool:
        """Check if message is stale."""
        if msg_time is None:
            return True
        return (time() - msg_time) > self.max_stale_time_sec

    def _get_bbox_center_size(self, bbox: BoundingBox2D) -> Tuple[float, float, float, float]:
        """Extract center and size from BoundingBox2D, handling different schemas."""
        try:
            # Try schema 1: center.x, center.y
            cx = getattr(bbox.center, 'x', None)
            cy = getattr(bbox.center, 'y', None)
            if cx is not None and cy is not None:
                return float(cx), float(cy), float(bbox.size_x), float(bbox.size_y)
        except Exception:
            pass

        try:
            # Try schema 2: center.position.x, center.position.y
            cx = getattr(bbox.center.position, 'x', None)
            cy = getattr(bbox.center.position, 'y', None)
            if cx is not None and cy is not None:
                return float(cx), float(cy), float(bbox.size_x), float(bbox.size_y)
        except Exception:
            pass

        raise ValueError('Unsupported BoundingBox2D schema')

    def _get_detection_data(self, det: Detection2D) -> Tuple[float, float, str]:
        """Extract center_x, center_y, class_id, score from Detection2D."""
        score = 0.0
        class_id = 'unknown'

        try:
            if det.results and len(det.results) > 0:
                hyp = det.results[0].hypothesis
                score = float(hyp.score)
                class_id = str(hyp.class_id)
        except Exception:
            pass

        try:
            cx, cy, sx, sy = self._get_bbox_center_size(det.bbox)
            return cx, cy, sx, sy, class_id, score
        except Exception as e:
            self.get_logger().error(f'Failed to extract bbox: {e}')
            return 0.0, 0.0, 0.0, 0.0, class_id, score

    def _process_frame(self):
        """Main fusion processing triggered by image callback."""
        if self.image_msg is None:
            return

        self._frame_count += 1

        # Convert image to cv2
        try:
            frame = self.bridge.imgmsg_to_cv2(self.image_msg, desired_encoding='bgr8')
        except CvBridgeError:
            try:
                frame = self.bridge.imgmsg_to_cv2(self.image_msg, desired_encoding='passthrough')
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            except Exception as e:
                self.get_logger().error(f'Failed to convert image: {e}')
                return

        h, w = frame.shape[:2]

        # Start with frame copy
        fused_image = frame.copy()

        # Apply masks
        self._apply_masks(fused_image, w, h)

        # Draw detections
        vehicle_count, person_count, traffic_light_count = self._draw_detections(fused_image)

        # Draw debug text
        self._draw_debug_text(fused_image, w, h, vehicle_count, person_count, traffic_light_count)

        # Publish fused image
        if self.publish_fused_image:
            try:
                out_msg = self.bridge.cv2_to_imgmsg(fused_image, encoding='bgr8')
                out_msg.header = self.image_msg.header
                self.fused_image_pub.publish(out_msg)
            except CvBridgeError as e:
                self.get_logger().error(f'Failed to publish fused image: {e}')

        # Publish fused detections
        if self.publish_fused_detections:
            fused_dets = self._create_fused_detections()
            self.fused_detections_pub.publish(fused_dets)

        # Log stats every 30 frames
        if self._frame_count % 30 == 0:
            self.get_logger().info(
                f'Processed frames={self._frame_count} '
                f'vehicles={vehicle_count} persons={person_count} traffic_lights={traffic_light_count}'
            )

    def _apply_masks(self, fused_image: np.ndarray, w: int, h: int):
        """Apply drivable area and lane line masks to fused image."""
        # Drivable mask
        if self.drivable_mask_msg is not None and not self._is_stale(self.drivable_mask_time):
            try:
                drivable_mask = self.bridge.imgmsg_to_cv2(self.drivable_mask_msg, desired_encoding='mono8')
                if drivable_mask.shape != (h, w):
                    drivable_mask = cv2.resize(drivable_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                # Create BGR overlay (cyan-ish for drivable area)
                drivable_overlay = np.zeros_like(fused_image)
                drivable_overlay[:, :] = [0, 255, 255]  # Yellow in BGR
                mask_bin = drivable_mask > 128
                fused_image[mask_bin] = cv2.addWeighted(
                    fused_image[mask_bin], 1.0 - self.drivable_alpha,
                    drivable_overlay[mask_bin], self.drivable_alpha,
                    0
                )
            except Exception as e:
                self.get_logger().error(f'Failed to apply drivable mask: {e}')

        # Lane mask
        if self.lane_mask_msg is not None and not self._is_stale(self.lane_mask_time):
            try:
                lane_mask = self.bridge.imgmsg_to_cv2(self.lane_mask_msg, desired_encoding='mono8')
                if lane_mask.shape != (h, w):
                    lane_mask = cv2.resize(lane_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                # Create BGR overlay (magenta-ish for lane line)
                lane_overlay = np.zeros_like(fused_image)
                lane_overlay[:, :] = [255, 0, 255]  # Magenta in BGR
                mask_bin = lane_mask > 128
                fused_image[mask_bin] = cv2.addWeighted(
                    fused_image[mask_bin], 1.0 - self.lane_alpha,
                    lane_overlay[mask_bin], self.lane_alpha,
                    0
                )
            except Exception as e:
                self.get_logger().error(f'Failed to apply lane mask: {e}')

    def _draw_detections(self, fused_image: np.ndarray) -> Tuple[int, int, int]:
        """Draw vehicle, person, and traffic light detections."""
        vehicle_count = 0
        person_count = 0
        traffic_light_count = 0

        # Draw vehicle detections
        if self.vehicle_detections_msg is not None and not self._is_stale(self.vehicle_detections_time):
            for det in self.vehicle_detections_msg.detections:
                try:
                    cx, cy, sx, sy, class_id, score = self._get_detection_data(det)
                    if score < self.min_vehicle_confidence:
                        continue

                    # Filter by vehicle_class_ids if specified
                    if self.vehicle_class_ids is not None:
                        try:
                            class_id_int = int(class_id)
                            if class_id_int not in self.vehicle_class_ids:
                                continue
                        except ValueError:
                            pass

                    x1 = int(cx - sx / 2)
                    y1 = int(cy - sy / 2)
                    x2 = int(cx + sx / 2)
                    y2 = int(cy + sy / 2)

                    # Map class_id to name if available
                    class_name = class_id
                    if isinstance(self.yolopv2_class_id_map, dict) and class_id in self.yolopv2_class_id_map:
                        class_name = self.yolopv2_class_id_map[class_id]
                    else:
                        try:
                            class_id_int = int(class_id)
                            if isinstance(self.yolopv2_class_id_map, dict) and str(class_id_int) in self.yolopv2_class_id_map:
                                class_name = self.yolopv2_class_id_map[str(class_id_int)]
                            else:
                                class_name = f'yolopv2_cls_{class_id}'
                        except ValueError:
                            class_name = f'yolopv2_cls_{class_id}'

                    label = f'{class_name} {score:.2f}'
                    cv2.rectangle(fused_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(fused_image, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    vehicle_count += 1
                except Exception as e:
                    self.get_logger().error(f'Failed to draw vehicle det: {e}')

        # Draw person detections
        if self.person_detections_msg is not None and not self._is_stale(self.person_detections_time):
            for det in self.person_detections_msg.detections:
                try:
                    cx, cy, sx, sy, class_id, score = self._get_detection_data(det)
                    if score < self.min_person_confidence:
                        continue

                    x1 = int(cx - sx / 2)
                    y1 = int(cy - sy / 2)
                    x2 = int(cx + sx / 2)
                    y2 = int(cy + sy / 2)

                    label = f'person {score:.2f}'
                    cv2.rectangle(fused_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(fused_image, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    person_count += 1
                except Exception as e:
                    self.get_logger().error(f'Failed to draw person det: {e}')

        # Draw traffic light detections
        if self.traffic_light_detections_msg is not None and not self._is_stale(self.traffic_light_detections_time):
            for det in self.traffic_light_detections_msg.detections:
                try:
                    cx, cy, sx, sy, class_id, score = self._get_detection_data(det)
                    if score < self.min_traffic_light_confidence:
                        continue

                    x1 = int(cx - sx / 2)
                    y1 = int(cy - sy / 2)
                    x2 = int(cx + sx / 2)
                    y2 = int(cy + sy / 2)

                    label = f'traffic light {score:.2f}'
                    cv2.rectangle(fused_image, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(fused_image, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    traffic_light_count += 1
                except Exception as e:
                    self.get_logger().error(f'Failed to draw traffic light det: {e}')

        return vehicle_count, person_count, traffic_light_count

    def _draw_debug_text(
        self, fused_image: np.ndarray, w: int, h: int,
        vehicle_count: int, person_count: int, traffic_light_count: int
    ):
        """Draw debug information on fused image."""
        y_offset = 30
        line_height = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (200, 200, 200)  # Light gray

        # Line 1: Detection counts
        text1 = f'vehicles: {vehicle_count} | persons: {person_count} | lights: {traffic_light_count}'
        cv2.putText(fused_image, text1, (10, y_offset), font, font_scale, color, 1)

        # Line 2: Mask status
        drivable_status = 'ok' if (self.drivable_mask_msg and not self._is_stale(self.drivable_mask_time)) else 'missing/stale'
        lane_status = 'ok' if (self.lane_mask_msg and not self._is_stale(self.lane_mask_time)) else 'missing/stale'
        text2 = f'drivable: {drivable_status} | lane: {lane_status}'
        cv2.putText(fused_image, text2, (10, y_offset + line_height), font, font_scale, color, 1)

        # Line 3: Resolution
        text3 = f'resolution: {w}x{h}'
        cv2.putText(fused_image, text3, (10, y_offset + 2 * line_height), font, font_scale, color, 1)

    def _create_fused_detections(self) -> Detection2DArray:
        """Create fused detection array from all sources."""
        fused = Detection2DArray()
        fused.header = self.image_msg.header

        # Collect all detections with filtering
        all_detections: List[Tuple[Detection2D, str]] = []

        # Vehicle detections
        if self.vehicle_detections_msg is not None and not self._is_stale(self.vehicle_detections_time):
            for det in self.vehicle_detections_msg.detections:
                try:
                    cx, cy, sx, sy, class_id, score = self._get_detection_data(det)
                    if score < self.min_vehicle_confidence:
                        continue

                    # Filter by vehicle_class_ids if specified
                    if self.vehicle_class_ids is not None:
                        try:
                            class_id_int = int(class_id)
                            if class_id_int not in self.vehicle_class_ids:
                                continue
                        except ValueError:
                            pass

                    # Normalize class_id
                    if isinstance(self.yolopv2_class_id_map, dict) and class_id in self.yolopv2_class_id_map:
                        norm_class_id = self.yolopv2_class_id_map[class_id]
                    else:
                        try:
                            class_id_int = int(class_id)
                            if isinstance(self.yolopv2_class_id_map, dict) and str(class_id_int) in self.yolopv2_class_id_map:
                                norm_class_id = self.yolopv2_class_id_map[str(class_id_int)]
                            else:
                                norm_class_id = f'yolopv2_cls_{class_id}'
                        except ValueError:
                            norm_class_id = f'yolopv2_cls_{class_id}'

                    # Copy detection and update header + class_id
                    det_copy = Detection2D()
                    det_copy.header = self.image_msg.header
                    det_copy.bbox = det.bbox
                    if det.results and len(det.results) > 0:
                        hyp_copy = ObjectHypothesisWithPose()
                        try:
                            hyp_copy.hypothesis.class_id = norm_class_id
                            hyp_copy.hypothesis.score = score
                        except Exception:
                            try:
                                hyp_copy.class_id = norm_class_id
                                hyp_copy.score = score
                            except Exception:
                                pass
                        det_copy.results = [hyp_copy]

                    all_detections.append((det_copy, 'vehicle'))
                except Exception as e:
                    self.get_logger().error(f'Failed to add vehicle detection to fused: {e}')

        # Person detections
        if self.person_detections_msg is not None and not self._is_stale(self.person_detections_time):
            for det in self.person_detections_msg.detections:
                try:
                    cx, cy, sx, sy, class_id, score = self._get_detection_data(det)
                    if score < self.min_person_confidence:
                        continue

                    det_copy = Detection2D()
                    det_copy.header = self.image_msg.header
                    det_copy.bbox = det.bbox
                    if det.results and len(det.results) > 0:
                        hyp_copy = ObjectHypothesisWithPose()
                        try:
                            hyp_copy.hypothesis.class_id = 'person'
                            hyp_copy.hypothesis.score = score
                        except Exception:
                            try:
                                hyp_copy.class_id = 'person'
                                hyp_copy.score = score
                            except Exception:
                                pass
                        det_copy.results = [hyp_copy]

                    all_detections.append((det_copy, 'person'))
                except Exception as e:
                    self.get_logger().error(f'Failed to add person detection to fused: {e}')

        # Traffic light detections
        if self.traffic_light_detections_msg is not None and not self._is_stale(self.traffic_light_detections_time):
            for det in self.traffic_light_detections_msg.detections:
                try:
                    cx, cy, sx, sy, class_id, score = self._get_detection_data(det)
                    if score < self.min_traffic_light_confidence:
                        continue

                    det_copy = Detection2D()
                    det_copy.header = self.image_msg.header
                    det_copy.bbox = det.bbox
                    if det.results and len(det.results) > 0:
                        hyp_copy = ObjectHypothesisWithPose()
                        try:
                            hyp_copy.hypothesis.class_id = 'traffic light'
                            hyp_copy.hypothesis.score = score
                        except Exception:
                            try:
                                hyp_copy.class_id = 'traffic light'
                                hyp_copy.score = score
                            except Exception:
                                pass
                        det_copy.results = [hyp_copy]

                    all_detections.append((det_copy, 'traffic_light'))
                except Exception as e:
                    self.get_logger().error(f'Failed to add traffic light detection to fused: {e}')

        fused.detections = [det for det, _ in all_detections]
        return fused


def main(args=None):
    rclpy.init(args=args)
    node = FusionVisualizerNode()
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
