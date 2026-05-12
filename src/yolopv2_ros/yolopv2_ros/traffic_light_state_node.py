import math
from time import monotonic

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray

try:
    import cv2
    import numpy as np
    from cv_bridge import CvBridge, CvBridgeError
except Exception:
    cv2 = None
    np = None
    CvBridge = None
    CvBridgeError = Exception


VALID_STATES = {'RED', 'YELLOW', 'GREEN', 'UNKNOWN'}


class TrafficLightStateNode(Node):
    """Convert traffic light Detection2DArray output into behavior String state."""

    def __init__(self):
        super().__init__('traffic_light_state_node')

        self.declare_parameter('detection_topic', '/yolo/traffic_light/detections')
        self.declare_parameter('image_topic', '/camera/image_1280x720')
        self.declare_parameter('output_topic', '/traffic_light_state')
        self.declare_parameter('debug_topic', '/traffic_light_state_debug')
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('confidence_threshold', 0.3)
        self.declare_parameter('stale_timeout_s', 0.5)
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('timeout_state', 'UNKNOWN')
        self.declare_parameter('unknown_state', 'UNKNOWN')
        self.declare_parameter('use_roi_color_fallback', False)
        self.declare_parameter('red_aliases', ['red', 'red_light', 'traffic_light_red', 'stop'])
        self.declare_parameter('yellow_aliases', ['yellow', 'yellow_light', 'traffic_light_yellow'])
        self.declare_parameter('green_aliases', ['green', 'green_light', 'traffic_light_green', 'go'])
        self.declare_parameter('state_priority', ['RED', 'YELLOW', 'GREEN', 'UNKNOWN'])

        self.detection_topic = str(self.get_parameter('detection_topic').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.debug_topic = str(self.get_parameter('debug_topic').value)
        self.publish_debug = bool(self.get_parameter('publish_debug').value)
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self.stale_timeout_s = max(0.0, float(self.get_parameter('stale_timeout_s').value))
        publish_rate_hz = max(0.1, float(self.get_parameter('publish_rate_hz').value))
        self.timeout_state = self._normalize_output_state(str(self.get_parameter('timeout_state').value))
        self.unknown_state = self._normalize_output_state(str(self.get_parameter('unknown_state').value))
        self.use_roi_color_fallback = bool(self.get_parameter('use_roi_color_fallback').value)
        self.state_priority = self._load_state_priority()

        self.alias_to_state = {}
        self._add_aliases('RED', self.get_parameter('red_aliases').value)
        self._add_aliases('YELLOW', self.get_parameter('yellow_aliases').value)
        self._add_aliases('GREEN', self.get_parameter('green_aliases').value)

        self.bridge = CvBridge() if CvBridge is not None else None
        self.latest_image = None
        self.latest_state = self.unknown_state
        self.latest_debug = self._format_debug('', 0.0, self.latest_state, 0, 'no_detection_received')
        self.last_detection_time = None
        self.last_logged_state = None
        self.last_logged_reason = None

        self.state_pub = self.create_publisher(String, self.output_topic, 10)
        self.debug_pub = self.create_publisher(String, self.debug_topic, 10) if self.publish_debug else None
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            self.detection_topic,
            self._detections_callback,
            10,
        )

        self.image_sub = None
        if self.use_roi_color_fallback:
            if self.bridge is None or cv2 is None or np is None:
                self.get_logger().warn('ROI color fallback requested but cv_bridge/OpenCV is unavailable; fallback disabled.')
                self.use_roi_color_fallback = False
            else:
                self.image_sub = self.create_subscription(Image, self.image_topic, self._image_callback, 10)

        self.timer = self.create_timer(1.0 / publish_rate_hz, self._publish_state)

        self.get_logger().info(
            'traffic_light_state_node started. '
            f'detection_topic={self.detection_topic} output_topic={self.output_topic} '
            f'confidence_threshold={self.confidence_threshold:.2f} timeout_state={self.timeout_state} '
            f'use_roi_color_fallback={self.use_roi_color_fallback}'
        )

    def _load_state_priority(self):
        priority = []
        for state in self.get_parameter('state_priority').value:
            norm = self._normalize_output_state(str(state))
            if norm not in priority:
                priority.append(norm)
        for state in ['RED', 'YELLOW', 'GREEN', 'UNKNOWN']:
            if state not in priority:
                priority.append(state)
        return priority

    def _add_aliases(self, state, aliases):
        for alias in aliases:
            key = self._normalize_class_id(alias)
            if key:
                self.alias_to_state[key] = state

    def _image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError:
            try:
                image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                if image is not None and len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image is not None and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                self.latest_image = image
            except Exception as exc:
                self.get_logger().warn(f'Failed to decode image for ROI fallback: {exc}')

    def _detections_callback(self, msg):
        self.last_detection_time = monotonic()
        candidates = []
        detection_count = len(getattr(msg, 'detections', []))

        for det in getattr(msg, 'detections', []):
            selected_class, score = self._best_result(det)
            if not self._valid_score(score) or score < self.confidence_threshold:
                continue

            state = self._state_from_class_id(selected_class)
            reason = 'class_alias' if state != self.unknown_state else 'class_not_color'
            if state == self.unknown_state and self.use_roi_color_fallback:
                fallback_state, fallback_reason = self._state_from_roi(det)
                if fallback_state != self.unknown_state:
                    state = fallback_state
                reason = fallback_reason

            candidates.append({
                'class_id': selected_class,
                'score': score,
                'state': state,
                'reason': reason,
            })

        if not candidates:
            self.latest_state = self.unknown_state
            self.latest_debug = self._format_debug('', 0.0, self.latest_state, detection_count, 'no_detection_above_threshold')
            return

        selected = self._select_candidate(candidates)
        self.latest_state = selected['state']
        self.latest_debug = self._format_debug(
            selected['class_id'],
            selected['score'],
            selected['state'],
            detection_count,
            selected['reason'],
        )

    def _best_result(self, det):
        best_class = ''
        best_score = 0.0
        for result in getattr(det, 'results', []):
            class_id, score = self._extract_class_score(result)
            if self._valid_score(score) and score >= best_score:
                best_class = class_id
                best_score = score
        return best_class, best_score

    def _extract_class_score(self, result):
        try:
            hypothesis = getattr(result, 'hypothesis', None)
            if hypothesis is not None:
                class_id = getattr(hypothesis, 'class_id', None)
                if class_id is None:
                    class_id = getattr(hypothesis, 'id', '')
                score = getattr(hypothesis, 'score', 0.0)
            else:
                class_id = getattr(result, 'class_id', None)
                if class_id is None:
                    class_id = getattr(result, 'id', '')
                score = getattr(result, 'score', 0.0)
            return str(class_id).strip(), float(score)
        except Exception:
            return '', 0.0

    def _state_from_class_id(self, class_id):
        return self.alias_to_state.get(self._normalize_class_id(class_id), self.unknown_state)

    def _normalize_class_id(self, value):
        return str(value).strip().lower().replace('-', '_').replace(' ', '_')

    def _normalize_output_state(self, value):
        state = str(value).strip().upper()
        if state not in VALID_STATES:
            return 'UNKNOWN'
        return state

    def _select_candidate(self, candidates):
        priority_index = {state: idx for idx, state in enumerate(self.state_priority)}
        return sorted(
            candidates,
            key=lambda item: (
                priority_index.get(item['state'], priority_index.get('UNKNOWN', 99)),
                -item['score'],
            ),
        )[0]

    def _state_from_roi(self, det):
        if self.latest_image is None:
            return self.unknown_state, 'roi_fallback_no_image'

        crop = self._crop_detection_roi(det)
        if crop is None or crop.size == 0:
            return self.unknown_state, 'roi_fallback_invalid_bbox'

        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            active = (saturation > 80) & (value > 100)
            active_pixels = max(1, int(active.sum()))

            hue = hsv[:, :, 0]
            red = (((hue <= 10) | (hue >= 170)) & active).sum()
            yellow = (((hue >= 15) & (hue <= 35)) & active).sum()
            green = (((hue >= 40) & (hue <= 90)) & active).sum()

            scores = {
                'RED': float(red) / active_pixels,
                'YELLOW': float(yellow) / active_pixels,
                'GREEN': float(green) / active_pixels,
            }
            best_state = max(scores, key=scores.get)
            best_score = scores[best_state]

            if best_state == 'GREEN':
                competing = max(scores['RED'], scores['YELLOW'])
                if best_score < 0.18 or best_score < competing * 1.5:
                    return self.unknown_state, 'roi_fallback_uncertain_green'
            elif best_score < 0.12:
                return self.unknown_state, 'roi_fallback_uncertain'

            return best_state, f'roi_fallback_{best_state.lower()}'
        except Exception:
            return self.unknown_state, 'roi_fallback_error'

    def _crop_detection_roi(self, det):
        bbox = getattr(det, 'bbox', None)
        if bbox is None or self.latest_image is None:
            return None

        cx, cy = self._bbox_center_xy(bbox)
        sx = float(getattr(bbox, 'size_x', 0.0))
        sy = float(getattr(bbox, 'size_y', 0.0))
        if sx <= 1.0 or sy <= 1.0:
            return None

        height, width = self.latest_image.shape[:2]
        x1 = max(0, int(round(cx - sx / 2.0)))
        y1 = max(0, int(round(cy - sy / 2.0)))
        x2 = min(width, int(round(cx + sx / 2.0)))
        y2 = min(height, int(round(cy + sy / 2.0)))
        if x2 <= x1 or y2 <= y1:
            return None
        return self.latest_image[y1:y2, x1:x2]

    def _bbox_center_xy(self, bbox):
        center = getattr(bbox, 'center', None)
        if center is None:
            return 0.0, 0.0
        if hasattr(center, 'x') and hasattr(center, 'y'):
            return float(center.x), float(center.y)
        position = getattr(center, 'position', None)
        if position is not None:
            return float(getattr(position, 'x', 0.0)), float(getattr(position, 'y', 0.0))
        return 0.0, 0.0

    def _publish_state(self):
        state = self.latest_state
        debug = self.latest_debug
        if self.last_detection_time is None:
            state = self.timeout_state
            debug = self._format_debug('', 0.0, state, 0, 'timeout_no_detection_received')
        elif monotonic() - self.last_detection_time > self.stale_timeout_s:
            state = self.timeout_state
            debug = self._format_debug('', 0.0, state, 0, 'timeout_stale_detection')

        self.state_pub.publish(String(data=state))
        if self.debug_pub is not None:
            self.debug_pub.publish(String(data=debug))

        reason = self._reason_from_debug(debug)
        if state != self.last_logged_state or reason != self.last_logged_reason:
            self.get_logger().info(f'traffic light state={state} reason={reason}')
            self.last_logged_state = state
            self.last_logged_reason = reason

    def _format_debug(self, selected_class, score, state, number_of_detections, reason):
        return (
            f'selected_class={selected_class} '
            f'score={float(score):.3f} '
            f'state={state} '
            f'number_of_detections={int(number_of_detections)} '
            f'reason={reason}'
        )

    def _reason_from_debug(self, debug_text):
        for token in debug_text.split():
            if token.startswith('reason='):
                return token.split('=', 1)[1]
        return ''

    def _valid_score(self, score):
        return math.isfinite(float(score))


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightStateNode()
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
