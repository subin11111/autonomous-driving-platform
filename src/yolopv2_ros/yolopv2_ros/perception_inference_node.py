import argparse
import os
import sys
from pathlib import Path


cv2 = None
torch = None
np = None

time_synchronized = None
select_device = None
increment_path = None
scale_coords = None
xyxy2xywh = None
non_max_suppression = None
split_for_trace_model = None
driving_area_mask = None
lane_line_mask = None
plot_one_box = None
show_seg_result = None
AverageMeter = None
letterbox = None
LoadImages = None


def _iter_unique_paths(paths):
    seen = set()
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path).expanduser()
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        yield path


def _discover_yolopv2_root(preferred_root=None):
    this_file = Path(__file__).resolve()
    candidates = []

    if preferred_root:
        candidates.append(preferred_root)

    env_root = os.getenv('YOLOPV2_ROOT')
    if env_root:
        candidates.append(env_root)

    cwd = Path.cwd()
    candidates.extend([
        cwd,
        cwd / 'YOLOPv2',
        Path.home() / 'YOLOPv2',
    ])

    for parent in this_file.parents:
        candidates.append(parent)
        candidates.append(parent / 'YOLOPv2')

    for candidate in _iter_unique_paths(candidates):
        if (candidate / 'utils' / 'utils.py').is_file():
            return candidate.resolve()
    return None


def _ensure_yolopv2_on_sys_path(root_path):
    if root_path is None:
        return
    root_str = str(root_path)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _load_runtime_dependencies():
    global cv2, torch, np
    global time_synchronized, select_device, increment_path, scale_coords, xyxy2xywh
    global non_max_suppression, split_for_trace_model, driving_area_mask, lane_line_mask
    global plot_one_box, show_seg_result, AverageMeter, letterbox, LoadImages

    # Import heavy dependencies lazily so importing this module stays fast.
    if torch is not None and cv2 is not None and select_device is not None:
        return

    import cv2 as _cv2
    import torch as _torch
    import numpy as _np

    from utils.utils import (  # type: ignore[import-not-found]
        time_synchronized as _time_synchronized,
        select_device as _select_device,
        increment_path as _increment_path,
        scale_coords as _scale_coords,
        xyxy2xywh as _xyxy2xywh,
        non_max_suppression as _non_max_suppression,
        split_for_trace_model as _split_for_trace_model,
        driving_area_mask as _driving_area_mask,
        lane_line_mask as _lane_line_mask,
        plot_one_box as _plot_one_box,
        show_seg_result as _show_seg_result,
        AverageMeter as _AverageMeter,
        letterbox as _letterbox,
        LoadImages as _LoadImages,
    )

    cv2 = _cv2
    torch = _torch
    np = _np
    time_synchronized = _time_synchronized
    select_device = _select_device
    increment_path = _increment_path
    scale_coords = _scale_coords
    xyxy2xywh = _xyxy2xywh
    non_max_suppression = _non_max_suppression
    split_for_trace_model = _split_for_trace_model
    driving_area_mask = _driving_area_mask
    lane_line_mask = _lane_line_mask
    plot_one_box = _plot_one_box
    show_seg_result = _show_seg_result
    AverageMeter = _AverageMeter
    letterbox = _letterbox
    LoadImages = _LoadImages



def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolopv2-root', type=str, default=None, help='path to YOLOPv2 repository root')
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--ros', action='store_true', help='deprecated: node is ROS-only and always runs in topic mode')
    parser.add_argument('--ros-input-topic', default='/camera/image_raw', help='ROS2 input image topic')
    parser.add_argument('--ros-output-topic', default='/yolopv2/result_image', help='ROS2 output image topic')
    parser.add_argument('--ros-drivable-mask-topic', default='/yolopv2/drivable_mask', help='ROS2 drivable-area mask image topic (mono8)')
    parser.add_argument('--ros-lane-mask-topic', default='/yolopv2/lane_mask', help='ROS2 lane-line mask image topic (mono8)')
    parser.add_argument('--ros-detections-topic', default='/yolopv2/detections', help='ROS2 detections topic')
    parser.add_argument('--ros-queue-size', type=int, default=10, help='ROS2 publisher/subscriber queue size')
    parser.add_argument('--ros-node-name', default='yolopv2_node', help='ROS2 node name')
    return parser


def resolve_weights_path(weights_arg):
    if isinstance(weights_arg, (list, tuple)):
        if not weights_arg:
            raise ValueError('weights path is empty')
        return weights_arg[0]
    return weights_arg


def _resolve_input_path(path_value, yolopv2_root=None, must_exist=True):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)

    candidates = []
    if yolopv2_root:
        candidates.append(yolopv2_root / path)
    candidates.extend([
        Path.cwd() / path,
        Path(__file__).resolve().parents[4] / path,
    ])

    for candidate in _iter_unique_paths(candidates):
        if candidate.exists():
            return str(candidate.resolve())

    if must_exist and yolopv2_root:
        matches = sorted(yolopv2_root.rglob(path.name))
        if matches:
            return str(matches[0].resolve())

    fallback_base = yolopv2_root if yolopv2_root else Path.cwd()
    return str((fallback_base / path).resolve())


def _resolve_output_path(path_value, yolopv2_root=None):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    base = yolopv2_root if yolopv2_root else Path.cwd()
    return str((base / path).resolve())


def configure_runtime_paths(opt):
    selected_root = _discover_yolopv2_root(opt.yolopv2_root)
    _ensure_yolopv2_on_sys_path(selected_root)

    if selected_root is None:
        raise FileNotFoundError(
            'YOLOPv2 root was not found. Provide --yolopv2-root or set YOLOPV2_ROOT. '
            'Expected to find utils/utils.py under the root.'
        )

    opt.yolopv2_root = str(selected_root)

    resolved_weights = _resolve_input_path(resolve_weights_path(opt.weights), selected_root, must_exist=True)
    if not Path(resolved_weights).is_file():
        raise FileNotFoundError(
            f'weights file not found: {resolved_weights}. '
            'Update --weights or place the file under your YOLOPv2 root.'
        )
    opt.weights = [resolved_weights]


def load_model(opt):
    _load_runtime_dependencies()
    stride = 32
    device = select_device(opt.device)
    model = torch.jit.load(resolve_weights_path(opt.weights), map_location=device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = model.to(device)
    if half:
        model.half()  # to FP16
    model.eval()
    return model, device, half, stride


def preprocess_frame(im0, img_size, stride):
    _load_runtime_dependencies()
    im0 = cv2.resize(im0, (1280, 720), interpolation=cv2.INTER_LINEAR)
    img = letterbox(im0, img_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    return img, im0


def infer_detections(img, im0, model, device, half, opt):
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    [pred, anchor_grid], seg, ll = model(img)
    pred = split_for_trace_model(pred, anchor_grid)
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    det = pred[0]
    if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    da_seg_mask = driving_area_mask(seg)
    ll_seg_mask = lane_line_mask(ll)
    return det, da_seg_mask, ll_seg_mask


def detections_to_msg(det, header, Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose):
    msg = Detection2DArray()
    msg.header = header
    if det is None or len(det) == 0:
        return msg
    det = det.cpu()
    for *xyxy, conf, cls in det:
        detection = Detection2D()
        detection.header = header
        x1, y1, x2, y2 = [float(x) for x in xyxy]
        bbox = BoundingBox2D()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        # vision_msgs schema differs by version: center may expose x/y directly or via position.x/y.
        if hasattr(bbox.center, 'x') and hasattr(bbox.center, 'y'):
            bbox.center.x = cx
            bbox.center.y = cy
        elif hasattr(bbox.center, 'position'):
            bbox.center.position.x = cx
            bbox.center.position.y = cy
            if hasattr(bbox.center, 'theta'):
                bbox.center.theta = 0.0
        else:
            raise AttributeError('Unsupported BoundingBox2D.center schema in vision_msgs')
        bbox.size_x = x2 - x1
        bbox.size_y = y2 - y1
        detection.bbox = bbox
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = str(int(cls))
        hypothesis.hypothesis.score = float(conf)
        detection.results.append(hypothesis)
        msg.detections.append(detection)
    return msg


def draw_detections(im0, det):
    if det is None or len(det) == 0:
        return im0
    for *xyxy, conf, cls in det:
        label = f'{int(cls)}:{float(conf):.2f}'
        plot_one_box(xyxy, im0, label=label, line_thickness=2)
    return im0


def overlay_segmentation(im0, da_seg_mask, ll_seg_mask):
    show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)
    return im0


def mask_to_mono8(mask):
    mask_u8 = (mask.astype(np.uint8) * 255)
    return np.ascontiguousarray(mask_u8)


def ros_spin(opt, ros_args):
    _load_runtime_dependencies()
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose  # type: ignore[import-not-found]
        from cv_bridge import CvBridge
    except ImportError as exc:
        raise ImportError(
            'ROS2 python dependencies are missing. Install rclpy/sensor_msgs/vision_msgs/cv_bridge and source your ROS2 environment.'
        ) from exc

    class Yolopv2Node(Node):
        def __init__(self):
            super().__init__(opt.ros_node_name)
            self.bridge = CvBridge()
            self.model, self.device, self.half, self.stride = load_model(opt)
            self.frame_count = 0
            if self.device.type != 'cpu':
                self.model(torch.zeros(1, 3, opt.img_size, opt.img_size).to(self.device).type_as(next(self.model.parameters())))
            self.result_image_publisher = self.create_publisher(Image, opt.ros_output_topic, opt.ros_queue_size)
            self.drivable_mask_publisher = self.create_publisher(Image, opt.ros_drivable_mask_topic, opt.ros_queue_size)
            self.lane_mask_publisher = self.create_publisher(Image, opt.ros_lane_mask_topic, opt.ros_queue_size)
            self.detections_publisher = self.create_publisher(Detection2DArray, opt.ros_detections_topic, opt.ros_queue_size)
            self.subscription = self.create_subscription(Image, opt.ros_input_topic, self.image_callback, opt.ros_queue_size)
            self.get_logger().info(
                f'YOLOPv2 ROS node started. input={opt.ros_input_topic}, '
                f'result_image={opt.ros_output_topic}, drivable_mask={opt.ros_drivable_mask_topic}, '
                f'lane_mask={opt.ros_lane_mask_topic}, detections={opt.ros_detections_topic}'
            )

        def image_callback(self, msg):
            try:
                im0 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception:
                im0 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                if len(im0.shape) == 2:
                    im0 = cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)
                elif im0.shape[2] == 4:
                    im0 = cv2.cvtColor(im0, cv2.COLOR_BGRA2BGR)

            img, im0 = preprocess_frame(im0, opt.img_size, self.stride)
            with torch.no_grad():
                det, da_seg_mask, ll_seg_mask = infer_detections(img, im0, self.model, self.device, self.half, opt)

            vis = draw_detections(im0.copy(), det)
            vis = overlay_segmentation(vis, da_seg_mask, ll_seg_mask)
            detection_msg = detections_to_msg(det, msg.header, Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose)
            self.detections_publisher.publish(detection_msg)

            result_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            result_msg.header = msg.header
            self.result_image_publisher.publish(result_msg)

            da_mask_msg = self.bridge.cv2_to_imgmsg(mask_to_mono8(da_seg_mask), encoding='mono8')
            da_mask_msg.header = msg.header
            self.drivable_mask_publisher.publish(da_mask_msg)

            ll_mask_msg = self.bridge.cv2_to_imgmsg(mask_to_mono8(ll_seg_mask), encoding='mono8')
            ll_mask_msg.header = msg.header
            self.lane_mask_publisher.publish(ll_mask_msg)

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.get_logger().info(f'processed frames: {self.frame_count}, detections: {len(detection_msg.detections)}')

    rclpy.init(args=ros_args)
    node = Yolopv2Node()
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


def main(args=None):
    opt, ros_args = make_parser().parse_known_args(args=args)
    configure_runtime_paths(opt)
    _load_runtime_dependencies()
    print(opt)
    ros_spin(opt, ros_args)


if __name__ == '__main__':
    main()
