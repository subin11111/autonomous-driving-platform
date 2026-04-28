import argparse
import struct
from typing import List, Optional, Sequence, Tuple

import numpy as np


def deg2rad(deg: float) -> float:
    return deg * np.pi / 180.0


def rotation_matrix_y(pitch_rad: float) -> np.ndarray:
    """
    Rotation about y axis.
    Used to tilt camera optical axis downward from vehicle frame perspective.
    """
    c = np.cos(pitch_rad)
    s = np.sin(pitch_rad)
    return np.array(
        [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ],
        dtype=np.float64,
    )


def build_intrinsic_from_fov(image_width: int, image_height: int, hfov_deg: float) -> np.ndarray:
    """
    Build approximate camera intrinsic matrix from horizontal FOV.
    """
    hfov = deg2rad(hfov_deg)
    fx = (image_width / 2.0) / np.tan(hfov / 2.0)
    fy = fx
    cx = image_width / 2.0
    cy = image_height / 2.0

    return np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


class SimpleGroundProjector:
    """
    Monocular pixel to vehicle-frame ground point projection.

    Vehicle frame:
      x: forward
      y: left
      z: up
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        hfov_deg: float = 90.0,
        cam_x: float = 2.05,
        cam_y: float = 0.0,
        cam_z: float = 1.45,
        pitch_deg: float = 12.0,
    ):
        self.w = image_width
        self.h = image_height
        self.K = build_intrinsic_from_fov(image_width, image_height, hfov_deg)
        self.K_inv = np.linalg.inv(self.K)

        # Camera origin in vehicle frame.
        self.C = np.array([cam_x, cam_y, cam_z], dtype=np.float64)

        # Camera frame definition:
        #   x_c: image right
        #   y_c: image down
        #   z_c: optical axis forward
        #
        # Mapping to vehicle frame:
        #   z_c -> x_v
        #   x_c -> -y_v
        #   y_c -> -z_v
        R0 = np.array(
            [
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0],
            ],
            dtype=np.float64,
        )

        Rp = rotation_matrix_y(deg2rad(pitch_deg))
        self.R_cam_to_vehicle = R0 @ Rp

    def pixel_to_ground(self, u: float, v: float) -> Optional[Tuple[float, float, float]]:
        """
        Intersect pixel ray with ground plane z=0.
        Returns (x, y, 0) in vehicle frame, or None if invalid.
        """
        pix = np.array([u, v, 1.0], dtype=np.float64)

        ray_cam = self.K_inv @ pix
        ray_cam = ray_cam / np.linalg.norm(ray_cam)

        ray_vehicle = self.R_cam_to_vehicle @ ray_cam
        ray_vehicle = ray_vehicle / np.linalg.norm(ray_vehicle)

        dz = ray_vehicle[2]
        cz = self.C[2]

        if abs(dz) < 1e-8:
            return None

        s = -cz / dz
        if s <= 0:
            return None

        P = self.C + s * ray_vehicle

        if P[0] < 0:
            return None

        return float(P[0]), float(P[1]), 0.0


def sample_mask_pixels(
    mask: np.ndarray,
    sample_step: int = 6,
    min_v_ratio: float = 0.45,
    max_points: int = 2000,
) -> List[Tuple[int, int]]:
    """
    Sample sparse pixel coordinates (u, v) from binary mask.
    Only keeps lower part of the image for stable ground intersections.
    """
    if mask is None or mask.size == 0:
        return []

    if mask.ndim == 3:
        # Accept either BGR/RGB-style mask or stacked channels; any non-zero means valid.
        mask_gray = np.any(mask > 0, axis=2).astype(np.uint8)
    else:
        mask_gray = mask

    h, w = mask_gray.shape[:2]
    min_v = int(h * min_v_ratio)

    if mask_gray.dtype != np.uint8:
        mask_u8 = (mask_gray > 0).astype(np.uint8)
    else:
        mask_u8 = (mask_gray > 0).astype(np.uint8)

    ys, xs = np.where(mask_u8[min_v:, :] > 0)
    if len(xs) == 0:
        return []

    ys = ys + min_v

    keep = ((xs % sample_step) == 0) & ((ys % sample_step) == 0)
    xs = xs[keep]
    ys = ys[keep]

    if len(xs) == 0:
        return []

    if len(xs) > max_points:
        idx = np.linspace(0, len(xs) - 1, max_points, dtype=np.int32)
        xs = xs[idx]
        ys = ys[idx]

    return [(int(u), int(v)) for u, v in zip(xs, ys)]


def mask_to_ground_points(
    mask: np.ndarray,
    projector: SimpleGroundProjector,
    sample_step: int = 6,
    min_v_ratio: float = 0.45,
    max_points: int = 2000,
    min_x: float = 0.0,
    max_x: float = 40.0,
    max_abs_y: float = 12.0,
) -> List[Tuple[float, float, float]]:
    """
    Convert binary mask pixels into projected vehicle-frame ground points.
    """
    pixels = sample_mask_pixels(
        mask=mask,
        sample_step=sample_step,
        min_v_ratio=min_v_ratio,
        max_points=max_points,
    )

    points: List[Tuple[float, float, float]] = []
    for u, v in pixels:
        p = projector.pixel_to_ground(u, v)
        if p is None:
            continue

        x, y, z = p
        if x < min_x or x > max_x:
            continue
        if abs(y) > max_abs_y:
            continue

        points.append((x, y, z))

    points.sort(key=lambda pt: (pt[0], abs(pt[1])))
    return points


def points_to_pointcloud2(
    points_xyz: Sequence[Tuple[float, float, float]],
    header,
    PointCloud2,
    PointField,
):
    """
    Build PointCloud2 message with x,y,z float32 fields.
    """
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    point_step = 16
    data = b"".join(struct.pack("<fffxxxx", float(x), float(y), float(z)) for x, y, z in points_xyz)
    width = len(points_xyz)

    return PointCloud2(
        header=header,
        height=1,
        width=width,
        fields=fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=point_step * width,
        data=data,
        is_dense=True,
    )


def build_default_projector_from_image(
    image: np.ndarray,
    hfov_deg: float = 90.0,
    cam_x: float = 2.05,
    cam_y: float = 0.0,
    cam_z: float = 1.45,
    pitch_deg: float = 12.0,
) -> SimpleGroundProjector:
    h, w = image.shape[:2]
    return SimpleGroundProjector(
        image_width=w,
        image_height=h,
        hfov_deg=hfov_deg,
        cam_x=cam_x,
        cam_y=cam_y,
        cam_z=cam_z,
        pitch_deg=pitch_deg,
    )


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ros-node-name', default='mask_ground_projection_node', help='ROS2 node name')
    parser.add_argument('--input-lane-mask-topic', default='/yolopv2/lane_mask', help='input lane mask topic from perception node')
    parser.add_argument('--input-drivable-mask-topic', default='/yolopv2/drivable_mask', help='input drivable mask topic from perception node')
    parser.add_argument('--output-lane-points-topic', default='/perception/real_world_lane_points', help='output lane PointCloud2 topic')
    parser.add_argument('--output-drivable-points-topic', default='/perception/real_world_drivable_points', help='output drivable PointCloud2 topic')
    parser.add_argument('--output-rviz-topic', default='/masked_ray_ground', help='output PointCloud2 topic for RViz visualization')
    parser.add_argument('--output-frame-id', default='ego_vehicle', help='output frame_id override')
    parser.add_argument('--ros-queue-size', type=int, default=10, help='ROS2 pub/sub queue size')

    parser.add_argument('--hfov-deg', type=float, default=90.0, help='camera horizontal FOV [deg]')
    parser.add_argument('--cam-x', type=float, default=2.05, help='camera x in vehicle frame [m]')
    parser.add_argument('--cam-y', type=float, default=0.0, help='camera y in vehicle frame [m]')
    parser.add_argument('--cam-z', type=float, default=1.45, help='camera height from ground [m]')
    parser.add_argument('--pitch-deg', type=float, default=12.0, help='camera downward pitch [deg]')

    parser.add_argument('--sample-step', type=int, default=6, help='pixel sampling stride')
    parser.add_argument('--min-v-ratio', type=float, default=0.45, help='min vertical image ratio to keep')
    parser.add_argument('--max-points', type=int, default=2000, help='max output points per frame')
    parser.add_argument('--min-x', type=float, default=0.0, help='min x filter [m]')
    parser.add_argument('--max-x', type=float, default=40.0, help='max x filter [m]')
    parser.add_argument('--max-abs-y', type=float, default=12.0, help='max abs(y) filter [m]')
    return parser


def ros_spin(opt, ros_args):
    try:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from sensor_msgs.msg import Image, PointCloud2, PointField
    except ImportError as exc:
        raise ImportError(
            'ROS2 python dependencies are missing. Install rclpy/sensor_msgs/cv_bridge and source your ROS2 environment.'
        ) from exc

    class MaskGroundProjectionNode(Node):
        def __init__(self):
            super().__init__(opt.ros_node_name)
            self.bridge = CvBridge()
            self.projector = None
            self.frame_count = 0

            self.lane_output_pub = self.create_publisher(PointCloud2, opt.output_lane_points_topic, opt.ros_queue_size)
            self.drivable_output_pub = self.create_publisher(PointCloud2, opt.output_drivable_points_topic, opt.ros_queue_size)
            self.rviz_output_pub = self.create_publisher(PointCloud2, opt.output_rviz_topic, opt.ros_queue_size)

            self.lane_sub = self.create_subscription(
                Image,
                opt.input_lane_mask_topic,
                self.lane_mask_callback,
                opt.ros_queue_size,
            )
            self.drivable_sub = self.create_subscription(
                Image,
                opt.input_drivable_mask_topic,
                self.drivable_mask_callback,
                opt.ros_queue_size,
            )

            self.get_logger().info(
                f'mask projector started. lane_in={opt.input_lane_mask_topic}, drivable_in={opt.input_drivable_mask_topic}, '
                f'lane_out={opt.output_lane_points_topic}, drivable_out={opt.output_drivable_points_topic}, '
                f'rviz_out={opt.output_rviz_topic}, frame={opt.output_frame_id}'
            )

        def _build_projector_if_needed(self, width: int, height: int):
            if self.projector is not None:
                return
            self.projector = SimpleGroundProjector(
                image_width=width,
                image_height=height,
                hfov_deg=opt.hfov_deg,
                cam_x=opt.cam_x,
                cam_y=opt.cam_y,
                cam_z=opt.cam_z,
                pitch_deg=opt.pitch_deg,
            )
            self.get_logger().info(
                f'projector init: size={width}x{height}, hfov={opt.hfov_deg}, cam=({opt.cam_x},{opt.cam_y},{opt.cam_z}), pitch={opt.pitch_deg}'
            )

        def _project_and_publish(self, msg, publisher, tag):
            try:
                mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            except Exception as exc:
                self.get_logger().error(f'[{tag}] failed to decode mask image: {exc}')
                return

            if mask is None or np.size(mask) == 0:
                return

            if mask.ndim == 3:
                mask_bin = np.any(mask > 0, axis=2).astype(np.uint8)
            else:
                mask_bin = (mask > 0).astype(np.uint8)

            height, width = mask_bin.shape[:2]
            self._build_projector_if_needed(width, height)

            points = mask_to_ground_points(
                mask=mask_bin,
                projector=self.projector,
                sample_step=opt.sample_step,
                min_v_ratio=opt.min_v_ratio,
                max_points=opt.max_points,
                min_x=opt.min_x,
                max_x=opt.max_x,
                max_abs_y=opt.max_abs_y,
            )

            out_header = msg.header
            if opt.output_frame_id:
                out_header.frame_id = opt.output_frame_id

            cloud = points_to_pointcloud2(points, out_header, PointCloud2, PointField)
            publisher.publish(cloud)
            self.rviz_output_pub.publish(cloud)

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.get_logger().info(f'processed={self.frame_count}, source={tag}, points={len(points)}')

        def lane_mask_callback(self, msg):
            self._project_and_publish(msg, self.lane_output_pub, 'lane')

        def drivable_mask_callback(self, msg):
            self._project_and_publish(msg, self.drivable_output_pub, 'drivable')

    rclpy.init(args=ros_args)
    node = MaskGroundProjectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


def main(args=None):
    opt, ros_args = make_parser().parse_known_args(args=args)
    ros_spin(opt, ros_args)


if __name__ == '__main__':
    main()
