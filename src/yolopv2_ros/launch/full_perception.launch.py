from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.conditions import IfCondition
import os


def generate_launch_description():
    # Raw camera input
    raw_image_topic = LaunchConfiguration('raw_image_topic', default='/camera/image_raw')

    # Resized image (unified resolution for all perception nodes)
    resized_image_topic = LaunchConfiguration('resized_image_topic', default='/camera/image_1280x720')
    resize_width = LaunchConfiguration('resize_width', default='1280')
    resize_height = LaunchConfiguration('resize_height', default='720')

    # YOLOPv2 arguments
    yolopv2_root = LaunchConfiguration('yolopv2_root', default='/home/subin/YOLOPv2')
    yolopv2_weights = LaunchConfiguration('yolopv2_weights', default='/home/subin/YOLOPv2/data/weights/yolopv2.pt')
    yolopv2_device = LaunchConfiguration('yolopv2_device', default='0')
    yolopv2_img_size = LaunchConfiguration('yolopv2_img_size', default='640')
    yolopv2_conf_thres = LaunchConfiguration('yolopv2_conf_thres', default='0.3')
    yolopv2_iou_thres = LaunchConfiguration('yolopv2_iou_thres', default='0.45')
    yolopv2_vehicle_classes = LaunchConfiguration('yolopv2_vehicle_classes', default='')

    # Ultralytics YOLO arguments
    yolo_model_name = LaunchConfiguration('yolo_model_name', default='yolov8n.pt')
    yolo_device = LaunchConfiguration('yolo_device', default='cpu')
    yolo_confidence_threshold = LaunchConfiguration('yolo_confidence_threshold', default='0.25')
    yolo_iou_threshold = LaunchConfiguration('yolo_iou_threshold', default='0.45')

    # Fusion visualizer arguments
    fused_image_topic = LaunchConfiguration('fused_image_topic', default='/perception/fused_debug_image')
    fused_detections_topic = LaunchConfiguration('fused_detections_topic', default='/perception/fused_detections')
    publish_fused_image = LaunchConfiguration('publish_fused_image', default='true')
    publish_fused_detections = LaunchConfiguration('publish_fused_detections', default='true')

    ld = LaunchDescription([
        DeclareLaunchArgument('raw_image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('resized_image_topic', default_value='/camera/image_1280x720'),
        DeclareLaunchArgument('resize_width', default_value='1280'),
        DeclareLaunchArgument('resize_height', default_value='720'),
        # video source options
        DeclareLaunchArgument('use_video_source', default_value='true'),
        DeclareLaunchArgument('video_path', default_value='/home/subin/test.mp4'),
        DeclareLaunchArgument('video_output_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('video_loop', default_value='true'),
        DeclareLaunchArgument('video_fps', default_value='30.0'),
        DeclareLaunchArgument('video_frame_id', default_value='camera'),
        DeclareLaunchArgument('yolopv2_root', default_value='/home/subin/YOLOPv2'),
        DeclareLaunchArgument('yolopv2_weights', default_value='/home/subin/YOLOPv2/data/weights/yolopv2.pt'),
        DeclareLaunchArgument('yolopv2_device', default_value='0'),
        DeclareLaunchArgument('yolopv2_img_size', default_value='640'),
        DeclareLaunchArgument('yolopv2_conf_thres', default_value='0.3'),
        DeclareLaunchArgument('yolopv2_iou_thres', default_value='0.45'),
        DeclareLaunchArgument('yolopv2_vehicle_classes', default_value=''),
        DeclareLaunchArgument('yolo_model_name', default_value='yolov8n.pt'),
        DeclareLaunchArgument('yolo_device', default_value='cpu'),
        DeclareLaunchArgument('yolo_confidence_threshold', default_value='0.25'),
        DeclareLaunchArgument('yolo_iou_threshold', default_value='0.45'),
        DeclareLaunchArgument('fused_image_topic', default_value='/perception/fused_debug_image'),
        DeclareLaunchArgument('fused_detections_topic', default_value='/perception/fused_detections'),
        DeclareLaunchArgument('publish_fused_image', default_value='true'),
        DeclareLaunchArgument('publish_fused_detections', default_value='true'),

        # Optional video_to_topic node (publishes /camera/image_raw)
        Node(
            package='yolopv2_ros',
            executable='video_to_topic',
            name='video_to_topic_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_video_source')),
            parameters=[{
                'video_path': LaunchConfiguration('video_path'),
                'output_topic': LaunchConfiguration('video_output_topic'),
                'loop': LaunchConfiguration('video_loop'),
                'fps': LaunchConfiguration('video_fps'),
                'frame_id': LaunchConfiguration('video_frame_id'),
            }]
        ),

        # Image resize node: /camera/image_raw -> /camera/image_1280x720
        Node(
            package='yolopv2_ros',
            executable='image_resize',
            name='image_resize_node',
            output='screen',
            parameters=[{
                'input_image_topic': raw_image_topic,
                'output_image_topic': resized_image_topic,
                'output_width': resize_width,
                'output_height': resize_height,
                'queue_size': 10,
                'publish_even_if_same_size': True,
            }]
        ),

        # YOLOPv2 node: uses resized image
        Node(
            package='yolopv2_ros',
            executable='perception_inference',
            name='yolopv2_node',
            output='screen',
            arguments=[
                '--ros-input-topic', resized_image_topic,
                '--ros-output-topic', '/yolopv2/result_image',
                '--ros-drivable-mask-topic', '/yolopv2/drivable_mask',
                '--ros-lane-mask-topic', '/yolopv2/lane_mask',
                '--ros-detections-topic', '/yolopv2/vehicle_detections',
                '--ros-node-name', 'yolopv2_node',
                '--yolopv2-root', yolopv2_root,
                '--weights', yolopv2_weights,
                '--img-size', yolopv2_img_size,
                '--conf-thres', yolopv2_conf_thres,
                '--iou-thres', yolopv2_iou_thres,
                '--device', yolopv2_device,
            ]
        ),

        # Pedestrian detector node: uses resized image
        Node(
            package='yolopv2_ros',
            executable='pedestrian_detector',
            name='pedestrian_detector_node',
            output='screen',
            parameters=[{
                'input_image_topic': resized_image_topic,
                'detection_topic': '/yolo/person/detections',
                'result_image_topic': '/yolo/person/result_image',
                'model_name': yolo_model_name,
                'device': ParameterValue(yolo_device, value_type=str),
                'confidence_threshold': yolo_confidence_threshold,
                'iou_threshold': yolo_iou_threshold,
                'queue_size': 10,
                'publish_result_image': True,
            }]
        ),

        # Traffic light detector node: uses resized image
        Node(
            package='yolopv2_ros',
            executable='traffic_light_detector',
            name='traffic_light_detector_node',
            output='screen',
            parameters=[{
                'input_image_topic': resized_image_topic,
                'detection_topic': '/yolo/traffic_light/detections',
                'result_image_topic': '/yolo/traffic_light/result_image',
                'model_name': yolo_model_name,
                'device': ParameterValue(yolo_device, value_type=str),
                'confidence_threshold': yolo_confidence_threshold,
                'iou_threshold': yolo_iou_threshold,
                'queue_size': 10,
                'publish_result_image': True,
            }]
        ),

        # Fusion visualizer node: uses resized image
        Node(
            package='yolopv2_ros',
            executable='fusion_visualizer',
            name='fusion_visualizer_node',
            output='screen',
            parameters=[{
                'image_topic': resized_image_topic,
                'drivable_mask_topic': '/yolopv2/drivable_mask',
                'lane_mask_topic': '/yolopv2/lane_mask',
                'vehicle_detections_topic': '/yolopv2/vehicle_detections',
                'person_detections_topic': '/yolo/person/detections',
                'traffic_light_detections_topic': '/yolo/traffic_light/detections',
                'fused_image_topic': fused_image_topic,
                'fused_detections_topic': fused_detections_topic,
                'publish_fused_image': publish_fused_image,
                'publish_fused_detections': publish_fused_detections,
                'queue_size': 10,
            }]
        ),
    ])

    return ld

