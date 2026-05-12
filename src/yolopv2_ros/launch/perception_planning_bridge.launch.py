from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    raw_image_topic = LaunchConfiguration('raw_image_topic', default='/camera/image_raw')
    resized_image_topic = LaunchConfiguration('resized_image_topic', default='/camera/image_1280x720')

    lane_mask_topic = LaunchConfiguration('lane_mask_topic', default='/yolopv2/lane_mask')
    drivable_mask_topic = LaunchConfiguration('drivable_mask_topic', default='/yolopv2/drivable_mask')
    detections_topic = LaunchConfiguration('detections_topic', default='/yolopv2/detections')

    lane_points_topic = LaunchConfiguration('lane_points_topic', default='/perception/real_world_lane_points')
    drivable_points_topic = LaunchConfiguration('drivable_points_topic', default='/perception/real_world_drivable_points')
    closest_obstacle_topic = LaunchConfiguration('closest_obstacle_topic', default='/perception/closest_obstacle')

    yolopv2_root = LaunchConfiguration('yolopv2_root', default='/home/subin/YOLOPv2')
    yolopv2_weights = LaunchConfiguration('yolopv2_weights', default='/home/subin/YOLOPv2/data/weights/yolopv2.pt')
    yolopv2_device = LaunchConfiguration('yolopv2_device', default='0')
    yolopv2_img_size = LaunchConfiguration('yolopv2_img_size', default='640')
    yolopv2_conf_thres = LaunchConfiguration('yolopv2_conf_thres', default='0.3')
    yolopv2_iou_thres = LaunchConfiguration('yolopv2_iou_thres', default='0.45')

    yolo_model_name = LaunchConfiguration('yolo_model_name', default='yolov8n.pt')
    yolo_device = LaunchConfiguration('yolo_device', default='cpu')
    yolo_confidence_threshold = LaunchConfiguration('yolo_confidence_threshold', default='0.25')
    yolo_iou_threshold = LaunchConfiguration('yolo_iou_threshold', default='0.45')

    enable_video_source = LaunchConfiguration('use_video_source', default='true')
    enable_projection = LaunchConfiguration('enable_projection', default='true')
    enable_traffic_light_state = LaunchConfiguration('enable_traffic_light_state', default='true')
    traffic_light_detections_topic = LaunchConfiguration(
        'traffic_light_detections_topic',
        default='/yolo/traffic_light/detections',
    )
    traffic_light_state_topic = LaunchConfiguration('traffic_light_state_topic', default='/traffic_light_state')
    traffic_light_confidence_threshold = LaunchConfiguration('traffic_light_confidence_threshold', default='0.3')
    traffic_light_image_topic = LaunchConfiguration('traffic_light_image_topic', default='/camera/image_1280x720')
    use_roi_color_fallback = LaunchConfiguration('use_roi_color_fallback', default='false')
    enable_speed_stub = LaunchConfiguration('enable_speed_stub', default='false')

    return LaunchDescription([
        DeclareLaunchArgument('raw_image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('resized_image_topic', default_value='/camera/image_1280x720'),
        DeclareLaunchArgument('resize_width', default_value='1280'),
        DeclareLaunchArgument('resize_height', default_value='720'),

        DeclareLaunchArgument('use_video_source', default_value='true'),
        DeclareLaunchArgument('video_path', default_value='/home/subin/test.mp4'),
        DeclareLaunchArgument('video_output_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('video_loop', default_value='true'),
        DeclareLaunchArgument('video_fps', default_value='30.0'),
        DeclareLaunchArgument('video_frame_id', default_value='camera'),

        DeclareLaunchArgument('lane_mask_topic', default_value='/yolopv2/lane_mask'),
        DeclareLaunchArgument('drivable_mask_topic', default_value='/yolopv2/drivable_mask'),
        DeclareLaunchArgument('detections_topic', default_value='/yolopv2/detections'),
        DeclareLaunchArgument('lane_points_topic', default_value='/perception/real_world_lane_points'),
        DeclareLaunchArgument('drivable_points_topic', default_value='/perception/real_world_drivable_points'),
        DeclareLaunchArgument('closest_obstacle_topic', default_value='/perception/closest_obstacle'),

        DeclareLaunchArgument('yolopv2_root', default_value='/home/subin/YOLOPv2'),
        DeclareLaunchArgument('yolopv2_weights', default_value='/home/subin/YOLOPv2/data/weights/yolopv2.pt'),
        DeclareLaunchArgument('yolopv2_device', default_value='0'),
        DeclareLaunchArgument('yolopv2_img_size', default_value='640'),
        DeclareLaunchArgument('yolopv2_conf_thres', default_value='0.3'),
        DeclareLaunchArgument('yolopv2_iou_thres', default_value='0.45'),

        DeclareLaunchArgument('yolo_model_name', default_value='yolov8n.pt'),
        DeclareLaunchArgument('yolo_device', default_value='cpu'),
        DeclareLaunchArgument('yolo_confidence_threshold', default_value='0.25'),
        DeclareLaunchArgument('yolo_iou_threshold', default_value='0.45'),

        DeclareLaunchArgument('enable_projection', default_value='true'),
        DeclareLaunchArgument('enable_traffic_light_state', default_value='true'),
        DeclareLaunchArgument('traffic_light_detections_topic', default_value='/yolo/traffic_light/detections'),
        DeclareLaunchArgument('traffic_light_state_topic', default_value='/traffic_light_state'),
        DeclareLaunchArgument('traffic_light_confidence_threshold', default_value='0.3'),
        DeclareLaunchArgument('traffic_light_image_topic', default_value='/camera/image_1280x720'),
        DeclareLaunchArgument('use_roi_color_fallback', default_value='false'),
        DeclareLaunchArgument('enable_speed_stub', default_value='false'),

        Node(
            package='yolopv2_ros',
            executable='video_to_topic',
            name='video_to_topic_node',
            output='screen',
            condition=IfCondition(enable_video_source),
            parameters=[{
                'video_path': LaunchConfiguration('video_path'),
                'output_topic': LaunchConfiguration('video_output_topic'),
                'loop': LaunchConfiguration('video_loop'),
                'fps': LaunchConfiguration('video_fps'),
                'frame_id': LaunchConfiguration('video_frame_id'),
            }],
        ),

        Node(
            package='yolopv2_ros',
            executable='image_resize',
            name='image_resize_node',
            output='screen',
            parameters=[{
                'input_image_topic': raw_image_topic,
                'output_image_topic': resized_image_topic,
                'output_width': ParameterValue(LaunchConfiguration('resize_width'), value_type=int),
                'output_height': ParameterValue(LaunchConfiguration('resize_height'), value_type=int),
                'queue_size': 10,
                'publish_even_if_same_size': True,
            }],
        ),

        Node(
            package='yolopv2_ros',
            executable='perception_inference',
            name='yolopv2_node',
            output='screen',
            arguments=[
                '--ros-input-topic', resized_image_topic,
                '--ros-output-topic', '/yolopv2/result_image',
                '--ros-drivable-mask-topic', drivable_mask_topic,
                '--ros-lane-mask-topic', lane_mask_topic,
                '--ros-detections-topic', detections_topic,
                '--ros-node-name', 'yolopv2_node',
                '--yolopv2-root', yolopv2_root,
                '--weights', yolopv2_weights,
                '--img-size', yolopv2_img_size,
                '--conf-thres', yolopv2_conf_thres,
                '--iou-thres', yolopv2_iou_thres,
                '--device', yolopv2_device,
            ],
        ),

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
            }],
        ),

        Node(
            package='yolopv2_ros',
            executable='traffic_light_detector',
            name='traffic_light_detector_node',
            output='screen',
            parameters=[{
                'input_image_topic': resized_image_topic,
                'detection_topic': traffic_light_detections_topic,
                'result_image_topic': '/yolo/traffic_light/result_image',
                'model_name': yolo_model_name,
                'device': ParameterValue(yolo_device, value_type=str),
                'confidence_threshold': yolo_confidence_threshold,
                'iou_threshold': yolo_iou_threshold,
                'queue_size': 10,
                'publish_result_image': True,
            }],
        ),

        Node(
            package='yolopv2_ros',
            executable='traffic_light_state',
            name='traffic_light_state_node',
            output='screen',
            condition=IfCondition(enable_traffic_light_state),
            parameters=[{
                'detection_topic': traffic_light_detections_topic,
                'image_topic': traffic_light_image_topic,
                'output_topic': traffic_light_state_topic,
                'debug_topic': '/traffic_light_state_debug',
                'publish_debug': True,
                'confidence_threshold': traffic_light_confidence_threshold,
                'stale_timeout_s': 0.5,
                'publish_rate_hz': 10.0,
                'timeout_state': 'UNKNOWN',
                'unknown_state': 'UNKNOWN',
                'use_roi_color_fallback': ParameterValue(use_roi_color_fallback, value_type=bool),
            }],
        ),

        Node(
            package='yolopv2_ros',
            executable='fusion_visualizer',
            name='fusion_visualizer_node',
            output='screen',
            parameters=[{
                'image_topic': resized_image_topic,
                'drivable_mask_topic': drivable_mask_topic,
                'lane_mask_topic': lane_mask_topic,
                'vehicle_detections_topic': detections_topic,
                'person_detections_topic': '/yolo/person/detections',
                'traffic_light_detections_topic': traffic_light_detections_topic,
                'fused_image_topic': '/perception/fused_debug_image',
                'fused_detections_topic': '/perception/fused_detections',
                'publish_fused_image': True,
                'publish_fused_detections': True,
                'queue_size': 10,
            }],
        ),

        Node(
            package='yolopv2_ros',
            executable='masked_ray_ground_projection',
            name='mask_ground_projection_node',
            output='screen',
            condition=IfCondition(enable_projection),
            arguments=[
                '--input-lane-mask-topic', lane_mask_topic,
                '--input-drivable-mask-topic', drivable_mask_topic,
                '--input-detections-topic', detections_topic,
                '--output-lane-points-topic', lane_points_topic,
                '--output-drivable-points-topic', drivable_points_topic,
                '--output-vehicle-bbox-points-topic', closest_obstacle_topic,
            ],
        ),

        # Test-only stub: behavior_node speed input when no real speed publisher exists.
        ExecuteProcess(
            condition=IfCondition(enable_speed_stub),
            cmd=['ros2', 'topic', 'pub', '-r', '10', '/vehicle/current_speed_mps', 'std_msgs/msg/Float32', '{data: 0.0}'],
            output='screen',
        ),
    ])
