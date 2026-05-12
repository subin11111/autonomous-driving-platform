from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    perception_launch = PathJoinSubstitution([
        FindPackageShare('yolopv2_ros'),
        'launch',
        'perception_planning_bridge.launch.py',
    ])
    neuro_launch = PathJoinSubstitution([
        FindPackageShare('neuro_decision'),
        'launch',
        'neuro_decision.launch.py',
    ])
    vehicle_bridge_launch = PathJoinSubstitution([
        FindPackageShare('vehicle_serial_bridge'),
        'launch',
        'mcu_serial_bridge.launch.py',
    ])

    return LaunchDescription([
        DeclareLaunchArgument('video_path', default_value='/home/subin/test.mp4'),
        DeclareLaunchArgument('mock_serial', default_value='true'),
        DeclareLaunchArgument('enable_vehicle_bridge', default_value='true'),
        DeclareLaunchArgument('enable_perception', default_value='true'),
        DeclareLaunchArgument('enable_neuro_decision', default_value='true'),
        DeclareLaunchArgument('use_roi_color_fallback', default_value='false'),
        DeclareLaunchArgument('enable_traffic_light_state', default_value='true'),
        DeclareLaunchArgument('enable_traffic_light', default_value='false'),
        DeclareLaunchArgument('lane_timeout_s', default_value='2.0'),
        DeclareLaunchArgument('drivable_timeout_s', default_value='2.0'),
        DeclareLaunchArgument('obstacle_timeout_s', default_value='2.0'),
        DeclareLaunchArgument('traffic_light_timeout_s', default_value='2.0'),
        DeclareLaunchArgument('speed_timeout_s', default_value='1.0'),
        DeclareLaunchArgument('detection_timeout_s', default_value='2.0'),
        DeclareLaunchArgument('mcu_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('baudrate', default_value='115200'),
        DeclareLaunchArgument('traffic_light_image_topic', default_value='/camera/image_1280x720'),
        DeclareLaunchArgument('traffic_light_state_topic', default_value='/traffic_light_state'),
        DeclareLaunchArgument('yolopv2_device', default_value='cpu'),
        DeclareLaunchArgument('yolo_device', default_value='cpu'),
        DeclareLaunchArgument('enable_speed_stub', default_value='false'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(perception_launch),
            condition=IfCondition(LaunchConfiguration('enable_perception')),
            launch_arguments={
                'video_path': LaunchConfiguration('video_path'),
                'yolopv2_device': LaunchConfiguration('yolopv2_device'),
                'yolo_device': LaunchConfiguration('yolo_device'),
                'enable_traffic_light_state': LaunchConfiguration('enable_traffic_light_state'),
                'use_roi_color_fallback': LaunchConfiguration('use_roi_color_fallback'),
                'traffic_light_image_topic': LaunchConfiguration('traffic_light_image_topic'),
                'traffic_light_state_topic': LaunchConfiguration('traffic_light_state_topic'),
                'enable_speed_stub': LaunchConfiguration('enable_speed_stub'),
            }.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(neuro_launch),
            condition=IfCondition(LaunchConfiguration('enable_neuro_decision')),
            launch_arguments={
                'enable_traffic_light': LaunchConfiguration('enable_traffic_light'),
                'lane_timeout_s': LaunchConfiguration('lane_timeout_s'),
                'drivable_timeout_s': LaunchConfiguration('drivable_timeout_s'),
                'obstacle_timeout_s': LaunchConfiguration('obstacle_timeout_s'),
                'traffic_light_timeout_s': LaunchConfiguration('traffic_light_timeout_s'),
                'speed_timeout_s': LaunchConfiguration('speed_timeout_s'),
                'detection_timeout_s': LaunchConfiguration('detection_timeout_s'),
            }.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(vehicle_bridge_launch),
            condition=IfCondition(LaunchConfiguration('enable_vehicle_bridge')),
            launch_arguments={
                'port': LaunchConfiguration('mcu_port'),
                'baudrate': LaunchConfiguration('baudrate'),
                'mock_serial': LaunchConfiguration('mock_serial'),
            }.items(),
        ),
    ])
