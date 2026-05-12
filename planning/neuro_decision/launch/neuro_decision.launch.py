from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    behavior_node = Node(
        package='neuro_decision',
        executable='behavior_node',
        name='behavior_node',
        output='screen',
        parameters=[{
            'speed_topic': LaunchConfiguration('speed_topic'),
            'max_desired_speed_mps': ParameterValue(LaunchConfiguration('max_desired_speed_mps'), value_type=float),
            'desired_speed_straight_mps': ParameterValue(LaunchConfiguration('desired_speed_straight_mps'), value_type=float),
            'desired_speed_gentle_turn_mps': ParameterValue(LaunchConfiguration('desired_speed_gentle_turn_mps'), value_type=float),
            'desired_speed_sharp_turn_mps': ParameterValue(LaunchConfiguration('desired_speed_sharp_turn_mps'), value_type=float),
            'enable_obstacle_avoidance': ParameterValue(LaunchConfiguration('enable_obstacle_avoidance'), value_type=bool),
            'enable_lane_change': ParameterValue(LaunchConfiguration('enable_lane_change'), value_type=bool),
            'enable_traffic_light': ParameterValue(LaunchConfiguration('enable_traffic_light'), value_type=bool),
            'enable_stopline': ParameterValue(LaunchConfiguration('enable_stopline'), value_type=bool),
            'enable_cutin_detection': ParameterValue(LaunchConfiguration('enable_cutin_detection'), value_type=bool),
            'require_drivable_for_lane_keeping': ParameterValue(LaunchConfiguration('require_drivable_for_lane_keeping'), value_type=bool),
        }],
    )

    steering_command_node = Node(
        package='neuro_decision',
        executable='steering_command_node',
        name='steering_command_node',
        output='screen',
        parameters=[{
            'wheelbase': ParameterValue(LaunchConfiguration('wheelbase'), value_type=float),
            'max_steering_angle_deg': ParameterValue(LaunchConfiguration('max_steering_angle_deg'), value_type=float),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('speed_topic', default_value='/vehicle/current_speed_mps', description='현재 차량 속도 토픽 (std_msgs/Float32, m/s)'),
        DeclareLaunchArgument('max_desired_speed_mps', default_value='1.40', description='목표 속도 상한 [m/s]'),
        DeclareLaunchArgument('desired_speed_straight_mps', default_value='1.10', description='직선 목표 속도 [m/s]'),
        DeclareLaunchArgument('desired_speed_gentle_turn_mps', default_value='0.75', description='완만한 곡선 목표 속도 [m/s]'),
        DeclareLaunchArgument('desired_speed_sharp_turn_mps', default_value='0.45', description='급한 곡선 목표 속도 [m/s]'),
        DeclareLaunchArgument('enable_obstacle_avoidance', default_value='false', description='Local Lattice 회피 활성화 여부'),
        DeclareLaunchArgument('enable_lane_change', default_value='false', description='차선 변경 활성화 여부'),
        DeclareLaunchArgument('enable_traffic_light', default_value='false', description='신호등 판단 활성화 여부'),
        DeclareLaunchArgument('enable_stopline', default_value='false', description='정지선 판단 활성화 여부'),
        DeclareLaunchArgument('enable_cutin_detection', default_value='false', description='Cut-in 감지 활성화 여부'),
        DeclareLaunchArgument('require_drivable_for_lane_keeping', default_value='false', description='lane keeping에 drivable area 필수 여부'),
        DeclareLaunchArgument('wheelbase', default_value='0.95', description='차량 휠베이스 [m]'),
        DeclareLaunchArgument('max_steering_angle_deg', default_value='20.0', description='최대 조향각 [deg]'),
        behavior_node,
        steering_command_node,
    ])
