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
    )

    steering_command_node = Node(
        package='neuro_decision',
        executable='steering_command_node',
        name='steering_command_node',
        output='screen',
        parameters=[{
            'wheelbase': ParameterValue(LaunchConfiguration('wheelbase'), value_type=float),
            'max_steering_angle_rad': ParameterValue(LaunchConfiguration('max_steering_angle_rad'), value_type=float),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('wheelbase', default_value='0.95', description='차량 휠베이스 [m]'),
        DeclareLaunchArgument('max_steering_angle_rad', default_value='0.75', description='최대 조향각 [rad]'),
        behavior_node,
        steering_command_node,
    ])