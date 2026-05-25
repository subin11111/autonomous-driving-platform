from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('baudrate', default_value='115200'),
        DeclareLaunchArgument('mock_serial', default_value='false'),
        DeclareLaunchArgument('serial_timeout', default_value='0.01'),
        DeclareLaunchArgument('write_timeout', default_value='0.01'),
        DeclareLaunchArgument('command_publish_period_s', default_value='0.05'),
        DeclareLaunchArgument('desired_speed_topic', default_value='/desired_speed'),
        DeclareLaunchArgument('desired_steering_angle_deg_topic', default_value='/desired_steering_angle_deg'),
        DeclareLaunchArgument('behavior_state_topic', default_value='/behavior_state'),
        DeclareLaunchArgument('max_abs_speed_mps', default_value='1.40'),
        DeclareLaunchArgument('max_abs_steering_deg', default_value='20.0'),
        DeclareLaunchArgument('allow_reverse', default_value='true'),
        DeclareLaunchArgument('speed_input_timeout_s', default_value='0.5'),
        DeclareLaunchArgument('steering_input_timeout_s', default_value='0.5'),
        DeclareLaunchArgument('state_input_timeout_s', default_value='0.5'),
        Node(
            package='vehicle_serial_bridge',
            executable='mcu_serial_bridge',
            name='mcu_serial_bridge',
            output='screen',
            parameters=[{
                'port': LaunchConfiguration('port'),
                'baudrate': ParameterValue(LaunchConfiguration('baudrate'), value_type=int),
                'mock_serial': ParameterValue(LaunchConfiguration('mock_serial'), value_type=bool),
                'serial_timeout': ParameterValue(LaunchConfiguration('serial_timeout'), value_type=float),
                'write_timeout': ParameterValue(LaunchConfiguration('write_timeout'), value_type=float),
                'command_publish_period_s': ParameterValue(
                    LaunchConfiguration('command_publish_period_s'),
                    value_type=float,
                ),
                'desired_speed_topic': LaunchConfiguration('desired_speed_topic'),
                'desired_steering_angle_deg_topic': LaunchConfiguration('desired_steering_angle_deg_topic'),
                'behavior_state_topic': LaunchConfiguration('behavior_state_topic'),
                'max_abs_speed_mps': ParameterValue(LaunchConfiguration('max_abs_speed_mps'), value_type=float),
                'max_abs_steering_deg': ParameterValue(
                    LaunchConfiguration('max_abs_steering_deg'),
                    value_type=float,
                ),
                'allow_reverse': ParameterValue(LaunchConfiguration('allow_reverse'), value_type=bool),
                'speed_input_timeout_s': ParameterValue(
                    LaunchConfiguration('speed_input_timeout_s'),
                    value_type=float,
                ),
                'steering_input_timeout_s': ParameterValue(
                    LaunchConfiguration('steering_input_timeout_s'),
                    value_type=float,
                ),
                'state_input_timeout_s': ParameterValue(
                    LaunchConfiguration('state_input_timeout_s'),
                    value_type=float,
                ),
            }],
        ),
    ])
