from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('baudrate', default_value='115200'),
        DeclareLaunchArgument('cmd_topic', default_value='/cmd_vel'),
        DeclareLaunchArgument('linear_deadband', default_value='0.05'),
        DeclareLaunchArgument('angular_deadband', default_value='0.05'),
        DeclareLaunchArgument('watchdog_timeout', default_value='0.5'),
        DeclareLaunchArgument('serial_timeout', default_value='0.01'),
        DeclareLaunchArgument('write_timeout', default_value='0.01'),
        DeclareLaunchArgument('send_center_command', default_value='true'),
        DeclareLaunchArgument('mock_serial', default_value='false'),
        Node(
            package='vehicle_serial_bridge',
            executable='mcu_serial_bridge',
            name='mcu_serial_bridge',
            output='screen',
            parameters=[{
                'port': LaunchConfiguration('port'),
                'baudrate': ParameterValue(LaunchConfiguration('baudrate'), value_type=int),
                'cmd_topic': LaunchConfiguration('cmd_topic'),
                'linear_deadband': ParameterValue(LaunchConfiguration('linear_deadband'), value_type=float),
                'angular_deadband': ParameterValue(LaunchConfiguration('angular_deadband'), value_type=float),
                'watchdog_timeout': ParameterValue(LaunchConfiguration('watchdog_timeout'), value_type=float),
                'serial_timeout': ParameterValue(LaunchConfiguration('serial_timeout'), value_type=float),
                'write_timeout': ParameterValue(LaunchConfiguration('write_timeout'), value_type=float),
                'send_center_command': ParameterValue(LaunchConfiguration('send_center_command'), value_type=bool),
                'mock_serial': ParameterValue(LaunchConfiguration('mock_serial'), value_type=bool),
            }],
        ),
    ])