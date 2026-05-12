from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


CONFIRM_TEXT = 'I_UNDERSTAND_THIS_CAN_MOVE_THE_VEHICLE'


def _as_bool(value):
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _launch_setup(context, *args, **kwargs):
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

    enable_vehicle_bridge = _as_bool(LaunchConfiguration('enable_vehicle_bridge').perform(context))
    mock_serial = _as_bool(LaunchConfiguration('mock_serial').perform(context))
    require_hardware_confirm = _as_bool(LaunchConfiguration('require_hardware_confirm').perform(context))
    hardware_confirm_text = LaunchConfiguration('hardware_confirm_text').perform(context)

    actions = []

    if _as_bool(LaunchConfiguration('enable_perception').perform(context)):
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(perception_launch),
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
            )
        )

    if _as_bool(LaunchConfiguration('enable_neuro_decision').perform(context)):
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(neuro_launch),
                launch_arguments={
                    'enable_traffic_light': LaunchConfiguration('enable_traffic_light'),
                    'lane_timeout_s': LaunchConfiguration('lane_timeout_s'),
                    'drivable_timeout_s': LaunchConfiguration('drivable_timeout_s'),
                    'obstacle_timeout_s': LaunchConfiguration('obstacle_timeout_s'),
                    'traffic_light_timeout_s': LaunchConfiguration('traffic_light_timeout_s'),
                    'speed_timeout_s': LaunchConfiguration('speed_timeout_s'),
                    'detection_timeout_s': LaunchConfiguration('detection_timeout_s'),
                }.items(),
            )
        )

    if not enable_vehicle_bridge:
        actions.append(LogInfo(msg='Hardware-safe launch: vehicle bridge disabled. No serial port will be opened.'))
        return actions

    if not mock_serial:
        if not require_hardware_confirm or hardware_confirm_text != CONFIRM_TEXT:
            return [
                LogInfo(
                    msg=(
                        'Refusing to start real MCU serial bridge. To use mock_serial:=false, pass '
                        'enable_vehicle_bridge:=true require_hardware_confirm:=true '
                        f'hardware_confirm_text:={CONFIRM_TEXT}. Run preflight checks first.'
                    )
                ),
                EmitEvent(event=Shutdown(reason='hardware confirmation missing')),
            ]

        actions.append(
            LogInfo(
                msg='Hardware confirmation accepted. Starting real MCU serial bridge; ensure the vehicle is secured.'
            )
        )
    else:
        actions.append(LogInfo(msg='Starting MCU bridge with mock_serial=true. No serial port will be opened.'))

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(vehicle_bridge_launch),
            launch_arguments={
                'port': LaunchConfiguration('mcu_port'),
                'baudrate': LaunchConfiguration('baudrate'),
                'mock_serial': LaunchConfiguration('mock_serial'),
            }.items(),
        )
    )
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('video_path', default_value='/home/subin/test.mp4'),
        DeclareLaunchArgument('enable_perception', default_value='true'),
        DeclareLaunchArgument('enable_neuro_decision', default_value='true'),
        DeclareLaunchArgument('enable_vehicle_bridge', default_value='false'),
        DeclareLaunchArgument('mock_serial', default_value='true'),
        DeclareLaunchArgument('require_hardware_confirm', default_value='false'),
        DeclareLaunchArgument('hardware_confirm_text', default_value=''),
        DeclareLaunchArgument('mcu_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('baudrate', default_value='115200'),
        DeclareLaunchArgument('lane_timeout_s', default_value='2.0'),
        DeclareLaunchArgument('drivable_timeout_s', default_value='2.0'),
        DeclareLaunchArgument('obstacle_timeout_s', default_value='2.0'),
        DeclareLaunchArgument('traffic_light_timeout_s', default_value='2.0'),
        DeclareLaunchArgument('speed_timeout_s', default_value='1.0'),
        DeclareLaunchArgument('detection_timeout_s', default_value='2.0'),
        DeclareLaunchArgument('enable_traffic_light', default_value='false'),
        DeclareLaunchArgument('enable_traffic_light_state', default_value='true'),
        DeclareLaunchArgument('use_roi_color_fallback', default_value='false'),
        DeclareLaunchArgument('traffic_light_image_topic', default_value='/camera/image_1280x720'),
        DeclareLaunchArgument('traffic_light_state_topic', default_value='/traffic_light_state'),
        DeclareLaunchArgument('yolopv2_device', default_value='cpu'),
        DeclareLaunchArgument('yolo_device', default_value='cpu'),
        DeclareLaunchArgument('enable_speed_stub', default_value='false'),
        OpaqueFunction(function=_launch_setup),
    ])
