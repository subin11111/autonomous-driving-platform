import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # 1. neuro_decision 패키지의 'share' 폴더 위치를 찾습니다. (설치된 config 파일을 찾기 위해)
    pkg_dir = get_package_share_directory('neuro_decision')
    
    # 2. config 폴더 안에 있는 'ekf.yaml' 지시서 파일의 경로를 완성합니다.
    config_file = os.path.join(pkg_dir, 'config', 'ekf.yaml')

    # 3. 각 노드 실행 준비
    behavior_node = Node(
        package='neuro_decision',
        executable='behavior_node', # (setup.py에 console_scripts로 behavior_node로 등록되어 있다고 가정합니다)
        name='behavior_node',
        output='screen'
    )

    pure_pursuit_node = Node(
        package='neuro_decision',
        executable='pure_pursuit', # (executable name is 'pure_pursuit' but file is 'pure_pursuit_node.py')
        name='pure_pursuit_node',
        output='screen',
        parameters=[{
            'wheelbase': ParameterValue(LaunchConfiguration('wheelbase'), value_type=float),
            'max_steering_angle_rad': ParameterValue(LaunchConfiguration('max_steering_angle_rad'), value_type=float),
        }]
    )

    speed_control_node = Node(
        package='neuro_decision',
        executable='speed_control',
        name='speed_control_node',
        output='screen'
    )

    # 4. robot_localization 패키지의 EKF(확장 칼만 필터) 노드를 실행 준비
    localization_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[config_file] # ⭐ 방금 위에서 완성한 경로로 지시서 전달!
    )

    return LaunchDescription([
        # 실행 모드를 sim으로 기본 설정 (나중에 real로 바꿀 수 있게 준비)
        DeclareLaunchArgument('mode', default_value='sim', description='실행 모드: sim 또는 real'),
        DeclareLaunchArgument('wheelbase', default_value='2.9', description='차량 휠베이스 [m]'),
        DeclareLaunchArgument('max_steering_angle_rad', default_value='1.22', description='최대 조향각 [rad]'),
        
        # 준비된 4개의 노드를 일괄 가동!
        behavior_node,
        pure_pursuit_node,
        speed_control_node,
        localization_node,
    ])