import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 우리가 방금 만든 ekf.yaml 지시서 파일의 위치를 찾습니다.
    pkg_dir = get_package_share_directory('neuro_decision')
    config_file = os.path.join(pkg_dir, 'config', 'ekf.yaml')

    # 2. robot_localization 패키지의 ekf_node를 실행할 준비를 합니다.
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[config_file] # 지시서 전달!
    )

    return LaunchDescription([
        ekf_node
    ])