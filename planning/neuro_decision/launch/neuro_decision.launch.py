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
        executable='pure_pursuit_node', # 수정: 'pure_pursuit'에서 'pure_pursuit_node'로
        name='pure_pursuit_node',
        output='screen',
        parameters=[{
            'wheelbase': ParameterValue(LaunchConfiguration('wheelbase'), value_type=float),
            'max_steering_angle_rad': ParameterValue(LaunchConfiguration('max_steering_angle_rad'), value_type=float),
        }]
    )

    speed_control_node = Node(
        package='neuro_decision',
        executable='speed_control_node', # 수정: 'speed_control'에서 'speed_control_node'로
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

    # 5. Perception 노드: YOLO 기반 차선/도로 감지
    perception_inference_node = Node(
        package='yolopv2_ros',
        executable='perception_inference_node',
        name='perception_inference_node',
        output='screen',
        arguments=[]
    )

    # 6. Perception 노드: 2D 마스크를 3D 포인트로 변환
    masked_ray_ground_projection_node = Node(
        package='yolopv2_ros',
        executable='masked_ray_ground_projection',
        name='masked_ray_ground_projection_node',
        output='screen',
        arguments=[]
    )

    return LaunchDescription([
        # 실행 모드를 sim으로 기본 설정 (나중에 real로 바꿀 수 있게 준비)
        DeclareLaunchArgument('mode', default_value='sim', description='실행 모드: sim 또는 real'),
        DeclareLaunchArgument('wheelbase', default_value='2.9', description='차량 휠베이스 [m]'),
        DeclareLaunchArgument('max_steering_angle_rad', default_value='1.22', description='최대 조향각 [rad]'),
        
        # Perception 노드 먼저 시작 (다른 노드들의 입력 역할)
        perception_inference_node,
        masked_ray_ground_projection_node,
        
        # 제어 노드
        behavior_node,
        pure_pursuit_node,
        speed_control_node,
        localization_node,
    ])