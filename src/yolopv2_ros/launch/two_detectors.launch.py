from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    input_image_topic = LaunchConfiguration('input_image_topic', default='/camera/image_raw')
    model_name = LaunchConfiguration('model_name', default='yolov8n.pt')
    device = LaunchConfiguration('device', default='cpu')
    confidence_threshold = LaunchConfiguration('confidence_threshold', default='0.25')
    iou_threshold = LaunchConfiguration('iou_threshold', default='0.45')
    queue_size = LaunchConfiguration('queue_size', default='10')
    publish_result_image = LaunchConfiguration('publish_result_image', default='true')

    ld = LaunchDescription([
        DeclareLaunchArgument('input_image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('model_name', default_value='yolov8n.pt'),
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.25'),
        DeclareLaunchArgument('iou_threshold', default_value='0.45'),
        DeclareLaunchArgument('queue_size', default_value='10'),
        DeclareLaunchArgument('publish_result_image', default_value='true'),

        Node(
            package='yolopv2_ros',
            executable='pedestrian_detector',
            name='pedestrian_detector_node',
            output='screen',
            parameters=[{
                'input_image_topic': input_image_topic,
                'model_name': model_name,
                'device': device,
                'confidence_threshold': confidence_threshold,
                'iou_threshold': iou_threshold,
                'queue_size': queue_size,
                'publish_result_image': publish_result_image,
                'detection_topic': '/yolo/person/detections',
                'result_image_topic': '/yolo/person/result_image',
            }]
        ),

        Node(
            package='yolopv2_ros',
            executable='traffic_light_detector',
            name='traffic_light_detector_node',
            output='screen',
            parameters=[{
                'input_image_topic': input_image_topic,
                'model_name': model_name,
                'device': device,
                'confidence_threshold': confidence_threshold,
                'iou_threshold': iou_threshold,
                'queue_size': queue_size,
                'publish_result_image': publish_result_image,
                'detection_topic': '/yolo/traffic_light/detections',
                'result_image_topic': '/yolo/traffic_light/result_image',
            }]
        ),
    ])

    return ld
