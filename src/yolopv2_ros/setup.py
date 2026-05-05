import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'yolopv2_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='subin',
    maintainer_email='ju27586@konkuk.ac.kr',
    description='Ultralytics YOLO pretrained detectors for ROS2 (pedestrian, traffic light)',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'perception_inference = yolopv2_ros.perception_inference_node:main',
            'perception_inference_node = yolopv2_ros.perception_inference_node:main',
            'image_resize = yolopv2_ros.image_resize_node:main',
            'video_to_topic = yolopv2_ros.video_to_topic:main',
            'masked_ray_ground_projection = yolopv2_ros.masked_ray_ground_projection:main',
            'pedestrian_detector = yolopv2_ros.pedestrian_detector_node:main',
            'traffic_light_detector = yolopv2_ros.traffic_light_detector_node:main',
            'fusion_visualizer = yolopv2_ros.fusion_visualizer_node:main',
        ],
    },
)
