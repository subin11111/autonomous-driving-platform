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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='subin',
    maintainer_email='ju27586@konkuk.ac.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'perception_inference_node = yolopv2_ros.perception_inference_node:main',
            'video_to_topic = yolopv2_ros.video_to_topic:main',
            'masked_ray_ground_projection = yolopv2_ros.masked_ray_ground_projection:main',
        ],
    },
)
