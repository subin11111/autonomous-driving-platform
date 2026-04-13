import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'neuro_decision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='junghun',
    maintainer_email='junghun@todo.todo',
    description='Decision and control package for autonomous driving in ROS 2/CARLA.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'behavior_node = neuro_decision.behavior_node:main',
            'pure_pursuit = neuro_decision.pure_pursuit_node:main',
            'speed_control = neuro_decision.speed_control_node:main',
        ],
    },
)
