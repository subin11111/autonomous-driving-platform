import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'vehicle_serial_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='subin',
    maintainer_email='ju27586@konkuk.ac.kr',
    description='ROS2 serial bridge for MCU CMD line vehicle control',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mcu_serial_bridge = vehicle_serial_bridge.mcu_serial_bridge:main',
        ],
    },
)
