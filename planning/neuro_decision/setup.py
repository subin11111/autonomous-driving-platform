from setuptools import find_packages, setup

package_name = 'neuro_decision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='junghun',
    maintainer_email='junghun@example.com',
    description='Neuro decision nodes',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'behavior_node = neuro_decision.behavior_node:main',
            'steering_command_node = neuro_decision.steering_command_node:main',
        ],
    },
)