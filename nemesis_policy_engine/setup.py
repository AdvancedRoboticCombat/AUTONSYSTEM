import os
from setuptools import find_packages, setup

package_name = 'nemesis_policy_engine'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), ['config/saw_policy.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ryan-ly',
    maintainer_email='walkingpresent@github.com',
    description='Flies auton_training hammer-saw policies on Nemesis',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'saw_policy_node = nemesis_policy_engine.saw_policy:main',
        ],
    },
)
