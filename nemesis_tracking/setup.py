from setuptools import find_packages, setup

package_name = 'nemesis_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='ryan-ly',
    maintainer_email='walkingpresent@github.com',
    description='Kalman fusion tracking for Nemesis',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'kalman_node = nemesis_tracking.kalman_node:main',
        ],
    },
)
