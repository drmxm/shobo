from setuptools import setup
package_name='shobo_bringup'
setup(
  name=package_name, version='0.1.0', packages=[package_name],
  data_files=[
    ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
    (f'share/{package_name}', ['package.xml']),
    (f'share/{package_name}/launch', ['launch/perception.launch.py']),
    (f'share/{package_name}/config', ['config/rgb_cam.yaml','config/ir_cam.yaml']),
  ],
  zip_safe=True,
)
