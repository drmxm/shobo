# perception.launch.py
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    share_dir = get_package_share_directory('shobo_bringup')
    rgb_cfg = os.path.join(share_dir, 'config', 'rgb_cam.yaml')

    rgb_device = os.environ.get('RGB_DEV')
    if not rgb_device:
        # Prefer the first present UVC node; fall back to /dev/video0 so errors are obvious.
        for candidate in ('/dev/video1', '/dev/video0', '/dev/video2'):
            if os.path.exists(candidate):
                rgb_device = candidate
                break
        else:
            rgb_device = '/dev/video0'

    ir_cfg  = os.path.join(share_dir, 'config', 'ir_cam.yaml')
    detectors_cfg = os.path.join(share_dir, 'config', 'detectors.yaml')
    engine_path = os.environ.get('YOLO_ENGINE', '/work/ros2_ws/yolov8n.engine')
    publish_annotated = os.environ.get('PUBLISH_ANNOTATED', 'true').lower() not in ('0', 'false', 'no')

    return LaunchDescription([
        Node(
            package='shobo_sensors', executable='uvc_cam_node', name='rgb_cam',
            parameters=[rgb_cfg, {'rgb.device': rgb_device}],
            remappings=[('image_raw','/sensors/rgb/image_raw'),
                        ('camera_info','/sensors/rgb/camera_info')]
        ),
        Node(
            package='shobo_sensors', executable='csi_cam_node', name='ir_cam',
            parameters=[ir_cfg],
            remappings=[('image_raw','/sensors/ir/image_raw'),
                        ('camera_info','/sensors/ir/camera_info')]
        ),
        Node(
            package='shobo_detectors', executable='detector_node', name='trt_detector_rgb',
            parameters=[detectors_cfg, {
              'engine_path': engine_path,
              'publish_annotated': publish_annotated
            }]
        ),
        Node(
            package='shobo_detectors', executable='detector_node', name='trt_detector_ir',
            parameters=[detectors_cfg, {
              'engine_path': engine_path,
              'publish_annotated': publish_annotated,
              'input_topic': '/sensors/ir/image_raw'
            }]
        )
    ])
