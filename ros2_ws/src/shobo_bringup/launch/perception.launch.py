import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    share_dir = get_package_share_directory('shobo_bringup')
    rgb_cfg = os.path.join(share_dir, 'config', 'rgb_cam.yaml')
    ir_cfg  = os.path.join(share_dir, 'config', 'ir_cam.yaml')
    return LaunchDescription([
        Node(
            package='shobo_sensors', executable='uvc_cam_node', name='rgb_cam',
            parameters=[rgb_cfg],
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
            package='shobo_detectors', executable='detector_node', name='trt_detector',
            parameters=[{
              'input_topic': '/sensors/rgb/image_raw',
              'annotated_topic': '/perception/annotated',
              'detections_topic': '/perception/detections',
              'engine_path': '/work/ros2_ws/yolov8n.engine',
              'input_binding': 'images',
              'output_binding': 'output0',
              'conf_th': 0.35,
              'iou_th': 0.50
            }]
        )
    ])
