from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='shobo_sensors', executable='uvc_cam_node', name='rgb_cam',
            parameters=['config/rgb_cam.yaml'],
            remappings=[('image_raw','/sensors/rgb/image_raw'),
                        ('camera_info','/sensors/rgb/camera_info')]
        ),
        Node(
            package='shobo_sensors', executable='csi_cam_node', name='ir_cam',
            parameters=['config/ir_cam.yaml'],
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
