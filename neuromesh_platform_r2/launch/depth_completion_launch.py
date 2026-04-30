from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'robot_name',
            default_value='robot',
            description='Name of the robot'
        ),
        DeclareLaunchArgument(
            'rgbd_topic',
            default_value='/camera/rgbd',
            description='RGBD combined message topic from RealSense'
        ),
        DeclareLaunchArgument(
            'vggt_pointcloud_topic',
            default_value='/vggt/pointcloud',
            description='Point cloud topic from VGGT neural network'
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='/camera/depth/camera_info',
            description='Camera info topic for intrinsics'
        ),
        DeclareLaunchArgument(
            'output_pointcloud_topic',
            default_value='/depth_completion/pointcloud',
            description='Output completed point cloud topic'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Logging level'
        ),
        DeclareLaunchArgument(
            'engine_width',
            default_value='392',
            description='Neural network output width'
        ),
        DeclareLaunchArgument(
            'engine_height',
            default_value='518',
            description='Neural network output height'
        ),
        
        # Depth completion node
        Node(
            package='neuromesh_platform_r2',
            executable='depth_completion_node',
            name='depth_completion_node',
            output='screen',
            parameters=[{
                'rgbd_topic': LaunchConfiguration('rgbd_topic'),
                'vggt_pointcloud_topic': LaunchConfiguration('vggt_pointcloud_topic'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'output_pointcloud_topic': LaunchConfiguration('output_pointcloud_topic'),
                'depth_epsilon': 0.001,
                'speckle_window_size': 100,
                'speckle_range': 4,
                'engine_width': LaunchConfiguration('engine_width'),
                'engine_height': LaunchConfiguration('engine_height'),
            }],
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        ),
    ])