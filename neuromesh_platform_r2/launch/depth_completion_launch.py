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
            'depth_robot1_topic',
            default_value='/depth_robot1',
            description='Normalized depth image from VGGT decoder (relative to robot_name)'
        ),
        DeclareLaunchArgument(
            'decoder_sync_depth_topic',
            default_value='/decoder_sync_depth',
            description='Metric depth from sensor synchronized with decoder output (relative to robot_name)'
        ),
        DeclareLaunchArgument(
            'pointcloud_rgb_topic',
            default_value='/pointcloud_rgb',
            description='RGB pointcloud from VGGT decoder (relative to robot_name)'
        ),
        DeclareLaunchArgument(
            'pointcloud_current_topic',
            default_value='/pointcloud_current',
            description='Subsampled pointcloud from VGGT decoder (relative to robot_name)'
        ),
        DeclareLaunchArgument(
            'pointcloud_neighbor_topic',
            default_value='/pointcloud_neighbor',
            description='Neighbor robot pointcloud (relative to robot_name)'
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='/camera_info',
            description='Camera info topic for depth image intrinsics (relative to robot_name)'
        ),
        DeclareLaunchArgument(
            'output_pointcloud_topic',
            default_value='/pointcloud_current_rgb_scaled',
            description='Output scaled RGB pointcloud topic (relative to robot_name)'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Logging level'
        ),
        DeclareLaunchArgument(
            'min_scale',
            default_value='0.1',
            description='Minimum allowed scale factor'
        ),
        DeclareLaunchArgument(
            'max_scale',
            default_value='10.0',
            description='Maximum allowed scale factor'
        ),
        DeclareLaunchArgument(
            'min_shift',
            default_value='-5.0',
            description='Minimum allowed shift value'
        ),
        DeclareLaunchArgument(
            'max_shift',
            default_value='5.0',
            description='Maximum allowed shift value'
        ),
        DeclareLaunchArgument(
            'min_valid_points',
            default_value='100',
            description='Minimum number of valid points for scale recovery'
        ),
        DeclareLaunchArgument(
            'max_time_diff',
            default_value='0.1',
            description='Maximum time difference for message synchronization (seconds)'
        ),
        DeclareLaunchArgument(
            'voxel_leaf_size',
            default_value='0.02',
            description='Voxel leaf size for pointcloud subsampling (meters)'
        ),
        DeclareLaunchArgument(
            'enable_subsampling',
            default_value='false',
            description='Enable voxel grid subsampling of output pointclouds'
        ),
        
        # Depth completion node
        Node(
            package='neuromesh_platform_r2',
            executable='depth_completion_node',
            name='depth_completion_node',
            output='screen',
            parameters=[{
                'robot_name': LaunchConfiguration('robot_name'),
                'depth_robot1_topic': LaunchConfiguration('depth_robot1_topic'),
                'decoder_sync_depth_topic': LaunchConfiguration('decoder_sync_depth_topic'),
                'pointcloud_rgb_topic': LaunchConfiguration('pointcloud_rgb_topic'),
                'pointcloud_current_topic': LaunchConfiguration('pointcloud_current_topic'),
                'pointcloud_neighbor_topic': LaunchConfiguration('pointcloud_neighbor_topic'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'output_pointcloud_topic': LaunchConfiguration('output_pointcloud_topic'),
                'depth_epsilon': 0.001,
                'min_scale': LaunchConfiguration('min_scale'),
                'max_scale': LaunchConfiguration('max_scale'),
                'min_shift': LaunchConfiguration('min_shift'),
                'max_shift': LaunchConfiguration('max_shift'),
                'min_valid_points': LaunchConfiguration('min_valid_points'),
                'max_time_diff': LaunchConfiguration('max_time_diff'),
                'voxel_leaf_size': LaunchConfiguration('voxel_leaf_size'),
                'enable_subsampling': LaunchConfiguration('enable_subsampling'),
            }],
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        ),
    ])