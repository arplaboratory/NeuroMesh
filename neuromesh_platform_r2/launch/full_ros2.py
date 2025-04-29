import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
import launch_ros.actions
import launch
from launch.actions import LogInfo, DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.substitutions import LaunchConfiguration



def generate_launch_description():

    agent_num = LaunchConfiguration('agent_num')
    agent_num_arg = DeclareLaunchArgument(
        name='agent_num', default_value='1',
        description=(
            'Which agent we are running'
        )
    )

    agent_list = LaunchConfiguration('agent_list')
    agent_list_arg = DeclareLaunchArgument(
        name='agent_list', default_value='agent1,agent2',
        description=(
            'List of all agents present (including self)'
        )
    )

    composable_nodes = [
        ComposableNode(
            package='image_proc',
            plugin='image_proc::ResizeNode',
            name='resizer_node',
            remappings=[
                ('image/image_raw', ['/',  EnvironmentVariable('USER'), '/cam1/color/image_raw']),
                ('image/camera_info', ['/', EnvironmentVariable('USER'), '/cam1/color/camera_info']),
                ('resize/image_raw', ['camera',agent_num])
            ],
            parameters=[{'height': 320, 'width': 1024, 'use_scale': False}],
        )
    ]

    return LaunchDescription([
        agent_num_arg,
        agent_list_arg,

        launch_ros.actions.Node(
            package='neuromesh_platform_r2', executable='toy_implementation', name=["agent", agent_num], output='screen',
            parameters=[{'id': ["agent", agent_num],
                        'output_topic': ["agent", agent_num, "_gnn_output"],
                        'agents': agent_list}],

            remappings=[('camera', ['camera', agent_num]),
                        ('tensorrt_input', ['tensorrt_input', agent_num]),
                        ('tensorrt_output', ['tensorrt_output', agent_num])]),

        launch_ros.actions.Node(
            package='tensorrt_engine', executable='tensorrt_engine', name=["engine", agent_num], output='screen',
            parameters=[{'model_path': get_package_share_directory('tensorrt_engine') + "/models/depth_anything_vitb14.trt",
                        'input_dimensions' : "1,3,320,1024",
                        'output_dimensions' : "1,1,320,1024",
                        'tensor_type': "fp32",
                        }],

            remappings=[('tensorrt_input', ['tensorrt_input', agent_num]),
                        ('tensorrt_output', ['tensorrt_output', agent_num])]),


        launch_ros.actions.Node(
            package='realsense2_camera', executable='realsense2_camera_node', name=["cam", agent_num], output='screen',

            remappings=[
                ('/color/image_raw', ['/',  EnvironmentVariable('USER'), '/cam1/color/image_raw']),
                ('/color/camera_info', ['/', EnvironmentVariable('USER'), '/cam1/color/camera_info'])
            ],),



        # image processing container
        ComposableNodeContainer(
            name='image_proc_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=composable_nodes,
            output='screen'
        )
    ])
