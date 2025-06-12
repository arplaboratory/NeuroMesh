import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo, DeclareLaunchArgument, TimerAction, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, TextSubstitution
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def launch_setup(context):
    name = LaunchConfiguration('name').perform(context)
    agent_list = LaunchConfiguration('agent_list').perform(context)
    agent_num = LaunchConfiguration('agent_num').perform(context)
    color_raw_topic = LaunchConfiguration('color_raw_topic').perform(context)
    log_level = LaunchConfiguration('log_level').perform(context)

    remappings = []
    for agent in agent_list.split(','):
        if agent != name:
            remappings.append((f'features_{agent}', f'/{agent}/features_{agent}'))

    composable_nodes = [
    ComposableNode(
            package= 'neuromesh_platform_r2',
            namespace= name,
            name= ['vggt_neuromesh'],
            plugin='vggtNode::VggtToyImplementation',
            parameters=[{'id': name,
                        'encoder_model_name' : 'vggt_encoder',
                        'decoder_model_name' : 'vggt_decoder',
                        'encoder_cycle_length' : 3000,
                        'decoder_cycle_length' : 3000,
                        'agents': agent_list,
                        'ints_to_floats': True,
                        'vggt_decoder_output_dimensions' : "1,2,9;1,2,392,518,1;1,2,392,518;1,2,392,518,3;1,2,392,518",
                        }],

            remappings=[('camera', color_raw_topic),] + remappings,
        ),

    ComposableNode(
            package= 'tensorrt_engine',
            namespace= name,
            name=["engine", agent_num],
            plugin="tensorrt_engine_node::TensorRTEngineNode",
            parameters=[{'model_names' : 'vggt_encoder,vggt_decoder',
                        'vggt_encoder.model_path': get_package_share_directory('tensorrt_engine') + "/models/vggt_onnx_2x/vggt_image_encoder_2x.engine",
                        'vggt_encoder.input_dimensions' : "1,3,392,518",
                        'vggt_encoder.output_dimensions' : "1,1036,1024",
                         'vggt_encoder.tensor_type': "fp32",
                        'vggt_decoder.model_path': get_package_share_directory('tensorrt_engine') + "/models/vggt_onnx_2x/vggt_aggregator_2x.engine",
                        'vggt_decoder.input_dimensions' : "2,1036,1024",
                        'vggt_decoder.output_dimensions' : "1,2,9;1,2,392,518,1;1,2,392,518;1,2,392,518,3;1,2,392,518",
                         'vggt_decoder.tensor_type': "fp32",
                        }],
        ),
    ]

    return [ComposableNodeContainer(
        name='vggt_container',
        namespace=name,
        package='rclcpp_components',
        executable='component_container_mt',  # Use multi-threaded executor
        composable_node_descriptions=composable_nodes,
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        additional_env={'ROS_DOMAIN_ID': EnvironmentVariable('ROS_DOMAIN_ID', default_value='0')},
        # Use multi-threaded executor with 4 threads
        ros_arguments=['--ros-args', '--log-level', log_level, '-p', 'use_intra_process_comms:=true'],
    )]

def generate_launch_description():
    name_arg = DeclareLaunchArgument(
        name='name', default_value='khonsu',
        description=(
            'Which robot we are running'
    )
    )

    color_raw_topic_arg = DeclareLaunchArgument(
        name='color_raw_topic',
        default_value=[
            TextSubstitution(text='/'),
            LaunchConfiguration('name'),
            TextSubstitution(text='/sensors/camera_0/camera/color/image_raw')
        ],
        description='Camera topic to be remapped',
    )

    agent_num_arg = DeclareLaunchArgument(
        name='agent_num', default_value='1',
        description=(
            'Which agent we are running'
    )
    )

    agent_list_arg = DeclareLaunchArgument(
        name='agent_list', default_value='khonsu,anubis',
        description=(
            'List of all agents present (including self) - default 2 robots for VGGT'
        )
    )

    log_level_arg = DeclareLaunchArgument(
        name='log_level', default_value='INFO',
        description=(
            'Log level for all nodes (DEBUG, INFO, WARN, ERROR, FATAL)'
        )
    )

    opaque_function_action = OpaqueFunction(function=launch_setup)

    return LaunchDescription([
        name_arg,
        agent_num_arg,
        agent_list_arg,
        color_raw_topic_arg,
        log_level_arg,
        opaque_function_action,
    ])