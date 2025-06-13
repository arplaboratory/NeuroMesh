import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def launch_setup(context):
    name = LaunchConfiguration('robot_name').perform(context)
    agent_list = LaunchConfiguration('agent_list').perform(context)
    agent_num = LaunchConfiguration('agent_num').perform(context)
    color_raw_topic = LaunchConfiguration('color_raw_topic').perform(context)
    log_level = LaunchConfiguration('log_level').perform(context)
    
    # Path to config file
    config_file = os.path.join(
        get_package_share_directory('neuromesh_platform_r2'),
        'config',
        'vggt_config.yaml'
    )
    
    # Build remappings for inter-robot feature topics
    remappings = []
    for agent in agent_list.split(','):
        if agent != name:
            remappings.append((f'features_{agent}', f'/{agent}/features_{agent}'))
    
    composable_nodes = [
        # TensorRT Engine Node for Encoder
        ComposableNode(
            package='tensorrt_engine',
            namespace=name,
            name="tensorrt_encoder",
            plugin="tensorrt_engine_node::TensorRTEngineNode",
            parameters=[{
                'model_names': 'vggt_image_encoder_2x.engine',
                'vggt_image_encoder_2x.engine.model_path': os.path.join(
                    get_package_share_directory('tensorrt_engine'),
                    'models/vggt_onnx_2x/vggt_image_encoder_2x.engine'
                ),
                'vggt_image_encoder_2x.engine.input_dimensions': "1,3,392,518",
                'vggt_image_encoder_2x.engine.output_dimensions': "1,1036,1024",
                'vggt_image_encoder_2x.engine.tensor_type': "fp32",
            }],
            remappings=[
                ('tensorrt_request', 'tensorrt_request_encoder'),
                ('tensorrt_output', 'tensorrt_output_encoder')
            ],
        ),
        
        # TensorRT Engine Node for Decoder
        ComposableNode(
            package='tensorrt_engine',
            namespace=name,
            name="tensorrt_decoder",
            plugin="tensorrt_engine_node::TensorRTEngineNode",
            parameters=[{
                'model_names': 'vggt_aggregator_2x.engine',
                'vggt_aggregator_2x.engine.model_path': os.path.join(
                    get_package_share_directory('tensorrt_engine'),
                    'models/vggt_onnx_2x/vggt_aggregator_2x.engine'
                ),
                'vggt_aggregator_2x.engine.input_dimensions': "2,1036,1024",
                'vggt_aggregator_2x.engine.output_dimensions': "1,2,9;1,2,392,518,1;1,2,392,518;1,2,392,518,3;1,2,392,518",
                'vggt_aggregator_2x.engine.tensor_type': "fp32",
            }],
            remappings=[
                ('tensorrt_request', 'tensorrt_request_decoder'),
                ('tensorrt_output', 'tensorrt_output_decoder')
            ],
        ),
        # VGGT Encoder Node
        ComposableNode(
            package='neuromesh_platform_r2',
            namespace=name,
            name='vggt_encoder',
            plugin='neuromesh::VggtEncoderNode',
            parameters=[
                config_file,
                {
                    'robot_name': name,
                    'color_raw_topic': color_raw_topic,
                    'vggt.encoder.model_path': os.path.join(
                        get_package_share_directory('tensorrt_engine'),
                        'models/vggt_onnx_2x/vggt_image_encoder_2x.engine'
                    )
                }
            ],
            remappings=[('camera', color_raw_topic)],
        ),
        # VGGT Decoder Node
        ComposableNode(
            package='neuromesh_platform_r2',
            namespace=name,
            name='vggt_decoder',
            plugin='neuromesh::VggtDecoderNode',
            parameters=[
                config_file,
                {
                    'robot_name': name,
                    'vggt.robot_names': agent_list.split(','),
                    'vggt.decoder.model_path': os.path.join(
                        get_package_share_directory('tensorrt_engine'),
                        'models/vggt_onnx_2x/vggt_aggregator_2x.engine'
                    )
                }
            ],
            remappings=remappings,
        ),    
    ]
    
    return [ComposableNodeContainer(
        name='vggt_separated_container',
        namespace=name,
        package='rclcpp_components',
        executable='component_container_mt',  # Use multi-threaded executor
        composable_node_descriptions=composable_nodes,
        output='screen',
        # Use multi-threaded executor with intra-process communication
        arguments=['--ros-args', '--log-level', log_level, '-p', 'use_intra_process_comms:=true'],
    )]

def generate_launch_description():
    robot_name_arg = DeclareLaunchArgument(
        name='robot_name',
        default_value='khonsu',
        description='Which robot we are running'
    )
    
    color_raw_topic_arg = DeclareLaunchArgument(
        name='color_raw_topic',
        default_value=[
            TextSubstitution(text='/'),
            LaunchConfiguration('robot_name'),
            TextSubstitution(text='/sensors/camera_0/camera/color/image_raw')
        ],
        description='Camera topic to be remapped',
    )
    
    agent_num_arg = DeclareLaunchArgument(
        name='agent_num',
        default_value='1',
        description='Which agent we are running'
    )
    
    agent_list_arg = DeclareLaunchArgument(
        name='agent_list',
        default_value='khonsu,anubis',
        description='List of all agents present (including self) - default 2 robots for VGGT'
    )
    
    log_level_arg = DeclareLaunchArgument(
        name='log_level',
        default_value='INFO',
        description='Log level for all nodes (DEBUG, INFO, WARN, ERROR, FATAL)'
    )
    
    opaque_function_action = OpaqueFunction(function=launch_setup)
    
    return LaunchDescription([
        robot_name_arg,
        agent_num_arg,
        agent_list_arg,
        color_raw_topic_arg,
        log_level_arg,
        opaque_function_action,
    ])