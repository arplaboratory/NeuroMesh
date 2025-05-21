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

    pos1_yaml_path = os.path.join(get_package_share_directory('neuromesh_platform_r2'), 'config', 'pos1.yaml')
    pos2_yaml_path = os.path.join(get_package_share_directory('neuromesh_platform_r2'), 'config', 'pos2.yaml')

    remappings = []
    for agent in agent_list.split(','):
        if agent != name:
            remappings.append((f'features_{agent}', f'/{agent}/features_{agent}'))

    composable_nodes = [
    ComposableNode(
            package= 'neuromesh_platform_r2',
            namespace= name,
            name= ['neuromesh'],
            plugin='neuromeshNode::ToyImplementation',
            parameters=[{'id': name,
                        'encoder_model_name' : 'dust3r_encoder',
                        'decoder_model_name' : 'dust3r_decoder',
                        'encoder_cycle_length' : 3000,
                        'decoder_cycle_length' : 3000,
                        'agents': agent_list,
                        'ints_to_floats': True,
                        'pos1_yaml_path': pos1_yaml_path,
                        'pos2_yaml_path': pos2_yaml_path,
                        'dust3r_decoder_output_dimensions' : "2,384,512,3;2,384,512;2,384,512,3;2,384,512",
                        }],

            remappings=[('camera', color_raw_topic),] + remappings,
        ),

    ComposableNode(
            package= 'tensorrt_engine',
            namespace= name,
            name=["engine", agent_num],
            plugin="tensorrt_engine_node::TensorRTEngineNode",
            parameters=[{'model_names' : 'dust3r_encoder,dust3r_decoder',
                        'dust3r_encoder.model_path': get_package_share_directory('tensorrt_engine') + "/models/dust3r_encoder_single_mini_params.trt",
                        'dust3r_encoder.input_dimensions' : "1,3,512,384",
                        'dust3r_encoder.output_dimensions' : "1,768,1024",
                         'dust3r_encoder.tensor_type': "fp32",
                        'dust3r_decoder.model_path': get_package_share_directory('tensorrt_engine') + "/models/dust3r_decoder_tensor_params.trt",
                        'dust3r_decoder.input_dimensions' : "2,768,1024;2,768,1024;2,768,2;2,768,2",
                        'dust3r_decoder.output_dimensions' : "2,384,512,3;2,384,512;2,384,512,3;2,384,512",
                         'dust3r_decoder.tensor_type': "fp32",
                        }],
        ),
    ]

    return [ComposableNodeContainer(
        name='neuromesh_container',
        namespace=name,
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=composable_nodes,
        output='screen',
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
        name='agent_list', default_value='khonsu,anubis,ra',
        description=(
            'List of all agents present (including self)'
        )
    )

    opaque_function_action = OpaqueFunction(function=launch_setup)

    return LaunchDescription([
        name_arg,
        agent_num_arg,
        agent_list_arg,
        color_raw_topic_arg,
        opaque_function_action,
    ])
