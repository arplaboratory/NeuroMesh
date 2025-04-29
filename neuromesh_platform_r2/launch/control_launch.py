import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
#import launch_ros.actions
#import launch
from launch.actions import LogInfo, DeclareLaunchArgument, TimerAction
from launch.substitutions import EnvironmentVariable
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    agent_num = LaunchConfiguration('agent_num')
    agent_num_arg = DeclareLaunchArgument(
        name='agent_num', default_value='2',
        description=(
            'Which agent we are running'
    )
    )

    agent_list = LaunchConfiguration('agent_list')
    agent_list_arg = DeclareLaunchArgument(
        name='agent_list', default_value='agent1,agent2,agent3',
        description=(
            'List of all agents present (including self)'
        )
    )

    feature_subscribe_topic = LaunchConfiguration('feature_subscribe_topic')
    feature_subscribe_topic_arg = DeclareLaunchArgument(
            name='feature_subscribe_topic',
            default_value='/features_agent2_local2',
            description='Remapped topic name'
        ),

    composable_nodes = [

    ComposableNode(
            package= 'neuromesh_platform_r2',
            namespace= 'khonsu',
            name= ['agent', agent_num],
            plugin='ControlneuromeshNode::ControlImplementation',
            parameters=[{'id': ["agent", agent_num],
                        'encoder_model_name' : 'encoder',
                        'decoder_model_name' : 'gnn_post_decoder',
                        'encoder_cycle_length' : 100,
                        'decoder_cycle_length' : 100,
                        'output_topic': ["agent", agent_num, "_gnn_output"],
                        'agents': agent_list,
                        'goal_x': 0.0,
                        'goal_y': 2.0,
                        'goal_z': 0.0}],

            remappings=[('odom_position_topic', '/khonsu/odom'),
                        ('odom_velocity_topic', '/khonsu/odom'),
                        #('features_agent1', ['features_agent1_local', agent_num]), #temporary
                        ('features_agent1', '/thoth/features_agent1'),
                        ('features_agent2', '/khonsu/features_agent2'),
                        ('features_agent3', '/mau/features_agent3'),
                        ('robot_velocity', '/khonsu/cmd_vel/ext')],
        ),

    ComposableNode(
            package= 'tensorrt_engine',
            namespace= 'khonsu',
            name=["engine", agent_num],
            plugin="tensorrt_engine_node::TensorRTEngineNode",
            parameters=[{'model_names' : 'encoder,gnn_post_decoder',
                        'encoder.model_path': get_package_share_directory('tensorrt_engine') + "/models/encoder_control.trt",
                        'encoder.input_dimensions' : "1,8",
                        'encoder.output_dimensions' : "1,32",
                         'encoder.tensor_type': "fp32",
                        'gnn_post_decoder.model_path': get_package_share_directory('tensorrt_engine') + "/models/gnn_post_combined_control.trt",
                        'gnn_post_decoder.input_dimensions' : "1,32;1,32",
                        'gnn_post_decoder.output_dimensions' : "1,4",
                         'gnn_post_decoder.tensor_type': "fp32",
                        }],

            remappings=[('tensorrt_input', ['tensorrt_input', agent_num]),
                        ('tensorrt_output', ['tensorrt_output', agent_num])],
        )
    ]

    return LaunchDescription([
        agent_num_arg,
        agent_list_arg,

        # image processing container
        ComposableNodeContainer(
            name='neuromesh_container2_khonsu',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            #executable='component_container_isolated',
            composable_node_descriptions=composable_nodes,
            output='screen',
            )
    ])
