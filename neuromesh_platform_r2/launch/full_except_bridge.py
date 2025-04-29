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

    pos1_yaml_path = os.path.join(get_package_share_directory('neuromesh_platform_r2'), 'config', 'pos1.yaml')
    pos2_yaml_path = os.path.join(get_package_share_directory('neuromesh_platform_r2'), 'config', 'pos2.yaml')

    agent_num = LaunchConfiguration('agent_num')
    agent_num_arg = DeclareLaunchArgument(
        name='agent_num', default_value='1',
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
            default_value='/features_agent1_local1',
            description='Remapped topic name'
        ),

    composable_nodes = [
    #Realsense driver node
    #ComposableNode(
    #    namespace='camera',
    #    package='realsense2_camera',
    #    plugin='realsense2_camera::RealSenseNodeFactory',
    #    parameters=[config_file],
    #    ),

    # ComposableNode(
    #         package='image_proc',
    #         namespace='race13',
    #         plugin='image_proc::ResizeNode',
    #         name=['resizer_node', agent_num],
    #         remappings=[
    #             ('image/image_raw', ['/race13/color/image_raw']),#,  EnvironmentVariable('USER'), '/cam1/color/image_raw']),
    #             ('image/camera_info', ['/race13/color/camera_info',]),# EnvironmentVariable('USER'), '/cam1/color/camera_info']),
    #             ('resize/image_raw', ['camera', agent_num])
    #         ],
    #         parameters=[{'height': 512, 'width': 384, 'use_scale': False}],
    #     ),

        # ComposableNode(
        #    package= 'neuromesh_platform_r2',
        #    namespace= '',
        #    name=['agent', agent_num],
        #    plugin='qos_node::CameraInfoRepublisher',
        #    parameters=[{'qos_reliability': 'reliable',
        #                 'qos_durability': 'transient_local'}]
        #    ),

    ComposableNode(
            package= 'neuromesh_platform_r2',
            namespace= 'khonsu',
            name= ['agent', agent_num],
            plugin='neuromeshNode::ToyImplementation',
            parameters=[{'id': ["agent", agent_num],
                        'encoder_model_name' : 'dust3r_encoder',
                        'decoder_model_name' : 'dust3r_decoder',
                        'encoder_cycle_length' : 3500,
                        'decoder_cycle_length' : 3500,
                        'output_topic': ["agent", agent_num, "_gnn_output"],
                        'agents': agent_list,
                        'ints_to_floats': True,
                        'pos1_yaml_path': pos1_yaml_path,
                        'pos2_yaml_path': pos2_yaml_path,
                        'dust3r_decoder_output_dimensions' : "2,384,512,3;2,384,512;2,384,512,3;2,384,512",
                        # 'tensor_batch_size': 1,
                        # 'tensor_channels': 3,
                        # 'tensor_height': 220,
                        # 'tensor_width': 322,
                        #'output_qos_profile_': 'SYSTEM_DEFAULT',
                        }],

            remappings=[('camera', '/khonsu/camera/color/image_raw'),#['camera', agent_num]),
                        ('tensorrt_input', ['tensorrt_input', agent_num]),
                        #('tensorrt_output', ['tensorrt_output', agent_num]),
                        ('features_agent3', ['features_agent3_local', agent_num]), #temporary
                       #('features_agent2', ['features_agent2_local', agent_num])
                        ('features_agent1', LaunchConfiguration('feature_subscribe_topic')),
                        ('features_agent2', '/race15/features_agent2_local2'),
                        ('res1_pts3d_cloud', 'res1_pts3d_cloud'),
                        ('res2_pts3d_cloud', 'res2_pts3d_cloud')],
        ),

    ComposableNode(
            package= 'tensorrt_engine',
            namespace= 'khonsu',
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

            remappings=[('tensorrt_input', ['tensorrt_input', agent_num]),
                        ('tensorrt_output', ['tensorrt_output', agent_num])],
        ),

        #ComposableNode(
        #   package= 'neuromesh_platform_r2',
        #   namespace= 'race13',
        #   name= 'point_cloud_to_depth',
        #   plugin='point_cloud_to_depth_image::PointCloudToDepthImage',
        #
        #   remappings=[('point_cloud_topic', '/race13/res1_pts3d_cloud'),
        #               ('camera_info_topic', '/race13/color/camera_info')],

        #)
    ]

    return LaunchDescription([
        agent_num_arg,
        agent_list_arg,

         #launch_ros.actions.Node(
         #    package='neuromesh_platform_r2', executable='qos_node', output='screen',
         #    parameters=[{'qos_reliability': 'reliable',
         #                 'qos_durability': 'transient_local'}]),

         #launch_ros.actions.Node(
         #   package='neuromesh_platform_r2', executable='toy_implementation', name=["agent", agent_num], output='screen',
         #   parameters=[{'id': ["agent", agent_num],
         #               'output_topic': ["agent", agent_num, "_gnn_output"],
         #               'agents': agent_list,
         #               'ints_to_floats': True,
                        #'image_qos_profile': 'SYSTEM_DEFAULT'
         #               }],

         #   remappings=[('camera', ['camera', agent_num]),
         #               ('tensorrt_input', ['tensorrt_input', agent_num]),
         #               ('tensorrt_output', ['tensorrt_output', agent_num]),
         #               ('features_agent1', ['features_agent1_local', agent_num]), #temporary
         #               ('features_agent2', ['features_agent2_local', agent_num])]),

         #launch_ros.actions.Node(
         #   package='tensorrt_engine', executable='tensorrt_engine', name=["engine", agent_num], output='screen',
         #parameters=[{'model_path': get_package_share_directory('tensorrt_engine') + "/models/dust3r_vitb14.trt",
         #               'input_dimensions' : "1,3,320,1024",
         #               'output_dimensions' : "1,1,320,1024",
         #               'tensor_type': "fp32",
         #               }],
         #  remappings=[('tensorrt_input', ['tensorrt_input', agent_num]),
         #              ('tensorrt_output', ['tensorrt_output', agent_num])]),


        # image processing container
        ComposableNodeContainer(
            name='neuromesh_container5',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            #executable='component_container_isolated',
            composable_node_descriptions=composable_nodes,
            output='screen',
            #arguments=[{'--ros-args', '--log-level', 'debug'}],
                       #{ '--use_multithreaded_executor'}]
            )
    ])
