import os
from ast import literal_eval

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def launch_setup(context):
    agent_list = LaunchConfiguration("agent_list").perform(context)
    name = LaunchConfiguration("name").perform(context)
    goal_file = LaunchConfiguration("goal_file").perform(context)
    start_file = LaunchConfiguration("start_file").perform(context)
    start_position = LaunchConfiguration("start_position")
    odom_republisher = LaunchConfiguration("odom_republisher")
    publish_static_map = LaunchConfiguration("publish_static_map").perform(context)
    planning_frame = LaunchConfiguration("planning_frame")

    launch_list = []

    # If using static map read in start location from yaml, publish map transform
    start_x = 0.0
    start_y = 0.0
    complete_agent_list = []
    with open(start_file) as f:
        start_positions = yaml.safe_load(f)
        start_x = start_positions[name]["goals"][0]["position"]["x"]
        start_y = start_positions[name]["goals"][0]["position"]["y"]
        for k, v in start_positions.items():
            complete_agent_list.append(k)

    print(f"{name} start positions, x: {start_x}, y: {start_y}")

    tf2_static_pub = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_tf2_static_pub",
        arguments=[
            "--x",
            str(start_x),
            "--y",
            str(start_y),
            "--z",
            "0",
            "--yaw",
            "0",
            "--pitch",
            "0",
            "--roll",
            "0",
            "--frame-id",
            "map",
            "--child-frame-id",
            name + "/map",
        ],
        condition=IfCondition(publish_static_map),
    )
    launch_list.append(tf2_static_pub)

    # LaunchConfiguration('agent_list') was serialized into string
    if isinstance(agent_list, str):
        try:
            agent_list = literal_eval(agent_list)
        except (ValueError, SyntaxError):
            agent_list = agent_list.split(",")

    remappings = []
    for i, agent in enumerate(agent_list):
        remappings.append((f"/{name}/features_{agent}", f"/{agent}/features_{agent}"))
        remappings.append(
            (f"/{name}/gnn_output_{agent}", f"/{agent}/gnn_output_{agent}")
        )

    remappings.append(("position_topic", f"/{name}/odometry/global"))

    composable_nodes = []
    composable_nodes.append(
        ComposableNode(
            package="neuromesh_platform_r2",
            plugin="goal_sender::GoalSenderNode",
            name="starting_poses_sender",
            namespace=name,
            parameters=[
                {
                    "robot_id": name,
                    "start_poses_yaml_file": start_file,
                }
            ],
            condition=IfCondition(start_position),
        )
    )
    composable_nodes.append(
        ComposableNode(
            package="tensorrt_engine",
            namespace=name,
            name=["engine", LaunchConfiguration("agent_num")],
            plugin="tensorrt_engine_node::TensorRTEngineNode",
            parameters=[
                {
                    "model_names": "encoder,decoder1,decoder2",
                    "encoder.model_path": get_package_share_directory("tensorrt_engine")
                    + "/models/encoder_local.trt",
                    "encoder.input_dimensions": "1,1,5",
                    "encoder.output_dimensions": "1,1,16",
                    "encoder.tensor_type": "fp32",
                    "decoder1.model_path": get_package_share_directory(
                        "tensorrt_engine"
                    )
                    + "/models/multi_head_gat_layer1.trt",
                    "decoder1.input_dimensions": "1,1,16;1,4,16;1,1,16",
                    "decoder1.output_dimensions": "1,1,16",
                    "decoder1.tensor_type": "fp32",
                    "decoder2.model_path": get_package_share_directory(
                        "tensorrt_engine"
                    )
                    + "/models/multi_head_gat_layer2.trt",
                    "decoder2.input_dimensions": "1,16;1,4,16;1,1,16",
                    "decoder2.output_dimensions": "1,1,5",
                    "decoder2.tensor_type": "fp32",
                }
            ],
        )
    )
    composable_nodes.append(
        ComposableNode(
            package="neuromesh_platform_r2",
            namespace=name,
            name=["neuromesh"],
            plugin="GATneuromeshNode::GATImplementation",
            parameters=[
                {
                    "id": name,
                    "encoder_model_name": "encoder",
                    "decoder_model1_name": "decoder1",
                    "decoder_model2_name": "decoder2",
                    "encoder_cycle_length": 3000,
                    "decoder_cycle_length": 3000,
                    "output_topic": "gnn_output_",
                    "agents": ",".join(agent_list),
                    "goal_poses_yaml_file": goal_file,
                    "planning_frame": planning_frame,
                    "goals_sending_delay": 2.0,
                }
            ],
            remappings=remappings,
        )
    )

    # If the number of agents is less than complete list, add the odom_republisher nodes
    if len(complete_agent_list) > len(agent_list):
        print(f"Number of agents is less than {len(complete_agent_list)}. Adding odom_republisher nodes!")
        missing_agents = set(complete_agent_list) - set(agent_list)
        for missing_agent in missing_agents:
            composable_nodes.append(
                ComposableNode(
                    package="neuromesh_platform_r2",
                    plugin="odom_republisher::OdomRepublisher",
                    name=f"odom_republisher_{missing_agent}",
                    remappings=[
                        ("original/odom", f"/{name}/odometry/global"),
                        ("republished/odom", f"/{missing_agent}/odometry/global"),
                    ],
                    condition=IfCondition(odom_republisher),
                )
            )
            print(
                f"Added odom_republisher for {missing_agent} using {name}'s odometry"
            )

    launch_list.append(
        ComposableNodeContainer(
            name="neuromesh_container",
            namespace=name,
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=composable_nodes,
            output="screen",
        )
    )

    return launch_list


def generate_launch_description():
    name_arg = DeclareLaunchArgument(
        name="name", default_value="warty", description=("Which agent we are running")
    )

    planning_frame_arg = DeclareLaunchArgument(
        name="planning_frame",
        default_value="map",
        description=("Which frame to send goals in"),
    )

    agent_num_arg = DeclareLaunchArgument(
        name="agent_num", default_value="1", description=("Which agent we are running")
    )

    agent_list_arg = DeclareLaunchArgument(
        name="agent_list",
        default_value="khonsu,osiris,ra,sekhmet,anubis",
        description=("List of all agents present (including self)"),
    )

    publish_static_map_arg = DeclareLaunchArgument(
        name="publish_static_map",
        default_value="False",
        description="run publish static map",
    )

    goal_file_arg = DeclareLaunchArgument(
        name="goal_file",
        default_value=os.path.join(
            get_package_share_directory("neuromesh_platform_r2"),
            "config",
            "goal_poses.yaml",
        ),
        description=("Location of config containing starting positions"),
    )
    start_file_arg = DeclareLaunchArgument(
        name="start_file",
        default_value=os.path.join(
            get_package_share_directory("neuromesh_platform_r2"),
            "config",
            "start_poses.yaml",
        ),
        description=("Location of config containing starting positions"),
    )

    start_position_arg = DeclareLaunchArgument(
        name="start_position",
        default_value="False",
        description=("Whether or not to run start goals node"),
    )
    odom_republisher_arg = DeclareLaunchArgument(
        name="odom_republisher",
        default_value="False",
        description=(
            "Whether or not to start the sekhmet republisher needed for < 5 robots"
        ),
    )

    opaque_function_action = OpaqueFunction(function=launch_setup)

    return LaunchDescription(
        [
            name_arg,
            agent_num_arg,
            agent_list_arg,
            goal_file_arg,
            start_file_arg,
            start_position_arg,
            publish_static_map_arg,
            odom_republisher_arg,
            planning_frame_arg,
            opaque_function_action,
        ]
    )
