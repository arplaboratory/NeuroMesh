from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="mission_maestro_bridge",
                executable="mission_maestro_bridge",
                name="mission_maestro_bridge",
                output="screen",
                parameters=[
                    {
                        "mission_goal_topic": "mission_goals",
                        "mission_yaml_topic": "mission_yaml",
                        "mission_command_topic": "mission_command",
                        "mission_yaml_service": "maestro_yaml",
                        "mission_command_service": "maestro_command",
                        "radius": 2.0,
                    }
                ],
            )
        ]
    )
