from __future__ import annotations

from typing import Optional

import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from std_msgs.msg import String, UInt8
import yaml


class MissionMaestroBridge(Node):
    def __init__(self) -> None:
        super().__init__("mission_maestro_bridge")

        self.declare_parameter("mission_goal_topic", "mission_goals")
        self.declare_parameter("mission_yaml_topic", "mission_yaml")
        self.declare_parameter("mission_command_topic", "mission_command")
        self.declare_parameter("mission_yaml_service", "maestro_yaml")
        self.declare_parameter("mission_command_service", "maestro_command")
        self.declare_parameter("radius", 2.0)

        mission_goal_topic = self.get_parameter("mission_goal_topic").get_parameter_value().string_value
        mission_yaml_topic = self.get_parameter("mission_yaml_topic").get_parameter_value().string_value
        mission_command_topic = self.get_parameter("mission_command_topic").get_parameter_value().string_value
        self._yaml_service_name = self.get_parameter("mission_yaml_service").get_parameter_value().string_value
        self._command_service_name = self.get_parameter("mission_command_service").get_parameter_value().string_value
        self._radius = self.get_parameter("radius").get_parameter_value().double_value

        self._maestro_available = False
        self._maestro_yaml_srv = None
        self._maestro_cmd_srv = None
        self._yaml_client = None
        self._command_client = None

        try:
            from arl_mission_maestro.srv import MaestroCommand, MaestroMissionYaml

            self._maestro_yaml_srv = MaestroMissionYaml
            self._maestro_cmd_srv = MaestroCommand
            self._yaml_client = self.create_client(MaestroMissionYaml, self._yaml_service_name)
            self._command_client = self.create_client(MaestroCommand, self._command_service_name)
            self._maestro_available = True
            self.get_logger().info(
                f"Mission Maestro bridge enabled: services '{self._yaml_service_name}' and "
                f"'{self._command_service_name}'"
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.get_logger().warn(
                "arl_mission_maestro not available in this environment. "
                f"Bridge will stay passive. Details: {exc}"
            )

        self.create_subscription(PoseArray, mission_goal_topic, self._on_pose_array, 10)
        self.create_subscription(String, mission_yaml_topic, self._on_yaml, 10)
        self.create_subscription(UInt8, mission_command_topic, self._on_command, 10)

        self.get_logger().info(
            f"Listening on topics: goals='{mission_goal_topic}', yaml='{mission_yaml_topic}', "
            f"command='{mission_command_topic}'"
        )

    def _wait_for_service(self, service_type: str) -> bool:
        if not self._maestro_available:
            return False
        client = self._yaml_client if service_type == "yaml" else self._command_client
        if client is None:
            return False
        if client.service_is_ready():
            return True
        ok = client.wait_for_service(timeout_sec=0.5)
        if not ok:
            name = self._yaml_service_name if service_type == "yaml" else self._command_service_name
            self.get_logger().warn(f"Mission Maestro service not ready: {name}")
        return ok

    def _call_yaml_service(self, yaml_payload: str) -> None:
        if not self._wait_for_service("yaml"):
            return
        req = self._maestro_yaml_srv.Request()
        req.yaml_as_string = yaml_payload
        self._yaml_client.call_async(req)
        self.get_logger().info("Forwarded mission YAML to maestro_yaml service")

    def _call_command_service(self, command: int) -> None:
        if not self._wait_for_service("command"):
            return
        req = self._maestro_cmd_srv.Request()
        req.command = int(command)
        self._command_client.call_async(req)
        self.get_logger().info(f"Forwarded mission command to maestro_command service: {command}")

    def _build_yaml_from_goals(self, msg: PoseArray) -> str:
        frame_id = msg.header.frame_id if msg.header.frame_id else "map"
        waypoints = []
        for idx, pose in enumerate(msg.poses):
            waypoints.append(
                {
                    "name": f"waypoint{idx + 1}",
                    "pose": [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
                    "radius": float(self._radius),
                }
            )

        payload = {
            "version": 2.0,
            "frameid": frame_id,
            "waypoints": waypoints,
        }
        return yaml.safe_dump(payload, sort_keys=False)

    def _on_pose_array(self, msg: PoseArray) -> None:
        if not self._maestro_available:
            return
        yaml_payload = self._build_yaml_from_goals(msg)
        self._call_yaml_service(yaml_payload)

    def _on_yaml(self, msg: String) -> None:
        if not self._maestro_available:
            return
        if not msg.data:
            return
        self._call_yaml_service(msg.data)

    def _on_command(self, msg: UInt8) -> None:
        if not self._maestro_available:
            return
        self._call_command_service(int(msg.data))


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = MissionMaestroBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
