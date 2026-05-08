#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <string>
#include <vector>

namespace goal_sender
{
class GoalSenderNode : public rclcpp::Node
{
public:
  explicit GoalSenderNode(const rclcpp::NodeOptions & options)
  : Node("goal_sender_node", options)
  {
    this->declare_parameter("start_poses_yaml_file", "start_poses.yaml");
    this->declare_parameter("id", "");
    this->declare_parameter("robot_id", "");
    this->declare_parameter("initialization_timeout", 10.0);
    this->declare_parameter("mission_goal_topic", "mission_goals");
    this->declare_parameter("mission_yaml_topic", "mission_yaml");
    this->declare_parameter("mission_command_topic", "mission_command");
    this->declare_parameter("publish_mission_yaml", true);
    this->declare_parameter("start_command_value", 0);

    std::string robot_id = this->get_parameter("robot_id").as_string();
    std::string id = this->get_parameter("id").as_string();
    id_ = !robot_id.empty() ? robot_id : id;
    if (id_.empty()) {
      id_ = this->get_namespace();
      if (!id_.empty() && id_[0] == '/') {
        id_.erase(0, 1);
      }
    }

    std::string start_poses_yaml_file = this->get_parameter("start_poses_yaml_file").as_string();
    yaml_file_path_ = resolveYamlPath(start_poses_yaml_file);
    publish_mission_yaml_ = this->get_parameter("publish_mission_yaml").as_bool();
    start_command_value_ = this->get_parameter("start_command_value").as_int();

    std::string mission_goal_topic = this->get_parameter("mission_goal_topic").as_string();
    std::string mission_yaml_topic = this->get_parameter("mission_yaml_topic").as_string();
    std::string mission_command_topic = this->get_parameter("mission_command_topic").as_string();

    goal_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(mission_goal_topic, 10);
    command_pub_ = this->create_publisher<std_msgs::msg::UInt8>(mission_command_topic, 10);
    if (publish_mission_yaml_) {
      yaml_pub_ = this->create_publisher<std_msgs::msg::String>(mission_yaml_topic, 10);
    }

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    std::string odom_topic = "/" + id_ + "/odom";
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic,
      10,
      std::bind(&GoalSenderNode::odom_callback, this, std::placeholders::_1));

    // Set initialization timeout
    double timeout = this->get_parameter("initialization_timeout").as_double();
    init_timeout_ = this->now() + rclcpp::Duration::from_seconds(timeout);
      
    timer_ = this->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&GoalSenderNode::timerCallback, this));

    loadGoalsFromYaml();

    RCLCPP_INFO(this->get_logger(), "Waiting for robot %s initialization...", id_.c_str());
  }

private:
  std::string resolveYamlPath(const std::string & yaml_file) const
  {
    std::ifstream direct_file(yaml_file);
    if (direct_file.good()) {
      return yaml_file;
    }

    const std::string package_share =
      ament_index_cpp::get_package_share_directory("neuromesh_platform_r2");
    const std::string config_candidate = package_share + "/config/" + yaml_file;
    std::ifstream config_file(config_candidate);
    if (config_file.good()) {
      return config_candidate;
    }

    return yaml_file;
  }

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    if (!odom_received_) {
      odom_received_ = true;
      RCLCPP_INFO(this->get_logger(), "Received first odometry message for robot %s", id_.c_str());
    }
  }

  bool check_transform_available()
  {
    try {
      geometry_msgs::msg::TransformStamped transform = 
        tf_buffer_->lookupTransform("map", id_ + "/odom", tf2::TimePointZero);
      
      if (!tf_available_) {
        RCLCPP_INFO(this->get_logger(), "Transform from odom to map is now available for robot %s", 
          id_.c_str());
        tf_available_ = true;
      }
      return true;
    }
    catch (const tf2::TransformException & ex) {
      return false;
    }
  }

  void loadGoalsFromYaml()
  {
    try {
      YAML::Node config = YAML::LoadFile(yaml_file_path_);

      if (!config[id_]) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Robot ID '%s' not found in YAML file", id_.c_str());
        return;
      }

      const YAML::Node& robot_goals = config[id_]["goals"];

      if (!robot_goals || !robot_goals.IsSequence()) {
        RCLCPP_ERROR(
          this->get_logger(),
          "No valid goals found for robot '%s'", id_.c_str());
        return;
      }

      for (const auto& goal : robot_goals) {
        geometry_msgs::msg::Pose pose;
        
        if (goal["position"] && 
            goal["position"]["x"] && 
            goal["position"]["y"]) {
          
          pose.position.x = goal["position"]["x"].as<double>();
          pose.position.y = goal["position"]["y"].as<double>();
          
          goal_poses_.push_back(pose);
        } else {
          RCLCPP_WARN(
            this->get_logger(),
            "Skipping malformed goal entry for robot '%s'", id_.c_str());
        }
      }
      
      RCLCPP_INFO(
        this->get_logger(),
        "Loaded %zu goals for robot '%s'", goal_poses_.size(), id_.c_str());
    } catch (const YAML::Exception& e) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to load YAML file: %s", e.what());
    }
  }

  YAML::Node createMissionYaml()
  {
    YAML::Node yaml_string;
    yaml_string["version"] = 2.0;
    yaml_string["frameid"] = id_ + "/map";
    
    yaml_string["waypoints"] = YAML::Node(YAML::NodeType::Sequence);

    for (size_t i = 0; i < goal_poses_.size(); ++i) {
      YAML::Node wp_node;
      const auto& pose = goal_poses_[i];

      std::vector<float> pose_data{
        static_cast<float>(pose.position.x),
        static_cast<float>(pose.position.y),
        static_cast<float>(pose.position.z)
      };
      
      wp_node["name"] = "waypoint" + std::to_string(i + 1);
      wp_node["pose"] = pose_data;
      wp_node["pose"].SetStyle(YAML::EmitterStyle::Flow);
      wp_node["radius"] = 2.0;
      
      yaml_string["waypoints"].push_back(wp_node);
    }

    return yaml_string;
  }

  void sendGoals()
  {
    if (goals_sent_ || goal_poses_.empty()) {
      return;
    }

    YAML::Node mission_yaml = createMissionYaml();

    geometry_msgs::msg::PoseArray goal_array;
    goal_array.header.stamp = this->now();
    goal_array.header.frame_id = id_ + "/map";
    goal_array.poses = goal_poses_;
    goal_pub_->publish(goal_array);

    if (publish_mission_yaml_ && yaml_pub_) {
      std_msgs::msg::String yaml_msg;
      yaml_msg.data = YAML::Dump(mission_yaml);
      yaml_pub_->publish(yaml_msg);
    }

    std_msgs::msg::UInt8 command_msg;
    command_msg.data = static_cast<uint8_t>(start_command_value_);
    command_pub_->publish(command_msg);

    RCLCPP_INFO(
      this->get_logger(),
      "Published %zu starting goals for robot %s using generic mission topics.",
      goal_poses_.size(), id_.c_str());
    
    goals_sent_ = true;
  }

  void timerCallback()
  {
    if (goals_sent_) {
      timer_->cancel();
      return;
    }

    check_transform_available();

    bool robot_ready = odom_received_ && tf_available_;

    if (robot_ready)
    {
      RCLCPP_INFO(this->get_logger(), "Robot %s is ready, sending goals...", id_.c_str());
      sendGoals();
    }
    else if (this->now() > init_timeout_) {
      RCLCPP_ERROR(this->get_logger(), 
        "Robot %s initialization timeout. Status:", id_.c_str());
      RCLCPP_ERROR(this->get_logger(), "Odometry received: %s", 
        odom_received_ ? "yes" : "no");
      RCLCPP_ERROR(this->get_logger(), "Transform available: %s", 
        tf_available_ ? "yes" : "no");
      timer_->cancel();
    }
  }

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr goal_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr yaml_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr command_pub_;

  rclcpp::TimerBase::SharedPtr timer_;

  std::string yaml_file_path_;
  std::string id_;
  double waypoint_radius_;
  int start_command_value_{0};
  bool publish_mission_yaml_{true};
  bool goals_sent_{false};
  bool odom_received_{false};
  bool tf_available_{false};
  rclcpp::Time init_timeout_;
  std::vector<geometry_msgs::msg::Pose> goal_poses_;
};

} // namespace goal_sender

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(goal_sender::GoalSenderNode)
