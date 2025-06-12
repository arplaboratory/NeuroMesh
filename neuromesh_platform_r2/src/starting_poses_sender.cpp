#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include <arl_mission_maestro/srv/maestro_mission_yaml.hpp>
#include <arl_mission_maestro/srv/maestro_command.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <vector>
#include <string>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace goal_sender
{
class GoalSenderNode : public rclcpp::Node
{
public:
  explicit GoalSenderNode(const rclcpp::NodeOptions & options)
  : Node("goal_sender_node", options)
  {
    // Declare parameters
    this->declare_parameter("start_poses_yaml_file", "start_poses.yaml");
    this->declare_parameter("id", "id");
    this->declare_parameter("initialization_timeout", 10.0);
    
    // Get parameters
    id_ = this->get_parameter("id").as_string();
    
    std::string start_poses_yaml_file = this->get_parameter("start_poses_yaml_file").as_string();

    // Setup TF listener
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

     // Create odometry subscriber
     std::string odom_topic = "/" + id_ + "/odom";
     odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
       odom_topic, 
       10, 
       std::bind(&GoalSenderNode::odom_callback, this, std::placeholders::_1)
     );
    
    // Create service clients
    waypoint_yaml_request_ = this->create_client<arl_mission_maestro::srv::MaestroMissionYaml>(
      "maestro_yaml");
    waypoint_command_request_ = this->create_client<arl_mission_maestro::srv::MaestroCommand>(
      "maestro_command");

    // Set initialization timeout
    double timeout = this->get_parameter("initialization_timeout").as_double();
    init_timeout_ = this->now() + rclcpp::Duration::from_seconds(timeout);
      
    // Create timer for periodic checking of service availability
    timer_ = this->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&GoalSenderNode::timerCallback, this));
      
    // Load goals from YAML file
    loadGoalsFromYaml();

    RCLCPP_INFO(this->get_logger(), "Waiting for robot %s initialization...", id_.c_str());
  }

private:
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
      // Check transform from odom to map
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
      
      // Check if the id exists in the YAML file
      if (!config[id_]) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Robot ID '%s' not found in YAML file", id_.c_str());
        return;
      }

      // Get goals specific to this robot
      const YAML::Node& robot_goals = config[id_]["goals"];
      
      if (!robot_goals || !robot_goals.IsSequence()) {
        RCLCPP_ERROR(
          this->get_logger(),
          "No valid goals found for robot '%s'", id_.c_str());
        return;
      }

      for (const auto& goal : robot_goals) {
        geometry_msgs::msg::Pose pose;
        
        // Check if position node exists and has required fields
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
    
    // Create a waypoints sequence node
    yaml_string["waypoints"] = YAML::Node(YAML::NodeType::Sequence);
    
    // Add each goal pose as a waypoint
    for (size_t i = 0; i < goal_poses_.size(); ++i) {
      YAML::Node wp_node;
      const auto& pose = goal_poses_[i];
      
      // Create pose data vector
      std::vector<float> pose_data{
        static_cast<float>(pose.position.x),
        static_cast<float>(pose.position.y),
        static_cast<float>(pose.position.z)
      };
      
      wp_node["name"] = "waypoint" + std::to_string(i + 1);
      wp_node["pose"] = pose_data;
      wp_node["pose"].SetStyle(YAML::EmitterStyle::Flow);
      wp_node["radius"] = 2.0;
      
      // Add waypoint to the sequence
      yaml_string["waypoints"].push_back(wp_node);
    }
    
    return yaml_string;
  }

  void sendGoals()
  {
    if (goals_sent_ || goal_poses_.empty()) {
      return;
    }
    // Prepare the goal pose format from yaml file
    YAML::Node mission_yaml = createMissionYaml();

    // Prepare YAML service request
    auto waypoint_yaml = std::make_shared<arl_mission_maestro::srv::MaestroMissionYaml::Request>();
    auto waypoint_command = std::make_shared<arl_mission_maestro::srv::MaestroCommand::Request>();
    
    waypoint_yaml->yaml_as_string = YAML::Dump(mission_yaml);
    waypoint_command->command = 0;

    if (!waypoint_yaml_request_->wait_for_service(std::chrono::seconds(1))) {
      RCLCPP_ERROR(this->get_logger(), "Waypoint client not reachable via service.");
      return;
    }
    auto waypoint_yaml_result = waypoint_yaml_request_->async_send_request(waypoint_yaml);

    if (!waypoint_command_request_->wait_for_service(std::chrono::seconds(1))) {
      RCLCPP_ERROR(this->get_logger(), "Waypoint client not reachable via service.");
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Sending goals to Maestro...");

    auto waypoint_command_result = waypoint_command_request_->async_send_request(waypoint_command);
    
    goals_sent_ = true;
  }

  void timerCallback()
  {
    // If goals are already sent, stop checking
    if (goals_sent_) {
      timer_->cancel();
      return;
    }

    // Update TF availability
    check_transform_available();
    
    // Check if all required conditions are met
    bool robot_ready = odom_received_ && tf_available_;
    
    if (robot_ready && 
        waypoint_yaml_request_->service_is_ready() &&
        waypoint_command_request_->service_is_ready()) 
    {
      RCLCPP_INFO(this->get_logger(), "Robot %s is ready, sending goals...", id_.c_str());
      sendGoals();
    }
    // Check for timeout
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

  // Service clients
  rclcpp::Client<arl_mission_maestro::srv::MaestroMissionYaml>::SharedPtr waypoint_yaml_request_;
  rclcpp::Client<arl_mission_maestro::srv::MaestroCommand>::SharedPtr waypoint_command_request_;

  // TF buffer and listener
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Odometry subscriber
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

  // Timer for periodic checking
  rclcpp::TimerBase::SharedPtr timer_;
  
  // Member variables
  std::string yaml_file_path_;
  std::string id_;
  double waypoint_radius_;
  bool goals_sent_{false};
  bool odom_received_{false};
  bool tf_available_{false};
  rclcpp::Time init_timeout_;
  std::vector<geometry_msgs::msg::Pose> goal_poses_;
};

} // namespace goal_sender

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(goal_sender::GoalSenderNode)
