#pragma once

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include "types/navigation_goal.hpp"

class ServiceInterface {
public:
  virtual ~ServiceInterface() = default;
  virtual bool sendGoal(const NavigationGoal & goal) = 0;
  virtual std::string type() const = 0;
};

class NullServiceAdapter : public ServiceInterface {
public:
  explicit NullServiceAdapter(rclcpp::Logger logger)
  : logger_(logger)
  {
  }

  bool sendGoal(const NavigationGoal & goal) override
  {
    RCLCPP_WARN(
      logger_,
      "Navigation adapter is disabled; dropping goal in frame '%s' at (%.3f, %.3f, %.3f)",
      goal.planning_frame.c_str(), goal.x, goal.y, goal.z);
    return false;
  }

  std::string type() const override
  {
    return "none";
  }

private:
  rclcpp::Logger logger_;
};

class PoseStampedTopicAdapter : public ServiceInterface {
public:
  explicit PoseStampedTopicAdapter(const rclcpp::Node::SharedPtr & node)
  : node_(node)
  {
    if (!node_->has_parameter("navigation_goal_topic")) {
      node_->declare_parameter<std::string>("navigation_goal_topic", "goal_pose");
    }
    node_->get_parameter("navigation_goal_topic", goal_topic_);
    publisher_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(goal_topic_, 10);
  }

  bool sendGoal(const NavigationGoal & goal) override
  {
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = node_->now();
    msg.header.frame_id = goal.planning_frame;
    msg.pose.position.x = goal.x;
    msg.pose.position.y = goal.y;
    msg.pose.position.z = goal.z;
    msg.pose.orientation.w = 1.0;

    publisher_->publish(msg);
    RCLCPP_INFO(
      node_->get_logger(),
      "Published navigation goal on '%s' in frame '%s' at (%.3f, %.3f, %.3f)",
      goal_topic_.c_str(), goal.planning_frame.c_str(), goal.x, goal.y, goal.z);
    return true;
  }

  std::string type() const override
  {
    return "topic";
  }

private:
  rclcpp::Node::SharedPtr node_;
  std::string goal_topic_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr publisher_;
};

class AdapterFactory {
public:
  static std::shared_ptr<ServiceInterface>
  create(const std::string & adapter_type, const rclcpp::Node::SharedPtr & node)
  {
    std::string normalized = adapter_type;
    std::transform(
      normalized.begin(), normalized.end(), normalized.begin(),
      [](unsigned char c) {return static_cast<char>(std::tolower(c));});

    if (
      normalized.empty() || normalized == "none" || normalized == "noop" ||
      normalized == "null")
    {
      return std::make_shared<NullServiceAdapter>(node->get_logger());
    }

    if (
      normalized == "topic" || normalized == "pose_topic" ||
      normalized == "posestamped" || normalized == "missionmaestro" ||
      normalized == "mavmanager")
    {
      if (normalized == "missionmaestro" || normalized == "mavmanager") {
        RCLCPP_WARN(
          node->get_logger(),
          "Adapter type '%s' is treated as the generic topic adapter. "
          "Use 'topic' and bridge it externally if you need a specific navigation backend.",
          adapter_type.c_str());
      }
      return std::make_shared<PoseStampedTopicAdapter>(node);
    }

    RCLCPP_WARN(
      node->get_logger(),
      "Unknown navigation adapter '%s'; using no-op adapter instead.",
      adapter_type.c_str());
    return std::make_shared<NullServiceAdapter>(node->get_logger());
  }
};