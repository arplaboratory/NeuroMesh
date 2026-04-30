#ifndef GAT_neuromesh_NODE_HEADER_H
#define GAT_neuromesh_NODE_HEADER_H

#include "rclcpp/rclcpp.hpp"

#include "geometry_msgs/msg/pose.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "neuromesh_interfaces/msg/comm_message.hpp"
#include "neuromesh_interfaces/msg/feature.hpp"
#include "neuromesh_interfaces/msg/state_vector.hpp"
#include "neuromesh_interfaces/msg/tensor.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"
#include "yaml-cpp/yaml.h"
#include <Eigen/Geometry>
#include <cmath>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <iostream>
#include <random>
#include <sstream>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace GATneuromeshNode {
class GATneuromeshNode : public rclcpp::Node {
  // FUNCTIONS
public:
  // Constructor
  GATneuromeshNode(const rclcpp::NodeOptions &options);

protected:
  void
  feature_callback(const neuromesh_interfaces::msg::Feature::SharedPtr msg);

  virtual std::future<
      std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
  performInference(
      const std::string &model_name,
      const std::vector<neuromesh_interfaces::msg::Tensor> &tensors);

  // Convert tensor of features to a feature message
  neuromesh_interfaces::msg::Feature
  buildFeatureMessage(const neuromesh_interfaces::msg::Tensor &tensor);

  // Aggregates features from available agents (and itself) into a single tensor
  // PLACEHOLDER
  virtual bool buildDecoderTensor(
      std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr>
          &agent_features,
      neuromesh_interfaces::msg::Tensor &own_feature,
      neuromesh_interfaces::msg::Tensor &aggregated_tensor);

  // Adds a subscription to the feature_subscriptions_ map
  void createSubscription(
      std::map<std::string, rclcpp::Subscription<
                                neuromesh_interfaces::msg::Feature>::SharedPtr>
          &subscription_map,
      std::string id, rclcpp::QoS qos);

  // Remove subscription form the feature_subscriptions_ map
  void removeSubscription(
      std::map<std::string, rclcpp::Subscription<
                                neuromesh_interfaces::msg::Feature>::SharedPtr>
          subscription_map,
      std::string id);

  // Adds a subscription to the gnn_subscriptions_ map
  void createGNNSubscription(
      std::map<std::string, rclcpp::Subscription<
                                neuromesh_interfaces::msg::Feature>::SharedPtr>
          &gnn_subscription_map,
      std::string id, rclcpp::QoS qos);

  // Remove subscription form the gnn_subscriptions_ map
  void removeGNNSubscription(
      std::map<std::string, rclcpp::Subscription<
                                neuromesh_interfaces::msg::Feature>::SharedPtr>
          gnn_subscription_map,
      std::string id);

  // Convert string to ROS2 QoS profile
  rmw_qos_profile_t parseQoSString(const std::string &str);

  // Split agent string parameter into vector of agent ids
  std::set<std::string> splitAgentString(std::string str);

  // Parameter handling
  void load_goal_poses_from_yaml();

  // Calculate distances to goal poses
  neuromesh_interfaces::msg::Tensor
  calculate_goal_distances(const std::vector<float> &state_vector);

  // Set encoder status as to whether or not it's already been run this cycle
  void run_encoder_cycle();

  // Methods for GNN result handling
  void run_decoder_cycle();
  neuromesh_interfaces::msg::Feature
  build_gnn_msg(const neuromesh_interfaces::msg::Tensor &gnn_result);
  void
  gnn_result_callback(const neuromesh_interfaces::msg::Feature::SharedPtr msg);
  void prepare_second_stage_decoding();

  // transpose tensor
  // works for ints
  neuromesh_interfaces::msg::Tensor
  convert_to_nchw(const neuromesh_interfaces::msg::Tensor &input);

  // measure time
  void startClock(std::string phase);
  void stopClock(std::string phase);
  int64_t checkClock(std::string phase);
  std::map<std::string, std::pair<int64_t, bool>>
      times; // int = milliseconds, bool = has timer stopped. if bool=false,
             // then int=starting time

  // VARIABLES

  // Lists of agent ids that describe 1) all agents 2) the available ones
  std::set<std::string> all_agents;
  std::set<std::string> available_agents;

  // publishers and subscriptions
  rclcpp::Publisher<neuromesh_interfaces::msg::Feature>::SharedPtr
      feature_publisher_;

  std::map<std::string,
           rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr>
      feature_subscriptions_;
  std::map<std::string,
           rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr>
      gnn_subscriptions_;

  // repeating function to keep track of cycles
  rclcpp::TimerBase::SharedPtr decoder_timer_; // process features directly
  rclcpp::TimerBase::SharedPtr
      encoder_timer_; // update bool to be ready for encoder to run

  bool fresh_encoder_cycle;    // if true encoder is ready to run.
  bool waypoint_cmd_sent_;     // tracking sending waypoints only once
  double goals_sending_delay_; // Delay sending goals to robots
  std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
      encoder_result;
  std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
      gnn_result_future;

  // Publishers and subscribers for GNN results
  rclcpp::Publisher<neuromesh_interfaces::msg::Feature>::SharedPtr
      gnn_result_publisher_;
  rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr
      gnn_result_subscriber_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr
      second_decoder_result_publisher_;

  // variables for features
  std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr>
      feature_buffer_; // To store all features
  std::map<std::string, double>
      feature_buffer_timestamp_; // To store timestamps of all features

  void pos_callback(const nav_msgs::msg::Odometry::SharedPtr msg);

  Eigen::Vector3f quaternion_to_euler(const geometry_msgs::msg::Quaternion &q);

  // parameters
  std::string encoder_model_name_;
  std::string decoder_model1_name_;
  std::string decoder_model2_name_;
  std::string topic_prefix_;
  std::string output_topic_;
  int decoder_cycle_length_;
  int encoder_cycle_length_;
  int encoder_await_length_;
  std::string id_;
  std::string image_qos_profile_;
  std::string features_qos_profile_;
  std::string output_qos_profile_;
  std::string agents_;
  bool to_nchw_;
  bool pos_callback_complete = false;
  neuromesh_interfaces::msg::Tensor encoder_output_tensor;
  std::string goal_poses_yaml_file;
  std::string planning_frame_;

  std::map<std::string, neuromesh_interfaces::msg::StateVector> current_states_;
  std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr>
      received_features_;

  std::vector<geometry_msgs::msg::Pose> goal_poses_;

  std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr>
      received_gnn_results_;

  std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
      second_decoder_result_future;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};
} // namespace GATneuromeshNode

#endif
