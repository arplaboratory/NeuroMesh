#ifndef CONTROL_neuromesh_NODE_HEADER_H
#define CONTROL_neuromesh_NODE_HEADER_H

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
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <random>

namespace ControlneuromeshNode {
class ControlneuromeshNode : public rclcpp::Node {
  // FUNCTIONS
public:
  // Constructor
  ControlneuromeshNode(const rclcpp::NodeOptions &options);

protected:
  void
  feature_callback(const neuromesh_interfaces::msg::CommMessage::SharedPtr msg);

  // perform inference on tensor using model called model_name
  // PLACEHOLDER
  // virtual std::future<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>
  // performInference(const std::string& model_name, const
  // neuromesh_interfaces::msg::Tensor& tensor);
  virtual std::future<
      std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
  performInference(
      const std::string &model_name,
      const std::vector<neuromesh_interfaces::msg::Tensor> &tensors);

  // Convert tensor of features to a feature message
  neuromesh_interfaces::msg::CommMessage
  buildFeatureMessage(const neuromesh_interfaces::msg::Tensor &tensor,
                      const neuromesh_interfaces::msg::Tensor &obs);

  // Aggregates features from available agents (and itself) into a single tensor
  // PLACEHOLDER
  virtual bool buildDecoderTensor(
      const std::map<std::string, neuromesh_interfaces::msg::CommMessage>
          &agent_features,
      neuromesh_interfaces::msg::Tensor &aggregated_tensor);

  // Adds a subscription to the feature_subscriptions_ map
  void createSubscription(
      std::map<std::string,
               rclcpp::Subscription<neuromesh_interfaces::msg::CommMessage>::
                   SharedPtr> &subscription_map,
      std::string id, rclcpp::QoS qos);

  // Remove subscription form the feature_subscriptions_ map
  void removeSubscription(
      std::map<std::string,
               rclcpp::Subscription<
                   neuromesh_interfaces::msg::CommMessage>::SharedPtr>
          subscription_map,
      std::string id);

  // Convert string to ROS2 QoS profile
  rmw_qos_profile_t parseQoSString(const std::string &str);

  // Split agent string parameter into vector of agent ids
  std::set<std::string> splitAgentString(std::string str);

  // Set encoder status as to whether or not it's already been run this cycle
  void run_encoder_cycle();

  void run_decoder_cycle();

  // Build observation
  neuromesh_interfaces::msg::Tensor
  build_obs(const std::set<std::string> &controllable_agents);

  // Build features from observation
  neuromesh_interfaces::msg::Tensor
  build_features_from_obs(const neuromesh_interfaces::msg::Tensor &obs);

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
  rclcpp::Publisher<neuromesh_interfaces::msg::CommMessage>::SharedPtr
      feature_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr velocity_publisher_;

  std::map<std::string, rclcpp::Subscription<
                            neuromesh_interfaces::msg::CommMessage>::SharedPtr>
      feature_subscriptions_;

  // repeating function to keep track of cycles
  rclcpp::TimerBase::SharedPtr decoder_timer_; // process features directly
  rclcpp::TimerBase::SharedPtr
      encoder_timer_; // update bool to be ready for encoder to run

  bool fresh_encoder_cycle; // if true encoder is ready to run.
  std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
      encoder_result;
  std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
      gnn_result_future;

  // variables for features
  std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr>
      feature_buffer_; // To store all features
  std::map<std::string, double>
      feature_buffer_timestamp_; // To store timestamps of all features

  Eigen::Quaternionf current_side_;

  std::vector<float> apply_current_side(const std::vector<float> &vec) const;
  void update_current_side();

  double beta_distribution(double alpha, double beta, std::mt19937 &gen);

  void transform_output(float *input, size_t size, float min_value,
                        float max_value);

  void pos_callback(const nav_msgs::msg::Odometry::SharedPtr msg);

  void vel_callback(const nav_msgs::msg::Odometry::SharedPtr msg);

  Eigen::Vector3f quaternion_to_euler(const geometry_msgs::msg::Quaternion &q);

  // parameters
  std::string encoder_model_name_;
  std::string decoder_model_name_;
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
  std::string velocity_topic;
  bool to_nchw_;
  neuromesh_interfaces::msg::Tensor obs;
  std::random_device rd_;
  std::mt19937 gen_;
  geometry_msgs::msg::Point hardcoded_goal_;

  std::map<std::string, geometry_msgs::msg::Pose> goal_poses_;
  std::map<std::string, neuromesh_interfaces::msg::StateVector> current_states_;
  std::map<std::string, neuromesh_interfaces::msg::CommMessage>
      received_features_;

  std::mutex state_mutex_;
  bool position_updated_ = false;
  const std::chrono::milliseconds max_wait_time_{1};
};
} // namespace ControlneuromeshNode

#endif
