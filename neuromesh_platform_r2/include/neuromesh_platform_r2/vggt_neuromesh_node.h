#ifndef VGGT_NEUROMESH_NODE_HEADER_H
#define VGGT_NEUROMESH_NODE_HEADER_H

#include "rclcpp/rclcpp.hpp"

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "neuromesh_interfaces/msg/feature.hpp"
#include "neuromesh_interfaces/msg/tensor.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace vggtNode {
class vggtNode : public rclcpp::Node {
  // FUNCTIONS
public:
  // Constructor
  vggtNode(const rclcpp::NodeOptions &options);

protected:
  // callback for feature subscription
  void feature_callback(const neuromesh_interfaces::msg::Feature::SharedPtr msg);

  // callback for camera subscription
  void camera_callback(const sensor_msgs::msg::Image::SharedPtr msg);

  // Process received features. Run once per decoder_cycle_length_
  void process_features();
  
  // Process encoder results and publish features
  void process_encoder_result();

  void broadcast_transform();

  void revertTensorDimensions(neuromesh_interfaces::msg::Tensor &tensor);

  // perform inference on tensor using model called model_name
  virtual std::future<
      std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
  performInference(
      const std::string &model_name,
      const std::vector<neuromesh_interfaces::msg::Tensor> &tensors);

  // Convert tensor of features to a feature message
  neuromesh_interfaces::msg::Feature
  buildFeatureMessage(const neuromesh_interfaces::msg::Tensor &tensor);

  // Aggregates features from available agents (and itself) into a single tensor
  virtual bool buildDecoderTensor(
      std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr>
          buffer,
      std::map<std::string, double> buffer_timestamp,
      neuromesh_interfaces::msg::Tensor &own_tensor,
      neuromesh_interfaces::msg::Tensor &neighbour_tensor);

  // Converts image message to tensor with VGGT preprocessing (resize to 392x518)
  neuromesh_interfaces::msg::Tensor
  imageToTensor(const sensor_msgs::msg::Image::SharedPtr msg);

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

  // Convert string to ROS2 QoS profile
  rmw_qos_profile_t parseQoSString(const std::string &str);

  // Split agent string parameter into vector of agent ids
  std::set<std::string> splitAgentString(std::string str);

  // Set encoder status as to whether or not it's already been run this cycle
  void run_encoder_cycle();

  int encoder_cycle_count_ = 0;
  std::shared_ptr<neuromesh_interfaces::msg::Tensor> first_encoder_result_;

  // transpose tensor (works for ints)
  neuromesh_interfaces::msg::Tensor
  convert_to_nchw(const neuromesh_interfaces::msg::Tensor &input);

  // Converts a vector of unsigned integers from an image to floats
  neuromesh_interfaces::msg::Tensor
  tensor_ints_to_floats(neuromesh_interfaces::msg::Tensor &input);

  std::vector<uint> string_to_dims_single(std::string in);
  std::vector<std::vector<uint>> string_to_dims(std::string in);

  // Create depth image message from decoder output
  sensor_msgs::msg::Image::SharedPtr
  createDepthImage(const neuromesh_interfaces::msg::Tensor &depth_tensor,
                   int robot_idx, const std_msgs::msg::Header &header);

  // Create pointcloud from world_points tensor
  sensor_msgs::msg::PointCloud2::SharedPtr
  createPointCloud(const neuromesh_interfaces::msg::Tensor &world_points_tensor,
                   const neuromesh_interfaces::msg::Tensor &world_points_conf_tensor,
                   int robot_idx, const std_msgs::msg::Header &header,
                   bool use_rgb = false,
                   const sensor_msgs::msg::Image::SharedPtr rgb_image = nullptr);

  // measure time
  void startClock(std::string phase);
  void stopClock(std::string phase);
  int64_t checkClock(std::string phase);
  std::map<std::string, std::pair<int64_t, bool>> times;

  // VARIABLES

  // Lists of agent ids that describe 1) all agents 2) the available ones
  std::set<std::string> all_agents;
  std::set<std::string> available_agents;

  // publishers and subscriptions
  rclcpp::Publisher<neuromesh_interfaces::msg::Feature>::SharedPtr
      feature_publisher_;
  rclcpp::Publisher<neuromesh_interfaces::msg::Tensor>::SharedPtr
      output_publisher_;
  
  // VGGT-specific publishers
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr
      depth_robot1_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr
      depth_robot2_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pointcloud_current_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pointcloud_neighbor_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pointcloud_current_rgb_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pointcloud_neighbor_rgb_publisher_;

  std::map<std::string,
           rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr>
      feature_subscriptions_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_subscription_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr transform_timer_;

  // repeating function to keep track of cycles
  rclcpp::TimerBase::SharedPtr decoder_timer_;
  rclcpp::TimerBase::SharedPtr encoder_timer_;
  rclcpp::TimerBase::SharedPtr encoder_result_timer_;

  bool fresh_encoder_cycle;
  std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
      encoder_result;
  std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
      decoder_result_future;

  // variables for features
  std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr>
      feature_buffer_;
  std::map<std::string, double> feature_buffer_timestamp_;

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
  bool to_nchw_;
  bool ints_to_floats_;
  std::string decoder_output_dimensions_str;
  neuromesh_interfaces::msg::Tensor cached_image_tensor;

  std::vector<std::vector<uint>> decoder_output_dims;

  sensor_msgs::msg::Image::SharedPtr latest_camera_msg_;
  std::mutex camera_msg_mutex_;

  // VGGT-specific dimensions (392x518 input resolution)
  uint32_t tensor_batch_size_ = 1;
  uint32_t tensor_channels_ = 3;
  uint32_t tensor_height_ = 392;
  uint32_t tensor_width_ = 518;
};
} // namespace vggtNode

#endif // VGGT_NEUROMESH_NODE_HEADER_H