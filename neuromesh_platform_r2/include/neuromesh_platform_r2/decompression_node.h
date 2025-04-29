#ifndef DECOMPRESSION_NODE_H
#define DECOMPRESSION_NODE_H

#include "neuromesh_interfaces/msg/compressed_tensor.hpp"
#include "neuromesh_interfaces/msg/tensor.hpp"
#include <rclcpp/rclcpp.hpp>

#include "neuromesh_interfaces/msg/compressed_feature.hpp"
#include "neuromesh_interfaces/msg/feature.hpp"

class DecompressionNode : public rclcpp::Node {

public:
  DecompressionNode();

protected:
  /**
   * @brief Decompresses a tensor.
   * @param msg The CompressedTensor to be decompressed
   * @return The decompressed tensor.
   */
  neuromesh_interfaces::msg::Tensor decompress(
      const std::shared_ptr<neuromesh_interfaces::msg::CompressedTensor> msg);

  /**
   * @brief This function is called whenever a new tensor is received. It will
   * decompress the tensor and publish it.
   *
   * @param msg The tensor message.
   */
  void tensor_callback(
      const std::shared_ptr<neuromesh_interfaces::msg::CompressedTensor> msg);

  /**
   * @brief This function is called whenever a new feature is received. It will
   * decompress the feature and publish it.
   *
   * @param msg The feature message.
   */
  void feature_callback(
      const std::shared_ptr<neuromesh_interfaces::msg::CompressedFeature> msg);

  /**
   * @brief Takes a string given as a parameter to the node and converts it to a
   * QoS profile.
   *
   * NOTE: implementation taken from
   * https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox/nvblox_ros_common/src/qos.cpp#L26
   *
   * @param str string corresponding to the QoS profile
   * @return QoS profile
   */
  rmw_qos_profile_t parseQoSString(const std::string &str);

  rclcpp::Subscription<neuromesh_interfaces::msg::CompressedTensor>::SharedPtr
      tensor_subscription_;
  rclcpp::Publisher<neuromesh_interfaces::msg::Tensor>::SharedPtr
      tensor_publisher_;

  rclcpp::Subscription<neuromesh_interfaces::msg::CompressedFeature>::SharedPtr
      feature_subscription_;
  rclcpp::Publisher<neuromesh_interfaces::msg::Feature>::SharedPtr
      feature_publisher_;

  std::string input_name_;
  std::string output_name_;
  bool use_features_;
  std::string input_qos_;
};

#endif // DECOMPRESSION_NODE_H
