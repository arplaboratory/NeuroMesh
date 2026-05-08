#ifndef GAT_IMPLEMENTATION_HEADER_H
#define GAT_IMPLEMENTATION_HEADER_H

#include "neuromesh_interfaces/srv/tensor_request.hpp"
#include "neuromesh_platform_r2/gat_neuromesh_node.h"
#include "rclcpp/rclcpp.hpp"
#include <map>
#include <memory>
#include <vector>

namespace GATneuromeshNode {
class GATImplementation : public GATneuromeshNode {
public:
  GATImplementation(const rclcpp::NodeOptions &options);

protected:
  // Perform inference
  std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
  performInference(
      const std::string &model_name,
      const std::vector<neuromesh_interfaces::msg::Tensor> &tensors);

  // Build decoder
  bool buildDecoderTensor(
      std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr>
          &agent_features,
      neuromesh_interfaces::msg::Tensor &own_feature,
      neuromesh_interfaces::msg::Tensor &aggregated_tensor);

  rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr
      feature_subscription_;
  rclcpp::Publisher<neuromesh_interfaces::msg::Feature>::SharedPtr
      feature_publisher_;
  rclcpp::Client<neuromesh_interfaces::srv::TensorRequest>::SharedPtr
      tensor_client_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pos_sub_;

  std::map<std::string, geometry_msgs::msg::Pose> goal_poses_;
  std::map<std::string, neuromesh_interfaces::msg::StateVector> current_states_;
};
} // namespace GATneuromeshNode
#endif
