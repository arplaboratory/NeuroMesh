#ifndef VGGT_TOY_IMPLEMENTATION_HEADER_H
#define VGGT_TOY_IMPLEMENTATION_HEADER_H

#include "neuromesh_interfaces/srv/tensor_request.hpp"
#include "neuromesh_platform_r2/vggt_neuromesh_node.h"

namespace vggtNode {
class VggtToyImplementation : public vggtNode {
public:
  VggtToyImplementation(const rclcpp::NodeOptions &options);

protected:
  std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
  performInference(
      const std::string &model_name,
      const std::vector<neuromesh_interfaces::msg::Tensor> &tensors) override;

  bool buildDecoderTensor(
      std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr> buffer,
      std::map<std::string, double> buffer_timestamp,
      neuromesh_interfaces::msg::Tensor &own_tensor,
      neuromesh_interfaces::msg::Tensor &neighbour_tensor) override;

  // topic publishers and subscribers
  rclcpp::Publisher<neuromesh_interfaces::msg::Tensor>::SharedPtr tensor_publisher_;
  rclcpp::Subscription<neuromesh_interfaces::msg::Tensor>::SharedPtr tensor_subscriber_;

  void tensor_callback(const neuromesh_interfaces::msg::Tensor::SharedPtr msg);

  // service
  rclcpp::Client<neuromesh_interfaces::srv::TensorRequest>::SharedPtr tensor_client_;
};
} // namespace vggtNode
#endif // VGGT_TOY_IMPLEMENTATION_HEADER_H