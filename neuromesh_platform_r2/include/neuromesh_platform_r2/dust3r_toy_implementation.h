#ifndef TOY_IMPLEMENTATION_HEADER_H
#define TOY_IMPLEMENTATION_HEADER_H

#include "neuromesh_interfaces/srv/tensor_request.hpp"
#include "neuromesh_platform_r2/dust3r_neuromesh_node.h"

namespace neuromeshNode {
class ToyImplementation : public neuromeshNode {
public:
  ToyImplementation(const rclcpp::NodeOptions &options);

protected:
  // std::future<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>
  // performInference(const std::string& model_name, const
  // neuromesh_interfaces::msg::Tensor& tensor);
  std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
  performInference(
      const std::string &model_name,
      const std::vector<neuromesh_interfaces::msg::Tensor> &tensors);

  bool buildDecoderTensor(
      std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr> buffer,
      std::map<std::string, double> buffer_timestamp,
      neuromesh_interfaces::msg::Tensor &own_tensor,
      neuromesh_interfaces::msg::Tensor &neighbour_tensor,
      neuromesh_interfaces::msg::Tensor &pos1,
      neuromesh_interfaces::msg::Tensor &pos2, const YAML::Node &pos1_yaml,
      const YAML::Node &pos2_yaml);
  // neuromesh_interfaces::msg::Tensor imageToTensor(const
  // sensor_msgs::msg::Image::SharedPtr msg);

  // topic publisehrs and subscribers
  rclcpp::Publisher<neuromesh_interfaces::msg::Tensor>::SharedPtr tensor_publisher_;
  rclcpp::Subscription<neuromesh_interfaces::msg::Tensor>::SharedPtr tensor_subscriber_;

  void tensor_callback(const neuromesh_interfaces::msg::Tensor::SharedPtr msg);

  // service
  rclcpp::Client<neuromesh_interfaces::srv::TensorRequest>::SharedPtr tensor_client_;
  // void
  // service_callback(rclcpp::Client<neuromesh_interfaces::srv::TensorRequest>::SharedFuture
  // future);
};
} // namespace neuromeshNode
#endif // TOY_IMPLEMENTATION_HEADER_H
