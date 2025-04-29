#ifndef TOY_IMPLEMENTATION_HEADER_H
#define TOY_IMPLEMENTATION_HEADER_H

#include "neuromesh_interfaces/srv/tensor_request.hpp"
#include "neuromesh_platform_r2/neuromesh_node.h"

namespace neuromeshNode {
class ToyImplementation : public neuromeshNode {
public:
  ToyImplementation(const rclcpp::NodeOptions &options);

protected:
  /**
   * @brief Sends a tensor to the engine for inference.
   *
   * NOTE: This implements a placeholder that is not implemented in
   * neuromesh_node.
   *
   * @param model_name A string to indicate which model to use, as implemented
   * by the user.
   * @param tensor The input to the model.
   * @return A future that will contain the output tensor.
   */
  std::future<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>
  performInference(const std::string &model_name,
                   const neuromesh_interfaces::msg::Tensor &tensor);

  /**
   * @brief Aggregates features from feature buffer into a single tensor.
   *
   * NOTE: This implements a placeholder that is not implemented in
   * neuromesh_node.
   *
   * @param buffer The buffer of features (typically this->feature_buffer_).
   * @return A single tensor containing all the features.
   */
  neuromesh_interfaces::msg::Tensor buildDecoderTensor(
      std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr>
          buffer);

  // topic publishers and subscribers
  rclcpp::Publisher<neuromesh_interfaces::msg::Tensor>::SharedPtr
      tensor_publisher_;
  rclcpp::Subscription<neuromesh_interfaces::msg::Tensor>::SharedPtr
      tensor_subscriber_;

  void tensor_callback(const neuromesh_interfaces::msg::Tensor::SharedPtr msg);

  // service
  rclcpp::Client<neuromesh_interfaces::srv::TensorRequest>::SharedPtr
      tensor_client_;
  // void
  // service_callback(rclcpp::Client<neuromesh_interfaces::srv::TensorRequest>::SharedFuture
  // future);
};
} // namespace neuromeshNode
#endif // TOY_IMPLEMENTATION_HEADER_H
