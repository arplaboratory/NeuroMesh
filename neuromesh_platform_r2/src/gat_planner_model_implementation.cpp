#include "neuromesh_platform_r2/gat_planner_implementation.h"

namespace GATPlannerNeuromeshNode {
GATPlannerImplementation::GATPlannerImplementation(const rclcpp::NodeOptions &options)
    : GATPlannerNeuromeshNode(options) {

  // Subscribe to odom topic to get current pose of robot
  pos_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "position_topic", 10,
      std::bind(&GATPlannerImplementation::pos_callback, this, std::placeholders::_1));

  // Subscribe to gnn_features topic received from the other neighbour robots
  gnn_result_subscriber_ =
      this->create_subscription<neuromesh_interfaces::msg::Feature>(
          "gnn_result_topic", 10,
          std::bind(&GATPlannerImplementation::gnn_result_callback, this,
                    std::placeholders::_1));

  this->tensor_client_ =
      create_client<neuromesh_interfaces::srv::TensorRequest>(
          "tensorrt_request");
}

std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
GATPlannerImplementation::performInference(
    const std::string &model_name,
    const std::vector<neuromesh_interfaces::msg::Tensor> &tensors) {

  if (!tensor_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_ERROR(this->get_logger(), "Engine not reachable via service.");
    // Complex stuff just to return future that resolves to empty tensor
    std::promise<
        std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
        prom;
    std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
        r = prom.get_future();
    std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> t(
        1, std::make_shared<neuromesh_interfaces::msg::Tensor>());
    t[0]->result = 3; // Cannot reach engine error code
    prom.set_value(t);
    return r;
  }

  // Create a request to send to the service server
  auto request =
      std::make_shared<neuromesh_interfaces::srv::TensorRequest::Request>();
  request->model_name = model_name;
  request->tensor1 = tensors;

  // Call the service and wait for the response
  std::shared_future<
      std::shared_ptr<neuromesh_interfaces::srv::TensorRequest::Response>>
      future = tensor_client_->async_send_request(request);

  std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
      return_tensors = std::async(std::launch::async, [future]() {
        std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>
            output_tensors;
        for (const auto &tensor : future.get()->tensor2) {
          output_tensors.emplace_back(
              std::make_shared<neuromesh_interfaces::msg::Tensor>(tensor));
        }
        return output_tensors;
      });
  
  // RCLCPP_INFO(this->get_logger(), "Here all good");
  // std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> r_tensors = return_tensors.get();
  // auto tensor_ptr = r_tensors[0];  // Assume we're just printing the first tensor
  // if (tensor_ptr->data_type != 9) {  // 9 = float32
  //     RCLCPP_WARN(this->get_logger(), "Tensor data_type is not float32 (got %d)", tensor_ptr->data_type);
  //     return;
  // }

  // size_t num_floats = tensor_ptr->data.size() / sizeof(float);
  // const float* float_data = reinterpret_cast<const float*>(tensor_ptr->data.data());

  // std::ostringstream oss;
  // oss << "return_tensors: [";
  // for (size_t i = 0; i < num_floats; ++i) {
  //     oss << float_data[i];
  //     if (i < num_floats - 1)
  //         oss << ", ";
  // }
  // oss << "]";

  // RCLCPP_INFO(this->get_logger(), "return_tensors = %s", oss.str().c_str());

  return return_tensors;
}

bool GATPlannerImplementation::buildDecoderTensor(
    std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr>
        &agent_features,
    neuromesh_interfaces::msg::Tensor &own_feature,
    neuromesh_interfaces::msg::Tensor &aggregated_tensor) {
  if (agent_features.size() != 5) {
    RCLCPP_INFO(this->get_logger(), "Expected 5 feature messages, got %zu",
                agent_features.size());
    return false;
  }

  std::vector<std::string> sorted_ids;
  for (const auto &pair : agent_features) {
    sorted_ids.push_back(pair.first);
  }
  std::sort(sorted_ids.begin(), sorted_ids.end());

  own_feature.name = "own_features";
  aggregated_tensor.name = "other_features";
  own_feature.data_type = 9;
  aggregated_tensor.data_type = 9;

  for (const std::string &id : sorted_ids) {
    const auto &feature_msg = agent_features.at(id);
    if (id == this->id_) {
      own_feature = feature_msg->tensor;
    } else {
      if (aggregated_tensor.data.empty()) {
        aggregated_tensor = feature_msg->tensor;
      } else {
        aggregated_tensor.data.insert(aggregated_tensor.data.end(),
                                      feature_msg->tensor.data.begin(),
                                      feature_msg->tensor.data.end());
      }
    }
  }

  return true;
}
} // namespace GATPlannerNeuromeshNode

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(GATPlannerNeuromeshNode::GATPlannerImplementation)
