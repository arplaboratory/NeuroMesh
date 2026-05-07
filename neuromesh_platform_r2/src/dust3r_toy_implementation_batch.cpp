#include "neuromesh_platform_r2/dust3r_toy_implementation.h"
#include <chrono>
#include <rclcpp/callback_group.hpp>

namespace neuromeshNode {
ToyImplementation::ToyImplementation(const rclcpp::NodeOptions &options)
    : neuromeshNode(options) {

  std::string tensor_qos_profile_;

  this->declare_parameter<std::string>("tensor_qos_profile", "default");
  this->get_parameter("tensor_qos_profile", tensor_qos_profile_);

  auto tensor_qos =
      rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(tensor_qos_profile_));

  auto main_opt = rclcpp::SubscriptionOptions();

  auto tensor_opt = rclcpp::SubscriptionOptions();

  // redefine subscriptions with the callback groups

  auto image_qos =
      rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(output_qos_profile_));

  camera_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "camera", image_qos,
      std::bind(&ToyImplementation::camera_callback, this,
                std::placeholders::_1),
      // parseQoSString(image_qos_profile_),
      main_opt);

  decoder_timer_ = this->create_wall_timer(
      std::chrono::duration<int, std::milli>(decoder_cycle_length_),
      std::bind(&ToyImplementation::process_features, this));

  encoder_timer_ = this->create_wall_timer(
      std::chrono::duration<int, std::milli>(decoder_cycle_length_),
      std::bind(&ToyImplementation::run_encoder_cycle, this));

  // define our subscriptions
  this->tensor_client_ =
      create_client<neuromesh_interfaces::srv::TensorRequest>("tensorrt_request");
}

std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
ToyImplementation::performInference(
    const std::string &model_name,
    const std::vector<neuromesh_interfaces::msg::Tensor> &tensors
) {
  if (!tensor_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_ERROR(this->get_logger(), "Engine not reachable via service.");
    // Complex stuff just to return future that resolves to empty tensor
    std::promise<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
        prom;
    std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> r =
        prom.get_future();
    std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> t(
        1, std::make_shared<neuromesh_interfaces::msg::Tensor>());
    t[0]->result = 3; // Cannot reach engine error code
    prom.set_value(t);
    return r;
  }

  // Create a request to send to the service server
  auto request = std::make_shared<neuromesh_interfaces::srv::TensorRequest::Request>();
  request->model_name = model_name;
  request->tensor1 = tensors;
  RCLCPP_INFO(rclcpp::get_logger("logger.neuromesh"), "Requesting tensor: %s", model_name.c_str());
  for (size_t i = 0; i < tensors.size(); i++) {
    std::stringstream dims;
    for(size_t j = 0; j < tensors[i].shape.dims.size(); j++) {
        dims << tensors[i].shape.dims[j] << ",";
    }
    RCLCPP_INFO(rclcpp::get_logger("logger.neuromesh"), "> tensor[%d]: (%s) %ld dims: %s", i, tensors[i].name.c_str(), tensors[i].data.size(), dims.str().c_str());
  }

  // Call the service and wait for the response
  std::shared_future<
      std::shared_ptr<neuromesh_interfaces::srv::TensorRequest::Response>>
      future = tensor_client_->async_send_request(request);

  std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
      return_tensors = std::async(std::launch::async, [future]() {
        std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>
            output_tensors;
        for (const auto &tensor : future.get()->tensor2) {
            std::stringstream dims;
            for(size_t j = 0; j < tensor.shape.dims.size(); j++) {
                dims << tensor.shape.dims[j] << ",";
            }
            RCLCPP_INFO(rclcpp::get_logger("logger.neuromesh"), "Tensors returned after inference: %s tensor: %ld dims: %s", tensor.name.c_str(), tensor.data.size(), dims.str().c_str());
            output_tensors.emplace_back( std::make_shared<neuromesh_interfaces::msg::Tensor>(tensor));
        }
        return output_tensors;
      });

  return return_tensors;
}

bool ToyImplementation::buildDecoderTensor(
    std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr> buffer,
    std::map<std::string, double> buffer_timestamp,
    neuromesh_interfaces::msg::Tensor &own_tensor,
    neuromesh_interfaces::msg::Tensor &neighbour_tensor,
    neuromesh_interfaces::msg::Tensor &pos1, neuromesh_interfaces::msg::Tensor &pos2,
    const YAML::Node &pos1_yaml, const YAML::Node &pos2_yaml) {
  // RCLCPP_INFO(this->get_logger(), "Building decoder tensor with %d features.",
  //             buffer.size());
  startClock("decoder_tensor");

  // Dynamic M based on number of neighbors
  std::vector<std::string> neighbor_ids;
  for (const auto &[key, _] : buffer) {
    if (key != id_) {
      //RCLCPP_INFO(this->get_logger(), "key value: %s", key.c_str());
      neighbor_ids.push_back(key);
    }
  }

  // Check if we have enough features
  if (neighbor_ids.size() < all_agents.size()) {
    RCLCPP_WARN(
      this->get_logger(),
      "Num neighbors %ld/%ld, returning",
      neighbor_ids.size(),
      all_agents.size()
    );
    return false;
  } else {
    RCLCPP_INFO(this->get_logger(), "Neighbors found");
  }

  RCLCPP_INFO(this->get_logger(), "Generating tensor for decoding");
  const uint32_t feature_dim = 786432; // Feature dimension
  const size_t M = neighbor_ids.size();

  // Get our own feature
  auto own_feature = buffer[id_];

  // Create batched tensor for own features
  own_tensor.name = "own_input";
  own_tensor.data.clear();
  own_tensor.data.reserve(M * feature_dim);

  // Broadcast own feature M times
  for (size_t i = 0; i < M; i++) {
    own_tensor.data.insert(own_tensor.data.end(),
                           own_feature->tensor.data.begin(),
                           own_feature->tensor.data.end());
  }

  // Set shape for own tensor
  own_tensor.shape.dims = {static_cast<long>(M), all_agents.size()+1, static_cast<long>(feature_dim)};
  own_tensor.data_type = own_feature->tensor.data_type;
  own_tensor.strides = own_feature->tensor.strides;

  // Process neighbor features
  neighbour_tensor.name = "neighbor_input";
  neighbour_tensor.data.clear();
  neighbour_tensor.data.reserve(
      M * feature_dim); // Reserve space for M feature vectors

  // Get data type and strides from the first neighbor feature
  auto first_neighbor_feature = buffer[neighbor_ids[0]];
  neighbour_tensor.data_type = first_neighbor_feature->tensor.data_type;
  neighbour_tensor.strides = first_neighbor_feature->tensor.strides;

  // Collect features from all M neighbors
  for (size_t n = 0; n < M; n++) {
    auto neighbor_feature = buffer[neighbor_ids[n]];

    // Append this neighbor's feature vector to the neighbour_tensor
    neighbour_tensor.data.insert(neighbour_tensor.data.end(),
                                 neighbor_feature->tensor.data.begin(),
                                 neighbor_feature->tensor.data.end());
  }

  // Set shape for neighbor tensor
  neighbour_tensor.shape.dims = {static_cast<long>(M), all_agents.size()+1,
                                 static_cast<long>(feature_dim)};

  // Process pos1
  if (pos1_yaml["pos1"]) {
    auto pos1_data =
        pos1_yaml["pos1"].as<std::vector<std::vector<std::vector<int>>>>();
    if (!pos1_data.empty() && !pos1_data[0].empty()) {
      pos1.data_type = 7; // INT32
      pos1.shape.dims = {2, static_cast<long>(pos1_data[0].size()), 2};

      // Resize data to accommodate M copies of the original data
      pos1.data.resize(pos1.shape.dims[0] * pos1.shape.dims[1] *
                       pos1.shape.dims[2] * sizeof(int));

      int *data_ptr = reinterpret_cast<int *>(pos1.data.data());

      // Repeat the original pos1 data M times
      for (size_t i = 0; i < M; i++) {
        for (const auto &row : pos1_data[0]) {
          if (row.size() == 2) {
            *data_ptr++ = row[0];
            *data_ptr++ = row[1];
          } else {
            RCLCPP_ERROR(this->get_logger(), "Invalid data format in pos1");
            return false;
          }
        }
      }
    } else {
      RCLCPP_ERROR(this->get_logger(), "Empty data in pos1");
      return false;
    }
  } else {
    RCLCPP_WARN(this->get_logger(), "pos1 not found in YAML config");
    return false;
  }

  // Process pos2 (similar to pos1)
  if (pos2_yaml["pos2"]) {
    auto pos2_data =
        pos2_yaml["pos2"].as<std::vector<std::vector<std::vector<int>>>>();
    if (!pos2_data.empty() && !pos2_data[0].empty()) {
      pos2.data_type = 7; // INT32
      pos2.shape.dims = {2,
                         static_cast<long>(pos2_data[0].size()), 2};

      // Resize data to accommodate M copies of the original data
      pos2.data.resize(pos2.shape.dims[0] * pos2.shape.dims[1] *
                       pos2.shape.dims[2] * sizeof(int));

      int *data_ptr = reinterpret_cast<int *>(pos2.data.data());

      // Repeat the original pos2 data M times
      for (size_t i = 0; i < M; i++) {
        for (const auto &row : pos2_data[0]) {
          if (row.size() == 2) {
            *data_ptr++ = row[0];
            *data_ptr++ = row[1];
          } else {
            RCLCPP_ERROR(this->get_logger(), "Invalid data format in pos2");
            return false;
          }
        }
      }
    } else {
      RCLCPP_ERROR(this->get_logger(), "Empty data in pos2");
      return false;
    }
  } else {
    RCLCPP_WARN(this->get_logger(), "pos2 not found in YAML config");
    return false;
  }

  RCLCPP_INFO(this->get_logger(), "Generated tensor for decoding");

  stopClock("decoder_tensor");

  return true;
}

} // namespace neuromeshNode

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(neuromeshNode::ToyImplementation)
