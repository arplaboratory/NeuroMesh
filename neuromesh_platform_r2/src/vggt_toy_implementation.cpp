#include "neuromesh_platform_r2/vggt_toy_implementation.h"
#include <chrono>
#include <rclcpp/callback_group.hpp>

namespace vggtNode {
VggtToyImplementation::VggtToyImplementation(const rclcpp::NodeOptions &options) 
    : vggtNode(options) {
    
    std::string tensor_qos_profile_;

    this->declare_parameter<std::string>("tensor_qos_profile", "default");
    this->get_parameter("tensor_qos_profile", tensor_qos_profile_);

    auto main_opt = rclcpp::SubscriptionOptions();

    // Redefine subscriptions with callback groups
    auto image_qos = rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(output_qos_profile_));

    camera_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        "camera",
        image_qos,
        std::bind(&VggtToyImplementation::camera_callback, this, std::placeholders::_1),
        main_opt);

    decoder_timer_ = this->create_wall_timer(
        std::chrono::duration<int,std::milli>(decoder_cycle_length_), 
        std::bind(&VggtToyImplementation::process_features, this));

    encoder_timer_ = this->create_wall_timer(
        std::chrono::duration<int,std::milli>(encoder_cycle_length_), 
        std::bind(&VggtToyImplementation::run_encoder_cycle, this));

    // Define TensorRT service client
    this->tensor_client_ = create_client<neuromesh_interfaces::srv::TensorRequest>("tensorrt_request");

    RCLCPP_INFO(this->get_logger(), "VGGT Toy Implementation initialized");
}

std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> 
VggtToyImplementation::performInference(
    const std::string& model_name,
    const std::vector<neuromesh_interfaces::msg::Tensor>& tensors) {
    
    if (!tensor_client_->wait_for_service(std::chrono::seconds(1))) {
        RCLCPP_ERROR(this->get_logger(), "TensorRT engine not reachable via service.");
        
        // Return future that resolves to error tensor
        std::promise<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> prom;
        std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> result = prom.get_future();
        std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> error_tensors(1, 
            std::make_shared<neuromesh_interfaces::msg::Tensor>());
        error_tensors[0]->result = 3; // Cannot reach engine error code
        prom.set_value(error_tensors);
        return result;
    }

    // Create service request
    auto request = std::make_shared<neuromesh_interfaces::srv::TensorRequest::Request>();
    request->model_name = model_name;
    request->tensor1 = tensors;
    
    RCLCPP_INFO(this->get_logger(), "Sending TensorRT request for model: %s with %zu tensors", 
                model_name.c_str(), tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        RCLCPP_INFO(this->get_logger(), "  Tensor %zu: dims=[%s], data_size=%zu bytes", i,
                    tensors[i].shape.dims.empty() ? "empty" : 
                    std::accumulate(tensors[i].shape.dims.begin(), tensors[i].shape.dims.end(), std::string(),
                        [](const std::string& a, uint32_t b) { return a.empty() ? std::to_string(b) : a + ", " + std::to_string(b); }).c_str(),
                    tensors[i].data.size());
    }

    // Call TensorRT service
    auto future_and_request_id = tensor_client_->async_send_request(request);
    std::shared_future<std::shared_ptr<neuromesh_interfaces::srv::TensorRequest::Response>> future = 
        future_and_request_id.future.share();

    // Process response asynchronously
    std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> return_tensors = 
        std::async(std::launch::async, [this, future, model_name]() {
            std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> output_tensors;
            
            // Wait for response with timeout
            auto status = future.wait_for(std::chrono::seconds(30));
            if (status == std::future_status::timeout) {
                RCLCPP_ERROR(this->get_logger(), "TensorRT service timeout for model: %s", model_name.c_str());
                auto error_tensor = std::make_shared<neuromesh_interfaces::msg::Tensor>();
                error_tensor->result = 4; // Timeout error
                output_tensors.push_back(error_tensor);
                return output_tensors;
            }
            
            auto response = future.get();
            RCLCPP_INFO(this->get_logger(), "Received TensorRT response for model: %s with %zu output tensors", 
                        model_name.c_str(), response->tensor2.size());
            for (const auto& tensor : response->tensor2) {
                output_tensors.emplace_back(std::make_shared<neuromesh_interfaces::msg::Tensor>(tensor));
            }
            return output_tensors;
        });

    return return_tensors;
}

bool VggtToyImplementation::buildDecoderTensor(
    std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr> buffer,
    std::map<std::string, double> buffer_timestamp,
    neuromesh_interfaces::msg::Tensor& own_tensor,
    neuromesh_interfaces::msg::Tensor& neighbour_tensor) {
    
    RCLCPP_DEBUG(this->get_logger(), "Building VGGT decoder tensor with %zu features", buffer.size());
    startClock("vggt_decoder_tensor");

    // VGGT requires exactly 2 agents (current + 1 neighbor)
    if (buffer.size() < 2) {
        RCLCPP_WARN(this->get_logger(), "Not enough features for VGGT decoder (need 2, have %zu)", buffer.size());
        return false;
    }

    // Find our own feature
    auto own_feature_it = buffer.find(id_);
    if (own_feature_it == buffer.end()) {
        RCLCPP_WARN(this->get_logger(), "Own feature not found in buffer for agent: %s", id_.c_str());
        return false;
    }

    // Find neighbor feature (select the most recent one if multiple neighbors)
    neuromesh_interfaces::msg::Feature::SharedPtr neighbor_feature = nullptr;
    std::string neighbor_id;
    double most_recent_timestamp = 0.0;

    for (const auto& [agent_id, feature] : buffer) {
        if (agent_id != id_) {
            double timestamp = buffer_timestamp[agent_id];
            if (timestamp > most_recent_timestamp) {
                most_recent_timestamp = timestamp;
                neighbor_feature = feature;
                neighbor_id = agent_id;
            }
        }
    }

    if (!neighbor_feature) {
        RCLCPP_WARN(this->get_logger(), "No neighbor feature found for VGGT decoder");
        return false;
    }

    // Extract feature tensors
    auto own_feature_tensor = own_feature_it->second->tensor;
    auto neighbor_feature_tensor = neighbor_feature->tensor;

    // Validate feature dimensions for VGGT: should be 1x1036x1024
    if (own_feature_tensor.shape.dims.size() < 3 || 
        own_feature_tensor.shape.dims[0] != 1 || 
        own_feature_tensor.shape.dims[1] != 1036 || 
        own_feature_tensor.shape.dims[2] != 1024) {
        RCLCPP_WARN(this->get_logger(), "Invalid own feature dimensions for VGGT: [%u, %u, %u]", 
                   own_feature_tensor.shape.dims[0], own_feature_tensor.shape.dims[1], own_feature_tensor.shape.dims[2]);
        return false;
    }

    if (neighbor_feature_tensor.shape.dims.size() < 3 || 
        neighbor_feature_tensor.shape.dims[0] != 1 || 
        neighbor_feature_tensor.shape.dims[1] != 1036 || 
        neighbor_feature_tensor.shape.dims[2] != 1024) {
        RCLCPP_WARN(this->get_logger(), "Invalid neighbor feature dimensions for VGGT: [%u, %u, %u]", 
                   neighbor_feature_tensor.shape.dims[0], neighbor_feature_tensor.shape.dims[1], neighbor_feature_tensor.shape.dims[2]);
        return false;
    }

    // Create combined tensor for VGGT decoder input: 2x1036x1024
    neuromesh_interfaces::msg::Tensor combined_tensor;
    combined_tensor.shape.dims = {2, 1036, 1024};
    combined_tensor.data_type = 9; // float32
    
    // Extract float data from tensors
    std::vector<float> own_float_data(own_feature_tensor.data.size() / sizeof(float));
    std::memcpy(own_float_data.data(), own_feature_tensor.data.data(), own_feature_tensor.data.size());
    
    std::vector<float> neighbor_float_data(neighbor_feature_tensor.data.size() / sizeof(float));
    std::memcpy(neighbor_float_data.data(), neighbor_feature_tensor.data.data(), neighbor_feature_tensor.data.size());

    // Concatenate features: first our own, then neighbor's
    std::vector<float> combined_float_data;
    combined_float_data.reserve(2 * 1036 * 1024);
    combined_float_data.insert(combined_float_data.end(), own_float_data.begin(), own_float_data.end());
    combined_float_data.insert(combined_float_data.end(), neighbor_float_data.begin(), neighbor_float_data.end());
    
    // Convert back to uint8 data
    combined_tensor.data.resize(combined_float_data.size() * sizeof(float));
    std::memcpy(combined_tensor.data.data(), combined_float_data.data(), combined_float_data.size() * sizeof(float));

    // Set metadata
    combined_tensor.name = "vggt_combined_features_" + id_ + "_" + neighbor_id;

    // For VGGT decoder, we pass the same combined tensor as both inputs
    own_tensor = combined_tensor;
    neighbour_tensor = combined_tensor;

    stopClock("vggt_decoder_tensor");
    RCLCPP_DEBUG(this->get_logger(), "VGGT decoder tensor built in %ld ms", checkClock("vggt_decoder_tensor"));
    RCLCPP_INFO(this->get_logger(), "Built VGGT decoder tensor: [%u, %u, %u] with %zu elements", 
               combined_tensor.shape.dims[0], combined_tensor.shape.dims[1], combined_tensor.shape.dims[2],
               combined_float_data.size());

    return true;
}

void VggtToyImplementation::tensor_callback(const neuromesh_interfaces::msg::Tensor::SharedPtr /* msg */) {
    RCLCPP_DEBUG(this->get_logger(), "Received tensor callback");
    // Handle any additional tensor processing if needed
}

} // namespace vggtNode

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vggtNode::VggtToyImplementation)