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
    
    RCLCPP_INFO(this->get_logger(), "=== START performInference for model: %s ===", model_name.c_str());
    
    try {
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

        RCLCPP_INFO(this->get_logger(), "TensorRT service is available");

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
        RCLCPP_INFO(this->get_logger(), "Calling async_send_request...");
        auto future_and_request_id = tensor_client_->async_send_request(request);
        std::shared_future<std::shared_ptr<neuromesh_interfaces::srv::TensorRequest::Response>> future = 
            future_and_request_id.future.share();
        RCLCPP_INFO(this->get_logger(), "Service request sent successfully");

        // Process response asynchronously
        std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> return_tensors = 
            std::async(std::launch::async, [this, future, model_name]() {
                RCLCPP_INFO(this->get_logger(), "Async task started for model: %s", model_name.c_str());
                std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> output_tensors;
                
                try {
                    // Wait for response with timeout
                    RCLCPP_INFO(this->get_logger(), "Waiting for TensorRT response with 30s timeout...");
                    auto status = future.wait_for(std::chrono::seconds(30));
                    if (status == std::future_status::timeout) {
                        RCLCPP_ERROR(this->get_logger(), "TensorRT service timeout for model: %s", model_name.c_str());
                        auto error_tensor = std::make_shared<neuromesh_interfaces::msg::Tensor>();
                        error_tensor->result = 4; // Timeout error
                        output_tensors.push_back(error_tensor);
                        return output_tensors;
                    }
                    
                    RCLCPP_INFO(this->get_logger(), "Getting response from future...");
                    auto response = future.get();
                    RCLCPP_INFO(this->get_logger(), "Received TensorRT response for model: %s with %zu output tensors", 
                                model_name.c_str(), response->tensor2.size());
                    
                    for (size_t i = 0; i < response->tensor2.size(); ++i) {
                        RCLCPP_INFO(this->get_logger(), "Processing output tensor %zu", i);
                        output_tensors.emplace_back(std::make_shared<neuromesh_interfaces::msg::Tensor>(response->tensor2[i]));
                    }
                    
                    RCLCPP_INFO(this->get_logger(), "Async task completed successfully for model: %s", model_name.c_str());
                    return output_tensors;
                    
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "Exception in async task for model %s: %s", 
                                model_name.c_str(), e.what());
                    auto error_tensor = std::make_shared<neuromesh_interfaces::msg::Tensor>();
                    error_tensor->result = 5; // Exception error
                    output_tensors.push_back(error_tensor);
                    return output_tensors;
                } catch (...) {
                    RCLCPP_ERROR(this->get_logger(), "Unknown exception in async task for model: %s", model_name.c_str());
                    auto error_tensor = std::make_shared<neuromesh_interfaces::msg::Tensor>();
                    error_tensor->result = 5; // Exception error
                    output_tensors.push_back(error_tensor);
                    return output_tensors;
                }
            });

        RCLCPP_INFO(this->get_logger(), "=== END performInference (returning future) ===");
        return return_tensors;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in performInference: %s", e.what());
        
        // Return future that resolves to error tensor
        std::promise<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> prom;
        std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> result = prom.get_future();
        std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> error_tensors(1, 
            std::make_shared<neuromesh_interfaces::msg::Tensor>());
        error_tensors[0]->result = 6; // Exception error
        prom.set_value(error_tensors);
        return result;
    }
}

bool VggtToyImplementation::buildDecoderTensor(
    std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr> buffer,
    std::map<std::string, double> buffer_timestamp,
    neuromesh_interfaces::msg::Tensor& own_tensor,
    neuromesh_interfaces::msg::Tensor& neighbour_tensor) {
    
    RCLCPP_INFO(this->get_logger(), "=== START buildDecoderTensor ===");
    RCLCPP_INFO(this->get_logger(), "Building VGGT decoder tensor with %zu features", buffer.size());
    
    try {
        startClock("vggt_decoder_tensor");

        // VGGT requires exactly 2 agents (current + 1 neighbor)
        if (buffer.size() < 2) {
            RCLCPP_WARN(this->get_logger(), "Not enough features for VGGT decoder (need 2, have %zu)", buffer.size());
            return false;
        }

        // Log all agents in buffer
        RCLCPP_INFO(this->get_logger(), "Agents in buffer:");
        for (const auto& [agent_id, feature] : buffer) {
            RCLCPP_INFO(this->get_logger(), "  - Agent: %s, Feature ptr: %p", 
                       agent_id.c_str(), feature.get());
        }

        // Find our own feature
        RCLCPP_INFO(this->get_logger(), "Looking for own feature with id: %s", id_.c_str());
        auto own_feature_it = buffer.find(id_);
        if (own_feature_it == buffer.end()) {
            RCLCPP_WARN(this->get_logger(), "Own feature not found in buffer for agent: %s", id_.c_str());
            return false;
        }
        RCLCPP_INFO(this->get_logger(), "Found own feature");

        // Validate own feature pointer
        if (!own_feature_it->second) {
            RCLCPP_ERROR(this->get_logger(), "Own feature pointer is null!");
            return false;
        }

        // Find neighbor feature (select the most recent one if multiple neighbors)
        neuromesh_interfaces::msg::Feature::SharedPtr neighbor_feature = nullptr;
        std::string neighbor_id;
        double most_recent_timestamp = 0.0;

        RCLCPP_INFO(this->get_logger(), "Searching for neighbor feature...");
        for (const auto& [agent_id, feature] : buffer) {
            if (agent_id != id_) {
                double timestamp = buffer_timestamp[agent_id];
                RCLCPP_INFO(this->get_logger(), "  Checking neighbor: %s, timestamp: %f", 
                           agent_id.c_str(), timestamp);
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
        RCLCPP_INFO(this->get_logger(), "Selected neighbor: %s with timestamp: %f", 
                   neighbor_id.c_str(), most_recent_timestamp);

        // Extract feature tensors
        RCLCPP_INFO(this->get_logger(), "Extracting feature tensors...");
        auto own_feature_tensor = own_feature_it->second->tensor;
        auto neighbor_feature_tensor = neighbor_feature->tensor;

        // Log tensor sizes
        RCLCPP_INFO(this->get_logger(), "Own tensor data size: %zu bytes", own_feature_tensor.data.size());
        RCLCPP_INFO(this->get_logger(), "Neighbor tensor data size: %zu bytes", neighbor_feature_tensor.data.size());

        // Validate feature dimensions for VGGT: should be 1x1036x1024
        RCLCPP_INFO(this->get_logger(), "Validating tensor dimensions...");
        if (own_feature_tensor.shape.dims.size() < 3) {
            RCLCPP_ERROR(this->get_logger(), "Own feature tensor has only %zu dimensions, expected at least 3", 
                        own_feature_tensor.shape.dims.size());
            return false;
        }
        
        RCLCPP_INFO(this->get_logger(), "Own feature dimensions: [%u, %u, %u]", 
                   own_feature_tensor.shape.dims[0], own_feature_tensor.shape.dims[1], own_feature_tensor.shape.dims[2]);
        
        if (own_feature_tensor.shape.dims[0] != 1 || 
            own_feature_tensor.shape.dims[1] != 1036 || 
            own_feature_tensor.shape.dims[2] != 1024) {
            RCLCPP_WARN(this->get_logger(), "Invalid own feature dimensions for VGGT: [%u, %u, %u]", 
                       own_feature_tensor.shape.dims[0], own_feature_tensor.shape.dims[1], own_feature_tensor.shape.dims[2]);
            return false;
        }

        if (neighbor_feature_tensor.shape.dims.size() < 3) {
            RCLCPP_ERROR(this->get_logger(), "Neighbor feature tensor has only %zu dimensions, expected at least 3", 
                        neighbor_feature_tensor.shape.dims.size());
            return false;
        }
        
        RCLCPP_INFO(this->get_logger(), "Neighbor feature dimensions: [%u, %u, %u]", 
                   neighbor_feature_tensor.shape.dims[0], neighbor_feature_tensor.shape.dims[1], neighbor_feature_tensor.shape.dims[2]);
        
        if (neighbor_feature_tensor.shape.dims[0] != 1 || 
            neighbor_feature_tensor.shape.dims[1] != 1036 || 
            neighbor_feature_tensor.shape.dims[2] != 1024) {
            RCLCPP_WARN(this->get_logger(), "Invalid neighbor feature dimensions for VGGT: [%u, %u, %u]", 
                       neighbor_feature_tensor.shape.dims[0], neighbor_feature_tensor.shape.dims[1], neighbor_feature_tensor.shape.dims[2]);
            return false;
        }

        // Validate data sizes
        size_t expected_size = 1 * 1036 * 1024 * sizeof(float);
        RCLCPP_INFO(this->get_logger(), "Expected tensor data size: %zu bytes", expected_size);
        
        if (own_feature_tensor.data.size() != expected_size) {
            RCLCPP_ERROR(this->get_logger(), "Own tensor data size mismatch! Expected: %zu, Got: %zu", 
                        expected_size, own_feature_tensor.data.size());
            return false;
        }
        
        if (neighbor_feature_tensor.data.size() != expected_size) {
            RCLCPP_ERROR(this->get_logger(), "Neighbor tensor data size mismatch! Expected: %zu, Got: %zu", 
                        expected_size, neighbor_feature_tensor.data.size());
            return false;
        }

        // Create combined tensor for VGGT decoder input: 2x1036x1024
        RCLCPP_INFO(this->get_logger(), "Creating combined tensor...");
        neuromesh_interfaces::msg::Tensor combined_tensor;
        combined_tensor.shape.dims = {2, 1036, 1024};
        combined_tensor.data_type = 9; // float32
        
        // Extract float data from tensors
        RCLCPP_INFO(this->get_logger(), "Extracting float data from own tensor...");
        std::vector<float> own_float_data(own_feature_tensor.data.size() / sizeof(float));
        if (!own_feature_tensor.data.empty()) {
            std::memcpy(own_float_data.data(), own_feature_tensor.data.data(), own_feature_tensor.data.size());
            RCLCPP_INFO(this->get_logger(), "Successfully extracted %zu floats from own tensor", own_float_data.size());
        } else {
            RCLCPP_ERROR(this->get_logger(), "Own tensor data is empty!");
            return false;
        }
        
        RCLCPP_INFO(this->get_logger(), "Extracting float data from neighbor tensor...");
        std::vector<float> neighbor_float_data(neighbor_feature_tensor.data.size() / sizeof(float));
        if (!neighbor_feature_tensor.data.empty()) {
            std::memcpy(neighbor_float_data.data(), neighbor_feature_tensor.data.data(), neighbor_feature_tensor.data.size());
            RCLCPP_INFO(this->get_logger(), "Successfully extracted %zu floats from neighbor tensor", neighbor_float_data.size());
        } else {
            RCLCPP_ERROR(this->get_logger(), "Neighbor tensor data is empty!");
            return false;
        }

        // Concatenate features: first our own, then neighbor's
        RCLCPP_INFO(this->get_logger(), "Concatenating features...");
        std::vector<float> combined_float_data;
        combined_float_data.reserve(2 * 1036 * 1024);
        combined_float_data.insert(combined_float_data.end(), own_float_data.begin(), own_float_data.end());
        combined_float_data.insert(combined_float_data.end(), neighbor_float_data.begin(), neighbor_float_data.end());
        RCLCPP_INFO(this->get_logger(), "Combined float data size: %zu elements", combined_float_data.size());
        
        // Convert back to uint8 data
        RCLCPP_INFO(this->get_logger(), "Converting back to uint8 data...");
        size_t combined_data_size = combined_float_data.size() * sizeof(float);
        combined_tensor.data.resize(combined_data_size);
        std::memcpy(combined_tensor.data.data(), combined_float_data.data(), combined_data_size);
        RCLCPP_INFO(this->get_logger(), "Combined tensor data size: %zu bytes", combined_tensor.data.size());

        // Set metadata
        combined_tensor.name = "vggt_combined_features_" + id_ + "_" + neighbor_id;
        RCLCPP_INFO(this->get_logger(), "Combined tensor name: %s", combined_tensor.name.c_str());

        // For VGGT decoder, we pass the same combined tensor as both inputs
        own_tensor = combined_tensor;
        neighbour_tensor = combined_tensor;

        stopClock("vggt_decoder_tensor");
        RCLCPP_INFO(this->get_logger(), "VGGT decoder tensor built in %ld ms", checkClock("vggt_decoder_tensor"));
        RCLCPP_INFO(this->get_logger(), "Built VGGT decoder tensor: [%u, %u, %u] with %zu elements", 
                   combined_tensor.shape.dims[0], combined_tensor.shape.dims[1], combined_tensor.shape.dims[2],
                   combined_float_data.size());
        
        RCLCPP_INFO(this->get_logger(), "=== END buildDecoderTensor SUCCESS ===");
        return true;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in buildDecoderTensor: %s", e.what());
        return false;
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown exception in buildDecoderTensor");
        return false;
    }
}

void VggtToyImplementation::tensor_callback(const neuromesh_interfaces::msg::Tensor::SharedPtr /* msg */) {
    RCLCPP_DEBUG(this->get_logger(), "Received tensor callback");
    // Handle any additional tensor processing if needed
}

} // namespace vggtNode

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vggtNode::VggtToyImplementation)