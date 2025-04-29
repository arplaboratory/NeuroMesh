#include "neuromesh_platform_r2/dust3r_toy_implementation.h"
#include <chrono>
#include <rclcpp/callback_group.hpp>

namespace neuromeshNode {
ToyImplementation::ToyImplementation(const rclcpp::NodeOptions &options) : neuromeshNode(options){
    
    std::string tensor_qos_profile_;

    this->declare_parameter<std::string>("tensor_qos_profile", "default");
    this->get_parameter("tensor_qos_profile", tensor_qos_profile_);

    auto tensor_qos = rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(tensor_qos_profile_));
    
    auto main_opt = rclcpp::SubscriptionOptions();

    auto tensor_opt = rclcpp::SubscriptionOptions();

    //redefine subscriptions with the callback groups

    auto image_qos = rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(output_qos_profile_));

    camera_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
			"camera",
			image_qos,
			std::bind(&ToyImplementation::camera_callback, this, std::placeholders::_1),
            // parseQoSString(image_qos_profile_),
            main_opt);

    decoder_timer_ = this->create_wall_timer(std::chrono::duration<int,std::milli>(decoder_cycle_length_), 
                                            std::bind(&ToyImplementation::process_features, this));

    encoder_timer_ = this->create_wall_timer(std::chrono::duration<int,std::milli>(decoder_cycle_length_), 
                                          std::bind(&ToyImplementation::run_encoder_cycle, this));

    //define our subscriptions
    this->tensor_client_ = create_client<neuromesh_interfaces::srv::TensorRequest>("tensorrt_request");

}

std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> ToyImplementation::performInference(
    const std::string& model_name,
    const std::vector<neuromesh_interfaces::msg::Tensor>& tensors)
{
    if (!tensor_client_->wait_for_service(std::chrono::seconds(1))) {
        RCLCPP_ERROR(this->get_logger(), "Engine not reachable via service.");
        // Complex stuff just to return future that resolves to empty tensor
        std::promise<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> prom;
        std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> r = prom.get_future();
        std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> t(1, std::make_shared<neuromesh_interfaces::msg::Tensor>());
        t[0]->result = 3; // Cannot reach engine error code
        prom.set_value(t);
        return r;
    }

    // Create a request to send to the service server
    auto request = std::make_shared<neuromesh_interfaces::srv::TensorRequest::Request>();
    request->model_name = model_name;
    request->tensor1 = tensors;

    // Call the service and wait for the response
    std::shared_future<std::shared_ptr<neuromesh_interfaces::srv::TensorRequest::Response>> future = tensor_client_->async_send_request(request);

    std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> return_tensors = std::async(std::launch::async, [future]() {
        std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> output_tensors;
        for (const auto& tensor : future.get()->tensor2) {
            output_tensors.emplace_back(std::make_shared<neuromesh_interfaces::msg::Tensor>(tensor));
        }
        return output_tensors;
    });

    return return_tensors;
}

bool ToyImplementation::buildDecoderTensor(
    std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr> buffer,
    std::map<std::string, double> buffer_timestamp,
    neuromesh_interfaces::msg::Tensor& own_tensor,
    neuromesh_interfaces::msg::Tensor& neighbour_tensor,
    neuromesh_interfaces::msg::Tensor& pos1,
    neuromesh_interfaces::msg::Tensor& pos2,
    const YAML::Node& pos1_yaml,
    const YAML::Node& pos2_yaml)
{
    RCLCPP_DEBUG(this->get_logger(), "Inside buildDecoderTensor function");
    RCLCPP_INFO(this->get_logger(), "Building decoder tensor with %d features.", buffer.size());
    startClock("decoder_tensor");

    // Check if we have our own feature and at least one other feature
    if (buffer.size() != 2 || buffer_timestamp.size() != 2) {
        RCLCPP_WARN(this->get_logger(), "Not enough features to build decoder tensor");
        return false;
    }

    // Find the neighbor with the largest timestamp difference
    std::string neighbor_id;
    double max_time_diff = 0;
    double own_timestamp = buffer_timestamp[id_];

    for (const auto& [key, timestamp] : buffer_timestamp) {
        if (key != id_) {
            double time_diff = std::abs(timestamp - own_timestamp);
            if (time_diff > max_time_diff) {
                max_time_diff = time_diff;
                neighbor_id = key;
            }
        }
    }

    if (neighbor_id.empty()) {
        RCLCPP_WARN(this->get_logger(), "No valid neighbor found");
        return false;
    }

    // Build the decoder tensor using our own feature and the selected neighbor's feature
    own_tensor = buffer[id_]->tensor;
    neighbour_tensor = buffer[neighbor_id]->tensor;

    //RCLCPP_INFO(this->get_logger(), "Own tensor data_type: %d", own_tensor.data_type);
    //RCLCPP_INFO(this->get_logger(), "Neighbor tensor data_type: %d", neighbor_tensor.data_type);

    // For own tensor
    own_tensor.name = "own_input";
    // Data type, strides, and shape should already be set from the buffer
    // But we can verify and set them explicitly if needed:
    own_tensor.data_type = buffer[id_]->tensor.data_type;
    own_tensor.strides = buffer[id_]->tensor.strides;
    own_tensor.shape = buffer[id_]->tensor.shape;

    // For neighbor_tensor
    neighbour_tensor.name = "neighbor_input";
    // Similarly, set these explicitly:
    neighbour_tensor.data_type = buffer[neighbor_id]->tensor.data_type;
    neighbour_tensor.strides = buffer[neighbor_id]->tensor.strides;
    neighbour_tensor.shape = buffer[neighbor_id]->tensor.shape;

    // Preparing shape1 and shape2 tensors
    // Use std::vector<int> for shape1 and shape2
    // std::vector<int> shape_data = {384, 512};

    // // Set the data type for shape1 and shape2
    // shape1.data_type = 5;
    // shape2.data_type = 5;

    // // Resize the data vector and copy the values
    // shape1.data.resize(shape_data.size() * sizeof(int));
    // std::memcpy(shape1.data.data(), shape_data.data(), shape_data.size() * sizeof(int));

    // shape2.data.resize(shape_data.size() * sizeof(int));
    // std::memcpy(shape2.data.data(), shape_data.data(), shape_data.size() * sizeof(int));

    // // Set the shape of the tensors
    // shape1.shape.dims = {2};  // 2 because we have two values: 384 and 512
    // shape2.shape.dims = {2};

    // Preparing pos1 and pos2 tensors
    // if (pos1_yaml["pos1"]) {
    //     std::vector<int> pos1_data = pos1_yaml["pos1"].as<std::vector<int>>();
    //     pos1.data_type = 5; // Assuming float data type
    //     pos1.data.resize(pos1_data.size() * sizeof(int));
    //     std::memcpy(pos1.data.data(), pos1_data.data(), pos1_data.size() * sizeof(int));
    //     pos1.shape.dims = {static_cast<int>(pos1_data.size())};
    // } else {
    //     RCLCPP_WARN(this->get_logger(), "pos1 not found in YAML config");
    // }

    // // Load pos2
    // if (pos2_yaml["pos2"]) {
    //     std::vector<int> pos2_data = pos2_yaml["pos2"].as<std::vector<int>>();
    //     pos2.data_type = 5; // Assuming float data type
    //     pos2.data.resize(pos2_data.size() * sizeof(int));
    //     std::memcpy(pos2.data.data(), pos2_data.data(), pos2_data.size() * sizeof(int));
    //     pos2.shape.dims = {static_cast<int>(pos2_data.size())};
    // } else {
    //     RCLCPP_WARN(this->get_logger(), "pos2 not found in YAML config");
    // }

    if (pos1_yaml["pos1"]) {
    auto pos1_data = pos1_yaml["pos1"].as<std::vector<std::vector<std::vector<int>>>>();
    if (!pos1_data.empty() && !pos1_data[0].empty()) {
        pos1.data_type = 7; // INT32
        pos1.shape.dims = {1, static_cast<int>(pos1_data[0].size()), 2};
        pos1.data.resize(pos1.shape.dims[0] * pos1.shape.dims[1] * pos1.shape.dims[2] * sizeof(int));
        
        int* data_ptr = reinterpret_cast<int*>(pos1.data.data());
        for (const auto& row : pos1_data[0]) {
            if (row.size() == 2) {
                *data_ptr++ = row[0];
                *data_ptr++ = row[1];
            } else {
                RCLCPP_ERROR(this->get_logger(), "Invalid data format in pos1: each inner array should have exactly 2 elements");
                return false;
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

    // Load pos2 (similar to pos1)
    if (pos2_yaml["pos2"]) {
        auto pos2_data = pos2_yaml["pos2"].as<std::vector<std::vector<std::vector<int>>>>();
        if (!pos2_data.empty() && !pos2_data[0].empty()) {
            pos2.data_type = 7; // INT32
            pos2.shape.dims = {1, static_cast<int>(pos2_data[0].size()), 2};
            pos2.data.resize(pos2.shape.dims[0] * pos2.shape.dims[1] * pos2.shape.dims[2] * sizeof(int));
            
            int* data_ptr = reinterpret_cast<int*>(pos2.data.data());
            for (const auto& row : pos2_data[0]) {
                if (row.size() == 2) {
                    *data_ptr++ = row[0];
                    *data_ptr++ = row[1];
                } else {
                    RCLCPP_ERROR(this->get_logger(), "Invalid data format in pos2: each inner array should have exactly 2 elements");
                    return false;
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

    stopClock("decoder_tensor");
    RCLCPP_DEBUG(this->get_logger(), "Built decoder tensor in %lims", times["decoder_tensor"].first);

    //decoder_tensor = tensor_ints_to_floats(decoder_tensor);

    RCLCPP_DEBUG(this->get_logger(), "Returning decoder tensor");

    return true;
}


/*int main(int argc, char** argv)
{   
    rclcpp::init(argc, argv);

    rclcpp::executors::MultiThreadedExecutor executor;
	auto toy_implementation = std::make_shared<ToyImplementation>();
	
    executor.add_node(toy_implementation);
    executor.spin();

    rclcpp::shutdown();

	return 0;
}*/

}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(neuromeshNode::ToyImplementation)
