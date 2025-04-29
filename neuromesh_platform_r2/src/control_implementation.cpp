#include "neuromesh_platform_r2/control_implementation.h"

namespace ControlneuromeshNode {
ControlImplementation::ControlImplementation(const rclcpp::NodeOptions &options) : ControlneuromeshNode(options) {

    // Subscribe to position and velocity topics
    pos_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "odom_position_topic", 10, std::bind(&ControlImplementation::pos_callback, this, std::placeholders::_1));

    vel_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "odom_velocity_topic", 10, std::bind(&ControlImplementation::vel_callback, this, std::placeholders::_1));

    this->tensor_client_ = create_client<neuromesh_interfaces::srv::TensorRequest>("tensorrt_request");
}

std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> ControlImplementation::performInference(
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

bool ControlImplementation::buildDecoderTensor(
    const std::map<std::string, neuromesh_interfaces::msg::CommMessage>& agent_features,
    neuromesh_interfaces::msg::Tensor& aggregated_tensor) {

    // Check if we have exactly 3 agents
    if (agent_features.size() != 2) {
        RCLCPP_WARN(this->get_logger(), "Expected 2 agents, but got %zu", agent_features.size());
        return false;
    }

    // Check if all agents have features
    for (const auto& [agent_id, feature] : agent_features) {
        if (feature.tensor.data.empty()) {
            RCLCPP_WARN(this->get_logger(), "Agent %s has no features", agent_id.c_str());
            return false;
        }
    }

    // Initialize aggregated tensor
    const auto& first_feature = agent_features.begin()->second.tensor;
    aggregated_tensor.shape.dims = {3, static_cast<uint32_t>(first_feature.data.size())};
    aggregated_tensor.data_type = first_feature.data_type;
    aggregated_tensor.data.clear();

    // Reserve space for all data
    aggregated_tensor.data.reserve(3 * first_feature.data.size());

    // Aggregate features from all agents
    for (const auto& [agent_id, feature] : agent_features) {
        if (feature.tensor.data.size() != first_feature.data.size()) {
            RCLCPP_WARN(this->get_logger(), "Inconsistent feature size for agent %s", agent_id.c_str());
            return false;
        }
        aggregated_tensor.data.insert(aggregated_tensor.data.end(),  
                                      feature.tensor.data.begin(), 
                                      feature.tensor.data.end());
    }

    RCLCPP_INFO(this->get_logger(), "Building decoder tensor with %d features.", agent_features.size());
    startClock("decoder_tensor");


    return true;
    

    // Find the latest timestamp
    /*auto latest_timestamp = std::max_element(
        feature_buffer_timestamp_.begin(), feature_buffer_timestamp_.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; }
    )->second;*/

    // Initialize aggregated tensor
    
    //for (int kk=0; kk<agent_features.begin()->second.tensor.data.size() ; ++kk){
    //	RCLCPP_WARN(this->get_logger(), "kk=%d",kk);
    //    aggregated_tensor.data.push_back(0.0);
    //}

    
        // Calculate time difference in seconds
        // double time_diff = (latest_timestamp - feature_buffer_timestamp_[agent_id]);

        // aggregated_tensor.data.push_back(static_cast<float>(time_diff));
	/*if (agent_id != "agent1"){ 
	    std::transform(aggregated_tensor.data.begin(), aggregated_tensor.data.end(),
                           feature.tensor.data.begin(), aggregated_tensor.data.begin(), std::plus<float>());
	    std::transform(aggregated_tensor.data.begin(), aggregated_tensor.data.end(), 
               agent_features.at("agent1").tensor.data.begin(), aggregated_tensor.data.begin(), 
               [](float a, float b) { return a - b; });
	} else {
	    std::transform(aggregated_tensor.data.begin(), aggregated_tensor.data.end(),
			    feature.tensor.data.begin(), aggregated_tensor.data.begin(), std::plus<float>());
	}*/
        
}
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(ControlneuromeshNode::ControlImplementation)
