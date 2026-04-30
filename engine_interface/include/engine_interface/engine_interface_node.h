#ifndef ENGINE_INTERFACE_NODE_H
#define ENGINE_INTERFACE_NODE_H

#define INFERENCE_HELPER_ENABLE_TENSORRT

#include <cstdio>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include <pluginlib/class_loader.hpp>

#include "neuromesh_interfaces/msg/tensor.hpp"
#include "neuromesh_interfaces/srv/tensor_request.hpp"
#include "std_msgs/msg/header.hpp"

#include "engine_interface/inference_engine_base.hpp"


namespace engine_interface {

class EngineInterfaceNode : public rclcpp::Node {
public:
    EngineInterfaceNode(
        rclcpp::NodeOptions options, 
        const std::string& plugin_class = "engine_interface::BaseEngine"
    );
    ~EngineInterfaceNode() override = default;

private:
    pluginlib::ClassLoader<BaseEngine> engine_loader_;
    std::shared_ptr<BaseEngine> engine_;

    rclcpp::Publisher<neuromesh_interfaces::msg::Tensor>::SharedPtr
        tensor_publisher_;
    rclcpp::Subscription<neuromesh_interfaces::msg::Tensor>::SharedPtr
        tensor_subscription_;
    rclcpp::Service<neuromesh_interfaces::srv::TensorRequest>::SharedPtr service_;

    // params
    std::string models_param;
    std::vector<std::string> model_names;

    std::unordered_map<std::string, std::string> model_paths;
    std::unordered_map<std::string, std::vector<std::vector<uint>>>
        input_dimensions;
    std::unordered_map<std::string, std::vector<std::vector<uint>>>
        output_dimensions;
    std::unordered_map<std::string, std::string> input_dimensions_strings;
    std::unordered_map<std::string, std::string> output_dimensions_strings;
    std::unordered_map<std::string, std::string> tensor_type_params;
    std::string tensor_qos_param;

    // engine
    std::unordered_map<std::string, std::shared_ptr<BaseEngine>> engines;
    std::unordered_map<std::string, std::vector<uint32_t>> input_lengths;
    std::unordered_map<std::string, std::vector<uint32_t>> output_lengths;
    std::unordered_map<std::string, int> tensor_typelengths;

    std::vector<int> failed_models;

    //  callbacks
    void tensor_request_callback(
        const std::shared_ptr<neuromesh_interfaces::srv::TensorRequest::Request>
            request,
        const std::shared_ptr<neuromesh_interfaces::srv::TensorRequest::Response>
            response);

    // execution
    std::vector<neuromesh_interfaces::msg::Tensor>
    execute(const std::string &model,
        const std::vector<neuromesh_interfaces::msg::Tensor> &tensor_msgs);

    // helper functions
    int tensor_string_to_typelength(std::string input);
    std::vector<std::string> string_to_vector(std::string in);
    std::vector<uint> string_to_dims_single(std::string in);
    std::vector<std::vector<uint>> string_to_dims(std::string in);

    // Convert string to ROS2 QoS profile
    rmw_qos_profile_t parseQoSString(const std::string &str);
};
} // namespace engine_interface
#endif
