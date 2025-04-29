#ifndef CONTROL_IMPLEMENTATION_HEADER_H
#define CONTROL_IMPLEMENTATION_HEADER_H

#include "neuromesh_platform_r2/control_neuromesh_node.h"
#include "neuromesh_interfaces/srv/tensor_request.hpp"
#include <vector>
#include <map>
#include <memory>
#include "rclcpp/rclcpp.hpp"

namespace ControlneuromeshNode {
class ControlImplementation : public ControlneuromeshNode {
public:
    ControlImplementation(const rclcpp::NodeOptions &options);

protected:

    // Perform inference
    std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> performInference(
    const std::string& model_name,
    const std::vector<neuromesh_interfaces::msg::Tensor>& tensors);

    // Build decoder
    bool buildDecoderTensor(const std::map<std::string, neuromesh_interfaces::msg::CommMessage>& agent_features,
                            neuromesh_interfaces::msg::Tensor& aggregated_tensor);

    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr goal_subscription_;
    rclcpp::Subscription<neuromesh_interfaces::msg::StateVector>::SharedPtr state_subscription_;
    rclcpp::Subscription<neuromesh_interfaces::msg::CommMessage>::SharedPtr feature_subscription_;
    rclcpp::Publisher<neuromesh_interfaces::msg::CommMessage>::SharedPtr feature_publisher_;
    rclcpp::Publisher<neuromesh_interfaces::msg::CommMessage>::SharedPtr result_publisher_;
    rclcpp::Client<neuromesh_interfaces::srv::TensorRequest>::SharedPtr tensor_client_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pos_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr vel_sub_;


    std::map<std::string, geometry_msgs::msg::Pose> goal_poses_;
    std::map<std::string, neuromesh_interfaces::msg::StateVector> current_states_;
};
}
#endif