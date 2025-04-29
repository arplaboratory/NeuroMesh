#ifndef COMPRESSION_NODE_H
#define COMPRESSION_NODE_H

#include <rclcpp/rclcpp.hpp>
#include "neuromesh_interfaces/msg/compressed_tensor.hpp"
#include "neuromesh_interfaces/msg/tensor.hpp"

#include "neuromesh_interfaces/msg/compressed_feature.hpp"
#include "neuromesh_interfaces/msg/feature.hpp"


class CompressionNode : public rclcpp::Node
{

public:
    CompressionNode();

protected:
    /**
    * @brief Compresses a tensor.
    * @param msg The tensor to be compressed
    * @return The compressed tensor.
    */
    neuromesh_interfaces::msg::CompressedTensor compress(const std::shared_ptr<neuromesh_interfaces::msg::Tensor> msg);

    /**
    * @brief This function is called whenever a new tensor is received. It will compress the tensor and publish it.
    * @param msg The tensor message.
    */
    void tensor_callback(const std::shared_ptr<neuromesh_interfaces::msg::Tensor> msg);

    /**
    * @brief This function is called whenever a new feature is received. It will compress the feature and publish it.
    * @param msg The feature message.
    */
    void feature_callback(const std::shared_ptr<neuromesh_interfaces::msg::Feature> msg);

    /**
     * @brief Takes a string given as a parameter to the node and converts it to a QoS profile.
     * 
     * NOTE: implementation taken from https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox/nvblox_ros_common/src/qos.cpp#L26
     * 
     * @param str string corresponding to the QoS profile
     * @return QoS profile
     */
    rmw_qos_profile_t parseQoSString(const std::string& str);

    rclcpp::Subscription<neuromesh_interfaces::msg::Tensor>::SharedPtr tensor_subscription_;
    rclcpp::Publisher<neuromesh_interfaces::msg::CompressedTensor>::SharedPtr tensor_publisher_;

    rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr feature_subscription_;
    rclcpp::Publisher<neuromesh_interfaces::msg::CompressedFeature>::SharedPtr feature_publisher_;

    std::string input_name_;
    std::string output_name_;
    int level_;
    bool use_features_;
    std::string output_qos_;

};

#endif // COMPRESSION_NODE_H