#ifndef VISUALIZATION_NODE_H
#define VISUALIZATION_NODE_H

#include "rclcpp/rclcpp.hpp"

#include "neuromesh_interfaces/msg/tensor.hpp"
#include "neuromesh_interfaces/msg/feature.hpp"
#include <sensor_msgs/msg/image.hpp>

class VisualizationNode : public rclcpp::Node
{
    //FUNCTIONS

public:
    VisualizationNode();


protected:

    /**
    * @brief This function is called whenever a new tensor is received. It will convert the tensor to a visualization format and publish it.
    * @param msg The tensor message.
    */
    void tensor_callback(const neuromesh_interfaces::msg::Tensor::SharedPtr msg);
    /**
    * @brief This function is called whenever a new feature is received. It will convert the feature to a visualization format and publish it.
    * @param msg The feature message.
    */
    void feature_callback(const neuromesh_interfaces::msg::Feature::SharedPtr msg);

    /**
    * @brief Converts a tensor to ROS image format
    * @param msg The tensor to be converted
    * @return The ROS image.
    */
    sensor_msgs::msg::Image tensorToImage(const neuromesh_interfaces::msg::Tensor::SharedPtr msg);

    //VARIABLES

    rclcpp::Subscription<neuromesh_interfaces::msg::Tensor>::SharedPtr tensor_subscription_;
    rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr feature_subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;

    std::string input_topic_;
    std::string output_topic_;
    std::string datatype_;
    std::string modeltype_;
    int height_;
    int width_;
    bool use_features_;
};


#endif //VISUALIZATION_NODE_H