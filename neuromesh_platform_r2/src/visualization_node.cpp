#include "neuromesh_platform_r2/visualization_node.h"

VisualizationNode::VisualizationNode() : Node("visualization_node") {
  this->declare_parameter<std::string>("input_topic", "tensorrt_output");
  this->declare_parameter<std::string>("output_topic", "visualization");
  this->declare_parameter<std::string>("datatype", "fp32");
  this->declare_parameter<std::string>("modeltype", "depth");
  this->declare_parameter<int>("height", 0);
  this->declare_parameter<int>("width", 0);
  this->declare_parameter<bool>("use_features", false);

  this->get_parameter("input_topic", input_topic_);
  this->get_parameter("output_topic", output_topic_);
  this->get_parameter("datatype", datatype_);
  this->get_parameter("modeltype", modeltype_);
  this->get_parameter("height", height_);
  this->get_parameter("width", width_);
  this->get_parameter("use_features", use_features_);

  // check for compatibility

  if (datatype_ == "fp32") {
    RCLCPP_INFO(this->get_logger(), "Visualization node: datatype is fp32");
  } else {
    RCLCPP_ERROR(this->get_logger(),
                 "Visualization node: datatype not supported");
    rclcpp::shutdown();
  }

  if (modeltype_ == "depth") {
    RCLCPP_INFO(this->get_logger(), "Visualization node: modeltype is depth");
  } else {
    RCLCPP_ERROR(this->get_logger(),
                 "Visualization node: modeltype not supported");
    rclcpp::shutdown();
  }

  if (height_ == 0 || width_ == 0) {
    RCLCPP_ERROR(this->get_logger(),
                 "Visualization node: height or width not set");
    rclcpp::shutdown();
  }

  if (use_features_) {
    feature_subscription_ =
        this->create_subscription<neuromesh_interfaces::msg::Feature>(
            input_topic_, 10,
            std::bind(&VisualizationNode::feature_callback, this,
                      std::placeholders::_1));
  } else {
    // create subscription
    tensor_subscription_ =
        this->create_subscription<neuromesh_interfaces::msg::Tensor>(
            input_topic_, 10,
            std::bind(&VisualizationNode::tensor_callback, this,
                      std::placeholders::_1));
  }
  // create publisher
  publisher_ =
      this->create_publisher<sensor_msgs::msg::Image>(output_topic_, 10);
}

void VisualizationNode::tensor_callback(
    const neuromesh_interfaces::msg::Tensor::SharedPtr msg) {
  // convert tensor to image
  sensor_msgs::msg::Image image_msg = tensorToImage(msg);

  // publish image
  publisher_->publish(image_msg);
}

void VisualizationNode::feature_callback(
    const neuromesh_interfaces::msg::Feature::SharedPtr msg) {
  // convert tensor to image
  sensor_msgs::msg::Image image_msg = tensorToImage(
      std::make_shared<neuromesh_interfaces::msg::Tensor>(msg->tensor));

  // publish image
  publisher_->publish(image_msg);
}

sensor_msgs::msg::Image VisualizationNode::tensorToImage(
    const neuromesh_interfaces::msg::Tensor::SharedPtr msg) {

  if (modeltype_ != "depth" || datatype_ != "fp32") {
    RCLCPP_ERROR(this->get_logger(), "Visualization node: tensorToImage: "
                                     "modeltype or datatype not supported");
    rclcpp::shutdown();
  }

  sensor_msgs::msg::Image image_msg;

  image_msg.height = height_;
  image_msg.width = width_;
  image_msg.encoding = "32FC1";
  image_msg.is_bigendian = false;
  image_msg.step = width_ * sizeof(float);
  image_msg.data = msg->data;

  return image_msg;
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto visualization_node = std::make_shared<VisualizationNode>();
  rclcpp::spin(visualization_node);
  rclcpp::shutdown();
  return 0;
}
