#include "neuromesh_platform_r2/compression_node.h"
#include "gzip/compress.hpp"

CompressionNode::CompressionNode() : Node("compression_node") {
  // Declare parameters
  this->declare_parameter<std::string>("input_name", "compressed_tensor");
  this->declare_parameter<std::string>("output_name", "tensor");
  this->declare_parameter<int>("level", 9); // 1-9
  this->declare_parameter<bool>("use_features", true);
  this->declare_parameter<std::string>("output_qos", "SENSOR_DATA");

  // Get parameters
  this->get_parameter("input_name", input_name_);
  this->get_parameter("output_name", output_name_);
  this->get_parameter("level", level_);
  this->get_parameter("use_features", use_features_);
  this->get_parameter("output_qos", output_qos_);

  auto output_qos =
      rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(output_qos_));

  if (!use_features_) {
    // Create publisher
    tensor_publisher_ =
        this->create_publisher<neuromesh_interfaces::msg::CompressedTensor>(
            output_name_, output_qos);

    // Create subscriber
    tensor_subscription_ =
        this->create_subscription<neuromesh_interfaces::msg::Tensor>(
            input_name_, 10,
            std::bind(&CompressionNode::tensor_callback, this,
                      std::placeholders::_1));

  } else {

    // Create publisher
    feature_publisher_ =
        this->create_publisher<neuromesh_interfaces::msg::CompressedFeature>(
            output_name_, output_qos);

    // Create subscriber
    feature_subscription_ =
        this->create_subscription<neuromesh_interfaces::msg::Feature>(
            input_name_, 10,
            std::bind(&CompressionNode::feature_callback, this,
                      std::placeholders::_1));
  }
}

void CompressionNode::tensor_callback(
    const std::shared_ptr<neuromesh_interfaces::msg::Tensor> msg) {
  // compress tensor
  auto tensor = compress(msg);

  int new_size =
      (int)(100 * (float)tensor.data.size() / (float)msg->data.size());

  RCLCPP_INFO(this->get_logger(), "Tensor is %i%% of original", new_size);

  // Publish tensor
  tensor_publisher_->publish(tensor);
}

void CompressionNode::feature_callback(
    const std::shared_ptr<neuromesh_interfaces::msg::Feature> msg) {
  neuromesh_interfaces::msg::Tensor::SharedPtr tensor =
      std::make_shared<neuromesh_interfaces::msg::Tensor>(msg->tensor);

  neuromesh_interfaces::msg::CompressedFeature feature =
      neuromesh_interfaces::msg::CompressedFeature();

  feature.tensor = this->compress(tensor);
  feature.id = msg->id;
  feature.timestamp = msg->timestamp;

  int new_size = (int)(100 * (float)feature.tensor.data.size() /
                       (float)msg->tensor.data.size());
  RCLCPP_INFO(this->get_logger(), "Tensor is %i%% of original", new_size);

  feature_publisher_->publish(feature);
}

neuromesh_interfaces::msg::CompressedTensor CompressionNode::compress(
    const std::shared_ptr<neuromesh_interfaces::msg::Tensor> msg) {

  char *input_ptr = reinterpret_cast<char *>(msg->data.data());

  std::string output = gzip::compress(input_ptr, msg->data.size(), level_);

  neuromesh_interfaces::msg::CompressedTensor compressed_tensor;

  compressed_tensor.name = msg->name;
  compressed_tensor.shape = msg->shape;
  compressed_tensor.data_type = msg->data_type;
  compressed_tensor.strides = msg->strides;

  compressed_tensor.data = std::vector<uint8_t>(output.begin(), output.end());

  return compressed_tensor;
}

// Convert string to ROS2 QoS profile
// from
// https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox/nvblox_ros_common/src/qos.cpp#L26
rmw_qos_profile_t CompressionNode::parseQoSString(const std::string &str) {
  std::string profile = str;
  // Convert to upper case.
  std::transform(profile.begin(), profile.end(), profile.begin(), ::toupper);

  if (profile == "SYSTEM_DEFAULT") {
    return rmw_qos_profile_system_default;
  }
  if (profile == "DEFAULT") {
    return rmw_qos_profile_default;
  }
  if (profile == "PARAMETER_EVENTS") {
    return rmw_qos_profile_parameter_events;
  }
  if (profile == "SERVICES_DEFAULT") {
    return rmw_qos_profile_services_default;
  }
  if (profile == "PARAMETERS") {
    return rmw_qos_profile_parameters;
  }
  if (profile == "SENSOR_DATA") {
    return rmw_qos_profile_sensor_data;
  }
  RCLCPP_WARN_STREAM(rclcpp::get_logger("parseQosString"),
                     "Unknown QoS profile: " << profile
                                             << ". Returning profile: DEFAULT");
  return rmw_qos_profile_default;
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto compression_node = std::make_shared<CompressionNode>();
  rclcpp::spin(compression_node);
  rclcpp::shutdown();
  return 0;
}
