#include "neuromesh_platform_r2/decompression_node.h"
#include "gzip/decompress.hpp"


DecompressionNode::DecompressionNode() : Node("decompression_node")
{
    // Declare parameters
    this->declare_parameter<std::string>("input_name", "compressed_tensor");
    this->declare_parameter<std::string>("output_name", "tensor");
    this->declare_parameter<bool>("use_features", true);
    this->declare_parameter<std::string>("input_qos", "SENSOR_DATA");

    // Get parameters
    this->get_parameter("input_name", input_name_);
    this->get_parameter("output_name", output_name_);
    this->get_parameter("use_features", use_features_);
    this->get_parameter("input_qos", input_qos_);


    auto input_qos = rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(input_qos_));

    if(!use_features_){
    // Create publisher
    tensor_publisher_ = this->create_publisher<neuromesh_interfaces::msg::Tensor>(output_name_, 10);

    // Create subscriber
    tensor_subscription_ = this->create_subscription<neuromesh_interfaces::msg::CompressedTensor>(
        input_name_,
        input_qos,
        std::bind(&DecompressionNode::tensor_callback, this, std::placeholders::_1)
    );

    }else{

    // Create publisher
    feature_publisher_ = this->create_publisher<neuromesh_interfaces::msg::Feature>(output_name_, 10);

    // Create subscriber
    feature_subscription_ = this->create_subscription<neuromesh_interfaces::msg::CompressedFeature>(
        input_name_,
        input_qos,
        std::bind(&DecompressionNode::feature_callback, this, std::placeholders::_1)
    );
    }
}

void DecompressionNode::tensor_callback(const std::shared_ptr<neuromesh_interfaces::msg::CompressedTensor> msg)
{
    // Decompress tensor
    auto tensor = decompress(msg);

    // Publish tensor
    tensor_publisher_->publish(tensor);
}

void DecompressionNode::feature_callback(const std::shared_ptr<neuromesh_interfaces::msg::CompressedFeature> msg)
{
    neuromesh_interfaces::msg::CompressedTensor::SharedPtr tensor = std::make_shared<neuromesh_interfaces::msg::CompressedTensor>(msg->tensor);

    neuromesh_interfaces::msg::Feature feature = neuromesh_interfaces::msg::Feature();

    feature.tensor = decompress(tensor);
    feature.id = msg->id;
    feature.timestamp = msg->timestamp;

    feature_publisher_->publish(feature);
}


neuromesh_interfaces::msg::Tensor DecompressionNode::decompress(const std::shared_ptr<neuromesh_interfaces::msg::CompressedTensor> msg)
{

    char* input_ptr = reinterpret_cast<char*>(msg->data.data());

    std::string output = gzip::decompress(input_ptr, msg->data.size());

    neuromesh_interfaces::msg::Tensor tensor;

    tensor.name = msg->name;
    tensor.shape = msg->shape;
    tensor.data_type = msg->data_type;
    tensor.strides = msg->strides;

    tensor.data = std::vector<uint8_t>(output.begin(), output.end());

    return tensor;

}

//Convert string to ROS2 QoS profile
//from https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox/nvblox_ros_common/src/qos.cpp#L26
rmw_qos_profile_t DecompressionNode::parseQoSString(const std::string& str)
{
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
  RCLCPP_WARN_STREAM(
    rclcpp::get_logger("parseQosString"),
    "Unknown QoS profile: " << profile << ". Returning profile: DEFAULT");
  return rmw_qos_profile_default;
}

int main(int argc, char** argv)
{
	rclcpp::init(argc, argv);
	auto decompression_node = std::make_shared<DecompressionNode>();
	rclcpp::spin(decompression_node);
	rclcpp::shutdown();
	return 0;
}