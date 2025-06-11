#include "tensorrt_engine/engine_node.h"
#include "tensorrt_engine/trt_engine.h"

#include "cv_bridge/cv_bridge.h"

#include <sstream>

namespace tensorrt_engine_node {
std::vector<uint> string_to_dims(std::string in) {
  std::stringstream stream(in);
  std::string element;

  std::vector<uint> out;

  while (getline(stream, element, ',')) {
    out.push_back(std::stoi(element));
  }
  return out;
}
std::vector<std::string> string_to_vector(std::string in) {
  std::stringstream stream(in);
  std::string element;

  std::vector<std::string> out;

  while (getline(stream, element, ',')) {
    out.push_back(element);
  }
  return out;
}

void TensorRTEngineNode::tensor_request_callback(
    const std::shared_ptr<neuromesh_interfaces::srv::TensorRequest::Request>
        request,
    const std::shared_ptr<neuromesh_interfaces::srv::TensorRequest::Response>
        response) {
  RCLCPP_DEBUG(this->get_logger(), "Recieved service request.");
  RCLCPP_DEBUG(this->get_logger(), "Input tensor size: %zu",
               request->tensor1.data.size());
  RCLCPP_DEBUG(this->get_logger(), "Expected input size for model %s: %d",
               request->model_name.c_str(), input_lengths[request->model_name]);
  response->tensor2 = execute(request->model_name, request->tensor1);

  // tensor_publisher_->publish(response->tensor2);
}

TensorRTEngineNode::TensorRTEngineNode(rclcpp::NodeOptions options)
    : Node("TensorRTEngineNode",
           options.allow_undeclared_parameters(true)
               .automatically_declare_parameters_from_overrides(true)) {
  // params
  // this->declare_parameter<std::string>("model_names", ""); //only declared
  // parameters
  this->declare_parameter<std::string>("tensor_qos_profile", "default");

  // set param vars
  this->get_parameter("model_names", models_param); // values separated by comma
  model_names = string_to_vector(models_param);

  for (std::vector<std::string>::iterator it = model_names.begin();
       it != model_names.end(); it++) {
    std::string m = *it;
    this->get_parameter(m + ".model_path", model_paths[m]);
    this->get_parameter(m + ".input_dimensions", input_dimensions_strings[m]);
    this->get_parameter(m + ".output_dimensions", output_dimensions_strings[m]);
    this->get_parameter(m + ".tensor_type", tensor_type_params[m]);

    RCLCPP_DEBUG(this->get_logger(), "Loading parameters.");
    RCLCPP_DEBUG(this->get_logger(), "Model Name: %s", m.c_str());
    RCLCPP_DEBUG(this->get_logger(), "Input Dimensions: %s",
                 input_dimensions_strings[m].c_str());
    RCLCPP_DEBUG(this->get_logger(), "Output Dimensions: %s",
                 output_dimensions_strings[m].c_str());
    RCLCPP_DEBUG(this->get_logger(), "Tensor Type: %s",
                 tensor_type_params[m].c_str());

    // here we set default values
    if (!tensor_type_params.count(m)) {
      tensor_type_params[m] = "fp32";
    }

    if (!model_paths.count(m) || !input_dimensions_strings.count(m) ||
        !output_dimensions_strings.count(m)) {
      RCLCPP_WARN(this->get_logger(),
                  "Parameters incomplete. Could not set up model %s",
                  m.c_str());
      failed_models.push_back(std::distance(model_names.begin(), it));

      input_dimensions_strings.erase(m);
      output_dimensions_strings.erase(m);

      continue;
    }

    // expand strings to dims
    input_dimensions[m] = string_to_dims(input_dimensions_strings[m]);
    output_dimensions[m] = string_to_dims(output_dimensions_strings[m]);

    // set input and output lengths
    input_lengths[m] = 1;
    for (uint32_t i : input_dimensions[m]) {
      input_lengths[m] *= i;
    }

    output_lengths[m] = 1;
    for (uint32_t i : output_dimensions[m]) {
      output_lengths[m] *= i;
    }

    // set tensor type
    tensor_typelengths[m] = tensor_string_to_typelength(tensor_type_params[m]);
  }

  // iterate backwords over failed models backwords to properly erase from
  // model_names
  for (std::vector<int>::reverse_iterator it = failed_models.rbegin();
       it != failed_models.rend(); it++) {
    model_names.erase(model_names.begin() + *it); // erase by index
  }

  this->get_parameter("tensor_qos_profile", tensor_qos_param);
  auto tensor_qos =
      rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(tensor_qos_param));

  // // Publisher and subscriber setup
  tensor_publisher_ = this->create_publisher<neuromesh_interfaces::msg::Tensor>(
      "tensorrt_output", tensor_qos);

  // tensor_subscription_ =
  // this->create_subscription<neuromesh_interfaces::msg::Tensor>(
  // 	"tensorrt_input", tensor_qos,
  // std::bind(&TensorRTEngineNode::tensor_callback, this,
  // std::placeholders::_1));

  service_ = this->create_service<neuromesh_interfaces::srv::TensorRequest>(
      "tensorrt_request",
      std::bind(&TensorRTEngineNode::tensor_request_callback, this,
                std::placeholders::_1, std::placeholders::_2));

  for (std::vector<std::string>::iterator it = model_names.begin();
       it != model_names.end(); it++) {
    std::string m = *it;
    // initialize engine
    // TODO try/catch engine creation failure
    engines[m].reset(new TRTEngine(model_paths[m], input_dimensions[m],
                                   tensor_typelengths[m]));

    if (engines[m].get() == NULL) {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize engine %s.",
                   m.c_str());
    }
  }
}

neuromesh_interfaces::msg::Tensor TensorRTEngineNode::execute(
    const std::string model,
    const neuromesh_interfaces::msg::Tensor &tensor_msg) {
  RCLCPP_DEBUG(this->get_logger(), "Execute function for model %s",
               model.c_str());
  if (input_lengths[model] * tensor_typelengths[model] !=
      tensor_msg.data.size()) {
    RCLCPP_ERROR(this->get_logger(),
                 "Input tensor size does not match engine input size %i and %i",
                 input_lengths[model], tensor_msg.data.size());

    auto output_msg = neuromesh_interfaces::msg::Tensor();
    output_msg.result = 2;

    return output_msg;
  }

  // TODO generalize for all types
  // process image
  const uint8_t *data_ptr = tensor_msg.data.data();

  std::vector<uint8_t> output_data;
  output_data.reserve(output_lengths[model] * tensor_typelengths[model]);

  RCLCPP_DEBUG(this->get_logger(), "Input Lengths %i", input_lengths[model]);

  RCLCPP_DEBUG(this->get_logger(), "Output Lengths %i", output_lengths[model]);

  RCLCPP_DEBUG(this->get_logger(), "Tensor Type Lengths %i",
               tensor_typelengths[model]);

  engines[model]->runInference(
      data_ptr, input_lengths[model] * tensor_typelengths[model],
      output_data.data(), output_lengths[model] * tensor_typelengths[model]);
  // publish output
  RCLCPP_DEBUG(this->get_logger(), "Inference run successfully");

  // TODO fill more of the output_msg
  auto output_msg = neuromesh_interfaces::msg::Tensor();

  uint8_t *char_ptr = reinterpret_cast<uint8_t *>(output_data.data());
  output_msg.data = std::vector<uint8_t>(
      char_ptr, char_ptr + (output_lengths[model] * tensor_typelengths[model]));

  output_msg.name = tensor_msg.name + "_inference";
  output_msg.shape.dims = output_dimensions[model];

  output_msg.result = 0;

  // float32
  output_msg.data_type = 9;

  // Add header
  // output_msg.header = std_msgs::msg::Header();
  // output_msg.header.stamp = this->now();
  // output_msg.header.frame_id = "tensor_frame"; // Set an appropriate frame_id

  return output_msg;
}

int TensorRTEngineNode::tensor_string_to_typelength(std::string input) {

  if (input == "fp32")
    return 4;
  else if (input == "uint8")
    return 1;
  else if (input == "int8")
    return 1;
  else if (input == "int32")
    return 4;
  else if (input == "int64")
    return 8;
  else
    return -1;
}

// Convert string to ROS2 QoS profile
// from
// https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox/nvblox_ros_common/src/qos.cpp#L26
rmw_qos_profile_t TensorRTEngineNode::parseQoSString(const std::string &str) {
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
} // namespace tensorrt_engine_node

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(tensorrt_engine_node::TensorRTEngineNode)
