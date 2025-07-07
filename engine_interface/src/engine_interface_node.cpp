#include "engine_interface/engine_interface_node.h"

#include <sstream>

namespace engine_interface {

EngineInterfaceNode::EngineInterfaceNode(
  rclcpp::NodeOptions options,
  const std::string& plugin_package,
  const std::string& plugin_class
)
  : Node("EngineInterfaceNode", 
    options.allow_undeclared_parameters(true)
        .automatically_declare_parameters_from_overrides(true)),
    engine_loader_(plugin_package, "engine_interface::InferenceEngineBase")
{
  // set param vars
  this->declare_parameter<std::string>("tensor_qos_profile", "default");
  this->declare_parameter<std::string>("engine_plugin_package", plugin_package);
  this->declare_parameter<std::string>("engine_type", plugin_class);

  this->get_parameter("model_names", models_param);
  model_names = string_to_vector(models_param);
  std::string plugin_pkg = this->get_parameter("engine_plugin_package").as_string();
  std::string plugin_cls = this->get_parameter("engine_type").as_string();

  // loader
  if (plugin_pkg != plugin_package) {
    engine_loader_ = pluginlib::ClassLoader<InferenceEngineBase>(plugin_pkg, "engine_interface::InferenceEngineBase");
  }

  for (const auto& m : model_names) {
          std::string engine_type = this->declare_parameter(m + ".engine_type", plugin_cls);
          try {
              engines[m] = engine_loader_.createSharedInstance(engine_type);
              engines[m]->loadModel(model_paths[m], input_dimensions[m], tensor_typelengths[m]);
          } catch (const pluginlib::PluginlibException& ex) {
              RCLCPP_ERROR(this->get_logger(), "Failed to load engine plugin: %s", ex.what());
          }
      }

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
  }

}
std::vector<neuromesh_interfaces::msg::Tensor> EngineInterfaceNode::execute(
    const std::string &model,
    const std::vector<neuromesh_interfaces::msg::Tensor> &tensor_msgs) 
{
  RCLCPP_DEBUG(this->get_logger(), "Execute function for model %s",
               model.c_str());
    
  if (tensor_msgs.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Expected at least 1 input tensor, got 0");
    return {[]() {
      neuromesh_interfaces::msg::Tensor t;
      t.result = 2;
      return t;
    }()};
  }
  // Prepare input tensors and sizes
  std::vector<const void *> inputTensors;
  std::vector<int> inputSizes;

  for (const auto &tensor_msg : tensor_msgs) {
    inputTensors.push_back(tensor_msg.data.data());
    inputSizes.push_back(static_cast<int>(tensor_msg.data.size()));
    RCLCPP_DEBUG(this->get_logger(), "tensor_msg.data.size() %ld",
                 tensor_msg.data.size());
  }

  {
    float myfloat;
    std::memcpy(&myfloat, inputTensors.at(0), 4);
    RCLCPP_DEBUG(this->get_logger(), "myfloat is %f", myfloat);
  }

  size_t totalInputSize = std::accumulate(inputSizes.begin(), inputSizes.end(), 0);
  size_t expectedInputSize = 0;
  for (size_t i = 0; i < input_lengths[model].size(); ++i) {
    RCLCPP_DEBUG(this->get_logger(), "input_lengths[model] %d",
                 input_lengths[model][i]);
    RCLCPP_DEBUG(this->get_logger(), "tensor_typelengths %d",
                 tensor_typelengths[model]);
    expectedInputSize += input_lengths[model][i] * tensor_typelengths[model];
  }

  if (totalInputSize != expectedInputSize) {
    RCLCPP_ERROR(
        this->get_logger(),
        "Total input tensor size does not match engine input size %zu and %zu",
        totalInputSize, expectedInputSize);
    return {[]() {
      neuromesh_interfaces::msg::Tensor t;
      t.result = 2;
      return t;
    }()};
  }

  // Prepare output buffers
  std::vector<std::vector<uint8_t>> outputDataVectors;
  std::vector<void *> outputTensors;
  std::vector<int> outputSizes;

  for (size_t i = 0; i < output_dimensions[model].size(); ++i) {
    uint32_t outputSize = 1;
    for (uint32_t dim : output_dimensions[model][i]) {
      outputSize *= dim;
    }
    outputSize *= tensor_typelengths[model];

    outputDataVectors.emplace_back(outputSize);
    outputTensors.push_back(outputDataVectors.back().data());
    outputSizes.push_back(outputSize);
  }

  // Run inference
  engines[model]->runInference(inputTensors, inputSizes, outputTensors,
                               outputSizes);

  RCLCPP_DEBUG(this->get_logger(), "Inference run successfully");

  // Prepare output messages
  std::vector<neuromesh_interfaces::msg::Tensor> output_msgs;
  for (size_t i = 0; i < outputDataVectors.size(); ++i) {
    neuromesh_interfaces::msg::Tensor output_msg;
    output_msg.name = tensor_msgs[0].name + "_output_" + std::to_string(i);
    output_msg.data = std::move(outputDataVectors[i]);
    output_msg.shape.dims = output_dimensions[model][i];
    output_msg.result = 0;
    output_msg.data_type = 9; // float32
    output_msgs.push_back(std::move(output_msg));
    RCLCPP_DEBUG(this->get_logger(), "I: %ld", i);
    RCLCPP_DEBUG(this->get_logger(), "MODEL: %s", model.c_str());
    RCLCPP_DEBUG(this->get_logger(), "OUTPUT TENSOR: %d", output_dimensions[model][i][0]);
  }

  RCLCPP_DEBUG(this->get_logger(), "Returning output messages");

  return output_msgs;
}

void EngineInterfaceNode::tensor_request_callback(
    const std::shared_ptr<neuromesh_interfaces::srv::TensorRequest::Request>
        request,
    const std::shared_ptr<neuromesh_interfaces::srv::TensorRequest::Response>
        response) {
  auto now = this->get_clock()->now();
  double timestamp = now.seconds() + now.nanoseconds() / 1e9;
  RCLCPP_DEBUG(this->get_logger(), "Received service request.");
  RCLCPP_DEBUG(this->get_logger(), "Time of receiving service call %.9f",
              timestamp);
  RCLCPP_DEBUG(this->get_logger(), "Number of input tensors: %zu",
               request->tensor1.size());
  for (size_t i = 0; i < request->tensor1.size(); i++) {
    RCLCPP_DEBUG(this->get_logger(), "Tensor size (%ld): %ld", i, (request->tensor1[i]).data.size());
  }

  std::vector<neuromesh_interfaces::msg::Tensor> input_tensors =
      request->tensor1;
  response->tensor2 = execute(request->model_name, input_tensors);
}

std::vector<std::string> EngineInterfaceNode::string_to_vector(std::string in) {
  std::stringstream stream(in);
  std::string element;

  std::vector<std::string> out;

  while (getline(stream, element, ',')) {
    out.push_back(element);
  }
  return out;
}

int EngineInterfaceNode::tensor_string_to_typelength(std::string input) {

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
rmw_qos_profile_t EngineInterfaceNode::parseQoSString(const std::string &str) {
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
} // namespace engine_interface

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(engine_interface::EngineInterfaceNode)
