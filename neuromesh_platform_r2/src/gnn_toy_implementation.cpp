#include "neuromesh_platform_r2/dust3r_toy_implementation.h"
#include <chrono>
#include <rclcpp/callback_group.hpp>

namespace neuromeshNode {
ToyImplementation::ToyImplementation(const rclcpp::NodeOptions &options)
    : neuromeshNode(options) {

  std::string tensor_qos_profile_;

  this->declare_parameter<std::string>("tensor_qos_profile", "default");
  this->get_parameter("tensor_qos_profile", tensor_qos_profile_);

  auto tensor_qos =
      rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(tensor_qos_profile_));

  // create two threads
  // main_callback_group_ =
  // this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  // tensor_output_callback_group =
  // this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  auto main_opt = rclcpp::SubscriptionOptions();
  // main_opt.callback_group = main_callback_group_;
  auto tensor_opt = rclcpp::SubscriptionOptions();
  // tensor_opt.callback_group = tensor_output_callback_group;

  // redefine subscriptions with the callback groups

  auto image_qos =
      rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(output_qos_profile_));

  camera_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "camera", image_qos,
      std::bind(&ToyImplementation::camera_callback, this,
                std::placeholders::_1),
      // parseQoSString(image_qos_profile_),
      main_opt);

  // feature_subscriptions_["features"] =
  // this->create_subscription<neuromesh_interfaces::msg::Feature>(
  //     "features",
  //     tensor_qos,
  //     std::bind(&ToyImplementation::feature_callback, this,
  //     std::placeholders::_1), tensor_opt);

  decoder_timer_ = this->create_wall_timer(
      std::chrono::duration<int, std::milli>(decoder_cycle_length_),
      std::bind(&ToyImplementation::process_features, this));

  encoder_timer_ = this->create_wall_timer(
      std::chrono::duration<int, std::milli>(decoder_cycle_length_),
      std::bind(&ToyImplementation::run_encoder_cycle, this));

  // define our subscriptions
  this->tensor_client_ =
      create_client<neuromesh_interfaces::srv::TensorRequest>(
          "tensorrt_request");
  // tensor_subscriber_ =
  // this->create_subscription<neuromesh_interfaces::msg::Tensor>("tensorrt_output",
  //     tensor_qos,
  //     std::bind(&ToyImplementation::tensor_callback, this,
  //     std::placeholders::_1), tensor_opt);

  // tensor_publisher_ =
  // this->create_publisher<neuromesh_interfaces::msg::Tensor>("tensorrt_input",
  // tensor_qos);
}

std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
ToyImplementation::performInference(
    const std::string &model_name,
    const std::vector<neuromesh_interfaces::msg::Tensor> &tensors) {
  if (!tensor_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_ERROR(this->get_logger(), "Engine not reachable via service.");
    // Complex stuff just to return future that resolves to empty tensor
    std::promise<
        std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
        prom;
    std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
        r = prom.get_future();
    std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> t(
        1, std::make_shared<neuromesh_interfaces::msg::Tensor>());
    t[0]->result = 3; // Cannot reach engine error code
    prom.set_value(t);
    return r;
  }

  // Create a request to send to the service server
  auto request =
      std::make_shared<neuromesh_interfaces::srv::TensorRequest::Request>();
  request->model_name = model_name;
  request->tensor1 = tensors;

  // Call the service and wait for the response
  std::shared_future<
      std::shared_ptr<neuromesh_interfaces::srv::TensorRequest::Response>>
      future = tensor_client_->async_send_request(request);

  std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
      return_tensors = std::async(std::launch::async, [future]() {
        std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>
            output_tensors;
        for (const auto &tensor : future.get()->tensor2) {
          output_tensors.emplace_back(
              std::make_shared<neuromesh_interfaces::msg::Tensor>(tensor));
        }
        return output_tensors;
      });

  return return_tensors;
}

bool ToyImplementation::buildDecoderTensor(
    std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr> buffer,
    std::map<std::string, double> buffer_timestamp,
    neuromesh_interfaces::msg::Tensor &decoder_tensor) {
  RCLCPP_DEBUG(this->get_logger(), "Inside buildFullDecoderTensor function");
  RCLCPP_INFO(this->get_logger(),
              "Building full decoder tensor with %zu features.", buffer.size());
  startClock("full_decoder_tensor");

  // Check if we have features for all robots (except self) in both buffers
  if (buffer.size() != all_agents.size() - 1 ||
      buffer_timestamp.size() != all_agents.size() - 1) {
    RCLCPP_WARN(this->get_logger(),
                "Not enough features to build full decoder tensor. Expected "
                "%zu, got %zu",
                all_agents.size(), buffer.size());
    return false;
  }

  // Build the decoder tensor using all features
  neuromesh_interfaces::msg::Tensor own_tensor = buffer[id_]->tensor;

  decoder_tensor.data_type = own_tensor.data_type;
  decoder_tensor.name = "full_decoder_input";
  decoder_tensor.strides = own_tensor.strides;
  decoder_tensor.shape = own_tensor.shape;

  // decoder_tensor.header = std_msgs::msg::Header();
  // decoder_tensor.header.stamp = this->now();
  // decoder_tensor.header.frame_id = "full_tensor_frame";

  // Calculate length of a single tensor in bytes
  int single_tensor_length = 1;
  for (int i : own_tensor.shape.dims) {
    single_tensor_length *= i;
  }

  // Reserve space for all tensors
  decoder_tensor.data.reserve(single_tensor_length * all_agents.size());

  // Combine all tensors
  for (const auto &[agent_id, feature] : buffer) {
    decoder_tensor.data.insert(decoder_tensor.data.end(),
                               feature->tensor.data.begin(),
                               feature->tensor.data.end());
  }

  // Update shape to reflect the combination of all tensors
  // decoder_tensor.shape.dims[0] *= all_agents.size();  // Assuming the first
  // dimension is the batch size

  stopClock("full_decoder_tensor");
  RCLCPP_DEBUG(this->get_logger(), "Built full decoder tensor in %lims",
               times["full_decoder_tensor"].first);

  // decoder_tensor = tensor_ints_to_floats(decoder_tensor);

  RCLCPP_DEBUG(this->get_logger(), "Full decoder tensor size %zu",
               decoder_tensor.data.size());
  RCLCPP_DEBUG(this->get_logger(), "Returning full decoder tensor");

  return true;
}

/*int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::executors::MultiThreadedExecutor executor;
        auto toy_implementation = std::make_shared<ToyImplementation>();

    executor.add_node(toy_implementation);
    executor.spin();

    rclcpp::shutdown();

        return 0;
}*/

} // namespace neuromeshNode

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(neuromeshNode::ToyImplementation)
