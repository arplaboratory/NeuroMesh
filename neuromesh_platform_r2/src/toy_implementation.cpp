#include "neuromesh_platform_r2/toy_implementation.h"
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

std::future<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>
ToyImplementation::performInference(
    const std::string &model_name,
    const neuromesh_interfaces::msg::Tensor &tensor) {
  if (!tensor_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_ERROR(this->get_logger(), "Engine not reachable via service.");

    // Create a future that resolves to empty tensor
    std::promise<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> prom;
    std::future<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> r =
        prom.get_future();

    neuromesh_interfaces::msg::Tensor t = neuromesh_interfaces::msg::Tensor();
    t.result = 3; // cannot reach engine error code

    prom.set_value(std::make_shared<neuromesh_interfaces::msg::Tensor>(t));

    return r;
  }

  // Create a request to send to the service server
  auto request =
      std::make_shared<neuromesh_interfaces::srv::TensorRequest::Request>();
  request->model_name = model_name;
  request->tensor1 = tensor;

  // Call the service and wait for the response
  RCLCPP_DEBUG(this->get_logger(), "Service request sent");
  std::shared_future<
      std::shared_ptr<neuromesh_interfaces::srv::TensorRequest::Response>>
      future = tensor_client_->async_send_request(request);
  RCLCPP_DEBUG(this->get_logger(), "Received response");

  std::future<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>
      return_tensor = std::async(std::launch::async, [future]() {
        // TODO this would be better with no copy
        //  std::shared_ptr<neuromesh_interfaces::srv::TensorRequest::Response>
        //  r_ptr = future.get();
        //  std::shared_ptr<neuromesh_interfaces::msg::Tensor> t_ptr =
        //  std::make_shared<neuromesh_interfaces::msg::Tensor>(r_ptr,
        //  &r_ptr->tensor2);

        neuromesh_interfaces::msg::Tensor tensor = future.get()->tensor2;
        std::shared_ptr<neuromesh_interfaces::msg::Tensor> t_ptr =
            std::make_shared<neuromesh_interfaces::msg::Tensor>(tensor);

        return t_ptr;
      });

  // std::future<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>
  // return_tensor = std::async(std::launch::async, [future]()
  // {
  //     try
  //     {
  //         auto r_ptr = future.get();
  //         if (r_ptr)
  //         {
  //             std::shared_ptr<neuromesh_interfaces::msg::Tensor> t_ptr =
  //             std::make_shared<neuromesh_interfaces::msg::Tensor>(r_ptr->tensor2);
  //             return t_ptr;
  //         }
  //         else
  //         {
  //             std::cerr << "The future.get() got an empty tensor" <<
  //             std::endl; return
  //             std::shared_ptr<neuromesh_interfaces::msg::Tensor>(nullptr);
  //         }
  //     }
  //     catch (const std::exception &e)
  //     {
  //         std::cerr << "Exception while waiting for the future: " << e.what()
  //         << std::endl; return
  //         std::shared_ptr<neuromesh_interfaces::msg::Tensor>(nullptr);
  //     }
  // });

  RCLCPP_DEBUG(this->get_logger(), "Returning tensor");

  return return_tensor;
}

neuromesh_interfaces::msg::Tensor ToyImplementation::buildDecoderTensor(
    std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr>
        buffer) {
  RCLCPP_DEBUG(this->get_logger(), "Inside buildDecoderTensor function");
  RCLCPP_INFO(this->get_logger(), "Building decoder tensor with %d features.",
              buffer.size());
  startClock("decoder_tensor");
  neuromesh_interfaces::msg::Tensor first_tensor =
      buffer.begin()->second->tensor;

  neuromesh_interfaces::msg::Tensor tensor =
      neuromesh_interfaces::msg::Tensor();
  tensor.data_type = first_tensor.data_type;
  tensor.name = "decoder_input";
  tensor.strides = first_tensor.strides;
  tensor.shape = first_tensor.shape;

  tensor.header = std_msgs::msg::Header();
  tensor.header.stamp = this->now();
  tensor.header.frame_id = "tensor_frame"; // Set an appropriate frame_id

  // calculate length of a tensor in bytes
  int length = 1;
  for (int i : first_tensor.shape.dims) {
    length *= i;
  }

  // average the tensors

  for (int i = 0; i < length; i++) {
    for (int j = 0; j < 3; j++) {

      float sum = 0;
      for (auto feature : buffer) {
        sum += feature.second->tensor.data[i];
      }

      tensor.data.push_back(sum / buffer.size());
    }
  }

  stopClock("decoder_tensor");
  RCLCPP_DEBUG(this->get_logger(), "Built decoder tensor in %ims",
               times["decoder_tensor"].first);

  tensor = tensor_ints_to_floats(tensor);

  RCLCPP_DEBUG(this->get_logger(), "Decoder tensor size %d ", tensor);

  RCLCPP_DEBUG(this->get_logger(), "Returning decoder tensor");

  return tensor;
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
