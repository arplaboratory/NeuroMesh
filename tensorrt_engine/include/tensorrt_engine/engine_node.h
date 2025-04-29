#ifndef TENSORRT_ENGINE_H
#define TENSORRT_ENGINE_H

#define INFERENCE_HELPER_ENABLE_TENSORRT

#include <cstdio>
#include <string>

#include "rclcpp/rclcpp.hpp"

#include "neuromesh_interfaces/msg/tensor.hpp"
#include "neuromesh_interfaces/srv/tensor_request.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "std_msgs/msg/header.hpp"

#include <opencv2/opencv.hpp>

#include "tensorrt_engine/trt_engine.h"

namespace tensorrt_engine_node {
class TensorRTEngineNode : public rclcpp::Node {
public:
  TensorRTEngineNode(rclcpp::NodeOptions options);
  ~TensorRTEngineNode() {};

private:
  rclcpp::Publisher<neuromesh_interfaces::msg::Tensor>::SharedPtr
      tensor_publisher_;
  rclcpp::Subscription<neuromesh_interfaces::msg::Tensor>::SharedPtr
      tensor_subscription_;
  rclcpp::Service<neuromesh_interfaces::srv::TensorRequest>::SharedPtr service_;

  // params
  std::string models_param;
  std::vector<std::string> model_names;

  std::unordered_map<std::string, std::string> model_paths;
  std::unordered_map<std::string, std::vector<uint>> input_dimensions;
  std::unordered_map<std::string, std::vector<uint>> output_dimensions;
  std::unordered_map<std::string, std::string> input_dimensions_strings;
  std::unordered_map<std::string, std::string> output_dimensions_strings;
  std::unordered_map<std::string, std::string> tensor_type_params;
  std::string tensor_qos_param;

  // engine
  std::unordered_map<std::string, std::shared_ptr<TRTEngine>> engines;

  std::unordered_map<std::string, uint32_t> input_lengths;
  std::unordered_map<std::string, uint32_t> output_lengths;
  std::unordered_map<std::string, int> tensor_typelengths;

  std::vector<int> failed_models;

  // functions
  //  callbacks
  void tensor_request_callback(
      const std::shared_ptr<neuromesh_interfaces::srv::TensorRequest::Request>
          request,
      const std::shared_ptr<neuromesh_interfaces::srv::TensorRequest::Response>
          response);

  void
  tensor_callback(const std::shared_ptr<neuromesh_interfaces::msg::Tensor> msg);

  // execution
  neuromesh_interfaces::msg::Tensor
  execute(const std::string model,
          const neuromesh_interfaces::msg::Tensor &tensor_msg);

  // helper functions
  int tensor_string_to_typelength(std::string input);
  std::vector<int> construct_dims(int width, int height, bool rgb, bool nchw);

  // Convert string to ROS2 QoS profile
  rmw_qos_profile_t parseQoSString(const std::string &str);
};
} // namespace tensorrt_engine_node
#endif
