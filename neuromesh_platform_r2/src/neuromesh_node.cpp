#include "neuromesh_platform_r2/neuromesh_node.h"
#include "chrono"
#include "rclcpp/rclcpp.hpp"

namespace neuromeshNode {
neuromeshNode ::neuromeshNode(const rclcpp::NodeOptions &options)
    : Node("neuromesh_node", options) {
  // Declare node parameters
  this->declare_parameter<std::string>("encoder_model_name",
                                       "default_encoder_model");
  this->declare_parameter<std::string>("decoder_model_name",
                                       "default_decoder_model");
  this->declare_parameter<std::string>("topic_prefix", "features_");
  this->declare_parameter<std::string>("output_topic", "gnn_output");
  this->declare_parameter<int>("decoder_cycle_length", 1000);
  this->declare_parameter<int>("encoder_cycle_length", 1000);
  this->declare_parameter<int>("encoder_await_length", 10000);
  this->declare_parameter<std::string>("id", "default_id");
  this->declare_parameter<std::string>("image_qos_profile", "default");
  this->declare_parameter<std::string>("features_qos_profile", "default");
  this->declare_parameter<std::string>("output_qos_profile", "default");
  this->declare_parameter<std::string>("agents", "");
  this->declare_parameter<bool>("to_nchw", true);
  this->declare_parameter<bool>("ints_to_floats", true);

  // Get node parameters
  this->get_parameter("encoder_model_name", encoder_model_name_);
  this->get_parameter("decoder_model_name", decoder_model_name_);
  this->get_parameter("topic_prefix", topic_prefix_);
  this->get_parameter("output_topic", output_topic_);
  this->get_parameter("decoder_cycle_length", decoder_cycle_length_);
  this->get_parameter("encoder_cycle_length", encoder_cycle_length_);
  this->get_parameter("encoder_await_length", encoder_await_length_);
  this->get_parameter("id", id_);
  this->get_parameter("image_qos_profile", image_qos_profile_);
  this->get_parameter("features_qos_profile",
                      features_qos_profile_); // for both input and output
  this->get_parameter("output_qos_profile", output_qos_profile_);
  this->get_parameter("agents", agents_);
  this->get_parameter("to_nchw", to_nchw_);
  this->get_parameter("ints_to_floats", ints_to_floats_);

  auto feature_qos =
      rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(features_qos_profile_));
  auto output_qos =
      rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(output_qos_profile_));
  feature_publisher_ =
      this->create_publisher<neuromesh_interfaces::msg::Feature>(
          topic_prefix_ + id_, feature_qos);
  output_publisher_ = this->create_publisher<neuromesh_interfaces::msg::Tensor>(
      output_topic_, output_qos);

  // PLACEHOLDER: update available_agents

  all_agents = splitAgentString(agents_);
  RCLCPP_INFO(this->get_logger(), "Agents:");
  for (const auto &agent : all_agents) {
    RCLCPP_INFO(this->get_logger(), "%s", agent.c_str());
  }
  all_agents.erase(id_); // remove self from list

  available_agents = all_agents;

  for (std::string id : all_agents) {
    RCLCPP_INFO(this->get_logger(),
                "Going through all agents to create subscriptions");
    RCLCPP_INFO(this->get_logger(), "Id: %s", id.c_str());
    this->createSubscription(feature_subscriptions_, id, feature_qos);
  }

  auto image_qos =
      rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(output_qos_profile_));
  camera_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "camera", image_qos,
      std::bind(&neuromeshNode::camera_callback, this, std::placeholders::_1));

  decoder_timer_ = this->create_wall_timer(
      std::chrono::duration<int, std::milli>(decoder_cycle_length_),
      std::bind(&neuromeshNode::process_features, this));
  encoder_timer_ = this->create_wall_timer(
      std::chrono::duration<int, std::milli>(encoder_cycle_length_),
      std::bind(&neuromeshNode::run_encoder_cycle, this));
  fresh_encoder_cycle = true;
}

// callback for feature subscription
void neuromeshNode::feature_callback(
    const neuromesh_interfaces::msg::Feature::SharedPtr msg) {
  std::string id = msg->id;

  RCLCPP_INFO(this->get_logger(),
              "(PRE-AVILABILITY CHECK) Received feature from robot: %s",
              msg->id.c_str());

  // if id is not an available agent
  if (!available_agents.count(id)) {
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Received feature from robot: %s",
              msg->id.c_str());
  feature_buffer_[id] = msg; // overwrite any previous feature from this robot
}

// callback for camera subscription
void neuromeshNode::camera_callback(
    const sensor_msgs::msg::Image::SharedPtr msg) {
  RCLCPP_DEBUG(this->get_logger(), "Received image");
  if (!fresh_encoder_cycle) {
    RCLCPP_DEBUG(this->get_logger(),
                 "Skipping image, already ran encoder this cycle");
    return;
  }

  fresh_encoder_cycle = false;

  RCLCPP_DEBUG(this->get_logger(), "Running encoder on image.");

  neuromesh_interfaces::msg::Tensor image_tensor = imageToTensor(msg);

  // RCLCPP_INFO(this->get_logger(), "Printing image tensor:");
  // const float* float_data = reinterpret_cast<const
  // float*>(image_tensor.data.data()); for (size_t i = 0; i <
  // std::min(static_cast<size_t>(100), image_tensor.data.size() /
  // sizeof(float)); ++i) { 	RCLCPP_INFO(this->get_logger(), "Element %zu: %f",
  // i, float_data[i]);
  // }

  startClock("encoder_inference");
  RCLCPP_DEBUG(this->get_logger(), "Performing Inference on Encoder");
  encoder_result = performInference(encoder_model_name_, image_tensor);
  RCLCPP_DEBUG(this->get_logger(), "Finished performing Inference on Encoder");
  fresh_encoder_cycle = false;
}

// Process received features. Run once per cycle_length_
void neuromeshNode::process_features() {
  // cycle gets split in half into calling the inference and then handling its
  // return
  if (gnn_result_future.valid()) {
    std::shared_ptr<neuromesh_interfaces::msg::Tensor> gnn_result =
        gnn_result_future.get();
    stopClock("decoder_inference");

    if (gnn_result->result != 0) {
      RCLCPP_WARN(this->get_logger(), "Decoder inference failed.");
      feature_buffer_.clear();
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Publishing final decoder result");

    output_publisher_->publish(*gnn_result.get());

    feature_buffer_.clear();

    stopClock("feature_handling");
    RCLCPP_DEBUG(this->get_logger(), "Feature handling took %lims",
                 (times["feature_handling"].first));
    RCLCPP_DEBUG(this->get_logger(), "Decoder inference took %lims",
                 (times["decoder_inference"].first));
  }

  if (!feature_buffer_.empty()) {
    startClock("feature_handling");
    RCLCPP_DEBUG(this->get_logger(), "Running decoder on features.");

    // Process feature buffer here
    RCLCPP_DEBUG(this->get_logger(), "Building Decoder Tensor");
    neuromesh_interfaces::msg::Tensor decoder_tensor =
        buildDecoderTensor(feature_buffer_);

    startClock("decoder_inference");
    RCLCPP_DEBUG(this->get_logger(), "Performing Inference on Decoder");
    gnn_result_future = performInference(decoder_model_name_, decoder_tensor);
    RCLCPP_DEBUG(this->get_logger(),
                 "Finished performing inference on Decoder");
  }
}

// perform inference on tensor using model called model_name
// PLACEHOLDER
std::future<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>
neuromeshNode::performInference(
    const std::string &model_name,
    const neuromesh_interfaces::msg::Tensor &tensor) {
  std::promise<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> prom;
  std::future<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> r =
      prom.get_future();

  std::shared_ptr<neuromesh_interfaces::msg::Tensor> t =
      std::make_shared<neuromesh_interfaces::msg::Tensor>(
          neuromesh_interfaces::msg::Tensor());
  t->result = 1; // placeholder function not defined error code
  prom.set_value(t);

  return r;
}

// Convert tensor of features to a feature message
neuromesh_interfaces::msg::Feature neuromeshNode::buildFeatureMessage(
    const neuromesh_interfaces::msg::Tensor &tensor) {
  neuromesh_interfaces::msg::Feature feature_msg =
      neuromesh_interfaces::msg::Feature();

  feature_msg.tensor = tensor;
  feature_msg.id = id_;
  feature_msg.timestamp = this->get_clock()->now();

  return feature_msg;
}

// Aggregates features from available agents (and itself) into a single tensor
// PLACEHOLDER
neuromesh_interfaces::msg::Tensor neuromeshNode::buildDecoderTensor(
    std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr>
        buffer) {
  return neuromesh_interfaces::msg::Tensor();
}

// Converts image message to tensor
neuromesh_interfaces::msg::Tensor
neuromeshNode::imageToTensor(const sensor_msgs::msg::Image::SharedPtr msg) {
  // neuromesh_interfaces::msg::Tensor tensor =
  // neuromesh_interfaces::msg::Tensor();

  // //find dimensions
  // uint channels = sensor_msgs::image_encodings::numChannels(msg->encoding);
  // const uint& height = msg->height;
  // const uint& width = msg->width;

  // uint bitdepth = sensor_msgs::image_encodings::bitDepth(msg->encoding);
  // uint bytedepth = bitdepth / 8;

  // //copy data
  // tensor.data = std::vector<uint8_t>(msg->data.begin(), msg->data.end());

  // //determine datatype

  // enum dtype{ unsigned_int, signed_int, float_ };
  // dtype datatype;

  // //all encodings:
  // https://github.com/ros2/common_interfaces/blob/foxy/sensor_msgs/include/sensor_msgs/image_encodings.hpp
  // //case: is a regular or bayer encoding
  // if( sensor_msgs::image_encodings::isColor(msg->encoding) ||
  // sensor_msgs::image_encodings::isMono(msg->encoding) ||
  // sensor_msgs::image_encodings::isBayer(msg->encoding) ){ 	datatype =
  // unsigned_int;
  // }
  // //case: float encoding from opencv
  // else if(msg->encoding.find("FC") != std::string::npos){
  // 	datatype = float_;
  // }
  // //case: signed int encoding from opencv
  // else if (msg->encoding.find("SC")){
  // 	datatype = signed_int;
  // }
  // //case: unsigned int encoding from opencv
  // else if (msg->encoding.find("UC")){
  // 	datatype = unsigned_int;
  // }else{
  // 	RCLCPP_ERROR(this->get_logger(), "Could not determine datatype of
  // image.");
  // }

  // //datatypes are described in Tensor.msg
  // //Set datatype
  // switch(datatype)
  // {
  // 	case unsigned_int:
  // 		if (bitdepth == 8)
  // 			tensor.data_type = 2; // "uint8"
  // 		if (bitdepth == 16)
  // 			tensor.data_type = 4; // "uint16"
  // 		if (bitdepth == 32)
  // 			tensor.data_type = 6; // "uint32"
  // 		if (bitdepth == 64)
  // 			tensor.data_type = 8; // "uint64"
  // 		break;

  // 	case signed_int:
  // 		if (bitdepth == 8)
  // 			tensor.data_type = 1; // "int8"
  // 		if (bitdepth == 16)
  // 			tensor.data_type = 3; // "int16"
  // 		if (bitdepth == 32)
  // 			tensor.data_type = 5; // "int32"
  // 		if (bitdepth == 64)
  // 			tensor.data_type = 7; // "int64"
  // 		break;

  // 	case float_:
  // 		if (bitdepth == 32)
  // 			tensor.data_type = 9; // "float32"
  // 		if (bitdepth == 64)
  // 			tensor.data_type = 10; // "float64"
  // 		break;
  // }

  // //Set shape
  // tensor.shape.dims = std::vector<uint32_t>{height, width, channels}; //TODO:
  // make sure the order is correct tensor.shape.rank; //TODO: I don't know what
  // rank means tensor.strides = std::vector<uint64_t>{channels * width *
  // bytedepth, channels * bytedepth, bytedepth}; //TODO: make sure the order is
  // correct

  // //fill data
  // tensor.data = std::vector<uint8_t>(msg->data.begin(), msg->data.end());

  // //set name
  // tensor.name = msg->header.frame_id + std::to_string(msg->header.stamp.sec)
  // + "." + std::to_string(msg->header.stamp.nanosec);

  // if(to_nchw_){
  // 	RCLCPP_DEBUG(this->get_logger(), "Converting to NCHW");
  // 	tensor = convert_to_nchw(tensor);
  // }

  // if(ints_to_floats_){
  // 	RCLCPP_DEBUG(this->get_logger(), "Converting ints to floats");
  // 	tensor = tensor_ints_to_floats(tensor);
  // }

  // return tensor;

  neuromesh_interfaces::msg::Tensor tensor =
      neuromesh_interfaces::msg::Tensor();

  uint channels = sensor_msgs::image_encodings::numChannels(msg->encoding);
  uint height = msg->height;
  uint width = msg->width;

  // RCLCPP_INFO(this->get_logger(), "Input image dimensions: %ux%u with %u
  // channels", width, height, channels);

  // Crop
  uint cx = width / 2;
  uint cy = height / 2;
  uint halfw = ((2 * cx) / 16) * 8;
  uint halfh = ((2 * cy) / 16) * 8;

  // Adjust crop for 3:4 aspect ratio if needed
  if (width == height) {
    halfh = static_cast<uint>(3 * halfw / 4);
  }

  uint crop_width = 2 * halfw;
  uint crop_height = 2 * halfh;

  // RCLCPP_INFO(this->get_logger(), "Cropped dimensions: %ux%u", crop_width,
  // crop_height);

  std::vector<float> cropped_data(crop_width * crop_height * channels);

  for (uint y = 0; y < crop_height; ++y) {
    for (uint x = 0; x < crop_width; ++x) {
      uint src_x = cx - halfw + x;
      uint src_y = cy - halfh + y;
      for (uint c = 0; c < channels; ++c) {
        uint src_index = (src_y * width + src_x) * channels + c;
        if (src_index >= msg->data.size()) {
          RCLCPP_ERROR(this->get_logger(), "Index out of bounds: %u >= %zu",
                       src_index, msg->data.size());
          continue;
        }
        float pixel_value = static_cast<float>(msg->data[src_index]);
        // Normalize to [-1, 1] with safeguard against division by zero
        pixel_value = (pixel_value / 255.0f - 0.5f) /
                      std::max(0.5f, std::numeric_limits<float>::epsilon());
        cropped_data[(y * crop_width + x) * channels + c] = pixel_value;
      }
    }
  }

  // Set tensor data
  tensor.data.resize(crop_width * crop_height * channels * sizeof(float));
  std::memcpy(tensor.data.data(), cropped_data.data(), tensor.data.size());

  // Set datatype to float32
  tensor.data_type = 9; // float32

  // Set shape (NCHW format)
  tensor.shape.dims =
      std::vector<uint32_t>{1, channels, crop_height, crop_width};
  tensor.shape.rank = 4;

  // Set strides
  tensor.strides =
      std::vector<uint64_t>{channels * crop_height * crop_width * sizeof(float),
                            crop_height * crop_width * sizeof(float),
                            crop_width * sizeof(float), sizeof(float)};

  // Set name
  tensor.name = msg->header.frame_id + std::to_string(msg->header.stamp.sec) +
                "." + std::to_string(msg->header.stamp.nanosec);

  // Add header
  tensor.header = std_msgs::msg::Header();
  tensor.header.stamp = this->now();
  tensor.header.frame_id = "tensor_frame"; // Set an appropriate frame_id
  // Debug print
  // const float* float_data = reinterpret_cast<const
  // float*>(tensor.data.data()); RCLCPP_INFO(this->get_logger(), "Printing
  // inside imageToTensor");
  // for (size_t i = 0; i < std::min(static_cast<size_t>(100),
  // tensor.data.size() / sizeof(float)); ++i) {
  //     RCLCPP_INFO(this->get_logger(), "Element %zu: %f", i, float_data[i]);
  // }
  return tensor;
}

// Adds a subscription to the feature_subscriptions_ map
void neuromeshNode::createSubscription(
    std::map<std::string, rclcpp::Subscription<
                              neuromesh_interfaces::msg::Feature>::SharedPtr>
        &subscription_map,
    std::string id, rclcpp::QoS qos) {
  std::string topic = topic_prefix_ + id;

  RCLCPP_INFO(this->get_logger(), "creating subscription for topic %s",
              topic.c_str());

  rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr
      feature_subscription_ =
          this->create_subscription<neuromesh_interfaces::msg::Feature>(
              topic, qos,
              std::bind(&neuromeshNode::feature_callback, this,
                        std::placeholders::_1));

  subscription_map.insert({id, feature_subscription_});
}

// Remove subscription form the feature_subscriptions_ map
void neuromeshNode::removeSubscription(
    std::map<std::string, rclcpp::Subscription<
                              neuromesh_interfaces::msg::Feature>::SharedPtr>
        subscription_map,
    std::string id) {
  subscription_map.erase(id);
}

// Convert string to ROS2 QoS profile
// from
// https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox/nvblox_ros_common/src/qos.cpp#L26
rmw_qos_profile_t neuromeshNode::parseQoSString(const std::string &str) {
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

// Split agent string parameter into vector of agent ids
std::set<std::string> neuromeshNode::splitAgentString(std::string str) {
  std::set<std::string> agents;
  const std::string delimiter = ",";

  size_t pos = 0;
  std::string token;
  while ((pos = str.find(delimiter)) != std::string::npos) {
    token = str.substr(0, pos);
    agents.insert(token);
    str.erase(0, pos + delimiter.length());
  }
  agents.insert(str);
  return agents;
}

void neuromeshNode::run_encoder_cycle() {

  if (fresh_encoder_cycle == true) {
    return;
  }

  if (!encoder_result.valid()) {
    if (encoder_await_length_ <= checkClock("encoder_inference")) {
      fresh_encoder_cycle = true;
    }
    return;
  }
  stopClock("encoder_inference");
  std::shared_ptr<neuromesh_interfaces::msg::Tensor> feature_tensor =
      encoder_result.get();
  RCLCPP_INFO(this->get_logger(), "Encoder Feature Tensor size: %ld",
              feature_tensor->data.size());
  RCLCPP_DEBUG(this->get_logger(), "Encoder took %lims",
               times["encoder_inference"].first);

  if (feature_tensor->result != 0) {
    RCLCPP_WARN(this->get_logger(), "Encoder inference failed.");
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Publishing features.");

  neuromesh_interfaces::msg::Feature feature_msg =
      buildFeatureMessage(*feature_tensor.get());

  // neuromesh_interfaces::msg::Feature feature_msg =
  // buildFeatureMessage(*combined_tensor.get());

  feature_publisher_->publish(feature_msg);
  feature_buffer_[this->id_] =
      std::make_shared<neuromesh_interfaces::msg::Feature>(feature_msg);

  // Reset for the next cycle
  // encoder_cycle_count_ = 0;
  fresh_encoder_cycle = true;
}

void neuromeshNode::startClock(std::string phase) {
  int64_t now_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();

  times[phase] = {now_time, false};
}
void neuromeshNode::stopClock(std::string phase) {
  if (times[phase].second) {
    return; // clock already stopped
  }
  int64_t start_time = times[phase].first;
  int64_t now_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();

  times[phase] = {now_time - start_time, true};
}

int64_t neuromeshNode::checkClock(std::string phase) {
  if (times[phase].second) {
    return times[phase].first; // clock already stopped
  }

  // stopclock calculations without saving
  int64_t start_time = times[phase].first;
  int64_t now_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();

  return now_time - start_time;
}

// TODO generalize to a transpose function
// tested for use with integers
neuromesh_interfaces::msg::Tensor
neuromeshNode::convert_to_nchw(const neuromesh_interfaces::msg::Tensor &input) {
  std::vector<uint8_t> new_data;
  const std::vector<uint8_t> &old_data = input.data;

  std::vector<uint64_t> strides = input.strides;
  std::vector<uint32_t> dims = input.shape.dims;
  unsigned long bytedepth = input.strides[0] / input.shape.dims[0];

  for (int i = 0; i < dims[2]; i++) {
    for (int j = 0; j < dims[0]; j++) {
      for (int k = 0; k < dims[1]; k++) {
        new_data.push_back(
            old_data[(j * strides[0]) + (k * strides[1]) + (i * strides[2])]);
      }
    }
  }

  neuromesh_interfaces::msg::Tensor new_tensor = std::move(input);
  new_tensor.data = new_data;
  new_tensor.shape.dims = {input.shape.dims[2], input.shape.dims[0],
                           input.shape.dims[1]};
  new_tensor.strides = {dims[1] * dims[2] * bytedepth, dims[2] * bytedepth,
                        bytedepth};

  return new_tensor;
}

float int_to_scaled_float(int i) { return static_cast<float>(i) / 255.0; }

neuromesh_interfaces::msg::Tensor
neuromeshNode::tensor_ints_to_floats(neuromesh_interfaces::msg::Tensor &input) {

  std::vector<float> float_data;

  std::transform(input.data.begin(), input.data.end(),
                 std::back_inserter(float_data), int_to_scaled_float);

  neuromesh_interfaces::msg::Tensor new_tensor = std::move(input);
  new_tensor.data_type = 9; // float32

  for (int i = 0; i < new_tensor.strides.size(); i++) {
    new_tensor.strides[i] *= sizeof(float);
  }

  uint8_t *char_ptr = reinterpret_cast<uint8_t *>(float_data.data());
  new_tensor.data = std::vector<uint8_t>(
      char_ptr,
      char_ptr + (float_data.size() * sizeof(float))); // std::move(float_data)

  return new_tensor;
}
} // namespace neuromeshNode
