#include "chrono"
#include "cv_bridge/cv_bridge.h"
#include "neuromesh_platform_r2/dust3r_neuromesh_node.h"
#include "rclcpp/rclcpp.hpp"
#include <opencv4/opencv2/core.hpp>    // Core functionality
#include <opencv4/opencv2/highgui.hpp> // For imshow, waitKey
#include <opencv4/opencv2/imgproc.hpp> // Image processing

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

  // Declare tensor dimension parameters
  this->declare_parameter<int>("tensor_batch_size", 1);
  this->declare_parameter<int>("tensor_channels", 3);
  this->declare_parameter<int>("tensor_height", 192);
  this->declare_parameter<int>("tensor_width", 320);

  // Get tensor dimension parameters
  int batch_size, channels, height, width;
  this->get_parameter("tensor_batch_size", batch_size);
  this->get_parameter("tensor_channels", channels);
  this->get_parameter("tensor_height", height);
  this->get_parameter("tensor_width", width);

  // Convert to uint32_t and store in member variables
  tensor_batch_size_ = static_cast<uint32_t>(batch_size);
  tensor_channels_ = static_cast<uint32_t>(channels);
  tensor_height_ = static_cast<uint32_t>(height);
  tensor_width_ = static_cast<uint32_t>(width);

  // Define and get pos1 and pos2 yaml file paths
  this->declare_parameter<std::string>("pos1_yaml_path", "");
  this->declare_parameter<std::string>("pos2_yaml_path", "");

  this->get_parameter("pos1_yaml_path", pos1_yaml_path);
  this->get_parameter("pos2_yaml_path", pos2_yaml_path);

  this->declare_parameter("dust3r_decoder_output_dimensions",
                          "1,384,512,3;1,384,512;1,384,512,3;1,384,512");
  this->get_parameter("dust3r_decoder_output_dimensions",
                      decoder_output_dimensions_str);

  auto feature_qos =
      rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(features_qos_profile_));
  auto output_qos =
      rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(output_qos_profile_));
  feature_publisher_ =
      this->create_publisher<neuromesh_interfaces::msg::Feature>(
          topic_prefix_ + id_, feature_qos);
  output_publisher_ = this->create_publisher<neuromesh_interfaces::msg::Tensor>(
      output_topic_, output_qos);
  pts3d_res1_publisher_ =
      this->create_publisher<neuromesh_interfaces::msg::Tensor>(
          "res1_pts3d_topic", output_qos);
  conf_res1_publisher_ =
      this->create_publisher<neuromesh_interfaces::msg::Tensor>(
          "res1_conf_topic", output_qos);
  pts3d_res2_publisher_ =
      this->create_publisher<neuromesh_interfaces::msg::Tensor>(
          "res2_pts3d_topic", output_qos);
  conf_res2_publisher_ =
      this->create_publisher<neuromesh_interfaces::msg::Tensor>(
          "res2_conf_topic", output_qos);
  pts3d_res1_cloud_publisher_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>("res1_pts3d_cloud",
                                                            output_qos);
  pts3d_res2_cloud_publisher_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>("res2_pts3d_cloud",
                                                            output_qos);

  // Add TransformBroadcaster
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  // Add a timer to broadcast the transform periodically
  transform_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&neuromeshNode::broadcast_transform, this));

  // PLACEHOLDER: update available_agents

  all_agents = splitAgentString(agents_);
  // RCLCPP_INFO(this->get_logger(), "Agents:");
  for (const auto &agent : all_agents) {
    // RCLCPP_INFO(this->get_logger(), "%s", agent.c_str());
  }
  all_agents.erase(id_); // remove self from list

  available_agents = all_agents;

  // Print all agent names in agent list
  for (std::string id : all_agents) {
    // RCLCPP_INFO(this->get_logger(), "Going through all agents to create
    // subscriptions"); RCLCPP_INFO(this->get_logger(), "Id: %s", id.c_str());
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

  // Printing to debug decoder output dimensions
  decoder_output_dims = string_to_dims(decoder_output_dimensions_str);
  // RCLCPP_INFO(this->get_logger(), "Printing decoder_output_dims:");
  for (size_t i = 0; i < decoder_output_dims.size(); ++i) {
    std::string dim_str = "";
    for (size_t j = 0; j < decoder_output_dims[i].size(); ++j) {
      dim_str += std::to_string(decoder_output_dims[i][j]);
      if (j < decoder_output_dims[i].size() - 1) {
        dim_str += ", ";
      }
    }
    // RCLCPP_INFO(this->get_logger(), "  Dimension %zu: [%s]", i,
    // dim_str.c_str());
  }
}

// Transform function for displaying point cloud in map frame
void neuromeshNode::broadcast_transform() {
  geometry_msgs::msg::TransformStamped t;

  t.header.stamp = this->now();
  t.header.frame_id =
      "cam1_color_optical_frame"; // Or whatever your parent frame is
  t.child_frame_id = id_ + "/map";

  // Set the translation
  t.transform.translation.x = 0.0; // Replace with actual values
  t.transform.translation.y = 0.0;
  t.transform.translation.z = 0.0;

  // Set the rotation (as quaternion)
  t.transform.rotation.x = 0.0; // Replace with actual values
  t.transform.rotation.y = 0.0;
  t.transform.rotation.z = 0.0;
  t.transform.rotation.w = 1.0;

  // Send the transform
  tf_broadcaster_->sendTransform(t);
}

// callback for feature subscription
void neuromeshNode::feature_callback(
    const neuromesh_interfaces::msg::Feature::SharedPtr msg) {
  std::string id = msg->id;

  // RCLCPP_INFO(this->get_logger(), "(PRE-AVILABILITY CHECK) Received feature
  // from robot: %s", msg->id.c_str());

  // if id is not an available agent
  if (!available_agents.count(id)) {
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Received feature from robot: %s",
              msg->id.c_str());

  feature_buffer_[id] = msg; // overwrite any previous feature from this robot
  feature_buffer_timestamp_[id] =
      rclcpp::Time(msg->timestamp).seconds(); // Store time stamp of feature
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
  // RCLCPP_INFO(this->get_logger(), "image tensor size %ld",
  // image_tensor.data.size());

  std::vector<neuromesh_interfaces::msg::Tensor> image_tensors = {image_tensor};

  startClock("encoder_inference");
  RCLCPP_DEBUG(this->get_logger(), "Performing Inference on Encoder");
  encoder_result = performInference(encoder_model_name_, image_tensors);
  RCLCPP_DEBUG(this->get_logger(), "Finished performing Inference on Encoder");
  fresh_encoder_cycle = false;
}

// Process received features. Run once per cycle_length_
void neuromeshNode::process_features() {
  // cycle gets split in half into calling the inference and then handling its
  // return
  if (gnn_result_future.valid()) {
    std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>
        gnn_results = gnn_result_future.get();
    stopClock("decoder_inference");

    for (size_t i = 0; i < gnn_results.size(); ++i) {
      auto &tensor = gnn_results[i];

      std::cout << "Tensor " << i << " details:" << std::endl;

      // Dimensions with more readable output
      std::cout << "Dimensions (" << tensor->shape.dims.size() << "): ";
      for (auto dim : tensor->shape.dims) {
        std::cout << dim << " ";
      }
      std::cout << std::endl;

      std::cout << "Total data size: " << tensor->data.size() << std::endl;

      // Improved data point printing with type casting and error checking
      if (!tensor->data.empty()) {
        const float *float_data =
            reinterpret_cast<const float *>(tensor->data.data());
        size_t float_count = tensor->data.size() / sizeof(float);

        std::cout << "Float32 interpretation (first 10 or fewer):" << std::endl;
        for (size_t j = 0; j < std::min(float_count, size_t(10)); ++j) {
          std::cout << float_data[j] << " ";
        }
        std::cout << std::endl;
      } else {
        std::cout << "No data points in this tensor." << std::endl;
      }

      std::cout << "------------------------" << std::endl;
    }

    if (gnn_results.size() != 4) {
      RCLCPP_WARN(this->get_logger(),
                  "Unexpected number of tensors in the inference result.");
      feature_buffer_.clear();
      return;
    }

    // Extract each tensor from the result vector
    std::shared_ptr<neuromesh_interfaces::msg::Tensor> res1_pts3d =
        gnn_results[0];
    std::shared_ptr<neuromesh_interfaces::msg::Tensor> res1_conf =
        gnn_results[1];
    std::shared_ptr<neuromesh_interfaces::msg::Tensor> res2_pts3d =
        gnn_results[2];
    std::shared_ptr<neuromesh_interfaces::msg::Tensor> res2_conf =
        gnn_results[3];

    if (res1_pts3d->result != 0 || res1_conf->result != 0 ||
        res2_pts3d->result != 0 || res2_conf->result != 0) {
      RCLCPP_WARN(this->get_logger(), "Decoder inference failed.");
      feature_buffer_.clear();
      return;
    }
    // reshape the res1_pts3d to 3xHxW format
    auto res1_pts3d_shape = res1_pts3d->shape.dims;
    auto res2_pts3d_shape = res2_pts3d->shape.dims;
    std::cout << decoder_output_dims.size() << std::endl;
    std::cout << decoder_output_dims[0].size() << std::endl;

    if ((decoder_output_dims.size() == 4) &&
        (decoder_output_dims[0].size() == 4)) {
      int num_batches = decoder_output_dims[0][0]; // Number of batches
      int height = decoder_output_dims[0][1];
      int width = decoder_output_dims[0][2];
      int numpoints_per_batch = height * width;
      int total_points = num_batches * numpoints_per_batch;

      // For res1 point cloud
      auto combined_res1_pts3d_msg =
          std::make_unique<sensor_msgs::msg::PointCloud2>();
      combined_res1_pts3d_msg->header.stamp = this->now();
      combined_res1_pts3d_msg->header.frame_id = id_ + "/map";
      combined_res1_pts3d_msg->height = 1; // Unorganized point cloud
      combined_res1_pts3d_msg->width = total_points;
      combined_res1_pts3d_msg->is_bigendian = false;
      combined_res1_pts3d_msg->is_dense = true;
      combined_res1_pts3d_msg->point_step =
          3 * sizeof(float) + 3 * sizeof(uint8_t);
      combined_res1_pts3d_msg->row_step =
          combined_res1_pts3d_msg->point_step * combined_res1_pts3d_msg->width;

      // Set up point cloud fields
      sensor_msgs::PointCloud2Modifier modifier1(*combined_res1_pts3d_msg);
      modifier1.setPointCloud2Fields(
          6, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1,
          sensor_msgs::msg::PointField::FLOAT32, "z", 1,
          sensor_msgs::msg::PointField::FLOAT32, "r", 1,
          sensor_msgs::msg::PointField::UINT8, "g", 1,
          sensor_msgs::msg::PointField::UINT8, "b", 1,
          sensor_msgs::msg::PointField::UINT8);
      modifier1.resize(total_points);

      // Iterators for the combined point cloud
      sensor_msgs::PointCloud2Iterator<float> res1_iter_x(
          *combined_res1_pts3d_msg, "x");
      sensor_msgs::PointCloud2Iterator<float> res1_iter_y(
          *combined_res1_pts3d_msg, "y");
      sensor_msgs::PointCloud2Iterator<float> res1_iter_z(
          *combined_res1_pts3d_msg, "z");
      sensor_msgs::PointCloud2Iterator<uint8_t> res1_iter_r(
          *combined_res1_pts3d_msg, "r");
      sensor_msgs::PointCloud2Iterator<uint8_t> res1_iter_g(
          *combined_res1_pts3d_msg, "g");
      sensor_msgs::PointCloud2Iterator<uint8_t> res1_iter_b(
          *combined_res1_pts3d_msg, "b");

      const float *res1_pts3d_data =
          reinterpret_cast<const float *>(res1_pts3d->data.data());

      // Process points from all batches
      for (int batch = 0; batch < num_batches; ++batch) {
        for (int i = 0; i < numpoints_per_batch; ++i, ++res1_iter_x,
                 ++res1_iter_y, ++res1_iter_z, ++res1_iter_r, ++res1_iter_g,
                 ++res1_iter_b) {
          // Calculate the index for the current batch and point
          int batch_offset = (batch * numpoints_per_batch + i) * 3;

          *res1_iter_x = res1_pts3d_data[batch_offset];
          *res1_iter_y = res1_pts3d_data[batch_offset + 2];
          *res1_iter_z = -res1_pts3d_data[batch_offset + 1];

          // Optional: Add color if needed
          *res1_iter_r = 255; // Red channel
          *res1_iter_g = 0;   // Green channel
          *res1_iter_b = 0;   // Blue channel
        }
      }

      // Publish the combined point cloud
      pts3d_res1_cloud_publisher_->publish(std::move(combined_res1_pts3d_msg));

      // For res2 point cloud
      auto combined_res2_pts3d_msg =
          std::make_unique<sensor_msgs::msg::PointCloud2>();
      combined_res2_pts3d_msg->header.stamp = this->now();
      combined_res2_pts3d_msg->header.frame_id = id_ + "/map";
      combined_res2_pts3d_msg->height = 1; // Unorganized point cloud
      combined_res2_pts3d_msg->width = total_points;
      combined_res2_pts3d_msg->is_bigendian = false;
      combined_res2_pts3d_msg->is_dense = true;
      combined_res2_pts3d_msg->point_step =
          3 * sizeof(float) + 3 * sizeof(uint8_t);
      combined_res2_pts3d_msg->row_step =
          combined_res2_pts3d_msg->point_step * combined_res2_pts3d_msg->width;

      // Set up point cloud fields
      sensor_msgs::PointCloud2Modifier modifier2(*combined_res2_pts3d_msg);
      modifier2.setPointCloud2Fields(
          6, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1,
          sensor_msgs::msg::PointField::FLOAT32, "z", 1,
          sensor_msgs::msg::PointField::FLOAT32, "r", 1,
          sensor_msgs::msg::PointField::UINT8, "g", 1,
          sensor_msgs::msg::PointField::UINT8, "b", 1,
          sensor_msgs::msg::PointField::UINT8);
      modifier2.resize(total_points);

      // Iterators for the combined point cloud
      sensor_msgs::PointCloud2Iterator<float> res2_iter_x(
          *combined_res2_pts3d_msg, "x");
      sensor_msgs::PointCloud2Iterator<float> res2_iter_y(
          *combined_res2_pts3d_msg, "y");
      sensor_msgs::PointCloud2Iterator<float> res2_iter_z(
          *combined_res2_pts3d_msg, "z");
      sensor_msgs::PointCloud2Iterator<uint8_t> res2_iter_r(
          *combined_res2_pts3d_msg, "r");
      sensor_msgs::PointCloud2Iterator<uint8_t> res2_iter_g(
          *combined_res2_pts3d_msg, "g");
      sensor_msgs::PointCloud2Iterator<uint8_t> res2_iter_b(
          *combined_res2_pts3d_msg, "b");

      const float *res2_pts3d_data =
          reinterpret_cast<const float *>(res2_pts3d->data.data());

      // Process points from all batches
      for (int batch = 0; batch < num_batches; ++batch) {
        for (int i = 0; i < numpoints_per_batch; ++i, ++res2_iter_x,
                 ++res2_iter_y, ++res2_iter_z, ++res2_iter_r, ++res2_iter_g,
                 ++res2_iter_b) {
          // Calculate the index for the current batch and point
          int batch_offset = (batch * numpoints_per_batch + i) * 3;

          *res2_iter_x = res2_pts3d_data[batch_offset];
          *res2_iter_y = res2_pts3d_data[batch_offset + 2];
          *res2_iter_z = -res2_pts3d_data[batch_offset + 1];

          // Optional: Add color if needed
          *res2_iter_r = 255; // Red channel
          *res2_iter_g = 0;   // Green channel
          *res2_iter_b = 0;   // Blue channel
        }
      }

      // Publish the combined point cloud
      pts3d_res2_cloud_publisher_->publish(std::move(combined_res2_pts3d_msg));
    }

    std::cout << "After publishing the point cloud" << std::endl;

    // Publish each tensor individually
    // RCLCPP_INFO(this->get_logger(), "Publishing inference results.");
    pts3d_res1_publisher_->publish(*res1_pts3d); // Publish res1["pts3d"]
    conf_res1_publisher_->publish(*res1_conf);   // Publish res1["conf"]
    pts3d_res2_publisher_->publish(*res2_pts3d); // Publish res2["pts3d"]
    conf_res2_publisher_->publish(*res2_conf);   // Publish res2["conf"]

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
    neuromesh_interfaces::msg::Tensor own_tensor, neighbour_tensor, pos1, pos2;
    YAML::Node pos1_yaml = YAML::LoadFile(pos1_yaml_path);
    YAML::Node pos2_yaml = YAML::LoadFile(pos2_yaml_path);
    if (buildDecoderTensor(feature_buffer_, feature_buffer_timestamp_,
                           own_tensor, neighbour_tensor, pos1, pos2, pos1_yaml,
                           pos2_yaml)) {
      startClock("decoder_inference");
      RCLCPP_DEBUG(this->get_logger(), "Performing Inference on Decoder");
      std::vector<neuromesh_interfaces::msg::Tensor> decoder_tensors = {
          own_tensor, neighbour_tensor, pos1, pos2};
      gnn_result_future =
          performInference(decoder_model_name_, decoder_tensors);
      RCLCPP_DEBUG(this->get_logger(),
                   "Finished performing inference on Decoder");
    } else {
      RCLCPP_DEBUG(this->get_logger(), "Could not build Decoder Tensor");
    }
  }
}

// perform inference on tensor using model called model_name
// PLACEHOLDER
std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
neuromeshNode::performInference(
    const std::string &model_name,
    const std::vector<neuromesh_interfaces::msg::Tensor> &tensors) {
  std::promise<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
      prom;
  std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
      r = prom.get_future();
  std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> t(
      1, std::make_shared<neuromesh_interfaces::msg::Tensor>());
  t[0]->result = 1; // Cannot reach engine error code
  prom.set_value(std::move(t));
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
bool neuromeshNode::buildDecoderTensor(
    std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr> buffer,
    std::map<std::string, double> buffer_timestamp,
    neuromesh_interfaces::msg::Tensor &own_tensor,
    neuromesh_interfaces::msg::Tensor &neighbour_tensor,
    neuromesh_interfaces::msg::Tensor &pos1,
    neuromesh_interfaces::msg::Tensor &pos2, const YAML::Node &pos1_yaml,
    const YAML::Node &pos2_yaml) {
  return true;
}

neuromesh_interfaces::msg::Tensor
neuromeshNode::imageToTensor(const sensor_msgs::msg::Image::SharedPtr msg) {
  neuromesh_interfaces::msg::Tensor tensor =
      neuromesh_interfaces::msg::Tensor();

  // Convert ROS Image to OpenCV format
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception &e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return tensor;
  }

  cv::Mat image = cv_ptr->image;
  uint channels = image.channels();

  // Always resize to 512x384 with aspect ratio preservation
  cv::Mat resized_image;
  cv::Size target_size(512, 384);

  // Compute the scaling factor while maintaining aspect ratio
  double scale_width = static_cast<double>(target_size.width) / image.cols;
  double scale_height = static_cast<double>(target_size.height) / image.rows;
  double scale = std::min(scale_width, scale_height);

  cv::Size scaled_size(static_cast<int>(image.cols * scale),
                       static_cast<int>(image.rows * scale));

  // Resize the image
  cv::resize(image, resized_image, scaled_size, 0, 0, cv::INTER_CUBIC);

  // Create a blank canvas of the target size
  cv::Mat canvas = cv::Mat::zeros(target_size, resized_image.type());

  // Compute the position to center the scaled image
  int x_offset = (target_size.width - scaled_size.width) / 2;
  int y_offset = (target_size.height - scaled_size.height) / 2;

  // Copy the scaled image onto the canvas
  resized_image.copyTo(canvas(
      cv::Rect(x_offset, y_offset, scaled_size.width, scaled_size.height)));

  // Convert to float and normalize
  cv::Mat float_image;
  canvas.convertTo(float_image, CV_32F, 1.0 / 255.0);

  // Normalize to [-1, 1]
  float_image = (float_image - 0.5f) / 0.5f;

  // Create buffer for normalized data in NCHW format
  std::vector<float> tensor_data(channels * target_size.height *
                                 target_size.width);

  // Convert from HWC to NCHW format
  for (uint c = 0; c < channels; ++c) {
    for (uint h = 0; h < target_size.height; ++h) {
      for (uint w = 0; w < target_size.width; ++w) {
        float pixel_value = float_image.at<cv::Vec3f>(h, w)[c];
        tensor_data[c * target_size.height * target_size.width +
                    h * target_size.width + w] = pixel_value;
      }
    }
  }

  // Set tensor data
  tensor.data.resize(channels * target_size.height * target_size.width *
                     sizeof(float));
  std::memcpy(tensor.data.data(), tensor_data.data(), tensor.data.size());

  // Set datatype to float32
  tensor.data_type = 9; // float32

  // Set shape (NCHW format) - always 1 x 3 x 512 x 384
  tensor.shape.dims =
      std::vector<uint32_t>{1, channels, target_size.height, target_size.width};
  tensor.shape.rank = 4;

  // Set strides
  tensor.strides = std::vector<uint64_t>{
      channels * target_size.height * target_size.width * sizeof(float),
      target_size.height * target_size.width * sizeof(float),
      target_size.width * sizeof(float), sizeof(float)};

  // Set name using frame_id and timestamp
  tensor.name = msg->header.frame_id + std::to_string(msg->header.stamp.sec) +
                "." + std::to_string(msg->header.stamp.nanosec);

  return tensor;
}

// Adds a subscription to the feature_subscriptions_ map
void neuromeshNode::createSubscription(
    std::map<std::string, rclcpp::Subscription<
                              neuromesh_interfaces::msg::Feature>::SharedPtr>
        &subscription_map,
    std::string id, rclcpp::QoS qos) {
  std::string topic = topic_prefix_ + id;

  // RCLCPP_INFO(this->get_logger(), "creating subscription for topic %s",
  // topic.c_str());

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

  std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>
      output_tensors = encoder_result.get();

  if (output_tensors.size() != 1) {
    RCLCPP_WARN(this->get_logger(),
                "Encoder model did not return 1 output tensor as expected.");
    return;
  }

  std::shared_ptr<neuromesh_interfaces::msg::Tensor> feature_tensor =
      output_tensors[0];

  // RCLCPP_INFO(this->get_logger(), "Encoder Feature Tensor size: %ld",
  // feature_tensor->data.size());
  RCLCPP_DEBUG(this->get_logger(), "Encoder took %lims",
               times["encoder_inference"].first);

  if (feature_tensor->result != 0) {
    RCLCPP_WARN(this->get_logger(), "Encoder inference failed.");
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Publishing features.");
  neuromesh_interfaces::msg::Feature feature_msg =
      buildFeatureMessage(*feature_tensor.get());
  feature_publisher_->publish(feature_msg);
  feature_buffer_[this->id_] =
      std::make_shared<neuromesh_interfaces::msg::Feature>(feature_msg);
  feature_buffer_timestamp_[this->id_] =
      rclcpp::Time(feature_msg.timestamp).seconds();

  // Reset for the next cycle
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

std::vector<uint> neuromeshNode::string_to_dims_single(std::string in) {
  std::stringstream stream(in);
  std::string element;
  std::vector<uint> out;

  while (getline(stream, element, ',')) {
    out.push_back(std::stoi(element));
  }
  return out;
}

std::vector<std::vector<uint>> neuromeshNode::string_to_dims(std::string in) {
  std::stringstream stream(in);
  std::string element;
  std::vector<std::vector<uint>> out;

  while (getline(stream, element, ';')) {
    std::vector<uint> dims = string_to_dims_single(element);
    out.push_back(dims);
  }
  return out;
}
} // namespace neuromeshNode
