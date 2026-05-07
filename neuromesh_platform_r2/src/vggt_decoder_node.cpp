#include "neuromesh_platform_r2/vggt_decoder_node.h"
#include <limits>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Quaternion.h>

namespace neuromesh {

// Add this helper function to your class
sensor_msgs::msg::PointCloud2
subsample_pointcloud(const sensor_msgs::msg::PointCloud2 &input_cloud,
                     float leaf_size = 0.05f) {
  // Convert ROS PointCloud2 to PCL
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(input_cloud, *pcl_cloud);

  // Apply voxel grid filter
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setInputCloud(pcl_cloud);
  voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  voxel_filter.filter(*filtered_cloud);

  // Convert back to ROS PointCloud2
  sensor_msgs::msg::PointCloud2 output_cloud;
  pcl::toROSMsg(*filtered_cloud, output_cloud);
  output_cloud.header = input_cloud.header;

  return output_cloud;
}

sensor_msgs::msg::PointCloud2
subsample_rgb_pointcloud(const sensor_msgs::msg::PointCloud2 &input_cloud,
                         float leaf_size = 0.05f) {
  // Convert ROS PointCloud2 to PCL with RGB support
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg(input_cloud, *pcl_cloud);

  // Apply voxel grid filter that preserves color information
  pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
  voxel_filter.setInputCloud(pcl_cloud);
  voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  voxel_filter.filter(*filtered_cloud);

  // Convert back to ROS PointCloud2
  sensor_msgs::msg::PointCloud2 output_cloud;
  pcl::toROSMsg(*filtered_cloud, output_cloud);
  output_cloud.header = input_cloud.header;

  return output_cloud;
}

VggtDecoderNode::VggtDecoderNode(const rclcpp::NodeOptions &options)
    : Node("vggt_decoder", options), processing_in_progress_(false) {

  // Declare and get parameters
  this->declare_parameter<std::string>("robot_name", "");
  this->declare_parameter<double>("vggt.decoder.cycle_interval", 3.0);
  this->declare_parameter<double>("vggt.decoder.feature_age_threshold", 10.0);
  this->declare_parameter<int>("vggt.decoder.num_robots", 2);
  this->declare_parameter<std::string>("vggt.decoder.model_path", "");
  this->declare_parameter<std::vector<std::string>>("vggt.robot_names",
                                                    std::vector<std::string>{});
  this->declare_parameter<double>("vggt.tensorrt.timeout", 30.0);
  this->declare_parameter<int>("vggt.encoder.image_width", 518);
  this->declare_parameter<int>("vggt.encoder.image_height", 392);
  
  // Declare frame_id parameter
  this->declare_parameter<std::string>("frame_id", "");
  
  // Declare voxel leaf size parameter
  this->declare_parameter<double>("vggt.decoder.voxel_leaf_size", 0.02);

  // Get parameters
  robot_name_ = this->get_parameter("robot_name").as_string();
  decoder_cycle_interval_ =
      this->get_parameter("vggt.decoder.cycle_interval").as_double();
  feature_age_threshold_ =
      this->get_parameter("vggt.decoder.feature_age_threshold").as_double();
  num_robots_ = this->get_parameter("vggt.decoder.num_robots").as_int();
  decoder_model_path_ =
      this->get_parameter("vggt.decoder.model_path").as_string();
  robot_names_ = this->get_parameter("vggt.robot_names").as_string_array();
  tensorrt_timeout_ = this->get_parameter("vggt.tensorrt.timeout").as_double();
  image_width_ = this->get_parameter("vggt.encoder.image_width").as_int();
  image_height_ = this->get_parameter("vggt.encoder.image_height").as_int();
  
  // Get frame_id parameter (default to robot_name + "_base_link" if not specified)
  frame_id_ = this->get_parameter("frame_id").as_string();
  if (frame_id_.empty()) {
    frame_id_ = "/" + robot_name_ + "/vggt";
  }
  
  // Get voxel leaf size parameter
  voxel_leaf_size_ = static_cast<float>(this->get_parameter("vggt.decoder.voxel_leaf_size").as_double());
  
  RCLCPP_INFO(this->get_logger(), "VGGT Decoder voxel leaf size: %.3f m", voxel_leaf_size_);

  // Set depth dimensions (same as image for VGGT)
  depth_width_ = image_width_;
  depth_height_ = image_height_;

  // Extract model name from path
  size_t last_slash = decoder_model_path_.find_last_of("/");
  decoder_model_name_ = (last_slash != std::string::npos)
                            ? decoder_model_path_.substr(last_slash + 1)
                            : decoder_model_path_;

  // Ensure current robot is in the robot list
  if (std::find(robot_names_.begin(), robot_names_.end(), robot_name_) ==
      robot_names_.end()) {
    robot_names_.insert(robot_names_.begin(), robot_name_);
  }

  // Adjust robot list to match num_robots
  if (robot_names_.size() > static_cast<size_t>(num_robots_)) {
    robot_names_.resize(num_robots_);
  }

  RCLCPP_INFO(this->get_logger(), "VggtDecoderNode initialized for robot: %s",
              robot_name_.c_str());
  RCLCPP_INFO(this->get_logger(), "Decoder model: %s",
              decoder_model_name_.c_str());
  RCLCPP_INFO(this->get_logger(), "Decoder cycle interval: %.2f seconds",
              decoder_cycle_interval_);
  RCLCPP_INFO(this->get_logger(), "Feature age threshold: %.2f seconds",
              feature_age_threshold_);
  RCLCPP_INFO(this->get_logger(), "Number of robots: %d", num_robots_);
  RCLCPP_INFO(this->get_logger(), "Frame ID: %s", frame_id_.c_str());
  RCLCPP_INFO(this->get_logger(), "Robot names: %s",
              std::accumulate(robot_names_.begin(), robot_names_.end(),
                              std::string(),
                              [](const std::string &a, const std::string &b) {
                                return a.empty() ? b : a + ", " + b;
                              })
                  .c_str());

  // Create TensorRT client for decoder
  tensorrt_client_ =
      this->create_client<neuromesh_interfaces::srv::TensorRequest>(
          "tensorrt_request_decoder");

  // Wait for TensorRT service
  while (!tensorrt_client_->wait_for_service(std::chrono::seconds(1))) {
    if (!rclcpp::ok()) {
      RCLCPP_ERROR(this->get_logger(),
                   "Interrupted while waiting for TensorRT service. Exiting.");
      return;
    }
    RCLCPP_INFO(this->get_logger(), "Waiting for TensorRT service...");
  }

  // Create feature subscriptions for all robots
  for (const auto &robot : robot_names_) {
    std::string feature_topic = "/" + robot + "/features_" + robot;
    auto qos = rclcpp::QoS(10).reliability(rclcpp::ReliabilityPolicy::Reliable);

    feature_subs_[robot] =
        this->create_subscription<neuromesh_interfaces::msg::Feature>(
            feature_topic, qos,
            [this](const neuromesh_interfaces::msg::Feature::SharedPtr msg) {
              this->feature_callback(msg);
            });

    RCLCPP_INFO(this->get_logger(), "Subscribed to features from: %s",
                feature_topic.c_str());
  }

  // Create RGB image subscription for the current robot
  std::string rgb_topic = "/" + robot_name_ + "/resized_rgb";
  auto rgb_qos =
      rclcpp::QoS(10).reliability(rclcpp::ReliabilityPolicy::Reliable);
  rgb_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      rgb_topic, rgb_qos,
      std::bind(&VggtDecoderNode::rgb_image_callback, this,
                std::placeholders::_1));
  RCLCPP_INFO(this->get_logger(), "Subscribed to RGB images from: %s",
              rgb_topic.c_str());
  
  // Create encoder sync depth subscription for the current robot
  std::string encoder_sync_depth_topic = "/" + robot_name_ + "/encoder_sync_depth";
  auto encoder_sync_depth_qos =
      rclcpp::QoS(10).reliability(rclcpp::ReliabilityPolicy::Reliable);
  encoder_sync_depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      encoder_sync_depth_topic, encoder_sync_depth_qos,
      std::bind(&VggtDecoderNode::encoder_sync_depth_callback, this,
                std::placeholders::_1));
  RCLCPP_INFO(this->get_logger(), "Subscribed to encoder sync depth from: %s",
              encoder_sync_depth_topic.c_str());

  // Create output publishers
  for (int i = 0; i < num_robots_; ++i) {
    std::string depth_topic =
        "/" + robot_name_ + "/depth_robot" + std::to_string(i + 1);
    depth_pubs_["robot" + std::to_string(i + 1)] =
        this->create_publisher<sensor_msgs::msg::Image>(depth_topic, 10);
  }

  // Create point cloud publishers
  std::string pc_current_topic = "/" + robot_name_ + "/pointcloud_current";
  std::string pc_neighbor_topic = "/" + robot_name_ + "/pointcloud_neighbor";
  std::string pc_rgb_topic = "/" + robot_name_ + "/pointcloud_rgb";

  pointcloud_pubs_["current"] =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(pc_current_topic,
                                                            10);
  pointcloud_pubs_["neighbor"] =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(pc_neighbor_topic,
                                                            10);
  rgb_pointcloud_pub_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(pc_rgb_topic, 10);
  
  // Create decoder sync depth publisher
  std::string decoder_sync_depth_topic = "/" + robot_name_ + "/decoder_sync_depth";
  decoder_sync_depth_pub_ =
      this->create_publisher<sensor_msgs::msg::Image>(decoder_sync_depth_topic, 10);
  RCLCPP_INFO(this->get_logger(), "Publishing decoder sync depth to: %s", decoder_sync_depth_topic.c_str());

  // Create timer for decoder processing
  decoder_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(decoder_cycle_interval_),
      std::bind(&VggtDecoderNode::decoder_timer_callback, this));

  RCLCPP_INFO(this->get_logger(), "VggtDecoderNode setup complete");
}

void VggtDecoderNode::feature_callback(
    const neuromesh_interfaces::msg::Feature::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(feature_mutex_);

  RCLCPP_DEBUG(this->get_logger(), "Received features from robot: %s",
               msg->id.c_str());

  // Update feature buffer
  feature_buffer_[msg->id] = msg;
  feature_timestamps_[msg->id] = this->now();
}

void VggtDecoderNode::rgb_image_callback(
    const sensor_msgs::msg::Image::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(rgb_mutex_);

  try {
    // Convert ROS image to OpenCV
    cv_bridge::CvImagePtr cv_ptr =
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);

    // Add to buffer
    RgbImageData rgb_data;
    rgb_data.image = cv_ptr->image.clone();
    rgb_data.timestamp = msg->header.stamp;

    rgb_image_buffer_.push_back(rgb_data);

    // Keep only the last MAX_RGB_BUFFER_SIZE images
    while (rgb_image_buffer_.size() > MAX_RGB_BUFFER_SIZE) {
      rgb_image_buffer_.pop_front();
    }

    RCLCPP_DEBUG(this->get_logger(),
                 "Received RGB image with timestamp: %d.%d, buffer size: %zu",
                 msg->header.stamp.sec, msg->header.stamp.nanosec,
                 rgb_image_buffer_.size());

  } catch (cv_bridge::Exception &e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception in RGB callback: %s",
                 e.what());
  }
}

void VggtDecoderNode::encoder_sync_depth_callback(
    const sensor_msgs::msg::Image::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(encoder_sync_depth_mutex_);
  
  // Add to buffer
  EncoderSyncDepthData depth_data;
  depth_data.depth_image = msg;
  depth_data.timestamp = rclcpp::Time(msg->header.stamp);
  
  encoder_sync_depth_buffer_.push_back(depth_data);
  
  // Keep only the last MAX_ENCODER_SYNC_DEPTH_BUFFER_SIZE images
  while (encoder_sync_depth_buffer_.size() > MAX_ENCODER_SYNC_DEPTH_BUFFER_SIZE) {
    encoder_sync_depth_buffer_.pop_front();
  }
  
  RCLCPP_DEBUG(this->get_logger(), 
               "Received encoder sync depth with timestamp: %d.%d, buffer size: %zu",
               msg->header.stamp.sec, msg->header.stamp.nanosec,
               encoder_sync_depth_buffer_.size());
}

void VggtDecoderNode::decoder_timer_callback() {
  // Check if previous decoding is still in progress
  if (processing_in_progress_.load()) {
    RCLCPP_WARN(this->get_logger(),
                "Previous decoding still in progress, skipping this cycle");
    return;
  }

  // Check if decoder result is ready
  if (decoder_result_.valid() &&
      decoder_result_.wait_for(std::chrono::milliseconds(0)) ==
          std::future_status::ready) {

    RCLCPP_INFO(this->get_logger(),
                "Decoder result ready, processing outputs...");
    try {
      auto results = decoder_result_.get();
      process_decoder_output(results);
      stop_clock("decoder_inference");
      RCLCPP_INFO(this->get_logger(), "Decoder inference took: %ld ms",
                  check_clock("decoder_inference"));
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Error processing decoder result: %s",
                   e.what());
    }
  }

  // Check if we have features from current robot (required)
  bool has_self_features = false;
  {
    std::lock_guard<std::mutex> lock(feature_mutex_);
    has_self_features =
        feature_buffer_.find(robot_name_) != feature_buffer_.end();
  }

  if (!has_self_features) {
    RCLCPP_DEBUG(this->get_logger(),
                 "No features from self (%s), skipping decoder",
                 robot_name_.c_str());
    return;
  }

  // Try to build decoder tensor
  neuromesh_interfaces::msg::Tensor decoder_input;
  if (build_decoder_tensor(decoder_input)) {
    RCLCPP_INFO(this->get_logger(),
                "Starting decoder inference with tensor size: %zu bytes",
                decoder_input.data.size());

    processing_in_progress_ = true;
    start_clock("decoder_inference");
    decoder_result_ = perform_inference({decoder_input});
  }
}

bool VggtDecoderNode::build_decoder_tensor(
    neuromesh_interfaces::msg::Tensor &output_tensor) {
  std::lock_guard<std::mutex> lock(feature_mutex_);

  // Get self features (required)
  if (feature_buffer_.find(robot_name_) == feature_buffer_.end()) {
    RCLCPP_ERROR(this->get_logger(), "No features available from self (%s)",
                 robot_name_.c_str());
    return false;
  }

  auto self_features = feature_buffer_[robot_name_];

  // Prepare aggregated tensor data
  std::vector<float> aggregated_data;
  size_t feature_size = self_features->tensor.data.size() / sizeof(float);

  RCLCPP_INFO(this->get_logger(), "Building decoder tensor for %d robots",
              num_robots_);

  // Aggregate features from all robots
  for (int i = 0; i < num_robots_; ++i) {
    const float *feature_data = nullptr;
    std::string source_robot = robot_name_; // Default to self

    if (i < static_cast<int>(robot_names_.size())) {
      const std::string &target_robot = robot_names_[i];

      if (feature_buffer_.find(target_robot) != feature_buffer_.end()) {
        double age_seconds = 0.0;
        if (check_feature_freshness(target_robot, age_seconds)) {
          // Use target robot's features
          feature_data = reinterpret_cast<const float *>(
              feature_buffer_[target_robot]->tensor.data.data());
          source_robot = target_robot;
        } else {
          // Features too old, use self features
          RCLCPP_WARN(this->get_logger(),
                      "Features from %s are %.2f seconds old (threshold: "
                      "%.2f). Using self features.",
                      target_robot.c_str(), age_seconds,
                      feature_age_threshold_);
          feature_data = reinterpret_cast<const float *>(
              self_features->tensor.data.data());
        }
      } else {
        // No features available, use self features
        RCLCPP_WARN(this->get_logger(),
                    "No features available from %s. Using self features.",
                    target_robot.c_str());
        feature_data =
            reinterpret_cast<const float *>(self_features->tensor.data.data());
      }
    } else {
      // Beyond configured robots, use self features
      feature_data =
          reinterpret_cast<const float *>(self_features->tensor.data.data());
    }

    // Copy feature data
    aggregated_data.insert(aggregated_data.end(), feature_data,
                           feature_data + feature_size);
    RCLCPP_DEBUG(this->get_logger(), "Robot %d: using features from %s", i,
                 source_robot.c_str());
  }

  // Build output tensor
  output_tensor.name = "aggregated_features";
  output_tensor.shape.dims = {static_cast<uint32_t>(num_robots_), 1036,
                              1024}; // VGGT-specific dimensions
  output_tensor.data_type = 9;       // float32

  // Calculate strides
  output_tensor.strides.push_back(1036 * 1024 * sizeof(float));
  output_tensor.strides.push_back(1024 * sizeof(float));
  output_tensor.strides.push_back(sizeof(float));

  // Copy data
  output_tensor.data.resize(aggregated_data.size() * sizeof(float));
  std::memcpy(output_tensor.data.data(), aggregated_data.data(),
              output_tensor.data.size());

  RCLCPP_INFO(
      this->get_logger(),
      "Built decoder tensor with shape [%d, 1036, 1024], size: %zu bytes",
      num_robots_, output_tensor.data.size());

  return true;
}

bool VggtDecoderNode::check_feature_freshness(const std::string &robot_name,
                                              double &age_seconds) {
  auto it = feature_timestamps_.find(robot_name);
  if (it == feature_timestamps_.end()) {
    age_seconds = std::numeric_limits<double>::max();
    return false;
  }

  auto age = this->now() - it->second;
  age_seconds = age.seconds();

  return age_seconds < feature_age_threshold_;
}

std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>
VggtDecoderNode::perform_inference(
    const std::vector<neuromesh_interfaces::msg::Tensor> &inputs) {

  auto request =
      std::make_shared<neuromesh_interfaces::srv::TensorRequest::Request>();
  request->model_name = decoder_model_name_;
  request->tensor1 = inputs;

  // Create promise and future
  auto promise = std::make_shared<std::promise<
      std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>>();
  auto future = promise->get_future();

  // Send async request
  auto response_future = tensorrt_client_->async_send_request(request);

  // Handle response in a separate thread to avoid blocking
  std::thread(
      [this, promise](auto response_future) {
        try {
          // Wait for response with timeout
          auto status = response_future.wait_for(
              std::chrono::duration<double>(tensorrt_timeout_));

          if (status == std::future_status::ready) {
            auto response = response_future.get();

            if (response) {
              std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>
                  result;
              for (const auto &tensor : response->tensor2) {
                result.push_back(
                    std::make_shared<neuromesh_interfaces::msg::Tensor>(
                        tensor));
              }
              promise->set_value(result);
            } else {
              RCLCPP_ERROR(this->get_logger(),
                           "TensorRT inference failed - null response");
              promise->set_value({});
            }
          } else {
            RCLCPP_ERROR(this->get_logger(), "TensorRT request timed out");
            promise->set_value({});
          }
        } catch (const std::exception &e) {
          RCLCPP_ERROR(this->get_logger(), "Exception in TensorRT request: %s",
                       e.what());
          promise->set_value({});
        }

        processing_in_progress_ = false;
      },
      std::move(response_future))
      .detach();

  return future;
}

void VggtDecoderNode::process_decoder_output(
    const std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>
        &outputs) {

  if (outputs.empty()) {
    RCLCPP_ERROR(this->get_logger(), "No decoder outputs received");
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Processing %zu decoder outputs",
              outputs.size());

  // VGGT decoder outputs (based on the model):
  // 0: pose_enc (1x2x9)
  // 1: depth (1x2x392x518x1)
  // 2: depth_conf (1x2x392x518)
  // 3: world_points (1x2x392x518x3)
  // 4: world_points_conf (1x2x392x518)

  const float *depth_data = nullptr;
  const float *depth_conf_data = nullptr;
  const float *world_points_data = nullptr;
  const float *world_points_conf_data = nullptr;

  // Extract data pointers from tensors
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto &tensor = outputs[i];
    RCLCPP_DEBUG(this->get_logger(), "Output %zu: %s, shape: [%s]", i,
                 tensor->name.c_str(),
                 std::accumulate(tensor->shape.dims.begin(),
                                 tensor->shape.dims.end(), std::string(),
                                 [](const std::string &a, uint32_t b) {
                                   return a.empty()
                                              ? std::to_string(b)
                                              : a + ", " + std::to_string(b);
                                 })
                     .c_str());

    if (tensor->name == "depth" || i == 1) {
      depth_data = reinterpret_cast<const float *>(tensor->data.data());
    } else if (tensor->name == "depth_conf" || i == 2) {
      depth_conf_data = reinterpret_cast<const float *>(tensor->data.data());
    } else if (tensor->name == "world_points" || i == 3) {
      world_points_data = reinterpret_cast<const float *>(tensor->data.data());
    } else if (tensor->name == "world_points_conf" || i == 4) {
      world_points_conf_data =
          reinterpret_cast<const float *>(tensor->data.data());
    }
  }


  // Process outputs for each robot
  for (int robot_idx = 0; robot_idx < num_robots_; ++robot_idx) {
    if (depth_data && depth_conf_data) {
      // Calculate offsets for this robot's data
      size_t pixel_count = depth_width_ * depth_height_;
      const float *robot_depth = depth_data + robot_idx * pixel_count;
      const float *robot_depth_conf = depth_conf_data + robot_idx * pixel_count;

      // Create and publish depth image
      auto depth_image = create_depth_image(robot_depth, robot_depth_conf,
                                            depth_width_, depth_height_);
      depth_image.header.stamp = this->now();
      depth_image.header.frame_id = frame_id_;

      std::string robot_key = "robot" + std::to_string(robot_idx + 1);
      if (depth_pubs_.find(robot_key) != depth_pubs_.end()) {
        depth_pubs_[robot_key]->publish(depth_image);
      }
    }

    if (world_points_data && world_points_conf_data) {
      // Calculate offsets for this robot's data
      size_t pixel_count = depth_width_ * depth_height_;
      const float *robot_points =
          world_points_data + robot_idx * pixel_count * 3;
      const float *robot_points_conf =
          world_points_conf_data + robot_idx * pixel_count;

      // Create and publish point cloud
      auto pointcloud =
          create_point_cloud(robot_points, robot_points_conf, depth_width_,
                             depth_height_, frame_id_);

      // Apply subsampling before publishing
      auto subsampled_pointcloud =
          subsample_pointcloud(pointcloud, voxel_leaf_size_);

      // Publish to appropriate topic
      if (robot_idx == 0 &&
          pointcloud_pubs_.find("current") != pointcloud_pubs_.end()) {
        pointcloud_pubs_["current"]->publish(subsampled_pointcloud);
      } else if (robot_idx == 1 &&
                 pointcloud_pubs_.find("neighbor") != pointcloud_pubs_.end()) {
        pointcloud_pubs_["neighbor"]->publish(subsampled_pointcloud);
      }

      // Create RGB point cloud if we have color data for current robot
      if (robot_idx == 0) {
        // Get the timestamp of the current robot's features
        rclcpp::Time feature_timestamp;
        {
          std::lock_guard<std::mutex> feature_lock(feature_mutex_);
          if (feature_buffer_.find(robot_name_) != feature_buffer_.end()) {
            feature_timestamp = feature_buffer_[robot_name_]->timestamp;
          } else {
            feature_timestamp = this->now();
          }
        }

        // Find the RGB image with the closest timestamp
        cv::Mat rgb_image;
        if (find_closest_rgb_image(feature_timestamp, rgb_image)) {
          auto rgb_pointcloud =
              create_rgb_point_cloud(robot_points, robot_points_conf, rgb_image,
                                     depth_width_, depth_height_, frame_id_);
          rgb_pointcloud_pub_->publish(rgb_pointcloud);
          RCLCPP_DEBUG(this->get_logger(),
                       "Published RGB point cloud");
        } else {
          RCLCPP_INFO(this->get_logger(),
                      "No RGB image found within acceptable timestamp range, "
                      "skipping RGB point cloud");
        }
        
        // Find and publish decoder sync depth corresponding to the feature timestamp
        sensor_msgs::msg::Image encoder_sync_depth;
        if (find_closest_encoder_sync_depth(feature_timestamp, encoder_sync_depth)) {
          // Create decoder sync depth message with same content as encoder sync depth
          // but with updated timestamp to match the decoder output
          auto decoder_sync_depth = encoder_sync_depth;
          decoder_sync_depth.header.stamp = this->now();
          decoder_sync_depth.header.frame_id = frame_id_;
          decoder_sync_depth_pub_->publish(decoder_sync_depth);
          RCLCPP_DEBUG(this->get_logger(),
                       "Published decoder sync depth with timestamp: %d.%d",
                       decoder_sync_depth.header.stamp.sec, decoder_sync_depth.header.stamp.nanosec);
        } else {
          RCLCPP_DEBUG(this->get_logger(),
                       "No encoder sync depth found within acceptable timestamp range, "
                       "skipping decoder sync depth");
        }
      }
    }
  }

  RCLCPP_INFO(this->get_logger(), "Decoder outputs processed and published");
}

sensor_msgs::msg::Image
VggtDecoderNode::create_depth_image(const float *depth_data,
                                    const float *confidence_data, int width,
                                    int height) {

  sensor_msgs::msg::Image depth_image;
  depth_image.width = width;
  depth_image.height = height;
  depth_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  depth_image.step = width * sizeof(float);
  depth_image.data.resize(width * height * sizeof(float));

  // Apply confidence threshold and copy depth values
  float *output = reinterpret_cast<float *>(depth_image.data.data());
  const float confidence_threshold = 0.5f;

  for (int i = 0; i < width * height; ++i) {
    if (confidence_data[i] > confidence_threshold) {
      output[i] = depth_data[i];
    } else {
      output[i] = std::numeric_limits<float>::quiet_NaN();
    }
  }

  return depth_image;
}

sensor_msgs::msg::PointCloud2
VggtDecoderNode::create_point_cloud(const float *world_points,
                                    const float *confidence_data, int width,
                                    int height, const std::string &frame_id) {

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->width = width * height;
  cloud->height = 1;
  cloud->is_dense = false;
  cloud->points.reserve(width * height);

  const float confidence_threshold = 0.5f;

  for (int i = 0; i < width * height; ++i) {
    if (confidence_data[i] > confidence_threshold) {
      pcl::PointXYZ point;
      point.x = world_points[i * 3 + 0];
      point.y = world_points[i * 3 + 1];
      point.z = world_points[i * 3 + 2];

      // Filter out invalid points
      if (!std::isnan(point.x) && !std::isnan(point.y) &&
          !std::isnan(point.z) && std::isfinite(point.x) &&
          std::isfinite(point.y) && std::isfinite(point.z)) {
        cloud->points.push_back(point);
      }
    }
  }

  // Convert to ROS message
  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(*cloud, cloud_msg);
  cloud_msg.header.stamp = this->now();
  cloud_msg.header.frame_id = frame_id;

  return cloud_msg;
}

sensor_msgs::msg::PointCloud2 VggtDecoderNode::create_rgb_point_cloud(
    const float *world_points, const float *confidence_data,
    const cv::Mat &rgb_image, int width, int height,
    const std::string &frame_id) {

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  cloud->width = width * height;
  cloud->height = 1;
  cloud->is_dense = false;
  cloud->points.reserve(width * height);

  const float confidence_threshold = 0.5f;

  // Ensure RGB image matches expected dimensions
  cv::Mat resized_rgb;
  if (rgb_image.cols != width || rgb_image.rows != height) {
    cv::resize(rgb_image, resized_rgb, cv::Size(width, height));
  } else {
    resized_rgb = rgb_image;
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = y * width + x;

      if (confidence_data[idx] > confidence_threshold) {
        pcl::PointXYZRGB point;
        point.x = world_points[idx * 3 + 0];
        point.y = world_points[idx * 3 + 1];
        point.z = world_points[idx * 3 + 2];

        // Get RGB values
        cv::Vec3b color = resized_rgb.at<cv::Vec3b>(y, x);
        point.r = color[0];
        point.g = color[1];
        point.b = color[2];

        // Filter out invalid points
        if (!std::isnan(point.x) && !std::isnan(point.y) &&
            !std::isnan(point.z) && std::isfinite(point.x) &&
            std::isfinite(point.y) && std::isfinite(point.z)) {
          cloud->points.push_back(point);
        }
      }
    }
  }

  // Convert to ROS message
  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(*cloud, cloud_msg);
  cloud_msg.header.stamp = this->now();
  cloud_msg.header.frame_id = frame_id;

  return cloud_msg;
}

void VggtDecoderNode::start_clock(const std::string &name) {
  clock_map_[name] = std::chrono::high_resolution_clock::now();
}

void VggtDecoderNode::stop_clock(const std::string &name) {
  if (clock_map_.find(name) != clock_map_.end()) {
    auto duration =
        std::chrono::high_resolution_clock::now() - clock_map_[name];
    RCLCPP_DEBUG(this->get_logger(), "%s took %ld ms", name.c_str(),
                 std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                     .count());
  }
}

long VggtDecoderNode::check_clock(const std::string &name) {
  if (clock_map_.find(name) != clock_map_.end()) {
    auto duration =
        std::chrono::high_resolution_clock::now() - clock_map_[name];
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration)
        .count();
  }
  return -1;
}

bool VggtDecoderNode::find_closest_rgb_image(const rclcpp::Time &target_time,
                                             cv::Mat &rgb_image) {
  std::lock_guard<std::mutex> lock(rgb_mutex_);

  if (rgb_image_buffer_.empty()) {
    return false;
  }

  // Find the RGB image with the closest timestamp
  double min_time_diff = std::numeric_limits<double>::max();
  size_t best_idx = 0;

  for (size_t i = 0; i < rgb_image_buffer_.size(); ++i) {
    double time_diff =
        std::abs((rgb_image_buffer_[i].timestamp - target_time).seconds());
    if (time_diff < min_time_diff) {
      min_time_diff = time_diff;
      best_idx = i;
    }
  }

  // Maximum allowed time difference (0.1 seconds)
  const double MAX_TIME_DIFF = 0.1;

  if (min_time_diff > MAX_TIME_DIFF) {
    RCLCPP_DEBUG(this->get_logger(),
                 "No RGB image found within %.2f seconds of target time. "
                 "Closest was %.3f seconds away.",
                 MAX_TIME_DIFF, min_time_diff);
    return false;
  }

  rgb_image = rgb_image_buffer_[best_idx].image.clone();
  RCLCPP_DEBUG(this->get_logger(),
               "Found RGB image with time difference: %.3f seconds",
               min_time_diff);

  return true;
}

bool VggtDecoderNode::find_closest_encoder_sync_depth(const rclcpp::Time &target_time,
                                                      sensor_msgs::msg::Image &depth_image) {
  std::lock_guard<std::mutex> lock(encoder_sync_depth_mutex_);

  if (encoder_sync_depth_buffer_.empty()) {
    RCLCPP_DEBUG(this->get_logger(), "No encoder sync depth available in buffer");
    return false;
  }

  // Find the encoder sync depth with the closest timestamp
  double min_time_diff = std::numeric_limits<double>::max();
  size_t best_idx = 0;

  for (size_t i = 0; i < encoder_sync_depth_buffer_.size(); ++i) {
    double time_diff = std::abs((encoder_sync_depth_buffer_[i].timestamp - target_time).seconds());
    if (time_diff < min_time_diff) {
      min_time_diff = time_diff;
      best_idx = i;
    }
  }

  // Maximum allowed time difference (0.5 seconds)
  const double MAX_TIME_DIFF = 0.5;

  if (min_time_diff > MAX_TIME_DIFF) {
    RCLCPP_DEBUG(this->get_logger(),
                 "No encoder sync depth found within %.2f seconds of target time. "
                 "Closest was %.3f seconds away.",
                 MAX_TIME_DIFF, min_time_diff);
    return false;
  }

  depth_image = *encoder_sync_depth_buffer_[best_idx].depth_image;
  RCLCPP_DEBUG(this->get_logger(),
               "Found encoder sync depth with time difference: %.3f seconds",
               min_time_diff);

  return true;
}

} // namespace neuromesh

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(neuromesh::VggtDecoderNode)
