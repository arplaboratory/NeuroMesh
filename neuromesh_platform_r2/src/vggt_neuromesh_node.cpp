#include "neuromesh_platform_r2/vggt_neuromesh_node.h"
#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include "chrono"

namespace vggtNode {
vggtNode::vggtNode(const rclcpp::NodeOptions &options): Node("vggt_node", options)
{
    // Declare node parameters
    this->declare_parameter<std::string>("encoder_model_name", "vggt_encoder");
    this->declare_parameter<std::string>("decoder_model_name", "vggt_decoder");
    this->declare_parameter<std::string>("topic_prefix", "features_");
    this->declare_parameter<std::string>("output_topic", "vggt_output");
    this->declare_parameter<int>("decoder_cycle_length", 3000);
    this->declare_parameter<int>("encoder_cycle_length", 3000);
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
    this->get_parameter("features_qos_profile", features_qos_profile_);
    this->get_parameter("output_qos_profile", output_qos_profile_);
    this->get_parameter("agents", agents_);
    this->get_parameter("to_nchw", to_nchw_);
    this->get_parameter("ints_to_floats", ints_to_floats_);

    // Declare VGGT decoder output dimensions parameter
    this->declare_parameter("vggt_decoder_output_dimensions", "1,2,9;1,2,392,518,1;1,2,392,518;1,2,392,518,3;1,2,392,518");
    this->get_parameter("vggt_decoder_output_dimensions", decoder_output_dimensions_str);

    auto feature_qos = rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(features_qos_profile_));
    auto output_qos = rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(output_qos_profile_));
    
    // Create publishers
    feature_publisher_ = this->create_publisher<neuromesh_interfaces::msg::Feature>(topic_prefix_ + id_, feature_qos);
    output_publisher_ = this->create_publisher<neuromesh_interfaces::msg::Tensor>(output_topic_, output_qos);
    
    // VGGT-specific publishers
    depth_robot1_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("depth_robot1", output_qos);
    depth_robot2_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("depth_robot2", output_qos);
    pointcloud_current_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("pointcloud_current", output_qos);
    pointcloud_neighbor_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("pointcloud_neighbor", output_qos);
    pointcloud_current_rgb_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("pointcloud_current_rgb", output_qos);
    pointcloud_neighbor_rgb_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("pointcloud_neighbor_rgb", output_qos);

    // Add TransformBroadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Add a timer to broadcast the transform periodically
    transform_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&vggtNode::broadcast_transform, this));

    // Parse agent list
    all_agents = splitAgentString(agents_);
    RCLCPP_INFO(this->get_logger(), "VGGT Agents:");
    for (const auto& agent : all_agents) {
        RCLCPP_INFO(this->get_logger(), "%s", agent.c_str());
    }
    all_agents.erase(id_); // remove self from list

    // Parse decoder output dimensions
    decoder_output_dims = string_to_dims(decoder_output_dimensions_str);

    // Create camera subscription
    auto image_qos = rclcpp::QoS(rclcpp::KeepLast(10), parseQoSString(image_qos_profile_));
    camera_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        "camera", image_qos,
        std::bind(&vggtNode::camera_callback, this, std::placeholders::_1));

    // Create feature subscriptions for other agents
    for (const auto& agent : all_agents) {
        createSubscription(feature_subscriptions_, agent, feature_qos);
    }

    // Create timers
    decoder_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(decoder_cycle_length_),
        std::bind(&vggtNode::process_features, this));

    encoder_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(encoder_cycle_length_),
        std::bind(&vggtNode::run_encoder_cycle, this));

    fresh_encoder_cycle = true;

    RCLCPP_INFO(this->get_logger(), "VGGT Node initialized with encoder: %s, decoder: %s", 
                encoder_model_name_.c_str(), decoder_model_name_.c_str());
}

void vggtNode::camera_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "=== Camera callback triggered ===");
    
    {
        std::lock_guard<std::mutex> lock(camera_msg_mutex_);
        latest_camera_msg_ = msg;
    }

    RCLCPP_INFO(this->get_logger(), "fresh_encoder_cycle: %s, encoder_cycle_count: %d", 
                fresh_encoder_cycle ? "true" : "false", encoder_cycle_count_);

    if (fresh_encoder_cycle) {
        RCLCPP_INFO(this->get_logger(), "Starting encoder inference...");
        startClock("encoder_inference");
        
        // Convert image to tensor with VGGT preprocessing
        auto tensor = imageToTensor(msg);
        RCLCPP_INFO(this->get_logger(), "Image converted to tensor: dims=[%s], data_size=%zu", 
                    tensor.shape.dims.empty() ? "empty" : 
                    std::accumulate(tensor.shape.dims.begin(), tensor.shape.dims.end(), std::string(),
                        [](const std::string& a, uint32_t b) { return a.empty() ? std::to_string(b) : a + ", " + std::to_string(b); }).c_str(),
                    tensor.data.size());
        
        // Perform encoder inference
        std::vector<neuromesh_interfaces::msg::Tensor> input_tensors = {tensor};
        RCLCPP_INFO(this->get_logger(), "Calling performInference for encoder...");
        // Always create a new future for each encoder cycle
        encoder_result = performInference(encoder_model_name_, input_tensors);
        
        fresh_encoder_cycle = false;
        encoder_cycle_count_++;
        RCLCPP_INFO(this->get_logger(), "Encoder inference started, cycle count: %d", encoder_cycle_count_);
    } else {
        RCLCPP_DEBUG(this->get_logger(), "Skipping encoder inference - not a fresh cycle");
    }
}

neuromesh_interfaces::msg::Tensor 
vggtNode::imageToTensor(const sensor_msgs::msg::Image::SharedPtr msg) {
    neuromesh_interfaces::msg::Tensor tensor;
    
    try {
        // Convert ROS image to OpenCV
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        cv::Mat image = cv_ptr->image;
        
        // Resize to VGGT input dimensions: 392x518
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(tensor_width_, tensor_height_));
        
        // Normalize to [-1, 1] range
        cv::Mat normalized_image;
        resized_image.convertTo(normalized_image, CV_32F, 2.0/255.0, -1.0);
        
        // Set tensor dimensions [1, 3, 392, 518]
        tensor.shape.dims = {tensor_batch_size_, tensor_channels_, tensor_height_, tensor_width_};
        
        // Convert HWC to CHW format for neural network input
        std::vector<cv::Mat> channels(3);
        cv::split(normalized_image, channels);
        
        // Prepare float data
        std::vector<float> float_data;
        float_data.reserve(tensor_batch_size_ * tensor_channels_ * tensor_height_ * tensor_width_);
        
        // Add data in CHW order
        for (int c = 0; c < 3; ++c) {
            float* channel_data = reinterpret_cast<float*>(channels[c].data);
            size_t channel_size = tensor_height_ * tensor_width_;
            for (size_t i = 0; i < channel_size; ++i) {
                float_data.push_back(channel_data[i]);
            }
        }
        
        // Convert float data to uint8 data
        tensor.data_type = 9; // float32
        tensor.data.resize(float_data.size() * sizeof(float));
        std::memcpy(tensor.data.data(), float_data.data(), float_data.size() * sizeof(float));
        
        // Set metadata
        tensor.name = msg->header.frame_id + std::to_string(msg->header.stamp.sec) + 
                     std::to_string(msg->header.stamp.nanosec);
        
        RCLCPP_DEBUG(this->get_logger(), "Image converted to tensor: [%u, %u, %u, %u]", 
                    tensor.shape.dims[0], tensor.shape.dims[1], tensor.shape.dims[2], tensor.shape.dims[3]);
                    
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
    
    return tensor;
}

void vggtNode::feature_callback(const neuromesh_interfaces::msg::Feature::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "=== Feature callback from agent: %s ===", msg->id.c_str());
    RCLCPP_INFO(this->get_logger(), "Feature tensor dims: [%s], data_size: %zu",
                msg->tensor.shape.dims.empty() ? "empty" : 
                std::accumulate(msg->tensor.shape.dims.begin(), msg->tensor.shape.dims.end(), std::string(),
                    [](const std::string& a, uint32_t b) { return a.empty() ? std::to_string(b) : a + ", " + std::to_string(b); }).c_str(),
                msg->tensor.data.size());
    
    feature_buffer_[msg->id] = msg;
    feature_buffer_timestamp_[msg->id] = this->get_clock()->now().seconds();
    
    RCLCPP_INFO(this->get_logger(), "Feature buffer updated, current size: %zu", feature_buffer_.size());
}

void vggtNode::process_features() {
    auto now = std::chrono::system_clock::now();
    auto time_since_epoch = now.time_since_epoch();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time_since_epoch);
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(time_since_epoch - seconds);
    
    RCLCPP_INFO(this->get_logger(), "=== START process_features at %ld.%09ld ===", 
                seconds.count(), nanoseconds.count());
    
    try {
        // Check if we have our own encoder result
        RCLCPP_INFO(this->get_logger(), "Checking encoder result validity: %s", 
                    encoder_result.valid() ? "valid" : "invalid");
        if (encoder_result.valid() && 
            encoder_result.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            
            RCLCPP_INFO(this->get_logger(), "Encoder result is ready, retrieving...");
            auto encoder_results = encoder_result.get();
            RCLCPP_INFO(this->get_logger(), "Retrieved %zu encoder results", encoder_results.size());
            // Note: After get(), encoder_result becomes invalid and will remain so until next encoder cycle
            
            if (!encoder_results.empty() && encoder_results[0]) {
                RCLCPP_INFO(this->get_logger(), "Encoder result tensor dims: [%s], data_size: %zu",
                            encoder_results[0]->shape.dims.empty() ? "empty" : 
                            std::accumulate(encoder_results[0]->shape.dims.begin(), encoder_results[0]->shape.dims.end(), std::string(),
                                [](const std::string& a, uint32_t b) { return a.empty() ? std::to_string(b) : a + ", " + std::to_string(b); }).c_str(),
                            encoder_results[0]->data.size());
                
                // Check if the tensor has valid data
                if (encoder_results[0]->data.empty() || encoder_results[0]->shape.dims.empty()) {
                    RCLCPP_ERROR(this->get_logger(), "Encoder returned empty tensor data or dimensions!");
                } else {
                    // Store our own feature
                    auto own_feature = buildFeatureMessage(*encoder_results[0]);
                    feature_buffer_[id_] = std::make_shared<neuromesh_interfaces::msg::Feature>(own_feature);
                    feature_buffer_timestamp_[id_] = this->get_clock()->now().seconds();
                    
                    // Publish our feature for other agents
                    feature_publisher_->publish(own_feature);
                    
                    stopClock("encoder_inference");
                    RCLCPP_INFO(this->get_logger(), "Encoder inference took: %ld ms", checkClock("encoder_inference"));
                    RCLCPP_INFO(this->get_logger(), "Published feature for agent: %s with tensor size: %zu bytes", 
                                id_.c_str(), own_feature.tensor.data.size());
                }
            } else {
                RCLCPP_WARN(this->get_logger(), "Encoder results empty or null!");
            }
        } else {
            RCLCPP_DEBUG(this->get_logger(), "Encoder result not ready yet");
        }
        
        // Check if we have features from neighbor agents for decoder
        RCLCPP_INFO(this->get_logger(), "Current feature buffer size: %zu", feature_buffer_.size());
        if (feature_buffer_.size() >= 2) { // Need at least 2 agents (self + 1 neighbor)
            RCLCPP_INFO(this->get_logger(), "Have enough features for decoder, building decoder tensor...");
            neuromesh_interfaces::msg::Tensor own_tensor, neighbor_tensor;
            
            if (buildDecoderTensor(feature_buffer_, feature_buffer_timestamp_, own_tensor, neighbor_tensor)) {
                RCLCPP_INFO(this->get_logger(), "Decoder tensor built successfully, starting decoder inference...");
                startClock("decoder_inference");
                
                // Perform decoder inference - VGGT expects single concatenated tensor [2, 1036, 1024]
                std::vector<neuromesh_interfaces::msg::Tensor> decoder_inputs = {own_tensor};
                RCLCPP_INFO(this->get_logger(), "Calling performInference with decoder model: %s", decoder_model_name_.c_str());
                RCLCPP_INFO(this->get_logger(), "Decoder input tensor size: %zu bytes", decoder_inputs[0].data.size());
                
                decoder_result_future = performInference(decoder_model_name_, decoder_inputs);
                RCLCPP_INFO(this->get_logger(), "Decoder inference started");
            } else {
                RCLCPP_WARN(this->get_logger(), "Failed to build decoder tensor");
            }
        } else {
            RCLCPP_DEBUG(this->get_logger(), "Not enough features yet (need 2, have %zu)", feature_buffer_.size());
        }
        
        // Check if decoder result is ready (non-blocking check)
        if (decoder_result_future.valid() && 
            decoder_result_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            
            RCLCPP_INFO(this->get_logger(), "Decoder result is ready, retrieving...");
            auto decoder_results = decoder_result_future.get();
            RCLCPP_INFO(this->get_logger(), "Retrieved %zu decoder results", decoder_results.size());
            
            if (decoder_results.size() >= 5) { // pose_enc, depth, depth_conf, world_points, world_points_conf
                stopClock("decoder_inference");
                RCLCPP_INFO(this->get_logger(), "Decoder inference took: %ld ms", checkClock("decoder_inference"));
                
                // Extract decoder outputs
                auto pose_enc = decoder_results[0];
                auto depth = decoder_results[1];
                auto depth_conf = decoder_results[2];
                auto world_points = decoder_results[3];
                auto world_points_conf = decoder_results[4];
                
                // Create header for outputs
                std_msgs::msg::Header header;
                header.stamp = this->get_clock()->now();
                header.frame_id = id_ + "/map";
                
                // Publish depth images
                auto depth_robot1 = createDepthImage(*depth, 0, header);
                auto depth_robot2 = createDepthImage(*depth, 1, header);
                if (depth_robot1) depth_robot1_publisher_->publish(*depth_robot1);
                if (depth_robot2) depth_robot2_publisher_->publish(*depth_robot2);
                
                // Publish pointclouds
                auto pc_current = createPointCloud(*world_points, *world_points_conf, 0, header);
                auto pc_neighbor = createPointCloud(*world_points, *world_points_conf, 1, header);
                if (pc_current) pointcloud_current_publisher_->publish(*pc_current);
                if (pc_neighbor) pointcloud_neighbor_publisher_->publish(*pc_neighbor);
                
                // Publish RGB pointclouds
                sensor_msgs::msg::Image::SharedPtr rgb_image;
                {
                    std::lock_guard<std::mutex> lock(camera_msg_mutex_);
                    rgb_image = latest_camera_msg_;
                }
                
                if (rgb_image) {
                    auto pc_current_rgb = createPointCloud(*world_points, *world_points_conf, 0, header, true, rgb_image);
                    auto pc_neighbor_rgb = createPointCloud(*world_points, *world_points_conf, 1, header, true, rgb_image);
                    if (pc_current_rgb) pointcloud_current_rgb_publisher_->publish(*pc_current_rgb);
                    if (pc_neighbor_rgb) pointcloud_neighbor_rgb_publisher_->publish(*pc_neighbor_rgb);
                }
            }
        } else if (decoder_result_future.valid()) {
            RCLCPP_DEBUG(this->get_logger(), "Decoder result not ready yet, will check again next cycle");
        }
        
        RCLCPP_INFO(this->get_logger(), "=== END process_features ===");
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in process_features: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown exception in process_features");
    }
}

bool vggtNode::buildDecoderTensor(
    std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr> buffer,
    std::map<std::string, double> buffer_timestamp,
    neuromesh_interfaces::msg::Tensor &own_tensor,
    neuromesh_interfaces::msg::Tensor &neighbour_tensor) {
    
    // Find our own feature and one neighbor feature
    auto own_it = buffer.find(id_);
    if (own_it == buffer.end()) {
        RCLCPP_WARN(this->get_logger(), "Own feature not found in buffer");
        return false;
    }
    
    // Find a neighbor feature (just take the first one that's not us)
    neuromesh_interfaces::msg::Feature::SharedPtr neighbor_feature = nullptr;
    for (const auto& [agent_id, feature] : buffer) {
        if (agent_id != id_) {
            neighbor_feature = feature;
            break;
        }
    }
    
    if (!neighbor_feature) {
        RCLCPP_WARN(this->get_logger(), "No neighbor feature found in buffer");
        return false;
    }
    
    // Build tensors from features
    own_tensor = own_it->second->tensor;
    neighbour_tensor = neighbor_feature->tensor;
    
    // Ensure correct dimensions for VGGT decoder: 2x1036x1024
    if (own_tensor.shape.dims.size() >= 3 && own_tensor.shape.dims[1] == 1036 && own_tensor.shape.dims[2] == 1024) {
        // Concatenate tensors for decoder input
        neuromesh_interfaces::msg::Tensor combined_tensor;
        combined_tensor.shape.dims = {2, 1036, 1024};
        combined_tensor.data_type = 9; // float32
        
        // Extract float data from own tensor
        std::vector<float> own_float_data(own_tensor.data.size() / sizeof(float));
        std::memcpy(own_float_data.data(), own_tensor.data.data(), own_tensor.data.size());
        
        // Extract float data from neighbor tensor  
        std::vector<float> neighbor_float_data(neighbour_tensor.data.size() / sizeof(float));
        std::memcpy(neighbor_float_data.data(), neighbour_tensor.data.data(), neighbour_tensor.data.size());
        
        // Combine float data
        std::vector<float> combined_float_data;
        combined_float_data.insert(combined_float_data.end(), own_float_data.begin(), own_float_data.end());
        combined_float_data.insert(combined_float_data.end(), neighbor_float_data.begin(), neighbor_float_data.end());
        
        // Convert back to uint8 data
        combined_tensor.data.resize(combined_float_data.size() * sizeof(float));
        std::memcpy(combined_tensor.data.data(), combined_float_data.data(), combined_float_data.size() * sizeof(float));
        
        own_tensor = combined_tensor;
        neighbour_tensor = combined_tensor; // For decoder, we pass the same combined tensor
        
        RCLCPP_DEBUG(this->get_logger(), "Built decoder tensor with dimensions [%u, %u, %u]",
                    combined_tensor.shape.dims[0], combined_tensor.shape.dims[1], combined_tensor.shape.dims[2]);
        return true;
    } else {
        RCLCPP_WARN(this->get_logger(), "Invalid feature tensor dimensions for VGGT decoder");
        return false;
    }
}

sensor_msgs::msg::Image::SharedPtr 
vggtNode::createDepthImage(const neuromesh_interfaces::msg::Tensor &depth_tensor,
                          int robot_idx, const std_msgs::msg::Header &header) {
    auto depth_image = std::make_shared<sensor_msgs::msg::Image>();
    
    // Set header
    depth_image->header = header;
    depth_image->header.frame_id = "cam1_color_optical_frame";
    
    // Set image properties for depth (392x518x1)
    depth_image->height = 392;
    depth_image->width = 518;
    depth_image->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    depth_image->is_bigendian = false;
    depth_image->step = depth_image->width * sizeof(float);
    
    // Extract depth data for specific robot (robot_idx: 0 or 1)
    // Tensor shape: 1x2x392x518x1
    if (depth_tensor.shape.dims.size() >= 5 && 
        depth_tensor.shape.dims[0] == 1 && depth_tensor.shape.dims[1] == 2 &&
        depth_tensor.shape.dims[2] == 392 && depth_tensor.shape.dims[3] == 518) {
        
        size_t depth_slice_size = 392 * 518;
        size_t start_idx = robot_idx * depth_slice_size * sizeof(float);
        size_t data_size = depth_slice_size * sizeof(float);
        
        if (start_idx + data_size <= depth_tensor.data.size()) {
            depth_image->data.resize(data_size);
            std::memcpy(depth_image->data.data(), 
                       &depth_tensor.data[start_idx], 
                       data_size);
        }
    }
    
    return depth_image;
}

sensor_msgs::msg::PointCloud2::SharedPtr 
vggtNode::createPointCloud(const neuromesh_interfaces::msg::Tensor &world_points_tensor,
                          const neuromesh_interfaces::msg::Tensor &world_points_conf_tensor,
                          int robot_idx, const std_msgs::msg::Header &header,
                          bool use_rgb, const sensor_msgs::msg::Image::SharedPtr rgb_image) {
    auto pointcloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
    
    // Set header
    pointcloud->header = header;
    
    // Tensor shape: 1x2x392x518x3 for world_points
    if (world_points_tensor.shape.dims.size() >= 5 &&
        world_points_tensor.shape.dims[0] == 1 && world_points_tensor.shape.dims[1] == 2 &&
        world_points_tensor.shape.dims[2] == 392 && world_points_tensor.shape.dims[3] == 518 &&
        world_points_tensor.shape.dims[4] == 3) {
        
        size_t height = 392;
        size_t width = 518;
        size_t points_per_robot = height * width;
        size_t start_idx = robot_idx * points_per_robot * 3; // 3 for XYZ
        
        // Set up point cloud fields
        sensor_msgs::PointCloud2Modifier modifier(*pointcloud);
        if (use_rgb && rgb_image) {
            modifier.setPointCloud2Fields(4, 
                "x", 1, sensor_msgs::msg::PointField::FLOAT32,
                "y", 1, sensor_msgs::msg::PointField::FLOAT32, 
                "z", 1, sensor_msgs::msg::PointField::FLOAT32,
                "rgb", 1, sensor_msgs::msg::PointField::UINT32);
        } else {
            modifier.setPointCloud2Fields(3,
                "x", 1, sensor_msgs::msg::PointField::FLOAT32,
                "y", 1, sensor_msgs::msg::PointField::FLOAT32,
                "z", 1, sensor_msgs::msg::PointField::FLOAT32);
        }
        
        // Count valid points (where confidence > threshold)
        size_t valid_points = 0;
        float conf_threshold = 0.5; // Confidence threshold
        
        // Extract confidence data as float array
        std::vector<float> conf_data(world_points_conf_tensor.data.size() / sizeof(float));
        std::memcpy(conf_data.data(), world_points_conf_tensor.data.data(), world_points_conf_tensor.data.size());
        
        for (size_t i = 0; i < points_per_robot; ++i) {
            size_t conf_idx = robot_idx * points_per_robot + i;
            if (conf_idx < conf_data.size() &&
                conf_data[conf_idx] > conf_threshold) {
                valid_points++;
            }
        }
        
        modifier.resize(valid_points);
        
        // Fill point cloud data
        sensor_msgs::PointCloud2Iterator<float> iter_x(*pointcloud, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(*pointcloud, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(*pointcloud, "z");
        
        std::unique_ptr<sensor_msgs::PointCloud2Iterator<uint32_t>> iter_rgb_ptr;
        if (use_rgb && rgb_image) {
            iter_rgb_ptr = std::make_unique<sensor_msgs::PointCloud2Iterator<uint32_t>>(*pointcloud, "rgb");
        }
        
        // Extract world points data as float array
        std::vector<float> points_data(world_points_tensor.data.size() / sizeof(float));
        std::memcpy(points_data.data(), world_points_tensor.data.data(), world_points_tensor.data.size());
        
        // Convert RGB image to OpenCV for color extraction
        cv::Mat rgb_cv;
        if (use_rgb && rgb_image) {
            try {
                cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(rgb_image, sensor_msgs::image_encodings::RGB8);
                cv::resize(cv_ptr->image, rgb_cv, cv::Size(518, 392)); // Resize to match point cloud resolution
            } catch (cv_bridge::Exception& e) {
                RCLCPP_WARN(this->get_logger(), "Failed to convert RGB image: %s", e.what());
                use_rgb = false;
            }
        }
        
        for (size_t i = 0; i < points_per_robot; ++i) {
            size_t conf_idx = robot_idx * points_per_robot + i;
            if (conf_idx < conf_data.size() &&
                conf_data[conf_idx] > conf_threshold) {
                
                size_t xyz_base_idx = start_idx + i * 3;
                if (xyz_base_idx + 2 < points_data.size()) {
                    *iter_x = points_data[xyz_base_idx];
                    *iter_y = points_data[xyz_base_idx + 1];
                    *iter_z = points_data[xyz_base_idx + 2];
                    
                    if (use_rgb && rgb_image && !rgb_cv.empty() && iter_rgb_ptr) {
                        // Map point index back to image coordinates
                        int img_y = i / width;
                        int img_x = i % width;
                        
                        if (img_y < rgb_cv.rows && img_x < rgb_cv.cols) {
                            cv::Vec3b color = rgb_cv.at<cv::Vec3b>(img_y, img_x);
                            uint32_t rgb_value = (color[0] << 16) | (color[1] << 8) | color[2];
                            **iter_rgb_ptr = rgb_value;
                            ++(*iter_rgb_ptr);
                        }
                    }
                    
                    ++iter_x;
                    ++iter_y;
                    ++iter_z;
                }
            }
        }
    }
    
    return pointcloud;
}

std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> 
vggtNode::performInference(const std::string &model_name, 
                          const std::vector<neuromesh_interfaces::msg::Tensor> &tensors) {
    // Placeholder implementation - in real system this would call TensorRT service
    std::promise<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> prom;
    std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> result = prom.get_future();
    std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> output_tensors(1, std::make_shared<neuromesh_interfaces::msg::Tensor>());
    output_tensors[0]->result = 1; // Cannot reach engine error code
    prom.set_value(std::move(output_tensors));
    return result;
}

neuromesh_interfaces::msg::Feature 
vggtNode::buildFeatureMessage(const neuromesh_interfaces::msg::Tensor &tensor) {
    RCLCPP_INFO(this->get_logger(), "=== buildFeatureMessage called ===");
    RCLCPP_INFO(this->get_logger(), "Input tensor dims: [%s], data_size: %zu", 
                tensor.shape.dims.empty() ? "empty" : 
                std::accumulate(tensor.shape.dims.begin(), tensor.shape.dims.end(), std::string(),
                    [](const std::string& a, uint32_t b) { return a.empty() ? std::to_string(b) : a + ", " + std::to_string(b); }).c_str(),
                tensor.data.size());
    
    neuromesh_interfaces::msg::Feature feature_msg;
    feature_msg.tensor = tensor;
    feature_msg.id = id_;
    feature_msg.timestamp = this->get_clock()->now();
    
    RCLCPP_INFO(this->get_logger(), "Built feature message for agent: %s", id_.c_str());
    return feature_msg;
}

void vggtNode::createSubscription(
    std::map<std::string, rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr> &subscription_map,
    std::string id, rclcpp::QoS qos) {
    
    auto callback = [this, id](const neuromesh_interfaces::msg::Feature::SharedPtr msg) {
        this->feature_callback(msg);
    };
    
    subscription_map[id] = this->create_subscription<neuromesh_interfaces::msg::Feature>(
        topic_prefix_ + id, qos, callback);
}

void vggtNode::removeSubscription(
    std::map<std::string, rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr> subscription_map,
    std::string id) {
    subscription_map.erase(id);
}

rmw_qos_profile_t vggtNode::parseQoSString(const std::string &str) {
    std::string profile = str;
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
                       "Unknown QoS profile: " << profile << ". Returning profile: DEFAULT");
    return rmw_qos_profile_default;
}

std::set<std::string> vggtNode::splitAgentString(std::string str) {
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

void vggtNode::run_encoder_cycle() {
    auto now = std::chrono::system_clock::now();
    auto time_since_epoch = now.time_since_epoch();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time_since_epoch);
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(time_since_epoch - seconds);
    
    RCLCPP_INFO(this->get_logger(), "=== run_encoder_cycle called at %ld.%09ld, setting fresh_encoder_cycle = true ===",
                seconds.count(), nanoseconds.count());
    fresh_encoder_cycle = true;
}

void vggtNode::broadcast_transform() {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = this->get_clock()->now();
    t.header.frame_id = "cam1_color_optical_frame";
    t.child_frame_id = id_ + "/map";
    
    t.transform.translation.x = 0.0;
    t.transform.translation.y = 0.0;
    t.transform.translation.z = 0.0;
    t.transform.rotation.x = 0.0;
    t.transform.rotation.y = 0.0;
    t.transform.rotation.z = 0.0;
    t.transform.rotation.w = 1.0;
    
    tf_broadcaster_->sendTransform(t);
}

void vggtNode::revertTensorDimensions(neuromesh_interfaces::msg::Tensor &tensor) {
    // Implementation for reverting tensor dimensions if needed
}

neuromesh_interfaces::msg::Tensor 
vggtNode::convert_to_nchw(const neuromesh_interfaces::msg::Tensor &input) {
    // Implementation for converting tensor to NCHW format
    return input; // Placeholder
}

neuromesh_interfaces::msg::Tensor 
vggtNode::tensor_ints_to_floats(neuromesh_interfaces::msg::Tensor &input) {
    // Implementation for converting ints to floats
    return input; // Placeholder
}

std::vector<uint> vggtNode::string_to_dims_single(std::string in) {
    std::stringstream stream(in);
    std::string element;
    std::vector<uint> out;
    
    while (getline(stream, element, ',')) {
        out.push_back(std::stoi(element));
    }
    return out;
}

std::vector<std::vector<uint>> vggtNode::string_to_dims(std::string in) {
    std::stringstream stream(in);
    std::string element;
    std::vector<std::vector<uint>> out;
    
    while (getline(stream, element, ';')) {
        std::vector<uint> dims = string_to_dims_single(element);
        out.push_back(dims);
    }
    return out;
}

void vggtNode::startClock(std::string phase) {
    int64_t now_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    times[phase] = {now_time, false};
}

void vggtNode::stopClock(std::string phase) {
    if (times[phase].second) {
        return; // clock already stopped
    }
    int64_t start_time = times[phase].first;
    int64_t now_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    times[phase] = {now_time - start_time, true};
}

int64_t vggtNode::checkClock(std::string phase) {
    if (times[phase].second) {
        return times[phase].first; // clock already stopped
    }
    
    // stopclock calculations without saving
    int64_t start_time = times[phase].first;
    int64_t now_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return now_time - start_time;
}

} // namespace vggtNode