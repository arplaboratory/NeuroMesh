#include "neuromesh_platform_r2/vggt_encoder_node.h"

namespace neuromesh {

VggtEncoderNode::VggtEncoderNode(const rclcpp::NodeOptions& options)
    : Node("vggt_encoder", options),
      processing_in_progress_(false) {
    
    // Declare and get parameters
    this->declare_parameter<std::string>("robot_name", "");
    this->declare_parameter<std::string>("color_raw_topic", "");
    this->declare_parameter<double>("vggt.encoder.cycle_interval", 3.0);
    this->declare_parameter<std::string>("vggt.encoder.model_path", "");
    this->declare_parameter<int>("vggt.encoder.image_width", 518);
    this->declare_parameter<int>("vggt.encoder.image_height", 392);
    this->declare_parameter<double>("vggt.tensorrt.timeout", 30.0);
    
    // Get parameters
    robot_name_ = this->get_parameter("robot_name").as_string();
    color_raw_topic_ = this->get_parameter("color_raw_topic").as_string();
    encoder_cycle_interval_ = this->get_parameter("vggt.encoder.cycle_interval").as_double();
    encoder_model_path_ = this->get_parameter("vggt.encoder.model_path").as_string();
    image_width_ = this->get_parameter("vggt.encoder.image_width").as_int();
    image_height_ = this->get_parameter("vggt.encoder.image_height").as_int();
    tensorrt_timeout_ = this->get_parameter("vggt.tensorrt.timeout").as_double();
    
    // Extract model name from path
    size_t last_slash = encoder_model_path_.find_last_of("/");
    encoder_model_name_ = (last_slash != std::string::npos) ? 
                          encoder_model_path_.substr(last_slash + 1) : encoder_model_path_;
    
    RCLCPP_INFO(this->get_logger(), "VggtEncoderNode initialized for robot: %s", robot_name_.c_str());
    RCLCPP_INFO(this->get_logger(), "Encoder model: %s", encoder_model_name_.c_str());
    RCLCPP_INFO(this->get_logger(), "Encoder cycle interval: %.2f seconds", encoder_cycle_interval_);
    RCLCPP_INFO(this->get_logger(), "Image dimensions: %dx%d", image_width_, image_height_);
    
    // Create TensorRT client for encoder
    tensorrt_client_ = this->create_client<neuromesh_interfaces::srv::TensorRequest>("tensorrt_request_encoder");
    
    // Wait for TensorRT service
    while (!tensorrt_client_->wait_for_service(std::chrono::seconds(1))) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for TensorRT service. Exiting.");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Waiting for TensorRT service...");
    }
    
    // Create camera subscription
    auto camera_qos = rclcpp::QoS(10).reliability(rclcpp::ReliabilityPolicy::BestEffort);
    camera_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        color_raw_topic_, camera_qos,
        std::bind(&VggtEncoderNode::camera_callback, this, std::placeholders::_1));
    
    // Create feature publisher
    std::string feature_topic = "/" + robot_name_ + "/features_" + robot_name_;
    feature_pub_ = this->create_publisher<neuromesh_interfaces::msg::Feature>(
        feature_topic, rclcpp::QoS(10).reliability(rclcpp::ReliabilityPolicy::Reliable));
    
    // Create timer for encoder processing
    encoder_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(encoder_cycle_interval_),
        std::bind(&VggtEncoderNode::encoder_timer_callback, this));
    
    RCLCPP_INFO(this->get_logger(), "VggtEncoderNode setup complete. Publishing features to: %s", 
                feature_topic.c_str());
}

void VggtEncoderNode::camera_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(image_mutex_);
    latest_image_ = msg;
    RCLCPP_DEBUG(this->get_logger(), "Received camera image with timestamp: %d.%d", 
                 msg->header.stamp.sec, msg->header.stamp.nanosec);
}

void VggtEncoderNode::encoder_timer_callback() {
    // Check if previous encoding is still in progress
    if (processing_in_progress_.load()) {
        RCLCPP_WARN(this->get_logger(), "Previous encoding still in progress, skipping this cycle");
        return;
    }
    
    // Check if encoder result is ready
    if (encoder_result_.valid() && 
        encoder_result_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
        
        RCLCPP_INFO(this->get_logger(), "Encoder result ready, processing...");
        try {
            auto results = encoder_result_.get();
            if (!results.empty() && results[0]) {
                publish_features(*results[0]);
                stop_clock("encoder_inference");
                RCLCPP_INFO(this->get_logger(), "Encoder inference took: %ld ms", 
                           check_clock("encoder_inference"));
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing encoder result: %s", e.what());
        }
    }
    
    // Start new encoding cycle if we have an image
    std::shared_ptr<sensor_msgs::msg::Image> image_to_process;
    {
        std::lock_guard<std::mutex> lock(image_mutex_);
        image_to_process = latest_image_;
    }
    
    if (image_to_process) {
        RCLCPP_INFO(this->get_logger(), "Starting new encoder cycle");
        processing_in_progress_ = true;
        process_image();
    } else {
        RCLCPP_DEBUG(this->get_logger(), "No image available for encoding");
    }
}

void VggtEncoderNode::process_image() {
    try {
        // Get latest image
        std::shared_ptr<sensor_msgs::msg::Image> image;
        {
            std::lock_guard<std::mutex> lock(image_mutex_);
            image = latest_image_;
        }
        
        if (!image) {
            processing_in_progress_ = false;
            return;
        }
        
        // Convert ROS image to OpenCV
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::RGB8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            processing_in_progress_ = false;
            return;
        }
        
        // Preprocess image
        auto preprocessed = preprocess_image(cv_ptr->image);
        
        // Create tensor for encoder input
        neuromesh_interfaces::msg::Tensor input_tensor;
        input_tensor.name = "input_image";
        input_tensor.shape.dims = {1, 3, static_cast<uint32_t>(image_height_), 
                                   static_cast<uint32_t>(image_width_)};
        input_tensor.data_type = 9;  // float32
        
        // Calculate strides
        input_tensor.strides.push_back(3 * image_height_ * image_width_ * sizeof(float));
        input_tensor.strides.push_back(image_height_ * image_width_ * sizeof(float));
        input_tensor.strides.push_back(image_width_ * sizeof(float));
        input_tensor.strides.push_back(sizeof(float));
        
        // Copy preprocessed data
        input_tensor.data.resize(preprocessed.size() * sizeof(float));
        std::memcpy(input_tensor.data.data(), preprocessed.data(), input_tensor.data.size());
        
        // Start inference
        start_clock("encoder_inference");
        encoder_result_ = perform_inference({input_tensor});
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in process_image: %s", e.what());
        processing_in_progress_ = false;
    }
}

std::vector<float> VggtEncoderNode::preprocess_image(const cv::Mat& image) {
    // Resize image to encoder input size
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(image_width_, image_height_), 0, 0, cv::INTER_LINEAR);
    
    // Convert to float and normalize to [0, 1]
    cv::Mat normalized;
    resized.convertTo(normalized, CV_32FC3, 1.0/255.0, 0.0);

    // Convert HWC to CHW format
    std::vector<float> preprocessed(3 * image_height_ * image_width_);
    std::vector<cv::Mat> channels(3);
    cv::split(normalized, channels);
    
    // Copy each channel
    for (int c = 0; c < 3; ++c) {
        std::memcpy(&preprocessed[c * image_height_ * image_width_], 
                   channels[c].data, 
                   image_height_ * image_width_ * sizeof(float));
    }
    
    return preprocessed;
}

std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> 
VggtEncoderNode::perform_inference(const std::vector<neuromesh_interfaces::msg::Tensor>& inputs) {
    
    auto request = std::make_shared<neuromesh_interfaces::srv::TensorRequest::Request>();
    request->model_name = encoder_model_name_;
    request->tensor1 = inputs;
    
    // Create promise and future
    auto promise = std::make_shared<std::promise<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>>>();
    auto future = promise->get_future();
    
    // Send async request
    auto response_future = tensorrt_client_->async_send_request(request);
    
    // Handle response in a separate thread to avoid blocking
    std::thread([this, promise](auto response_future) {
        try {
            // Wait for response with timeout
            auto status = response_future.wait_for(std::chrono::duration<double>(tensorrt_timeout_));
            
            if (status == std::future_status::ready) {
                auto response = response_future.get();
                
                if (response) {
                    std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>> result;
                    for (const auto& tensor : response->tensor2) {
                        result.push_back(std::make_shared<neuromesh_interfaces::msg::Tensor>(tensor));
                    }
                    promise->set_value(result);
                } else {
                    RCLCPP_ERROR(this->get_logger(), "TensorRT inference failed - null response");
                    promise->set_value({});
                }
            } else {
                RCLCPP_ERROR(this->get_logger(), "TensorRT request timed out");
                promise->set_value({});
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in TensorRT request: %s", e.what());
            promise->set_value({});
        }
        
        processing_in_progress_ = false;
    }, std::move(response_future)).detach();
    
    return future;
}

void VggtEncoderNode::publish_features(const neuromesh_interfaces::msg::Tensor& tensor) {
    auto feature_msg = build_feature_message(tensor);
    feature_pub_->publish(feature_msg);
    
    RCLCPP_INFO(this->get_logger(), "Published features for robot: %s, tensor shape: [%s], size: %zu bytes",
                robot_name_.c_str(),
                tensor.shape.dims.empty() ? "empty" : 
                std::accumulate(tensor.shape.dims.begin(), tensor.shape.dims.end(), std::string(),
                    [](const std::string& a, uint32_t b) { 
                        return a.empty() ? std::to_string(b) : a + ", " + std::to_string(b); 
                    }).c_str(),
                tensor.data.size());
}

neuromesh_interfaces::msg::Feature VggtEncoderNode::build_feature_message(
    const neuromesh_interfaces::msg::Tensor& tensor) {
    
    neuromesh_interfaces::msg::Feature feature;
    feature.id = robot_name_;
    feature.timestamp = this->now();
    feature.tensor = tensor;
    
    return feature;
}

void VggtEncoderNode::start_clock(const std::string& name) {
    clock_map_[name] = std::chrono::high_resolution_clock::now();
}

void VggtEncoderNode::stop_clock(const std::string& name) {
    if (clock_map_.find(name) != clock_map_.end()) {
        auto duration = std::chrono::high_resolution_clock::now() - clock_map_[name];
        RCLCPP_DEBUG(this->get_logger(), "%s took %ld ms", name.c_str(),
                    std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
    }
}

long VggtEncoderNode::check_clock(const std::string& name) {
    if (clock_map_.find(name) != clock_map_.end()) {
        auto duration = std::chrono::high_resolution_clock::now() - clock_map_[name];
        return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    }
    return -1;
}

}  // namespace neuromesh

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(neuromesh::VggtEncoderNode)