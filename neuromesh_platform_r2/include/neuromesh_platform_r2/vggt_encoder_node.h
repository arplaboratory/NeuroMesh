#ifndef NEUROMESH_PLATFORM_R2__VGGT_ENCODER_NODE_H_
#define NEUROMESH_PLATFORM_R2__VGGT_ENCODER_NODE_H_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <neuromesh_interfaces/msg/feature.hpp>
#include <neuromesh_interfaces/srv/tensor_request.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <future>
#include <mutex>
#include <string>
#include <vector>

namespace neuromesh {

class VggtEncoderNode : public rclcpp::Node {
public:
    explicit VggtEncoderNode(const rclcpp::NodeOptions& options);
    ~VggtEncoderNode() = default;

private:
    // Callbacks
    void camera_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void encoder_timer_callback();
    
    // Processing functions
    std::vector<float> preprocess_image(const cv::Mat& image, cv::Mat& resized_rgb);
    void process_image();
    void publish_features(const neuromesh_interfaces::msg::Tensor& tensor);
    neuromesh_interfaces::msg::Feature build_feature_message(const neuromesh_interfaces::msg::Tensor& tensor);
    
    // QoS configuration
    rclcpp::QoS create_image_qos();
    
    // Async inference
    std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> 
        perform_inference(const std::vector<neuromesh_interfaces::msg::Tensor>& inputs);
    
    // Subscriptions
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub_;
    
    // Publishers
    rclcpp::Publisher<neuromesh_interfaces::msg::Feature>::SharedPtr feature_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr resized_rgb_pub_;
    
    // TensorRT client
    rclcpp::Client<neuromesh_interfaces::srv::TensorRequest>::SharedPtr tensorrt_client_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr encoder_timer_;
    
    // Configuration parameters
    std::string robot_name_;
    std::string color_raw_topic_;
    double encoder_cycle_interval_;  // seconds
    std::string encoder_model_path_;
    std::string encoder_model_name_;
    int image_width_;
    int image_height_;
    double tensorrt_timeout_;  // seconds
    
    // State management
    sensor_msgs::msg::Image::SharedPtr latest_image_;
    std::mutex image_mutex_;
    std::atomic<bool> processing_in_progress_;
    std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> encoder_result_;
    
    // Store resized RGB image and timestamp for synchronized publishing
    cv::Mat pending_resized_rgb_;
    rclcpp::Time pending_timestamp_;
    
    // Timing utilities
    std::map<std::string, std::chrono::high_resolution_clock::time_point> clock_map_;
    void start_clock(const std::string& name);
    void stop_clock(const std::string& name);
    long check_clock(const std::string& name);
};

}  // namespace neuromesh

#endif  // NEUROMESH_PLATFORM_R2__VGGT_ENCODER_NODE_H_