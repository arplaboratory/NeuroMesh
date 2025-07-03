#ifndef NEUROMESH_PLATFORM_R2__VGGT_DECODER_NODE_H_
#define NEUROMESH_PLATFORM_R2__VGGT_DECODER_NODE_H_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <neuromesh_interfaces/msg/feature.hpp>
#include <neuromesh_interfaces/srv/tensor_request.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <future>
#include <mutex>
#include <string>
#include <vector>
#include <map>
#include <deque>

namespace neuromesh {

class VggtDecoderNode : public rclcpp::Node {
public:
    explicit VggtDecoderNode(const rclcpp::NodeOptions& options);
    ~VggtDecoderNode() = default;

private:
    // Callbacks
    void feature_callback(const neuromesh_interfaces::msg::Feature::SharedPtr msg);
    void rgb_image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void decoder_timer_callback();
    
    // Processing functions
    std::vector<float> aggregate_features();
    bool check_feature_freshness(const std::string& robot_name, double& age_seconds);
    void process_decoder_output(const std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>& outputs);
    bool build_decoder_tensor(neuromesh_interfaces::msg::Tensor& output_tensor);
    bool find_closest_rgb_image(const rclcpp::Time& target_time, cv::Mat& rgb_image);
    
    // Output generation
    sensor_msgs::msg::Image create_depth_image(const float* depth_data, 
                                              const float* confidence_data,
                                              int width, int height);
    sensor_msgs::msg::PointCloud2 create_point_cloud(const float* world_points,
                                                     const float* confidence_data,
                                                     int width, int height,
                                                     const std::string& frame_id);
    sensor_msgs::msg::PointCloud2 create_rgb_point_cloud(const float* world_points,
                                                         const float* confidence_data,
                                                         const cv::Mat& rgb_image,
                                                         int width, int height,
                                                         const std::string& frame_id);
    
    // Async inference
    std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> 
        perform_inference(const std::vector<neuromesh_interfaces::msg::Tensor>& inputs);
    
    // Subscriptions
    std::map<std::string, rclcpp::Subscription<neuromesh_interfaces::msg::Feature>::SharedPtr> feature_subs_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_image_sub_;
    
    // Publishers
    std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> depth_pubs_;
    std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> pointcloud_pubs_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr rgb_pointcloud_pub_;
    
    // TensorRT client
    rclcpp::Client<neuromesh_interfaces::srv::TensorRequest>::SharedPtr tensorrt_client_;
    
    // Feature buffer
    std::map<std::string, neuromesh_interfaces::msg::Feature::SharedPtr> feature_buffer_;
    std::map<std::string, rclcpp::Time> feature_timestamps_;
    std::mutex feature_mutex_;
    
    // RGB image buffer for point cloud coloring
    struct RgbImageData {
        cv::Mat image;
        rclcpp::Time timestamp;
    };
    std::deque<RgbImageData> rgb_image_buffer_;
    std::mutex rgb_mutex_;
    static constexpr size_t MAX_RGB_BUFFER_SIZE = 5;
    
    // Configuration parameters
    std::string robot_name_;
    std::string frame_id_;
    double decoder_cycle_interval_;  // seconds
    double feature_age_threshold_;   // seconds (default: 10.0)
    std::vector<std::string> robot_names_;
    int num_robots_;
    std::string decoder_model_path_;
    std::string decoder_model_name_;
    double tensorrt_timeout_;  // seconds
    int image_width_;
    int image_height_;
    
    // Output dimensions
    int depth_width_;
    int depth_height_;
    
    // Timer for processing
    rclcpp::TimerBase::SharedPtr decoder_timer_;
    
    // State management
    std::atomic<bool> processing_in_progress_;
    std::future<std::vector<std::shared_ptr<neuromesh_interfaces::msg::Tensor>>> decoder_result_;
    
    // Timing utilities
    std::map<std::string, std::chrono::high_resolution_clock::time_point> clock_map_;
    void start_clock(const std::string& name);
    void stop_clock(const std::string& name);
    long check_clock(const std::string& name);
};

}  // namespace neuromesh

#endif  // NEUROMESH_PLATFORM_R2__VGGT_DECODER_NODE_H_