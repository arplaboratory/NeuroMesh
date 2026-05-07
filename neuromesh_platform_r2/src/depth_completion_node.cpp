/*
  Current Input Topics

  All topics are prefixed with /{robot_name}/:

  1. {robot_name}/depth_robot1 (sensor_msgs/Image)
    - Normalized depth from VGGT decoder (518x392 resolution)
    - TYPE_32FC1 encoding
  2. {robot_name}/decoder_sync_depth (sensor_msgs/Image)
    - Metric depth from sensor synchronized with decoder output
    - Original camera resolution
    - TYPE_32FC1 encoding
  3. {robot_name}/pointcloud_rgb (sensor_msgs/PointCloud2)
    - RGB pointcloud from VGGT decoder (current robot)
    - Contains XYZ coordinates and RGB color
  4. {robot_name}/pointcloud_current (sensor_msgs/PointCloud2)
    - Subsampled pointcloud from VGGT decoder (current robot)
    - Contains only XYZ coordinates (no color)
  5. {robot_name}/pointcloud_neighbor (sensor_msgs/PointCloud2)
    - Pointcloud from neighbor robot (no color)
    - Contains only XYZ coordinates
  6. {robot_name}/camera_info (sensor_msgs/CameraInfo)
    - Camera intrinsics for the depth sensor
    - Used to convert depth images to pointclouds

  Current Output Topics

  All topics are prefixed with /{robot_name}/:

  1. {robot_name}/pointcloud_current_rgb_scaled (sensor_msgs/PointCloud2)
    - Main output: Scaled RGB pointcloud with metric scale
    - Same structure as input but with corrected scale
  2. {robot_name}/pointcloud_current_scaled (sensor_msgs/PointCloud2)
    - Scaled subsampled pointcloud with metric scale
    - No color information
  3. {robot_name}/pointcloud_neighbor_scaled (sensor_msgs/PointCloud2)
    - Scaled neighbor pointcloud with metric scale
    - No color information

  Processing Pipeline

  1. Synchronization: Wait for all 5 input messages with matching timestamps
  2. Scale Recovery:
    - Resize decoder_sync_depth to match depth_robot1 resolution if needed
    - Compute scale ratios for each valid pixel pair
    - Create histogram of ratios and find mode
    - Refine scale estimate using median of peak values
  3. Apply Scale:
    - Scale RGB pointcloud
    - Scale current pointcloud (subsampled)
    - Scale neighbor pointcloud
  4. Publish: Output all scaled pointclouds

  The node essentially recovers the metric scale relationship between VGGT's normalized output and real-world measurements, then applies this scale to make all outputs metrically accurate.
*/

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

class DepthCompletionNode : public rclcpp::Node {
public:
    DepthCompletionNode() : Node("depth_completion_node") {
        // Declare parameters
        this->declare_parameter<std::string>("robot_name", "");
        this->declare_parameter<std::string>("depth_robot1_topic", "/depth_robot1");
        this->declare_parameter<std::string>("decoder_sync_depth_topic", "/decoder_sync_depth");
        this->declare_parameter<std::string>("pointcloud_rgb_topic", "/pointcloud_rgb");
        this->declare_parameter<std::string>("pointcloud_current_topic", "/pointcloud_current");
        this->declare_parameter<std::string>("pointcloud_neighbor_topic", "/pointcloud_neighbor");
        this->declare_parameter<std::string>("output_pointcloud_topic", "/pointcloud_current_rgb_scaled");
        this->declare_parameter<std::string>("camera_info_topic", "/camera_info");
        this->declare_parameter<double>("depth_epsilon", 0.001);
        this->declare_parameter<double>("min_scale", 0.1);
        this->declare_parameter<double>("max_scale", 10.0);
        this->declare_parameter<double>("min_shift", -5.0);
        this->declare_parameter<double>("max_shift", 5.0);
        this->declare_parameter<int>("min_valid_points", 100);
        this->declare_parameter<double>("max_time_diff", 0.1);
        this->declare_parameter<double>("voxel_leaf_size", 0.02);
        this->declare_parameter<bool>("enable_subsampling", false);

        // Get parameters
        robot_name_ = this->get_parameter("robot_name").as_string();
        voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();
        enable_subsampling_ = this->get_parameter("enable_subsampling").as_bool();
        auto depth_robot1_topic = "/" + robot_name_ + this->get_parameter("depth_robot1_topic").as_string();
        auto decoder_sync_depth_topic = "/" + robot_name_ + this->get_parameter("decoder_sync_depth_topic").as_string();
        auto pointcloud_rgb_topic = "/" + robot_name_ + this->get_parameter("pointcloud_rgb_topic").as_string();
        auto pointcloud_current_topic = "/" + robot_name_ + this->get_parameter("pointcloud_current_topic").as_string();
        auto pointcloud_neighbor_topic = "/" + robot_name_ + this->get_parameter("pointcloud_neighbor_topic").as_string();
        auto output_topic = "/" + robot_name_ + this->get_parameter("output_pointcloud_topic").as_string();
        auto camera_info_topic = "/" + robot_name_ + this->get_parameter("camera_info_topic").as_string();

        // Publishers
        scaled_pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            output_topic, 10);
        
        // Publisher for scaled neighbor pointcloud
        scaled_pointcloud_neighbor_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/" + robot_name_ + "/pointcloud_neighbor_scaled", 10);
        
        // Publisher for scaled current pointcloud (subsampled, no color)
        scaled_pointcloud_current_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/" + robot_name_ + "/pointcloud_current_scaled", 10);

        // Subscribers using message filters for synchronization
        depth_robot1_sub_.subscribe(this, depth_robot1_topic);
        decoder_sync_depth_sub_.subscribe(this, decoder_sync_depth_topic);
        pointcloud_rgb_sub_.subscribe(this, pointcloud_rgb_topic);
        pointcloud_current_sub_.subscribe(this, pointcloud_current_topic);
        pointcloud_neighbor_sub_.subscribe(this, pointcloud_neighbor_topic);

        // Synchronizer for all five topics
        sync_ = std::make_shared<Synchronizer>(SyncPolicy(10), depth_robot1_sub_, decoder_sync_depth_sub_, 
            pointcloud_rgb_sub_, pointcloud_current_sub_, pointcloud_neighbor_sub_);
        sync_->registerCallback(std::bind(&DepthCompletionNode::syncCallback, this,
            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, 
            std::placeholders::_4, std::placeholders::_5));

        // Subscribe to camera info topic
        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_info_topic, 10,
            std::bind(&DepthCompletionNode::cameraInfoCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Depth Completion Node initialized for robot: %s", robot_name_.c_str());
        RCLCPP_INFO(this->get_logger(), "Subscribing to:");
        RCLCPP_INFO(this->get_logger(), "  - %s", depth_robot1_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "  - %s", decoder_sync_depth_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "  - %s", pointcloud_rgb_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "  - %s", pointcloud_current_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "  - %s", pointcloud_neighbor_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "  - %s", camera_info_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Publishing to: %s", output_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Subsampling: %s (leaf size: %.3f m)", 
                    enable_subsampling_ ? "enabled" : "disabled", voxel_leaf_size_);
    }

private:
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::PointCloud2, 
        sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2>;
    using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scaled_pointcloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scaled_pointcloud_neighbor_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scaled_pointcloud_current_pub_;

    // Subscribers
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_robot1_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> decoder_sync_depth_sub_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> pointcloud_rgb_sub_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> pointcloud_current_sub_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> pointcloud_neighbor_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    
    // Synchronizer
    std::shared_ptr<Synchronizer> sync_;

    // Robot name
    std::string robot_name_;
    
    // Voxel leaf size for pointcloud subsampling
    double voxel_leaf_size_;
    bool enable_subsampling_;

    // Camera intrinsics
    bool camera_info_received_ = false;
    double fx_ = 0.0, fy_ = 0.0, cx_ = 0.0, cy_ = 0.0;
    int original_width_ = 0, original_height_ = 0;
    
    // Scaled intrinsics for depth_robot1 (518x392)
    double fx_scaled_ = 0.0, fy_scaled_ = 0.0, cx_scaled_ = 0.0, cy_scaled_ = 0.0;
    const int VGGT_WIDTH = 518;
    const int VGGT_HEIGHT = 392;

    void syncCallback(const sensor_msgs::msg::Image::ConstSharedPtr& depth_robot1_msg,
                      const sensor_msgs::msg::Image::ConstSharedPtr& decoder_sync_depth_msg,
                      const sensor_msgs::msg::PointCloud2::ConstSharedPtr& pointcloud_rgb_msg,
                      const sensor_msgs::msg::PointCloud2::ConstSharedPtr& pointcloud_current_msg,
                      const sensor_msgs::msg::PointCloud2::ConstSharedPtr& pointcloud_neighbor_msg) {
        
        try {
            // Convert depth images to OpenCV
            cv_bridge::CvImagePtr cv_depth_robot1 = cv_bridge::toCvCopy(depth_robot1_msg, sensor_msgs::image_encodings::TYPE_32FC1);
            cv_bridge::CvImagePtr cv_decoder_sync_depth = cv_bridge::toCvCopy(decoder_sync_depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);

            // Recover scale by comparing depth_robot1 with decoder_sync_depth using histogram alignment
            float scale;
            bool alignment_success = recoverScaleHistogram(cv_depth_robot1->image, cv_decoder_sync_depth->image, scale);
            
            if (!alignment_success) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "Failed to recover scale, using default scale=1.0");
                scale = 1.0f;
            }
            
            // Apply scale to the RGB pointcloud
            auto scaled_rgb_pointcloud = scalePointcloud(pointcloud_rgb_msg, scale);
            if (enable_subsampling_) {
                scaled_rgb_pointcloud = subsamplePointcloudRGB(scaled_rgb_pointcloud);
            }
            scaled_pointcloud_pub_->publish(*scaled_rgb_pointcloud);
            
            // Apply scale to the current pointcloud (subsampled, no color)
            auto scaled_current_pointcloud = scalePointcloudNoColor(pointcloud_current_msg, scale);
            if (enable_subsampling_) {
                scaled_current_pointcloud = subsamplePointcloud(scaled_current_pointcloud);
            }
            scaled_pointcloud_current_pub_->publish(*scaled_current_pointcloud);
            
            // Apply scale to the neighbor pointcloud (no color)
            auto scaled_neighbor_pointcloud = scalePointcloudNoColor(pointcloud_neighbor_msg, scale);
            if (enable_subsampling_) {
                scaled_neighbor_pointcloud = subsamplePointcloud(scaled_neighbor_pointcloud);
            }
            scaled_pointcloud_neighbor_pub_->publish(*scaled_neighbor_pointcloud);

            RCLCPP_DEBUG(this->get_logger(), "Applied scale=%.3f to all pointclouds%s", 
                         scale, enable_subsampling_ ? " with subsampling" : "");

        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in syncCallback: %s", e.what());
        }
    }

    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        if (!camera_info_received_) {
            fx_ = msg->k[0];  // K[0,0]
            fy_ = msg->k[4];  // K[1,1]
            cx_ = msg->k[2];  // K[0,2]
            cy_ = msg->k[5];  // K[1,2]
            original_width_ = msg->width;
            original_height_ = msg->height;
            
            // Calculate scaled intrinsics for VGGT resolution (518x392)
            double scale_x = static_cast<double>(VGGT_WIDTH) / original_width_;
            double scale_y = static_cast<double>(VGGT_HEIGHT) / original_height_;
            
            fx_scaled_ = fx_ * scale_x;
            fy_scaled_ = fy_ * scale_y;
            cx_scaled_ = cx_ * scale_x;
            cy_scaled_ = cy_ * scale_y;
            
            camera_info_received_ = true;
            RCLCPP_INFO(this->get_logger(), "Camera intrinsics received for %dx%d: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f", 
                        original_width_, original_height_, fx_, fy_, cx_, cy_);
            RCLCPP_INFO(this->get_logger(), "Scaled intrinsics for %dx%d: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f", 
                        VGGT_WIDTH, VGGT_HEIGHT, fx_scaled_, fy_scaled_, cx_scaled_, cy_scaled_);
        }
    }

    bool recoverScaleHistogram(const cv::Mat& depth_robot1, const cv::Mat& decoder_sync_depth, float& scale) {
        auto epsilon = this->get_parameter("depth_epsilon").as_double();
        auto min_valid_points = this->get_parameter("min_valid_points").as_int();
        auto min_scale = this->get_parameter("min_scale").as_double();
        auto max_scale = this->get_parameter("max_scale").as_double();

        // Resize decoder_sync_depth to match depth_robot1 dimensions if needed
        cv::Mat decoder_sync_depth_resized;
        if (decoder_sync_depth.size() != depth_robot1.size()) {
            cv::resize(decoder_sync_depth, decoder_sync_depth_resized, depth_robot1.size(), 0, 0, cv::INTER_LINEAR);
        } else {
            decoder_sync_depth_resized = decoder_sync_depth;
        }

        // Collect scale ratios from valid pixel pairs
        std::vector<float> scale_ratios;
        
        for (int y = 0; y < depth_robot1.rows; y++) {
            for (int x = 0; x < depth_robot1.cols; x++) {
                float d1 = depth_robot1.at<float>(y, x);
                float d2 = decoder_sync_depth_resized.at<float>(y, x);
                
                // Check if both depths are valid
                if (d1 > epsilon && d2 > epsilon && !std::isnan(d1) && !std::isnan(d2) && 
                    std::isfinite(d1) && std::isfinite(d2)) {
                    float ratio = d2 / d1;  // decoder_sync_depth = scale * depth_robot1
                    
                    // Only keep ratios within reasonable bounds
                    if (ratio >= min_scale && ratio <= max_scale) {
                        scale_ratios.push_back(ratio);
                    }
                }
            }
        }

        if (static_cast<int>(scale_ratios.size()) < min_valid_points) {
            RCLCPP_WARN(this->get_logger(), "Insufficient valid points for scale recovery: %zu < %ld", 
                       scale_ratios.size(), min_valid_points);
            return false;
        }

        // Create histogram of scale ratios
        const int num_bins = 100;
        float bin_width = (max_scale - min_scale) / num_bins;
        std::vector<int> histogram(num_bins, 0);
        
        // Fill histogram
        for (float ratio : scale_ratios) {
            int bin_idx = static_cast<int>((ratio - min_scale) / bin_width);
            bin_idx = std::max(0, std::min(num_bins - 1, bin_idx));
            histogram[bin_idx]++;
        }
        
        // Find the bin with maximum count (mode)
        int max_bin_idx = 0;
        int max_count = 0;
        for (int i = 0; i < num_bins; i++) {
            if (histogram[i] > max_count) {
                max_count = histogram[i];
                max_bin_idx = i;
            }
        }
        
        // Refine scale estimate using values in the peak bin and neighboring bins
        std::vector<float> peak_values;
        float peak_min = min_scale + (max_bin_idx - 1) * bin_width;
        float peak_max = min_scale + (max_bin_idx + 2) * bin_width;
        
        for (float ratio : scale_ratios) {
            if (ratio >= peak_min && ratio <= peak_max) {
                peak_values.push_back(ratio);
            }
        }
        
        // Calculate median of peak values for robust scale estimate
        if (peak_values.empty()) {
            scale = min_scale + (max_bin_idx + 0.5f) * bin_width;
        } else {
            std::sort(peak_values.begin(), peak_values.end());
            scale = peak_values[peak_values.size() / 2];
        }
        
        // Calculate confidence metric (percentage of values in peak)
        float confidence = static_cast<float>(peak_values.size()) / scale_ratios.size() * 100.0f;
        
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "Histogram scale recovery: scale=%.3f, confidence=%.1f%%, valid_points=%zu, peak_points=%zu", 
            scale, confidence, scale_ratios.size(), peak_values.size());
        
        return true;
    }

    sensor_msgs::msg::PointCloud2::SharedPtr scalePointcloud(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& input_pc, float scale) {
        
        // Convert ROS pointcloud to PCL
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*input_pc, *input_cloud);
        
        // Create output pointcloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        output_cloud->header = input_cloud->header;
        output_cloud->width = input_cloud->width;
        output_cloud->height = input_cloud->height;
        output_cloud->is_dense = input_cloud->is_dense;
        output_cloud->points.reserve(input_cloud->points.size());
        
        // Apply scale to each point
        for (const auto& point : input_cloud->points) {
            pcl::PointXYZRGB scaled_point;
            
            // Scale all coordinates uniformly
            scaled_point.x = point.x * scale;
            scaled_point.y = point.y * scale;
            scaled_point.z = point.z * scale;
            
            // Preserve color information
            scaled_point.r = point.r;
            scaled_point.g = point.g;
            scaled_point.b = point.b;
            
            output_cloud->points.push_back(scaled_point);
        }
        
        // Convert back to ROS message
        auto output_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(*output_cloud, *output_msg);
        output_msg->header = input_pc->header;
        
        RCLCPP_DEBUG(this->get_logger(), "Scaled RGB pointcloud by factor %.3f", scale);
        
        return output_msg;
    }
    
    sensor_msgs::msg::PointCloud2::SharedPtr scalePointcloudNoColor(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& input_pc, float scale) {
        
        // Convert ROS pointcloud to PCL (no color)
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*input_pc, *input_cloud);
        
        // Create output pointcloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        output_cloud->header = input_cloud->header;
        output_cloud->width = input_cloud->width;
        output_cloud->height = input_cloud->height;
        output_cloud->is_dense = input_cloud->is_dense;
        output_cloud->points.reserve(input_cloud->points.size());
        
        // Apply scale to each point
        for (const auto& point : input_cloud->points) {
            pcl::PointXYZ scaled_point;
            
            // Scale all coordinates uniformly
            scaled_point.x = point.x * scale;
            scaled_point.y = point.y * scale;
            scaled_point.z = point.z * scale;
            
            output_cloud->points.push_back(scaled_point);
        }
        
        // Convert back to ROS message
        auto output_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(*output_cloud, *output_msg);
        output_msg->header = input_pc->header;
        
        RCLCPP_DEBUG(this->get_logger(), "Scaled neighbor pointcloud by factor %.3f", scale);
        
        return output_msg;
    }
    
    sensor_msgs::msg::PointCloud2::SharedPtr subsamplePointcloud(
        const sensor_msgs::msg::PointCloud2::SharedPtr& input_pc) {
        
        // Convert ROS pointcloud to PCL (no color)
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*input_pc, *input_cloud);
        
        // Create voxel grid filter
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(input_cloud);
        voxel_filter.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
        
        // Apply filter
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        voxel_filter.filter(*filtered_cloud);
        
        // Convert back to ROS message
        auto output_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(*filtered_cloud, *output_msg);
        output_msg->header = input_pc->header;
        
        RCLCPP_DEBUG(this->get_logger(), "Subsampled pointcloud from %zu to %zu points (leaf size: %.3f)", 
                     input_cloud->points.size(), filtered_cloud->points.size(), voxel_leaf_size_);
        
        return output_msg;
    }
    
    sensor_msgs::msg::PointCloud2::SharedPtr subsamplePointcloudRGB(
        const sensor_msgs::msg::PointCloud2::SharedPtr& input_pc) {
        
        // Convert ROS pointcloud to PCL with RGB
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*input_pc, *input_cloud);
        
        // Create voxel grid filter that preserves color
        pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
        voxel_filter.setInputCloud(input_cloud);
        voxel_filter.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
        
        // Apply filter
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        voxel_filter.filter(*filtered_cloud);
        
        // Convert back to ROS message
        auto output_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(*filtered_cloud, *output_msg);
        output_msg->header = input_pc->header;
        
        RCLCPP_DEBUG(this->get_logger(), "Subsampled RGB pointcloud from %zu to %zu points (leaf size: %.3f)", 
                     input_cloud->points.size(), filtered_cloud->points.size(), voxel_leaf_size_);
        
        return output_msg;
    }

    sensor_msgs::msg::PointCloud2::SharedPtr depthImageToPointCloud(
        const cv::Mat& depth_image, const std::string& frame_id, bool use_scaled_intrinsics) {
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        
        if (!camera_info_received_) {
            RCLCPP_WARN(this->get_logger(), "Camera info not received, cannot create pointcloud from depth");
            auto empty_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
            empty_msg->header.stamp = this->now();
            empty_msg->header.frame_id = frame_id;
            return empty_msg;
        }
        
        // Select appropriate intrinsics based on image size
        double fx, fy, cx, cy;
        if (use_scaled_intrinsics || (depth_image.cols == VGGT_WIDTH && depth_image.rows == VGGT_HEIGHT)) {
            // Use scaled intrinsics for VGGT resolution (518x392)
            fx = fx_scaled_;
            fy = fy_scaled_;
            cx = cx_scaled_;
            cy = cy_scaled_;
            RCLCPP_DEBUG(this->get_logger(), "Using scaled intrinsics for %dx%d image", depth_image.cols, depth_image.rows);
        } else {
            // Use original intrinsics
            fx = fx_;
            fy = fy_;
            cx = cx_;
            cy = cy_;
            RCLCPP_DEBUG(this->get_logger(), "Using original intrinsics for %dx%d image", depth_image.cols, depth_image.rows);
        }
        
        // Convert depth image to pointcloud using appropriate intrinsics
        for (int v = 0; v < depth_image.rows; v++) {
            for (int u = 0; u < depth_image.cols; u++) {
                float depth = depth_image.at<float>(v, u);
                
                if (depth > 0 && !std::isnan(depth) && std::isfinite(depth)) {
                    pcl::PointXYZ point;
                    
                    // Convert pixel coordinates to 3D coordinates
                    point.z = depth;
                    point.x = (u - cx) * depth / fx;
                    point.y = (v - cy) * depth / fy;
                    
                    cloud->points.push_back(point);
                }
            }
        }
        
        // Convert to ROS message
        auto output_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(*cloud, *output_msg);
        output_msg->header.stamp = this->now();
        output_msg->header.frame_id = frame_id;
        
        RCLCPP_DEBUG(this->get_logger(), "Created pointcloud with %zu points from %dx%d depth image", 
                     cloud->points.size(), depth_image.cols, depth_image.rows);
        
        return output_msg;
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DepthCompletionNode>());
    rclcpp::shutdown();
    return 0;
}