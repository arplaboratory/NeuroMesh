#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <realsense2_camera_msgs/msg/rgbd.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

class DepthCompletionNode : public rclcpp::Node {
public:
    DepthCompletionNode() : Node("depth_completion_node") {
        // Declare parameters
        this->declare_parameter<std::string>("rgbd_topic", "/camera/rgbd");
        this->declare_parameter<std::string>("vggt_pointcloud_topic", "/vggt/pointcloud");
        this->declare_parameter<std::string>("camera_info_topic", "/camera/depth/camera_info");
        this->declare_parameter<std::string>("output_pointcloud_topic", "/depth_completion/pointcloud");
        this->declare_parameter<double>("depth_epsilon", 0.001);
        this->declare_parameter<int>("speckle_window_size", 100);
        this->declare_parameter<int>("speckle_range", 4);
        this->declare_parameter<int>("engine_width", 392);
        this->declare_parameter<int>("engine_height", 518);

        // Get parameters
        auto rgbd_topic = this->get_parameter("rgbd_topic").as_string();
        auto vggt_pc_topic = this->get_parameter("vggt_pointcloud_topic").as_string();
        auto camera_info_topic = this->get_parameter("camera_info_topic").as_string();
        auto output_topic = this->get_parameter("output_pointcloud_topic").as_string();

        // Publishers
        pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            output_topic, 10);

        // Subscribers using message filters for synchronization
        rgbd_sub_.subscribe(this, rgbd_topic);
        vggt_pc_sub_.subscribe(this, vggt_pc_topic);
        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_info_topic, 10,
            std::bind(&DepthCompletionNode::cameraInfoCallback, this, std::placeholders::_1));

        // Synchronizer
        sync_ = std::make_shared<Synchronizer>(SyncPolicy(10), rgbd_sub_, vggt_pc_sub_);
        sync_->registerCallback(std::bind(&DepthCompletionNode::syncCallback, this,
            std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Depth Completion Node initialized");
    }

private:
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        realsense2_camera_msgs::msg::RGBD, sensor_msgs::msg::PointCloud2>;
    using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;

    // Subscribers
    message_filters::Subscriber<realsense2_camera_msgs::msg::RGBD> rgbd_sub_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> vggt_pc_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    
    // Synchronizer
    std::shared_ptr<Synchronizer> sync_;

    // Camera intrinsics
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    cv::Mat scaled_camera_matrix_;
    bool camera_info_received_ = false;
    int original_width_ = 0;
    int original_height_ = 0;

    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        if (!camera_info_received_) {
            // Store original image dimensions
            original_width_ = msg->width;
            original_height_ = msg->height;
            
            // Extract camera matrix
            camera_matrix_ = cv::Mat(3, 3, CV_64F);
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    camera_matrix_.at<double>(i, j) = msg->k[i * 3 + j];
                }
            }

            // Extract distortion coefficients
            dist_coeffs_ = cv::Mat(msg->d.size(), 1, CV_64F);
            for (size_t i = 0; i < msg->d.size(); i++) {
                dist_coeffs_.at<double>(i, 0) = msg->d[i];
            }

            camera_info_received_ = true;
            RCLCPP_INFO(this->get_logger(), "Camera intrinsics received for %dx%d image", 
                        original_width_, original_height_);
        }
    }

    void syncCallback(const realsense2_camera_msgs::msg::RGBD::ConstSharedPtr& rgbd_msg,
                      const sensor_msgs::msg::PointCloud2::ConstSharedPtr& vggt_pc_msg) {
        
        if (!camera_info_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Camera info not received yet, skipping depth completion");
            return;
        }

        try {
            // Convert ROS images to OpenCV
            cv_bridge::CvImagePtr cv_rgb = cv_bridge::toCvCopy(rgbd_msg->rgb, "bgr8");
            cv_bridge::CvImagePtr cv_depth = cv_bridge::toCvCopy(rgbd_msg->depth, "16UC1");

            // Process depth completion
            cv::Mat completed_depth = processDepthCompletion(
                cv_rgb->image, cv_depth->image, vggt_pc_msg);

            // Convert to point cloud and publish
            // The completed_depth is at engine resolution, so resize RGB to match
            auto engine_width = this->get_parameter("engine_width").as_int();
            auto engine_height = this->get_parameter("engine_height").as_int();
            cv::Mat rgb_resized;
            cv::resize(cv_rgb->image, rgb_resized, cv::Size(engine_width, engine_height), 0, 0, cv::INTER_LINEAR);
            
            auto output_pc = createPointCloud(rgb_resized, completed_depth, rgbd_msg->header);
            pointcloud_pub_->publish(*output_pc);

        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    cv::Mat processDepthCompletion(const cv::Mat& rgb, const cv::Mat& depth,
                                   const sensor_msgs::msg::PointCloud2::ConstSharedPtr& vggt_pc) {
        // 1. Resize images to engine dimensions (neural network output size)
        auto engine_width = this->get_parameter("engine_width").as_int();
        auto engine_height = this->get_parameter("engine_height").as_int();
        
        cv::Mat rgb_resized, depth_resized;
        cv::resize(rgb, rgb_resized, cv::Size(engine_width, engine_height), 0, 0, cv::INTER_LINEAR);
        cv::resize(depth, depth_resized, cv::Size(engine_width, engine_height), 0, 0, cv::INTER_NEAREST);
        
        // 2. Update camera intrinsics for the resized image
        double scale_x = static_cast<double>(engine_width) / rgb.cols;
        double scale_y = static_cast<double>(engine_height) / rgb.rows;
        scaled_camera_matrix_ = camera_matrix_.clone();
        scaled_camera_matrix_.at<double>(0, 0) *= scale_x;  // fx
        scaled_camera_matrix_.at<double>(1, 1) *= scale_y;  // fy
        scaled_camera_matrix_.at<double>(0, 2) *= scale_x;  // cx
        scaled_camera_matrix_.at<double>(1, 2) *= scale_y;  // cy
        
        // 3. Undistort images
        cv::Mat rgb_undistorted, depth_undistorted;
        cv::undistort(rgb_resized, rgb_undistorted, scaled_camera_matrix_, dist_coeffs_);
        cv::undistort(depth_resized, depth_undistorted, scaled_camera_matrix_, dist_coeffs_);

        // 4. Filter speckle noise
        cv::Mat depth_filtered;
        filterSpeckleNoise(depth_undistorted, depth_filtered);

        // 5. Project VGGT point cloud to depth image
        cv::Mat vggt_depth_map = projectPointCloudToDepthMap(vggt_pc, depth_filtered.size());

        // 6. Perform least squares depth completion
        cv::Mat completed_depth = leastSquaresDepthCompletion(depth_filtered, vggt_depth_map);

        return completed_depth;
    }

    void filterSpeckleNoise(const cv::Mat& depth_in, cv::Mat& depth_out) {
        auto max_speckle_size = this->get_parameter("speckle_window_size").as_int();
        auto max_diff = this->get_parameter("speckle_range").as_int();
        
        // Convert to signed 16-bit for cv::filterSpeckles
        cv::Mat depth_s16;
        depth_in.convertTo(depth_s16, CV_16S);
        
        // Apply OpenCV's filterSpeckles
        // This removes small noise regions in disparity/depth maps
        cv::filterSpeckles(depth_s16, 0, max_speckle_size, max_diff);
        
        // Convert back to unsigned 16-bit
        depth_s16.convertTo(depth_out, CV_16U);
    }

    cv::Mat calculateDisparity(const cv::Mat& depth) {
        auto epsilon = this->get_parameter("depth_epsilon").as_double();
        
        cv::Mat disparity;
        cv::Mat depth_float;
        depth.convertTo(depth_float, CV_32F, 0.001); // Convert mm to meters

        // Calculate disparity = 1 / (depth + epsilon)
        disparity = 1.0 / (depth_float + epsilon);

        return disparity;
    }

    cv::Mat projectPointCloudToDepthMap(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& pc_msg,
                                       const cv::Size& image_size) {
        cv::Mat depth_map = cv::Mat::zeros(image_size, CV_16U);
        
        // Convert point cloud to PCL format
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*pc_msg, *cloud);
        
        // Use scaled camera intrinsics for projection
        float fx = scaled_camera_matrix_.at<double>(0, 0);
        float fy = scaled_camera_matrix_.at<double>(1, 1);
        float cx = scaled_camera_matrix_.at<double>(0, 2);
        float cy = scaled_camera_matrix_.at<double>(1, 2);
        
        // Project each point to image plane
        for (const auto& point : cloud->points) {
            if (point.z <= 0 || std::isnan(point.z)) continue;
            
            int u = static_cast<int>(fx * point.x / point.z + cx);
            int v = static_cast<int>(fy * point.y / point.z + cy);
            
            if (u >= 0 && u < image_size.width && v >= 0 && v < image_size.height) {
                uint16_t depth_mm = static_cast<uint16_t>(point.z * 1000.0f);
                // Keep closest depth value if multiple points project to same pixel
                if (depth_map.at<uint16_t>(v, u) == 0 || depth_mm < depth_map.at<uint16_t>(v, u)) {
                    depth_map.at<uint16_t>(v, u) = depth_mm;
                }
            }
        }
        
        return depth_map;
    }

    cv::Mat leastSquaresDepthCompletion(const cv::Mat& sensor_depth, const cv::Mat& predicted_depth) {
        auto epsilon = this->get_parameter("depth_epsilon").as_double();
        
        // Convert depth to disparity
        cv::Mat sensor_float, predicted_float;
        sensor_depth.convertTo(sensor_float, CV_32F, 0.001); // mm to meters
        predicted_depth.convertTo(predicted_float, CV_32F, 0.001); // mm to meters
        
        // Create mask for valid pixels where both sensor and predicted data exist
        cv::Mat valid_mask = (sensor_float > epsilon) & (predicted_float > epsilon);
        
        // Calculate disparities for valid pixels
        cv::Mat sensor_disparity = cv::Mat::zeros(sensor_float.size(), CV_32F);
        cv::Mat predicted_disparity = cv::Mat::zeros(predicted_float.size(), CV_32F);
        
        sensor_float.forEach<float>([&](float& pixel, const int pos[]) {
            if (valid_mask.at<uint8_t>(pos[0], pos[1]) && pixel > epsilon) {
                sensor_disparity.at<float>(pos[0], pos[1]) = 1.0f / pixel;
                predicted_disparity.at<float>(pos[0], pos[1]) = 1.0f / predicted_float.at<float>(pos[0], pos[1]);
            }
        });
        
        // Extract valid disparity values into vectors
        std::vector<float> sensor_disp_vec, predicted_disp_vec;
        for (int y = 0; y < valid_mask.rows; y++) {
            for (int x = 0; x < valid_mask.cols; x++) {
                if (valid_mask.at<uint8_t>(y, x)) {
                    sensor_disp_vec.push_back(sensor_disparity.at<float>(y, x));
                    predicted_disp_vec.push_back(predicted_disparity.at<float>(y, x));
                }
            }
        }
        
        // Default alignment parameters
        float scale = 1.0f, shift = 0.0f;
        
        if (sensor_disp_vec.size() >= 10) {
            // Create matrices for SVD: sensor_disp = scale * predicted_disp + shift
            // We need to solve: [predicted_disp, 1] * [scale; shift] = sensor_disp
            int n = sensor_disp_vec.size();
            cv::Mat A(n, 2, CV_32F);
            cv::Mat b(n, 1, CV_32F);
            
            // Fill matrices
            for (int i = 0; i < n; i++) {
                A.at<float>(i, 0) = predicted_disp_vec[i];  // predicted disparity
                A.at<float>(i, 1) = 1.0f;                   // for shift term
                b.at<float>(i, 0) = sensor_disp_vec[i];     // sensor disparity
            }
            
            // Solve using SVD: A * x = b, where x = [scale; shift]
            cv::Mat x;
            cv::solve(A, b, x, cv::DECOMP_SVD);
            
            scale = x.at<float>(0, 0);
            shift = x.at<float>(1, 0);
            
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "SVD alignment: scale=%.3f, shift=%.3f, valid_points=%d", 
                scale, shift, n);
        }
        
        // Apply alignment to all predicted disparities
        cv::Mat completed_depth = sensor_depth.clone();
        
        for (int y = 0; y < completed_depth.rows; y++) {
            for (int x = 0; x < completed_depth.cols; x++) {
                // Fill missing sensor depths with aligned predicted depths
                if (sensor_float.at<float>(y, x) <= epsilon && predicted_float.at<float>(y, x) > epsilon) {
                    // Calculate aligned disparity
                    float pred_disp = 1.0f / predicted_float.at<float>(y, x);
                    float aligned_disp = scale * pred_disp + shift;
                    
                    // TODO: Convert back to depth
                    if (aligned_disp > epsilon) {
                        float aligned_depth = 1.0f / aligned_disp;
                        uint16_t depth_mm = static_cast<uint16_t>(aligned_depth * 1000.0f);
                        
                        if (depth_mm > 0 && depth_mm < 65535) {
                            completed_depth.at<uint16_t>(y, x) = depth_mm;
                        }
                    }
                }
            }
        }
        
        return completed_depth;
    }

    sensor_msgs::msg::PointCloud2::SharedPtr createPointCloud(
        const cv::Mat& rgb, const cv::Mat& depth, const std_msgs::msg::Header& header) {
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        // Use scaled camera intrinsics since images are at engine resolution
        float fx = scaled_camera_matrix_.at<double>(0, 0);
        float fy = scaled_camera_matrix_.at<double>(1, 1);
        float cx = scaled_camera_matrix_.at<double>(0, 2);
        float cy = scaled_camera_matrix_.at<double>(1, 2);

        for (int v = 0; v < depth.rows; v++) {
            for (int u = 0; u < depth.cols; u++) {
                uint16_t depth_value = depth.at<uint16_t>(v, u);
                if (depth_value == 0) continue;

                float z = depth_value * 0.001f; // mm to meters
                float x = (u - cx) * z / fx;
                float y = (v - cy) * z / fy;

                pcl::PointXYZRGB point;
                point.x = x;
                point.y = y;
                point.z = z;

                cv::Vec3b color = rgb.at<cv::Vec3b>(v, u);
                point.r = color[2];
                point.g = color[1];
                point.b = color[0];

                cloud->push_back(point);
            }
        }

        auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(*cloud, *msg);
        msg->header = header;
        
        return msg;
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DepthCompletionNode>());
    rclcpp::shutdown();
    return 0;
}
