#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

class ImageResizer : public rclcpp::Node
{
public:
    ImageResizer() : Node("image_resizer")
    {
	height_ = this->declare_parameter<int>("height", 480); // Default value of 480
        width_ = this->declare_parameter<int>("width", 640);   // Default value of 640

        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "camera", 10,
            [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                // Resize the image
                cv_bridge::CvImagePtr cv_ptr;
                try
                {
                    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
                }
                catch (cv_bridge::Exception &e)
                {
                    RCLCPP_ERROR_STREAM(get_logger(), "cv_bridge exception: " << e.what());
                    return;
                }

                cv::Mat resized_image;
                cv::resize(cv_ptr->image, resized_image, cv::Size(height_, width_)); // Example resizing to 640x480

		// Print the size of the resized image
	       	//RCLCPP_INFO(this->get_logger(), "Resized image size: %dx%d", resized_image.cols, resized_image.rows);

                // Publish the resized image
                sensor_msgs::msg::Image::SharedPtr resized_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", resized_image).toImageMsg();
                pub_->publish(*resized_msg);
            });

        pub_ = this->create_publisher<sensor_msgs::msg::Image>("output", 10);
    }

private:
    int height_;
    int width_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ImageResizer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
