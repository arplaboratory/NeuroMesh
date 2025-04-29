#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>

namespace odom_republisher
{
class OdomRepublisher : public rclcpp::Node 
{
public:
    OdomRepublisher(const rclcpp::NodeOptions& options) : Node("odom_republisher", options)
    {
        this->declare_parameter<std::string>("robot_name", "khonsu");
        this->get_parameter("robot_name", robot_name_);
        RCLCPP_DEBUG(this->get_logger(), "Robot name: %s", robot_name_.c_str());

        this->declare_parameter<std::string>("frame_id", robot_name_ + "/map");
        this->get_parameter("frame_id", frame_id_);
        RCLCPP_DEBUG(this->get_logger(), "Frame ID: %s", frame_id_.c_str());

        this->declare_parameter<double>("offset_x", 3.0);
        this->get_parameter("offset_x", offset_x_);
        RCLCPP_DEBUG(this->get_logger(), "Offset X: %f", offset_x_);

        this->declare_parameter<double>("offset_y", 3.0);
        this->get_parameter("offset_y", offset_y_);
        RCLCPP_DEBUG(this->get_logger(), "Offset Y: %f", offset_y_);

        // Declare the robot name parameter
        // Subscription to original odometry topic
        subscription_ = create_subscription<nav_msgs::msg::Odometry>(
            "/original/odom", 10, 
            std::bind(&OdomRepublisher::odomCallback, this, std::placeholders::_1)
        );

        // Publisher for modified odometry topic
        publisher_ = create_publisher<nav_msgs::msg::Odometry>(
            "/republished/odom", 10
        );
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // Create a new odometry message
        auto modified_msg = std::make_shared<nav_msgs::msg::Odometry>();

        modified_msg->header.frame_id = frame_id_;
        
        modified_msg->pose.pose.position.x = msg->pose.pose.position.x + offset_x_;
        modified_msg->pose.pose.position.y = msg->pose.pose.position.y + offset_y_;
        
        // Publish the modified message
        publisher_->publish(*modified_msg);

        RCLCPP_DEBUG(this->get_logger(), 
            "Original Pos: (%.2f, %.2f), Modified Pos: (%.2f, %.2f)",
            msg->pose.pose.position.x, msg->pose.pose.position.y,
            modified_msg->pose.pose.position.x, modified_msg->pose.pose.position.y
        );
    }

    // ROS2 communication objects
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr publisher_;
    std::string robot_name_; // robot name to be used for namespacing frame_id
    std::string frame_id_;
    double offset_x_;
    double offset_y_;
};
}

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(odom_republisher::OdomRepublisher)