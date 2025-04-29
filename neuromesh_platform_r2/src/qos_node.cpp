#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

namespace qos_node {
class CameraInfoRepublisher : public rclcpp::Node
{
public:
    CameraInfoRepublisher(const rclcpp::NodeOptions& options)
    : Node("camera_info_republisher", options)
    {
	// Declare parameters for QoS settings
        this->declare_parameter("qos_reliability", "reliable");
        this->declare_parameter("qos_durability", "volatile");

        // Get QoS settings from parameters
        std::string qos_reliability, qos_durability;
        this->get_parameter("qos_reliability", qos_reliability);
        this->get_parameter("qos_durability", qos_durability);

        // Subscribe to original camera_info topic with its existing QoS settings
        subscription = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/color/camera_info", 
            rclcpp::QoS(rclcpp::KeepLast(10)), // Change to the desired QoS settings
            std::bind(&CameraInfoRepublisher::cameraInfoCallback, this, std::placeholders::_1));

        // Get the QoS settings of the subscription
        auto qos = subscription->get_actual_qos();

	// Modify the qos settings
	qos.reliability(parseReliability(qos_reliability));
	qos.durability(parseDurability(qos_durability));

	// Publish topic with modified qos settings
        publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
            "/camera/color/camera_info_modified", qos);
    }

private:
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        // Republish camera_info message with modified QoS
        publisher_->publish(*msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr subscription;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr publisher_;

    // Helper functions for parsing reliability and durability settings
    rmw_qos_reliability_policy_t parseReliability(const std::string& reliability) {
        if (reliability == "reliable") {
            return RMW_QOS_POLICY_RELIABILITY_RELIABLE;
        }
        return RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;
    }

    rmw_qos_durability_policy_t parseDurability(const std::string& durability) {
        if (durability == "transient_local") {
            return RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL;
        }
        return RMW_QOS_POLICY_DURABILITY_VOLATILE;
    }
};
}

/*int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraInfoRepublisher>());
    rclcpp::shutdown();
    return 0;
}*/

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(qos_node::CameraInfoRepublisher)

