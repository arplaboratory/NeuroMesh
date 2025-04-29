#include "neuromesh_platform_r2/neuromesh_node.h"

int main(int argc, char** argv)
{
	rclcpp::init(argc, argv);
	auto neuromesh_node = std::make_shared<neuromeshNode>();
	rclcpp::spin(neuromesh_node);
	rclcpp::shutdown();
	return 0;
}