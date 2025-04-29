#include "tensorrt_engine/engine_node.h"

int main(int argc, char ** argv)
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<TensorRTEngineNode>());
	rclcpp::shutdown();

	printf("shutting down tflite engine node\n");
	return 0;
};