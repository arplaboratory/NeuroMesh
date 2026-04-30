cmake_minimum_required(VERSION 3.5)
project(neuromesh_platform_r2)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set (dependencies
 "std_msgs"
 "rclcpp"
 "sensor_msgs"
 "neuromesh_interfaces"
 "rclcpp_components"
 "yaml-cpp"
 "tf2_ros"
 "geometry_msgs"
 "cv_bridge"
 "OpenCV"
 "Eigen3"
 "nav_msgs"
 "tf2_geometry_msgs"
 "tf2_sensor_msgs"
 "visualization_msgs"
 "interactive_markers"
 "rviz_rendering"
 "rviz_common"
 "std_srvs"
 "pcl_conversions"
 "realsense2_camera_msgs"
 "message_filters"
 )


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(ZLIB)
find_package(OpenCV 4 REQUIRED)
find_package(PCL REQUIRED)
find_package(tf2 REQUIRED)
foreach(dep ${dependencies})
  find_package(${dep} REQUIRED)
endforeach()

include_directories(include)
include_directories(
    include ${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS})

add_executable(visualization_node
  src/visualization_node.cpp
)

add_executable(decompression_node
  src/decompression_node.cpp
)

add_executable(compression_node
  src/compression_node.cpp
)

# Modified Codes
add_library(dust3r_example SHARED
 src/dust3r_toy_implementation_batch.cpp
 src/dust3r_neuromesh_node_batch.cpp)

set_target_properties(dust3r_example PROPERTIES
 COMPILE_DEFINITIONS "COMPOSITION_BUILDING_DLL"
)

ament_target_dependencies(dust3r_example
 ${dependencies}
 )

target_link_libraries(dust3r_example ${YAML_CPP_LIBRARIES})
rclcpp_components_register_nodes(dust3r_example "neuromeshNode::ToyImplementation")

# VGGT Codes
add_library(vggt_example SHARED
 src/vggt_toy_implementation.cpp
 src/vggt_neuromesh_node.cpp)

set_target_properties(vggt_example PROPERTIES
 COMPILE_DEFINITIONS "COMPOSITION_BUILDING_DLL"
)

 ament_target_dependencies(vggt_example
 ${dependencies}
 )

target_link_libraries(vggt_example ${YAML_CPP_LIBRARIES})
rclcpp_components_register_nodes(vggt_example "vggtNode::VggtToyImplementation")

# VGGT Separated Nodes
add_library(vggt_separated SHARED
 src/vggt_encoder_node.cpp
 src/vggt_decoder_node.cpp)

set_target_properties(vggt_separated PROPERTIES
 COMPILE_DEFINITIONS "COMPOSITION_BUILDING_DLL"
)

ament_target_dependencies(vggt_separated
 ${dependencies}
 )

target_link_libraries(vggt_separated ${YAML_CPP_LIBRARIES} ${PCL_LIBRARIES})
rclcpp_components_register_nodes(vggt_separated 
  "neuromesh::VggtEncoderNode"
  "neuromesh::VggtDecoderNode")

#Control Codes
add_library(control_implementation SHARED
  src/control_implementation.cpp
  src/control_neuromesh_node.cpp)

set_target_properties(control_implementation PROPERTIES
  COMPILE_DEFINITIONS "COMPOSITION_BUILDING_DLL"
)

ament_target_dependencies(control_implementation
    ${dependencies}
  )

target_link_libraries(control_implementation ${YAML_CPP_LIBRARIES})

rclcpp_components_register_nodes(control_implementation "ControlneuromeshNode::ControlImplementation")

# GAT Codes
add_library(gat_example SHARED
 src/gat_model_implementation.cpp
 src/gat_model_neuromesh_node.cpp
 src/gat_planner_model_implementation.cpp
 src/gat_planner_model_neuromesh_node.cpp)
set_target_properties(gat_example PROPERTIES
 COMPILE_DEFINITIONS "COMPOSITION_BUILDING_DLL"
)
ament_target_dependencies(gat_example
 ${dependencies}
 )
target_link_libraries(gat_example ${YAML_CPP_LIBRARIES})
rclcpp_components_register_nodes(gat_example 
  "GATneuromeshNode::GATImplementation"
  "GATPlannerNeuromeshNode::GATPlannerImplementation")

# odom_republisher
add_library(odom_republisher SHARED
  src/odom_republisher.cpp)

set_target_properties(odom_republisher PROPERTIES
  COMPILE_DEFINITIONS "COMPOSITION_BUILDING_DLL"
  )

# Link dependencies
ament_target_dependencies(odom_republisher
  "rclcpp"
  "nav_msgs"
  "rclcpp_components"
)

rclcpp_components_register_nodes(odom_republisher "odom_republisher::OdomRepublisher")


# Add include folder
target_include_directories(visualization_node PUBLIC "include/")
target_include_directories(decompression_node PUBLIC "include/")
target_include_directories(compression_node PUBLIC "include/")

ament_target_dependencies(
  visualization_node
	"std_msgs"
  "rclcpp"
  "sensor_msgs"
  "neuromesh_interfaces"
)

ament_target_dependencies(
  decompression_node
	"std_msgs"
  "rclcpp"
  "sensor_msgs"
  "neuromesh_interfaces"
)

ament_target_dependencies(
  compression_node
	"std_msgs"
  "rclcpp"
  "sensor_msgs"
  "neuromesh_interfaces"
)

target_link_libraries(compression_node ZLIB::ZLIB)
target_link_libraries(decompression_node ZLIB::ZLIB)

# Depth completion node
add_executable(depth_completion_node
  src/depth_completion_node.cpp
)

target_include_directories(depth_completion_node PUBLIC "include/")

ament_target_dependencies(depth_completion_node
  "rclcpp"
  "sensor_msgs"
  "cv_bridge"
  "PCL"
  "pcl_conversions"
  "message_filters"
  "realsense2_camera_msgs"
)

target_link_libraries(depth_completion_node
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS
  odom_republisher
  dust3r_example
  gat_example
  vggt_example
  vggt_separated
  #control_implementation
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  )

install(TARGETS
	#neuromesh_node
	#dust3r_exmaple
	visualization_node decompression_node compression_node depth_completion_node
	#resizer_node qos_node
	DESTINATION lib/${PROJECT_NAME})

install(FILES
  launch/full_except_bridge.py
  launch/dust3r_model_neuromesh_launch.py
  launch/vggt_model_neuromesh_launch.py
  launch/gat_model_neuromesh_launch.py
  launch/gat_planner_model_neuromesh_launch.py
  launch/vggt_separated_launch.py
  launch/depth_completion_launch.py
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

install(PROGRAMS 
    scripts/dust3r_model_neuromesh_launch.sh
    scripts/vggt_model_neuromesh_launch.sh
    scripts/vggt_separated_launch.sh
    DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)
ament_export_libraries(vggt_separated dust3r_example vggt_example odom_republisher)
ament_package()
