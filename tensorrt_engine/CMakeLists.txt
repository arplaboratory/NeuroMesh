cmake_minimum_required(VERSION 3.5)
project(tensorrt_engine)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()



# find dependencies
find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rclcpp REQUIRED)
find_package(neuromesh_interfaces REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(CUDA REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

#add_executable(tensorrt_engine src/engine_node.cpp src/main.cpp src/trt_engine.cpp  )
#target_include_directories(tensorrt_engine PUBLIC "include/")

include_directories(include)

add_library(tensorrt_engine SHARED
  src/engine_modified.cpp src/trt_engine_modified.cpp)

set_target_properties(tensorrt_engine PROPERTIES
  COMPILE_DEFINITIONS "COMPOSITION_BUILDING_DLL"
)

ament_target_dependencies(
  tensorrt_engine
    rclcpp
    std_msgs
    sensor_msgs
    cv_bridge
    neuromesh_interfaces
    rclcpp_components
  )

  #install(TARGETS tensorrt_engine
  #DESTINATION lib/${PROJECT_NAME})

target_include_directories(tensorrt_engine PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} tensorrt)
target_link_libraries(tensorrt_engine ${OpenCV_LIBS} ${CUDA_LIBRARIES} nvinfer nvonnxparser nvinfer_plugin cudnn)

rclcpp_components_register_nodes(tensorrt_engine "tensorrt_engine_node::TensorRTEngineNode")

 install(TARGETS
  tensorrt_engine
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME})

ament_package()
