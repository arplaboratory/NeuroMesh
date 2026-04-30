cmake_minimum_required(VERSION 3.5)
project(tensorrt_engine)
set(CMAKE_BUILD_TYPE Release)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set (dependencies
  "rclcpp"
  "std_msgs"
  "sensor_msgs"
  "cv_bridge"
  "neuromesh_interfaces"
  "rclcpp_components"
  "engine_interface"
  "pluginlib"
)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
foreach(dep ${dependencies})
  find_package(${dep} REQUIRED)
endforeach()

include_directories(include)

add_library(tensorrt_engine SHARED
  src/trt_engine_modified.cpp)

target_include_directories(tensorrt_engine PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})


target_link_libraries(tensorrt_engine ${OpenCV_LIBS} ${CUDA_LIBRARIES} nvinfer nvonnxparser nvinfer_plugin cudnn)
# set_target_properties(tensorrt_engine PROPERTIES
#   COMPILE_DEFINITIONS "COMPOSITION_BUILDING_DLL"
# )

ament_target_dependencies(
  tensorrt_engine
  ${dependencies}
  )

if(BUILD_TESTING)
  message("Building tests for tensorrt_engine")
  find_package(ament_cmake_gmock REQUIRED)
  ament_add_gmock(test_trt_engine test/test_trt_engine.cpp)
  target_link_libraries(test_trt_engine tensorrt_engine)
endif()

install(FILES plugin_description.xml
  DESTINATION share/${PROJECT_NAME})
pluginlib_export_plugin_description_file(engine_interface plugin_description.xml)

install(TARGETS
  tensorrt_engine
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME})

ament_export_libraries(tensorrt_engine)
ament_export_targets(export_${PROJECT_NAME})
ament_package()
