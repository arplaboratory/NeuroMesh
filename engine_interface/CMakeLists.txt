cmake_minimum_required(VERSION 3.5)
project(engine_interface)
set(CMAKE_BUILD_TYPE Release)

if(BUILD_TESTING)
  message("Building tests for engine_interface")
  find_package(ament_cmake_gmock REQUIRED)
  ament_add_gmock(test_engine_interface test/test_engine_interface.cpp)
  target_link_libraries(test_engine_interface engine_interface)
endif()

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
  "neuromesh_interfaces"
  "rclcpp_components"
  "pluginlib"
)

find_package(ament_cmake REQUIRED)
foreach(dep ${dependencies})
  find_package(${dep} REQUIRED)
endforeach()

include_directories(include)

add_library(${PROJECT_NAME} SHARED
  src/engine_interface_node.cpp
  src/inference_engine_base.cpp 
)

target_include_directories(engine_interface PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(
  engine_interface
  ${dependencies}
)

rclcpp_components_register_nodes(engine_interface "engine_interface_node::EngineInterfaceNode")

 install(TARGETS
  engine_interface
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL 
  )
  
install(DIRECTORY include/ 
  DESTINATION include)


ament_export_include_directories(include)
ament_export_libraries(engine_interface)
ament_export_dependencies(
  ${dependencies}
)
ament_package()
