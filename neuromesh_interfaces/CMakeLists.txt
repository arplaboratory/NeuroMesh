cmake_minimum_required(VERSION 3.5)
project(neuromesh_interfaces)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

set(srv_files
    "srv/TensorRequest.srv"
)

set (msg_files
      "msg/Tensor.msg"
      "msg/TensorShape.msg"
      "msg/Feature.msg"
      "msg/CompressedTensor.msg"
      "msg/CompressedFeature.msg"
      )


rosidl_generate_interfaces(${PROJECT_NAME}
	${msg_files}
  ${srv_files}
  DEPENDENCIES std_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)


# rosidl_generate_interfaces(${PROJECT_NAME}
#     ${srv_files}
#     DEPENDENCIES builtin_interfaces std_msgs)


ament_package()
