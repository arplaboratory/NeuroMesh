cmake_minimum_required(VERSION 3.5)
project(onnx_engine)
set(CMAKE_BUILD_TYPE Release)

if(BUILD_TESTING)
  message("Building tests for onnx_engine")
  find_package(ament_cmake_gmock REQUIRED)
  ament_add_gmock(test_onnx_engine test/test_onnx_engine.cpp)
  target_link_libraries(test_onnx_engine onnx_engine)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set (dependencies
  "engine_interface"
  "pluginlib"
)

find_package(ament_cmake REQUIRED)
find_package(PkgConfig REQUIRED)
foreach(dep ${dependencies})
  find_package(${dep} REQUIRED)
endforeach()

include_directories(include)

# Using Pkg Config to find ONNX Runtime
pkg_check_modules(ONNXRUNTIME REQUIRED libonnxruntime)
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
link_directories(${ONNXRUNTIME_LIBRARY_DIRS})

# Using find_package to find ONNX Runtime
# message("Finding ONNX Runtime")
# message(onnxruntime_INCLUDE_DIRS: ${onnxruntime_INCLUDE_DIRS})
# message(onnxruntime_LIBRARIES: ${onnxruntime_LIBRARIES})
# find_package(onnxruntime REQUIRED)
# include_directories(${onnxruntime_INCLUDE_DIRS})

add_library(onnx_engine SHARED
  src/onnx_engine.cpp)

target_include_directories(onnx_engine PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(onnx_engine ${ONNXRUNTIME_LIBRARIES})
# target_link_libraries(onnx_engine
#   ${onnxruntime_LIBRARIES}
# )
# set_target_properties(onnx_engine PROPERTIES
#   COMPILE_DEFINITIONS "COMPOSITION_BUILDING_DLL"
# )
ament_target_dependencies(
  onnx_engine
  ${dependencies}
)

install(FILES plugin_description.xml
  DESTINATION share/${PROJECT_NAME})
pluginlib_export_plugin_description_file(engine_interface plugin_description.xml)

install(TARGETS
  onnx_engine
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME})

install(FILES plugin_description.xml 
  DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(onnx_engine)
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(
  ${dependencies}
)
ament_package()
