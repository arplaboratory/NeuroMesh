cmake_minimum_required(VERSION 3.18)
project(neuromesh_tensorrt_sdk VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/resolve_tensorrt.cmake)
add_library(neuromesh_tensorrt_sdk INTERFACE)
add_library(NeuroMesh::tensorrt_sdk ALIAS neuromesh_tensorrt_sdk)
target_link_libraries(neuromesh_tensorrt_sdk INTERFACE
  TensorRT::nvinfer TensorRT::nvonnxparser TensorRT::plugin CUDA::cudart)
set_target_properties(neuromesh_tensorrt_sdk PROPERTIES EXPORT_NAME tensorrt_sdk)
install(TARGETS neuromesh_tensorrt_sdk EXPORT NeuroMeshTensorRTSDKTargets)
install(EXPORT NeuroMeshTensorRTSDKTargets NAMESPACE NeuroMesh::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuromesh_tensorrt_sdk)
install(FILES cmake/resolve_tensorrt.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuromesh_tensorrt_sdk)
configure_package_config_file(cmake/config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/neuromesh_tensorrt_sdkConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuromesh_tensorrt_sdk)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/neuromesh_tensorrt_sdkConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/neuromesh_tensorrt_sdkConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/neuromesh_tensorrt_sdkConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuromesh_tensorrt_sdk)
install(FILES package.xml DESTINATION share/${PROJECT_NAME})
