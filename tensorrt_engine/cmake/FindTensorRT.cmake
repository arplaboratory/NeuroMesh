# Locate the official NVIDIA TensorRT SDK and expose versioned imported targets.
# TensorRT_ROOT may point at an unpacked SDK. System package paths remain eligible.
find_path(TensorRT_INCLUDE_DIR
  NAMES NvInfer.h NvInferVersion.h
  HINTS ${TensorRT_ROOT} ENV TensorRT_ROOT
  PATH_SUFFIXES include include/x86_64-linux-gnu)
find_library(TensorRT_NVINFER_LIBRARY
  NAMES nvinfer
  HINTS ${TensorRT_ROOT} ENV TensorRT_ROOT
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
find_library(TensorRT_ONNXPARSER_LIBRARY
  NAMES nvonnxparser
  HINTS ${TensorRT_ROOT} ENV TensorRT_ROOT
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
find_library(TensorRT_PLUGIN_LIBRARY
  NAMES nvinfer_plugin
  HINTS ${TensorRT_ROOT} ENV TensorRT_ROOT
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)

if(TensorRT_INCLUDE_DIR)
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_version_lines
    REGEX "^#define NV_TENSORRT_(MAJOR|MINOR|PATCH|BUILD) ")
  foreach(_component MAJOR MINOR PATCH BUILD)
    string(REGEX MATCH "NV_TENSORRT_${_component} +([0-9]+)"
      _match "${_trt_version_lines}")
    if(CMAKE_MATCH_1 STREQUAL "")
      set(TensorRT_VERSION_${_component} 0)
    else()
      set(TensorRT_VERSION_${_component} "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  set(TensorRT_VERSION
    "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}.${TensorRT_VERSION_BUILD}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
  REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_NVINFER_LIBRARY
    TensorRT_ONNXPARSER_LIBRARY TensorRT_PLUGIN_LIBRARY
  VERSION_VAR TensorRT_VERSION)

if(TensorRT_FOUND)
  if(NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES
      IMPORTED_LOCATION "${TensorRT_NVINFER_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
  endif()
  if(NOT TARGET TensorRT::nvonnxparser)
    add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvonnxparser PROPERTIES
      IMPORTED_LOCATION "${TensorRT_ONNXPARSER_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES TensorRT::nvinfer)
  endif()
  if(NOT TARGET TensorRT::plugin)
    add_library(TensorRT::plugin UNKNOWN IMPORTED)
    set_target_properties(TensorRT::plugin PROPERTIES
      IMPORTED_LOCATION "${TensorRT_PLUGIN_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES TensorRT::nvinfer)
  endif()
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_NVINFER_LIBRARY
  TensorRT_ONNXPARSER_LIBRARY TensorRT_PLUGIN_LIBRARY)
