find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime libonnxruntime.so.1.23.2)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime REQUIRED_VARS ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY)
if(ONNXRuntime_FOUND AND NOT TARGET ONNXRuntime::Runtime)
  add_library(ONNXRuntime::Runtime UNKNOWN IMPORTED)
  set_target_properties(ONNXRuntime::Runtime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
endif()
mark_as_advanced(ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY)
