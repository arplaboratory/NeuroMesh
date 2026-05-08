# Build Guide

## System dependencies

Install ROS 2 Humble base tools and common development dependencies:

- `python3-colcon-common-extensions`
- `python3-rosdep`
- CMake and compiler toolchain

## Build steps

1. Run `rosdep install` for workspace dependencies.
2. Build with `colcon build --symlink-install`.
3. Source `install/setup.bash`.

## Optional engines

- `onnx_engine` for ONNX Runtime backend
- `tensorrt_engine` for TensorRT backend

## Optional navigation bridge

`mission_maestro_bridge` is optional. It only activates forwarding when `arl_mission_maestro` is present.
