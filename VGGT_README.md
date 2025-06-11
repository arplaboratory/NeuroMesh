# VGGT Model Implementation for 2-Robot Setup

This implementation provides a ROS2 node for running VGGT (Visual Geometry Grounded Transformation) models in a multi-robot environment.

## Overview

The VGGT implementation includes:
- **Encoder**: Processes RGB images (392x518 resolution) to extract patch tokens (1x1036x1024)
- **Decoder**: Aggregates features from 2 robots to produce depth maps, 3D points, and confidence maps
- **Multi-robot communication**: Feature sharing between robots via ROS2 topics
- **Output publishers**: Depth images, pointclouds (with/without RGB colors)

## Model Specifications

### Encoder (vggt_image_encoder_2x.engine)
- **Input**: Images with dimensions 1x3x392x518
- **Output**: patch_tokens with dimensions 1x1036x1024

### Decoder (vggt_aggregator_2x.engine)  
- **Input**: patch_tokens with dimensions 2x1036x1024 (from 2 robots)
- **Outputs**:
  - pose_enc: 1x2x9
  - depth: 1x2x392x518x1 
  - depth_conf: 1x2x392x518
  - world_points: 1x2x392x518x3
  - world_points_conf: 1x2x392x518

## Files Created

### Core Implementation
- `include/neuromesh_platform_r2/vggt_neuromesh_node.h` - Base VGGT node header
- `src/vggt_neuromesh_node.cpp` - Base VGGT node implementation
- `include/neuromesh_platform_r2/vggt_toy_implementation.h` - TensorRT integration header
- `src/vggt_toy_implementation.cpp` - TensorRT integration implementation

### Launch Files
- `launch/vggt_model_neuromesh_launch.py` - Main launch file for VGGT setup
- `scripts/vggt_model_neuromesh_launch.sh` - Launch script for easy execution

## Usage

### Launch Single Robot
```bash
# Terminal 1 - Robot 1 (khonsu)
./scripts/vggt_model_neuromesh_launch.sh khonsu 1

# Terminal 2 - Robot 2 (anubis) 
./scripts/vggt_model_neuromesh_launch.sh anubis 2
```

### Launch with Custom Parameters
```bash
ros2 launch neuromesh_platform_r2 vggt_model_neuromesh_launch.py \
    name:=khonsu \
    agent_num:=1 \
    agent_list:="khonsu,anubis" \
    color_raw_topic:=/khonsu/sensors/camera_0/camera/color/image_raw
```

## Published Topics

Each robot publishes the following topics:

### Feature Sharing
- `/{robot_name}/features_{robot_name}` - Feature messages for multi-robot coordination

### Depth Outputs
- `/{robot_name}/depth_robot1` - Depth image for current robot
- `/{robot_name}/depth_robot2` - Depth image for neighbor robot

### Point Clouds
- `/{robot_name}/pointcloud_current` - 3D pointcloud for current robot
- `/{robot_name}/pointcloud_neighbor` - 3D pointcloud for neighbor robot  
- `/{robot_name}/pointcloud_current_rgb` - RGB pointcloud for current robot
- `/{robot_name}/pointcloud_neighbor_rgb` - RGB pointcloud for neighbor robot

## Subscribed Topics

- `/{robot_name}/sensors/camera_0/camera/color/image_raw` - Input RGB camera images
- `/other_robot/features_other_robot` - Features from neighbor robots

## Key Features

### Image Preprocessing
- Automatic resizing to 392x518 resolution (VGGT input requirement)
- Normalization to [-1, 1] range
- HWC to CHW format conversion for neural network input

### Multi-Robot Coordination
- Feature sharing between robots via ROS2 topics
- Temporal synchronization of features
- Automatic neighbor selection for decoder input

### Output Processing
- Depth image generation from decoder outputs
- 3D pointcloud creation with confidence filtering
- RGB pointcloud generation using original camera colors
- Separate outputs for current robot vs neighbor robot data

### Performance Monitoring
- Built-in timing measurements for encoder/decoder inference
- Debug logging for feature processing and tensor operations

## Configuration

### Default Agent Setup
- **Robot 1**: khonsu (agent_num: 1)
- **Robot 2**: anubis (agent_num: 2)
- **Agent List**: "khonsu,anubis"

### Model Paths
- **Encoder**: `tensorrt_engine/models/vggt_onnx_2x/vggt_image_encoder_2x.engine`
- **Decoder**: `tensorrt_engine/models/vggt_onnx_2x/vggt_aggregator_2x.engine`

### Cycle Timing
- **Encoder Cycle**: 3000ms (configurable)
- **Decoder Cycle**: 3000ms (configurable)

## Coordinate Frames

- **Depth Images**: Published in camera optical frame (`cam1_color_optical_frame`)
- **Point Clouds**: Published in robot map frame (`{robot_name}/map`)
- **Transform Broadcasting**: Automatic TF2 transform broadcasting between frames

## Dependencies

- ROS2 (tested with appropriate ROS2 distribution)
- TensorRT Engine service for model inference
- OpenCV for image processing
- cv_bridge for ROS-OpenCV conversion
- sensor_msgs for point cloud generation
- neuromesh_interfaces for custom message types

## Build Instructions

The VGGT implementation is integrated into the existing neuromesh_platform_r2 package:

```bash
cd /path/to/workspace
colcon build --packages-select neuromesh_platform_r2
source install/setup.bash
```

## Notes

- Requires exactly 2 robots for proper operation
- TensorRT engine files must be present in the specified model directory
- Camera topics should follow the expected naming convention
- Point cloud confidence threshold is set to 0.5 (configurable in code)