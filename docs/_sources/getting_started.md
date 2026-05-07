# Getting Started

NeuroMesh is a unified neural inference framework for decentralized multi-robot
collaborative learning. It enables robot teams to share neural information and
computational resources to achieve objectives that surpass individual robots'
capabilities.

## Key Features

- **General**: Compatible with multiple task domains (perception, control, planning) and robotic platforms
- **Modular**: Flexible implementation that allows easy integration with existing robotic systems
- **Decentralized**: Independent operation across multiple robots without a central coordinator
- **Efficient**: Optimized for real-time operation on resource-constrained platforms
- **Interoperable**: Works with heterogeneous robot teams (aerial and ground robots)

## Prerequisites

### Third-Party Software

| Dependency | Version |
|---|---|
| TensorRT | 8.6.1 |
| NVIDIA CUDA | 12.1 |
| ROS2 | Jazzy |
| Zenoh | 1.0.0-dev |
| zenoh-bridge-ros2dds | v1.0.0-dev-34-gca4a1f2 |

Optional:
- **tmux** - for conveniently launching multi-pane sessions
- **Rviz2** - for visualizing point cloud outputs

## Environment Setup

Export the following environment variables on every terminal where you run ROS2
nodes. The `ROS_DOMAIN_ID` must be **unique** on each agent:

```bash
export ROS_DOMAIN_ID=80
export ROBOT_NAME=race16
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

## Build

```bash
colcon build --packages-up-to neuromesh_platform_r2
source install/setup.bash
```

To build individual packages:

```bash
# Interfaces only
colcon build --packages-select neuromesh_interfaces

# Engine interface + plugins
colcon build --packages-select engine_interface tensorrt_engine onnx_engine

# Platform
colcon build --packages-select neuromesh_platform_r2
```

## Quick Launch: VGGT

```bash
# Robot 1
./scripts/vggt_model_neuromesh_launch.sh khonsu 1

# Robot 2 (separate machine)
./scripts/vggt_model_neuromesh_launch.sh anubis 2
```

Or via ROS2 launch:

```bash
ros2 launch neuromesh_platform_r2 vggt_model_neuromesh_launch.py \
    name:=khonsu \
    agent_num:=1 \
    agent_list:="khonsu,anubis" \
    color_raw_topic:=/khonsu/sensors/camera_0/camera/color/image_raw
```

## Quick Launch: DUSt3R

```bash
ros2 launch neuromesh_platform_r2 dust3r_model_neuromesh_launch.py name:=$ROBOT_NAME
```

## Quick Launch: GAT

```bash
ros2 launch neuromesh_platform_r2 gat_planner_model_neuromesh_launch.py name:=$ROBOT_NAME
```
