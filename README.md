# NeuroMesh: A Unified Neural Inference Framework for Decentralized Multi-Robot Collaborative Learning

NeuroMesh is a general, modular, and decentralized framework for deploying multi-robot collaborative learning algorithms in real-world settings. Robot teams collaborate by sharing neural information and computational resources to achieve objectives that surpass individual robots' capabilities.

**Full documentation:** [`docs/`](docs/index.md)

## Key Features

- **General** — compatible with multiple task domains (perception, control, planning) and robotic platforms
- **Modular** — easy integration with existing robotic systems via a plugin-based inference engine
- **Decentralized** — independent operation across robots without a central coordinator
- **Efficient** — optimized for real-time operation on resource-constrained platforms
- **Interoperable** — works with heterogeneous robot teams (aerial and ground robots)

## Requirements

| Dependency | Version |
|---|---|
| NVIDIA CUDA | 12.1 |
| TensorRT | 8.6.1 |
| ONNX Runtime | 1.10.0 (built from source) |
| ROS2 | Jazzy |
| Zenoh | 1.0.0-dev |
| zenoh-bridge-ros2dds | v1.0.0-dev-34-gca4a1f2 |

Optional: **tmux** (multi-pane launch scripts), **Rviz2** (visualization).

Each agent also needs a ROS2 camera node and, if running goal assignment, a ROS2 control node.

## Quick Start

### 1. Install ONNX Runtime

ONNX Runtime must be built from source before the ROS2 workspace. A GCC 13 compatibility patch is included:

```bash
git clone --recursive https://github.com/microsoft/onnxruntime.git -b v1.10.0
cd onnxruntime && touch COLCON_IGNORE
git apply /path/to/neuromesh/onnx_engine/onnxruntime_gcc13.patch
./build.sh --config Release --build_shared_lib --parallel
cd build/Linux/Release && cmake --install . --prefix /usr/local
```

> See [`docs/build.md`](docs/build.md) for details on the patch and troubleshooting.

### 2. Convert Models to TensorRT

ONNX model files must be converted to TensorRT `.engine` files on each robot (engines are GPU-specific):

```bash
trtexec --onnx=vggt_image_encoder_2x.onnx \
        --saveEngine=tensorrt_engine/models/vggt/vggt_image_encoder_2x.engine \
        --fp16
```

> See [`docs/build.md`](docs/build.md) and [`tensorrt_engine/models/README.md`](tensorrt_engine/models/README.md).

### 3. Build the ROS2 Workspace

```bash
colcon build --packages-up-to neuromesh_platform_r2
source install/setup.bash
```

### 4. Configure Environment (per robot, per terminal)

```bash
export ROS_DOMAIN_ID=80          # must be unique per robot
export ROBOT_NAME=race16
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

### 5. Configure Zenoh

Edit `neuromesh_platform_r2/config/zenoh_config.json5`:
- Set `plugins -> ros2dds -> domain` to your `ROS_DOMAIN_ID`
- Add all robot IPs to `connect -> endpoints`

## Demos

### Collaborative Perception (DUSt3R)

```bash
ros2 launch neuromesh_platform_r2 dust3r_model_neuromesh_launch.py name:=$ROBOT_NAME
```

To visualize in Rviz2, subscribe to `/$ROBOT_NAME/res1_pts3d_cloud` or `/$ROBOT_NAME/res2_pts3d_cloud` in the `$ROBOT_NAME/map` frame.

> Full deployment walkthrough: [`docs/deployment/real_world.md`](docs/deployment/real_world.md)

### Collaborative Perception (VGGT)

```bash
# Robot 1
./neuromesh_platform_r2/scripts/vggt_model_neuromesh_launch.sh khonsu 1

# Robot 2 (separate machine)
./neuromesh_platform_r2/scripts/vggt_model_neuromesh_launch.sh anubis 2
```

> Model specs, topics, and parameters: [`docs/deployment/vggt_setup.md`](docs/deployment/vggt_setup.md)

### Collaborative Goal Assignment (GAT)

```bash
ros2 launch neuromesh_platform_r2 gat_planner_model_neuromesh_launch.py name:=$ROBOT_NAME
```

## Repository Structure

```
neuromesh/
├── neuromesh_interfaces/     # ROS2 message & service definitions
├── engine_interface/         # Abstract inference engine + ROS2 service node
├── tensorrt_engine/          # TensorRT backend plugin
├── onnx_engine/              # ONNX Runtime backend plugin
├── neuromesh_platform_r2/    # Application nodes, launch files, configs
└── docs/                     # Full Sphinx documentation
```

> Architecture overview: [`docs/architecture.md`](docs/architecture.md)
