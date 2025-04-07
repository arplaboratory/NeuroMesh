# NeuroMesh: A Unified Neural Inference Framework for Decentralized Multi-Robot Collaborative Learning
NeuroMesh is a general, modular, and decentralized framework designed for deploying multi-robot collaborative learning algorithms in real-world settings. This open-source framework enables robot teams to collaborate by sharing neural information and computational resources for collectively achieving objectives that surpass individual robots' capabilities.

## Key Features
- General: Compatible with multiple task domains (perception, control, planning) and robotic platforms
- Modular: Flexible implementation that allows easy integration with existing robotic systems
- Decentralized: Independent operation across multiple robots without a central coordinator
- Efficient: Optimized for real-time operation on resource-constrained platforms
- Interoperable: Works with heterogeneous robot teams (aerial and ground robots)

## Dependency
- NVIDIA TensorRT
- NVIDIA CUDA (TODO: version)
- ROS2 Humble
- Zenoh (TODO: version)
- zenoh-bridge-ros2dds v1.0.0-dev-34-gca4a1f2
## Installation
```
# Clone the repository
git clone https://github.com/arplaboratory/NeuroMesh.git
cd NeuroMesh

# Install dependencies
./scripts/install_dependencies.sh

# Build the package
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```
## Examples

### Collaborative Depth Estimation

### Collaborative Goal Assignment



