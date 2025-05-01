# NeuroMesh: A Unified Neural Inference Framework for Decentralized Multi-Robot Collaborative Learning

NeuroMesh is a general, modular, and decentralized framework designed for deploying multi-robot collaborative learning algorithms in real-world settings. This open-source framework enables robot teams to collaborate by sharing neural information and computational resources for collectively achieving objectives that surpass individual robots' capabilities.

## Key Features

- General: Compatible with multiple task domains (perception, control, planning) and robotic platforms
- Modular: Flexible implementation that allows easy integration with existing robotic systems
- Decentralized: Independent operation across multiple robots without a central coordinator
- Efficient: Optimized for real-time operation on resource-constrained platforms
- Interoperable: Works with heterogeneous robot teams (aerial and ground robots)

## Real World Deployment

We demonstrate the capabilities of neuromesh using two examples, collaborative perception using DUSt3R and collaborative goal assignment using Graph Attention Network (GAT).

Follow these instructions while having real world agents ready (ground robots and/or UAVs).

## Requirements

### Third-Party Software

- TensorRT 8.6.1
- NVIDIA CUDA 12.1
- ROS2 Humble
- Zenoh 1.0.0-dev
- zenoh-bridge-ros2dds v1.0.0-dev-34-gca4a1f2
- tmux (optional: it is used for conveniently starting the required ROS2 nodes and other software, we will be using it in our demonstrations below)
- Rviz2 (optional: for visualizations)

### Agent Dependent Software

- A ROS2 node publishing images from camera.
- A ROS2 node that controls the agent based on output of neuromesh (if needed).

## Perception using DUSt3R

On each real world agent perform these steps to setup neuromesh:

1. Build neuromesh

```
colcon build --packages-up-to neuromesh_platform_r2
```

2. Export these environment variables on every terminal window where you start a ROS2 node. Note down the `ROS_DOMAIN_ID`. It must be unique on each agent:

```
export ROS_DOMAIN_ID=80
export ROBOT_NAME=bumblebee
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

3. Modify `config/zenoh_config.json5`:

- Add the `ROS_DOMAIN_ID` that you noted above in `plugins -> ros2dds -> domain` section as indicated in the file.
- Put the IPs of all the robots on the network in `connect -> endpoints` section as indicated in the file. You can find the IPs using `ifconfig` command.

4. Make sure the topic for the camera node is correct in `launch/dust3r_model_neuromesh_launch.py` in `neuromesh_platform_r2` ROS2 `ComposableNode` near the top of the file.

5. Get the encoder and decoder onnx model files on the robots.

6. Convert the encoder and decoder onnx model file to trt on each robot and save them in the models folder in the tensorrt_engine package.

7. Build the tensorrt_engine package using the same build commands in step 1.

8. Modify `dust3r_model_neuromesh_launch.sh` to launch the following software in order. Make sure there is sufficient time between them that they start sequentially. You can run these commands manually if that's what you prefer:

- Export variables in step 2 and source ROS2 workspace on all the panes.
- Camera node
- Zenoh node
  - Make sure that the config file location is correct
- NeuroMesh pipeline
  - It can be run using `ros2 launch neuromesh_platform_r2 full_except_bridge.py feature_subscribe_topic:=/race15/features_agent2_local2`
  - `feature_subscribe_topic` parameter is used to subscribe to the features of neighboring robot.
  - Feature remapping can also be done in the launch file instead of passing in `feature_subscribe_topic`.
- NeuroMesh DUSt3R exmaple launch file
  - It can be run using `ros2 launch neuromesh_platform_r2 dust3r_model_neuromesh_launch.py name:=$ROBOT_NAME`

9. Visualize point cloud output (optional)

- Open Rviz2 and subscribe to the global frame. By default it is set to `$ROBOT_NAME/map`.
- Select `/$ROBOT_NAME/res1_pts3d_cloud` or `/$ROBOT_NAME/res2_pts3d_cloud` topics in the PointCloud2 visualization topic in the displays panel.
- You should be able to see the point cloud generated from the framework.
