# Real-World Deployment

NeuroMesh is demonstrated using two examples:

1. **Collaborative Perception** using DUSt3R (3D point cloud reconstruction from multiple robots)
2. **Collaborative Goal Assignment** using Graph Attention Network (GAT)

Follow these instructions while having real-world agents ready (ground robots and/or UAVs).

## Requirements

### Third-Party Software

| Dependency | Version |
|---|---|
| TensorRT | 8.6.1 |
| NVIDIA CUDA | 12.1 |
| ROS2 | Jazzy |
| Zenoh | 1.0.0-dev |
| zenoh-bridge-ros2dds | v1.0.0-dev-34-gca4a1f2 |

Optional:
- **tmux** — convenient multi-pane terminal session management
- **Rviz2** — point cloud visualization

### Per-Agent Software

- A ROS2 node publishing camera images (`sensor_msgs/Image`)
- A ROS2 control node (if running goal assignment)

## Setup Steps (Per Robot)

### 1. Build NeuroMesh

```bash
colcon build --packages-up-to neuromesh_platform_r2
source install/setup.bash
```

### 2. Export Environment Variables

Set these on every terminal that runs a ROS2 node. The `ROS_DOMAIN_ID` must be
**unique per robot**:

```bash
export ROS_DOMAIN_ID=80
export ROBOT_NAME=race16
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

### 3. Configure Zenoh

Edit `config/zenoh_config.json5`:

- Set `plugins -> ros2dds -> domain` to match your `ROS_DOMAIN_ID`
- Add the IP addresses of all robots in `connect -> endpoints`:

  ```json
  // connect section in zenoh_config.json5
  "connect": {
    "endpoints": [
      "tcp/192.168.1.10:7447",
      "tcp/192.168.1.11:7447"
    ]
  }
  ```

  Find IPs using `ifconfig` on each robot.

### 4. Verify Camera Topic

Confirm the camera topic name in `launch/dust3r_model_neuromesh_launch.py` (in
the `ComposableNode` parameters near the top of the file) matches your robot's
camera topic.

### 5. Prepare Models

1. Obtain ONNX encoder and decoder model files
2. Convert to TensorRT on each robot (see [Build Instructions](../build.md))
3. Place `.engine` files in `tensorrt_engine/models/`

### 6. Rebuild After Model Placement

```bash
colcon build --packages-select tensorrt_engine neuromesh_platform_r2
```

## Perception using DUSt3R

Launch on each robot using the tmux script (recommended for multi-pane setup):

```bash
./scripts/dust3r_model_neuromesh_launch.sh
```

Or manually in order (with sufficient time between each step):

```bash
# Pane 1 — environment
export ROS_DOMAIN_ID=80; export ROBOT_NAME=bumblebee
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source install/setup.bash

# Pane 2 — camera node (depends on your hardware)
ros2 run <your_camera_package> <camera_node>

# Pane 3 — Zenoh bridge
zenoh-bridge-ros2dds -c config/zenoh_config.json5

# Pane 4 — NeuroMesh pipeline
ros2 launch neuromesh_platform_r2 full_except_bridge.py \
    feature_subscribe_topic:=/peer_robot/features_peer_robot

# Pane 5 — DUSt3R example
ros2 launch neuromesh_platform_r2 dust3r_model_neuromesh_launch.py \
    name:=$ROBOT_NAME
```

### Visualize Point Cloud (Optional)

1. Open Rviz2
2. Set the global frame to `$ROBOT_NAME/map`
3. Add a **PointCloud2** display and subscribe to:
   - `/$ROBOT_NAME/res1_pts3d_cloud`, or
   - `/$ROBOT_NAME/res2_pts3d_cloud`

## Collaborative Goal Assignment using GAT

```bash
ros2 launch neuromesh_platform_r2 gat_planner_model_neuromesh_launch.py \
    name:=$ROBOT_NAME
```

Start positions and goal positions are configured in:
- `config/gat_planner_start_pos.yaml`
- `config/gat_planner_goal1_pos.yaml`
