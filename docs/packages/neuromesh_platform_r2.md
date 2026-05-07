# `neuromesh_platform_r2`

The main application package. Contains all ROS2 nodes that implement
collaborative multi-robot perception and control using neural networks.

## Node Reference

### `VggtEncoderNode`

Subscribes to a camera topic, preprocesses images, and publishes encoded feature
tensors for multi-robot perception.

**Subscribed Topics**

| Topic | Type | Description |
|---|---|---|
| `color_raw_topic` (param) | `sensor_msgs/Image` | Input RGB image |

**Published Topics**

| Topic | Type | Description |
|---|---|---|
| `/{robot_name}/features_{robot_name}` | `neuromesh_interfaces/Feature` | Encoded patch tokens |

**Parameters**

| Parameter | Default | Description |
|---|---|---|
| `robot_name` | `khonsu` | Unique robot identifier |
| `color_raw_topic` | `/khonsu/color_raw` | Camera image topic |
| `encoder_cycle_interval` | `3.0` | Encoding period (seconds) |
| `encoder_model_path` | — | Path to `.engine` encoder file |

---

### `VggtDecoderNode`

Aggregates feature tensors from N robots and runs the VGGT decoder to produce
depth maps and 3D point clouds.

**Subscribed Topics**

| Topic | Type | Description |
|---|---|---|
| `/{robot}/features_{robot}` | `Feature` | Per-robot encoded features (one subscription per robot) |

**Published Topics**

| Topic | Type | Description |
|---|---|---|
| `/{robot_name}/depth_robot1` | `sensor_msgs/Image` | Depth image (self) |
| `/{robot_name}/depth_robot2` | `sensor_msgs/Image` | Depth image (neighbor) |
| `/{robot_name}/pointcloud_current` | `sensor_msgs/PointCloud2` | 3D points (self) |
| `/{robot_name}/pointcloud_neighbor` | `sensor_msgs/PointCloud2` | 3D points (neighbor) |
| `/{robot_name}/pointcloud_current_rgb` | `sensor_msgs/PointCloud2` | RGB pointcloud (self) |
| `/{robot_name}/pointcloud_neighbor_rgb` | `sensor_msgs/PointCloud2` | RGB pointcloud (neighbor) |

**Parameters**

| Parameter | Default | Description |
|---|---|---|
| `robot_name` | `khonsu` | Unique robot identifier |
| `agent_list` | `"khonsu,anubis"` | Comma-separated list of all robots |
| `decoder_cycle_interval` | `3.0` | Decoding period (seconds) |
| `feature_age_threshold` | `10.0` | Max age (seconds) before falling back to self features |
| `decoder_model_path` | — | Path to `.engine` decoder file |

---

### `Dust3rNeuromeshNode`

Collaborative 3D scene reconstruction using DUSt3R. Subscribes to camera images
and neighbor features, produces fused point clouds.

**Key Topics**

| Topic | Description |
|---|---|
| `/{robot_name}/res1_pts3d_cloud` | Point cloud from view 1 |
| `/{robot_name}/res2_pts3d_cloud` | Point cloud from view 2 |

---

### `GatNeuromeshNode`

Runs a Graph Attention Network (GAT) over shared robot states to produce a
fused state representation.

---

### `GatPlannerNeuromeshNode`

Uses the GAT model output to generate goal assignments for multi-robot
coordination.

---

### `ControlNeuromeshNode`

Translates NeuroMesh planner output into robot velocity commands.

---

### `CompressionNode` / `DecompressionNode`

gzip-based compression nodes for reducing feature message bandwidth. Uses the
bundled `gzip-hpp` header-only library.

---

### `VisualizationNode`

Publishes Rviz2 markers and visualization messages for debugging.

## Launch Files

| File | Description |
|---|---|
| `full_ros2.py` | Full NeuroMesh stack (all nodes) |
| `full_except_bridge.py` | Full stack without zenoh bridge |
| `vggt_model_neuromesh_launch.py` | VGGT perception (monolithic) |
| `vggt_separated_launch.py` | VGGT with separated encoder/decoder nodes |
| `dust3r_model_neuromesh_launch.py` | DUSt3R collaborative perception |
| `gat_model_neuromesh_launch.py` | GAT goal assignment |
| `gat_planner_model_neuromesh_launch.py` | GAT planner |
| `control_launch.py` | Robot control integration |
| `depth_completion_launch.py` | Depth completion pipeline |

## Configuration Files

| File | Description |
|---|---|
| `config/vggt_config.yaml` | VGGT encoder/decoder parameters |
| `config/zenoh_config.json5` | Zenoh network bridge configuration |
| `config/example_qos_override.yaml` | Example QoS profile overrides |
| `config/gat_planner_goal1_pos.yaml` | GAT planner goal position 1 |
| `config/gat_planner_start_pos.yaml` | GAT planner start position |

## Build

```bash
colcon build --packages-select neuromesh_platform_r2
source install/setup.bash
```
