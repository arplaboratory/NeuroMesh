# VGGT Encoder-Decoder Separation Design Document

## Executive Summary

This document outlines the design for refactoring the Visual Geometry Grounded Transformation (VGGT) system from a single monolithic node into two separate ROS2 nodes: `vggt_encoder_node` and `vggt_decoder_node`. This separation aims to eliminate async scheduling conflicts, improve modularity, and enable better scalability for multi-robot systems.

## Architecture Overview

### Current Architecture (Monolithic)
```
┌─────────────────────────────────────────┐
│          vggt_neuromesh_node            │
│  ┌─────────────┐    ┌─────────────┐    │
│  │   Encoder   │───►│   Decoder   │    │
│  └─────────────┘    └─────────────┘    │
│         │                   │           │
│         └───────┬───────────┘           │
│                 ▼                       │
│          TensorRT Service               │
└─────────────────────────────────────────┘
```

### Proposed Architecture (Separated)
```
┌─────────────────────┐         ┌─────────────────────┐
│  vggt_encoder_node  │         │  vggt_decoder_node  │
│  ┌─────────────┐    │         │  ┌─────────────┐    │
│  │   Encoder   │    │◄────────┤  │   Decoder   │    │
│  └─────────────┘    │Features │  └─────────────┘    │
│         │           │  Topic  │         │           │
│         ▼           │         │         ▼           │
│  TensorRT Service   │         │  TensorRT Service   │
└─────────────────────┘         └─────────────────────┘
```

## Node Design Details

### 1. VGGT Encoder Node

#### Purpose
Processes camera images and generates feature representations for multi-robot perception.

#### Inputs
- **Camera Image**: Subscribes to `color_raw_topic` (sensor_msgs/Image)
- **Configuration**: ROS parameters for encoder settings

#### Outputs
- **Features**: Publishes to `/robot_name/features_robot_name` (neuromesh_msgs/NeuroMeshFeatures)

#### Core Components
```cpp
class VggtEncoderNode : public rclcpp::Node {
private:
    // Subscriptions
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub_;
    
    // Publishers
    rclcpp::Publisher<neuromesh_msgs::msg::NeuroMeshFeatures>::SharedPtr feature_pub_;
    
    // TensorRT client
    rclcpp::Client<neuromesh_interfaces::srv::TensorrtRequest>::SharedPtr tensorrt_client_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr encoder_timer_;
    
    // Configuration
    double encoder_cycle_interval_;  // Configurable processing interval
    std::string encoder_model_path_;
    
    // State management
    std::shared_ptr<sensor_msgs::msg::Image> latest_image_;
    std::mutex image_mutex_;
    bool processing_in_progress_;
};
```

#### Key Methods
- `camera_callback()`: Stores latest image (non-blocking)
- `encoder_timer_callback()`: Triggers encoding at configured intervals
- `process_image()`: Preprocesses image and calls TensorRT
- `publish_features()`: Publishes encoded features with timestamp

### 2. VGGT Decoder Node

#### Purpose
Aggregates features from N robots and generates depth maps, point clouds, and pose information.

#### Inputs
- **Self Features**: From local encoder node
- **Neighbor Features**: From other robots' encoder nodes
- **Configuration**: ROS parameters for decoder settings

#### Outputs
- **Depth Images**: `/robot_name/depth_robotX` (sensor_msgs/Image)
- **Point Clouds**: `/robot_name/pointcloud_current`, `/robot_name/pointcloud_neighbor`
- **RGB Point Clouds**: With color information
- **Pose Information**: If needed

#### Core Components
```cpp
class VggtDecoderNode : public rclcpp::Node {
private:
    // Subscriptions
    std::map<std::string, rclcpp::Subscription<neuromesh_msgs::msg::NeuroMeshFeatures>::SharedPtr> feature_subs_;
    
    // Publishers
    std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> depth_pubs_;
    std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> pointcloud_pubs_;
    
    // TensorRT client
    rclcpp::Client<neuromesh_interfaces::srv::TensorrtRequest>::SharedPtr tensorrt_client_;
    
    // Feature buffer
    std::map<std::string, neuromesh_msgs::msg::NeuroMeshFeatures> feature_buffer_;
    std::map<std::string, rclcpp::Time> feature_timestamps_;
    std::mutex feature_mutex_;
    
    // Configuration
    double decoder_cycle_interval_;
    double feature_age_threshold_;  // Max age for neighbor features (default: 10s)
    std::vector<std::string> robot_names_;
    int num_robots_;  // Configurable N
    
    // Timer for processing
    rclcpp::TimerBase::SharedPtr decoder_timer_;
};
```

#### Key Methods
- `feature_callback()`: Updates feature buffer for each robot
- `decoder_timer_callback()`: Triggers decoding at intervals
- `aggregate_features()`: Builds decoder tensor from N robot features
- `check_feature_freshness()`: Validates feature timestamps
- `process_decoder_output()`: Converts tensor outputs to ROS messages

### 3. Feature Aggregation Strategy

```cpp
std::vector<float> VggtDecoderNode::aggregate_features() {
    std::lock_guard<std::mutex> lock(feature_mutex_);
    std::vector<float> aggregated_tensor;
    
    // Get current robot's features (always index 0)
    auto self_features = feature_buffer_[robot_name_];
    
    // Aggregate features from all robots
    for (int i = 0; i < num_robots_; i++) {
        if (i == 0) {
            // Always use self features for index 0
            aggregated_tensor.insert(aggregated_tensor.end(), 
                                   self_features.data.begin(), 
                                   self_features.data.end());
        } else {
            // Use neighbor features or fallback to self
            std::string neighbor_name = robot_names_[i];
            
            if (feature_buffer_.count(neighbor_name) > 0) {
                auto age = this->now() - feature_timestamps_[neighbor_name];
                
                if (age.seconds() < feature_age_threshold_) {
                    // Use neighbor features
                    aggregated_tensor.insert(aggregated_tensor.end(),
                                           feature_buffer_[neighbor_name].data.begin(),
                                           feature_buffer_[neighbor_name].data.end());
                } else {
                    // Features too old, use self features
                    RCLCPP_WARN(this->get_logger(), 
                              "Features from %s are %.2f seconds old (threshold: %.2f). Using self features.",
                              neighbor_name.c_str(), age.seconds(), feature_age_threshold_);
                    aggregated_tensor.insert(aggregated_tensor.end(),
                                           self_features.data.begin(),
                                           self_features.data.end());
                }
            } else {
                // No features available, use self features
                RCLCPP_WARN(this->get_logger(), 
                          "No features available from %s. Using self features.",
                          neighbor_name.c_str());
                aggregated_tensor.insert(aggregated_tensor.end(),
                                       self_features.data.begin(),
                                       self_features.data.end());
            }
        }
    }
    
    return aggregated_tensor;
}
```

## Configuration System

### Unified Configuration File
Create a YAML configuration file that both nodes can reference:

```yaml
# config/vggt_config.yaml
vggt:
  # Encoder settings
  encoder:
    cycle_interval: 3.0  # seconds
    image_width: 518
    image_height: 392
    model_path: "$(find neuromesh_platform_r2)/../../tensorrt_engine/models/vggt_onnx_2x/vggt_image_encoder_2x.engine"
    
  # Decoder settings
  decoder:
    cycle_interval: 3.0  # seconds
    feature_age_threshold: 10.0  # seconds
    num_robots: 2  # Configurable N
    model_path: "$(find neuromesh_platform_r2)/../../tensorrt_engine/models/vggt_onnx_2x/vggt_aggregator_2x.engine"
    
  # Common settings
  robot_names: ["khonsu", "anubis"]  # List of all robots
  
  # TensorRT service settings
  tensorrt:
    service_name: "tensorrt_request"
    timeout: 30.0  # seconds
```

### Launch File Structure

```python
# launch/vggt_separated_launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, LoadComposableNodes
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('neuromesh_platform_r2')
    config_file = os.path.join(pkg_dir, 'config', 'vggt_config.yaml')
    
    # Declare launch arguments
    robot_name = LaunchConfiguration('robot_name')
    color_raw_topic = LaunchConfiguration('color_raw_topic')
    
    # Component container
    container = Node(
        package='rclcpp_components',
        executable='component_container_mt',
        name='vggt_container',
        output='screen',
        parameters=[config_file]
    )
    
    # Load encoder node
    encoder_node = ComposableNode(
        package='neuromesh_platform_r2',
        plugin='neuromesh::VggtEncoderNode',
        name='vggt_encoder',
        parameters=[
            config_file,
            {'robot_name': robot_name,
             'color_raw_topic': color_raw_topic}
        ],
        extra_arguments=[{'use_intra_process_comms': True}]
    )
    
    # Load decoder node
    decoder_node = ComposableNode(
        package='neuromesh_platform_r2',
        plugin='neuromesh::VggtDecoderNode',
        name='vggt_decoder',
        parameters=[
            config_file,
            {'robot_name': robot_name}
        ],
        extra_arguments=[{'use_intra_process_comms': True}]
    )
    
    # Load components
    load_components = LoadComposableNodes(
        target_container='vggt_container',
        composable_node_descriptions=[encoder_node, decoder_node]
    )
    
    return LaunchDescription([
        DeclareLaunchArgument('robot_name', default_value='khonsu'),
        DeclareLaunchArgument('color_raw_topic', default_value='/khonsu/color_raw'),
        container,
        load_components
    ])
```

## Implementation Roadmap

### Phase 1: Infrastructure Setup (Week 1)
1. Create new header files:
   - `include/neuromesh_platform_r2/vggt_encoder_node.h`
   - `include/neuromesh_platform_r2/vggt_decoder_node.h`
2. Create base implementations with minimal functionality
3. Set up configuration file structure
4. Create unit tests framework

### Phase 2: Encoder Node Implementation (Week 2)
1. Implement camera subscription and buffering
2. Port image preprocessing from current implementation
3. Integrate TensorRT service client for encoder
4. Implement feature publishing with proper timestamps
5. Add configurable processing intervals
6. Test encoder node independently

### Phase 3: Decoder Node Implementation (Week 3)
1. Implement multi-robot feature subscription
2. Port feature aggregation logic with N-robot support
3. Implement feature freshness checking with warnings
4. Integrate TensorRT service client for decoder
5. Port output processing (depth, pointcloud generation)
6. Test decoder node with simulated features

### Phase 4: Integration and Testing (Week 4)
1. Create integrated launch files
2. Test end-to-end system with 2 robots
3. Test with N robots (N > 2)
4. Verify backward compatibility with existing topics
5. Performance benchmarking vs monolithic node
6. Documentation and code cleanup

### Phase 5: Optional Enhancements
1. Implement health monitoring/watchdog system
2. Add dynamic reconfiguration support
3. Create diagnostic publishers
4. Add feature quality metrics
5. Implement feature compression for bandwidth optimization

## Migration Strategy

### Backward Compatibility
- Maintain existing topic names and message formats
- Keep same output formats for depth images and point clouds
- Ensure launch file parameters are compatible

### Gradual Migration
1. Deploy new nodes alongside existing monolithic node
2. Compare outputs to ensure consistency
3. Switch over once validated
4. Deprecate monolithic node

## Testing Strategy

### Unit Tests
- Test encoder preprocessing independently
- Test decoder tensor building with various feature combinations
- Test feature age validation logic

### Integration Tests
- Test encoder-decoder communication
- Test multi-robot feature aggregation
- Test failure scenarios (missing features, old features)

### System Tests
- End-to-end testing with real camera data
- Multi-robot coordination testing
- Performance comparison with monolithic approach

## Benefits of This Design

1. **Modularity**: Clean separation of encoder and decoder logic
2. **Scalability**: Easy to extend to N robots
3. **Maintainability**: Simpler codebase with focused responsibilities
4. **Flexibility**: Independent configuration and deployment
5. **Performance**: No async scheduling conflicts
6. **Debugging**: Easier to isolate issues to specific nodes

## Conclusion

This refactoring will transform the VGGT system into a more modular, scalable architecture while maintaining backward compatibility. The separation of encoder and decoder nodes eliminates the current async scheduling issues and provides a cleaner foundation for future enhancements.