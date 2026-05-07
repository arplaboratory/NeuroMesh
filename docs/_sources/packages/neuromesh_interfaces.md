# `neuromesh_interfaces`

ROS2 message and service definitions shared across all NeuroMesh packages.

## Messages

### `Tensor`

Raw float tensor for passing neural network inputs and outputs through the
`TensorRequest` service.

| Field | Type | Description |
|---|---|---|
| `header` | `std_msgs/Header` | Timestamp and frame ID |
| `shape` | `TensorShape` | Tensor dimensions |
| `data` | `float32[]` | Flattened tensor data |

### `TensorShape`

| Field | Type | Description |
|---|---|---|
| `dims` | `uint32[]` | Size of each dimension |

### `Feature`

Encoded neural feature vector, used for inter-robot communication.

| Field | Type | Description |
|---|---|---|
| `header` | `std_msgs/Header` | Timestamp and frame ID |
| `robot_name` | `string` | Source robot name |
| `data` | `float32[]` | Feature vector data |
| `shape` | `uint32[]` | Feature tensor shape |

### `CompressedFeature`

Bandwidth-optimized feature message using gzip compression.

### `CompressedTensor`

Compressed version of `Tensor` for network-efficient transmission.

### `StateVector`

Robot state vector (e.g., pose, velocity) for control applications.

### `CommMessage`

General-purpose communication message for inter-robot coordination.

## Services

### `TensorRequest`

The primary service for neural model inference.

**Request:**

| Field | Type | Description |
|---|---|---|
| `model_name` | `string` | Name of the model to run |
| `inputs` | `Tensor[]` | Input tensors |

**Response:**

| Field | Type | Description |
|---|---|---|
| `outputs` | `Tensor[]` | Output tensors from model |
| `success` | `bool` | Whether inference succeeded |
| `message` | `string` | Error message if failed |

## Build

```bash
colcon build --packages-select neuromesh_interfaces
```
