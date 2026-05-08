# Architecture

```mermaid
graph TD
    NI[neuromesh_interfaces<br/>ROS 2 msgs/srvs] --> EI[engine_interface]
    EI --> TRT[tensorrt_engine]
    EI --> ONNX[onnx_engine]
    NP[neuromesh_platform_r2] --> EI
    NP --> NI
    NP --> MB[mission_maestro_bridge optional]
```

NeuroMesh separates model execution from application logic.

- `engine_interface`: runtime backend abstraction.
- `neuromesh_platform_r2`: application nodes and orchestration.
- `mission_maestro_bridge`: optional integration bridge.
