# Neuromesh Changelog

All notable changes to the Neuromesh project will be documented here.

Tags follow the format `<target>-v<major>.<minor>.<patch>-<variant>`.

---

## [pre-v0.0.1-m]
**Branch:** `vggt-no-arl` -> `main`

Initial pre-release targeting the Humble ROS distribution.

### Added
- Initial Humble compatibility support

---

## [pre-v0.1.0-m]
**Branch:** `vggt-no-arl-split` -> `main`

MVP for decoder / encoder split.

### Added
- Tested decoder / encoder split implementation
- Configurations for split architecture

---

## [gq-v0.1.0-tc]

First tagged release targeting the GQ robot platforms.

### Added
- Initial release for GQ platforms

---

## [v1.0.0-main] — current

First release support ROS2 Humble after RAL acceptance.

### Added
- Refactored inference engine to support both tensorRT and ONNX runtimes using pluggable backends.
- Removed dependency on specific autonomous robot platforms and stacks using a generic adapter interface.
- Provide full documentation for installation, configuration, and usage of the Neuromesh library.
