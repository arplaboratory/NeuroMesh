# Contributing

Thank you for your interest in contributing to NeuroMesh!

## Workflow

### Branching

- Branch from `jazzy-main` for all new work.
- Use descriptive branch names: `feature/my-feature`, `fix/issue-description`.

### Development

1. Install dependencies (see the [Build](build.md) guide).
2. Make your changes with tests where applicable.
3. Run `colcon build --symlink-install` and verify no regressions.

### Pull Request

1. Push your branch and open a PR against `jazzy-main`.
2. Summarize the changes, motivation, and any relevant test results.
3. Assign at least one reviewer from the ARPL team.

### Code Style

- Follow ROS 2 C++ and Python style conventions.
- Run `ament_clang_format` / `ament_flake8` before committing.

### Reporting Issues

Open a GitHub issue with:
- A clear description of the bug or feature request.
- Steps to reproduce (for bugs).
- Expected vs. actual behaviour.
