#!/bin/bash

# Start a new tmux session
SESSION="dust3r_neuromesh_launch"
tmux new-session -d -s $SESSION

# Rename the first window
tmux rename-window -t $SESSION:0 'dust3r_neuromesh_launch'

# Create a layout with 7 panes
tmux split-window -h -t $SESSION:0.0    # Split horizontally (creates pane 1)
tmux split-window -v -t $SESSION:0.1    # Split right pane vertically (creates pane 2)
tmux split-window -v -t $SESSION:0.2    # Split bottom-right pane vertically (creates pane 3)

# Pane 0
tmux send-keys -t $SESSION:0.0 'source install/setup.bash' C-m
tmux send-keys -t $SESSION:0.0 'ros2 run zenoh_bridge_ros2dds zenoh_bridge_ros2dds -c config/zenoh_config.json5' C-m

# Pane 1
tmux send-keys -t $SESSION:0.1 'source install/setup.bash' C-m
tmux send-keys -t $SESSION:0.1 "sleep 10; ros2 launch neuromesh_platform_r2 dust3r_model_neuromesh_launch.py name:=$ROBOT_NAME" C-m

# Pane 2 TODO: @yash please correct the following launch arguments
tmux send-keys -t $SESSION:0.2 'source install/setup.bash' C-m
tmux send-keys -t $SESSION:0.2 "sleep 5; ros2 launch neuromesh_platform_r2 full_except_bridge.py feature_subscribe_topic:=/race15/feature_agent2_local2" C-m

# Attach to the tmux session
tmux attach -t $SESSION
