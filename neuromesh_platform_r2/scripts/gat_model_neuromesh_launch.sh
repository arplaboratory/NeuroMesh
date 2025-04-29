#!/bin/bash

# Check if a robot name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <robot_name>"
    exit 1
fi

# Use the provided robot name
ROBOT_NAME=$1

# Start a new tmux session
SESSION="gat_neuromesh_launch"
tmux new-session -d -s $SESSION

# Rename the first window
tmux rename-window -t $SESSION:0 'gat_neuromesh_launch'

# Create a layout with 7 panes
tmux split-window -h -t $SESSION:0.0    # Split horizontally (creates pane 1)
tmux split-window -v -t $SESSION:0.1    # Split right pane vertically (creates pane 2)
tmux split-window -v -t $SESSION:0.2    # Split bottom-right pane vertically (creates pane 3)
tmux split-window -v -t $SESSION:0.3    # Split bottom-right pane vertically again (creates pane 4)
tmux split-window -v -t $SESSION:0.4    # Split left pane vertically (creates pane 5)
tmux split-window -v -t $SESSION:0.5    # Split bottom-left pane vertically (creates pane 6)

# Pane 2 (middle-right top)
tmux send-keys -t $SESSION:0.2 'sleep 30; ./ros2_bag_record.sh' C-m

# Attach to the tmux session
tmux attach -t $SESSION
