#!/bin/bash

# VGGT Model Launch Script for 2-Robot Setup
# Usage: ./vggt_model_neuromesh_launch.sh [robot_name] [agent_num]

ROBOT_NAME=${1:-khonsu}
AGENT_NUM=${2:-1}
AGENT_LIST="khonsu,anubis"

echo "Launching VGGT model for robot: $ROBOT_NAME (agent $AGENT_NUM)"
echo "Agent list: $AGENT_LIST"

ros2 launch neuromesh_platform_r2 vggt_model_neuromesh_launch.py \
    name:=$ROBOT_NAME \
    agent_num:=$AGENT_NUM \
    agent_list:=$AGENT_LIST