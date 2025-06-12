#!/bin/bash

# Launch VGGT separated encoder/decoder nodes
# Usage: ./vggt_separated_launch.sh [robot_name] [agent_num] [agent_list] [color_raw_topic]

# Default values
ROBOT_NAME=${1:-khonsu}
AGENT_NUM=${2:-1}
AGENT_LIST=${3:-khonsu,anubis}
COLOR_RAW_TOPIC=${4:-/${ROBOT_NAME}/sensors/camera_0/camera/color/image_raw}

echo "Launching VGGT separated nodes for robot: $ROBOT_NAME"
echo "Agent number: $AGENT_NUM"
echo "Agent list: $AGENT_LIST"
echo "Camera topic: $COLOR_RAW_TOPIC"

ros2 launch neuromesh_platform_r2 vggt_separated_launch.py \
    name:=$ROBOT_NAME \
    agent_num:=$AGENT_NUM \
    agent_list:=$AGENT_LIST \
    color_raw_topic:=$COLOR_RAW_TOPIC