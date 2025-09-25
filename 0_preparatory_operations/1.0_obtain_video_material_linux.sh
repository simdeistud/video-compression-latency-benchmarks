#!/bin/bash

# Define the destination directory relative to the script's location
DEST_DIR="$(dirname "$0")/../1_video_material"

# Create the directory if it doesn't exist
mkdir -p "$DEST_DIR"

# List of URLs to download
urls=(
    "https://media.xiph.org/video/derf/ElFuente/Netflix_Tango_4096x2160_60fps_10bit_420.y4m"
    "https://media.xiph.org/video/derf/ElFuente/Netflix_Narrator_4096x2160_60fps_10bit_420.y4m"
    "https://media.xiph.org/video/derf/ElFuente/Netflix_Crosswalk_4096x2160_60fps_10bit_420.y4m"
    "https://media.xiph.org/video/derf/ElFuente/Netflix_SquareAndTimelapse_4096x2160_60fps_10bit_420.y4m"
)

# Download each file with a numbered filename
for i in "${!urls[@]}"; do
    index=$((i + 1))
    wget -O "$DEST_DIR/$index.y4m" "${urls[$i]}"
done

