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
    "https://media.xiph.org/video/derf/y4m/aspen_1080p.y4m"
    "https://media.xiph.org/video/derf/y4m/controlled_burn_1080p.y4m"
    "https://media.xiph.org/video/derf/y4m/crowd_run_2160p50.y4m"
    "https://media.xiph.org/video/derf/y4m/FourPeople_1280x720_60.y4m"
    "https://media.xiph.org/video/derf/y4m/old_town_cross_444_720p50.y4m"
    "https://media.xiph.org/video/derf/y4m/riverbed_1080p25.y4m"
    "https://media.xiph.org/video/derf/y4m/snow_mnt_1080p.y4m"
    "https://media.xiph.org/video/derf/y4m/720p5994_stockholm_ter.y4m"
    "https://media.xiph.org/video/derf/y4m/vidyo1_720p_60fps.y4m"
    "https://media.xiph.org/video/derf/ElFuente/Netflix_BoxingPractice_4096x2160_60fps_10bit_420.y4m"
    "https://media.xiph.org/video/derf/ElFuente/Netflix_FoodMarket2_4096x2160_60fps_10bit_420.y4m"
)

# Download each file with a numbered filename
for i in "${!urls[@]}"; do
    index=$((i + 1))
    wget -O "$DEST_DIR/$index.y4m" "${urls[$i]}"
done

