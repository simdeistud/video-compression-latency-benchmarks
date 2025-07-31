#!/bin/bash

echo "file 1.y4m" > ../1_video_material/files.txt
echo "file 2.y4m" >> ../1_video_material/files.txt
echo "file 3.y4m" >> ../1_video_material/files.txt
echo "file 4.y4m" >> ../1_video_material/files.txt
# Creates the y4m master video file
ffmpeg \
  -f concat -safe 0 -i ../1_video_material/files.txt \
  -vf "fps=30,select='not(mod(n\,2))',format=yuv420p" \
  -frames:v 500 \
  -pix_fmt yuv420p \
  -f yuv4mpegpipe ../1_video_material/video_material.y4m

# Creates ultrahd (3840x2160), fullhd (1920x1080), and hd (1280x720) copies of the material in YUV420p format
ffmpeg \
  -i ../1_video_material/video_material.y4m \
  -vf "scale=3840:2160" \
  -pix_fmt yuv420p \
  -f yuv4mpegpipe ../1_video_material/video_material_ultrahd.y4m

ffmpeg -i ../1_video_material/video_material.y4m \
  -vf "scale=1920:1080" \
  -pix_fmt yuv420p \
  -f yuv4mpegpipe ../1_video_material/video_material_fullhd.y4m

ffmpeg -i ../1_video_material/video_material.y4m \
  -vf "scale=1280:720" \
  -pix_fmt yuv420p \
  -f yuv4mpegpipe ../1_video_material/video_material_hd.y4m

# Creates ultrahd (3840x2160), fullhd (1920x1080), and hd (1280x720) copies of the material in RGB24 format
ffmpeg \
  -i ../1_video_material/video_material.y4m \
  -vf "scale=3840:2160" \
  -pix_fmt rgb24 \
  -f rawvideo ../1_video_material/video_material_ultrahd.rgb

ffmpeg -i ../1_video_material/video_material.y4m \
  -vf "scale=1920:1080" \
  -pix_fmt yuv420p \
  -f rawvideo ../1_video_material/video_material_fullhd.rgb

ffmpeg -i ../1_video_material/video_material.y4m \
  -vf "scale=1280:720" \
  -pix_fmt yuv420p \
  -f rawvideo ../1_video_material/video_material_hd.rgb

rm ../1_video_material/files.txt
