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
  
# Creates ultrahd (3840x2160), fullhd (1920x1080), and hd (1280x720) frames of the material in JPEG format
ffmpeg -i ../1_video_material/video_material_ultrahd.y4m \
  -frames:v 1 \
  ../1_video_material/frame_ultrahd.jpg

ffmpeg -i ../1_video_material/video_material_fullhd.y4m \
  -frames:v 1 \
  ../1_video_material/frame_fullhd.jpg

ffmpeg -i ../1_video_material/video_material_hd.y4m \
  -frames:v 1 \
  ../1_video_material/frame_hd.jpg
  
# Creates ultrahd (3840x2160), fullhd (1920x1080), and hd (1280x720) frames of the material in PNG format
ffmpeg -i ../1_video_material/video_material_ultrahd.y4m \
  -frames:v 1 \
  ../1_video_material/frame_ultrahd.png

ffmpeg -i ../1_video_material/video_material_fullhd.y4m \
  -frames:v 1 \
  ../1_video_material/frame_fullhd.png

ffmpeg -i ../1_video_material/video_material_hd.y4m \
  -frames:v 1 \
  ../1_video_material/frame_hd.png

# Creates ultrahd (3840x2160), fullhd (1920x1080), and hd (1280x720) frames of the material in TIFF format
ffmpeg -i ../1_video_material/video_material_ultrahd.y4m \
  -frames:v 1 \
  ../1_video_material/frame_ultrahd.tiff

ffmpeg -i ../1_video_material/video_material_fullhd.y4m \
  -frames:v 1 \
  ../1_video_material/frame_fullhd.tiff

ffmpeg -i ../1_video_material/video_material_hd.y4m \
  -frames:v 1 \
  ../1_video_material/frame_hd.tiff
  
  # Creates ultrahd (3840x2160), fullhd (1920x1080), and hd (1280x720) frames of the material in PPM format
ffmpeg -i ../1_video_material/video_material_ultrahd.y4m \
  -pix_fmt rgb24 \
  -frames:v 1 \
  -f rawvideo ../1_video_material/frame_ultrahd.ppm

ffmpeg -i ../1_video_material/video_material_fullhd.y4m \
  -pix_fmt rgb24 \
  -frames:v 1 \
  -f rawvideo ../1_video_material/frame_fullhd.ppm
  
ffmpeg -i ../1_video_material/video_material_hd.y4m \
  -pix_fmt rgb24 \
  -frames:v 1 \
  -f rawvideo ../1_video_material/frame_hd.ppm

# Creates ultrahd (3840x2160), fullhd (1920x1080), and hd (1280x720) frames of the material in RGB24 format
ffmpeg -i ../1_video_material/video_material_ultrahd.y4m \
  -pix_fmt rgb24 \
  -frames:v 1 \
  -f rawvideo ../1_video_material/frame_ultrahd.rgb

ffmpeg -i ../1_video_material/video_material_fullhd.y4m \
  -pix_fmt rgb24 \
  -frames:v 1 \
  -f rawvideo ../1_video_material/frame_fullhd.rgb

ffmpeg -i ../1_video_material/video_material_hd.y4m \
  -pix_fmt rgb24 \
  -frames:v 1 \
  -f rawvideo ../1_video_material/frame_hd.rgb
  
# Creates ultrahd (3840x2160), fullhd (1920x1080), and hd (1280x720) frames of the material in YUV444p planar format
ffmpeg -i ../1_video_material/video_material_ultrahd.y4m \
  -pix_fmt yuv444p \
  -frames:v 1 \
  -f rawvideo ../1_video_material/frame_ultrahd_444.yuv

ffmpeg -i ../1_video_material/video_material_fullhd.y4m \
  -pix_fmt yuv444p \
  -frames:v 1 \
  -f rawvideo ../1_video_material/frame_fullhd_444.yuv

ffmpeg -i ../1_video_material/video_material_hd.y4m \
  -pix_fmt yuv444p \
  -frames:v 1 \
  -f rawvideo ../1_video_material/frame_hd_444.yuv

# Creates ultrahd (3840x2160), fullhd (1920x1080), and hd (1280x720) frames of the material in YUV422p planar format
ffmpeg -i ../1_video_material/video_material_ultrahd.y4m \
  -pix_fmt yuv422p \
  -frames:v 1 \
  -f rawvideo ../1_video_material/frame_ultrahd_422.yuv

ffmpeg -i ../1_video_material/video_material_fullhd.y4m \
  -pix_fmt yuv422p \
  -frames:v 1 \
  -f rawvideo ../1_video_material/frame_fullhd_422.yuv

ffmpeg -i ../1_video_material/video_material_hd.y4m \
  -pix_fmt yuv422p \
  -frames:v 1 \
  -f rawvideo ../1_video_material/frame_hd_422.yuv

# Creates ultrahd (3840x2160), fullhd (1920x1080), and hd (1280x720) frames of the material in YUV420p planar format
ffmpeg -i ../1_video_material/video_material_ultrahd.y4m \
  -pix_fmt yuv420p \
  -frames:v 1 \
  -f rawvideo ../1_video_material/frame_ultrahd_420.yuv

ffmpeg -i ../1_video_material/video_material_fullhd.y4m \
  -pix_fmt yuv420p \
  -frames:v 1 \
  -f rawvideo ../1_video_material/frame_fullhd_420.yuv

ffmpeg -i ../1_video_material/video_material_hd.y4m \
  -pix_fmt yuv420p \
  -frames:v 1 \
  -f rawvideo ../1_video_material/frame_hd_420.yuv

# Creates ultrahd (3840x2160), fullhd (1920x1080), and hd (1280x720) frames of the material in NV12 format
ffmpeg -i ../1_video_material/video_material_ultrahd.y4m \
  -pix_fmt nv12 \
  -frames:v 1 \
  -f rawvideo ../1_video_material/frame_ultrahd_nv12.yuv

ffmpeg -i ../1_video_material/video_material_fullhd.y4m \
  -pix_fmt nv12 \
  -frames:v 1 \
  -f rawvideo ../1_video_material/frame_fullhd_nv12.yuv

ffmpeg -i ../1_video_material/video_material_hd.y4m \
  -pix_fmt nv12 \
  -frames:v 1 \
  -f rawvideo ../1_video_material/frame_hd_nv12.yuv



rm ../1_video_material/files.txt
