#!/bin/bash

# Dowload libjpeg version 9f (14-01-2024)
mkdir ../2_libraries/libjpeg
wget -O ../2_libraries/libjpeg/src.zip https://ijg.org/files/jpegsr9f.zip
unzip ../2_libraries/libjpeg/src.zip -d ../2_libraries/libjpeg
rm ../2_libraries/libjpeg/src.zip
cp -r ../2_libraries/libjpeg/jpeg-9f/* ../2_libraries/libjpeg/
rm -r ../2_libraries/libjpeg/jpeg-9f

# Dowload libjpeg-turbo version 3.1.1 (10-06-2025)
mkdir ../2_libraries/libjpeg-turbo
wget -O ../2_libraries/libjpeg-turbo/src.zip https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/3.1.1.zip
unzip ../2_libraries/libjpeg-turbo/src.zip -d ../2_libraries/libjpeg-turbo
rm ../2_libraries/libjpeg-turbo/src.zip
cp -r ../2_libraries/libjpeg-turbo/libjpeg-turbo-3.1.1/* ../2_libraries/libjpeg-turbo/
rm -r ../2_libraries/libjpeg-turbo/libjpeg-turbo-3.1.1

# Dowload gpujpeg version 0.27.7 (10-07-2025)
mkdir ../2_libraries/gpujpeg
wget -O ../2_libraries/gpujpeg/src.zip https://github.com/CESNET/GPUJPEG/archive/refs/tags/v0.27.7.zip
unzip ../2_libraries/gpujpeg/src.zip -d ../2_libraries/gpujpeg
rm ../2_libraries/gpujpeg/src.zip
cp -r ../2_libraries/gpujpeg/GPUJPEG-0.27.7/* ../2_libraries/gpujpeg/
rm -r ../2_libraries/gpujpeg/GPUJPEG-0.27.7

# Download jpegli + libjxl version 0.11.1 (26-11-2024)
mkdir ../2_libraries/libjxl
mkdir ../2_libraries/jpegli
wget -O ../2_libraries/libjxl/lib.zip https://github.com/libjxl/libjxl/releases/download/v0.11.1/jxl-x64-windows-static.zip
unzip ../2_libraries/libjxl/lib.zip -d ../2_libraries/libjxl
rm ../2_libraries/libjxl/lib.zip

# Download libavif version 1.3.0 (09-05-2025)
mkdir ../2_libraries/libavif
wget -O ../2_libraries/libavif/src.zip https://github.com/AOMediaCodec/libavif/archive/refs/tags/v1.3.0.zip
unzip ../2_libraries/libavif/src.zip -d ../2_libraries/libavif
rm ../2_libraries/libavif/src.zip
cp -r ../2_libraries/libavif/libavif-1.3.0/* ../2_libraries/libavif/
rm -r ../2_libraries/libavif/libavif-1.3.0

# Download libwebp version 1.6.0 (30-06-2025)
mkdir ../2_libraries/libwebp
wget -O ../2_libraries/libwebp/src.zip https://github.com/webmproject/libwebp/archive/refs/tags/v1.6.0.zip
unzip ../2_libraries/libwebp/src.zip -d ../2_libraries/libwebp
rm ../2_libraries/libwebp/src.zip
cp -r ../2_libraries/libwebp/libwebp-1.6.0/* ../2_libraries/libwebp/
rm -r ../2_libraries/libwebp/libwebp-1.6.0

# Download libx264 version 0.165.r3222 (14-06-2025) [WINDOWS ONLY!!!]
mkdir ../2_libraries/libx264
wget -O ../2_libraries/libx264/lib.zip https://github.com/ShiftMediaProject/x264/releases/download/0.165.r3222/libx264_0.165.r3222_msvc17.zip
unzip ../2_libraries/libx264/lib.zip -d ../2_libraries/libx264/
rm ../2_libraries/libx264/lib.zip