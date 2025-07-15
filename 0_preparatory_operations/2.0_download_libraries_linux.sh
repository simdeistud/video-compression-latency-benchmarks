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

# Dowload jpegli git version
git -C ../2_libraries clone https://github.com/google/jpegli.git

# Dowload sjpeg git version
git -C ../2_libraries clone https://github.com/webmproject/sjpeg.git

# Dowload gpujpeg version 0.27.7 (10-07-2025)
mkdir ../2_libraries/gpujpeg
wget -O ../2_libraries/gpujpeg/src.zip https://github.com/CESNET/GPUJPEG/archive/refs/tags/v0.25.7.zip
unzip ../2_libraries/gpujpeg/src.zip -d ../2_libraries/gpujpeg
rm ../2_libraries/gpujpeg/src.zip
cp -r ../2_libraries/gpujpeg/GPUJPEG-0.25.7/* ../2_libraries/gpujpeg/
rm -r ../2_libraries/gpujpeg/GPUJPEG-0.25.7

# Download libjxl version 0.11.1 (26-11-2024)
mkdir ../2_libraries/libjxl
wget -O ../2_libraries/libjxl/src.zip https://github.com/libjxl/libjxl/archive/refs/tags/v0.11.1.zip
unzip ../2_libraries/libjxl/src.zip -d ../2_libraries/libjxl
rm ../2_libraries/libjxl/src.zip
cp -r ../2_libraries/libjxl/libjxl-0.11.1/* ../2_libraries/libjxl/
rm -r ../2_libraries/libjxl/libjxl-0.11.1

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