#!/bin/bash
mkdir ../2_libraries

# Dowload libjpeg version 9f (14-01-2024)
mkdir ../2_libraries/libjpeg
wget -O ../2_libraries/libjpeg/src.tar.gz https://ijg.org/files/jpegsrc.v9f.tar.gz
tar -xf ../2_libraries/libjpeg/src.tar.gz -C ../2_libraries/libjpeg
rm ../2_libraries/libjpeg/src.tar.gz
cp -r ../2_libraries/libjpeg/jpeg-9f/* ../2_libraries/libjpeg/
rm -r ../2_libraries/libjpeg/jpeg-9f

# Dowload libjpeg-turbo version 3.1.1 (10-06-2025)
mkdir ../2_libraries/libjpeg-turbo
wget -O ../2_libraries/libjpeg-turbo/src.zip https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/3.1.1.zip
unzip ../2_libraries/libjpeg-turbo/src.zip -d ../2_libraries/libjpeg-turbo
rm ../2_libraries/libjpeg-turbo/src.zip
cp -r ../2_libraries/libjpeg-turbo/libjpeg-turbo-3.1.1/* ../2_libraries/libjpeg-turbo/
rm -r ../2_libraries/libjpeg-turbo/libjpeg-turbo-3.1.1

# Download libjxl + jpegli version 0.11.1 (26-11-2024)
mkdir ../2_libraries/jpegli-jpegli
git clone --branch v0.11.1 --depth 1 --recursive --shallow-submodules https://github.com/libjxl/libjxl.git ../2_libraries/libjxl-jpegli

# Dowload gpujpeg version 0.27.7 (10-07-2025)
mkdir ../2_libraries/gpujpeg
wget -O ../2_libraries/gpujpeg/src.zip https://github.com/CESNET/GPUJPEG/archive/refs/tags/v0.27.7.zip
unzip ../2_libraries/gpujpeg/src.zip -d ../2_libraries/gpujpeg
rm ../2_libraries/gpujpeg/src.zip
cp -r ../2_libraries/gpujpeg/GPUJPEG-0.27.7/* ../2_libraries/gpujpeg/
rm -r ../2_libraries/gpujpeg/GPUJPEG-0.27.7

# Download libavif version 1.3.0 (09-05-2025)
mkdir ../2_libraries/libavif
wget -O ../2_libraries/libavif/src.zip https://github.com/AOMediaCodec/libavif/archive/refs/tags/v1.3.0.zip
unzip ../2_libraries/libavif/src.zip -d ../2_libraries/libavif
rm ../2_libraries/libavif/src.zip
cp -r ../2_libraries/libavif/libavif-1.3.0/* ../2_libraries/libavif/
rm -r ../2_libraries/libavif/libavif-1.3.0

# Download libavif for AV2
cp -r ../2_libraries/libavif ../2_libraries/libavif-av2

# Download libwebp version 1.6.0 (30-06-2025)
mkdir ../2_libraries/libwebp
wget -O ../2_libraries/libwebp/src.zip https://github.com/webmproject/libwebp/archive/refs/tags/v1.6.0.zip
unzip ../2_libraries/libwebp/src.zip -d ../2_libraries/libwebp
rm ../2_libraries/libwebp/src.zip
cp -r ../2_libraries/libwebp/libwebp-1.6.0/* ../2_libraries/libwebp/
rm -r ../2_libraries/libwebp/libwebp-1.6.0

# Download libx264 stable
git clone --branch stable --depth 1 --recursive --shallow-submodules https://code.videolan.org/videolan/x264.git ../2_libraries/x264

# Download VC-2 HQ
git clone https://github.com/bbc/vc2hqencode.git ../2_libraries/vc2-hq/encode
git clone https://github.com/bbc/vc2hqdecode.git ../2_libraries/vc2-hq/decode

# Download VC-2 reference
git clone https://github.com/bbc/vc2-reference.git ../2_libraries/vc2-reference
cp ../2_libraries/vc2-hq/encode/m4/ax_cxx_compile_stdcxx_11.m4 ../2_libraries/vc2-reference/m4/ax_cxx_compile_stdcxx_11.m4
