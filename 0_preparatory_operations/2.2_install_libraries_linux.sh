#!/bin/bash

mkdir ../2_libraries/_installdir

# Install libjpeg
mkdir ../2_libraries/_installdir/libjpeg
make --directory=../2_libraries/libjpeg/ install

# Install libjpeg-turbo
mkdir ../2_libraries/_installdir/libjpeg-turbo
cmake --install ../2_libraries/libjpeg-turbo/build

# Install jpegli
mkdir ../2_libraries/_installdir/jpegli
cmake --install ../2_libraries/jpegli/build

# Install libjxl
mkdir ../2_libraries/_installdir/libjxl
cmake --install ../2_libraries/libjxl/build

# Install gpujpeg
mkdir ../2_libraries/_installdir/gpujpeg
cmake --install ../2_libraries/gpujpeg/build

# Install libwebp
mkdir ../2_libraries/_installdir/libwebp
cmake --install ../2_libraries/libwebp/build

# Install libavif
mkdir ../2_libraries/_installdir/libavif
cmake --install ../2_libraries/libavif/build

# Install libavif-av2
mkdir ../2_libraries/_installdir/libavif-av2
cmake --install ../2_libraries/libavif-av2/build

# Install libx264
mkdir ../2_libraries/_installdir/x264
make --directory=../2_libraries/x264/ install

# Install VC-2 reference
mkdir ../2_libraries/_installdir/vc2-reference
make --directory=../2_libraries/vc2-reference/ install

# Install VC-2 HQ
mkdir -p ../2_libraries/_installdir/vc2-hq/encode
mkdir -p ../2_libraries/_installdir/vc2-hq/decode
make --directory=../2_libraries/vc2-hq/encode install
make --directory=../2_libraries/vc2-hq/decode install

# Install VC-2 Vulkan
mkdir -p ../2_libraries/_installdir/vc2-vulkan/enc
mkdir -p ../2_libraries/_installdir/vc2-vulkan/dec
make --directory=../2_libraries/vc2-vulkan/enc install
make --directory=../2_libraries/vc2-vulkan/dec install

# Install libvmaf
meson install -C ../2_libraries/libvmaf/libvmaf/build
