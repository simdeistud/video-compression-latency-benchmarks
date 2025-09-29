cmake_minimum_required(VERSION 3.10)
project(libvpx-vp8_benchmarks C)

add_executable(libvpx-vp8_encode libvpx-vp8_encode.c)
target_link_libraries(libvpx-vp8_encode avcodec avutil)

add_executable(libvpx-vp8_decode libvpx-vp8_decode.c)
target_link_libraries(libvpx-vp8_decode avcodec avutil)

add_executable(libvpx-vp8_decode_native libvpx-vp8_decode_native.c)
target_link_libraries(libvpx-vp8_decode_native avcodec avutil)


