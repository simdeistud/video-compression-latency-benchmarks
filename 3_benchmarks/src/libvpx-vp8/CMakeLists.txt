cmake_minimum_required(VERSION 3.10)
project(libvpx-vp8_benchmarks C)

add_executable(libvpx-vp8_encode libvpx-vp8_encode.c)
target_link_libraries(libvpx-vp8_encode avcodec avutil)


