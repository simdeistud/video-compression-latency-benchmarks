cmake_minimum_required(VERSION 3.10)
project(libvpx-vp9_benchmarks C)

add_executable(libvpx-vp9_encode libvpx-vp9_encode.c)
target_link_libraries(libvpx-vp9_encode avcodec avutil)


