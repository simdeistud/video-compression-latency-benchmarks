cmake_minimum_required(VERSION 3.10)
project(x264_benchmarks C)

add_executable(x264_encode x264_encode.c)
target_link_libraries(x264_encode avcodec avutil)


