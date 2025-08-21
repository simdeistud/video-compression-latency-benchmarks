cmake_minimum_required(VERSION 3.10)
project(libsvt-av1_benchmarks C)

add_executable(libsvt-av1_encode libsvt-av1_encode.c)
target_link_libraries(libsvt-av1_encode avcodec avutil)


