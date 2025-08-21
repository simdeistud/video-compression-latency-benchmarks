cmake_minimum_required(VERSION 3.10)
project(av1_nvenc_benchmarks C)

add_executable(av1_nvenc_encode av1_nvenc_encode.c)
target_link_libraries(av1_nvenc_encode avcodec avutil)


