cmake_minimum_required(VERSION 3.10)
project(libaom-av1_benchmarks C)

add_executable(libaom-av1_encode libaom-av1_encode.c)
target_link_libraries(libaom-av1_encode avcodec avutil)


