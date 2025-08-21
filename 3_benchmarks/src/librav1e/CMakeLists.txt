cmake_minimum_required(VERSION 3.10)
project(librav1e_benchmarks C)

add_executable(librav1e_encode librav1e_encode.c)
target_link_libraries(librav1e_encode avcodec avutil)


