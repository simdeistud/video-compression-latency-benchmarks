cmake_minimum_required(VERSION 3.10)
project(nvjpeg_benchmarks C)

add_compile_options(-O0 -g)

add_executable(nvjpeg_encode nvjpeg_encode.c)
target_link_libraries(nvjpeg_encode nvjpeg cudart)

add_executable(nvjpeg_decode nvjpeg_decode.c)
target_link_libraries(nvjpeg_decode nvjpeg cudart)
