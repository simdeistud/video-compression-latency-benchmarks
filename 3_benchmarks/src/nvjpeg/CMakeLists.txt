cmake_minimum_required(VERSION 3.10)
project(nvjpeg_benchmarks CXX)

add_executable(nvjpeg_encode nvjpeg_encode.cpp)
target_link_libraries(nvjpeg_encode nvjpeg cudart)

add_executable(nvjpeg_decode nvjpeg_decode.cpp)
target_link_libraries(nvjpeg_decode nvjpeg cudart)
