cmake_minimum_required(VERSION 3.10)
project(h264qsv_benchmarks C)

add_executable(h264qsv_encode_device h264qsv_encode_device.c)
target_link_libraries(h264qsv_encode_device avcodec avutil)

add_executable(h264qsv_encode_host h264qsv_encode_host.c)
target_link_libraries(h264qsv_encode_host avcodec avutil)

