cmake_minimum_required(VERSION 3.10)
project(libjpeg_benchmarks C)

set(LIBJPEG_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../../2_libraries/_installdir/libjpeg/include")
set(LIBJPEG_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/../../../2_libraries/_installdir/libjpeg/lib")

include_directories(${LIBJPEG_INCLUDE_DIR})
link_directories(${LIBJPEG_LIBRARY_DIR})
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath=${LIBJPEG_LIBRARY_DIR}")

add_executable(libjpeg_encode libjpeg_encode.c)
target_link_libraries(libjpeg_encode jpeg)

add_executable(libjpeg_decode libjpeg_decode.c)
target_link_libraries(libjpeg_decode jpeg)