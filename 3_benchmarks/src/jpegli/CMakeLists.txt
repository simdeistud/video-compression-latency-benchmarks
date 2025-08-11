cmake_minimum_required(VERSION 3.10)
project(jpegli_benchmarks C)

set(LIBJPEG_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../../2_libraries/_installdir/jpegli/include")
set(LIBJPEG_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/../../../2_libraries/_installdir/jpegli/lib")

include_directories(${LIBJPEG_INCLUDE_DIR})
link_directories(${LIBJPEG_LIBRARY_DIR})
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath=${LIBJPEG_LIBRARY_DIR}")

add_executable(jpegli_encode jpegli_encode.c)
target_link_libraries(jpegli_encode jpeg)

add_executable(jpegli_decode jpegli_decode.c)
target_link_libraries(jpegli_decode jpeg)
