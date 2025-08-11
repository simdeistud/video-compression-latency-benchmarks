cmake_minimum_required(VERSION 3.10)
project(gpujpeg_benchmarks C)

set(GPUJPEG_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../../2_libraries/_installdir/gpujpeg/include")
set(GPUJPEG_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/../../../2_libraries/_installdir/gpujpeg/lib")

include_directories(${GPUJPEG_INCLUDE_DIR})
link_directories(${GPUJPEG_LIBRARY_DIR})
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath=${GPUJPEG_LIBRARY_DIR}")

add_executable(gpujpeg_encode gpujpeg_encode.c)
target_link_libraries(gpujpeg_encode gpujpeg cudart)

add_executable(gpujpeg_decode gpujpeg_decode.c)
target_link_libraries(gpujpeg_decode gpujpeg cudart)
