cmake_minimum_required(VERSION 3.10)
project(libavif_benchmarks CXX)

set(LIBAVIF_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../../2_libraries/_installdir/libavif/include")
set(LIBAVIF_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/../../../2_libraries/_installdir/libavif/lib")

include_directories(${LIBAVIF_INCLUDE_DIR})
link_directories(${LIBAVIF_LIBRARY_DIR})
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath=${LIBAVIF_LIBRARY_DIR}")

add_executable(libavif_encode libavif_encode.cpp)
target_link_libraries(libavif_encode avif m)

add_executable(libavif_decode libavif_decode.cpp)
target_link_libraries(libavif_decode avif m)
