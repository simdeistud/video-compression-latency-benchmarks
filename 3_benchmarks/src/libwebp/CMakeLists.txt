cmake_minimum_required(VERSION 3.10)
project(libwebp_benchmarks C)

set(LIBWEBP_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../../2_libraries/_installdir/libwebp/include")
set(LIBWEBP_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/../../../2_libraries/_installdir/libwebp/lib")

include_directories(${LIBWEBP_INCLUDE_DIR})
link_directories(${LIBWEBP_LIBRARY_DIR})
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath=${LIBWEBP_LIBRARY_DIR}")

add_executable(libwebp_encode libwebp_encode.c)
target_link_libraries(libwebp_encode webp)

add_executable(libwebp_decode libwebp_decode.c)
target_link_libraries(libwebp_decode webp)
