cmake_minimum_required(VERSION 3.10)
project(x264_benchmarks C)

add_executable(x264_encode x264_encode.c)
target_link_libraries(x264_encode avcodec avutil)

# Define paths
set(LIBX264_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../../2_libraries/_installdir/x264/include")
set(LIBX264_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/../../../2_libraries/_installdir/x264/lib")

# Add second executable
add_executable(x264_encode_native x264_encode_native.c)

# Apply include and link directories only to this target
target_include_directories(x264_encode_native PRIVATE ${LIBX264_INCLUDE_DIR})
target_link_directories(x264_encode_native PRIVATE ${LIBX264_LIBRARY_DIR})
target_link_libraries(x264_encode_native x264)

# Set RPATH for runtime
set_target_properties(x264_encode_native PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${LIBX264_LIBRARY_DIR}"
)

