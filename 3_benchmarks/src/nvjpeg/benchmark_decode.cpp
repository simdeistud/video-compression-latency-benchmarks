#include <nvjpeg.h>

#include <cuda_runtime.h>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdint.h>
#include <thread>
#include <vector>
#include <cstdlib>

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA error \"" << cudaGetErrorString(err) << "\" at: " __FILE__ ":"      \
                      << __LINE__ << "\n";                                                         \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

#define CHECK_NVJPEG(call)                                                                     \
    do {                                                                                       \
        nvjpegStatus_t stat = call;                                                            \
        if (stat != NVJPEG_STATUS_SUCCESS) {                                                   \
            std::cerr << "nvJPEG error \"" << static_cast<int>(stat) << "\" at: " __FILE__ ":" \
                      << __LINE__ << "\n";                                                     \
            std::exit(EXIT_FAILURE);                                                           \
        }                                                                                      \
    } while (0)

int device_malloc(void*, void** ptr, size_t size, cudaStream_t)
{
    CHECK_CUDA(cudaMalloc(ptr, size));
    return 0;
}

int device_free(void*, void* ptr, size_t, cudaStream_t)
{
    CHECK_CUDA(cudaFree(ptr));
    return 0;
}

int pinned_malloc(void*, void** ptr, size_t size, cudaStream_t)
{
    CHECK_CUDA(cudaMallocHost(ptr, size));
    return 0;
}

int pinned_free(void*, void* ptr, size_t, cudaStream_t)
{
    CHECK_CUDA(cudaFreeHost(ptr));
    return 0;
}

struct nvjpeg_state {
    void startup()
    {
        device_allocator = {&device_malloc, &device_free, nullptr};
        pinned_allocator = {&pinned_malloc, &pinned_free, nullptr};

        const nvjpegBackend_t backend = NVJPEG_BACKEND_GPU_HYBRID;
        const int flags               = 0;
        CHECK_NVJPEG(nvjpegCreateEx(backend, nullptr, nullptr, flags, &nvjpeg_handle));
        CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state));
        CHECK_NVJPEG(nvjpegDecoderCreate(nvjpeg_handle, backend, &nvjpeg_decoder));
        CHECK_NVJPEG(
            nvjpegDecoderStateCreate(nvjpeg_handle, nvjpeg_decoder, &nvjpeg_decoupled_state));
        CHECK_NVJPEG(nvjpegBufferDeviceCreateV2(nvjpeg_handle, &device_allocator, &device_buffer));
        CHECK_NVJPEG(nvjpegBufferPinnedCreateV2(nvjpeg_handle, &pinned_allocator, &pinned_buffer));
        CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_stream));
        CHECK_NVJPEG(nvjpegDecodeParamsCreate(nvjpeg_handle, &nvjpeg_decode_params));

        CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state, device_buffer));
        CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(nvjpeg_decoupled_state, pinned_buffer));
    }

    void cleanup()
    {
        CHECK_NVJPEG(nvjpegDecodeParamsDestroy(nvjpeg_decode_params));
        CHECK_NVJPEG(nvjpegJpegStreamDestroy(jpeg_stream));
        CHECK_NVJPEG(nvjpegBufferPinnedDestroy(pinned_buffer));
        CHECK_NVJPEG(nvjpegBufferDeviceDestroy(device_buffer));
        CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_decoupled_state));
        CHECK_NVJPEG(nvjpegDecoderDestroy(nvjpeg_decoder));
        CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_state));
        CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle));
    }

    nvjpegDevAllocatorV2_t device_allocator;
    nvjpegPinnedAllocatorV2_t pinned_allocator;

    nvjpegHandle_t nvjpeg_handle;
    nvjpegJpegState_t nvjpeg_state;
    nvjpegJpegDecoder_t nvjpeg_decoder;
    nvjpegJpegState_t nvjpeg_decoupled_state;

    nvjpegBufferDevice_t device_buffer;
    nvjpegBufferPinned_t pinned_buffer;
    nvjpegJpegStream_t jpeg_stream;
    nvjpegDecodeParams_t nvjpeg_decode_params;
};

double bench_nvjpeg(const uint8_t* file_data, size_t file_size, int iterations)
{
    cudaStream_t stream = 0;

    nvjpeg_state state;
    state.startup();

    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int channels;
    nvjpegChromaSubsampling_t subsampling;
    CHECK_NVJPEG(nvjpegGetImageInfo(
        state.nvjpeg_handle,
        reinterpret_cast<const unsigned char*>(file_data),
        file_size,
        &channels,
        &subsampling,
        widths,
        heights));

    nvjpegImage_t d_img;
    for (int c = 0; c < channels; ++c) {
        CHECK_CUDA(cudaMalloc(&d_img.channel[c], widths[c] * heights[c]));
        d_img.pitch[c] = widths[c];
    }

    CHECK_NVJPEG(
        nvjpegDecodeParamsSetOutputFormat(state.nvjpeg_decode_params, NVJPEG_OUTPUT_UNCHANGED));

    const auto run_iter_test = [&]() {
        const int save_metadata = 0;
        const int save_stream   = 0;
        CHECK_NVJPEG(nvjpegJpegStreamParse(
            state.nvjpeg_handle,
            reinterpret_cast<const unsigned char*>(file_data),
            file_size,
            save_metadata,
            save_stream,
            state.jpeg_stream));

        CHECK_NVJPEG(nvjpegDecodeJpegHost(
            state.nvjpeg_handle,
            state.nvjpeg_decoder,
            state.nvjpeg_decoupled_state,
            state.nvjpeg_decode_params,
            state.jpeg_stream));

        CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(
            state.nvjpeg_handle,
            state.nvjpeg_decoder,
            state.nvjpeg_decoupled_state,
            state.jpeg_stream,
            stream));

        CHECK_NVJPEG(nvjpegDecodeJpegDevice(
            state.nvjpeg_handle,
            state.nvjpeg_decoder,
            state.nvjpeg_decoupled_state,
            &d_img,
            stream));

        CHECK_CUDA(cudaStreamSynchronize(stream));
    };

    run_iter_test(); // test run;
    
    const auto run_iter = [&]() {
        const int save_metadata = 0;
        const int save_stream   = 0;
        nvjpegJpegStreamParse(
            state.nvjpeg_handle,
            reinterpret_cast<const unsigned char*>(file_data),
            file_size,
            save_metadata,
            save_stream,
            state.jpeg_stream);

        nvjpegDecodeJpegHost(
            state.nvjpeg_handle,
            state.nvjpeg_decoder,
            state.nvjpeg_decoupled_state,
            state.nvjpeg_decode_params,
            state.jpeg_stream);

        nvjpegDecodeJpegTransferToDevice(
            state.nvjpeg_handle,
            state.nvjpeg_decoder,
            state.nvjpeg_decoupled_state,
            state.jpeg_stream,
            stream);

        nvjpegDecodeJpegDevice(
            state.nvjpeg_handle,
            state.nvjpeg_decoder,
            state.nvjpeg_decoupled_state,
            &d_img,
            stream);

        cudaStreamSynchronize(stream);
    };

    const auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        run_iter();
    }
    const auto end_time = std::chrono::high_resolution_clock::now();

    const double total_seconds = std::chrono::duration<double>(end_time - start_time).count();

    for (int c = 0; c < channels; ++c) {
        CHECK_CUDA(cudaFree(d_img.channel[c]));
    }

    state.cleanup();
    return total_seconds;
}

int main(int argc, const char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: benchmark_decode <input file> <output file> <iterations>\n";
        return EXIT_FAILURE;
    }
        std::filesystem::path input_file_path(argv[1]);

        std::ifstream input_file(input_file_path);
        if (!input_file.is_open()) {
            std::cerr << "cannot open \"" << input_file_path << "\"\n";
            return EXIT_FAILURE;
        }

        input_file.seekg(0, std::ios_base::end);
        const std::streampos input_file_size = input_file.tellg();
        input_file.seekg(0);
        uint8_t* input_file_data = nullptr;
        CHECK_CUDA(cudaMallocHost(&input_file_data, input_file_size));
        input_file.read(reinterpret_cast<char*>(input_file_data), input_file_size);
        input_file.close();
        
        int iterations = atoi(argv[3]);

        double total_time = bench_nvjpeg(input_file_data, input_file_size, iterations);

        CHECK_CUDA(cudaFreeHost(input_file_data));
        
        std::cout << total_time << "\n";
    
}
