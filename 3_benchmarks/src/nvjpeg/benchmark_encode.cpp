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
#include "../../include/img_utils.h"
#include "../../include/cuda_img_utils.h"

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

double bench_nvjpeg(const uint8_t* file_data, size_t file_size, int iterations)
{
    cudaStream_t stream = 0;
    nvjpegHandle_t handle;
    nvjpegInputFormat_t input_format = NVJPEG_INPUT_RGB;
    nvjpegEncoderState_t encoder_state;
    nvjpegEncoderParams_t encoder_params;
    nvjpegImage_t input_image;
    
    CHECK_NVJPEG(nvjpegCreateSimple(&handle));
    
    CHECK_NVJPEG(nvjpegEncoderStateCreate(
      handle,
      &encoder_state,
      stream));
    
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(
      handle,
      &encoder_params,
      stream));
      
    nvjpegEncoderParamsSetSamplingFactors(encoder_params, NVJPEG_CSS_420, stream); 
    
    size_t input_image_size = 0;
    input_image.channel[0] = (unsigned char*) cuda_img_load("/home/simonedeiana/Documents/TIROCINIO/c/2_benchmarks/samples/image_sample_ultrahd.rgb", &input_image_size);
    /*input_image.channel[1] = input_image.channel[0] + 3840 * 2160;
    input_image.channel[2] = input_image.channel[0] + 3840 * 2160 * 2;
    input_image.channel[3] = input_image.channel[0] + 3840 * 2160 * 3;*/
    input_image.pitch[0] = 3840 * 3;
    /*input_image.pitch[1] = 3840;
    input_image.pitch[2] = 3840;
    input_image.pitch[3] = 3840;*/

    const auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        CHECK_NVJPEG(nvjpegEncodeImage(handle, encoder_state, encoder_params,
    &input_image, NVJPEG_INPUT_RGBI, 3840, 2160, stream));
    }
    const auto end_time = std::chrono::high_resolution_clock::now();

    const double total_seconds = std::chrono::duration<double>(end_time - start_time).count();

    CHECK_CUDA(cudaFree(input_image.channel[0]));
    
    // get compressed stream size
size_t length;
nvjpegEncodeRetrieveBitstream(handle, encoder_state, NULL, &length, stream);
// get stream itself
cudaStreamSynchronize(stream);
unsigned char* compressed_image = (unsigned char*) malloc(length);
nvjpegEncodeRetrieveBitstream(handle, encoder_state, compressed_image, &length, 0);

// write stream to file
cudaStreamSynchronize(stream);
img_save("test.jpeg", compressed_image, length);
    
    CHECK_NVJPEG(nvjpegEncoderParamsDestroy(encoder_params));

    CHECK_NVJPEG(nvjpegEncoderStateDestroy(encoder_state));
    
    CHECK_NVJPEG(nvjpegDestroy(handle));
    return total_seconds;
}

int main(int argc, const char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: benchmark_encode <input_file> <output_file> <resolution> <quality> <subsampling> <iterations>\n";
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
        
        //int resolution = atoi(argv[3]);
        //int quality = atoi(argv[4]);
        //int subsampling = atoi(argv[5]);
        int iterations = atoi(argv[3]);

        double total_time = bench_nvjpeg(input_file_data, input_file_size, iterations);

        CHECK_CUDA(cudaFreeHost(input_file_data));
        
        std::cout << total_time << "\n";
    
}
