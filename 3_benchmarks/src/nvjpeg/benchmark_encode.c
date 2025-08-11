#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <nvjpeg.h>
#include <cuda_runtime.h>
#include "../../include/benchmark.h"
#include "../../include/img_utils.h"
#include "../../include/cuda_img_utils.h"
#include "../../include/subs_utils.h"
#include "../../include/res_utils.h"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s at: %d:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            return 1; \
        } \
    } while (0)

#define CHECK_NVJPEG(call) \
    do { \
        nvjpegStatus_t stat = call; \
        if (stat != NVJPEG_STATUS_SUCCESS) { \
            fprintf(stderr, "nvJPEG error %d at: %d:%d\n", stat, __FILE__, __LINE__); \
            return 1; \
        } \
    } while (0)

int encoder_initialization(nvjpegHandle_t* handle)
{
    CHECK_NVJPEG(nvjpegCreateSimple(handle));
    return 0;
}

int encoder_setup(nvjpegHandle_t handle, nvjpegEncoderState_t* encoder_state, cudaStream_t stream)
{        
  CHECK_NVJPEG(nvjpegEncoderStateCreate(handle, encoder_state, stream));
  return 0;
}

int encoding_parameters_setup(nvjpegHandle_t handle, nvjpegEncoderParams_t* encoder_params, cudaStream_t stream, int quality, subsampling_t subsampling)
{
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle, encoder_params, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(*encoder_params, NVJPEG_ENCODING_BASELINE_DCT, stream)); // we only encode with baseline
    CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(*encoder_params, quality, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(*encoder_params, 0, stream));  // we only encode without optimized Huffman
    CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(*encoder_params, subsampling_to_nvjpeg(subsampling), stream));
      
    return 0;
}

int encoding_parameters_cleanup()
{
    return 0;
}

int encoder_cleanup()
{
    return 0;
}

int encoder_destroy()
{
    
    return 0;
}

int encode_image_with_nvjpeg(const char* input_filename, const char* output_filename, int quality, subsampling_t subsampling, int width, int height, int iterations)
{
    uint8_t* input_buffer = NULL;
    size_t input_buffer_size = 0;
    uint8_t* output_buffer = NULL;
    size_t output_buffer_size = 0;
    cudaStream_t stream = 0;
    nvjpegHandle_t handle;
    nvjpegEncoderState_t encoder_state;
    nvjpegEncoderParams_t encoder_params;
    nvjpegImage_t nv_image;
    nvjpegInputFormat_t input_format = NVJPEG_INPUT_RGBI;
    clock_t start_time, end_time;
    double total_time;
    int i;

    if (encoder_initialization(&handle)) {
        return 1;
    }

    if(encoder_setup(handle, &encoder_state, stream)){
      return 1;
    } 
    
    input_buffer = (uint8_t*) cuda_pinned_img_load(input_filename, &input_buffer_size);
    
    encoding_parameters_setup(handle, &encoder_params, stream, quality, subsampling);
    
    if (!input_buffer) {
        fprintf(stderr, "Error: Failed to load data from file: %s\n", input_filename);
        return 1;
    }
    
    nv_image.channel[0] = (unsigned char*) input_buffer;
    nv_image.pitch[0] = width * 3;
    
    CHECK_NVJPEG(nvjpegEncodeImage(handle, encoder_state, encoder_params, &nv_image, input_format, width, height, stream));

    /* Measure encoding time */
    start_time = clock();
    for (i = 0; i < iterations; i++) {
    
        nvjpegEncodeImage(handle, encoder_state, encoder_params, &nv_image, input_format, width, height, stream);
        
    }
    end_time = clock();
    
    nvjpegEncodeRetrieveBitstream(handle, encoder_state, NULL, &output_buffer_size, stream);
    cudaStreamSynchronize(stream);
    output_buffer = (unsigned char*) malloc(output_buffer_size);
    nvjpegEncodeRetrieveBitstream(handle, encoder_state, output_buffer, &output_buffer_size, stream);
    
    img_save(output_filename, output_buffer, output_buffer_size);
    
    /* Calculate the total time and compressed size */
    total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    if (cuda_img_destroy(input_buffer)) {
          return 1;
    }
    encoding_parameters_cleanup();
    encoder_cleanup();
    encoder_destroy();
    
    printf("%f\n", total_time);
    
    return 0;
}

int main(int argc, char* argv[])
{
    const char* input_filename;
    int width, height;
    resolution_t resolution;
    int quality;
    subsampling_t subsampling;
    const char* output_filename;
    int iterations;
    input_filename = NULL;
    output_filename = NULL;
    width = 0;
    height = 0;
    resolution = RESOLUTION_INVALID;
    iterations = 0;

    if (argc != 7) {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <resolution> <quality> <subsampling> <iterations>\n", argv[0]);
        return 1;
    }
    

    input_filename = argv[1];
    output_filename = argv[2];
    resolution = string_to_resolution(argv[3]);
    quality = atoi(argv[4]);
    subsampling = string_to_subsampling(argv[5]);
    iterations = atoi(argv[6]);
    

    if (resolution == RESOLUTION_INVALID) {
        fprintf(stderr, "Error: Invalid resolution. Use one of: 1280x720, 1920x1080, 3840x2160.\n");
        return 1;
    }

    if (quality < 1 || quality > 100) {
        fprintf(stderr, "Error: Quality must be between 1 and 100.\n");
        return 1;
    }

    if (subsampling == SUBSAMPLING_INVALID) {
        fprintf(stderr, "Error: Invalid subsampling type. Use one of: 444, 422, 420, grayscale.\n");
        return 1;
    }
    
    if (subsampling == SUBSAMPLING_GRAYSCALE) {
        fprintf(stderr, "Warning: grayscale should only be used with raw grayscale data (e.g. monochrome non-debayered sensor data).\n");
    }

    if (iterations <= 0) {
        fprintf(stderr, "Error: Iterations must be a positive integer.\n");
        return 1;
    }
    

    /* Get resolution dimensions */
    resolution_to_dimensions(resolution, &width, &height);
    

    /* Encode the image */
    encode_image_with_nvjpeg(input_filename, output_filename, quality, subsampling, width, height, iterations);

    return 0;
}


