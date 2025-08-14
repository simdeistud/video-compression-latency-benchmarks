#include <nvjpeg.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "../img_utils.h"

// --------- helpers ----------
#define CHECK_CUDA(cmd)                                                        \
  do {                                                                         \
    cudaError_t e = (cmd);                                                     \
    if (e != cudaSuccess) {                                                    \
      std::cerr << "CUDA error: " << cudaGetErrorString(e)                     \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CHECK_NVJPEG(cmd)                                                      \
  do {                                                                         \
    nvjpegStatus_t s = (cmd);                                                  \
    if (s != NVJPEG_STATUS_SUCCESS) {                                          \
      std::cerr << "nvJPEG error: " << (int)s                                  \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Encodes RGB24 (interleaved) frame to JPEG.
// rgb_host: pointer to host memory containing width*height*3 bytes (RGBRGB...)
// quality: [1..100]
// subsampling: one of NVJPEG_CSS_444, NVJPEG_CSS_422, NVJPEG_CSS_420, etc.
// Returns compressed bytes in 'jpeg'.
void encode_rgb24_nvjpeg(const unsigned char *inbuf,
                         int img_w, int img_h,
                         int q,
                         nvjpegChromaSubsampling_t css,
                         std::vector<unsigned char> &outbuf) {
    nvjpegHandle_t handle = nullptr;
    nvjpegEncoderState_t enc_state = nullptr;
    nvjpegEncoderParams_t enc_params = nullptr;

    // 1) Create nvJPEG handle and encoder state/params
    CHECK_NVJPEG(nvjpegCreateSimple(&handle)); // docs intro [1](https://docs.nvidia.com/cuda/nvjpeg/index.html)
    CHECK_NVJPEG(nvjpegEncoderStateCreate(handle, &enc_state, /*stream*/ nullptr));
    // encoder state [1](https://docs.nvidia.com/cuda/nvjpeg/index.html)
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle, &enc_params, /*stream*/ nullptr));
    // encoder params [1](https://docs.nvidia.com/cuda/nvjpeg/index.html)

    // 2) Set encode parameters (quality, subsampling, etc.)
    CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(enc_params, q, /*stream*/ nullptr));
    // set quality [2](https://www.clear.rice.edu/comp422/resources/cuda/html/nvjpeg/index.html)
    // Explicitly set subsampling (important with nvjpegEncodeImage)
    CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(enc_params, css, /*stream*/ nullptr));
    // set subsampling [3](https://stackoverflow.com/questions/65929613/nvjpeg-encode-packed-bgr)
    // Optional: disable optimized Huffman to improve throughput for tiny images
    // CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(enc_params, 0, nullptr));                  // perf tip [5](https://github.com/NVIDIA/CUDALibrarySamples/issues/89)

    // (Optional) Baseline vs progressive:
    // CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(enc_params, NVJPEG_ENCODING_BASELINE, nullptr));   // API exists in docs

    // 3) Copy the RGB frame to device memory (nvJPEG encoder consumes device memory in samples)
    const size_t pitch = static_cast<size_t>(img_w) * 3; // bytes per row for RGB24
    const size_t bytes = pitch * static_cast<size_t>(img_h);
    unsigned char *d_rgb = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_rgb), bytes));
    CHECK_CUDA(cudaMemcpy(d_rgb, inbuf, bytes, cudaMemcpyHostToDevice));

    // 4) Prepare nvjpegImage_t for interleaved RGB input
    nvjpegImage_t img{};
    img.channel[0] = d_rgb; // interleaved stored in channel[0]
    img.pitch[0] = static_cast<unsigned int>(pitch);

    // 5) Encode (NVJPEG_INPUT_RGBI = interleaved RGB)
    CHECK_NVJPEG(nvjpegEncodeImage(handle,
        enc_state,
        enc_params,
        &img,
        NVJPEG_INPUT_RGBI, // interleaved RGB24 input
        img_w,
        img_h,
        /*stream*/ nullptr));
    // encode call [4](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/4_CUDA_Libraries/nvJPEG_encoder/nvJPEG_encoder.cpp)

    // 6) Retrieve the bitstream size and bytes
    size_t bitstream_length = 0;
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, enc_state, nullptr, &bitstream_length, nullptr));
    // query size [1](https://docs.nvidia.com/cuda/nvjpeg/index.html)
    outbuf.resize(bitstream_length);
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, enc_state, outbuf.data(), &bitstream_length, nullptr));
    // get bytes [4](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/4_CUDA_Libraries/nvJPEG_encoder/nvJPEG_encoder.cpp)

    // 7) Cleanup
    CHECK_CUDA(cudaFree(d_rgb));
    CHECK_NVJPEG(nvjpegEncoderParamsDestroy(enc_params));
    CHECK_NVJPEG(nvjpegEncoderStateDestroy(enc_state));
    CHECK_NVJPEG(nvjpegDestroy(handle));
}

int main(int argc, char **argv) {
    /* Input image related data */
    uint8_t *inbuf;
    size_t inbuf_size = 0;
    int img_h = 0;
    int img_w = 0;

    /* Output image related data */
    uint8_t *outbuf;
    size_t outbuf_size = 0;

    /* Encoder data */
    nvjpegHandle_t handle = nullptr;
    nvjpegEncoderState_t enc_state = nullptr;
    nvjpegEncoderParams_t enc_params = nullptr;
    cudaStream_t stream = nullptr;
    uint8_t *inbuf_gpu;
    nvjpegImage_t img{};
    int q = 0;
    nvjpegChromaSubsampling_t css = NVJPEG_CSS_UNKNOWN;

    /* Benchmark data */
    clock_t start_time, end_time;
    double total_time = 0.0; /* Initialize total_time */
    int iterations = 1;

    if (argc != 6) {
        std::cerr << "\nProgram Input Syntax:\n\n"
                << "  ┌────────────────────────────────────────────────────────────┐\n"
                << "  │ Filepath          : RGB24 filepath                         │\n"
                << "  │ Resolution        : 3840 | 1920 | 1280  (16:9 only!)       │\n"
                << "  │ Iterations        : 1 ... n                                │\n"
                << "  │ Subsampling       : 444 | 422 | 420 | 0 (grayscale)        │\n"
                << "  │ Quality           : 0 ... 100                              │\n"
                << "  └────────────────────────────────────────────────────────────┘\n";
        return EXIT_FAILURE;
    }

    img_w = std::atoi(argv[2]);
    switch (img_w) {
        case 3840: img_h = 2160;
            break;
        case 1920: img_h = 1080;
            break;
        case 1280: img_h = 720;
            break;
        default: img_h = 0;
            break;
    }

    iterations = std::atoi(argv[3]);

    std::string subs = argv[4];
    if (subs == "444") css = NVJPEG_CSS_444;
    else if (subs == "422") css = NVJPEG_CSS_422;
    else if (subs == "420") css = NVJPEG_CSS_420;
    else css = NVJPEG_CSS_GRAY;

    q = std::atoi(argv[5]);

    img_load(argv[0], &inbuf, &inbuf_size);

    // 1) Create nvJPEG handle and encoder state/params
    CHECK_NVJPEG(nvjpegCreateSimple(&handle));
    CHECK_NVJPEG(nvjpegEncoderStateCreate(handle, &enc_state, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle, &enc_params, stream));

    // 2) Set encode parameters (quality, subsampling, etc.)
    /* Disable optimized Huffman to improve speed */
    CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(enc_params, NVJPEG_ENCODING_BASELINE_DCT, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(enc_params, css, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(enc_params, 0, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(enc_params, q, stream));


    // 3) Copy the RGB frame to device memory (nvJPEG encoder consumes device memory in samples)
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&inbuf_gpu), inbuf_size));
    CHECK_CUDA(cudaMemcpy(inbuf_gpu, inbuf, inbuf_size, cudaMemcpyHostToDevice));

    // 4) Prepare nvjpegImage_t for interleaved RGB input
    img.channel[0] = inbuf_gpu; // interleaved stored in channel[0]
    img.pitch[0] = static_cast<unsigned int>(static_cast<size_t>(img_w) * 3);

    // 5) Make test encode
    CHECK_NVJPEG(nvjpegEncodeImage(handle,
        enc_state,
        enc_params,
        &img,
        NVJPEG_INPUT_RGBI, // interleaved RGB24 input
        img_w,
        img_h,
        nullptr));

    start_time = clock();
    /* Compression begins here, parameters and input image
       cannot be changed until it has finished             */
    for (int i = 0; i < iterations; i++) {
        nvjpegEncodeImage(handle,
                          enc_state,
                          enc_params,
                          &img,
                          NVJPEG_INPUT_RGBI, // interleaved RGB24 input
                          img_w,
                          img_h,
                          nullptr);
    }
    /* Compression ends here, a new image can be loaded in
       the input buffer and parameters can be changed
       (if not they will remain the same)                  */
    end_time = clock();

    /* The benchmark only measures the encoding time, not the transfers between host and device memory.
     * Retrieving the bitstream is present only for testing purposes but is not included in the measurement
     * loop.
     */
    // 6) Retrieve the bitstream size and bytes
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, enc_state, nullptr, &outbuf_size, nullptr));
    outbuf = static_cast<uint8_t *>(malloc(outbuf_size));
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, enc_state, outbuf, &outbuf_size, nullptr));

    // 7) Cleanup
    CHECK_CUDA(cudaFree(inbuf_gpu));
    CHECK_NVJPEG(nvjpegEncoderParamsDestroy(enc_params));
    CHECK_NVJPEG(nvjpegEncoderStateDestroy(enc_state));
    CHECK_NVJPEG(nvjpegDestroy(handle));

    img_save("out.jpeg", &outbuf, outbuf_size);

    total_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

    img_destroy(inbuf);
    img_destroy(outbuf);

    std::cout << total_time << std::endl;

    return 0;
}
