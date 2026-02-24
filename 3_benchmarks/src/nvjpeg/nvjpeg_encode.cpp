#include <nvjpeg.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>

// ---------- error-check helpers ----------
#define CHECK_CUDA(cmd)                                                         \
do {                                                                            \
    cudaError_t e = (cmd);                                                      \
    if (e != cudaSuccess) {                                                     \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(e),     \
                __FILE__, __LINE__);                                            \
        return 1;                                                               \
    }                                                                           \
} while (0)

#define CHECK_NVJPEG(cmd)                                                       \
do {                                                                            \
    nvjpegStatus_t s = (cmd);                                                   \
    if (s != NVJPEG_STATUS_SUCCESS) {                                           \
        fprintf(stderr, "nvJPEG error: %d at %s:%d\n", (int)s,                  \
                __FILE__, __LINE__);                                            \
        return 1;                                                               \
    }                                                                           \
} while (0)

int main(int argc, char** argv)
{
    /* I/O */
    uint8_t* inbuf = NULL; // raw RGB
    uint8_t* pinned_inbuf = NULL; // raw RGB pinned buffer
    uint8_t* outbuf = NULL; // compressed output
    std::size_t inbuf_size = 0;
    std::size_t outbuf_size = 0;


    /* nvJPEG encoder state */
    nvjpegHandle_t handle = NULL;
    nvjpegEncoderState_t encoder_state = NULL;
    nvjpegEncoderParams_t encoder_params = NULL;

    /* CUDA objs */
    cudaStream_t stream = 0;

    const int width = 1280, height = 720, bpp = 3, quality = 50;
    const int pitch = width * 3;

    // RGB24 Gradient
    inbuf_size = width * height * bpp;
    inbuf = reinterpret_cast<uint8_t*>(malloc(inbuf_size));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t idx = (y * width + x) * 3;

            // Simple gradient:
            // R = horizontal ramp
            // G = vertical ramp
            // B = mix of both
            uint8_t r = static_cast<uint8_t>((x * 255) / (width - 1));
            uint8_t g = static_cast<uint8_t>((y * 255) / (height - 1));
            uint8_t b = static_cast<uint8_t>((r + g) / 2);

            inbuf[idx + 0] = r;
            inbuf[idx + 1] = g;
            inbuf[idx + 2] = b;
        }
    }


    // -------------------- SETUP --------------------
    CHECK_NVJPEG(nvjpegCreateSimple(&handle));
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CHECK_NVJPEG(nvjpegEncoderStateCreate(handle, &encoder_state, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle, &encoder_params, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(encoder_params, NVJPEG_ENCODING_BASELINE_DCT, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(encoder_params, quality, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(encoder_params, 0, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(encoder_params, NVJPEG_CSS_420, stream));

    for (int i = 0; i < 10; i++)
    {
        nvjpegImage_t curr_src;
        uint8_t* curr_img = reinterpret_cast<uint8_t*>(malloc(inbuf_size));
        memcpy(curr_img, inbuf, inbuf_size);
        uint8_t* curr_out = NULL;
        curr_src.channel[0] = curr_img;
        curr_src.pitch[0] = pitch;
        CHECK_NVJPEG(nvjpegEncodeImage(handle, encoder_state, encoder_params,
            &curr_src, NVJPEG_INPUT_RGBI, width, height, stream));
        CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, encoder_state, NULL, &outbuf_size, stream));
        curr_out = reinterpret_cast<uint8_t*>(malloc(outbuf_size));
        CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, encoder_state,
            curr_out, &outbuf_size, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream)); // ensure finish
        free(curr_img);
        outbuf = reinterpret_cast<uint8_t*>(malloc(outbuf_size));
        memcpy(outbuf, curr_out, outbuf_size);
        free(curr_out);
    }

    // -------------------- CLEANUP --------------------

    CHECK_NVJPEG(nvjpegEncoderParamsDestroy(encoder_params));
    CHECK_NVJPEG(nvjpegEncoderStateDestroy(encoder_state));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_NVJPEG(nvjpegDestroy(handle));

    // Output from last buffer
    if (outbuf)
    {
        std::ofstream outFile("compressed.jpeg", std::ios::binary);
        outFile.write(reinterpret_cast<const char*>(outbuf), outbuf_size);
        outFile.close();
    }

    if (pinned_inbuf) cudaFreeHost(pinned_inbuf);
    free(outbuf);
    free(inbuf);

    return 0;
}