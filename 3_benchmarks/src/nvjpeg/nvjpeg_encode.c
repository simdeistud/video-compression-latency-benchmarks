// nvjpeg_bench_throughput.c
#include <nvjpeg.h>
#include <cuda_runtime_api.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "../img_utils.h"

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

static void print_syntax(void)
{
    printf("\nProgram Input Syntax:\n\n");
    printf("  ┌──────────────────────────────────────┐\n");
    printf("  │ -w  <width [px]>                     │\n");
    printf("  │ -h  <height [px]>                    │\n");
    printf("  │ -s  <subsampling [444|422|420]>      │\n");
    printf("  │ -q  <quality [10...100]>             │\n");
    printf("  │ -i  <iterations [1...n]>             │\n");
    printf("  │ -b  <benchmark mode>                 │\n");
    printf("  │ -o  <output mode [FILEPATH|-]>       │\n");
    printf("  └──────────────────────────────────────┘\n");
}

static nvjpegChromaSubsampling_t get_subsampling(const char* subsampling_str)
{
    if (!subsampling_str) return NVJPEG_CSS_UNKNOWN;
    switch (atoi(subsampling_str))
    {
    case 444: return NVJPEG_CSS_444;
    case 422: return NVJPEG_CSS_422;
    case 420: return NVJPEG_CSS_420;
    default: return NVJPEG_CSS_UNKNOWN;
    }
}

static double timespec_diff_sec(struct timespec a, struct timespec b)
{
    double sec = (double)(b.tv_sec - a.tv_sec);
    double nsec = (double)(b.tv_nsec - a.tv_nsec) / 1e9;
    return sec + nsec;
}

int main(int argc, char** argv)
{
    /* Inputs */
    unsigned char* inbuf = NULL; // raw RGB from stdin
    size_t inbuf_size = 0;

    /* Double-buffer pinned inputs */
    unsigned char* pinned_in[2] = {NULL, NULL};

    /* Double-buffer bitstreams */
    unsigned char* bitstreams[2] = {NULL, NULL};
    size_t caps[2] = {0, 0};
    size_t last_size[2] = {0, 0};

    /* nvJPEG encoder state */
    nvjpegHandle_t handle = NULL;
    nvjpegEncoderState_t encoder_state = NULL;
    nvjpegEncoderParams_t encoder_params = NULL;

    /* CUDA objs */
    cudaStream_t stream = 0;
    cudaEvent_t evt_start = 0, evt_stop = 0;

    /* Benchmark bookkeeping */
    struct timespec setup_t0, setup_t1;
    struct timespec cleanup_t0, cleanup_t1;
    double gpu_ms_total = 0.0;
    double e2e_sec_total = 0.0;

    /* CLI */
    int width = 0, height = 0, quality = 0, iterations = 0;
    char *subsampling_str = NULL, *output = NULL;
    int benchmark = 0;
    int opt;


    while ((opt = getopt(argc, argv, "w:h:s:q:i:bo:")) != -1)
    {
        switch (opt)
        {
        case 'w': width = atoi(optarg);
            break;
        case 'h': height = atoi(optarg);
            break;
        case 's': subsampling_str = optarg;
            break;
        case 'q': quality = atoi(optarg);
            break;
        case 'i': iterations = atoi(optarg);
            break;
        case 'b': benchmark = 1;
            break;
        case 'o': output = optarg;
            break;
        default: print_syntax();
            return 1;
        }
    }

    if (width <= 0 || height <= 0 ||
        quality < 10 || quality > 100 ||
        iterations <= 0 ||
        get_subsampling(subsampling_str) == NVJPEG_CSS_UNKNOWN)
    {
        fprintf(stderr, "Invalid arguments\n");
        print_syntax();
        return 1;
    }

    // Input raw RGB
    if (img_load_stdin(&inbuf, &inbuf_size))
    {
        fprintf(stderr, "Error reading input\n");
        return 1;
    }
    size_t expected = (size_t)width * (size_t)height * 3;
    if (inbuf_size < expected)
    {
        fprintf(stderr, "Error: not enough RGB data\n");
        free(inbuf);
        return 1;
    }

    // -------------------- SETUP --------------------
    clock_gettime(CLOCK_MONOTONIC, &setup_t0);

    CHECK_NVJPEG(nvjpegCreateSimple(&handle));
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUDA(cudaEventCreate(&evt_start));
    CHECK_CUDA(cudaEventCreate(&evt_stop));

    CHECK_NVJPEG(nvjpegEncoderStateCreate(handle, &encoder_state, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle, &encoder_params, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(encoder_params, NVJPEG_ENCODING_BASELINE_DCT, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(encoder_params, quality, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(encoder_params, 0, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(encoder_params, get_subsampling(subsampling_str), stream));

    // Create pinned input double buffers
    for (int i = 0; i < 2; i++)
    {
        CHECK_CUDA(cudaHostAlloc((void**)&pinned_in[i], inbuf_size, cudaHostAllocDefault));
        memcpy(pinned_in[i], inbuf, inbuf_size);
    }

    clock_gettime(CLOCK_MONOTONIC, &setup_t1);

    const int pitch = width * 3;

    // Warm-up: produce worst-case bitstream sizes for both buffers
    for (int i = 0; i < 2; i++)
    {
        nvjpegImage_t src;
        memset(&src, 0, sizeof(src));
        src.channel[0] = pinned_in[i];
        src.pitch[0] = pitch;

        size_t out_size = 0;

        CHECK_NVJPEG(nvjpegEncodeImage(handle, encoder_state, encoder_params,
            &src, NVJPEG_INPUT_RGBI, width, height, stream));
        CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, encoder_state, NULL, &out_size, stream));

        if (out_size > caps[i])
        {
            bitstreams[i] = (unsigned char*)realloc(bitstreams[i], out_size);
            caps[i] = out_size;
        }

        CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, encoder_state,
            bitstreams[i], &out_size, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream)); // ensure finish

        last_size[i] = out_size;
    }

    // -------------------- MEASURED LOOP --------------------
    for (int iter = 0; iter < iterations; iter++)
    {
        int idx = iter & 1;

        nvjpegImage_t src;
        memset(&src, 0, sizeof(src));
        src.channel[0] = pinned_in[idx];
        src.pitch[0] = pitch;

        size_t out_size = 0;

        struct timespec e2e_t0, e2e_t1;
        clock_gettime(CLOCK_MONOTONIC, &e2e_t0);

        CHECK_CUDA(cudaEventRecord(evt_start, stream));

        CHECK_NVJPEG(nvjpegEncodeImage(handle, encoder_state, encoder_params,
            &src, NVJPEG_INPUT_RGBI, width, height, stream));

        CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, encoder_state,
            NULL, &out_size, stream));

        if (out_size > caps[idx])
        {
            bitstreams[idx] = (unsigned char*)realloc(bitstreams[idx], out_size);
            caps[idx] = out_size;
        }

        CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, encoder_state,
            bitstreams[idx], &out_size, stream));

        CHECK_CUDA(cudaEventRecord(evt_stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        clock_gettime(CLOCK_MONOTONIC, &e2e_t1);

        last_size[idx] = out_size;

        float gpu_ms = 0.f;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, evt_start, evt_stop));
        gpu_ms_total += gpu_ms;

        e2e_sec_total += timespec_diff_sec(e2e_t0, e2e_t1);
    }

    // -------------------- CLEANUP --------------------
    clock_gettime(CLOCK_MONOTONIC, &cleanup_t0);

    CHECK_NVJPEG(nvjpegEncoderParamsDestroy(encoder_params));
    CHECK_NVJPEG(nvjpegEncoderStateDestroy(encoder_state));
    CHECK_CUDA(cudaEventDestroy(evt_start));
    CHECK_CUDA(cudaEventDestroy(evt_stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_NVJPEG(nvjpegDestroy(handle));

    clock_gettime(CLOCK_MONOTONIC, &cleanup_t1);

    if (benchmark)
    {
        double setup_time = timespec_diff_sec(setup_t0, setup_t1);
        double cleanup_time = timespec_diff_sec(cleanup_t0, cleanup_t1);
        double gpu_sec = gpu_ms_total / 1000.0;
        double total_time = setup_time + e2e_sec_total + cleanup_time;

        printf("setup:%f\n", setup_time);
        printf("encoding_gpu_stream:%f\n", gpu_sec);
        printf("encoding_end_to_end_ram_to_ram:%f\n", e2e_sec_total);
        printf("cleanup:%f\n", cleanup_time);
        printf("total:%f\n", total_time);
    }

    // Output from last buffer
    if (output)
    {
        int idx = (iterations - 1) & 1;
        if (!strcmp(output, "-"))
        {
            fwrite(bitstreams[idx], 1, last_size[idx], stdout);
        }
        else
        {
            img_save(output, &bitstreams[idx], last_size[idx]);
        }
    }

    for (int i = 0; i < 2; i++)
    {
        if (pinned_in[i]) cudaFreeHost(pinned_in[i]);
        free(bitstreams[i]);
    }
    free(inbuf);

    return 0;
}
