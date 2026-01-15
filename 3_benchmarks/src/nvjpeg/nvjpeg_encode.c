
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
        default:  return NVJPEG_CSS_UNKNOWN;
    }
}

static double timespec_diff_sec(struct timespec a, struct timespec b)
{
    // returns (b - a) in seconds
    double sec  = (double)(b.tv_sec - a.tv_sec);
    double nsec = (double)(b.tv_nsec - a.tv_nsec) / 1e9;
    return sec + nsec;
}

int main(int argc, char** argv)
{
    /* Input image related data */
    unsigned char* inbuf = NULL;
    size_t inbuf_size = 0;

    /* Optionally pinned input copy to reduce staging overhead */
    unsigned char* pinned_in = NULL;
    int use_pinned = 0;

    /* Output bitstream buffer (reused, grow-on-demand) */
    unsigned char* bitstream = NULL;
    size_t bitstream_cap = 0;
    size_t last_out_size = 0;

    /* nvJPEG encoder objects (reused across iterations) */
    nvjpegHandle_t handle = NULL;
    nvjpegEncoderState_t encoder_state = NULL;
    nvjpegEncoderParams_t encoder_params = NULL;

    /* CUDA stream + timing events */
    cudaStream_t stream = 0;
    cudaEvent_t evt_start = 0, evt_stop = 0;

    /* Benchmark data */
    struct timespec setup_t0 = {0}, setup_t1 = {0};
    struct timespec cleanup_t0 = {0}, cleanup_t1 = {0};

    /* Input parsing */
    int width = 0, height = 0, quality = 0, iterations = 0;
    char *subsampling_str = NULL, *output = NULL;
    int benchmark = 0;
    int opt;

    while ((opt = getopt(argc, argv, "w:h:s:q:i:bo:")) != -1)
    {
        switch (opt)
        {
        case 'w': width = atoi(optarg); break;
        case 'h': height = atoi(optarg); break;
        case 's': subsampling_str = optarg; break;
        case 'q': quality = atoi(optarg); break;
        case 'i': iterations = atoi(optarg); break;
        case 'b': benchmark = 1; break;
        case 'o': output = optarg; break;
        default:
            fprintf(stderr, "Usage error\n");
            print_syntax();
            return 1;
        }
    }

    if (width <= 0 || height <= 0 || quality < 10 || quality > 100 || iterations <= 0 ||
        get_subsampling(subsampling_str) == NVJPEG_CSS_UNKNOWN)
    {
        fprintf(stderr, "Invalid arguments\n");
        print_syntax();
        return 1;
    }

    if (img_load_stdin(&inbuf, &inbuf_size))
    {
        fprintf(stderr, "Error: Failed to load image from stdin\n");
        return 1;
    }

    // Expect raw interleaved RGB of size width*height*3
    size_t expected = (size_t)width * (size_t)height * 3;
    if (inbuf_size < expected)
    {
        fprintf(stderr, "Error: Input buffer smaller than width*height*3 (%zu < %zu)\n",
                inbuf_size, expected);
        free(inbuf);
        return 1;
    }

    // Setup (handle, stream, events, encoder state/params)
    clock_gettime(CLOCK_MONOTONIC, &setup_t0);
    CHECK_NVJPEG(nvjpegCreateSimple(&handle));
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUDA(cudaEventCreate(&evt_start));
    CHECK_CUDA(cudaEventCreate(&evt_stop));

    // Reuse state & params across iterations for steady-state throughput
    CHECK_NVJPEG(nvjpegEncoderStateCreate(handle, &encoder_state, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle, &encoder_params, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(encoder_params, NVJPEG_ENCODING_BASELINE_DCT, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(encoder_params, quality, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(encoder_params, 0, stream)); // keep same as your original
    CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(encoder_params, get_subsampling(subsampling_str), stream));

    // Try to place input in pinned host memory (optional), fallback to original on failure
    if (cudaHostAlloc((void**)&pinned_in, inbuf_size, cudaHostAllocDefault) == cudaSuccess) {
        memcpy(pinned_in, inbuf, inbuf_size);
        use_pinned = 1;
    } else {
        pinned_in = inbuf; // just alias to avoid branching in the hot path
        use_pinned = 0;
    }
    clock_gettime(CLOCK_MONOTONIC, &setup_t1);

    const int pitch = width * 3;

    // Prepare nvjpegImage (RGB interleaved -> only channel[0] and pitch[0] are relevant)
    nvjpegImage_t source;
    memset(&source, 0, sizeof(source));
    source.channel[0] = pinned_in;
    source.pitch[0]   = pitch;

    // Warm-up iterations (not counted)
    int warmup = iterations / 3;
    if (warmup < 1) warmup = 1;
    if (warmup > 3) warmup = 3;
    if (warmup > iterations) warmup = iterations;

    for (int i = 0; i < warmup; ++i) {
        size_t out_size = 0;
        CHECK_NVJPEG(nvjpegEncodeImage(handle, encoder_state, encoder_params,
                                       &source, NVJPEG_INPUT_RGBI, width, height, stream));
        CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, encoder_state, NULL, &out_size, stream));
        // Ensure bitstream capacity (grow-on-demand)
        if (out_size > bitstream_cap) {
            unsigned char* new_buf = (unsigned char*)realloc(bitstream, out_size);
            if (!new_buf) {
                fprintf(stderr, "realloc failed for output bitstream (size=%zu)\n", out_size);
                // Cleanup on failure
                if (use_pinned && pinned_in != NULL && pinned_in != inbuf) cudaFreeHost(pinned_in);
                free(inbuf);
                free(bitstream);
                CHECK_NVJPEG(nvjpegEncoderParamsDestroy(encoder_params));
                CHECK_NVJPEG(nvjpegEncoderStateDestroy(encoder_state));
                CHECK_CUDA(cudaEventDestroy(evt_start));
                CHECK_CUDA(cudaEventDestroy(evt_stop));
                CHECK_CUDA(cudaStreamDestroy(stream));
                CHECK_NVJPEG(nvjpegDestroy(handle));
                return 1;
            }
            bitstream = new_buf;
            bitstream_cap = out_size;
        }
        CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, encoder_state, bitstream, &out_size, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream)); // warm-up: block until done
        last_out_size = out_size; // keep the most recent
    }

    // Measured iterations
    int measured = iterations - warmup;
    double encode_gpu_ms_total = 0.0;

    for (int i = 0; i < measured; ++i)
    {
        size_t out_size = 0;

        CHECK_CUDA(cudaEventRecord(evt_start, stream));
        CHECK_NVJPEG(nvjpegEncodeImage(handle, encoder_state, encoder_params,
                                       &source, NVJPEG_INPUT_RGBI, width, height, stream));
        CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, encoder_state, NULL, &out_size, stream));

        // Ensure bitstream capacity (grow-on-demand)
        if (out_size > bitstream_cap) {
            unsigned char* new_buf = (unsigned char*)realloc(bitstream, out_size);
            if (!new_buf) {
                fprintf(stderr, "realloc failed for output bitstream (size=%zu)\n", out_size);
                if (use_pinned && pinned_in != NULL && pinned_in != inbuf) cudaFreeHost(pinned_in);
                free(inbuf);
                free(bitstream);
                CHECK_NVJPEG(nvjpegEncoderParamsDestroy(encoder_params));
                CHECK_NVJPEG(nvjpegEncoderStateDestroy(encoder_state));
                CHECK_CUDA(cudaEventDestroy(evt_start));
                CHECK_CUDA(cudaEventDestroy(evt_stop));
                CHECK_CUDA(cudaStreamDestroy(stream));
                CHECK_NVJPEG(nvjpegDestroy(handle));
                return 1;
            }
            bitstream = new_buf;
            bitstream_cap = out_size;
        }

        CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, encoder_state, bitstream, &out_size, stream));
        CHECK_CUDA(cudaEventRecord(evt_stop, stream));
        CHECK_CUDA(cudaEventSynchronize(evt_stop));

        float iter_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&iter_ms, evt_start, evt_stop));
        encode_gpu_ms_total += (double)iter_ms;

        last_out_size = out_size;
    }

    // Cleanup (nvJPEG & CUDA objects)
    clock_gettime(CLOCK_MONOTONIC, &cleanup_t0);
    CHECK_NVJPEG(nvjpegEncoderParamsDestroy(encoder_params));
    CHECK_NVJPEG(nvjpegEncoderStateDestroy(encoder_state));
    CHECK_CUDA(cudaEventDestroy(evt_start));
    CHECK_CUDA(cudaEventDestroy(evt_stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_NVJPEG(nvjpegDestroy(handle));
    clock_gettime(CLOCK_MONOTONIC, &cleanup_t1);

    // Benchmark reporting
    if (benchmark)
    {
        double setup_time   = timespec_diff_sec(setup_t0, setup_t1);
        double cleanup_time = timespec_diff_sec(cleanup_t0, cleanup_t1);
        double encoding_time = encode_gpu_ms_total / 1000.0; // seconds
        double total_time = setup_time + encoding_time + cleanup_time;

        double avg_ms_per_image = (measured > 0)
                                  ? (encode_gpu_ms_total / (double)measured)
                                  : 0.0;

        // Throughput metrics
        const double pixels_total = (double)width * (double)height * (double)measured;
        const double mpix_per_s = (encoding_time > 0.0)
                                  ? (pixels_total / 1e6) / encoding_time
                                  : 0.0;
        const double input_MB = (pixels_total * 3.0) / (1024.0 * 1024.0);
        const double MBps_in = (encoding_time > 0.0)
                               ? input_MB / encoding_time
                               : 0.0;

        printf("setup:%f\nencoding:%f\ncleanup:%f\ntotal:%f\n", setup_time, encoding_time, cleanup_time, total_time);
        printf("measured_iters:%d\nwarmup_iters:%d\n", measured, warmup);
        printf("avg_ms_per_image:%f\n", avg_ms_per_image);
        printf("throughput_MPix_per_s:%f\nthroughput_input_MB_per_s:%f\n", mpix_per_s, MBps_in);
    }

    // Emit the last encoded bitstream if requested
    if (output != NULL)
    {
        if (strcmp(output, "-") == 0)
        {
            size_t written = fwrite(bitstream, 1, last_out_size, stdout);
            if (written != last_out_size)
            {
                perror("Couldn't write to stdout");
                // fallthrough to frees and return error
                if (use_pinned && pinned_in != NULL && pinned_in != inbuf) cudaFreeHost(pinned_in);
                free(inbuf);
                free(bitstream);
                return 1;
            }
        }
        else
        {
            // img_save takes ownership of the pointer? (Your helper likely copies; we follow your original usage)
            if (img_save(output, &bitstream, last_out_size))
            {
                if (use_pinned && pinned_in != NULL && pinned_in != inbuf) cudaFreeHost(pinned_in);
                free(inbuf);
                free(bitstream);
                return 1;
            }
        }
    }

    // Free resources
    if (use_pinned && pinned_in != NULL && pinned_in != inbuf) cudaFreeHost(pinned_in);
    free(inbuf);
    free(bitstream);

    return 0;
}
