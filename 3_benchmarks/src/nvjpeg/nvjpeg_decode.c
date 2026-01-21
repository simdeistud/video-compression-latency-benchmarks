
// nvjpeg_decode_nocache.c
#include <nvjpeg.h>
#include <cuda_runtime_api.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "../img_utils.h"

// ---------- allocators required by nvjpegCreateEx ----------
static int dev_malloc(void **p, size_t s)
{
    return (int)cudaMalloc(p, s);
}
static int dev_free(void *p)
{
    return (int)cudaFree(p);
}
// Use simple malloc/free for nvJPEG internal pinned allocator here;
// we will explicitly allocate our I/O buffers as pinned via cudaHostAlloc.
static int host_malloc(void **p, size_t s, unsigned int flags)
{
    (void)flags;
    *p = malloc(s);
    return (*p) ? 0 : 1;
}
static int host_free(void *p)
{
    free(p);
    return 0;
}

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
    printf("  │ -i  <iterations [1...n]>             │\n");
    printf("  │ -b  <benchmark mode>                 │\n");
    printf("  │ -o  <output mode [FILEPATH|-]>       │\n");
    printf("  └──────────────────────────────────────┘\n");
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
    /* Compressed JPEG input */
    unsigned char* inbuf = NULL;     // pageable
    size_t inbuf_size = 0;

    /* Output image (RGB interleaved) */
    int widths[NVJPEG_MAX_COMPONENT] = {0};
    int heights[NVJPEG_MAX_COMPONENT] = {0};
    int components = 0;
    nvjpegChromaSubsampling_t subsampling;
    nvjpegOutputFormat_t output_format = NVJPEG_OUTPUT_RGBI;

    /* Last output pointer/size (for optional write) */
    unsigned char* last_outbuf = NULL;
    size_t last_out_size = 0;

    /* nvJPEG handle (shared) */
    nvjpegHandle_t handle = NULL;

    /* CUDA stream + timing events */
    cudaStream_t stream = 0;
    cudaEvent_t evt_start = 0, evt_stop = 0;

    /* Benchmark data */
    struct timespec setup_t0 = {0}, setup_t1 = {0};
    struct timespec cleanup_t0 = {0}, cleanup_t1 = {0};

    /* CLI */
    int iterations = 0;
    char* output = NULL;
    int benchmark = 0;
    int opt;
    while ((opt = getopt(argc, argv, "i:bo:")) != -1)
    {
        switch (opt)
        {
        case 'i': iterations = atoi(optarg); break;
        case 'b': benchmark = 1; break;
        case 'o': output = optarg; break;
        default:
            fprintf(stderr, "Usage error\n");
            print_syntax();
            return 1;
        }
    }
    if (iterations <= 0) {
        fprintf(stderr, "Invalid -i iterations\n");
        print_syntax();
        return 1;
    }

    // Load compressed JPEG bitstream from stdin to pageable RAM (one-time)
    if (img_load_stdin(&inbuf, &inbuf_size))
    {
        fprintf(stderr, "Error: Failed to load image (JPEG) from stdin\n");
        return 1;
    }
    if (inbuf_size == 0) {
        fprintf(stderr, "Error: Empty input\n");
        free(inbuf);
        return 1;
    }

    // ---- nvJPEG GPU backend setup ----
    clock_gettime(CLOCK_MONOTONIC, &setup_t0);

    nvjpegDevAllocator_t dev_alloc = { dev_malloc, dev_free };
    nvjpegPinnedAllocator_t pinned_alloc = { host_malloc, host_free };
    CHECK_NVJPEG(nvjpegCreateEx(
        NVJPEG_BACKEND_GPU_HYBRID,   // or NVJPEG_BACKEND_HARDWARE if supported
        &dev_alloc,
        &pinned_alloc,
        NVJPEG_FLAGS_DEFAULT,
        &handle
    ));

    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUDA(cudaEventCreate(&evt_start));
    CHECK_CUDA(cudaEventCreate(&evt_stop));

    // Probe JPEG info once
    CHECK_NVJPEG(nvjpegGetImageInfo(handle, inbuf, inbuf_size,
                                    &components, &subsampling,
                                    widths, heights));

    const int out_w = widths[0];
    const int out_h = heights[0];
    const int out_pitch = out_w * 3;                        // RGB interleaved
    const size_t out_size = (size_t)out_pitch * (size_t)out_h;

    // Pinned input buffers (ping-pong). Copy once from pageable to pinned.
    unsigned char* in_alts[2] = { NULL, NULL };
    CHECK_CUDA(cudaHostAlloc(&in_alts[0], inbuf_size, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&in_alts[1], inbuf_size, cudaHostAllocDefault));
    memcpy(in_alts[0], inbuf, inbuf_size);
    memcpy(in_alts[1], inbuf, inbuf_size);

    // Pinned host output buffers (ping-pong)
    unsigned char* out_alts[2] = { NULL, NULL };
    CHECK_CUDA(cudaHostAlloc(&out_alts[0], out_size, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&out_alts[1], out_size, cudaHostAllocDefault));

    // Device staging buffers (ping-pong) for decode output
    unsigned char* d_out_alts[2] = { NULL, NULL };
    CHECK_CUDA(cudaMalloc((void**)&d_out_alts[0], out_size));
    CHECK_CUDA(cudaMalloc((void**)&d_out_alts[1], out_size));

    clock_gettime(CLOCK_MONOTONIC, &setup_t1);

    // Initial decode (sanity)
    {
        nvjpegJpegState_t jpeg_state = NULL;
        CHECK_NVJPEG(nvjpegJpegStateCreate(handle, &jpeg_state));

        nvjpegImage_t outimg; memset(&outimg, 0, sizeof(outimg));
        outimg.channel[0] = d_out_alts[0];    // device destination
        outimg.pitch[0]   = out_pitch;

        CHECK_NVJPEG(nvjpegDecode(handle, jpeg_state,
                                  in_alts[0], inbuf_size,
                                  output_format, &outimg, stream));
        // bring to host
        CHECK_CUDA(cudaMemcpyAsync(out_alts[0], d_out_alts[0], out_size,
                                   cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_NVJPEG(nvjpegJpegStateDestroy(jpeg_state));
    }

    // Decode loop with both timings: GPU stream time and end-to-end RAM->RAM
    double decode_gpu_ms_total = 0.0;
    double e2e_sec_total = 0.0;

    for (int i = 0; i < iterations; ++i)
    {
        const int idx = i & 1;

        nvjpegJpegState_t jpeg_state = NULL;
        CHECK_NVJPEG(nvjpegJpegStateCreate(handle, &jpeg_state));

        nvjpegImage_t outimg; memset(&outimg, 0, sizeof(outimg));
        outimg.channel[0] = d_out_alts[idx];  // device staging
        outimg.pitch[0]   = out_pitch;

        // ---- End-to-end timer (RAM JPEG -> RAM RGB) ----
        struct timespec e2e_t0, e2e_t1;
        clock_gettime(CLOCK_MONOTONIC, &e2e_t0);

        // GPU-segment timer
        CHECK_CUDA(cudaEventRecord(evt_start, stream));

        // Decode to device
        CHECK_NVJPEG(nvjpegDecode(handle, jpeg_state,
                                  in_alts[idx], inbuf_size,
                                  output_format, &outimg, stream));

        // Device -> Host (pinned) copy
        CHECK_CUDA(cudaMemcpyAsync(out_alts[idx], d_out_alts[idx], out_size,
                                   cudaMemcpyDeviceToHost, stream));

        CHECK_CUDA(cudaEventRecord(evt_stop, stream));
        // Make sure host buffer is ready
        CHECK_CUDA(cudaStreamSynchronize(stream));

        clock_gettime(CLOCK_MONOTONIC, &e2e_t1);
        e2e_sec_total += timespec_diff_sec(e2e_t0, e2e_t1);

        float iter_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&iter_ms, evt_start, evt_stop));
        decode_gpu_ms_total += (double)iter_ms;

        CHECK_NVJPEG(nvjpegJpegStateDestroy(jpeg_state));

        last_outbuf = out_alts[idx];
        last_out_size = out_size;
    }

    // Cleanup of global objects
    clock_gettime(CLOCK_MONOTONIC, &cleanup_t0);
    CHECK_CUDA(cudaEventDestroy(evt_start));
    CHECK_CUDA(cudaEventDestroy(evt_stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_NVJPEG(nvjpegDestroy(handle));
    clock_gettime(CLOCK_MONOTONIC, &cleanup_t1);

    if (benchmark)
    {
        double setup_time   = timespec_diff_sec(setup_t0, setup_t1);
        double cleanup_time = timespec_diff_sec(cleanup_t0, cleanup_t1);
        double decoding_time_gpu = decode_gpu_ms_total / 1000.0; // seconds
        double decoding_time_e2e = e2e_sec_total;                // seconds
        double total_time = setup_time + decoding_time_e2e + cleanup_time;

        printf("setup:%f\n",   setup_time);
        printf("decoding_gpu_stream:%f\n", decoding_time_gpu);
        printf("decoding_end_to_end_ram_to_ram:%f\n", decoding_time_e2e);
        printf("cleanup:%f\n", cleanup_time);
        printf("total:%f\n",   total_time);
    }

    // Output (last decoded image)
    if (output != NULL)
    {
        if (strcmp(output, "-") == 0)
        {
            size_t written = fwrite(last_outbuf, 1, last_out_size, stdout);
            if (written != last_out_size) {
                perror("Couldn't write to stdout");
                // fallthrough to frees and return error
                free(inbuf);
                cudaFreeHost(in_alts[0]); cudaFreeHost(in_alts[1]);
                cudaFreeHost(out_alts[0]); cudaFreeHost(out_alts[1]);
                cudaFree(d_out_alts[0]);   cudaFree(d_out_alts[1]);
                return 1;
            }
        }
        else
        {
            unsigned char* tmp = last_outbuf;
            if (img_save(output, &tmp, last_out_size))
            {
                free(inbuf);
                cudaFreeHost(in_alts[0]); cudaFreeHost(in_alts[1]);
                cudaFreeHost(out_alts[0]); cudaFreeHost(out_alts[1]);
                cudaFree(d_out_alts[0]);   cudaFree(d_out_alts[1]);
                return 1;
            }
        }
    }

    // Free buffers
    free(inbuf);
    cudaFreeHost(in_alts[0]); cudaFreeHost(in_alts[1]);
    cudaFreeHost(out_alts[0]); cudaFreeHost(out_alts[1]);
    cudaFree(d_out_alts[0]);   cudaFree(d_out_alts[1]);

    return 0;
}
