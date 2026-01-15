
// nvjpeg_decode_nocache.c
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
    unsigned char* inbuf = NULL;
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

    // Load compressed JPEG bitstream from stdin
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

    // Setup
    clock_gettime(CLOCK_MONOTONIC, &setup_t0);
    CHECK_NVJPEG(nvjpegCreateSimple(&handle));
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUDA(cudaEventCreate(&evt_start));
    CHECK_CUDA(cudaEventCreate(&evt_stop));

    // Probe image info once to allocate output surfaces
    // (Do not mutate bitstream contents across iterations; alternate buffers only)
    CHECK_NVJPEG(nvjpegGetImageInfo(handle, inbuf, inbuf_size, &components, &subsampling, widths, heights));
    const int out_w = widths[0];
    const int out_h = heights[0];
    const int out_pitch = out_w * 3;             // RGB interleaved
    const size_t out_size = (size_t)out_pitch * (size_t)out_h; // single plane for RGBI

    // Prepare two alternating input buffers (same content, different addresses)
    unsigned char* in_alts[2] = { NULL, NULL };
    in_alts[0] = (unsigned char*)malloc(inbuf_size);
    in_alts[1] = (unsigned char*)malloc(inbuf_size);
    if (!in_alts[0] || !in_alts[1]) {
        fprintf(stderr, "malloc failed for alternate inputs\n");
        free(inbuf);
        if (in_alts[0]) free(in_alts[0]);
        if (in_alts[1]) free(in_alts[1]);
        CHECK_CUDA(cudaEventDestroy(evt_start));
        CHECK_CUDA(cudaEventDestroy(evt_stop));
        CHECK_CUDA(cudaStreamDestroy(stream));
        CHECK_NVJPEG(nvjpegDestroy(handle));
        return 1;
    }
    memcpy(in_alts[0], inbuf, inbuf_size);
    memcpy(in_alts[1], inbuf, inbuf_size);

    // Prepare two alternating output buffers
    unsigned char* out_alts[2] = { NULL, NULL };
    out_alts[0] = (unsigned char*)malloc(out_size);
    out_alts[1] = (unsigned char*)malloc(out_size);
    if (!out_alts[0] || !out_alts[1]) {
        fprintf(stderr, "malloc failed for outputs\n");
        free(inbuf);
        free(in_alts[0]); free(in_alts[1]);
        if (out_alts[0]) free(out_alts[0]);
        if (out_alts[1]) free(out_alts[1]);
        CHECK_CUDA(cudaEventDestroy(evt_start));
        CHECK_CUDA(cudaEventDestroy(evt_stop));
        CHECK_CUDA(cudaStreamDestroy(stream));
        CHECK_NVJPEG(nvjpegDestroy(handle));
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &setup_t1);

    // Initial decode test (optional sanity)
    {
        nvjpegJpegState_t jpeg_state = NULL;
        CHECK_NVJPEG(nvjpegJpegStateCreate(handle, &jpeg_state));
        nvjpegImage_t outimg; memset(&outimg, 0, sizeof(outimg));
        outimg.channel[0] = out_alts[0];
        outimg.pitch[0]   = out_pitch;
        CHECK_NVJPEG(nvjpegDecode(handle, jpeg_state, in_alts[0], inbuf_size, output_format, &outimg, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_NVJPEG(nvjpegJpegStateDestroy(jpeg_state));
    }

    // Decoding loop with isolation:
    // - Recreate nvjpegJpegState_t each iteration (prevents internal reuse)
    // - Alternate input/output buffers (different host addresses)
    double decode_gpu_ms_total = 0.0;

    for (int i = 0; i < iterations; ++i)
    {
        const int idx = i & 1;

        nvjpegJpegState_t jpeg_state = NULL;
        CHECK_NVJPEG(nvjpegJpegStateCreate(handle, &jpeg_state));

        nvjpegImage_t outimg; memset(&outimg, 0, sizeof(outimg));
        outimg.channel[0] = out_alts[idx];
        outimg.pitch[0]   = out_pitch;

        CHECK_CUDA(cudaEventRecord(evt_start, stream));
        CHECK_NVJPEG(nvjpegDecode(handle, jpeg_state, in_alts[idx], inbuf_size, output_format, &outimg, stream));
        CHECK_CUDA(cudaEventRecord(evt_stop, stream));
        CHECK_CUDA(cudaEventSynchronize(evt_stop));

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
        double decoding_time = decode_gpu_ms_total / 1000.0; // seconds
        double total_time = setup_time + decoding_time + cleanup_time;
        printf("setup:%f\ndecoding:%f\ncleanup:%f\ntotal:%f\n",
               setup_time, decoding_time, cleanup_time, total_time);
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
                free(in_alts[0]); free(in_alts[1]);
                free(out_alts[0]); free(out_alts[1]);
                return 1;
            }
        }
        else
        {
            // img_save likely copies; follow your original ownership pattern
            unsigned char* tmp = last_outbuf;
            if (img_save(output, &tmp, last_out_size))
            {
                free(inbuf);
                free(in_alts[0]); free(in_alts[1]);
                free(out_alts[0]); free(out_alts[1]);
                return 1;
            }
        }
    }

    // Free host buffers
    free(inbuf);
    free(in_alts[0]); free(in_alts[1]);
    free(out_alts[0]); free(out_alts[1]);

    return 0;
}
