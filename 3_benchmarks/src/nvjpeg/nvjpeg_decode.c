#include <nvjpeg.h>
#include <cuda_runtime_api.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "../img_utils.h"

// ---------- error-check helpers ----------
#define CHECK_CUDA(cmd)                                                        \
  do {                                                                         \
    cudaError_t e = (cmd);                                                     \
    if (e != cudaSuccess) {                                                    \
      printf("CUDA error: %s \n", cudaGetErrorString(e));                      \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_NVJPEG(cmd)                                                      \
  do {                                                                         \
    nvjpegStatus_t s = (cmd);                                                  \
    if (s != NVJPEG_STATUS_SUCCESS) {                                          \
      printf("nvJPEG error: %d \n", (int)s);                                   \
    }                                                                          \
  } while (0)

void print_syntax()
{
    printf("\nProgram Input Syntax:\n\n");
    printf("  ┌──────────────────────────────────────┐\n");
    printf("  │ -i  <iterations [1...n]>             │\n");
    printf("  │ -b  <benchmark mode>                 │\n");
    printf("  │ -o  <output mode [FILEPATH|-]>       │\n");
    printf("  └──────────────────────────────────────┘\n");
}

int main(int argc, char** argv)
{
    /* Input image related data */
    unsigned char* inbuf = NULL;
    size_t inbuf_size = 0;

    /* Output image related data */
    nvjpegImage_t outimg;
    unsigned char* outbuf = NULL;
    size_t outbuf_size = 0;
    /* Decoder data */
    nvjpegHandle_t handle;
    nvjpegJpegState_t jpeg_handle;
    nvjpegChromaSubsampling_t subsampling;
    int components = 0;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    nvjpegOutputFormat_t output_format = NVJPEG_OUTPUT_RGBI;

    /* Benchmark data */
    clock_t setup_start_time, setup_end_time;
    clock_t decoding_start_time, decoding_end_time;
    clock_t cleanup_start_time, cleanup_end_time;

    int iterations = 0;
    char* output = NULL;
    int benchmark = 0;
    int opt;

    while ((opt = getopt(argc, argv, "i:bo:")) != -1)
    {
        switch (opt)
        {
        case 'i': iterations = atoi(optarg);
            break;
        case 'b': benchmark = 1;
            break;
        case 'o': output = optarg;
            break;
        default:
            fprintf(stderr, "Usage error\n");
            print_syntax();
            exit(EXIT_FAILURE);
        }
    }

    if (img_load_stdin(&inbuf, &inbuf_size))
    {
        fprintf(stderr, "Error: Failed to load image from stdin\n");
        return 1;
    }

    setup_start_time = clock();
    /* Decoder setup starts here */
    CHECK_NVJPEG(nvjpegCreateSimple(&handle));
    CHECK_NVJPEG(nvjpegJpegStateCreate(handle, &jpeg_handle));
    CHECK_NVJPEG(nvjpegGetImageInfo(handle, inbuf, inbuf_size, &components, &subsampling, widths, heights));
    outimg.pitch[0] = widths[0] * 3;
    outimg.pitch[1] = widths[0] * 3;
    outimg.pitch[2] = widths[0] * 3;
    outbuf_size = outimg.pitch[0] * heights[0] * components;
    outbuf = (unsigned char*) malloc(outbuf_size);
    outimg.channel[0] = outbuf;
    outimg.channel[1] = outimg.channel[0] + outimg.pitch[0] * heights[0];
    outimg.channel[2] = outimg.channel[1] + outimg.pitch[0] * heights[0];
    /* Decoder setup ends here */
    setup_end_time = clock();

    CHECK_NVJPEG(nvjpegDecode(handle, jpeg_handle, inbuf, inbuf_size, output_format, &outimg, 0));

    decoding_start_time = clock();
    /* Decompression benchmark begins here, parameters and input image
    cannot be changed until it has finished */
    for (int i = 0; i < iterations; i++)
    {
        nvjpegDecode(handle, jpeg_handle, inbuf, inbuf_size, output_format, &outimg, 0);
    }
    /* Decompression ends here, a new image can be loaded in
    the input buffer and parameters can be changed
    (if not they will remain the same) */
    decoding_end_time = clock();

    cleanup_start_time = clock();
    /* Encoder cleanup begins here */
    CHECK_NVJPEG(nvjpegJpegStateDestroy(jpeg_handle));
    CHECK_NVJPEG(nvjpegDestroy(handle));
    /* Encoder cleanup ends here */
    cleanup_end_time = clock();

    if (benchmark)
    {
        double cleanup_time, decoding_time, setup_time, total_time;
        setup_time = (double)(setup_end_time - setup_start_time) / CLOCKS_PER_SEC;
        decoding_time = (double)(decoding_end_time - decoding_start_time) / CLOCKS_PER_SEC;
        cleanup_time = (double)(cleanup_end_time - cleanup_start_time) / CLOCKS_PER_SEC;
        total_time = setup_time + decoding_time + cleanup_time;
        printf("setup:%f\ndecoding:%f\ncleanup:%f\ntotal:%f\n", setup_time, decoding_time, cleanup_time, total_time);
    }

    if (output == NULL)
    {
        return 0;
    }

    if (strcmp(output, "-") == 0)
    {
        size_t written = fwrite(outbuf, 1, outbuf_size, stdout);
        if (written != outbuf_size)
        {
            perror("Couldn't write to stdout");
            return 1;
        }
    }
    else
    {
        if (img_save(output, &outbuf, outbuf_size))
        {
            return 1;
        }
    }

    return 0;
}
