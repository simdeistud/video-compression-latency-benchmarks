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
    printf("  │ -w  <width [px]>                     │\n");
    printf("  │ -h  <height [px]>                    │\n");
    printf("  │ -s  <subsampling [444|422|420]>      │\n");
    printf("  │ -q  <quality [10...100]>             │\n");
    printf("  │ -i  <iterations [1...n]>             │\n");
    printf("  │ -b  <benchmark mode>                 │\n");
    printf("  │ -o  <output mode [FILEPATH|-]>       │\n");
    printf("  └──────────────────────────────────────┘\n");
}

nvjpegChromaSubsampling_t get_subsampling(const char* subsampling_str)
{
    switch (atoi(subsampling_str))
    {
    case 444: return NVJPEG_CSS_444;
    case 422: return NVJPEG_CSS_422;
    case 420: return NVJPEG_CSS_420;
    default: return NVJPEG_CSS_UNKNOWN;
    }
}

int main(int argc, char** argv)
{
    /* Input image related data */
    nvjpegImage_t source;
    unsigned char* inbuf = NULL;
    size_t inbuf_size = 0;

    /* Output image related data */
    unsigned char* outbuf = NULL;
    size_t outbuf_size = 0;

    /* Encoder data */
    nvjpegHandle_t handle;
    nvjpegEncoderState_t encoder_state;
    nvjpegEncoderParams_t encoder_params;
    cudaStream_t stream = 0;

    /* Benchmark data */
    clock_t setup_start_time, setup_end_time;
    clock_t encoding_start_time, encoding_end_time;
    clock_t cleanup_start_time, cleanup_end_time;

    /* Input parsing */
    int width = 0, height = 0, quality = 0, restart_interval = 0, iterations = 0;
    char *subsampling_str = NULL, *dct_str = NULL, *output = NULL;
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
    /* Encoder setup starts here */
    CHECK_NVJPEG(nvjpegCreateSimple(&handle));
    CHECK_NVJPEG(nvjpegEncoderStateCreate(handle, &encoder_state, stream));
    /* Setting up the compression parameters */
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle, &encoder_params, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(encoder_params, NVJPEG_ENCODING_BASELINE_DCT, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(encoder_params, quality, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(encoder_params, 0, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(encoder_params, get_subsampling(subsampling_str), stream));
    /* Setting up image parameters */
    source.pitch[0] = width * 3;
    source.pitch[1] = width * 3;
    source.pitch[2] = width * 3;
    source.channel[0] = inbuf;
    source.channel[1] = source.channel[0] + source.pitch[0] * height;
    source.channel[2] = source.channel[1] + source.pitch[0] * height;
    /* Encoder setup ends here */
    setup_end_time = clock();

    /* Test run to see if everything works */
    CHECK_NVJPEG(nvjpegEncodeImage(handle, encoder_state, encoder_params, &source, NVJPEG_INPUT_RGBI, width, height, stream));
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, encoder_state, NULL, &outbuf_size, stream));
    outbuf = (unsigned char*) malloc(outbuf_size);
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, encoder_state, outbuf, &outbuf_size, stream));

    encoding_start_time = clock();
    /* Compression begins here, parameters and input image
       cannot be changed until it has finished             */
    for (int i = 0; i < iterations; i++)
    {
        nvjpegEncodeImage(handle, encoder_state, encoder_params, &source, NVJPEG_INPUT_RGBI, width, height, stream);
        nvjpegEncodeRetrieveBitstream(handle, encoder_state, NULL, &outbuf_size, stream);
        nvjpegEncodeRetrieveBitstream(handle, encoder_state, outbuf, &outbuf_size, stream);
    }
    /* Compression ends here, a new image can be loaded in
       the input buffer and parameters can be changed
       (if not they will remain the same)                  */
    encoding_end_time = clock();

    cleanup_start_time = clock();
    /* Encoder cleanup begins here */
    CHECK_NVJPEG(nvjpegEncoderParamsDestroy(encoder_params));
    CHECK_NVJPEG(nvjpegDestroy(handle));
    /* Encoder cleanup ends here */
    cleanup_end_time = clock();

    if (benchmark)
    {
        double cleanup_time, encoding_time, setup_time, total_time;
        setup_time = (double)(setup_end_time - setup_start_time) / CLOCKS_PER_SEC;
        encoding_time = (double)(encoding_end_time - encoding_start_time) / CLOCKS_PER_SEC;
        cleanup_time = (double)(cleanup_end_time - cleanup_start_time) / CLOCKS_PER_SEC;
        total_time = setup_time + encoding_time + cleanup_time;
        printf("setup:%f\nencoding:%f\ncleanup:%f\ntotal:%f\n", setup_time, encoding_time, cleanup_time, total_time);
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
