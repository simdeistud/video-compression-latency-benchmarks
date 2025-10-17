#include "libgpujpeg/gpujpeg.h"
#include "../img_utils.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

void print_syntax()
{
    printf("\nProgram Input Syntax:\n\n");
    printf("  ┌────────────────────────────────────────────┐\n");
    printf("  │ -w  <width [px]>                           │\n");
    printf("  │ -h  <height [px]>                          │\n");
    printf("  │ -s  <subsampling [444|422|420]>            │\n");
    printf("  │ -q  <quality [10...100]>                   │\n");
    printf("  │ -f  <pixel format [planar|interleaved]>    │\n");
    printf("  │ -r  <restart intervals [0...n]>            │\n");
    printf("  │ -i  <iterations [1...n]>                   │\n");
    printf("  │ -b  <benchmark mode>                       │\n");
    printf("  │ -o  <output mode [FILEPATH|-]>             │\n");
    printf("  └────────────────────────────────────────────┘\n");
}

int get_sampnum(const char* subsampling_str)
{
    switch (atoi(subsampling_str))
    {
    case 444: return GPUJPEG_SUBSAMPLING_444; // 0x11111100U
    case 422: return GPUJPEG_SUBSAMPLING_422; // 0x21111100U
    case 420: return GPUJPEG_SUBSAMPLING_420; // 0x22111100U
    default: return GPUJPEG_SUBSAMPLING_UNKNOWN;
    }
}

int is_interleaved(const char* fmt_str)
{
    if (strcmp(fmt_str, "planar") == 0)
    {
        return 1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    /* Input image related data */
    uint8_t* inbuf = NULL;
    size_t inbuf_size = 0;

    /* Output image related data */
    uint8_t* outbuf = NULL;
    size_t outbuf_size = 0;

    /* Encoder data */
    struct gpujpeg_parameters param;
    struct gpujpeg_image_parameters param_image;
    struct gpujpeg_encoder* encoder;
    struct gpujpeg_encoder_input encoder_input;

    /* Benchmark data */
    clock_t setup_start_time, setup_end_time;
    clock_t encoding_start_time, encoding_end_time;
    clock_t cleanup_start_time, cleanup_end_time;

    /* Input parsing */
    int width = 0, height = 0, quality = 0, restart_interval = 0, iterations = 0, format = 0;
    char *subsampling_str = NULL, *fmt_str = NULL, *output = NULL;
    int benchmark = 0;
    int opt;

    while ((opt = getopt(argc, argv, "w:h:s:q:f:r:i:bo:")) != -1)
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
        case 'f': fmt_str = optarg;
            break;
        case 'r': restart_interval = atoi(optarg);
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
    /* We use the default CUDA device, if you want to test on different ones, change the first argument of gpujpeg_init_device */
    if (gpujpeg_init_device(0, 0))
    {
        perror("Failed to initialize GPU device");
        return 1;
    }

    encoder = gpujpeg_encoder_create(0);
    if (encoder == NULL)
    {
        perror("Failed to create encoder");
        return 1;
    }

    /* Setting up the input image parameters */
    gpujpeg_image_set_default_parameters(&param_image);
    param_image.width = width;
    param_image.height = height;
    param_image.color_space = GPUJPEG_RGB;
    param_image.pixel_format = GPUJPEG_444_U8_P012;

    /* Setting up the compression parameters */
    gpujpeg_set_default_parameters(&param);
    param.quality = quality;
    param.interleaved = is_interleaved(fmt_str);
    param.segment_info = param.interleaved;
    param.restart_interval = restart_interval;
    gpujpeg_parameters_chroma_subsampling(&param, get_sampnum(subsampling_str));
    param.color_space_internal = GPUJPEG_YCBCR;
    gpujpeg_encoder_input_set_image(&encoder_input, inbuf);
    /* Encoder setup ends here */
    setup_end_time = clock();

    /* Test run to see if everything works */
    if (gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input, &outbuf, &outbuf_size))
    {
        perror("Failed to encode image");
        return 1;
    }

    encoding_start_time = clock();
    /* Compression begins here, parameters and input image
       cannot be changed until it has finished             */
    for (int i = 0; i < iterations; i++)
    {
        gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input, &outbuf, &outbuf_size);
    }
    /* Compression ends here, a new image can be loaded in
       the input buffer and parameters can be changed
       (if not they will remain the same)                  */
    encoding_end_time = clock();

    /* The cleanup process also deallocates the compressed image.
     * For this reason, we perform a full temporary copy so we can
     * correctly output it */
    uint8_t* tmp = (uint8_t*)malloc(outbuf_size);
    memcpy(tmp, outbuf, outbuf_size);

    cleanup_start_time = clock();
    /* Encoder cleanup begins here */
    gpujpeg_encoder_destroy(encoder);
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
        size_t written = fwrite(tmp, 1, outbuf_size, stdout);
        if (written != outbuf_size)
        {
            perror("Couldn't write to stdout");
            return 1;
        }
    }
    else
    {
        if (img_save(output, &tmp, outbuf_size))
        {
            return 1;
        }
    }

    return 0;
}
