#include "../img_utils.h"
#include "avif/avif_cxx.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <avif/avif.h>


void print_syntax()
{
    printf("\nProgram Input Syntax:\n\n");
    printf("  ┌──────────────────────────────────────┐\n");
    printf("  │ -w  <width [px]>                     │\n");
    printf("  │ -h  <height [px]>                    │\n");
    printf("  │ -s  <subsampling [444|422|420]>      │\n");
    printf("  │ -q  <quality [10...100]>             │\n");
    printf("  │ -f  <speed [0...10]>                 │\n");
    printf("  │ -c  <codec [aom|dav1d|svt]>          │\n");
    printf("  │ -t  <threads [1...n]>                │\n");
    printf("  │ -i  <iterations [1...n]>             │\n");
    printf("  │ -b  <benchmark mode>                 │\n");
    printf("  │ -o  <output mode [FILEPATH|-]>       │\n");
    printf("  └──────────────────────────────────────┘\n");
}

avifPixelFormat get_subsampling(const char* subsampling_str)
{
    switch (atoi(subsampling_str))
    {
    case 444: return AVIF_PIXEL_FORMAT_YUV444;
    case 422: return AVIF_PIXEL_FORMAT_YUV422;
    case 420: return AVIF_PIXEL_FORMAT_YUV420;
    default: return AVIF_PIXEL_FORMAT_NONE;
    }
}

avifCodecChoice get_codec(const char* codec_str)
{
    if (strcmp("rav1e", codec_str) != 0)
    {
        return AVIF_CODEC_CHOICE_RAV1E;
    }
    if (strcmp("svt", codec_str) != 0)
    {
        return AVIF_CODEC_CHOICE_SVT;
    }
    if (strcmp("aom", codec_str) != 0)
    {
        return AVIF_CODEC_CHOICE_AOM;
    }
    return AVIF_CODEC_CHOICE_AUTO;
}

int main(int argc, char** argv)
{
    /* Input image related data */
    avifRGBImage rgb = {0};
    unsigned char* inbuf = NULL;
    size_t inbuf_size = 0;

    /* Output image related data */
    avifRWData avifOutput = AVIF_DATA_EMPTY;
    avifImage* image = NULL;
    unsigned char* outbuf = NULL;
    size_t outbuf_size = 0;

    /* Encoder data */
    avifEncoder* encoder = NULL;
    /* Benchmark data */
    clock_t setup_start_time, setup_end_time;
    clock_t encoding_start_time, encoding_end_time;
    clock_t cleanup_start_time, cleanup_end_time;

    /* Input parsing */
    int width = 0, height = 0, quality = 0, threads = 1, speed = 0, iterations = 0;
    char *subsampling_str = NULL, *codec_str = NULL, *output = NULL;
    int benchmark = 0;
    int opt;

    while ((opt = getopt(argc, argv, "w:h:s:q:f:c:t:i:bo:")) != -1)
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
        case 'f': speed = atoi(optarg);
            break;
        case 'c': codec_str = optarg;
            break;
        case 't': threads = atoi(optarg);
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
    image = avifImageCreate(width, height, 8, get_subsampling(subsampling_str));
    encoder = avifEncoderCreate();
    encoder->codecChoice = get_codec(codec_str);
    encoder->maxThreads = threads;
    encoder->speed = speed;
    encoder->quality = quality;
    /* Encoder setup ends here */
    setup_end_time = clock();

    /* Test run to see if everything works */
    avifRGBImageSetDefaults(&rgb, image);
    rgb.pixels = inbuf;
    rgb.maxThreads = encoder->maxThreads;
    rgb.avoidLibYUV = 1;
    avifImageRGBToYUV(image, &rgb);
    avifEncoderAddImage(encoder, image, 1, AVIF_ADD_IMAGE_FLAG_SINGLE);
    avifEncoderFinish(encoder, &avifOutput);
    outbuf = avifOutput.data;
    outbuf_size = avifOutput.size;

    encoding_start_time = clock();
    /* Compression begins here, parameters and input image
       cannot be changed until it has finished             */
    for (int i = 0; i < iterations; i++)
    {

    }
    /* Compression ends here, a new image can be loaded in
       the input buffer and parameters can be changed
       (if not they will remain the same)                  */
    encoding_end_time = clock();

    cleanup_start_time = clock();
    /* Encoder cleanup begins here */
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
