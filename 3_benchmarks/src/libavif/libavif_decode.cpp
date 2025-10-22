#include "../img_utils.h"
#include "avif/avif.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

void print_syntax()
{
    printf("\nProgram Input Syntax:\n\n");
    printf("  ┌──────────────────────────────────────┐\n");
    printf("  │ -c  <codec [aom|dav1d|gav1]>         │\n");
    printf("  │ -t  <threads [1...n]>                │\n");
    printf("  │ -i  <iterations [1...n]>             │\n");
    printf("  │ -b  <benchmark mode>                 │\n");
    printf("  │ -o  <output mode [FILEPATH|-]>       │\n");
    printf("  └──────────────────────────────────────┘\n");
}

avifCodecChoice get_codec(const char* codec_str)
{
    if (strcmp("dav1d", codec_str) == 0)
    {
        return AVIF_CODEC_CHOICE_DAV1D;
    }
    if (strcmp("gav1", codec_str) == 0)
    {
        return AVIF_CODEC_CHOICE_LIBGAV1;
    }
    if (strcmp("aom", codec_str) == 0)
    {
        return AVIF_CODEC_CHOICE_AOM;
    }
    return AVIF_CODEC_CHOICE_AUTO;
}

int main(int argc, char** argv)
{
    /* Input image related data */
    unsigned char* inbuf = NULL;
    size_t inbuf_size = 0;

    /* Output image related data */
    avifRGBImage rgb;
    avifResult result;
    unsigned char* outbuf = NULL;
    size_t outbuf_size = 0;

    /* Decoder data */
    avifDecoder* decoder = NULL;
    int width = 0, height = 0;

    /* Benchmark data */
    clock_t setup_start_time, setup_end_time;
    clock_t decoding_start_time, decoding_end_time;
    clock_t cleanup_start_time, cleanup_end_time;

    int threads = 0, iterations = 0;
    char *codec_str = NULL, *output = NULL;
    int benchmark = 0;
    int opt;

    while ((opt = getopt(argc, argv, "c:t:i:bo:")) != -1)
    {
        switch (opt)
        {
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
    /* Decoder setup starts here */
    decoder = avifDecoderCreate();
    result = avifDecoderSetIOMemory(decoder, inbuf, inbuf_size);
    if (result != AVIF_RESULT_OK) {
        fprintf(stderr, "Cannot set IO on avifDecoder\n");
        return 1;
    }
    decoder->codecChoice = get_codec(codec_str);
    decoder->maxThreads = threads;
    /* Encoder setup ends here */
    setup_end_time = clock();

    /* Test decode */
    result = avifDecoderParse(decoder);
    if (result != AVIF_RESULT_OK) {
        fprintf(stderr, "Failed to decode image: %s\n", avifResultToString(result));
        return 1;
    }
    avifDecoderNextImage(decoder);
    avifRGBImageSetDefaults(&rgb, decoder->image);
    rgb.rowBytes = rgb.width * 3;
    rgb.maxThreads = decoder->maxThreads;
    rgb.format = AVIF_RGB_FORMAT_RGB;
    outbuf_size = rgb.width * rgb.height * 3;
    outbuf = (unsigned char*) malloc(outbuf_size);
    rgb.pixels = outbuf;
    result = avifImageYUVToRGB(decoder->image, &rgb);
    if (result != AVIF_RESULT_OK) {
        fprintf(stderr, "Conversion from YUV failed: (%s)\n", avifResultToString(result));
        return 1;
    }

    decoding_start_time = clock();
    /* Decompression benchmark begins here, parameters and input image
    cannot be changed until it has finished */
    for (int i = 0; i < iterations; i++)
    {
        avifDecoderParse(decoder);
        avifDecoderNextImage(decoder);
        avifImageYUVToRGB(decoder->image, &rgb);
    }
    /* Decompression ends here, a new image can be loaded in
    the input buffer and parameters can be changed
    (if not they will remain the same) */
    decoding_end_time = clock();

    cleanup_start_time = clock();
    /* Encoder cleanup begins here */
    //avifRGBImageFreePixels(&rgb); // Only use in conjunction with avifRGBImageAllocatePixels()
    avifDecoderDestroy(decoder);
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
