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
    printf("  │ -t  <multithread [0|1]>              │\n");
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
    unsigned char* outbuf = NULL;
    size_t outbuf_size = 0;

    /* Decoder data */
    Decode
    WebPDecoderConfig config;
    int width = 0, height = 0;

    /* Benchmark data */
    clock_t setup_start_time, setup_end_time;
    clock_t decoding_start_time, decoding_end_time;
    clock_t cleanup_start_time, cleanup_end_time;

    int threaded = 0, iterations = 0;
    char* output = NULL;
    int benchmark = 0;
    int opt;

    while ((opt = getopt(argc, argv, "t:i:bo:")) != -1)
    {
        switch (opt)
        {
        case 't': threaded = atoi(optarg);
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
    WebPInitDecoderConfig(&config);
    WebPGetFeatures(inbuf, inbuf_size, &config.input);
    WebPGetInfo(inbuf, inbuf_size, &width, &height);
    outbuf_size = width * height * 3;
    outbuf = (unsigned char*) malloc(outbuf_size);
    config.options.no_fancy_upsampling = 1;
    config.options.use_scaling = 0;
    config.options.use_threads = threaded;
    config.output.colorspace = MODE_RGB;
    config.output.u.RGBA.rgba = outbuf;
    config.output.u.RGBA.stride = width * 3;
    config.output.u.RGBA.size = outbuf_size;
    config.output.is_external_memory = 1;
    /* Encoder setup ends here */
    setup_end_time = clock();

    /* Test decode */
    WebPDecode(inbuf, inbuf_size, &config);

    decoding_start_time = clock();
    /* Decompression benchmark begins here, parameters and input image
    cannot be changed until it has finished */
    for (int i = 0; i < iterations; i++)
    {
        WebPDecode(inbuf, inbuf_size, &config);
    }
    /* Decompression ends here, a new image can be loaded in
    the input buffer and parameters can be changed
    (if not they will remain the same) */
    decoding_end_time = clock();

    cleanup_start_time = clock();
    /* Encoder cleanup begins here */
    WebPFreeDecBuffer(&config.output);
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
