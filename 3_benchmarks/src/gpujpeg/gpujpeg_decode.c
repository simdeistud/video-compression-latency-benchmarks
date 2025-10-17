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
    printf("  ┌──────────────────────────────────────┐\n");
    printf("  │ -i  <iterations [1...n]>             │\n");
    printf("  │ -b  <benchmark mode>                 │\n");
    printf("  │ -o  <output mode [FILEPATH|-]>       │\n");
    printf("  └──────────────────────────────────────┘\n");
}

int main(int argc, char* argv[])
{
    /* Input image related data */
    uint8_t* inbuf = NULL;
    size_t inbuf_size = 0;

    /* Output image related data */
    struct gpujpeg_decoder_output decoder_output;

    /* Decoder data */
    struct gpujpeg_parameters param;
    struct gpujpeg_image_parameters param_image;
    struct gpujpeg_decoder* decoder;

    /* Benchmark data */
    clock_t setup_start_time, setup_end_time;
    clock_t decoding_start_time, decoding_end_time;
    clock_t cleanup_start_time, cleanup_end_time;

    int iterations = 0;
    char *output = NULL;
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
    /* We use the default CUDA device, if you want to test on different ones, change the first argument of gpujpeg_init_device */
    if (gpujpeg_init_device(0, 0))
    {
        perror("Failed to initialize GPU device");
        return 1;
    }

    decoder = gpujpeg_decoder_create(0);
    if (decoder == NULL)
    {
        perror("Failed to create decoder");
        return 1;
    }
    gpujpeg_set_default_parameters(&param);
    gpujpeg_image_set_default_parameters(&param_image);
    gpujpeg_decoder_init(decoder, &param, &param_image);
    gpujpeg_decoder_output_set_default(&decoder_output);
    gpujpeg_decoder_set_output_format(decoder, GPUJPEG_RGB, GPUJPEG_444_U8_P012);
    /* Encoder setup ends here */
    setup_end_time = clock();

    /* Test run to see if everything works */
    if (gpujpeg_decoder_decode(decoder, inbuf, inbuf_size, &decoder_output))
    {
        perror("Failed to decode image");
        return 1;
    }

    decoding_start_time = clock();
    /* Decompression begins here, parameters and input image
    cannot be changed until it has finished */
    for (int i = 0; i < iterations; i++)
    {
        gpujpeg_decoder_decode(decoder, inbuf, inbuf_size, &decoder_output);
    }
    /* Decompression ends here, a new image can be loaded in
    the input buffer and parameters can be changed
    (if not they will remain the same) */
    decoding_end_time = clock();

    /* The cleanup process also deallocates the compressed image.
     * For this reason, we perform a full temporary copy so we can
     * correctly output it */
    size_t outbuf_size = decoder_output.data_size;
    uint8_t* outbuf = (uint8_t*)malloc(outbuf_size);
    memcpy(outbuf, decoder_output.data, decoder_output.data_size);

    cleanup_start_time = clock();
    /* Encoder cleanup begins here */
    gpujpeg_decoder_destroy(decoder);
    /* Encoder cleanup ends here */
    cleanup_end_time = clock();

    if (benchmark)
    {
        double cleanup_time, encoding_time, setup_time, total_time;
        setup_time = (double)(setup_end_time - setup_start_time) / CLOCKS_PER_SEC;
        encoding_time = (double)(decoding_end_time - decoding_start_time) / CLOCKS_PER_SEC;
        cleanup_time = (double)(cleanup_end_time - cleanup_start_time) / CLOCKS_PER_SEC;
        total_time = setup_time + encoding_time + cleanup_time;
        printf("setup:%f\ndecoding:%f\ncleanup:%f\ntotal:%f\n", setup_time, encoding_time, cleanup_time, total_time);
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
