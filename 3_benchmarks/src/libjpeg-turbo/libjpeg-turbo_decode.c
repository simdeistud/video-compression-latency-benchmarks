#include "../img_utils.h"
#include "jpeglib.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

void print_syntax()
{
    printf("\nProgram Input Syntax:\n\n");
    printf("  ┌──────────────────────────────────────┐\n");
    printf("  │ -d  <DCT method [int|fast|float]>    │\n");
    printf("  │ -i  <iterations [1...n]>             │\n");
    printf("  │ -b  <benchmark mode>                 │\n");
    printf("  │ -o  <output mode [FILEPATH|-]>       │\n");
    printf("  └──────────────────────────────────────┘\n");
}

J_DCT_METHOD get_dct(const char* dct_str)
{
    if (strcmp(dct_str, "int") == 0) return JDCT_ISLOW;
    if (strcmp(dct_str, "fast") == 0) return JDCT_IFAST;
    if (strcmp(dct_str, "float") == 0) return JDCT_FLOAT;
    return -1;
}

int main(int argc, char** argv)
{
    /* Input image related data */
    JSAMPLE* inbuf = NULL;
    size_t inbuf_size = 0;
    JSAMPROW row_pointer[1];

    /* Output image related data */
    JSAMPLE* outbuf = NULL;
    size_t outbuf_size = 0;

    /* Decoder data */
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    /* Benchmark data */
    clock_t setup_start_time, setup_end_time;
    clock_t decoding_start_time, decoding_end_time;
    clock_t cleanup_start_time, cleanup_end_time;

    int iterations = 0;
    char *dct_str = NULL, *output = NULL;
    int benchmark = 0;
    int opt;

    while ((opt = getopt(argc, argv, "d:i:bo:")) != -1)
    {
        switch (opt)
        {
        case 'd': dct_str = optarg;
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
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    /* Setting up decompression parameters */
    cinfo.dct_method = get_dct(dct_str);
    jpeg_mem_src(&cinfo, inbuf, inbuf_size);
    /* Image source has to be set before every run even though it's the same pointer... */
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK)
    {
        fprintf(stderr, "Error: Failed to read JPEG header from file: %s\n", argv[1]);
        jpeg_destroy_decompress(&cinfo);
        img_destroy(inbuf);
        return 1;
    } /* Reading the JPEG header is mandatory before starting the decompression */
    jpeg_start_decompress(&cinfo);
    /* Encoder setup ends here */
    setup_end_time = clock();

    outbuf_size = cinfo.output_width * cinfo.output_height * cinfo.output_components;
    /* These cinfo fields are filled only AFTER decompression has started */
    outbuf = (JSAMPLE*)malloc(outbuf_size);
    while (cinfo.output_scanline < cinfo.output_height)
    {
        row_pointer[0] = outbuf + (cinfo.output_scanline * cinfo.output_width * cinfo.output_components);
        jpeg_read_scanlines(&cinfo, row_pointer, 1);
    }
    jpeg_finish_decompress(&cinfo);

    decoding_start_time = clock();
    /* Decompression benchmark begins here, parameters and input image
    cannot be changed until it has finished */
    for (int i = 0; i < iterations; i++)
    {
        jpeg_mem_src(&cinfo, inbuf, inbuf_size);
        jpeg_read_header(&cinfo, TRUE);
        jpeg_start_decompress(&cinfo);
        while (cinfo.output_scanline < cinfo.output_height)
        {
            row_pointer[0] = outbuf + (cinfo.output_scanline * cinfo.output_width * cinfo.output_components);
            jpeg_read_scanlines(&cinfo, row_pointer, 1);
        }
        jpeg_finish_decompress(&cinfo);
    }
    /* Decompression ends here, a new image can be loaded in
    the input buffer and parameters can be changed
    (if not they will remain the same) */
    decoding_end_time = clock();

    cleanup_start_time = clock();
    /* Encoder cleanup begins here */
    jpeg_destroy_decompress(&cinfo);
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
