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
    printf("  │ -w  <width [px]>                     │\n");
    printf("  │ -h  <height [px]>                    │\n");
    printf("  │ -s  <subsampling [444|422|420]>      │\n");
    printf("  │ -q  <quality [10...100]>             │\n");
    printf("  │ -d  <DCT method [int|fast|float]>    │\n");
    printf("  │ -r  <restart intervals [0...n]>      │\n");
    printf("  │ -i  <iterations [1...n]>             │\n");
    printf("  │ -b  <benchmark mode>                 │\n");
    printf("  │ -o  <output mode [FILEPATH|-]>       │\n");
    printf("  └──────────────────────────────────────┘\n");
}

int get_vsamp(const char* subsampling_str)
{
    switch (atoi(subsampling_str))
    {
    case 444:
    case 422: return 1;
    case 420: return 2;
    default: return 1;
    }
}

int get_hsamp(const char* subsampling_str)
{
    switch (atoi(subsampling_str))
    {
    case 444: return 1;
    case 422:
    case 420: return 2;
    default: return 0;
    }
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

    /* Encoder data */
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    /* Benchmark data */
    clock_t setup_start_time, setup_end_time;
    clock_t encoding_start_time, encoding_end_time;
    clock_t cleanup_start_time, cleanup_end_time;

    /* Input parsing */
    int width = 0, height = 0, quality = 0, restart_interval = 0, iterations = 0;
    char *subsampling_str = NULL, *dct_str = NULL, *output = NULL;
    int benchmark = 0;
    int opt;

    while ((opt = getopt(argc, argv, "w:h:s:q:d:r:i:bo:")) != -1)
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
        case 'd': dct_str = optarg;
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
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_mem_dest(&cinfo, &outbuf, &outbuf_size);

    /* Setting up the input image parameters */
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    /* Setting up the compression parameters */
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE); /* Compression quality, chosen by the user */
    jpeg_set_colorspace(&cinfo, JCS_YCbCr); /* Output colorspace, we choose YUV */
    cinfo.comp_info[0].v_samp_factor = get_vsamp(subsampling_str); /* Chroma subsampling options, chosen by the user */
    cinfo.comp_info[0].h_samp_factor = get_hsamp(subsampling_str);
    cinfo.arith_code = 0; /* Arithmetic or Huffman encoding, always choose Huffman because much faster */
    cinfo.dct_method = get_dct(dct_str); /* DCT method, chosen by the user */
    cinfo.restart_interval = restart_interval; /* Presence of restart intervals, chosen by the user */
    /* Encoder setup ends here */
    setup_end_time = clock();

    /* Test run to see if everything works */
    jpeg_start_compress(&cinfo, TRUE);
    while (cinfo.next_scanline < cinfo.image_height)
    {
        row_pointer[0] = &inbuf[cinfo.next_scanline * cinfo.image_width * cinfo.input_components];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }
    jpeg_finish_compress(&cinfo);

    encoding_start_time = clock();
    /* Compression begins here, parameters and input image
       cannot be changed until it has finished             */
    for (int i = 0; i < iterations; i++)
    {
        jpeg_start_compress(&cinfo, TRUE);
        while (cinfo.next_scanline < cinfo.image_height)
        {
            row_pointer[0] = &inbuf[cinfo.next_scanline * cinfo.image_width * cinfo.input_components];
            jpeg_write_scanlines(&cinfo, row_pointer, 1);
        }
        jpeg_finish_compress(&cinfo);
    }
    /* Compression ends here, a new image can be loaded in
       the input buffer and parameters can be changed
       (if not they will remain the same)                  */
    encoding_end_time = clock();

    cleanup_start_time = clock();
    /* Encoder cleanup begins here */
    jpeg_destroy_compress(&cinfo);
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
