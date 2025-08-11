#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <jpeglib.h>
#include "../../include/benchmark.h"
#include "../../include/img_utils.h"
#include "../../include/subs_utils.h"
#include "../../include/res_utils.h"

int decoder_initialization(struct jpeg_decompress_struct* cinfo, struct jpeg_error_mgr* jerr)
{
    cinfo->err = jpeg_std_error(jerr);
    jpeg_create_decompress(cinfo);
    return 0;
}

int decoder_setup(struct jpeg_decompress_struct* cinfo, J_DCT_METHOD dct)
{
    cinfo->dct_method = dct;
    return 0;
}

int decoder_destroy(struct jpeg_decompress_struct* cinfo)
{
    jpeg_abort_decompress(cinfo);
    return 0;
}

int decode_image_with_libjpeg(const char* input_filename, const char* output_filename, J_DCT_METHOD dct, int iterations)
{
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    unsigned char* image_buffer = NULL;
    JSAMPROW row_pointer[1];
    int row_stride;
    clock_t start_time, end_time;
    double total_time;
    unsigned char* input_buffer = NULL;
    size_t input_buffer_size = 0;
    unsigned char* output_buffer = NULL;
    size_t output_buffer_size = 0;
    int i;

    input_buffer = (unsigned char*) img_load(input_filename, &input_buffer_size);
    
    if (!input_buffer) {
        fprintf(stderr, "Error: Failed to load data from file: %s\n", input_filename);
        return 1;
    }

    if (decoder_initialization(&cinfo, &jerr)) {
        img_destroy(input_buffer);
        return 1;
    }
    
    decoder_setup(&cinfo, dct);

    /* Measure decoding time */
    start_time = clock();
    for (i = 0; i < iterations; i++) {
    jpeg_mem_src(&cinfo, input_buffer, input_buffer_size);
        if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
            fprintf(stderr, "Error: Failed to read JPEG header from file: %s\n", input_filename);
            decoder_destroy(&cinfo);
            img_destroy(input_buffer);
            return 1;
        }
        
        jpeg_start_decompress(&cinfo);
        output_buffer_size = cinfo.output_width * cinfo.output_height * cinfo.output_components;
        output_buffer = (unsigned char*)malloc(output_buffer_size);
        row_stride = cinfo.output_width * cinfo.output_components;

        while (cinfo.output_scanline < cinfo.output_height) {
            unsigned char* buffer_array[1];
            buffer_array[0] = output_buffer + (cinfo.output_scanline) * row_stride;
            jpeg_read_scanlines(&cinfo, buffer_array, 1);
        }

        jpeg_finish_decompress(&cinfo);
    }
    end_time = clock();

    /* Calculate the total time */
    total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    /* Save the decompressed image */
    if (img_save(output_filename, output_buffer, output_buffer_size)) {
        fprintf(stderr, "Error: Failed to save decompressed image to file: %s\n", output_filename);
        free(output_buffer);
        decoder_destroy(&cinfo);
        return;
    }

    /* Clean up resources */
    img_destroy(input_buffer);
    free(output_buffer);
    decoder_destroy(&cinfo);
    
    printf("%f\n", total_time);

    return 0;
}

int main(int argc, char* argv[])
{
    const char* input_filename;
    const char* output_filename;
    const char* dct_str = NULL;
    J_DCT_METHOD dct;
    int iterations;

    if (argc != 5) {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <dct_algorithm>  <iterations>\n", argv[0]);
        return 1;
    }

    input_filename = argv[1];
    output_filename = argv[2];
    dct_str = argv[3];
    iterations = atoi(argv[4]);
    
    if (strcmp(dct_str, "dct_fast") == 0) {
        dct = JDCT_FASTEST;
    } else if (strcmp(dct_str, "dct_default") == 0) {
        dct = JDCT_DEFAULT;
    } else {
        /* Invalid string input */
        fprintf(stderr, "Error: Invalid DCT algorithm '%s'. Use 'dct_default' or 'dct_fast'.\n", dct_str);
        return 1;
    }

    if (iterations <= 0) {
        fprintf(stderr, "Error: Iterations must be a positive integer.\n");
        return 1;
    }

    /* Create the CSV file with the header */

    decode_image_with_libjpeg(input_filename, output_filename, dct, iterations);

    return 0;
}
