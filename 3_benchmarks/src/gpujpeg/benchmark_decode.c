#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "libgpujpeg/gpujpeg.h"
#include "../../include/benchmark.h"
#include "../../include/img_utils.h"
#include "../../include/subs_utils.h"
#include "../../include/res_utils.h"

int decoder_initialization()
{
    /* We use the default CUDA device, if you want to test on different ones, change the first argument of gpujpeg_init_device */
    if (gpujpeg_init_device(0, 0)) {
    perror("Failed to initialize GPU device");
    return 1;
  }
    
    return 0;
}

int decoder_setup(struct gpujpeg_decoder** decoder)
{
    *decoder = gpujpeg_decoder_create(0);
    if (*decoder == NULL ){
      perror("Failed to create decoder");
      return 1;
    }
    return 0;
}

int decoding_parameters_setup(struct gpujpeg_decoder** decoder, struct gpujpeg_parameters* param, struct gpujpeg_image_parameters* param_image, struct gpujpeg_decoder_output* decoder_output)
{
  //gpujpeg_decoder_init(*decoder, param, param_image);
  gpujpeg_decoder_set_output_format(*decoder, GPUJPEG_RGB,
                GPUJPEG_444_U8_P012);
  gpujpeg_decoder_output_set_default(decoder_output);
}

int decoder_destroy(struct gpujpeg_decoder* decoder)
{
    gpujpeg_decoder_destroy(decoder);
    return 0;
}

int decode_image_with_gpujpeg(const char* input_filename, const char* output_filename, int iterations)
{
    uint8_t* input_buffer = NULL;
    size_t input_buffer_size = 0;
    uint8_t* output_buffer = NULL;
    size_t output_buffer_size = 0;
    struct gpujpeg_parameters param;
    struct gpujpeg_image_parameters param_image;
    struct gpujpeg_decoder* decoder;
    struct gpujpeg_decoder_output decoder_output;
    clock_t start_time, end_time;
    double total_time;
    int i;

    if (decoder_initialization()) {
        return 1;
    }
    
    if(decoder_setup(&decoder)){
      return 1;
    }
    
    input_buffer = (unsigned char*) img_load(input_filename, &input_buffer_size);
    
    if (!input_buffer) {
        fprintf(stderr, "Error: Failed to load data from file: %s\n", input_filename);
        return 1;
    }
    
    decoding_parameters_setup(&decoder, &param, &param_image, &decoder_output);
    
    if (gpujpeg_decoder_decode(decoder, input_buffer, input_buffer_size, &decoder_output)){
      perror("Failed to decode image");
        return 1;
    }

    /* Measure decoding time */
    start_time = clock();
    for (i = 0; i < iterations; i++) {
      gpujpeg_decoder_decode(decoder, input_buffer, input_buffer_size, &decoder_output);
    }
    end_time = clock();

    /* Calculate the total time */
    total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    img_destroy(input_buffer);
    
    /* Save the decompressed image */
    if (img_save(output_filename, decoder_output.data, decoder_output.data_size)) {
        fprintf(stderr, "Error: Failed to save decompressed image to file: %s\n", output_filename);
        //free(output_buffer);
        decoder_destroy(decoder);
        return 1;
    }

    /* Clean up resources */
    //free(output_buffer);
    decoder_destroy(decoder);
    
    printf("%f\n", total_time);

    return 0;
}

int main(int argc, char* argv[])
{
    const char* input_filename;
    const char* output_filename;
    const char* dct_str = NULL;
    int iterations;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <iterations>\n", argv[0]);
        return 1;
    }

    input_filename = argv[1];
    output_filename = argv[2];
    iterations = atoi(argv[3]);

    if (iterations <= 0) {
        fprintf(stderr, "Error: Iterations must be a positive integer.\n");
        return 1;
    }


    decode_image_with_gpujpeg(input_filename, output_filename, iterations);

    return 0;
}
