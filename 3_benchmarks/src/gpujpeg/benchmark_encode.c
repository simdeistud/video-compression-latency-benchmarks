#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "libgpujpeg/gpujpeg.h"
#include "../../include/benchmark.h"
#include "../../include/img_utils.h"
#include "../../include/subs_utils.h"
#include "../../include/res_utils.h"

int encoder_initialization()
{
    /* We use the default CUDA device, if you want to test on different ones, change the first argument of gpujpeg_init_device */
    if (gpujpeg_init_device(0, 0)) {
    perror("Failed to initialize GPU device");
    return 1;
  }
    
    return 0;
}

int encoder_setup(struct gpujpeg_encoder** encoder)
{        
    *encoder = gpujpeg_encoder_create(0);
  if (*encoder == NULL) {
    perror("Failed to create encoder");
    return 1;
  }
  return 0;
}

int encoding_parameters_setup(struct gpujpeg_parameters* param, struct gpujpeg_image_parameters* param_image, struct gpujpeg_encoder_input* encoder_input, uint8_t* input_buffer, int width, int height, int quality, subsampling_t subsampling, int interleaved)
{
  gpujpeg_set_default_parameters(param);
  param->quality = quality;
  param->interleaved = interleaved;
  param->segment_info = interleaved;
  /* param.restart_interval = 0; */
  
  gpujpeg_image_set_default_parameters(param_image);
  if (subsampling == SUBSAMPLING_GRAYSCALE){
    param_image->width = width;
    param_image->height = height;
    param_image->pixel_format = GPUJPEG_U8;
  } else {
    param_image->width = width;
    param_image->height = height;
    param_image->pixel_format = GPUJPEG_444_U8_P012;
    gpujpeg_parameters_chroma_subsampling(param, subsampling_to_gpujpeg(subsampling));
  }
  
  /* param_image->color_space = GPUJPEG_RGB; */
  
  gpujpeg_encoder_input_set_image(encoder_input, input_buffer);
     
    return 0;
}

int encoding_parameters_cleanup()
{
    return 0;
}

int encoder_cleanup()
{
    return 0;
}

int encoder_destroy(struct gpujpeg_encoder* encoder)
{
    gpujpeg_encoder_destroy(encoder);
    encoder = NULL;
    return 0;
}

int encode_image_with_gpujpeg(const char* input_filename, const char* output_filename, int quality, subsampling_t subsampling, int width, int height, int interleaved, int iterations)
{
    uint8_t* input_buffer = NULL;
    size_t input_buffer_size = 0;
    uint8_t* output_buffer = NULL;
    size_t output_buffer_size = 0;
    struct gpujpeg_parameters param;
    struct gpujpeg_image_parameters param_image;
    struct gpujpeg_encoder* encoder;
    struct gpujpeg_encoder_input encoder_input;
    clock_t start_time, end_time;
    double total_time;
    int i;

    if (encoder_initialization()) {
        img_destroy(input_buffer);
        return 1;
    }

    if(encoder_setup(&encoder)){
      return 1;
    }
    
    input_buffer = (unsigned char*) img_load(input_filename, &input_buffer_size);
    
    if (!input_buffer) {
        fprintf(stderr, "Error: Failed to load data from file: %s\n", input_filename);
        return 1;
    }
    
    encoding_parameters_setup(&param, &param_image, &encoder_input, input_buffer, width, height, quality, subsampling, interleaved);
    
    if(gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input, &output_buffer, &output_buffer_size)){
      perror("Failed to encode image");
        return 1;
    }

    /* Measure encoding time */
    start_time = clock();
    for (i = 0; i < iterations; i++) {
    
        gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input, &output_buffer, &output_buffer_size);
        
    }
    end_time = clock();
    
    img_save(output_filename, output_buffer, output_buffer_size);
    
    /* Calculate the total time and compressed size */
    total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    if (img_destroy(input_buffer)) {
          return 1;
    }
    encoding_parameters_cleanup();
    encoder_cleanup();
    encoder_destroy(encoder);
    
    printf("%f\n", total_time);
    
    return 0;
}

int main(int argc, char* argv[])
{
    const char* input_filename;
    int width, height;
    resolution_t resolution;
    
    int quality;
    subsampling_t subsampling;
    const char* output_filename;
    int interleaved;
    int iterations;
    input_filename = NULL;
    output_filename = NULL;
    width = 0;
    height = 0;
    resolution = RESOLUTION_INVALID;
    iterations = 0;

    if (argc != 8) {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <resolution> <quality> <subsampling> <interleaved> <iterations>\n", argv[0]);
        return 1;
    }
    

    input_filename = argv[1];
    output_filename = argv[2];
    resolution = string_to_resolution(argv[3]);
    quality = atoi(argv[4]);
    subsampling = string_to_subsampling(argv[5]);
    interleaved = atoi(argv[6]);
    iterations = atoi(argv[7]);
    

    if (resolution == RESOLUTION_INVALID) {
        fprintf(stderr, "Error: Invalid resolution. Use one of: 1280x720, 1920x1080, 3840x2160.\n");
        return 1;
    }

    if (quality < 1 || quality > 100) {
        fprintf(stderr, "Error: Quality must be between 1 and 100.\n");
        return 1;
    }

    if (subsampling == SUBSAMPLING_INVALID) {
        fprintf(stderr, "Error: Invalid subsampling type. Use one of: 444, 422, 420, grayscale.\n");
        return 1;
    }
    
    if (subsampling == SUBSAMPLING_GRAYSCALE) {
        fprintf(stderr, "Warning: grayscale should only be used with raw grayscale data (e.g. monochrome non-debayered sensor data).\n");
    }
    
    if (interleaved < 0 || interleaved > 1) {
      fprintf(stderr, "Error: Invalid interleaved setting. Use one of: 1, 0.\n");
        return 1;
    }

    if (iterations <= 0) {
        fprintf(stderr, "Error: Iterations must be a positive integer.\n");
        return 1;
    }
    

    /* Get resolution dimensions */
    resolution_to_dimensions(resolution, &width, &height);
    

    /* Encode the image */
    encode_image_with_gpujpeg(input_filename, output_filename, quality, subsampling, width, height, interleaved, iterations);

    return 0;
}


