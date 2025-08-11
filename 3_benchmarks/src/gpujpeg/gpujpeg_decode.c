#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "libgpujpeg/gpujpeg.h"
#include "../img_utils.h"

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
    
}

int main(int argc, char* argv[])
{  
  /* Input image related data */
  uint8_t* inbuf = NULL;
  size_t inbuf_size = 0;

  /* Output image related data */
  uint8_t* outbuf = NULL;
  size_t outbuf_size = 0;

  /* Decoder data */
  struct gpujpeg_parameters param;
  struct gpujpeg_image_parameters param_image;
  struct gpujpeg_decoder* decoder;
  struct gpujpeg_decoder_output decoder_output;
  
  /* Benchmark data */
  clock_t start_time, end_time;
  double total_time = 0.0; /* Initialize total_time */
  int iterations = 1;
  
  /* Input parsing */
  
  if (argc == 1) {
    printf("\nProgram Input Syntax:\n\n");
    printf("  ┌────────────────────────────────────────────────────────────┐\n");
    printf("  │ Filepath          : JPEG imagepath                         │\n");
    printf("  │ Iterations        : 1 ... n                                │\n");
    printf("  └────────────────────────────────────────────────────────────┘\n");
    return 1;
  }
  
  /* Parsing iterations argv[2] */
  if((iterations = atoi(argv[2])) < 1){
    fprintf(stderr, "Error: No iterations\n");
    return 1;
  }
  
  if(img_load(argv[1], (char**)inbuf, &inbuf_size)){
    fprintf(stderr, "Error: Failed to load image from file: %s\n", argv[1]);
    return 1;
  }

  /* Initializing the decoder */
  /* We use the default CUDA device, if you want to test on different ones, change the first argument of gpujpeg_init_device */
  if (gpujpeg_init_device(0, 0)) {
    perror("Failed to initialize GPU device");
    return 1;
  }
  
  decoder = gpujpeg_decoder_create(0);
  if (decoder == NULL ){
    perror("Failed to create decoder");
    return 1;
  }    
  
  gpujpeg_decoder_set_output_format(decoder, GPUJPEG_RGB, GPUJPEG_444_U8_P012);
  gpujpeg_decoder_output_set_default(&decoder_output);
    
  if (gpujpeg_decoder_decode(decoder, inbuf, inbuf_size, &decoder_output)){
    perror("Failed to decode image");
    return 1;
  }

  start_time = clock();
  /* Decompression begins here, parameters and input image
  cannot be changed until it has finished */
  for (int i = 0; i < iterations; i++) {
    gpujpeg_decoder_decode(decoder, inbuf, inbuf_size, &decoder_output);
  }
  /* Decompression ends here, a new image can be loaded in
  the input buffer and parameters can be changed
  (if not they will remain the same) */
  end_time = clock();

  total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
  
  img_destroy(inbuf);
  
  printf("%f\n", total_time);

  return 0;
}
