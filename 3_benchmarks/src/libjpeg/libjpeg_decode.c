#include "../img_utils.h"
#include "jpeglib.h"
#include <time.h>

int main(int argc, char** argv)
{
  /* Input image related data */
  unsigned char* inbuf = NULL;
  JSAMPROW row_pointer[1];
  int img_h = 0;
  int img_w = 0;
  size_t inbuf_size = 0;

  /* Output image related data */
  unsigned char* outbuf = NULL;
  size_t outbuf_size = 0;

  /* Decoder data */
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  J_DCT_METHOD dct = 0;
  
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
    printf("  │ DCT Method        : 0 = SLOW | 1 = FAST | 2 = FLOAT        │\n");
    printf("  └────────────────────────────────────────────────────────────┘\n");
    return 1;
  }
  
  /* Parsing iterations argv[2] */
  if((iterations = atoi(argv[2])) < 1){
    fprintf(stderr, "Error: No iterations\n");
    return 1;
  }

  /* Parsing DCT algorithm argv[3] */
  dct = atoi(argv[3]);

  if(img_load(argv[1], (char**)inbuf, &inbuf_size)){
    fprintf(stderr, "Error: Failed to load image from file: %s\n", argv[1]);
    return 1;
  }

  /* Initializing the decoder */
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, inbuf, inbuf_size);

  /* Setting up the input image parameters */
  if(jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
    fprintf(stderr, "Error: Failed to read JPEG header from file: %s\n", argv[1]);
    jpeg_destroy_decompress(&cinfo);
    img_destroy(inbuf);
    return 1;
  }

  /* Setting up decompression parameteres */
  cinfo.dct_method = dct;

  /* Setting up output buffer */
  outbuf_size = cinfo.output_width * cinfo.output_height * cinfo.output_components;
  outbuf = (unsigned char*)malloc(outbuf_size);
  
  start_time = clock();
  /* Decompression begins here, parameters and input image
  cannot be changed until it has finished */
  for(int i = 0; i < iterations; i++){
    jpeg_start_decompress(&cinfo);
    while (cinfo.output_scanline < cinfo.output_height) {
      unsigned char* buffer_array[1];
      buffer_array[0] = outbuf + (cinfo.output_scanline * cinfo.output_width * cinfo.output_components);
      jpeg_read_scanlines(&cinfo, buffer_array, 1);
    }
    jpeg_finish_decompress(&cinfo);
  }
  /* Decompression ends here, a new image can be loaded in
  the input buffer and parameters can be changed
  (if not they will remain the same) */
  end_time = clock();
  
  total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
  
  jpeg_destroy_decompress(&cinfo);
  
  img_destroy(inbuf);
  
  printf("%f\n", total_time);
  
  return 0;
}

