#include "img_utils.h"
#include "jpeglib.h"

int main(int argc, char** argv)
{
  /* Input image related data */
  JSAMPLE * input_img = NULL;
  JSAMPROW row_pointer[1];
  int img_h = 0;
  int img_w = 0;
  size_t input_img_size = 0;

  /* Output image related data */
  char* output_img = NULL;
  size_t output_img_size = 0;

  /* Decoder data */
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  
  /* Initializing the decoder */
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, infile);
  
  /* Setting up the input image parameters */
  jpeg_read_header(&cinfo, TRUE);
  
  /* Setting up decompression paramteres */
  cinfo.dct_method = JDCT_DEFAULT;
  
  /* Decompression begins here, parameters and input image
  cannot be changed until it has finished */
  jpeg_start_decompress(&cinfo);
  while (cinfo.output_scanline < cinfo.output_height) {
    jpeg_read_scanlines(&cinfo, output_img, 1);
  }
  jpeg_finish_decompress(&cinfo);
  /* Decompression ends here, a new image can be loaded in
  the input buffer and parameters can be changed
  (if not they will remain the same) */
  
  jpeg_destroy_decompress(&cinfo);
  
  return 0;
}

