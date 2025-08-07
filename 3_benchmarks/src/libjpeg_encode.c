#include "img_utils.h"
#include "jpeglib.h"

int main(int argc, char** argv)
{
  /* Input image related data */
  char* input_img = NULL;
  JSAMPROW row_pointer[1];
  int img_h = 0;
  int img_w = 0;
  size_t input_img_size = 0;

  /* Output image related data */
  char* output_img = NULL;
  size_t output_img_size = 0;

  /* Encoder data */
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  
  /* Initializing the encoder */
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, outfile);
  
  /* Setting up the input image parameters */
  cinfo.image_width = img_h;
  cinfo.image_height = img_w;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  
  /* Setting up the compression parameters */
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE);  /* Compression quality, chosen by the user */
  jpeg_set_colorspace (&cinfo, JCS_RGB);    /* We always use RGB */
  cinfo->comp_info[0].v_samp_factor = 2;    /* Chroma subsampling options, 4:2:0 in this case */
  cinfo->comp_info[0].h_samp_factor = 2;
  cinfo.arith_code = FALSE;                 /* Arthmetic or Huffman encoding, chosen by the user */
  cinfo.dct_method = JDCT_DEFAULT;          /* Default or fast DCT, chosen by the user */
  cinfo.restart_interval = 0;               /* Presence of restart intervals, chosen by the user */
  
  /* Compression begins here, parameters and input image
     cannot be changed until it has finished             */
  jpeg_start_compress(&cinfo, TRUE);
  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = input_img[cinfo.next_scanline];
    jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }
  jpeg_finish_compress(&cinfo);
  /* Compression ends here, a new image can be loaded in
     the input buffer and parameters can be changed
     (if not they will remain the same)                  */
  
  jpeg_destroy_compress(&cinfo);
  
  return 0;
}
