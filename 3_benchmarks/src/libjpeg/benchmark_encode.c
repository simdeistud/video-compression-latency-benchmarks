#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <jpeglib.h>
#include "../../include/benchmark.h"
#include "../../include/img_utils.h"
#include "../../include/subs_utils.h"
#include "../../include/res_utils.h"

/*
 * Initializes the libjpeg compression object.
 *
 * Parameters:
 * cinfo: Pointer to the jpeg_compress_struct.
 * jerr: Pointer to the jpeg_error_mgr.
 *
 * Returns:
 * 0 on success, non-zero on failure.
 */
int encoder_initialization(struct jpeg_compress_struct *cinfo, struct jpeg_error_mgr *jerr) {
  cinfo->err = jpeg_std_error(jerr);
  jpeg_create_compress(cinfo);

  return 0;
}

/*
 * Performs any initial setup required for the encoder.
 * Currently, this function does nothing.
 *
 * Returns:
 * 0 on success.
 */
int encoder_setup(void) {
  return 0;
}

/*
 * Sets up the encoding parameters for the JPEG compression.
 *
 * Parameters:
 * cinfo: Pointer to the jpeg_compress_struct.
 * width: Width of the input image.
 * height: Height of the input image.
 * pixel_size: Number of color components per pixel (e.g., 3 for RGB).
 * quality: JPEG quality setting (1-100).
 * subsampling: Subsampling type (e.g., SUBSAMPLING_444, SUBSAMPLING_GRAYSCALE).
 * dct: DCT method to use (e.g., JDCT_DEFAULT, JDCT_FASTEST).
 *
 * Returns:
 * 0 on success, non-zero on failure (e.g., invalid subsampling).
 */
int encoding_parameters_setup(struct jpeg_compress_struct *cinfo, int width, int height, int pixel_size, int quality, subsampling_t subsampling, J_DCT_METHOD dct) {
  int h_samp_factor, v_samp_factor;

  /* Setting mandatory parameters */
  cinfo->image_width = width;
  cinfo->image_height = height;
  cinfo->input_components = pixel_size;
  cinfo->in_color_space = JCS_RGB;
  jpeg_set_defaults(cinfo);

  /* Setting optional parameters */
  subsampling_to_libjpeg(subsampling, &h_samp_factor, &v_samp_factor);
  if (subsampling == SUBSAMPLING_GRAYSCALE) {
    cinfo->jpeg_color_space = JCS_GRAYSCALE;
    jpeg_set_defaults(cinfo); /* Re-apply defaults for grayscale */
  } else if (h_samp_factor > 0 && v_samp_factor > 0) {
    cinfo->comp_info[0].h_samp_factor = h_samp_factor;
    cinfo->comp_info[0].v_samp_factor = v_samp_factor;
  } else {
    fprintf(stderr, "Error: Invalid subsampling type.\n");
    return 1;
  }
  jpeg_set_quality(cinfo, quality, TRUE);
  cinfo->dct_method = dct;

  return 0;
}

/*
 * Cleans up encoding parameters by resetting them to default values.
 *
 * Parameters:
 * cinfo: Pointer to the jpeg_compress_struct.
 *
 * Returns:
 * 0 on success.
 */
int encoding_parameters_cleanup(struct jpeg_compress_struct *cinfo) {
  jpeg_set_defaults(cinfo);
  return 0;
}

/*
 * Cleans up resources allocated during the encoding process.
 *
 * Parameters:
 * cinfo: Pointer to the jpeg_compress_struct (may not be used in all cases).
 * output_buffer: Pointer to the output buffer allocated by libjpeg.
 * It will be freed if it's not NULL.
 *
 * Returns:
 * 0 on success.
 */
int encoder_cleanup(struct jpeg_compress_struct *cinfo, unsigned char *output_buffer) {
  if (output_buffer != NULL) {
    free(output_buffer);
    output_buffer = NULL; /* Prevent dangling pointer */
  }
  return 0;
}

/*
 * Destroys the libjpeg compression object, freeing associated memory.
 *
 * Parameters:
 * cinfo: Pointer to the jpeg_compress_struct.
 *
 * Returns:
 * 0 on success.
 */
int encoder_destroy(struct jpeg_compress_struct *cinfo) {
  jpeg_destroy_compress(cinfo);
  return 0;
}

/*
 * Encodes an image from the specified input file to a JPEG file.
 *
 * Parameters:
 * input_filename: Path to the input image file.
 * output_filename: Path to the output JPEG file.
 * quality: JPEG quality setting (1-100).
 * subsampling: Subsampling type.
 * width: Width of the input image.
 * height: Height of the input image.
 * dct: DCT method to use.
 * iterations: Number of times to perform the encoding (for benchmarking).
 *
 * Returns:
 * 0 on success, non-zero on failure (e.g., file loading error, encoding error).
 */
int encode_image_with_libjpeg(const char *input_filename, const char *output_filename, int quality, subsampling_t subsampling, int width, int height, J_DCT_METHOD dct, int iterations) {
  unsigned char *input_buffer = NULL;
  size_t input_buffer_size = 0;
  unsigned char *output_buffer = NULL;
  size_t output_buffer_size = 0;
  int pixel_size = 3; /* RGB has 3 bytes per pixel */
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPROW row_pointer[1];
  clock_t start_time, end_time;
  double total_time = 0.0; /* Initialize total_time */
  int i;

  input_buffer = (unsigned char *)img_load(input_filename, &input_buffer_size);

  if (!input_buffer) {
    fprintf(stderr, "Error: Failed to load data from file: %s\n", input_filename);
    return 1;
  }

  if (encoder_initialization(&cinfo, &jerr) != 0) {
    img_destroy(input_buffer);
    return 1;
  }

  encoder_setup();

  if (encoding_parameters_setup(&cinfo, width, height, pixel_size, quality, subsampling, dct) != 0) {
    jpeg_destroy_compress(&cinfo);
    img_destroy(input_buffer);
    return 1;
  }

  /* Measure encoding time */
  start_time = clock();
  for (i = 0; i < iterations; i++) {

    jpeg_mem_dest(&cinfo, &output_buffer, &output_buffer_size);

    jpeg_start_compress(&cinfo, TRUE);

    /* Write scanlines */
    while (cinfo.next_scanline < cinfo.image_height) {
      row_pointer[0] = &input_buffer[cinfo.next_scanline * width * pixel_size];
      if (jpeg_write_scanlines(&cinfo, row_pointer, 1) != 1) {
        fprintf(stderr, "Error: Failed to write scanline %d.\n", cinfo.next_scanline);
        encoder_cleanup(&cinfo, output_buffer);
        encoder_destroy(&cinfo);
        encoding_parameters_cleanup(&cinfo);
        img_destroy(input_buffer);
        return 1;
      }
    }

    jpeg_finish_compress(&cinfo);
  }
  end_time = clock();

  total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

  img_save(output_filename, output_buffer, output_buffer_size);

  encoding_parameters_cleanup(&cinfo);
  encoder_cleanup(&cinfo, output_buffer);
  encoder_destroy(&cinfo);
  img_destroy(input_buffer); /* Free input buffer */

  printf("%f\n", total_time);

  return 0;
}

/*
 * Main function to drive the JPEG encoding process.
 * Takes command-line arguments for input file, output file, resolution,
 * quality, subsampling, DCT algorithm, and number of iterations.
 */
int main(int argc, char *argv[]) {
  const char *input_filename;
  int width, height;
  resolution_t resolution;
  int quality;
  subsampling_t subsampling;
  const char *output_filename;
  int iterations;
  const char *dct_str = NULL;
  J_DCT_METHOD dct;

  if (argc != 8) {
    fprintf(stderr, "Usage: %s <input_file> <output_file> <resolution> <quality> <subsampling> <dct_algorithm> <iterations>\n", argv[0]);
    return 1;
  }

  input_filename = argv[1];
  output_filename = argv[2];
  resolution = string_to_resolution(argv[3]);
  quality = atoi(argv[4]);
  subsampling = string_to_subsampling(argv[5]);
  dct_str = argv[6];
  iterations = atoi(argv[7]);

  if (resolution == RESOLUTION_INVALID) {
    fprintf(stderr, "Error: Invalid resolution. Use one of: 1280x720, 1920x1080, 3840x2160.\n");
    return 1;
  }

  if (quality < 1 || quality > 100) {
    fprintf(stderr, "Error: Quality must be between 1 and 100.\n");
    return 1;
  }
  
  if (subsampling == SUBSAMPLING_GRAYSCALE) {
    fprintf(stderr, "Error: grayscale subsampling should only be used with grayscale input images!.\n");
    return 1;
  }

  if (subsampling == SUBSAMPLING_INVALID) {
    fprintf(stderr, "Error: Invalid subsampling type. Use one of: 444, 422, 420, grayscale.\n");
    return 1;
  }

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

  resolution_to_dimensions(resolution, &width, &height);

  encode_image_with_libjpeg(input_filename, output_filename, quality, subsampling, width, height, dct, iterations);

  return 0;
}
