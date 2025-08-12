#include "../img_utils.h"
#include "jpeglib.h"
#include <time.h>

int main(int argc, char **argv) {
    /* Input image related data */
    JSAMPLE *inbuf = NULL;
    size_t inbuf_size = 0;
    JSAMPROW row_pointer[1];

    /* Output image related data */
    JSAMPLE *outbuf = NULL;
    size_t outbuf_size = 0;

    /* Decoder data */
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    J_DCT_METHOD dct = 0;
    int row_stride;

    /* Benchmark data */
    clock_t start_time, end_time;
    double total_time = 0.0;
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
    if ((iterations = atoi(argv[2])) < 1) {
        fprintf(stderr, "Error: No iterations\n");
        return 1;
    }

    /* Parsing DCT algorithm argv[3] */
    dct = atoi(argv[3]);

    if (img_load(argv[1], &inbuf, &inbuf_size)) {
        fprintf(stderr, "Error: Failed to load image from file: %s\n", argv[1]);
        return 1;
    }

    /* Initializing the decoder */
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    /* Setting up decompression parameters */
    cinfo.dct_method = dct;

    /* Test run to see if everything works and to set image parameters.
     * This benchmark measures the decoding time given that the most
     * important metadata of the decoded raw image (height, length, bits per channel)
     * are already known beforehand. This benchmark is not made to measure
     * decoding random images, for which including these preparatory steps
     * in the benchmark would be reasonable!
     */
    jpeg_mem_src(&cinfo, inbuf, inbuf_size); /* Image source has to be set before every run even though it's the same pointer... */
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        fprintf(stderr, "Error: Failed to read JPEG header from file: %s\n", argv[1]);
        jpeg_destroy_decompress(&cinfo);
        img_destroy(inbuf);
        return 1;
    } /* Reading the JPEG header is mandatory before starting the decompression */
    jpeg_start_decompress(&cinfo);
    outbuf_size = cinfo.output_width * cinfo.output_height * cinfo.output_components; /* These cinfo fields are filled only AFTER decompression has started */
    outbuf = (JSAMPLE *) malloc(outbuf_size);
    while (cinfo.output_scanline < cinfo.output_height) {
        row_pointer[0] = outbuf + (cinfo.output_scanline * cinfo.output_width * cinfo.output_components);
        jpeg_read_scanlines(&cinfo, row_pointer, 1);
    }
    jpeg_finish_decompress(&cinfo);

    start_time = clock();
    /* Decompression benchmark begins here, parameters and input image
    cannot be changed until it has finished */
    for (int i = 0; i < iterations; i++) {
        jpeg_mem_src(&cinfo, inbuf, inbuf_size);
        jpeg_read_header(&cinfo, TRUE);
        jpeg_start_decompress(&cinfo);
        while (cinfo.output_scanline < cinfo.output_height) {
            row_pointer[0] = outbuf + (cinfo.output_scanline * cinfo.output_width * cinfo.output_components);
            jpeg_read_scanlines(&cinfo, row_pointer, 1);
        }
        jpeg_finish_decompress(&cinfo);
    }
    /* Decompression ends here, a new image can be loaded in
    the input buffer and parameters can be changed
    (if not they will remain the same) */
    end_time = clock();

    total_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;

    jpeg_destroy_decompress(&cinfo);

    img_destroy(inbuf);
    img_destroy(outbuf);

    printf("%f\n", total_time);

    return 0;
}
