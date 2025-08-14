#include "../img_utils.h"
#include "jpeglib.h"
#include <time.h>

int main(int argc, char **argv) {
    /* Input image related data */
    JSAMPLE *inbuf = NULL;
    size_t inbuf_size = 0;
    JSAMPROW row_pointer[1];
    int img_h = 0;
    int img_w = 0;

    /* Output image related data */
    JSAMPLE *outbuf = NULL;
    size_t outbuf_size = 0;

    /* Encoder data */
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    int v_sampling = 0;
    int h_sampling = 0;
    J_COLOR_SPACE colorspace = 0;
    int q = 0;
    J_DCT_METHOD dct = 0;
    int arith = 0;
    int restart_interval = 0;

    /* Benchmark data */
    clock_t start_time, end_time;
    double total_time = 0.0; /* Initialize total_time */
    int iterations = 1;

    /* Input parsing */

    if (argc == 1) {
        printf("\nProgram Input Syntax:\n\n");
        printf("  ┌────────────────────────────────────────────────────────────┐\n");
        printf("  │ Filepath          : RGB24 filepath                         │\n");
        printf("  │ Resolution        : 3840 | 1920 | 1280  (16:9 only!)       │\n");
        printf("  │ Iterations        : 1 ... n                                │\n");
        printf("  │ Subsampling       : 444 | 422 | 420 | 0 (grayscale)        │\n");
        printf("  │ Quality           : 0 ... 100                              │\n");
        printf("  │ Compression       : 0 = Huffman | 1 = Arithmetic           │\n");
        printf("  │ DCT Method        : 0 = SLOW | 1 = FAST | 2 = FLOAT        │\n");
        printf("  │ Restart Interval  : 0 ... n                                │\n");
        printf("  └────────────────────────────────────────────────────────────┘\n");
        return 1;
    }


    /* Parsing Resolution argv[2] */
    if ((img_w = atoi(argv[2])) == 0) {
        fprintf(stderr, "Error: Failed to parse image resolution: %s\n", argv[2]);
        return 1;
    }
    switch (img_w) {
        case 3840: img_h = 2160;
            break;
        case 1920: img_h = 1080;
            break;
        case 1280: img_h = 720;
            break;
        default: fprintf(stderr, "Error: Resolution not supported: %s\n", argv[2]);
            return 1;
    }

    /* Parsing iterations argv[3] */
    if ((iterations = atoi(argv[3])) < 1) {
        fprintf(stderr, "Error: No iterations");
        return 1;
    }

    /* Parsing subsampling argv[4] */
    switch (atoi(argv[4])) {
        case 444: v_sampling = 1;
            h_sampling = 1;
            colorspace = JCS_YCbCr;
            break;
        case 422: v_sampling = 1;
            h_sampling = 2;
            colorspace = JCS_YCbCr;
            break;
        case 420: v_sampling = 2;
            h_sampling = 2;
            colorspace = JCS_YCbCr;
            break;
        default: v_sampling = 1;
            h_sampling = 1;
            colorspace = JCS_GRAYSCALE;
            break;
    }

    /* Parsing quality argv[5] */
    q = atoi(argv[5]);

    /* Parsing Huffman encoding argv[6] */
    arith = atoi(argv[6]);

    /* Parsing DCT algorithm argv[7] */
    dct = atoi(argv[7]);

    /* Parsing restart interval argv[8] */
    restart_interval = atoi(argv[8]);

    if (img_load(argv[1], &inbuf, &inbuf_size)) {
        fprintf(stderr, "Error: Failed to load image from file: %s\n", argv[1]);
        return 1;
    }

    /* Initializing the encoder */
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_mem_dest(&cinfo, &outbuf, &outbuf_size);

    /* Setting up the input image parameters */
    cinfo.image_width = img_w;
    cinfo.image_height = img_h;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    /* Setting up the compression parameters */
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, q, TRUE); /* Compression quality, chosen by the user */
    jpeg_set_colorspace(&cinfo, colorspace); /* Output colorspace, chosen by the user */
    cinfo.comp_info[0].v_samp_factor = v_sampling; /* Chroma subsampling options, chosen by the user */
    cinfo.comp_info[0].h_samp_factor = h_sampling;
    cinfo.arith_code = arith; /* Arithmetic or Huffman encoding, chosen by the user */
    cinfo.dct_method = dct; /* Default or fast DCT, chosen by the user */
    cinfo.restart_interval = restart_interval; /* Presence of restart intervals, chosen by the user */

    start_time = clock();
    /* Compression begins here, parameters and input image
       cannot be changed until it has finished             */
    for (int i = 0; i < iterations; i++) {
        jpeg_start_compress(&cinfo, TRUE);
        while (cinfo.next_scanline < cinfo.image_height) {
            row_pointer[0] = &inbuf[cinfo.next_scanline * cinfo.image_width * cinfo.input_components];
            jpeg_write_scanlines(&cinfo, row_pointer, 1);
        }
        jpeg_finish_compress(&cinfo);
    }
    /* Compression ends here, a new image can be loaded in
       the input buffer and parameters can be changed
       (if not they will remain the same)                  */
    end_time = clock();

    total_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;

    jpeg_destroy_compress(&cinfo);

    img_destroy(inbuf);
    img_destroy(outbuf);

    printf("%f\n", total_time);

    return 0;
}
