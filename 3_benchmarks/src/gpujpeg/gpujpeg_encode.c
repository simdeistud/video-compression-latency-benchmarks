#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "libgpujpeg/gpujpeg.h"
#include "../img_utils.h"

int main(int argc, char **argv) {
    /* Input image related data */
    uint8_t *inbuf = NULL;
    size_t inbuf_size = 0;
    int img_h = 0;
    int img_w = 0;

    /* Output image related data */
    uint8_t *outbuf = NULL;
    size_t outbuf_size = 0;

    /* Encoder data */
    struct gpujpeg_parameters param;
    struct gpujpeg_image_parameters param_image;
    struct gpujpeg_encoder *encoder;
    struct gpujpeg_encoder_input encoder_input;
    unsigned int subsampling = 0;
    int q = 0;
    int interleaved = 0;
    int restart_interval = 0;

    /* Benchmark data */
    clock_t start_time, end_time;
    double total_time = 0.0; /* Initialize total_time */
    int iterations = 1;

    /* Input parsing */

    if (argc != 8) {
        printf("\nProgram Input Syntax:\n\n");
        printf("  ┌────────────────────────────────────────────────────────────┐\n");
        printf("  │ Filepath          : RGB24 filepath                         │\n");
        printf("  │ Resolution        : 3840 | 1920 | 1280  (16:9 only!)       │\n");
        printf("  │ Iterations        : 1 ... n                                │\n");
        printf("  │ Subsampling       : 444 | 422 | 420 | 0 (grayscale)        │\n");
        printf("  │ Quality           : 0 ... 100                              │\n");
        printf("  │ Interleaved       : 0 = NO | 1 = YES                       │\n");
        printf("  │ Restart Interval  : 0 = NO | 1 = YES                       │\n");
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
        case 444: subsampling = 0x11111100U;
            break;
        case 422: subsampling = 0x21111100U;
            break;
        case 420: subsampling = 0x22111100U;
            break;
        default: subsampling = 0U;
            break;
    }

    /* Parsing quality argv[5] */
    q = atoi(argv[5]);

    /* Parsing is interleaved argv[6] */
    interleaved = atoi(argv[6]);

    /* Parsing restart interval argv[7] */
    restart_interval = atoi(argv[7]);

    if (img_load(argv[1], &inbuf, &inbuf_size)) {
        fprintf(stderr, "Error: Failed to load image from file: %s\n", argv[1]);
        return 1;
    }

    /* Initializing the encoder */
    /* We use the default CUDA device, if you want to test on different ones, change the first argument of gpujpeg_init_device */
    if (gpujpeg_init_device(0, 0)) {
        perror("Failed to initialize GPU device");
        return 1;
    }

    encoder = gpujpeg_encoder_create(0);
    if (encoder == NULL) {
        perror("Failed to create encoder");
        return 1;
    }

    /* Setting up the input image parameters */
    gpujpeg_image_set_default_parameters(&param_image);
    param_image.width = img_w;
    param_image.height = img_h;
    param_image.color_space = GPUJPEG_RGB;
    param_image.pixel_format = GPUJPEG_444_U8_P012;
    gpujpeg_parameters_chroma_subsampling(&param, subsampling);
    gpujpeg_encoder_input_set_image(&encoder_input, inbuf);

    /* Setting up the compression parameters */
    gpujpeg_set_default_parameters(&param);
    param.quality = q;
    param.interleaved = interleaved;
    param.segment_info = interleaved;
    param.restart_interval = restart_interval ? (img_w != 3840 ? 8 : 32) : 0;

    /* Test run to see if everything works */
    if (gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input, &outbuf, &outbuf_size)) {
        perror("Failed to encode image");
        return 1;
    }
    img_save("out.jpeg", &outbuf, outbuf_size);

    start_time = clock();
    /* Compression begins here, parameters and input image
       cannot be changed until it has finished             */
    for (int i = 0; i < iterations; i++) {
        gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input, &outbuf, &outbuf_size);
    }
    /* Compression ends here, a new image can be loaded in
       the input buffer and parameters can be changed
       (if not they will remain the same)                  */
    end_time = clock();

    total_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;

    gpujpeg_encoder_destroy(encoder);

    img_destroy(inbuf);

    printf("%f\n", total_time);

    return 0;
}