#include <stdint.h>
#include "../img_utils.h"
#include "x264.h"
#include "x264_config.h"
#include <time.h>

int main(int argc, char **argv) {
    /* Input image related data */
    uint8_t *inbuf = NULL;
    x264_picture_t inpic;
    size_t inbuf_size = 0;
    int img_h = 0;
    int img_w = 0;

    /* Output image related data */
    x264_picture_t outpic;
    size_t outbuf_size = 0;

    /* Encoder data */
    x264_t* encoder = NULL;
    x264_param_t encoder_params;
    x264_nal_t nal;
    char* profile;
    int q;
    char* qval = argv[6];
    int speed;
    int subsampling;

    /* Benchmark data */
    clock_t start_time, end_time;
    double total_time = 0.0; /* Initialize total_time */
    int iterations = 1;

    /* Input parsing */

    if (argc == 1) {
        printf("\nProgram Input Syntax:\n\n");
        printf("  ┌───────────────────────────────────────────────────────────────────┐\n");
        printf("  │ Filepath          : YUV filepath                                  │\n");
        printf("  │ Resolution        : 3840 | 1920 | 1280  (16:9 only!)              │\n");
        printf("  │ Iterations        : 1 ... n                                       │\n");
        printf("  │ Subsampling       : 444 | 422 | 420                               │\n");
        printf("  │ Quality           : 0 = QP | 1 = CRF | 2 = CBR                    │\n");
        printf("  │ Quality value     : 0 ... 52 QP | 1 ... 63 CRF | kb/s CBR         │\n");
        printf("  │ Speed             : 0 = ultrafast | 1 = superfast | 2 = veryfast  │\n");
        printf("  └───────────────────────────────────────────────────────────────────┘\n");
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
        case 444: profile = "high444";
            subsampling = X264_CSP_I444;
            break;
        case 422: profile = "high422";
            subsampling = X264_CSP_I422;
            break;
        case 420: profile = "baseline";
            subsampling = X264_CSP_I420;
            break;
        default: profile = "baseline";
            subsampling = X264_CSP_I400;
            break;
    }

    /* Parsing quality metric argv[5] */
    q = atoi(argv[5]);

    /* Parsing speed algorithm argv[7] */
    speed = atoi(argv[7]);

    if (img_load(argv[1], &inbuf, &inbuf_size)) {
        fprintf(stderr, "Error: Failed to load image from file: %s\n", argv[1]);
        return 1;
    }

    /* Setting up the compression parameters */
    x264_param_default(&encoder_params);

    /* Initializing the encoder */
    encoder = x264_encoder_open(&encoder_params);
    x264_param_default_preset(&encoder_params, argv[7] , "realtime,fastdecode" );
    encoder_params.i_width  = img_w;
    encoder_params.i_height = img_h;
    encoder_params.i_fps_num = 30;
    encoder_params.i_fps_den = 30;
    encoder_params.i_timebase_num = 30;
    encoder_params.i_timebase_den = 30;
    encoder_params.i_csp = subsampling;
    // All-intra
    x264_param_parse(&encoder_params, "keyint", "1");
    x264_param_parse(&encoder_params, "min-keyint", "1");
    x264_param_parse(&encoder_params, "scenecut", "0");
    x264_param_parse(&encoder_params, "bframes", "0");
    x264_param_parse(&encoder_params, "ref", "1");
    // Latency
    x264_param_parse(&encoder_params, "rc-lookahead", "0");
    x264_param_parse(&encoder_params, "sync-lookahead", "0");
    x264_param_parse(&encoder_params, "frame-threads", "1");
    x264_param_parse(&encoder_params, "sliced-threads", "1");
    // Bitstream
    x264_param_parse(&encoder_params, "repeat-headers", "1");
    x264_param_parse(&encoder_params, "annexb", "1");
    if (q == 0) {
        x264_param_parse(&encoder_params, "qp", qval);       // typical: 20–26
        x264_param_parse(&encoder_params, "nal-hrd", "none");
    }
    if (q == 1) {
        x264_param_parse(&encoder_params, "crf", qval);
        x264_param_parse(&encoder_params, "nal-hrd", "none");
    }
    if (q == 2) {
        x264_param_parse(&encoder_params, "bitrate",      qval);
        x264_param_parse(&encoder_params, "vbv-maxrate",  qval);
        x264_param_parse(&encoder_params, "vbv-bufsize",  qval);
        x264_param_parse(&encoder_params, "vbv-init",     "0.0");
        x264_param_parse(&encoder_params, "nal-hrd",      "cbr");
    }
    x264_param_apply_fastfirstpass(&encoder_params);
    x264_param_apply_profile(&encoder_params, profile);

    /* Setting up the input image parameters */
    x264_picture_init(&inpic);
    inpic.img.i_csp = subsampling;
    inpic.img.i_plane = 3;
    inpic.img.plane[0] = inbuf;
    inpic.img.plane[1] = inbuf + (img_w * 3);
    inpic.img.plane[2] = inpic.img.plane[1] + (img_w * 3);
    inpic.img.i_stride[0] = img_w * 3;     // in bytes
    inpic.img.i_stride[1] = img_w * 3;
    inpic.img.i_stride[2] = img_w * 3;
    inpic.i_pts = 0;

    x264_nal_t *nal = NULL; int i_nal = 0;
    x264_encoder_encode(h, &nal, &i_nal, &inpic, &outpic); // discard output


    /* Test run to see if everything works */
    x264_encoder_encode(encoder, x264_nal_t **pp_nal, int *pi_nal, x264_picture_t *pic_in, x264_picture_t *pic_out );

    start_time = clock();
    /* Compression begins here, parameters and input image
       cannot be changed until it has finished             */
    for (int i = 0; i < iterations; i++) {

    }
    /* Compression ends here, a new image can be loaded in
       the input buffer and parameters can be changed
       (if not they will remain the same)                  */
    end_time = clock();

    total_time = (double) (end_time - start_time) / CLOCKS_PER_SEC;

    /* Clean the created objects */
    x264_param_cleanup(&encoder_params);
    x264_encoder_close(encoder);

    printf("%f\n", total_time);

    return 0;
}

