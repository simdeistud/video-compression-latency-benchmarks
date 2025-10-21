#include "../img_utils.h"
#include "webp/encode.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>


void print_syntax()
{
    printf("\nProgram Input Syntax:\n\n");
    printf("  ┌──────────────────────────────────────┐\n");
    printf("  │ -w  <width [px]>                     │\n");
    printf("  │ -h  <height [px]>                    │\n");
    printf("  │ -q  <quality_factor [0...100]>       │\n");
    printf("  │ -s  <speed [0...6]>                  │\n");
    printf("  │ -t  <multithread [0|1]>              │\n");
    printf("  │ -i  <iterations [1...n]>             │\n");
    printf("  │ -b  <benchmark mode>                 │\n");
    printf("  │ -o  <output mode [FILEPATH|-]>       │\n");
    printf("  └──────────────────────────────────────┘\n");
}

int main(int argc, char** argv)
{
    /* Input image related data */
    WebPPicture pic;
    unsigned char* inbuf = NULL;
    size_t inbuf_size = 0;

    /* Output image related data */
    WebPMemoryWriter wrt;
    unsigned char* outbuf = NULL;
    size_t outbuf_size = 0;

    /* Encoder data */
    WebPConfig config;

    /* Benchmark data */
    clock_t setup_start_time, setup_end_time;
    clock_t encoding_start_time, encoding_end_time;
    clock_t cleanup_start_time, cleanup_end_time;

    /* Input parsing */
    int width = 0, height = 0, speed = 0, threaded = 0, iterations = 0;
    float quality_factor = 0;
    char *output = NULL;
    int benchmark = 0;
    int opt;

    while ((opt = getopt(argc, argv, "w:h:q:s:t:i:bo:")) != -1)
    {
        switch (opt)
        {
        case 'w': width = atoi(optarg);
            break;
        case 'h': height = atoi(optarg);
            break;
        case 'q': quality_factor = atof(optarg);
            break;
        case 's': speed = atoi(optarg);
            break;
        case 't': threaded = atoi(optarg);
            break;
        case 'i': iterations = atoi(optarg);
            break;
        case 'b': benchmark = 1;
            break;
        case 'o': output = optarg;
            break;
        default:
            fprintf(stderr, "Usage error\n");
            print_syntax();
            exit(EXIT_FAILURE);
        }
    }

    if (img_load_stdin(&inbuf, &inbuf_size))
    {
        fprintf(stderr, "Error: Failed to load image from stdin\n");
        return 1;
    }

    setup_start_time = clock();
    /* Encoder setup starts here */
    if (!WebPConfigPreset(&config, WEBP_PRESET_DEFAULT, quality_factor)) {
        return 0;   // version error
    }
    config.method = speed;
    config.thread_level = threaded;
    WebPValidateConfig(&config);
    if (!WebPPictureInit(&pic)) {
        return 0;  // version error
    }
    pic.width = width;
    pic.height = height;
    /*if (!WebPPictureAlloc(&pic)) {
        return 0;   // memory error
    }*/
    WebPPictureImportRGB(&pic, inbuf, width * 3);
    WebPMemoryWriterInit(&wrt);
    pic.writer = WebPMemoryWrite;
    pic.custom_ptr = &wrt;
    outbuf = wrt.mem;
    /* Encoder setup ends here */
    setup_end_time = clock();

    /* Test run to see if everything works */
    WebPEncode(&config, &pic);
    WebPPictureFree(&pic);
    //WebPMemoryWriterClear(&wrt);
    outbuf = wrt.mem;
    outbuf_size = wrt.size;

    encoding_start_time = clock();
    /* Compression begins here, parameters and input image
       cannot be changed until it has finished             */
    for (int i = 0; i < iterations; i++)
    {

    }
    /* Compression ends here, a new image can be loaded in
       the input buffer and parameters can be changed
       (if not they will remain the same)                  */
    encoding_end_time = clock();

    cleanup_start_time = clock();
    /* Encoder cleanup begins here */
    /* Encoder cleanup ends here */
    cleanup_end_time = clock();

    if (benchmark)
    {
        double cleanup_time, encoding_time, setup_time, total_time;
        setup_time = (double)(setup_end_time - setup_start_time) / CLOCKS_PER_SEC;
        encoding_time = (double)(encoding_end_time - encoding_start_time) / CLOCKS_PER_SEC;
        cleanup_time = (double)(cleanup_end_time - cleanup_start_time) / CLOCKS_PER_SEC;
        total_time = setup_time + encoding_time + cleanup_time;
        printf("setup:%f\nencoding:%f\ncleanup:%f\ntotal:%f\n", setup_time, encoding_time, cleanup_time, total_time);
    }

    if (output == NULL)
    {
        return 0;
    }

    if (strcmp(output, "-") == 0)
    {
        size_t written = fwrite(outbuf, 1, outbuf_size, stdout);
        if (written != outbuf_size)
        {
            perror("Couldn't write to stdout");
            return 1;
        }
    }
    else
    {
        if (img_save(output, &outbuf, outbuf_size))
        {
            return 1;
        }
    }

    return 0;
}
