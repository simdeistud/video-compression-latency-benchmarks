#include "libgpujpeg/gpujpeg.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <fstream>

int main(int argc, char** argv)
{
    /* I/O */
    uint8_t* inbuf = NULL;
    size_t inbuf_size = 0;
    uint8_t* outbuf = NULL;
    size_t outbuf_size = 0;

    /* Encoder data */
    struct gpujpeg_parameters param;
    struct gpujpeg_image_parameters param_image;
    struct gpujpeg_encoder* encoder;
    struct gpujpeg_encoder_input encoder_input;

    /* Input parsing */
    const int width = 1280, height = 720, quality = 50, restart_interval = 8, bpp = 3;

    // RGB24 Gradient
    inbuf_size = width * height * bpp;
    inbuf = reinterpret_cast<uint8_t*>(malloc(inbuf_size));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t idx = (y * width + x) * 3;

            // Simple gradient:
            // R = horizontal ramp
            // G = vertical ramp
            // B = mix of both
            uint8_t r = static_cast<uint8_t>((x * 255) / (width - 1));
            uint8_t g = static_cast<uint8_t>((y * 255) / (height - 1));
            uint8_t b = static_cast<uint8_t>((r + g) / 2);

            inbuf[idx + 0] = r;
            inbuf[idx + 1] = g;
            inbuf[idx + 2] = b;
        }
    }

    /* Encoder setup starts here */
    /* We use the default CUDA device, if you want to test on different ones, change the first argument of gpujpeg_init_device */
    if (gpujpeg_init_device(0, 0))
    {
        perror("Failed to initialize GPU device");
        return 1;
    }

    encoder = gpujpeg_encoder_create(0);
    if (encoder == NULL)
    {
        perror("Failed to create encoder");
        return 1;
    }

    /* Setting up the input image parameters */
    gpujpeg_image_set_default_parameters(&param_image);
    param_image.width = width;
    param_image.height = height;
    param_image.color_space = GPUJPEG_RGB;
    param_image.pixel_format = GPUJPEG_444_U8_P012;

    /* Setting up the compression parameters */
    gpujpeg_set_default_parameters(&param);
    param.quality = quality;
    param.interleaved = 1;
    param.segment_info = param.interleaved;
    param.restart_interval = restart_interval;
    gpujpeg_parameters_chroma_subsampling(&param, GPUJPEG_SUBSAMPLING_420);
    /* Encoder setup ends here */

    /* Compression begins here, parameters and input image
       cannot be changed until it has finished             */
    for (int i = 0; i < 10; i++)
    {
        uint8_t* curr_img = reinterpret_cast<uint8_t*>(malloc(inbuf_size));
        memcpy(curr_img, inbuf, inbuf_size);
        uint8_t* curr_out = NULL;
        gpujpeg_encoder_input_set_image(&encoder_input, curr_img);
        gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input, &curr_out, &outbuf_size);
        free(curr_img);
        outbuf = reinterpret_cast<uint8_t*>(malloc(outbuf_size));
        memcpy(outbuf, curr_out, outbuf_size);
        //free(curr_out); // OUTPUT IS FREED ONLY ON ENCODER DESTRUCTION!
    }
    /* Compression ends here, a new image can be loaded in
       the input buffer and parameters can be changed
       (if not they will remain the same)                  */

    /* Encoder cleanup begins here */
    gpujpeg_encoder_destroy(encoder);
    /* Encoder cleanup ends here */

    if (outbuf)
    {
        std::ofstream outFile("compressed.jpeg", std::ios::binary);
        outFile.write(reinterpret_cast<const char*>(outbuf), outbuf_size);
        outFile.close();
    }

    return 0;
}