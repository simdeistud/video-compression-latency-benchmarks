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

    struct gpujpeg_parameters param;
    struct gpujpeg_image_parameters param_image;

    /* Encoder data */
    struct gpujpeg_encoder* encoder;
    struct gpujpeg_encoder_input encoder_input;

    /* Output image related data */
    struct gpujpeg_decoder_output decoder_output;

    /* Decoder data */
    struct gpujpeg_decoder* decoder;

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

    /* Compression begins here */
    gpujpeg_encoder_input_set_image(&encoder_input, inbuf);
    outbuf = reinterpret_cast<uint8_t*>(malloc(outbuf_size));
    gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input, &outbuf, &outbuf_size);
    free(inbuf);
    /* Compression ends here */
    
    inbuf = reinterpret_cast<uint8_t*>(malloc(outbuf_size));
    memcpy(inbuf, outbuf, outbuf_size);
    gpujpeg_encoder_destroy(encoder);
    outbuf_size = 0;

    decoder = gpujpeg_decoder_create(0);
    if (decoder == NULL)
    {
        perror("Failed to create decoder");
        return 1;
    }
    gpujpeg_set_default_parameters(&param);
    gpujpeg_image_set_default_parameters(&param_image);
    gpujpeg_decoder_init(decoder, &param, &param_image);
    gpujpeg_decoder_set_output_format(decoder, GPUJPEG_RGB, GPUJPEG_444_U8_P012);
    outbuf_size = width * height * bpp;
    /* Encoder setup ends here */

    /* Decompression begins here, parameters and input image
    cannot be changed until it has finished */
    outbuf = reinterpret_cast<uint8_t*>(malloc(outbuf_size));;
    gpujpeg_decoder_output_set_custom(&decoder_output, outbuf);
    gpujpeg_decoder_decode(decoder, inbuf, inbuf_size, &decoder_output);
    free(inbuf);
    inbuf_size = 0;
    /* Decompression ends here*/

    inbuf_size = outbuf_size;
    inbuf = reinterpret_cast<uint8_t*>(malloc(inbuf_size));
    memcpy(inbuf, outbuf, outbuf_size);
    free(outbuf);
    outbuf_size = 0;

    /* Encoder cleanup begins here */
    gpujpeg_decoder_destroy(decoder);
    /* Encoder cleanup ends here */


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

    /* Compression begins here */
    gpujpeg_encoder_input_set_image(&encoder_input, inbuf);
    outbuf = reinterpret_cast<uint8_t*>(malloc(outbuf_size));
    gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input, &outbuf, &outbuf_size);
    free(inbuf);
    /* Compression ends here */

    if (outbuf)
    {
        std::ofstream outFile("raw-compressed-raw-compressed.jpeg", std::ios::binary);
        outFile.write(reinterpret_cast<const char*>(outbuf), outbuf_size);
        outFile.close();
    }


    return 0;
}