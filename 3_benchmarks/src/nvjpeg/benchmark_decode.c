// main.c
// Compile with:
// gcc main.c -o jpeg_decoder -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnvjpeg -lcudart -std=c99
// Adjust CUDA paths if necessary.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <nvjpeg.h>

// Helper macro for checking CUDA API calls
#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err = call;                                                       \
        if (cudaSuccess != err) {                                                     \
            fprintf(stderr, "CUDA Error:\n    File: %s\n    Line: %d\n    Reason: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (0)

// Helper macro for checking nvJPEG API calls
#define NVJPEG_CHECK(call)                                                              \
    do {                                                                                \
        nvjpegStatus_t status = call;                                                   \
        if (NVJPEG_STATUS_SUCCESS != status) {                                          \
            fprintf(stderr, "nvJPEG Error:\n    File: %s\n    Line: %d\n    Reason: %d\n", \
                    __FILE__, __LINE__, status);                                        \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while (0)

// Function to save decoded image data as a PPM file
// This function assumes interleaved RGB data (R G B R G B ...)
void save_to_ppm(const char *filename, unsigned char *image_data, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Unable to open PPM file %s for writing.\n", filename);
        return;
    }

    // Write PPM header
    // P6 means binary color PPM
    // Width Height
    // Max_color_value (255 for 8-bit per channel)
    fprintf(fp, "P6\n%d %d\n255\n", width, height);

    // Write pixel data
    // For RGB, each pixel is 3 bytes
    fwrite(image_data, 1, width * height * 3, fp);

    fclose(fp);
    printf("Successfully saved decoded image to %s\n", filename);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_jpeg_file> <output_file> <iterations>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *input_filename = argv[1];
    const char *output_filename = argv[2];
    const int iterations = atoi(argv[3]);

    // nvJPEG handles and parameters
    nvjpegHandle_t nvjpeg_handle;
    nvjpegJpegState_t jpeg_state;
    nvjpegDecodeParams_t decode_params;

    // CUDA stream
    cudaStream_t stream = NULL; // Using default stream

    // Host and device buffers for JPEG data
    unsigned char *jpeg_data_host = NULL;
    unsigned char *jpeg_data_device = NULL;
    size_t jpeg_data_size = 0;

    // Image dimensions and channels
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int num_components;
    nvjpegChromaSubsampling_t subsampling;

    // Output image buffer (decoded)
    nvjpegImage_t output_image_desc;
    unsigned char *decoded_image_data_device = NULL;
    unsigned char *decoded_image_data_host = NULL;

    printf("Initializing nvJPEG...\n");
    // 1. Create nvJPEG handle
    // Using simple create for backend selection (GPU or CPU - typically GPU)
    NVJPEG_CHECK(nvjpegCreateSimple(&nvjpeg_handle));

    // 2. Create JPEG state (acts as a workspace for decoding)
    NVJPEG_CHECK(nvjpegJpegStateCreate(nvjpeg_handle, &jpeg_state));

    // 3. Create decode parameters
    NVJPEG_CHECK(nvjpegDecodeParamsCreate(nvjpeg_handle, &decode_params));

    // Set output format to interleaved RGB
    // Other options include NVJPEG_OUTPUT_BGR, NVJPEG_OUTPUT_YUV, NVJPEG_OUTPUT_UNCHANGED etc.
    NVJPEG_CHECK(nvjpegDecodeParamsSetOutputFormat(decode_params, NVJPEG_OUTPUT_RGBI));
    // NVJPEG_OUTPUT_BGRI can also be used if BGR is preferred.

    // Optional: Allow ROI decoding (Region of Interest)
    // nvjpegDecodeParamsSetROI(decode_params, offset_x, offset_y, roi_width, roi_height);

    printf("Reading JPEG file: %s\n", input_filename);
    // 4. Read JPEG file from disk
    FILE *fp_jpeg = fopen(input_filename, "rb");
    if (!fp_jpeg) {
        fprintf(stderr, "Error: Unable to open JPEG file %s\n", input_filename);
        NVJPEG_CHECK(nvjpegDecodeParamsDestroy(decode_params));
        NVJPEG_CHECK(nvjpegJpegStateDestroy(jpeg_state));
        NVJPEG_CHECK(nvjpegDestroy(nvjpeg_handle));
        return EXIT_FAILURE;
    }

    fseek(fp_jpeg, 0, SEEK_END);
    jpeg_data_size = ftell(fp_jpeg);
    fseek(fp_jpeg, 0, SEEK_SET);

    jpeg_data_host = (unsigned char *)malloc(jpeg_data_size);
    if (!jpeg_data_host) {
        fprintf(stderr, "Error: Failed to allocate host memory for JPEG data.\n");
        fclose(fp_jpeg);
        NVJPEG_CHECK(nvjpegDecodeParamsDestroy(decode_params));
        NVJPEG_CHECK(nvjpegJpegStateDestroy(jpeg_state));
        NVJPEG_CHECK(nvjpegDestroy(nvjpeg_handle));
        return EXIT_FAILURE;
    }

    if (fread(jpeg_data_host, 1, jpeg_data_size, fp_jpeg) != jpeg_data_size) {
        fprintf(stderr, "Error: Failed to read JPEG data from file.\n");
        free(jpeg_data_host);
        fclose(fp_jpeg);
        NVJPEG_CHECK(nvjpegDecodeParamsDestroy(decode_params));
        NVJPEG_CHECK(nvjpegJpegStateDestroy(jpeg_state));
        NVJPEG_CHECK(nvjpegDestroy(nvjpeg_handle));
        return EXIT_FAILURE;
    }
    fclose(fp_jpeg);

    printf("JPEG file size: %zu bytes\n", jpeg_data_size);

    // 5. Get image information (width, height, channels, subsampling)
    // This can be done before allocating device memory for the JPEG data,
    // but nvjpegGetImageInfo needs the data to be accessible.
    // For simplicity here, we'll do it after reading to host.
    // Alternatively, one could use nvjpegGetFileInfo for some basic info from header.
    NVJPEG_CHECK(nvjpegGetImageInfo(
        nvjpeg_handle,
        jpeg_data_host,
        jpeg_data_size,
        &num_components,
        &subsampling,
        widths,  // Array to store width per component
        heights  // Array to store height per component
    ));

    printf("Image Info:\n  Width: %d\n  Height: %d\n  Components: %d\n", widths[0], heights[0], num_components);
    if (num_components == 1) {
        printf("  Detected Grayscale image. Output will be 3-channel RGB (grayscale replicated).\n");
    } else if (num_components == 3) {
        printf("  Detected Color image (likely YCbCr internally, will be converted to RGB).\n");
    } else {
        fprintf(stderr, "Warning: Unsupported number of components: %d. Expected 1 (grayscale) or 3 (color).\n", num_components);
        // Continue, but decoding might behave unexpectedly or fail.
    }

    // Assuming the first component's dimensions are the image dimensions
    int image_width = widths[0];
    int image_height = heights[0];

    // 6. Allocate device memory for JPEG data and copy from host
    CUDA_CHECK(cudaMallocHost((void **)&jpeg_data_device, jpeg_data_size));
    CUDA_CHECK(cudaMemcpy(jpeg_data_device, jpeg_data_host, jpeg_data_size, cudaMemcpyHostToDevice));

    // 7. Allocate device memory for the decoded output image
    // For RGBI, pitch[0] is width * 3 bytes (R, G, B interleaved)
    // channel[0] points to the start of the RGBI data.
    output_image_desc.pitch[0] = (unsigned int)image_width * 3; // 3 bytes per pixel for RGB
    CUDA_CHECK(cudaMalloc((void **)&(output_image_desc.channel[0]), output_image_desc.pitch[0] * image_height));
    decoded_image_data_device = output_image_desc.channel[0]; // Convenience pointer

    // For other output formats (e.g., planar YUV), multiple channels and pitches would be set:
    // output_image_desc.channel[1] = ...; output_image_desc.pitch[1] = ...;
    // output_image_desc.channel[2] = ...; output_image_desc.pitch[2] = ...;

    printf("Decoding JPEG image...\n");
    // 8. Decode the JPEG image
    // nvjpegDecode can take raw JPEG data from host or device. Here we use device memory.
    clock_t start_time, end_time;
    double total_time;
    start_time = clock();
    for(int i = 0; i < iterations; i++){
    NVJPEG_CHECK(nvjpegDecode(
        nvjpeg_handle,
        jpeg_state,
        jpeg_data_device, // Input JPEG data on device
        jpeg_data_size,
        NVJPEG_OUTPUT_RGBI, // Desired output format
        &output_image_desc, // Description of the output buffer
        stream              // CUDA stream for asynchronous execution (NULL for default/sync)
    ));}
    end_time = clock();
    total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Execution time: %f\n", total_time);

    // If using a non-NULL stream, synchronize it before accessing output data on host
    if (stream) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    printf("Decoding complete. Copying decoded image to host...\n");

    // 9. Allocate host memory for the decoded image and copy from device
    size_t decoded_image_size = output_image_desc.pitch[0] * image_height;
    decoded_image_data_host = (unsigned char *)malloc(decoded_image_size);
    if (!decoded_image_data_host) {
        fprintf(stderr, "Error: Failed to allocate host memory for decoded image.\n");
        // Perform cleanup
        if (decoded_image_data_device) CUDA_CHECK(cudaFree(decoded_image_data_device));
        CUDA_CHECK(cudaFree(jpeg_data_device));
        free(jpeg_data_host);
        NVJPEG_CHECK(nvjpegDecodeParamsDestroy(decode_params));
        NVJPEG_CHECK(nvjpegJpegStateDestroy(jpeg_state));
        NVJPEG_CHECK(nvjpegDestroy(nvjpeg_handle));
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaMemcpy(decoded_image_data_host, decoded_image_data_device, decoded_image_size, cudaMemcpyDeviceToHost));

    // 10. Save the decoded image to a PPM file
    save_to_ppm(output_filename, decoded_image_data_host, image_width, image_height);

    printf("Cleaning up resources...\n");
    // 11. Clean up
    free(decoded_image_data_host);
    CUDA_CHECK(cudaFree(decoded_image_data_device)); // or cudaFree(output_image_desc.channel[0])
    CUDA_CHECK(cudaFree(jpeg_data_device));
    free(jpeg_data_host);

    NVJPEG_CHECK(nvjpegDecodeParamsDestroy(decode_params));
    NVJPEG_CHECK(nvjpegJpegStateDestroy(jpeg_state));
    NVJPEG_CHECK(nvjpegDestroy(nvjpeg_handle));

    printf("Program finished successfully.\n");
    return EXIT_SUCCESS;
}

