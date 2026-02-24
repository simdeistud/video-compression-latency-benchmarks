#include <nvjpeg.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <fstream>

// ---------- allocators required by nvjpegCreateEx ----------
static int dev_malloc(void** p, size_t s)
{
    return (int)cudaMalloc(p, s);
}
static int dev_free(void* p)
{
    return (int)cudaFree(p);
}
// Use simple malloc/free for nvJPEG internal pinned allocator here;
// we will explicitly allocate our I/O buffers as pinned via cudaHostAlloc.
static int host_malloc(void** p, size_t s, unsigned int flags)
{
    (void)flags;
    *p = malloc(s);
    return (*p) ? 0 : 1;
}
static int host_free(void* p)
{
    free(p);
    return 0;
}

// ---------- error-check helpers ----------
#define CHECK_CUDA(cmd)                                                         \
do {                                                                            \
    cudaError_t e = (cmd);                                                      \
    if (e != cudaSuccess) {                                                     \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(e),     \
                __FILE__, __LINE__);                                            \
        return 1;                                                               \
    }                                                                           \
} while (0)

#define CHECK_NVJPEG(cmd)                                                       \
do {                                                                            \
    nvjpegStatus_t s = (cmd);                                                   \
    if (s != NVJPEG_STATUS_SUCCESS) {                                           \
        fprintf(stderr, "nvJPEG error: %d at %s:%d\n", (int)s,                  \
                __FILE__, __LINE__);                                            \
        return 1;                                                               \
    }                                                                           \
} while (0)

int main(int argc, char** argv)
{
    /* Compressed JPEG input */
    uint8_t* inbuf = NULL;     // pageable
    size_t inbuf_size = 0;

    /* Output image (RGB interleaved) */
    int widths[NVJPEG_MAX_COMPONENT] = { 0 };
    int heights[NVJPEG_MAX_COMPONENT] = { 0 };
    int components = 0;
    nvjpegChromaSubsampling_t subsampling;
    nvjpegOutputFormat_t output_format = NVJPEG_OUTPUT_RGBI;

    /* Last output pointer/size (for optional write) */
    uint8_t* last_outbuf = NULL;
    size_t last_out_size = 0;

    /* nvJPEG handle (shared) */
    nvjpegHandle_t handle = NULL;

    /* CUDA stream + timing events */
    cudaStream_t stream = 0;

    // Open file
    FILE* fp = nullptr;
    errno_t err = fopen_s(&fp, "compressed.jpeg", "rb");
    if (!fp) {
        std::perror("Failed to open file");
        return 1;
    }
    // Determine file size
    if (std::fseek(fp, 0, SEEK_END) != 0) {
        std::perror("fseek failed");
        std::fclose(fp);
        return 1;
    }
    inbuf_size = static_cast<size_t>(std::ftell(fp));
    // Allocate buffer
    inbuf = reinterpret_cast<uint8_t*>(malloc(inbuf_size));
    // Read the file
    std::rewind(fp);
    size_t read_bytes = std::fread(inbuf, 1, inbuf_size, fp);
    std::fclose(fp);
    if (read_bytes != inbuf_size) {
        std::fprintf(stderr, "Error: only read %zu of %zu bytes\n", read_bytes, inbuf_size);
        delete[] inbuf;
        return 1;
    }

    // ---- nvJPEG GPU backend setup ----
    nvjpegDevAllocator_t dev_alloc = { dev_malloc, dev_free };
    nvjpegPinnedAllocator_t pinned_alloc = { host_malloc, host_free };
    CHECK_NVJPEG(nvjpegCreateEx(
        NVJPEG_BACKEND_GPU_HYBRID,   // or NVJPEG_BACKEND_HARDWARE if supported
        &dev_alloc,
        &pinned_alloc,
        NVJPEG_FLAGS_DEFAULT,
        &handle
    ));

    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Probe JPEG info once
    CHECK_NVJPEG(nvjpegGetImageInfo(handle, inbuf, inbuf_size,
        &components, &subsampling,
        widths, heights));

    const int out_w = widths[0];
    const int out_h = heights[0];
    const int out_pitch = out_w * 3;                        // RGB interleaved
    const size_t out_size = (size_t)out_pitch * (size_t)out_h;

    // Pinned input buffers (ping-pong). Copy once from pageable to pinned.
    uint8_t* inbuf_pinned = NULL;
    CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&inbuf_pinned), inbuf_size, cudaHostAllocDefault));
    memcpy(inbuf_pinned, inbuf, inbuf_size);

    // Pinned host output buffers (ping-pong)
    uint8_t* outbuf_pinned = NULL;
    CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&outbuf_pinned), out_size, cudaHostAllocDefault));

    // Device staging buffers (ping-pong) for decode output
    uint8_t* midbuf = NULL;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&midbuf), out_size));

    for (int i = 0; i < 100; ++i)
    {
        nvjpegJpegState_t jpeg_state = NULL;
        CHECK_NVJPEG(nvjpegJpegStateCreate(handle, &jpeg_state));

        nvjpegImage_t outimg; 
        memset(&outimg, 0, sizeof(outimg));
        outimg.channel[0] = midbuf;  // device staging
        outimg.pitch[0] = out_pitch;

        // Decode to device
        CHECK_NVJPEG(nvjpegDecode(handle, jpeg_state,
            inbuf_pinned, inbuf_size,
            output_format, &outimg, stream));

        // Device -> Host (pinned) copy
        CHECK_CUDA(cudaMemcpyAsync(outbuf_pinned, midbuf, out_size,
            cudaMemcpyDeviceToHost, stream));

        // Make sure host buffer is ready
        CHECK_CUDA(cudaStreamSynchronize(stream));

        CHECK_NVJPEG(nvjpegJpegStateDestroy(jpeg_state));
    }

    // Cleanup of global objects
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_NVJPEG(nvjpegDestroy(handle));
    
    if (outbuf_pinned)
    {
        std::ofstream outFile("compressed.jpeg", std::ios::binary);
        outFile.write(reinterpret_cast<const char*>(outbuf_pinned), out_size);
        outFile.close();
    }

    // Free buffers
    free(inbuf);
    cudaFreeHost(inbuf_pinned);
    cudaFreeHost(outbuf_pinned);
    cudaFree(midbuf);

    return 0;
}