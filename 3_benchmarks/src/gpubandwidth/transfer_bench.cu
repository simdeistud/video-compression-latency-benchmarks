
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

int main(int argc, char **argv) {
    size_t num_bytes = 0;
    int upload = 1;            // 1 = H->D, 0 = D->H
    int iters = 100;           // default iteration count

    // Parse CLI
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bytes") && i + 1 < argc) {
            num_bytes = atol(argv[++i]);
        } else if (!strcmp(argv[i], "--direction") && i + 1 < argc) {
            const char *d = argv[++i];
            if (!strcmp(d, "upload")) upload = 1;
            else if (!strcmp(d, "download")) upload = 0;
            else return 1;
        } else if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
            iters = atoi(argv[++i]);
        }
    }

    if (num_bytes == 0 || iters <= 0)
        return 1;

    // Allocate pinned host memory
    void *h_ptr;
    cudaMallocHost(&h_ptr, num_bytes);

    // Allocate device memory
    void *d_ptr;
    cudaMalloc(&d_ptr, num_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *samples = (float*)malloc(sizeof(float) * iters);

    // Warm-up transfer
    if (upload)
        cudaMemcpy(d_ptr, h_ptr, num_bytes, cudaMemcpyHostToDevice);
    else
        cudaMemcpy(h_ptr, d_ptr, num_bytes, cudaMemcpyDeviceToHost);

    // Timed iterations
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        if (upload)
            cudaMemcpy(d_ptr, h_ptr, num_bytes, cudaMemcpyHostToDevice);
        else
            cudaMemcpy(h_ptr, d_ptr, num_bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&samples[i], start, stop);
    }

    // Compute mean and variance
    double mean = 0.0;
    for (int i = 0; i < iters; i++)
        mean += samples[i];
    mean /= iters;

    double var = 0.0;
    for (int i = 0; i < iters; i++) {
        double diff = samples[i] - mean;
        var += diff * diff;
    }
    var /= iters;    // population variance

    // Output: mean_ms variance_ms
    printf("%.6f\n", mean);

    // Cleanup
    free(samples);
    cudaFreeHost(h_ptr);
    cudaFree(d_ptr);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
