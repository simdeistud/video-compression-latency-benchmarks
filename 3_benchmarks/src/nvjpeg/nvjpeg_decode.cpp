#include <nvjpeg.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "../img_utils.h"

// ---------- error-check helpers ----------
#define CHECK_CUDA(cmd)                                                        \
  do {                                                                         \
    cudaError_t e = (cmd);                                                     \
    if (e != cudaSuccess) {                                                    \
      std::cerr << "CUDA error: " << cudaGetErrorString(e)                     \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CHECK_NVJPEG(cmd)                                                      \
  do {                                                                         \
    nvjpegStatus_t s = (cmd);                                                  \
    if (s != NVJPEG_STATUS_SUCCESS) {                                          \
      std::cerr << "nvJPEG error: " << (int)s                                  \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Decode a JPEG bitstream (host memory) to interleaved RGB on host.
// jpeg_data/jpeg_size: compressed JPEG in host memory
// out_rgb: filled with width*height*3 bytes (RGBRGB...)
// out_w/out_h: dimensions of the decoded image
void decode_jpeg_to_rgb_host(const unsigned char* jpeg_data, size_t jpeg_size,
                             std::vector<unsigned char>& out_rgb,
                             int& out_w, int& out_h)
{
  // 1) Create nvJPEG handle and decode state
  nvjpegHandle_t handle = nullptr;
  nvjpegJpegState_t jpeg_state = nullptr;
  CHECK_NVJPEG(nvjpegCreateSimple(&handle));                      // create library handle (host-side)  [1](https://docs.nvidia.com/cuda/nvjpeg/contents.html)
  CHECK_NVJPEG(nvjpegJpegStateCreate(handle, &jpeg_state));       // create decode state               [1](https://docs.nvidia.com/cuda/nvjpeg/contents.html)

  // 2) Get image info from bitstream (width/height per component)
  int nComponents = 0;
  nvjpegChromaSubsampling_t subsampling;
  int widths[NVJPEG_MAX_COMPONENT] = {0};
  int heights[NVJPEG_MAX_COMPONENT] = {0};
  CHECK_NVJPEG(nvjpegGetImageInfo(handle,
                                  jpeg_data,
                                  jpeg_size,
                                  &nComponents,
                                  &subsampling,
                                  widths,
                                  heights));                      // query sizes from JPEG header      [1](https://docs.nvidia.com/cuda/nvjpeg/contents.html)
  out_w = widths[0];
  out_h = heights[0];

  // 3) Allocate device output buffer for interleaved RGB
  // For NVJPEG_OUTPUT_RGBI, decoded pixels are written to channel[0] only,
  // with pitch[0] = width*3 bytes.                               [1](https://docs.nvidia.com/cuda/nvjpeg/contents.html)
  const size_t pitch = static_cast<size_t>(out_w) * 3;
  const size_t dev_bytes = pitch * static_cast<size_t>(out_h);

  unsigned char* d_rgb = nullptr;
  CHECK_CUDA(cudaMalloc(&d_rgb, dev_bytes));

  nvjpegImage_t out_img{};
  out_img.channel[0] = d_rgb;                       // interleaved buffer in channel[0]
  out_img.pitch[0]   = static_cast<unsigned int>(pitch);

  // 4) Decode to interleaved RGB on device
  CHECK_NVJPEG(nvjpegDecode(handle,
                            jpeg_state,
                            jpeg_data,
                            jpeg_size,
                            NVJPEG_OUTPUT_RGBI,      // interleaved RGB output
                            &out_img,
                            /*stream*/ nullptr));     // synchronous stream is fine here [1](https://docs.nvidia.com/cuda/nvjpeg/contents.html)
  CHECK_CUDA(cudaDeviceSynchronize());

  // 5) Copy result back to host memory
  out_rgb.resize(dev_bytes);
  CHECK_CUDA(cudaMemcpy(out_rgb.data(),
                        d_rgb,
                        dev_bytes,
                        cudaMemcpyDeviceToHost));

  // 6) Cleanup
  CHECK_CUDA(cudaFree(d_rgb));
  CHECK_NVJPEG(nvjpegJpegStateDestroy(jpeg_state));
  CHECK_NVJPEG(nvjpegDestroy(handle));
}

// Simple demo: read a JPEG file into host memory and decode it using the function above.
// Writes raw RGB24 to <out.raw>.
int main(int argc, char** argv)
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <in.jpg> <out.raw>\n"
              << "Decodes <in.jpg> to raw interleaved RGB24 (width*height*3 bytes) in <out.raw>.\n";
    return EXIT_FAILURE;
  }

  const std::string in_jpeg = argv[1];
  const std::string out_raw = argv[2];

  // Read JPEG file to host buffer (to simulate 'already in host memory')
  std::ifstream f(in_jpeg, std::ios::binary);
  if (!f) {
    std::cerr << "Failed to open: " << in_jpeg << std::endl;
    return EXIT_FAILURE;
  }
  f.seekg(0, std::ios::end);
  size_t jpeg_size = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);

  std::vector<unsigned char> jpeg(jpeg_size);
  f.read(reinterpret_cast<char*>(jpeg.data()), jpeg_size);

  // Decode
  std::vector<unsigned char> rgb;
  int w = 0, h = 0;
  decode_jpeg_to_rgb_host(jpeg.data(), jpeg.size(), rgb, w, h);

  // Write raw RGB
  std::ofstream out(out_raw, std::ios::binary);
  out.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
  std::cout << "Decoded " << in_j