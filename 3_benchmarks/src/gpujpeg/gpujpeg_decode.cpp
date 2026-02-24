#include "libgpujpeg/gpujpeg.h"
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <fstream>

// ---- minimal RGB24 viewer plumbing (add above main) ----
#include <stdint.h>

struct RgbViewCtx {
    const uint8_t* bits;   // pointer to RGB (possibly padded) data
    int w, h;
    BITMAPINFO bi;         // prepared BITMAPINFO
};

LRESULT CALLBACK RgbWndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(h, &ps);
        auto* ctx = reinterpret_cast<RgbViewCtx*>(GetWindowLongPtr(h, GWLP_USERDATA));
        if (ctx && ctx->bits) {
            StretchDIBits(dc,
                0, 0, ctx->w, ctx->h,
                0, 0, ctx->w, ctx->h,
                ctx->bits,
                &ctx->bi,
                DIB_RGB_COLORS,
                SRCCOPY);
        }
        EndPaint(h, &ps);
        return 0;
    }
    case WM_KEYDOWN:
        if (w == VK_ESCAPE) { PostQuitMessage(0); return 0; }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(h, m, w, l);
}
// ---- end plumbing ----

int main(int argc, char* argv[])
{
    /* I/O */
    uint8_t* inbuf = NULL;
    size_t inbuf_size = 0;
    uint8_t* outbuf = NULL;
    size_t outbuf_size = 0;

    /* Output image related data */
    struct gpujpeg_decoder_output decoder_output;

    /* Decoder data */
    struct gpujpeg_parameters param;
    struct gpujpeg_image_parameters param_image;
    struct gpujpeg_decoder* decoder;


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


    /* Decoder setup starts here */
    /* We use the default CUDA device, if you want to test on different ones, change the first argument of gpujpeg_init_device */
    if (gpujpeg_init_device(0, 0))
    {
        perror("Failed to initialize GPU device");
        return 1;
    }

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
    const int width = 1280, height = 720, bpp = 3;
    outbuf_size = width * height * bpp;
    /* Encoder setup ends here */

    /* Decompression begins here, parameters and input image
    cannot be changed until it has finished */
    for (int i = 0; i < 100; i++)
    {
        uint8_t* curr_img = reinterpret_cast<uint8_t*>(malloc(inbuf_size));
        memcpy(curr_img, inbuf, inbuf_size);
        uint8_t* curr_out = reinterpret_cast<uint8_t*>(malloc(outbuf_size));;
        gpujpeg_decoder_output_set_custom(&decoder_output, curr_out);
        gpujpeg_decoder_decode(decoder, curr_img, inbuf_size, &decoder_output);
        free(curr_img);
        outbuf = reinterpret_cast<uint8_t*>(malloc(outbuf_size));
        memcpy(outbuf, curr_out, outbuf_size);
        free(curr_out);
    }
    /* Decompression ends here, a new image can be loaded in
    the input buffer and parameters can be changed
    (if not they will remain the same) */

    /* Encoder cleanup begins here */
    gpujpeg_decoder_destroy(decoder);
    /* Encoder cleanup ends here */

    if (outbuf)
    {
        std::ofstream outFile("uncompressed.rgb", std::ios::binary);
        outFile.write(reinterpret_cast<const char*>(outbuf), outbuf_size);
        outFile.close();
    }

    // ---- RGB24 minimal viewer (insert right before return 0) ----

    if (outbuf && outbuf_size == static_cast<size_t>(width * height * 3)) {
        const int src_stride = width * 3;
        const int dst_stride = (src_stride + 3) & ~3; // 4‑byte DIB alignment

        std::unique_ptr<uint8_t[]> bgr_aligned; // alive during message loop
        const uint8_t* dib_bits = nullptr;

        // Make a destination buffer that is both BGR and DWORD-aligned
        bgr_aligned.reset(new uint8_t[static_cast<size_t>(dst_stride) * height]);

        for (int y = 0; y < height; ++y) {
            const uint8_t* s = outbuf + static_cast<size_t>(y) * src_stride;
            uint8_t* d = bgr_aligned.get() + static_cast<size_t>(y) * dst_stride;

            // RGB -> BGR in-place to dest
            for (int x = 0; x < width; ++x) {
                const uint8_t r = s[3 * x + 0];
                const uint8_t g = s[3 * x + 1];
                const uint8_t b = s[3 * x + 2];
                d[3 * x + 0] = b;   // B
                d[3 * x + 1] = g;   // G
                d[3 * x + 2] = r;   // R
            }
            // zero pad to DWORD
            std::memset(d + src_stride, 0, dst_stride - src_stride);
        }
        dib_bits = bgr_aligned.get();

        // BITMAPINFO (top-down)
        RgbViewCtx ctx{};
        ctx.w = width; ctx.h = height; ctx.bits = dib_bits;
        ZeroMemory(&ctx.bi, sizeof(ctx.bi));
        ctx.bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        ctx.bi.bmiHeader.biWidth = width;
        ctx.bi.bmiHeader.biHeight = -height;      // top-down
        ctx.bi.bmiHeader.biPlanes = 1;
        ctx.bi.bmiHeader.biBitCount = 24;
        ctx.bi.bmiHeader.biCompression = BI_RGB;
        ctx.bi.bmiHeader.biSizeImage = static_cast<DWORD>(dst_stride * height);

        ctx.bits = dib_bits;

        // Register a tiny window class
        WNDCLASS wc{};
        wc.lpfnWndProc = RgbWndProc;
        wc.hInstance = GetModuleHandle(nullptr);
        wc.lpszClassName = TEXT("RGB24ViewerClass");
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        RegisterClass(&wc);

        // Create window sized to match client area to the image
        RECT wr{ 0, 0, width, height };
        AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);
        HWND hwnd = CreateWindow(
            wc.lpszClassName, TEXT("RGB24 (ESC to close)"),
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT, CW_USEDEFAULT,
            wr.right - wr.left, wr.bottom - wr.top,
            nullptr, nullptr, wc.hInstance, nullptr);

        // Pass context to the window
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&ctx));
        ShowWindow(hwnd, SW_SHOW);
        UpdateWindow(hwnd);

        // Standard message loop
        MSG msg;
        while (GetMessage(&msg, nullptr, 0, 0) > 0) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }
    // ---- end viewer ----

    return 0;



    return 0;
}