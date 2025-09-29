// vp8_decode_benchmark.c
// Benchmark decoding a single VP8 frame (from IVF) using libvpx.
// Fast-path settings: no postproc, no error concealment, configurable threads.
// Usage: ./vp8_decode_benchmark <frame.ivf> <iterations> [threads]
//
// Build: gcc -O3 -march=native vp8_decode_benchmark.c -o vp8_decode_benchmark -lvpx -lm
// (On very old toolchains you might need -lrt for clock_gettime.)

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <errno.h>
#include <time.h>

#include <vpx/vpx_decoder.h>
#include <vpx/vp8dx.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int read_exact(FILE *f, void *buf, size_t n) {
    return fread(buf, 1, n, f) == n ? 0 : -1;
}

static uint32_t le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t le16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s <frame.ivf> <iterations> [threads]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *path = argv[1];
    char *endp = NULL;
    long iterations = strtol(argv[2], &endp, 10);
    if (!argv[2][0] || *endp || iterations <= 0) {
        die("Invalid iterations (must be positive integer).");
    }
    int threads = 1;
    if (argc == 4) {
        long t = strtol(argv[3], &endp, 10);
        if (!argv[3][0] || *endp || t <= 0) die("Invalid threads (must be positive integer).");
        threads = (int)t;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open '%s': %s\n", path, strerror(errno));
        return EXIT_FAILURE;
    }

    // Parse IVF header (32 bytes)
    uint8_t ivf_hdr[32];
    if (read_exact(f, ivf_hdr, 32) < 0) die("Failed to read IVF header.");
    if (memcmp(ivf_hdr, "DKIF", 4) != 0) die("Not an IVF file (missing DKIF).");

    const uint16_t version = le16(ivf_hdr + 4);
    const uint16_t header_len = le16(ivf_hdr + 6);
    const uint32_t fourcc = le32(ivf_hdr + 8);
    const uint16_t width  = le16(ivf_hdr + 12);
    const uint16_t height = le16(ivf_hdr + 14);
    // Skip rest of header if larger than 32 (some files pad)
    if (header_len > 32) {
        if (fseek(f, header_len - 32, SEEK_CUR) != 0) die("Failed to seek to first frame.");
    }

    if (fourcc != 0x30385056u) { // 'VP80' little-endian
        fprintf(stderr, "FourCC not VP80 (got 0x%08x). This tool expects VP8.\n", fourcc);
        return EXIT_FAILURE;
    }

    // Read first frame header (12 bytes) => size(4) + pts(8)
    uint8_t frm_hdr[12];
    if (read_exact(f, frm_hdr, 12) < 0) die("Failed to read IVF frame header.");
    uint32_t frame_size = le32(frm_hdr + 0);
    // uint64_t timestamp = (uint64_t)le32(frm_hdr+4) | ((uint64_t)le32(frm_hdr+8) << 32); // unused

    if (frame_size == 0) die("Frame size is zero.");
    uint8_t *frame = (uint8_t *)malloc(frame_size);
    if (!frame) die("Out of memory for frame.");
    if (read_exact(f, frame, frame_size) < 0) die("Failed to read frame payload.");
    fclose(f);

    // Initialize decoder with fast settings:
    // - No post-processing flag set (default).
    // - No error concealment flag set (default).
    // - Configurable thread count for VP8 row-based threading.
    vpx_codec_ctx_t codec;
    vpx_codec_dec_cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.threads = threads; // exploit available cores

    const vpx_codec_iface_t *iface = vpx_codec_vp8_dx();
    const unsigned long init_flags = 0; // No VPX_CODEC_USE_POSTPROC, no VPX_CODEC_USE_ERROR_CONCEALMENT
    if (vpx_codec_dec_init(&codec, iface, &cfg, init_flags) != VPX_CODEC_OK) {
        fprintf(stderr, "Decoder init failed: %s\n", vpx_codec_error(&codec));
        free(frame);
        return EXIT_FAILURE;
    }

    // Warm-up decode (once), to populate internal state and avoid first-iteration skew.
    if (vpx_codec_decode(&codec, frame, frame_size, NULL, 0) != VPX_CODEC_OK) {
        fprintf(stderr, "Warm-up decode error: %s\n", vpx_codec_error(&codec));
        vpx_codec_destroy(&codec);
        free(frame);
        return EXIT_FAILURE;
    }
    // Drain any output frames (we won't touch pixels, just drain to keep pipeline consistent)
    {
        vpx_codec_iter_t it = NULL;
        while (vpx_codec_get_frame(&codec, &it)) { /* no-op */ }
    }

    // Benchmark loop
    double t0 = now_ms();
    for (long i = 0; i < iterations; ++i) {
        if (vpx_codec_decode(&codec, frame, frame_size, NULL, 0) != VPX_CODEC_OK) {
            fprintf(stderr, "Decode error at iter %ld: %s\n", i, vpx_codec_error(&codec));
            vpx_codec_destroy(&codec);
            free(frame);
            return EXIT_FAILURE;
        }
        // Drain decoded frame(s). For a single input frame, at most one image is emitted.
        vpx_codec_iter_t it = NULL;
        while (vpx_codec_get_frame(&codec, &it)) {
            // Intentionally ignore the image to avoid cache pollution from touching pixels.
            // const vpx_image_t *img = ...
        }
    }
    double t1 = now_ms();

    double total_ms = t1 - t0;
    double avg_ms = total_ms / (double)iterations;
    double fps = 1000.0 / avg_ms;

    fprintf(stdout,
            "File: %s | VP8 %ux%u | IVF v%u hdr=%u | threads=%d\n"
            "Decoded %ld iterations in %.3f ms => %.4f ms/frame (%.2f fps)\n",
            path, (unsigned)width, (unsigned)height, (unsigned)version, (unsigned)header_len,
            threads, iterations, total_ms, avg_ms, fps);

    vpx_codec_destroy(&codec);
    free(frame);
    return EXIT_SUCCESS;
}
