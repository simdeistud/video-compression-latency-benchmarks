// vp8_decode_benchmark_ffmpeg.c
// Benchmark decoding a single VP8 frame (from IVF) using FFmpeg libavcodec.
// Fast-path settings: threads, FAST flag, skip loop filter.
// Usage: ./vp8_decode_benchmark_ffmpeg <frame.ivf> <iterations> [threads]
//
// Build (pkg-config preferred):
//   cc -O3 -march=native vp8_decode_benchmark_ffmpeg.c -o vp8_decode_benchmark_ffmpeg \
//      $(pkg-config --cflags --libs libavcodec libavutil)
// or fallback:
//   cc -O3 -march=native vp8_decode_benchmark_ffmpeg.c -o vp8_decode_benchmark_ffmpeg \
//      -lavcodec -lavutil -lpthread -lm
//
// Note: Requires an IVF file whose first frame is a VP8 keyframe.

#define POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>

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
    if (!argv[2][0] || *endp || iterations <= 0) die("Invalid iterations (must be > 0).");
    int threads = 1;
    if (argc == 4) {
        long t = strtol(argv[3], &endp, 10);
        if (!argv[3][0] || *endp || t <= 0) die("Invalid threads (must be > 0).");
        threads = (int)t;
    }

    // Silence FFmpeg logs (avoid noise & overhead)
    av_log_set_level(AV_LOG_QUIET);

    // Read IVF first frame
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open '%s': %s\n", path, strerror(errno));
        return EXIT_FAILURE;
    }

    uint8_t ivf_hdr[32];
    if (read_exact(f, ivf_hdr, 32) < 0) die("Failed to read IVF header.");
    if (memcmp(ivf_hdr, "DKIF", 4) != 0) die("Not an IVF file (missing DKIF).");

    const uint16_t version = le16(ivf_hdr + 4);
    const uint16_t header_len = le16(ivf_hdr + 6);
    const uint32_t fourcc = le32(ivf_hdr + 8);
    const uint16_t width  = le16(ivf_hdr + 12);
    const uint16_t height = le16(ivf_hdr + 14);
    if (fourcc != 0x30385056u) { // 'VP80'
        fprintf(stderr, "FourCC not VP80 (0x%08x). Expected VP8.\n", fourcc);
        return EXIT_FAILURE;
    }
    if (header_len > 32) {
        if (fseek(f, header_len - 32, SEEK_CUR) != 0) die("Failed to seek to first frame.");
    }

    uint8_t frm_hdr[12];
    if (read_exact(f, frm_hdr, 12) < 0) die("Failed to read IVF frame header.");
    uint32_t frame_size = le32(frm_hdr + 0);
    if (frame_size == 0) die("Frame size is zero.");

    uint8_t *frame = (uint8_t *)malloc(frame_size);
    if (!frame) die("Out of memory for frame.");
    if (read_exact(f, frame, frame_size) < 0) die("Failed to read frame payload.");
    fclose(f);

    // FFmpeg decoder setup
    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_VP8);
    if (!codec) die("VP8 decoder not found in libavcodec.");

    AVCodecContext *ctx = avcodec_alloc_context3(codec);
    if (!ctx) die("Failed to alloc codec context.");

    // Fastest settings:
    ctx->thread_count = threads;
    ctx->thread_type  = FF_THREAD_SLICE | FF_THREAD_FRAME; // use whatever the decoder supports
    ctx->flags2      |= AV_CODEC_FLAG2_FAST;               // non-strict speedups where available
    ctx->skip_loop_filter = AVDISCARD_ALL;                 // skip in-loop deblocking for speed
    ctx->err_recognition = 0;                              // no extra error checking

    // Width/height are optional (decoder can infer), but setting helps some paths
    ctx->width  = width ? width : 0;
    ctx->height = height ? height : 0;

    if (avcodec_open2(ctx, codec, NULL) < 0) die("Failed to open VP8 decoder.");

    // Reusable frame
    AVFrame *avf = av_frame_alloc();
    if (!avf) die("Failed to alloc AVFrame.");

    // Prepare immutable AVPacket pointing to our buffer (no copy)
    AVPacket pkt_template;
    memset(&pkt_template, 0, sizeof(pkt_template));
    pkt_template.data = frame;
    pkt_template.size = (int)frame_size;

    // Warm-up (not timed)
    {
        AVPacket pkt = pkt_template; // shallow copy
        if (avcodec_send_packet(ctx, &pkt) < 0) die("Warm-up send_packet failed.");
        int rcv;
        do {
            rcv = avcodec_receive_frame(ctx, avf);
        } while (rcv == 0);
        avcodec_flush_buffers(ctx);
    }

    // Benchmark loop
    double t0 = now_ms();
    for (long i = 0; i < iterations; ++i) {
        AVPacket pkt = pkt_template; // shallow copy, no ownership transfer
        if (avcodec_send_packet(ctx, &pkt) < 0) {
            fprintf(stderr, "send_packet failed at iter %ld\n", i);
            return EXIT_FAILURE;
        }
        // Drain frame(s); for a single input frame there's at most one
        int rcv;
        do {
            rcv = avcodec_receive_frame(ctx, avf);
            if (rcv == 0) {
                // Do nothing with pixels to avoid cache pollution
            } else if (rcv == AVERROR(EAGAIN) || rcv == AVERROR_EOF) {
                break;
            } else {
                fprintf(stderr, "receive_frame error at iter %ld: %d\n", i, rcv);
                return EXIT_FAILURE;
            }
        } while (rcv == 0);

        // Keep decoder state consistent for the next identical single-frame decode
        avcodec_flush_buffers(ctx);
    }
    double t1 = now_ms();

    double total_ms = t1 - t0;
    double avg_ms   = total_ms / (double)iterations;
    double fps      = 1000.0 / avg_ms;

    fprintf(stdout,
            "File: %s | VP8 %ux%u | IVF v%u hdr=%u | threads=%d\n"
            "Decoded %ld iterations in %.3f ms -> %.4f ms/frame (%.2f fps)\n",
            path, (unsigned)ctx->width, (unsigned)ctx->height,
            (unsigned)version, (unsigned)header_len, threads,
            iterations, total_ms, avg_ms, fps);

    av_frame_free(&avf);
    avcodec_free_context(&ctx);
    free(frame);
    return EXIT_SUCCESS;
}
