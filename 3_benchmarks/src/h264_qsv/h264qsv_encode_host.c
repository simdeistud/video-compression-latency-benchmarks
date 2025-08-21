// qsv_encode_auto.c
// Single-frame h264_qsv encode micro-benchmark (CPU->GPU upload INCLUDED).
// Usage:
//   qsv_encode_auto <raw_frame.yuv> <WxH> <yuv420p|nv12>
//                   <rc_mode: icq|cbr> <rc_value> <iterations> [output -]
//
// Output (stdout): a single line with avg ms-per-encode, like: 2.341
//
// Notes:
// - ICQ: rc_value in [1..51] (lower = better quality). CBR: rc_value = kbps (>0).
// - Fast/low-latency pipeline: low_power=1, async_depth=1, look_ahead=0, bf=0, gop=1.
// - This variant does NOT create a QSV hwdevice; the encoder internally uploads
//   system-memory NV12 frames to GPU each iteration (included in timing).

#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/time.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

static off_t file_size_of(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    return st.st_size;
}

static enum AVPixelFormat input_pixfmt_from_str(const char *s) {
    if (!strcmp(s, "yuv420p")) return AV_PIX_FMT_YUV420P;
    if (!strcmp(s, "nv12"))    return AV_PIX_FMT_NV12;
    return AV_PIX_FMT_NONE;
}

static void planar420_to_nv12(uint8_t *dstY, int dstYls,
                              uint8_t *dstUV, int dstUVls,
                              const uint8_t *srcY, int srcYls,
                              const uint8_t *srcU, int srcUls,
                              const uint8_t *srcV, int srcVls,
                              int w, int h)
{
    // Copy Y plane
    for (int y = 0; y < h; ++y) {
        memcpy(dstY + y*dstYls, srcY + y*srcYls, (size_t)w);
    }
    // Interleave UV into NV12 chroma
    int cw = w/2, ch = h/2;
    for (int y = 0; y < ch; ++y) {
        uint8_t *dst = dstUV + y*dstUVls;
        const uint8_t *u = srcU + y*srcUls;
        const uint8_t *v = srcV + y*srcVls;
        for (int x = 0; x < cw; ++x) {
            dst[2*x+0] = u[x];
            dst[2*x+1] = v[x];
        }
    }
}

typedef enum { RC_ICQ = 0, RC_CBR = 1 } RcMode;

static RcMode rc_mode_from_str(const char *s) {
    if (!strcmp(s, "icq")) return RC_ICQ;
    if (!strcmp(s, "cbr")) return RC_CBR;
#if 0
    // Could add vbr/cqp later if needed
    if (!strcmp(s, "vbr")) return RC_VBR;
    if (!strcmp(s, "cqp")) return RC_CQP;
#endif
    return (RcMode)-1;
}

static int open_qsv_ctx(AVCodecContext **pctx,
                        int w, int h,
                        enum AVPixelFormat sw_pix_fmt,
                        RcMode rc_mode, int rc_val)
{
    const AVCodec *codec = avcodec_find_encoder_by_name("h264_qsv");
    if (!codec) { fprintf(stderr, "h264_qsv not available.\n"); return -1; }

    AVCodecContext *ctx = avcodec_alloc_context3(codec);
    if (!ctx) { fprintf(stderr, "Cannot alloc codec ctx.\n"); return -1; }

    // Base config: all-intra, zero-latency oriented
    ctx->width  = w;
    ctx->height = h;
    // NOTE: for system-memory upload path, codec ctx pix_fmt must be the SW format (NV12).
    ctx->pix_fmt = AV_PIX_FMT_NV12;
    ctx->time_base = (AVRational){1, 30};
    ctx->framerate = (AVRational){30, 1};
    ctx->gop_size = 1;         // all-intra
    ctx->max_b_frames = 0;     // no B-frames

    // QSV private options
    av_opt_set_int(ctx->priv_data, "async_depth", 1, 0); // minimize pipeline depth
    av_opt_set_int(ctx->priv_data, "look_ahead",  0, 0); // disable LA for latency
    av_opt_set_int(ctx->priv_data, "low_power",   1, 0); // VDEnc fast path
    av_opt_set     (ctx->priv_data, "preset", "veryfast", 0);

    // Rate control
    if (rc_mode == RC_ICQ) {
        if (rc_val < 1 || rc_val > 51) {
            fprintf(stderr, "ICQ quality must be 1..51.\n");
            avcodec_free_context(&ctx);
            return -1;
        }
        // ICQ selection: set global_quality, ensure qscale flag is unset, look_ahead=0
        ctx->flags &= ~AV_CODEC_FLAG_QSCALE;
        ctx->global_quality = rc_val; // ICQQuality in [1..51]
        // bitrate fields should be zero in pure ICQ
        ctx->bit_rate = ctx->rc_max_rate = ctx->rc_buffer_size = 0;
    } else { // CBR
        if (rc_val <= 0) {
            fprintf(stderr, "CBR bitrate (kbps) must be > 0.\n");
            avcodec_free_context(&ctx);
            return -1;
        }
        const int64_t bps = (int64_t)rc_val * 1000;
        ctx->bit_rate       = bps;
        ctx->rc_max_rate    = bps;
        ctx->rc_buffer_size = bps; // small VBV for tight CBR
    }

    if (avcodec_open2(ctx, codec, NULL) < 0) {
        fprintf(stderr, "Failed to open h264_qsv.\n");
        avcodec_free_context(&ctx);
        return -1;
    }
    *pctx = ctx;
    return 0;
}

int main(int argc, char **argv) {
    // Args:
    // <raw_frame.yuv> <WxH> <yuv420p|nv12> <rc_mode: icq|cbr> <rc_value> <iterations> [output -]
    if (argc < 7 || argc > 8) {
        fprintf(stderr,
                "Args:\n"
                "  qsv_encode_auto <raw_frame.yuv> <WxH> <yuv420p|nv12>\n"
                "                  <rc_mode: icq|cbr> <rc_value> <iterations> [output -]\n"
                "Notes:\n"
                "  - icq: rc_value = 1..51 (lower = higher quality)\n"
                "  - cbr: rc_value = bitrate in kbps (>0)\n");
        return 1;
    }
    const char *raw_path = argv[1];
    int width=0, height=0;
    if (sscanf(argv[2], "%dx%d", &width, &height) != 2 || width<=0 || height<=0) {
        fprintf(stderr, "Invalid WxH.\n"); return 1;
    }
    enum AVPixelFormat in_pf = input_pixfmt_from_str(argv[3]);
    if (in_pf == AV_PIX_FMT_NONE) { fprintf(stderr, "Use yuv420p or nv12.\n"); return 1; }

    RcMode rc_mode = rc_mode_from_str(argv[4]);
    if ((int)rc_mode == -1) { fprintf(stderr, "rc_mode must be icq or cbr.\n"); return 1; }
    int rc_value = atoi(argv[5]);

    long iterations_l = strtol(argv[6], NULL, 10);
    if (iterations_l <= 0 || iterations_l > 100000000L) {
        fprintf(stderr, "iterations must be in 1..1e8.\n"); return 1;
    }
    const int iterations = (int)iterations_l;
    const char *out_path = (argc == 8) ? argv[7] : NULL;
    const int do_write = (out_path && strcmp(out_path, "-") != 0);

    // Load one frame
    const int nv12_size = width*height + (width*height)/2;
    off_t fsz = file_size_of(raw_path);
    if (fsz < (in_pf == AV_PIX_FMT_YUV420P ? (off_t)(width*height*3/2) : (off_t)nv12_size)) {
        fprintf(stderr, "Input file smaller than one frame.\n"); return 1;
    }
    FILE *in = fopen(raw_path, "rb");
    if (!in) { fprintf(stderr, "Cannot open input.\n"); return 1; }

    // Staging buffers
    uint8_t *nv12buf = (uint8_t*)av_malloc((size_t)nv12_size);
    if (!nv12buf) { fclose(in); fprintf(stderr, "OOM nv12buf.\n"); return 1; }

    if (in_pf == AV_PIX_FMT_NV12) {
        size_t rd = fread(nv12buf, 1, (size_t)nv12_size, in);
        fclose(in);
        if (rd != (size_t)nv12_size) { av_free(nv12buf); fprintf(stderr, "Short read.\n"); return 1; }
    } else {
        // yuv420p -> NV12 repack
        const int ysz = width*height;
        const int csz = ysz/4;
        uint8_t *y = (uint8_t*)av_malloc(ysz);
        uint8_t *u = (uint8_t*)av_malloc(csz);
        uint8_t *v = (uint8_t*)av_malloc(csz);
        if (!y || !u || !v) { fclose(in); av_free(nv12buf); fprintf(stderr, "OOM planar.\n"); return 1; }
        size_t rd1 = fread(y, 1, (size_t)ysz, in);
        size_t rd2 = fread(u, 1, (size_t)csz, in);
        size_t rd3 = fread(v, 1, (size_t)csz, in);
        fclose(in);
        if (rd1 != (size_t)ysz || rd2 != (size_t)csz || rd3 != (size_t)csz) {
            av_free(y); av_free(u); av_free(v); av_free(nv12buf);
            fprintf(stderr, "Short read.\n"); return 1;
        }
        // Fill NV12 layout
        uint8_t *dstY  = nv12buf;
        uint8_t *dstUV = nv12buf + ysz;
        planar420_to_nv12(dstY, width, dstUV, width, y, width, u, width/2, v, width/2, width, height);
        av_free(y); av_free(u); av_free(v);
    }

    // Open encoder
    AVCodecContext *ctx = NULL;
    if (open_qsv_ctx(&ctx, width, height, AV_PIX_FMT_NV12, rc_mode, rc_value) < 0) {
        av_free(nv12buf); return 1;
    }

    AVFrame  *frame = av_frame_alloc();
    AVPacket *pkt   = av_packet_alloc();
    if (!frame || !pkt) {
        if (frame) av_frame_free(&frame);
        if (pkt)   av_packet_free(&pkt);
        avcodec_free_context(&ctx);
        av_free(nv12buf);
        fprintf(stderr, "OOM frame/pkt.\n");
        return 1;
    }
    frame->format = ctx->pix_fmt; // NV12
    frame->width  = ctx->width;
    frame->height = ctx->height;
    if (av_image_fill_arrays(frame->data, frame->linesize, nv12buf,
                             AV_PIX_FMT_NV12, width, height, 1) < 0) {
        fprintf(stderr, "av_image_fill_arrays failed.\n");
        av_packet_free(&pkt); av_frame_free(&frame);
        avcodec_free_context(&ctx); av_free(nv12buf);
        return 1;
    }

    // Timed encode loop (includes host->device upload done by encoder)
    int64_t t0 = av_gettime_relative();
    int64_t total_bytes = 0;
    for (int i = 0; i < iterations; ++i) {
        frame->pts = i;
        if (avcodec_send_frame(ctx, frame) < 0) { fprintf(stderr, "send_frame failed.\n"); goto fail; }
        for (;;) {
            int r = avcodec_receive_packet(ctx, pkt);
            if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
            if (r < 0) { fprintf(stderr, "receive_packet failed.\n"); goto fail; }
            total_bytes += pkt->size;
            av_packet_unref(pkt);
        }
    }
    // Flush
    avcodec_send_frame(ctx, NULL);
    for (;;) {
        int r = avcodec_receive_packet(ctx, pkt);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
        if (r < 0) { fprintf(stderr, "flush receive_packet failed.\n"); goto fail; }
        total_bytes += pkt->size;
        av_packet_unref(pkt);
    }
    int64_t t1 = av_gettime_relative();
    double avg_ms = (iterations > 0) ? ((t1 - t0)/1000.0 / iterations) : 0.0;

    printf("%.3f\n", avg_ms);
    fflush(stdout);

    // Optional: write ONE frame after timing (re-open cleanly)
    if (do_write) {
        FILE *out = fopen(out_path, "wb");
        if (!out) { fprintf(stderr, "Cannot open output.\n"); goto cleanup; }
        // Reopen for a pristine one-frame output
        avcodec_free_context(&ctx);
        if (open_qsv_ctx(&ctx, width, height, AV_PIX_FMT_NV12, rc_mode, rc_value) < 0) {
            fclose(out); goto cleanup;
        }
        frame->pts = 0;
        if (avcodec_send_frame(ctx, frame) < 0) { fprintf(stderr, "send_frame(out) failed.\n"); fclose(out); goto cleanup; }
        for (;;) {
            int r = avcodec_receive_packet(ctx, pkt);
            if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
            if (r < 0) { fprintf(stderr, "receive_packet(out) failed.\n"); fclose(out); goto cleanup; }
            fwrite(pkt->data, 1, (size_t)pkt->size, out);
            av_packet_unref(pkt);
        }
        avcodec_send_frame(ctx, NULL);
        for (;;) {
            int r = avcodec_receive_packet(ctx, pkt);
            if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
            if (r < 0) break;
            fwrite(pkt->data, 1, (size_t)pkt->size, out);
            av_packet_unref(pkt);
        }
        fclose(out);
    }

cleanup:
    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&ctx);
    av_free(nv12buf);
    return 0;

fail:
    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&ctx);
    av_free(nv12buf);
    return 1;
}

