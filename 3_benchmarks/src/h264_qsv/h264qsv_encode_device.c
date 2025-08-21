// qsv_encode_preloaded.c
// Single-frame h264_qsv encode micro-benchmark (GPU-only encode time).
// Upload to GPU happens ONCE before timing; the timed loop reuses the same
// QSV surface each iteration.
//
// Usage:
//   qsv_encode_preloaded <raw_frame.yuv> <WxH> <yuv420p|nv12>
//                        <rc_mode: icq|cbr> <rc_value> <iterations> [output -]
//
// Output (stdout): avg ms-per-encode
//
// Requirements:
// - Working QSV device (Intel iGPU / driver).
// - Frames allocated as AV_PIX_FMT_QSV via AVHWFramesContext; we upload a single
//   NV12 SW frame to this surface BEFORE timing and then re-use it.
//
// Fast path: low_power=1, async_depth=1, look_ahead=0, bf=0, gop=1.

#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_qsv.h>
#include <libavutil/imgutils.h>
#include <libavutil/time.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

static off_t file_size_of(const char *path) {
    struct stat st; if (stat(path, &st) != 0) return -1; return st.st_size;
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
    for (int y = 0; y < h; ++y) memcpy(dstY + y*dstYls, srcY + y*srcYls, (size_t)w);
    int cw = w/2, ch = h/2;
    for (int y = 0; y < ch; ++y) {
        uint8_t *dst = dstUV + y*dstUVls;
        const uint8_t *u = srcU + y*srcUls;
        const uint8_t *v = srcV + y*srcVls;
        for (int x = 0; x < cw; ++x) { dst[2*x+0] = u[x]; dst[2*x+1] = v[x]; }
    }
}

typedef enum { RC_ICQ = 0, RC_CBR = 1 } RcMode;
static RcMode rc_mode_from_str(const char *s) {
    if (!strcmp(s, "icq")) return RC_ICQ;
    if (!strcmp(s, "cbr")) return RC_CBR;
    return (RcMode)-1;
}

static int open_qsv_ctx_gpu(AVCodecContext **pctx, AVBufferRef *qsv_dev,
                            AVBufferRef *frames_ref,
                            int w, int h, RcMode rc_mode, int rc_val)
{
    const AVCodec *codec = avcodec_find_encoder_by_name("h264_qsv");
    if (!codec) { fprintf(stderr, "h264_qsv not available.\n"); return -1; }
    AVCodecContext *ctx = avcodec_alloc_context3(codec);
    if (!ctx) { fprintf(stderr, "Cannot alloc codec ctx.\n"); return -1; }

    ctx->width  = w;
    ctx->height = h;
    // For GPU path, encoder expects AV_PIX_FMT_QSV frames
    ctx->pix_fmt = AV_PIX_FMT_QSV;
    ctx->time_base = (AVRational){1, 30};
    ctx->framerate = (AVRational){30, 1};
    ctx->gop_size = 1;
    ctx->max_b_frames = 0;

    // Attach device/frames to context so encoder doesn't create its own pool
    ctx->hw_device_ctx = av_buffer_ref(qsv_dev);
    ctx->hw_frames_ctx = av_buffer_ref(frames_ref);

    av_opt_set_int(ctx->priv_data, "async_depth", 1, 0);
    av_opt_set_int(ctx->priv_data, "look_ahead",  0, 0);
    av_opt_set_int(ctx->priv_data, "low_power",   1, 0);
    av_opt_set     (ctx->priv_data, "preset", "veryfast", 0);

    if (rc_mode == RC_ICQ) {
        if (rc_val < 1 || rc_val > 51) {
            fprintf(stderr, "ICQ quality must be 1..51.\n");
            avcodec_free_context(&ctx); return -1;
        }
        ctx->flags &= ~AV_CODEC_FLAG_QSCALE;
        ctx->global_quality = rc_val;
        ctx->bit_rate = ctx->rc_max_rate = ctx->rc_buffer_size = 0;
    } else {
        if (rc_val <= 0) { fprintf(stderr, "CBR bitrate (kbps) must be >0.\n"); avcodec_free_context(&ctx); return -1; }
        const int64_t bps = (int64_t)rc_val * 1000;
        ctx->bit_rate       = bps;
        ctx->rc_max_rate    = bps;
        ctx->rc_buffer_size = bps;
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
    // <raw_frame.yuv> <WxH> <yuv420p|nv12> <rc_mode: icq|cbr> <rc_value> <iterations> [output -]
    if (argc < 7 || argc > 8) {
        fprintf(stderr,
                "Args:\n"
                "  qsv_encode_preloaded <raw_frame.yuv> <WxH> <yuv420p|nv12>\n"
                "                       <rc_mode: icq|cbr> <rc_value> <iterations> [output -]\n");
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

    const int nv12_size = width*height + (width*height)/2;
    off_t fsz = file_size_of(raw_path);
    if (fsz < (in_pf == AV_PIX_FMT_YUV420P ? (off_t)(width*height*3/2) : (off_t)nv12_size)) {
        fprintf(stderr, "Input file smaller than one frame.\n"); return 1;
    }

    // Read and pack to NV12 in system memory
    FILE *in = fopen(raw_path, "rb");
    if (!in) { fprintf(stderr, "Cannot open input.\n"); return 1; }
    uint8_t *nv12buf = (uint8_t*)av_malloc((size_t)nv12_size);
    if (!nv12buf) { fclose(in); fprintf(stderr, "OOM nv12buf.\n"); return 1; }
    if (in_pf == AV_PIX_FMT_NV12) {
        size_t rd = fread(nv12buf, 1, (size_t)nv12_size, in);
        fclose(in);
        if (rd != (size_t)nv12_size) { av_free(nv12buf); fprintf(stderr, "Short read.\n"); return 1; }
    } else {
        const int ysz = width*height, csz = ysz/4;
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
        uint8_t *dstY  = nv12buf;
        uint8_t *dstUV = nv12buf + ysz;
        planar420_to_nv12(dstY, width, dstUV, width, y, width, u, width/2, v, width/2, width, height);
        av_free(y); av_free(u); av_free(v);
    }

    // Create QSV device
    AVBufferRef *qsv_dev = NULL;
    // device string: NULL/"auto" lets FFmpeg pick the proper Intel adapter
    if (av_hwdevice_ctx_create(&qsv_dev, AV_HWDEVICE_TYPE_QSV, NULL, NULL, 0) < 0) {
        fprintf(stderr, "Failed to create QSV device.\n");
        av_free(nv12buf); return 1;
    }

    // Create frames context (QSV surfaces) with NV12 as sw_format
    AVBufferRef *frames_ref = av_hwframe_ctx_alloc(qsv_dev);
    if (!frames_ref) { fprintf(stderr, "av_hwframe_ctx_alloc failed.\n"); av_free(nv12buf); av_buffer_unref(&qsv_dev); return 1; }
    AVHWFramesContext *frames_ctx = (AVHWFramesContext*)frames_ref->data;
    frames_ctx->format    = AV_PIX_FMT_QSV;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;
    frames_ctx->width     = width;
    frames_ctx->height    = height;
    frames_ctx->initial_pool_size = 2; // enough with async_depth=1
    if (av_hwframe_ctx_init(frames_ref) < 0) {
        fprintf(stderr, "av_hwframe_ctx_init failed.\n");
        av_buffer_unref(&frames_ref); av_buffer_unref(&qsv_dev); av_free(nv12buf); return 1;
    }

    // Allocate one QSV surface and upload once (outside timing)
    AVFrame *hwframe = av_frame_alloc();
    AVFrame *swframe = av_frame_alloc();
    AVPacket *pkt    = av_packet_alloc();
    if (!hwframe || !swframe || !pkt) {
        fprintf(stderr, "OOM frames/pkt.\n");
        if (hwframe) av_frame_free(&hwframe);
        if (swframe) av_frame_free(&swframe);
        if (pkt)     av_packet_free(&pkt);
        av_buffer_unref(&frames_ref); av_buffer_unref(&qsv_dev); av_free(nv12buf);
        return 1;
    }
    if (av_hwframe_get_buffer(frames_ref, hwframe, 0) < 0) {
        fprintf(stderr, "av_hwframe_get_buffer failed.\n");
        av_frame_free(&hwframe); av_frame_free(&swframe); av_packet_free(&pkt);
        av_buffer_unref(&frames_ref); av_buffer_unref(&qsv_dev); av_free(nv12buf); return 1;
    }
    // Prepare SW NV12 frame that wraps nv12buf (for upload)
    swframe->format = AV_PIX_FMT_NV12;
    swframe->width  = width;
    swframe->height = height;
    if (av_image_fill_arrays(swframe->data, swframe->linesize, nv12buf,
                             AV_PIX_FMT_NV12, width, height, 1) < 0) {
        fprintf(stderr, "av_image_fill_arrays failed.\n");
        av_frame_free(&hwframe); av_frame_free(&swframe); av_packet_free(&pkt);
        av_buffer_unref(&frames_ref); av_buffer_unref(&qsv_dev); av_free(nv12buf); return 1;
    }
    // Upload once
    if (av_hwframe_transfer_data(hwframe, swframe, 0) < 0) {
        fprintf(stderr, "av_hwframe_transfer_data (upload) failed.\n");
        av_frame_free(&hwframe); av_frame_free(&swframe); av_packet_free(&pkt);
        av_buffer_unref(&frames_ref); av_buffer_unref(&qsv_dev); av_free(nv12buf); return 1;
    }

    // Open encoder bound to this hw frames context
    AVCodecContext *ctx = NULL;
    if (open_qsv_ctx_gpu(&ctx, qsv_dev, frames_ref, width, height, rc_mode, rc_value) < 0) {
        av_frame_free(&hwframe); av_frame_free(&swframe); av_packet_free(&pkt);
        av_buffer_unref(&frames_ref); av_buffer_unref(&qsv_dev); av_free(nv12buf); return 1;
    }

    // Timed loop: re-use same QSV surface (GPU->GPU only)
    int64_t t0 = av_gettime_relative();
    int64_t total_bytes = 0;
    for (int i = 0; i < iterations; ++i) {
        hwframe->pts = i;
        // Send a ref to avoid the encoder modifying our single surface header
        AVFrame *to_send = av_frame_alloc();
        if (!to_send) { fprintf(stderr, "OOM.\n"); goto fail; }
        if (av_frame_ref(to_send, hwframe) < 0) { fprintf(stderr, "av_frame_ref failed.\n"); av_frame_free(&to_send); goto fail; }
        to_send->pts = hwframe->pts;

        if (avcodec_send_frame(ctx, to_send) < 0) { fprintf(stderr, "send_frame failed.\n"); av_frame_free(&to_send); goto fail; }
        av_frame_free(&to_send);

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

    // Optional: write one frame after timing
    if (do_write) {
        FILE *out = fopen(out_path, "wb");
        if (!out) { fprintf(stderr, "Cannot open output.\n"); goto cleanup; }
        // Clean open
        avcodec_free_context(&ctx);
        if (open_qsv_ctx_gpu(&ctx, qsv_dev, frames_ref, width, height, rc_mode, rc_value) < 0) {
            fclose(out); goto cleanup;
        }
        hwframe->pts = 0;
        if (avcodec_send_frame(ctx, hwframe) < 0) { fprintf(stderr, "send_frame(out) failed.\n"); fclose(out); goto cleanup; }
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
    av_frame_free(&swframe);
    av_frame_free(&hwframe);
    avcodec_free_context(&ctx);
    av_buffer_unref(&frames_ref);
    av_buffer_unref(&qsv_dev);
    av_free(nv12buf);
    return 0;

fail:
    av_packet_free(&pkt);
    av_frame_free(&swframe);
    av_frame_free(&hwframe);
    avcodec_free_context(&ctx);
    av_buffer_unref(&frames_ref);
    av_buffer_unref(&qsv_dev);
    av_free(nv12buf);
    return 1;
}

