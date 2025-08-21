// av1nvenc_encode_device.c
// Single-frame NVENC AV1 micro-benchmark (GPU-preloaded; measures GPU->GPU encode).
// Usage:
//   av1nvenc_pre <raw.yuv> <WxH> <yuv420p|p010le> <iters> [output -]
//
// Output (stdout):
//   <avg_ms_per_encode>
//
// Notes:
// - Loads ONE planar YUV frame (8-bit yuv420p or 10-bit p010le) into RAM, uploads to CUDA,
//   and reuses HW frames to benchmark NVENC AV1 throughput.
// - All-intra (gop=1), tune=ULL, preset=p1 for maximum throughput, no explicit rate-control.
// - Requires FFmpeg with av1_nvenc + CUDA (nv-codec-headers). See NVIDIA FFmpeg HW docs. [1][2]
//
// Refs:
// [1] NVIDIA "Using FFmpeg with NVIDIA GPU HW Acceleration" (build/runtime)            https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/ffmpeg-with-nvidia-gpu/index.html
// [2] NVENC AV1 options as exposed by FFmpeg (preset/tune/tiles/surfaces/pixel fmts)   https://gist.github.com/Nicklas373/6cf92b3594561bff326697a93ffb2e5b
// [3] NVENC presets P1..P7 (performance<->quality)                                     https://docs.nvidia.com/video-technologies/video-codec-sdk/12.1/nvenc-video-encoder-api-prog-guide/index.html
// [4] FFmpeg HWAccelIntro (CUDA hw frames & transfer semantics)                        https://trac.ffmpeg.org/wiki/HWAccelIntro

#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/time.h>
#include <libavutil/hwcontext.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static off_t file_size_of(const char *path){ struct stat st; if(stat(path,&st)!=0) return -1; return st.st_size; }

static enum AVPixelFormat pix_fmt_from_str(const char *s){
    if (!strcmp(s,"yuv420p")) return AV_PIX_FMT_YUV420P;
    if (!strcmp(s,"p010le") || !strcmp(s,"yuv420p10le")) return AV_PIX_FMT_P010LE;
    return AV_PIX_FMT_NONE;
}

static int open_av1nvenc_ctx_cuda(AVCodecContext **pctx, AVBufferRef *hw_frames_ref,
                                  int w, int h, enum AVPixelFormat sw_pf)
{
    const AVCodec *codec = avcodec_find_encoder_by_name("av1_nvenc");
    if (!codec) { fprintf(stderr, "av1_nvenc not available.\n"); return -1; }

    AVCodecContext *ctx = avcodec_alloc_context3(codec);
    if (!ctx) { fprintf(stderr, "Cannot alloc codec ctx.\n"); return -1; }

    ctx->codec_type   = AVMEDIA_TYPE_VIDEO;
    ctx->width        = w;
    ctx->height       = h;
    ctx->time_base    = (AVRational){1, 30};
    ctx->framerate    = (AVRational){30, 1};
    ctx->gop_size     = 1;           // all intra
    ctx->max_b_frames = 0;           // low-latency intra path
    ctx->pix_fmt      = AV_PIX_FMT_CUDA;
    ctx->hw_frames_ctx = av_buffer_ref(hw_frames_ref);

    // NVENC AV1 knobs via FFmpeg AVOptions:
    // - preset: p1..p7 (p1=fastest); -tune: ull (ultra low latency). [2][3]
    av_opt_set(ctx->priv_data, "preset", "p1", 0);
    av_opt_set(ctx->priv_data, "tune",   "ull", 0);

    // No explicit RC (avoid bitrate/CBR when GOP=1); let encoder choose defaults.
    // Tiles/surfaces may be tuned if needed, e.g.: av_opt_set_int(ctx->priv_data,"surfaces",4,0);

    if (avcodec_open2(ctx, codec, NULL) < 0) {
        if (ctx->hw_frames_ctx) av_buffer_unref(&ctx->hw_frames_ctx);
        avcodec_free_context(&ctx);
        fprintf(stderr, "Failed to open av1_nvenc.\n");
        return -1;
    }

    *pctx = ctx;
    return 0;
}

int main(int argc, char **argv){
    if (argc < 6 || argc > 7) {
        fprintf(stderr,
            "Args:\n"
            "  av1nvenc_pre <raw.yuv> <WxH> <yuv420p|p010le> <iters> [output -]\n");
        return 1;
    }

    const char *raw_path = argv[1];
    int width=0, height=0;
    if (sscanf(argv[2], "%dx%d", &width, &height) != 2 || width<=0 || height<=0) {
        fprintf(stderr, "Invalid WxH.\n"); return 1;
    }

    enum AVPixelFormat sw_pf = pix_fmt_from_str(argv[3]);
    if (sw_pf == AV_PIX_FMT_NONE) { fprintf(stderr, "Use yuv420p or p010le.\n"); return 1; }

    long iters_l = strtol(argv[4], NULL, 10);
    if (iters_l <= 0 || iters_l > 100000000L) { fprintf(stderr, "iters 1..1e8.\n"); return 1; }
    const int iterations = (int)iters_l;

    const char *out_path = (argc == 7) ? argv[5] : NULL;
    const int do_write   = (out_path && strcmp(out_path, "-") != 0);

    int frame_size = av_image_get_buffer_size(sw_pf, width, height, 1);
    if (frame_size <= 0) { fprintf(stderr, "Bad frame size.\n"); return 1; }

    off_t fsz = file_size_of(raw_path);
    if (fsz < frame_size) { fprintf(stderr, "Input smaller than one frame.\n"); return 1; }

    FILE *in = fopen(raw_path, "rb");
    if (!in) { fprintf(stderr, "Cannot open input.\n"); return 1; }

    uint8_t *rawbuf = (uint8_t*)av_malloc((size_t)frame_size);
    if (!rawbuf) { fclose(in); fprintf(stderr, "OOM raw buffer.\n"); return 1; }
    size_t rd = fread(rawbuf, 1, (size_t)frame_size, in);
    fclose(in);
    if (rd != (size_t)frame_size) { av_free(rawbuf); fprintf(stderr, "Short read.\n"); return 1; }

    // Prepare SW frame
    AVFrame *sw = av_frame_alloc();
    if (!sw) { av_free(rawbuf); fprintf(stderr, "OOM sw frame.\n"); return 1; }
    sw->format = sw_pf; sw->width = width; sw->height = height;
    if (av_image_fill_arrays(sw->data, sw->linesize, rawbuf, sw_pf, width, height, 1) < 0) {
        fprintf(stderr, "av_image_fill_arrays failed.\n"); av_free(rawbuf); av_frame_free(&sw); return 1;
    }

    // CUDA device & frames
    AVBufferRef *hw_dev = NULL;
    if (av_hwdevice_ctx_create(&hw_dev, AV_HWDEVICE_TYPE_CUDA, NULL, NULL, 0) < 0) {
        fprintf(stderr, "av_hwdevice_ctx_create(CUDA) failed.\n");
        av_free(rawbuf); av_frame_free(&sw); return 1;
    }
    AVBufferRef *hw_frames = av_hwframe_ctx_alloc(hw_dev);
    if (!hw_frames) {
        fprintf(stderr, "av_hwframe_ctx_alloc failed.\n");
        av_buffer_unref(&hw_dev); av_free(rawbuf); av_frame_free(&sw); return 1;
    }
    AVHWFramesContext *fctx = (AVHWFramesContext*)hw_frames->data;
    fctx->format    = AV_PIX_FMT_CUDA;    // NVENC takes CUDA frames. [4]
    fctx->sw_format = sw_pf;              // backing SW format
    fctx->width     = width;
    fctx->height    = height;
    fctx->initial_pool_size = 4;

    if (av_hwframe_ctx_init(hw_frames) < 0) {
        fprintf(stderr, "av_hwframe_ctx_init failed.\n");
        av_buffer_unref(&hw_frames); av_buffer_unref(&hw_dev);
        av_free(rawbuf); av_frame_free(&sw); return 1;
    }

    // Pre-allocate & upload N HW frames
    const int NPOOL = 2;
    AVFrame *hw[NPOOL]; memset(hw, 0, sizeof(hw));
    for (int i=0;i<NPOOL;++i){
        hw[i] = av_frame_alloc();
        if (!hw[i]) { fprintf(stderr,"OOM hw frame.\n"); goto fail; }
        if (av_hwframe_get_buffer(hw_frames, hw[i], 0) < 0) { fprintf(stderr,"hwframe_get_buffer failed.\n"); goto fail; }
        if (av_hwframe_transfer_data(hw[i], sw, 0) < 0) { fprintf(stderr,"hwframe_transfer_data failed.\n"); goto fail; }
    }

    // Open av1_nvenc
    AVCodecContext *ctx = NULL;
    if (open_av1nvenc_ctx_cuda(&ctx, hw_frames, width, height, sw_pf) < 0) goto fail;
    AVPacket *pkt = av_packet_alloc(); if (!pkt) { fprintf(stderr,"OOM pkt.\n"); goto fail2; }

    // Timed loop
    int64_t t0 = av_gettime_relative();
    for (int i=0; i<iterations; ++i) {
        AVFrame *f = hw[i & (NPOOL-1)];
        f->pts = i;
        if (avcodec_send_frame(ctx, f) < 0) { fprintf(stderr,"send_frame failed.\n"); goto fail3; }
        for (;;) {
            int r = avcodec_receive_packet(ctx, pkt);
            if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
            if (r < 0) { fprintf(stderr,"receive_packet failed.\n"); goto fail3; }
            av_packet_unref(pkt);
        }
    }
    avcodec_send_frame(ctx, NULL);
    for (;;) {
        int r = avcodec_receive_packet(ctx, pkt);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
        if (r < 0) break;
        av_packet_unref(pkt);
    }
    int64_t t1 = av_gettime_relative();
    double avg_ms = ((t1 - t0)/1e6) * 1000.0 / (double)iterations;
    printf("%.3f\n", avg_ms); fflush(stdout);

    // Optional: write a single frame outside timing
    if (do_write) {
        avcodec_free_context(&ctx);
        if (open_av1nvenc_ctx_cuda(&ctx, hw_frames, width, height, sw_pf) == 0) {
            FILE *out = fopen(out_path,"wb");
            if (out) {
                hw[0]->pts = 0;
                if (avcodec_send_frame(ctx, hw[0]) == 0) {
                    for(;;){
                        int r = avcodec_receive_packet(ctx, pkt);
                        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
                        if (r < 0) break;
                        fwrite(pkt->data, 1, (size_t)pkt->size, out);
                        av_packet_unref(pkt);
                    }
                }
                avcodec_send_frame(ctx, NULL);
                for(;;){
                    int r = avcodec_receive_packet(ctx, pkt);
                    if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
                    if (r < 0) break;
                    fwrite(pkt->data, 1, (size_t)pkt->size, out);
                    av_packet_unref(pkt);
                }
                fclose(out);
            }
        }
    }

    // Cleanup
    av_packet_free(&pkt);
fail3:
    avcodec_free_context(&ctx);
fail2:
    for (int i=0;i<NPOOL;++i) if (hw[i]) av_frame_free(&hw[i]);
    av_buffer_unref(&hw_frames); av_buffer_unref(&hw_dev);
    av_free(rawbuf); av_frame_free(&sw);
    return 0;

fail:
    for (int i=0;i<NPOOL;++i) if (hw[i]) av_frame_free(&hw[i]);
    if (hw_frames) av_buffer_unref(&hw_frames);
    if (hw_dev)    av_buffer_unref(&hw_dev);
    av_free(rawbuf); if (sw) av_frame_free(&sw);
    return 1;
}
