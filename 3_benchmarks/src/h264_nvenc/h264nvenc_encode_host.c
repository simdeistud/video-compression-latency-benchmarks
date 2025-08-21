// nvenc_encode_hostupload.c
// Single-frame NVENC micro-benchmark (host->device upload INCLUDED each iter).
// Usage:
//   nvenc_host <raw.yuv> <WxH> <yuv420p|yuv444p> <cq|cbr|cbr_strict> <rc_value> <iters> [output -]
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef enum { RC_CQ = 0, RC_CBR = 1, RC_CBR_STRICT = 2 } RcMode;

static off_t file_size_of(const char *path) {
  struct stat st; if (stat(path, &st) != 0) return -1; return st.st_size;
}

static enum AVPixelFormat pix_fmt_from_str(const char *s) {
  if (!strcmp(s, "yuv420p")) return AV_PIX_FMT_YUV420P;
  if (!strcmp(s, "yuv444p")) return AV_PIX_FMT_YUV444P; // NVENC supports 4:4:4 profile
  return AV_PIX_FMT_NONE;
}

static const char* nvenc_profile_for_pixfmt(enum AVPixelFormat pf) {
  return (pf == AV_PIX_FMT_YUV444P) ? "high444p" : "high";
}

// Open h264_nvenc for SW-frame input; libavcodec will upload to VRAM internally.
static int open_nvenc_ctx(AVCodecContext **pctx, int w, int h,
                          enum AVPixelFormat sw_pf, RcMode rc_mode, int rc_val)
{
  const AVCodec *codec = avcodec_find_encoder_by_name("h264_nvenc");
  if (!codec) { fprintf(stderr, "h264_nvenc not available.\n"); return -1; }

  AVCodecContext *ctx = avcodec_alloc_context3(codec);
  if (!ctx) { fprintf(stderr, "Cannot alloc codec ctx.\n"); return -1; }

  ctx->codec_type   = AVMEDIA_TYPE_VIDEO;
  ctx->width        = w;
  ctx->height       = h;
  ctx->time_base    = (AVRational){1, 30};
  ctx->framerate    = (AVRational){30, 1};
  ctx->gop_size     = 1;   // all-intra
  ctx->max_b_frames = 0;   // no B-frames
  ctx->pix_fmt      = sw_pf;

  // NVENC options
  // Preset scale p1..p7 exists; p1 = fastest. Use p1 for throughput benchmarking.
  av_opt_set(ctx->priv_data, "preset", "p1", 0);                     // NVENC preset ladder. [6](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.1/nvenc-video-encoder-api-prog-guide/index.html)
  av_opt_set(ctx->priv_data, "profile", nvenc_profile_for_pixfmt(sw_pf), 0);
  av_opt_set(ctx->priv_data, "zerolatency", "1", 0);                 // NVENC low-latency flags. [3](https://stackoverflow.com/questions/59779205/best-settings-for-h264-nvenc-to-minimize-latency-ffmpeg)
  av_opt_set(ctx->priv_data, "delay", "0", 0);                       // ditto. [3](https://stackoverflow.com/questions/59779205/best-settings-for-h264-nvenc-to-minimize-latency-ffmpeg)

  // Rate control
  if (rc_mode == RC_CQ) {
    av_opt_set(ctx->priv_data, "rc", "vbr", 0);                      // CQ uses VBR mode with 'cq' knob. [2](https://superuser.com/questions/1236275/how-can-i-use-crf-encoding-with-nvenc-in-ffmpeg)
    av_opt_set_int(ctx->priv_data, "cq", rc_val, 0);
    ctx->bit_rate = 0; ctx->rc_max_rate = 0; ctx->rc_buffer_size = 0;
  } else {
    const int kbps = rc_val; const int64_t bps = (int64_t)kbps * 1000;
    av_opt_set(ctx->priv_data, "rc", "cbr", 0);
    ctx->bit_rate = bps; ctx->rc_max_rate = bps;
    // VBV: looser for CBR, tight for CBR_STRICT
    ctx->rc_buffer_size = (rc_mode == RC_CBR) ? (int64_t)kbps*2000 : (int64_t)kbps*1000;
  }

  if (avcodec_open2(ctx, codec, NULL) < 0) {
    avcodec_free_context(&ctx);
    fprintf(stderr, "Failed to open h264_nvenc.\n");
    return -1;
  }
  *pctx = ctx;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 7 || argc > 8) {
    fprintf(stderr,
      "Args:\n"
      "  nvenc_host <raw.yuv> <WxH> <yuv420p|yuv444p> <cq|cbr|cbr_strict> <rc_value> <iters> [output -]\n");
    return 1;
  }
  const char *raw_path = argv[1];
  int width=0, height=0; if (sscanf(argv[2], "%dx%d", &width,&height)!=2 || width<=0||height<=0) {
    fprintf(stderr,"Invalid WxH.\n"); return 1;
  }
  enum AVPixelFormat sw_pf = pix_fmt_from_str(argv[3]);
  if (sw_pf == AV_PIX_FMT_NONE) { fprintf(stderr,"Use yuv420p|yuv444p.\n"); return 1; }

  RcMode rc_mode;
  if (!strcmp(argv[4],"cq")) rc_mode = RC_CQ;
  else if (!strcmp(argv[4],"cbr")) rc_mode = RC_CBR;
  else if (!strcmp(argv[4],"cbr_strict")) rc_mode = RC_CBR_STRICT;
  else { fprintf(stderr,"rc_mode must be cq|cbr|cbr_strict.\n"); return 1; }

  int rc_val = atoi(argv[5]);
  if (rc_mode == RC_CQ) {
    if (rc_val < 0 || rc_val > 51) { fprintf(stderr,"cq must be 0..51.\n"); return 1; }
  } else if (rc_val <= 0) { fprintf(stderr,"Bitrate (kbps) > 0.\n"); return 1; }

  long iters_l = strtol(argv[6], NULL, 10);
  if (iters_l <= 0 || iters_l > 100000000L) { fprintf(stderr,"iters 1..1e8.\n"); return 1; }
  const int iterations = (int)iters_l;

  const char *out_path = (argc == 8) ? argv[7] : NULL;
  const int do_write = (out_path && strcmp(out_path, "-")!=0);

  int frame_size = av_image_get_buffer_size(sw_pf, width, height, 1);
  if (frame_size <= 0) { fprintf(stderr,"Bad frame size.\n"); return 1; }
  off_t fsz = file_size_of(raw_path); if (fsz < frame_size) { fprintf(stderr,"Input smaller than one frame.\n"); return 1; }

  FILE *in = fopen(raw_path,"rb"); if (!in) { fprintf(stderr,"Cannot open input.\n"); return 1; }
  uint8_t *rawbuf = (uint8_t*)av_malloc((size_t)frame_size);
  if (!rawbuf) { fclose(in); fprintf(stderr,"OOM raw buffer.\n"); return 1; }
  size_t rd = fread(rawbuf,1,(size_t)frame_size,in); fclose(in);
  if (rd != (size_t)frame_size) { av_free(rawbuf); fprintf(stderr,"Short read.\n"); return 1; }

  AVCodecContext *ctx = NULL;
  if (open_nvenc_ctx(&ctx, width, height, sw_pf, rc_mode, rc_val) < 0) { av_free(rawbuf); return 1; }

  AVFrame *frame = av_frame_alloc(); AVPacket *pkt = av_packet_alloc();
  if (!frame || !pkt) { if(frame) av_frame_free(&frame); if(pkt) av_packet_free(&pkt);
    avcodec_free_context(&ctx); av_free(rawbuf); fprintf(stderr,"OOM frame/pkt.\n"); return 1; }
  frame->format = sw_pf; frame->width = width; frame->height = height;
  if (av_image_fill_arrays(frame->data, frame->linesize, rawbuf, sw_pf, width, height, 1) < 0) {
    fprintf(stderr,"av_image_fill_arrays failed.\n");
    av_packet_free(&pkt); av_frame_free(&frame); avcodec_free_context(&ctx); av_free(rawbuf); return 1;
  }

  int64_t t0 = av_gettime_relative();
  for (int i=0; i<iterations; ++i) {
    frame->pts = i;
    if (avcodec_send_frame(ctx, frame) < 0) { fprintf(stderr,"send_frame failed.\n"); goto fail; }
    for (;;) {
      int r = avcodec_receive_packet(ctx, pkt);
      if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
      if (r < 0) { fprintf(stderr,"receive_packet failed.\n"); goto fail; }
      av_packet_unref(pkt);
    }
  }
  // Flush
  avcodec_send_frame(ctx, NULL);
  for (;;) {
    int r = avcodec_receive_packet(ctx, pkt);
    if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
    if (r < 0) break;
    av_packet_unref(pkt);
  }
  int64_t t1 = av_gettime_relative();
  double avg_ms = ((t1 - t0) / 1e6) * 1000.0 / (double)iterations;
  printf("%.3f\n", avg_ms); fflush(stdout);

  if (do_write) {
    // re-open encoder to emit a one-frame bitstream without polluting timing
    avcodec_free_context(&ctx);
    if (open_nvenc_ctx(&ctx, width,height, sw_pf, rc_mode, rc_val) < 0) goto cleanup;
    FILE *out = fopen(out_path,"wb"); if(!out){ fprintf(stderr,"Cannot open output.\n"); goto cleanup; }
    frame->pts = 0;
    if (avcodec_send_frame(ctx, frame) == 0) {
      for (;;) {
        int r = avcodec_receive_packet(ctx, pkt);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
        if (r < 0) break;
        fwrite(pkt->data,1,(size_t)pkt->size,out);
        av_packet_unref(pkt);
      }
      avcodec_send_frame(ctx, NULL);
      for (;;) {
        int r = avcodec_receive_packet(ctx, pkt);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
        if (r < 0) break;
        fwrite(pkt->data,1,(size_t)pkt->size,out);
        av_packet_unref(pkt);
      }
    }
    fclose(out);
  }

cleanup:
  av_packet_free(&pkt); av_frame_free(&frame); avcodec_free_context(&ctx); av_free(rawbuf); return 0;
fail:
  av_packet_free(&pkt); av_frame_free(&frame); avcodec_free_context(&ctx); av_free(rawbuf); return 1;
}

