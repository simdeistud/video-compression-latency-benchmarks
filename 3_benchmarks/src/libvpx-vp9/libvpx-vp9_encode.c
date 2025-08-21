// Single-frame VP9 (libvpx-vp9) encode micro-benchmark.
// Encodes the SAME raw frame 'iterations' times, measures encode time only.
//
// Usage:
// vp9_encode_frame <raw_frame.yuv> <WxH> <yuv420p|yuv422p|yuv444p> <cpu_mode(0|1)>
//   <rc_mode: crf|cbr|cbr_strict> <rc_value> <iterations> [output -]
//
// Output (stdout):
//   <avg_ms_per_encode>
//
// Notes:
// - Loads exactly ONE raw planar 8-bit frame into RAM.
// - Encoder configured for all-intra (GOP=1), realtime/low-latency.
// - Timed section excludes setup/teardown and disk I/O.
// - If output path is given, a ONE-FRAME stream is encoded/written AFTER timing.

#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

typedef enum { RC_CRF = 0, RC_CBR = 1, RC_CBR_STRICT = 2 } RcMode;

static off_t file_size_of(const char *path) {
  struct stat st;
  if (stat(path, &st) != 0) return -1;
  return st.st_size;
}

static enum AVPixelFormat pix_fmt_from_str(const char *s) {
  if (!strcmp(s, "yuv420p")) return AV_PIX_FMT_YUV420P;
  if (!strcmp(s, "yuv422p")) return AV_PIX_FMT_YUV422P;
  if (!strcmp(s, "yuv444p")) return AV_PIX_FMT_YUV444P;
  return AV_PIX_FMT_NONE;
}

static int vp9_profile_for_pixfmt(enum AVPixelFormat pf) {
  // VP9 profile 0: 8-bit 4:2:0; profile 1: 8-bit 4:2:2 / 4:4:4.
  switch (pf) {
    case AV_PIX_FMT_YUV420P: return 0;
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUV444P: return 1;
    default: return 0;
  }
}

static RcMode rc_mode_from_str(const char *s) {
  if (!strcmp(s, "crf")) return RC_CRF;
  if (!strcmp(s, "cbr")) return RC_CBR;
  if (!strcmp(s, "cbr_strict") || !strcmp(s, "strict_cbr")) return RC_CBR_STRICT;
  return (RcMode)-1;
}

static int open_vp9_ctx(AVCodecContext **pctx,
                        int w, int h, enum AVPixelFormat pf,
                        int use_all_cpus, RcMode rc_mode, int rc_value)
{
  const AVCodec *codec = avcodec_find_encoder_by_name("libvpx-vp9");
  if (!codec) { fprintf(stderr, "libvpx-vp9 not available.\n"); return -1; }

  AVCodecContext *ctx = avcodec_alloc_context3(codec);
  if (!ctx) { fprintf(stderr, "Cannot alloc codec ctx.\n"); return -1; }

  // Base config: all-intra, realtime, low-latency
  ctx->codec_type = AVMEDIA_TYPE_VIDEO;
  ctx->codec_id   = codec->id;
  ctx->width  = w;
  ctx->height = h;
  ctx->pix_fmt = pf;
  ctx->time_base = (AVRational){1, 30};
  ctx->framerate = (AVRational){30, 1};
  ctx->gop_size = 1;      // all-intra
  ctx->max_b_frames = 0;  // VP9: no B-frames
  ctx->thread_count = use_all_cpus ? 0 : 1; // 0 = auto/all
  ctx->thread_type  = FF_THREAD_SLICE;
  ctx->flags &= ~AV_CODEC_FLAG_GLOBAL_HEADER;

  // VP9-specific threading & realtime knobs
  av_opt_set(ctx->priv_data, "deadline", "realtime", 0);
  // cpu-used: 0(slowest)...8(fastest typical). Choose fast when using all cores.
  av_opt_set_int(ctx->priv_data, "cpu-used", use_all_cpus ? 8 : 2, 0);
  av_opt_set_int(ctx->priv_data, "lag-in-frames", 0, 0); // zero lookahead
  av_opt_set_int(ctx->priv_data, "row-mt", 1, 0);        // row-based MT on
  // Tiling to scale threads (log2): 4 cols (2^2), 2 rows (2^1) when using all CPUs.
  if (use_all_cpus) {
    av_opt_set_int(ctx->priv_data, "tile-columns", 2, 0);
    av_opt_set_int(ctx->priv_data, "tile-rows", 1, 0);
  } else {
    av_opt_set_int(ctx->priv_data, "tile-columns", 0, 0);
    av_opt_set_int(ctx->priv_data, "tile-rows", 0, 0);
  }
  // Disable alt-ref (primarily 2-pass) for this all-intra, low-latency benchmark.
  av_opt_set_int(ctx->priv_data, "auto-alt-ref", 0, 0);

  // Profile selection to match input subsampling
  av_opt_set_int(ctx->priv_data, "profile", vp9_profile_for_pixfmt(pf), 0);

  // Rate control
  if (rc_mode == RC_CRF) {
    // VP9 CRF 0..63, lower = better. Use pure CQ (b:v = 0).
    av_opt_set_int(ctx->priv_data, "crf", rc_value, 0);
    ctx->bit_rate = 0;
    ctx->rc_max_rate = 0;
    ctx->rc_buffer_size = 0;
  } else {
    const int kbps = rc_value;
    const int64_t bps = (int64_t)kbps * 1000;
    ctx->bit_rate     = bps;
    ctx->rc_max_rate  = bps;
    ctx->rc_buffer_size = (int64_t)(kbps * 2) * 1000; // 2x VBV

    // Engage CBR in wrapper: set minrate/maxrate/bufsize
    av_opt_set_int(ctx->priv_data, "minrate", bps, 0);
    av_opt_set_int(ctx->priv_data, "maxrate", bps, 0);
    av_opt_set_int(ctx->priv_data, "bufsize", ctx->rc_buffer_size, 0);

    if (rc_mode == RC_CBR_STRICT) {
      av_opt_set_int(ctx->priv_data, "undershoot-pct", 0, 0);
      av_opt_set_int(ctx->priv_data, "overshoot-pct", 0, 0);
      av_opt_set_int(ctx->priv_data, "rc_init_occupancy", ctx->rc_buffer_size, 0);
      av_opt_set_int(ctx->priv_data, "lag-in-frames", 0, 0);
      // Optional: turn off AQ in strictest mode
      av_opt_set_int(ctx->priv_data, "aq-mode", 0, 0);
    }
  }

  if (avcodec_open2(ctx, codec, NULL) < 0) {
    avcodec_free_context(&ctx);
    fprintf(stderr, "Failed to open libvpx-vp9.\n");
    return -1;
  }
  *pctx = ctx;
  return 0;
}

int main(int argc, char **argv) {
  // <raw_frame.yuv> <WxH> <yuv420p|yuv422p|yuv444p> <cpu_mode(0|1)>
  // <rc_mode: crf|cbr|cbr_strict> <rc_value> <iterations> [output -]
  if (argc < 8 || argc > 9) {
    fprintf(stderr,
      "Args:\n"
      " vp9_encode_frame <raw_frame.yuv> <WxH> <yuv420p|yuv422p|yuv444p> <cpu_mode(0|1)>\n"
      "  <rc_mode: crf|cbr|cbr_strict> <rc_value> <iterations> [output -]\n"
      "Notes:\n"
      " - crf: rc_value = CRF in 0..63 (VP9)\n"
      " - cbr/cbr_strict: rc_value = bitrate in kbps (>0)\n");
    return 1;
  }

  const char *raw_path = argv[1];
  int width=0, height=0;
  if (sscanf(argv[2], "%dx%d", &width, &height) != 2 || width<=0 || height<=0) {
    fprintf(stderr, "Invalid WxH.\n"); return 1;
  }

  enum AVPixelFormat pix_fmt = pix_fmt_from_str(argv[3]);
  if (pix_fmt == AV_PIX_FMT_NONE) {
    fprintf(stderr, "Unsupported subsampling. Use yuv420p|yuv422p|yuv444p.\n");
    return 1;
  }

  int use_all_cpus = atoi(argv[4]) ? 1 : 0;
  RcMode rc_mode = rc_mode_from_str(argv[5]);
  if ((int)rc_mode == -1) { fprintf(stderr, "rc_mode must be crf|cbr|cbr_strict.\n"); return 1; }

  int rc_value = atoi(argv[6]);
  if (rc_mode == RC_CRF) {
    if (rc_value < 0 || rc_value > 63) { fprintf(stderr, "CRF must be 0..63 (VP9).\n"); return 1; }
  } else {
    if (rc_value <= 0) { fprintf(stderr, "Bitrate (kbps) must be > 0.\n"); return 1; }
  }

  long iterations_l = strtol(argv[7], NULL, 10);
  if (iterations_l <= 0 || iterations_l > 100000000L) {
    fprintf(stderr, "iterations must be in 1..1e8.\n"); return 1;
  }
  const int iterations = (int)iterations_l;

  const char *out_path = (argc == 9) ? argv[8] : NULL;
  const int do_write = (out_path && strcmp(out_path, "-") != 0);

  // Load one frame
  int frame_size = av_image_get_buffer_size(pix_fmt, width, height, 1);
  if (frame_size <= 0) { fprintf(stderr, "Bad frame size.\n"); return 1; }
  off_t fsz = file_size_of(raw_path);
  if (fsz < frame_size) { fprintf(stderr, "Input file smaller than one frame.\n"); return 1; }

  FILE *in = fopen(raw_path, "rb");
  if (!in) { fprintf(stderr, "Cannot open input.\n"); return 1; }
  uint8_t *rawbuf = (uint8_t*)av_malloc((size_t)frame_size);
  if (!rawbuf) { fclose(in); fprintf(stderr, "OOM allocating raw buffer.\n"); return 1; }
  size_t rd = fread(rawbuf, 1, (size_t)frame_size, in);
  fclose(in);
  if (rd != (size_t)frame_size) {
    av_free(rawbuf); fprintf(stderr, "Short read.\n"); return 1;
  }

  // Open encoder
  AVCodecContext *ctx = NULL;
  if (open_vp9_ctx(&ctx, width, height, pix_fmt, use_all_cpus, rc_mode, rc_value) < 0) {
    av_free(rawbuf); return 1;
  }

  AVFrame *frame = av_frame_alloc();
  AVPacket *pkt = av_packet_alloc();
  if (!frame || !pkt) {
    if (frame) av_frame_free(&frame);
    if (pkt) av_packet_free(&pkt);
    avcodec_free_context(&ctx);
    av_free(rawbuf);
    fprintf(stderr, "OOM allocating frame/packet.\n");
    return 1;
  }

  frame->format = ctx->pix_fmt;
  frame->width  = ctx->width;
  frame->height = ctx->height;
  if (av_image_fill_arrays(frame->data, frame->linesize, rawbuf, pix_fmt, width, height, 1) < 0) {
    fprintf(stderr, "av_image_fill_arrays failed.\n");
    av_packet_free(&pkt); av_frame_free(&frame); avcodec_free_context(&ctx); av_free(rawbuf);
    return 1;
  }

  // --- Timed encode loop (no disk I/O) ---
  int64_t t0 = av_gettime_relative();
  int64_t total_bytes = 0;
  int total_pkts = 0;

  for (int i = 0; i < iterations; ++i) {
    frame->pts = i;
    if (avcodec_send_frame(ctx, frame) < 0) {
      fprintf(stderr, "send_frame failed.\n"); goto fail;
    }
    for (;;) {
      int r = avcodec_receive_packet(ctx, pkt);
      if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
      if (r < 0) { fprintf(stderr, "receive_packet failed.\n"); goto fail; }
      total_bytes += pkt->size;
      total_pkts++;
      av_packet_unref(pkt);
    }
  }

  // Flush
  if (avcodec_send_frame(ctx, NULL) < 0) { fprintf(stderr, "flush send_frame(NULL) failed.\n"); goto fail; }
  for (;;) {
    int r = avcodec_receive_packet(ctx, pkt);
    if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
    if (r < 0) { fprintf(stderr, "flush receive_packet failed.\n"); goto fail; }
    total_bytes += pkt->size;
    total_pkts++;
    av_packet_unref(pkt);
  }

  int64_t t1 = av_gettime_relative();
  double total_s = (t1 - t0) / 1e6;
  double avg_ms = (iterations > 0) ? (total_s * 1000.0 / iterations) : 0.0;

  printf("%.3f\n", avg_ms);
  fflush(stdout);

  // --- Optional: write ONE-FRAME bitstream after timing ---
  if (do_write) {
    avcodec_free_context(&ctx);
    if (open_vp9_ctx(&ctx, width, height, pix_fmt, use_all_cpus, rc_mode, rc_value) < 0) {
      fprintf(stderr, "Reopen encoder failed for output.\n"); goto cleanup;
    }
    FILE *out = fopen(out_path, "wb");
    if (!out) { fprintf(stderr, "Cannot open output for write.\n"); goto cleanup; }
    setvbuf(out, NULL, _IOFBF, 1<<20);
    frame->pts = 0;
    if (avcodec_send_frame(ctx, frame) < 0) { fprintf(stderr, "send_frame(out) failed.\n"); fclose(out); goto cleanup; }
    for (;;) {
      int r = avcodec_receive_packet(ctx, pkt);
      if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
      if (r < 0) { fprintf(stderr, "receive_packet(out) failed.\n"); fclose(out); goto cleanup; }
      if (pkt->size > 0) {
        size_t wr = fwrite(pkt->data, 1, (size_t)pkt->size, out);
        if (wr != (size_t)pkt->size) { fprintf(stderr, "Short write.\n"); }
      }
      av_packet_unref(pkt);
    }
    // flush
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
  av_free(rawbuf);
  return 0;

fail:
  av_packet_free(&pkt);
  av_frame_free(&frame);
  avcodec_free_context(&ctx);
  av_free(rawbuf);
  return 1;
}
