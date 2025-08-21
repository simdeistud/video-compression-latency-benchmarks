#!/usr/bin/env bash
# encode_all_intra_silent.sh
# Args: INPUT.yuv RESOLUTION FRAMES
# Example: ./encode_all_intra_silent.sh input.yuv 1920x1080 30

set -u
LC_ALL=C

FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"

if ! command -v "$FFMPEG_BIN" >/dev/null 2>&1; then
  echo "ffmpeg not found" >&2
  exit 1
fi

if [ $# -lt 2 ]; then
  echo "Usage: $0 <input.yuv> <WxH> <frames>" >&2
  exit 1
fi

INPUT="$1"
RES="$2"
FRAMES="$3"

if [ ! -f "$INPUT" ]; then
  echo "Input not found: $INPUT" >&2
  exit 1
fi

BASE="$(basename -- "$INPUT")"
STEM="${BASE%.*}"
OUT_PREFIX="${STEM}__"

# List encoders once
ENC_LIST="$("$FFMPEG_BIN" -hide_banner -encoders 2>/dev/null | sed 's/^[[:space:]]*//')"

have_encoder() {
  # match encoder token as a whole word in the encoders list
  grep -Eq "(^|[[:space:]])$1([[:space:]]|$)" <<<"$ENC_LIST"
}

# Silent runner: prints "<label> <seconds>s" or "<label> FAILED"
run() {
  local label="$1"; shift
  local status elapsed avg_time_ms
  TIMEFORMAT='%3R'
  { time "$FFMPEG_BIN" -y -hide_banner -nostats -loglevel error "$@" >/dev/null 2>&1; } \
    2> .__time.$$ 
  status=$?
  if [ $status -eq 0 ]; then
    elapsed="$(tr -d '\n\r' < .__time.$$)"
    avg_time_ms=$(awk "BEGIN {printf \"%.2f\", ($elapsed * 1000) / $FRAMES}")
    printf "%s %ss (avg/frame: %sms)\n" "$label" "$elapsed" "$avg_time_ms"
  else
    printf "%s FAILED\n" "$label"
  fi
  rm -f .__time.$$
}


# Common raw input
COMMON_IN=(-i "$INPUT")

########################################
# Software encoders
########################################

have_encoder mpeg2video && run mpeg2video \
  "${COMMON_IN[@]}" -c:v mpeg2video -intra 1 -g 1 -bf 0 -qscale:v 2 \
  "${OUT_PREFIX}mpeg2video.m2v"

have_encoder mpeg4 && run mpeg4 \
  "${COMMON_IN[@]}" -c:v mpeg4 -g 1 -bf 0 -qscale:v 2 \
  "${OUT_PREFIX}mpeg4.m4v"

have_encoder libxvid && run libxvid \
  "${COMMON_IN[@]}" -c:v libxvid -g 1 -bf 0 -qscale:v 2 \
  "${OUT_PREFIX}libxvid.m4v"

have_encoder msmpeg4 && run msmpeg4 \
  "${COMMON_IN[@]}" -c:v msmpeg4 -g 1 -bf 0 -qscale:v 2 \
  "${OUT_PREFIX}msmpeg4.avi"

have_encoder mjpeg && run mjpeg \
  "${COMMON_IN[@]}" -c:v mjpeg -q:v 3 \
  "${OUT_PREFIX}mjpeg.avi"

have_encoder libx264 && run libx264 \
  "${COMMON_IN[@]}" -c:v libx264 -preset ultrafast -tune zerolatency \
  -g 1 -bf 0 -sc_threshold 0 \
  "${OUT_PREFIX}libx264.mp4"

# VP8 (libvpx) all-intra realtime
have_encoder libvpx && run libvpx_vp8 \
  "${COMMON_IN[@]}" -c:v libvpx -deadline realtime -cpu-used 8 -lag-in-frames 0 \
  -g 1 -auto-alt-ref 0 -b:v 0 -crf 30 \
  "${OUT_PREFIX}libvpx.webm"

# VP9 (libvpx-vp9) all-intra realtime
have_encoder libvpx-vp9 && run libvpx_vp9 \
  "${COMMON_IN[@]}" -c:v libvpx-vp9 -deadline realtime -cpu-used 8 -row-mt 1 \
  -lag-in-frames 0 -g 1 -auto-alt-ref 0 -b:v 0 -crf 32 \
  "${OUT_PREFIX}libvpx-vp9.webm"

# AV1 software
have_encoder libaom-av1 && run libaom_av1 \
  "${COMMON_IN[@]}" -c:v libaom-av1 -usage realtime -cpu-used 8 -row-mt 1 \
  -lag-in-frames 0 -g 1 -b:v 0 -crf 34 \
  "${OUT_PREFIX}libaom-av1.mkv"

have_encoder librav1e && run librav1e \
  "${COMMON_IN[@]}" -c:v librav1e -rav1e-params speed=10:low_latency=true:keyint=1 \
  "${OUT_PREFIX}librav1e.mkv"

have_encoder libsvtav1 && run libsvtav1 \
  "${COMMON_IN[@]}" -c:v libsvtav1 -preset 12 -g 1 \
  "${OUT_PREFIX}libsvtav1.mkv"

########################################
# NVIDIA NVENC (CUDA)
########################################
have_encoder h264_nvenc && run h264_nvenc \
  -init_hw_device cuda=cu -filter_hw_device cu \
  "${COMMON_IN[@]}" -vf format=yuv420p,hwupload \
  -c:v h264_nvenc -preset p1 -tune ll -g 1 -bf 0 -rc constqp -qp 23 \
  "${OUT_PREFIX}h264_nvenc.mp4"

have_encoder av1_nvenc && run av1_nvenc \
  -init_hw_device cuda=cu -filter_hw_device cu \
  "${COMMON_IN[@]}" -vf format=yuv420p,hwupload \
  -c:v av1_nvenc -preset p1 -tune ll -g 1 -bf 0 -rc constqp -qp 28 \
  "${OUT_PREFIX}av1_nvenc.mkv"

########################################
# Intel Quick Sync Video (QSV)
########################################
have_encoder mpeg2_qsv && run mpeg2_qsv \
  -init_hw_device qsv=hw -filter_hw_device hw \
  "${COMMON_IN[@]}" -vf format=nv12,hwupload \
  -c:v mpeg2_qsv -g 1 -bf 0 -look_ahead 0 -preset veryfast \
  "${OUT_PREFIX}mpeg2_qsv.m2v"

have_encoder h264_qsv && run h264_qsv \
  -init_hw_device qsv=hw -filter_hw_device hw \
  "${COMMON_IN[@]}" -vf format=nv12,hwupload \
  -c:v h264_qsv -g 1 -bf 0 -look_ahead 0 -preset veryfast \
  "${OUT_PREFIX}h264_qsv.mp4"

have_encoder vp9_qsv && run vp9_qsv \
  -init_hw_device qsv=hw -filter_hw_device hw \
  "${COMMON_IN[@]}" -vf format=nv12,hwupload \
  -c:v vp9_qsv -g 1 -bf 0 -low_power 1 \
  "${OUT_PREFIX}vp9_qsv.webm"

have_encoder av1_qsv && run av1_qsv \
  -init_hw_device qsv=hw -filter_hw_device hw \
  "${COMMON_IN[@]}" -vf format=nv12,hwupload \
  -c:v av1_qsv -g 1 -bf 0 -look_ahead 0 -low_power 1 \
  "${OUT_PREFIX}av1_qsv.mkv"

# MJPEG QSV
have_encoder mjpeg_qsv && run mjpeg_qsv \
  -init_hw_device qsv=hw -filter_hw_device hw \
  "${COMMON_IN[@]}" -vf format=nv12,hwupload \
  -c:v mjpeg_qsv -q:v 3 \
  "${OUT_PREFIX}mjpeg_qsv.avi"

########################################
# VAAPI (Intel/AMD iGPU)
########################################
VA_NODE="/dev/dri/renderD128"
if [ -e "$VA_NODE" ]; then
  have_encoder mpeg2_vaapi && run mpeg2_vaapi \
    -init_hw_device vaapi=va:"$VA_NODE" -filter_hw_device va \
    "${COMMON_IN[@]}" -vf format=nv12,hwupload \
    -c:v mpeg2_vaapi -g 1 -bf 0 -rc_mode cqp -qp 18 \
    "${OUT_PREFIX}mpeg2_vaapi.m2v"

  have_encoder mjpeg_vaapi && run mjpeg_vaapi \
    -init_hw_device vaapi=va:"$VA_NODE" -filter_hw_device va \
    "${COMMON_IN[@]}" -vf format=nv12,hwupload \
    -c:v mjpeg_vaapi -q:v 3 \
    "${OUT_PREFIX}mjpeg_vaapi.avi"

  have_encoder h264_vaapi && run h264_vaapi \
    -init_hw_device vaapi=va:"$VA_NODE" -filter_hw_device va \
    "${COMMON_IN[@]}" -vf format=nv12,hwupload \
    -c:v h264_vaapi -g 1 -bf 0 -rc_mode cqp -qp 23 \
    "${OUT_PREFIX}h264_vaapi.mp4"

  have_encoder vp8_vaapi && run vp8_vaapi \
    -init_hw_device vaapi=va:"$VA_NODE" -filter_hw_device va \
    "${COMMON_IN[@]}" -vf format=nv12,hwupload \
    -c:v vp8_vaapi -g 1 -bf 0 \
    "${OUT_PREFIX}vp8_vaapi.webm"

  have_encoder vp9_vaapi && run vp9_vaapi \
    -init_hw_device vaapi=va:"$VA_NODE" -filter_hw_device va \
    "${COMMON_IN[@]}" -vf format=nv12,hwupload \
    -c:v vp9_vaapi -g 1 -bf 0 \
    "${OUT_PREFIX}vp9_vaapi.webm"

  have_encoder av1_vaapi && run av1_vaapi \
    -init_hw_device vaapi=va:"$VA_NODE" -filter_hw_device va \
    "${COMMON_IN[@]}" -vf format=nv12,hwupload \
    -c:v av1_vaapi -g 1 -bf 0 -rc_mode cqp -qp 28 \
    "${OUT_PREFIX}av1_vaapi.mkv"
fi

########################################
# V4L2 M2M (SoC)
########################################
have_encoder mpeg4_v4l2m2m && run mpeg4_v4l2m2m \
  "${COMMON_IN[@]}" -c:v mpeg4_v4l2m2m -g 1 -bf 0 \
  "${OUT_PREFIX}mpeg4_v4l2m2m.m4v"

have_encoder h264_v4l2m2m && run h264_v4l2m2m \
  "${COMMON_IN[@]}" -c:v h264_v4l2m2m -g 1 -bf 0 \
  "${OUT_PREFIX}h264_v4l2m2m.mp4"

have_encoder vp8_v4l2m2m && run vp8_v4l2m2m \
  "${COMMON_IN[@]}" -c:v vp8_v4l2m2m -g 1 -bf 0 \
  "${OUT_PREFIX}vp8_v4l2m2m.webm"

