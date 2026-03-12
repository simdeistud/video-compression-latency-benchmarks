
#!/usr/bin/env bash
set -euo pipefail

# Usage: ./jpegturbo_to_lossless.sh <subsampling:{420,422,444}> <quality:0-100> <input_video> <output.{mkv|zip}>
# Example: ./jpegturbo_to_lossless.sh 420 75 input.mp4 out.mkv
#          ./jpegturbo_to_lossless.sh 444 90 input.y4m out.zip

if [ $# -ne 4 ]; then
  echo "Usage: $0 <subsampling:{420,422,444}> <quality:0-100> <input_video> <output.{mkv|zip}>"
  exit 1
fi

SUBS="$1"      # 420 | 422 | 444
QUALITY="$2"   # 0..100
INPUT="$3"
OUTPUT="$4"

jpegturbo="/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/libjpeg-turbo/cmake-build-debug/libjpeg-turbo_encode"

# --- Probe input video geometry and frame rate (keep rate as a rational) ---
WIDTH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width  -of csv=p=0 "$INPUT")
HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$INPUT")
FPS_R=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$INPUT")

echo "Detected: ${WIDTH}x${HEIGHT} @ ${FPS_R} fps"

# --- Working directories ---
WORKDIR=$(mktemp -d)
PPM_DIR="$WORKDIR/ppm"    # per-frame PPM (for reliable splitting)
JPEG_DIR="$WORKDIR/jpeg"  # per-frame JPEG from jpegturbo
mkdir -p "$PPM_DIR" "$JPEG_DIR"

echo "Workdir: $WORKDIR"
echo "[1/3] Extracting frames -> PPM (one file per frame)..."

# Rationale: image2+PPM guarantees one file per frame (PPM has headers).
# We keep RGB24 here and later convert each file to raw RGB on-the-fly.
ffmpeg -hide_banner -loglevel error \
  -i "$INPUT" \
  -pix_fmt rgb24 \
  -f image2 "$PPM_DIR/frame_%06d.ppm"

echo "[2/3] PPM -> raw rgb24 -> jpegturbo_encode (-> per-frame JPEG)..."

# Encode each frame with your jpegturbo (stdin: raw rgb24, stdout: JPEG -> file)
# To avoid writing .rgb files, we convert each PPM to raw rgb24 via ffmpeg per file.
shopt -s nullglob
for f in "$PPM_DIR"/frame_*.ppm; do
  base=$(basename "$f" .ppm)
  out="$JPEG_DIR/$base.jpg"

  # PPM -> raw rgb24 to stdout, then pipe to jpegturbo
  ffmpeg -hide_banner -loglevel error -y \
    -i "$f" -f rawvideo -pix_fmt rgb24 - \
  | "$jpegturbo" \
      -w "$WIDTH" \
      -h "$HEIGHT" \
      -s "$SUBS" \
      -d "int" \
      -q "$QUALITY" \
      -r 0 \
      -i 1 \
      -o "$out"
done

echo "[3/3] Packaging losslessly..."

ext="${OUTPUT##*.}"
shopt -s nocasematch

if [[ "$ext" == "mkv" ]]; then
  # Lossless, playback-friendly option: FFV1 in Matroska.
  # We decode the JPEGs and write FFV1 (mathematically lossless), no added loss.
  # Match the chroma format to your JPEG subsampling to avoid any resampling.
  case "$SUBS" in
    420) PIXFMT=yuv420p ;;
    422) PIXFMT=yuv422p ;;
    444) PIXFMT=yuv444p ;;
    *)   echo "Unsupported subsampling: $SUBS" ; exit 2 ;;
  esac

  # Note: JPEG decodes to full-range (yuvj*). We preserve full-range signaling.
  # -coder 1 enables range coder (FFV1 v3 default); -g 1 intra-only; -slicecrc improves resiliency.
  ffmpeg -hide_banner -loglevel error -y \
    -framerate "$FPS_R" \
    -i "$JPEG_DIR/frame_%06d.jpg" \
    -pix_fmt "$PIXFMT" -color_range pc \
    -c:v ffv1 -level 3 -g 1 -slices 16 -slicecrc 1 -coder 1 \
    -map_metadata -1 -an \
    "$OUTPUT"

  echo "MKV/FFV1 written to: $OUTPUT"

elif [[ "$ext" == "zip" ]]; then
  # Byte-identical preservation of the JPEGs: store them in a zip (no recompression).
  ( cd "$JPEG_DIR" && zip -q -0 -r "$OUTPUT" . )
  echo "ZIP (store) written to: $OUTPUT"
  echo "Note: ZIP is not a video container; it preserves the exact JPEG bitstreams."

else
  echo "Unsupported output extension: .$ext"
  echo "Use .mkv for lossless playable video (FFV1) or .zip for exact JPEG bitstreams."
  exit 3
fi

echo "Done."
echo "Temporary working directory kept at: $WORKDIR"
echo "Subfolders:"
echo "  - $PPM_DIR  (per-frame PPM)"
echo "  - $JPEG_DIR (per-frame JPEG from jpegturbo)"
