import os
import subprocess
import time
import shutil
import re
from os import cpu_count

import psutil
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from benchmark_common import get_compression_metrics
from vmaf_torch import VMAF
from vmaf_torch.utils import yuv_to_tensor

# Set plot style
sns.set_theme(style="whitegrid")
#%%
# --- Main Configuration ---

# Number of times to run encode/decode for each parameter set to average the time
N_ITERATIONS = 10

# Resolutions to test (Name: (Width, Height))
RESOLUTIONS = {
    '1280x720': (1280, 720),
}

RESOLUTION_TO_FILENAME = {
    '1280x720': 'frame_hd.rgb',
}

# Parameters to benchmark
QUALITIES = range(10, 101, 1)
SPEEDS = [10]
SUBSAMPLINGS = ['4:2:0']
THREADS = [1, cpu_count()/2, cpu_count()]
ENCODERS = ['rav1e']
DECODERS = ['dav1d', 'gav1']

SUBSAMPLING_TO_PARAM = {
    '4:2:0': '420'
}

# --- File and Directory Setup ---
# Create directories to hold our generated files
os.makedirs('raw_images', exist_ok=True)
os.makedirs('temp_files', exist_ok=True)
#%%
# Create directories to hold our generated files
os.makedirs('raw_images', exist_ok=True)
raw_imgs_path = "/home/simone/Documenti/video-compression-latency-benchmarks/1_video_material"
os.makedirs('temp_files', exist_ok=True)
#%%
results = []
total_combinations = len(RESOLUTIONS) * len(QUALITIES) * len(SPEEDS) * len(SUBSAMPLINGS) * len(THREADS) * len(ENCODERS) + len(RESOLUTIONS) * len(QUALITIES) * len(SPEEDS) * len(SUBSAMPLINGS) * len(THREADS) * len(DECODERS)

# Since other than AOM, there are no codecs that support both encoding and decoding, the benchmark works as follows:
# The encoders encode the image, and it gets decoded by AOM (since it's the reference one)
# AOM encodes the image (since it's the reference one), and it gets decoded by the decoders
# For this reason, we only measure the generic processing time. Difference in PSNR for encoders and decoders implicitly include AOM's errors

progress = 1
for encoder in ENCODERS:
    for res_name, (w, h) in RESOLUTIONS.items():
        original_rgb_path = os.path.join(raw_imgs_path, RESOLUTION_TO_FILENAME[res_name])

        # Read the entire source image into memory once per resolution
        with open(original_rgb_path, 'rb') as f:
            original_rgb_data = f.read()

        for quality in QUALITIES:
            for speed in SPEEDS:
                for subsample in SUBSAMPLINGS:
                    for threadnum in THREADS:
                        # --- Encoding ---
                        # 1. Benchmark Encoding in memory
                        enc_cmd = [
                            '/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/libavif/cmake-build-debug/libavif_encode',
                            '-w', f'{w}',
                            '-h', f'{h}',
                            '-s', SUBSAMPLING_TO_PARAM[subsample],
                            '-q', f'{quality}',
                            '-f', f'{speed}',
                            '-c', f'{encoder}',
                            '-t', f'{threadnum}',
                            '-i', f'{N_ITERATIONS}',
                            '-b']
                        enc_proc = subprocess.run(enc_cmd, input=original_rgb_data, capture_output=True, check=True)
                        enc_timings = enc_proc.stdout.decode()

                        # 2. Obtain encoded image
                        enc_cmd = [
                            '/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/libavif/cmake-build-debug/libavif_encode',
                            '-w', f'{w}',
                            '-h', f'{h}',
                            '-s', SUBSAMPLING_TO_PARAM[subsample],
                            '-q', f'{quality}',
                            '-f', f'{speed}',
                            '-c', f'{encoder}',
                            '-t', f'{threadnum}',
                            '-i', '1',
                            '-o', '-']
                        enc_proc = subprocess.run(enc_cmd, input=original_rgb_data, capture_output=True, check=True)
                        compressed_jpg_data = enc_proc.stdout
                        compressed_size = len(compressed_jpg_data)

                        # --- Decoding ---
                        # Obtain decoded image
                        dec_cmd = [
                            '/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/libavif/cmake-build-debug/libavif_decode',
                            '-c', 'aom',
                            '-t', f'{threadnum}',
                            '-i', "1",
                            '-o', '-']
                        dec_proc = subprocess.run(dec_cmd, input=compressed_jpg_data, capture_output=True, check=True)
                        decompressed_rgb_data = dec_proc.stdout

                        # --- Metrics extraction ---
                        enc_setup_time = enc_timings.splitlines()[0].split(":")[1].strip()
                        avg_encode_time = float(enc_timings.splitlines()[1].split(":")[1].strip()) / N_ITERATIONS
                        enc_cleanup_time = enc_timings.splitlines()[2].split(":")[1].strip()

                        psnr, ssim = get_compression_metrics(original_rgb_data, decompressed_rgb_data, width=w, height=h)

                        enc_str = ""
                        for s in enc_cmd:
                            enc_str += s + " "

                        # --- Results recording ---
                        results.append({
                            'Codec': encoder,
                            'Resolution': res_name,
                            'Quality': quality,
                            'Speed': speed,
                            'Threads': threadnum,
                            'Subsampling': subsample,
                            'Iterations': N_ITERATIONS,
                            'Avg Processing Time (s)': avg_encode_time,
                            'Setup Time (s)': enc_setup_time,
                            'Cleanup Time (s)': enc_cleanup_time,
                            'Compressed Size (KB)': compressed_size / 1000,
                            'PSNR': psnr,
                            'SSIM': ssim,
                            'CMD': enc_str,
                        })
                        print(f'Combination {progress}/{total_combinations}')
                        progress += 1

for decoder in DECODERS:
    for res_name, (w, h) in RESOLUTIONS.items():
        original_rgb_path = os.path.join(raw_imgs_path, RESOLUTION_TO_FILENAME[res_name])

        # Read the entire source image into memory once per resolution
        with open(original_rgb_path, 'rb') as f:
            original_rgb_data = f.read()

        for quality in QUALITIES:
            for speed in SPEEDS:
                for subsample in SUBSAMPLINGS:
                    for threadnum in THREADS:
                        # --- Encoding ---
                        # Obtain encoded image
                        enc_cmd = [
                            '/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/libavif/cmake-build-debug/libavif_encode',
                            '-w', f'{w}',
                            '-h', f'{h}',
                            '-s', SUBSAMPLING_TO_PARAM[subsample],
                            '-q', f'{quality}',
                            '-f', f'{speed}',
                            '-c', 'aom',
                            '-t', f'{threadnum}',
                            '-i', '1',
                            '-o', '-']
                        enc_proc = subprocess.run(enc_cmd, input=original_rgb_data, capture_output=True, check=True)
                        compressed_jpg_data = enc_proc.stdout
                        compressed_size = len(compressed_jpg_data)

                        # --- Decoding ---
                        # 1. Benchmark Decoding in memory
                        dec_cmd = [
                            '/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/libavif/cmake-build-debug/libavif_decode',
                            '-c', f'{decoder}',
                            '-t', f'{threadnum}',
                            '-i', f'{N_ITERATIONS}',
                            '-b']
                        dec_proc = subprocess.run(dec_cmd, input=compressed_jpg_data, capture_output=True, check=True)
                        dec_timings = dec_proc.stdout.decode()

                        # 2. Obtain decoded image
                        dec_cmd = [
                            '/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/libavif/cmake-build-debug/libavif_decode',
                            '-c', f'{decoder}',
                            '-t', f'{threadnum}',
                            '-i', "1",
                            '-o', '-']
                        dec_proc = subprocess.run(dec_cmd, input=compressed_jpg_data, capture_output=True, check=True)
                        decompressed_rgb_data = dec_proc.stdout

                        # --- Metrics extraction ---
                        dec_setup_time = dec_timings.splitlines()[0].split(":")[1].strip()
                        avg_decode_time = float(dec_timings.splitlines()[1].split(":")[1].strip()) / N_ITERATIONS
                        dec_cleanup_time = dec_timings.splitlines()[2].split(":")[1].strip()

                        psnr, ssim = get_compression_metrics(original_rgb_data, decompressed_rgb_data, width=w, height=h)

                        dec_str = ""
                        for s in dec_cmd:
                            dec_str += s + " "

                        # --- Results recording ---
                        results.append({
                            'Codec': decoder,
                            'Resolution': res_name,
                            'Quality': quality,
                            'Speed': speed,
                            'Threads': threadnum,
                            'Subsampling': subsample,
                            'Iterations': N_ITERATIONS,
                            'Avg Processing Time (s)': avg_decode_time,
                            'Setup Time (s)': dec_setup_time,
                            'Cleanup Time (s)': dec_cleanup_time,
                            'Compressed Size (KB)': compressed_size / 1000,
                            'PSNR': psnr,
                            'SSIM': ssim,
                            'CMD': dec_str,
                        })
                        print(f'Combination {progress}/{total_combinations}')
                        progress += 1

# Create and save the DataFrame
df = pd.DataFrame(results)
df.to_csv('results_libavif_LoLa.csv', index=False)
print("\n✅ Benchmarking complete! Results saved to 'results_libavif_LoLa.csv'")
df.head()
