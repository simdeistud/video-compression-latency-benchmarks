import os
import subprocess
import time
import shutil
import re
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
# %%
# --- Main Configuration ---

# Number of times to run encode/decode for each parameter set to average the time
N_ITERATIONS = 100

# Resolutions to test (Name: (Width, Height))
RESOLUTIONS = {
    '1280x720': (1280, 720),
    '1920x1080': (1920, 1080),
    '3840x2160': (3840, 2160),
}

DATASET_SIZE = 1000

# Parameters to benchmark
QUALITIES = range(40, 96, 1)
SUBSAMPLINGS = ['4:4:4']

SUBSAMPLING_TO_PARAM = {
    '4:4:4': '444'
}

# --- File and Directory Setup ---
# Create directories to hold our generated files
os.makedirs('raw_images', exist_ok=True)
os.makedirs('temp_files', exist_ok=True)
# %%
# Create directories to hold our generated files
os.makedirs('raw_images', exist_ok=True)
raw_imgs_path = "/home/simone/Documenti/video-compression-latency-benchmarks/1_video_material/RAISE"
os.makedirs('temp_files', exist_ok=True)
# %%
results = []
total_combinations = len(RESOLUTIONS) * len(QUALITIES) * len(SUBSAMPLINGS)

for frame in range(1, 101, 1):
    progress = 1
    RESOLUTION_TO_FILENAME = {
        '1280x720': f'hd_{frame}.rgb',
        '1920x1080': f'fullhd_{frame}.rgb',
        '3840x2160': f'ultrahd_{frame}.rgb',
    }
    for res_name, (w, h) in RESOLUTIONS.items():
        original_rgb_path = os.path.join(raw_imgs_path, RESOLUTION_TO_FILENAME[res_name])

        # Read the entire source image into memory once per resolution
        with open(original_rgb_path, 'rb') as f:
            original_rgb_data = f.read()

        for quality in QUALITIES:
            for subsample in SUBSAMPLINGS:
                # --- Encoding ---
                # 1. Benchmark Encoding in memory
                enc_cmd = [
                    '/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/nvjpeg2k/cmake-build-release/nvjpeg2k_encode',
                    '-w', f'{w}',
                    '-h', f'{h}',
                    '-q', str(quality),
                    '-i', f'{N_ITERATIONS}',
                    '-b',
                    '-o', '/dev/null']
                enc_proc = subprocess.run(enc_cmd, input=original_rgb_data, capture_output=True, check=True)
                enc_timings = enc_proc.stdout.decode()

                # 2. Obtain encoded image
                enc_cmd = [
                    '/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/nvjpeg2k/cmake-build-release/nvjpeg2k_encode',
                    '-w', f'{w}',
                    '-h', f'{h}',
                    '-q', str(quality),
                    '-i', '1',
                    '-o', '-']
                enc_proc = subprocess.run(enc_cmd, input=original_rgb_data, capture_output=True, check=True)
                compressed_jpg_data = enc_proc.stdout
                compressed_size = len(compressed_jpg_data)

                # --- Decoding ---
                # 1. Benchmark Decoding in memory
                dec_cmd = [
                    '/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/nvjpeg2k/cmake-build-release/nvjpeg2k_decode',
                    '-i', f'{N_ITERATIONS}',
                    '-b',
                    '-o', '/dev/null']
                dec_proc = subprocess.run(dec_cmd, input=compressed_jpg_data, capture_output=True, check=True)
                dec_timings = dec_proc.stdout.decode()

                # 2. Obtain decoded image
                dec_cmd = [
                    '/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/nvjpeg2k/cmake-build-release/nvjpeg2k_decode',
                    '-i', "1",
                    '-o', '-']
                dec_proc = subprocess.run(dec_cmd, input=compressed_jpg_data, capture_output=True, check=True)
                decompressed_rgb_data = dec_proc.stdout

                # --- Metrics extraction ---
                avg_encode_time = float(enc_timings.strip()) / N_ITERATIONS
                avg_decode_time = float(dec_timings.strip()) / N_ITERATIONS

                psnr, ssim = get_compression_metrics(original_rgb_data, decompressed_rgb_data, width=w, height=h)

                enc_str = ""
                for s in enc_cmd:
                    enc_str += s + " "
                dec_str = ""
                for s in dec_cmd:
                    dec_str += s + " "

                # --- Results recording ---
                results.append({
                    'Resolution': res_name,
                    'Quality': quality,
                    'Subsampling': subsample,
                    'Iterations': N_ITERATIONS,
                    'Avg Encode Time (s)': avg_encode_time,
                    'Avg Decode Time (s)': avg_decode_time,
                    'Compressed Size (KB)': compressed_size / 1000,
                    'PSNR': psnr,
                    'SSIM': ssim,
                    'Encode CMD': enc_str,
                    'Decode CMD': dec_str,
                    'Frame' : frame,
                })
                print(f'Combination {progress}/{total_combinations} - {frame}')
                progress += 1
# Create and save the DataFrame
df = pd.DataFrame(results)
df.to_csv('results_nvjpeg2k_LoLa.csv', index=False)
print("\n✅ Benchmarking complete! Results saved to 'results_nvjpeg2k_LoLa.csv'")
df.head()
