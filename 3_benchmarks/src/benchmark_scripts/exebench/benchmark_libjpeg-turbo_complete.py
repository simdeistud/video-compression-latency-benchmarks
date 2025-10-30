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
#%%
# --- Main Configuration ---

# Number of times to run encode/decode for each parameter set to average the time
N_ITERATIONS = 10 # 100 is a good number of iterations, wrt to 1000 it changes almost nothing

# Resolutions to test (Name: (Width, Height))
RESOLUTIONS = {
    '1280x720': (1280, 720),
    '1920x1080': (1920, 1080),
    '3840x2160': (3840, 2160),
}

RESOLUTION_TO_FILENAME = {
    '1280x720': 'frame_hd.rgb',
    '1920x1080': 'frame_fullhd.rgb',
    '3840x2160': 'frame_ultrahd.rgb',
}

# Parameters to benchmark
QUALITIES = range(10, 101, 10)
DCT_METHODS = ['int', 'fast', 'float']
SUBSAMPLINGS = ['4:4:4', '4:2:2', '4:2:0'] # For end user viewing purposes, 444 and 422 make no sense
RESTART_INTERVALS = [0, 1, 2, 4, 8, 16, 32, 64, 128]

SUBSAMPLING_TO_PARAM = {
    '4:4:4': '444',
    '4:2:2': '422',
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
total_combinations = len(RESOLUTIONS) * len(QUALITIES) * len(DCT_METHODS) * len(SUBSAMPLINGS) * len(RESTART_INTERVALS)

progress = 1
for res_name, (w, h) in RESOLUTIONS.items():
    original_rgb_path = os.path.join(raw_imgs_path, RESOLUTION_TO_FILENAME[res_name])

    # Read the entire source image into memory once per resolution
    with open(original_rgb_path, 'rb') as f:
        original_rgb_data = f.read()

    for quality in QUALITIES:
        for dct in DCT_METHODS:
            for subsample in SUBSAMPLINGS:
                for restart_interval in RESTART_INTERVALS:
                    # --- Encoding ---
                    # 1. Benchmark Encoding in memory
                    enc_cmd = [
                        '/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/libjpeg-turbo/cmake-build-debug/libjpeg-turbo_encode',
                        '-w', f'{w}',
                        '-h', f'{h}',
                        '-s', SUBSAMPLING_TO_PARAM[subsample],
                        '-d', dct,
                        '-q', str(quality),
                        '-r', f'{restart_interval}',
                        '-i', f'{N_ITERATIONS}',
                        '-b']
                    enc_proc = subprocess.run(enc_cmd, input=original_rgb_data, capture_output=True, check=True)
                    enc_timings = enc_proc.stdout.decode()

                    # 2. Obtain encoded image
                    enc_cmd = [
                        '/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/libjpeg-turbo/cmake-build-debug/libjpeg-turbo_encode',
                        '-w', f'{w}',
                        '-h', f'{h}',
                        '-s', SUBSAMPLING_TO_PARAM[subsample],
                        '-d', dct,
                        '-q', str(quality),
                        '-r', f'{restart_interval}',
                        '-i', '1',
                        '-o', '-']
                    enc_proc = subprocess.run(enc_cmd, input=original_rgb_data, capture_output=True, check=True)
                    compressed_jpg_data = enc_proc.stdout
                    compressed_size = len(compressed_jpg_data)

                    # --- Decoding ---
                    # 1. Benchmark Decoding in memory
                    dec_cmd = [
                        '/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/libjpeg-turbo/cmake-build-debug/libjpeg-turbo_decode',
                        '-d', dct,
                        '-i', f'{N_ITERATIONS}',
                        '-b']
                    dec_proc = subprocess.run(dec_cmd, input=compressed_jpg_data, capture_output=True, check=True)
                    dec_timings = dec_proc.stdout.decode()

                    # 2. Obtain decoded image
                    dec_cmd = [
                        '/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/libjpeg-turbo/cmake-build-debug/libjpeg-turbo_decode',
                        '-d', dct,
                        '-i', "1",
                        '-o', '-']
                    dec_proc = subprocess.run(dec_cmd, input=compressed_jpg_data, capture_output=True, check=True)
                    decompressed_rgb_data = dec_proc.stdout

                    # --- Metrics extraction ---
                    enc_setup_time = enc_timings.splitlines()[0].split(":")[1].strip()
                    avg_encode_time = float(enc_timings.splitlines()[1].split(":")[1].strip()) / N_ITERATIONS
                    enc_cleanup_time = enc_timings.splitlines()[2].split(":")[1].strip()
                    dec_setup_time = dec_timings.splitlines()[0].split(":")[1].strip()
                    avg_decode_time = float(dec_timings.splitlines()[1].split(":")[1].strip()) / N_ITERATIONS
                    dec_cleanup_time = dec_timings.splitlines()[2].split(":")[1].strip()

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
                        'DCT': dct,
                        'Restart Interval': restart_interval,
                        'Subsampling': subsample,
                        'Iterations': N_ITERATIONS,
                        'Avg Encode Time (s)': avg_encode_time,
                        'Encoder Setup Time (s)': enc_setup_time,
                        'Encoder Cleanup Time (s)': enc_cleanup_time,
                        'Avg Decode Time (s)': avg_decode_time,
                        'Decoder Setup Time (s)': dec_setup_time,
                        'Decoder Cleanup Time (s)': dec_cleanup_time,
                        'Compressed Size (KB)': compressed_size / 1000,
                        'PSNR': psnr,
                        'SSIM': ssim,
                        'Encode CMD': enc_str,
                        'Decode CMD': dec_str,
                    })
                    print(f'Combination {progress}/{total_combinations}')
                    progress += 1
# Create and save the DataFrame
df = pd.DataFrame(results)
df.to_csv('results_libjpeg-turbo_complete.csv', index=False)
print("\n✅ Benchmarking complete! Results saved to 'results_libjpeg-turbo_complete.csv'")
df.head()
