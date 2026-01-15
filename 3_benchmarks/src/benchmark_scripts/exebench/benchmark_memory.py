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

# Resolutions to test (Name: (Width, Height))
RESOLUTIONS = {
    '1280x720': (1280, 720),
    '1920x1080': (1920, 1080),
    '3840x2160': (3840, 2160),
}

# The average jpeg size around quality=90 for the various resolutions
RESOLUTIONS_TO_COMPRESSED = {
    '1280x720': 200000,
    '1920x1080': 500000,
    '3840x2160': 2000000,
}

results = []
progress = 1
for res_name, (w, h) in RESOLUTIONS.items():
    # --- RGB Upload ---
    up_RGB_cmd = [
        '/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/gpubandwidth/transfer_bench',
        '--bytes', f'{w*h}',
        '--direction', 'upload']
    up_RGB_proc = subprocess.run(up_RGB_cmd, capture_output=True, check=True)
    up_RGB_timing = up_RGB_proc.stdout.decode()

    # --- RGB Download ---
    down_RGB_cmd = [
        '/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/gpubandwidth/transfer_bench',
        '--bytes', f'{w*h}',
        '--direction', 'download']
    down_RGB_proc = subprocess.run(down_RGB_cmd, capture_output=True, check=True)
    down_RGB_timing = down_RGB_proc.stdout.decode()

    # --- JPEG Upload ---
    up_JPEG_cmd = [
        '/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/gpubandwidth/transfer_bench',
        '--bytes', f'{RESOLUTIONS_TO_COMPRESSED[res_name]}',
        '--direction', 'upload']
    up_JPEG_proc = subprocess.run(up_JPEG_cmd, capture_output=True, check=True)
    up_JPEG_timing = up_JPEG_proc.stdout.decode()

    # --- JPEG Download ---
    down_JPEG_cmd = [
        '/home/simone/Documenti/video-compression-latency-benchmarks/3_benchmarks/src/gpubandwidth/transfer_bench',
        '--bytes', f'{RESOLUTIONS_TO_COMPRESSED[res_name]}',
        '--direction', 'download']
    down_JPEG_proc = subprocess.run(down_JPEG_cmd, capture_output=True, check=True)
    down_JPEG_timing = down_JPEG_proc.stdout.decode()

    # --- Results recording ---
    results.append({
        'Resolution': res_name,
        'Avg RGB Upload Time (ms)': up_RGB_timing,
        'Avg RGB Download Time (ms)': down_RGB_timing,
        'Avg JPEG Upload Time (ms)': up_JPEG_timing,
        'Avg JPEG Download Time (ms)': down_JPEG_timing,
    })
    progress += 1
# Create and save the DataFrame
df = pd.DataFrame(results)
df.to_csv('results_memory.csv', index=False)
print("\n✅ Benchmarking complete! Results saved to 'results_memory.csv'")
df.head()
