from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import yuvio
from PIL import Image

def rgb_to_array(rgb24_img, width, height):
    # Convert bytes to a PIL Image
    image = Image.frombytes('RGB', (width, height), rgb24_img)
    # Convert PIL Image to NumPy array
    return np.array(image)

def yuv_to_array(yuv_img, width, height):
    # Convert bytes to a PIL Image
    image = Image.frombytes('YCbCr', (width, height), yuv_img)
    # Convert PIL Image to NumPy array
    return np.array(image)

def get_compression_metrics(raw_img, comp_img, width, height):
    original_img = rgb_to_array(raw_img, width, height)
    decompressed_img = rgb_to_array(comp_img, width, height)

    # --- PSNR and SSIM ---
    psnr = peak_signal_noise_ratio(original_img, decompressed_img, data_range=255)
    # Explicitly set win_size to avoid error with smaller resolutions
    win_size = min(original_img.shape[0], original_img.shape[1], 7)  # Use a small odd number, 7 is typical
    ssim = structural_similarity(original_img, decompressed_img, win_size=win_size, channel_axis=2, data_range=255)

    # --- VMAF ---
    vmaf = 0.0
    #vmaf_cmd = [
    #    '/home/simone/Documenti/video-compression-latency-benchmarks/2_libraries/_installdir/libvmaf/bin/vmaf',
    #    '-q',
    #    '-r', original_path,
    #    '-d', decompressed_path,
    #    '-w', width,
    #    '-h', height,
    #    '-p', subsampling,
    #    '-b', '8'
    #]
    #try:
    #    # Capture stdout and stderr for debugging
    #    result = subprocess.run(vmaf_cmd, check=True, capture_output=True, text=True, timeout=30)
    #    with open(vmaf_log, 'r') as f:
    #        log_content = f.read()
    #    vmaf_match = re.search(r'<vmaf version=".*">(\d+\.\d+)</vmaf>', log_content)
    #    vmaf = float(vmaf_match.group(1)) if vmaf_match else 0.0
    #except subprocess.CalledProcessError as e:
    #    print(f"⚠️ ffmpeg failed with exit code {e.returncode}. stderr:\n{e.stderr}")
    #    print(f"⚠️ Could not calculate VMAF for {os.path.basename(original_path)}: {e}")
    #except Exception as e:
    #    print(f"⚠️ Could not calculate VMAF for {os.path.basename(original_path)}: {e}")
    return psnr, ssim, vmaf