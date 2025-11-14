from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import yuvio
from PIL import Image
from vmaf_torch import VMAF
from vmaf_torch.utils import yuv_to_tensor

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

    return psnr, ssim