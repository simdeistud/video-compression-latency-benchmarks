from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import yuvio
import fast_ssim as m
from PIL import Image

def rgb_to_array(rgb24_img, width, height):
    # Convert bytes to a PIL Image
    image = Image.frombytes('RGB', (width, height), rgb24_img)
    # Convert PIL Image to NumPy array
    return np.array(image)

def get_compression_metrics(raw_rgb24img, comp_rgb24img, width, height):
    original_img = rgb_to_array(raw_rgb24img, width, height)
    decompressed_img = rgb_to_array(comp_rgb24img, width, height)
    # --- PSNR and SSIM ---
    #psnr = peak_signal_noise_ratio(original_img, decompressed_img, data_range=255)
    psnr_fast = m.psnr(original_img, decompressed_img, data_range=255)
    # Explicitly set win_size to avoid error with smaller resolutions
    #win_size = min(original_img.shape[0], original_img.shape[1], 7)  # Use a small odd number, 7 is typical
    #ssim = structural_similarity(original_img, decompressed_img, win_size=win_size, channel_axis=2, data_range=255)
    ssim_fast = m.ssim(original_img, decompressed_img, data_range=255)
    return psnr_fast, ssim_fast