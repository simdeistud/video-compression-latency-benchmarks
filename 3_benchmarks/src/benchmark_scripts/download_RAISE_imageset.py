#!/usr/bin/env python3
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import rawpy
from PIL import Image

# =========================
# HARD-CODED PARAMETERS
# =========================
INPUT_CSV = "/home/simone/Scaricati/RAISE_370.csv"   # <-- percorso CSV
OUTDIR = "/home/simone/Documenti/video-compression-latency-benchmarks/1_video_material/RAISE"            # <-- directory output
NEF_FIELD = "NEF"               # colonna con URL http/https alla NEF
CSV_SEP = ","                   # ";" se necessario
CSV_ENCODING = None             # es. "utf-8" o "iso-8859-1"; None = autodetect
CSV_ENCODING_ERRORS = "strict"  # o "ignore"/"replace"
TIMEOUT_CONNECT = 15.0
TIMEOUT_READ = 120.0
# =========================

SIZES = {
    "hd":     (1280, 720),
    "fullhd": (1920, 1080),
    "ultrahd": (3840, 2160),
}

def download_bytes(url: str, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ)) -> bytes:
    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    return resp.content

def nef_to_rgb_array(nef_bytes: bytes) -> np.ndarray:
    """Decodifica NEF in RGB uint8 (RGB24), senza salvare il RAW su disco."""
    with rawpy.imread(io.BytesIO(nef_bytes)) as raw:
        rgb = raw.postprocess(
            output_bps=8,           # 8 bpc -> RGB24
            no_auto_bright=True,    # evita guadagno globale
            gamma=(1.0, 1.0),       # lineare; rimuovi per curva sRGB di default
            use_camera_wb=True,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8, copy=False)
    return np.ascontiguousarray(rgb)  # HxWx3, uint8

def center_crop_to_aspect(img_np: np.ndarray, target_aspect: float) -> np.ndarray:
    """Crop centrato al rapporto dato (es. 16/9) mantenendo area massima."""
    h, w, c = img_np.shape
    assert c == 3
    current_aspect = w / h
    if current_aspect > target_aspect:
        # troppo largo: riduci larghezza
        new_w = int(round(h * target_aspect))
        new_h = h
    else:
        # troppo alto: riduci altezza
        new_w = w
        new_h = int(round(w / target_aspect))
    x0 = (w - new_w) // 2
    y0 = (h - new_h) // 2
    return img_np[y0:y0 + new_h, x0:x0 + new_w, :]

def pil_from_np_rgb(img_np: np.ndarray) -> Image.Image:
    return Image.fromarray(img_np, mode="RGB")

def resize_rgb(img_pil: Image.Image, wh: tuple[int, int]) -> Image.Image:
    return img_pil.resize(wh, resample=Image.Resampling.LANCZOS)

def save_rgb24_raw(img_pil: Image.Image, path: Path) -> None:
    """Salva bytes RGB24 raw (interleaved, row-major) senza header."""
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(img_pil.tobytes())

def process_row(idx: int, url: str, outdir: Path) -> None:
    nef_bytes = download_bytes(url)
    rgb = nef_to_rgb_array(nef_bytes)

    # Crop 16:9 centrato (massimo rettangolo)
    cropped = center_crop_to_aspect(rgb, target_aspect=16/9)

    # PIL per resize
    pil_img = pil_from_np_rgb(cropped)

    for tag, (tw, th) in SIZES.items():
        resized = resize_rgb(pil_img, (tw, th))
        out_path = outdir / f"{tag}_{idx}.rgb"   # nome_<indice>.rgb
        save_rgb24_raw(resized, out_path)

def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # Caricamento CSV
    try:
        df = pd.read_csv(INPUT_CSV, sep=CSV_SEP, encoding=CSV_ENCODING, encoding_errors=CSV_ENCODING_ERRORS)
    except TypeError:
        # pandas < 2.0: nessun encoding_errors
        df = pd.read_csv(INPUT_CSV, sep=CSV_SEP, encoding=CSV_ENCODING)

    if NEF_FIELD not in df.columns:
        print(f'ERROR: CSV missing required column "{NEF_FIELD}". Columns: {list(df.columns)}', file=sys.stderr)
        sys.exit(1)

    for idx, row in df.iterrows():
        try:
            url = str(row[NEF_FIELD]).strip()
            if not url.lower().startswith(("http://", "https://")):
                raise ValueError(f"Invalid NEF URL: {url}")
            process_row(idx, url, outdir)
            print(f"[OK] row {idx}: wrote hd_{idx}.rgb, fullhd_{idx}.rgb, ultrahd_{idx}.rgb")
        except Exception as e:
            print(f"[FAIL] row {idx}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()