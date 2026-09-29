# -*- coding: utf-8 -*-
"""
Calibration for mixed NEF (flats / dark-flats) + FITS (darks / lights),
with optional Bayer-first or demosaic-first stacking, alignment, and RGB/BW output.

Optionally, the program can plate solve only the final stacked FITS product and
display the annotated solution in a Matplotlib window.

Folder example (with --src-root Data):
Data/
    Flat/       # NEF flats
    Dark_Flat/  # NEF dark flats
    Dark/       # FITS darks
    Light/      # FITS lights

Key features
-----------
- Flats & Dark-Flats: NEF (Bayer, read via rawpy).
- Lights & Darks: FITS (Bayer).
- Two stack modes:
  * demosaic-first:
    - Debayer (and optionally collapse to B&W) early.
    - Masters are debayered (mono or RGB).
    - Lights are calibrated in debayered space.
    - Alignment and stacking in debayered space.
  * bayer-first:
    - All frames remain mosaiced (2D Bayer) for masters, calibration, and stacking.
    - Only the final stack is demosaiced to RGB/BW.
    - Also saves the stacked Bayer frame.

- No APS-C crop by default (full frame).
- If flat masters (NEF) are smaller than FITS lights/darks, those are
  center-cropped to flat size (e.g., FITS 7378x4924 -> NEF 7360x4912).
- "Keep X% negatives" subtraction rule.
- Flat coefficient: sigma-clipped median normalization to 1, low-side clamp only.
- Saves both calibrated-unaligned and calibrated-aligned lights + final stacks.
"""

from __future__ import annotations

import argparse
import math
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Tuple

import astroalign as aa
import cv2
import numpy as np
import rawpy
from astropy.io import fits
from astropy.stats import sigma_clip
from tqdm import tqdm

# ------------------
# Constants & Maps
# ------------------

_PATTERNS: Tuple[str, ...] = ("RGGB", "BGGR", "GRBG", "GBRG")
_PAT2PREFIX = {"RGGB": "RG", "BGGR": "BG", "GRBG": "GR", "GBRG": "GB"}

# Effective CFA after a 90° CW rotation (frame.T[::-1, :])
_ROT90CW = {"RGGB": "GRBG", "GRBG": "GBRG", "GBRG": "BGGR", "BGGR": "RGGB"}

OutputColor = Literal["bw", "rgb"]


# ------------------
# Config dataclasses
# ------------------

@dataclass
class FlatClampConfig:
    # Low-side clamp only; high-side intentionally omitted.
    clip_low_k: float = 10.0
    min_coeff: float = 1e-6
    use_mad: bool = True
    pct_low: Optional[float] = None  # e.g., 0.1 (%)


@dataclass
class PipelineConfig:
    sigma_clip_sigma: float = 3.0
    sigma_clip_maxiters: int = 5
    keep_neg_frac: float = 0.01
    per_channel_shift: bool = True
    rotate_portraits: bool = True
    compressed_fits: bool = False

    # Output options
    output_color: OutputColor = "bw"
    bayer_pattern: Optional[str] = "AUTO"

    # Optional APS-C crop (disabled by default)
    crop_to_apsc: bool = False
    apsc_crop_factor: float = 1.5

    clamp: Optional[FlatClampConfig] = None


# ------------------
# Small utilities
# ------------------

def _safe_name(p: Path) -> str:
    return p.name.replace(" ", "_")


def _utcnow_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_ASCII_MAP = {
    "±": "+/-", "σ": "sigma", "°": "deg", "×": "x", "µ": "u",
    "–": "-", "—": "-", "…": "...", "’": "'", "“": '"', "”": '"',
}


def _fits_ascii(s: str) -> str:
    for k, v in _ASCII_MAP.items():
        s = s.replace(k, v)
    return s.encode("ascii", "ignore").decode("ascii")


def _ensure_float32(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return float(1.4826 * mad)


def _extract_exptime(fpath: Path) -> Optional[float]:
    """Try to read EXPTIME/EXPOSURE from FITS; NEF will just return None."""
    keys = ("EXPTIME", "EXPOSURE", "TIME-EX", "ITIME")

    def _try(ext: int) -> Optional[float]:
        try:
            hdr = fits.getheader(str(fpath), ext=ext)
        except Exception:
            return None
        for k in keys:
            if k in hdr:
                try:
                    val = float(hdr[k])
                    if math.isfinite(val):
                        return val
                except Exception:
                    pass
        return None

    try:
        return _try(0) or _try(1)
    except Exception:
        return None


def center_crop_to_shape(arr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Center-crop arr to target_shape. Supports (H,W) and (3,H,W)/(H,W,3).
    Used to match FITS masters/lights to NEF-based flat size.
    """
    target_shape = tuple(target_shape)
    if arr.shape == target_shape:
        return arr

    if arr.ndim == 2 and len(target_shape) == 2:
        H, W = arr.shape
        h, w = target_shape
        if h > H or w > W:
            raise ValueError(f"Target shape {target_shape} larger than source {arr.shape}")
        y0 = (H - h) // 2
        x0 = (W - w) // 2
        return arr[y0:y0 + h, x0:x0 + w]

    if arr.ndim == 3 and len(target_shape) == 3:
        # (3,H,W)
        if arr.shape[0] == 3 and target_shape[0] == 3:
            _, H, W = arr.shape
            _, h, w = target_shape
            if h > H or w > W:
                raise ValueError(f"Target shape {target_shape} larger than source {arr.shape}")
            y0 = (H - h) // 2
            x0 = (W - w) // 2
            return arr[:, y0:y0 + h, x0:x0 + w]
        # (H,W,3)
        if arr.shape[-1] == 3 and target_shape[-1] == 3:
            H, W, _ = arr.shape
            h, w, _ = target_shape
            if h > H or w > W:
                raise ValueError(f"Target shape {target_shape} larger than source {arr.shape}")
            y0 = (H - h) // 2
            x0 = (W - w) // 2
            return arr[y0:y0 + h, x0:x0 + w, :]

    raise ValueError(f"Incompatible shapes: arr {arr.shape}, target {target_shape}")


# ------------------
# Bayer / demosaic
# ------------------

def _get_bayer_code(pattern: str):
    pattern = pattern.upper()
    if pattern not in _PATTERNS:
        raise ValueError(f"Unsupported Bayer pattern '{pattern}'")
    pat2 = _PAT2PREFIX[pattern]
    names_rgb = [
        f"COLOR_BAYER_{pat2}2RGB_EA",
        f"COLOR_BAYER_{pat2}2RGB",
    ]
    names_bgr = [
        f"COLOR_BAYER_{pat2}2BGR_EA",
        f"COLOR_BAYER_{pat2}2BGR",
    ]
    for nm in names_rgb:
        if hasattr(cv2, nm):
            return getattr(cv2, nm), (lambda x: x)
    for nm in names_bgr:
        if hasattr(cv2, nm):
            return getattr(cv2, nm), (lambda x: x[..., ::-1])  # BGR -> RGB
    raise RuntimeError(f"OpenCV lacks Bayer conversion codes for {pattern}")


def _demosaic_rgb(image_2d: np.ndarray, pattern: str) -> np.ndarray:
    """Demosaic 2D mosaiced image to RGB float32 (no normalization)."""
    if image_2d.dtype != np.uint16:
        img16 = np.clip(image_2d, 0, 65535).astype(np.uint16)
    else:
        img16 = image_2d
    code, to_rgb = _get_bayer_code(pattern)
    out = cv2.cvtColor(img16, code)
    out = to_rgb(out)
    return out.astype(np.float32)


def _auto_detect_cfa(image_2d: np.ndarray) -> str:
    """Pick Bayer pattern with lowest chroma high-frequency energy."""
    best_pat = "RGGB"
    best_score = float("inf")
    for pat in _PATTERNS:
        try:
            rgb = _demosaic_rgb(image_2d, pat)
            r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            cr = r - g
            cb = b - g
            lap_cr = cv2.Laplacian(cr, cv2.CV_32F)
            lap_cb = cv2.Laplacian(cb, cv2.CV_32F)
            score = float(np.nanmean(np.abs(lap_cr)) + np.nanmean(np.abs(lap_cb)))
            if score < best_score:
                best_pat = pat
                best_score = score
        except Exception:
            continue
    return best_pat


# ------------------
# Generic raw loader (FITS or NEF)
# ------------------

def load_mosaiced_raw(path: Path) -> np.ndarray:
    """
    Load a mosaiced frame from either FITS or NEF.
    - FITS: returns the primary/first image data.
    - NEF: returns raw_image_visible as float32 (2D Bayer).
    """
    suffix = path.suffix.lower()

    # FITS / compressed FITS
    if suffix in (".fits", ".fit", ".fz") or path.name.lower().endswith((".fits.fz", ".fits.gz")):
        return fits.getdata(path, memmap=False)

    # NEF
    if suffix == ".nef":
        with rawpy.imread(str(path)) as raw:
            img = raw.raw_image_visible.astype(np.float32)
        return img

    raise ValueError(f"Unsupported file type for {path}")


def load_fits_with_header(path: Path) -> tuple[np.ndarray, fits.Header]:
    """
    Load FITS image data together with the header of the image HDU.

    Used for LIGHT frames so that the original header can be copied
    to calibrated and aligned outputs (and then extended with new
    keywords and HISTORY).
    """
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)) and hdu.data is not None:
                data = np.array(hdu.data)
                header = hdu.header.copy()
                return data, header
        # Fallback: just take the primary header/data
        data = np.array(hdul[0].data)
        header = hdul[0].header.copy()
        return data, header


# ------------------
# Geometry / rotation / orientation
# ------------------

def _rotate_if_needed(frame: np.ndarray, rotate_portraits: bool) -> tuple[np.ndarray, bool]:
    """Rotate to landscape (90° CW) if H>W. Returns (frame, rotated_flag)."""
    if not rotate_portraits:
        return frame, False

    if frame.ndim == 2:
        H, W = frame.shape
        if H > W:
            return frame.T[::-1, :], True
        return frame, False

    if frame.ndim == 3:
        if frame.shape[0] == 3:  # (3,H,W)
            _, H, W = frame.shape
            if H > W:
                return np.transpose(frame, (0, 2, 1))[:, ::-1, :], True
            return frame, False
        if frame.shape[-1] == 3:  # (H,W,3)
            H, W, _ = frame.shape
            if H > W:
                fr = np.transpose(frame, (1, 0, 2))[:, ::-1, :]
                return fr, True
            return frame, False

    raise ValueError(f"Unsupported frame shape for rotation: {frame.shape}")


def central_crop_factor(img: np.ndarray, factor: float, keep_even: bool = True) -> np.ndarray:
    """Center crop by division factor. (Used only if APS-C crop is enabled)."""
    if factor <= 1.0:
        return img
    if img.ndim == 2:
        H, W = img.shape
        h = int(round(H / factor))
        w = int(round(W / factor))
        if keep_even:
            h -= (h % 2)
            w -= (w % 2)
        y0 = (H - h) // 2
        x0 = (W - w) // 2
        return img[y0:y0 + h, x0:x0 + w]
    if img.ndim == 3 and img.shape[0] == 3:
        _, H, W = img.shape
        h = int(round(H / factor))
        w = int(round(W / factor))
        if keep_even:
            h -= (h % 2)
            w -= (w % 2)
        y0 = (H - h) // 2
        x0 = (W - w) // 2
        return img[:, y0:y0 + h, x0:x0 + w]
    if img.ndim == 3 and img.shape[-1] == 3:
        H, W, _ = img.shape
        h = int(round(H / factor))
        w = int(round(W / factor))
        if keep_even:
            h -= (h % 2)
            w -= (w % 2)
        y0 = (H - h) // 2
        x0 = (W - w) // 2
        return img[y0:y0 + h, x0:x0 + w, :]
    raise ValueError(f"Unsupported shape for central crop: {img.shape}")


def apply_final_orient(arr: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply a final orientation to a 2D or (3,H,W)/(H,W,3) array *after* stacking.
    mode: 'none' | 'v' | 'h' | 'vh' | 'rot180'
    v -> vertical flip (top<->bottom)
    h -> horizontal flip (left<->right)
    vh -> both (same as 180° rotation)
    rot180 -> alias of 'vh'
    """
    if mode in (None, "", "none"):
        return arr

    if arr.ndim == 2:
        V = lambda a: a[::-1, :]
        H = lambda a: a[:, ::-1]
    elif arr.ndim == 3 and arr.shape[0] == 3:
        V = lambda a: a[:, ::-1, :]
        H = lambda a: a[:, :, ::-1]
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        V = lambda a: a[::-1, :, :]
        H = lambda a: a[:, ::-1, :]
    else:
        raise ValueError(f"Unsupported shape for orientation: {arr.shape}")

    if mode == "v":
        return V(arr)
    if mode == "h":
        return H(arr)
    if mode in ("vh", "rot180"):
        return V(H(arr))
    raise ValueError(f"Unknown final orientation mode: {mode}")


# ------------------
# Preprocess helpers
# ------------------

def _preprocess_frame(
    arr: np.ndarray,
    cfg: PipelineConfig,
    bayer_pattern: Optional[str],
) -> tuple[np.ndarray, Optional[str]]:
    """
    DEMOSAIC-FIRST path:
    Rotate -> (optional APS-C crop) -> demosaic if needed ->
    collapse to mono if cfg.output_color == 'bw'.

    Returns (frame, detected_pattern_if_any).
    Output shape: (H,W) for mono, (3,H,W) for RGB.
    """
    detected_pat = None

    # Normalize possible shapes (2D or 3xHxW or HxWx3)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = np.transpose(arr, (2, 0, 1))  # (3,H,W)

    fr, rotated = _rotate_if_needed(_ensure_float32(arr), cfg.rotate_portraits)

    if cfg.crop_to_apsc:
        fr = central_crop_factor(fr, cfg.apsc_crop_factor, keep_even=True)

    # Demosaic if still 2D
    if fr.ndim == 2:
        pat = bayer_pattern
        if not pat or str(pat).upper() == "AUTO":
            detected_pat = _auto_detect_cfa(np.nan_to_num(fr, nan=0.0))
            pat = detected_pat
        else:
            pat = str(pat).upper()
        if rotated:
            pat = _ROT90CW.get(pat, pat)
        rgb = _demosaic_rgb(
            np.clip(np.nan_to_num(fr, nan=0.0), 0, 65535).astype(np.uint16),
            pat,
        )
        fr = np.transpose(rgb, (2, 0, 1))  # (3,H,W)

    # Collapse to mono if B&W output
    if cfg.output_color == "bw":
        if fr.ndim == 3 and fr.shape[0] == 3:
            fr = np.nansum(fr, axis=0)  # (H,W)
        elif fr.ndim != 2:
            raise ValueError(f"Unexpected frame shape after preprocess: {fr.shape}")

    return fr.astype(np.float32), detected_pat


def _preprocess_bayer_frame(
    arr: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    """
    BAYER-FIRST path:
    Rotate -> (optional APS-C crop), but keep data mosaiced (2D).
    """
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Bayer-first path expects 2D mosaiced data, got {arr.shape}")

    fr, _ = _rotate_if_needed(_ensure_float32(arr), cfg.rotate_portraits)

    if cfg.crop_to_apsc:
        fr = central_crop_factor(fr, cfg.apsc_crop_factor, keep_even=True)

    return fr.astype(np.float32)


def _make_detection_image(frame: np.ndarray) -> np.ndarray:
    """Build a 2D detection image for astroalign from 2D or (3,H,W) RGB."""
    if frame.ndim == 2:
        return np.nan_to_num(frame, nan=0.0)
    if frame.ndim == 3 and frame.shape[0] == 3:
        return np.nan_to_num(np.nansum(frame, axis=0), nan=0.0)
    raise ValueError(f"Unsupported frame shape for detection image: {frame.shape}")


def _make_detection_from_bayer(frame: np.ndarray, pattern: str) -> np.ndarray:
    """
    Build a 2D detection image from a Bayer (2D) frame by demosaicing
    with the provided Bayer pattern and summing RGB channels.

    Used in BAYER-FIRST mode to make astroalign more robust.
    """
    if frame.ndim != 2:
        raise ValueError(f"_make_detection_from_bayer expects 2D, got {frame.shape}")
    pat = pattern.upper()
    img16 = np.clip(np.nan_to_num(frame, nan=0.0), 0, 65535).astype(np.uint16)
    rgb = _demosaic_rgb(img16, pat)  # (H,W,3)
    det = np.nansum(rgb.astype(np.float32), axis=2)
    return np.nan_to_num(det, nan=0.0)


# -----------------------------------------------
# 1) Build MASTER via Astropy sigma_clip
# -----------------------------------------------

def create_master_sigmaclip(
    group: str | Path,
    source_root: Optional[str],
    output_root: str,
    sigma: float,
    maxiters: int,
    delete_inputs: bool,
    compressed: bool,
    cfg: PipelineConfig,
    final_orient: str = "none",
    target_shape: Optional[Tuple[int, ...]] = None,
    bayer_mode: bool = False,
) -> Path:
    """
    Create a master frame using sigma_clip.

    - If bayer_mode=False (DEMOSAIC-FIRST):
      * Frames are debayered via _preprocess_frame().
      * If cfg.output_color == 'bw' → B&W master (H,W).
      * If cfg.output_color == 'rgb' → RGB master (3,H,W),
        built per-channel (R,G,B each combined with B&W logic).

    - If bayer_mode=True (BAYER-FIRST):
      * Frames stay mosaiced (2D) via _preprocess_bayer_frame().
      * Master is 2D Bayer mono (H,W).

    Supports both FITS and NEF inputs via load_mosaiced_raw().
    Optionally center-crops everything to target_shape.
    """
    group = Path(group)
    if source_root is None or group.is_absolute() or group.exists():
        src_dir = group
    else:
        src_dir = Path(source_root) / group

    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect FITS / NEF files (case-insensitive, recursive, supports .fit.gz)
    # Collect FITS / NEF files (robust, macOS-safe)
    files: list[Path] = []

    for p in src_dir.iterdir():
        if not p.is_file():
            continue

        name = p.name.lower()
        if name.endswith((".fit", ".fits", ".fit.gz", ".fits.gz", ".fits.fz", ".fz", ".nef")):
            files.append(p)

    files.sort()

    if not files:
        raise FileNotFoundError(f"No FITS/NEF files found in {src_dir}")

    exptimes: list[float] = []

    # First frame + Bayer pattern tracking
    first_raw = load_mosaiced_raw(files[0])

    # This will hold the actual pattern we want to write into the header
    used_bayer_pat: Optional[str] = None

    if bayer_mode:
        # Bayer-first: keep 2D mosaiced
        first = _preprocess_bayer_frame(first_raw, cfg)  # 2D

        # Decide pattern: explicit from cfg or AUTO-detect from first frame
        if cfg.bayer_pattern and cfg.bayer_pattern.upper() != "AUTO":
            used_bayer_pat = cfg.bayer_pattern.upper()
        else:
            # Auto-detect from the first Bayer frame
            first_u16 = np.clip(
                np.nan_to_num(first, nan=0.0),
                0,
                65535
            ).astype(np.uint16)
            used_bayer_pat = _auto_detect_cfa(first_u16)

    else:
        # Demosaic-first: _preprocess_frame can AUTO-detect
        first, detected_pat = _preprocess_frame(first_raw, cfg, cfg.bayer_pattern)  # 2D or 3xHxW

        if cfg.bayer_pattern and cfg.bayer_pattern.upper() != "AUTO":
            used_bayer_pat = cfg.bayer_pattern.upper()
        else:
            # If AUTO, use what _preprocess_frame detected (may be None if input already RGB)
            used_bayer_pat = detected_pat

    if target_shape is not None:
        first = center_crop_to_shape(first, target_shape)

    xt0 = _extract_exptime(files[0])
    if xt0 is not None:
        exptimes.append(xt0)

    frame_shape = first.shape
    ndim = first.ndim
    N = len(files)

    # ---------------------------------------------------
    # Bayer-first: 2D stack
    # ---------------------------------------------------
    if bayer_mode:
        if ndim != 2:
            raise ValueError(f"Bayer mode expects 2D frame, got {frame_shape}")
        H, W = frame_shape
        stack = np.empty((N, H, W), dtype=np.float32)
        stack[0] = first

        for i, f in enumerate(tqdm(files[1:], desc=f"Stacking Bayer { _safe_name(src_dir) }"), start=1):
            raw = load_mosaiced_raw(f)
            arr = _preprocess_bayer_frame(raw, cfg)
            if target_shape is not None:
                arr = center_crop_to_shape(arr, target_shape)
            if arr.shape != frame_shape:
                raise ValueError(f"Shape mismatch in {f.name}: {arr.shape} vs {frame_shape}")
            stack[i] = arr
            xt = _extract_exptime(f)
            if xt is not None:
                exptimes.append(xt)

        clipped = sigma_clip(stack, sigma=sigma, maxiters=maxiters, axis=0, masked=True)
        master = np.ma.median(clipped, axis=0).filled(0).astype(np.float32)
        mode_label = "bayer"

    # ---------------------------------------------------
    # Demosaic-first: B&W or RGB
    # ---------------------------------------------------
    else:
        if ndim == 2:
            # B&W master
            stack = np.empty((N,) + frame_shape, dtype=np.float32)
            stack[0] = first
            for i, f in enumerate(tqdm(files[1:], desc=f"Stacking { _safe_name(src_dir) }"), start=1):
                raw = load_mosaiced_raw(f)
                arr, _ = _preprocess_frame(raw, cfg, cfg.bayer_pattern)
                if target_shape is not None:
                    arr = center_crop_to_shape(arr, target_shape)
                if arr.shape != frame_shape:
                    raise ValueError(f"Shape mismatch in {f.name}: {arr.shape} vs {frame_shape}")
                stack[i] = arr
                xt = _extract_exptime(f)
                if xt is not None:
                    exptimes.append(xt)
            clipped = sigma_clip(stack, sigma=sigma, maxiters=maxiters, axis=0, masked=True)
            master = np.ma.median(clipped, axis=0).filled(0).astype(np.float32)
            mode_label = "bw"

        elif ndim == 3 and first.shape[0] == 3:
            # RGB master per channel
            _, H, W = frame_shape
            stack = np.empty((N, 3, H, W), dtype=np.float32)
            stack[0] = first
            for i, f in enumerate(tqdm(files[1:], desc=f"Stacking RGB { _safe_name(src_dir) }"), start=1):
                raw = load_mosaiced_raw(f)
                arr, _ = _preprocess_frame(raw, cfg, cfg.bayer_pattern)
                if target_shape is not None:
                    arr = center_crop_to_shape(arr, target_shape)
                if arr.shape != frame_shape:
                    raise ValueError(f"Shape mismatch in {f.name}: {arr.shape} vs {frame_shape}")
                stack[i] = arr
                xt = _extract_exptime(f)
                if xt is not None:
                    exptimes.append(xt)

            master = np.empty((3, H, W), dtype=np.float32)
            for c in range(3):
                clipped_c = sigma_clip(
                    stack[:, c, :, :],
                    sigma=sigma,
                    maxiters=maxiters,
                    axis=0,
                    masked=True,
                )
                master[c] = np.ma.median(clipped_c, axis=0).filled(0).astype(np.float32)
            mode_label = "rgb"
        else:
            raise ValueError(f"Unexpected frame shape after preprocess: {frame_shape}")

    # Final orientation
    master = apply_final_orient(master, final_orient)

    # Save master
    suffix = "BAYER" if bayer_mode else mode_label.upper()
    out_path = Path(output_root) / f"master_{_safe_name(src_dir)}_{mode_label}.fits"

    if compressed:
        phdu = fits.PrimaryHDU()
        chdu = fits.CompImageHDU(data=master, compression_type="RICE_1")
        hdul = fits.HDUList([phdu, chdu])
    else:
        hdu = fits.PrimaryHDU(master)
        hdul = fits.HDUList([hdu])

    hdr = hdul[0].header
    hdr.add_history(_fits_ascii(f"Master via sigma_clip + masked median (mode={suffix}, per-channel for RGB)"))
    hdr["GRPPATH"] = str(src_dir)
    hdr["NFRAMES"] = int(N)
    hdr["SIGMA"] = float(sigma)
    hdr["NITER"] = int(maxiters)
    hdr["DATE"] = _utcnow_str()
    hdr["NDIM"] = int(master.ndim)
    for i, d in enumerate(master.shape):
        hdr[f"DIM{i}"] = int(d)
    hdr["OUTMODE"] = suffix
    hdr["FINORIEN"] = (final_orient.upper(), "Final orientation after stack")
    if exptimes:
        expt_arr = np.array(exptimes, dtype=float)
        hdr["MEDXPT"] = float(np.median(expt_arr))
        hdr["MINXPT"] = float(np.min(expt_arr))
        hdr["MAXXPT"] = float(np.max(expt_arr))
        # Record Bayer pattern if known
    if used_bayer_pat is not None:
        hdr["BAYERPAT"] = (
            used_bayer_pat,
            "Bayer CFA pattern used (AUTO-detected or from cfg.bayer_pattern)"
        )

    hdul.writeto(out_path, overwrite=True)
    print(f"[master-{mode_label}] Saved -> {out_path}")

    if delete_inputs:
        import shutil
        shutil.rmtree(src_dir)
        print(f"[master-{mode_label}] Deleted source folder -> {src_dir}")

    return out_path


# -------------------------------------------------------------------
# Subtraction with "keep X% negatives"
# -------------------------------------------------------------------

def subtract_with_percent_negatives(
    frame: np.ndarray,
    master: np.ndarray,
    keep_neg_frac: float = 0.01,
    per_channel: bool = True,
):
    """Subtract master; shift so only `keep_neg_frac` of negatives remain; negatives -> NaN."""
    frame = _ensure_float32(frame)
    master = _ensure_float32(master)
    if frame.shape != master.shape:
        raise ValueError(f"Shapes must match: frame {frame.shape} vs master {master.shape}")

    sub0 = frame - master

    def _shift_for_plane(diff2d: np.ndarray) -> float:
        neg = diff2d[diff2d < 0]
        if neg.size == 0:
            return 0.0
        q = np.quantile(neg, 1.0 - keep_neg_frac)
        return -float(q)

    if sub0.ndim == 2:
        sh = _shift_for_plane(sub0)
        out = sub0 + sh
        out[out < 0] = np.nan
        return out, sh

    if sub0.ndim == 3 and sub0.shape[0] == 3 and per_channel:
        out = sub0.copy()
        shifts = np.zeros(3, dtype=np.float32)
        for c in range(3):
            sh = _shift_for_plane(sub0[c])
            shifts[c] = sh
            plane = sub0[c] + sh
            plane[plane < 0] = np.nan
            out[c] = plane
        return out, shifts

    neg = sub0[sub0 < 0]
    if neg.size == 0:
        out = sub0
        out[out < 0] = np.nan
        return out, 0.0
    q = np.quantile(neg, 1.0 - keep_neg_frac)
    sh = -float(q)
    out = sub0 + sh
    out[out < 0] = np.nan
    return out, sh


# --------------------------------------------------------------
# Flat' = Flat − DarkFlat
# --------------------------------------------------------------

def make_calibrated_flat(
    flat_master_path: str | Path,
    dark_flat_master_path: str | Path,
    keep_neg_frac: float,
    per_channel: bool,
    output_root: str,
    out_name: str,
) -> Path:
    flat = _ensure_float32(fits.getdata(flat_master_path, memmap=False))
    dark_flat = _ensure_float32(fits.getdata(dark_flat_master_path, memmap=False))

    if flat.shape != dark_flat.shape:
        raise ValueError(f"Flat {flat.shape} vs Dark_Flat {dark_flat.shape} mismatch")

    cal_flat, shift = subtract_with_percent_negatives(
        flat, dark_flat, keep_neg_frac=keep_neg_frac, per_channel=per_channel
    )

    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    hdu = fits.PrimaryHDU(cal_flat)
    hdr = hdu.header
    hdr.add_history(_fits_ascii("Flat - DarkFlat with shift so only X% negatives remain; negatives -> NaN"))
    hdr["KEEPNFR"] = float(keep_neg_frac)
    if np.isscalar(shift):
        hdr["SHIFT"] = float(shift)
    else:
        for i, s in enumerate(np.ravel(shift)):
            hdr[f"SHIFT{i}"] = float(s)
    hdr["DATE"] = _utcnow_str()
    hdu.writeto(out_path, overwrite=True)
    print(f"[flat] Saved corrected flat -> {out_path}")
    return out_path


# ------------------------------
# Flat coefficient (median=1)
# ------------------------------

def make_flat_coefficient(
    corrected_flat_path: str | Path,
    output_root: str,
    out_name: str,
    sigma: float,
    maxiters: int,
    clamp: Optional[FlatClampConfig],
) -> Path:
    """Build coefficient map so sigma-clipped median == 1.
    Supports 2D mono or (3,H,W) RGB. Low-side clamp only.
    """
    if clamp is None:
        clamp = FlatClampConfig()

    flat = _ensure_float32(fits.getdata(corrected_flat_path, memmap=False))
    flat[~np.isfinite(flat)] = np.nan
    flat[flat <= 0] = np.nan

    def _robust(x):
        return _robust_sigma(x) if clamp.use_mad else float(np.nanstd(x))

    def _clamp_low_only(cmap: np.ndarray) -> np.ndarray:
        med_c = float(np.nanmedian(cmap))
        sig_c = float(_robust(cmap))
        low = med_c - clamp.clip_low_k * sig_c
        if clamp.pct_low is not None:
            low = max(low, float(np.nanpercentile(cmap, clamp.pct_low)))
        low = max(low, clamp.min_coeff)
        return np.clip(cmap, low, np.inf)

    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    # RGB flat
    if flat.ndim == 3 and flat.shape[0] == 3:
        flat_ma = sigma_clip(flat, sigma=sigma, maxiters=maxiters, axis=None, masked=True, copy=True)
        med = np.array(
            [np.ma.median(flat_ma[0]), np.ma.median(flat_ma[1]), np.ma.median(flat_ma[2])],
            dtype=np.float32,
        )
        coeff = np.empty_like(flat, dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            coeff[0] = med[0] / flat[0]
            coeff[1] = med[1] / flat[1]
            coeff[2] = med[2] / flat[2]
        coeff = _clamp_low_only(coeff)
        coeff[~np.isfinite(coeff)] = np.nan
        coeff[coeff <= 0] = np.nan

        hdu = fits.PrimaryHDU(coeff)
        hdr = hdu.header
        hdr.add_history(_fits_ascii("Flat coeff (RGB): per-channel sigma-clipped median normalization; low-side K*sigma clamp"))
        hdr["MEDFR"] = float(med[0])
        hdr["MEDFG"] = float(med[1])
        hdr["MEDFB"] = float(med[2])
        hdr["ROBUST"] = bool(clamp.use_mad)
        hdr["CLKLOWK"] = float(clamp.clip_low_k)
        if clamp.pct_low is not None:
            hdr["PCTLOW"] = float(clamp.pct_low)
        hdr["MINCOEF"] = float(clamp.min_coeff)
        hdr["LOWONLY"] = (True, "flat coeff uses only low-side clamp")
        hdr["DATE"] = _utcnow_str()
        hdu.writeto(out_path, overwrite=True)
        print(f"[flatcoef] Saved -> {out_path}")
        return out_path

    # Mono flat
    if flat.ndim != 2:
        raise ValueError(f"Unsupported flat shape {flat.shape}; expected (H,W) or (3,H,W)")

    sc = sigma_clip(flat, sigma=sigma, maxiters=maxiters, masked=True, copy=True)
    med = float(np.ma.median(sc))
    with np.errstate(divide="ignore", invalid="ignore"):
        coeff = (med / flat).astype(np.float32)
    coeff = _clamp_low_only(coeff)
    coeff[~np.isfinite(coeff)] = np.nan
    coeff[coeff <= 0] = np.nan

    hdu = fits.PrimaryHDU(coeff)
    hdr = hdu.header
    hdr.add_history(_fits_ascii("Flat coeff (mono): sigma-clipped median normalization; low-side K*sigma clamp"))
    hdr["MEDF"] = float(med)
    hdr["ROBUST"] = bool(clamp.use_mad)
    hdr["CLKLOWK"] = float(clamp.clip_low_k)
    if clamp.pct_low is not None:
        hdr["PCTLOW"] = float(clamp.pct_low)
    hdr["MINCOEF"] = float(clamp.min_coeff)
    hdr["LOWONLY"] = (True, "flat coeff uses only low-side clamp")
    hdr["DATE"] = _utcnow_str()
    hdu.writeto(out_path, overwrite=True)
    print(f"[flatcoef] Saved -> {out_path}")
    return out_path


# --------------------------------------------------------------
# Helpers for saving, alignment
# --------------------------------------------------------------

def _to_uint16_image(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=65535.0, neginf=0.0)
    arr = np.clip(arr, 0, 65535)
    return arr.astype(np.uint16)


# --------------------------------------------------------------
# DEMOSAIC-FIRST: Calibrate, align, stack
# --------------------------------------------------------------

def calibrate_fits_light_group_demosaic(
    light_group: str | Path,
    dark_master_path: str | Path,
    flat_coeff_path: str | Path,
    in_root: Optional[str],
    out_root: str,
    cfg: PipelineConfig,
) -> Path:
    """
    DEMOSAIC-FIRST mode:
    - Calibrate lights in debayered space (mono or RGB).
    - Save calibrated unaligned in .../<group>/calibrated/
    - Align with astroalign, save in .../<group>/aligned/
    - Stack aligned via sigma-clipped median * N into .../<group>/stack/stack_sigma_medxN.fits

    For each light, the original FITS header is copied to the calibrated
    and aligned products, and then extended with new HISTORY / keywords.
    """
    light_group = Path(light_group)
    in_dir = light_group if (in_root is None or light_group.is_absolute() or light_group.exists()) else Path(in_root) / light_group

    safe_group = _safe_name(in_dir)
    base_dir = Path(out_root) / safe_group
    unaligned_dir = base_dir / "calibrated"
    aligned_dir = base_dir / "aligned"
    stack_dir = base_dir / "stack"
    unaligned_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)
    stack_dir.mkdir(parents=True, exist_ok=True)

    master_dark = _ensure_float32(fits.getdata(dark_master_path, memmap=False))
    flat_coeff = _ensure_float32(fits.getdata(flat_coeff_path, memmap=False))

    files: list[Path] = []
    for pat in ("*.fits", "*.fit", "*.fits.fz", "*.fits.gz", "*.fz"):
        files.extend(in_dir.glob(pat))
    files.sort()
    if not files:
        raise FileNotFoundError(f"No FITS files found in {in_dir}")

    target_shape = master_dark.shape
    if flat_coeff.shape != target_shape:
        raise ValueError(f"Flat coeff shape {flat_coeff.shape} != dark master shape {target_shape}")

    calibrated_list: list[np.ndarray] = []
    orig_headers: list[fits.Header] = []

    # 1) Calibrate (debayered) & save unaligned
    for p in tqdm(files, desc=f"[demosaic] Calibrating {safe_group}"):
        raw, base_hdr = load_fits_with_header(p)
        frame, _ = _preprocess_frame(raw, cfg, cfg.bayer_pattern)

        if frame.shape != target_shape:
            frame = center_crop_to_shape(frame, target_shape)

        if frame.shape != master_dark.shape or frame.shape != flat_coeff.shape:
            raise ValueError(
                f"Shape mismatch for {p.name}: {frame.shape} vs dark {master_dark.shape}, flat {flat_coeff.shape}"
            )

        sub, shift = subtract_with_percent_negatives(
            frame, master_dark, keep_neg_frac=cfg.keep_neg_frac, per_channel=cfg.per_channel_shift
        )
        calibrated = sub * flat_coeff
        calibrated[~np.isfinite(calibrated)] = np.nan

        calibrated_list.append(calibrated.astype(np.float32))
        orig_headers.append(base_hdr)

        out_data = _to_uint16_image(calibrated)
        out_path = unaligned_dir / p.name
        exptime = _extract_exptime(p)

        # Copy original header, then extend
        hdr = base_hdr.copy()
        hdr.add_history(
            _fits_ascii("Calibrated (debayered): dark-subtracted (X% negatives) + flat-fielded (median=1, low-clamped)")
        )
        hdr["KEEPNFR"] = float(cfg.keep_neg_frac)
        if np.isscalar(shift):
            hdr["SHIFT"] = float(shift)
        else:
            for i, s in enumerate(np.ravel(shift)):
                hdr[f"SHIFT{i}"] = float(s)
        hdr["DATE"] = _utcnow_str()
        hdr["U16SAVE"] = (True, "Saved as uint16 (unaligned)")
        hdr["OUTCOLOR"] = cfg.output_color.upper()
        if exptime is not None:
            hdr["EXPTIME"] = (float(exptime), "exposure time [s]")
            hdr["EXPOSURE"] = (float(exptime), "exposure time [s] (alias)")

        if cfg.compressed_fits:
            phdu = fits.PrimaryHDU()
            chdu = fits.CompImageHDU(data=out_data, header=hdr, compression_type="RICE_1")
            fits.HDUList([phdu, chdu]).writeto(out_path, overwrite=True)
        else:
            hdu = fits.PrimaryHDU(out_data, header=hdr)
            fits.HDUList([hdu]).writeto(out_path, overwrite=True)

        tqdm.write(f" -> saved unaligned {out_path}")

    if not calibrated_list:
        raise RuntimeError(f"No calibrated frames were produced for {in_dir}")

    # 2) Align with astroalign
    N = len(calibrated_list)
    ref_frame = calibrated_list[0]
    ref_det = _make_detection_image(ref_frame)
    aligned_list: list[np.ndarray] = []

    for idx, (p, frame, base_hdr) in enumerate(zip(files, calibrated_list, orig_headers)):
        if idx == 0:
            aligned = frame
        else:
            src_det = _make_detection_image(frame)
            try:
                tform, _ = aa.find_transform(src_det, ref_det)
                if frame.ndim == 2:
                    aligned = aa.apply_transform(tform, frame, ref_det.shape)
                elif frame.ndim == 3 and frame.shape[0] == 3:
                    chans = []
                    for c in range(3):
                        chan_aligned = aa.apply_transform(tform, frame[c], ref_det.shape)
                        chans.append(chan_aligned)
                    aligned = np.stack(chans, axis=0)
                else:
                    raise ValueError(f"Unexpected frame shape for alignment: {frame.shape}")
            except Exception as e:
                print(f"[align] Warning: astroalign failed for {p.name}: {e}. Using unaligned frame.")
                aligned = frame

        aligned_list.append(aligned.astype(np.float32))

        out_aligned_data = _to_uint16_image(aligned)
        out_aligned_path = aligned_dir / p.name

        hdr = base_hdr.copy()
        hdr.add_history(
            _fits_ascii("Calibrated + astroalign-registered to first calibrated frame (debayered)")
        )
        hdr["DATE"] = _utcnow_str()
        hdr["U16SAVE"] = (True, "Saved as uint16 (aligned)")
        hdr["OUTCOLOR"] = cfg.output_color.upper()

        if cfg.compressed_fits:
            phdu = fits.PrimaryHDU()
            chdu = fits.CompImageHDU(data=out_aligned_data, header=hdr, compression_type="RICE_1")
            fits.HDUList([phdu, chdu]).writeto(out_aligned_path, overwrite=True)
        else:
            hdu = fits.PrimaryHDU(out_aligned_data, header=hdr)
            fits.HDUList([hdu]).writeto(out_aligned_path, overwrite=True)

        tqdm.write(f" -> saved aligned {out_aligned_path}")

    # 3) Stack aligned frames via sigma-clipped median * N
    stack_arr = np.stack(aligned_list, axis=0)  # (N,H,W) or (N,3,H,W)
    clipped = sigma_clip(
        stack_arr,
        sigma=cfg.sigma_clip_sigma,
        maxiters=cfg.sigma_clip_maxiters,
        axis=0,
        masked=True,
    )
    median = np.ma.median(clipped, axis=0).filled(0).astype(np.float32)
    stacked = median * float(N)

    stack_path = stack_dir / "stack_sigma_medxN.fits"
    hdu = fits.PrimaryHDU(stacked)
    hdr = hdu.header
    hdr.add_history(_fits_ascii("Sigma-clipped median stack * N from aligned calibrated frames (debayered)"))
    hdr["NFRAMES"] = int(N)
    hdr["SIGMA"] = float(cfg.sigma_clip_sigma)
    hdr["MAXITER"] = int(cfg.sigma_clip_maxiters)
    hdr["OUTCOLOR"] = cfg.output_color.upper()
    hdr["DATE"] = _utcnow_str()
    hdu.writeto(stack_path, overwrite=True)

    print(f"[stack-demosaic] Saved stacked image -> {stack_path}")
    return stack_path


# --------------------------------------------------------------
# BAYER-FIRST: Calibrate, align, stack in Bayer space
# --------------------------------------------------------------

def calibrate_fits_light_group_bayer(
    light_group: str | Path,
    dark_master_path: str | Path,
    flat_coeff_path: str | Path,
    in_root: Optional[str],
    out_root: str,
    cfg: PipelineConfig,
) -> Path:
    """
    BAYER-FIRST mode:
    - Lights remain mosaiced (2D Bayer) through calibration, alignment, stacking.
    - Save calibrated unaligned Bayer frames in .../<group>/calibrated_bayer/
    - Save aligned Bayer frames in .../<group>/aligned_bayer/
    - Save stacked Bayer frame in .../<group>/stack/stack_bayer.fits
    - Auto-demosaic final stacked Bayer to RGB/BW in same stack dir:
      stack_demosaic_rgb.fits or stack_demosaic_bw.fits

    For each light, the original FITS header is copied to the calibrated
    and aligned Bayer products, and then extended with new HISTORY / keywords.
    """
    light_group = Path(light_group)
    in_dir = light_group if (in_root is None or light_group.is_absolute() or light_group.exists()) else Path(in_root) / light_group

    safe_group = _safe_name(in_dir)
    base_dir = Path(out_root) / safe_group
    unaligned_dir = base_dir / "calibrated_bayer"
    aligned_dir = base_dir / "aligned_bayer"
    stack_dir = base_dir / "stack"
    unaligned_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)
    stack_dir.mkdir(parents=True, exist_ok=True)

    master_dark = _ensure_float32(fits.getdata(dark_master_path, memmap=False))
    flat_coeff = _ensure_float32(fits.getdata(flat_coeff_path, memmap=False))

    files: list[Path] = []
    for pat in ("*.fits", "*.fit", "*.fits.fz", "*.fits.gz", "*.fz"):
        files.extend(in_dir.glob(pat))
    files.sort()
    if not files:
        raise FileNotFoundError(f"No FITS files found in {in_dir}")

    target_shape = master_dark.shape
    if flat_coeff.shape != target_shape:
        raise ValueError(f"Flat coeff shape {flat_coeff.shape} != dark master shape {target_shape}")

    calibrated_list: list[np.ndarray] = []
    orig_headers: list[fits.Header] = []

    # 1) Calibrate in Bayer space
    for p in tqdm(files, desc=f"[bayer] Calibrating {safe_group}"):
        raw, base_hdr = load_fits_with_header(p)
        frame = _preprocess_bayer_frame(raw, cfg)

        if frame.shape != target_shape:
            frame = center_crop_to_shape(frame, target_shape)

        if frame.shape != master_dark.shape or frame.shape != flat_coeff.shape:
            raise ValueError(
                f"Shape mismatch for {p.name}: {frame.shape} vs dark {master_dark.shape}, flat {flat_coeff.shape}"
            )

        sub, shift = subtract_with_percent_negatives(
            frame, master_dark, keep_neg_frac=cfg.keep_neg_frac, per_channel=False
        )
        calibrated = sub * flat_coeff
        calibrated[~np.isfinite(calibrated)] = np.nan

        calibrated_list.append(calibrated.astype(np.float32))
        orig_headers.append(base_hdr)

        out_data = _to_uint16_image(calibrated)
        out_path = unaligned_dir / p.name
        exptime = _extract_exptime(p)

        hdr = base_hdr.copy()
        hdr.add_history(
            _fits_ascii("Calibrated (Bayer): dark-subtracted (X% negatives) + flat-fielded (median=1, low-clamped)")
        )
        hdr["KEEPNFR"] = float(cfg.keep_neg_frac)
        if np.isscalar(shift):
            hdr["SHIFT"] = float(shift)
        hdr["DATE"] = _utcnow_str()
        hdr["U16SAVE"] = (True, "Saved as uint16 (Bayer, unaligned)")
        hdr["OUTMODE"] = "BAYER"
        if exptime is not None:
            hdr["EXPTIME"] = (float(exptime), "exposure time [s]")
            hdr["EXPOSURE"] = (float(exptime), "exposure time [s] (alias)")

        if cfg.compressed_fits:
            phdu = fits.PrimaryHDU()
            chdu = fits.CompImageHDU(data=out_data, header=hdr, compression_type="RICE_1")
            fits.HDUList([phdu, chdu]).writeto(out_path, overwrite=True)
        else:
            hdu = fits.PrimaryHDU(out_data, header=hdr)
            fits.HDUList([hdu]).writeto(out_path, overwrite=True)

        tqdm.write(f" -> saved unaligned Bayer {out_path}")

    if not calibrated_list:
        raise RuntimeError(f"No calibrated frames were produced for {in_dir}")

    # 2) Align Bayer frames
    N = len(calibrated_list)
    ref_frame = calibrated_list[0]

    # Decide Bayer pattern for alignment (from config or auto-detected on first frame)
    if cfg.bayer_pattern and cfg.bayer_pattern.upper() != "AUTO":
        bayer_pat = cfg.bayer_pattern.upper()
    else:
        bayer_pat = _auto_detect_cfa(np.clip(np.nan_to_num(ref_frame, nan=0.0), 0, 65535).astype(np.uint16))

    ref_det = _make_detection_from_bayer(ref_frame, bayer_pat)
    aligned_list: list[np.ndarray] = []

    for idx, (p, frame, base_hdr) in enumerate(zip(files, calibrated_list, orig_headers)):
        if idx == 0:
            aligned = frame
        else:
            src_det = _make_detection_from_bayer(frame, bayer_pat)
            try:
                tform, _ = aa.find_transform(src_det, ref_det)
                aligned = aa.apply_transform(tform, frame, ref_det.shape)
            except Exception as e:
                print(f"[align-bayer] Warning: astroalign failed for {p.name}: {e}. Using unaligned frame.")
                aligned = frame

        aligned_list.append(aligned.astype(np.float32))

        out_aligned_data = _to_uint16_image(aligned)
        out_aligned_path = aligned_dir / p.name

        hdr = base_hdr.copy()
        hdr.add_history(
            _fits_ascii("Calibrated (Bayer) + astroalign-registered to first Bayer calibrated frame")
        )
        hdr["DATE"] = _utcnow_str()
        hdr["U16SAVE"] = (True, "Saved as uint16 (Bayer, aligned)")
        hdr["OUTMODE"] = "BAYER"
        hdr["BAYERPAT"] = bayer_pat

        if cfg.compressed_fits:
            phdu = fits.PrimaryHDU()
            chdu = fits.CompImageHDU(data=out_aligned_data, header=hdr, compression_type="RICE_1")
            fits.HDUList([phdu, chdu]).writeto(out_aligned_path, overwrite=True)
        else:
            hdu = fits.PrimaryHDU(out_aligned_data, header=hdr)
            fits.HDUList([hdu]).writeto(out_aligned_path, overwrite=True)

        tqdm.write(f" -> saved aligned Bayer {out_aligned_path}")

    # 3) Stack Bayer frames
    stack_arr = np.stack(aligned_list, axis=0)  # (N,H,W)
    clipped = sigma_clip(
        stack_arr,
        sigma=cfg.sigma_clip_sigma,
        maxiters=cfg.sigma_clip_maxiters,
        axis=0,
        masked=True,
    )
    median = np.ma.median(clipped, axis=0).filled(0).astype(np.float32)
    stacked_bayer = median * float(N)

    stack_bayer_path = stack_dir / "stack_bayer.fits"
    hdu = fits.PrimaryHDU(stacked_bayer)
    hdr = hdu.header
    hdr.add_history(_fits_ascii("Sigma-clipped median stack * N from aligned calibrated Bayer frames"))
    hdr["NFRAMES"] = int(N)
    hdr["SIGMA"] = float(cfg.sigma_clip_sigma)
    hdr["MAXITER"] = int(cfg.sigma_clip_maxiters)
    hdr["OUTMODE"] = "BAYER"
    hdr["BAYERPAT"] = bayer_pat
    hdr["DATE"] = _utcnow_str()
    hdu.writeto(stack_bayer_path, overwrite=True)

    print(f"[stack-bayer] Saved stacked Bayer image -> {stack_bayer_path}")

    # 4) Demosaic the stacked Bayer frame for final RGB/BW master
    rgb = _demosaic_rgb(stacked_bayer, bayer_pat)  # (H,W,3)

    if cfg.output_color == "rgb":
        out_arr = np.transpose(rgb, (2, 0, 1))  # (3,H,W)
        stack_rgb_path = stack_dir / "stack_demosaic_rgb.fits"
        hdu2 = fits.PrimaryHDU(out_arr.astype(np.float32))
        hdr2 = hdu2.header
        hdr2.add_history(_fits_ascii(f"Demosaiced final Bayer stack to RGB using pattern {bayer_pat}"))
        hdr2["OUTCOLOR"] = "RGB"
        hdr2["BAYERPAT"] = bayer_pat
        hdr2["DATE"] = _utcnow_str()
        hdu2.writeto(stack_rgb_path, overwrite=True)
        print(f"[stack-bayer] Saved demosaiced RGB stack -> {stack_rgb_path}")
        return stack_rgb_path
    else:
        bw = np.nansum(rgb, axis=2)  # (H,W)
        stack_bw_path = stack_dir / "stack_demosaic_bw.fits"
        hdu2 = fits.PrimaryHDU(bw.astype(np.float32))
        hdr2 = hdu2.header
        hdr2.add_history(
            _fits_ascii(
                f"Demosaiced final Bayer stack to RGB then collapsed to B&W (sum) using pattern {bayer_pat}"
            )
        )
        hdr2["OUTCOLOR"] = "BW"
        hdr2["BAYERPAT"] = bayer_pat
        hdr2["DATE"] = _utcnow_str()
        hdu2.writeto(stack_bw_path, overwrite=True)
        print(f"[stack-bayer] Saved demosaiced BW stack -> {stack_bw_path}")
        return stack_bw_path


# ---------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------

def batch_calibrate_fits_lights_demosaic(
    flat_coeff_path: str | Path,
    mapping: dict[str, str],
    in_root: Optional[str],
    out_root: str,
    cfg: PipelineConfig,
) -> list[Path]:
    flat_coeff_path = Path(flat_coeff_path)
    final_stacks: list[Path] = []
    for light_group, dark_master_path in mapping.items():
        final_stacks.append(
            calibrate_fits_light_group_demosaic(
                light_group=light_group,
                dark_master_path=dark_master_path,
                flat_coeff_path=flat_coeff_path,
                in_root=in_root,
                out_root=out_root,
                cfg=cfg,
            )
        )
    return final_stacks


def batch_calibrate_fits_lights_bayer(
    flat_coeff_path: str | Path,
    mapping: dict[str, str],
    in_root: Optional[str],
    out_root: str,
    cfg: PipelineConfig,
) -> list[Path]:
    flat_coeff_path = Path(flat_coeff_path)
    final_stacks: list[Path] = []
    for light_group, dark_master_path in mapping.items():
        final_stacks.append(
            calibrate_fits_light_group_bayer(
                light_group=light_group,
                dark_master_path=dark_master_path,
                flat_coeff_path=flat_coeff_path,
                in_root=in_root,
                out_root=out_root,
                cfg=cfg,
            )
        )
    return final_stacks


# ---------------------------------------------------------
# Optional plate solving of the single final stack
# ---------------------------------------------------------

def _positive_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def _prepare_plate_solve_input(final_stack: Path, run_dir: Path) -> Path:
    """Return a 2-D FITS image suitable for source extraction.

    Monochrome final stacks are solved directly. RGB cubes are collapsed only
    for source extraction; the calibrated final stack is not modified.
    """
    data, header = load_fits_with_header(final_stack)
    data = np.asarray(data)

    if data.ndim == 2:
        return final_stack
    if data.ndim != 3:
        raise ValueError(
            f"Plate solving requires a 2-D image or RGB cube; got shape {data.shape}"
        )

    if data.shape[0] == 3:
        luminance = np.nanmean(data.astype(np.float32), axis=0)
    elif data.shape[-1] == 3:
        luminance = np.nanmean(data.astype(np.float32), axis=-1)
    else:
        raise ValueError(f"Cannot identify the RGB channel axis in shape {data.shape}")

    luminance = np.nan_to_num(luminance, nan=0.0, posinf=0.0, neginf=0.0)
    solve_input = run_dir / f"{final_stack.stem}_platesolve_input.fits"
    solve_header = header.copy()
    solve_header.add_history(
        _fits_ascii("Temporary RGB-mean image created only for plate-solving source extraction")
    )
    fits.PrimaryHDU(luminance.astype(np.float32), header=solve_header).writeto(
        solve_input, overwrite=False
    )
    return solve_input


def _show_plate_solution(annotated: Path, final_stack: Path) -> None:
    """Display the annotated solution in a blocking Matplotlib window."""
    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required to display the plate-solved image. "
            "Install it with: python3 -m pip install matplotlib"
        ) from exc

    image = mpimg.imread(annotated)
    figure, axis = plt.subplots(num="Plate-solved final image", figsize=(12, 8))
    axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
    axis.set_title(f"Plate solution: {final_stack.name}")
    axis.axis("off")
    figure.tight_layout()
    plt.show(block=True)


def plate_solve_final_stack(final_stack: str | Path, args: argparse.Namespace) -> Path:
    """Plate solve one final stack, annotate it, and show it with Matplotlib."""
    final_stack = Path(final_stack).expanduser().resolve()
    if not final_stack.is_file():
        raise FileNotFoundError(f"Final stack not found: {final_stack}")

    executables: dict[str, str] = {}
    for name in ("solve-field", "plot-constellations"):
        executable = shutil.which(name)
        if not executable:
            raise RuntimeError(
                f"{name} is not installed or not on PATH. Install the full Astrometry.net package."
            )
        executables[name] = executable

    script_dir = Path(__file__).resolve().parent
    index_dirs = [
        directory.expanduser().resolve()
        for directory in (args.index_dir or [script_dir / "Astrometry-Indexes"])
    ]
    index_files: list[Path] = []
    for directory in index_dirs:
        files = sorted(directory.glob("index-*.fits")) if directory.is_dir() else []
        if not files:
            raise FileNotFoundError(
                f"No index-*.fits files directly inside {directory}; "
                "point --index-dir to the actual Astrometry.net index folder"
            )
        if "\n" in str(directory) or "\r" in str(directory):
            raise ValueError("Index directory cannot contain newlines")
        index_files.extend(files)

    output_root = (
        args.solve_output.expanduser().resolve()
        if args.solve_output
        else final_stack.parent / "plate_solve_outputs"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"{final_stack.stem}_", dir=output_root))

    astrometry_cfg = run_dir / "astrometry.cfg"
    astrometry_cfg.write_text(
        "\n".join(
            [f"cpulimit {args.cpulimit}", *[f"index {path}" for path in index_files], ""]
        ),
        encoding="utf-8",
    )

    solve_input = _prepare_plate_solve_input(final_stack, run_dir)
    solved_fits = run_dir / f"{final_stack.stem}_solved.fits"
    command = [
        executables["solve-field"],
        "--config", str(astrometry_cfg),
        "--dir", str(run_dir),
        "--out", "field",
        "--new-fits", str(solved_fits),
        "--pnm", str(run_dir / "field.pnm"),
        "--downsample", str(args.downsample),
        "--cpulimit", str(args.cpulimit),
        "--no-verify",
        "--crpix-center",
    ]
    if not args.blind_scale:
        command += [
            "--scale-units", args.scale_units,
            "--scale-low", str(args.scale_low),
            "--scale-high", str(args.scale_high),
        ]
    if args.ra is not None:
        command += [f"--ra={args.ra}", f"--dec={args.dec}", "--radius", str(args.radius)]
    if args.parity:
        command += ["--parity", args.parity]
    command.append(str(solve_input))

    (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    print(f"[plate-solve] Final stack only: {final_stack}")
    print(f"[plate-solve] Output: {run_dir}")
    print(f"[plate-solve] Running {shlex.join(command)}", flush=True)

    with (run_dir / "solve.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise

    marker = run_dir / "field.solved"
    solved_ok = (
        marker.exists()
        and marker.read_bytes()[:1] == b"\x01"
        and solved_fits.is_file()
    )
    if not solved_ok:
        raise RuntimeError(
            f"No complete plate solution (exit {return_code}). See {run_dir / 'solve.log'}. "
            "Check index scales and hints, or use --blind-scale."
        )
    if return_code:
        print(
            f"[plate-solve] Warning: solve-field returned {return_code}; check solve.log.",
            file=sys.stderr,
        )

    annotated = run_dir / f"{final_stack.stem}_annotated.png"
    annotation_command = [
        executables["plot-constellations"],
        "-w", str(run_dir / "field.wcs"),
        "-i", str(run_dir / "field.pnm"),
        "-o", str(annotated),
        "-N", "-B", "-F", "0", "-G", "15",
    ]
    with (run_dir / "annotation.log").open("w", encoding="utf-8") as log:
        annotation_result = subprocess.run(
            annotation_command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if annotation_result.returncode or not annotated.is_file():
        raise RuntimeError(
            f"Plate solution succeeded, but annotation failed. See {run_dir / 'annotation.log'}"
        )

    print(f"[plate-solve] Solved FITS: {solved_fits}")
    print(f"[plate-solve] Annotated image: {annotated}")
    _show_plate_solution(annotated, final_stack)
    return annotated


# --------------------
# CLI / Main
# --------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Calibration for NEF (flats) + FITS (lights) with alignment & stacking "
            "(demosaic-first or bayer-first)."
        )
    )
    ap.add_argument("--flats", help="Folder for FLATS group (relative to --src-root)")
    ap.add_argument("--darks", help="Folder for DARKS group (relative to --src-root)")
    ap.add_argument("--darkflats", help="Folder for DARK-FLATS group (relative to --src-root)")
    ap.add_argument("--lights", help="Folder for LIGHTS group (relative to --src-root)")
    ap.add_argument("--src-root", default=None, help="Root of source groups (e.g. 'Data')")
    ap.add_argument("--out-root", default="calibrated", help="Output root for calibrated images")
    ap.add_argument("--masters-root", default="masters", help="Output root for master frames")
    ap.add_argument("--delete-inputs", action="store_true", help="Delete group folders after master creation")
    ap.add_argument("--rotate-portraits", action="store_true", help="Rotate frames 90° CW when H>W")
    ap.add_argument("--no-rotate", dest="rotate_portraits", action="store_false", help="Disable portrait rotation")
    ap.set_defaults(rotate_portraits=True)

    # Output selection (optional; prompt if not provided)
    ap.add_argument("--color", choices=["bw", "rgb"], help="Output color mode")

    # Stack mode
    ap.add_argument(
        "--stack-mode",
        choices=["demosaic-first", "bayer-first"],
        default="demosaic-first",
        help="Stack in debayered space or in Bayer space then demosaic at the end.",
    )

    # Calibration params
    ap.add_argument("--sigma", type=float, default=3.0, help="Sigma for sigma_clip")
    ap.add_argument("--maxiters", type=int, default=5, help="Max iterations for sigma_clip")
    ap.add_argument("--keep-neg-frac", type=float, default=0.01, help="Fraction of negatives to keep after shift")
    ap.add_argument(
        "--per-channel-shift",
        action="store_true",
        help="Apply shift per channel (RGB) in debayered mode",
    )
    ap.add_argument(
        "--global-shift",
        dest="per_channel_shift",
        action="store_false",
        help="Use a single global shift (no per-channel)",
    )
    ap.set_defaults(per_channel_shift=True)

    # Flat coefficient clamp params (LOW SIDE ONLY)
    ap.add_argument("--clip-low-k", type=float, default=6.0, help="Low clamp K*sigma for coefficients")
    ap.add_argument("--min-coeff", type=float, default=1e-6, help="Minimum coefficient after clamp")
    ap.add_argument("--mad", dest="use_mad", action="store_true", help="Use MAD-based robust sigma")
    ap.add_argument("--no-mad", dest="use_mad", action="store_false", help="Use np.nanstd for sigma")
    ap.set_defaults(use_mad=True)
    ap.add_argument("--pct-low", type=float, default=None, help="Optional percentile low clamp (e.g., 0.1)")

    # Bayer pattern / APS-C crop
    ap.add_argument("--bayer", default="AUTO", help="Bayer pattern for demosaic: AUTO | RGGB | BGGR | GRBG | GBRG")
    ap.add_argument(
        "--apsc-crop",
        action="store_true",
        help="Enable APS-C-style center crop by factor (--apsc-factor). Disabled by default.",
    )
    ap.add_argument("--apsc-factor", type=float, default=1.5, help="Center crop factor if --apsc-crop is used")

    # FITS IO
    ap.add_argument("--compressed", action="store_true", help="Write FITS with RICE_1 compression for outputs")

    # Final orientations for masters
    ap.add_argument(
        "--final-orient-flats",
        choices=["none", "v", "h", "vh", "rot180"],
        default="none",
        help="Final orientation for FLATS master (applied after stack)",
    )
    ap.add_argument(
        "--final-orient-darkflats",
        choices=["none", "v", "h", "vh", "rot180"],
        default="none",
        help="Final orientation for DARK-FLATS master (applied after stack)",
    )
    ap.add_argument(
        "--final-orient-darks",
        choices=["none", "v", "h", "vh", "rot180"],
        default="none",
        help="Final orientation for DARKS master (applied after stack)",
    )

    # Optional local plate solving. Only the final stack is passed to solve-field.
    solve = ap.add_argument_group("plate solving of the final stack")
    solve.add_argument(
        "--plate-solve",
        action="store_true",
        help="Plate solve the final stacked FITS and show its annotation with Matplotlib",
    )
    solve.add_argument(
        "--index-dir",
        type=Path,
        action="append",
        help=(
            "Astrometry.net index folder; defaults to Astrometry-Indexes beside this script; "
            "repeat for multiple folders"
        ),
    )
    solve.add_argument(
        "--solve-output",
        type=Path,
        help="Plate-solving output folder; defaults to plate_solve_outputs beside the final stack",
    )
    solve.add_argument(
        "--scale-units",
        choices=["arcsecperpix", "degwidth"],
        default="arcsecperpix",
        help="Plate-scale units: arcseconds/pixel or total image width in degrees",
    )
    solve.add_argument("--scale-low", type=_positive_number, help="Lower plate-scale bound")
    solve.add_argument("--scale-high", type=_positive_number, help="Upper plate-scale bound")
    solve.add_argument(
        "--blind-scale",
        action="store_true",
        help="Solve without a plate-scale constraint (slower and more memory intensive)",
    )
    solve.add_argument("--ra", help="Approximate J2000 RA, in decimal degrees or hh:mm:ss")
    solve.add_argument("--dec", help="Approximate J2000 Dec, in degrees or signed dd:mm:ss")
    solve.add_argument(
        "--radius",
        type=_positive_number,
        default=5.0,
        help="RA/Dec search radius in degrees (default: 5)",
    )
    solve.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Source-extraction downsampling factor (default: 2; 1 disables)",
    )
    solve.add_argument(
        "--cpulimit",
        type=int,
        default=180,
        help="Astrometry.net CPU-time limit in seconds (default: 180)",
    )
    solve.add_argument("--parity", choices=["pos", "neg"], help="Omit to try both parities")

    args = ap.parse_args()
    if args.plate_solve:
        if args.blind_scale and (args.scale_low is not None or args.scale_high is not None):
            ap.error("Use either --blind-scale or scale bounds, not both")
        if not args.blind_scale and (args.scale_low is None or args.scale_high is None):
            ap.error(
                "Plate solving requires both --scale-low and --scale-high, "
                "or an explicit --blind-scale"
            )
        if not args.blind_scale and args.scale_low >= args.scale_high:
            ap.error("--scale-low must be below --scale-high")
        if (args.ra is None) != (args.dec is None):
            ap.error("Supply both --ra and --dec, or neither")
        if args.downsample < 1 or args.cpulimit < 1:
            ap.error("--downsample and --cpulimit must be positive integers")
    return args


def _prompt_if_missing(val: Optional[str], prompt_text: str) -> str:
    if val is not None and len(val.strip()) > 0:
        return val
    return input(prompt_text).strip()


def _prompt_color_choice(existing: Optional[str]) -> OutputColor:
    if existing in ("bw", "rgb"):
        return existing  # type: ignore[return-value]
    print("Choose output mode: [1] B&W [2] RGB")
    while True:
        choice = input("> ").strip()
        if choice == "1":
            return "bw"
        if choice == "2":
            return "rgb"
        print("Please type 1 (B&W) or 2 (RGB).")


def main():
    args = parse_args()

    # Interactive fallbacks
    color_mode: OutputColor = _prompt_color_choice(args.color)
    flats = _prompt_if_missing(args.flats, "Enter FLATS folder path (e.g. Flat): ")
    darks = _prompt_if_missing(args.darks, "Enter DARKS folder path (e.g. Dark): ")
    darkflats = _prompt_if_missing(args.darkflats, "Enter DARK-FLATS folder path (e.g. Dark_Flat): ")
    lights = _prompt_if_missing(args.lights, "Enter LIGHTS folder path (e.g. Light): ")
    src_root = args.src_root if args.src_root is not None else input("Enter source root (or blank for 'as given'): ").strip() or None

    clamp = FlatClampConfig(
        clip_low_k=args.clip_low_k,
        min_coeff=args.min_coeff,
        use_mad=args.use_mad,
        pct_low=args.pct_low,
    )

    cfg = PipelineConfig(
        sigma_clip_sigma=args.sigma,
        sigma_clip_maxiters=args.maxiters,
        keep_neg_frac=args.keep_neg_frac,
        per_channel_shift=args.per_channel_shift,
        rotate_portraits=args.rotate_portraits,
        compressed_fits=args.compressed,
        output_color=color_mode,
        bayer_pattern=args.bayer.upper() if args.bayer else None,
        crop_to_apsc=args.apsc_crop,
        apsc_crop_factor=args.apsc_factor,
        clamp=clamp,
    )

    # 1) Build masters
    if args.stack_mode == "demosaic-first":
        # Debayered masters
        m_flat = create_master_sigmaclip(
            flats,
            source_root=src_root,
            output_root=args.masters_root,
            sigma=cfg.sigma_clip_sigma,
            maxiters=cfg.sigma_clip_maxiters,
            delete_inputs=args.delete_inputs,
            compressed=False,
            cfg=cfg,
            final_orient=args.final_orient_flats,
            target_shape=None,
            bayer_mode=False,
        )
        flat_shape = fits.getdata(m_flat, memmap=False).shape

        m_dark = create_master_sigmaclip(
            darks,
            source_root=src_root,
            output_root=args.masters_root,
            sigma=cfg.sigma_clip_sigma,
            maxiters=cfg.sigma_clip_maxiters,
            delete_inputs=args.delete_inputs,
            compressed=False,
            cfg=cfg,
            final_orient=args.final_orient_darks,
            target_shape=flat_shape,
            bayer_mode=False,
        )

        m_dflat = create_master_sigmaclip(
            darkflats,
            source_root=src_root,
            output_root=args.masters_root,
            sigma=cfg.sigma_clip_sigma,
            maxiters=cfg.sigma_clip_maxiters,
            delete_inputs=args.delete_inputs,
            compressed=False,
            cfg=cfg,
            final_orient=args.final_orient_darkflats,
            target_shape=flat_shape,
            bayer_mode=False,
        )
    else:
        # Bayer masters
        m_flat = create_master_sigmaclip(
            flats,
            source_root=src_root,
            output_root=args.masters_root,
            sigma=cfg.sigma_clip_sigma,
            maxiters=cfg.sigma_clip_maxiters,
            delete_inputs=args.delete_inputs,
            compressed=False,
            cfg=cfg,
            final_orient=args.final_orient_flats,
            target_shape=None,
            bayer_mode=True,
        )
        flat_shape = fits.getdata(m_flat, memmap=False).shape

        m_dark = create_master_sigmaclip(
            darks,
            source_root=src_root,
            output_root=args.masters_root,
            sigma=cfg.sigma_clip_sigma,
            maxiters=cfg.sigma_clip_maxiters,
            delete_inputs=args.delete_inputs,
            compressed=False,
            cfg=cfg,
            final_orient=args.final_orient_darks,
            target_shape=flat_shape,
            bayer_mode=True,
        )

        m_dflat = create_master_sigmaclip(
            darkflats,
            source_root=src_root,
            output_root=args.masters_root,
            sigma=cfg.sigma_clip_sigma,
            maxiters=cfg.sigma_clip_maxiters,
            delete_inputs=args.delete_inputs,
            compressed=False,
            cfg=cfg,
            final_orient=args.final_orient_darkflats,
            target_shape=flat_shape,
            bayer_mode=True,
        )

    # 2) Corrected flat = Flat - DarkFlat
    flat_minus_darkflat = make_calibrated_flat(
        flat_master_path=m_flat,
        dark_flat_master_path=m_dflat,
        keep_neg_frac=cfg.keep_neg_frac,
        per_channel=(cfg.per_channel_shift and args.stack_mode == "demosaic-first"),
        output_root=args.masters_root,
        out_name=f"master_Flat_minus_DarkFlat_{args.stack_mode}.fits",
    )

    # 3) Flat coefficient
    flat_coeff_path = make_flat_coefficient(
        corrected_flat_path=flat_minus_darkflat,
        output_root=args.masters_root,
        out_name=f"flat_coefficient_{args.stack_mode}.fits",
        sigma=cfg.sigma_clip_sigma,
        maxiters=cfg.sigma_clip_maxiters,
        clamp=cfg.clamp,
    )

    # 4) Calibrate lights, align & stack
    light_to_dark = {lights: str(m_dark)}

    if args.stack_mode == "demosaic-first":
        final_stacks = batch_calibrate_fits_lights_demosaic(
            flat_coeff_path=flat_coeff_path,
            mapping=light_to_dark,
            in_root=src_root,
            out_root=args.out_root,
            cfg=cfg,
        )
    else:
        final_stacks = batch_calibrate_fits_lights_bayer(
            flat_coeff_path=flat_coeff_path,
            mapping=light_to_dark,
            in_root=src_root,
            out_root=args.out_root,
            cfg=cfg,
        )

    if args.plate_solve:
        if len(final_stacks) != 1:
            raise RuntimeError(
                f"Expected exactly one final stack for plate solving; got {len(final_stacks)}"
            )
        plate_solve_final_stack(final_stacks[0], args)


if __name__ == "__main__":
    main()
