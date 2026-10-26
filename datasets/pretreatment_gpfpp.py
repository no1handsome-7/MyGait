#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenGait GPF++ pretreatment.

This script upgrades the original 12-channel GPF into a 28-channel gait-native
field designed for GaitBase/HPP-style gait networks.  It intentionally keeps the
runtime dependency minimal: numpy + OpenCV.  RAFT/DINO/diffusion can later
replace the internal Farneback/structural-affinity functions without changing the
OpenGait IO contract.

Input folder assumption:
    rgb_root/subject/type/view/frame.png
    sil_root/subject/type/view/frame.png

Output folder:
    out_root/subject/type/view/
        000-gpf.pkl     : float16/float32 [T, 28, H, W]
        001-sils.pkl    : uint8           [T, 1,  H, W]
        002-phase.pkl   : float32         [T, 5]
                          [sin_phi, cos_phi, cycle_id, phase_quality, period]
        003-parts.pkl   : float16/float32 [T, K, H, W]
        004-quality.pkl : dict with frame-level and sequence-level quality

Core GPF++ channel order:
Geometry++, 10 channels:
  00 mask
  01 edge
  02 signed distance field
  03 normal x
  04 normal y
  05 curvature
  06 left boundary distance
  07 right boundary distance
  08 centerline distance
  09 body width profile
Motion++, 10 channels:
  10 flow dx relative
  11 flow dy relative
  12 flow magnitude
  13 flow direction sin
  14 flow direction cos
  15 motion amplitude
  16 motion acceleration
  17 forward-backward consistency
  18 periodic motion energy
  19 motion reliability
Frequency / cycle, 4 channels:
  20 phase sin, weak spatial cue
  21 phase cos, weak spatial cue
  22 gait-frequency energy
  23 second-harmonic energy
Appearance-suppressed residual / affinity, 4 channels:
  24 stable structural residual
  25 texture risk inverse
  26 affinity displacement x
  27 affinity displacement y
"""

import argparse
import json
import logging
import multiprocessing as mp
import pickle
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
GPFPP_CHANNELS = [
    "mask",
    "edge",
    "sdf",
    "normal_x",
    "normal_y",
    "curvature",
    "left_boundary_distance",
    "right_boundary_distance",
    "centerline_distance",
    "body_width_profile",
    "flow_dx_relative",
    "flow_dy_relative",
    "flow_magnitude",
    "flow_direction_sin",
    "flow_direction_cos",
    "motion_amplitude",
    "motion_acceleration",
    "forward_backward_consistency",
    "periodic_motion_energy",
    "motion_reliability",
    "phase_sin_weak",
    "phase_cos_weak",
    "gait_frequency_energy",
    "second_harmonic_energy",
    "stable_structural_residual",
    "texture_risk_inverse",
    "affinity_displacement_x",
    "affinity_displacement_y",
]


@dataclass(frozen=True)
class SeqKey:
    sid: str
    seq: str
    view: str

    def as_tuple(self) -> Tuple[str, str, str]:
        return self.sid, self.seq, self.view


def list_frames(root: Path) -> Dict[SeqKey, List[Path]]:
    groups: Dict[SeqKey, List[Path]] = defaultdict(list)
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        try:
            sid, seq, view = p.parts[-4], p.parts[-3], p.parts[-2]
        except IndexError:
            continue
        groups[SeqKey(sid, seq, view)].append(p)
    return {k: sorted(v) for k, v in groups.items()}


def pair_rgb_sil(rgb_paths: Sequence[Path], sil_paths: Sequence[Path]) -> List[Tuple[Path, Path]]:
    rgb_by_stem = {p.stem: p for p in rgb_paths}
    sil_by_stem = {p.stem: p for p in sil_paths}
    common = sorted(set(rgb_by_stem) & set(sil_by_stem))
    if common:
        return [(rgb_by_stem[s], sil_by_stem[s]) for s in common]
    n = min(len(rgb_paths), len(sil_paths))
    return list(zip(sorted(rgb_paths)[:n], sorted(sil_paths)[:n]))


def read_rgb(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def read_sil(path: Path, threshold: int) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return (img > threshold).astype(np.uint8) * 255


def _moving_average_1d(x: np.ndarray, radius: int = 2) -> np.ndarray:
    if len(x) <= 2 or radius <= 0:
        return x.astype(np.float32)
    k = np.ones(radius * 2 + 1, dtype=np.float32)
    k /= k.sum()
    pad = np.pad(x.astype(np.float32), (radius, radius), mode="edge")
    return np.convolve(pad, k, mode="valid").astype(np.float32)


def _foreground_bbox(mask_u8: np.ndarray, min_sil_area: int):
    mask = mask_u8 > 0
    if int(mask.sum()) < min_sil_area:
        return None
    ys, xs = np.where(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def sequence_align_by_sil(
    rgbs_bgr: Sequence[np.ndarray],
    sils_u8: Sequence[np.ndarray],
    out_h: int,
    out_w: int,
    min_sil_area: int,
    smooth_center_radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Silhouette-driven body canonicalization with sequence-level center smoothing."""
    records = []
    centers = []

    for rgb_bgr, sil_u8 in zip(rgbs_bgr, sils_u8):
        if rgb_bgr is None or sil_u8 is None:
            continue
        if rgb_bgr.shape[:2] != sil_u8.shape[:2]:
            rgb_bgr = cv2.resize(rgb_bgr, (sil_u8.shape[1], sil_u8.shape[0]), interpolation=cv2.INTER_LINEAR)
        bbox = _foreground_bbox(sil_u8, min_sil_area)
        if bbox is None:
            continue
        y0, y1, _, _ = bbox
        rgb_crop = rgb_bgr[y0:y1]
        sil_crop = sil_u8[y0:y1]
        crop_h, crop_w = sil_crop.shape[:2]
        if crop_h <= 1 or crop_w <= 1:
            continue
        new_w = max(1, int(round(crop_w * float(out_h) / float(crop_h))))
        rgb_rs = cv2.resize(rgb_crop, (new_w, out_h), interpolation=cv2.INTER_LINEAR)
        sil_rs = cv2.resize(sil_crop, (new_w, out_h), interpolation=cv2.INTER_NEAREST)
        col_mass = sil_rs.sum(axis=0).astype(np.float64)
        if col_mass.sum() <= 0:
            continue
        x_center = int(np.searchsorted(np.cumsum(col_mass), col_mass.sum() / 2.0))
        records.append((rgb_rs, sil_rs))
        centers.append(float(x_center))

    if not records:
        return np.empty((0, out_h, out_w, 3), dtype=np.uint8), np.empty((0, out_h, out_w), dtype=np.uint8)

    centers_np = _moving_average_1d(np.asarray(centers, dtype=np.float32), radius=smooth_center_radius)
    aligned_rgb, aligned_sil = [], []
    pad = out_w

    for (rgb_rs, sil_rs), x_center in zip(records, centers_np):
        rgb_pad = np.pad(rgb_rs, ((0, 0), (pad, pad), (0, 0)), mode="constant", constant_values=0)
        sil_pad = np.pad(sil_rs, ((0, 0), (pad, pad)), mode="constant", constant_values=0)
        xc = int(round(float(x_center))) + pad
        left = int(xc - out_w // 2)
        right = left + out_w
        rgb_out = rgb_pad[:, left:right]
        sil_out = sil_pad[:, left:right]
        if rgb_out.shape[:2] != (out_h, out_w) or sil_out.shape != (out_h, out_w):
            continue
        aligned_rgb.append(rgb_out.astype(np.uint8))
        aligned_sil.append(((sil_out > 0).astype(np.uint8) * 255))

    if not aligned_rgb:
        return np.empty((0, out_h, out_w, 3), dtype=np.uint8), np.empty((0, out_h, out_w), dtype=np.uint8)
    return np.stack(aligned_rgb, axis=0), np.stack(aligned_sil, axis=0)


def normalize01(x: np.ndarray, eps: float = 1e-6, percentile: Optional[float] = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if percentile is None:
        denom = float(np.max(np.abs(x)))
    else:
        denom = float(np.percentile(np.abs(x), percentile))
    if denom < eps:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip(x / denom, 0.0, 1.0).astype(np.float32)


def signed_distance_field(mask01: np.ndarray, clip_dist: float = 8.0) -> np.ndarray:
    mask_u8 = (mask01 > 0.5).astype(np.uint8)
    dist_in = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(1 - mask_u8, cv2.DIST_L2, 3)
    return np.clip((dist_in - dist_out) / clip_dist, -1.0, 1.0).astype(np.float32)


def body_edge(mask01: np.ndarray, ksize: int = 3) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)
    grad = cv2.morphologyEx((mask01 > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    return grad.astype(np.float32)


def contour_normals_and_curvature(sdf: np.ndarray, edge: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Sobel(sdf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sdf, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx = np.clip(gx / mag, -1.0, 1.0)
    ny = np.clip(gy / mag, -1.0, 1.0)
    div_x = cv2.Sobel(nx, cv2.CV_32F, 1, 0, ksize=3)
    div_y = cv2.Sobel(ny, cv2.CV_32F, 0, 1, ksize=3)
    curv = np.clip((div_x + div_y) / 4.0, -1.0, 1.0)
    edge_d = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(np.float32)
    return (nx * edge_d).astype(np.float32), (ny * edge_d).astype(np.float32), (curv * edge_d).astype(np.float32)


def boundary_coordinate_fields(mask01: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = mask01.shape
    left_d = np.zeros((h, w), dtype=np.float32)
    right_d = np.zeros((h, w), dtype=np.float32)
    center_d = np.zeros((h, w), dtype=np.float32)
    width_map = np.zeros((h, w), dtype=np.float32)
    xs_all = np.arange(w, dtype=np.float32)
    for y in range(h):
        xs = np.where(mask01[y] > 0.5)[0]
        if len(xs) == 0:
            continue
        left, right = float(xs[0]), float(xs[-1])
        width = max(right - left + 1.0, 1.0)
        center = 0.5 * (left + right)
        inside = mask01[y] > 0.5
        left_d[y, inside] = np.clip((xs_all[inside] - left) / width, 0.0, 1.0)
        right_d[y, inside] = np.clip((right - xs_all[inside]) / width, 0.0, 1.0)
        center_d[y, inside] = np.clip((xs_all[inside] - center) / (0.5 * width), -1.0, 1.0)
        width_map[y, inside] = np.clip(width / float(w), 0.0, 1.0)
    return left_d, right_d, center_d, width_map


def estimate_phase_meta(masks01: np.ndarray, min_period: int = 8, max_period: int = 48) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    """Return phase_info [T,5], metadata and the lower-body centroid signal."""
    t_len, h, w = masks01.shape
    if t_len == 0:
        return np.zeros((0, 5), dtype=np.float32), {"period": 0.0, "phase_quality": 0.0, "freq_bin": 0}, np.zeros((0,), dtype=np.float32)

    lower = masks01[:, h // 2 :, :]
    xs = np.arange(w, dtype=np.float32)[None, None, :]
    mass = lower.sum(axis=(1, 2)) + 1e-6
    centroid = (lower * xs).sum(axis=(1, 2)) / mass
    signal = centroid - np.mean(centroid)
    k_best = 0
    quality = 0.0

    if t_len >= max(4, min_period) and float(np.std(signal)) > 1e-4:
        fft = np.fft.rfft(signal)
        mag = np.abs(fft)
        bins = np.arange(len(mag))
        valid = bins > 0
        periods = np.zeros_like(bins, dtype=np.float32)
        periods[valid] = float(t_len) / bins[valid]
        valid = valid & (periods >= min_period) & (periods <= max_period)
        if valid.any():
            scores = mag * valid
            k_best = int(np.argmax(scores))
            sorted_scores = np.sort(scores[valid])
            top1 = float(sorted_scores[-1])
            top2 = float(sorted_scores[-2]) if len(sorted_scores) > 1 else 0.0
            quality = float(top1 / (top1 + top2 + 1e-6))
            phi = float(np.angle(fft[k_best]))
            period = float(t_len) / float(k_best)
            phase = 2.0 * np.pi * k_best * np.arange(t_len, dtype=np.float32) / float(t_len) + phi
        else:
            period = float(np.clip(t_len, min_period, max_period))
            phase = 2.0 * np.pi * np.arange(t_len, dtype=np.float32) / period
    else:
        period = float(np.clip(max(t_len, 1), min_period, max_period))
        phase = 2.0 * np.pi * np.arange(t_len, dtype=np.float32) / period

    unwrapped = np.unwrap(phase.astype(np.float32))
    cycle_id = np.floor((unwrapped - unwrapped.min()) / (2.0 * np.pi)).astype(np.float32)
    phase_info = np.stack(
        [
            np.sin(phase),
            np.cos(phase),
            cycle_id,
            np.full(t_len, quality, dtype=np.float32),
            np.full(t_len, period, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return phase_info, {"period": period, "phase_quality": quality, "freq_bin": int(k_best)}, signal.astype(np.float32)


def _masked_gray(rgbs_bgr: np.ndarray, masks01: np.ndarray) -> np.ndarray:
    gray = np.stack([cv2.cvtColor(rgbs_bgr[t], cv2.COLOR_BGR2GRAY) for t in range(rgbs_bgr.shape[0])], axis=0)
    return (gray.astype(np.float32) / 255.0) * masks01


def motion_amplitude(gray01: np.ndarray, masks01: np.ndarray) -> np.ndarray:
    t_len = gray01.shape[0]
    if t_len <= 1:
        return np.zeros_like(gray01, dtype=np.float32)
    amp = np.zeros_like(gray01, dtype=np.float32)
    for t in range(t_len):
        vals = []
        if t > 0:
            vals.append(np.abs(gray01[t] - gray01[t - 1]))
        if t < t_len - 1:
            vals.append(np.abs(gray01[t + 1] - gray01[t]))
        amp[t] = np.mean(vals, axis=0) if vals else 0.0
    return normalize01(amp * masks01, percentile=95)


def motion_acceleration_from_amp(amp: np.ndarray, masks01: np.ndarray) -> np.ndarray:
    if amp.shape[0] <= 2:
        return np.zeros_like(amp, dtype=np.float32)
    acc = np.zeros_like(amp, dtype=np.float32)
    acc[1:-1] = amp[2:] - 2.0 * amp[1:-1] + amp[:-2]
    denom = max(float(np.percentile(np.abs(acc), 95)), 1e-6)
    return np.clip(acc / denom, -1.0, 1.0).astype(np.float32) * masks01


def _torso_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    vertical = (yy > 0.22) & (yy < 0.58)
    central = np.abs(xx) < 0.45
    return (mask > 0.5) & vertical & central


def optical_flow_relative(
    frames01: np.ndarray,
    masks01: np.ndarray,
    flow_clip: float = 4.0,
    winsize: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward/backward checked relative flow in canonical body coordinates."""
    t_len, h, w = frames01.shape
    dx = np.zeros((t_len, h, w), dtype=np.float32)
    dy = np.zeros((t_len, h, w), dtype=np.float32)
    mag = np.zeros((t_len, h, w), dtype=np.float32)
    fb = np.zeros((t_len, h, w), dtype=np.float32)

    if t_len <= 1:
        return dx, dy, mag, dx.copy(), dy.copy(), fb

    frames = np.clip(frames01 * 255.0, 0, 255).astype(np.uint8)
    raw_fx = np.zeros_like(dx)
    raw_fy = np.zeros_like(dy)
    raw_bx = np.zeros_like(dx)
    raw_by = np.zeros_like(dy)

    for t in range(1, t_len):
        fwd = cv2.calcOpticalFlowFarneback(
            frames[t - 1], frames[t], None, 0.5, 3, winsize, 3, 5, 1.1, 0
        )
        bwd = cv2.calcOpticalFlowFarneback(
            frames[t], frames[t - 1], None, 0.5, 3, winsize, 3, 5, 1.1, 0
        )
        raw_fx[t], raw_fy[t] = fwd[..., 0], fwd[..., 1]
        raw_bx[t], raw_by[t] = bwd[..., 0], bwd[..., 1]

        tm = _torso_mask(masks01[t])
        if int(tm.sum()) < 4:
            tm = masks01[t] > 0.5
        gx = float(np.median(raw_fx[t][tm])) if int(tm.sum()) > 0 else 0.0
        gy = float(np.median(raw_fy[t][tm])) if int(tm.sum()) > 0 else 0.0
        rel_x = raw_fx[t] - gx
        rel_y = raw_fy[t] - gy
        dx[t] = np.clip(rel_x / flow_clip, -1.0, 1.0) * masks01[t]
        dy[t] = np.clip(rel_y / flow_clip, -1.0, 1.0) * masks01[t]
        mag[t] = np.sqrt(rel_x * rel_x + rel_y * rel_y) * masks01[t]
        cycle_err = np.sqrt((raw_fx[t] + raw_bx[t]) ** 2 + (raw_fy[t] + raw_by[t]) ** 2)
        fb[t] = np.exp(-cycle_err / 1.5).astype(np.float32) * masks01[t]

    dx[0], dy[0], mag[0], fb[0] = dx[1], dy[1], mag[1], fb[1]
    mag01 = normalize01(mag, percentile=95) * masks01
    return dx.astype(np.float32), dy.astype(np.float32), mag01.astype(np.float32), raw_fx, raw_fy, fb.astype(np.float32)


def periodic_energy_maps(signal: np.ndarray, masks01: np.ndarray, freq_bin: int, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel-wise FFT energy at gait fundamental and second harmonic."""
    t_len = signal.shape[0]
    if t_len < 4 or freq_bin <= 0:
        z = np.zeros_like(signal, dtype=np.float32)
        return z, z, z
    x = signal.astype(np.float32) * masks01
    x = x - x.mean(axis=0, keepdims=True)
    fft = np.fft.rfft(x, axis=0)
    total = np.sqrt(np.sum(np.abs(fft[1:]) ** 2, axis=0)).astype(np.float32) + eps
    k1 = int(np.clip(freq_bin, 1, fft.shape[0] - 1))
    k2 = int(np.clip(freq_bin * 2, 1, fft.shape[0] - 1))
    e1 = (np.abs(fft[k1]).astype(np.float32) / total) * masks01.mean(axis=0)
    e2 = (np.abs(fft[k2]).astype(np.float32) / total) * masks01.mean(axis=0)
    e1 = normalize01(e1, percentile=98)
    e2 = normalize01(e2, percentile=98)
    e1_t = np.broadcast_to(e1[None], signal.shape).astype(np.float32) * masks01
    e2_t = np.broadcast_to(e2[None], signal.shape).astype(np.float32) * masks01
    return e1_t, e2_t, np.maximum(e1_t, e2_t).astype(np.float32)


def structural_residual_and_texture_risk(rgbs_bgr: np.ndarray, masks01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Appearance-invariance tested residual: consensus is kept, variant-sensitive texture is suppressed."""
    t_len, h, w = masks01.shape
    residuals = []
    for t in range(t_len):
        bgr = rgbs_bgr[t]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        lab_l = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) / 255.0
        hsv_v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32) / 255.0
        smooth = cv2.bilateralFilter((gray * 255).astype(np.uint8), d=5, sigmaColor=35, sigmaSpace=3).astype(np.float32) / 255.0
        variants = [gray, lab_l, hsv_v, smooth]
        per_frame = []
        for v in variants:
            low = cv2.GaussianBlur(v, (0, 0), sigmaX=2.0, sigmaY=2.0)
            res = (v - low) * masks01[t]
            denom = max(float(np.percentile(np.abs(res), 95)), 1e-6)
            per_frame.append(np.clip(res / denom, -1.0, 1.0))
        residuals.append(np.stack(per_frame, axis=0))
    stack = np.stack(residuals, axis=0).astype(np.float32)  # [T,V,H,W]
    stable = stack.mean(axis=1)
    risk = stack.std(axis=1)
    risk01 = normalize01(risk, percentile=95) * masks01
    inv = (1.0 - risk01).clip(0.0, 1.0) * masks01
    stable = np.clip(stable * inv, -1.0, 1.0).astype(np.float32)
    return stable, inv.astype(np.float32)


def part_maps_from_body_coordinates(masks01: np.ndarray, centerline: np.ndarray, num_parts: int = 8, sigma_scale: float = 0.45) -> np.ndarray:
    """Soft body-coordinate maps. They are priors for pooling, not hard parsing labels."""
    t_len, h, w = masks01.shape
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    centers = np.linspace(0.07, 0.93, num_parts, dtype=np.float32)
    sigma = max(0.04, 1.0 / max(num_parts, 1) * sigma_scale)
    parts = np.zeros((t_len, num_parts, h, w), dtype=np.float32)
    for k, c in enumerate(centers):
        vertical = np.exp(-0.5 * ((yy - c) / sigma) ** 2).astype(np.float32)
        vertical /= vertical.max() + 1e-6
        # Slightly emphasize side regions in the arm/leg zones, central regions in torso zones.
        if 0.20 < c < 0.55:
            lateral = 0.65 + 0.35 * np.clip(np.abs(xx), 0, 1)
        elif c >= 0.55:
            lateral = 0.75 + 0.25 * np.clip(np.abs(xx), 0, 1)
        else:
            lateral = np.ones((1, w), dtype=np.float32)
        parts[:, k] = masks01 * vertical[None] * lateral[None]
    s = parts.sum(axis=1, keepdims=True) + 1e-6
    return (parts / s * masks01[:, None]).astype(np.float32)


def build_gpfpp(
    rgbs_bgr: np.ndarray,
    sils_u8: np.ndarray,
    num_parts: int,
    min_period: int,
    max_period: int,
    flow_clip: float,
    phase_spatial_strength: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    t_len, h, w = sils_u8.shape
    masks01 = (sils_u8 > 0).astype(np.float32)
    gray01 = _masked_gray(rgbs_bgr, masks01)

    # Geometry++
    edges = np.zeros_like(masks01, dtype=np.float32)
    sdfs = np.zeros_like(masks01, dtype=np.float32)
    nxs = np.zeros_like(masks01, dtype=np.float32)
    nys = np.zeros_like(masks01, dtype=np.float32)
    curvs = np.zeros_like(masks01, dtype=np.float32)
    left_d = np.zeros_like(masks01, dtype=np.float32)
    right_d = np.zeros_like(masks01, dtype=np.float32)
    center_d = np.zeros_like(masks01, dtype=np.float32)
    width_p = np.zeros_like(masks01, dtype=np.float32)
    for t in range(t_len):
        edges[t] = body_edge(masks01[t])
        sdfs[t] = signed_distance_field(masks01[t])
        nxs[t], nys[t], curvs[t] = contour_normals_and_curvature(sdfs[t], edges[t])
        left_d[t], right_d[t], center_d[t], width_p[t] = boundary_coordinate_fields(masks01[t])

    # Phase/cycle metadata. Phase remains weak and primarily supports temporal aggregation.
    phase, phase_meta, _ = estimate_phase_meta(masks01, min_period=min_period, max_period=max_period)
    phase_sin_map = phase_spatial_strength * phase[:, 0, None, None].astype(np.float32) * masks01
    phase_cos_map = phase_spatial_strength * phase[:, 1, None, None].astype(np.float32) * masks01

    # Motion++
    amp = motion_amplitude(gray01, masks01)
    acc = motion_acceleration_from_amp(amp, masks01)
    dx, dy, flow_mag, _, _, fb_cons = optical_flow_relative(gray01, masks01, flow_clip=flow_clip)
    dir_norm = np.sqrt(dx * dx + dy * dy) + 1e-6
    dir_sin = np.clip(dy / dir_norm, -1.0, 1.0) * masks01
    dir_cos = np.clip(dx / dir_norm, -1.0, 1.0) * masks01
    gait_e, harmonic_e, periodic_motion = periodic_energy_maps(amp, masks01, int(phase_meta.get("freq_bin", 0)))
    motion_reliability = (0.50 * fb_cons + 0.30 * periodic_motion + 0.20 * masks01).clip(0.0, 1.0) * masks01

    # Appearance-suppressed residual / affinity.
    stable_residual, texture_inv = structural_residual_and_texture_risk(rgbs_bgr, masks01)
    structural_frames = np.clip(0.50 * edges + 0.30 * np.abs(stable_residual) + 0.20 * masks01, 0.0, 1.0)
    aff_dx, aff_dy, _, _, _, aff_fb = optical_flow_relative(structural_frames, masks01, flow_clip=flow_clip, winsize=7)
    aff_dx = aff_dx * (0.5 + 0.5 * aff_fb)
    aff_dy = aff_dy * (0.5 + 0.5 * aff_fb)

    gpf = np.stack(
        [
            masks01,
            edges,
            sdfs,
            nxs,
            nys,
            curvs,
            left_d,
            right_d,
            center_d,
            width_p,
            dx,
            dy,
            flow_mag,
            dir_sin,
            dir_cos,
            amp,
            acc,
            fb_cons,
            periodic_motion,
            motion_reliability,
            phase_sin_map,
            phase_cos_map,
            gait_e,
            harmonic_e,
            stable_residual,
            texture_inv,
            aff_dx,
            aff_dy,
        ],
        axis=1,
    ).astype(np.float32)

    parts = part_maps_from_body_coordinates(masks01, center_d, num_parts=num_parts)

    area = masks01.mean(axis=(1, 2)).astype(np.float32)
    area_med = float(np.median(area)) if len(area) else 0.0
    mask_quality = np.clip(area / max(area_med, 1e-6), 0.0, 1.0).astype(np.float32)
    flow_quality = fb_cons.sum(axis=(1, 2)) / (masks01.sum(axis=(1, 2)) + 1e-6)
    occlusion_score = np.clip(1.0 - mask_quality, 0.0, 1.0).astype(np.float32)
    texture_risk = 1.0 - (texture_inv.sum(axis=(1, 2)) / (masks01.sum(axis=(1, 2)) + 1e-6))
    phase_quality = np.full(t_len, float(phase_meta.get("phase_quality", 0.0)), dtype=np.float32)
    quality_frame = np.stack([mask_quality, flow_quality, phase_quality, occlusion_score, texture_risk], axis=1).astype(np.float32)
    quality = {
        "frame_names": ["mask_quality", "flow_quality", "phase_quality", "occlusion_score", "texture_risk"],
        "frame": quality_frame,
        "sequence": {
            "mask_quality": float(mask_quality.mean()) if t_len else 0.0,
            "flow_quality": float(flow_quality.mean()) if t_len else 0.0,
            "phase_quality": float(phase_meta.get("phase_quality", 0.0)),
            "occlusion_score": float(occlusion_score.mean()) if t_len else 0.0,
            "texture_risk": float(texture_risk.mean()) if t_len else 0.0,
            "period": float(phase_meta.get("period", 0.0)),
            "freq_bin": int(phase_meta.get("freq_bin", 0)),
        },
    }
    return gpf, phase.astype(np.float32), parts.astype(np.float32), quality


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def process_one_sequence(
    item: Tuple[SeqKey, List[Tuple[Path, Path]]],
    out_root: Path,
    out_h: int,
    out_w: int,
    num_parts: int,
    sil_threshold: int,
    min_sil_area: int,
    min_period: int,
    max_period: int,
    flow_clip: float,
    phase_spatial_strength: float,
    dtype: str,
    overwrite: bool,
) -> Tuple[SeqKey, int, str]:
    key, pairs = item
    sid, seq, view = key.as_tuple()
    out_dir = out_root / sid / seq / view
    paths = {
        "gpf": out_dir / "000-gpf.pkl",
        "sils": out_dir / "001-sils.pkl",
        "phase": out_dir / "002-phase.pkl",
        "parts": out_dir / "003-parts.pkl",
        "quality": out_dir / "004-quality.pkl",
    }
    if not overwrite and all(p.exists() for p in paths.values()):
        return key, -1, "skip"

    rgbs, sils = [], []
    for rgb_p, sil_p in pairs:
        rgb = read_rgb(rgb_p)
        sil = read_sil(sil_p, sil_threshold)
        if rgb is None or sil is None:
            continue
        rgbs.append(rgb)
        sils.append(sil)
    if len(rgbs) == 0:
        return key, 0, "empty"

    rgbs_np, sils_np = sequence_align_by_sil(
        rgbs, sils, out_h=out_h, out_w=out_w, min_sil_area=min_sil_area
    )
    if rgbs_np.shape[0] == 0:
        return key, 0, "empty"

    gpf, phase, parts, quality = build_gpfpp(
        rgbs_bgr=rgbs_np,
        sils_u8=sils_np,
        num_parts=num_parts,
        min_period=min_period,
        max_period=max_period,
        flow_clip=flow_clip,
        phase_spatial_strength=phase_spatial_strength,
    )

    if dtype == "float16":
        gpf_to_save = gpf.astype(np.float16)
        parts_to_save = parts.astype(np.float16)
    elif dtype == "float32":
        gpf_to_save = gpf.astype(np.float32)
        parts_to_save = parts.astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    save_pickle(gpf_to_save, paths["gpf"])
    save_pickle(sils_np[:, None].astype(np.uint8), paths["sils"])
    save_pickle(phase.astype(np.float32), paths["phase"])
    save_pickle(parts_to_save, paths["parts"])
    save_pickle(quality, paths["quality"])

    meta = {
        "sid": sid,
        "seq": seq,
        "view": view,
        "num_valid_frames": int(rgbs_np.shape[0]),
        "gpf_shape": list(gpf_to_save.shape),
        "sils_shape": list(sils_np[:, None].shape),
        "phase_shape": list(phase.shape),
        "parts_shape": list(parts_to_save.shape),
        "quality_frame_shape": list(quality["frame"].shape),
        "gpf_channels": GPFPP_CHANNELS,
        "phase_columns": ["sin_phi", "cos_phi", "cycle_id", "phase_quality", "period"],
        "quality_frame_columns": quality["frame_names"],
        "note": "004-quality.pkl is diagnostic metadata; keep data_in_use false unless a custom transform consumes it.",
    }
    with open(out_dir / "meta-gpfpp.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return key, int(rgbs_np.shape[0]), "ok"


def pretreat_gpfpp(
    rgb_root: Path,
    sil_root: Path,
    out_root: Path,
    out_h: int = 64,
    out_w: int = 32,
    num_parts: int = 8,
    sil_threshold: int = 10,
    min_sil_area: int = 20,
    min_period: int = 8,
    max_period: int = 48,
    flow_clip: float = 4.0,
    phase_spatial_strength: float = 0.25,
    dtype: str = "float16",
    workers: int = 4,
    overwrite: bool = False,
) -> None:
    logging.info("Listing RGB frames from %s", rgb_root)
    rgb_groups = list_frames(rgb_root)
    logging.info("Listing silhouette frames from %s", sil_root)
    sil_groups = list_frames(sil_root)
    common_keys = sorted(set(rgb_groups) & set(sil_groups), key=lambda k: k.as_tuple())
    jobs = []
    for key in common_keys:
        pairs = pair_rgb_sil(rgb_groups[key], sil_groups[key])
        if pairs:
            jobs.append((key, pairs))
    logging.info("Prepared %d paired sequence jobs.", len(jobs))

    worker_fn = partial(
        process_one_sequence,
        out_root=out_root,
        out_h=out_h,
        out_w=out_w,
        num_parts=num_parts,
        sil_threshold=sil_threshold,
        min_sil_area=min_sil_area,
        min_period=min_period,
        max_period=max_period,
        flow_clip=flow_clip,
        phase_spatial_strength=phase_spatial_strength,
        dtype=dtype,
        overwrite=overwrite,
    )

    ok = skip = empty = 0
    if workers <= 1:
        iterator = map(worker_fn, jobs)
        for _, _, status in tqdm(iterator, total=len(jobs), desc="GPF++ pretreat"):
            ok += int(status == "ok")
            skip += int(status == "skip")
            empty += int(status == "empty")
    else:
        with mp.Pool(workers) as pool:
            for _, _, status in tqdm(pool.imap_unordered(worker_fn, jobs), total=len(jobs), desc="GPF++ pretreat"):
                ok += int(status == "ok")
                skip += int(status == "skip")
                empty += int(status == "empty")
    logging.info("Done. ok=%d, skip=%d, empty=%d", ok, skip, empty)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("OpenGait GPF++ pretreatment")
    parser.add_argument("--rgb_root", required=True, type=str, help="RGB root: subject/type/view/frame.png")
    parser.add_argument("--sil_root", required=True, type=str, help="Silhouette root: subject/type/view/frame.png")
    parser.add_argument("--output_root", required=True, type=str, help="Output OpenGait-style pkl root")
    parser.add_argument("--img_h", default=64, type=int)
    parser.add_argument("--img_w", default=32, type=int)
    parser.add_argument("--num_parts", default=8, type=int)
    parser.add_argument("--sil_threshold", default=10, type=int)
    parser.add_argument("--min_sil_area", default=20, type=int)
    parser.add_argument("--min_period", default=8, type=int)
    parser.add_argument("--max_period", default=48, type=int)
    parser.add_argument("--flow_clip", default=4.0, type=float)
    parser.add_argument("--phase_spatial_strength", default=0.25, type=float)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_file", default="pretreatment_gpfpp.log", type=str)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        filename=args.log_file,
        filemode="w",
        format="[%(asctime)s - %(levelname)s]: %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    pretreat_gpfpp(
        rgb_root=Path(args.rgb_root),
        sil_root=Path(args.sil_root),
        out_root=Path(args.output_root),
        out_h=args.img_h,
        out_w=args.img_w,
        num_parts=args.num_parts,
        sil_threshold=args.sil_threshold,
        min_sil_area=args.min_sil_area,
        min_period=args.min_period,
        max_period=args.max_period,
        flow_clip=args.flow_clip,
        phase_spatial_strength=args.phase_spatial_strength,
        dtype=args.dtype,
        workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
