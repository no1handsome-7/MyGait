#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenGait GPF pretreatment.

Create a gait-native representation from RGB frames + silhouettes.

Input folder assumption:
    rgb_root/subject/type/view/frame.png
    sil_root/subject/type/view/frame.png

Output folder:
    out_root/subject/type/view/
        000-gpf.pkl    : np.ndarray, float16/float32, [T, C, H, W]
        001-sils.pkl   : np.ndarray, uint8,          [T, 1, H, W]
        002-phase.pkl  : np.ndarray, float32,        [T, 2]
        003-parts.pkl  : np.ndarray, float16/float32,[T, K, H, W]

Default GPF channel order:
    00 silhouette mask            [0, 1]
    01 body edge                  [0, 1]
    02 contour normal x           [-1, 1]
    03 contour normal y           [-1, 1]
    04 signed distance field      [-1, 1]
    05 motion amplitude           [0, 1]
    06 phase sin                  [-1, 1]
    07 phase cos                  [-1, 1]
    08 optical-flow dx            [-1, 1]
    09 optical-flow dy            [-1, 1]
    10 motion reliability         [0, 1]
    11 mid-frequency residual     [-1, 1]

This file intentionally uses only numpy + opencv, so it can be run before
any model-side training/inference code is added.
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class SeqKey:
    sid: str
    seq: str
    view: str

    def as_tuple(self) -> Tuple[str, str, str]:
        return self.sid, self.seq, self.view


def list_frames(root: Path) -> Dict[SeqKey, List[Path]]:
    """Collect frames with OpenGait-style subject/type/view hierarchy."""
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
    """Pair RGB and silhouette frames by filename stem; fallback to sorted zip."""
    rgb_by_stem = {p.stem: p for p in rgb_paths}
    sil_by_stem = {p.stem: p for p in sil_paths}
    common = sorted(set(rgb_by_stem) & set(sil_by_stem))
    if common:
        return [(rgb_by_stem[s], sil_by_stem[s]) for s in common]
    n = min(len(rgb_paths), len(sil_paths))
    return list(zip(sorted(rgb_paths)[:n], sorted(sil_paths)[:n]))


def read_rgb(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


def read_sil(path: Path, threshold: int) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return (img > threshold).astype(np.uint8) * 255


def align_by_sil(
    rgb_bgr: np.ndarray,
    sil_u8: np.ndarray,
    out_h: int,
    out_w: int,
    min_sil_area: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    OpenGait-like body alignment driven by silhouette:
    vertical crop by foreground, resize to target height preserving ratio,
    then horizontal center crop/pad by silhouette mass center.
    """
    if rgb_bgr.shape[:2] != sil_u8.shape[:2]:
        rgb_bgr = cv2.resize(rgb_bgr, (sil_u8.shape[1], sil_u8.shape[0]), interpolation=cv2.INTER_LINEAR)

    mask = sil_u8 > 0
    if int(mask.sum()) < min_sil_area:
        return None

    rows = np.where(mask.sum(axis=1) > 0)[0]
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    rgb_crop = rgb_bgr[y0:y1]
    sil_crop = sil_u8[y0:y1]
    crop_h, crop_w = sil_crop.shape[:2]
    if crop_h <= 1 or crop_w <= 1:
        return None

    new_w = max(1, int(round(crop_w * float(out_h) / float(crop_h))))
    rgb_rs = cv2.resize(rgb_crop, (new_w, out_h), interpolation=cv2.INTER_LINEAR)
    sil_rs = cv2.resize(sil_crop, (new_w, out_h), interpolation=cv2.INTER_NEAREST)

    col_mass = sil_rs.sum(axis=0).astype(np.float64)
    if col_mass.sum() <= 0:
        return None
    x_center = int(np.searchsorted(np.cumsum(col_mass), col_mass.sum() / 2.0))

    # Pad generously, then crop a fixed-width window.
    pad = out_w
    rgb_pad = np.pad(rgb_rs, ((0, 0), (pad, pad), (0, 0)), mode="constant", constant_values=0)
    sil_pad = np.pad(sil_rs, ((0, 0), (pad, pad)), mode="constant", constant_values=0)

    x_center += pad
    left = int(x_center - out_w // 2)
    right = left + out_w
    rgb_out = rgb_pad[:, left:right]
    sil_out = sil_pad[:, left:right]

    if rgb_out.shape[:2] != (out_h, out_w) or sil_out.shape != (out_h, out_w):
        return None

    sil_out = ((sil_out > 0).astype(np.uint8) * 255)
    return rgb_out, sil_out


def normalize01(x: np.ndarray, eps: float = 1e-6, percentile: Optional[float] = None) -> np.ndarray:
    x = x.astype(np.float32)
    if percentile is None:
        denom = float(x.max())
    else:
        denom = float(np.percentile(x, percentile))
    if denom < eps:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip(x / denom, 0.0, 1.0).astype(np.float32)


def signed_distance_field(mask01: np.ndarray, clip_dist: float = 8.0) -> np.ndarray:
    mask_u8 = (mask01 > 0.5).astype(np.uint8)
    dist_in = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(1 - mask_u8, cv2.DIST_L2, 3)
    sdf = np.clip((dist_in - dist_out) / clip_dist, -1.0, 1.0)
    return sdf.astype(np.float32)


def contour_normals_from_sdf(sdf: np.ndarray, edge_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    gx = cv2.Sobel(sdf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sdf, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx = np.clip(gx / mag, -1.0, 1.0)
    ny = np.clip(gy / mag, -1.0, 1.0)
    if edge_mask is not None:
        nx = nx * edge_mask
        ny = ny * edge_mask
    return nx.astype(np.float32), ny.astype(np.float32)


def body_edge(mask01: np.ndarray, ksize: int = 3) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)
    grad = cv2.morphologyEx((mask01 > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    return grad.astype(np.float32)


def estimate_phase(
    masks01: np.ndarray,
    min_period: int = 8,
    max_period: int = 40,
) -> np.ndarray:
    """
    Estimate a rough walking phase from the horizontal centroid trajectory of
    the lower-body silhouette. Returns [T, 2] = [sin(phase), cos(phase)].
    """
    t_len, h, w = masks01.shape
    if t_len == 0:
        return np.zeros((0, 2), dtype=np.float32)

    lower = masks01[:, h // 2 :, :]
    xs = np.arange(w, dtype=np.float32)[None, None, :]
    mass = lower.sum(axis=(1, 2)) + 1e-6
    centroid = (lower * xs).sum(axis=(1, 2)) / mass
    signal = centroid - np.mean(centroid)

    if t_len >= max(4, min_period) and float(np.std(signal)) > 1e-4:
        fft = np.fft.rfft(signal)
        mag = np.abs(fft)
        bins = np.arange(len(mag))
        valid = bins > 0
        periods = np.zeros_like(bins, dtype=np.float32)
        periods[valid] = float(t_len) / bins[valid]
        valid = valid & (periods >= min_period) & (periods <= max_period)
        if valid.any():
            k = int(np.argmax(mag * valid))
            phi = float(np.angle(fft[k]))
            phase = 2.0 * np.pi * k * np.arange(t_len, dtype=np.float32) / float(t_len) + phi
        else:
            period = float(np.clip(t_len, min_period, max_period))
            phase = 2.0 * np.pi * np.arange(t_len, dtype=np.float32) / period
    else:
        period = float(np.clip(max(t_len, 1), min_period, max_period))
        phase = 2.0 * np.pi * np.arange(t_len, dtype=np.float32) / period

    return np.stack([np.sin(phase), np.cos(phase)], axis=1).astype(np.float32)


def motion_amplitude(gray01: np.ndarray, masks01: np.ndarray) -> np.ndarray:
    """Symmetric frame-difference motion cue, normalized sequence-wise."""
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
    amp = amp * masks01
    return normalize01(amp, percentile=95)


def optical_flow_fields(gray01: np.ndarray, masks01: np.ndarray, flow_clip: float = 4.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dense Farneback flow on aligned masked grayscale frames.
    dx/dy are clipped and scaled to [-1, 1]; reliability is [0, 1].
    """
    t_len, h, w = gray01.shape
    dx = np.zeros((t_len, h, w), dtype=np.float32)
    dy = np.zeros((t_len, h, w), dtype=np.float32)
    mag = np.zeros((t_len, h, w), dtype=np.float32)

    if t_len <= 1:
        return dx, dy, mag

    frames = np.clip(gray01 * masks01 * 255.0, 0, 255).astype(np.uint8)
    for t in range(1, t_len):
        flow = cv2.calcOpticalFlowFarneback(
            frames[t - 1],
            frames[t],
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=7,
            iterations=3,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )
        fx = np.clip(flow[..., 0] / flow_clip, -1.0, 1.0)
        fy = np.clip(flow[..., 1] / flow_clip, -1.0, 1.0)
        dx[t] = fx * masks01[t]
        dy[t] = fy * masks01[t]
        mag[t] = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2) * masks01[t]

    dx[0], dy[0], mag[0] = dx[1], dy[1], mag[1]
    rel = normalize01(mag, percentile=95) * masks01
    return dx.astype(np.float32), dy.astype(np.float32), rel.astype(np.float32)


def mid_frequency_residual(gray01: np.ndarray, masks01: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Signed mid/high residual after Gaussian smoothing, clipped to [-1, 1]."""
    out = np.zeros_like(gray01, dtype=np.float32)
    for t in range(gray01.shape[0]):
        low = cv2.GaussianBlur(gray01[t], (0, 0), sigmaX=sigma, sigmaY=sigma)
        res = (gray01[t] - low) * masks01[t]
        # Robust scaling per frame; keep sign.
        denom = max(float(np.percentile(np.abs(res), 95)), 1e-6)
        out[t] = np.clip(res / denom, -1.0, 1.0)
    return out.astype(np.float32)


def part_maps_from_masks(masks01: np.ndarray, num_parts: int = 6, sigma_scale: float = 0.55) -> np.ndarray:
    """
    Soft HPP/body-slot maps. This is deliberately stable and annotation-free.
    Later, this can be replaced by parsing/slot attention without changing IO.
    """
    t_len, h, w = masks01.shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    band_h = float(h) / float(num_parts)
    parts = np.zeros((t_len, num_parts, h, w), dtype=np.float32)
    sigma = max(1.0, band_h * sigma_scale)

    for k in range(num_parts):
        center = (k + 0.5) * band_h
        stripe = np.exp(-0.5 * ((yy - center) / sigma) ** 2).astype(np.float32)
        stripe = stripe / (stripe.max() + 1e-6)
        parts[:, k] = masks01 * stripe[None, :, :]
    return parts


def build_gpf(
    rgbs_bgr: np.ndarray,
    sils_u8: np.ndarray,
    num_parts: int,
    min_period: int,
    max_period: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        rgbs_bgr: [T, H, W, 3], uint8, aligned.
        sils_u8 : [T, H, W], uint8 {0,255}, aligned.

    Returns:
        gpf   : [T, 12, H, W], float32
        phase : [T, 2], float32
        parts : [T, K, H, W], float32
    """
    t_len, h, w = sils_u8.shape
    masks01 = (sils_u8 > 0).astype(np.float32)

    gray = np.stack([cv2.cvtColor(rgbs_bgr[t], cv2.COLOR_BGR2GRAY) for t in range(t_len)], axis=0)
    gray01 = gray.astype(np.float32) / 255.0

    edges = np.zeros_like(masks01, dtype=np.float32)
    sdfs = np.zeros_like(masks01, dtype=np.float32)
    nxs = np.zeros_like(masks01, dtype=np.float32)
    nys = np.zeros_like(masks01, dtype=np.float32)

    for t in range(t_len):
        edges[t] = body_edge(masks01[t])
        sdfs[t] = signed_distance_field(masks01[t])
        # Slightly dilate edge to make contour normals visible to small CNN stems.
        edge_for_normal = cv2.dilate(edges[t].astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(np.float32)
        nxs[t], nys[t] = contour_normals_from_sdf(sdfs[t], edge_for_normal)

    phase = estimate_phase(masks01, min_period=min_period, max_period=max_period)
    phase_sin_map = np.broadcast_to(phase[:, 0, None, None], (t_len, h, w)).astype(np.float32) * masks01
    phase_cos_map = np.broadcast_to(phase[:, 1, None, None], (t_len, h, w)).astype(np.float32) * masks01

    amp = motion_amplitude(gray01, masks01)
    dx, dy, rel = optical_flow_fields(gray01, masks01)
    mid_res = mid_frequency_residual(gray01, masks01)
    parts = part_maps_from_masks(masks01, num_parts=num_parts)

    gpf = np.stack(
        [
            masks01,
            edges,
            nxs,
            nys,
            sdfs,
            amp,
            phase_sin_map,
            phase_cos_map,
            dx,
            dy,
            rel,
            mid_res,
        ],
        axis=1,
    ).astype(np.float32)

    return gpf, phase, parts


def save_pickle(obj: np.ndarray, path: Path) -> None:
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
    dtype: str,
    overwrite: bool,
) -> Tuple[SeqKey, int, str]:
    key, pairs = item
    sid, seq, view = key.as_tuple()
    out_dir = out_root / sid / seq / view
    gpf_path = out_dir / "000-gpf.pkl"
    sil_path = out_dir / "001-sils.pkl"
    phase_path = out_dir / "002-phase.pkl"
    parts_path = out_dir / "003-parts.pkl"

    if not overwrite and gpf_path.exists() and sil_path.exists() and phase_path.exists() and parts_path.exists():
        return key, -1, "skip"

    aligned_rgbs: List[np.ndarray] = []
    aligned_sils: List[np.ndarray] = []

    for rgb_p, sil_p in pairs:
        rgb = read_rgb(rgb_p)
        sil = read_sil(sil_p, sil_threshold)
        if rgb is None or sil is None:
            continue
        aligned = align_by_sil(rgb, sil, out_h=out_h, out_w=out_w, min_sil_area=min_sil_area)
        if aligned is None:
            continue
        rgb_a, sil_a = aligned
        aligned_rgbs.append(rgb_a)
        aligned_sils.append(sil_a)

    if len(aligned_rgbs) == 0:
        return key, 0, "empty"

    rgbs_np = np.stack(aligned_rgbs, axis=0).astype(np.uint8)
    sils_np = np.stack(aligned_sils, axis=0).astype(np.uint8)

    gpf, phase, parts = build_gpf(
        rgbs_bgr=rgbs_np,
        sils_u8=sils_np,
        num_parts=num_parts,
        min_period=min_period,
        max_period=max_period,
    )

    if dtype == "float16":
        gpf = gpf.astype(np.float16)
        parts = parts.astype(np.float16)
    elif dtype == "float32":
        gpf = gpf.astype(np.float32)
        parts = parts.astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # User-requested shape: T x 1 x H x W.
    sils_np = sils_np[:, None, :, :].astype(np.uint8)

    save_pickle(gpf, gpf_path)
    save_pickle(sils_np, sil_path)
    save_pickle(phase.astype(np.float32), phase_path)
    save_pickle(parts, parts_path)

    meta = {
        "sid": sid,
        "seq": seq,
        "view": view,
        "num_valid_frames": int(len(aligned_rgbs)),
        "gpf_shape": list(gpf.shape),
        "sils_shape": list(sils_np.shape),
        "phase_shape": list(phase.shape),
        "parts_shape": list(parts.shape),
        "gpf_channels": [
            "silhouette_mask",
            "body_edge",
            "contour_normal_x",
            "contour_normal_y",
            "signed_distance_field",
            "motion_amplitude",
            "phase_sin",
            "phase_cos",
            "flow_dx",
            "flow_dy",
            "motion_reliability",
            "mid_frequency_residual",
        ],
        "note": "Do not include this json in OpenGait data_in_use; OpenGait loads .pkl only.",
    }
    with open(out_dir / "meta-gpf.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return key, len(aligned_rgbs), "ok"


def pretreat_gpf(
    rgb_root: Path,
    sil_root: Path,
    out_root: Path,
    out_h: int = 64,
    out_w: int = 32,
    num_parts: int = 6,
    sil_threshold: int = 10,
    min_sil_area: int = 20,
    min_period: int = 8,
    max_period: int = 40,
    dtype: str = "float16",
    workers: int = 4,
    overwrite: bool = False,
) -> None:
    logging.info("Listing RGB frames from %s", rgb_root)
    rgb_groups = list_frames(rgb_root)
    logging.info("Listing silhouette frames from %s", sil_root)
    sil_groups = list_frames(sil_root)

    common_keys = sorted(set(rgb_groups) & set(sil_groups), key=lambda k: k.as_tuple())
    logging.info("Found %d paired sequences.", len(common_keys))

    jobs = []
    for key in common_keys:
        pairs = pair_rgb_sil(rgb_groups[key], sil_groups[key])
        if len(pairs) > 0:
            jobs.append((key, pairs))

    logging.info("Prepared %d non-empty sequence jobs.", len(jobs))
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
        dtype=dtype,
        overwrite=overwrite,
    )

    ok = skip = empty = 0
    if workers <= 1:
        iterator = map(worker_fn, jobs)
        for key, n, status in tqdm(iterator, total=len(jobs), desc="GPF pretreat"):
            ok += int(status == "ok")
            skip += int(status == "skip")
            empty += int(status == "empty")
    else:
        with mp.Pool(workers) as pool:
            for key, n, status in tqdm(pool.imap_unordered(worker_fn, jobs), total=len(jobs), desc="GPF pretreat"):
                ok += int(status == "ok")
                skip += int(status == "skip")
                empty += int(status == "empty")

    logging.info("Done. ok=%d, skip=%d, empty=%d", ok, skip, empty)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("OpenGait GPF pretreatment")
    parser.add_argument("--rgb_root", required=True, type=str, help="Raw/aligned RGB root: subject/type/view/frame.png")
    parser.add_argument("--sil_root", required=True, type=str, help="Silhouette root: subject/type/view/frame.png")
    parser.add_argument("--output_root", required=True, type=str, help="Output OpenGait-style pkl root")
    parser.add_argument("--img_h", default=64, type=int)
    parser.add_argument("--img_w", default=32, type=int)
    parser.add_argument("--num_parts", default=6, type=int)
    parser.add_argument("--sil_threshold", default=10, type=int)
    parser.add_argument("--min_sil_area", default=20, type=int)
    parser.add_argument("--min_period", default=8, type=int)
    parser.add_argument("--max_period", default=40, type=int)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_file", default="pretreatment_gpf.log", type=str)
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

    pretreat_gpf(
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
        dtype=args.dtype,
        workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
