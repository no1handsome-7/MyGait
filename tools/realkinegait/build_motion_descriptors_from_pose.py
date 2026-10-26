#!/usr/bin/env python3
"""Generate deterministic motion-text JSON and numeric descriptors from pose pkl files.

Input pose pkl format: list/array of T frames, each frame [J,2/3] in COCO-like order.
Output:
  - motion_text.json
  - 03-motion_text_raw.pkl

This version is compatible with extract_pose_hf.py and safely handles missing joints.
Descriptor layout is unchanged: 10 primary joints * [x,y,vx,vy] + 4 angles = 44-D.
"""
import argparse
import warnings

import numpy as np

from common import iter_opengait_sequences, load_pkl, save_json, save_pkl


COCO = {
    "nose": 0,
    "l_shoulder": 5, "r_shoulder": 6,
    "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10,
    "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14,
    "l_ankle": 15, "r_ankle": 16,
}

PRIMARY_NAMES = [
    "l_wrist", "r_wrist",
    "l_ankle", "r_ankle",
    "l_knee", "r_knee",
    "l_hip", "r_hip",
    "l_shoulder", "r_shoulder",
]
AUX_NAMES = ["l_elbow", "r_elbow"]


def _get(pose, name):
    idx = COCO.get(name, None)
    if idx is None or idx >= pose.shape[1]:
        return None
    arr = pose[:, idx, :2].astype(np.float32)
    return np.where(np.isfinite(arr), arr, np.nan).astype(np.float32)


def _safe_nanmean(x, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(x, axis=axis)


def _safe_nanmax(x, axis, keepdims=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmax(x, axis=axis, keepdims=keepdims)


def _fallback_center(pose):
    xy = pose[:, :, :2].astype(np.float32)
    xy = np.where(np.isfinite(xy), xy, np.nan)
    center = _safe_nanmean(xy, axis=1)
    return np.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _nan_traj(T):
    return np.full((T, 2), np.nan, dtype=np.float32)


def _angle(a, b, c):
    ba = a - b
    bc = c - b
    num = (ba * bc).sum(-1)
    den = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-6
    return np.arccos(np.clip(num / den, -1, 1))


def _angle_safe(a, b, c, T):
    if a is None or b is None or c is None:
        return np.zeros((T,), dtype=np.float32)
    out = _angle(a, b, c).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def descriptors(pose):
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim != 3:
        raise ValueError(f"pose should be [T,J,C], got {pose.shape}")
    if pose.shape[-1] < 2:
        raise ValueError(f"pose last dim should contain at least x/y, got {pose.shape}")

    T = pose.shape[0]
    pts = {n: _get(pose, n) for n in PRIMARY_NAMES + AUX_NAMES}

    torso = [pts[n] for n in ["l_hip", "r_hip", "l_shoulder", "r_shoulder"] if pts[n] is not None]
    if torso:
        center = _safe_nanmean(np.stack(torso, 0), axis=0)
        fallback = _fallback_center(pose)
        bad = ~np.isfinite(center).all(axis=1)
        center[bad] = fallback[bad]
    else:
        center = _fallback_center(pose)

    xy = pose[:, :, :2].astype(np.float32)
    xy = np.where(np.isfinite(xy), xy, np.nan)
    dist = np.linalg.norm(xy - center[:, None], axis=-1)
    scale = _safe_nanmax(dist, axis=1, keepdims=True)
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0) + 1e-6

    feat = []
    for n in PRIMARY_NAMES:
        p_raw = pts[n] if pts[n] is not None else _nan_traj(T)
        p = (p_raw - center) / scale
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        v = np.vstack([np.zeros((1, 2), np.float32), np.diff(p, axis=0)])
        feat.extend([p[:, 0], p[:, 1], v[:, 0], v[:, 1]])

    feat.extend([
        _angle_safe(pts["l_hip"], pts["l_knee"], pts["l_ankle"], T),
        _angle_safe(pts["r_hip"], pts["r_knee"], pts["r_ankle"], T),
        _angle_safe(pts["l_shoulder"], pts["l_elbow"], pts["l_wrist"], T),
        _angle_safe(pts["r_shoulder"], pts["r_elbow"], pts["r_wrist"], T),
    ])

    return np.nan_to_num(np.stack(feat, -1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def text_from_desc(desc):
    x_lf = desc[:, 8]
    x_rf = desc[:, 12]
    anti = (
        np.corrcoef(x_lf, x_rf)[0, 1]
        if len(desc) > 3 and np.std(x_lf) > 1e-6 and np.std(x_rf) > 1e-6
        else 0.0
    )
    speed = float(np.mean(np.abs(np.diff((x_lf + x_rf) * 0.5)))) if len(desc) > 1 else 0.0
    arm_swing = float(np.mean(np.abs(desc[:, 2] - desc[:, 6]))) if desc.shape[1] > 6 else 0.0
    return {
        "gait_phase": "estimated from ankle horizontal trajectories",
        "left_leg": "periodic stance/swing proxy from left ankle motion",
        "right_leg": "periodic stance/swing proxy from right ankle motion",
        "left_arm": "arm swing estimated from wrist velocity",
        "right_arm": "arm swing estimated from wrist velocity",
        "torso_pelvis": "torso and pelvis center used as motion normalization anchor",
        "coordination": "anti-phase arm/leg pattern likely" if anti < -0.15 else "coordination uncertain or weakly anti-phase",
        "cadence_proxy": round(speed, 4),
        "arm_swing_proxy": round(arm_swing, 4),
        "confidence": "medium; generated from 2D pose numeric trajectories",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pose-name", default="02-pose.pkl")
    ap.add_argument("--out-pkl-name", default="03-motion_text_raw.pkl")
    ap.add_argument("--out-json-name", default="motion_text.json")
    args = ap.parse_args()

    for pid, typ, view, seq_dir in iter_opengait_sequences(args.root):
        pose_path = seq_dir / args.pose_name
        if not pose_path.exists():
            continue
        pose = load_pkl(pose_path)
        desc = descriptors(pose)
        save_pkl([d for d in desc], seq_dir / args.out_pkl_name)
        save_json(text_from_desc(desc), seq_dir / args.out_json_name)
        print("wrote", seq_dir)


if __name__ == "__main__":
    main()
