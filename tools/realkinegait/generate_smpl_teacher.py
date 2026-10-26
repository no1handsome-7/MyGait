#!/usr/bin/env python3
"""Convert external SMPL/SMPL-X estimates into RealKineGait teacher pkl files.

This script intentionally does not bundle SMPL weights or estimator code. Run an
estimator such as 4DHumans/HMR2.0/SMPLer-X first, then point --pred-root to its
per-sequence .npz/.pkl outputs. The output 05-smpl.pkl is a list of per-frame
vectors, typically [betas, body_pose, global_orient, camera] or 3D joints.
"""
import argparse
from pathlib import Path

import numpy as np

from common import iter_opengait_sequences, load_pkl, save_pkl


def load_pred(path):
    if path.suffix == '.npz':
        d = np.load(path, allow_pickle=True)
        keys = [k for k in ['betas', 'body_pose', 'global_orient', 'transl', 'camera', 'joints3d'] if k in d]
        arrs = []
        for k in keys:
            a = np.asarray(d[k])
            if a.ndim == 1:
                a = a[None]
            arrs.append(a.reshape(a.shape[0], -1))
        return np.concatenate(arrs, -1).astype(np.float32)
    if path.suffix == '.pkl':
        obj = load_pkl(path)
        if isinstance(obj, dict):
            keys = [k for k in ['betas', 'body_pose', 'global_orient', 'transl', 'camera', 'joints3d'] if k in obj]
            arrs = [np.asarray(obj[k]).reshape(np.asarray(obj[k]).shape[0], -1) for k in keys]
            return np.concatenate(arrs, -1).astype(np.float32)
        return np.asarray(obj, dtype=np.float32).reshape(len(obj), -1)
    raise ValueError(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Prepared OpenGait-style output root')
    ap.add_argument('--pred-root', required=True, help='External estimator predictions arranged as pid/type/view.npz or pid/type/view/*.npz')
    ap.add_argument('--rgb-name', default='00-rgb.pkl')
    ap.add_argument('--out-name', default='05-smpl.pkl')
    ap.add_argument('--target-dim', type=int, default=256)
    args = ap.parse_args()
    pred_root = Path(args.pred_root)
    for pid, typ, view, seq_dir in iter_opengait_sequences(args.root):
        candidates = [pred_root / pid / typ / f'{view}.npz', pred_root / pid / typ / f'{view}.pkl', pred_root / pid / typ / view / 'smpl.npz', pred_root / pid / typ / view / 'smpl.pkl']
        pred = next((p for p in candidates if p.exists()), None)
        if pred is None:
            continue
        feat = load_pred(pred)
        T = len(load_pkl(seq_dir / args.rgb_name))
        if len(feat) < T:
            pad = np.repeat(feat[-1:], T - len(feat), 0)
            feat = np.concatenate([feat, pad], 0)
        feat = feat[:T]
        # Fixed dimensionality through truncation/padding; the model has a learnable projector.
        if feat.shape[1] < args.target_dim:
            feat = np.pad(feat, ((0, 0), (0, args.target_dim - feat.shape[1])))
        feat = feat[:, :args.target_dim].astype(np.float32)
        save_pkl([x for x in feat], seq_dir / args.out_name)
        print('wrote', seq_dir / args.out_name)

if __name__ == '__main__':
    main()
