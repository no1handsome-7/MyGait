#!/usr/bin/env python3
"""Extract 2D pose from RGB pkl using HuggingFace ViTPose."""
import argparse
import numpy as np
import torch
from pathlib import Path

# 对于 transformers >= 4.48 内置 ViTPose 支持
from transformers import VitPoseForPoseEstimation, AutoProcessor
from common import iter_opengait_sequences, load_pkl, save_pkl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--model', default='usyd-community/vitpose-base-simple',
                    help='HuggingFace ViTPose model ID')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = VitPoseForPoseEstimation.from_pretrained(args.model).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    for pid, typ, view, seq_dir in iter_opengait_sequences(args.root):
        rgb_path = seq_dir / '00-rgb.pkl'
        out_path = seq_dir / '02-pose.pkl'
        if not rgb_path.exists():
            continue
        if out_path.exists():
            print(f'skip (exists): {seq_dir}')
            continue

        rgb = load_pkl(rgb_path)  # list of [C,H,W] or [H,W,C]
        T = len(rgb)
        all_kps = []

        for start in range(0, T, args.batch_size):
            batch_frames = rgb[start:start + args.batch_size]
            images = []
            for frame in batch_frames:
                f = np.asarray(frame)
                if f.ndim == 3 and f.shape[0] == 3:
                    f = f.transpose(1, 2, 0)  # CHW -> HWC
                images.append(f)

            h, w = images[0].shape[:2]
            # 为每帧提供整图作为 bounding box
            boxes = [[[0, 0, w, h]]] * len(images)

            inputs = processor(images=images, boxes=boxes, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # 解码关键点
            results = processor.post_process_pose_estimation(
                outputs, boxes=boxes)

            for res in results:
                if len(res) > 0:
                    kps = res[0]['keypoints'].cpu().numpy()   # [17, 2]
                    scores = res[0]['scores'].cpu().numpy()    # [17]
                    kp3 = np.concatenate(
                        [kps, scores[:, None]], axis=-1)       # [17, 3]
                else:
                    kp3 = np.zeros((17, 3), dtype=np.float32)
                all_kps.append(kp3.astype(np.float32))

        save_pkl(all_kps, out_path)
        print(f'wrote {out_path} ({T} frames)')


if __name__ == '__main__':
    main()