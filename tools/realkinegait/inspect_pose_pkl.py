#!/usr/bin/env python3
"""Inspect OpenGait 02-pose.pkl files and report COCO-17 shape/score quality."""
import argparse
from pathlib import Path

import numpy as np

from common import iter_opengait_sequences, load_pkl


COCO17_NAMES = [
    "nose",
    "l_eye", "r_eye",
    "l_ear", "r_ear",
    "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow",
    "l_wrist", "r_wrist",
    "l_hip", "r_hip",
    "l_knee", "r_knee",
    "l_ankle", "r_ankle",
]
IMPORTANT = ["l_elbow", "r_elbow", "l_knee", "r_knee", "l_ankle", "r_ankle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pose-name", default="02-pose.pkl")
    ap.add_argument("--max-show", type=int, default=20)
    args = ap.parse_args()

    total = 0
    bad_shape = []
    low_score = []

    for pid, typ, view, seq_dir in iter_opengait_sequences(args.root):
        pose_path = seq_dir / args.pose_name
        if not pose_path.exists():
            continue

        total += 1
        pose = np.asarray(load_pkl(pose_path), dtype=np.float32)

        if pose.ndim != 3 or pose.shape[1] < 17 or pose.shape[2] < 3:
            bad_shape.append((str(seq_dir), tuple(pose.shape)))
            continue

        scores = np.nan_to_num(pose[:, :17, 2], nan=0.0, posinf=0.0, neginf=0.0)
        means = {name: float(scores[:, COCO17_NAMES.index(name)].mean()) for name in IMPORTANT}
        if any(v < 0.05 for v in means.values()):
            low_score.append((str(seq_dir), means))

    print(f"checked sequences: {total}")
    print(f"bad shape: {len(bad_shape)}")
    for item in bad_shape[:args.max_show]:
        print("BAD_SHAPE", item)

    print(f"low important-joint confidence: {len(low_score)}")
    for seq, means in low_score[:args.max_show]:
        print("LOW_SCORE", seq, means)


if __name__ == "__main__":
    main()
