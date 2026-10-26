#!/usr/bin/env python3
"""Create a RealKineGait OpenGait pkl tree with deterministic modality names.

This script is intentionally conservative because official dataset layouts differ.
It expects a source OpenGait-style tree where each sequence directory already contains
one or more .pkl modality files. It copies/renames selected files into:
  pid/type/view/00-rgb.pkl
  pid/type/view/01-sil.pkl
  pid/type/view/02-pose.pkl
  pid/type/view/03-motion_text.pkl
  pid/type/view/04-nuisance_text.pkl
  pid/type/view/05-smpl.pkl
  pid/type/view/06-pointcloud.pkl
  pid/type/view/07-event.pkl

You can run it repeatedly; generated files are overwritten only when source exists.
"""
import argparse
import shutil
from pathlib import Path

from common import iter_opengait_sequences

MODS = [
    ('rgb', '00-rgb.pkl'),
    ('sil', '01-sil.pkl'),
    ('pose', '02-pose.pkl'),
    ('motion_text', '03-motion_text.pkl'),
    ('nuisance_text', '04-nuisance_text.pkl'),
    ('smpl', '05-smpl.pkl'),
    ('pointcloud', '06-pointcloud.pkl'),
    ('event', '07-event.pkl'),
]

DEFAULT_PATTERNS = {
    'rgb': ['rgb.pkl', '00-rgb.pkl', 'imgs.pkl', 'image.pkl'],
    'sil': ['sil.pkl', '01-sil.pkl', 'sils.pkl', 'silhouette.pkl'],
    'pose': ['pose.pkl', '02-pose.pkl', 'keypoints.pkl'],
    'motion_text': ['03-motion_text.pkl', 'motion_text.pkl'],
    'nuisance_text': ['04-nuisance_text.pkl', 'nuisance_text.pkl'],
    'smpl': ['05-smpl.pkl', 'smpl.pkl'],
    'pointcloud': ['06-pointcloud.pkl', 'pointcloud.pkl', 'points.pkl', 'pc.pkl'],
    'event': ['07-event.pkl', 'event.pkl', 'events.pkl'],
}


def find_file(seq_dir: Path, mod: str, override: str = None):
    if override:
        p = seq_dir / override
        return p if p.exists() else None
    for name in DEFAULT_PATTERNS[mod]:
        p = seq_dir / name
        if p.exists():
            return p
    # Last fallback: keyword search.
    for p in sorted(seq_dir.glob('*.pkl')):
        if mod.lower() in p.name.lower():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-root', required=True)
    ap.add_argument('--dst-root', required=True)
    for mod, _ in MODS:
        ap.add_argument(f'--{mod}-file', default=None, help=f'Optional source filename for {mod}')
    args = ap.parse_args()
    dst_root = Path(args.dst_root)
    copied = 0
    for pid, typ, view, seq_dir in iter_opengait_sequences(args.src_root):
        out_dir = dst_root / pid / typ / view
        out_dir.mkdir(parents=True, exist_ok=True)
        for mod, out_name in MODS:
            src = find_file(seq_dir, mod, getattr(args, f'{mod}_file'))
            if src is not None:
                shutil.copy2(src, out_dir / out_name)
                copied += 1
    print(f'Copied {copied} modality files to {dst_root}')

if __name__ == '__main__':
    main()
