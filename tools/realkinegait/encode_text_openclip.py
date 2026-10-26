#!/usr/bin/env python3
"""Encode RealKineGait JSON text into per-frame OpenGait pkl embeddings.

Writes:
  03-motion_text.pkl      (if motion_text.json exists)
  04-nuisance_text.pkl    (if nuisance_text.json exists)
Every pkl is a list of length T, matching 00-rgb.pkl.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from common import iter_opengait_sequences, load_json, load_pkl, save_pkl


def flatten_json_text(obj):
    if isinstance(obj, dict):
        return '; '.join(f'{k}: {flatten_json_text(v)}' for k, v in obj.items())
    if isinstance(obj, list):
        return '; '.join(flatten_json_text(v) for v in obj)
    return str(obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--rgb-name', default='00-rgb.pkl')
    ap.add_argument('--model', default='ViT-B-32')
    ap.add_argument('--pretrained', default='laion2b_s34b_b79k')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    import open_clip
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()

    def encode(text):
        toks = tokenizer([text]).to(device)
        with torch.no_grad():
            feat = model.encode_text(toks).float()
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return feat[0].cpu().numpy().astype(np.float32)

    for _, _, _, seq_dir in iter_opengait_sequences(args.root):
        rgb_path = seq_dir / args.rgb_name
        if not rgb_path.exists():
            continue
        T = len(load_pkl(rgb_path))
        for json_name, out_name in [('motion_text.json', '03-motion_text.pkl'), ('nuisance_text.json', '04-nuisance_text.pkl')]:
            jp = seq_dir / json_name
            if not jp.exists():
                continue
            emb = encode(flatten_json_text(load_json(jp)))
            save_pkl([emb.copy() for _ in range(T)], seq_dir / out_name)
            print('encoded', seq_dir / out_name)

if __name__ == '__main__':
    main()
