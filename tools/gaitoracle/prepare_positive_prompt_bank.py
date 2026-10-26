#!/usr/bin/env python3
"""Prepare fixed gait-positive prompt bank features for GaitOracle.

This script encodes a small human-designed gait prompt bank once.
The output .npy is used by GaitOracle's PositiveTextGate.

Example:
  python tools/gaitoracle/prepare_positive_prompt_bank.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --out tools/gaitoracle/gait_positive_prompt_feat.npy \
    --device cuda:0
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch


GAIT_POSITIVE_PROMPTS = [
    "Focus on arm swing, leg swing, stride rhythm, cadence, torso stability, and body boundary motion.",
    "Preserve gait-relevant dynamic biometric cues and suppress clothing color, texture, coat, bag, and background.",
    "Attend to walking cycle, left-right limb alternation, center-body trajectory, and temporal consistency.",
    "Use motion-rich regions including arms, legs, torso, and body contour changes for clothing-invariant gait recognition.",
]


@torch.no_grad()
def encode_text_qwen(model_name, prompts, device='cuda:0', dtype='bfloat16'):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    torch_dtype = torch.float16 if dtype == 'float16' else torch.bfloat16 if dtype == 'bfloat16' else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device.startswith('cuda') else None,
    )
    if not device.startswith('cuda'):
        model = model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    feats = []
    for text in prompts:
        inputs = processor(text=[text], padding=True, return_tensors='pt').to(device)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1][0]
        mask = inputs.attention_mask[0].bool()
        feat = hidden[mask].mean(0).float().cpu().numpy()
        feats.append(feat.astype(np.float32))
    return np.stack(feats, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--out', default='tools/gaitoracle/gait_positive_prompt_feat.npy')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
    args = parser.parse_args()

    feats = encode_text_qwen(args.model, GAIT_POSITIVE_PROMPTS, args.device, args.dtype)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, feats)
    print(f'wrote {out} shape={feats.shape}')


if __name__ == '__main__':
    main()
