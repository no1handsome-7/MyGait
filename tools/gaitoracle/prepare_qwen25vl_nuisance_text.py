#!/usr/bin/env python3
"""Generate Qwen2.5-VL nuisance captions and features for GaitOracle.

This script intentionally asks Qwen to describe only non-gait nuisance factors,
not identity or motion. The generated factors are used as negative text
supervision through GRL/NBDetach.

Outputs per sequence folder:
  qwen25vl_nuisance_text.json
  qwen25vl_clothing_feat.pkl
  qwen25vl_carrying_feat.pkl
  qwen25vl_background_feat.pkl
  qwen25vl_occlusion_feat.pkl
  qwen25vl_clothing_mask.pkl
  qwen25vl_carrying_mask.pkl
  qwen25vl_background_mask.pkl
  qwen25vl_occlusion_mask.pkl

Example:
  python tools/gaitoracle/prepare_qwen25vl_nuisance_text.py \
    --dataset-root /path/to/SUSTech1K/processed \
    --reference-glob '*sync*Camera-RGB_raw.pkl' \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --frames 10 \
    --device cuda:0 \
    --skip-existing
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image
import torch


NUISANCE_PROMPT = (
    "You are assisting clothing-invariant gait recognition. "
    "Given these walking frames, describe only non-gait nuisance factors that a gait recognition model should ignore. "
    "Do not describe walking rhythm, stride, leg swing, arm swing, identity, body build, gender, age, or biometric traits. "
    "Focus only on clothing, carried objects, bags, coats, loose garments, background, occlusion, illumination, and camera/view artifacts. "
    "Be conservative. If a factor is not clearly visible, set present to false. "
    "Return exactly one compact JSON object with keys: clothing, carrying, background, occlusion. "
    "Each key must contain: present, description, confidence. "
    "present must be true or false. confidence must be low, medium, or high. "
    "No markdown. No extra explanation."
)


FACTORS = ["clothing", "carrying", "background", "occlusion"]


def find_sequence_dirs(root: Path) -> Iterable[Path]:
    for pid in sorted(root.iterdir()):
        if not pid.is_dir():
            continue
        for typ in sorted(pid.iterdir()):
            if not typ.is_dir():
                continue
            for view in sorted(typ.iterdir()):
                if view.is_dir():
                    yield view


def load_pkl(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def to_pil(frame) -> Image.Image:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.max() <= 2.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def sample_frames(seq: List, num_frames: int) -> List[Image.Image]:
    """Use the first num_frames frames. If shorter, use all frames."""
    if len(seq) == 0:
        raise ValueError('empty sequence')
    idx = list(range(min(len(seq), num_frames)))
    return [to_pil(seq[i]) for i in idx]


def resolve_reference_pkl(seq_dir: Path, reference_pkl: Optional[str], reference_glob: Optional[str]) -> Path:
    if reference_pkl is not None:
        p = seq_dir / reference_pkl
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    if reference_glob is not None:
        matched = sorted(seq_dir.glob(reference_glob))
        if not matched:
            raise FileNotFoundError(f'No pkl files matched {reference_glob!r} in {seq_dir}')
        return matched[0]

    pkls = sorted(seq_dir.glob('*.pkl'))
    rgb_candidates = [
        p for p in pkls
        if ('rgb' in p.name.lower() or 'camera-rgb' in p.name.lower())
        and 'ratio' not in p.name.lower()
        and 'hw' not in p.name.lower()
    ]
    if rgb_candidates:
        sync = [p for p in rgb_candidates if 'sync' in p.name.lower()]
        return sync[0] if sync else rgb_candidates[0]
    if not pkls:
        raise FileNotFoundError(f'No pkl files found in {seq_dir}')
    return pkls[0]


def clean_caption(caption: str) -> str:
    caption = caption.strip()
    if caption.startswith("```json"):
        caption = caption[len("```json"):].strip()
    elif caption.startswith("```"):
        caption = caption[len("```"):].strip()
    if caption.endswith("```"):
        caption = caption[:-3].strip()
    return caption


def safe_json_loads(text: str):
    text = clean_caption(text)
    try:
        return json.loads(text)
    except Exception:
        # Try extracting the first json object.
        lo, hi = text.find('{'), text.rfind('}')
        if lo >= 0 and hi > lo:
            return json.loads(text[lo:hi + 1])
        raise


def conf_to_weight(present, confidence):
    if not bool(present):
        return 0.0
    confidence = str(confidence).lower()
    if confidence == 'high':
        return 1.0
    if confidence == 'medium':
        return 0.5
    return 0.0


def get_qwen_objects(model_name: str, device: str, dtype: str):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

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
    return model, processor, process_vision_info


@torch.no_grad()
def generate_caption(model, processor, process_vision_info, images, prompt, device, max_new_tokens):
    content = [{'type': 'image', 'image': img} for img in images]
    content.append({'type': 'text', 'text': prompt})
    messages = [{'role': 'user', 'content': content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt').to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


@torch.no_grad()
def encode_text_with_qwen(model, processor, text: str, device: str) -> np.ndarray:
    inputs = processor(text=[text], padding=True, return_tensors='pt').to(device)
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden = outputs.hidden_states[-1][0]
    mask = inputs.attention_mask[0].bool()
    feat = hidden[mask].mean(0).float().cpu().numpy()
    return feat.astype(np.float32)


def write_feature(seq_dir: Path, feature: np.ndarray, seq_len: int, out_name: str):
    repeated = [feature.copy() for _ in range(seq_len)]
    with open(seq_dir / out_name, 'wb') as f:
        pickle.dump(repeated, f)


def write_mask(seq_dir: Path, value: float, seq_len: int, out_name: str):
    repeated = [np.array([value], dtype=np.float32) for _ in range(seq_len)]
    with open(seq_dir / out_name, 'wb') as f:
        pickle.dump(repeated, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--reference-pkl', default=None)
    parser.add_argument('--reference-glob', default=None)
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--max-sequences', type=int, default=-1)
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    root = Path(args.dataset_root)
    seq_dirs = list(find_sequence_dirs(root))
    if args.max_sequences > 0:
        seq_dirs = seq_dirs[:args.max_sequences]
    print(f'Found {len(seq_dirs)} sequence folders under {root}')

    if args.dry_run:
        for seq_dir in seq_dirs[:10]:
            print(seq_dir)
        return

    model, processor, process_vision_info = get_qwen_objects(args.model, args.device, args.dtype)

    for idx, seq_dir in enumerate(seq_dirs, 1):
        out_json = seq_dir / 'qwen25vl_nuisance_text.json'
        out_feats = [seq_dir / f'qwen25vl_{k}_feat.pkl' for k in FACTORS]
        out_masks = [seq_dir / f'qwen25vl_{k}_mask.pkl' for k in FACTORS]
        if args.skip_existing and out_json.exists() and all(p.exists() for p in out_feats + out_masks):
            continue

        try:
            ref_pkl = resolve_reference_pkl(seq_dir, args.reference_pkl, args.reference_glob)
            seq = load_pkl(ref_pkl)
            images = sample_frames(seq, args.frames)
            raw = generate_caption(model, processor, process_vision_info, images, NUISANCE_PROMPT, args.device, args.max_new_tokens)
            parsed = safe_json_loads(raw)

            meta = {
                'sequence_dir': str(seq_dir),
                'reference_pkl': ref_pkl.name,
                'model': args.model,
                'prompt': NUISANCE_PROMPT,
                'raw_caption': raw,
                'parsed': parsed,
            }

            for factor in FACTORS:
                item = parsed.get(factor, {})
                present = item.get('present', False)
                desc = item.get('description', 'none')
                conf = item.get('confidence', 'low')
                weight = conf_to_weight(present, conf)
                text_for_embedding = (
                    'Non-gait nuisance factor for clothing-invariant gait recognition. '
                    f'Factor: {factor}. Description: {desc}. Confidence: {conf}.'
                )
                feat = encode_text_with_qwen(model, processor, text_for_embedding, args.device)
                write_feature(seq_dir, feat, len(seq), f'qwen25vl_{factor}_feat.pkl')
                write_mask(seq_dir, weight, len(seq), f'qwen25vl_{factor}_mask.pkl')
                meta[f'{factor}_feature_dim'] = int(feat.shape[0])
                meta[f'{factor}_mask_weight'] = float(weight)

            out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
            print(f'[{idx}/{len(seq_dirs)}] wrote nuisance features in {seq_dir}')
        except Exception as exc:
            print(f'[{idx}/{len(seq_dirs)}] ERROR {seq_dir}: {exc}')


if __name__ == '__main__':
    main()
