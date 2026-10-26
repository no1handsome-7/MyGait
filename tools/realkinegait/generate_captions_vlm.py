#!/usr/bin/env python3
"""Generate nuisance/motion captions for RealKineGait.

Backends:
  qwen: local Qwen2.5-VL via transformers.
  openai: OpenAI Responses API; set OPENAI_API_KEY and --model accordingly.

The script samples a few RGB frames per sequence and writes nuisance_text.json and/or
motion_text_vlm.json. For core training, numeric pose-derived motion descriptors are
usually more stable; use VLM motion captions as auxiliary metadata.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from common import image_to_data_url, iter_opengait_sequences, load_pkl, save_json
from prompts import MOTION_PROMPT, NUISANCE_PROMPT


def sample_frames(rgb_seq, k=4):
    n = len(rgb_seq)
    if n == 0:
        return []
    idxs = np.linspace(0, n - 1, min(k, n)).round().astype(int).tolist()
    return [np.asarray(rgb_seq[i]) for i in idxs]


def call_openai(images, prompt, model):
    from openai import OpenAI
    client = OpenAI()
    content = [{'type': 'input_text', 'text': prompt}]
    for img in images:
        content.append({'type': 'input_image', 'image_url': image_to_data_url(img)})
    resp = client.responses.create(model=model, input=[{'role': 'user', 'content': content}], temperature=0)
    return resp.output_text


def load_qwen(model_name):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype='auto', device_map='auto')
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def call_qwen(images, prompt, model_pack):
    model, processor = model_pack
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': img} for img in images] + [{'type': 'text', 'text': prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, padding=True, return_tensors='pt').to(model.device)
    ids = model.generate(**inputs, max_new_tokens=512)
    ids = ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def parse_json_or_wrap(text):
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        return json.loads(text[start:end])
    except Exception:
        return {'raw': text.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--rgb-name', default='00-rgb.pkl')
    ap.add_argument('--backend', choices=['qwen', 'openai'], default='qwen')
    ap.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--frames', type=int, default=4)
    ap.add_argument('--tasks', nargs='+', default=['nuisance'], choices=['nuisance', 'motion'])
    args = ap.parse_args()

    qwen_pack = None
    if args.backend == 'qwen':
        qwen_pack = load_qwen(args.model)

    for _, _, _, seq_dir in iter_opengait_sequences(args.root):
        rgb_path = seq_dir / args.rgb_name
        if not rgb_path.exists():
            continue
        rgb = load_pkl(rgb_path)
        imgs = sample_frames(rgb, args.frames)
        if not imgs:
            continue
        if 'nuisance' in args.tasks:
            txt = call_qwen(imgs, NUISANCE_PROMPT, qwen_pack) if args.backend == 'qwen' else call_openai(imgs, NUISANCE_PROMPT, args.model)
            save_json(parse_json_or_wrap(txt), seq_dir / 'nuisance_text.json')
        if 'motion' in args.tasks:
            txt = call_qwen(imgs, MOTION_PROMPT, qwen_pack) if args.backend == 'qwen' else call_openai(imgs, MOTION_PROMPT, args.model)
            save_json(parse_json_or_wrap(txt), seq_dir / 'motion_text_vlm.json')
        print('captioned', seq_dir)

if __name__ == '__main__':
    main()
