#!/usr/bin/env python3
"""Create a simple TRAIN_SET/TEST_SET JSON split from a prepared root."""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--train-ratio', type=float, default=0.7)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    labels = sorted([p.name for p in Path(args.root).iterdir() if p.is_dir()])
    random.Random(args.seed).shuffle(labels)
    k = int(len(labels) * args.train_ratio)
    obj = {'TRAIN_SET': sorted(labels[:k]), 'TEST_SET': sorted(labels[k:])}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(obj, indent=2), encoding='utf-8')
    print(args.out, len(obj['TRAIN_SET']), len(obj['TEST_SET']))

if __name__ == '__main__':
    main()
