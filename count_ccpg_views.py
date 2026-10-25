#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count all view categories in a CCPG-style OpenGait pkl dataset.

Example path:
    /data3/user25/code/OpenGait/data/CCPG-GPF-128x64-pkl/008/U0_D0_BG/03_0/000-gpf.pkl

Expected hierarchy:
    root/subject/sequence_type/view/000-gpf.pkl

Usage:
    python count_ccpg_views.py \
        --root /data3/user25/code/OpenGait/data/CCPG-GPF-128x64-pkl \
        --output-json ccpg_view_stats.json \
        --output-yaml ccpg_view_values.yaml

Optionally patch a training config automatically:
    python count_ccpg_views.py \
        --root /data3/user25/code/OpenGait/data/CCPG-GPF-128x64-pkl \
        --config-in gpf_structured_viewprompt_full_10ch.template.yaml \
        --config-out gpf_structured_viewprompt_full_10ch.yaml
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def natural_key(s: str):
    # "03_0" -> ["", 3, "_", 0, ""], "090" -> ["", 90, ""]
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(s))]


def collect_views(root: Path, filename: str = "000-gpf.pkl"):
    root = root.expanduser().resolve()
    paths = sorted(root.glob(f"*/*/*/{filename}"))

    view_counter = Counter()
    subject_counter = Counter()
    seq_type_counter = Counter()
    examples = defaultdict(list)

    bad_paths = []
    for p in paths:
        try:
            view = p.parent.name
            seq_type = p.parent.parent.name
            subject = p.parent.parent.parent.name
        except Exception:
            bad_paths.append(str(p))
            continue

        view_counter[view] += 1
        subject_counter[subject] += 1
        seq_type_counter[seq_type] += 1
        if len(examples[view]) < 3:
            examples[view].append(str(p))

    view_values = sorted(view_counter.keys(), key=natural_key)
    return {
        "root": str(root),
        "filename": filename,
        "total_sequences": int(sum(view_counter.values())),
        "num_views": int(len(view_values)),
        "view_values": view_values,
        "views": [
            {
                "view": v,
                "count": int(view_counter[v]),
                "examples": examples[v],
            }
            for v in view_values
        ],
        "num_subjects": int(len(subject_counter)),
        "num_sequence_types": int(len(seq_type_counter)),
        "sequence_types": [
            {"type": k, "count": int(seq_type_counter[k])}
            for k in sorted(seq_type_counter.keys(), key=natural_key)
        ],
        "bad_paths": bad_paths[:20],
    }


def write_yaml_snippet(stats, path: Path):
    values = stats["view_values"]
    lines = []
    lines.append("# Copy this snippet into model_cfg.")
    lines.append(f"view_prompt_num_views: {len(values)}")
    lines.append("view_values:")
    for v in values:
        lines.append(f"  - \"{v}\"")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def patch_config(stats, config_in: Path, config_out: Path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for --config-in/--config-out patching.") from exc

    cfg = yaml.safe_load(config_in.read_text(encoding="utf-8"))
    model_cfg = cfg.setdefault("model_cfg", {})
    model_cfg["use_view_prompt"] = True
    model_cfg["view_prompt_num_views"] = int(stats["num_views"])
    model_cfg["view_values"] = list(stats["view_values"])

    # Keep non-phase GPF channels only for learnable branches.
    model_cfg["channel_indices"] = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]
    model_cfg["phase_indices"] = []
    model_cfg["use_phase_film"] = False
    model_cfg["phase_input_index"] = None

    # Add view CE and regularization losses if absent.
    loss_cfg = cfg.setdefault("loss_cfg", [])
    prefixes = {item.get("log_prefix") for item in loss_cfg if isinstance(item, dict)}

    if "view_softmax" not in prefixes:
        loss_cfg.append({
            "type": "CrossEntropyLoss",
            "loss_term_weight": 0.03,
            "scale": 16,
            "log_prefix": "view_softmax",
            "log_accuracy": True,
        })

    if "view_prompt_reg" not in prefixes:
        loss_cfg.append({
            "type": "ViewPromptRegularizationLoss",
            "loss_term_weight": 1.0,
            "balance_weight": 0.002,
            "orthogonality_weight": 0.005,
            "log_prefix": "view_prompt_reg",
        })

    config_out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser("Count CCPG view categories and optionally patch GPFGait config.")
    parser.add_argument("--root", required=True, type=str, help="Dataset root, e.g. CCPG-GPF-128x64-pkl")
    parser.add_argument("--filename", default="000-gpf.pkl", type=str)
    parser.add_argument("--output-json", default="ccpg_view_stats.json", type=str)
    parser.add_argument("--output-yaml", default="ccpg_view_values.yaml", type=str)
    parser.add_argument("--config-in", default=None, type=str, help="Optional input yaml config to patch")
    parser.add_argument("--config-out", default=None, type=str, help="Optional output yaml config")
    args = parser.parse_args()

    stats = collect_views(Path(args.root), filename=args.filename)

    print(f"Root: {stats['root']}")
    print(f"Total sequences: {stats['total_sequences']}")
    print(f"Subjects: {stats['num_subjects']}")
    print(f"Sequence types: {stats['num_sequence_types']}")
    print(f"Views: {stats['num_views']}")
    print("")
    for row in stats["views"]:
        print(f"{row['view']:>12s}  {row['count']:8d}")

    out_json = Path(args.output_json)
    out_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved JSON stats to: {out_json}")

    out_yaml = Path(args.output_yaml)
    write_yaml_snippet(stats, out_yaml)
    print(f"Saved YAML snippet to: {out_yaml}")

    if args.config_in or args.config_out:
        if not args.config_in or not args.config_out:
            raise ValueError("--config-in and --config-out must be provided together.")
        patch_config(stats, Path(args.config_in), Path(args.config_out))
        print(f"Saved patched config to: {args.config_out}")


if __name__ == "__main__":
    main()
