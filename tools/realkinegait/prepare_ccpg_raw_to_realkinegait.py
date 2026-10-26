#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert raw CCPG RGB/Sil image folders into RealKineGait/OpenGait pkl format.

Expected raw layout:
  <rgb_root>/<pid>/<condition>/<view_seq>/<frame>.jpg
  <sil_root>/<pid>/<condition>/<view_seq>/<frame>.jpg or .png

Example:
  /data3/user25/data_desk/CCPG_D_MASK_FACE_SHOE/006/U0_D0_BG/07_0/00000005.jpg

Output layout:
  <out_root>/<pid>/<condition>/<view_seq>/00-rgb.pkl
  <out_root>/<pid>/<condition>/<view_seq>/01-sil.pkl
  <out_root>/<pid>/<condition>/<view_seq>/meta_ccpg_preprocess.json

The pkl files are list[np.ndarray], one resized uint8 frame per element.
RGB frames are [H,W,3], silhouettes are [H,W].
"""

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_frames(seq_dir: Path) -> List[Path]:
    return sorted([p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def iter_sequences(root: Path) -> Iterable[Tuple[str, str, str, Path]]:
    """Yield pid, condition, view_seq, sequence_dir from a 3-level CCPG tree."""
    for pid_dir in sorted(root.iterdir()):
        if not pid_dir.is_dir():
            continue
        for cond_dir in sorted(pid_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            for view_dir in sorted(cond_dir.iterdir()):
                if not view_dir.is_dir():
                    continue
                if list_frames(view_dir):
                    yield pid_dir.name, cond_dir.name, view_dir.name, view_dir


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_sil(path: Path, thr: int) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    if thr >= 0:
        arr = (arr > thr).astype(np.uint8) * 255
    return arr


def find_sil_frame(sil_seq_dir: Path, rgb_frame: Path) -> Optional[Path]:
    """Try exact stem match first, then common extensions."""
    if not sil_seq_dir.exists():
        return None
    # exact file name
    p = sil_seq_dir / rgb_frame.name
    if p.exists():
        return p
    # same stem with other extensions
    for ext in IMG_EXTS:
        q = sil_seq_dir / f"{rgb_frame.stem}{ext}"
        if q.exists():
            return q
    return None


def bbox_from_sil(sil: np.ndarray, margin: float = 0.15) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(sil > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    w, h = x2 - x1, y2 - y1
    mx, my = int(round(w * margin)), int(round(h * margin))
    H, W = sil.shape[:2]
    x1 = max(0, x1 - mx)
    x2 = min(W, x2 + mx)
    y1 = max(0, y1 - my)
    y2 = min(H, y2 + my)
    return x1, y1, x2, y2


def union_bbox(boxes: List[Tuple[int, int, int, int]], H: int, W: int) -> Optional[Tuple[int, int, int, int]]:
    if not boxes:
        return None
    x1 = max(0, min(b[0] for b in boxes))
    y1 = max(0, min(b[1] for b in boxes))
    x2 = min(W, max(b[2] for b in boxes))
    y2 = min(H, max(b[3] for b in boxes))
    return x1, y1, x2, y2


def resize_direct(arr: np.ndarray, out_hw: Tuple[int, int], is_sil: bool) -> np.ndarray:
    h, w = out_hw
    pil_mode = "L" if is_sil else None
    im = Image.fromarray(arr, mode=pil_mode) if pil_mode else Image.fromarray(arr)
    resample = Image.NEAREST if is_sil else Image.BILINEAR
    return np.asarray(im.resize((w, h), resample=resample))


def resize_pad(arr: np.ndarray, out_hw: Tuple[int, int], is_sil: bool, fill: int = 0) -> np.ndarray:
    """Resize with aspect ratio preserved, then pad to [H,W]."""
    out_h, out_w = out_hw
    in_h, in_w = arr.shape[:2]
    if in_h <= 0 or in_w <= 0:
        shape = (out_h, out_w) if is_sil else (out_h, out_w, 3)
        return np.zeros(shape, dtype=np.uint8)

    scale = min(out_w / in_w, out_h / in_h)
    new_w = max(1, int(round(in_w * scale)))
    new_h = max(1, int(round(in_h * scale)))

    im = Image.fromarray(arr, mode="L") if is_sil else Image.fromarray(arr)
    resample = Image.NEAREST if is_sil else Image.BILINEAR
    resized = np.asarray(im.resize((new_w, new_h), resample=resample))

    if is_sil:
        canvas = np.full((out_h, out_w), fill, dtype=np.uint8)
    else:
        canvas = np.full((out_h, out_w, 3), fill, dtype=np.uint8)

    y0 = (out_h - new_h) // 2
    x0 = (out_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w, ...] = resized
    return canvas


def save_pkl(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def process_one(args_tuple):
    (pid, cond, view, rgb_seq_dir, rgb_root, sil_root, out_root, img_hw, sil_hw,
     crop_mode, bbox_margin, sil_thr, keep_aspect, frame_step, max_frames,
     min_frames, overwrite) = args_tuple

    out_dir = out_root / pid / cond / view
    rgb_out = out_dir / "00-rgb.pkl"
    sil_out = out_dir / "01-sil.pkl"

    if rgb_out.exists() and sil_out.exists() and not overwrite:
        return {"seq": str(out_dir), "status": "skip_exists"}

    rgb_frames = list_frames(rgb_seq_dir)
    if frame_step > 1:
        rgb_frames = rgb_frames[::frame_step]
    if max_frames > 0:
        rgb_frames = rgb_frames[:max_frames]
    if len(rgb_frames) < min_frames:
        return {"seq": str(rgb_seq_dir), "status": "skip_too_short", "frames": len(rgb_frames)}

    sil_seq_dir = sil_root / pid / cond / view if sil_root is not None else None

    rgbs, sils = [], []
    bboxes = []
    raw_sils = []
    raw_rgbs = []
    missing_sil = 0

    for rf in rgb_frames:
        rgb = read_rgb(rf)
        sil = None
        if sil_seq_dir is not None:
            sf = find_sil_frame(sil_seq_dir, rf)
            if sf is not None:
                sil = read_sil(sf, sil_thr)
            else:
                missing_sil += 1
        if sil is None:
            # Fallback: full-body dummy mask, so RGB-only preprocessing still works.
            sil = np.ones(rgb.shape[:2], dtype=np.uint8) * 255

        raw_rgbs.append(rgb)
        raw_sils.append(sil)
        box = bbox_from_sil(sil, bbox_margin)
        bboxes.append(box)

    seq_box = None
    if crop_mode == "sequence":
        valid = [b for b in bboxes if b is not None]
        H, W = raw_sils[0].shape[:2]
        seq_box = union_bbox(valid, H, W)

    for rgb, sil, box in zip(raw_rgbs, raw_sils, bboxes):
        use_box = None
        if crop_mode == "frame":
            use_box = box
        elif crop_mode == "sequence":
            use_box = seq_box

        if use_box is not None:
            x1, y1, x2, y2 = use_box
            rgb = rgb[y1:y2, x1:x2]
            sil = sil[y1:y2, x1:x2]

        if keep_aspect:
            rgb_r = resize_pad(rgb, tuple(img_hw), is_sil=False)
            sil_r = resize_pad(sil, tuple(sil_hw), is_sil=True)
        else:
            rgb_r = resize_direct(rgb, tuple(img_hw), is_sil=False)
            sil_r = resize_direct(sil, tuple(sil_hw), is_sil=True)

        if sil_thr >= 0:
            sil_r = (sil_r > 127).astype(np.uint8) * 255

        rgbs.append(rgb_r.astype(np.uint8))
        sils.append(sil_r.astype(np.uint8))

    save_pkl(rgbs, rgb_out)
    save_pkl(sils, sil_out)

    meta = {
        "pid": pid,
        "condition": cond,
        "view": view,
        "frames": len(rgbs),
        "rgb_shape": list(rgbs[0].shape),
        "sil_shape": list(sils[0].shape),
        "crop_mode": crop_mode,
        "bbox_margin": bbox_margin,
        "keep_aspect": keep_aspect,
        "frame_step": frame_step,
        "max_frames": max_frames,
        "missing_sil_frames": missing_sil,
        "rgb_source": str(rgb_seq_dir),
        "sil_source": str(sil_seq_dir) if sil_seq_dir is not None else None,
    }
    with open(out_dir / "meta_ccpg_preprocess.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return {"seq": str(out_dir), "status": "ok", "frames": len(rgbs), "missing_sil": missing_sil}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-root", required=True, help="Raw CCPG RGB root: root/pid/condition/view/*.jpg")
    ap.add_argument("--sil-root", default=None, help="Raw CCPG silhouette root with the same pid/condition/view layout. If omitted, dummy full masks are used.")
    ap.add_argument("--out-root", required=True, help="Output RealKineGait/OpenGait pkl root")
    ap.add_argument("--img-size", nargs=2, type=int, default=[128, 64], metavar=("H", "W"), help="RGB output size, default 128 64")
    ap.add_argument("--sil-size", nargs=2, type=int, default=None, metavar=("H", "W"), help="Sil output size; default equals --img-size")
    ap.add_argument("--crop", choices=["none", "frame", "sequence"], default="frame", help="Crop by silhouette bbox before resizing")
    ap.add_argument("--bbox-margin", type=float, default=0.15)
    ap.add_argument("--sil-thr", type=int, default=10, help="Silhouette threshold. Use -1 to disable binarization.")
    ap.add_argument("--keep-aspect", action="store_true", help="Preserve crop aspect ratio and pad to target size")
    ap.add_argument("--frame-step", type=int, default=1, help="Keep every k-th frame to save space; default 1")
    ap.add_argument("--max-frames", type=int, default=0, help="Optional cap per sequence; 0 means no cap")
    ap.add_argument("--min-frames", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rgb_root = Path(args.rgb_root)
    sil_root = Path(args.sil_root) if args.sil_root else None
    out_root = Path(args.out_root)
    sil_hw = args.sil_size if args.sil_size is not None else args.img_size

    seqs = list(iter_sequences(rgb_root))
    print(f"Found {len(seqs)} RGB sequences under {rgb_root}")
    if args.dry_run:
        for x in seqs[:10]:
            print(x[0], x[1], x[2], x[3])
        return

    jobs = [
        (pid, cond, view, seq_dir, rgb_root, sil_root, out_root, args.img_size, sil_hw,
         args.crop, args.bbox_margin, args.sil_thr, args.keep_aspect, args.frame_step,
         args.max_frames, args.min_frames, args.overwrite)
        for pid, cond, view, seq_dir in seqs
    ]

    ok = skip = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                res = fut.result()
                st = res["status"]
                if st == "ok":
                    ok += 1
                elif st.startswith("skip"):
                    skip += 1
                else:
                    fail += 1
                if i % 200 == 0 or st != "ok":
                    print(f"[{i}/{len(futs)}] {st}: {res['seq']}")
            except Exception as e:
                fail += 1
                print(f"[ERROR] {e}")

    print(f"Done. ok={ok}, skip={skip}, fail={fail}, out={out_root}")


if __name__ == "__main__":
    main()
