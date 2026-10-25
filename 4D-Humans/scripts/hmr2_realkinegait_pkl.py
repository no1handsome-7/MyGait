#!/usr/bin/env python3
"""Run HMR2/4D-Humans directly on RealKineGait 00-rgb.pkl files.

This script avoids unpacking RGB pkl files into image folders. It traverses an
OpenGait/RealKineGait-style tree:

    ROOT/pid/type/view/00-rgb.pkl

and writes HMR2 predictions to:

    OUT_ROOT/pid/type/view/smpl.npz

The resulting npz files are compatible with tools/realkinegait/generate_smpl_teacher.py
when that script looks for pred_root / pid / type / view / smpl.npz.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import pickle
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import cv2
import numpy as np
import torch

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import DEFAULT_CHECKPOINT, download_models, load_hmr2
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset
def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.0):
    """Convert weak-perspective crop camera to full-image camera translation.

    This is copied from hmr2.utils.renderer.cam_crop_to_full to avoid importing
    hmr2.utils.renderer, which requires pyrender/trimesh even though this script
    does not render anything.
    """
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam


from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy


Record = Dict[str, np.ndarray]


def iter_realkinegait_sequences(root: Path, rgb_name: str) -> Iterator[Tuple[str, str, str, Path]]:
    """Yield (pid, type, view, rgb_pkl_path) from ROOT/pid/type/view/rgb_name."""
    if not root.exists():
        raise FileNotFoundError(f"root does not exist: {root}")

    for pid_dir in sorted(root.iterdir()):
        if not pid_dir.is_dir():
            continue
        for typ_dir in sorted(pid_dir.iterdir()):
            if not typ_dir.is_dir():
                continue
            for view_dir in sorted(typ_dir.iterdir()):
                if not view_dir.is_dir():
                    continue
                rgb_pkl = view_dir / rgb_name
                if rgb_pkl.exists():
                    yield pid_dir.name, typ_dir.name, view_dir.name, rgb_pkl


def frame_to_cv2(frame, input_color: str = "rgb") -> np.ndarray:
    """Convert a frame from pkl to HxWx3 uint8 BGR image for OpenCV/detectron2."""
    arr = np.asarray(frame)

    if arr.ndim == 0:
        raise ValueError("frame is scalar; expected image array")

    # Common pkl layouts: CHW or HWC.
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[-1]:
        arr = arr.transpose(1, 2, 0)

    # Grayscale -> 3 channels.
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)

    if arr.ndim != 3:
        raise ValueError(f"unsupported frame shape: {arr.shape}")

    # Single-channel HWC -> 3 channels. RGBA/BGRA -> RGB/BGR by dropping alpha.
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[-1] >= 4:
        arr = arr[..., :3]
    elif arr.shape[-1] != 3:
        raise ValueError(f"unsupported channel count in frame shape: {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32, copy=False)
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        vmin = float(arr.min())
        vmax = float(arr.max())

        # Most datasets store either [0, 1] floats or [0, 255] values.
        # This extra branch also handles roughly [-1, 1] normalized images.
        if vmin >= -1.5 and vmax <= 1.5:
            if vmin < 0.0:
                arr = (arr + 1.0) * 127.5
            else:
                arr = arr * 255.0

        arr = np.clip(arr, 0, 255).astype(np.uint8)

    arr = np.ascontiguousarray(arr)

    if input_color == "rgb":
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if input_color == "bgr":
        return arr
    raise ValueError(f"input_color must be 'rgb' or 'bgr', got {input_color!r}")


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def build_detector(detector_name: str):
    if detector_name == "vitdet":
        from detectron2.config import LazyConfig
        import hmr2

        cfg_path = Path(hmr2.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        return DefaultPredictor_Lazy(detectron2_cfg)

    if detector_name == "regnety":
        from detectron2 import model_zoo

        detectron2_cfg = model_zoo.get_config(
            "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py",
            trained=True,
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        return DefaultPredictor_Lazy(detectron2_cfg)

    raise ValueError(f"unknown detector: {detector_name}")


def choose_person_box(det_instances, det_thresh: float, person_strategy: str) -> Optional[np.ndarray]:
    # Detectron2 may keep Instances on GPU; move them to CPU for numpy/indexing safety.
    if hasattr(det_instances, "to"):
        det_instances = det_instances.to("cpu")

    valid = (det_instances.pred_classes == 0) & (det_instances.scores > det_thresh)
    valid_count = int(valid.detach().cpu().sum().item()) if isinstance(valid, torch.Tensor) else int(np.sum(valid))
    if valid_count == 0:
        return None

    boxes = det_instances.pred_boxes.tensor[valid].detach().cpu().numpy()
    scores = det_instances.scores[valid].detach().cpu().numpy()

    if len(boxes) == 0:
        return None

    if person_strategy == "score":
        keep = int(np.argmax(scores))
    elif person_strategy == "first":
        keep = 0
    elif person_strategy == "largest":
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = int(np.argmax(areas))
    else:
        raise ValueError(f"unknown person_strategy: {person_strategy}")

    return boxes[keep : keep + 1].astype(np.float32)


def default_record(save_joints3d: bool = False) -> Record:
    rec: Record = {
        "betas": np.zeros((10,), dtype=np.float32),
        "body_pose": np.tile(np.eye(3, dtype=np.float32), (23, 1, 1)),
        "global_orient": np.eye(3, dtype=np.float32)[None, ...],
        "transl": np.zeros((3,), dtype=np.float32),
        "camera": np.zeros((3,), dtype=np.float32),
    }
    if save_joints3d:
        rec["joints3d"] = np.zeros((44, 3), dtype=np.float32)
    return rec


def predict_frame(
    img_cv2: np.ndarray,
    model,
    model_cfg,
    detector,
    device: torch.device,
    det_thresh: float,
    person_strategy: str,
    quiet_dataset: bool,
    save_joints3d: bool,
) -> Optional[Record]:
    det_out = detector(img_cv2)
    det_instances = det_out["instances"]
    boxes = choose_person_box(det_instances, det_thresh=det_thresh, person_strategy=person_strategy)
    if boxes is None:
        return None

    dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    if quiet_dataset:
        with contextlib.redirect_stdout(io.StringIO()):
            batch = next(iter(dataloader))
    else:
        batch = next(iter(dataloader))

    batch = recursive_to(batch, device)

    with torch.no_grad():
        out = model(batch)

    pred_cam = out["pred_cam"]
    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    img_size = batch["img_size"].float()
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    pred_cam_t_full = cam_crop_to_full(
        pred_cam,
        box_center,
        box_size,
        img_size,
        scaled_focal_length,
    )

    smpl_params = out["pred_smpl_params"]
    rec: Record = {
        "betas": to_numpy(smpl_params["betas"][0]).astype(np.float32),
        "body_pose": to_numpy(smpl_params["body_pose"][0]).astype(np.float32),
        "global_orient": to_numpy(smpl_params["global_orient"][0]).astype(np.float32),
        "transl": to_numpy(pred_cam_t_full[0]).astype(np.float32),
        "camera": to_numpy(pred_cam[0]).astype(np.float32),
    }

    if save_joints3d:
        rec["joints3d"] = to_numpy(out["pred_keypoints_3d"][0]).astype(np.float32)

    return rec


def process_sequence(
    rgb_pkl: Path,
    out_npz: Path,
    model,
    model_cfg,
    detector,
    device: torch.device,
    input_color: str,
    det_thresh: float,
    person_strategy: str,
    quiet_dataset: bool,
    save_joints3d: bool,
    max_frames: Optional[int],
    log_every: int,
) -> Tuple[int, int]:
    with open(rgb_pkl, "rb") as f:
        frames = pickle.load(f)

    if max_frames is not None:
        frames = frames[:max_frames]

    records = []
    detected = []
    last_valid = None

    total = len(frames)
    for idx, frame in enumerate(frames):
        img_cv2 = frame_to_cv2(frame, input_color=input_color)
        rec = predict_frame(
            img_cv2=img_cv2,
            model=model,
            model_cfg=model_cfg,
            detector=detector,
            device=device,
            det_thresh=det_thresh,
            person_strategy=person_strategy,
            quiet_dataset=quiet_dataset,
            save_joints3d=save_joints3d,
        )

        is_detected = rec is not None
        if rec is None:
            rec = last_valid
        else:
            last_valid = rec

        records.append(rec)
        detected.append(is_detected)

        if log_every > 0 and ((idx + 1) % log_every == 0 or idx + 1 == total):
            print(f"    frames {idx + 1}/{total}, detected={sum(detected)}")

    first_valid = next((r for r in records if r is not None), None)
    if first_valid is None:
        first_valid = default_record(save_joints3d=save_joints3d)

    records = [r if r is not None else first_valid for r in records]

    out = {
        key: np.stack([r[key] for r in records], axis=0).astype(np.float32)
        for key in ["betas", "body_pose", "global_orient", "transl", "camera"]
    }
    if save_joints3d:
        out["joints3d"] = np.stack([r["joints3d"] for r in records], axis=0).astype(np.float32)

    # generate_smpl_teacher.py ignores unknown keys, but this is useful for QA.
    out["detected"] = np.asarray(detected, dtype=np.bool_)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **out)

    return len(frames), int(np.sum(detected))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HMR2 directly on RealKineGait/OpenGait 00-rgb.pkl files and save smpl.npz.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True, type=Path, help="Input root: RealKineGait dataset root, e.g. data/RealKineGait/CCPG")
    parser.add_argument("--out-root", required=True, type=Path, help="Output root for smpl.npz, e.g. data/RealKineGait_smpl_raw/CCPG")
    parser.add_argument("--rgb-name", default="00-rgb.pkl", help="RGB pkl file name inside each sequence directory")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="HMR2 checkpoint path")
    parser.add_argument("--detector", default="vitdet", choices=["vitdet", "regnety"], help="Person detector")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")
    parser.add_argument("--input-color", default="rgb", choices=["rgb", "bgr"], help="Color order stored in 00-rgb.pkl")
    parser.add_argument("--det-thresh", default=0.5, type=float, help="Minimum person detection score")
    parser.add_argument("--person-strategy", default="largest", choices=["largest", "score", "first"], help="How to choose one person when multiple are detected")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if smpl.npz already exists")
    parser.add_argument("--no-download", action="store_true", help="Do not call download_models before loading HMR2")
    parser.add_argument("--save-joints3d", action="store_true", help="Also save pred_keypoints_3d as joints3d. Off by default to keep teacher vector compact.")
    parser.add_argument("--max-seqs", type=int, default=None, help="Only process the first N sequences; useful for debugging")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N discovered sequences; useful for resuming chunks")
    parser.add_argument("--max-frames-per-seq", type=int, default=None, help="Debug only: limit frames per sequence")
    parser.add_argument("--log-every", type=int, default=50, help="Print progress every N frames; 0 disables per-frame progress")
    parser.add_argument("--verbose-dataset", action="store_true", help="Allow ViTDetDataset to print crop downsampling details")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.no_download:
        download_models(CACHE_DIR_4DHUMANS)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading HMR2 checkpoint: {args.checkpoint}")
    model, model_cfg = load_hmr2(args.checkpoint)
    model = model.to(device)
    model.eval()
    print(f"Using device: {device}")

    print(f"Loading detector: {args.detector}")
    detector = build_detector(args.detector)

    seqs = list(iter_realkinegait_sequences(args.root, args.rgb_name))
    if args.start_index:
        seqs = seqs[args.start_index :]
    if args.max_seqs is not None:
        seqs = seqs[: args.max_seqs]

    if len(seqs) == 0:
        raise RuntimeError(f"No sequences with {args.rgb_name} were found under {args.root}")

    print(f"Found {len(seqs)} sequences")

    done = 0
    skipped = 0
    failed = 0

    for seq_idx, (pid, typ, view, rgb_pkl) in enumerate(seqs, start=1):
        out_npz = args.out_root / pid / typ / view / "smpl.npz"

        if out_npz.exists() and out_npz.stat().st_size > 0 and not args.overwrite:
            print(f"[{seq_idx}/{len(seqs)}] SKIP {pid}/{typ}/{view}: {out_npz}")
            skipped += 1
            continue

        print(f"[{seq_idx}/{len(seqs)}] RUN  {pid}/{typ}/{view}")
        try:
            n_frames, n_detected = process_sequence(
                rgb_pkl=rgb_pkl,
                out_npz=out_npz,
                model=model,
                model_cfg=model_cfg,
                detector=detector,
                device=device,
                input_color=args.input_color,
                det_thresh=args.det_thresh,
                person_strategy=args.person_strategy,
                quiet_dataset=not args.verbose_dataset,
                save_joints3d=args.save_joints3d,
                max_frames=args.max_frames_per_seq,
                log_every=args.log_every,
            )
            print(f"    saved {out_npz} | frames={n_frames}, detected={n_detected}")
            done += 1
        except Exception as exc:
            failed += 1
            print(f"    ERROR {pid}/{typ}/{view}: {type(exc).__name__}: {exc}")

    print(f"Finished. done={done}, skipped={skipped}, failed={failed}")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
