import cv2
import pickle
import argparse
import numpy as np
from pathlib import Path


def resize_with_padding(img, target_size=(256, 128)):
    """
    OpenGait-style RGB resize with padding.

    Args:
        img: RGB image, [H, W, 3]
        target_size: (target_h, target_w)

    Return:
        padded RGB image, [target_h, target_w, 3]
    """
    h, w, _ = img.shape
    target_h, target_w = target_size

    new_w = int(w * target_h / h)

    resized = cv2.resize(
        img,
        (new_w, target_h),
        interpolation=cv2.INTER_AREA
    )

    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x_offset = (target_w - resized.shape[1]) // 2

    if x_offset < 0:
        x_offset = abs(x_offset)
        padded = resized[:, x_offset:x_offset + target_w, :]
    else:
        padded[:, x_offset:x_offset + resized.shape[1], :] = resized

    return padded


def avi_to_rgb_pkl(
    avi_path: Path,
    out_pkl: Path,
    target_size=(256, 128)
):
    """
    Convert one RGB avi video to pkl.

    Output pkl shape:
        [T, 3, H, W]

    Channel order:
        RGB
    """

    cap = cv2.VideoCapture(str(avi_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open avi: {avi_path}")

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # OpenCV reads BGR by default, convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = resize_with_padding(
            frame,
            target_size=target_size
        )

        # [H, W, C] -> [C, H, W]
        frame = frame.transpose(2, 0, 1)

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames found in avi: {avi_path}")

    frames = np.asarray(frames, dtype=np.uint8)

    out_pkl.parent.mkdir(parents=True, exist_ok=True)

    tmp_pkl = out_pkl.with_suffix(".pkl.tmp")

    with open(tmp_pkl, "wb") as f:
        pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)

    tmp_pkl.replace(out_pkl)

    print(f"[OK] {avi_path}")
    print(f"     -> {out_pkl}")
    print(f"     shape = {frames.shape}")


def build_output_path(
    avi_path: Path,
    src_root: Path,
    dst_root: Path
):
    """
    Example:

    src:
        CCGR-MINI-RGB-V1/1/AS1/180_1.avi/180_1rgb_f-1-AS1.avi

    dst:
        CCGR-MINI/1/AS1/180_1.avi/1-AS1-180_1.avi-rgb.pkl
    """

    rel = avi_path.relative_to(src_root)

    # rel:
    #   1/AS1/180_1.avi/180_1rgb_f-1-AS1.avi
    rel_parent = rel.parent

    parts = rel_parent.parts

    if len(parts) != 3:
        raise RuntimeError(
            f"Unexpected directory structure: {rel_parent}. "
            f"Expected: ID/TYPE/VIEW.avi"
        )

    subject_id = parts[0]       # 1
    seq_type = parts[1]         # AS1
    view_name = parts[2]        # 180_1.avi

    out_name = f"{subject_id}-{seq_type}-{view_name}-rgb.pkl"

    out_pkl = dst_root / rel_parent / out_name

    return out_pkl


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src_root",
        required=True,
        help="Root of CCGR-MINI raw RGB avi, e.g. /data3/user25/code/OpenGait/data/CCGR-MINI-RGB-V1"
    )

    parser.add_argument(
        "--dst_root",
        required=True,
        help="Target CCGR-MINI root, e.g. /data3/user25/code/OpenGait/data/CCGR-MINI"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Output image height."
    )

    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Output image width."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing rgb pkl files."
    )

    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    if not dst_root.exists():
        raise FileNotFoundError(f"Target root not found: {dst_root}")

    avi_files = sorted(
        p for p in src_root.rglob("*.avi")
        if p.is_file()
    )

    print(f"Source root: {src_root}")
    print(f"Target root: {dst_root}")
    print(f"Found {len(avi_files)} avi files.")

    for avi_path in avi_files:
        try:
            out_pkl = build_output_path(
                avi_path=avi_path,
                src_root=src_root,
                dst_root=dst_root
            )

            if out_pkl.exists() and not args.overwrite:
                print(f"[Skip] exists: {out_pkl}")
                continue

            avi_to_rgb_pkl(
                avi_path=avi_path,
                out_pkl=out_pkl,
                target_size=(args.height, args.width)
            )

        except Exception as e:
            print(f"[Failed] {avi_path}")
            print(f"Reason: {e}")


if __name__ == "__main__":
    main()
