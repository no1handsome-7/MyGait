import base64
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np


def load_pkl(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pkl(obj: Any, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_json(obj: Any, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def iter_opengait_sequences(root: str):
    root = Path(root)
    for pid in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for typ_dir in sorted((root / pid).iterdir()):
            if not typ_dir.is_dir():
                continue
            for view_dir in sorted(typ_dir.iterdir()):
                if view_dir.is_dir():
                    yield pid, typ_dir.name, view_dir.name, view_dir


def read_image_any(path: str):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def image_to_data_url(img: np.ndarray, max_side: int = 448) -> str:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return f'data:image/jpeg;base64,{b64}'


def sequence_key(pid: str, typ: str, view: str) -> str:
    return f'{pid}/{typ}/{view}'


def ensure_same_length(*seqs):
    lens = [len(s) for s in seqs if s is not None]
    if not lens:
        return 0
    m = min(lens)
    return m
