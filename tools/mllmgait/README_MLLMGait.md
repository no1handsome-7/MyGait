# MLLMGait: OpenGait implementation notes

This patch adds `MLLMGait`, an OpenGait-compatible research prototype for clothing-change gait recognition.

## Implemented idea

- **Qwen2.5-VL guidance**: generate sequence-level motion descriptions and cache Qwen text features with `prepare_qwen25vl_motion_text.py`.
- **RGB clothing-irrelevant suppression**: RGB features are fused with temporal frame-difference features and optional silhouette features through text-conditioned FiLM gates.
- **RGB + PointCloud default fusion**: RGB is the default appearance/motion branch and point cloud is the default 3D structure branch for SUSTech1K; silhouette is optional.
- **Enhanced temporal module**: each body part uses Transformer + attention pooling instead of plain max pooling.
- **Training losses**: ID CE, triplet, cross-modal consistency, Qwen text guidance contrast, and optional clothing-adversarial regularization.

## Files added

```text
opengait/modeling/models/mllmgait.py
opengait/modeling/losses/cross_modal_consistency.py
opengait/modeling/losses/text_guidance.py
configs/mllmgait/mllmgait_ccpg_rgbsil.yaml
configs/mllmgait/mllmgait_sustech1k_rgbpc.yaml
configs/mllmgait/mllmgait_sustech1k_rgbpcsil.yaml  # compatibility alias, now also RGB+PC
tools/mllmgait/prepare_qwen25vl_motion_text.py
tools/mllmgait/README_MLLMGait.md
```

## Prepare Qwen2.5-VL motion text features

Install optional dependencies:

```bash
pip install transformers accelerate qwen-vl-utils pillow
```

Example for CCPG:

```bash
python tools/mllmgait/prepare_qwen25vl_motion_text.py \
  --dataset-root /path/to/CCPG/processed \
  --reference-pkl 00-rgb.pkl \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 \
  --skip-existing
```

The script writes `qwen25vl_motion_feat.pkl` in each sequence folder. The pkl is a frame-repeated list of a sequence-level Qwen feature, so OpenGait can sample it like other modalities.

## Training

CCPG RGB + Sil + Qwen text feature:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4 opengait/main.py \
  --cfgs configs/mllmgait/mllmgait_ccpg_rgbsil.yaml \
  --phase train
```

SUSTech1K PointCloud + Sil + Qwen text feature:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4 opengait/main.py \
  --cfgs configs/mllmgait/mllmgait_sustech1k_rgbpc.yaml
configs/mllmgait/mllmgait_sustech1k_rgbpcsil.yaml  # compatibility alias, now also RGB+PC \
  --phase train
```

## Important data-order note

OpenGait sorts `.pkl` files lexicographically inside each sequence folder, then applies `data_in_use`. The model indices (`rgb_index`, `sil_index`, `pc_index`, `text_index`) refer to the **selected order after filtering**, not the original sorted-file order.

For example, if sorted files are:

```text
00-rgb.pkl
01-sil.pkl
02-qwen25vl_motion_feat.pkl
```

and `data_in_use: [true, true, true]`, then:

```text
rgb_index = 0
sil_index = 1
text_index = 2
```

If your dataset has a different order, change the YAML indices and transform list accordingly.


## SUSTech1K default modality update

For this project, SUSTech1K now defaults to **RGB + PointCloud**, not RGB/PC/Sil.

Default selected order after `data_in_use` in `mllmgait_sustech1k_rgbpc.yaml`:

```text
ipts[0] = synchronized point cloud
ipts[1] = synchronized RGB
ipts[2] = qwen25vl_motion_feat.pkl
```

The old config filename `mllmgait_sustech1k_rgbpcsil.yaml` is retained only as a compatibility alias and now contains the same RGB+PC settings.
