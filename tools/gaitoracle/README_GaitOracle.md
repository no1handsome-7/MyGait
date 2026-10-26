# GaitOracle for OpenGait

GaitOracle is an upgraded framework for clothing-invariant gait recognition.

It implements five modules:

1. **Foundation RGB Encoder**
   - default: lightweight dense CNN for compatibility;
   - optional: timm foundation encoders such as DINOv2 / EVA-CLIP / DINOv3-like dense features.

2. **Positive-Negative Text Supervisor**
   - positive branch: fixed gait-positive prompt bank for motion-region attention;
   - negative branch: Qwen2.5-VL nuisance captions + GRL/NBDetach to remove clothing, carrying, background and occlusion cues.

3. **Geometry / Silhouette Anchor**
   - RGB+PC: point cloud acts as 3D structure-motion anchor;
   - RGB+Sil: silhouette acts as contour-motion anchor;
   - RGB-only can still run, but anchor losses are disabled.

4. **Cycle-Aware Temporal Encoder**
   - temporal transformer;
   - temporal difference tokens;
   - frequency tokens;
   - future latent prediction objective.

5. **Verifier-Guided Training**
   - rule-based attention verifier uses silhouette or RGB motion magnitude as weak attention sanity check;
   - Qwen-generated nuisance masks weight unreliable captions down.

## Prepare positive gait prompt bank

```bash
python tools/gaitoracle/prepare_positive_prompt_bank.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --out tools/gaitoracle/gait_positive_prompt_feat.npy \
  --device cuda:0 \
  --dtype bfloat16
```

If the file is absent, GaitOracle falls back to randomly initialized prompt embeddings.

## Prepare Qwen nuisance features

SUSTech1K RGB reference example:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/gaitoracle/prepare_qwen25vl_nuisance_text.py \
  --dataset-root /path/to/SUSTech1K/processed \
  --reference-glob '*sync*Camera-RGB_raw.pkl' \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 \
  --dtype bfloat16 \
  --frames 10 \
  --skip-existing
```

CCPG RGB reference example:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/gaitoracle/prepare_qwen25vl_nuisance_text.py \
  --dataset-root /path/to/CCPG/processed \
  --reference-glob '*rgb*.pkl' \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 \
  --dtype bfloat16 \
  --frames 10 \
  --skip-existing
```

Generated files per sequence:

```text
qwen25vl_nuisance_text.json
qwen25vl_clothing_feat.pkl
qwen25vl_carrying_feat.pkl
qwen25vl_background_feat.pkl
qwen25vl_occlusion_feat.pkl
qwen25vl_clothing_mask.pkl
qwen25vl_carrying_mask.pkl
qwen25vl_background_mask.pkl
qwen25vl_occlusion_mask.pkl
```

## Training

RGB+PC SUSTech1K:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  opengait/main.py \
  --cfgs configs/gaitoracle/gaitoracle_sustech1k_rgbpc.yaml \
  --phase train
```

RGB+Sil CCPG:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  opengait/main.py \
  --cfgs configs/gaitoracle/gaitoracle_ccpg_rgbsil.yaml \
  --phase train
```

## Notes

- Text features are used only during training.
- Inference uses RGB plus optional PC/Sil.
- The optional `rgb_backbone.type: timm` requires `pip install timm`.
- The default configs use the dependency-free tiny dense RGB encoder for initial debugging.
