"""GaitMoE V3 Visualization Utilities.

New plots for V3 architecture:
  1. Expert usage (reused from V2Plus visualize module)
  2. Appearance expert spatial attention heatmaps on RGB frames
  3. Fusion cross-attention maps (query → RGB/PC token preference)
  4. Quality score distribution

Usage:
  python gait_moe_visualize_v3.py --viz_data ./viz_output_v3/viz_data.pth
"""

import os
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Reuse existing utilities
try:
    from .gait_moe_visualize import (
        compute_expert_usage_from_logits,
        print_expert_usage_table,
        plot_expert_usage,
        safe_torch_load,
    )
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from gait_moe_visualize import (
        compute_expert_usage_from_logits,
        print_expert_usage_table,
        plot_expert_usage,
        safe_torch_load,
    )



# ================================================================
#  Spatial Attention Heatmap
# ================================================================

def _to_numpy_rgb_batch(rgb):
    """Convert RGB tensors to [N, H, W, 3] numpy images for imshow."""
    if rgb is None:
        return None
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().float().numpy()
    else:
        rgb = np.asarray(rgb)

    if rgb.ndim == 3:
        if rgb.shape[-1] == 3:
            rgb = rgb[None]
        elif rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))[None]
        else:
            return None
    elif rgb.ndim == 4:
        if rgb.shape[-1] == 3:
            pass
        elif rgb.shape[1] == 3:
            rgb = np.transpose(rgb, (0, 2, 3, 1))
        else:
            return None
    else:
        return None

    rgb = rgb.astype(np.float32)

    # Display normalization. If input was mean/std normalized, this does not
    # perfectly invert dataset normalization, but it makes overlays readable.
    # For true original colors, save raw pre-normalized RGB from the dataset/transform.
    if rgb.max() > 2.0:
        rgb = rgb / 255.0
    if rgb.min() < 0.0 or rgb.max() > 1.0:
        lo = rgb.min(axis=(1, 2, 3), keepdims=True)
        hi = rgb.max(axis=(1, 2, 3), keepdims=True)
        rgb = (rgb - lo) / np.maximum(hi - lo, 1e-6)
    return np.clip(rgb, 0.0, 1.0)


def _normalize_attention_map(attn_2d: np.ndarray):
    attn_2d = np.asarray(attn_2d, dtype=np.float32)
    attn_2d = attn_2d - np.nanmin(attn_2d)
    denom = np.nanmax(attn_2d)
    if denom > 1e-8:
        attn_2d = attn_2d / denom
    return np.clip(attn_2d, 0.0, 1.0)


def _resize_attention(attn_2d: np.ndarray, out_hw):
    H_img, W_img = out_hw
    try:
        import cv2
        return cv2.resize(attn_2d, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        t = torch.as_tensor(attn_2d, dtype=torch.float32)[None, None]
        t = torch.nn.functional.interpolate(
            t, size=(H_img, W_img), mode='bilinear', align_corners=False)
        return t[0, 0].cpu().numpy()


def plot_all_expert_attention_on_rgb(rgb_first: torch.Tensor,
                                     app_attn_all: torch.Tensor,
                                     Hf: int,
                                     Wf: int,
                                     save_dir: str,
                                     alpha: float = 0.45,
                                     max_experts: int = None):
    """Plot all appearance experts' attention on six randomly selected RGB frames.

    Args:
        rgb_first: [N, 3, H, W] or [N, H, W, 3], where N is normally 6.
        app_attn_all: [N, E, S] or [N, E, Hf, Wf].
    Output:
        app_expert_attention_overlay_6seq.png
        app_expert_attention_seqXX.png for each selected sequence
    """
    if not HAS_MPL:
        print("  matplotlib not found; skip RGB attention overlay.")
        return

    os.makedirs(save_dir, exist_ok=True)
    rgb_np = _to_numpy_rgb_batch(rgb_first)
    attn = torch.as_tensor(app_attn_all).detach().cpu().float()

    if rgb_np is None or attn.numel() == 0:
        print("  No valid RGB/all-expert attention data to plot.")
        return

    if attn.dim() == 4:
        attn = attn.flatten(2)  # [N,E,H,W] -> [N,E,S]
    if attn.dim() != 3:
        print(f"  Invalid app_attn_all shape: {tuple(attn.shape)}")
        return

    N = min(len(rgb_np), attn.size(0))
    E = attn.size(1)
    if max_experts is not None:
        E = min(E, int(max_experts))
    if N <= 0 or E <= 0:
        return

    ncols = E + 1
    fig, axes = plt.subplots(N, ncols, figsize=(2.8 * ncols, 2.9 * N), squeeze=False)

    for i in range(N):
        img = rgb_np[i]
        H_img, W_img = img.shape[:2]

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Seq {i}\\nRGB", fontsize=9)
        axes[i, 0].axis("off")

        for e in range(E):
            a = attn[i, e]
            if a.numel() != Hf * Wf:
                a = a.flatten()
                if a.numel() < Hf * Wf:
                    pad = torch.zeros(Hf * Wf - a.numel())
                    a = torch.cat([a, pad], dim=0)
                a = a[:Hf * Wf]
            hm = a.reshape(Hf, Wf).numpy()
            hm = _normalize_attention_map(hm)
            hm = _resize_attention(hm, (H_img, W_img))

            ax = axes[i, e + 1]
            ax.imshow(img)
            ax.imshow(hm, cmap="jet", alpha=alpha, vmin=0, vmax=1)
            ax.set_title(f"Expert {e}", fontsize=9)
            ax.axis("off")

    fig.suptitle("Appearance Experts: Attention Heatmaps Overlaid on First RGB Frame",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(save_dir, "app_expert_attention_overlay_6seq.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved all-expert RGB attention overlay → {out}")

    # Also save one clearer figure per sequence.
    for i in range(N):
        img = rgb_np[i]
        H_img, W_img = img.shape[:2]
        rows = 2 if E + 1 <= 9 else 3
        cols = int(np.ceil((E + 1) / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
        axes_flat = axes.flatten()

        axes_flat[0].imshow(img)
        axes_flat[0].set_title("Original RGB", fontsize=10)
        axes_flat[0].axis("off")

        for e in range(E):
            hm = attn[i, e]
            if hm.numel() != Hf * Wf:
                hm = hm.flatten()[:Hf * Wf]
            hm = hm.reshape(Hf, Wf).numpy()
            hm = _normalize_attention_map(hm)
            hm = _resize_attention(hm, (H_img, W_img))

            ax = axes_flat[e + 1]
            ax.imshow(img)
            ax.imshow(hm, cmap="jet", alpha=alpha, vmin=0, vmax=1)
            ax.set_title(f"Expert {e}", fontsize=10)
            ax.axis("off")

        for j in range(E + 1, len(axes_flat)):
            axes_flat[j].axis("off")

        plt.tight_layout()
        out_i = os.path.join(save_dir, f"app_expert_attention_seq{i:02d}.png")
        plt.savefig(out_i, dpi=150, bbox_inches="tight")
        plt.close()


def plot_spatial_attention_heatmaps(app_spatial_attn: Dict[str, torch.Tensor],
                                    Hf: int, Wf: int,
                                    save_dir: str,
                                    max_experts: int = 8,
                                    rgb_frames: Optional[np.ndarray] = None):
    """Legacy fallback: plot mean spatial attention heatmaps.

    The preferred V4 path is plot_all_expert_attention_on_rgb(), which uses
    viz_rgb_first + viz_app_attn_all.
    """
    if not app_spatial_attn:
        print("  No spatial attention data to plot.")
        return

    E = min(len(app_spatial_attn), max_experts)
    expert_keys = sorted(app_spatial_attn.keys())[:E]

    if not HAS_MPL:
        print("\\n  Appearance Expert Spatial Attention (text mode):")
        for ek in expert_keys:
            sa = app_spatial_attn[ek]
            mean_sa = sa.mean(0)
            if mean_sa.numel() == Hf * Wf:
                hm = mean_sa.reshape(Hf, Wf)
                top_pos = torch.topk(hm.flatten(), min(3, hm.numel())).indices
                coords = [(p.item() // Wf, p.item() % Wf) for p in top_pos]
                print(f"    {ek}: top positions (h,w) = {coords}")
        return

    fig, axes = plt.subplots(2, (E + 1) // 2, figsize=(4 * ((E + 1) // 2), 8),
                             squeeze=False)
    axes = axes.flatten()
    cmap = LinearSegmentedColormap.from_list('attn', ['#f0f0f0', '#ff6b35', '#c40233'])

    last_idx = -1
    for idx, ek in enumerate(expert_keys):
        sa = app_spatial_attn[ek]
        mean_sa = sa.float().mean(0)
        if mean_sa.numel() != Hf * Wf:
            continue
        hm = mean_sa.reshape(Hf, Wf).numpy()
        ax = axes[idx]
        im = ax.imshow(hm, cmap=cmap, vmin=0, vmax=max(hm.max(), 0.01),
                       aspect='auto', interpolation='bilinear')
        ax.set_title(f'{ek}\\nmax={hm.max():.3f}', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        last_idx = idx

    for j in range(last_idx + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Appearance Expert Spatial Attention Heatmaps - Legacy Fallback',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'app_spatial_attention_legacy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved legacy spatial attention heatmap → {path}")


# ================================================================
#  Fusion Cross-Attention Analysis
# ================================================================

def plot_fusion_cross_attention(cross_attn: torch.Tensor,
                                S_rgb: int, N_pts: int,
                                save_path: str,
                                Hf: int = None, Wf: int = None):
    """Analyze and plot fusion cross-attention weights.

    Shows per-query average attention on RGB tokens vs PC tokens.

    Args:
        cross_attn: [N, K, S]  (S = S_rgb + N_pts)
        S_rgb: number of RGB spatial tokens
        N_pts: number of PC point tokens
        save_path: output file path
        Hf, Wf: feature map spatial dims for per-query spatial heatmap
    """
    if cross_attn is None or not HAS_MPL:
        return

    ca = cross_attn.float()                                    # [N, K, S]
    K = ca.size(1)

    # Per-query modality preference
    rgb_attn = ca[:, :, :S_rgb].mean(dim=(0, 2))              # [K]
    pc_attn = ca[:, :, S_rgb:].mean(dim=(0, 2))               # [K]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: RGB vs PC preference per query
    x = np.arange(K)
    w = 0.35
    axes[0].bar(x - w / 2, rgb_attn.numpy(), w, label='RGB tokens',
                color='#E8835C', alpha=0.85, edgecolor='white')
    axes[0].bar(x + w / 2, pc_attn.numpy(), w, label='PC tokens',
                color='#4C72B0', alpha=0.85, edgecolor='white')
    axes[0].set_xlabel('Query Token Index')
    axes[0].set_ylabel('Mean Cross-Attention Weight')
    axes[0].set_title('Fusion Queries: RGB vs PC Attention', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].legend()

    # Per-query spatial heatmap on RGB tokens (if Hf, Wf provided)
    if Hf is not None and Wf is not None and S_rgb == Hf * Wf:
        rgb_ca = ca[:, :, :S_rgb].mean(0)                      # [K, S_rgb]
        hm = rgb_ca.reshape(K, Hf, Wf).numpy()
        # Show top-4 queries by total RGB attention
        top4 = rgb_attn.topk(min(4, K)).indices.numpy()
        for plot_i, qi in enumerate(top4):
            ax_sub = fig.add_axes([0.55 + plot_i * 0.11, 0.15, 0.1, 0.3])
            ax_sub.imshow(hm[qi], cmap='hot', aspect='auto')
            ax_sub.set_title(f'Q{qi}', fontsize=8)
            ax_sub.axis('off')
    else:
        axes[1].text(0.5, 0.5, f'S_rgb={S_rgb}, N_pts={N_pts}\n'
                     f'Total tokens={S_rgb + N_pts}',
                     ha='center', va='center', fontsize=12)
        axes[1].set_title('Token Composition')
        axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved fusion cross-attention → {save_path}")


# ================================================================
#  Query Selection Statistics
# ================================================================

def plot_query_selection(query_scores: torch.Tensor, output_k: int,
                         save_path: str):
    """Plot how often each query is selected across the test set.

    Args:
        query_scores: [N, num_queries]
        output_k: K used in top-k selection
    """
    if not HAS_MPL or query_scores is None:
        return

    Q = query_scores.size(1)
    _, topk_idx = query_scores.topk(output_k, dim=-1)          # [N, K]
    counts = torch.zeros(Q)
    for i in range(Q):
        counts[i] = (topk_idx == i).float().sum()
    freq = counts / (query_scores.size(0) * output_k)

    fig, ax = plt.subplots(figsize=(max(10, Q * 0.2), 4))
    x = np.arange(Q)
    ideal = 1.0 / Q * output_k      # expected if perfectly balanced
    ax.bar(x, freq.numpy(), color='#55A868', alpha=0.8, edgecolor='white')
    ax.axhline(y=output_k / Q, color='red', linestyle='--', alpha=0.5,
               label=f'Ideal ({output_k / Q:.3f})')
    ax.set_xlabel('Query Index')
    ax.set_ylabel('Selection Frequency')
    ax.set_title(f'Learnable Query Selection Distribution  (K={output_k} / Q={Q})',
                 fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved query selection → {save_path}")


# ================================================================
#  Main entry point
# ================================================================

def generate_v3_plots(viz_data: dict, output_dir: str, model=None):
    """Generate all V3-specific plots from viz_data dict.

    Called by GaitMoEOracleV3._save_and_plot_viz() or from CLI.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Expert usage (appearance + structure)
    usage = {}
    for tag in ['app', 'str']:
        key = f'viz_{tag}_logits'
        if key in viz_data and isinstance(viz_data[key], torch.Tensor):
            usage[tag] = compute_expert_usage_from_logits(viz_data[key], top_k=2)
    if usage:
        print_expert_usage_table(usage, title="V3 Test-Set Expert Usage")
        plot_expert_usage(usage, os.path.join(output_dir, 'expert_usage.png'))

    # 2) Spatial attention heatmaps
    hw = viz_data.get('viz_spatial_hw', None)
    if hw is not None:
        if isinstance(hw, torch.Tensor):
            Hf, Wf = int(hw[0].item()), int(hw[1].item())
        else:
            Hf, Wf = int(hw[0]), int(hw[1])
    else:
        Hf, Wf = 16, 11       # fallback guess

    # Preferred V4 visualization: six random sequences, first RGB frame,
    # all appearance experts overlaid on the same image.
    if 'viz_rgb_first' in viz_data and 'viz_app_attn_all' in viz_data:
        plot_all_expert_attention_on_rgb(
            viz_data['viz_rgb_first'], viz_data['viz_app_attn_all'],
            Hf, Wf, output_dir)
    elif 'app_spatial_attn' in viz_data:
        plot_spatial_attention_heatmaps(
            viz_data['app_spatial_attn'], Hf, Wf, output_dir)

    # 3) Fusion cross-attention
    if 'fusion_cross_attn' in viz_data:
        S_rgb = viz_data.get('S_rgb', Hf * Wf)
        N_pts = viz_data.get('N_pts', 256)
        plot_fusion_cross_attention(
            viz_data['fusion_cross_attn'], S_rgb, N_pts,
            os.path.join(output_dir, 'fusion_cross_attention.png'),
            Hf=Hf, Wf=Wf)

    # 4) Query selection
    if 'viz_query_scores' in viz_data:
        qs = viz_data['viz_query_scores']
        if not isinstance(qs, torch.Tensor):
            qs = torch.as_tensor(qs)
        output_k = model.output_k if model is not None else 16
        plot_query_selection(
            qs, output_k,
            os.path.join(output_dir, 'query_selection.png'))

    # 5) Quality distribution
    if 'viz_rgb_quality' in viz_data and HAS_MPL:
        rq = torch.as_tensor(viz_data['viz_rgb_quality']).cpu().numpy().flatten()
        pq = torch.as_tensor(viz_data['viz_pc_quality']).cpu().numpy().flatten()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(rq, bins=50, color='#E8835C', alpha=0.7, edgecolor='white')
        axes[0].set_title(f'RGB Quality (mean={rq.mean():.3f})')
        axes[0].axvline(x=rq.mean(), color='red', linestyle='--')
        axes[1].hist(pq, bins=50, color='#4C72B0', alpha=0.7, edgecolor='white')
        axes[1].set_title(f'PC Quality (mean={pq.mean():.3f})')
        axes[1].axvline(x=pq.mean(), color='red', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_distribution.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


# ================================================================
#  CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='GaitMoE V3 Visualization')
    parser.add_argument('--viz_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_k', type=int, default=16)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.viz_data) or '.'

    viz_data = safe_torch_load(args.viz_data, map_location='cpu')
    print(f"  Loaded V3 viz data: keys={list(viz_data.keys())}")

    class _FakeModel:
        output_k = args.output_k

    generate_v3_plots(viz_data, args.output_dir, model=_FakeModel())
    print(f"\n  All V3 plots saved in {args.output_dir}/")


if __name__ == '__main__':
    main()
