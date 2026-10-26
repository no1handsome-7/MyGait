#!/usr/bin/env python3
"""GaitMoEOracleV2Max — Publication-Quality Visualization Suite.

Generates 8 figures for CVPR-level submission:
  Fig 1. t-SNE Embedding Comparison (Base vs Full)
  Fig 2. Intra-/Inter-class Distance Distribution
  Fig 3. Heterogeneous Expert Body-Part Specialization
  Fig 4. Condition-Adaptive Expert Routing Heatmap
  Fig 5. Fusion Expert Modality Preference (RGB vs PC)
  Fig 6. QACT Quality-Condition Correlation
  Fig 7. QACT Context Token Attention Map
  Fig 8. Expert Routing Decision t-SNE

Usage:
  # After inference with collect_visualization: true
  python gait_moe_visualize_full.py --viz_dir ./viz_output_v2max_lrgait_vis

  # Compare two models (for t-SNE / distance figures)
  python gait_moe_visualize_full.py --viz_dir ./viz_output_v2max_lrgait \
                                    --base_dir ./viz_output_base_lrgait
"""

import os, sys, argparse, math
from collections import defaultdict
import numpy as np
import torch

# ---- Style Constants ----
DPI = 300
FIG_DIR_DEFAULT = './viz_figures/v2maxcalib/lrgait'

# Colorblind-friendly palette (Wong 2011, Nature Methods)
PALETTE = {
    'blue':    '#0072B2',
    'orange':  '#E69F00',
    'green':   '#009E73',
    'pink':    '#CC79A7',
    'cyan':    '#56B4E9',
    'red':     '#D55E00',
    'yellow':  '#F0E442',
    'gray':    '#999999',
}

EXPERT_TYPE_COLORS = {
    'PointwiseExpert':        PALETTE['blue'],
    'LocalConvExpert':        PALETTE['orange'],
    'GlobalAttentionExpert':  PALETTE['red'],
    'PartExpertBlock':        PALETTE['gray'],
    'SharedExpert':           PALETTE['green'],
}

EXPERT_TYPE_LABELS = {
    'PointwiseExpert':        'Pointwise',
    'LocalConvExpert':        'Local Conv',
    'GlobalAttentionExpert':  'Global Attn',
    'PartExpertBlock':        'Standard',
    'SharedExpert':           'Shared',
}

# 16-part body labels (HPP top-to-bottom horizontal slices)
BODY_PARTS_16 = [
    'Head', 'Forehead', 'Neck', 'Shoulder',
    'Upper Arm', 'Chest', 'Elbow', 'Waist',
    'Hip', 'Wrist', 'Thigh-U', 'Thigh-L',
    'Knee', 'Calf', 'Ankle', 'Foot',
]


def _setup_style():
    """Set publication-quality matplotlib defaults."""
    import matplotlib
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth': 0.8,
        'axes.grid': False,
        'grid.linewidth': 0.4,
        'lines.linewidth': 1.2,
    })


def load_viz_data(viz_dir):
    """Load saved viz data from a directory."""
    pth_path = os.path.join(viz_dir, 'viz_data.pth')
    if os.path.exists(pth_path):
        return torch.load(pth_path, map_location='cpu', weights_only=False)
    raise FileNotFoundError(f"No viz_data.pth in {viz_dir}")


# ==================================================================
#  Fig 1. t-SNE Embedding Comparison
# ==================================================================

def plot_tsne(viz_data, out_dir, info_dict=None, base_data=None,
              base_info=None, max_samples=3000):
    """t-SNE of final embeddings, optionally comparing Base vs Full."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    _setup_style()

    def _get_embeds_labels(data, info):
        if info is not None and 'embeddings' in info:
            emb = np.array(info['embeddings'])
            if emb.ndim == 3:
                emb = emb.mean(axis=-1)
            labels = np.array(info.get('labels', []))
            return emb, labels
        return None, None

    emb_full, lbl_full = _get_embeds_labels(viz_data, info_dict)
    emb_base, lbl_base = _get_embeds_labels(base_data, base_info)

    if emb_full is None:
        print("[Fig 1] No embeddings found. Skipping t-SNE.")
        return

    ncols = 2 if emb_base is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 3.2))
    if ncols == 1:
        axes = [axes]

    for ax, emb, lbl, title in zip(
        axes,
        [emb_base, emb_full] if emb_base is not None else [emb_full],
        [lbl_base, lbl_full] if lbl_base is not None else [lbl_full],
        ['(a) Base Model', '(b) Full Model'] if ncols == 2 else ['Embedding t-SNE'],
    ):
        if emb is None:
            continue
        n = min(len(emb), max_samples)
        idx = np.random.choice(len(emb), n, replace=False) if len(emb) > n else np.arange(n)
        e, l = emb[idx], lbl[idx] if len(lbl) > 0 else np.zeros(n)

        tsne = TSNE(n_components=2, perplexity=min(30, n - 1),
                     random_state=42, n_iter=1000, init='pca',
                     learning_rate='auto')
        pts = tsne.fit_transform(e)

        unique_labels = np.unique(l)
        cmap = plt.cm.get_cmap('tab20', len(unique_labels))
        for i, ul in enumerate(unique_labels):
            mask = l == ul
            ax.scatter(pts[mask, 0], pts[mask, 1], c=[cmap(i % 20)],
                       s=6, alpha=0.6, edgecolors='none', rasterized=True)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig.tight_layout(w_pad=1.5)
    savepath = os.path.join(out_dir, 'fig1_tsne.pdf')
    fig.savefig(savepath); fig.savefig(savepath.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"[Fig 1] t-SNE → {savepath}")


# ==================================================================
#  Fig 2. Intra-/Inter-class Distance Distribution
# ==================================================================

def plot_distance_distribution(viz_data, out_dir, info_dict=None,
                                base_info=None, max_pairs=50000):
    """Histogram + KDE of intra- and inter-class cosine distances."""
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    _setup_style()

    def _compute_distances(info, max_p):
        if info is None or 'embeddings' not in info:
            return None, None
        emb = np.array(info['embeddings'])
        if emb.ndim == 3:
            emb = emb.mean(axis=-1)
        labels = np.array(info.get('labels', []))
        if len(labels) == 0:
            return None, None

        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        n = len(emb_norm)
        intra, inter = [], []
        np.random.seed(42)
        count = 0
        for _ in range(max_p * 5):
            i, j = np.random.randint(0, n, 2)
            if i == j:
                continue
            d = float(np.dot(emb_norm[i], emb_norm[j]))
            if labels[i] == labels[j]:
                intra.append(d)
            else:
                inter.append(d)
            count += 1
            if count >= max_p:
                break
        return np.array(intra), np.array(inter)

    intra_full, inter_full = _compute_distances(info_dict, max_pairs)
    intra_base, inter_base = _compute_distances(base_info, max_pairs)

    if intra_full is None:
        print("[Fig 2] No embeddings found. Skipping distance distribution.")
        return

    ncols = 2 if intra_base is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 2.8))
    if ncols == 1:
        axes = [axes]

    for ax, intra, inter, title in zip(
        axes,
        [intra_base, intra_full] if intra_base is not None else [intra_full],
        [inter_base, inter_full] if inter_base is not None else [inter_full],
        ['(a) Base Model', '(b) Full Model'] if ncols == 2 else ['Distance Distribution'],
    ):
        if intra is None:
            continue
        bins = np.linspace(-0.3, 1.0, 80)
        ax.hist(inter, bins, alpha=0.35, color=PALETTE['blue'], density=True,
                label='Inter-class', edgecolor='none')
        ax.hist(intra, bins, alpha=0.35, color=PALETTE['red'], density=True,
                label='Intra-class', edgecolor='none')
        # KDE curves
        for data, color in [(inter, PALETTE['blue']), (intra, PALETTE['red'])]:
            if len(data) > 10:
                kde = gaussian_kde(data, bw_method=0.05)
                x = np.linspace(-0.3, 1.0, 300)
                ax.plot(x, kde(x), color=color, linewidth=1.5)
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Density')
        ax.set_title(title, fontweight='bold')
        ax.legend(frameon=False, loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout(w_pad=1.5)
    savepath = os.path.join(out_dir, 'fig2_distance_dist.pdf')
    fig.savefig(savepath); fig.savefig(savepath.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"[Fig 2] Distance distribution → {savepath}")


# ==================================================================
#  Fig 3. Heterogeneous Expert Body-Part Specialization
# ==================================================================

def plot_expert_body_part_specialization(viz_data, out_dir, parts_num=16):
    """Heatmap of each expert's learned body-part attention (sigmoid(part_logit)).

    Shows that heterogeneous experts develop distinct specializations:
      - Pointwise: isolated body regions
      - LocalConv: adjacent part groups (limb coordination)
      - GlobalAttn: holistic body structure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.gridspec as gridspec

    _setup_style()

    for branch, branch_label in [('app', 'Appearance'), ('str', 'Structure')]:
        key = f'{branch}_expert_info'
        if key not in viz_data or not viz_data[key]:
            continue

        expert_info = viz_data[key]
        n_experts = len(expert_info)
        ncols = min(4, n_experts)
        nrows = math.ceil(n_experts / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 3.0 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[np.newaxis, :]
        elif ncols == 1:
            axes = axes[:, np.newaxis]

        for idx in range(n_experts):
            row, col = idx // ncols, idx % ncols
            ax = axes[row, col]
            ek = f'expert_{idx}'
            info = expert_info[ek]
            etype = info.get('type', 'Unknown')
            pa = info.get('part_attention', np.ones(parts_num) * 0.5)

            # Draw body outline as stacked horizontal bars
            colors = plt.cm.YlOrRd(pa)
            labels = BODY_PARTS_16[:parts_num]

            y_pos = np.arange(parts_num)[::-1]
            bars = ax.barh(y_pos, pa, height=0.85, color=colors,
                           edgecolor='white', linewidth=0.3)

            ax.set_xlim(0, 1)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=6)
            ax.set_xlabel('Attention', fontsize=7)

            # Color-coded title by expert type
            color = EXPERT_TYPE_COLORS.get(etype, PALETTE['gray'])
            label = EXPERT_TYPE_LABELS.get(etype, etype)
            ax.set_title(f'E{idx}: {label}', fontsize=8, fontweight='bold',
                         color=color)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Hide unused axes
        for idx in range(n_experts, nrows * ncols):
            row, col = idx // ncols, idx % ncols
            axes[row, col].set_visible(False)

        fig.suptitle(f'{branch_label} Branch Expert Specialization',
                     fontsize=11, fontweight='bold', y=1.02)
        fig.tight_layout()
        savepath = os.path.join(out_dir, f'fig3_{branch}_expert_specialization.pdf')
        fig.savefig(savepath); fig.savefig(savepath.replace('.pdf', '.png'))
        plt.close(fig)
        print(f"[Fig 3] {branch_label} expert specialization → {savepath}")


# ==================================================================
#  Fig 4. Condition-Adaptive Expert Routing Heatmap
# ==================================================================

def plot_expert_routing_heatmap(viz_data, out_dir):
    """Heatmap: condition bins (rows) × experts (columns).

    Shows that routing is condition-dependent, not random.
    Uses brightness/distance from condition stats as row labels.
    """
    import matplotlib.pyplot as plt

    _setup_style()

    cond_stats = viz_data.get('viz_cond_stats')
    if cond_stats is None:
        print("[Fig 4] No condition stats. Skipping routing heatmap.")
        return
    cond_stats = cond_stats.numpy() if torch.is_tensor(cond_stats) else np.array(cond_stats)

    panels = []
    for tag, key in [('Appearance', 'viz_app_logits'),
                     ('Structure', 'viz_str_logits'),
                     ('Fusion', 'viz_fus_logits')]:
        logits = viz_data.get(key)
        if logits is None:
            continue
        logits = logits.numpy() if torch.is_tensor(logits) else np.array(logits)
        panels.append((tag, logits))

    if not panels:
        print("[Fig 4] No routing logits. Skipping.")
        return

    # Condition binning: brightness (col 0) × distance (col 3)
    brightness = cond_stats[:, 0]
    distance = cond_stats[:, 3]
    n_bins = 4

    bri_bins = np.quantile(brightness, np.linspace(0, 1, n_bins + 1))
    dist_bins = np.quantile(distance, np.linspace(0, 1, n_bins + 1))
    bri_labels = [f'Bri {i+1}' for i in range(n_bins)]
    dist_labels = [f'Dist {i+1}' for i in range(n_bins)]
    cond_labels = [f'{b} / {d}' for b in bri_labels for d in dist_labels]
    n_conds = n_bins * n_bins

    bri_idx = np.clip(np.digitize(brightness, bri_bins[1:-1]), 0, n_bins - 1)
    dist_idx = np.clip(np.digitize(distance, dist_bins[1:-1]), 0, n_bins - 1)
    cond_idx = bri_idx * n_bins + dist_idx

    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 4.0))
    if len(panels) == 1:
        axes = [axes]

    for ax, (tag, logits) in zip(axes, panels):
        E = logits.shape[1]
        top_k = 2
        top_idx = np.argsort(logits, axis=1)[:, -top_k:]

        heatmap = np.zeros((n_conds, E))
        counts = np.zeros(n_conds)
        for s in range(len(logits)):
            ci = cond_idx[s]
            for k in range(top_k):
                heatmap[ci, top_idx[s, k]] += 1
            counts[ci] += 1

        for c in range(n_conds):
            if counts[c] > 0:
                heatmap[c] /= counts[c]

        im = ax.imshow(heatmap, aspect='auto', cmap='YlOrRd',
                        interpolation='nearest')
        ax.set_xticks(range(E))
        ax.set_xticklabels([f'E{i}' for i in range(E)], fontsize=6)
        ax.set_yticks(range(n_conds))
        ax.set_yticklabels(cond_labels, fontsize=5.5)
        ax.set_xlabel('Expert Index')
        ax.set_ylabel('Condition (Brightness / Distance)')
        ax.set_title(f'{tag} MoE', fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Activation Freq.')

    fig.suptitle('Condition-Adaptive Expert Routing', fontsize=11,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    savepath = os.path.join(out_dir, 'fig4_routing_heatmap.pdf')
    fig.savefig(savepath); fig.savefig(savepath.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"[Fig 4] Routing heatmap → {savepath}")


# ==================================================================
#  Fig 5. Fusion Expert Modality Preference
# ==================================================================

def plot_fusion_modality_preference(viz_data, out_dir):
    """Grouped bar chart: per-expert RGB vs PC token preference.

    Shows that different fusion experts specialize in different modalities,
    enabling adaptive information routing.
    """
    import matplotlib.pyplot as plt

    _setup_style()

    fus_info = viz_data.get('fus_expert_info')
    if not fus_info:
        print("[Fig 5] No fusion expert info. Skipping.")
        return

    n_experts = len(fus_info)
    rgb_prefs, pc_prefs, labels = [], [], []
    for i in range(n_experts):
        ek = f'expert_{i}'
        if ek not in fus_info:
            continue
        info = fus_info[ek]
        rgb_prefs.append(info.get('rgb_preference', 0.5))
        pc_prefs.append(info.get('pc_preference', 0.5))
        labels.append(f'E{i}')

    if not labels:
        return

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    bars_rgb = ax.bar(x - w / 2, rgb_prefs, w, label='RGB Preference',
                      color=PALETTE['blue'], edgecolor='white', linewidth=0.5)
    bars_pc = ax.bar(x + w / 2, pc_prefs, w, label='PC Preference',
                     color=PALETTE['orange'], edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Mean Token Attention')
    ax.set_xlabel('Fusion Expert')
    ax.set_title('Fusion Expert Modality Preference', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, loc='upper right')
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate dominant modality
    for i in range(len(labels)):
        if abs(rgb_prefs[i] - pc_prefs[i]) > 0.05:
            dominant = 'RGB' if rgb_prefs[i] > pc_prefs[i] else 'PC'
            ratio = max(rgb_prefs[i], pc_prefs[i]) / (min(rgb_prefs[i], pc_prefs[i]) + 1e-8)
            ax.annotate(f'{dominant}\n×{ratio:.1f}', (x[i], max(rgb_prefs[i], pc_prefs[i]) + 0.02),
                        ha='center', fontsize=5.5, color='dimgray')

    fig.tight_layout()
    savepath = os.path.join(out_dir, 'fig5_fusion_modality_pref.pdf')
    fig.savefig(savepath); fig.savefig(savepath.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"[Fig 5] Fusion modality preference → {savepath}")


# ==================================================================
#  Fig 6. QACT Quality-Condition Correlation
# ==================================================================

def plot_qact_quality_condition(viz_data, out_dir):
    """Box plot: QACT quality scores stratified by brightness / distance bins.

    Validates that QACT learns:
      - RGB quality ↓ when brightness is low (night)
      - PC quality ↓ when distance is large (sparse point cloud)
    """
    import matplotlib.pyplot as plt

    _setup_style()

    rgb_q = viz_data.get('viz_rgb_quality')
    pc_q = viz_data.get('viz_pc_quality')
    cond = viz_data.get('viz_cond_stats')
    if rgb_q is None or cond is None:
        print("[Fig 6] No quality / condition data. Skipping.")
        return

    rgb_q = rgb_q.numpy().flatten() if torch.is_tensor(rgb_q) else np.array(rgb_q).flatten()
    pc_q = pc_q.numpy().flatten() if torch.is_tensor(pc_q) else np.array(pc_q).flatten()
    cond = cond.numpy() if torch.is_tensor(cond) else np.array(cond)

    brightness = cond[:, 0]
    distance = cond[:, 3]
    n_bins = 5

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # Panel (a): RGB quality vs brightness
    ax = axes[0]
    bri_edges = np.quantile(brightness, np.linspace(0, 1, n_bins + 1))
    bri_idx = np.clip(np.digitize(brightness, bri_edges[1:-1]), 0, n_bins - 1)
    groups = [rgb_q[bri_idx == b] for b in range(n_bins)]
    bp = ax.boxplot(groups, positions=range(n_bins), widths=0.6, patch_artist=True,
                    showfliers=False, medianprops={'color': 'black', 'linewidth': 1.2})
    for patch in bp['boxes']:
        patch.set_facecolor(PALETTE['cyan'])
        patch.set_alpha(0.7)
    ax.set_xticklabels([f'Q{i+1}' for i in range(n_bins)])
    ax.set_xlabel('Brightness Quintile (dark → bright)')
    ax.set_ylabel('RGB Quality Score')
    ax.set_title('(a) RGB Quality vs. Brightness', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel (b): PC quality vs distance
    ax = axes[1]
    dist_edges = np.quantile(distance, np.linspace(0, 1, n_bins + 1))
    dist_idx = np.clip(np.digitize(distance, dist_edges[1:-1]), 0, n_bins - 1)
    groups = [pc_q[dist_idx == d] for d in range(n_bins)]
    bp = ax.boxplot(groups, positions=range(n_bins), widths=0.6, patch_artist=True,
                    showfliers=False, medianprops={'color': 'black', 'linewidth': 1.2})
    for patch in bp['boxes']:
        patch.set_facecolor(PALETTE['orange'])
        patch.set_alpha(0.7)
    ax.set_xticklabels([f'Q{i+1}' for i in range(n_bins)])
    ax.set_xlabel('Distance Quintile (near → far)')
    ax.set_ylabel('PC Quality Score')
    ax.set_title('(b) PC Quality vs. Distance', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('QACT Quality Scores Reflect Input Conditions', fontsize=11,
                 fontweight='bold', y=1.05)
    fig.tight_layout()
    savepath = os.path.join(out_dir, 'fig6_qact_quality_condition.pdf')
    fig.savefig(savepath); fig.savefig(savepath.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"[Fig 6] QACT quality-condition → {savepath}")


# ==================================================================
#  Fig 7. QACT Context Token Attention Map
# ==================================================================

def plot_qact_token_attention(viz_data, out_dir, parts_num=16):
    """Heatmap: K context tokens × P body parts, for both RGB and PC.

    Shows that different context tokens attend to different body regions,
    capturing complementary quality signals (e.g., token 1 → legs for
    gait dynamics, token 2 → torso for clothing noise).
    """
    import matplotlib.pyplot as plt

    _setup_style()

    rgb_attn = viz_data.get('viz_qact_rgb_attn')
    pc_attn = viz_data.get('viz_qact_pc_attn')
    if rgb_attn is None:
        print("[Fig 7] No QACT attention data. Skipping.")
        return

    rgb_attn = rgb_attn.numpy() if torch.is_tensor(rgb_attn) else np.array(rgb_attn)
    pc_attn = pc_attn.numpy() if torch.is_tensor(pc_attn) else np.array(pc_attn)

    # Average over all test samples: [K, P]
    # Attention shape: [N, K, P] or [N, num_heads, K, P]
    if rgb_attn.ndim == 4:
        rgb_mean = rgb_attn.mean(axis=(0, 1))
        pc_mean = pc_attn.mean(axis=(0, 1))
    else:
        rgb_mean = rgb_attn.mean(axis=0)
        pc_mean = pc_attn.mean(axis=0)

    K = rgb_mean.shape[0]
    P = rgb_mean.shape[1]
    labels = BODY_PARTS_16[:P]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, max(2.0, 0.3 * K + 1.0)))

    for ax, attn_map, title, cmap in zip(
        axes,
        [rgb_mean, pc_mean],
        ['(a) RGB Self-Probe', '(b) PC Self-Probe'],
        ['Blues', 'Oranges'],
    ):
        im = ax.imshow(attn_map, aspect='auto', cmap=cmap, interpolation='nearest')
        ax.set_xticks(range(P))
        ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=5.5)
        ax.set_yticks(range(K))
        ax.set_yticklabels([f'Token {i}' for i in range(K)], fontsize=7)
        ax.set_xlabel('Body Part')
        ax.set_ylabel('Context Token')
        ax.set_title(title, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Attention')

    fig.suptitle('QACT Context Token Attention Patterns', fontsize=11,
                 fontweight='bold', y=1.04)
    fig.tight_layout()
    savepath = os.path.join(out_dir, 'fig7_qact_token_attention.pdf')
    fig.savefig(savepath); fig.savefig(savepath.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"[Fig 7] QACT token attention → {savepath}")


# ==================================================================
#  Fig 8. Expert Routing Decision t-SNE
# ==================================================================

def plot_expert_routing_tsne(viz_data, out_dir, max_samples=3000):
    """t-SNE of condition embeddings, colored by dominant expert.

    Shows that routing decisions form meaningful clusters in condition space,
    i.e., the router learns a structured condition → expert mapping.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    _setup_style()

    cond_emb = viz_data.get('viz_qact_cond_emb')
    fus_logits = viz_data.get('viz_fus_logits')
    if cond_emb is None or fus_logits is None:
        print("[Fig 8] No condition embedding or routing logits. Skipping.")
        return

    cond_emb = cond_emb.numpy() if torch.is_tensor(cond_emb) else np.array(cond_emb)
    fus_logits = fus_logits.numpy() if torch.is_tensor(fus_logits) else np.array(fus_logits)

    n = min(len(cond_emb), max_samples)
    idx = np.random.choice(len(cond_emb), n, replace=False) if len(cond_emb) > n else np.arange(n)
    emb = cond_emb[idx]
    logits = fus_logits[idx]
    dominant = logits.argmax(axis=1)

    tsne = TSNE(n_components=2, perplexity=min(30, n - 1),
                 random_state=42, n_iter=1000, init='pca', learning_rate='auto')
    pts = tsne.fit_transform(emb)

    E = logits.shape[1]
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    cmap = plt.cm.get_cmap('Set2', E)
    for e in range(E):
        mask = dominant == e
        if mask.sum() == 0:
            continue
        ax.scatter(pts[mask, 0], pts[mask, 1], c=[cmap(e)], s=8,
                   alpha=0.6, edgecolors='none', label=f'Expert {e}',
                   rasterized=True)

    ax.set_title('Fusion Router Decision Space', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(fontsize=6, markerscale=1.5, frameon=False,
              bbox_to_anchor=(1.02, 1), loc='upper left')

    fig.tight_layout()
    savepath = os.path.join(out_dir, 'fig8_routing_tsne.pdf')
    fig.savefig(savepath); fig.savefig(savepath.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"[Fig 8] Routing decision t-SNE → {savepath}")


# ==================================================================
#  Runner
# ==================================================================

def run_all_visualizations(viz_data, info_dict=None, out_dir=FIG_DIR_DEFAULT,
                           parts_num=16, base_data=None, base_info=None):
    """Run all 8 visualization generators."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  GaitMoEOracleV2Max Visualization Suite")
    print(f"  Output directory: {out_dir}")
    print(f"{'='*60}")

    plot_tsne(viz_data, out_dir, info_dict, base_data, base_info)
    plot_distance_distribution(viz_data, out_dir, info_dict, base_info)
    plot_expert_body_part_specialization(viz_data, out_dir, parts_num)
    plot_expert_routing_heatmap(viz_data, out_dir)
    plot_fusion_modality_preference(viz_data, out_dir)
    plot_qact_quality_condition(viz_data, out_dir)
    plot_qact_token_attention(viz_data, out_dir, parts_num)
    plot_expert_routing_tsne(viz_data, out_dir)

    print(f"\n{'='*60}")
    print(f"  All figures saved to: {out_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='GaitMoEOracleV2Max Visualization')
    parser.add_argument('--viz_dir', type=str, required=True,
                        help='Directory containing viz_data.pth from full model')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='(Optional) Directory containing viz_data.pth from base model')
    parser.add_argument('--out_dir', type=str, default=FIG_DIR_DEFAULT,
                        help='Output directory for figures')
    parser.add_argument('--parts_num', type=int, default=16)
    args = parser.parse_args()

    viz_data = load_viz_data(args.viz_dir)
    info_path = os.path.join(args.viz_dir, 'inference_info.pth')
    info_dict = torch.load(info_path, map_location='cpu',
                           weights_only=False) if os.path.exists(info_path) else None

    base_data, base_info = None, None
    if args.base_dir:
        base_data = load_viz_data(args.base_dir)
        bp = os.path.join(args.base_dir, 'inference_info.pth')
        base_info = torch.load(bp, map_location='cpu',
                               weights_only=False) if os.path.exists(bp) else None

    run_all_visualizations(viz_data, info_dict, args.out_dir,
                           args.parts_num, base_data, base_info)


if __name__ == '__main__':
    main()
