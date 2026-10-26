#!/usr/bin/env python3
"""GaitMoEOracleV2Max — Publication-Quality Visualization Suite.

Generates legacy V2Max figures plus QACT-Calib diagnostics:
  Fig 1. t-SNE Embedding Comparison (Base vs Full)
  Fig 2. Intra-/Inter-class Distance Distribution
  Fig 3. Heterogeneous Expert Body-Part Specialization
  Fig 4. Condition-Adaptive Expert Routing Heatmap
  Fig 5. Fusion Expert Modality Preference (RGB vs PC)
  Fig 6. QACT Quality-Condition Correlation
  Fig 7. QACT Context Token Attention Map
  Fig 8. Expert Routing Decision t-SNE
  Fig 9. QACT Quality Decomposition
  Fig 10. Quality-Gated Aux Ensemble
  Fig 11. QACT Condition Embedding Quality t-SNE

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
FIG_DIR_DEFAULT = './viz_figures/v2maxcalibvis/lrgait'

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


def _as_numpy(x):
    """Convert tensors/lists to numpy arrays without changing semantic shape."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        vals = [_as_numpy(v) for v in x if v is not None]
        if not vals:
            return None
        try:
            return np.concatenate(vals, axis=0)
        except Exception:
            try:
                return np.stack(vals, axis=0)
            except Exception:
                return np.array(vals, dtype=object)
    try:
        return np.asarray(x)
    except Exception:
        return x


def _safe_flatten_quality(x):
    x = _as_numpy(x)
    if x is None:
        return None
    x = np.asarray(x)
    if x.ndim == 0:
        return x.reshape(1)
    if x.ndim >= 2:
        return x.reshape(x.shape[0], -1)[:, 0]
    return x.reshape(-1)


def _safe_matrix(x):
    x = _as_numpy(x)
    if x is None:
        return None
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    return x.reshape(x.shape[0], -1) if x.ndim > 2 and x.shape[-1] not in (16, 32) else x


def _alias(data, canonical, candidates):
    if canonical in data and data[canonical] is not None:
        return
    for c in candidates:
        if c in data and data[c] is not None:
            data[canonical] = data[c]
            return


def _normalize_viz_data(data):
    """Normalize old V2Max and new V2Max-Calib visualization keys.

    Old scripts expected:
      viz_qact_rgb_attn / viz_qact_pc_attn / viz_cond_stats / fus_expert_info.
    New model may emit:
      viz_qact_rgb_self_attn / viz_qact_pc_self_attn / viz_qact_condition_stats.
    This function makes both formats available.
    """
    if data is None:
        return {}
    data = dict(data)

    _alias(data, 'viz_qact_rgb_attn',
           ['viz_qact_rgb_self_attn', 'qact_rgb_attn', 'rgb_self_attn_w'])
    _alias(data, 'viz_qact_pc_attn',
           ['viz_qact_pc_self_attn', 'qact_pc_attn', 'pc_self_attn_w'])
    _alias(data, 'viz_qact_rgb_self_attn',
           ['viz_qact_rgb_attn', 'qact_rgb_attn', 'rgb_self_attn_w'])
    _alias(data, 'viz_qact_pc_self_attn',
           ['viz_qact_pc_attn', 'qact_pc_attn', 'pc_self_attn_w'])

    _alias(data, 'viz_cond_stats',
           ['viz_qact_condition_stats', 'condition_stats', 'cond_stats',
            'viz_condition_stats'])
    _alias(data, 'viz_qact_condition_stats',
           ['viz_cond_stats', 'condition_stats', 'cond_stats'])

    # If quality is saved as [N,2], expose branch-wise fields too.
    q = data.get('viz_qact_quality', None)
    if q is not None:
        qn = _as_numpy(q)
        if qn is not None and np.asarray(qn).ndim >= 2 and np.asarray(qn).shape[-1] >= 2:
            qn = np.asarray(qn).reshape(np.asarray(qn).shape[0], -1)
            data.setdefault('viz_rgb_quality', torch.as_tensor(qn[:, :1]))
            data.setdefault('viz_pc_quality', torch.as_tensor(qn[:, 1:2]))

    # Conversely, combine branch-wise quality for the new diagnostic figures.
    if 'viz_qact_quality' not in data:
        rq = _safe_flatten_quality(data.get('viz_rgb_quality'))
        pq = _safe_flatten_quality(data.get('viz_pc_quality'))
        if rq is not None and pq is not None and len(rq) == len(pq):
            data['viz_qact_quality'] = torch.as_tensor(np.stack([rq, pq], axis=-1))

    return data


def load_viz_data(viz_dir):
    """Load and normalize saved viz_data.pth from a directory."""
    pth_path = os.path.join(viz_dir, 'viz_data.pth')
    if os.path.exists(pth_path):
        data = torch.load(pth_path, map_location='cpu', weights_only=False)
        return _normalize_viz_data(data)
    raise FileNotFoundError(f"No viz_data.pth in {viz_dir}")




def _make_tsne(n, random_state=42):
    """Create a TSNE object compatible with both old and new scikit-learn."""
    from sklearn.manifold import TSNE
    import inspect
    kwargs = dict(n_components=2, perplexity=max(1, min(30, n - 1)),
                  random_state=random_state, init='pca', learning_rate='auto')
    params = inspect.signature(TSNE).parameters
    if 'max_iter' in params:
        kwargs['max_iter'] = 1000
    else:
        kwargs['n_iter'] = 1000
    return TSNE(**kwargs)

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

        tsne = _make_tsne(n, random_state=42)
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

    Preferred rows use brightness/distance from viz_cond_stats.  If a legacy
    viz_data.pth lacks condition stats, the function falls back to RGB/PC
    quality bins so old experiments can still produce a routing heatmap.
    """
    import matplotlib.pyplot as plt

    _setup_style()

    cond_stats = _as_numpy(viz_data.get('viz_cond_stats'))
    rgb_q = _safe_flatten_quality(viz_data.get('viz_rgb_quality'))
    pc_q = _safe_flatten_quality(viz_data.get('viz_pc_quality'))

    panels = []
    for tag, key in [('Appearance', 'viz_app_logits'),
                     ('Structure', 'viz_str_logits'),
                     ('Fusion', 'viz_fus_logits')]:
        logits = _as_numpy(viz_data.get(key))
        if logits is None:
            continue
        logits = np.asarray(logits)
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        panels.append((tag, logits))

    if not panels:
        print("[Fig 4] No routing logits. Skipping.")
        return

    n = min(len(panels[0][1]), *(len(p[1]) for p in panels))
    n_bins = 4

    if cond_stats is not None and len(cond_stats) >= n and np.asarray(cond_stats).shape[1] >= 4:
        cond_stats = np.asarray(cond_stats)[:n]
        brightness = cond_stats[:, 0]
        distance = cond_stats[:, 3]
        bri_bins = np.quantile(brightness, np.linspace(0, 1, n_bins + 1))
        dist_bins = np.quantile(distance, np.linspace(0, 1, n_bins + 1))
        bri_idx = np.clip(np.digitize(brightness, bri_bins[1:-1]), 0, n_bins - 1)
        dist_idx = np.clip(np.digitize(distance, dist_bins[1:-1]), 0, n_bins - 1)
        cond_idx = bri_idx * n_bins + dist_idx
        cond_labels = [f'Bri {b+1} / Dist {d+1}' for b in range(n_bins) for d in range(n_bins)]
        ylabel = 'Condition (Brightness / Distance)'
    elif rgb_q is not None and pc_q is not None and len(rgb_q) >= n and len(pc_q) >= n:
        rgb_q, pc_q = rgb_q[:n], pc_q[:n]
        rb = np.quantile(rgb_q, np.linspace(0, 1, n_bins + 1))
        pb = np.quantile(pc_q, np.linspace(0, 1, n_bins + 1))
        r_idx = np.clip(np.digitize(rgb_q, rb[1:-1]), 0, n_bins - 1)
        p_idx = np.clip(np.digitize(pc_q, pb[1:-1]), 0, n_bins - 1)
        cond_idx = r_idx * n_bins + p_idx
        cond_labels = [f'RGB-Q {r+1} / PC-Q {p+1}' for r in range(n_bins) for p in range(n_bins)]
        ylabel = 'Condition (RGB Quality / PC Quality)'
        print("[Fig 4] No condition stats; using quality-bin fallback.")
    else:
        print("[Fig 4] No condition stats or quality fallback. Skipping routing heatmap.")
        return

    n_conds = n_bins * n_bins
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 4.2))
    if len(panels) == 1:
        axes = [axes]

    for ax, (tag, logits) in zip(axes, panels):
        logits = np.asarray(logits)[:n]
        E = logits.shape[1]
        top_k = min(2, E)
        top_idx = np.argsort(logits, axis=1)[:, -top_k:]

        heatmap = np.zeros((n_conds, E), dtype=np.float32)
        counts = np.zeros(n_conds, dtype=np.float32)
        for s in range(n):
            ci = int(cond_idx[s])
            for k in range(top_k):
                heatmap[ci, top_idx[s, k]] += 1
            counts[ci] += 1
        nonzero = counts > 0
        heatmap[nonzero] /= counts[nonzero, None]

        im = ax.imshow(heatmap, aspect='auto', cmap='YlOrRd',
                        interpolation='nearest')
        ax.set_xticks(range(E))
        ax.set_xticklabels([f'E{i}' for i in range(E)], fontsize=6)
        ax.set_yticks(range(n_conds))
        ax.set_yticklabels(cond_labels, fontsize=5.5)
        ax.set_xlabel('Expert Index')
        ax.set_ylabel(ylabel)
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

    Works with explicit fus_expert_info.  If unavailable, it uses a neutral
    0.5/0.5 placeholder rather than crashing, so the remaining figures still run.
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
        rgb_prefs.append(float(info.get('rgb_preference', 0.5)))
        pc_prefs.append(float(info.get('pc_preference', 0.5)))
        labels.append(f'E{i}')

    if not labels:
        print("[Fig 5] Fusion expert info is empty. Skipping.")
        return

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    ax.bar(x - w / 2, rgb_prefs, w, label='RGB Preference',
           color=PALETTE['blue'], edgecolor='white', linewidth=0.5)
    ax.bar(x + w / 2, pc_prefs, w, label='PC Preference',
           color=PALETTE['orange'], edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Mean Token Gate')
    ax.set_xlabel('Fusion Expert')
    ax.set_title('Fusion Expert Modality Preference', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, loc='upper right')
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i in range(len(labels)):
        if abs(rgb_prefs[i] - pc_prefs[i]) > 0.05:
            dominant = 'RGB' if rgb_prefs[i] > pc_prefs[i] else 'PC'
            ratio = max(rgb_prefs[i], pc_prefs[i]) / (min(rgb_prefs[i], pc_prefs[i]) + 1e-8)
            ax.annotate(f'{dominant}\\n×{ratio:.1f}',
                        (x[i], min(0.98, max(rgb_prefs[i], pc_prefs[i]) + 0.02)),
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
    """Box plot: QACT quality scores stratified by brightness / distance bins."""
    import matplotlib.pyplot as plt

    _setup_style()

    rgb_q = _safe_flatten_quality(viz_data.get('viz_rgb_quality'))
    pc_q = _safe_flatten_quality(viz_data.get('viz_pc_quality'))
    cond = _as_numpy(viz_data.get('viz_cond_stats'))
    if rgb_q is None or pc_q is None or cond is None:
        print("[Fig 6] No quality / condition data. Skipping.")
        return

    cond = np.asarray(cond)
    n = min(len(rgb_q), len(pc_q), len(cond))
    if n < 5 or cond.shape[1] < 4:
        print("[Fig 6] Not enough condition data. Skipping.")
        return
    rgb_q, pc_q, cond = rgb_q[:n], pc_q[:n], cond[:n]

    brightness = cond[:, 0]
    distance = cond[:, 3]
    n_bins = min(5, max(2, n // 2))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

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
    ax.set_xlabel('Brightness Quantile (dark → bright)')
    ax.set_ylabel('RGB Quality Score')
    ax.set_title('(a) RGB Quality vs. Brightness', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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
    ax.set_xlabel('Distance Quantile (near → far)')
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
    """Heatmap: K context tokens × P body parts for RGB/PC self-probe attention.

    Accepts both legacy keys:
      viz_qact_rgb_attn / viz_qact_pc_attn
    and Calib keys:
      viz_qact_rgb_self_attn / viz_qact_pc_self_attn.
    """
    import matplotlib.pyplot as plt

    _setup_style()

    rgb_attn = _as_numpy(viz_data.get('viz_qact_rgb_attn'))
    pc_attn = _as_numpy(viz_data.get('viz_qact_pc_attn'))
    if rgb_attn is None:
        print("[Fig 7] No QACT self-attention data. Skipping.")
        return

    rgb_attn = np.asarray(rgb_attn)
    pc_attn = np.asarray(pc_attn) if pc_attn is not None else None

    def _mean_token_part(a):
        # Supported shapes:
        #   [N,K,P], [N,H,K,P], [N,T,K,P], [N,T,H,K,P]
        if a is None:
            return None
        a = np.asarray(a)
        if a.ndim == 5:
            return a.mean(axis=(0, 1, 2))
        if a.ndim == 4:
            # Heuristic: second dim can be heads or time; both are averaged.
            return a.mean(axis=(0, 1))
        if a.ndim == 3:
            return a.mean(axis=0)
        if a.ndim == 2:
            return a
        return a.reshape(-1, a.shape[-1])

    rgb_mean = _mean_token_part(rgb_attn)
    pc_mean = _mean_token_part(pc_attn) if pc_attn is not None else None

    panels = [('(a) RGB Self-Probe', rgb_mean, 'Blues')]
    if pc_mean is not None:
        panels.append(('(b) PC Self-Probe', pc_mean, 'Oranges'))

    K = rgb_mean.shape[0]
    P = rgb_mean.shape[1]
    labels = BODY_PARTS_16[:P] if P <= len(BODY_PARTS_16) else [f'Part {i}' for i in range(P)]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), max(2.2, 0.30 * K + 1.0)))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, attn_map, cmap) in zip(axes, panels):
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
#  Fig 9. QACT Quality Decomposition
#  Fig 10. Quality-Gated Aux Ensemble
#  Fig 11. QACT Condition Embedding Quality t-SNE
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

    tsne = _make_tsne(n, random_state=42)
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
#  Fig 9. QACT Quality Decomposition
# ==================================================================

def plot_qact_quality_decomposition(viz_data, out_dir):
    """Compare learned quality, entropy quality, and pseudo quality targets."""
    import matplotlib.pyplot as plt

    _setup_style()

    learned = _as_numpy(viz_data.get('viz_qact_learned_quality'))
    entropy = _as_numpy(viz_data.get('viz_qact_entropy_quality'))
    target = _as_numpy(viz_data.get('viz_qact_quality_targets'))
    final_q = _as_numpy(viz_data.get('viz_qact_quality'))

    if learned is None and entropy is None and target is None:
        print("[Fig 9] No QACT quality decomposition data. Skipping.")
        return

    arrays = []
    names = []
    for name, arr in [('Final', final_q), ('Learned', learned), ('Entropy', entropy), ('Pseudo Target', target)]:
        if arr is not None:
            arr = np.asarray(arr).reshape(np.asarray(arr).shape[0], -1)
            if arr.shape[1] >= 2:
                arrays.append(arr[:, :2])
                names.append(name)

    if not arrays:
        print("[Fig 9] QACT quality arrays have unsupported shape. Skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    x = np.arange(len(names))
    width = 0.36

    rgb_means = [a[:, 0].mean() for a in arrays]
    pc_means = [a[:, 1].mean() for a in arrays]
    rgb_stds = [a[:, 0].std() for a in arrays]
    pc_stds = [a[:, 1].std() for a in arrays]

    ax = axes[0]
    ax.bar(x - width/2, rgb_means, width, yerr=rgb_stds, label='RGB',
           color=PALETTE['blue'], alpha=0.85, capsize=2)
    ax.bar(x + width/2, pc_means, width, yerr=pc_stds, label='PC',
           color=PALETTE['orange'], alpha=0.85, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('Quality Score')
    ax.set_title('(a) Quality Source Comparison', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1]
    if learned is not None and target is not None:
        learned = np.asarray(learned).reshape(np.asarray(learned).shape[0], -1)
        target = np.asarray(target).reshape(np.asarray(target).shape[0], -1)
        n = min(len(learned), len(target))
        ax.scatter(target[:n, 0], learned[:n, 0], s=8, alpha=0.35,
                   color=PALETTE['blue'], label='RGB', rasterized=True)
        ax.scatter(target[:n, 1], learned[:n, 1], s=8, alpha=0.35,
                   color=PALETTE['orange'], label='PC', rasterized=True)
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Pseudo Quality Target')
        ax.set_ylabel('Learned Quality')
        ax.set_title('(b) Calibration Scatter', fontweight='bold')
        ax.legend(frameon=False)
    else:
        arr = arrays[0]
        ax.hist(arr[:, 0], bins=30, alpha=0.45, color=PALETTE['blue'], label='RGB')
        ax.hist(arr[:, 1], bins=30, alpha=0.45, color=PALETTE['orange'], label='PC')
        ax.set_xlabel('Quality Score')
        ax.set_ylabel('Count')
        ax.set_title('(b) Quality Distribution', fontweight='bold')
        ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('QACT-Calib Quality Decomposition', fontsize=11,
                 fontweight='bold', y=1.05)
    fig.tight_layout()
    savepath = os.path.join(out_dir, 'fig9_qact_quality_decomposition.pdf')
    fig.savefig(savepath); fig.savefig(savepath.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"[Fig 9] QACT quality decomposition → {savepath}")


# ==================================================================
#  Fig 10. Quality-Gated Aux Ensemble
# ==================================================================

def plot_quality_gated_aux_weights(viz_data, out_dir):
    """Visualize adaptive RGB/PC auxiliary branch weights."""
    import matplotlib.pyplot as plt

    _setup_style()

    wr = _safe_flatten_quality(viz_data.get('viz_rgb_aux_weight'))
    wp = _safe_flatten_quality(viz_data.get('viz_pc_aux_weight'))
    cond = _as_numpy(viz_data.get('viz_cond_stats'))
    agreement = _safe_flatten_quality(viz_data.get('viz_qact_cross_agreement'))

    if wr is None or wp is None:
        print("[Fig 10] No quality-gated aux weights. Skipping.")
        return

    n = min(len(wr), len(wp))
    wr, wp = wr[:n], wp[:n]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    ax = axes[0]
    ax.hist(wr, bins=30, alpha=0.45, color=PALETTE['blue'], label='RGB aux weight')
    ax.hist(wp, bins=30, alpha=0.45, color=PALETTE['orange'], label='PC aux weight')
    ax.set_xlabel('Aux Weight')
    ax.set_ylabel('Count')
    ax.set_title('(a) Weight Distribution', fontweight='bold')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1]
    if cond is not None and len(cond) >= n and np.asarray(cond).shape[1] >= 4:
        cond = np.asarray(cond)[:n]
        brightness = cond[:, 0]
        distance = cond[:, 3]
        ax.scatter(brightness, wr, s=8, alpha=0.35, color=PALETTE['blue'],
                   label='RGB weight vs brightness', rasterized=True)
        ax.scatter(distance, wp, s=8, alpha=0.35, color=PALETTE['orange'],
                   label='PC weight vs distance', rasterized=True)
        ax.set_xlabel('Condition Statistic')
        ax.set_ylabel('Aux Weight')
        ax.set_title('(b) Weight vs. Condition', fontweight='bold')
        ax.legend(frameon=False, fontsize=6)
    elif agreement is not None and len(agreement) >= n:
        agreement = agreement[:n]
        ax.scatter(agreement, wr, s=8, alpha=0.35, color=PALETTE['blue'],
                   label='RGB aux', rasterized=True)
        ax.scatter(agreement, wp, s=8, alpha=0.35, color=PALETTE['orange'],
                   label='PC aux', rasterized=True)
        ax.set_xlabel('RGB-PC Agreement')
        ax.set_ylabel('Aux Weight')
        ax.set_title('(b) Weight vs. Agreement', fontweight='bold')
        ax.legend(frameon=False)
    else:
        ax.plot(np.sort(wr), color=PALETTE['blue'], label='RGB aux')
        ax.plot(np.sort(wp), color=PALETTE['orange'], label='PC aux')
        ax.set_xlabel('Sorted Sample Index')
        ax.set_ylabel('Aux Weight')
        ax.set_title('(b) Sorted Weights', fontweight='bold')
        ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Quality-Gated Test-Time Aux Ensemble', fontsize=11,
                 fontweight='bold', y=1.05)
    fig.tight_layout()
    savepath = os.path.join(out_dir, 'fig10_quality_gated_aux_weights.pdf')
    fig.savefig(savepath); fig.savefig(savepath.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"[Fig 10] Quality-gated aux weights → {savepath}")


# ==================================================================
#  Fig 11. QACT Condition Embedding Quality Space
# ==================================================================

def plot_qact_condition_embedding_quality_tsne(viz_data, out_dir, max_samples=3000):
    """t-SNE of QACT condition embeddings, colored by calibrated RGB/PC quality."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    _setup_style()

    cond_emb = _as_numpy(viz_data.get('viz_qact_cond_emb'))
    q = _as_numpy(viz_data.get('viz_qact_quality'))
    if cond_emb is None or q is None:
        print("[Fig 11] No QACT condition embedding / quality. Skipping.")
        return

    cond_emb = np.asarray(cond_emb)
    q = np.asarray(q).reshape(np.asarray(q).shape[0], -1)
    n = min(len(cond_emb), len(q), max_samples)
    if n < 5:
        print("[Fig 11] Not enough samples for t-SNE. Skipping.")
        return
    idx = np.random.choice(len(cond_emb), n, replace=False) if len(cond_emb) > n else np.arange(n)
    emb = cond_emb[idx].reshape(n, -1)
    qq = q[idx]

    tsne = _make_tsne(n, random_state=42)
    pts = tsne.fit_transform(emb)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, c, title, cmap in [
        (axes[0], qq[:, 0], '(a) RGB Quality', 'Blues'),
        (axes[1], qq[:, 1] if qq.shape[1] > 1 else qq[:, 0], '(b) PC Quality', 'Oranges'),
    ]:
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=c, cmap=cmap, s=8,
                        alpha=0.75, edgecolors='none', rasterized=True)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='Quality')
    fig.suptitle('QACT Condition Embedding Quality Space', fontsize=11,
                 fontweight='bold', y=1.04)
    fig.tight_layout()
    savepath = os.path.join(out_dir, 'fig11_qact_condition_quality_tsne.pdf')
    fig.savefig(savepath); fig.savefig(savepath.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"[Fig 11] QACT condition quality t-SNE → {savepath}")


# ==================================================================
#  Runner
# ==================================================================

def run_all_visualizations(viz_data, info_dict=None, out_dir=FIG_DIR_DEFAULT,
                           parts_num=16, base_data=None, base_info=None):
    """Run legacy V2Max figures and new QACT-Calib figures.

    Each figure is isolated; missing fields skip only that figure.
    """
    viz_data = _normalize_viz_data(viz_data)
    base_data = _normalize_viz_data(base_data) if base_data is not None else None
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  GaitMoEOracleV2Max-Calib Visualization Suite")
    print(f"  Output directory: {out_dir}")
    print(f"{'='*60}")

    steps = [
        ('Fig 1 t-SNE', lambda: plot_tsne(viz_data, out_dir, info_dict, base_data, base_info)),
        ('Fig 2 Distance', lambda: plot_distance_distribution(viz_data, out_dir, info_dict, base_info)),
        ('Fig 3 Expert Specialization', lambda: plot_expert_body_part_specialization(viz_data, out_dir, parts_num)),
        ('Fig 4 Routing Heatmap', lambda: plot_expert_routing_heatmap(viz_data, out_dir)),
        ('Fig 5 Fusion Modality', lambda: plot_fusion_modality_preference(viz_data, out_dir)),
        ('Fig 6 QACT Quality Condition', lambda: plot_qact_quality_condition(viz_data, out_dir)),
        ('Fig 7 QACT Attention', lambda: plot_qact_token_attention(viz_data, out_dir, parts_num)),
        ('Fig 8 Routing t-SNE', lambda: plot_expert_routing_tsne(viz_data, out_dir)),
        ('Fig 9 QACT Quality Decomposition', lambda: plot_qact_quality_decomposition(viz_data, out_dir)),
        ('Fig 10 Quality-Gated Aux', lambda: plot_quality_gated_aux_weights(viz_data, out_dir)),
        ('Fig 11 QACT Condition Quality t-SNE', lambda: plot_qact_condition_embedding_quality_tsne(viz_data, out_dir)),
    ]
    for name, fn in steps:
        try:
            fn()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[{name}] Failed but continuing: {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"  All available figures saved to: {out_dir}")
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
