"""GaitMoEOracle V2Max — Visualization Utilities.

Provides:
  - t-SNE embedding visualization
  - Intra-class / Inter-class distance analysis
  - Expert usage heatmaps
  - Expert part-attention by type
  - Fusion token preference plots
"""

import os
import numpy as np
import torch


def run_all_visualizations(viz_data, info_dict, viz_dir, parts_num=16):
    """Run all available visualizations and save to viz_dir."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Viz] matplotlib not available, skipping plots.")
        return

    # 1. Expert usage
    usage = {}
    for tag in ['app', 'str', 'fus']:
        k = f'viz_{tag}_logits'
        if k in viz_data and isinstance(viz_data[k], torch.Tensor):
            usage[tag] = compute_expert_usage_from_logits(viz_data[k], top_k=2)
    if usage:
        print_expert_usage_table(usage, title="Test-Set Expert Usage (V2Max)")
        plot_expert_usage(usage, os.path.join(viz_dir, 'expert_usage.png'))

    # 2. Expert part-attention by type
    if viz_data.get('app_expert_info'):
        plot_typed_part_attention(
            viz_data['app_expert_info'],
            os.path.join(viz_dir, 'part_attention_by_type.png'),
            parts_num)

    # 3. t-SNE
    if 'embeddings' in info_dict and 'labels' in info_dict:
        emb = info_dict['embeddings']
        labels = info_dict['labels']
        plot_tsne(emb, labels, os.path.join(viz_dir, 'tsne.png'))

    # 4. Intra/Inter class distance
    if 'embeddings' in info_dict and 'labels' in info_dict:
        emb = info_dict['embeddings']
        labels = info_dict['labels']
        plot_intra_inter_distance(
            emb, labels, os.path.join(viz_dir, 'intra_inter_distance.png'))

    print(f"[V2Max Visualization] All plots saved in {viz_dir}/")


# ================================================================
#  Expert Usage
# ================================================================

def compute_expert_usage_from_logits(logits, top_k=2):
    """Compute per-expert selection frequency."""
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    E = logits.size(1)
    _, topk_idx = logits.topk(top_k, dim=-1)
    usage = torch.zeros(E)
    for k in range(top_k):
        for i in range(E):
            usage[i] += (topk_idx[:, k] == i).float().sum()
    usage = usage / (logits.size(0) * top_k)
    return usage.numpy()


def print_expert_usage_table(usage_dict, title="Expert Usage"):
    lines = [f"\n{'='*60}", f"  {title}", f"{'='*60}"]
    for tag, usage in usage_dict.items():
        parts = [f"E{i}={u:.3f}" for i, u in enumerate(usage)]
        lines.append(f"  [{tag.upper():>3s}] {' | '.join(parts)}")
        entropy = -sum(
            u * np.log(max(u, 1e-8)) for u in usage)
        max_entropy = np.log(len(usage))
        lines.append(
            f"        Balance: {entropy/max_entropy:.3f} "
            f"(entropy={entropy:.3f}/{max_entropy:.3f})")
    lines.append(f"{'='*60}\n")
    print('\n'.join(lines), flush=True)


def plot_expert_usage(usage_dict, save_path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(usage_dict), figsize=(5 * len(usage_dict), 4))
    if len(usage_dict) == 1:
        axes = [axes]
    for ax, (tag, usage) in zip(axes, usage_dict.items()):
        colors = plt.cm.Set3(np.linspace(0, 1, len(usage)))
        ax.bar(range(len(usage)), usage, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(f'{tag.upper()} Expert Usage', fontsize=12)
        ax.set_xlabel('Expert ID')
        ax.set_ylabel('Selection Frequency')
        ax.set_ylim(0, max(usage) * 1.3 + 0.01)
        ax.axhline(y=1.0 / len(usage), color='red', linestyle='--',
                    alpha=0.5, label='Uniform')
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ================================================================
#  Typed Part Attention Heatmap
# ================================================================

def plot_typed_part_attention(expert_info, save_path, parts_num=16):
    """Plot part attention heatmap grouped by expert type."""
    import matplotlib.pyplot as plt

    types_order = ['PointwiseExpert', 'LocalConvExpert', 'GlobalAttentionExpert',
                   'PartExpertBlock', 'SharedExpert']
    type_short = {'PointwiseExpert': 'PW', 'LocalConvExpert': 'LC',
                  'GlobalAttentionExpert': 'GA', 'PartExpertBlock': 'Std',
                  'SharedExpert': 'Sh'}

    names, attns = [], []
    for eid, info in sorted(expert_info.items()):
        etype = info.get('type', 'Unknown')
        short = type_short.get(etype, etype[:3])
        names.append(f"{eid}\n({short})")
        attns.append(info['part_attention'])

    if not attns:
        return

    data = np.stack(attns)  # [num_experts, P]
    fig, ax = plt.subplots(figsize=(max(8, parts_num * 0.6), len(attns) * 0.5 + 1))
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xticks(range(parts_num))
    ax.set_xticklabels([f'P{i}' for i in range(parts_num)], fontsize=7)
    ax.set_xlabel('Body Part (top→bottom)', fontsize=10)
    ax.set_title('Expert Part Attention by Type', fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ================================================================
#  t-SNE Visualization
# ================================================================

def plot_tsne(embeddings, labels, save_path, max_samples=2000, perplexity=30):
    """Plot t-SNE of embeddings colored by identity label."""
    import matplotlib.pyplot as plt

    if isinstance(embeddings, np.ndarray):
        emb = embeddings
    else:
        emb = np.array(embeddings)

    if isinstance(labels, (list, tuple)):
        labels = np.array(labels)

    # Flatten: [N, C, P] -> [N, C*P]
    if emb.ndim == 3:
        N, C, P = emb.shape
        emb = emb.reshape(N, C * P)

    # Subsample if too large
    if len(emb) > max_samples:
        idx = np.random.choice(len(emb), max_samples, replace=False)
        emb = emb[idx]
        labels = labels[idx]

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[Viz] sklearn not available, skipping t-SNE.")
        return

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                n_iter=1000, init='pca', learning_rate='auto')
    coords = tsne.fit_transform(emb.astype(np.float32))

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    if n_classes <= 50:
        cmap = plt.cm.get_cmap('tab20', n_classes)
        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[cmap(i % 20)], s=8, alpha=0.7, label=str(lbl))
        if n_classes <= 20:
            ax.legend(fontsize=6, markerscale=2, ncol=2,
                      loc='upper right', framealpha=0.5)
    else:
        label_to_int = {l: i for i, l in enumerate(unique_labels)}
        colors = [label_to_int[l] for l in labels]
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=colors, cmap='nipy_spectral', s=5, alpha=0.6)
        plt.colorbar(sc, ax=ax, label='Identity', shrink=0.8)

    ax.set_title(f't-SNE Embedding ({n_classes} identities)', fontsize=13)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [t-SNE] Saved: {save_path}")


# ================================================================
#  Intra-class / Inter-class Distance
# ================================================================

def plot_intra_inter_distance(embeddings, labels, save_path,
                              max_classes=50, max_per_class=20):
    """Plot distribution of intra-class and inter-class distances."""
    import matplotlib.pyplot as plt

    if isinstance(embeddings, np.ndarray):
        emb = embeddings
    else:
        emb = np.array(embeddings)
    if isinstance(labels, (list, tuple)):
        labels = np.array(labels)

    if emb.ndim == 3:
        N, C, P = emb.shape
        emb = emb.reshape(N, C * P)

    unique_labels = np.unique(labels)
    if len(unique_labels) > max_classes:
        sel = np.random.choice(unique_labels, max_classes, replace=False)
        mask = np.isin(labels, sel)
        emb, labels = emb[mask], labels[mask]
        unique_labels = sel

    intra_dists = []
    inter_dists = []
    centroids = {}

    for lbl in unique_labels:
        mask = labels == lbl
        cls_emb = emb[mask]
        if len(cls_emb) > max_per_class:
            cls_emb = cls_emb[np.random.choice(len(cls_emb), max_per_class, replace=False)]
        centroids[lbl] = cls_emb.mean(axis=0)
        # Intra-class: pairwise distances within class
        for i in range(len(cls_emb)):
            for j in range(i + 1, min(len(cls_emb), i + 5)):
                d = np.linalg.norm(cls_emb[i] - cls_emb[j])
                intra_dists.append(d)

    # Inter-class: distances between centroids
    cent_list = list(centroids.values())
    for i in range(len(cent_list)):
        for j in range(i + 1, len(cent_list)):
            d = np.linalg.norm(cent_list[i] - cent_list[j])
            inter_dists.append(d)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(max(intra_dists, default=1),
                              max(inter_dists, default=1)) * 1.1, 60)
    ax.hist(intra_dists, bins=bins, alpha=0.6, color='#e74c3c',
            label=f'Intra-class (n={len(intra_dists)})', density=True)
    ax.hist(inter_dists, bins=bins, alpha=0.6, color='#2ecc71',
            label=f'Inter-class (n={len(inter_dists)})', density=True)
    ax.set_xlabel('Euclidean Distance', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Intra-class vs Inter-class Distance Distribution', fontsize=13)
    ax.legend(fontsize=10)

    # Add separation margin info
    intra_mean = np.mean(intra_dists) if intra_dists else 0
    inter_mean = np.mean(inter_dists) if inter_dists else 0
    margin = inter_mean - intra_mean
    ax.axvline(intra_mean, color='#e74c3c', linestyle='--', alpha=0.8)
    ax.axvline(inter_mean, color='#2ecc71', linestyle='--', alpha=0.8)
    ax.text(0.02, 0.95,
            f'Intra mean: {intra_mean:.2f}\n'
            f'Inter mean: {inter_mean:.2f}\n'
            f'Margin: {margin:.2f}',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Distance] Saved: {save_path}")
