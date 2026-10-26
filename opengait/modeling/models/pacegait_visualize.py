
"""PACE-Gait visualization suite.

Expected inputs:
- viz_data: dict saved by PACEGait._save_and_plot_viz
- info_dict: OpenGait inference dict containing embeddings/labels and optional metadata

The functions are intentionally robust: if a signal is missing, the remaining
plots are still generated.
"""

import os
from pathlib import Path
import numpy as np
import torch


def _to_np(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def run_all_visualizations(viz_data, info_dict, viz_dir, parts_num=16):
    import matplotlib
    matplotlib.use('Agg')
    _ensure_dir(viz_dir)
    print('[PACE-Gait Viz] writing plots to', viz_dir)

    usage = {}
    for tag in ['app', 'str', 'fus']:
        logits = viz_data.get(f'viz_{tag}_logits')
        if logits is not None:
            usage[tag] = compute_expert_usage_from_logits(logits, top_k=2)
    if usage:
        plot_expert_usage(usage, os.path.join(viz_dir, 'expert_usage.png'))
        plot_router_entropy(usage, os.path.join(viz_dir, 'router_balance_entropy.png'))

    for tag in ['app', 'str']:
        info = viz_data.get(f'{tag}_expert_info')
        if info:
            plot_typed_part_attention(info, os.path.join(viz_dir, f'{tag}_expert_part_attention.png'), parts_num)

    if viz_data.get('fusion_token_preference'):
        plot_fusion_token_preference(viz_data['fusion_token_preference'], os.path.join(viz_dir, 'fusion_token_preference.png'))

    emb = _to_np(info_dict.get('embeddings')) if isinstance(info_dict, dict) else None
    labels = _to_np(info_dict.get('labels')) if isinstance(info_dict, dict) else None
    if emb is not None and labels is not None:
        emb = emb.reshape(emb.shape[0], -1)
        plot_tsne(emb, labels, os.path.join(viz_dir, 'tsne_embeddings.png'))
        plot_intra_inter_distance(emb, labels, os.path.join(viz_dir, 'intra_inter_distance.png'))

    rq, pq = _to_np(viz_data.get('viz_rgb_quality')), _to_np(viz_data.get('viz_pc_quality'))
    stats = _to_np(viz_data.get('viz_condition_stats'))
    if rq is not None and pq is not None:
        plot_quality_histograms(rq, pq, os.path.join(viz_dir, 'quality_histograms.png'))
        if stats is not None:
            plot_quality_vs_condition(rq, pq, stats, os.path.join(viz_dir, 'quality_vs_condition.png'))

    r_attn = _to_np(viz_data.get('viz_qact_rgb_self_attn_w'))
    p_attn = _to_np(viz_data.get('viz_qact_pc_self_attn_w'))
    if r_attn is not None:
        plot_qact_attention(r_attn, os.path.join(viz_dir, 'qact_rgb_token_attention.png'), 'RGB QACT token-to-part attention')
    if p_attn is not None:
        plot_qact_attention(p_attn, os.path.join(viz_dir, 'qact_pc_token_attention.png'), 'Point-cloud QACT token-to-part attention')

    rctx = _to_np(viz_data.get('viz_qact_rgb_ctx'))
    pctx = _to_np(viz_data.get('viz_qact_pc_ctx'))
    if rctx is not None and pctx is not None:
        plot_context_token_diversity(rctx, pctx, os.path.join(viz_dir, 'qact_context_token_similarity.png'))
        if labels is not None:
            plot_context_tsne(rctx, pctx, labels, os.path.join(viz_dir, 'qact_context_tsne.png'))

    print('[PACE-Gait Viz] done')


def compute_expert_usage_from_logits(logits, top_k=2):
    logits = torch.as_tensor(logits).float()
    E = logits.size(1)
    top_k = min(top_k, E)
    _, topk_idx = logits.topk(top_k, dim=-1)
    usage = torch.zeros(E)
    for k in range(top_k):
        for i in range(E):
            usage[i] += (topk_idx[:, k] == i).float().sum()
    return (usage / max(1, logits.size(0) * top_k)).numpy()


def plot_expert_usage(usage_dict, save_path):
    import matplotlib.pyplot as plt
    n = len(usage_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    axes = np.atleast_1d(axes)
    for ax, (tag, usage) in zip(axes, usage_dict.items()):
        ax.bar(np.arange(len(usage)), usage, edgecolor='black', linewidth=0.5)
        ax.axhline(1.0 / len(usage), linestyle='--', linewidth=1, label='Uniform')
        ax.set_title(f'{tag.upper()} expert usage')
        ax.set_xlabel('Expert ID')
        ax.set_ylabel('Selection frequency')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_router_entropy(usage_dict, save_path):
    import matplotlib.pyplot as plt
    tags, vals = [], []
    for tag, usage in usage_dict.items():
        u = np.asarray(usage) + 1e-12
        ent = -(u * np.log(u)).sum() / np.log(len(u))
        tags.append(tag.upper())
        vals.append(ent)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.bar(tags, vals, edgecolor='black')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Normalized routing entropy')
    ax.set_title('MoE load-balance diagnostic')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_typed_part_attention(expert_info, save_path, parts_num=16):
    import matplotlib.pyplot as plt
    rows, names = [], []
    for eid, info in sorted(expert_info.items()):
        if 'part_attention' not in info:
            continue
        rows.append(np.asarray(info['part_attention']).reshape(-1))
        names.append(f"{eid}\n{info.get('type','Expert').replace('Expert','')}")
    if not rows:
        return
    data = np.vstack(rows)
    fig, ax = plt.subplots(figsize=(max(8, parts_num * 0.45), max(3, len(rows) * 0.45)))
    im = ax.imshow(data, aspect='auto', interpolation='nearest')
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xticks(np.arange(parts_num)); ax.set_xlabel('Body part index')
    ax.set_title('Expert part-attention diversity')
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_fusion_token_preference(pref, save_path):
    import matplotlib.pyplot as plt
    names, rgb_m, pc_m = [], [], []
    for eid, item in sorted(pref.items()):
        names.append(eid.replace('expert_', 'E'))
        rgb_m.append(item['rgb_mean']); pc_m.append(item['pc_mean'])
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.65), 4))
    ax.bar(x - w / 2, rgb_m, w, label='RGB tokens', edgecolor='black')
    ax.bar(x + w / 2, pc_m, w, label='PC tokens', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel('Mean token gate')
    ax.set_title('Fusion expert modality preference')
    ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close(fig)


def plot_tsne(emb, labels, save_path, max_points=3000):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    labels = np.asarray(labels).reshape(-1)
    emb = np.asarray(emb)
    if emb.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(emb.shape[0], max_points, replace=False)
        emb, labels = emb[idx], labels[idx]
    z = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=min(30, max(5, emb.shape[0] // 10)), random_state=0).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(z[:, 0], z[:, 1], c=labels, s=7, alpha=0.75)
    ax.set_title('t-SNE of identity embeddings')
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label='ID')
    fig.tight_layout(); fig.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close(fig)


def _pairwise_cosine_distance(x):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    sim = x @ x.T
    return 1 - sim


def plot_intra_inter_distance(emb, labels, save_path, max_points=2000):
    import matplotlib.pyplot as plt
    emb = np.asarray(emb); labels = np.asarray(labels).reshape(-1)
    if emb.shape[0] > max_points:
        rng = np.random.default_rng(1)
        idx = rng.choice(emb.shape[0], max_points, replace=False)
        emb, labels = emb[idx], labels[idx]
    D = _pairwise_cosine_distance(emb)
    same = labels[:, None] == labels[None, :]
    tri = np.triu(np.ones_like(same, dtype=bool), 1)
    intra = D[same & tri]
    inter = D[(~same) & tri]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(intra, bins=50, density=True, alpha=0.65, label='Intra-class')
    ax.hist(inter, bins=50, density=True, alpha=0.65, label='Inter-class')
    ax.set_xlabel('Cosine distance')
    ax.set_ylabel('Density')
    ax.set_title('Intra-/inter-class distance distribution')
    ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close(fig)


def plot_quality_histograms(rq, pq, save_path):
    import matplotlib.pyplot as plt
    rq = np.asarray(rq).reshape(-1); pq = np.asarray(pq).reshape(-1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rq, bins=40, alpha=0.6, density=True, label='RGB quality')
    ax.hist(pq, bins=40, alpha=0.6, density=True, label='PC quality')
    ax.set_xlabel('Predicted quality score')
    ax.set_ylabel('Density')
    ax.set_title('Learnable quality distribution')
    ax.legend(); fig.tight_layout(); fig.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close(fig)


def plot_quality_vs_condition(rq, pq, stats, save_path):
    import matplotlib.pyplot as plt
    rq, pq = np.asarray(rq).reshape(-1), np.asarray(pq).reshape(-1)
    stats = np.asarray(stats)
    brightness = stats[:, 0] if stats.shape[1] > 0 else np.arange(len(rq))
    distance = stats[:, 3] if stats.shape[1] > 3 else np.arange(len(rq))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(brightness, rq - pq, c=distance, s=9, alpha=0.75)
    ax.axhline(0, linewidth=1, linestyle='--')
    ax.set_xlabel('Brightness proxy')
    ax.set_ylabel('RGB quality - PC quality')
    ax.set_title('Quality reacts to illumination/distance')
    fig.colorbar(sc, ax=ax, label='Distance proxy')
    fig.tight_layout(); fig.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close(fig)


def plot_qact_attention(attn, save_path, title):
    import matplotlib.pyplot as plt
    a = np.asarray(attn)
    # MultiheadAttention average weights are usually [B, K, P].
    if a.ndim == 4:
        a = a.mean(axis=1)
    if a.ndim == 3:
        a = a.mean(axis=0)
    fig, ax = plt.subplots(figsize=(max(6, a.shape[-1] * 0.45), max(3.5, a.shape[0] * 0.45)))
    im = ax.imshow(a, aspect='auto', interpolation='nearest')
    ax.set_xlabel('Body part index')
    ax.set_ylabel('QACT token index')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout(); fig.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close(fig)


def plot_context_token_diversity(rctx, pctx, save_path):
    import matplotlib.pyplot as plt
    def sim(ctx):
        x = np.asarray(ctx).mean(axis=0)
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return x @ x.T
    sr, sp = sim(rctx), sim(pctx)
    data = np.block([[sr, np.zeros((sr.shape[0], sp.shape[1]))], [np.zeros((sp.shape[0], sr.shape[1])), sp]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(data, vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title('Context token cosine similarity')
    ax.set_xlabel('Token index'); ax.set_ylabel('Token index')
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout(); fig.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close(fig)


def plot_context_tsne(rctx, pctx, labels, save_path, max_points=2000):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    r = np.asarray(rctx).mean(axis=1)
    p = np.asarray(pctx).mean(axis=1)
    x = np.vstack([r, p])
    dom = np.array([0] * len(r) + [1] * len(p))
    if x.shape[0] > max_points:
        rng = np.random.default_rng(2)
        idx = rng.choice(x.shape[0], max_points, replace=False)
        x, dom = x[idx], dom[idx]
    z = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=min(30, max(5, x.shape[0] // 10)), random_state=2).fit_transform(x)
    fig, ax = plt.subplots(figsize=(6, 5))
    for d, name, marker in [(0, 'RGB context', 'o'), (1, 'PC context', '^')]:
        m = dom == d
        ax.scatter(z[m, 0], z[m, 1], s=9, alpha=0.7, label=name, marker=marker)
    ax.set_title('QACT context space by modality')
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--viz-data', required=True, help='pacegait_viz_data.pth')
    p.add_argument('--info', default=None, help='optional info_dict .pth with embeddings and labels')
    p.add_argument('--out', required=True)
    p.add_argument('--parts-num', type=int, default=16)
    args = p.parse_args()
    vd = torch.load(args.viz_data, map_location='cpu')
    info = torch.load(args.info, map_location='cpu') if args.info else {}
    run_all_visualizations(vd, info, args.out, args.parts_num)
