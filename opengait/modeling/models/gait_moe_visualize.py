# """GaitMoE Visualization Utilities.

# Standalone scripts for analyzing trained models:
#   1. Expert usage statistics (bar charts, anti-collapse check)
#   2. Appearance expert part-attention heatmaps
#   3. Fusion expert token preference (RGB vs PC by condition)
#   4. Routing decision analysis across conditions

# Works with both GaitMoEOracleV2 and V2Plus.

# Usage:
#   # After evaluation with collect_visualization=True, run:
#   python gait_moe_visualize.py --viz_data path/to/viz_data.pth --output_dir ./viz_output

#   # Or import functions in your evaluation script:
#   from gait_moe_visualize import (
#       plot_expert_usage, plot_part_attention_heatmap,
#       plot_fusion_token_preference, plot_routing_by_condition,
#       aggregate_viz_data, print_expert_usage_table
#   )
# """

# import os
# import math
# import argparse
# from collections import defaultdict
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import torch
# import torch.nn.functional as F

# try:
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#     import matplotlib.gridspec as gridspec
#     from matplotlib.colors import LinearSegmentedColormap
#     HAS_MPL = True
# except ImportError:
#     HAS_MPL = False
#     print("[WARNING] matplotlib not found. Plotting functions disabled, text output only.")


# # ================================================================
# #  1. Expert Usage Analysis
# # ================================================================

# def compute_expert_usage_from_logits(logits: torch.Tensor, top_k: int = 2) -> dict:
#     """Compute per-expert usage statistics from routing logits.

#     Args:
#         logits: [N, E] routing logits
#         top_k: number of experts selected per sample
#     Returns:
#         dict with 'usage' (per-expert fraction), 'entropy', 'balance_ratio'
#     """
#     if logits is None or logits.dim() != 2:
#         return {}
#     E = logits.size(1)
#     _, topk_idx = logits.topk(top_k, dim=-1)
#     usage = torch.zeros(E)
#     for k in range(top_k):
#         for i in range(E):
#             usage[i] += (topk_idx[:, k] == i).float().sum()
#     usage = usage / (logits.size(0) * top_k)
#     entropy = -(usage.clamp(min=1e-8) * usage.clamp(min=1e-8).log()).sum().item()
#     max_ent = math.log(E) if E > 1 else 1.0
#     return {
#         'usage': usage.numpy(),
#         'entropy': entropy,
#         'max_entropy': max_ent,
#         'balance_ratio': entropy / max_ent,
#         'max_usage': usage.max().item(),
#         'min_usage': usage.min().item(),
#         'cv': (usage.std() / usage.mean()).item() if usage.mean() > 0 else 0,
#     }


# def print_expert_usage_table(usage_dict: Dict[str, dict], title: str = "Expert Usage Report"):
#     """Print a formatted table of expert usage statistics to console.

#     Suitable for training logs.
#     """
#     print(f"\n{'='*70}")
#     print(f"  {title}")
#     print(f"{'='*70}")

#     for moe_name, stats in usage_dict.items():
#         if not stats:
#             continue
#         usage = stats['usage']
#         E = len(usage)
#         print(f"\n  [{moe_name.upper()} MoE] — {E} experts")
#         print(f"  Balance ratio: {stats['balance_ratio']:.3f} "
#               f"(1.0 = perfect, entropy={stats['entropy']:.3f}/{stats['max_entropy']:.3f})")
#         print(f"  Max usage: {stats['max_usage']:.3f}, "
#               f"Min usage: {stats['min_usage']:.3f}, "
#               f"CV: {stats['cv']:.3f}")

#         # Bar chart in text
#         max_bar = 40
#         ideal = 1.0 / E
#         for i, u in enumerate(usage):
#             bar_len = int(u * max_bar / max(usage.max(), 1e-8))
#             bar = '█' * bar_len + '░' * (max_bar - bar_len)
#             flag = ' ⚠️' if u < ideal * 0.3 or u > ideal * 3.0 else ''
#             print(f"    Expert {i}: [{bar}] {u:.3f}{flag}")

#         # Collapse warning
#         if stats['balance_ratio'] < 0.7:
#             print(f"  ⚠️  WARNING: Balance ratio {stats['balance_ratio']:.3f} < 0.7 — "
#                   f"possible expert collapse!")
#         if stats['max_usage'] > 0.5:
#             dominant = np.argmax(usage)
#             print(f"  ⚠️  WARNING: Expert {dominant} handles {stats['max_usage']*100:.1f}% "
#                   f"of all samples — dominant expert detected!")

#     print(f"\n{'='*70}\n")


# def plot_expert_usage(usage_dict: Dict[str, dict], save_path: str = None,
#                       title: str = "Expert Usage Distribution"):
#     """Plot expert usage bar charts for all MoE layers."""
#     if not HAS_MPL:
#         print_expert_usage_table(usage_dict, title)
#         return

#     n_moes = len([k for k, v in usage_dict.items() if v])
#     if n_moes == 0:
#         return

#     fig, axes = plt.subplots(1, n_moes, figsize=(5 * n_moes, 4), squeeze=False)
#     colors = {'app': '#4C72B0', 'str': '#55A868', 'fus': '#C44E52'}
#     labels = {'app': 'Appearance', 'str': 'Structure', 'fus': 'Fusion'}

#     idx = 0
#     for moe_name, stats in usage_dict.items():
#         if not stats:
#             continue
#         ax = axes[0, idx]
#         usage = stats['usage']
#         E = len(usage)
#         x = np.arange(E)
#         ideal = 1.0 / E

#         bars = ax.bar(x, usage, color=colors.get(moe_name, '#888888'),
#                        alpha=0.8, edgecolor='white', linewidth=0.5)
#         ax.axhline(y=ideal, color='red', linestyle='--', alpha=0.5,
#                    label=f'Ideal ({ideal:.3f})')

#         # Highlight collapsed experts
#         for i, (bar, u) in enumerate(zip(bars, usage)):
#             if u < ideal * 0.3:
#                 bar.set_edgecolor('red')
#                 bar.set_linewidth(2)

#         ax.set_xlabel('Expert Index')
#         ax.set_ylabel('Usage Fraction')
#         ax.set_title(f'{labels.get(moe_name, moe_name)} MoE\n'
#                      f'Balance: {stats["balance_ratio"]:.3f}')
#         ax.set_xticks(x)
#         ax.legend(fontsize=8)
#         ax.set_ylim(0, max(usage.max() * 1.2, ideal * 2))
#         idx += 1

#     fig.suptitle(title, fontsize=14, fontweight='bold')
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"  Saved expert usage plot to {save_path}")
#     plt.close()


# # ================================================================
# #  2. Appearance Expert Part-Attention Heatmaps
# # ================================================================

# PART_NAMES_16 = [
#     'Head', 'Neck', 'Upper Chest', 'Lower Chest',
#     'Upper Abdomen', 'Lower Abdomen', 'Upper Hip', 'Lower Hip',
#     'Upper Left Thigh', 'Upper Right Thigh', 'Lower Left Thigh', 'Lower Right Thigh',
#     'Left Knee', 'Right Knee', 'Left Shin', 'Right Shin',
# ]


# def get_part_attention_from_model(model) -> Dict[str, np.ndarray]:
#     """Extract part-attention patterns from a loaded model's appearance MoE.

#     Returns:
#         Dict mapping 'expert_i' -> numpy array of shape [P] with sigmoid(part_logit)
#     """
#     app_moe = getattr(model, 'app_moe', None)
#     if app_moe is None:
#         return {}
#     result = {}
#     for i, expert in enumerate(app_moe.experts):
#         if hasattr(expert, 'part_logit'):
#             result[f'expert_{i}'] = torch.sigmoid(expert.part_logit).detach().cpu().numpy()
#     return result


# def plot_part_attention_heatmap(part_attention: Dict[str, np.ndarray],
#                                  save_path: str = None, parts_num: int = 16,
#                                  part_names: List[str] = None):
#     """Plot heatmap showing which body parts each appearance expert attends to.

#     Each row is an expert, each column is a body part.
#     Helps verify experts specialize on different body regions.
#     """
#     if not HAS_MPL or not part_attention:
#         # Text fallback
#         print("\n  Appearance Expert Part-Attention:")
#         for name, attn in sorted(part_attention.items()):
#             top3 = np.argsort(attn)[-3:][::-1]
#             top3_str = ', '.join(f'P{j}({attn[j]:.2f})' for j in top3)
#             print(f"    {name}: top-3 parts = {top3_str}")
#         return

#     if part_names is None:
#         part_names = PART_NAMES_16[:parts_num] if parts_num <= 16 \
#             else [f'Part {i}' for i in range(parts_num)]

#     E = len(part_attention)
#     matrix = np.stack([part_attention[f'expert_{i}']
#                        for i in range(E) if f'expert_{i}' in part_attention])

#     fig, ax = plt.subplots(figsize=(max(10, parts_num * 0.6), max(4, E * 0.5)))
#     cmap = LinearSegmentedColormap.from_list('custom', ['#f0f0f0', '#4C72B0', '#C44E52'])
#     im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1)

#     ax.set_yticks(range(E))
#     ax.set_yticklabels([f'Expert {i}' for i in range(E)])
#     ax.set_xticks(range(parts_num))
#     ax.set_xticklabels(part_names, rotation=45, ha='right', fontsize=8)

#     # Annotate values
#     for i in range(E):
#         for j in range(parts_num):
#             val = matrix[i, j]
#             color = 'white' if val > 0.6 else 'black'
#             ax.text(j, i, f'{val:.2f}', ha='center', va='center',
#                     fontsize=6, color=color)

#     plt.colorbar(im, ax=ax, label='Attention Weight (sigmoid)')
#     ax.set_title('Appearance Expert × Body Part Attention\n'
#                  '(Row = expert, Column = body part)',
#                  fontsize=12, fontweight='bold')
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"  Saved part attention heatmap to {save_path}")
#     plt.close()


# def plot_spatial_gate_overlay(rgb_frame: np.ndarray, spatial_gate: np.ndarray,
#                                save_path: str = None, title: str = "Spatial Gate"):
#     """Overlay the PositiveTextGate spatial attention on an RGB frame.

#     Args:
#         rgb_frame: [H, W, 3] RGB image (uint8 or float)
#         spatial_gate: [Hf, Wf] spatial gate output
#     """
#     if not HAS_MPL:
#         return

#     import cv2
#     if rgb_frame.max() <= 1.0:
#         rgb_frame = (rgb_frame * 255).astype(np.uint8)

#     H, W = rgb_frame.shape[:2]
#     gate_resized = cv2.resize(spatial_gate, (W, H), interpolation=cv2.INTER_LINEAR)

#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#     axes[0].imshow(rgb_frame)
#     axes[0].set_title('Input RGB')
#     axes[0].axis('off')

#     axes[1].imshow(gate_resized, cmap='hot', vmin=0, vmax=1)
#     axes[1].set_title('Spatial Gate')
#     axes[1].axis('off')

#     axes[2].imshow(rgb_frame)
#     axes[2].imshow(gate_resized, cmap='hot', alpha=0.5, vmin=0, vmax=1)
#     axes[2].set_title('Overlay')
#     axes[2].axis('off')

#     fig.suptitle(title, fontsize=12, fontweight='bold')
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close()


# # ================================================================
# #  3. Fusion Expert Token Preference (RGB vs PC)
# # ================================================================

# def get_fusion_token_preference(model) -> Dict[str, dict]:
#     """Extract per-expert RGB vs PC token preference from fusion MoE.

#     Uses the learned token_logit to determine which experts prefer
#     RGB tokens vs PC tokens.
#     """
#     fus_moe = getattr(model, 'fus_moe', None)
#     if fus_moe is None:
#         return {}
#     result = {}
#     for i, expert in enumerate(fus_moe.experts):
#         if hasattr(expert, 'token_logit'):
#             logit = expert.token_logit.detach().cpu()
#             P = logit.size(0) // 2
#             attn = torch.sigmoid(logit)
#             rgb_attn = attn[:P].numpy()
#             pc_attn = attn[P:].numpy()
#             result[f'expert_{i}'] = {
#                 'rgb_mean': float(rgb_attn.mean()),
#                 'pc_mean': float(pc_attn.mean()),
#                 'rgb_per_part': rgb_attn,
#                 'pc_per_part': pc_attn,
#                 'preference': 'RGB' if rgb_attn.mean() > pc_attn.mean() else 'PC',
#                 'ratio': float(rgb_attn.mean() / max(pc_attn.mean(), 1e-8)),
#             }
#     return result


# def plot_fusion_token_preference(pref_dict: Dict[str, dict], save_path: str = None,
#                                    title: str = "Fusion Expert Token Preference"):
#     """Plot bar chart comparing RGB vs PC preference per fusion expert."""
#     if not pref_dict:
#         return
#     if not HAS_MPL:
#         print(f"\n  {title}:")
#         for name, info in sorted(pref_dict.items()):
#             print(f"    {name}: RGB={info['rgb_mean']:.3f}, "
#                   f"PC={info['pc_mean']:.3f}, Pref={info['preference']}")
#         return

#     E = len(pref_dict)
#     x = np.arange(E)
#     width = 0.35
#     rgb_vals = [pref_dict[f'expert_{i}']['rgb_mean'] for i in range(E)]
#     pc_vals = [pref_dict[f'expert_{i}']['pc_mean'] for i in range(E)]

#     fig, ax = plt.subplots(figsize=(max(8, E * 1.2), 5))
#     bars1 = ax.bar(x - width/2, rgb_vals, width, label='RGB Tokens',
#                    color='#E8835C', alpha=0.85, edgecolor='white')
#     bars2 = ax.bar(x + width/2, pc_vals, width, label='PC Tokens',
#                    color='#4C72B0', alpha=0.85, edgecolor='white')

#     ax.set_xlabel('Expert Index')
#     ax.set_ylabel('Mean Token Attention (sigmoid)')
#     ax.set_title(title, fontsize=12, fontweight='bold')
#     ax.set_xticks(x)
#     ax.set_xticklabels([f'E{i}\n({pref_dict[f"expert_{i}"]["preference"]})'
#                         for i in range(E)], fontsize=9)
#     ax.legend()
#     ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"  Saved fusion token preference to {save_path}")
#     plt.close()


# def plot_fusion_preference_by_condition(routing_data: List[dict],
#                                           conditions: List[str],
#                                           save_path: str = None):
#     """Analyze and plot which fusion experts are selected under different conditions.

#     Args:
#         routing_data: list of dicts, each with 'fus_routing' [top_k], 'condition' str
#         conditions: list of condition names (e.g., ['day', 'night', 'close', 'far'])
#     """
#     if not routing_data:
#         return

#     # Group by condition
#     by_condition = defaultdict(list)
#     for item in routing_data:
#         cond = item.get('condition', 'unknown')
#         routing = item.get('fus_routing', None)
#         if routing is not None:
#             by_condition[cond].append(routing)

#     if not by_condition:
#         return

#     # Compute per-condition expert distribution
#     E = max(r.max().item() + 1 for rs in by_condition.values() for r in rs)
#     cond_usage = {}
#     for cond, routings in by_condition.items():
#         all_r = torch.cat(routings, dim=0) if len(routings) > 1 else routings[0]
#         usage = torch.zeros(E)
#         for i in range(E):
#             usage[i] = (all_r == i).float().mean()
#         cond_usage[cond] = usage.numpy()

#     if not HAS_MPL:
#         print("\n  Fusion Expert Usage by Condition:")
#         for cond, usage in cond_usage.items():
#             top2 = np.argsort(usage)[-2:][::-1]
#             print(f"    {cond}: top experts = E{top2[0]}({usage[top2[0]]:.2f}), "
#                   f"E{top2[1]}({usage[top2[1]]:.2f})")
#         return

#     n_conds = len(cond_usage)
#     fig, ax = plt.subplots(figsize=(max(10, E * 1.5), 5))
#     x = np.arange(E)
#     width = 0.8 / n_conds
#     cmap = plt.cm.Set2

#     for idx, (cond, usage) in enumerate(sorted(cond_usage.items())):
#         offset = (idx - n_conds / 2 + 0.5) * width
#         ax.bar(x + offset, usage, width, label=cond,
#                color=cmap(idx / max(n_conds - 1, 1)), alpha=0.85, edgecolor='white')

#     ax.set_xlabel('Expert Index')
#     ax.set_ylabel('Selection Frequency')
#     ax.set_title('Fusion Expert Selection by Condition', fontsize=12, fontweight='bold')
#     ax.set_xticks(x)
#     ax.legend()
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"  Saved condition-based routing to {save_path}")
#     plt.close()


# # ================================================================
# #  4. Aggregation & Data Collection Helpers
# # ================================================================

# class VizDataCollector:
#     """Collects visualization data during evaluation.

#     Viz data is stored in model._viz_buffer (not in inference_feat,
#     because OpenGait's ddp_all_gather requires all inference_feat
#     values to be Tensors).

#     Usage:
#         collector = VizDataCollector()
#         model.collect_viz = True
#         for batch in dataloader:
#             output = model(batch)
#             # Drain viz data from model buffer
#             collector.drain_model_buffer(model, typs=batch_typs, views=batch_views)
#         collector.save('viz_data.pth')
#         collector.generate_all_plots('./viz_output/', model=model)
#     """

#     def __init__(self):
#         self.data = defaultdict(list)
#         self.metadata = defaultdict(list)  # typs, views, etc.

#     def drain_model_buffer(self, model, typs: list = None, views: list = None,
#                            conditions: list = None):
#         """Drain accumulated viz data from model._viz_buffer.

#         Call this after each forward pass (or batch) during evaluation.
#         """
#         buf = getattr(model, '_viz_buffer', [])
#         for viz_dict in buf:
#             self.add(viz_dict, typs=typs, views=views, conditions=conditions)
#         # Clear the buffer
#         if hasattr(model, '_viz_buffer'):
#             model._viz_buffer.clear()

#     def add(self, viz_dict: dict, typs: list = None, views: list = None,
#             conditions: list = None):
#         """Add one batch of visualization data."""
#         for key, val in viz_dict.items():
#             if isinstance(val, torch.Tensor):
#                 self.data[key].append(val.cpu())
#             elif isinstance(val, dict):
#                 # Nested dict (like part_attention, token_preference)
#                 self.data[key].append(val)
#             else:
#                 self.data[key].append(val)

#         if typs is not None:
#             self.metadata['typs'].extend(typs if isinstance(typs, list) else list(typs))
#         if views is not None:
#             self.metadata['views'].extend(views if isinstance(views, list) else list(views))
#         if conditions is not None:
#             self.metadata['conditions'].extend(conditions)

#     def save(self, path: str):
#         """Save collected data to disk."""
#         torch.save({'data': dict(self.data), 'metadata': dict(self.metadata)}, path)
#         print(f"  Saved visualization data to {path}")

#     @classmethod
#     def load(cls, path: str) -> 'VizDataCollector':
#         """Load from disk."""
#         obj = cls()
#         saved = torch.load(path, map_location='cpu')
#         obj.data = defaultdict(list, saved.get('data', {}))
#         obj.metadata = defaultdict(list, saved.get('metadata', {}))
#         return obj

#     def get_aggregated_usage(self) -> Dict[str, dict]:
#         """Aggregate expert usage from all batches."""
#         result = {}
#         for tag in ['app', 'str', 'fus']:
#             key = f'{tag}_logits'
#             if key in self.data and self.data[key]:
#                 tensors = [t for t in self.data[key] if isinstance(t, torch.Tensor)]
#                 if tensors:
#                     all_logits = torch.cat(tensors, dim=0)
#                     result[tag] = compute_expert_usage_from_logits(all_logits, top_k=2)
#         return result

#     def generate_all_plots(self, output_dir: str, model=None):
#         """Generate all visualization plots."""
#         os.makedirs(output_dir, exist_ok=True)
#         print(f"\n  Generating visualization plots in {output_dir}/")

#         # 1) Expert usage
#         usage = self.get_aggregated_usage()
#         if usage:
#             print_expert_usage_table(usage)
#             plot_expert_usage(usage, os.path.join(output_dir, 'expert_usage.png'))

#         # 2) Part attention (from model parameters)
#         if model is not None:
#             part_attn = get_part_attention_from_model(model)
#             if part_attn:
#                 plot_part_attention_heatmap(
#                     part_attn, os.path.join(output_dir, 'part_attention.png'))

#             # 3) Fusion token preference (from model parameters)
#             fus_pref = get_fusion_token_preference(model)
#             if fus_pref:
#                 plot_fusion_token_preference(
#                     fus_pref, os.path.join(output_dir, 'fusion_token_preference.png'))

#         # 4) Condition-based routing
#         if 'fus_routing' in self.data and 'conditions' in self.metadata:
#             routing_list = []
#             routings = self.data['fus_routing']
#             conditions = self.metadata['conditions']
#             for i, r in enumerate(routings):
#                 if isinstance(r, torch.Tensor) and i < len(conditions):
#                     for j in range(r.size(0)):
#                         routing_list.append({
#                             'fus_routing': r[j:j+1],
#                             'condition': conditions[min(i, len(conditions)-1)],
#                         })
#             if routing_list:
#                 all_conds = sorted(set(item['condition'] for item in routing_list))
#                 plot_fusion_preference_by_condition(
#                     routing_list, all_conds,
#                     os.path.join(output_dir, 'routing_by_condition.png'))

#         # 5) Quality score distribution
#         self._plot_quality_distribution(output_dir)

#         print(f"  All plots generated in {output_dir}/\n")

#     def _plot_quality_distribution(self, output_dir: str):
#         """Plot distribution of RGB vs PC quality scores."""
#         if 'rgb_quality' not in self.data or not HAS_MPL:
#             return
#         rq = torch.cat([t for t in self.data['rgb_quality']
#                         if isinstance(t, torch.Tensor)]).numpy().flatten()
#         pq = torch.cat([t for t in self.data['pc_quality']
#                         if isinstance(t, torch.Tensor)]).numpy().flatten()

#         fig, axes = plt.subplots(1, 2, figsize=(10, 4))
#         axes[0].hist(rq, bins=50, color='#E8835C', alpha=0.7, edgecolor='white')
#         axes[0].set_title('RGB Quality Score Distribution')
#         axes[0].set_xlabel('Quality Score')
#         axes[0].axvline(x=rq.mean(), color='red', linestyle='--', label=f'Mean={rq.mean():.3f}')
#         axes[0].legend()

#         axes[1].hist(pq, bins=50, color='#4C72B0', alpha=0.7, edgecolor='white')
#         axes[1].set_title('PC Quality Score Distribution')
#         axes[1].set_xlabel('Quality Score')
#         axes[1].axvline(x=pq.mean(), color='red', linestyle='--', label=f'Mean={pq.mean():.3f}')
#         axes[1].legend()

#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, 'quality_distribution.png'),
#                     dpi=150, bbox_inches='tight')
#         plt.close()


# # ================================================================
# #  5. Training Log Integration
# # ================================================================

# def format_expert_usage_for_log(visual_summary: dict) -> str:
#     """Format expert usage from visual_summary for training log output.

#     Called during training to print usage stats every N iterations.
#     """
#     lines = []
#     for tag in ['app', 'str', 'fus']:
#         usage_keys = sorted([k for k in visual_summary
#                              if k.startswith(f'scalar/{tag}_usage_e')])
#         if not usage_keys:
#             continue
#         vals = []
#         for k in usage_keys:
#             v = visual_summary[k]
#             v = v.item() if torch.is_tensor(v) else float(v)
#             vals.append(v)
#         balance_key = f'scalar/{tag}_balance_ratio'
#         br = visual_summary.get(balance_key, None)
#         br = br.item() if torch.is_tensor(br) else (float(br) if br is not None else -1)
#         usage_str = ' '.join(f'{v:.2f}' for v in vals)
#         lines.append(f"  {tag.upper()}: [{usage_str}] bal={br:.3f}")
#     return '\n'.join(lines) if lines else ''


# # ================================================================
# #  6. Model-Level Visualization Hook for V2 (Backward Compatible)
# # ================================================================

# def add_viz_hooks_to_v2(model):
#     """Add visualization data collection to an existing GaitMoEOracleV2 model.

#     Monkey-patches the model to collect viz data during inference.
#     Viz data is stored in model._viz_buffer (NOT in inference_feat,
#     because OpenGait's ddp_all_gather requires Tensors).

#     Usage:
#         model = load_model(...)
#         add_viz_hooks_to_v2(model)
#         model.collect_viz = True

#         collector = VizDataCollector()
#         for batch in eval_loader:
#             output = model(batch)
#             collector.drain_model_buffer(model, typs=..., views=...)
#         collector.generate_all_plots('./viz_output/', model=model)
#     """
#     original_forward = model.forward.__func__ if hasattr(model.forward, '__func__') else model.forward

#     def patched_forward(self, inputs):
#         result = original_forward(self, inputs)

#         if getattr(self, 'collect_viz', False) and not self.training:
#             viz = {}

#             # Collect expert part attention
#             if hasattr(self, 'app_moe') and self.app_moe is not None:
#                 viz['app_part_attention'] = get_part_attention_from_model(self)

#             # Collect fusion token preference
#             if hasattr(self, 'fus_moe') and self.fus_moe is not None:
#                 viz['fus_token_preference'] = get_fusion_token_preference(self)

#             # Expert routing from visual_summary
#             vs = result.get('visual_summary', {})
#             for tag in ['app', 'str', 'fus']:
#                 wkeys = sorted([k for k in vs if k.startswith(f'scalar/{tag}_e') and k.endswith('_w')])
#                 if wkeys:
#                     weights = torch.stack([vs[k] for k in wkeys])
#                     viz[f'{tag}_weights'] = weights.detach().cpu()

#             # Store in model buffer — collector reads from here
#             if not hasattr(self, '_viz_buffer'):
#                 self._viz_buffer = []
#             self._viz_buffer.append(viz)

#         return result

#     import types
#     model.forward = types.MethodType(patched_forward, model)
#     model.collect_viz = False
#     model._viz_buffer = []
#     return model


# # ================================================================
# #  CLI
# # ================================================================

# def main():
#     parser = argparse.ArgumentParser(description='GaitMoE Visualization')
#     parser.add_argument('--viz_data', type=str, required=True,
#                         help='Path to saved viz data (.pth)')
#     parser.add_argument('--output_dir', type=str, default='./viz_output',
#                         help='Output directory for plots')
#     parser.add_argument('--model_path', type=str, default=None,
#                         help='Optional: path to model checkpoint for parameter-based plots')
#     args = parser.parse_args()

#     collector = VizDataCollector.load(args.viz_data)

#     model = None
#     if args.model_path and os.path.exists(args.model_path):
#         try:
#             ckpt = torch.load(args.model_path, map_location='cpu')
#             # Model loading depends on your framework; adjust as needed
#             print(f"  Loaded checkpoint from {args.model_path}")
#         except Exception as e:
#             print(f"  Could not load model: {e}")

#     collector.generate_all_plots(args.output_dir, model=model)


# if __name__ == '__main__':
#     main()





"""GaitMoE Visualization Utilities.

Standalone scripts for analyzing trained models:
  1. Expert usage statistics (bar charts, anti-collapse check)
  2. Appearance expert part-attention heatmaps
  3. Fusion expert token preference (RGB vs PC by condition)
  4. Routing decision analysis across conditions

Works with both GaitMoEOracleV2 and V2Plus.

Usage:
  # After evaluation with collect_visualization=True, run:
  python gait_moe_visualize.py --viz_data path/to/viz_data.pth --output_dir ./viz_output

  # Or import functions in your evaluation script:
  from gait_moe_visualize import (
      plot_expert_usage, plot_part_attention_heatmap,
      plot_fusion_token_preference, plot_routing_by_condition,
      aggregate_viz_data, print_expert_usage_table
  )
"""

import os
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not found. Plotting functions disabled, text output only.")


def safe_torch_load(path, map_location='cpu'):
    """Load visualization data robustly across PyTorch versions.

    PyTorch 2.6 changed torch.load default behavior toward weights_only=True.
    Our viz_data may contain Python lists/dicts and older files may contain numpy
    arrays, so we explicitly fall back to weights_only=False for trusted local
    visualization files.
    """
    try:
        return torch.load(path, map_location=map_location)
    except Exception:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)


def validate_viz_data(viz_data, path='viz_data.pth'):
    """Fail loudly if a checkpoint is passed as --viz_data by mistake."""
    if not isinstance(viz_data, dict):
        raise ValueError(f"{path} is not a dict-like visualization file.")

    expected_raw = {
        'viz_app_logits', 'viz_str_logits', 'viz_fus_logits',
        'app_part_attention', 'fus_token_preference',
        'viz_rgb_quality', 'viz_pc_quality',
    }
    expected_collector = {'data', 'metadata'}

    keys = set(viz_data.keys())
    if keys & expected_raw or expected_collector <= keys:
        return

    preview = sorted(list(keys))[:20]
    raise ValueError(
        f"{path} does not look like a visualization data file. "
        f"Got keys={preview}. Did you pass a model checkpoint .pt by mistake?"
    )


# ================================================================
#  1. Expert Usage Analysis
# ================================================================

def compute_expert_usage_from_logits(logits: torch.Tensor, top_k: int = 2) -> dict:
    """Compute per-expert usage statistics from routing logits.

    Args:
        logits: [N, E] routing logits
        top_k: number of experts selected per sample
    Returns:
        dict with 'usage' (per-expert fraction), 'entropy', 'balance_ratio'
    """
    if logits is None or logits.dim() != 2:
        return {}
    E = logits.size(1)
    _, topk_idx = logits.topk(top_k, dim=-1)
    usage = torch.zeros(E)
    for k in range(top_k):
        for i in range(E):
            usage[i] += (topk_idx[:, k] == i).float().sum()
    usage = usage / (logits.size(0) * top_k)
    entropy = -(usage.clamp(min=1e-8) * usage.clamp(min=1e-8).log()).sum().item()
    max_ent = math.log(E) if E > 1 else 1.0
    return {
        'usage': usage.numpy(),
        'entropy': entropy,
        'max_entropy': max_ent,
        'balance_ratio': entropy / max_ent,
        'max_usage': usage.max().item(),
        'min_usage': usage.min().item(),
        'cv': (usage.std() / usage.mean()).item() if usage.mean() > 0 else 0,
    }


def print_expert_usage_table(usage_dict: Dict[str, dict], title: str = "Expert Usage Report"):
    """Print a formatted table of expert usage statistics to console.

    Suitable for training logs.
    """
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    for moe_name, stats in usage_dict.items():
        if not stats:
            continue
        usage = stats['usage']
        E = len(usage)
        print(f"\n  [{moe_name.upper()} MoE] — {E} experts")
        print(f"  Balance ratio: {stats['balance_ratio']:.3f} "
              f"(1.0 = perfect, entropy={stats['entropy']:.3f}/{stats['max_entropy']:.3f})")
        print(f"  Max usage: {stats['max_usage']:.3f}, "
              f"Min usage: {stats['min_usage']:.3f}, "
              f"CV: {stats['cv']:.3f}")

        # Bar chart in text
        max_bar = 40
        ideal = 1.0 / E
        for i, u in enumerate(usage):
            bar_len = int(u * max_bar / max(usage.max(), 1e-8))
            bar = '█' * bar_len + '░' * (max_bar - bar_len)
            flag = ' ⚠️' if u < ideal * 0.3 or u > ideal * 3.0 else ''
            print(f"    Expert {i}: [{bar}] {u:.3f}{flag}")

        # Collapse warning
        if stats['balance_ratio'] < 0.7:
            print(f"  ⚠️  WARNING: Balance ratio {stats['balance_ratio']:.3f} < 0.7 — "
                  f"possible expert collapse!")
        if stats['max_usage'] > 0.5:
            dominant = np.argmax(usage)
            print(f"  ⚠️  WARNING: Expert {dominant} handles {stats['max_usage']*100:.1f}% "
                  f"of all samples — dominant expert detected!")

    print(f"\n{'='*70}\n")


def plot_expert_usage(usage_dict: Dict[str, dict], save_path: str = None,
                      title: str = "Expert Usage Distribution"):
    """Plot expert usage bar charts for all MoE layers."""
    if not HAS_MPL:
        print_expert_usage_table(usage_dict, title)
        return

    n_moes = len([k for k, v in usage_dict.items() if v])
    if n_moes == 0:
        return

    fig, axes = plt.subplots(1, n_moes, figsize=(5 * n_moes, 4), squeeze=False)
    colors = {'app': '#4C72B0', 'str': '#55A868', 'fus': '#C44E52'}
    labels = {'app': 'Appearance', 'str': 'Structure', 'fus': 'Fusion'}

    idx = 0
    for moe_name, stats in usage_dict.items():
        if not stats:
            continue
        ax = axes[0, idx]
        usage = stats['usage']
        E = len(usage)
        x = np.arange(E)
        ideal = 1.0 / E

        bars = ax.bar(x, usage, color=colors.get(moe_name, '#888888'),
                       alpha=0.8, edgecolor='white', linewidth=0.5)
        ax.axhline(y=ideal, color='red', linestyle='--', alpha=0.5,
                   label=f'Ideal ({ideal:.3f})')

        # Highlight collapsed experts
        for i, (bar, u) in enumerate(zip(bars, usage)):
            if u < ideal * 0.3:
                bar.set_edgecolor('red')
                bar.set_linewidth(2)

        ax.set_xlabel('Expert Index')
        ax.set_ylabel('Usage Fraction')
        ax.set_title(f'{labels.get(moe_name, moe_name)} MoE\n'
                     f'Balance: {stats["balance_ratio"]:.3f}')
        ax.set_xticks(x)
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(usage.max() * 1.2, ideal * 2))
        idx += 1

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved expert usage plot to {save_path}")
    plt.close()


# ================================================================
#  2. Appearance Expert Part-Attention Heatmaps
# ================================================================

PART_NAMES_16 = [
    'Head', 'Neck', 'Upper Chest', 'Lower Chest',
    'Upper Abdomen', 'Lower Abdomen', 'Upper Hip', 'Lower Hip',
    'Upper Left Thigh', 'Upper Right Thigh', 'Lower Left Thigh', 'Lower Right Thigh',
    'Left Knee', 'Right Knee', 'Left Shin', 'Right Shin',
]


def get_part_attention_from_model(model) -> Dict[str, np.ndarray]:
    """Extract part-attention patterns from a loaded model's appearance MoE.

    Returns:
        Dict mapping 'expert_i' -> numpy array of shape [P] with sigmoid(part_logit)
    """
    app_moe = getattr(model, 'app_moe', None)
    if app_moe is None:
        return {}
    result = {}
    for i, expert in enumerate(app_moe.experts):
        if hasattr(expert, 'part_logit'):
            result[f'expert_{i}'] = torch.sigmoid(expert.part_logit).detach().cpu().numpy()
    return result


def plot_part_attention_heatmap(part_attention: Dict[str, np.ndarray],
                                 save_path: str = None, parts_num: int = 16,
                                 part_names: List[str] = None):
    """Plot heatmap showing which body parts each appearance expert attends to.

    Each row is an expert, each column is a body part.
    Helps verify experts specialize on different body regions.
    """
    if not HAS_MPL or not part_attention:
        # Text fallback
        print("\n  Appearance Expert Part-Attention:")
        for name, attn in sorted(part_attention.items()):
            top3 = np.argsort(attn)[-3:][::-1]
            top3_str = ', '.join(f'P{j}({attn[j]:.2f})' for j in top3)
            print(f"    {name}: top-3 parts = {top3_str}")
        return

    if part_names is None:
        part_names = PART_NAMES_16[:parts_num] if parts_num <= 16 \
            else [f'Part {i}' for i in range(parts_num)]

    E = len(part_attention)
    matrix = np.stack([part_attention[f'expert_{i}']
                       for i in range(E) if f'expert_{i}' in part_attention])

    fig, ax = plt.subplots(figsize=(max(10, parts_num * 0.6), max(4, E * 0.5)))
    cmap = LinearSegmentedColormap.from_list('custom', ['#f0f0f0', '#4C72B0', '#C44E52'])
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1)

    ax.set_yticks(range(E))
    ax.set_yticklabels([f'Expert {i}' for i in range(E)])
    ax.set_xticks(range(parts_num))
    ax.set_xticklabels(part_names, rotation=45, ha='right', fontsize=8)

    # Annotate values
    for i in range(E):
        for j in range(parts_num):
            val = matrix[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label='Attention Weight (sigmoid)')
    ax.set_title('Appearance Expert × Body Part Attention\n'
                 '(Row = expert, Column = body part)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved part attention heatmap to {save_path}")
    plt.close()


def plot_spatial_gate_overlay(rgb_frame: np.ndarray, spatial_gate: np.ndarray,
                               save_path: str = None, title: str = "Spatial Gate"):
    """Overlay the PositiveTextGate spatial attention on an RGB frame.

    Args:
        rgb_frame: [H, W, 3] RGB image (uint8 or float)
        spatial_gate: [Hf, Wf] spatial gate output
    """
    if not HAS_MPL:
        return

    import cv2
    if rgb_frame.max() <= 1.0:
        rgb_frame = (rgb_frame * 255).astype(np.uint8)

    H, W = rgb_frame.shape[:2]
    gate_resized = cv2.resize(spatial_gate, (W, H), interpolation=cv2.INTER_LINEAR)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(rgb_frame)
    axes[0].set_title('Input RGB')
    axes[0].axis('off')

    axes[1].imshow(gate_resized, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Spatial Gate')
    axes[1].axis('off')

    axes[2].imshow(rgb_frame)
    axes[2].imshow(gate_resized, cmap='hot', alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ================================================================
#  3. Fusion Expert Token Preference (RGB vs PC)
# ================================================================

def get_fusion_token_preference(model) -> Dict[str, dict]:
    """Extract per-expert RGB vs PC token preference from fusion MoE.

    Uses the learned token_logit to determine which experts prefer
    RGB tokens vs PC tokens.
    """
    fus_moe = getattr(model, 'fus_moe', None)
    if fus_moe is None:
        return {}
    result = {}
    for i, expert in enumerate(fus_moe.experts):
        if hasattr(expert, 'token_logit'):
            logit = expert.token_logit.detach().cpu()
            P = logit.size(0) // 2
            attn = torch.sigmoid(logit)
            rgb_attn = attn[:P].numpy()
            pc_attn = attn[P:].numpy()
            result[f'expert_{i}'] = {
                'rgb_mean': float(rgb_attn.mean()),
                'pc_mean': float(pc_attn.mean()),
                'rgb_per_part': rgb_attn,
                'pc_per_part': pc_attn,
                'preference': 'RGB' if rgb_attn.mean() > pc_attn.mean() else 'PC',
                'ratio': float(rgb_attn.mean() / max(pc_attn.mean(), 1e-8)),
            }
    return result


def plot_fusion_token_preference(pref_dict: Dict[str, dict], save_path: str = None,
                                   title: str = "Fusion Expert Token Preference"):
    """Plot bar chart comparing RGB vs PC preference per fusion expert."""
    if not pref_dict:
        return
    if not HAS_MPL:
        print(f"\n  {title}:")
        for name, info in sorted(pref_dict.items()):
            print(f"    {name}: RGB={info['rgb_mean']:.3f}, "
                  f"PC={info['pc_mean']:.3f}, Pref={info['preference']}")
        return

    E = len(pref_dict)
    x = np.arange(E)
    width = 0.35
    rgb_vals = [pref_dict[f'expert_{i}']['rgb_mean'] for i in range(E)]
    pc_vals = [pref_dict[f'expert_{i}']['pc_mean'] for i in range(E)]

    fig, ax = plt.subplots(figsize=(max(8, E * 1.2), 5))
    bars1 = ax.bar(x - width/2, rgb_vals, width, label='RGB Tokens',
                   color='#E8835C', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, pc_vals, width, label='PC Tokens',
                   color='#4C72B0', alpha=0.85, edgecolor='white')

    ax.set_xlabel('Expert Index')
    ax.set_ylabel('Mean Token Attention (sigmoid)')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'E{i}\n({pref_dict[f"expert_{i}"]["preference"]})'
                        for i in range(E)], fontsize=9)
    ax.legend()
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved fusion token preference to {save_path}")
    plt.close()


def plot_fusion_preference_by_condition(routing_data: List[dict],
                                          conditions: List[str],
                                          save_path: str = None):
    """Analyze and plot which fusion experts are selected under different conditions.

    Args:
        routing_data: list of dicts, each with 'fus_routing' [top_k], 'condition' str
        conditions: list of condition names (e.g., ['day', 'night', 'close', 'far'])
    """
    if not routing_data:
        return

    # Group by condition
    by_condition = defaultdict(list)
    for item in routing_data:
        cond = item.get('condition', 'unknown')
        routing = item.get('fus_routing', None)
        if routing is not None:
            by_condition[cond].append(routing)

    if not by_condition:
        return

    # Compute per-condition expert distribution
    E = max(r.max().item() + 1 for rs in by_condition.values() for r in rs)
    cond_usage = {}
    for cond, routings in by_condition.items():
        all_r = torch.cat(routings, dim=0) if len(routings) > 1 else routings[0]
        usage = torch.zeros(E)
        for i in range(E):
            usage[i] = (all_r == i).float().mean()
        cond_usage[cond] = usage.numpy()

    if not HAS_MPL:
        print("\n  Fusion Expert Usage by Condition:")
        for cond, usage in cond_usage.items():
            top2 = np.argsort(usage)[-2:][::-1]
            print(f"    {cond}: top experts = E{top2[0]}({usage[top2[0]]:.2f}), "
                  f"E{top2[1]}({usage[top2[1]]:.2f})")
        return

    n_conds = len(cond_usage)
    fig, ax = plt.subplots(figsize=(max(10, E * 1.5), 5))
    x = np.arange(E)
    width = 0.8 / n_conds
    cmap = plt.cm.Set2

    for idx, (cond, usage) in enumerate(sorted(cond_usage.items())):
        offset = (idx - n_conds / 2 + 0.5) * width
        ax.bar(x + offset, usage, width, label=cond,
               color=cmap(idx / max(n_conds - 1, 1)), alpha=0.85, edgecolor='white')

    ax.set_xlabel('Expert Index')
    ax.set_ylabel('Selection Frequency')
    ax.set_title('Fusion Expert Selection by Condition', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved condition-based routing to {save_path}")
    plt.close()


# ================================================================
#  4. Aggregation & Data Collection Helpers
# ================================================================

class VizDataCollector:
    """Collects visualization data during evaluation.

    Usage:
        collector = VizDataCollector()
        for batch in dataloader:
            output = model(batch)
            viz = output['inference_feat'].get('visualization', {})
            collector.add(viz, typs=batch_typs, views=batch_views)
        collector.save('viz_data.pth')
        collector.generate_all_plots('./viz_output/')
    """

    def __init__(self):
        self.data = defaultdict(list)
        self.metadata = defaultdict(list)  # typs, views, etc.

    def add(self, viz_dict: dict, typs: list = None, views: list = None,
            conditions: list = None):
        """Add one batch of visualization data."""
        for key, val in viz_dict.items():
            if isinstance(val, torch.Tensor):
                self.data[key].append(val.cpu())
            elif isinstance(val, dict):
                # Nested dict (like part_attention, token_preference)
                self.data[key].append(val)
            else:
                self.data[key].append(val)

        if typs is not None:
            self.metadata['typs'].extend(typs if isinstance(typs, list) else list(typs))
        if views is not None:
            self.metadata['views'].extend(views if isinstance(views, list) else list(views))
        if conditions is not None:
            self.metadata['conditions'].extend(conditions)

    def save(self, path: str):
        """Save collected data to disk."""
        torch.save({'data': dict(self.data), 'metadata': dict(self.metadata)}, path)
        print(f"  Saved visualization data to {path}")

    @classmethod
    def load(cls, path: str) -> 'VizDataCollector':
        """Load collector-format visualization data from disk."""
        obj = cls()
        saved = safe_torch_load(path, map_location='cpu')
        if not isinstance(saved, dict) or 'data' not in saved:
            keys = list(saved.keys())[:20] if isinstance(saved, dict) else type(saved)
            raise ValueError(
                f"{path} is not a VizDataCollector file. "
                f"Expected keys ['data', 'metadata'], got {keys}. "
                f"For auto-saved raw viz_data.pth, use the CLI main path instead."
            )
        obj.data = defaultdict(list, saved.get('data', {}))
        obj.metadata = defaultdict(list, saved.get('metadata', {}))
        return obj

    def get_aggregated_usage(self) -> Dict[str, dict]:
        """Aggregate expert usage from all batches."""
        result = {}
        for tag in ['app', 'str', 'fus']:
            key = f'{tag}_logits'
            if key in self.data and self.data[key]:
                tensors = [t for t in self.data[key] if isinstance(t, torch.Tensor)]
                if tensors:
                    all_logits = torch.cat(tensors, dim=0)
                    result[tag] = compute_expert_usage_from_logits(all_logits, top_k=2)
        return result

    def generate_all_plots(self, output_dir: str, model=None):
        """Generate all visualization plots."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n  Generating visualization plots in {output_dir}/")

        # 1) Expert usage
        usage = self.get_aggregated_usage()
        if usage:
            print_expert_usage_table(usage)
            plot_expert_usage(usage, os.path.join(output_dir, 'expert_usage.png'))

        # 2) Part attention (from model parameters)
        if model is not None:
            part_attn = get_part_attention_from_model(model)
            if part_attn:
                plot_part_attention_heatmap(
                    part_attn, os.path.join(output_dir, 'part_attention.png'))

            # 3) Fusion token preference (from model parameters)
            fus_pref = get_fusion_token_preference(model)
            if fus_pref:
                plot_fusion_token_preference(
                    fus_pref, os.path.join(output_dir, 'fusion_token_preference.png'))

        # 4) Condition-based routing
        if 'fus_routing' in self.data and 'conditions' in self.metadata:
            routing_list = []
            routings = self.data['fus_routing']
            conditions = self.metadata['conditions']
            for i, r in enumerate(routings):
                if isinstance(r, torch.Tensor) and i < len(conditions):
                    for j in range(r.size(0)):
                        routing_list.append({
                            'fus_routing': r[j:j+1],
                            'condition': conditions[min(i, len(conditions)-1)],
                        })
            if routing_list:
                all_conds = sorted(set(item['condition'] for item in routing_list))
                plot_fusion_preference_by_condition(
                    routing_list, all_conds,
                    os.path.join(output_dir, 'routing_by_condition.png'))

        # 5) Quality score distribution
        self._plot_quality_distribution(output_dir)

        print(f"  All plots generated in {output_dir}/\n")

    def _plot_quality_distribution(self, output_dir: str):
        """Plot distribution of RGB vs PC quality scores."""
        if 'rgb_quality' not in self.data or not HAS_MPL:
            return
        rq = torch.cat([t for t in self.data['rgb_quality']
                        if isinstance(t, torch.Tensor)]).numpy().flatten()
        pq = torch.cat([t for t in self.data['pc_quality']
                        if isinstance(t, torch.Tensor)]).numpy().flatten()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(rq, bins=50, color='#E8835C', alpha=0.7, edgecolor='white')
        axes[0].set_title('RGB Quality Score Distribution')
        axes[0].set_xlabel('Quality Score')
        axes[0].axvline(x=rq.mean(), color='red', linestyle='--', label=f'Mean={rq.mean():.3f}')
        axes[0].legend()

        axes[1].hist(pq, bins=50, color='#4C72B0', alpha=0.7, edgecolor='white')
        axes[1].set_title('PC Quality Score Distribution')
        axes[1].set_xlabel('Quality Score')
        axes[1].axvline(x=pq.mean(), color='red', linestyle='--', label=f'Mean={pq.mean():.3f}')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_distribution.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


# ================================================================
#  5. Training Log Integration
# ================================================================

def format_expert_usage_for_log(visual_summary: dict) -> str:
    """Format expert usage from visual_summary for training log output.

    Called during training to print usage stats every N iterations.
    """
    lines = []
    for tag in ['app', 'str', 'fus']:
        usage_keys = sorted([k for k in visual_summary
                             if k.startswith(f'scalar/{tag}_usage_e')])
        if not usage_keys:
            continue
        vals = []
        for k in usage_keys:
            v = visual_summary[k]
            v = v.item() if torch.is_tensor(v) else float(v)
            vals.append(v)
        balance_key = f'scalar/{tag}_balance_ratio'
        br = visual_summary.get(balance_key, None)
        br = br.item() if torch.is_tensor(br) else (float(br) if br is not None else -1)
        usage_str = ' '.join(f'{v:.2f}' for v in vals)
        lines.append(f"  {tag.upper()}: [{usage_str}] bal={br:.3f}")
    return '\n'.join(lines) if lines else ''


# ================================================================
#  6. Model-Level Visualization Hook for V2 (Backward Compatible)
# ================================================================

def add_viz_hooks_to_v2(model):
    """Add visualization data collection to an existing GaitMoEOracleV2 model.

    Monkey-patches the model to collect viz data during inference.
    Call this before evaluation.

    Usage:
        model = load_model(...)
        add_viz_hooks_to_v2(model)
        model.collect_viz = True
        # Now model.forward returns viz data in inference_feat['visualization']
    """
    original_forward = model.forward.__func__ if hasattr(model.forward, '__func__') else model.forward

    def patched_forward(self, inputs):
        result = original_forward(self, inputs)

        if getattr(self, 'collect_viz', False) and not self.training:
            viz = {}

            # Collect expert part attention
            if hasattr(self, 'app_moe') and self.app_moe is not None:
                viz['app_part_attention'] = get_part_attention_from_model(self)

            # Collect fusion token preference
            if hasattr(self, 'fus_moe') and self.fus_moe is not None:
                viz['fus_token_preference'] = get_fusion_token_preference(self)

            # Expert routing from visual_summary
            vs = result.get('visual_summary', {})
            for tag in ['app', 'str', 'fus']:
                wkeys = sorted([k for k in vs if k.startswith(f'scalar/{tag}_e') and k.endswith('_w')])
                if wkeys:
                    weights = torch.stack([vs[k] for k in wkeys])
                    viz[f'{tag}_weights'] = weights.detach().cpu()

            result['inference_feat']['visualization'] = viz

        return result

    import types
    model.forward = types.MethodType(patched_forward, model)
    model.collect_viz = False
    return model


# ================================================================
#  CLI
# ================================================================

def main():
    """CLI to generate plots from auto-saved viz_data.pth.

    Usage:
        python gait_moe_visualize.py --viz_data ./viz_output/viz_data.pth --output_dir ./viz_plots
    """
    parser = argparse.ArgumentParser(description='GaitMoE Visualization')
    parser.add_argument('--viz_data', type=str, required=True,
                        help='Path to viz_data.pth (auto-saved by model)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots (default: same dir as viz_data)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.viz_data) or '.'

    os.makedirs(args.output_dir, exist_ok=True)
    viz_data = safe_torch_load(args.viz_data, map_location='cpu')
    validate_viz_data(viz_data, args.viz_data)
    print(f"  Loaded viz data from {args.viz_data}")
    print(f"  Keys: {list(viz_data.keys())}")

    # 1) Expert usage from per-sample routing logits
    usage = {}
    for tag in ['app', 'str', 'fus']:
        key = f'viz_{tag}_logits'
        if key in viz_data:
            v = viz_data[key]
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v)
            usage[tag] = compute_expert_usage_from_logits(v, top_k=2)
    if usage:
        print_expert_usage_table(usage, title="Test-Set Expert Usage")
        plot_expert_usage(usage, os.path.join(args.output_dir, 'expert_usage.png'))

    # 2) Part attention heatmap
    if 'app_part_attention' in viz_data and viz_data['app_part_attention']:
        plot_part_attention_heatmap(
            viz_data['app_part_attention'],
            os.path.join(args.output_dir, 'part_attention.png'))

    # 3) Fusion token preference
    if 'fus_token_preference' in viz_data and viz_data['fus_token_preference']:
        plot_fusion_token_preference(
            viz_data['fus_token_preference'],
            os.path.join(args.output_dir, 'fusion_token_preference.png'))

    # 4) Quality distribution
    if 'viz_rgb_quality' in viz_data and HAS_MPL:
        rq = torch.as_tensor(viz_data['viz_rgb_quality']).numpy().flatten()
        pq = torch.as_tensor(viz_data['viz_pc_quality']).numpy().flatten()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(rq, bins=50, color='#E8835C', alpha=0.7, edgecolor='white')
        axes[0].set_title(f'RGB Quality (mean={rq.mean():.3f})')
        axes[0].axvline(x=rq.mean(), color='red', linestyle='--')
        axes[1].hist(pq, bins=50, color='#4C72B0', alpha=0.7, edgecolor='white')
        axes[1].set_title(f'PC Quality (mean={pq.mean():.3f})')
        axes[1].axvline(x=pq.mean(), color='red', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'quality_distribution.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\n  All plots saved in {args.output_dir}/")


if __name__ == '__main__':
    main()
