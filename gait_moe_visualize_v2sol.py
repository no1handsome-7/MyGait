#!/usr/bin/env python3
"""GaitMoEOracleV2Sol visualization suite.

Supports old V2Max/QACT figures and new CACT mechanism figures:
  Fig1 t-SNE, Fig2 distance distribution,
  Fig3 expert specialization, Fig4 condition-routing heatmap,
  Fig5 fusion modality preference, Fig6 CACT condition embedding t-SNE,
  Fig7 CACT token attention, Fig8 routing t-SNE,
  Fig9 CACT token-expert association, Fig10 condition-gated aux weights.
"""
import os, argparse, math
from collections import defaultdict
import numpy as np
import torch

DPI = 300
PALETTE = {'blue':'#0072B2','orange':'#E69F00','green':'#009E73','pink':'#CC79A7','cyan':'#56B4E9','red':'#D55E00','yellow':'#F0E442','gray':'#999999'}
BODY_PARTS_16 = ['Head','Forehead','Neck','Shoulder','Upper Arm','Chest','Elbow','Waist','Hip','Wrist','Thigh-U','Thigh-L','Knee','Calf','Ankle','Foot']

def _setup_style():
    import matplotlib
    matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'figure.dpi':DPI,'savefig.dpi':DPI,'savefig.bbox':'tight','pdf.fonttype':42,'ps.fonttype':42})

def _np(x):
    if x is None: return None
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray): return x
    if isinstance(x, (list,tuple)):
        vals=[_np(v) for v in x if v is not None]
        if not vals: return None
        try: return np.concatenate(vals, axis=0)
        except Exception:
            try: return np.stack(vals, axis=0)
            except Exception: return np.asarray(vals, dtype=object)
    try: return np.asarray(x)
    except Exception: return x

def _alias(d, k, cands):
    if k in d and d[k] is not None: return
    for c in cands:
        if c in d and d[c] is not None:
            d[k]=d[c]; return

def normalize_viz_data(d):
    d = {} if d is None else dict(d)
    _alias(d,'viz_cact_cond_emb',['viz_qact_cond_emb','viz_cond_emb'])
    _alias(d,'viz_cact_rgb_attn',['viz_qact_rgb_attn','viz_qact_rgb_self_attn'])
    _alias(d,'viz_cact_pc_attn',['viz_qact_pc_attn','viz_qact_pc_self_attn'])
    _alias(d,'viz_cact_token_energy',['token_energy','viz_qact_token_energy'])
    return d

def load_viz_data(viz_dir):
    p=os.path.join(viz_dir,'viz_data.pth')
    if not os.path.exists(p): raise FileNotFoundError(p)
    return normalize_viz_data(torch.load(p,map_location='cpu',weights_only=False))

def load_info(viz_dir):
    for name in ['inference_info.pth','inference_info.pt']:
        p=os.path.join(viz_dir,name)
        if os.path.exists(p): return torch.load(p,map_location='cpu',weights_only=False)
    return None

def _make_tsne(n):
    from sklearn.manifold import TSNE
    import inspect
    kwargs=dict(n_components=2,perplexity=max(1,min(30,n-1)),random_state=42,init='pca',learning_rate='auto')
    if 'max_iter' in inspect.signature(TSNE).parameters: kwargs['max_iter']=1000
    else: kwargs['n_iter']=1000
    return TSNE(**kwargs)

def _emb_labels(info):
    if not info: return None,None
    emb=_np(info.get('embeddings'))
    lab=_np(info.get('labels'))
    if emb is not None and emb.ndim>2: emb=emb.reshape(emb.shape[0],-1)
    return emb, lab

def plot_tsne(viz_data,out_dir,info_dict=None,base_data=None,base_info=None):
    import matplotlib.pyplot as plt
    _setup_style(); emb,lab=_emb_labels(info_dict)
    if emb is None or lab is None or len(emb)<5: print('[Fig1] missing embeddings/labels'); return
    n=min(len(emb),3000); idx=np.random.choice(len(emb),n,replace=False) if len(emb)>n else np.arange(n)
    pts=_make_tsne(n).fit_transform(emb[idx].reshape(n,-1)); labels=lab[idx].reshape(-1)
    fig,ax=plt.subplots(figsize=(4,3.2)); sc=ax.scatter(pts[:,0],pts[:,1],c=labels,s=5,cmap='tab20',alpha=.75,rasterized=True)
    ax.set_title('Embedding t-SNE'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    os.makedirs(out_dir,exist_ok=True); p=os.path.join(out_dir,'fig1_tsne_embedding.pdf'); fig.savefig(p); fig.savefig(p.replace('.pdf','.png')); plt.close(fig); print('[Fig1]',p)

def plot_distance_distribution(viz_data,out_dir,info_dict=None,base_info=None):
    import matplotlib.pyplot as plt
    _setup_style(); emb,lab=_emb_labels(info_dict)
    if emb is None or lab is None or len(emb)<5: print('[Fig2] missing embeddings/labels'); return
    x=torch.as_tensor(emb).float(); x=torch.nn.functional.normalize(x,dim=1); labels=np.asarray(lab).reshape(-1)
    dist=(1-(x@x.t())).cpu().numpy(); same=labels[:,None]==labels[None,:]; tri=np.triu(np.ones_like(same,dtype=bool),1)
    intra=dist[same&tri]; inter=dist[(~same)&tri]
    fig,ax=plt.subplots(figsize=(4,2.8)); ax.hist(intra,bins=60,density=True,alpha=.55,label='Intra-ID',color=PALETTE['blue']); ax.hist(inter,bins=60,density=True,alpha=.45,label='Inter-ID',color=PALETTE['orange'])
    ax.set_xlabel('Cosine Distance'); ax.set_ylabel('Density'); ax.set_title('Intra-/Inter-class Distance'); ax.legend(frameon=False)
    p=os.path.join(out_dir,'fig2_distance_distribution.pdf'); fig.savefig(p); fig.savefig(p.replace('.pdf','.png')); plt.close(fig); print('[Fig2]',p)

def plot_expert_body_part_specialization(viz_data,out_dir,parts_num=16):
    import matplotlib.pyplot as plt
    _setup_style(); labels=BODY_PARTS_16[:parts_num]
    for key,prefix,title in [('app_expert_info','app','Appearance'),('str_expert_info','str','Structure')]:
        info=viz_data.get(key); 
        if not info: print('[Fig3]',key,'missing'); continue
        mats=[]; names=[]
        for i in range(len(info)):
            item=info.get(f'expert_{i}',{})
            if 'part_attention' in item: mats.append(np.asarray(item['part_attention']).reshape(-1)[:parts_num]); names.append(f'E{i}')
        if not mats: continue
        mat=np.stack(mats); fig,ax=plt.subplots(figsize=(6,max(2,.35*len(mats)+1)))
        im=ax.imshow(mat,aspect='auto',cmap='viridis'); ax.set_xticks(range(parts_num)); ax.set_xticklabels(labels,rotation=60,ha='right',fontsize=6); ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.set_title(f'{title} Expert Body-Part Specialization'); fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
        p=os.path.join(out_dir,f'fig3_{prefix}_expert_specialization.pdf'); fig.savefig(p); fig.savefig(p.replace('.pdf','.png')); plt.close(fig); print('[Fig3]',p)

def _condition_groups(info,n):
    ty=_np(info.get('types')) if info else None
    if ty is not None and len(ty)>=n:
        return np.asarray([str(t) for t in ty[:n]]), 'Sequence Type'
    return np.asarray(['all']*n), 'All Samples'

def plot_expert_routing_heatmap(viz_data,out_dir,info_dict=None):
    import matplotlib.pyplot as plt
    _setup_style(); panels=[]
    for tag,key in [('Appearance','viz_app_logits'),('Structure','viz_str_logits'),('Fusion','viz_fus_logits')]:
        l=_np(viz_data.get(key));
        if l is not None: panels.append((tag,np.asarray(l).reshape(np.asarray(l).shape[0],-1)))
    if not panels: print('[Fig4] no routing logits'); return
    n=min(len(x[1]) for x in panels); groups,gname=_condition_groups(info_dict,n); uniq=list(dict.fromkeys(groups.tolist()))[:24]
    fig,axes=plt.subplots(1,len(panels),figsize=(3.4*len(panels),max(2.6,.22*len(uniq)+1.5)))
    if len(panels)==1: axes=[axes]
    for ax,(tag,logits) in zip(axes,panels):
        logits=logits[:n]; E=logits.shape[1]; top=np.argsort(logits,axis=1)[:,-min(2,E):]
        heat=np.zeros((len(uniq),E)); cnt=np.zeros(len(uniq))
        idx={u:i for i,u in enumerate(uniq)}
        for s,g in enumerate(groups[:n]):
            if g not in idx: continue
            ci=idx[g]; cnt[ci]+=1
            for e in top[s]: heat[ci,e]+=1
        heat[cnt>0]/=cnt[cnt>0,None]
        im=ax.imshow(heat,aspect='auto',cmap='YlOrRd'); ax.set_xticks(range(E)); ax.set_xticklabels([f'E{i}' for i in range(E)]); ax.set_yticks(range(len(uniq))); ax.set_yticklabels(uniq,fontsize=6); ax.set_title(f'{tag} MoE'); ax.set_xlabel('Expert'); ax.set_ylabel(gname); fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
    p=os.path.join(out_dir,'fig4_condition_routing_heatmap.pdf'); fig.savefig(p); fig.savefig(p.replace('.pdf','.png')); plt.close(fig); print('[Fig4]',p)

def plot_fusion_modality_preference(viz_data,out_dir):
    import matplotlib.pyplot as plt
    _setup_style(); info=viz_data.get('fus_expert_info')
    if not info: print('[Fig5] no fusion expert info'); return
    rgb=[]; pc=[]; labs=[]
    for i in range(len(info)):
        it=info.get(f'expert_{i}',{}); rgb.append(float(it.get('rgb_preference',.5))); pc.append(float(it.get('pc_preference',.5))); labs.append(f'E{i}')
    x=np.arange(len(labs)); w=.35; fig,ax=plt.subplots(figsize=(5,2.8)); ax.bar(x-w/2,rgb,w,label='RGB',color=PALETTE['blue']); ax.bar(x+w/2,pc,w,label='PC',color=PALETTE['orange']); ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0,1); ax.legend(frameon=False); ax.set_title('Fusion Expert Modality Preference')
    p=os.path.join(out_dir,'fig5_fusion_modality_preference.pdf'); fig.savefig(p); fig.savefig(p.replace('.pdf','.png')); plt.close(fig); print('[Fig5]',p)

def plot_cact_cond_tsne(viz_data,out_dir,info_dict=None):
    import matplotlib.pyplot as plt
    _setup_style(); emb=_np(viz_data.get('viz_cact_cond_emb'))
    if emb is None or len(emb)<5: print('[Fig6] no CACT cond emb'); return
    emb=np.asarray(emb).reshape(len(emb),-1); n=min(len(emb),3000); idx=np.random.choice(len(emb),n,replace=False) if len(emb)>n else np.arange(n); pts=_make_tsne(n).fit_transform(emb[idx])
    groups,_=_condition_groups(info_dict,n); colors=np.arange(n)
    fig,ax=plt.subplots(figsize=(4,3.2)); ax.scatter(pts[:,0],pts[:,1],c=colors,s=6,cmap='viridis',alpha=.75,rasterized=True); ax.set_title('CACT Condition Embedding t-SNE'); ax.set_xticks([]); ax.set_yticks([])
    p=os.path.join(out_dir,'fig6_cact_condition_tsne.pdf'); fig.savefig(p); fig.savefig(p.replace('.pdf','.png')); plt.close(fig); print('[Fig6]',p)

def plot_cact_token_attention(viz_data,out_dir,parts_num=16):
    import matplotlib.pyplot as plt
    _setup_style(); rgb=_np(viz_data.get('viz_cact_rgb_attn')); pc=_np(viz_data.get('viz_cact_pc_attn'))
    if rgb is None: print('[Fig7] no CACT attention'); return
    def mean(a):
        a=np.asarray(a)
        if a.ndim==5: return a.mean(axis=(0,1,2))
        if a.ndim==4: return a.mean(axis=(0,1))
        if a.ndim==3: return a.mean(axis=0)
        return a.reshape(-1,a.shape[-1])
    panels=[('RGB self-probe',mean(rgb),'Blues')]
    if pc is not None: panels.append(('PC self-probe',mean(pc),'Oranges'))
    fig,axes=plt.subplots(1,len(panels),figsize=(4*len(panels),3));
    if len(panels)==1: axes=[axes]
    for ax,(title,m,cmap) in zip(axes,panels):
        im=ax.imshow(m,aspect='auto',cmap=cmap); ax.set_title(title); ax.set_xlabel('Body Part'); ax.set_ylabel('CACT Token'); ax.set_xticks(range(min(m.shape[1],parts_num))); ax.set_xticklabels(BODY_PARTS_16[:min(m.shape[1],parts_num)],rotation=60,ha='right',fontsize=6); fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
    p=os.path.join(out_dir,'fig7_cact_token_attention.pdf'); fig.savefig(p); fig.savefig(p.replace('.pdf','.png')); plt.close(fig); print('[Fig7]',p)

def plot_routing_tsne(viz_data,out_dir):
    import matplotlib.pyplot as plt
    _setup_style(); l=_np(viz_data.get('viz_fus_logits'))
    if l is None or len(l)<5: print('[Fig8] no fusion logits'); return
    l=np.asarray(l).reshape(len(l),-1); n=min(len(l),3000); idx=np.random.choice(len(l),n,replace=False) if len(l)>n else np.arange(n); pts=_make_tsne(n).fit_transform(l[idx]); top=l[idx].argmax(1)
    fig,ax=plt.subplots(figsize=(4,3.2)); sc=ax.scatter(pts[:,0],pts[:,1],c=top,s=7,cmap='tab10',alpha=.8,rasterized=True); ax.set_title('Fusion Routing t-SNE'); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(sc,ax=ax,fraction=.046,pad=.04)
    p=os.path.join(out_dir,'fig8_routing_tsne.pdf'); fig.savefig(p); fig.savefig(p.replace('.pdf','.png')); plt.close(fig); print('[Fig8]',p)

def plot_cact_token_expert(viz_data,out_dir):
    import matplotlib.pyplot as plt
    _setup_style(); corr=_np(viz_data.get('viz_cact_token_expert_corr'))
    if corr is None:
        energy=_np(viz_data.get('viz_cact_token_energy')); logits=_np(viz_data.get('viz_fus_logits'))
        if energy is None or logits is None: print('[Fig9] no token-expert data'); return
        x=np.asarray(energy).reshape(len(energy),-1); y=np.exp(logits)/np.exp(logits).sum(1,keepdims=True); n=min(len(x),len(y)); x=x[:n]; y=y[:n]; x=(x-x.mean(0))/(x.std(0)+1e-6); y=(y-y.mean(0))/(y.std(0)+1e-6); corr=x.T@y/max(n-1,1)
    corr=np.asarray(corr); fig,ax=plt.subplots(figsize=(5,3.2)); im=ax.imshow(corr,aspect='auto',cmap='coolwarm',vmin=-np.nanmax(abs(corr)),vmax=np.nanmax(abs(corr))); ax.set_xlabel('Fusion Expert'); ax.set_ylabel('CACT Token'); ax.set_title('CACT Token × Expert Association'); fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
    p=os.path.join(out_dir,'fig9_cact_token_expert_association.pdf'); fig.savefig(p); fig.savefig(p.replace('.pdf','.png')); plt.close(fig); print('[Fig9]',p)

def plot_aux_weights(viz_data,out_dir):
    import matplotlib.pyplot as plt
    _setup_style(); wr=_np(viz_data.get('viz_rgb_aux_weight')); wp=_np(viz_data.get('viz_pc_aux_weight'))
    if wr is None or wp is None: print('[Fig10] no aux weights'); return
    wr=np.asarray(wr).reshape(-1); wp=np.asarray(wp).reshape(-1); fig,ax=plt.subplots(figsize=(4,2.8)); ax.hist(wr,bins=40,alpha=.55,label='RGB aux',color=PALETTE['blue']); ax.hist(wp,bins=40,alpha=.45,label='PC aux',color=PALETTE['orange']); ax.set_title('Condition-Gated Aux Weights'); ax.set_xlabel('Weight'); ax.set_ylabel('Count'); ax.legend(frameon=False)
    p=os.path.join(out_dir,'fig10_condition_gated_aux_weights.pdf'); fig.savefig(p); fig.savefig(p.replace('.pdf','.png')); plt.close(fig); print('[Fig10]',p)

def run_all_visualizations(viz_data, info_dict=None, out_dir='./viz_figures/v2sol', parts_num=16, base_data=None, base_info=None):
    os.makedirs(out_dir,exist_ok=True); viz_data=normalize_viz_data(viz_data)
    steps=[lambda:plot_tsne(viz_data,out_dir,info_dict,base_data,base_info),lambda:plot_distance_distribution(viz_data,out_dir,info_dict,base_info),lambda:plot_expert_body_part_specialization(viz_data,out_dir,parts_num),lambda:plot_expert_routing_heatmap(viz_data,out_dir,info_dict),lambda:plot_fusion_modality_preference(viz_data,out_dir),lambda:plot_cact_cond_tsne(viz_data,out_dir,info_dict),lambda:plot_cact_token_attention(viz_data,out_dir,parts_num),lambda:plot_routing_tsne(viz_data,out_dir),lambda:plot_cact_token_expert(viz_data,out_dir),lambda:plot_aux_weights(viz_data,out_dir)]
    for fn in steps:
        try: fn()
        except Exception as e: print('[Visualization] skipped:',type(e).__name__,e)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--viz_dir',required=True); ap.add_argument('--out_dir',default=None); ap.add_argument('--parts_num',type=int,default=16); args=ap.parse_args()
    viz=load_viz_data(args.viz_dir); info=load_info(args.viz_dir); out=args.out_dir or os.path.join(args.viz_dir,'figures')
    run_all_visualizations(viz,info,out,args.parts_num)
if __name__=='__main__': main()
