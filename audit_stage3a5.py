#!/usr/bin/env python3
"""
Stage 3a.5: Causal Mel Decoder Generalization + Ablation Audit.
Baselines: mean mel, global-only, shuffled content, wrong global.
Splits: train/val/test/unseen pair.
Metrics: mel L1, cosine, frame-cosine, delta-mel.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio, numpy as np, os, time, warnings, json
from collections import defaultdict
warnings.filterwarnings('ignore')

SR=44100; CONTENT_RATE=25; N_MELS=80
MEL_HOP=int(SR/CONTENT_RATE)  # 1764
device=torch.device('cpu')

# ── Model ──────────────────────────────────────────────────────────────
class AdaLNZero(nn.Module):
    def __init__(self,dim,cond_dim,eps=1e-5):
        super().__init__()
        self.norm=nn.LayerNorm(dim,eps=eps,elementwise_affine=False)
        self.proj=nn.Sequential(nn.SiLU(),nn.Linear(cond_dim,3*dim))
        nn.init.zeros_(self.proj[1].weight); nn.init.zeros_(self.proj[1].bias)
    def forward(self,x,cond):
        xn=self.norm(x); shift,scale,gate=self.proj(cond).chunk(3,dim=-1)
        return xn*(1+scale)+shift, gate

class CausalDecoderBlock(nn.Module):
    def __init__(self,dim=512,cond_dim=128,n_heads=8,ff_mult=4,dropout=0.1):
        super().__init__()
        self.adaln=AdaLNZero(dim,cond_dim); self.adaln2=AdaLNZero(dim,cond_dim)
        self.attn=nn.MultiheadAttention(dim,n_heads,dropout=dropout,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(dim,dim*ff_mult),nn.GELU(),nn.Dropout(dropout),
                              nn.Linear(dim*ff_mult,dim),nn.Dropout(dropout))
    def forward(self,x,cond):
        T=x.shape[1]; mask=torch.tril(torch.ones(T,T,device=x.device,dtype=torch.bool))
        xn,gate=self.adaln(x,cond); attn_out=self.attn(xn,xn,xn,attn_mask=~mask,need_weights=False)[0]
        x=x+gate*attn_out; xn2,gate2=self.adaln2(x,cond); ff_out=self.ff(xn2); x=x+gate2*ff_out
        return x

class CausalMelDecoder(nn.Module):
    def __init__(self,cd=768,cond_dim=128,hidden=512,n_layers=4,n_heads=8,n_mels=80):
        super().__init__()
        self.proj_in=nn.Linear(cd,hidden)
        self.blocks=nn.ModuleList([CausalDecoderBlock(hidden,cond_dim,n_heads) for _ in range(n_layers)])
        self.norm_out=nn.LayerNorm(hidden); self.proj_out=nn.Linear(hidden,n_mels)
    def forward(self,ce,ge):
        x=self.proj_in(ce); cond=ge.unsqueeze(1)
        for b in self.blocks: x=b(x,cond)
        x=self.norm_out(x); return self.proj_out(x).transpose(1,2)

decoder=CausalMelDecoder(); decoder.load_state_dict(torch.load("checkpoints/causal_mel_decoder.pt",map_location='cpu')); decoder.eval()

# ── Data ──────────────────────────────────────────────────────────────
DATA_DIR="/Users/asill/btrv5/data/mio_teacher"
meta=np.load("{}/meta.npz".format(DATA_DIR)); spk_names=meta['spk_names']; n=len(spk_names)
unique_spks=sorted(set(spk_names))
np.random.RandomState(42).shuffle(unique_spks)
tr_spks=set(unique_spks[:20]); vl_spks=set(unique_spks[20:25]); ts_spks=set(unique_spks[25:30])
tr_idx=[i for i in range(n) if spk_names[i] in tr_spks]
vl_idx=[i for i in range(n) if spk_names[i] in vl_spks]
ts_idx=[i for i in range(n) if spk_names[i] in ts_spks]

mel_ext=torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=1024,hop_length=MEL_HOP,n_mels=N_MELS,f_min=80,f_max=14000,center=False,power=1)

def load_data(idx):
    d=np.load("{}/sample_{:04d}.npz".format(DATA_DIR,idx))
    ce=torch.from_numpy(d['ce_768']).float()
    ge=torch.from_numpy(d['ge_128']).float()
    # Teacher mel target
    audio=d['audio']; w=audio-np.mean(audio)
    mel_tgt=mel_ext(torch.from_numpy(w).float().view(1,1,-1))
    mel_tgt=torch.log(mel_tgt.squeeze(1).clamp(min=1e-5)).squeeze(0)  # (n_mels,T)
    return ce,ge,mel_tgt,spk_names[idx]

# Compute mean mel from training set
all_tr_mels=[]
for i in tr_idx:
    _,_,mel,_=load_data(i); all_tr_mels.append(mel)
max_T_mel=max(m.shape[1] for m in all_tr_mels)
all_tr_mels_p=[F.pad(m,(0,max_T_mel-m.shape[1])) for m in all_tr_mels]
mean_mel=torch.stack(all_tr_mels_p).mean(dim=0)

# ── Evaluate ──────────────────────────────────────────────────────────
def compute_metrics(pred,target,prefix=""):
    """mel L1, cosine, frame-cosine, delta-mel L1."""
    Tp=min(pred.shape[1],target.shape[1])
    p=pred[:,:Tp]; t=target[:,:Tp]
    m={}
    m['l1']=F.l1_loss(p,t).item()
    m['cos']=F.cosine_similarity(p.flatten(),t.flatten(),dim=0).item()
    # Frame-wise cosine
    fcos=[]
    for i in range(Tp):
        c=F.cosine_similarity(p[:,i:i+1].flatten(),t[:,i:i+1].flatten(),dim=0)
        fcos.append(float(c))
    m['fcos_mean']=np.mean(fcos); m['fcos_p10']=np.percentile(fcos,10)
    m['fcos_p50']=np.median(fcos); m['fcos_p90']=np.percentile(fcos,90)
    # Delta-mel
    dp=p[:,1:]-p[:,:-1]; dt=t[:,1:]-t[:,:-1]
    m['delta_l1']=F.l1_loss(dp,dt).item() if Tp>1 else 0
    return m

def run_model(ce,ge):
    with torch.no_grad():
        mel=decoder(ce.unsqueeze(0),ge.unsqueeze(0)).squeeze(0)
    return mel

print("="*80)
print("  STAGE 3a.5: CAUSAL MEL DECODER AUDIT")
print("="*80)

results={}

for split_name,idxs in [("TRAIN",tr_idx),("VAL",vl_idx),("TEST",ts_idx)]:
    metrics=defaultdict(list)
    for idx in idxs:
        ce,ge,mel_tgt,spk=load_data(idx)
        T=ce.shape[0]; T_mel=mel_tgt.shape[1]
        # Trim to align
        Tc=min(T,int(T_mel*CONTENT_RATE/SR*mel_tgt.shape[1]/T_mel+1) or T)
        
        # Real model
        mel_pred=run_model(ce,ge)
        m=compute_metrics(mel_pred,mel_tgt)
        for k,v in m.items(): metrics[k].append(v)
    results[split_name]={k:np.mean(v) for k,v in metrics.items()}

# ── Baselines (on VAL set) ────────────────────────────────────────────
print()
print("--- Ablation Baselines (VAL set) ---")
bl_metrics=defaultdict(list)

# Compute random GE from another speaker
all_ge={}; all_ce={}
for i in vl_idx:
    ce,ge,_,spk=load_data(i)
    all_ge[i]=ge; all_ce[i]=ce

for idx in vl_idx:
    ce,ge,mel_tgt,spk=load_data(idx)
    
    # A: Real model
    mel=run_model(ce,ge); m=compute_metrics(mel,mel_tgt); bl_metrics['real'].append(m)
    
    # B: Mean mel baseline (match target length)
    mean_t=mean_mel[:,:mel_tgt.shape[1]]
    m=compute_metrics(mean_t.unsqueeze(0),mel_tgt.unsqueeze(0)); bl_metrics['mean_mel'].append(m)
    
    # C: Global-only (zero content)
    mel=run_model(torch.zeros_like(ce),ge); m=compute_metrics(mel,mel_tgt); bl_metrics['zero_content'].append(m)
    
    # D: Shuffled content (random CE from another sample)
    other_idx=vl_idx[(vl_idx.index(idx)+1)%len(vl_idx)]
    shuffled_ce=all_ce[other_idx]
    mel=run_model(shuffled_ce,ge); m=compute_metrics(mel,mel_tgt); bl_metrics['shuffled_ce'].append(m)
    
    # E: Wrong global
    wrong_ge=all_ge[vl_idx[(vl_idx.index(idx)+1)%len(vl_idx)]]
    mel=run_model(ce,wrong_ge); m=compute_metrics(mel,mel_tgt); bl_metrics['wrong_ge'].append(m)

print("  Baseline         Mel L1   Cosine   fCos p50  Δ-L1")
print("  "+"-"*55)
for name in ['real','mean_mel','zero_content','shuffled_ce','wrong_ge']:
    ms=bl_metrics[name]
    avg={k:np.mean([m[k] for m in ms]) for k in ms[0]}
    print("  {:<16s} {:.4f}   {:.4f}   {:.4f}   {:.4f}".format(
        name,avg['l1'],avg['cos'],avg['fcos_p50'],avg['delta_l1']))

# ── Split summary ─────────────────────────────────────────────────────
print()
print("--- Split Summary ---")
print("  Split     Mel L1   Cosine   fCos p50  Δ-L1")
print("  "+"-"*50)
for name in ['TRAIN','VAL','TEST']:
    r=results[name]
    print("  {:<8s} {:.4f}   {:.4f}   {:.4f}   {:.4f}".format(
        name,r['l1'],r['cos'],r['fcos_p50'],r['delta_l1']))

# ── Overfitting check ─────────────────────────────────────────────────
print()
gap=results['VAL']['l1']/results['TRAIN']['l1']
print("  val/train L1 gap: {:.2f}x".format(gap))
print("  → {} ".format("PASS (<2x)" if gap<2 else "OVERFIT"))

# ── Content usage check ──────────────────────────────────────────────
real_cos=bl_metrics['real'][0]['cos']; real_cos=np.mean([m['cos'] for m in bl_metrics['real']])
shuf_cos=np.mean([m['cos'] for m in bl_metrics['shuffled_ce']])
zero_cos=np.mean([m['cos'] for m in bl_metrics['zero_content']])
print("  Content usage: real-cos={:.3f} shuffled-cos={:.3f} zero-cos={:.3f}".format(real_cos,shuf_cos,zero_cos))
if real_cos>shuf_cos+0.05 and real_cos>zero_cos+0.05:
    print("  → DECODER USES CONTENT (real >> shuffled/zero)")
else:
    print("  → WEAK CONTENT USAGE — decoder relies on global/prior")

# ── Global sensitivity ────────────────────────────────────────────────
wrong_cos=np.mean([m['cos'] for m in bl_metrics['wrong_ge']])
print("  Global sensitivity: real-cos={:.3f} wrong-ge-cos={:.3f}".format(real_cos,wrong_cos))
if real_cos>wrong_cos+0.02:
    print("  → DECODER USES GLOBAL (correct global improves mel)")
else:
    print("  → WEAK GLOBAL USAGE")

print()
print("Done!")
