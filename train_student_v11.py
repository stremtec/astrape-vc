#!/usr/bin/env python3
"""
Stage 1 v1.1: Bigger causal content student with FSQ level CE + cosine loss.
- Larger TCN (hidden=384, 6 layers, kernel=7)
- Per-dim FSQ level classification
- Content embedding cosine distillation
- Alignment sweep
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, time
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

BATCH=32; EPOCHS=60
device=torch.device('cpu')

# ── Bigger TCN Encoder ────────────────────────────────────────────────
class ContentStudentV11(nn.Module):
    def __init__(self,in_dim=80,hidden=384,out_dim=5,num_layers=6,kernel=7):
        super().__init__()
        self.proj_in=nn.Conv1d(in_dim,hidden,1)
        layers=[]
        for i in range(num_layers):
            d=2**i; p=(kernel-1)*d
            layers.append(nn.Sequential(
                nn.Conv1d(hidden,hidden,kernel,dilation=d,padding=p,padding_mode='replicate'),
                nn.GroupNorm(min(16,hidden),hidden),nn.GELU(),
                nn.Conv1d(hidden,hidden,1)))
        self.layers=nn.ModuleList(layers)
        self.down=nn.Conv1d(hidden,hidden,3,stride=2,padding=1,padding_mode='replicate')
        self.fsq_head=nn.Conv1d(hidden,out_dim,1)
        # Per-dim level classifiers
        self.level_heads=nn.ModuleList([
            nn.Conv1d(hidden,8,1),  # dim 0: 8 levels
            nn.Conv1d(hidden,8,1),  # dim 1: 8 levels
            nn.Conv1d(hidden,8,1),  # dim 2: 8 levels
            nn.Conv1d(hidden,5,1),  # dim 3: 5 levels
            nn.Conv1d(hidden,5,1),  # dim 4: 5 levels
        ])
        self.embed_head=nn.Conv1d(out_dim,768,1)
    
    def forward(self,x):
        h=self.proj_in(x)
        for layer in self.layers:
            r=h; h=layer(h); h=h[:,:,:r.shape[2]]; h=h+r
        h=self.down(h)
        fsq=self.fsq_head(h)  # (B,5,T)
        levels=[head(h) for head in self.level_heads]  # list of (B,levels_i,T)
        embed=self.embed_head(fsq)
        return fsq, levels, embed

# ── Level targets from FSQ values ──────────────────────────────────────
LEVELS=[8,8,8,5,5]
def fsq_to_levels(fsq_5d):
    """Convert FSQ values to per-dim level indices (0 to levels[d]-1)."""
    # fsq_5d: (T,5) in [-1,1] range
    # Map [-1,1] → [0, levels[d]-1]
    levels_tensor=torch.tensor(LEVELS).float()
    half=(levels_tensor-1)/2
    indices=torch.clamp(torch.round((fsq_5d+1)/2*(levels_tensor-1)),0,levels_tensor-1).long()
    return indices[:,0],indices[:,1],indices[:,2],indices[:,3],indices[:,4]

# ── Data ──────────────────────────────────────────────────────────────
MEL_DIR="/Users/asill/btrv5/data/mio_mel"
meta=np.load("/Users/asill/btrv5/data/mio_teacher/meta.npz")
n=len(meta['spk_names'])
idxs=np.random.RandomState(42).permutation(n)
tr=idxs[:int(n*0.8)]; vl=idxs[int(n*0.8):]

def load(idx):
    d=np.load("{}/mel_{:04d}.npz".format(MEL_DIR,idx))
    return (torch.from_numpy(d['logmel']).float(),
            torch.from_numpy(d['fsq_5d']).float(),
            torch.from_numpy(d['ce_768']).float())

# ── Training ────────────────────────────────────────────────────────────
model=ContentStudentV11().to(device); model.train()
# Start from v1 checkpoint (partial load)
try: 
    v1=torch.load("checkpoints/causal_student_v1.pt",map_location='cpu')
    model.load_state_dict(v1,strict=False)
    print("Loaded v1 checkpoint (partial)")
except: print("Training from scratch")

opt=AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
sched=CosineAnnealingLR(opt,T_max=EPOCHS)
ce_loss=nn.CrossEntropyLoss()

print("Training Stage 1 v1.1 (bigger TCN + level CE + embed cosine)...")

from miocodec.model import MioCodecModel
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2'); teacher.eval()

for epoch in range(EPOCHS):
    model.train(); tr_loss=0; tr_fsq=0; tr_lev=0; tr_cos=0; nb=0
    perm=np.random.permutation(len(tr))
    
    for i in range(0,len(perm),BATCH):
        bi=perm[i:i+BATCH]
        mels=[]; fsqs=[]; ces=[]
        for j in bi:
            mel,fsq,ce=load(tr[j]); mels.append(mel); fsqs.append(fsq); ces.append(ce)
        
        max_T=max(f.shape[0] for f in fsqs)
        mel_b=torch.stack([F.pad(m,(0,max_T-m.shape[1])) for m in mels]).to(device)
        fsq_b=torch.stack([F.pad(f,(0,0,0,max_T-f.shape[0])) for f in fsqs]).transpose(1,2).to(device)
        ce_b=torch.stack([F.pad(ce,(0,0,0,max_T-ce.shape[0])) for ce in ces]).to(device)
        
        fsq_pred,level_preds,emb_pred=model(mel_b)
        Tp=min(fsq_pred.shape[2],fsq_b.shape[2])
        
        # FSQ MSE
        L_fsq=F.mse_loss(fsq_pred[:,:,:Tp],fsq_b[:,:,:Tp])
        
        # Per-dim level CE
        L_level=0
        for d in range(5):
            # Get level targets from fsq_b
            fsq_vals=fsq_b[:,d,:Tp]  # (B,T)
            half=(LEVELS[d]-1)/2
            level_targets=torch.clamp(torch.round((fsq_vals+1)/2*(LEVELS[d]-1)),0,LEVELS[d]-1).long()
            L_level=L_level+ce_loss(level_preds[d][:,:,:Tp],level_targets)
        
        # Content embedding cosine (via teacher FSQ path)
        with torch.inference_mode():
            fsq_t=fsq_pred[:,:,:Tp].transpose(1,2)  # (B,T,5)
            z_q,_=teacher.local_quantizer.fsq.encode(fsq_t.reshape(-1,5).unsqueeze(0))
        # Simplified: use embed_head output
        emb_p=emb_pred[:,:,:Tp]; emb_t=ce_b[:,:Tp,:].transpose(1,2)
        cos_sim=F.cosine_similarity(emb_p.reshape(emb_p.shape[0],-1),
                                     emb_t.reshape(emb_t.shape[0],-1),dim=1).mean()
        L_cos=1-cos_sim
        
        loss=L_fsq+0.3*L_level+0.5*L_cos
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tr_loss+=loss.item(); tr_fsq+=L_fsq.item(); tr_lev+=L_level.item(); tr_cos+=L_cos.item(); nb+=1
    sched.step()
    
    # Val
    model.eval(); v_cos=0; vn=0
    with torch.no_grad():
        for j in vl[:20]:
            mel,fsq,ce=load(j)
            mel=mel.unsqueeze(0).to(device); ce=ce.unsqueeze(0).to(device)
            _,_,emb=model(mel)
            emb_p=emb; emb_t=ce.transpose(1,2); Tp=min(emb_p.shape[2],emb_t.shape[2])
            cos=F.cosine_similarity(emb_p[:,:,:Tp].reshape(1,-1),emb_t[:,:,:Tp].reshape(1,-1),dim=1).mean()
            v_cos+=cos.item(); vn+=1
    model.train()
    
    if epoch%10==0 or epoch==EPOCHS-1:
        print("  E{:3d} loss={:.4f} fsq={:.4f} lev={:.4f} cos={:.4f} val_cos={:.4f}".format(
            epoch,tr_loss/max(nb,1),tr_fsq/max(nb,1),tr_lev/max(nb,1),tr_cos/max(nb,1),v_cos/max(vn,1)))

os.makedirs("checkpoints",exist_ok=True)
torch.save(model.state_dict(),"checkpoints/causal_student_v11.pt")

# ── Quick eval ─────────────────────────────────────────────────────────
print()
print("=== Content Embedding Quality ===")
model.eval()
d=load(120)  # test sample
mel,fsq_t,ce_t=d
mel=mel.unsqueeze(0).to(device)
with torch.inference_mode():
    fsq_pred,_,emb_pred=model(mel)
    fsq_s=fsq_pred.squeeze(0).T
    z_q,_=teacher.local_quantizer.fsq.encode(fsq_s.unsqueeze(0))
    ce_fsq=teacher.local_quantizer.proj_out(z_q).squeeze(0)
T=min(ce_fsq.shape[0],ce_t.shape[0])
cos_fsq=F.cosine_similarity(ce_fsq[:T].flatten(),ce_t[:T].flatten(),dim=0).item()

# Per-dim level accuracy
accs=[]
for d in range(5):
    half=(LEVELS[d]-1)/2
    s_lev=torch.clamp(torch.round((fsq_s[:T,d]+1)/2*(LEVELS[d]-1)),0,LEVELS[d]-1).long()
    t_lev=torch.clamp(torch.round((fsq_t[:T,d]+1)/2*(LEVELS[d]-1)),0,LEVELS[d]-1).long()
    acc=(s_lev==t_lev).float().mean().item()*100
    accs.append(acc)

print("  FSQ path cos: {:.4f} (v1 was 0.899)".format(cos_fsq))
print("  Level acc: d0={:.1f}% d1={:.1f}% d2={:.1f}% d3={:.1f}% d4={:.1f}%".format(*accs))
print("  Chance: d0-2=12.5% d3-4=20%")
print("Done!")
