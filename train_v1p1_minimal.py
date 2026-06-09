#!/usr/bin/env python3
"""Stage 1 v1.1-minimal: Same size as v1 + per-dim FSQ level CE loss only."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, time
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
BATCH=32; EPOCHS=80

# ── Same model as v1 ──────────────────────────────────────────────────
class StudentV1(nn.Module):
    def __init__(self,in_dim=80,hidden=256,out_dim=5,num_layers=4,kernel=5):
        super().__init__()
        self.proj_in=nn.Conv1d(in_dim,hidden,1)
        layers=[]
        for i in range(num_layers):
            d=2**i; p=(kernel-1)*d
            layers.append(nn.Sequential(nn.Conv1d(hidden,hidden,kernel,dilation=d,padding=p,padding_mode='replicate'),nn.GroupNorm(8,hidden),nn.GELU(),nn.Conv1d(hidden,hidden,1)))
        self.layers=nn.ModuleList(layers)
        self.down=nn.Conv1d(hidden,hidden,3,stride=2,padding=1,padding_mode='replicate')
        self.proj_out=nn.Conv1d(hidden,out_dim,1)
        self.embed_head=nn.Conv1d(out_dim,768,1)
    def forward(self,x):
        h=self.proj_in(x)
        for layer in self.layers: r=h; h=layer(h); h=h[:,:,:r.shape[2]]; h=h+r
        h=self.down(h); fsq=self.proj_out(h); embed=self.embed_head(fsq)
        return fsq, embed

LEVELS=[8,8,8,5,5]
ce=nn.CrossEntropyLoss()

MEL_DIR="/Users/asill/btrv5/data/mio_mel"
meta=np.load("/Users/asill/btrv5/data/mio_teacher/meta.npz")
n=len(meta['spk_names']); idxs=np.random.RandomState(42).permutation(n)
tr=idxs[:int(n*0.8)]; vl=idxs[int(n*0.8):]

def load(idx):
    d=np.load("{}/mel_{:04d}.npz".format(MEL_DIR,idx))
    return (torch.from_numpy(d['logmel']).float(),torch.from_numpy(d['fsq_5d']).float())

model=StudentV1()
try: model.load_state_dict(torch.load("checkpoints/causal_student_v1.pt",map_location='cpu'))
except: print("From scratch")
model.train()

opt=AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
sched=CosineAnnealingLR(opt,T_max=EPOCHS)

print("Training v1 + level CE (minimal change)...")
for epoch in range(EPOCHS):
    tr_l=0; tr_f=0; tr_c=0; nb=0
    perm=np.random.permutation(len(tr))
    for i in range(0,len(perm),BATCH):
        bi=perm[i:i+BATCH]
        mels=[]; fsqs=[]
        for j in bi: mel,fsq=load(tr[j]); mels.append(mel); fsqs.append(fsq)
        max_T=max(f.shape[0] for f in fsqs)
        xb=torch.stack([F.pad(m,(0,max_T-m.shape[1])) for m in mels])
        yb=torch.stack([F.pad(f,(0,0,0,max_T-f.shape[0])) for f in fsqs]).transpose(1,2)
        fsq_p,emb_p=model(xb)
        Tp=min(fsq_p.shape[2],yb.shape[2]); p=fsq_p[:,:,:Tp]; t=yb[:,:,:Tp]
        L_fsq=F.mse_loss(p,t)
        L_lev=0
        for d in range(5):
            half=(LEVELS[d]-1)/2
            targets=torch.clamp(torch.round((t[:,d]+1)/2*(LEVELS[d]-1)),0,LEVELS[d]-1).long()
            L_lev+=ce(p[:,d].unsqueeze(1).repeat(1,LEVELS[d],1)[:,:,:Tp],targets)*0.3
        loss=L_fsq+L_lev
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tr_l+=loss.item(); tr_f+=L_fsq.item(); tr_c+=L_lev.item(); nb+=1
    sched.step()
    if epoch%20==0 or epoch==EPOCHS-1:
        print("  E{:3d} loss={:.4f} fsq={:.4f} lev={:.4f}".format(epoch,tr_l/max(nb,1),tr_f/max(nb,1),tr_c/max(nb,1)))

os.makedirs("checkpoints",exist_ok=True); torch.save(model.state_dict(),"checkpoints/causal_student_v1p1.pt")

# Quick eval
print()
from miocodec.model import MioCodecModel
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2'); teacher.eval()
model.eval()
d=load(120); mel,fsq_t=d; mel=mel.unsqueeze(0)
with torch.inference_mode():
    fsq_p,_=model(mel); fsq_s=fsq_p.squeeze(0).T; T=min(fsq_s.shape[0],fsq_t.shape[0])
    z_q,_=teacher.local_quantizer.fsq.encode(fsq_s[:T].unsqueeze(0))
    ce_fsq=teacher.local_quantizer.proj_out(z_q).squeeze(0)
d2=np.load("{}/mel_{:04d}.npz".format(MEL_DIR,120)); ce_t=torch.from_numpy(d2['ce_768']).float()
cos=F.cosine_similarity(ce_fsq.flatten(),ce_t[:T].flatten(),dim=0).item()
accs=[]
for d in range(5):
    half=(LEVELS[d]-1)/2
    sl=torch.clamp(torch.round((fsq_s[:T,d]+1)/2*(LEVELS[d]-1)),0,LEVELS[d]-1).long()
    tl=torch.clamp(torch.round((fsq_t[:T,d]+1)/2*(LEVELS[d]-1)),0,LEVELS[d]-1).long()
    accs.append((sl==tl).float().mean().item()*100)
print("  FSQ cos: {:.4f} (v1=0.899)".format(cos))
print("  Level acc: {:.1f}/{:.1f}/{:.1f}/{:.1f}/{:.1f}%".format(*accs))
print("Done!")
