#!/usr/bin/env python3
"""V3-4k-mini: Fast training — preload data, short seqs, few epochs."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, os, time, math, gc
from torch.optim import AdamW

BATCH=4; EPOCHS=20; device='cpu'

class PositionalEncoding(nn.Module):
    def __init__(self,dim,max_len=200):
        super().__init__(); pe=torch.zeros(max_len,dim)
        pos=torch.arange(0,max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,dim,2).float()*(-math.log(10000.0)/dim))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1),:].contiguous()
class CTB(nn.Module):
    def __init__(self,dim=384,n_heads=8):
        super().__init__()
        self.norm1=nn.LayerNorm(dim); self.attn=nn.MultiheadAttention(dim,n_heads,batch_first=True)
        self.norm2=nn.LayerNorm(dim); self.ff=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))
    def forward(self,x):
        T=x.shape[1]; mask=torch.tril(torch.ones(T,T,device=x.device,dtype=torch.bool))
        xn=self.norm1(x); a=self.attn(xn,xn,xn,attn_mask=~mask,need_weights=False)[0]; x=x+a
        xn=self.norm2(x); x=x+self.ff(xn); return x

class ContentStudentV3(nn.Module):
    def __init__(self,in_dim=80,hidden=384,n_layers=6,n_heads=8):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv1d(in_dim,hidden,5,padding=2),nn.GELU(),nn.Conv1d(hidden,hidden,5,padding=2),nn.GELU())
        self.pos_enc=PositionalEncoding(hidden)
        self.blocks=nn.ModuleList([CTB(hidden,n_heads) for _ in range(n_layers)])
        self.norm=nn.LayerNorm(hidden); self.down=nn.Conv1d(hidden,hidden,3,stride=2,padding=1)
        self.content_head=nn.Conv1d(hidden,768,1)
    def forward(self,x):
        h=self.stem(x); h=h.transpose(1,2); h=self.pos_enc(h)
        for block in self.blocks: h=block(h)
        h=self.norm(h).transpose(1,2); h=self.down(h); return self.content_head(h)

# ── Preload ALL data ──────────────────────────────────────────────────
print("Preloading 4k dataset...")
OUT_DIR="/Users/asill/btrv5/data/mio_4k"; MEL_DIR="/Users/asill/btrv5/data/mio_4k_mel"
meta=np.load(f"{OUT_DIR}/meta.npz"); n=meta['n_samples'].item()
idxs=np.random.RandomState(42).permutation(n); tr=idxs[:int(n*0.85)]; vl=idxs[int(n*0.85):]

all_mel=[]; all_ce=[]
for i in range(n):
    md=np.load(f"{MEL_DIR}/m_{i:05d}.npz"); od=np.load(f"{OUT_DIR}/s_{i:05d}.npz")
    all_mel.append(torch.from_numpy(md['logmel']).float())
    all_ce.append(torch.from_numpy(od['ce_768']).float())
    if i%1000==0: print(f"  {i}/{n}")
print("Preloaded {} samples".format(n))

# ── Training ──────────────────────────────────────────────────────────
model=ContentStudentV3().to(device); model.train()
try: model.load_state_dict(torch.load("checkpoints/causal_student_v3.pt",map_location='cpu'),strict=False)
except: pass
opt=AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
print("Params:",sum(p.numel() for p in model.parameters()),"| Training mini...")

for epoch in range(EPOCHS):
    tr_l=0; nb=0; perm=np.random.permutation(len(tr))
    for i in range(0,len(perm),BATCH):
        bi=perm[i:i+BATCH]; mels=[]; ces=[]
        for j in bi:
            mel=all_mel[tr[j]]; ce=all_ce[tr[j]]
            mels.append(mel); ces.append(ce)
        max_T=max(m.shape[1] for m in mels); max_T25=max(c.shape[0] for c in ces)
        xb=torch.stack([F.pad(m,(0,max_T-m.shape[1])) for m in mels]).to(device)
        yb=torch.stack([F.pad(ce,(0,0,0,max_T25-ce.shape[0])) for ce in ces]).to(device)
        cp=model(xb); Tp=min(cp.shape[2],yb.shape[1])
        ep=cp[:,:,:Tp]; et=yb[:,:Tp,:].transpose(1,2)
        cos=F.cosine_similarity(ep.reshape(ep.shape[0],-1),et.reshape(et.shape[0],-1),dim=1).mean()
        loss=(1-cos)+0.3*F.l1_loss(ep,et)
        opt.zero_grad(); loss.backward(); opt.step(); tr_l+=loss.item(); nb+=1
    
    # Val
    model.eval(); vcos=0; vn=0
    with torch.no_grad():
        for j in vl[:30]:
            mel=all_mel[j].unsqueeze(0).to(device); ce=all_ce[j].unsqueeze(0).to(device)
            cp=model(mel); Tp=min(cp.shape[2],ce.shape[1])
            vcos+=F.cosine_similarity(cp[:,:,:Tp].reshape(1,-1),ce[:,:Tp,:].transpose(1,2).reshape(1,-1),dim=1).mean().item(); vn+=1
    model.train()
    print("  E{:3d} loss={:.4f} val_cos={:.4f}".format(epoch,tr_l/max(nb,1),vcos/max(vn,1)))
    torch.save(model.state_dict(),"checkpoints/causal_student_v3_4k.pt")

print("Done!")
