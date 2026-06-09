#!/usr/bin/env python3
"""
Content Student v3: 384dim Transformer, 1090 samples, decoder-aware loss.
Target: teacher content embedding + decoder mel consistency.
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, os, time, math
from torch.optim import AdamW; from torch.optim.lr_scheduler import CosineAnnealingLR

BATCH=4; EPOCHS=100; device='cpu'

# ── Models ──────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self,dim,max_len=2000):
        super().__init__(); pe=torch.zeros(max_len,dim)
        pos=torch.arange(0,max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,dim,2).float()*(-math.log(10000.0)/dim))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): T=x.size(1); return x+self.pe[:,:T,:].contiguous()
class CausalTransformerBlock(nn.Module):
    def __init__(self,dim=384,n_heads=8,ff_mult=4,dropout=0.1):
        super().__init__()
        self.norm1=nn.LayerNorm(dim); self.attn=nn.MultiheadAttention(dim,n_heads,dropout=dropout,batch_first=True)
        self.norm2=nn.LayerNorm(dim); self.ff=nn.Sequential(nn.Linear(dim,dim*ff_mult),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim*ff_mult,dim),nn.Dropout(dropout))
    def forward(self,x):
        T=x.shape[1]; mask=torch.tril(torch.ones(T,T,device=x.device,dtype=torch.bool))
        xn=self.norm1(x); a=self.attn(xn,xn,xn,attn_mask=~mask,need_weights=False)[0]; x=x+a
        xn=self.norm2(x); x=x+self.ff(xn); return x
class ContentStudentV3(nn.Module):
    def __init__(self,in_dim=80,hidden=384,n_layers=6,n_heads=8,kernel=5):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv1d(in_dim,hidden,kernel,padding=kernel//2),nn.GELU(),nn.Conv1d(hidden,hidden,kernel,padding=kernel//2),nn.GELU())
        self.pos_enc=PositionalEncoding(hidden)
        self.blocks=nn.ModuleList([CausalTransformerBlock(hidden,n_heads) for _ in range(n_layers)])
        self.norm=nn.LayerNorm(hidden); self.down=nn.Conv1d(hidden,hidden,3,stride=2,padding=1)
        self.content_head=nn.Conv1d(hidden,768,1); self.prefsq_head=nn.Conv1d(hidden,768,1)
    def forward(self,x):
        h=self.stem(x); h=h.transpose(1,2); h=self.pos_enc(h)
        for block in self.blocks: h=block(h)
        h=self.norm(h).transpose(1,2); h=self.down(h)
        return self.content_head(h),self.prefsq_head(h)

# ── Decoder (frozen) ───────────────────────────────────────────────────
class AdaLNZero(nn.Module):
    def __init__(self,dim,cond_dim,eps=1e-5):
        super().__init__()
        self.norm=nn.LayerNorm(dim,eps=eps,elementwise_affine=False)
        self.proj=nn.Sequential(nn.SiLU(),nn.Linear(cond_dim,3*dim))
        nn.init.zeros_(self.proj[1].weight); nn.init.zeros_(self.proj[1].bias)
    def forward(self,x,cond):
        xn=self.norm(x); shift,scale,gate=self.proj(cond).chunk(3,dim=-1)
        return xn*(1+scale)+shift,gate
class CDB(nn.Module):
    def __init__(self,dim=512,cond_dim=128,n_heads=8,ff_mult=4,dropout=0.1):
        super().__init__()
        self.adaln=AdaLNZero(dim,cond_dim); self.adaln2=AdaLNZero(dim,cond_dim)
        self.attn=nn.MultiheadAttention(dim,n_heads,dropout=dropout,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(dim,dim*ff_mult),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim*ff_mult,dim),nn.Dropout(dropout))
    def forward(self,x,cond):
        T=x.shape[1]; mask=torch.tril(torch.ones(T,T,device=x.device,dtype=torch.bool))
        xn,gate=self.adaln(x,cond); a=self.attn(xn,xn,xn,attn_mask=~mask,need_weights=False)[0]; x=x+gate*a
        xn2,gate2=self.adaln2(x,cond); x=x+gate2*self.ff(xn2); return x
class CausalMelDecoder(nn.Module):
    def __init__(self,cd=768,cond_dim=128,hidden=512,n_layers=4,n_heads=8,n_mels=80):
        super().__init__()
        self.proj_in=nn.Linear(cd,hidden)
        self.blocks=nn.ModuleList([CDB(hidden,cond_dim,n_heads) for _ in range(n_layers)])
        self.norm_out=nn.LayerNorm(hidden); self.proj_out=nn.Linear(hidden,n_mels)
    def forward(self,ce,ge):
        x=self.proj_in(ce); cond=ge.unsqueeze(1)
        for b in self.blocks: x=b(x,cond)
        x=self.norm_out(x); return self.proj_out(x).transpose(1,2)

# ── Load Data ───────────────────────────────────────────────────────────
MEL_DIR="/Users/asill/btrv5/data/mio_large_mel"
OUT_DIR="/Users/asill/btrv5/data/mio_large"
meta=np.load(f"{OUT_DIR}/meta.npz"); n=meta['n_samples'].item()
spk_names=meta['spk_names'][:n]
idxs=np.random.RandomState(42).permutation(n)
tr=idxs[:int(n*0.85)]; vl=idxs[int(n*0.85):]
print("Train: {} Val: {} ({} spk)".format(len(tr),len(vl),len(set(spk_names))))

def load(idx):
    md=np.load(f"{MEL_DIR}/mel_{idx:05d}.npz")
    od=np.load(f"{OUT_DIR}/sample_{idx:05d}.npz")
    return (torch.from_numpy(md['logmel']).float(),torch.from_numpy(od['ce_768']).float(),
            torch.from_numpy(od['pre_fsq_768']).float(),torch.from_numpy(od['ge_128']).float())

# ── Teacher mels: skip for v3-fast (data volume priority) ──────────────
teacher_mels={}
decoder=None

# ── Model ──────────────────────────────────────────────────────────────
model=ContentStudentV3().to(device); model.train()
try: model.load_state_dict(torch.load("checkpoints/causal_student_v2_final.pt",map_location='cpu'),strict=False)
except: pass

opt=AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5); sched=CosineAnnealingLR(opt,T_max=EPOCHS)
print("Params:",sum(p.numel() for p in model.parameters()),"| Training v3...")

for epoch in range(EPOCHS):
    tr_l=0; nb=0; perm=np.random.permutation(len(tr))
    for i in range(0,len(perm),BATCH):
        bi=perm[i:i+BATCH]
        mels=[]; ces=[]; pfs=[]; ges=[]; mel_tgts=[]
        for j in bi:
            mel,ce,pf,ge=load(tr[j]); mels.append(mel); ces.append(ce); pfs.append(pf); ges.append(ge)
            if j in teacher_mels: mel_tgts.append((j,teacher_mels[j]))
        
        max_T50=max(m.shape[1] for m in mels); max_T25=max(c.shape[0] for c in ces)
        xb=torch.stack([F.pad(m,(0,max_T50-m.shape[1])) for m in mels]).to(device)
        ce_b=torch.stack([F.pad(ce,(0,0,0,max_T25-ce.shape[0])) for ce in ces]).to(device)
        pf_b=torch.stack([F.pad(pf,(0,0,0,max_T25-pf.shape[0])) for pf in pfs]).to(device)
        ge_b=torch.stack(ges).to(device)
        
        ce_p,pf_p=model(xb); Tp=min(ce_p.shape[2],ce_b.shape[1])
        
        # Content embedding loss
        ep=ce_p[:,:,:Tp]; et=ce_b[:,:Tp,:].transpose(1,2)
        cos=F.cosine_similarity(ep.reshape(ep.shape[0],-1),et.reshape(et.shape[0],-1),dim=1).mean()
        L_ce=(1-cos)+0.3*F.l1_loss(ep,et)
        
        # Pre-FSQ loss
        pp=pf_p[:,:,:Tp]; pt=pf_b[:,:Tp,:].transpose(1,2)
        cos_pf=F.cosine_similarity(pp.reshape(pp.shape[0],-1),pt.reshape(pt.shape[0],-1),dim=1).mean()
        L_pf=(1-cos_pf)*0.3
        
        # Decoder-aware loss: SKIP for speed — data volume is the priority
        L_dec=0
        
        loss=L_ce+L_pf+L_dec
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tr_l+=loss.item(); nb+=1
    sched.step()
    
    if epoch%20==0 or epoch==EPOCHS-1:
        model.eval(); vcos=0; vn=0
        with torch.no_grad():
            for j in vl[:20]:
                mel,ce,_,_=load(j); mel=mel.unsqueeze(0).to(device); ce=ce.unsqueeze(0).to(device)
                cp,_=model(mel); Tp=min(cp.shape[2],ce.shape[1])
                ep=cp[:,:,:Tp]; et=ce[:,:Tp,:].transpose(1,2)
                vcos+=F.cosine_similarity(ep.reshape(1,-1),et.reshape(1,-1),dim=1).mean().item(); vn+=1
        model.train()
        print("  E{:3d} loss={:.4f} val_cos={:.4f}".format(epoch,tr_l/max(nb,1),vcos/max(vn,1)))

os.makedirs("checkpoints",exist_ok=True); torch.save(model.state_dict(),"checkpoints/causal_student_v3.pt")
print("Saved: checkpoints/causal_student_v3.pt")
print("Done!")
