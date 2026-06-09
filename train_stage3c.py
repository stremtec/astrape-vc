#!/usr/bin/env python3
"""
Stage 3c: Retrain causal mel decoder with student FSQ-path content.
Decoder learns to handle student content distribution, closing the 0.705→0.80+ gap.
Mix: 60% student content + 40% teacher content → teacher VC mel.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio, numpy as np, os, time, random, warnings
from scipy import signal as scipy_signal
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
warnings.filterwarnings('ignore')

SR=44100; N_MELS=80; MEL_HOP=int(SR/25); BATCH=8; EPOCHS=30
device=torch.device('cpu')

# ── Models ──────────────────────────────────────────────────────────────
class CausalContentStudentV1(nn.Module):
    def __init__(self,in_dim=80,hidden=256,out_dim=5,num_layers=4,kernel=5):
        super().__init__()
        self.proj_in=nn.Conv1d(in_dim,hidden,1)
        layers=[]
        for i in range(num_layers):
            d=2**i; p=(kernel-1)*d
            layers.append(nn.Sequential(nn.Conv1d(hidden,hidden,kernel,dilation=d,padding=p,padding_mode='replicate'),nn.GroupNorm(8,hidden),nn.GELU(),nn.Conv1d(hidden,hidden,1)))
        self.layers=nn.ModuleList(layers)
        self.down=nn.Conv1d(hidden,hidden,3,stride=2,padding=1,padding_mode='replicate')
        self.proj_out=nn.Conv1d(hidden,out_dim,1); self.embed_head=nn.Conv1d(out_dim,768,1)
    def forward(self,x):
        h=self.proj_in(x)
        for layer in self.layers: r=h; h=layer(h); h=h[:,:,:r.shape[2]]; h=h+r
        h=self.down(h); fsq=self.proj_out(h); embed=self.embed_head(fsq)
        return fsq, embed

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
        self.ff=nn.Sequential(nn.Linear(dim,dim*ff_mult),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim*ff_mult,dim),nn.Dropout(dropout))
    def forward(self,x,cond):
        T=x.shape[1]; mask=torch.tril(torch.ones(T,T,device=x.device,dtype=torch.bool))
        xn,gate=self.adaln(x,cond); attn_out=self.attn(xn,xn,xn,attn_mask=~mask,need_weights=False)[0]
        x=x+gate*attn_out; xn2,gate2=self.adaln2(x,cond); ff_out=self.ff(xn2); x=x+gate2*ff_out; return x

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

# ── Load models ────────────────────────────────────────────────────────
content_stu=CausalContentStudentV1()
content_stu.load_state_dict(torch.load("checkpoints/causal_student_v1.pt",map_location='cpu'))
content_stu.eval()
for p in content_stu.parameters(): p.requires_grad=False  # FREEZE

decoder=CausalMelDecoder()
decoder.load_state_dict(torch.load("checkpoints/causal_mel_decoder.pt",map_location='cpu'))  # start from self-recon
decoder.train()

from miocodec.model import MioCodecModel
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2'); teacher.eval()

mel_ext=torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=1024,hop_length=MEL_HOP,n_mels=N_MELS,f_min=80,f_max=14000,center=False,power=1)

def ext_mel(w): w=w-np.mean(w); m=mel_ext(torch.from_numpy(w).float().view(1,1,-1)); return torch.log(m.squeeze(1).clamp(min=1e-5))

# ── Data ──────────────────────────────────────────────────────────────
DATA_DIR="/Users/asill/btrv5/data/mio_teacher"
MEL_DIR="/Users/asill/btrv5/data/mio_mel"
meta=np.load("{}/meta.npz".format(DATA_DIR)); n=len(meta['spk_names']); spk_names=meta['spk_names']
unique_spks=sorted(set(spk_names))
np.random.RandomState(42).shuffle(unique_spks)
tr_spks=unique_spks[:20]

# Pre-compute student FSQ content for training set
print("Pre-computing student FSQ content...")
stu_ce={}
for i in range(n):
    if spk_names[i] not in tr_spks: continue
    mel_d=np.load("{}/mel_{:04d}.npz".format(MEL_DIR,i))
    logmel=torch.from_numpy(mel_d['logmel']).float().unsqueeze(0)
    with torch.inference_mode():
        fsq_pred=content_stu(logmel)[0].squeeze(0).T  # (T,5)
        z_q,_=teacher.local_quantizer.fsq.encode(fsq_pred.unsqueeze(0))
        ce_stu=teacher.local_quantizer.proj_out(z_q).squeeze(0)  # (T,768)
    stu_ce[i]=ce_stu
    if len(stu_ce)%50==0: print("  {}/{}".format(len(stu_ce),sum(1 for s in spk_names if s in tr_spks)))

# Teacher CE + GE cache
tgt_ge={}
for i in range(n):
    d=np.load("{}/sample_{:04d}.npz".format(DATA_DIR,i))
    tgt_ge[i]=torch.from_numpy(d['ge_128']).float()

# Build training pairs
pairs=[]
for i in range(n):
    if spk_names[i] not in tr_spks: continue
    # Self-recon with teacher content
    pairs.append({'src':i,'tgt':i,'use_student':False})
    # VC with student content (60%)
    other_spks=[s for s in tr_spks if s!=spk_names[i]]
    for tgt_spk in random.sample(other_spks,min(3,len(other_spks))):
        tgt_utts=[j for j in range(n) if spk_names[j]==tgt_spk]
        if tgt_utts:
            pairs.append({'src':i,'tgt':random.choice(tgt_utts),'use_student':True})
    # VC with teacher content for stability (40%)
    for tgt_spk in random.sample(other_spks,min(2,len(other_spks))):
        tgt_utts=[j for j in range(n) if spk_names[j]==tgt_spk]
        if tgt_utts:
            pairs.append({'src':i,'tgt':random.choice(tgt_utts),'use_student':False})

random.shuffle(pairs)
print("Train pairs: {} (student={}, teacher={})".format(len(pairs),
    sum(1 for p in pairs if p['use_student']),sum(1 for p in pairs if not p['use_student'])))

# ── Training ────────────────────────────────────────────────────────────
opt=AdamW(decoder.parameters(),lr=5e-4,weight_decay=1e-5)
sched=CosineAnnealingLR(opt,T_max=EPOCHS)

# Get teacher CE
teacher_ce={}
for i in range(n):
    d=np.load("{}/sample_{:04d}.npz".format(DATA_DIR,i))
    teacher_ce[i]=torch.from_numpy(d['ce_768']).float()

print("Training student-aware decoder...")
for epoch in range(EPOCHS):
    random.shuffle(pairs); tr_loss=0; nb=0
    for i in range(0,len(pairs),BATCH):
        batch=pairs[i:i+BATCH]
        ce_list=[]; ge_list=[]; mel_list=[]
        for p in batch:
            if p['use_student']:
                ce=stu_ce[p['src']]
            else:
                ce=teacher_ce[p['src']]
            ge=tgt_ge[p['tgt']]
            # Get VC mel: need to compute via teacher
            d_s=np.load("{}/sample_{:04d}.npz".format(DATA_DIR,p['src']))
            audio=d_s['audio']; alen=len(audio)
            x_s=torch.from_numpy(audio[:SR*3]).float().unsqueeze(0)
            with torch.inference_mode():
                ft_s=teacher.encode(x_s,return_content=True,return_global=False)
                ge_t=tgt_ge[p['tgt']].unsqueeze(0)
                wav=teacher.decode(global_embedding=ge_t.squeeze(0),content_token_indices=ft_s.content_token_indices,target_audio_length=alen)
            mel=ext_mel(wav.numpy())
            ce_list.append(ce); ge_list.append(ge); mel_list.append(mel.squeeze(0))
        
        max_Tc=max(ce.shape[0] for ce in ce_list)
        ce_b=torch.stack([F.pad(ce,(0,0,0,max_Tc-ce.shape[0])) for ce in ce_list]).to(device)
        ge_b=torch.stack(ge_list).to(device)
        max_Tm=max(mel.shape[1] for mel in mel_list)
        mel_b=torch.stack([F.pad(mel,(0,max_Tm-mel.shape[1])) for mel in mel_list]).to(device)
        
        mel_pred=decoder(ce_b,ge_b)
        Tp=min(mel_pred.shape[2],mel_b.shape[2])
        loss=F.l1_loss(mel_pred[:,:,:Tp],mel_b[:,:,:Tp])
        
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.0); opt.step()
        tr_loss+=loss.item(); nb+=1
    sched.step()
    if epoch%10==0 or epoch==EPOCHS-1:
        print("  E{:3d} loss={:.4f}".format(epoch,tr_loss/max(nb,1)))

os.makedirs("checkpoints",exist_ok=True)
torch.save(decoder.state_dict(),"checkpoints/causal_mel_decoder_s3c.pt")

# ── Test ──────────────────────────────────────────────────────────────
print()
print("=== Stage 3c Test ===")
decoder.eval()
for p in decoder.parameters(): p.requires_grad=False

import soundfile as sf; import glob
ROOT="/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"

d_src,sr_s=sf.read(glob.glob("{}/p255/p255_*_mic1.flac".format(ROOT))[0])
if d_src.ndim>1: d_src=d_src.mean(axis=1)
if sr_s!=SR: d_src=scipy_signal.resample(d_src,int(len(d_src)*SR/sr_s))
d_src=d_src[:SR*3]; alen=len(d_src)

d_tgt,sr_t=sf.read("/Users/asill/Downloads/origin.mp3")
if d_tgt.ndim>1: d_tgt=d_tgt.mean(axis=1)
if sr_t!=SR: d_tgt=scipy_signal.resample(d_tgt,int(len(d_tgt)*SR/sr_t))
d_tgt=d_tgt[:SR*3]

import torchaudio
mel_in=torchaudio.transforms.MelSpectrogram(sample_rate=16000,n_fft=512,hop_length=320,n_mels=80,f_min=80,f_max=7600,center=False,power=2)
audio_16k=scipy_signal.resample(d_src[:alen],int(alen*16000/SR))
mel=mel_in(torch.from_numpy(audio_16k).float().view(1,1,-1)).squeeze(1)
logmel=torch.log(mel.clamp(min=1e-5))

x_src=torch.from_numpy(d_src).float().unsqueeze(0)
x_tgt=torch.from_numpy(d_tgt).float().unsqueeze(0)

with torch.inference_mode():
    ft_src=teacher.encode(x_src,return_content=True,return_global=True)
    ft_tgt=teacher.encode(x_tgt,return_content=False,return_global=True)
    ge_tgt2=ft_tgt.global_embedding.unsqueeze(0)

    # Student content
    fsq_s=content_stu(logmel.unsqueeze(0))[0].squeeze(0).T
    z_q,_=teacher.local_quantizer.fsq.encode(fsq_s.unsqueeze(0))
    ce_s=teacher.local_quantizer.proj_out(z_q)

    # Teacher content
    ce_t=ft_src.content_embedding.unsqueeze(0)

    # VC mel target
    wav_vc=teacher.decode(global_embedding=ge_tgt2.squeeze(0),content_token_indices=ft_src.content_token_indices,target_audio_length=alen)
    mel_ref=ext_mel(wav_vc.numpy())

# Compare old decoder vs new decoder
dec_old=CausalMelDecoder(); dec_old.load_state_dict(torch.load("checkpoints/causal_mel_decoder.pt",map_location='cpu')); dec_old.eval()
for p in dec_old.parameters(): p.requires_grad=False

def test(dec,ce,ge,label):
    with torch.inference_mode():
        mel_pred=dec(ce,ge).squeeze(0)
    T=min(mel_pred.shape[1],mel_ref.shape[2])
    cos=F.cosine_similarity(mel_pred[:,:T].flatten(),mel_ref[:,:,:T].flatten(),dim=0).item()
    l1=F.l1_loss(mel_pred[:,:T],mel_ref[:,:,:T]).item()
    print("  {}: Cos={:.4f} L1={:.4f}".format(label,cos,l1))
    return cos

print("  Old decoder (self-recon):")
c_old_stu=test(dec_old,ce_s,ge_tgt2,"stuC+tgtG")
c_old_tea=test(dec_old,ce_t,ge_tgt2,"teaC+tgtG")

print("  New decoder (student-aware):")
c_new_stu=test(decoder,ce_s,ge_tgt2,"stuC+tgtG")
c_new_tea=test(decoder,ce_t,ge_tgt2,"teaC+tgtG")

print()
print("  Student content improvement: {:+.4f} ({:.0f}%→{:.0f}%)".format(
    c_new_stu-c_old_stu,c_old_stu*100,c_new_stu*100))
print("  Goal: 0.705 → 0.80+")
print("Done!")
