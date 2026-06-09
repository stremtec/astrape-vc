#!/usr/bin/env python3
"""
Stage 1.6: StudentProjOut — calibrate student 5D→768d to decoder-compatible space.
Freeze content encoder, train only projection layer with cosine + L1 + mel loss.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio, numpy as np, os, time
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

BATCH=32; EPOCHS=60
device=torch.device('cpu')

# ── Content Student (frozen) ──────────────────────────────────────────
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
    def forward(self,x):
        h=self.proj_in(x)
        for layer in self.layers: r=h; h=layer(h); h=h[:,:,:r.shape[2]]; h=h+r
        h=self.down(h); return self.proj_out(h)

# ── StudentProjOut (trainable) ────────────────────────────────────────
class StudentProjOut(nn.Module):
    """5-dim → 768d projection, initialized from teacher proj_out (Linear→Conv1d reshaped)."""
    def __init__(self,teacher_proj_out_state=None):
        super().__init__()
        self.proj=nn.Conv1d(5,768,1)
        if teacher_proj_out_state is not None:
            # teacher proj_out is Linear(5,768): weight=(768,5), bias=(768,)
            # Conv1d expects: weight=(768,5,1), bias=(768,)
            w=teacher_proj_out_state['weight'].unsqueeze(-1)  # (768,5,1)
            self.proj.weight.data.copy_(w)
            self.proj.bias.data.copy_(teacher_proj_out_state['bias'])
            print("  Initialized from teacher proj_out")
    
    def forward(self,z5):
        return self.proj(z5)  # (B,5,T)→(B,768,T)

# ── Load ──────────────────────────────────────────────────────────────
print("Loading models...")
content_stu=StudentV1()
content_stu.load_state_dict(torch.load("checkpoints/causal_student_v1.pt",map_location='cpu'),strict=False)
content_stu.eval()
for p in content_stu.parameters(): p.requires_grad=False  # FREEZE

from miocodec.model import MioCodecModel
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2'); teacher.eval()

# Get teacher proj_out weights
teacher_proj_state={'weight':teacher.local_quantizer.proj_out.weight.data.clone(),
                     'bias':teacher.local_quantizer.proj_out.bias.data.clone()}
proj_out=StudentProjOut(teacher_proj_state).to(device); proj_out.train()

# Also load causal mel decoder for mel loss
from train_stage3aG import CausalMelDecoder
mel_dec=CausalMelDecoder(); mel_dec.load_state_dict(torch.load("checkpoints/causal_mel_decoder.pt",map_location='cpu')); mel_dec.eval()
for p in mel_dec.parameters(): p.requires_grad=False

# ── Data ──────────────────────────────────────────────────────────────
MEL_DIR="/Users/asill/btrv5/data/mio_mel"
DATA_DIR="/Users/asill/btrv5/data/mio_teacher"
meta=np.load("{}/meta.npz".format(DATA_DIR)); n=len(meta['spk_names'])
idxs=np.random.RandomState(42).permutation(n)
tr=idxs[:int(n*0.8)]; vl=idxs[int(n*0.8):]

def load_data(idx):
    mel_d=np.load("{}/mel_{:04d}.npz".format(MEL_DIR,idx))
    d=np.load("{}/sample_{:04d}.npz".format(DATA_DIR,idx))
    return (torch.from_numpy(mel_d['logmel']).float(),
            torch.from_numpy(d['ce_768']).float(),
            torch.from_numpy(d['ge_128']).float())

# Pre-compute teacher mels for mel loss
print("Caching teacher mels...")
teacher_mels={}
mel_ext=torchaudio.transforms.MelSpectrogram(sample_rate=44100,n_fft=1024,hop_length=1764,n_mels=80,f_min=80,f_max=14000,center=False,power=1)
for i in range(n):
    d=np.load("{}/sample_{:04d}.npz".format(DATA_DIR,i))
    audio=d['audio']; w=audio-np.mean(audio)
    mel=mel_ext(torch.from_numpy(w).float().view(1,1,-1))
    teacher_mels[i]=torch.log(mel.squeeze(1).clamp(min=1e-5)).squeeze(0)  # (n_mels,T)
    if i%50==0: print("  {}/{}".format(i,n))

# ── Training ────────────────────────────────────────────────────────────
opt=AdamW(proj_out.parameters(),lr=1e-3,weight_decay=1e-5)
sched=CosineAnnealingLR(opt,T_max=EPOCHS)

print("Training StudentProjOut (content encoder frozen)...")
for epoch in range(EPOCHS):
    proj_out.train(); tr_l=0; tr_c=0; tr_l1=0; tr_mel=0; nb=0
    perm=np.random.permutation(len(tr))
    
    for i in range(0,len(perm),BATCH):
        bi=perm[i:i+BATCH]
        mels=[]; ces=[]; ges=[]
        for j in bi:
            mel,ce,ge=load_data(tr[j]); mels.append(mel); ces.append(ce); ges.append(ge)
        
        max_T=max(m.shape[0] for m in mels)  # T_content
        xb=torch.stack([F.pad(m,(0,max_T-m.shape[1])) for m in mels]).to(device)
        ce_b=torch.stack([F.pad(ce,(0,0,0,max_T-ce.shape[0])) for ce in ces]).to(device)
        ge_b=torch.stack(ges).to(device)
        
        # Student FSQ 5d
        with torch.inference_mode():
            z5=content_stu(xb)  # (B,5,T)
        # StudentProjOut → 768d
        emb_s=proj_out(z5)  # (B,768,T)
        
        # Align: trim to match
        Tp=min(emb_s.shape[2],ce_b.shape[1])
        emb_p=emb_s[:,:,:Tp]; emb_t=ce_b[:,:Tp,:].transpose(1,2)
        
        # Cosine loss
        cos=F.cosine_similarity(emb_p.reshape(emb_p.shape[0],-1),emb_t.reshape(emb_t.shape[0],-1),dim=1).mean()
        L_cos=1-cos
        
        # L1 loss
        L_l1=F.l1_loss(emb_p,emb_t)
        
        # Mel loss (decode + compare to teacher mel)
        mel_pred=mel_dec(emb_p.transpose(1,2),ge_b)  # (B,n_mels,T)
        # Get teacher mels
        mel_tgts=[]
        for j in bi:
            mt=teacher_mels[tr[j]]
            mel_tgts.append(mt)
        max_Tm=max(mt.shape[1] for mt in mel_tgts)
        mel_t=torch.stack([F.pad(mt,(0,max_Tm-mt.shape[1])) for mt in mel_tgts]).to(device)
        Tmp=min(mel_pred.shape[2],mel_t.shape[2])
        L_mel=F.l1_loss(mel_pred[:,:,:Tmp],mel_t[:,:,:Tmp])
        
        loss=L_cos+0.3*L_l1+0.1*L_mel
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(proj_out.parameters(),1.0); opt.step()
        tr_l+=loss.item(); tr_c+=L_cos.item(); tr_l1+=L_l1.item(); tr_mel+=L_mel.item(); nb+=1
    sched.step()
    
    if epoch%10==0 or epoch==EPOCHS-1:
        print("  E{:3d} loss={:.4f} cos={:.4f} l1={:.4f} mel={:.4f}".format(
            epoch,tr_l/max(nb,1),tr_c/max(nb,1),tr_l1/max(nb,1),tr_mel/max(nb,1)))

os.makedirs("checkpoints",exist_ok=True)
torch.save(proj_out.state_dict(),"checkpoints/student_proj_out.pt")

# ── Test ──────────────────────────────────────────────────────────────
print()
print("=== StudentProjOut Test ===")
proj_out.eval()

import soundfile as sf, glob; from scipy import signal as scipy_signal
SR=44100
mel_in=torchaudio.transforms.MelSpectrogram(sample_rate=16000,n_fft=512,hop_length=320,n_mels=80,f_min=80,f_max=7600,center=False,power=2)

d_src,sr=sf.read(glob.glob("/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed/p255/p255_*_mic1.flac")[0])
if d_src.ndim>1: d_src=d_src.mean(axis=1)
if sr!=SR: d_src=scipy_signal.resample(d_src,int(len(d_src)*SR/sr))
d_src=d_src[:SR*3]; alen=len(d_src)

d_tgt,sr=sf.read("/Users/asill/Downloads/origin.mp3")
if d_tgt.ndim>1: d_tgt=d_tgt.mean(axis=1)
if sr!=SR: d_tgt=scipy_signal.resample(d_tgt,int(len(d_tgt)*SR/sr))

x_s=torch.from_numpy(d_src).float().unsqueeze(0); x_t=torch.from_numpy(d_tgt[:SR*3]).float().unsqueeze(0)

with torch.inference_mode():
    ft_s=teacher.encode(x_s,return_content=True,return_global=True)
    ft_t=teacher.encode(x_t,return_content=False,return_global=True)
    ge_t=ft_t.global_embedding; ce_teacher=ft_s.content_embedding
    
    # Student paths
    a16=scipy_signal.resample(d_src[:alen],int(alen*16000/SR))
    mel=mel_in(torch.from_numpy(a16).float().view(1,1,-1))
    lm=torch.log(mel.squeeze(1).clamp(min=1e-5))
    z5=content_stu(lm.unsqueeze(0))
    
    # Hard FSQ
    z5_t=z5.squeeze(0).T
    z_q,_=teacher.local_quantizer.fsq.encode(z5_t.unsqueeze(0))
    ce_hard=teacher.local_quantizer.proj_out(z_q).squeeze(0)
    
    # StudentProjOut
    ce_stu=proj_out(z5).squeeze(0).T  # (T,768)
    
    # Decode
    wav_teacher=teacher.decode(global_embedding=ge_t,content_token_indices=ft_s.content_token_indices,target_audio_length=alen)
    wav_hard=teacher.decode(global_embedding=ge_t,content_embedding=ce_hard,target_audio_length=alen)
    wav_stu=teacher.decode(global_embedding=ge_t,content_embedding=ce_stu,target_audio_length=alen)

T=min(ce_teacher.shape[0],ce_hard.shape[0],ce_stu.shape[0])
cos_hard=F.cosine_similarity(ce_hard[:T].flatten(),ce_teacher[:T].flatten(),dim=0).item()
cos_stu=F.cosine_similarity(ce_stu[:T].flatten(),ce_teacher[:T].flatten(),dim=0).item()

fcos_hard=[F.cosine_similarity(ce_hard[i:i+1].flatten(),ce_teacher[i:i+1].flatten(),dim=0).item() for i in range(T)]
fcos_stu=[F.cosine_similarity(ce_stu[i:i+1].flatten(),ce_teacher[i:i+1].flatten(),dim=0).item() for i in range(T)]

print("  Content cos: Hard={:.4f}  StudentProjOut={:.4f}  Δ={:+.4f}".format(cos_hard,cos_stu,cos_stu-cos_hard))
print("  Frame median: Hard={:.3f}  StudentProjOut={:.3f}".format(np.median(fcos_hard),np.median(fcos_stu)))

sf.write('/Users/asill/Desktop/mio_stuproj_hard.wav',wav_hard.numpy()[:alen],SR)
sf.write('/Users/asill/Desktop/mio_stuproj_student.wav',wav_stu.numpy()[:alen],SR)
sf.write('/Users/asill/Desktop/mio_stuproj_teacher.wav',wav_teacher.numpy()[:alen],SR)
print("  Saved: Desktop/mio_stuproj_*.wav")
print("Done!")
