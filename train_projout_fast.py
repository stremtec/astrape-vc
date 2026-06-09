#!/usr/bin/env python3
"""Stage 1.6-minimal: StudentProjOut with cosine+L1 only (no mel loss → fast)."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, os, time
from torch.optim import AdamW; from torch.optim.lr_scheduler import CosineAnnealingLR
BATCH=32; EPOCHS=80; device='cpu'

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

class StudentProjOut(nn.Module):
    def __init__(self):
        super().__init__(); self.proj=nn.Conv1d(5,768,1)
    def forward(self,z5): return self.proj(z5)

print("Loading...")
stu=StudentV1(); stu.load_state_dict(torch.load("checkpoints/causal_student_v1.pt",map_location='cpu'),strict=False); stu.eval()
for p in stu.parameters(): p.requires_grad=False

from miocodec.model import MioCodecModel
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2'); teacher.eval()

# Init from teacher proj_out
w=teacher.local_quantizer.proj_out.weight.data.unsqueeze(-1)
b=teacher.local_quantizer.proj_out.bias.data
proj=StudentProjOut().to(device); proj.proj.weight.data.copy_(w); proj.proj.bias.data.copy_(b); proj.train()
print("Initialized from teacher proj_out")

MEL_DIR="/Users/asill/btrv5/data/mio_mel"
DATA_DIR="/Users/asill/btrv5/data/mio_teacher"
meta=np.load("{}/meta.npz".format(DATA_DIR)); n=len(meta['spk_names'])
idxs=np.random.RandomState(42).permutation(n)
tr=idxs[:int(n*0.8)]

def load(idx):
    md=np.load("{}/mel_{:04d}.npz".format(MEL_DIR,idx))
    d=np.load("{}/sample_{:04d}.npz".format(DATA_DIR,idx))
    return (torch.from_numpy(md['logmel']).float(),torch.from_numpy(d['ce_768']).float())

opt=AdamW(proj.parameters(),lr=2e-3,weight_decay=1e-5); sched=CosineAnnealingLR(opt,T_max=EPOCHS)

print("Training StudentProjOut (cos+L1 only)...")
for epoch in range(EPOCHS):
    proj.train(); tl=0; tc=0; nb=0
    perm=np.random.permutation(len(tr))
    for i in range(0,len(perm),BATCH):
        bi=perm[i:i+BATCH]; mels=[]; ces=[]
        for j in bi: mel,ce=load(tr[j]); mels.append(mel); ces.append(ce)
        max_T=max(m.shape[0] for m in mels)
        xb=torch.stack([F.pad(m,(0,max_T-m.shape[1])) for m in mels]).to(device)
        ce_b=torch.stack([F.pad(ce,(0,0,0,max_T-ce.shape[0])) for ce in ces]).to(device)
        with torch.no_grad(): z5=stu(xb).detach().clone()
        emb=proj(z5); Tp=min(emb.shape[2],ce_b.shape[1])
        ep=emb[:,:,:Tp]; et=ce_b[:,:Tp,:].transpose(1,2)
        cos=F.cosine_similarity(ep.reshape(ep.shape[0],-1),et.reshape(et.shape[0],-1),dim=1).mean()
        loss=(1-cos)+0.3*F.l1_loss(ep,et)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(proj.parameters(),1.0); opt.step()
        tl+=loss.item(); tc+=(1-cos).item(); nb+=1
    sched.step()
    if epoch%20==0 or epoch==EPOCHS-1:
        print("  E{:3d} loss={:.4f} cos_loss={:.4f}".format(epoch,tl/max(nb,1),tc/max(nb,1)))

os.makedirs("checkpoints",exist_ok=True); torch.save(proj.state_dict(),"checkpoints/student_proj_out.pt")

# Test
print()
print("=== Test ===")
proj.eval()
import soundfile as sf, glob, torchaudio; from scipy import signal as scipy_signal
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
    fs=teacher.encode(x_s,return_content=True,return_global=True)
    ft=teacher.encode(x_t,return_content=False,return_global=True)
    ge_t=ft.global_embedding; ce_t=fs.content_embedding
    a16=scipy_signal.resample(d_src[:alen],int(alen*16000/SR))
    mel=mel_in(torch.from_numpy(a16).float().view(1,1,-1))
    lm=torch.log(mel.squeeze(1).clamp(min=1e-5))
    z5=stu(lm.unsqueeze(0)); z5_t=z5.squeeze(0).T
    # Hard FSQ
    zq,_=teacher.local_quantizer.fsq.encode(z5_t.unsqueeze(0))
    ce_h=teacher.local_quantizer.proj_out(zq).squeeze(0)
    # StudentProjOut
    ce_s=proj(z5).squeeze(0).T
    wav_t=teacher.decode(global_embedding=ge_t,content_token_indices=fs.content_token_indices,target_audio_length=alen)
    wav_h=teacher.decode(global_embedding=ge_t,content_embedding=ce_h,target_audio_length=alen)
    wav_s=teacher.decode(global_embedding=ge_t,content_embedding=ce_s,target_audio_length=alen)

T=min(ce_t.shape[0],ce_h.shape[0],ce_s.shape[0])
cos_h=F.cosine_similarity(ce_h[:T].flatten(),ce_t[:T].flatten(),dim=0).item()
cos_s=F.cosine_similarity(ce_s[:T].flatten(),ce_t[:T].flatten(),dim=0).item()
fm_h=np.median([F.cosine_similarity(ce_h[i:i+1].flatten(),ce_t[i:i+1].flatten(),dim=0).item() for i in range(T)])
fm_s=np.median([F.cosine_similarity(ce_s[i:i+1].flatten(),ce_t[i:i+1].flatten(),dim=0).item() for i in range(T)])
print("  Hard FSQ:   cos={:.4f} fmed={:.3f}".format(cos_h,fm_h))
print("  StudentProj: cos={:.4f} fmed={:.3f}".format(cos_s,fm_s))
print("  Δ:          {:+.4f}".format(cos_s-cos_h))
sf.write('/Users/asill/Desktop/mio_proj_hard.wav',wav_h.numpy()[:alen],SR)
sf.write('/Users/asill/Desktop/mio_proj_student.wav',wav_s.numpy()[:alen],SR)
sf.write('/Users/asill/Desktop/mio_proj_teacher.wav',wav_t.numpy()[:alen],SR)
print("  Saved: Desktop/mio_proj_*.wav"); print("Done!")
