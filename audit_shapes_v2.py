#!/usr/bin/env python3
"""Shape audit + offset sweep for CausalContentStudent pipeline."""
import torch, torch.nn as nn, numpy as np, os
import torch.nn.functional as F

# ── Models ──────────────────────────────────────────────────────────────
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

print("=== SHAPE AUDIT ===")
MEL_DIR="/Users/asill/btrv5/data/mio_mel"
DATA_DIR="/Users/asill/btrv5/data/mio_teacher"

# Load one sample
mel_d=np.load("{}/mel_0000.npz".format(MEL_DIR))
d=np.load("{}/sample_0000.npz".format(DATA_DIR))

logmel=torch.from_numpy(mel_d['logmel']).float().unsqueeze(0)  # (1,80,T50)
ce_768=torch.from_numpy(d['ce_768']).float()  # (T25,768)
fsq_5d=torch.from_numpy(mel_d['fsq_5d']).float()  # (T25,5)

print("  logmel (cached):  {}".format(list(logmel.shape)))
print("  ce_768 (teacher): {}".format(list(ce_768.shape)))
print("  fsq_5d (teacher): {}".format(list(fsq_5d.shape)))

# Student forward
stu=StudentV1(); stu.load_state_dict(torch.load("checkpoints/causal_student_v1.pt",map_location='cpu'),strict=False); stu.eval()
with torch.no_grad():
    z5=stu(logmel)  # (1,5,T25_stu)

print("  z5 (student):     {}".format(list(z5.shape)))
print()

# Check alignment
T_teacher=ce_768.shape[0]  # teacher content frames
T_student=z5.shape[2]       # student content frames
print("  Teacher content frames (T25): {}".format(T_teacher))
print("  Student content frames (T25): {}".format(T_student))
print("  Mismatch: {}".format(T_teacher-T_student))
print()

# Per-frame alignment
print("  Input alignment:")
audio_len=len(d['audio'])
mel_len_50hz=logmel.shape[2]
expected_T25=mel_len_50hz//2  # stride-2 downsample
print("    Audio samples: {} → mel 50Hz: {} → expected 25Hz: {}".format(audio_len,mel_len_50hz,expected_T25))
print("    Teacher T25: {}".format(T_teacher))
print("    Student T25: {}".format(T_student))
print("    Offset: teacher-student = {}".format(T_teacher-T_student))
print()

# ── Offset Sweep ────────────────────────────────────────────────────────
print("=== OFFSET SWEEP ===")
from miocodec.model import MioCodecModel
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2'); teacher.eval()

with torch.no_grad():
    fsq_s=z5.squeeze(0).T  # (T_stu,5)
    zq,_=teacher.local_quantizer.fsq.encode(fsq_s.unsqueeze(0))
    ce_stu=teacher.local_quantizer.proj_out(zq).squeeze(0)  # (T_stu,768)

T=min(ce_stu.shape[0],ce_768.shape[0])

print("  Offset  Cos(full)  Cos(frame_med)  Best?")
print("  "+"-"*50)
best_cos=0; best_off=0
for off in range(-3,4):
    if off<0:
        s_slice=ce_stu[-off:T]
        t_slice=ce_768[:T+off]
    elif off>0:
        s_slice=ce_stu[:T-off]
        t_slice=ce_768[off:T]
    else:
        s_slice=ce_stu[:T]; t_slice=ce_768[:T]
    
    if s_slice.shape[0]<2: continue
    cos_full=F.cosine_similarity(s_slice.flatten(),t_slice.flatten(),dim=0).item()
    fcos=[F.cosine_similarity(s_slice[i:i+1].flatten(),t_slice[i:i+1].flatten(),dim=0).item() for i in range(s_slice.shape[0])]
    fmed=np.median(fcos)
    best=" ★" if cos_full>best_cos else ""
    if cos_full>best_cos: best_cos=cos_full; best_off=off
    print("  {:5d}   {:.4f}       {:.4f}        {}".format(off,cos_full,fmed,best))

print()
print("  Best offset: {} (cos={:.4f})".format(best_off,best_cos))
if best_off!=0:
    print("  WARNING: Non-zero best offset → likely alignment bug!")
else:
    print("  OK: Best at offset 0")

# ── Shape chain ─────────────────────────────────────────────────────────
print()
print("=== SHAPE CHAIN ===")
print("  audio → mel(50Hz): [B,80,T50] → student z5: [B,5,T25] → proj → [B,T25,768]")
print("  teacher content: [T25,768]")
print("  Decoder expects: [B,T25,768] for content, [B,128] for global")
print()

# Check mel→student dim consistency
print("  Mel batch check (5 samples):")
for i in [0,1,10,50,100]:
    md=np.load("{}/mel_{:04d}.npz".format(MEL_DIR,i))
    lm=torch.from_numpy(md['logmel']).float()  # (80,T50)
    dd=np.load("{}/sample_{:04d}.npz".format(DATA_DIR,i))
    ce=torch.from_numpy(dd['ce_768']).float()  # (T25,768)
    with torch.no_grad():
        z5_i=stu(lm.unsqueeze(0))
    Tt=ce.shape[0]; Ts=z5_i.shape[2]
    ok="OK" if abs(Tt-Ts)<=2 else "MISMATCH"
    print("    {}: mel={} teacher_T={} student_T={} diff={} {}".format(i,list(lm.shape),Tt,Ts,Tt-Ts,ok))

print()
print("Done!")
