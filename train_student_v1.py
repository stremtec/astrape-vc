#!/usr/bin/env python3
"""
Causal Content Student v1 — Mel + Transformer/TCN encoder.
Target: MioCodec teacher FSQ 5-dim + token + content embedding.
Architecture: causal logmel (50Hz) → TCN → downsample (25Hz) → FSQ head.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
import numpy as np, os, glob, time, soundfile as sf
from scipy import signal
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

SR=44100; TARGET_SR=16000  # Resample to 16k for mel
HOP_MS=20  # 50Hz frame rate
N_MELS=80
CONTENT_RATE=25  # Hz target
BATCH=32; EPOCHS=80

device=torch.device('cpu')
print("Device:", device)

# ── Causal Mel Frontend ───────────────────────────────────────────────
class CausalMelFrontend(nn.Module):
    """Streaming log-mel extraction at 50Hz, strictly causal."""
    def __init__(self, n_mels=80, hop_ms=20, sr=16000, n_fft=512):
        super().__init__()
        self.hop_samples=int(sr*hop_ms/1000)
        self.n_fft=n_fft
        # Use torchaudio MelSpectrogram with center=False for causal
        self.mel_spec=torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,n_fft=n_fft,hop_length=self.hop_samples,
            n_mels=n_mels,f_min=80,f_max=7600,center=False,power=2)
    
    def forward(self,x):
        # x: (B, C, T_audio) at 16kHz
        if x.shape[1]>1: x=x.mean(dim=1,keepdim=True)
        mel=self.mel_spec(x)  # (B, 1, n_mels, T) — squeeze channel
        mel=mel.squeeze(1)     # (B, n_mels, T)
        logmel=torch.log(mel.clamp(min=1e-5))
        return logmel  # (B, n_mels, T_50hz)

# ── TCN Encoder ────────────────────────────────────────────────────────
class CausalTCNEncoder(nn.Module):
    """Causal temporal conv network: 50Hz → 25Hz."""
    def __init__(self, in_dim=80, hidden=256, out_dim=5, num_layers=4, kernel=5):
        super().__init__()
        self.proj_in=nn.Conv1d(in_dim,hidden,1)
        
        layers=[]
        for i in range(num_layers):
            dilation=2**i
            pad=(kernel-1)*dilation  # causal: pad left only
            layers.append(nn.Sequential(
                nn.Conv1d(hidden,hidden,kernel,dilation=dilation,
                         padding=pad,padding_mode='replicate'),
                nn.GroupNorm(8,hidden),
                nn.GELU(),
                nn.Conv1d(hidden,hidden,1),  # pointwise
            ))
        self.layers=nn.ModuleList(layers)
        
        # Downsample 50Hz→25Hz: stride-2 conv
        self.down=nn.Conv1d(hidden,hidden,3,stride=2,padding=1,padding_mode='replicate')
        
        self.proj_out=nn.Conv1d(hidden,out_dim,1)  # FSQ 5-dim
    
    def forward(self,x):
        # x: (B, M, T_50hz)
        h=self.proj_in(x)
        for layer in self.layers:
            residue=h
            h=layer(h)
            # Trim causal padding from residue to match
            h=h[:,:,:residue.shape[2]]
            h=h+residue  # residual
        h=self.down(h)    # (B, H, T_25hz)
        return self.proj_out(h)  # (B, 5, T_25hz)

# ── Full Student ───────────────────────────────────────────────────────
class CausalContentStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel=CausalMelFrontend()
        self.encoder=CausalTCNEncoder()
        # Optional: content embedding projection
        self.embed_head=nn.Conv1d(5,768,1)  # FSQ 5d → 768d embedding
    
    def forward(self, audio_44k):
        # Resample 44.1k → 16k
        if audio_44k.shape[-1]>0:
            audio_16k=F.interpolate(audio_44k,scale_factor=TARGET_SR/SR,mode='linear')
        else:
            audio_16k=audio_44k
        mel=self.mel(audio_16k)  # (B,80,T_50)
        fsq=self.encoder(mel)     # (B,5,T_25)
        embed=self.embed_head(fsq)  # (B,768,T_25)
        return fsq, embed

# ── Data ──────────────────────────────────────────────────────────────
DATA_DIR="/Users/asill/btrv5/data/mio_teacher"

def load_sample(idx):
    d=np.load("{}/sample_{:04d}.npz".format(DATA_DIR,idx))
    return d['fsq_5d'],d['fsq_tokens'],d['ce_768'],d['audio']

meta=np.load("{}/meta.npz".format(DATA_DIR))
n_samples=len(meta['spk_names'])
idxs=np.random.RandomState(42).permutation(n_samples)
train_idx=idxs[:int(n_samples*0.8)]
val_idx=idxs[int(n_samples*0.8):]
print("Train: {} Val: {}".format(len(train_idx),len(val_idx)))

# ── Training ──────────────────────────────────────────────────────────
model=CausalContentStudent().to(device)
opt=AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5)
sched=CosineAnnealingLR(opt,T_max=EPOCHS)
mse=nn.MSELoss(); l1=nn.L1Loss()

print("Training CausalContentStudent v1 (mel+TCN)...")

HOP_44k=int(SR/CONTENT_RATE)  # 1764

for epoch in range(EPOCHS):
    model.train()
    tr_loss=0; nb=0
    perm=np.random.permutation(len(train_idx))
    
    for i in range(0,len(perm),BATCH):
        batch_idx=perm[i:i+BATCH]
        batch_data=[load_sample(train_idx[j]) for j in batch_idx]
        
        # Stack: need same T_content
        T_max=max(d[0].shape[0] for d in batch_data)
        
        xs=[]; ys=[]; y_lens=[]
        for fsq_5d,_,_,audio in batch_data:
            T=fsq_5d.shape[0]
            alen=T*HOP_44k
            a=audio[:alen] if len(audio)>=alen else np.pad(audio,(0,alen-len(audio)))
            xs.append(torch.from_numpy(a).float())
            ys.append(torch.from_numpy(fsq_5d).float())
            y_lens.append(T)
        
        # Pad to max in batch
        max_audio=max(x.shape[0] for x in xs)
        xs_padded=[F.pad(x,(0,max_audio-x.shape[0])) for x in xs]
        xb=torch.stack(xs_padded).unsqueeze(1).to(device)
        
        max_T=max(y_lens)
        ys_padded=[F.pad(y.T,(0,max_T-y.shape[0])).T for y in ys]  # pad T dim
        yb=torch.stack(ys_padded).transpose(1,2).to(device)  # (B,5,T)
        
        fsq_pred,_=model(xb)
        # Mask loss: only count valid frames
        mask=torch.zeros(yb.shape[0],yb.shape[2],device=device)
        for j,l in enumerate(y_lens):
            mask[j,:l]=1
        diff=(fsq_pred-yb)*mask.unsqueeze(1)
        loss=(diff**2).sum()/(mask.sum()*5+1e-8)
        
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        tr_loss+=loss.item(); nb+=1
    
    sched.step()
    
    # Val
    model.eval()
    vl=0; vn=0
    with torch.no_grad():
        for j in val_idx[:20]:
            fsq_5d,_,_,audio=load_sample(j)
            T=fsq_5d.shape[0]; alen=T*HOP_44k
            a=audio[:alen] if len(audio)>=alen else np.pad(audio,(0,alen-len(audio)))
            x=torch.from_numpy(a).float().view(1,1,-1).to(device)
            y=torch.from_numpy(fsq_5d).float().T.unsqueeze(0).to(device)
            pred,_=model(x)
            Tp=min(pred.shape[2],y.shape[2])
            vl+=mse(pred[:,:,:Tp],y[:,:,:Tp]).item(); vn+=1
    
    if epoch%10==0 or epoch==EPOCHS-1:
        print("  E{:3d} tr={:.4f} val={:.4f}".format(epoch,tr_loss/max(nb,1),vl/max(vn,1)))

# Save
os.makedirs("checkpoints",exist_ok=True)
torch.save(model.state_dict(),"checkpoints/causal_student_v1.pt")

# ── Teacher Decoder Plug-in Test ──────────────────────────────────────
print()
print("=== Teacher Decoder Plug-in Test ===")

from miocodec.model import MioCodecModel
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2')
teacher.eval()

d,sr=sf.read("/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed/p255/p255_001_mic1.flac")
if d.ndim>1: d=d.mean(axis=1)
if sr!=SR: d=signal.resample(d,int(len(d)*SR/sr))
d=d[:SR*3]; audio_len=len(d)

# Teacher encode
x_t=torch.from_numpy(d).float().unsqueeze(0)
with torch.inference_mode():
    ft=teacher.encode(x_t,return_content=True,return_global=True)
    ge=ft.global_embedding
    wav_teacher=teacher.decode(global_embedding=ge,
                              content_token_indices=ft.content_token_indices,
                              target_audio_length=audio_len)

# Student predict
T=ft.content_token_indices.shape[0]
alen=T*HOP_44k
a=d[:alen] if len(d)>=alen else np.pad(d,(0,alen-len(d)))
x_s=torch.from_numpy(a).float().view(1,1,-1).to(device)
with torch.inference_mode():
    fsq_pred,embed_pred=model(x_s)
    fsq_t=fsq_pred.squeeze(0).T  # (T,5)
    # Quantize through teacher FSQ
    z_q,_=teacher.local_quantizer.fsq.encode(fsq_t.unsqueeze(0))
    z_q=teacher.local_quantizer.proj_out(z_q)  # (1,T,768)
    # Decode with student content + teacher global
    wav_stu=teacher.decode(global_embedding=ge,
                          content_embedding=z_q.squeeze(0),
                          target_audio_length=audio_len)

# Metrics
from scipy.signal import stft
def measure(a):
    a=a-np.mean(a)
    f,_,Z=stft(a,fs=SR,nperseg=1024,noverlap=768)
    mag=np.abs(Z); total=mag.sum()+1e-8
    c=np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2],axis=0)
    c/=(mag[:len(f)//2].sum(axis=0)+1e-8)
    return np.mean(c)

wt=wav_teacher.cpu().numpy()[:audio_len]
ws=wav_stu.cpu().numpy()[:audio_len]
ct=measure(wt); cs=measure(ws)
print("  Teacher self-recon centroid: {:.0f}Hz".format(ct))
print("  Student→Teacher centroid:    {:.0f}Hz".format(cs))
print("  Delta: {:.0f}Hz".format(cs-ct))

# Token match
t_tokens=ft.content_token_indices.numpy()
# Get student tokens
stu_tokens=teacher.local_quantizer.fsq.codes_to_indices(z_q).squeeze(0).numpy()
match=(t_tokens[:len(stu_tokens)]==stu_tokens[:len(t_tokens)]).mean()*100
print("  Token match rate: {:.1f}%".format(match))

sf.write('/Users/asill/Desktop/mio_student_v1.wav',ws,SR)
sf.write('/Users/asill/Desktop/mio_teacher_ref.wav',wt,SR)
print("  Saved: Desktop/mio_student_v1.wav")

# Latency
t0=time.time()
with torch.inference_mode():
    _=model(x_s)
per_frame=(time.time()-t0)/T*1000
print("  Latency per content frame (40ms): {:.1f}ms".format(per_frame))
print()
print("Done!")
