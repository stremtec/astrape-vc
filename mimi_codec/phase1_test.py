"""Phase 1: Fixed KanadeSplitter + speaker invariance test."""
import sys, os; sys.path.insert(0,'/Users/asill/btrv5')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf
from moshi.models import loaders; from pathlib import Path
from scipy import signal

mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)

# Fixed KanadeSplitterV4
from mimi_codec.mimi_splitter import KanadeSplitterV4
splitter = KanadeSplitterV4(mimi, bottleneck=64)

from resemblyzer import VoiceEncoder
ve = VoiceEncoder()

STRIDE=1920; SR=24000
def load(path, dur=2):
    data,sr=sf.read(path)
    if sr!=SR: data=signal.resample(data,int(len(data)*SR/sr),axis=0)
    L=dur*SR-(dur*SR%STRIDE); data=data[:L]
    if data.ndim>1: data=data.mean(axis=1)
    return torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)

def load_any(path, dur=None):
    data,sr=sf.read(path)
    if sr!=SR: data=signal.resample(data,int(len(data)*SR/sr),axis=0)
    if dur is not None: L=dur*SR-(dur*SR%STRIDE); data=data[:L]
    else: L=len(data)-(len(data)%STRIDE); data=data[:L]
    if data.ndim>1: data=data.mean(axis=1)
    return torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)

base='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'

# === TEST 1: Speaker Invariance ===
# Same speaker, different texts → speaker cos should be HIGH
# Different speakers, same text → speaker cos should be LOW
print('=== TEST 1: Speaker Invariance ===')
spks=['p225','p226','p227']
utts=['001','002','003']

with torch.no_grad():
    for s in spks:
        spk_embs=[]
        for u in utts:
            a=load(f'{base}/{s}/{s}_{u}_mic1.flac')
            _, s_vec, _ = splitter(a)
            spk_embs.append(s_vec.squeeze())
        
        # Same speaker, different texts
        cos_same=[]
        for i in range(len(spk_embs)):
            for j in range(i+1, len(spk_embs)):
                cos_same.append(F.cosine_similarity(spk_embs[i], spk_embs[j], dim=0).item())
        print(f'  {s} same-spk diff-text cos: {sum(cos_same)/len(cos_same):.4f} (want HIGH)')

    # Different speakers, same text
    for u in utts:
        a225=load(f'{base}/p225/p225_{u}_mic1.flac')
        a226=load(f'{base}/p226/p226_{u}_mic1.flac')
        _,s225,_=splitter(a225); _,s226,_=splitter(a226)
        cos_diff=F.cosine_similarity(s225.squeeze(), s226.squeeze(), dim=-1).mean().item()
        print(f'  utt {u} p225↔p226 diff-spk same-text cos: {cos_diff:.4f} (want LOW)')

# === TEST 2: Content Invariance ===
# Same text, different speakers → content cos should be HIGH
print('\
=== TEST 2: Content Invariance ===')
with torch.no_grad():
    for u in utts:
        a225=load(f'{base}/p225/p225_{u}_mic1.flac')
        a226=load(f'{base}/p226/p226_{u}_mic1.flac')
        c225,_,_=splitter(a225)
        c226,_,_=splitter(a226)
        T=min(c225.shape[2],c226.shape[2])
        cos_c=F.cosine_similarity(c225[:,:,:T].reshape(-1),c226[:,:,:T].reshape(-1),dim=0)
        print(f'  utt {u} same-text diff-spk content cos: {cos_c.item():.4f} (want HIGH)')

# === TEST 3: Quick training with FIXED splitter ===
print('\
=== TEST 3: Fixed Splitter Training ===')
# Simple converter
class SimpleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.down=nn.Conv1d(512,512,4,stride=2,padding=1)
        self.sp=nn.Linear(512,512)
        self.ref=nn.Conv1d(512,512,3,padding=1)
    def forward(self,c,s):
        if c.shape[2]%2!=0: c=F.pad(c,(0,1))
        cz=self.down(c)
        # s: (D,) → (1, D, 1)
        sp=s.reshape(1,-1,1) if s.dim()==1 else s.unsqueeze(-1)
        sp_exp=sp.expand(cz.shape[0],-1,cz.shape[2])
        return cz+self.ref(cz+sp_exp)

cv=SimpleConv()
opt=torch.optim.AdamW(list(splitter.parameters())+list(cv.parameters()), lr=5e-4)

src=load(f'{base}/p225/p225_001_mic1.flac')
tgt=load(f'{base}/p226/p226_001_mic1.flac')

# Get z_q for target
with torch.no_grad():
    z_t=mimi.encode_to_latent(tgt,quantize=False)
    c_t=mimi.quantizer.encode(z_t)
    zq_tgt=mimi.quantizer.decode(c_t)

print('Training 30 steps...')
for step in range(30):
    c_s,s_s,_=splitter(src)
    c_t,s_t,_=splitter(tgt)
    
    # Content loss
    T=min(c_s.shape[2],c_t.shape[2])
    loss_c=F.mse_loss(c_s[:,:,:T],c_t[:,:,:T])
    cos_c=F.cosine_similarity(c_s[:,:,:T].reshape(-1),c_t[:,:,:T].reshape(-1),dim=0)
    loss_c+=(1-cos_c)**2*0.5
    
    # Speaker separation
    cos_s=F.cosine_similarity(s_s.squeeze(),s_t.squeeze(),dim=-1).mean()
    loss_s=torch.relu(cos_s-0.2)
    
    # Converter
    zvc=cv(c_s[:,:,:T],s_t.squeeze())
    Tq=min(zvc.shape[2],zq_tgt.shape[2])
    loss=F.mse_loss(zvc[:,:,:Tq],zq_tgt[:,:,:Tq])+0.3*loss_c+0.3*loss_s
    
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(splitter.parameters())+list(cv.parameters()),1.0)
    opt.step()
    
    if step%10==0:
        print(f'  step {step:3d}: loss={loss.item():.4f} c_cos={cos_c.item():.4f} s_cos={cos_s.item():.4f}')

# VC test
print('\
=== VC Test ===')
def vc_convert(sa,ta):
    with torch.no_grad():
        cs,_,_=splitter(sa); _,st,_=splitter(ta)
        zv=cv(cs,st.squeeze()); zu=mimi._to_encoder_framerate(zv)
        if mimi.decoder_transformer: (ztr,)=mimi.decoder_transformer(zu)
        else: ztr=zu
        return mimi.decoder(ztr)

src2=load_any(f'{base}/p225/p225_001_mic1.flac')
tgt_p=load_any(f'{base}/p226/p226_001_mic1.flac')

import subprocess
subprocess.run(['ffmpeg','-y','-i','/Users/asill/Downloads/origin.mp3','-ar','24000','-ac','1','-sample_fmt','s16','/tmp/tp1.wav'],capture_output=True)
tgt_c=load_any('/tmp/tp1.wav',dur=None)

for nm,ta in[('parallel',tgt_p),('cross',tgt_c)]:
    va=vc_convert(src2,ta)
    Tc=min(va.shape[2],src2.shape[2],ta.shape[2])
    zv=mimi.encode_to_latent(va[:,:,:Tc],quantize=False)
    zs=mimi.encode_to_latent(src2[:,:,:Tc],quantize=False)
    zt=mimi.encode_to_latent(ta[:,:,:Tc],quantize=False)
    T2=min(zv.shape[2],zs.shape[2],zt.shape[2])
    cs=F.cosine_similarity(zv[:,:,:T2].reshape(-1),zs[:,:,:T2].reshape(-1),dim=0)
    ct=F.cosine_similarity(zv[:,:,:T2].reshape(-1),zt[:,:,:T2].reshape(-1),dim=0)
    print(f'  {nm}: cos_src={cs:.4f} cos_tgt={ct:.4f} Δ={ct-cs:+.4f}')
    sf.write(f'/Users/asill/research5/phase1_{nm}.wav',va[0,0,:Tc].numpy(),SR)
print('✅')
