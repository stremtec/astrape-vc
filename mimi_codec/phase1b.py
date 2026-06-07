"""Phase 1 continued: Longer training + Phase 2 CodePredictor bidirectional."""
import sys,os,random; sys.path.insert(0,'/Users/asill/btrv5')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, subprocess, time
from moshi.models import loaders; from pathlib import Path
from scipy import signal

mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)
SR=24000; STRIDE=1920

def load_any(path,dur=None):
    data,sr=sf.read(path)
    if sr!=SR: data=signal.resample(data,int(len(data)*SR/sr),axis=0)
    if dur is not None: L=dur*SR-(dur*SR%STRIDE); data=data[:L]
    else: L=len(data)-(len(data)%STRIDE); data=data[:L]
    if data.ndim>1: data=data.mean(axis=1)
    return torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)

base='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'
spks=['p225','p226','p227','p228','p229']; utts=['001','002','003','004','005']

# === Fixed Splitter + Converter ===
from mimi_codec.mimi_splitter import KanadeSplitterV4
splitter = KanadeSplitterV4(mimi, bottleneck=64)

class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.down=nn.Conv1d(512,512,kernel_size=4,stride=2,padding=1)
        self.sp=nn.Linear(512,512)
        self.ref=nn.Conv1d(512,512,kernel_size=3,padding=1)
    def forward(self,c,s):
        if c.shape[2]%2!=0: c=F.pad(c,(0,1))
        cz=self.down(c)
        sp=s.reshape(1,-1,1) if s.dim()==1 else s.unsqueeze(-1)
        return cz+self.ref(cz+sp.expand(cz.shape[0],-1,cz.shape[2]))

cv=Conv()
opt=torch.optim.AdamW(list(splitter.parameters())+list(cv.parameters()), lr=8e-4)

# Cache z_q
zq={}
for s in spks:
    for u in utts:
        a=load_any(f'{base}/{s}/{s}_{u}_mic1.flac')
        with torch.no_grad():
            z=mimi.encode_to_latent(a,quantize=False)
            c=mimi.quantizer.encode(z)
            zq[(s,u)]=mimi.quantizer.decode(c)
print(f'Cached {len(zq)} z_q')

pairs=[(s,t,u) for s in spks for t in spks for u in utts if s!=t and (s,u) in zq and (t,u) in zq]
random.shuffle(pairs)
print(f'{len(pairs)} training pairs')

print('Training 100 steps...')
t0=time.time()
for step in range(100):
    random.shuffle(pairs); lt=0; lc=0; ls=0
    for s,t,u in pairs:
        a_s=load_any(f'{base}/{s}/{s}_{u}_mic1.flac')
        a_t=load_any(f'{base}/{t}/{t}_{u}_mic1.flac')
        
        c_s,s_s,_=splitter(a_s); c_t,s_t,_=splitter(a_t)
        z_t=zq[(t,u)]
        
        # Content loss
        T=min(c_s.shape[2],c_t.shape[2])
        loss_c=F.mse_loss(c_s[:,:,:T],c_t[:,:,:T])
        cos_c=F.cosine_similarity(c_s[:,:,:T].reshape(-1),c_t[:,:,:T].reshape(-1),dim=0)
        loss_c+=(1-cos_c)**2*0.5
        
        # Speaker separation
        cos_s=F.cosine_similarity(s_s.squeeze(),s_t.squeeze(),dim=-1).mean()
        loss_s=torch.relu(cos_s-0.1)
        
        # Converter
        zvc=cv(c_s[:,:,:T],s_t.squeeze())
        Tq=min(zvc.shape[2],z_t.shape[2])
        loss=F.mse_loss(zvc[:,:,:Tq],z_t[:,:,:Tq])+0.3*loss_c+0.5*loss_s
        
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(splitter.parameters())+list(cv.parameters()),1.0)
        opt.step()
        lt+=loss.item(); lc+=loss_c.item(); ls+=loss_s.item()
    
    if step%20==0:
        n=len(pairs)
        print(f'  step {step:3d}: loss={lt/n:.4f} c={lc/n:.4f} s={ls/n:.4f} c_cos={cos_c.item():.4f} s_cos={cos_s.item():.4f} [{time.time()-t0:.1f}s]')

print(f'Done [{time.time()-t0:.1f}s]')

# Test
subprocess.run(['ffmpeg','-y','-i','/Users/asill/Downloads/origin.mp3','-ar','24000','-ac','1','-sample_fmt','s16','/tmp/tp2.wav'],capture_output=True)

def convert(sa,ta):
    with torch.no_grad():
        cs,_,_=splitter(sa); _,st,_=splitter(ta)
        zv=cv(cs,st.squeeze()); zu=mimi._to_encoder_framerate(zv)
        if mimi.decoder_transformer: (ztr,)=mimi.decoder_transformer(zu)
        else: ztr=zu
        return mimi.decoder(ztr)

src_a=load_any(f'{base}/p225/p225_001_mic1.flac')
tgt_p=load_any(f'{base}/p226/p226_001_mic1.flac')
tgt_c=load_any('/tmp/tp2.wav',dur=None)

out='/Users/asill/research5'
for nm,ta in[('parallel',tgt_p),('cross',tgt_c)]:
    va=convert(src_a,ta); Tc=min(va.shape[2],src_a.shape[2],ta.shape[2])
    zv=mimi.encode_to_latent(va[:,:,:Tc],quantize=False)
    zs=mimi.encode_to_latent(src_a[:,:,:Tc],quantize=False)
    zt=mimi.encode_to_latent(ta[:,:,:Tc],quantize=False)
    T2=min(zv.shape[2],zs.shape[2],zt.shape[2])
    cs=F.cosine_similarity(zv[:,:,:T2].reshape(-1),zs[:,:,:T2].reshape(-1),dim=0)
    ct=F.cosine_similarity(zv[:,:,:T2].reshape(-1),zt[:,:,:T2].reshape(-1),dim=0)
    print(f'{nm}: cos_src={cs:.4f} cos_tgt={ct:.4f} Δ={ct-cs:+.4f}')
    sf.write(f'{out}/phase1b_{nm}.wav',va[0,0,:Tc].numpy(),SR)
print('✅')
