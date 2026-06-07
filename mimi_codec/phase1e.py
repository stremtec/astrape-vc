"""Phase 1e: Final training + VC test with correct architecture."""
import sys,os,random; sys.path.insert(0,'/Users/asill/btrv5')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, subprocess, time
from moshi.models import loaders; from pathlib import Path
from scipy import signal
mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)
SR=24000; STRIDE=1920

class LightSplitter(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_bn=nn.Sequential(nn.Conv1d(512,64,1),nn.GELU(),nn.Conv1d(64,512,1))
        self.s_net=nn.Sequential(nn.Conv1d(512,256,5,padding=2),nn.GELU(),nn.Conv1d(256,512,5,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(512,512))
    def forward(self,fs,fd):
        return fs+self.c_bn(fs), self.s_net(fd)

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

def load_any(path,dur=None):
    data,sr=sf.read(path)
    if sr!=SR: data=signal.resample(data,int(len(data)*SR/sr),axis=0)
    if dur is not None: L=dur*SR-(dur*SR%STRIDE); data=data[:L]
    else: L=len(data)-(len(data)%STRIDE); data=data[:L]
    if data.ndim>1: data=data.mean(axis=1)
    return torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)

base='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'
spks=['p225','p226','p227','p228','p229']; utts=['001','002','003','004','005']

# Cache raw features
print('Caching...')
raw={}
with torch.no_grad():
    for s in spks:
        for u in utts:
            a=load_any(f'{base}/{s}/{s}_{u}_mic1.flac')
            enc=mimi.encoder(a); h=enc.transpose(1,2)
            tt=mimi.encoder_transformer.transformer; sh=[]; dp=[]
            for i,layer in enumerate(tt.layers):
                h=layer(h)
                if i in[0,1,2]: sh.append(h)
                if i in[5,6,7]: dp.append(h)
            raw[(s,u)]=(torch.stack(sh,0).mean(0).transpose(1,2),
                        torch.stack(dp,0).mean(0).transpose(1,2),
                        mimi.quantizer.decode(mimi.quantizer.encode(mimi.encode_to_latent(a,quantize=False))))
print(f'Cached {len(raw)}')

sp=LightSplitter(); cv=Conv()
opt=torch.optim.AdamW(list(sp.parameters())+list(cv.parameters()), lr=1e-3)
pairs=[(s,t,u) for s in spks for t in spks for u in utts if s!=t]
random.shuffle(pairs)

t0=time.time()
for step in range(120):
    random.shuffle(pairs); lt=0
    for s,t,u in pairs:
        fs_s,fd_s,_=raw[(s,u)]; fs_t,fd_t,zq_t=raw[(t,u)]
        c_s,s_s=sp(fs_s,fd_s); c_t,s_t=sp(fs_t,fd_t)
        T=min(c_s.shape[2],c_t.shape[2])
        loss_c=F.mse_loss(c_s[:,:,:T],c_t[:,:,:T])
        cos_c=F.cosine_similarity(c_s[:,:,:T].reshape(-1),c_t[:,:,:T].reshape(-1),dim=0)
        loss_c+=(1-cos_c)**2*0.5
        cos_s=F.cosine_similarity(s_s,s_t,dim=-1).mean()
        loss_s=torch.relu(cos_s-0.1)
        zvc=cv(c_s[:,:,:T],s_t); Tq=min(zvc.shape[2],zq_t.shape[2])
        loss=F.mse_loss(zvc[:,:,:Tq],zq_t[:,:,:Tq])+0.3*loss_c+0.5*loss_s
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(sp.parameters())+list(cv.parameters()),1.0); opt.step()
        lt+=loss.item()
    if step%30==0:
        n=len(pairs)
        print(f'  step {step:3d}: loss={lt/n:.4f} c_cos={cos_c.item():.4f} s_cos={cos_s.item():.4f} [{time.time()-t0:.1f}s]')
print(f'Done [{time.time()-t0:.1f}s]')

# VC test
def extract_features(audio):
    with torch.no_grad():
        enc=mimi.encoder(audio); h=enc.transpose(1,2)
        tt=mimi.encoder_transformer.transformer; sh=[]; dp=[]
        for i,layer in enumerate(tt.layers):
            h=layer(h)
            if i in[0,1,2]: sh.append(h)
            if i in[5,6,7]: dp.append(h)
        return torch.stack(sh,0).mean(0).transpose(1,2), torch.stack(dp,0).mean(0).transpose(1,2)

def convert(sa,ta):
    with torch.no_grad():
        fs_s,_=extract_features(sa); _,fd_t=extract_features(ta)
        cs,ss=sp(fs_s,fd_t)
        zv=cv(cs,ss); zu=mimi._to_encoder_framerate(zv)
        if mimi.decoder_transformer: (ztr,)=mimi.decoder_transformer(zu)
        else: ztr=zu
        return mimi.decoder(ztr)

subprocess.run(['ffmpeg','-y','-i','/Users/asill/Downloads/origin.mp3','-ar','24000','-ac','1','-sample_fmt','s16','/tmp/tp5.wav'],capture_output=True)
src_a=load_any(f'{base}/p225/p225_001_mic1.flac')
tgt_p=load_any(f'{base}/p226/p226_001_mic1.flac')
tgt_c=load_any('/tmp/tp5.wav',dur=None)
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
    sf.write(f'{out}/phase1e_{nm}.wav',va[0,0,:Tc].numpy(),SR)
print('✅')
