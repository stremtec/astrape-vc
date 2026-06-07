"""Phase 1d: Cached raw features + lightweight trainable splitter."""
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

# Cache raw transformer features
print('Caching raw features...')
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
            fs=torch.stack(sh,0).mean(0).transpose(1,2)
            fd=torch.stack(dp,0).mean(0).transpose(1,2)
            z=mimi.encode_to_latent(a,quantize=False)
            codes=mimi.quantizer.encode(z)
            zq=mimi.quantizer.decode(codes)
            raw[(s,u)]=(fs,fd,zq)
print(f'Cached {len(raw)}')

class LightSplitter(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_bn=nn.Sequential(nn.Conv1d(512,64,1),nn.GELU(),nn.Conv1d(64,512,1))
        self.s_net=nn.Sequential(nn.Conv1d(512,256,5,padding=2),nn.GELU(),nn.Conv1d(256,512,5,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(512,512))
    def forward(self,fs,fd):
        c=fs+self.c_bn(fs); s=self.s_net(fd); return c,s

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

sp=LightSplitter(); cv=Conv()
opt=torch.optim.AdamW(list(sp.parameters())+list(cv.parameters()), lr=1e-3)
pairs=[(s,t,u) for s in spks for t in spks for u in utts if s!=t]
random.shuffle(pairs)

t0=time.time()
for step in range(80):
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
    if step%20==0:
        n=len(pairs)
        print(f'  step {step:3d}: loss={lt/n:.4f} c_cos={cos_c.item():.4f} s_cos={cos_s.item():.4f} [{time.time()-t0:.1f}s]')
print(f'Done [{time.time()-t0:.1f}s]')

# Copy weights to full splitter for inference
from mimi_codec.mimi_splitter import KanadeSplitterV4
full_sp = KanadeSplitterV4(mimi, bottleneck=64)
full_sp.content_bn.load_state_dict(sp.c_bn.state_dict())
# speaker_enc has different structure (conv vs conv+linear), skip for now
# Just use LightSplitter directly for inference with raw features
def convert(sa,ta):
    with torch.no_grad():
        enc_s=mimi.encoder(sa); h_s=enc_s.transpose(1,2)
        enc_t=mimi.encoder(ta); h_t=enc_t.transpose(1,2)
        tt=mimi.encoder_transformer.transformer
        sh_s=[]; dp_s=[]; sh_t=[]; dp_t=[]
        for i,layer in enumerate(tt.layers):
            h_s=layer(h_s); h_t=layer(h_t)
            if i in[0,1,2]: sh_s.append(h_s); sh_t.append(h_t)
            if i in[5,6,7]: dp_s.append(h_s); dp_t.append(h_t)
        fs_s=torch.stack(sh_s,0).mean(0).transpose(1,2)
        fd_t=torch.stack(dp_t,0).mean(0).transpose(1,2)
        c_s,_=sp(fs_s,fd_t)  # content from source, ignore speaker
        _,s_t=sp(fs_s,fd_t)   # speaker from target (using target deep features)
        
        # Actually: need target speaker only
        cs,_=sp(fs_s,fd_t)  # no: reuse sp for content from src shallow
        # Redo properly:
    
    # Quick inline fix
    with torch.no_grad():
        enc_s=mimi.encoder(sa); enc_t=mimi.encoder(ta)
        h_s=enc_s.transpose(1,2); h_t=enc_t.transpose(1,2)
        tt=mimi.encoder_transformer.transformer
        sh_s=[]; dp_t=[]
        for i,layer in enumerate(tt.layers):
            h_s=layer(h_s); h_t=layer(h_t)
            if i in[0,1,2]: sh_s.append(h_s)
            if i in[5,6,7]: dp_t.append(h_t)
        fs_s=torch.stack(sh_s,0).mean(0).transpose(1,2)
        fd_t=torch.stack(dp_t,0).mean(0).transpose(1,2)
        cs,ss=sp(fs_s,fd_t)  # content from src shallow, speaker from tgt deep
        zv=cv(cs,ss); zu=mimi._to_encoder_framerate(zv)
        if mimi.decoder_transformer: (ztr,)=mimi.decoder_transformer(zu)
        else: ztr=zu
        return mimi.decoder(ztr)

subprocess.run(['ffmpeg','-y','-i','/Users/asill/Downloads/origin.mp3','-ar','24000','-ac','1','-sample_fmt','s16','/tmp/tp4.wav'],capture_output=True)
src_a=load_any(f'{base}/p225/p225_001_mic1.flac')
tgt_p=load_any(f'{base}/p226/p226_001_mic1.flac')
tgt_c=load_any('/tmp/tp4.wav',dur=None)
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
    sf.write(f'{out}/phase1d_{nm}.wav',va[0,0,:Tc].numpy(),SR)
print('✅')
