#!/usr/bin/env python3
"""MioCodec extended evaluation: 10+ VC pairs with full metrics."""
import torch, time, numpy as np, soundfile as sf
from scipy import signal
import sys
sys.path.insert(0,'/Users/asill/btrvrc0/.venv/lib/python3.12/site-packages')
from miocodec.model import MioCodecModel

SR=44100
model=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2')
model.eval()

def load(path, dur=3):
    d,s=sf.read(path)
    if d.ndim>1: d=d.mean(axis=1)
    if s!=SR: d=signal.resample(d,int(len(d)*SR/s))
    return d[:int(SR*dur)]

def measure(a, sr=SR):
    a=a-np.mean(a)
    f,_,Z=signal.stft(a,fs=sr,nperseg=1024,noverlap=768)
    mag=np.abs(Z); total=mag.sum()+1e-8
    c=np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2],axis=0)/(mag[:len(f)//2].sum(axis=0)+1e-8)
    vh=mag[(f>=4000)&(f<8000)].sum()/total*100
    cr=np.max(np.abs(a))/(np.sqrt(np.mean(a**2))+1e-8)
    # Jitter
    fl,hp=int(sr*0.04),int(sr*0.01); fs=[]
    for i in range(0,len(a)-fl,hp):
        fr=a[i:i+fl]
        if np.sqrt(np.mean(fr**2))<0.001: fs.append(0); continue
        corr=np.correlate(fr,fr,mode='full'); corr=corr[len(corr)//2:]; corr=corr/(corr[0]+1e-8)
        pks=signal.find_peaks(corr,distance=10)[0]
        if len(pks)==0: fs.append(0); continue
        f0=sr/pks[0]; fs.append(f0 if 50<f0<400 else 0)
    fs=np.array(fs); v=fs>0
    j=np.mean(np.abs(np.diff(fs[v])))/np.mean(fs[v])*100 if v.sum()>3 else 0
    return np.mean(c),vh,cr,j

ROOT="/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"
import glob

# Test pairs
pairs=[
    ("p255","origin","m→f cross-lang"),
    ("p226","origin","f→f cross-lang"),
    ("p285","origin","f→f cross-lang"),
    ("p255","p226","m→f VCTK"),
    ("p255","p285","m→f VCTK"),
    ("p285","p255","f→m VCTK"),
    ("p226","p255","f→m VCTK"),
]

# Also test self-recon for a few
print("="*90)
print("  MioCodec Extended VC Evaluation")
print("="*90)
print(f"  {'Pair':<25s} {'Cent':>7s} {'Jitter':>7s} {'Crest':>6s} {'VHigh':>6s} {'Time':>6s} {'Notes':>20s}")
print("  "+"-"*85)

for src_spk, tgt_spk, desc in pairs:
    # Load source
    if src_spk.startswith('p'):
        files=sorted(glob.glob(f"{ROOT}/{src_spk}/{src_spk}_*_mic1.flac"))
        d_src=load(files[0],3)
    else:
        d_src=load(f"/Users/asill/Downloads/{src_spk}.mp3",3)
    
    # Load target  
    if tgt_spk.startswith('p'):
        files=sorted(glob.glob(f"{ROOT}/{tgt_spk}/{tgt_spk}_*_mic1.flac"))
        d_tgt=load(files[0],3)
    else:
        d_tgt=load(f"/Users/asill/Downloads/{tgt_spk}.mp3",3)
    
    src_len=len(d_src)
    x_src=torch.from_numpy(d_src).float()
    x_tgt=torch.from_numpy(d_tgt).float()
    
    with torch.inference_mode():
        t0=time.time()
        wav_vc=model.voice_conversion(x_src,x_tgt)
        elapsed=time.time()-t0
    
    vc_np=wav_vc.cpu().numpy()[:src_len]
    c,vh,cr,j=measure(vc_np)
    
    # Source/Target metrics
    sc,_,_,sj=measure(d_src)
    tc,_,_,tj=measure(d_tgt[:src_len])
    
    # Notes
    notes=""
    if c>tc+200: notes+="OVER_TGT "
    elif abs(c-tc)<200: notes+="MATCH_TGT ★ "
    elif c>sc+200: notes+="SHIFT_OK "
    if j<15: notes+="CLEAN "
    elif j<25: notes+="OK "
    
    print(f"  {src_spk}→{tgt_spk} ({desc:<15s}) {c:6.0f}Hz {j:6.1f}% {cr:5.1f} {vh:5.1f}% {elapsed:5.2f}s {notes:<20s}")

# Self-recon test
print()
print("  Self-recon tests:")
for spk in ['p255','p226','p285']:
    d_src=load(f"{ROOT}/{spk}/{spk}_001_mic1.flac",3) if spk.startswith('p') else load(f"/Users/asill/Downloads/{spk}.mp3",3)
    src_len=len(d_src)
    x_src=torch.from_numpy(d_src).float()
    with torch.inference_mode():
        feat=model.encode(x_src,return_content=True,return_global=True)
        wav_self=model.decode(global_embedding=feat.global_embedding,
                             content_token_indices=feat.content_token_indices,
                             target_audio_length=src_len)
    sn=wav_self.cpu().numpy()[:src_len]
    sc,svh,scr,sj=measure(d_src)
    rc,rvh,rcr,rj=measure(sn)
    print(f"  {spk}: src Cent={sc:.0f}Hz J={sj:.1f}% → recon Cent={rc:.0f}Hz J={rj:.1f}% ΔCent={rc-sc:.0f}Hz")

# Latency summary
print()
print("  Latency (~2-3s audio, CPU):")
t0=time.time()
x_test=torch.from_numpy(d_src).float()
with torch.inference_mode():
    _=model.encode(x_test,return_content=True,return_global=True)
enc_t=time.time()-t0
t0=time.time()
feat=model.encode(x_test,return_content=True,return_global=True)
with torch.inference_mode():
    _=model.decode(global_embedding=feat.global_embedding,
                  content_token_indices=feat.content_token_indices,
                  target_audio_length=len(d_src))
dec_t=time.time()-t0
print(f"  Encode: {enc_t*1000:.0f}ms, Decode wave: {dec_t*1000:.0f}ms, Total: {(enc_t+dec_t)*1000:.0f}ms")
rtf=(enc_t+dec_t)/(len(d_src)/SR)
print(f"  RTF: {rtf:.3f} ({'real-time' if rtf<1 else 'slower than real-time'})")
print()
print("Done!")
