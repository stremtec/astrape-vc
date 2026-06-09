#!/usr/bin/env python3
"""Large-scale teacher extraction: 109 speakers × 10 utt = ~1090 samples."""
import torch, numpy as np, os, glob, time, soundfile as sf
from scipy import signal
import torchaudio
from miocodec.model import MioCodecModel

SR=44100; TARGET_SR=16000
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2'); teacher.eval()

ROOT="/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"
OUT_DIR="/Users/asill/btrv5/data/mio_large"
MEL_DIR="/Users/asill/btrv5/data/mio_large_mel"
os.makedirs(OUT_DIR,exist_ok=True); os.makedirs(MEL_DIR,exist_ok=True)

spk_dirs=sorted([d for d in os.listdir(ROOT) if d.startswith('p')])
print("Speakers:",len(spk_dirs))

mel_spec=torchaudio.transforms.MelSpectrogram(sample_rate=TARGET_SR,n_fft=512,hop_length=320,n_mels=80,f_min=80,f_max=7600,center=False,power=2)

idx=0
for spk in spk_dirs:
    files=sorted(glob.glob(f"{ROOT}/{spk}/{spk}_*_mic1.flac"))[:10]  # 10 utt/spk
    for f in files:
        d,sr=sf.read(f)
        if d.ndim>1: d=d.mean(axis=1)
        if sr!=SR: d=signal.resample(d,int(len(d)*SR/sr))
        d=d[:SR*3]  # 3 seconds
        alen=len(d)
        pad=teacher._calculate_waveform_padding(alen)
        x=torch.from_numpy(d).float().unsqueeze(0)
        
        with torch.inference_mode():
            feat=teacher.encode(x,return_content=True,return_global=True)
            ce_768=feat.content_embedding.numpy()
            ct=feat.content_token_indices.numpy()
            ge_128=feat.global_embedding.numpy()
            
            # Intermediate: pre-FSQ
            local_ssl,global_ssl=teacher.forward_ssl_features(x,padding=pad)
            local_ssl_768=local_ssl.squeeze(0).numpy()
            local_enc=teacher.local_encoder(local_ssl)
            local_enc=teacher.conv_downsample(local_enc.transpose(1,2)).transpose(1,2)
            pre_fsq_768=local_enc.squeeze(0).numpy()
            fsq_5d=teacher.local_quantizer.proj_in(local_enc).squeeze(0).numpy()
        
        # Mel
        a16=signal.resample(d[:alen],int(alen*TARGET_SR/SR))
        mel=mel_spec(torch.from_numpy(a16).float().view(1,1,-1))
        logmel=torch.log(mel.squeeze(1).clamp(min=1e-5)).squeeze(0).numpy()
        
        np.savez_compressed(f"{OUT_DIR}/sample_{idx:05d}.npz",
            ce_768=ce_768,ct=ct,ge_128=ge_128,audio=d,
            local_ssl_768=local_ssl_768,pre_fsq_768=pre_fsq_768,fsq_5d=fsq_5d,spk=spk)
        np.savez_compressed(f"{MEL_DIR}/mel_{idx:05d}.npz",logmel=logmel,fsq_5d=fsq_5d)
        idx+=1
    
    if (spk_dirs.index(spk)+1)%20==0: print("  {}/{} speakers, {} samples".format(spk_dirs.index(spk)+1,len(spk_dirs),idx))

# Save metadata
spk_list=np.array([d for d in spk_dirs for _ in range(10)])
np.savez_compressed(f"{OUT_DIR}/meta.npz",spk_names=spk_list[:idx],n_samples=idx)
print("Done! {} samples from {} speakers".format(idx,len(spk_dirs)))
print("Saved to {}/ and {}/".format(OUT_DIR,MEL_DIR))
