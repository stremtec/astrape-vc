#!/usr/bin/env python3
"""Extract 40 utt/spk → ~4360 samples for V3-4k."""
import torch, numpy as np, os, glob, soundfile as sf
from scipy import signal; import torchaudio
from miocodec.model import MioCodecModel

SR=44100; TARGET_SR=16000
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2'); teacher.eval()
ROOT='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'
OUT_DIR='/Users/asill/btrv5/data/mio_4k'; MEL_DIR='/Users/asill/btrv5/data/mio_4k_mel'
os.makedirs(OUT_DIR,exist_ok=True); os.makedirs(MEL_DIR,exist_ok=True)

spk_dirs=sorted([d for d in os.listdir(ROOT) if d.startswith('p')])
mel_spec=torchaudio.transforms.MelSpectrogram(sample_rate=TARGET_SR,n_fft=512,hop_length=320,n_mels=80,f_min=80,f_max=7600,center=False,power=2)
idx=0
for spk in spk_dirs:
    files=sorted(glob.glob(f'{ROOT}/{spk}/{spk}_*_mic1.flac'))[:40]
    for f in files:
        d,sr=sf.read(f)
        if d.ndim>1: d=d.mean(axis=1)
        if sr!=SR: d=signal.resample(d,int(len(d)*SR/sr))
        d=d[:SR*3]; pad=teacher._calculate_waveform_padding(len(d))
        x=torch.from_numpy(d).float().unsqueeze(0)
        with torch.inference_mode():
            feat=teacher.encode(x,return_content=True,return_global=True)
            local_ssl,_=teacher.forward_ssl_features(x,padding=pad)
            local_enc=teacher.local_encoder(local_ssl)
            local_enc=teacher.conv_downsample(local_enc.transpose(1,2)).transpose(1,2)
            pre_fsq=local_enc.squeeze(0).numpy()
        a16=signal.resample(d[:len(d)],int(len(d)*TARGET_SR/SR))
        mel=mel_spec(torch.from_numpy(a16).float().view(1,1,-1))
        logmel=torch.log(mel.squeeze(1).clamp(min=1e-5)).squeeze(0).numpy()
        np.savez_compressed(f'{OUT_DIR}/s_{idx:05d}.npz',
            ce_768=feat.content_embedding.numpy(),ct=feat.content_token_indices.numpy(),
            ge_128=feat.global_embedding.numpy(),audio=d,pre_fsq_768=pre_fsq)
        np.savez_compressed(f'{MEL_DIR}/m_{idx:05d}.npz',logmel=logmel)
        idx+=1
    if (spk_dirs.index(spk)+1)%20==0: print(f'  {spk_dirs.index(spk)+1}/{len(spk_dirs)} spk, {idx} samples')

spk_list=np.concatenate([[spk]*min(40,len(glob.glob(f'{ROOT}/{spk}/{spk}_*_mic1.flac'))) for spk in spk_dirs])
np.savez_compressed(f'{OUT_DIR}/meta.npz',spk_names=spk_list[:idx],n_samples=idx)
print(f'Done: {idx} samples, {len(spk_dirs)} speakers')
