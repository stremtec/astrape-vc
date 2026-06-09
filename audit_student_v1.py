#!/usr/bin/env python3
"""Stage 1.5: CausalContentStudent v1 Generalization Audit."""
import torch, torch.nn as nn, numpy as np, os, time, warnings, json
warnings.filterwarnings('ignore')

SR=44100; HOP_44k=int(SR/25)
device=torch.device('cpu')

# ── Model definition ──────────────────────────────────────────────────
class CausalTCNEncoder(nn.Module):
    def __init__(self,in_dim=80,hidden=256,out_dim=5,num_layers=4,kernel=5):
        super().__init__()
        self.proj_in=nn.Conv1d(in_dim,hidden,1)
        layers=[]
        for i in range(num_layers):
            d=2**i; p=(kernel-1)*d
            layers.append(nn.Sequential(
                nn.Conv1d(hidden,hidden,kernel,dilation=d,padding=p,padding_mode='replicate'),
                nn.GroupNorm(8,hidden),nn.GELU(),nn.Conv1d(hidden,hidden,1)))
        self.layers=nn.ModuleList(layers)
        self.down=nn.Conv1d(hidden,hidden,3,stride=2,padding=1,padding_mode='replicate')
        self.proj_out=nn.Conv1d(hidden,out_dim,1)
        self.embed_head=nn.Conv1d(out_dim,768,1)
    def forward(self,x):
        h=self.proj_in(x)
        for layer in self.layers:
            r=h; h=layer(h); h=h[:,:,:r.shape[2]]; h=h+r
        h=self.down(h)
        fsq=self.proj_out(h)  # (B,5,T)
        embed=self.embed_head(fsq)  # (B,768,T)
        return fsq, embed

# ── Load data ─────────────────────────────────────────────────────────
MEL_DIR="/Users/asill/btrv5/data/mio_mel"
meta=np.load("/Users/asill/btrv5/data/mio_teacher/meta.npz")
spk_names=meta['spk_names']; spk_idxs=meta['spk_idxs']; n=len(spk_names)

# Split: 20 train spk, 5 val spk, 5 test spk (unseen)
unique_spks=sorted(set(spk_names))
np.random.RandomState(42).shuffle(unique_spks)
tr_spks=set(unique_spks[:20]); vl_spks=set(unique_spks[20:25]); ts_spks=set(unique_spks[25:30])

tr_idx=[i for i in range(n) if spk_names[i] in tr_spks]
vl_idx=[i for i in range(n) if spk_names[i] in vl_spks]
ts_idx=[i for i in range(n) if spk_names[i] in ts_spks]

print("Train: {} spk, {} utt | Val: {} spk, {} utt | Test: {} spk, {} utt".format(
    len(tr_spks),len(tr_idx),len(vl_spks),len(vl_idx),len(ts_spks),len(ts_idx)))

def load_mel(idx):
    d=np.load("{}/mel_{:04d}.npz".format(MEL_DIR,idx))
    return d['logmel'],d['fsq_5d'],d['fsq_tokens'],d['ce_768']

# ── Baselines ─────────────────────────────────────────────────────────
# Compute mean FSQ from training set
all_tr_fsq=[]
for i in tr_idx:
    _,fsq,_,_=load_mel(i)
    all_tr_fsq.append(fsq)
mean_fsq=np.mean(np.concatenate(all_tr_fsq,axis=0),axis=0)  # (5,)

# ── Load model ────────────────────────────────────────────────────────
model=CausalTCNEncoder().to(device)
model.load_state_dict(torch.load("checkpoints/causal_student_v1.pt",map_location='cpu'))
model.eval()

# ── Teacher for plug-in ───────────────────────────────────────────────
from miocodec.model import MioCodecModel
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2')
teacher.eval()

# ── Metric functions ──────────────────────────────────────────────────
def fsq_level_accuracy(pred,target,levels=[8,8,8,5,5]):
    """Per-dimension: what % of frames are quantized to the correct level?"""
    n_levels=np.array(levels)
    half=n_levels//2
    # Pred and target are (5,T)
    pred_q=np.clip(np.round(pred*half[:,None])/half[:,None],-1,1)
    target_q=np.clip(np.round(target*half[:,None])/half[:,None],-1,1)
    acc=(pred_q==target_q).mean(axis=1)
    return acc

def token_match(pred,token_target):
    """Exact token index match."""
    # pred: (5,T), need to convert to FSQ tokens
    pred_rounded=np.round(pred*np.array([4,4,4,2,2])[:,None])
    # Simplified: use L2 distance to codebook... too slow
    # Use per-dim level match as proxy
    return 0.0  # Token match is 0% for continuous→discrete at 12800 classes

def compute_metrics(pred_fsq,true_fsq,true_ce,mel_true=None,mel_pred=None):
    """All metrics for one sample."""
    m={}
    T=min(pred_fsq.shape[1],true_fsq.shape[0])
    p=pred_fsq[:,:T]; t=true_fsq[:T].T
    
    m['fsq_mse']=np.mean((p-t)**2)
    m['fsq_level_acc']=fsq_level_accuracy(p,t)
    # Content embedding cosine
    if true_ce is not None and T>0:
        with torch.no_grad():
            z_q,_=teacher.local_quantizer.fsq.encode(torch.from_numpy(p.T).float().unsqueeze(0))
            pred_ce=teacher.local_quantizer.proj_out(z_q).squeeze(0).numpy()  # (T,768)
            true_ce_t=true_ce[:T]
            cos=np.sum(pred_ce*true_ce_t,axis=1)/(np.linalg.norm(pred_ce,axis=1)*np.linalg.norm(true_ce_t,axis=1)+1e-8)
            m['content_cosine']=np.mean(cos)
    else:
        m['content_cosine']=0
    
    return m

# ── Evaluate ──────────────────────────────────────────────────────────
def eval_split(name,idxs):
    results=[]
    for idx in idxs:
        logmel,fsq_true,tokens_true,ce_true=load_mel(idx)
        T=fsq_true.shape[0]
        mel_t=torch.from_numpy(logmel).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            fsq_pred,_=model(mel_t)
        fsq_pred_np=fsq_pred.squeeze(0).cpu().numpy()
        
        m=compute_metrics(fsq_pred_np,fsq_true,ce_true)
        m['idx']=idx; m['spk']=str(spk_names[idx]); m['T']=T
        results.append(m)
    
    # Aggregate
    agg={}
    for k in ['fsq_mse','content_cosine']:
        vals=[r[k] for r in results]
        agg[k+'_mean']=np.mean(vals)
        agg[k+'_std']=np.std(vals)
    accs=np.array([r['fsq_level_acc'] for r in results])
    agg['level_acc_mean']=accs.mean(axis=0)
    agg['level_acc_std']=accs.std(axis=0)
    
    # Baselines
    mean_mse=[]
    for idx in idxs:
        _,fsq_true,_,_=load_mel(idx)
        T=fsq_true.shape[0]
        mean_pred=np.tile(mean_fsq,(T,1)).T
        mean_mse.append(np.mean((mean_pred[:,:T]-fsq_true[:T].T)**2))
    agg['mean_baseline_mse']=np.mean(mean_mse)
    agg['n']=len(results)
    
    print("  {}: n={} FSQ_MSE={:.4f}±{:.4f} MeanBL={:.4f} Cos={:.3f}".format(
        name,agg['n'],agg['fsq_mse_mean'],agg['fsq_mse_std'],
        agg['mean_baseline_mse'],agg['content_cosine_mean']))
    print("    Level acc: d0={:.1f}% d1={:.1f}% d2={:.1f}% d3={:.1f}% d4={:.1f}%".format(
        *[agg['level_acc_mean'][i]*100 for i in range(5)]))
    
    return agg, results

print()
print("="*80)
print("  CAUSAL CONTENT STUDENT v1 — GENERALIZATION AUDIT")
print("="*80)

tr_agg,tr_res=eval_split("TRAIN",tr_idx)
vl_agg,vl_res=eval_split("VAL",vl_idx)
ts_agg,ts_res=eval_split("TEST (unseen spk)",ts_idx)

# ── Teacher Plug-in (2 samples per split) ────────────────────────────
print()
print("--- Teacher Decoder Plug-in ---")
import soundfile as sf; from scipy import signal as scipy_signal; import torchaudio

mel_spec=torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,n_fft=512,hop_length=320,n_mels=80,f_min=80,f_max=7600,center=False,power=2)

for split_name,sample_idxs in [("TRAIN",tr_idx[:2]),("VAL",vl_idx[:2]),("TEST",ts_idx[:2])]:
    for idx in sample_idxs:
        d=np.load("/Users/asill/btrv5/data/mio_teacher/sample_{:04d}.npz".format(idx))
        audio=d['audio']; alen=len(audio)
        audio_16k=scipy_signal.resample(audio,int(alen*16000/SR))
        mel=mel_spec(torch.from_numpy(audio_16k).float().view(1,1,-1)).squeeze(1)
        logmel=torch.log(mel.clamp(min=1e-5))
        
        with torch.inference_mode():
            fsq_pred,_=model(logmel.unsqueeze(0))
            fsq_t=fsq_pred.squeeze(0).T
            z_q,_=teacher.local_quantizer.fsq.encode(fsq_t.unsqueeze(0))
            z_q=teacher.local_quantizer.proj_out(z_q)
            # For teacher plug-in, we need the original global embedding
            # Since we don't have it cached, use a dummy or skip
            # Actually, use the teacher's own global emb
            x_t=torch.from_numpy(audio[:SR*3]).float().unsqueeze(0)
            ft=teacher.encode(x_t,return_content=True,return_global=True)
            ge=ft.global_embedding
            wav_t=teacher.decode(global_embedding=ge,content_token_indices=ft.content_token_indices,target_audio_length=alen)
            wav_s=teacher.decode(global_embedding=ge,content_embedding=z_q.squeeze(0),target_audio_length=alen)
        
        from scipy.signal import stft
        def centroid(a):
            a=a-np.mean(a)
            f,_,Z=stft(a,fs=SR,nperseg=1024,noverlap=768)
            mag=np.abs(Z); total=mag.sum()+1e-8
            c=np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2],axis=0)/(mag[:len(f)//2].sum(axis=0)+1e-8)
            return np.mean(c)
        
        wt=wav_t.numpy()[:alen]; ws=wav_s.numpy()[:alen]
        ct=centroid(wt); cs=centroid(ws)
        print("  {} {}: T cent={:.0f}Hz S cent={:.0f}Hz Δ={:.0f}Hz".format(
            split_name,spk_names[idx],ct,cs,cs-ct))

# ── Summary ───────────────────────────────────────────────────────────
print()
print("="*80)
print("  SUMMARY")
print("="*80)
print("  FSQ MSE: train={:.4f} val={:.4f} test={:.4f}".format(
    tr_agg['fsq_mse_mean'],vl_agg['fsq_mse_mean'],ts_agg['fsq_mse_mean']))
print("  Mean baseline MSE: {:.4f}".format(tr_agg['mean_baseline_mse']))
print("  Ratio (student/baseline): train={:.2f}x val={:.2f}x test={:.2f}x".format(
    tr_agg['mean_baseline_mse']/tr_agg['fsq_mse_mean'],
    tr_agg['mean_baseline_mse']/vl_agg['fsq_mse_mean'],
    tr_agg['mean_baseline_mse']/ts_agg['fsq_mse_mean']))
print()
print("  Pass criteria check:")
overfit_ratio=vl_agg['fsq_mse_mean']/tr_agg['fsq_mse_mean']
print("  val/train MSE ratio: {:.2f}x (<2x = no severe overfit)".format(overfit_ratio))
print("  → {} ".format("PASS" if overfit_ratio<2 else "OVERFIT"))
print("  student vs mean baseline: {:.2f}x → {} ".format(
    tr_agg['mean_baseline_mse']/ts_agg['fsq_mse_mean'],
    "BEATS BASELINE" if ts_agg['fsq_mse_mean']<tr_agg['mean_baseline_mse'] else "FAILS"))
print()
print("Done!")
