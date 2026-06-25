"""Waveform-only training — paradigm shift test."""
import sys, warnings, logging, random, time, json
warnings.filterwarnings("ignore"); logging.disable(logging.INFO)
sys.path.insert(0, "external/MioCodec/src"); sys.path.insert(0, ".")
import torch, torchaudio, numpy as np, argparse
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from mcs_common import Batch, split_by_speaker, speaker_balanced_subset, move_batch, save_checkpoint, DEFAULT_DATA_DIR, multi_resolution_stft_loss
from train_mcs_q2d2 import MCSTransQ2D2Config, MCSTransQ2D2
from eval_mcs_trans_audio import load_mio, load_wave, SAMPLE_RATE

class DS(Dataset):
    def __init__(self, indices, speakers, source_files, max_s=3.0):
        self.idx=[int(i) for i in indices]; self.spk=speakers; self.src=source_files
        self.max_samp=int(max_s*SAMPLE_RATE); self.rng=random.Random(42)
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        import soundfile as sf
        idx=self.idx[i]; wav,_=sf.read(str(Path(self.src[idx])),dtype="float32")
        wav=torch.from_numpy(np.asarray(wav)); 
        if wav.ndim==2: wav=wav.mean(1)
        if _!=SAMPLE_RATE: wav=torchaudio.functional.resample(wav.unsqueeze(0),_,SAMPLE_RATE).squeeze(0)
        if wav.shape[0]>self.max_samp: start=self.rng.randint(0,wav.shape[0]-self.max_samp); wav=wav[start:start+self.max_samp]
        elif wav.shape[0]<self.max_samp: wav=F.pad(wav,(0,self.max_samp-wav.shape[0]))
        mel=torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,n_fft=2048,hop_length=882,n_mels=80,f_min=0.0,f_max=SAMPLE_RATE/2.0,power=1,center=False)(wav.unsqueeze(0))
        mel=torch.log(torch.clamp(mel,min=1e-5))
        tch=torch.from_numpy(np.load(Path("data/mio_vctk_full_compact")/f"s_{idx:05d}.npz",allow_pickle=False)["ce_768"].astype(np.float32))
        return mel[0].float(),tch.float(),str(self.spk[idx]),idx,str(self.src[idx])

def collate(samples, max_s=6.0):
    B=len(samples); MF=int(max_s*50); CF=int(max_s*25)
    mels=torch.zeros(B,80,MF); contents=[]; masks=torch.zeros(B,CF,dtype=torch.bool)
    speakers=[]; indices=[]; srcs=[]
    for i,(mel,tch,spk,idx,src) in enumerate(samples):
        mf=min(mel.shape[1],MF); mels[i,:,:mf]=mel[:,:mf]
        cf=min(tch.shape[0],CF)
        if tch.shape[0]<CF: tch=F.pad(tch,(0,0,0,CF-tch.shape[0]))
        contents.append(tch[:CF]); masks[i,:cf]=True
        speakers.append(spk); indices.append(idx); srcs.append(src)
    return Batch(mel=mels,content=torch.stack(contents),tokens=torch.zeros(B,CF,dtype=torch.long),mask=masks,speakers=speakers,indices=torch.tensor(indices,dtype=torch.long),crop_starts=torch.zeros(B,dtype=torch.long)),srcs

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",default="mps"); p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--steps",type=int,default=500); p.add_argument("--bs",type=int,default=2)
    p.add_argument("--max-s",type=float,default=6.0); p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--ww",type=float,default=2.0); p.add_argument("--grl",type=float,default=0.0); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--out",type=Path,default=Path("/Volumes/UNTITLED/btrv5_checkpoints/wave_only"))
    p.add_argument("--resume",type=Path,default=None)
    args=p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); dev=torch.device(args.device)
    meta=np.load(DEFAULT_DATA_DIR/"meta.npz",allow_pickle=False)
    n=int(meta["n_samples"]); spk=meta["spk_names"][:n].astype(str); src=meta["source_files"][:n].astype(str)
    ti,vi=split_by_speaker(spk,0.05,args.seed); pi=speaker_balanced_subset(vi,spk,128,args.seed)
    td=DS(ti,spk,src,args.max_s); pd=DS(pi,spk,src,args.max_s)
    tl=DataLoader(td,args.bs,shuffle=True,collate_fn=lambda x:collate(x,args.max_s))
    pl=DataLoader(pd,args.bs,shuffle=False,collate_fn=lambda x:collate(x,args.max_s))
    config=MCSTransQ2D2Config(n_layers=4,trans_dim=512,n_heads=8,ffn_dim=1024,window=256,use_rope=True,use_swiglu=True,q2d2_dim=6,q2d2_levels=(9,9,9,9,9,9),q2d2_grid="rhombic",grl_weight=args.grl,grl_num_speakers=len(set(spk)) if args.grl>0 else 0)
    spk_to_id = {s:i for i,s in enumerate(sorted(set(spk)))} if args.grl>0 else {}
    model=MCSTransQ2D2(config).to(dev)
    start_ep=0
    if args.resume:
        ck=torch.load(args.resume,map_location="cpu",weights_only=False)
        model.load_state_dict(ck["state_dict"],strict=False)
        start_ep=ck.get("metrics",{}).get("epoch",-1)+1
        print(f"Resumed from epoch {start_ep}",flush=True)
    mio=load_mio(dev).eval()
    for p in mio.parameters(): p.requires_grad_(False)
    nffts=(512,1024,2048); opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    args.out.mkdir(parents=True,exist_ok=True); best_wave=float("inf"); t0=time.time()
    print(f"Wave-only: w={args.ww}, center=False, max_s={args.max_s}",flush=True)
    for ep in range(start_ep, args.epochs):
        model.train(); tot={}
        for st,(batch,srcs) in enumerate(tl,1):
            if st>args.steps: break
            batch=move_batch(batch,dev); out=model(batch.mel,batch.mask)
            proj=out["projected"]
            ii=random.randrange(len(batch.speakers))
            sp=Path(srcs[ii])
            wl=torch.tensor(0.0,device=dev)
            if sp.exists():
                ow=load_wave(sp,SAMPLE_RATE,max_seconds=args.max_s).to(dev)
                with torch.no_grad(): fe=mio.encode(ow.unsqueeze(0),return_content=False,return_global=True); sl=mio._calculate_target_stft_length(ow.numel())
                ci=proj[ii].unsqueeze(0).transpose(1,2); nf=min(ci.shape[1],int(args.max_s*25))
                pw=mio.forward_wave(ci[:,:nf],fe.global_embedding.unsqueeze(0),stft_length=sl).squeeze(0)
                tl_=min(pw.shape[-1],ow.shape[-1]); wl=multi_resolution_stft_loss(pw[:tl_],ow[:tl_],nffts)
            with torch.no_grad():
                L=min(proj.shape[2],batch.content.shape[1],batch.mask.shape[1]); m=batch.mask[:,:L]
                cos=F.cosine_similarity(proj[:,:,:L].permute(0,2,1)[m],batch.content[:,:L][m],dim=-1).mean()
            loss=args.ww*wl
            grl_loss_val=0.0
            if args.grl>0 and model.speaker_classifier is not None:
                from train_mcs_q2d2 import grad_reverse
                grl_content=grad_reverse(proj,args.grl)
                spk_ids=torch.tensor([spk_to_id[s] for s in batch.speakers],device=dev,dtype=torch.long)
                sl=model.speaker_classifier(grl_content)
                gl=F.cross_entropy(sl,spk_ids); loss=loss+gl; grl_loss_val=float(gl.cpu())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot["loss"]=tot.get("loss",0)+float(loss.cpu()); tot["wave"]=tot.get("wave",0)+float(wl.cpu()); tot["cos"]=tot.get("cos",0)+float(cos.cpu()); tot["grl"]=tot.get("grl",0)+float(grl_loss_val)
            if st%200==0: d=max(st,1); print(f"E{ep:03d} s={st:04d} loss={tot['loss']/d:.4f} wave={tot['wave']/d:.4f} cos={tot['cos']/d:.4f}",flush=True)
        sch.step()
        model.eval(); pc=0.0; nb=0
        for batch,_ in pl:
            batch=move_batch(batch,dev); out=model(batch.mel,batch.mask)
            proj=out["projected"]; L=min(proj.shape[2],batch.content.shape[1],batch.mask.shape[1]); m=batch.mask[:,:L]
            pc+=float(F.cosine_similarity(proj[:,:,:L].permute(0,2,1)[m],batch.content[:,:L][m],dim=-1).mean().cpu()); nb+=1
        model.train(); pc/=max(nb,1); wv=tot.get("wave",0)/args.steps
        print(f"E{ep:03d} probe cos={pc:.4f} wave={wv:.4f}",flush=True)
        mf={"epoch":ep,"global_step":(ep+1)*args.steps,"probe":{"cos768":pc,"wave_loss":wv},"elapsed":time.time()-t0}
        save_checkpoint(args.out/"last.pt",model,opt,sch,ep,mf,args,best_wave)
        if wv<best_wave: best_wave=wv; save_checkpoint(args.out/"best.pt",model,opt,sch,ep,mf,args,best_wave)
        (args.out/"summary.json").write_text(json.dumps(mf,indent=2)+"\n")
    print(f"done best_wave={best_wave:.4f}",flush=True)

if __name__=="__main__": main()
