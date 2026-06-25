"""Cache WavLM CNN features for all VCTK samples.

Adds 'wavlm_cnn' key to each s_XXXXX.npz: (T, 512) float32 @ ~50Hz.
Downsampled from 137Hz via avg_pool(k=3,s=3), strictly causal.
"""
import sys,warnings,logging;warnings.filterwarnings('ignore');logging.disable(logging.INFO)
sys.path.insert(0,'external/MioCodec/src')
import torch,numpy as np,argparse
from pathlib import Path
from eval_mcs_trans_audio import load_mio,load_wave,SAMPLE_RATE

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default='data/mio_vctk_full_compact')
    ap.add_argument('--max-s',type=float,default=6.0)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    dev=torch.device(args.device)

    meta=np.load(Path(args.data_dir)/'meta.npz',allow_pickle=False)
    n=int(meta['n_samples']);srcs=meta['source_files'][:n].astype(str)
    end=n if args.limit==0 else min(n,args.start+args.limit)
    print(f'Caching WavLM CNN for samples {args.start}..{end-1} ({end-args.start} total)')

    mio=load_mio(dev).eval()
    fe=mio.ssl_feature_extractor.model.feature_extractor
    data_dir=Path(args.data_dir)

    for i in range(args.start,end):
        wav=load_wave(Path(str(srcs[i])),SAMPLE_RATE,max_seconds=args.max_s).to(dev)
        with torch.no_grad():
            cnn_out,_ = fe(wav.unsqueeze(0),length=None)
        # (1, T_137, 512) → downsample 137Hz → 46Hz via avg_pool(k=3,s=3)
        cnn = cnn_out.transpose(1,2)
        cnn_ds = torch.nn.functional.avg_pool1d(cnn,kernel_size=3,stride=3)
        cnn_ds = cnn_ds.transpose(1,2).squeeze(0).cpu().numpy().astype(np.float32)

        path=data_dir/f's_{i:05d}.npz'
        existing=np.load(path,allow_pickle=False)
        d=dict(existing)
        d['wavlm_cnn']=cnn_ds
        np.savez_compressed(path,**d)

        if (i-args.start)%500==0:print(f'  {i-args.start}/{end-args.start}')

    print(f'Done: {end-args.start} samples cached')

if __name__=='__main__':main()
