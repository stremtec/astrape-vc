"""Save SSL layers directly into each sample's .npz file."""
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
    print(f'Extracting SSL layers for samples {args.start}..{end-1} ({end-args.start} total)')

    mio=load_mio(dev).eval()
    ssl=mio.ssl_feature_extractor.model
    inter={}
    def h(name):
        def hook(m,i,o):inter[name]=o[0] if isinstance(o,tuple) else o
        return hook
    for layer in [0,4,8]:ssl.encoder.transformer.layers[layer].register_forward_hook(h(f'L{layer}'))

    data_dir=Path(args.data_dir)
    for i in range(args.start,end):
        wav=load_wave(Path(str(srcs[i])),SAMPLE_RATE,max_seconds=args.max_s).to(dev)
        inter.clear()
        with torch.no_grad():mio.encode(wav.unsqueeze(0),return_content=False,return_global=False)
        path=data_dir/f's_{i:05d}.npz'
        existing=np.load(path,allow_pickle=False)
        d=dict(existing)
        for l in [0,4,8]:
            feat=inter.get(f'L{l}')
            if feat is not None:d[f'ssl_L{l}']=feat[0].cpu().numpy().astype(np.float32)
        np.savez_compressed(path,**d)
        if (i-args.start)%500==0:print(f'  {i-args.start}/{end-args.start} samples saved')

    print(f'Done: {end-args.start} samples updated')

if __name__=='__main__':main()
