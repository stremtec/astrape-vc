"""Extract WavLM intermediate layer features for multi-layer distillation.

MioCodec's encode path: SSL(51Hz) → local_encoder → content_768(25Hz)
We extract SSL layers 0, 4, 8 BEFORE the local_encoder, at ~51Hz frame rate.

Stores: ssl_L0, ssl_L4, ssl_L8 as (N, T_ssl, 768) padded tensors in a single .npz.
"""
import sys,warnings,logging;warnings.filterwarnings('ignore');logging.disable(logging.INFO)
sys.path.insert(0,'external/MioCodec/src')
import torch,numpy as np,argparse
from pathlib import Path
from eval_mcs_trans_audio import load_mio,load_wave,SAMPLE_RATE

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default='data/mio_vctk_full_compact')
    ap.add_argument('--out',default='data/mio_vctk_full_compact/ssl_layers.npz')
    ap.add_argument('--max-s',type=float,default=6.0)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--skip',type=int,default=0)
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    dev=torch.device(args.device)

    meta=np.load(Path(args.data_dir)/'meta.npz',allow_pickle=False)
    n=int(meta['n_samples']);srcs=meta['source_files'][:n].astype(str)
    if args.limit>0:n=min(n,args.skip+args.limit)
    print(f'Extracting SSL layers for {n-args.skip} samples (idx {args.skip}..{n-1})...')

    mio=load_mio(dev).eval()
    ssl=mio.ssl_feature_extractor.model
    intermediates={}
    def hook_fn(name):
        def hook(module,input,output):
            intermediates[name]=output[0] if isinstance(output,tuple) else output
        return hook
    layers=[0,4,8]
    for i in layers:
        ssl.encoder.transformer.layers[i].register_forward_hook(hook_fn(f'layer_{i}'))

    # Collect all features
    all_L0=[];all_L4=[];all_L8=[]
    for i in range(args.skip,n):
        wav=load_wave(Path(str(srcs[i])),SAMPLE_RATE,max_seconds=args.max_s).to(dev)
        intermediates.clear()
        with torch.no_grad():
            mio.encode(wav.unsqueeze(0),return_content=False,return_global=False)
        l0=intermediates.get('layer_0');l4=intermediates.get('layer_4');l8=intermediates.get('layer_8')
        if l0 is None:
            print(f'skip {i}: no layer_0')
            continue
        all_L0.append(l0[0].cpu().numpy())
        all_L4.append(l4[0].cpu().numpy() if l4 is not None else np.zeros((l0.shape[1],l0.shape[2]),dtype=np.float32))
        all_L8.append(l8[0].cpu().numpy() if l8 is not None else np.zeros((l0.shape[1],l0.shape[2]),dtype=np.float32))
        if (i-args.skip)%500==0:
            print(f'  {i-args.skip}/{n-args.skip}')

    print(f'Saving {len(all_L0)} samples...')
    np.savez_compressed(args.out, L0=all_L0, L4=all_L4, L8=all_L8)
    print(f'Saved {args.out}')

if __name__=='__main__':main()
