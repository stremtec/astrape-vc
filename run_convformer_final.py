#!/usr/bin/env python3
"""Self-contained ConvFormer — model + training in one file, reverter-proof."""
import sys, argparse, time
from pathlib import Path
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

CFG = {
    'mel_bins': 80, 'dim': 512, 'conv_layers': 6, 'conv_expansion': 2,
    'transformer_layers': 3, 'n_heads': 8, 'dropout': 0.1,
    'fsq_levels': (8, 8, 8, 5, 5), 'content_dim': 768,
    'axis_emb_dim': 32, 'axis_head_hidden': 256,
}

class CausalConv1d(nn.Module):
    def __init__(self, dim, kernel_size, dilation=1):
        super().__init__()
        self.lc = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(dim, dim, kernel_size, dilation=dilation, groups=dim, bias=False)
    def forward(self, x): return self.conv(F.pad(x, (self.lc, 0)))

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, dilation=1, expansion=2):
        super().__init__()
        self.dwconv = CausalConv1d(dim, kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * expansion, bias=False)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(dim * expansion, dim, bias=False)
    def forward(self, x):
        r = x
        x = self.dwconv(x.transpose(1,2)).transpose(1,2)
        x = self.norm(x)
        return r + self.pw2(self.act(self.pw1(x)))

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads; self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
    def forward(self, x, rope_cos, rope_sin):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        d2 = self.head_dim // 2
        q_r, q_i = q[..., :d2], q[..., d2:]
        k_r, k_i = k[..., :d2], k[..., d2:]
        q_rot = torch.cat([q_r*rope_cos - q_i*rope_sin, q_r*rope_sin + q_i*rope_cos], dim=-1)
        k_rot = torch.cat([k_r*rope_cos - k_i*rope_sin, k_r*rope_sin + k_i*rope_cos], dim=-1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        out = F.scaled_dot_product_attention(
            q_rot.transpose(1,2), k_rot.transpose(1,2), v.transpose(1,2),
            attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, scale=self.scale)
        return self.proj(out.transpose(1,2).contiguous().view(B, T, D))

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=4, ff_mult=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*ff_mult, bias=False), nn.GELU(),
                                 nn.Linear(dim*ff_mult, dim, bias=False), nn.Dropout(dropout))
    def forward(self, x, rc, rs):
        return x + self.mlp(self.norm2(x + self.attn(self.norm1(x), rc, rs)))

class ConvFormer(nn.Module):
    def __init__(self):
        super().__init__()
        c = CFG; dim = c['dim']; n_axes = len(c['fsq_levels'])
        self.stem_conv = nn.Conv1d(c['mel_bins'], dim, 7, bias=False); self.stem_lc = 6
        self.downsample = nn.Conv1d(dim, dim, 2, stride=2)
        dilations = [1, 2, 4, 8][:c['conv_layers']]
        if len(dilations) < c['conv_layers']: dilations += [1]*(c['conv_layers']-len(dilations))
        self.conv_blocks = nn.ModuleList([ConvNeXtBlock(dim, 7, d, c['conv_expansion']) for d in dilations])
        self.content_proj = nn.Linear(c['content_dim'], dim, bias=False)
        self.rope = None
        self.xform = nn.ModuleList([TransformerBlock(dim, c['n_heads'], 2, c['dropout']) for _ in range(c['transformer_layers'])])
        self.axis_emb = nn.Embedding(n_axes, c['axis_emb_dim'])
        hh = c['axis_head_hidden']
        self.axis_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(dim+c['axis_emb_dim'], hh, bias=False), nn.GELU(),
            nn.Linear(hh, hh//2, bias=False), nn.GELU(), nn.Linear(hh//2, 1, bias=True)) for _ in range(n_axes)])
        self.proj_out = nn.Linear(n_axes, c['content_dim'])
        self.ordinal_heads = nn.ModuleList([nn.Linear(dim, L, bias=True) for L in c['fsq_levels']])
        self.residual_head = nn.Sequential(nn.Linear(dim, 256, bias=False), nn.GELU(), nn.Linear(256, c['content_dim'], bias=True))
        self.residual_gate = nn.Parameter(torch.tensor(0.0))
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)

    def _rope(self, T, dev):
        if self.rope is not None and self.rope[0].shape[1] >= T: return self.rope
        hd = CFG['dim'] // CFG['n_heads']; mx = max(T, 512)
        th = 1.0/(10000**(torch.arange(0, hd, 2, device=dev)/hd))
        fr = torch.arange(mx, device=dev).unsqueeze(1) * th.unsqueeze(0)
        cos = fr.cos().unsqueeze(0).unsqueeze(2); sin = fr.sin().unsqueeze(0).unsqueeze(2)
        self.rope = (cos, sin); return self.rope

    def forward(self, mel, content=None):
        B, _, T50 = mel.shape; dim = CFG['dim']
        h = self.stem_conv(F.pad(mel, (self.stem_lc,0))).transpose(1,2)
        h = self.downsample(h.transpose(1,2)).transpose(1,2); T25 = h.shape[1]
        if content is not None:
            c = self.content_proj(content)
            if c.shape[1] != T25: c = F.interpolate(c.transpose(1,2), size=T25, mode='nearest').transpose(1,2)
            h = h + c
        for blk in self.conv_blocks: h = blk(h)
        rc, rs = self._rope(T25, h.device)
        for blk in self.xform: h = blk(h, rc[:,:T25,:,:], rs[:,:T25,:,:])
        outs = []
        for i, head in enumerate(self.axis_heads):
            emb = self.axis_emb(torch.tensor(i, device=h.device)).view(1,1,-1).expand(B, T25, -1)
            outs.append(head(torch.cat([h, emb], dim=-1)))
        codes = torch.cat(outs, dim=-1)
        half = torch.tensor([(L-1)/2 for L in CFG['fsq_levels']], device=codes.device, dtype=codes.dtype)
        cs = codes - half.view(1,1,-1); cq = cs.round().clamp(-half, half) + half
        codes_q = cs + (cq - cs).detach()
        pre_res = self.proj_out(codes_q).transpose(1,2)
        ord_out = torch.cat([oh(h).transpose(1,2) for oh in self.ordinal_heads], dim=1)
        residual = self.residual_head(h)
        gate = self.residual_gate.sigmoid()
        proj = pre_res + gate * residual.transpose(1,2)
        return codes, proj, ord_out, pre_res, gate

# ── Data + Training ──
from astrape.data import MioContentDataset, ContentCollator, speaker_disjoint_split
from astrape.flat_ctc_training import speaker_balanced_subset
from astrape.mcss_training import indices_to_codes

def compute_loss(cp, pp, ol, gate, ti, ct, mask, fsq):
    T = min(cp.shape[1], pp.shape[2], ti.shape[1], ct.shape[1], mask.shape[1])
    cp=cp[:,:T]; pp=pp[:,:,:T]; ol=ol[:,:,:T]; ti=ti[:,:T]; mask=mask[:,:T]; ct=ct[:,:T,:].transpose(1,2)
    tgt = indices_to_codes(ti, fsq).float().to(cp.device)
    mf = mask.float(); tf = mf.sum().clamp(min=1)
    aw = torch.tensor([1.0,1.0,1.0,5.0,7.0], device=cp.device)
    cl = ((cp - tgt).abs() * mf.unsqueeze(-1) * aw.view(1,1,-1)).sum() / tf / len(fsq)
    ctl = ((pp - ct)**2 * mf.unsqueeze(1)).mean()
    offset = 0; ols = []
    for a, L in enumerate(fsq):
        t = tgt[..., a].long(); v = mask & (t>=0) & (t<L)
        if v.sum()>1: ols.append(F.cross_entropy(ol[:,offset:offset+L,:].transpose(1,2)[v], t[v], reduction='mean'))
        offset += L
    ol_ = torch.stack(ols).mean() if ols else torch.tensor(0.0, device=cp.device)
    cs = F.cosine_similarity(pp, ct, dim=1)
    return cl + ol_*0.5 + ctl*0.1 + (1.0 - cs*mf).mean()*0.1, {
        'code_l1': cl.item(), 'cos5': cs.mean().item(), 'ord': ol_.item(), 'content': ctl.item(), 'gate': gate.item()}

@torch.no_grad()
def evaluate(model, loader, dev, fsq):
    model.eval(); c5s=c7s=tot=0.0; pao=[0]*5; pan=0
    for batch in loader:
        if batch.token_indices is None: continue
        mel = batch.mel.to(dev); content = batch.content.to(dev)
        c, p, o, _, _ = model(mel, content)
        T = min(c.shape[1], p.shape[2], batch.token_indices.shape[1], batch.content.shape[1])
        pp = p[:,:,:T]; ct2 = batch.content[:,:T,:].to(dev).transpose(1,2)
        tgt = indices_to_codes(batch.token_indices[:,:T], fsq).float().to(dev)
        c5 = F.cosine_similarity(c[:,:T], tgt, dim=2)
        c7 = F.cosine_similarity(pp, ct2, dim=1)
        for b in range(mel.shape[0]):
            L = min(batch.input_lengths[b].item(), T)
            c5s += c5[b,:L].mean().item(); c7s += c7[b,:L].mean().item(); tot += 1
        pc = c[:,:T].cpu(); tc = tgt.cpu()
        for a in range(5):
            for b in range(mel.shape[0]):
                L = min(batch.input_lengths[b].item(), T)
                pa = pc[b,:L,a].round().clamp(0,7 if a<3 else 4)
                ta = tc[b,:L,a].round().clamp(0,7 if a<3 else 4)
                pao[a] += (pa==ta).sum().item()
        pan += sum(min(batch.input_lengths[b].item(),T) for b in range(mel.shape[0]))
    model.train(); ax = [pao[a]/pan if pan else 0 for a in range(5)]
    return c5s/tot if tot else 0, c7s/tot if tot else 0, ax

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu'); ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--target', type=float, default=0.95); ap.add_argument('--t0', type=int, default=20)
    ap.add_argument('--steps', type=int, default=30); ap.add_argument('--log-every', type=int, default=10)
    ap.add_argument('--data-dir', default='data/mio_vctk_full_compact')
    ap.add_argument('--resume', default=None)
    a = ap.parse_args()

    meta = np.load(Path(a.data_dir)/"meta.npz", allow_pickle=True)
    speakers = meta["spk_names"][:int(meta["n_samples"])].astype(str)
    train_idx, val_idx = speaker_disjoint_split(speakers, 0.10, 42)
    probe_idx = speaker_balanced_subset(val_idx, speakers, 64, 42)
    coll = ContentCollator(None, 42, include_transcripts=False)
    tl = DataLoader(MioContentDataset(a.data_dir,a.data_dir,train_idx), batch_size=2, shuffle=True, collate_fn=coll, num_workers=0)
    pl = DataLoader(MioContentDataset(a.data_dir,a.data_dir,probe_idx), batch_size=2, shuffle=False, collate_fn=coll, num_workers=0)
    print(f"Train={len(train_idx)} Probe={len(probe_idx)}", flush=True)

    model = ConvFormer().to(a.device)
    mcss = 'checkpoints/mcss.best.pt'
    if Path(mcss).exists():
        ckpt = torch.load(mcss, map_location=a.device, weights_only=False)
        sd = ckpt.get('state_dict', ckpt)
        if 'proj_out.weight' in sd:
            model.proj_out.weight.data.copy_(sd['proj_out.weight'])
            model.proj_out.bias.data.copy_(sd['proj_out.bias'])
            print("FSQ proj warm-started", flush=True)

    start_epoch = 0; best_probe = 0.0
    if a.resume and Path(a.resume).exists():
        ckpt = torch.load(a.resume, map_location=a.device, weights_only=False)
        sd = {k: v for k, v in ckpt['state_dict'].items() if v.shape == model.state_dict().get(k, torch.empty(0)).shape}
        model.load_state_dict(sd, strict=False)
        print(f"Loaded {len(sd)}/{len(ckpt['state_dict'])} compatible keys", flush=True)
        start_epoch = ckpt.get('epoch', 0) + 1
        best_probe = ckpt.get('probe_5d', 0.0)
        print(f"Resumed epoch {start_epoch-1} probe={best_probe:.4f}", flush=True)

    n = sum(p.numel() for p in model.parameters())
    print(f"Params={n:,} dim={CFG['dim']} conv={CFG['conv_layers']} xform={CFG['transformer_layers']} heads={CFG['n_heads']}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=a.t0, T_mult=1)
    if a.resume and Path(a.resume).exists():
        try: opt.load_state_dict(torch.load(a.resume, map_location=a.device, weights_only=False)['opt']); print("Opt resumed", flush=True)
        except: print("Opt fresh", flush=True)

    fsq = CFG['fsq_levels']; gs = start_epoch * a.steps; t0 = time.perf_counter()
    for epoch in range(start_epoch, a.epochs):
        model.train()
        for step, batch in enumerate(tl):
            if step >= a.steps: break
            if batch.token_indices is None: continue
            mel = batch.mel.to(a.device); content = batch.content.to(a.device)
            mask = batch.target_mask.to(a.device)
            c, p, o, _, gate = model(mel, content)
            loss, met = compute_loss(c, p, o, gate, batch.token_indices, content, mask, fsq)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); gs += 1
            if step % a.log_every == 0:
                dt = time.perf_counter() - t0; t0 = time.perf_counter()
                print(f"E{epoch:03d} step={step}/{a.steps} loss={loss.item():.4f} cos5={met['cos5']:.4f} content={met['content']:.4f} ord={met['ord']:.4f} gate={met['gate']:.4f} {dt:.3f}s/step", flush=True)

        pc5, pc7, pax = evaluate(model, pl, a.device, fsq)
        print(f"E{epoch:03d} probe_5d={pc5:.4f} cos768={pc7:.4f} ax=[{pax[0]:.3f} {pax[1]:.3f} {pax[2]:.3f} {pax[3]:.3f} {pax[4]:.3f}]", flush=True)
        ck = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt': opt.state_dict(), 'probe_5d': pc5, 'probe_768': pc7}
        torch.save(ck, 'checkpoints/convformer.last.pt')
        if pc5 > best_probe: best_probe = pc5; torch.save(ck, 'checkpoints/convformer.best.pt')
        if pc5 >= a.target: print(f"TARGET {a.target} REACHED at epoch {epoch}!", flush=True); break

if __name__ == '__main__': main()
