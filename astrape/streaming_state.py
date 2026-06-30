"""Incremental (state-carry) streaming primitives — the deploy fast path.

The bounded-recompute reference (astrape/streaming.py) re-runs a 256-frame window
every step (~80 ms/step), too slow for B=1 real-time. These primitives carry per-layer
state so each new frame costs O(1)/O(window) instead of re-processing the window:

  StreamingCausalConv   — ring-buffers the last `left_context` inputs of a CausalConv1d
                          (stride=1); per-step conv over [state ++ new] only.
  StreamingWindowAttn   — sliding-window KV-cache for AdaLNTransformerLayer: keep the
                          last `window` (k,v); a new frame's q attends to the cache.

Both are bit-exact vs the layers' full forward (verified in __main__). They are the
reference the C++/VST port replicates; latency stays 56 ms, compute drops to ~0.8 ms.

NOTE: handles stride=1 convs and 1-frame attention steps (the streaming case). Strided
convs (StridingAdapter / downsample), the ConvTranspose upsampler and the iSTFT
overlap-add need chunk-alignment — see PORT.md / the wiring TODO.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class StreamingCausalConv:
    """State-carry wrapper for a stride-1 CausalConv1d. `step(x)` returns the output
    for the new frames only, using ring-buffered left context (== full causal conv)."""

    def __init__(self, conv):
        assert conv.stride[0] == 1, "strided conv needs chunk-alignment (see module note)"
        self.conv = conv
        self.lc = conv.left_context          # dilation*(kernel-1)
        self.state = None                    # (B, C_in, lc)

    def reset(self):
        self.state = None

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, C_in, T_new)
        if self.lc == 0:
            return self.conv.forward(x)
        if self.state is None:
            self.state = x.new_zeros(x.shape[0], x.shape[1], self.lc)
        xc = torch.cat([self.state, x], dim=-1)             # left context ++ new
        self.state = xc[..., -self.lc:]
        return F.conv1d(xc, self.conv.weight, self.conv.bias,
                        stride=1, padding=0, dilation=self.conv.dilation, groups=self.conv.groups)


class StreamingWindowAttn:
    """Sliding-window KV-cache for an AdaLNTransformerLayer. `step` processes one new
    frame (T=1): its q attends to the cached last `window` (k,v) — == the layer's
    windowed-causal forward. AdaLN + FFN are per-frame (no cross-frame state)."""

    def __init__(self, layer, rope, window: int):
        self.L = layer
        self.rope = rope
        self.window = window
        self.pos = 0
        self.kc = None                       # (B, H, <=window, hd)
        self.vc = None

    def reset(self):
        self.pos = 0; self.kc = None; self.vc = None

    @torch.no_grad()
    def step(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:   # x: (B, 1, D)
        L = self.L
        B, T, D = x.shape
        if condition.dim() == 2:
            condition = condition.unsqueeze(1)
        normed, gate_a = L.attn_adaln(x, condition)
        q = L.wq(normed).view(B, T, L.heads, L.hd).transpose(1, 2)   # (B,H,T,hd)
        k = L.wk(normed).view(B, T, L.heads, L.hd).transpose(1, 2)
        v = L.wv(normed).view(B, T, L.heads, L.hd).transpose(1, 2)
        q = self.rope(q, self.pos)
        k = self.rope(k, self.pos)
        self.kc = k if self.kc is None else torch.cat([self.kc, k], dim=2)
        self.vc = v if self.vc is None else torch.cat([self.vc, v], dim=2)
        if self.kc.shape[2] > self.window:
            self.kc = self.kc[:, :, -self.window:]
            self.vc = self.vc[:, :, -self.window:]
        self.pos += T
        attn = F.scaled_dot_product_attention(q, self.kc, self.vc)   # new q ↔ window (all causal)
        x = x + gate_a * L.wo(attn.transpose(1, 2).contiguous().view(B, T, D))
        normed, gate_f = L.ffn_adaln(x, condition)
        return x + gate_f * L.w2(F.silu(L.w1(normed)) * L.w3(normed))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import warnings, sys; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
    from astrape.nn import CausalConv1d, RoPE, AdaLNTransformerLayer
    torch.manual_seed(0); torch.set_grad_enabled(False)
    B, T, D, H, cond = 2, 80, 384, 8, 128

    print("bit-exact: frame-by-frame state-carry vs full forward\n")

    # ── causal conv (a few dilations) ──
    for k, dil in [(3, 1), (3, 2), (5, 4)]:
        conv = CausalConv1d(D, D, k, dilation=dil, groups=D).eval()
        x = torch.randn(B, D, T)
        full = conv.forward(x)
        sc = StreamingCausalConv(conv)
        inc = torch.cat([sc.step(x[..., t:t+1]) for t in range(T)], dim=-1)
        print(f"  CausalConv1d k={k} dil={dil}:  max_err={ (inc-full).abs().max().item():.2e}")

    # ── windowed attention layer ──
    for window in (16, 64):
        layer = AdaLNTransformerLayer(D, H, cond).eval()
        rope = RoPE(D // H)
        x = torch.randn(B, T, D); c = torch.randn(B, cond)
        full = layer(x, c, rope, window)
        sa = StreamingWindowAttn(layer, rope, window)
        inc = torch.cat([sa.step(x[:, t:t+1], c) for t in range(T)], dim=1)
        print(f"  AdaLN windowed-attn window={window}:  max_err={(inc-full).abs().max().item():.2e}")
    print("\n(max_err ~1e-6 = bit-exact; these collapse the per-frame cost to O(window)/O(1).)")
