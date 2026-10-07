"""GQA + 레이어 간 KV 공유 + int8 KV cache."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GraphMoEConfig
from .quant import QuantLinear, fake_quant_int8_lastdim


def rope_cache(cfg: GraphMoEConfig, device=None):
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, device=device).float() / cfg.head_dim))
    t = torch.arange(cfg.max_seq_len, device=device).float()
    f = torch.outer(t, inv)
    return f.cos(), f.sin()                                  # [S, D/2]


def apply_rope(x, cos, sin):                                 # x [B,H,T,D]
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class Int8KVCache:
    """owner 레이어 1개당 1개. int8 값 + fp16 scale 을 max_seq_len 만큼 미리 할당."""

    def __init__(self, cfg: GraphMoEConfig, batch: int, device=None):
        shp = (batch, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim)
        self.k = torch.zeros(shp, dtype=torch.int8, device=device)
        self.v = torch.zeros(shp, dtype=torch.int8, device=device)
        self.ks = torch.zeros(*shp[:3], 1, dtype=torch.float16, device=device)
        self.vs = torch.zeros(*shp[:3], 1, dtype=torch.float16, device=device)
        self.len = 0

    def append(self, k, v):
        T = k.shape[2]
        kq, ks = fake_quant_int8_lastdim(k)
        vq, vs = fake_quant_int8_lastdim(v)
        s, e = self.len, self.len + T
        self.k[:, :, s:e], self.v[:, :, s:e] = kq, vq
        self.ks[:, :, s:e], self.vs[:, :, s:e] = ks.half(), vs.half()
        self.len = e

    def get(self, dtype):
        n = self.len
        return ((self.k[:, :, :n].to(dtype) * self.ks[:, :, :n].to(dtype)),
                (self.v[:, :, :n].to(dtype) * self.vs[:, :, :n].to(dtype)))

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.k, self.v, self.ks, self.vs))


class KVContext:
    """forward 1회 동안 owner 레이어의 K/V 를 sharer 레이어에 전달.

    caches 가 있으면 int8 cache(추론), 없으면 같은 양자화를 거친 임시 텐서(학습)."""

    def __init__(self, caches=None, past_len: int = 0):
        self.caches, self.past_len, self._tmp = caches, past_len, {}

    def write(self, layer, k, v):
        if self.caches is not None:
            c = self.caches[layer]
            c.append(k, v)
            kv = c.get(k.dtype)
        else:
            kq, ks = fake_quant_int8_lastdim(k)
            vq, vs = fake_quant_int8_lastdim(v)
            kv = (kq.to(k.dtype) * ks.to(k.dtype), vq.to(v.dtype) * vs.to(v.dtype))
            kv = (k + (kv[0] - k).detach(), v + (kv[1] - v).detach())   # STE
        self._tmp[layer] = kv
        return kv

    def read(self, owner):
        return self._tmp[owner]


class GQAttention(nn.Module):
    def __init__(self, cfg: GraphMoEConfig, layer: int):
        super().__init__()
        self.cfg, self.layer = cfg, layer
        self.owner = cfg.kv_owner()[layer]
        self.owns_kv = self.owner == layer
        H, Hk, D, d = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.d_model
        kw = dict(bits=cfg.bits_attn, group=cfg.group_size, qat=cfg.qat)
        self.q_proj = QuantLinear(d, H * D, **kw)
        self.o_proj = QuantLinear(H * D, d, **kw)
        if self.owns_kv:                                     # sharer 레이어는 K/V proj 자체가 없음
            self.k_proj = QuantLinear(d, Hk * D, **kw)
            self.v_proj = QuantLinear(d, Hk * D, **kw)

    def forward(self, x, rope, kv: KVContext):
        B, T, _ = x.shape
        cfg = self.cfg
        H, Hk, D = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        cos, sin = (r[kv.past_len: kv.past_len + T] for r in rope)

        q = apply_rope(self.q_proj(x).view(B, T, H, D).transpose(1, 2), cos, sin)
        if self.owns_kv:
            k = apply_rope(self.k_proj(x).view(B, T, Hk, D).transpose(1, 2), cos, sin)
            v = self.v_proj(x).view(B, T, Hk, D).transpose(1, 2)
            k, v = kv.write(self.layer, k, v)
        else:
            k, v = kv.read(self.owner)

        S = k.shape[2]
        pos_q = kv.past_len + torch.arange(T, device=x.device)
        mask = torch.arange(S, device=x.device)[None, :] <= pos_q[:, None]       # causal [T,S]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, H * D))
