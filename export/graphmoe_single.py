"""Graph-MoE 7B — 단일 파일 모델 정의.

자동 생성: tools/bundle_model.py  (직접 수정 금지 — graphmoe/*.py 를 고친 뒤 재생성)
포함: config · quant · attention · experts · model · budget
실행: python graphmoe_single.py   -> 7B 스펙 파라미터/메모리 리포트 + tiny 모델 forward 확인
"""
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ======================================================================================
# config.py
# ======================================================================================
@dataclass
class GraphMoEConfig:
    # --- backbone ---
    vocab_size: int = 64000            # ASSUMPTION: 스펙에 없음
    n_layers: int = 32
    d_model: int = 2560
    n_heads: int = 20                  # Q heads
    n_kv_heads: int = 4                # GQA
    head_dim: int = 128                # 20 * 128 = 2560
    max_seq_len: int = 8192            # default context 8K
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True        # ASSUMPTION: lm_head = embedding (RAM 절약)

    # --- KV sharing: 매 `kv_group` 레이어 중 앞 `kv_own`개만 K/V 생성, 나머지는 재사용 (5:3) ---
    kv_group: int = 8
    kv_own: int = 5                    # 5 own : 3 share   (ASSUMPTION: 8-layer 블록 해석)
    kv_bits: int = 8                   # int8 KV cache

    # --- shared SwiGLU (모든 레이어에서 항상 활성) ---
    ffn_dim: int = 4096

    # --- per-layer expert graph (tier 별 expert 수 / rank) ---
    # rank: 원 스펙(512/384/192)은 총 3.0B 밖에 안 나와 D1 결정으로 확대 (WORKGUIDE §5). 총 6.76B / 활성 2.32B
    n_domain: int = 8
    r_domain: int = 1728
    n_op: int = 8
    r_op: int = 1280
    n_ctrl: int = 4
    r_ctrl: int = 640

    # --- quantization bits ---
    bits_graph: int = 2                # expert factors (Graph QAT)
    bits_ffn_edge: int = 3             # shared FFN: 첫/끝 레이어 구간
    bits_ffn_mid: int = 2              # shared FFN: 중간 레이어  (=> 2~3-bit)
    ffn_edge_layers: int = 4           # ASSUMPTION: 앞/뒤 4개 레이어는 3-bit
    bits_attn: int = 3
    bits_embed: int = 4
    group_size: int = 64               # 그룹 양자화 단위; scale 은 fp16
    qat: bool = True                   # False 면 fake-quant 비활성 (FP 학습/디버그)
    grad_ckpt: bool = False            # 블록 단위 activation checkpointing (학습 메모리 절약)

    # --- routing ---
    aux_loss_coef: float = 0.01        # load-balance loss 계수 (tier 별 합산)

    def __post_init__(self):
        assert self.n_heads * self.head_dim == self.d_model
        assert self.n_heads % self.n_kv_heads == 0
        assert 0 < self.kv_own <= self.kv_group

    # ---- layer maps ----
    def kv_owner(self) -> List[int]:
        """layer i 가 사용할 K/V 를 만든 레이어 index (자기 자신이면 owner)."""
        owners, last = [], 0
        for i in range(self.n_layers):
            if i % self.kv_group < self.kv_own:
                last = i
            owners.append(last)
        return owners

    def ffn_bits(self, layer: int) -> int:
        edge = layer < self.ffn_edge_layers or layer >= self.n_layers - self.ffn_edge_layers
        return self.bits_ffn_edge if edge else self.bits_ffn_mid

    @property
    def n_logical_paths(self) -> int:
        return self.n_domain * self.n_op * self.n_ctrl     # 8*8*4 = 256

    @property
    def n_physical_factors(self) -> int:
        return self.n_domain + self.n_op + self.n_ctrl     # 20


def mini_config(**kw) -> GraphMoEConfig:
    """파일럿 학습용 소형 (~수백M). 구조·비율은 7B 와 동일, 규모만 축소."""
    base = dict(
        vocab_size=32000, n_layers=16, d_model=640, n_heads=10, n_kv_heads=2, head_dim=64,
        max_seq_len=2048, ffn_dim=1024, r_domain=128, r_op=96, r_ctrl=48,
        group_size=64, ffn_edge_layers=2,
    )
    base.update(kw)
    return GraphMoEConfig(**base)


def get_preset(name: str, **kw) -> GraphMoEConfig:
    presets = {"7b": GraphMoEConfig, "mini": mini_config, "tiny": lambda **k: tiny_config(**k)}
    return presets[name](**kw)


def tiny_config(**kw) -> GraphMoEConfig:
    """CPU 스모크 테스트용 축소판 (구조는 동일)."""
    base = dict(
        vocab_size=256, n_layers=8, d_model=128, n_heads=4, n_kv_heads=2, head_dim=32,
        max_seq_len=128, ffn_dim=256, r_domain=64, r_op=48, r_ctrl=32,
        group_size=32, ffn_edge_layers=1,
    )
    base.update(kw)
    return GraphMoEConfig(**base)


# ======================================================================================
# quant.py
# ======================================================================================
def quantize_codes(w: torch.Tensor, bits: int, group: int):
    """(codes uint-range float [..., ng, g], scale [..., ng, 1], group 크기) — 학습/export 공통 경로.

    levels = (code - h + .5) * scale,  h = 2**(bits-1)  (2-bit => 4 levels, 0 없음)
    scale 은 fp16 으로 반올림 => QAT 출력 == 배포 dequant 출력.
    """
    shape = w.shape
    g = group if shape[-1] % group == 0 else shape[-1]
    wg = w.reshape(*shape[:-1], shape[-1] // g, g)
    half = 2 ** (bits - 1)
    amax = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (amax / (half - 0.5)).half().to(w.dtype).clamp_min(1e-8)
    codes = torch.floor(wg / scale).clamp(-half, half - 1) + half       # 0 .. 2**bits-1
    return codes, scale, g


def dequantize_codes(codes, scale, bits, shape):
    half = 2 ** (bits - 1)
    return ((codes - half + 0.5) * scale).reshape(shape)


def fake_quant_weight(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """마지막 차원을 group 단위로 묶은 대칭 mid-rise 양자화 + STE."""
    codes, scale, _ = quantize_codes(w, bits, group)
    q = dequantize_codes(codes, scale, bits, w.shape)
    return w + (q - w).detach()                      # STE


def fake_quant_int8_lastdim(x: torch.Tensor):
    """token/head 별 absmax int8. (int8 값, fp scale) 반환 — KV cache 용."""
    scale = (x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0).half().to(x.dtype)  # cache 와 동일하게 fp16 scale
    q = torch.round(x / scale).clamp(-127, 127)
    return q.to(torch.int8), scale


class QuantLinear(nn.Linear):
    """weight 를 forward 마다 fake-quant. bits=None 이면 일반 Linear."""

    def __init__(self, in_f, out_f, bits=None, group=64, qat=True, bias=False):
        super().__init__(in_f, out_f, bias=bias)
        self.bits, self.group, self.qat = bits, group, qat

    def qweight(self):
        if self.bits is None or not self.qat:
            return self.weight
        return fake_quant_weight(self.weight, self.bits, self.group)

    def forward(self, x):
        return F.linear(x, self.qweight(), self.bias)

    def extra_repr(self):
        return super().extra_repr() + f", bits={self.bits}"


class QuantEmbedding(nn.Embedding):
    def __init__(self, num, dim, bits=4, group=64, qat=True):
        super().__init__(num, dim)
        self.bits, self.group, self.qat = bits, group, qat

    def qweight(self):
        return fake_quant_weight(self.weight, self.bits, self.group) if self.qat else self.weight

    def forward(self, ids):
        return F.embedding(ids, self.qweight())


class RMSNorm(nn.Module):
    """Norm 은 FP16 유지 대상 (배포 시 .half())."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


# ======================================================================================
# attention.py
# ======================================================================================
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


# ======================================================================================
# experts.py
# ======================================================================================
class DomainFactor(nn.Module):
    def __init__(self, d, r, **qk):
        super().__init__()
        self.gate_up = QuantLinear(d, 2 * r, **qk)
        self.down = QuantLinear(r, d, **qk)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        z = F.silu(g) * u
        return z, self.down(z)


class LatentFactor(nn.Module):
    """Operation / Control 공용: latent(r_in) -> latent(r) , latent(r) -> d_model."""

    def __init__(self, r_in, r, d, **qk):
        super().__init__()
        self.mix = QuantLinear(r_in, r, **qk)
        self.down = QuantLinear(r, d, **qk)

    def forward(self, z):
        z = F.silu(self.mix(z))
        return z, self.down(z)


def _dispatch(experts, idx, inp, r_out, d):
    """top-1 dispatch: idx[n] 번 expert 로 토큰을 보내고 (latent, out) 을 원래 순서로 되돌린다."""
    N = inp.shape[0]
    if N == 0:
        return inp.new_zeros(0, r_out), inp.new_zeros(0, d)
    order = idx.argsort(stable=True)                       # expert 별로 연속 배치 -> expert 당 matmul 1회
    counts = torch.bincount(idx, minlength=len(experts)).tolist()
    lats, outs = [], []
    for m, chunk in zip(experts, inp[order].split(counts)):
        if chunk.shape[0]:
            l, o = m(chunk)
            lats.append(l)
            outs.append(o)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(N, device=order.device)      # 원래 토큰 순서로 복원
    return torch.cat(lats)[inv], torch.cat(outs)[inv]


def _balance_loss(probs, idx, n):
    """Switch-style load balance: n * sum_e f_e * P_e."""
    f = F.one_hot(idx, n).float().mean(0)
    return n * (f * probs.float().mean(0)).sum()


class ExpertGraph(nn.Module):
    def __init__(self, cfg: GraphMoEConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        qk = dict(bits=cfg.bits_graph, group=cfg.group_size, qat=cfg.qat)
        self.domain = nn.ModuleList(DomainFactor(d, cfg.r_domain, **qk) for _ in range(cfg.n_domain))
        self.op = nn.ModuleList(LatentFactor(cfg.r_domain, cfg.r_op, d, **qk) for _ in range(cfg.n_op))
        self.ctrl = nn.ModuleList(LatentFactor(cfg.r_op, cfg.r_ctrl, d, **qk) for _ in range(cfg.n_ctrl))

        # Router: FP16 유지 대상 (양자화 없음). edge_* = 그래프 간선(이전 tier 선택 -> 다음 tier logit prior)
        self.r_domain = nn.Linear(d, cfg.n_domain, bias=False)
        self.r_op = nn.Linear(d, cfg.n_op, bias=False)
        self.r_ctrl = nn.Linear(d, cfg.n_ctrl, bias=False)
        self.r_ctrl_gate = nn.Linear(d, 1)                       # control top-0/1 on/off
        self.edge_do = nn.Parameter(torch.zeros(cfg.n_domain, cfg.n_op))
        self.edge_oc = nn.Parameter(torch.zeros(cfg.n_op, cfg.n_ctrl))

    def forward(self, x):
        """x [B,T,d] -> (y [B,T,d], aux_loss, route [B,T,3]  (ctrl 꺼짐 = -1))"""
        B, T, d = x.shape
        cfg = self.cfg
        xf = x.reshape(-1, d)

        # ---- routing (top-1 x3, control 은 top-0/1) ----
        p_d = F.softmax(self.r_domain(xf).float(), -1)
        i = p_d.argmax(-1)
        p_o = F.softmax((self.r_op(xf) + self.edge_do[i]).float(), -1)
        j = p_o.argmax(-1)
        p_c = F.softmax((self.r_ctrl(xf) + self.edge_oc[j]).float(), -1)
        k = p_c.argmax(-1)
        g_soft = torch.sigmoid(self.r_ctrl_gate(xf).float()).squeeze(-1)
        on = (g_soft > 0.5)
        g = on.float() + g_soft - g_soft.detach()                 # hard on/off, soft gradient (STE)

        gd = p_d.gather(1, i[:, None]).to(x.dtype)                # 선택된 expert 확률 = router 로 가는 gradient 경로
        go = p_o.gather(1, j[:, None]).to(x.dtype)

        # ---- expert graph ----
        u, y_d = _dispatch(self.domain, i, xf, cfg.r_domain, d)
        v, y_o = _dispatch(self.op, j, u, cfg.r_op, d)
        y = gd * y_d + go * y_o
        sel = on.nonzero(as_tuple=True)[0]
        if sel.numel() > 0:                                       # top-0 토큰은 control 연산을 건너뜀
            _, y_c = _dispatch(self.ctrl, k[sel], v[sel], cfg.r_ctrl, d)
            y = y.index_add(0, sel, g[sel, None].to(x.dtype) * y_c)

        aux = (_balance_loss(p_d, i, cfg.n_domain) + _balance_loss(p_o, j, cfg.n_op)
               + (_balance_loss(p_c[sel], k[sel], cfg.n_ctrl) if sel.numel() > 0 else x.new_zeros(())))
        route = torch.stack([i, j, torch.where(on, k, torch.full_like(k, -1))], -1).view(B, T, 3)
        return y.view(B, T, d), aux, route


# ======================================================================================
# model.py
# ======================================================================================
class SharedSwiGLU(nn.Module):
    def __init__(self, cfg: GraphMoEConfig, layer: int):
        super().__init__()
        kw = dict(bits=cfg.ffn_bits(layer), group=cfg.group_size, qat=cfg.qat)
        self.gate_up = QuantLinear(cfg.d_model, 2 * cfg.ffn_dim, **kw)
        self.down = QuantLinear(cfg.ffn_dim, cfg.d_model, **kw)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, cfg: GraphMoEConfig, layer: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GQAttention(cfg, layer)
        self.shared = SharedSwiGLU(cfg, layer)
        self.graph = ExpertGraph(cfg)

    def forward(self, x, rope, kv):
        x = x + self.attn(self.attn_norm(x), rope, kv)
        h = self.ffn_norm(x)
        y, aux, route = self.graph(h)
        return x + self.shared(h) + y, aux, route


class GraphMoE(nn.Module):
    def __init__(self, cfg: GraphMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = QuantEmbedding(cfg.vocab_size, cfg.d_model, cfg.bits_embed, cfg.group_size, cfg.qat)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        if not cfg.tie_embeddings:
            self.lm_head = QuantLinear(cfg.d_model, cfg.vocab_size, cfg.bits_embed, cfg.group_size, cfg.qat)
        self.register_buffer("_rope_cos", None, persistent=False)
        self.register_buffer("_rope_sin", None, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def new_caches(self, batch=1, device=None):
        """owner 레이어에만 int8 cache 를 만든다 (sharer 는 owner 것을 재사용)."""
        owners = set(self.cfg.kv_owner())
        return {i: Int8KVCache(self.cfg, batch, device) for i in owners}

    def _rope(self, device):
        if self._rope_cos is None or self._rope_cos.device != device:
            self._rope_cos, self._rope_sin = rope_cache(self.cfg, device)
        return self._rope_cos, self._rope_sin

    def forward(self, ids, caches=None, past_len=0, return_aux=True):
        x = self.embed(ids)
        rope = self._rope(ids.device)
        kv = KVContext(caches, past_len)
        aux_total, routes = x.new_zeros(()), []
        for blk in self.blocks:
            if self.cfg.grad_ckpt and self.training and caches is None:
                x, aux, route = checkpoint(blk, x, rope, kv, use_reentrant=False)
            else:
                x, aux, route = blk(x, rope, kv)
            aux_total = aux_total + aux
            routes.append(route)
        x = self.norm(x)
        w = self.embed.qweight() if self.cfg.tie_embeddings else None
        logits = F.linear(x, w) if w is not None else self.lm_head(x)
        out = {"logits": logits, "aux_loss": aux_total * self.cfg.aux_loss_coef / self.cfg.n_layers,
               "routes": torch.stack(routes, 0)}                      # [L,B,T,3]
        return out if return_aux else logits

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=32):
        """greedy decode (int8 KV cache 사용)."""
        self.eval()
        caches = self.new_caches(ids.shape[0], ids.device)
        logits = self(ids, caches, 0, return_aux=False)
        out, past = ids, ids.shape[1]
        for _ in range(max_new_tokens):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], 1)
            logits = self(nxt, caches, past, return_aux=False)
            past += 1
        return out


# ======================================================================================
# budget.py
# ======================================================================================
GB = 1e9


def _scale_bits(bits, group):                      # fp16 scale 오버헤드 (bit/weight)
    return bits + 16 / group


@dataclass
class Budget:
    params: dict          # 구성요소별 총 파라미터
    active: float         # 토큰당 활성 파라미터
    bytes_: dict          # 구성요소별 상주 bytes
    kv_bytes: int

    @property
    def total_params(self):
        return sum(self.params.values())

    @property
    def total_bytes(self):
        return sum(self.bytes_.values()) + self.kv_bytes


def compute_budget(cfg: GraphMoEConfig, ctx: int = None, batch: int = 1) -> Budget:
    ctx = ctx or cfg.max_seq_len
    d, L, g = cfg.d_model, cfg.n_layers, cfg.group_size
    H, Hk, D = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    owners = cfg.kv_owner()
    n_own = sum(1 for i, o in enumerate(owners) if i == o)

    attn = L * 2 * d * H * D + n_own * 2 * d * Hk * D
    ffn_layers = [3 * d * cfg.ffn_dim for _ in range(L)]
    dom = cfg.n_domain * 3 * d * cfg.r_domain
    op = cfg.n_op * (cfg.r_domain * cfg.r_op + cfg.r_op * d)
    ctl = cfg.n_ctrl * (cfg.r_op * cfg.r_ctrl + cfg.r_ctrl * d)
    graph = L * (dom + op + ctl)
    router = L * (d * (cfg.n_domain + cfg.n_op + cfg.n_ctrl + 1) + 1
                  + cfg.n_domain * cfg.n_op + cfg.n_op * cfg.n_ctrl)
    emb = cfg.vocab_size * d * (1 if cfg.tie_embeddings else 2)
    norms = (2 * L + 1) * d

    params = dict(attn=attn, shared_ffn=sum(ffn_layers), expert_graph=graph, router=router,
                  embedding=emb, norm=norms)

    # active/token: attn + shared ffn + (1 domain + 1 op + 1 ctrl) + router + embedding row(무시) + lm_head
    active = (attn + sum(ffn_layers) + router + norms
              + L * (3 * d * cfg.r_domain + cfg.r_domain * cfg.r_op + cfg.r_op * d
                     + cfg.r_op * cfg.r_ctrl + cfg.r_ctrl * d)
              + cfg.vocab_size * d)

    ffn_bytes = sum(3 * d * cfg.ffn_dim * _scale_bits(cfg.ffn_bits(i), g) / 8 for i in range(L))
    bytes_ = dict(
        attn=attn * _scale_bits(cfg.bits_attn, g) / 8,
        shared_ffn=ffn_bytes,
        expert_graph=graph * _scale_bits(cfg.bits_graph, g) / 8,
        router=router * 2, norm=norms * 2,                         # FP16
        embedding=emb * _scale_bits(cfg.bits_embed, g) / 8,
    )
    kv = n_own * batch * 2 * Hk * D * ctx * (cfg.kv_bits / 8 + 2 / D)   # int8 + fp16 scale/token/head
    return Budget(params, active, bytes_, int(kv))


def report(cfg: GraphMoEConfig, ctx: int = None):
    b = compute_budget(cfg, ctx)
    print(f"layers={cfg.n_layers}  d_model={cfg.d_model}  logical paths/layer={cfg.n_logical_paths}  "
          f"physical factors/layer={cfg.n_physical_factors}")
    print(f"KV-owner layers: {sum(1 for i,o in enumerate(cfg.kv_owner()) if i==o)}/{cfg.n_layers}")
    print("-- params")
    for k, v in b.params.items():
        print(f"  {k:13s} {v/GB:7.3f} B")
    print(f"  {'TOTAL':13s} {b.total_params/GB:7.3f} B   (target ≈ 6.8B)")
    print(f"  {'ACTIVE/token':13s} {b.active/GB:7.3f} B   (target ≈ 2.4B)")
    print(f"-- resident memory @ctx={ctx or cfg.max_seq_len}")
    for k, v in b.bytes_.items():
        print(f"  {k:13s} {v/GB:7.3f} GB")
    print(f"  {'kv_cache':13s} {b.kv_bytes/GB:7.3f} GB")
    print(f"  {'TOTAL':13s} {b.total_bytes/GB:7.3f} GB   (target ≈ 4.3~4.8 GB)")
    return b


# ======================================================================================
if __name__ == "__main__":
    report(GraphMoEConfig())                                   # 7B 스펙 수치 (모델은 만들지 않음)
    torch.manual_seed(0)
    m = GraphMoE(tiny_config()).eval()                          # 구조는 같고 규모만 축소한 확인용 모델
    ids = torch.randint(0, 256, (1, 16))
    out = m(ids)
    print(f"\n[tiny forward] logits {tuple(out['logits'].shape)}  routes {tuple(out['routes'].shape)}"
          f"  params {sum(p.numel() for p in m.parameters()):,}")
