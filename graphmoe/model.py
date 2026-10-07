"""Graph-MoE 본체: pre-norm 블록 = Attention(GQA, KV-share) + [Shared SwiGLU ∥ Expert Graph]."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import GQAttention, Int8KVCache, KVContext, rope_cache
from .config import GraphMoEConfig
from .experts import ExpertGraph
from .quant import QuantEmbedding, QuantLinear, RMSNorm


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
