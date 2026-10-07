"""파라미터 / 활성 파라미터 / 상주 RAM 을 모델 생성 없이 계산 (스펙 대비 점검용)."""
from dataclasses import dataclass

from .config import GraphMoEConfig

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


if __name__ == "__main__":
    report(GraphMoEConfig())
