"""Graph-MoE 7B 설정. 스펙에 없는 값은 ASSUMPTION 주석으로 표시."""
from dataclasses import dataclass, field
from typing import List


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
    n_domain: int = 8
    r_domain: int = 512
    n_op: int = 8
    r_op: int = 384
    n_ctrl: int = 4
    r_ctrl: int = 192

    # --- quantization bits ---
    bits_graph: int = 2                # expert factors (Graph QAT)
    bits_ffn_edge: int = 3             # shared FFN: 첫/끝 레이어 구간
    bits_ffn_mid: int = 2              # shared FFN: 중간 레이어  (=> 2~3-bit)
    ffn_edge_layers: int = 4           # ASSUMPTION: 앞/뒤 4개 레이어는 3-bit
    bits_attn: int = 3
    bits_embed: int = 4
    group_size: int = 64               # 그룹 양자화 단위; scale 은 fp16
    qat: bool = True                   # False 면 fake-quant 비활성 (FP 학습/디버그)

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


def tiny_config(**kw) -> GraphMoEConfig:
    """CPU 스모크 테스트용 축소판 (구조는 동일)."""
    base = dict(
        vocab_size=256, n_layers=8, d_model=128, n_heads=4, n_kv_heads=2, head_dim=32,
        max_seq_len=128, ffn_dim=256, r_domain=64, r_op=48, r_ctrl=32,
        group_size=32, ffn_edge_layers=1,
    )
    base.update(kw)
    return GraphMoEConfig(**base)
