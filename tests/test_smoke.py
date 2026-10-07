import torch

from graphmoe.budget import compute_budget
from graphmoe.config import GraphMoEConfig, tiny_config
from graphmoe.model import GraphMoE
from graphmoe.quant import fake_quant_weight


def _model(**kw):
    torch.manual_seed(0)
    return GraphMoE(tiny_config(**kw)).eval()


def test_config_invariants():
    c = GraphMoEConfig()
    assert c.n_logical_paths == 256 and c.n_physical_factors == 20
    owners = c.kv_owner()
    assert sum(i == o for i, o in enumerate(owners)) == 20          # 5:3 -> 4 groups * 5
    assert owners[5] == owners[6] == owners[7] == 4


def test_forward_backward():
    m = GraphMoE(tiny_config())
    ids = torch.randint(0, 256, (2, 16))
    out = m(ids)
    assert out["logits"].shape == (2, 16, 256)
    assert out["routes"].shape == (8, 2, 16, 3)
    (out["logits"].mean() + out["aux_loss"]).backward()
    assert m.blocks[0].graph.r_domain.weight.grad is not None       # router 로 gradient 도달
    assert m.blocks[0].graph.domain[0].gate_up.weight.grad is not None


def test_cache_matches_full_forward():
    m = _model()
    ids = torch.randint(0, 256, (1, 24))
    full = m(ids, return_aux=False)
    caches = m.new_caches(1)
    m(ids[:, :16], caches, 0, return_aux=False)
    tail = m(ids[:, 16:], caches, 16, return_aux=False)
    assert torch.allclose(full[:, 16:], tail, atol=1e-3), (full[:, 16:] - tail).abs().max()


def test_sharer_layers_have_no_kv_proj():
    m = _model()
    for i, blk in enumerate(m.blocks):
        assert hasattr(blk.attn, "k_proj") == (blk.attn.owner == i)


def test_quant_levels():
    w = torch.randn(4, 64)
    for bits in (2, 3, 4):
        groups = fake_quant_weight(w, bits, 32).reshape(-1, 32)
        assert all(g.round(decimals=5).unique().numel() <= 2 ** bits for g in groups)   # STE 의 ulp 오차 제거


def test_param_count_matches_budget():
    cfg = tiny_config()
    m = GraphMoE(cfg)
    actual = sum(p.numel() for p in m.parameters())
    assert actual == compute_budget(cfg).total_params


def test_generate():
    out = _model().generate(torch.randint(0, 256, (1, 8)), max_new_tokens=5)
    assert out.shape == (1, 13)
