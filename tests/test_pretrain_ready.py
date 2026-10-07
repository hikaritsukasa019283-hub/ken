import copy

import torch

from graphmoe.config import tiny_config
from graphmoe.data import BinSampler, ByteTokenizer, write_bin, write_synthetic_bin
from graphmoe.diagnostics import route_stats
from graphmoe.model import GraphMoE
from graphmoe.packing import (dequant_entry, export_packed, load_packed, pack_bits,
                              packed_nbytes, unpack_bits)
from graphmoe.quant import fake_quant_weight
from graphmoe.train import lr_at, parse, train


def test_pack_roundtrip():
    for bits in (2, 3, 4):
        for n in (8, 24, 64, 37):
            c = torch.randint(0, 2 ** bits, (5, n), dtype=torch.uint8)
            p = pack_bits(c, bits)
            assert p.shape[-1] == -(-n * bits // 8)
            assert torch.equal(unpack_bits(p, bits, n), c)


def test_export_matches_qat_weights():
    m = GraphMoE(tiny_config()).eval()
    blob = export_packed(m)
    mods = dict(m.named_modules())
    for name, e in blob["quant"].items():
        w = mods[name].weight
        ref = fake_quant_weight(w.float(), e["bits"], mods[name].group)
        assert torch.allclose(dequant_entry(e), ref, atol=1e-6), name
    m2 = GraphMoE(tiny_config(qat=False)).eval()
    load_packed(m2, blob)
    ids = torch.randint(0, 256, (1, 12))
    assert m2(ids, return_aux=False).shape == (1, 12, 256)
    fp32_bytes = sum(p.numel() * 4 for p in m.parameters())
    assert packed_nbytes(blob) < fp32_bytes / 6          # 2~4bit + fp16 이므로 크게 작아져야 함


def test_dispatch_matches_per_token_reference():
    m = GraphMoE(tiny_config()).eval()
    g = m.blocks[0].graph
    x = torch.randn(1, 6, 128)
    y, _, route = g(x)
    for t in range(6):                                    # 토큰 1개씩 따로 돌려도 동일해야 함
        yt, _, rt = g(x[:, t:t + 1])
        assert torch.equal(rt[0, 0], route[0, t])
        assert torch.allclose(yt[0, 0], y[0, t], atol=1e-5)


def test_grad_ckpt_same_grads():
    torch.manual_seed(0)
    a = GraphMoE(tiny_config()).train()
    b = copy.deepcopy(a); b.cfg.grad_ckpt = True
    ids = torch.randint(0, 256, (2, 12))
    for m in (a, b):
        o = m(ids); (o["logits"].square().mean() + o["aux_loss"]).backward()
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.grad is not None:
            assert torch.allclose(pa.grad, pb.grad, atol=1e-5, rtol=1e-3), n


def test_data_pipeline(tmp_path):
    p = str(tmp_path / "t.bin")
    n = write_bin(["hello world " * 20, "another doc " * 20], ByteTokenizer(), p)
    s = BinSampler(p, 16, seed=1)
    x, y = s.get_batch(4)
    assert x.shape == (4, 16) and torch.equal(x[:, 1:], y[:, :-1]) and n > 400


def test_route_stats_and_lr():
    cfg = tiny_config()
    m = GraphMoE(cfg).eval()
    st = route_stats(m(torch.randint(0, 256, (2, 32)))["routes"], cfg)
    assert 0 <= st["domain_entropy"] <= 1 and st["paths_used"] >= 1
    assert lr_at(0, 100, 1e-3, 10) < lr_at(9, 100, 1e-3, 10) <= 1e-3
    assert abs(lr_at(100, 100, 1e-3, 10) - 1e-4) < 1e-9


def test_training_reduces_loss_and_resumes(tmp_path):
    out = str(tmp_path / "run")
    args = parse(["--preset", "tiny", "--synthetic", "--steps", "60", "--batch", "8",
                  "--seq-len", "32", "--lr", "3e-3", "--out", out, "--log-every", "30",
                  "--save-every", "30"])
    h = train(args)
    assert h["last_loss"] < h["first_loss"] * 0.7, h
    args2 = parse(["--preset", "tiny", "--synthetic", "--steps", "70", "--batch", "8",
                   "--seq-len", "32", "--out", out, "--resume", "--log-every", "5"])
    h2 = train(args2)                                     # step 60 에서 이어서 10 스텝
    assert h2["first_loss"] < h["first_loss"]
