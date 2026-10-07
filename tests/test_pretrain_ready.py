import copy

import torch

from graphmoe.config import tiny_config
from graphmoe.data import BinSampler, ByteTokenizer, MixSampler, parse_mix, write_bin, write_synthetic_bin
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


def test_hf_streaming_and_max_tokens(tmp_path, monkeypatch):
    import sys, types
    from graphmoe.data import iter_hf_texts
    calls = {}

    def fake_load(name, config, data_dir=None, split=None, streaming=None):
        calls.update(name=name, split=split, streaming=streaming, data_dir=data_dir)
        return iter([{"text": "abc " * 50}, {"text": ""}, {"other": "x"}] + [{"text": "hello " * 50}] * 100)

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load))
    texts = list(iter_hf_texts("org/ds", "train", "text"))
    assert calls == {"name": "org/ds", "split": "train", "streaming": True, "data_dir": None}
    assert len(texts) == 101                              # 빈 text / 필드 없는 행은 건너뜀
    p = str(tmp_path / "hf.bin")
    n = write_bin(iter_hf_texts("org/ds"), ByteTokenizer(), p, max_tokens=1000)
    assert 1000 <= n < 1000 + 400                         # 문서 단위로 끊으므로 약간 초과 가능
    list(iter_hf_texts("Anthropic/hh-rlhf", "train", "chosen", data_dir="helpful-base"))
    assert calls["data_dir"] == "helpful-base" and calls["name"] == "Anthropic/hh-rlhf"


def _const_bin(path, value, n=5000, vocab=256):
    import json, numpy as np
    np.full(n, value, dtype=np.uint16).tofile(path)
    json.dump({"dtype": "uint16", "n_tokens": n, "vocab_size": vocab}, open(path + ".json", "w"))


def test_mix_sampler_ratio_determinism_and_usage(tmp_path):
    a, b = str(tmp_path / "ko.bin"), str(tmp_path / "math.bin")
    _const_bin(a, 5); _const_bin(b, 9)
    s = MixSampler([(a, 7), (b, 3)], seq_len=8, seed=1)                  # 가중치는 정규화됨 -> 0.7 / 0.3
    x, y = s.get_batch(2000)
    frac_ko = (x[:, 0] == 5).float().mean().item()
    assert abs(frac_ko - 0.7) < 0.05, frac_ko                           # 이항분포 sd≈0.01, 5σ 여유
    assert ((x == 5).all(1) | (x == 9).all(1)).all()                    # 한 샘플은 한 소스에서만
    assert s.usage["ko.bin"] == int((x[:, 0] == 5).sum())               # usage 집계가 실제 개수와 정확히 일치
    s2 = MixSampler([(a, 7), (b, 3)], seq_len=8, seed=1)
    assert torch.equal(s2.get_batch(50)[0], MixSampler([(a, 7), (b, 3)], 8, 1).get_batch(50)[0])


def test_mix_sampler_validation_and_parse(tmp_path):
    import pytest
    a = str(tmp_path / "a.bin"); _const_bin(a, 1)
    with pytest.raises(AssertionError):
        MixSampler([(a, 0)], 8)                                          # 가중치 0 거부
    with pytest.raises(AssertionError):
        MixSampler([], 8)
    assert parse_mix("ko.bin=0.7, math.bin=0.25") == [("ko.bin", 0.7), ("math.bin", 0.25)]
    assert parse_mix(r"C:\data\ko.bin=1") == [(r"C:\data\ko.bin", 1.0)]  # Windows 경로의 ':' 안전
    with pytest.raises(ValueError):
        parse_mix("ko.bin")


def test_train_with_mix(tmp_path):
    a, b = str(tmp_path / "a.bin"), str(tmp_path / "b.bin")
    write_synthetic_bin(a, vocab=256, n=20000, seed=0)
    write_synthetic_bin(b, vocab=256, n=20000, seed=1)
    h = train(parse(["--preset", "tiny", "--mix", f"{a}=0.6,{b}=0.4", "--steps", "4", "--batch", "8",
                     "--seq-len", "32", "--out", str(tmp_path / "run"), "--log-every", "2"]))
    assert h["last_loss"] > 0
    line = open(tmp_path / "run" / "log.jsonl").read().splitlines()[-1]
    import json
    mix = json.loads(line)["mix"]
    assert set(mix) == {"a.bin", "b.bin"} and abs(sum(mix.values()) - 1) < 1e-3
