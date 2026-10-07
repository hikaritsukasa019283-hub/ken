import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from make_arch_html import collect, render  # noqa: E402

from graphmoe.budget import compute_budget  # noqa: E402
from graphmoe.config import GraphMoEConfig, tiny_config  # noqa: E402


def test_render_has_no_placeholders_and_matches_budget():
    full, fragment = render()
    for doc in (full, fragment):
        assert "@@" not in doc
        assert "<title>Graph-MoE 7B 구조도</title>" in doc
    assert full.startswith("<!doctype html>") and "<html" not in fragment and "<body" not in fragment
    b = compute_budget(GraphMoEConfig())
    assert f"{b.total_params / 1e9:.2f}B" in fragment and f"{b.active / 1e9:.2f}B" in fragment
    data = json.loads(re.search(r"const D = (\{.*?\});\nlet uid", fragment, re.S).group(1))
    assert data["total_params"] == b.total_params and data["n_owner"] == 20 and data["n_sharer"] == 12
    assert data["per_layer"]["total"] * data["n_layers"] == data["params"]["expert_graph"]   # 레이어당 × 32 = 전체


def test_collect_follows_config():
    cfg = tiny_config()
    d = collect(cfg)
    assert d["n_layers"] == cfg.n_layers and d["r_domain"] == cfg.r_domain and len(d["owners"]) == cfg.n_layers
    full, _ = render(cfg)
    assert str(cfg.d_model) in full                       # 설정을 바꾸면 그림의 숫자도 바뀐다
