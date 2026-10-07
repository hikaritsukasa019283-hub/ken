import importlib.util
import py_compile
import subprocess
import sys
from pathlib import Path

import torch

from graphmoe.budget import compute_budget
from graphmoe.config import GraphMoEConfig, tiny_config
from graphmoe.model import GraphMoE

ROOT = Path(__file__).resolve().parent.parent


def _load_bundle(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from bundle_model import bundle
    finally:
        sys.path.pop(0)
    out = bundle(tmp_path / "graphmoe_single.py")
    spec = importlib.util.spec_from_file_location("graphmoe_single_t", out)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["graphmoe_single_t"] = mod
    spec.loader.exec_module(mod)
    return out, mod


def test_bundle_is_self_contained_and_compiles(tmp_path):
    out, _ = _load_bundle(tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "from ." not in text and "import graphmoe" not in text            # 패키지 의존 없음
    py_compile.compile(str(out), doraise=True)


def test_bundle_matches_package_numerically(tmp_path):
    _, mod = _load_bundle(tmp_path)
    torch.manual_seed(0)
    a = GraphMoE(tiny_config()).eval()
    b = mod.GraphMoE(mod.tiny_config()).eval()
    b.load_state_dict(a.state_dict())                                         # 같은 가중치
    ids = torch.randint(0, 256, (2, 20))
    ya, yb = a(ids), b(ids)
    assert torch.equal(ya["logits"], yb["logits"]) and torch.equal(ya["routes"], yb["routes"])
    ca, cb = a.new_caches(1), b.new_caches(1)                                  # int8 KV cache 경로도 동일
    la = a(ids[:1, :12], ca, 0, return_aux=False)
    lb = b(ids[:1, :12], cb, 0, return_aux=False)
    assert torch.equal(la, lb)
    assert mod.compute_budget(mod.GraphMoEConfig()).total_params == compute_budget(GraphMoEConfig()).total_params


def test_bundle_runs_as_script(tmp_path):
    out, _ = _load_bundle(tmp_path)
    r = subprocess.run([sys.executable, str(out)], capture_output=True, text=True, cwd=tmp_path, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "TOTAL" in r.stdout and "[tiny forward]" in r.stdout
