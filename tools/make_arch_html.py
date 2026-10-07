"""구조도 HTML 생성기: graphmoe 의 설정/계산 함수에서 실제 수치를 뽑아 tools/arch_template.html 에 채운다.

  python tools/make_arch_html.py            -> docs/architecture.html (단독 실행 가능한 전체 문서)
                                               docs/architecture.fragment.html (Artifact 게시용 조각: doctype/head/body 없음)

수치는 손으로 적지 않는다. config 나 budget 이 바뀌면 이 스크립트만 다시 돌리면 그림도 바뀐다.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graphmoe.budget import compute_budget  # noqa: E402
from graphmoe.config import GraphMoEConfig  # noqa: E402

HEAD_MARK, BODY_MARK = "<!--@@HEAD@@-->", "<!--@@BODY@@-->"


def collect(cfg: GraphMoEConfig) -> dict:
    b = compute_budget(cfg)
    d = cfg.d_model
    owners = cfg.kv_owner()
    ffn_bits = [cfg.ffn_bits(i) for i in range(cfg.n_layers)]
    per = dict(
        dom=cfg.n_domain * 3 * d * cfg.r_domain,
        op=cfg.n_op * (cfg.r_domain * cfg.r_op + cfg.r_op * d),
        ctl=cfg.n_ctrl * (cfg.r_op * cfg.r_ctrl + cfg.r_ctrl * d),
    )
    per["total"] = per["dom"] + per["op"] + per["ctl"]
    return dict(
        n_layers=cfg.n_layers, d_model=d, n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
        ffn_dim=cfg.ffn_dim, vocab=cfg.vocab_size, ctx=cfg.max_seq_len,
        owners=owners, ffn_bits=ffn_bits,
        n_owner=sum(i == o for i, o in enumerate(owners)), n_sharer=sum(i != o for i, o in enumerate(owners)),
        n_domain=cfg.n_domain, n_op=cfg.n_op, n_ctrl=cfg.n_ctrl,
        r_domain=cfg.r_domain, r_op=cfg.r_op, r_ctrl=cfg.r_ctrl,
        n_logical=cfg.n_logical_paths, n_physical=cfg.n_physical_factors,
        per_layer=per,
        params=b.params, bytes=b.bytes_, kv_bytes=b.kv_bytes,
        total_params=b.total_params, active=b.active, total_bytes=b.total_bytes,
        bits=dict(graph=cfg.bits_graph, ffn_edge=cfg.bits_ffn_edge, ffn_mid=cfg.bits_ffn_mid,
                  attn=cfg.bits_attn, embed=cfg.bits_embed, kv=cfg.kv_bits, group=cfg.group_size),
    )


def texts(D: dict) -> dict:
    edge = sum(1 for b in D["ffn_bits"] if b == D["bits"]["ffn_edge"])
    pct = lambda v, t: f"{(v - t) / t * 100:+.1f}%"
    return {
        "n_layers": D["n_layers"], "last_layer": D["n_layers"] - 1, "d_model": D["d_model"],
        "n_heads": D["n_heads"], "n_kv_heads": D["n_kv_heads"],
        "ctx": f"{D['ctx'] // 1024}K",
        "total": f"{D['total_params'] / 1e9:.2f}B", "active": f"{D['active'] / 1e9:.2f}B",
        "ram": f"{D['total_bytes'] / 1e9:.1f}GB",
        "expert_share": f"{D['params']['expert_graph'] / D['total_params'] * 100:.0f}%",
        "n_owner": D["n_owner"], "n_sharer": D["n_sharer"],
        "r_domain": D["r_domain"], "r_op": D["r_op"], "r_ctrl": D["r_ctrl"],
        "d_total": pct(D["total_params"] / 1e9, 6.8), "d_active": pct(D["active"] / 1e9, 2.4),
        "b_graph": D["bits"]["graph"], "b_edge": D["bits"]["ffn_edge"], "b_mid": D["bits"]["ffn_mid"],
        "b_attn": D["bits"]["attn"], "b_embed": D["bits"]["embed"], "b_kv": f"int{D['bits']['kv']}",
        "group": D["bits"]["group"], "edge": edge, "mid": D["n_layers"] - edge,
    }


def render(cfg: GraphMoEConfig = None):
    cfg = cfg or GraphMoEConfig()
    D = collect(cfg)
    tpl = (ROOT / "tools" / "arch_template.html").read_text(encoding="utf-8")
    for k, v in texts(D).items():
        tpl = tpl.replace(f"@@{k}@@", str(v))
    tpl = tpl.replace("/*@@DATA@@*/null", json.dumps(D, ensure_ascii=False))
    assert "@@" not in tpl.replace(HEAD_MARK, "").replace(BODY_MARK, "").replace("@@DATA@@", ""), "치환 안 된 자리표시자"
    head, body = tpl.split(HEAD_MARK, 1)[1].split(BODY_MARK, 1)
    fragment = (head + body).strip() + "\n"
    full = ('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + head.strip() + "\n</head>\n<body>\n" + body.strip() + "\n</body>\n</html>\n")
    return full, fragment


if __name__ == "__main__":
    full, fragment = render()
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    (out / "architecture.html").write_text(full, encoding="utf-8")
    (out / "architecture.fragment.html").write_text(fragment, encoding="utf-8")
    print(f"wrote docs/architecture.html ({len(full):,} B), docs/architecture.fragment.html ({len(fragment):,} B)")
