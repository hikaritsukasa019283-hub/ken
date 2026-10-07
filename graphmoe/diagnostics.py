"""라우터 건강 진단: tier 별 사용률 / 엔트로피 / dead expert / 경로 다양성."""
import math

import torch


def route_stats(routes: torch.Tensor, cfg) -> dict:
    """routes [L,B,T,3] (ctrl 꺼짐=-1) -> 스칼라 dict (로깅용)."""
    sizes = (cfg.n_domain, cfg.n_op, cfg.n_ctrl)
    out = {}
    for t, (name, n) in enumerate(zip(("domain", "op", "ctrl"), sizes)):
        r = routes[..., t]
        valid = r >= 0
        if name == "ctrl":
            out["ctrl_on_rate"] = valid.float().mean().item()
        ents, dead = [], 0
        for l in range(r.shape[0]):
            v = r[l][valid[l]]
            if v.numel() == 0:
                dead += n
                continue
            p = torch.bincount(v, minlength=n).float()
            dead += int((p == 0).sum())
            p = p / p.sum()
            ents.append(-(p[p > 0] * p[p > 0].log()).sum().item() / math.log(n))   # 0..1 정규화
        out[f"{name}_entropy"] = sum(ents) / max(len(ents), 1)
        out[f"{name}_dead"] = dead / r.shape[0]                                    # 레이어 평균 dead 수
    # 사용된 고유 (i,j,k) 경로 수 / 256, 레이어 평균
    code = routes[..., 0] * 1000 + routes[..., 1] * 10 + (routes[..., 2] + 1)
    out["paths_used"] = sum(c.unique().numel() for c in code) / code.shape[0]
    return out
