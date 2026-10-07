"""Per-layer Expert Graph: Domain -> Operation -> Control (top-0/1).

한 토큰의 경로(path) = (domain i, op j, ctrl k|None). 잠재(latent) 가 rank 순으로 좁아지며 흐르고
(512 -> 384 -> 192), 각 tier 가 자기 up-projection 으로 d_model 잔차에 기여한다:

    u = SwiGLU_d[i](x)          [r_d]     y_d = p_d * Down_d[i](u)
    v = silu(W_o[j] u)          [r_o]     y_o = p_o * Down_o[j](v)
    w = silu(W_c[k] v)          [r_c]     y_c = g   * Down_c[k](w)      (control 이 꺼지면 0)

v 는 (i,j) 에, w 는 (i,j,k) 에 의존 => 8*8*4 = 256 개 논리 경로, 물리 factor 는 8+8+4 = 20.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GraphMoEConfig
from .quant import QuantLinear


class DomainFactor(nn.Module):
    def __init__(self, d, r, **qk):
        super().__init__()
        self.gate_up = QuantLinear(d, 2 * r, **qk)
        self.down = QuantLinear(r, d, **qk)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        z = F.silu(g) * u
        return z, self.down(z)


class LatentFactor(nn.Module):
    """Operation / Control 공용: latent(r_in) -> latent(r) , latent(r) -> d_model."""

    def __init__(self, r_in, r, d, **qk):
        super().__init__()
        self.mix = QuantLinear(r_in, r, **qk)
        self.down = QuantLinear(r, d, **qk)

    def forward(self, z):
        z = F.silu(self.mix(z))
        return z, self.down(z)


def _dispatch(experts, idx, inp, r_out, d):
    """top-1 dispatch: idx[n] 번 expert 로 토큰을 보내고 (latent, out) 을 원래 순서로 되돌린다."""
    N = inp.shape[0]
    lat = inp.new_zeros(N, r_out)
    out = inp.new_zeros(N, d)
    for e, m in enumerate(experts):
        sel = (idx == e).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        l, o = m(inp[sel])
        lat = lat.index_copy(0, sel, l.to(lat.dtype))
        out = out.index_copy(0, sel, o.to(out.dtype))
    return lat, out


def _balance_loss(probs, idx, n):
    """Switch-style load balance: n * sum_e f_e * P_e."""
    f = F.one_hot(idx, n).float().mean(0)
    return n * (f * probs.float().mean(0)).sum()


class ExpertGraph(nn.Module):
    def __init__(self, cfg: GraphMoEConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        qk = dict(bits=cfg.bits_graph, group=cfg.group_size, qat=cfg.qat)
        self.domain = nn.ModuleList(DomainFactor(d, cfg.r_domain, **qk) for _ in range(cfg.n_domain))
        self.op = nn.ModuleList(LatentFactor(cfg.r_domain, cfg.r_op, d, **qk) for _ in range(cfg.n_op))
        self.ctrl = nn.ModuleList(LatentFactor(cfg.r_op, cfg.r_ctrl, d, **qk) for _ in range(cfg.n_ctrl))

        # Router: FP16 유지 대상 (양자화 없음). edge_* = 그래프 간선(이전 tier 선택 -> 다음 tier logit prior)
        self.r_domain = nn.Linear(d, cfg.n_domain, bias=False)
        self.r_op = nn.Linear(d, cfg.n_op, bias=False)
        self.r_ctrl = nn.Linear(d, cfg.n_ctrl, bias=False)
        self.r_ctrl_gate = nn.Linear(d, 1)                       # control top-0/1 on/off
        self.edge_do = nn.Parameter(torch.zeros(cfg.n_domain, cfg.n_op))
        self.edge_oc = nn.Parameter(torch.zeros(cfg.n_op, cfg.n_ctrl))

    def forward(self, x):
        """x [B,T,d] -> (y [B,T,d], aux_loss, route [B,T,3]  (ctrl 꺼짐 = -1))"""
        B, T, d = x.shape
        cfg = self.cfg
        xf = x.reshape(-1, d)

        # ---- routing (top-1 x3, control 은 top-0/1) ----
        p_d = F.softmax(self.r_domain(xf).float(), -1)
        i = p_d.argmax(-1)
        p_o = F.softmax((self.r_op(xf) + self.edge_do[i]).float(), -1)
        j = p_o.argmax(-1)
        p_c = F.softmax((self.r_ctrl(xf) + self.edge_oc[j]).float(), -1)
        k = p_c.argmax(-1)
        g_soft = torch.sigmoid(self.r_ctrl_gate(xf).float()).squeeze(-1)
        on = (g_soft > 0.5)
        g = on.float() + g_soft - g_soft.detach()                 # hard on/off, soft gradient (STE)

        gd = p_d.gather(1, i[:, None]).to(x.dtype)                # 선택된 expert 확률 = router 로 가는 gradient 경로
        go = p_o.gather(1, j[:, None]).to(x.dtype)

        # ---- expert graph ----
        u, y_d = _dispatch(self.domain, i, xf, cfg.r_domain, d)
        v, y_o = _dispatch(self.op, j, u, cfg.r_op, d)
        y = gd * y_d + go * y_o
        sel = on.nonzero(as_tuple=True)[0]
        if sel.numel() > 0:                                       # top-0 토큰은 control 연산을 건너뜀
            _, y_c = _dispatch(self.ctrl, k[sel], v[sel], cfg.r_ctrl, d)
            y = y.index_add(0, sel, g[sel, None].to(x.dtype) * y_c)

        aux = (_balance_loss(p_d, i, cfg.n_domain) + _balance_loss(p_o, j, cfg.n_op)
               + (_balance_loss(p_c[sel], k[sel], cfg.n_ctrl) if sel.numel() > 0 else x.new_zeros(())))
        route = torch.stack([i, j, torch.where(on, k, torch.full_like(k, -1))], -1).view(B, T, 3)
        return y.view(B, T, d), aux, route
