"""Fake-quant (STE) 기반 QAT 빌딩블록. 실제 배포 커널(packed 2/3/4-bit)은 TODO."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def fake_quant_weight(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """마지막 차원을 group 단위로 묶은 대칭 mid-rise 양자화 + STE.

    levels = {-(h-.5), ..., (h-.5)}*scale,  h = 2**(bits-1)  (2-bit => 4 levels, 0 없음)
    """
    shape = w.shape
    g = group if shape[-1] % group == 0 else shape[-1]
    wg = w.reshape(*shape[:-1], shape[-1] // g, g)
    half = 2 ** (bits - 1)
    amax = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / (half - 0.5)
    q = (torch.floor(wg / scale).clamp(-half, half - 1) + 0.5) * scale
    q = wg + (q - wg).detach()                      # STE
    return q.reshape(shape)


def fake_quant_int8_lastdim(x: torch.Tensor):
    """token/head 별 absmax int8. (int8 값, fp scale) 반환 — KV cache 용."""
    scale = (x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0).half().to(x.dtype)  # cache 와 동일하게 fp16 scale
    q = torch.round(x / scale).clamp(-127, 127)
    return q.to(torch.int8), scale


class QuantLinear(nn.Linear):
    """weight 를 forward 마다 fake-quant. bits=None 이면 일반 Linear."""

    def __init__(self, in_f, out_f, bits=None, group=64, qat=True, bias=False):
        super().__init__(in_f, out_f, bias=bias)
        self.bits, self.group, self.qat = bits, group, qat

    def qweight(self):
        if self.bits is None or not self.qat:
            return self.weight
        return fake_quant_weight(self.weight, self.bits, self.group)

    def forward(self, x):
        return F.linear(x, self.qweight(), self.bias)

    def extra_repr(self):
        return super().extra_repr() + f", bits={self.bits}"


class QuantEmbedding(nn.Embedding):
    def __init__(self, num, dim, bits=4, group=64, qat=True):
        super().__init__(num, dim)
        self.bits, self.group, self.qat = bits, group, qat

    def qweight(self):
        return fake_quant_weight(self.weight, self.bits, self.group) if self.qat else self.weight

    def forward(self, ids):
        return F.embedding(ids, self.qweight())


class RMSNorm(nn.Module):
    """Norm 은 FP16 유지 대상 (배포 시 .half())."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight
