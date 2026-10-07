"""실제 비트 패킹(2/3/4-bit) 과 배포용 export/import. 추론 커널의 '참조 구현(reference)'.

blob 구조 (torch.save 가능):
  {"cfg": dict, "quant": {name: {"codes": uint8[packed], "scale": fp16, "bits", "shape", "group"}},
   "fp16": {name: fp16 tensor}}      # router / norm / edge / 기타 비양자화 파라미터
"""
from dataclasses import asdict

import torch

from .config import GraphMoEConfig
from .quant import QuantEmbedding, QuantLinear, dequantize_codes, quantize_codes


def pack_bits(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """codes: 정수값(0..2^bits-1) 텐서 [..., n] -> uint8 [..., ceil(n*bits/8)] (LSB first)."""
    flat = codes.to(torch.uint8).reshape(-1, codes.shape[-1])
    shifts = torch.arange(bits, device=flat.device, dtype=torch.uint8)
    b = ((flat[..., None] >> shifts) & 1).reshape(flat.shape[0], -1)         # [R, n*bits]
    pad = (-b.shape[1]) % 8
    if pad:
        b = torch.cat([b, b.new_zeros(b.shape[0], pad)], 1)
    w = (1 << torch.arange(8, device=flat.device)).to(torch.uint8)
    out = (b.reshape(b.shape[0], -1, 8) * w).sum(-1).to(torch.uint8)
    return out.reshape(*codes.shape[:-1], -1)


def unpack_bits(packed: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    flat = packed.reshape(-1, packed.shape[-1])
    sh = torch.arange(8, device=flat.device, dtype=torch.uint8)
    b = ((flat[..., None] >> sh) & 1).reshape(flat.shape[0], -1)[:, : n * bits]
    w = (1 << torch.arange(bits, device=flat.device)).to(torch.uint8)
    out = (b.reshape(b.shape[0], n, bits) * w).sum(-1).to(torch.uint8)
    return out.reshape(*packed.shape[:-1], n)


def _quant_modules(model):
    for name, m in model.named_modules():
        if isinstance(m, (QuantLinear, QuantEmbedding)) and m.bits is not None:
            yield name, m


@torch.no_grad()
def export_packed(model) -> dict:
    blob = {"cfg": asdict(model.cfg), "quant": {}, "fp16": {}}
    qnames = set()
    for name, m in _quant_modules(model):
        codes, scale, g = quantize_codes(m.weight.float(), m.bits, m.group)
        # codes [..., ng, g] -> 행 단위로 이어 붙여 패킹
        flat = codes.reshape(*m.weight.shape[:-1], -1)
        blob["quant"][name] = dict(codes=pack_bits(flat, m.bits), scale=scale.half(),
                                   bits=m.bits, shape=tuple(m.weight.shape), group=g)
        qnames.add(name + ".weight")
    for k, v in model.state_dict().items():
        if k not in qnames:
            blob["fp16"][k] = v.half()
    return blob


@torch.no_grad()
def dequant_entry(e: dict) -> torch.Tensor:
    shape, g, bits = e["shape"], e["group"], e["bits"]
    codes = unpack_bits(e["codes"], bits, shape[-1]).float()
    codes = codes.reshape(*shape[:-1], shape[-1] // g, g)
    return dequantize_codes(codes, e["scale"].float(), bits, shape)


@torch.no_grad()
def load_packed(model, blob: dict):
    """검증/참조용: blob 을 dequant 해서 model 에 적재 (model.cfg.qat=False 로 쓰면 배포 수치 재현)."""
    sd = model.state_dict()
    for name, e in blob["quant"].items():
        sd[name + ".weight"].copy_(dequant_entry(e))
    for k, v in blob["fp16"].items():
        sd[k].copy_(v.to(sd[k].dtype))
    return model


def packed_nbytes(blob: dict) -> int:
    n = sum(e["codes"].numel() + e["scale"].numel() * 2 for e in blob["quant"].values())
    return n + sum(v.numel() * 2 for v in blob["fp16"].values())
