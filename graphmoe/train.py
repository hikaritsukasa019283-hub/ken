"""학습 루프 (AdamW + warmup/cosine + grad accum + bf16 + ckpt/resume + 라우터 진단).

  python -m graphmoe.train --preset tiny --synthetic --steps 50 --out runs/smoke
  python -m graphmoe.train --preset mini --data data/train.bin --val data/val.bin --out runs/mini0
"""
import argparse
import json
import math
import os
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

from .config import GraphMoEConfig, get_preset
from .data import BinSampler, MixSampler, parse_mix, write_synthetic_bin
from .diagnostics import route_stats
from .model import GraphMoE


def build_param_groups(model, weight_decay: float):
    """norm / router / edge / bias 는 decay 제외 (router·norm 은 FP16 유지 대상이라 특히 안정성 중요)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        skip = p.ndim < 2 or any(k in n for k in ("norm", "r_domain", "r_op", "r_ctrl", "edge_"))
        (no_decay if skip else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step, total, base, warmup, min_ratio=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def lm_loss(model, x, y):
    out = model(x)
    ce = F.cross_entropy(out["logits"].float().reshape(-1, out["logits"].shape[-1]), y.reshape(-1))
    return ce, out


@torch.no_grad()
def evaluate(model, sampler, batch, iters, device, amp):
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = sampler.get_batch(batch, device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
            ce, _ = lm_loss(model, x, y)
        tot += ce.item()
    model.train()
    loss = tot / iters
    return loss, math.exp(min(loss, 20))


def save_ckpt(path, model, opt, step, cfg):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "cfg": asdict(cfg), "torch_rng": torch.get_rng_state()}, path + ".tmp")
    os.replace(path + ".tmp", path)                                   # 원자적 교체


def load_ckpt(path, model, opt=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    if opt is not None:
        opt.load_state_dict(ck["opt"])
    return ck["step"]


def train(a) -> dict:
    torch.manual_seed(a.seed)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_preset(a.preset, grad_ckpt=a.grad_ckpt, aux_loss_coef=a.aux_coef)
    if a.synthetic:
        a.data = a.data or os.path.join(a.out, "synthetic.bin")
        if not os.path.exists(a.data):
            write_synthetic_bin(a.data, vocab=cfg.vocab_size)
    if a.mix:                                   # 여러 코퍼스를 비율로 혼합 (예: 한국어 웹/수학/영어)
        train_s = MixSampler(parse_mix(a.mix), a.seq_len, a.seed)
        val_s = (MixSampler(parse_mix(a.val_mix), a.seq_len, a.seed + 1) if a.val_mix
                 else BinSampler(a.val, a.seq_len, a.seed + 1) if a.val
                 else MixSampler(parse_mix(a.mix), a.seq_len, a.seed + 1))
    else:
        train_s = BinSampler(a.data, a.seq_len, a.seed)
        val_s = BinSampler(a.val or a.data, a.seq_len, a.seed + 1)
    assert train_s.vocab_size <= cfg.vocab_size, "데이터 vocab > 모델 vocab"

    model = GraphMoE(cfg).to(device).train()
    opt = torch.optim.AdamW(build_param_groups(model, a.wd), lr=a.lr, betas=(0.9, 0.95), eps=1e-8)
    step = 0
    ck_path = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(ck_path):
        step = load_ckpt(ck_path, model, opt)
        print(f"resumed from step {step}")
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    amp = a.bf16 and device != "cpu"
    hist = {"first_loss": None, "last_loss": None}
    t0 = time.time()

    while step < a.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, a.warmup)
        opt.zero_grad(set_to_none=True)
        tot_ce = 0.0
        for _ in range(a.accum):
            x, y = train_s.get_batch(a.batch, device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
                ce, out = lm_loss(model, x, y)
                loss = (ce + out["aux_loss"]) / a.accum
            loss.backward()
            tot_ce += ce.item() / a.accum
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip).item()
        if not math.isfinite(gn):                                     # NaN/Inf 가드: 스텝 건너뜀
            print(f"step {step}: non-finite grad, skipped")
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
        step += 1
        hist["first_loss"] = hist["first_loss"] if hist["first_loss"] is not None else tot_ce
        hist["last_loss"] = tot_ce

        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "loss": tot_ce, "aux": out["aux_loss"].item(), "gnorm": gn,
                   "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0,
                   **route_stats(out["routes"].detach(), cfg)}
            if a.mix:
                tot = sum(train_s.usage.values())
                rec["mix"] = {k: round(v / tot, 4) for k, v in train_s.usage.items()}
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()},
                             ensure_ascii=False))
            log.write(json.dumps(rec) + "\n"); log.flush()
        if a.eval_every and step % a.eval_every == 0:
            vl, ppl = evaluate(model, val_s, a.batch, a.eval_iters, device, amp)
            print(f"[eval] step {step} val_loss {vl:.4f} ppl {ppl:.2f}")
        if a.save_every and step % a.save_every == 0:
            save_ckpt(ck_path, model, opt, step, cfg)
    save_ckpt(ck_path, model, opt, step, cfg)
    return hist


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="tiny", choices=["tiny", "mini", "7b"])
    p.add_argument("--data"); p.add_argument("--val"); p.add_argument("--synthetic", action="store_true")
    p.add_argument("--mix", help="'a.bin=0.7,b.bin=0.25,...' 혼합 학습 데이터 (--data 대신)")
    p.add_argument("--val-mix", help="검증용 혼합 (미지정 시 --val, 그것도 없으면 --mix 재사용)")
    p.add_argument("--out", default="runs/default")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--aux-coef", type=float, default=0.01)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--device")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-iters", type=int, default=10)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse())
