"""토크나이저 인터페이스 + 토큰 .bin(memmap) 생성/샘플링.

CLI:  python -m graphmoe.data --tokenizer byte|hf:<name_or_path> --out data/train.bin  file1.txt file2.txt ...
"""
import argparse
import json
import os
from typing import Iterable, List

import numpy as np
import torch


class ByteTokenizer:
    """테스트/파이프라인 검증용. vocab=256."""
    vocab_size = 256
    eos_id = 0

    def encode(self, s: str) -> List[int]:
        return list(s.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


class HFTokenizer:
    """HF `tokenizers`/`transformers` 래퍼 (lazy import). 실제 학습용 vocab 은 여기서 주입."""

    def __init__(self, name_or_path: str):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name_or_path)
        self.vocab_size = len(self.tok)
        self.eos_id = self.tok.eos_token_id if self.tok.eos_token_id is not None else 0

    def encode(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def decode(self, ids):
        return self.tok.decode(list(ids))


def get_tokenizer(spec: str):
    return ByteTokenizer() if spec == "byte" else HFTokenizer(spec.removeprefix("hf:"))


def write_bin(docs: Iterable[str], tok, out_path: str) -> int:
    """문서들을 eos 로 이어 붙여 flat 토큰 .bin + .json(meta) 저장. 토큰 수 반환."""
    dtype = np.uint16 if tok.vocab_size < 2 ** 16 else np.uint32
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    with open(out_path, "wb") as f:
        for d in docs:
            ids = np.asarray(tok.encode(d) + [tok.eos_id], dtype=dtype)
            f.write(ids.tobytes())
            n += len(ids)
    with open(out_path + ".json", "w") as f:
        json.dump({"dtype": np.dtype(dtype).name, "n_tokens": n, "vocab_size": tok.vocab_size}, f)
    return n


class BinSampler:
    """flat 토큰 .bin 에서 무작위 윈도우를 뽑아 (x, y=x 한 칸 shift) 배치로 반환. 시드 고정 -> 재현 가능."""

    def __init__(self, path: str, seq_len: int, seed: int = 0):
        meta = json.load(open(path + ".json"))
        self.data = np.memmap(path, dtype=np.dtype(meta["dtype"]), mode="r")
        self.vocab_size, self.seq_len = meta["vocab_size"], seq_len
        assert len(self.data) > seq_len + 1, "데이터가 seq_len 보다 짧음"
        self.rng = np.random.default_rng(seed)

    def get_batch(self, batch: int, device="cpu"):
        hi = len(self.data) - self.seq_len - 1
        st = self.rng.integers(0, hi, size=batch)
        x = np.stack([self.data[s: s + self.seq_len] for s in st]).astype(np.int64)
        y = np.stack([self.data[s + 1: s + 1 + self.seq_len] for s in st]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def write_synthetic_bin(path: str, vocab: int = 256, n: int = 200_000, period: int = 16, seed: int = 0):
    """학습이 실제로 되는지 확인용: 주기 패턴 + 약간의 노이즈."""
    rng = np.random.default_rng(seed)
    base = rng.integers(1, vocab, size=period)
    arr = np.tile(base, n // period + 1)[:n]
    noise = rng.random(n) < 0.02
    arr[noise] = rng.integers(1, vocab, size=noise.sum())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr.astype(np.uint16).tofile(path)
    json.dump({"dtype": "uint16", "n_tokens": n, "vocab_size": vocab}, open(path + ".json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="byte")
    ap.add_argument("--out", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    tok = get_tokenizer(a.tokenizer)
    n = write_bin((open(p, encoding="utf-8").read() for p in a.files), tok, a.out)
    print(f"wrote {n:,} tokens -> {a.out}")
