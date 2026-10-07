"""토크나이저 인터페이스 + 토큰 .bin(memmap) 생성/샘플링.

CLI (로컬 텍스트):
  python -m graphmoe.data --tokenizer byte|file:<tok.json>|hf:<name_or_path> --out data/train.bin  file1.txt file2.txt ...
CLI (HuggingFace 데이터셋, 스트리밍 — 전체를 내려받지 않고 RAM/디스크 최소로 필요한 만큼만):
  python -m graphmoe.data --tokenizer hf:<tok> --hf-dataset <org/name> [--hf-config c] [--hf-data-dir d] \
        --split train --text-field text --max-tokens 200000000 --out data/train.bin
  예) Anthropic/hh-rlhf:  --hf-dataset Anthropic/hh-rlhf --hf-data-dir helpful-base --text-field chosen
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
    """'byte' | 'file:<tokenizer.json>' (graphmoe.tokenizer 로 학습한 것) | 'hf:<name_or_path>'
    접두사 없이 '*.json' 파일 경로만 줘도 학습한 토크나이저로 인식한다."""
    if spec == "byte":
        return ByteTokenizer()
    is_file = spec.startswith("file:") or (spec.endswith(".json") and not spec.startswith("hf:"))
    if is_file:
        path = spec.removeprefix("file:")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"토크나이저 파일을 찾을 수 없음: {path!r} (현재 폴더: {os.getcwd()}). "
                "graphmoe.tokenizer train 의 --out 경로와 같은지, 같은 폴더에서 실행했는지 확인하세요.")
        from .tokenizer import TrainedTokenizer
        return TrainedTokenizer(path)
    name = spec.removeprefix("hf:")
    try:
        return HFTokenizer(name)
    except Exception as e:                      # 허브에서 못 찾음 / transformers 미설치 / 네트워크 차단
        raise RuntimeError(
            f"HF 토크나이저 {name!r} 를 불러오지 못함: {e}\n"
            "-> 직접 학습한 토크나이저(json)라면 'file:경로.json' 으로 지정하세요.") from e


def iter_hf_texts(dataset: str, split: str = "train", text_field: str = "text", config: str = None,
                  data_dir: str = None):
    """HF 데이터셋을 streaming 으로 순회하며 text_field 만 yield (datasets lazy import).
    data_dir: 하위 폴더형 서브셋 (예: Anthropic/hh-rlhf 의 helpful-base)."""
    from datasets import load_dataset
    ds = load_dataset(dataset, config, data_dir=data_dir, split=split, streaming=True)
    for row in ds:
        t = row.get(text_field)
        if t:
            yield t


def write_bin(docs: Iterable[str], tok, out_path: str, max_tokens: int = None) -> int:
    """문서들을 eos 로 이어 붙여 flat 토큰 .bin + .json(meta) 저장. 토큰 수 반환.
    max_tokens 에 도달하면 중단 (스트리밍 데이터셋용)."""
    dtype = np.uint16 if tok.vocab_size < 2 ** 16 else np.uint32
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    with open(out_path, "wb") as f:
        for d in docs:
            ids = np.asarray(tok.encode(d) + [tok.eos_id], dtype=dtype)
            f.write(ids.tobytes())
            n += len(ids)
            if max_tokens and n >= max_tokens:
                break
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


class MixSampler:
    """여러 .bin 을 가중치 비율로 섞어 뽑는 샘플러 (BinSampler 와 같은 인터페이스).

    배치의 각 샘플마다 소스를 weights 확률로 고른 뒤 그 소스의 무작위 윈도우를 뽑는다.
    => 배치 크기와 무관하게 기대 비율이 유지되고, 시드 고정 시 재현 가능.
    usage[name] 에 소스별 누적 샘플 수를 기록 (실제 혼합 비율 점검용)."""

    def __init__(self, sources, seq_len: int, seed: int = 0):
        """sources: [(path, weight), ...]"""
        assert len(sources) >= 1, "소스가 비어 있음"
        ws = [float(w) for _, w in sources]
        assert all(w > 0 for w in ws), "가중치는 양수여야 함"
        self.names = [os.path.basename(p) for p, _ in sources]
        assert len(set(self.names)) == len(self.names), "소스 파일명이 중복됨 (usage 구분 불가)"
        self.weights = np.asarray(ws) / sum(ws)
        # 소스별 샘플러는 독립 시드 (소스 간 상관 방지), 소스 선택용 rng 는 별도
        self.samplers = [BinSampler(p, seq_len, seed * 1000 + i + 1) for i, (p, _) in enumerate(sources)]
        self.vocab_size = max(sm.vocab_size for sm in self.samplers)
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        self.usage = {n: 0 for n in self.names}

    def get_batch(self, batch: int, device="cpu"):
        pick = self.rng.choice(len(self.samplers), size=batch, p=self.weights)
        xs, ys = [], []
        for i in pick:
            x, y = self.samplers[i].get_batch(1)
            xs.append(x); ys.append(y)
            self.usage[self.names[i]] += 1
        return torch.cat(xs).to(device), torch.cat(ys).to(device)


def parse_mix(spec: str):
    """'ko.bin=0.7,math.bin=0.25,en.bin=0.05' -> [(path, weight), ...]  (Windows 경로의 ':' 때문에 '=' 사용)"""
    out = []
    for item in spec.split(","):
        path, _, w = item.strip().rpartition("=")
        if not path:
            raise ValueError(f"'경로=가중치' 형식이 아님: {item!r}")
        out.append((path, float(w)))
    return out


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
    ap.add_argument("files", nargs="*")
    ap.add_argument("--hf-dataset"); ap.add_argument("--hf-config"); ap.add_argument("--hf-data-dir")
    ap.add_argument("--split", default="train"); ap.add_argument("--text-field", default="text")
    ap.add_argument("--max-tokens", type=int)
    a = ap.parse_args()
    if bool(a.hf_dataset) == bool(a.files):
        ap.error("텍스트 파일들 또는 --hf-dataset 중 하나만 지정")
    tok = get_tokenizer(a.tokenizer)
    docs = (iter_hf_texts(a.hf_dataset, a.split, a.text_field, a.hf_config, a.hf_data_dir) if a.hf_dataset
            else (open(p, encoding="utf-8").read() for p in a.files))
    n = write_bin(docs, tok, a.out, a.max_tokens)
    print(f"wrote {n:,} tokens -> {a.out}")
