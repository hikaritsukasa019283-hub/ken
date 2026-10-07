"""한국어 + 수학 혼합 코퍼스로 byte-level BPE 토크나이저 학습 / 로드 / 점검.

설계 선택 (WORKGUIDE T11):
  - byte-level BPE  : 어떤 입력도 UNK 없이 표현 (한글/LaTeX/기호/이모지)
  - 숫자 개별 분리   : "12345" -> 1,2,3,4,5  (자릿수 규칙이 일정해져 산술/수식에 유리)
  - NFC 정규화       : 자모 분리형 한글을 음절로 통일 (그래서 비-NFC 입력은 decode 시 NFC 로 돌아옴)
  - special: <|endoftext|>=0 (eos), <|pad|>=1

CLI:
  python -m graphmoe.tokenizer train --out tok/ko_math.json --vocab-size 64000 --total-chars 100000000 \\
      --src hf=HuggingFaceFW/fineweb-2 config=kor_Hang field=text weight=0.7 \\
      --src hf=HuggingFaceTB/finemath config=finemath-4plus field=text weight=0.25 \\
      --src file=extra_en.txt weight=0.05
  로컬 파일(HF 에서 따로 받은 것): --src file=data\\kor_Hang\\*.parquet field=text weight=0.7   (.parquet/.jsonl/.jsonl.gz/.txt)
"""
import argparse
import json
import os
import unicodedata
from typing import Dict, Iterable, Iterator, List, Tuple

SPECIALS = ["<|endoftext|>", "<|pad|>"]


class TrainedTokenizer:
    """tokenizers JSON 로더. data.get_tokenizer('file:<path>') 로 사용 (ByteTokenizer/HFTokenizer 와 같은 인터페이스)."""

    def __init__(self, path: str):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(path)
        self.vocab_size = self.tok.get_vocab_size()
        self.eos_id = self.tok.token_to_id(SPECIALS[0])

    def encode(self, s: str) -> List[int]:
        return self.tok.encode(s).ids

    def decode(self, ids) -> str:
        return self.tok.decode([int(i) for i in ids])


def padded_vocab(n: int, multiple: int = 64) -> int:
    """GPU 효율/양자화 group(64) 정렬을 위해 모델 vocab_size 는 이 값으로."""
    return -(-n // multiple) * multiple


def mix_corpus(sources: List[dict], total_chars: int, holdout: int = 200):
    """sources: [{name, docs(iterable[str]), weight}] -> (학습용 문서 generator, heldout{name:[docs]}).

    소스별 문자 예산 = total_chars * weight/sum(weight). 사용량/예산이 가장 낮은 소스에서 다음 문서를 뽑아
    비율을 유지하며 인터리브한다. 각 소스의 앞 `holdout` 문서는 점검용으로 빼 두고 학습에 쓰지 않는다."""
    wsum = sum(s["weight"] for s in sources)
    assert wsum > 0 and all(s["weight"] > 0 for s in sources), "가중치는 양수"
    st = [{"name": s["name"], "it": iter(s["docs"]), "budget": total_chars * s["weight"] / wsum,
           "used": 0, "alive": True} for s in sources]
    held: Dict[str, List[str]] = {s["name"]: [] for s in st}
    for s in st:                                           # 홀드아웃 먼저 분리
        for _ in range(holdout):
            d = next(s["it"], None)
            if d is None:
                s["alive"] = False
                break
            held[s["name"]].append(d)

    def gen() -> Iterator[str]:
        while True:
            live = [s for s in st if s["alive"] and s["used"] < s["budget"]]
            if not live:
                return
            s = min(live, key=lambda x: x["used"] / x["budget"])
            d = next(s["it"], None)
            if d is None:
                s["alive"] = False
                continue
            s["used"] += len(d)
            yield d
    return gen(), held, st


def train_tokenizer(docs: Iterable[str], vocab_size: int, out_path: str) -> TrainedTokenizer:
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIALS, min_frequency=2, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(docs, trainer=trainer)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tok.save(out_path)
    return TrainedTokenizer(out_path)


def report(tok: TrainedTokenizer, held: Dict[str, List[str]]) -> dict:
    """홀드아웃 문서로 도메인별 압축률 + 무손실 왕복 + 숫자 분리 점검."""
    out = {"vocab_size": tok.vocab_size, "padded_vocab_size": padded_vocab(tok.vocab_size), "domains": {}}
    for name, docs in held.items():
        if not docs:
            continue
        ch = by = nt = ok = 0
        for d in docs:
            ids = tok.encode(d)
            ch += len(d); by += len(d.encode("utf-8")); nt += len(ids)
            ok += tok.decode(ids) == unicodedata.normalize("NFC", d)
        out["domains"][name] = {"docs": len(docs), "chars_per_token": ch / nt, "bytes_per_token": by / nt,
                                "roundtrip_ok": ok / len(docs)}
    out["digits_split_ok"] = tok.encode("12345") == [tok.encode(c)[0] for c in "12345"]
    return out


def _parse_src(spec: str) -> dict:
    kv = dict(p.split("=", 1) for p in spec.split(","))
    w = float(kv.pop("weight", 1.0))
    if "hf" in kv:
        from .data import iter_hf_texts
        docs = iter_hf_texts(kv["hf"], kv.get("split", "train"), kv.get("field", "text"),
                             kv.get("config"), kv.get("data_dir"))
        return {"name": kv["hf"].split("/")[-1] + (f":{kv['config']}" if kv.get("config") else ""),
                "docs": docs, "weight": w}
    if "file" in kv:                              # txt(줄=문서) / parquet / jsonl[.gz], 와일드카드 가능
        from .data import iter_local_docs
        path = kv["file"]
        return {"name": os.path.basename(path), "weight": w,
                "docs": iter_local_docs(path, kv.get("field", "text"), lines=True)}
    raise ValueError(f"hf=... 또는 file=... 필요: {spec!r}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--vocab-size", type=int, default=64000)
    t.add_argument("--total-chars", type=int, default=100_000_000,
                   help="학습에 쓸 총 문자 수. BPE 학습은 RAM 을 많이 쓰므로 PC 사양에 맞게 (기본 1억)")
    t.add_argument("--holdout", type=int, default=200)
    t.add_argument("--src", action="append", nargs="+", required=True, metavar="K=V",
                   help="소스 1개 = key=value 들. 쉼표로 잇거나 공백으로 나눠 써도 됨(PowerShell 이 쉼표 인자를 쪼개도 동작). "
                        "예) --src hf=org/name config=kor_Hang field=text weight=0.7  /  --src file=path.txt weight=0.1")
    a = ap.parse_args(argv)
    sources = [_parse_src(",".join(toks)) for toks in a.src]
    gen, held, st = mix_corpus(sources, a.total_chars, a.holdout)
    tok = train_tokenizer(gen, a.vocab_size, a.out)
    rep = report(tok, held)
    rep["used_chars"] = {s["name"]: s["used"] for s in st}
    json.dump(rep, open(a.out + ".report.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"\n=> 모델 cfg.vocab_size 를 {rep['padded_vocab_size']} 로 설정하세요 (tokenizer vocab {rep['vocab_size']}).")


if __name__ == "__main__":
    main()
