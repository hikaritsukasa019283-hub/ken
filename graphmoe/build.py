"""데이터 빌더: 문서 스트림 -> (품질 필터, 정확 중복 제거, 해시 기반 train/val 분리) -> 토큰 .bin 2개.

  - **분리는 문서 내용 해시**로 결정: 실행/재개/순서와 무관하게 같은 문서는 항상 같은 쪽 -> train·val 겹침 불가.
    (정확히 같은 문서는 중복 제거로 한 번만 남기므로 양쪽에 동시에 들어갈 수도 없다.
     *유사* 중복(near-dup, 보일러플레이트 등)은 잡지 못한다 -> WORKGUIDE §8)
  - **재개(--resume)**: 체크포인트마다 파일 크기·소비 문서 수·통계·해시 집합을 저장. 재개 시 파일을 체크포인트
    크기로 잘라내고 이어서 처리 -> 끊김 없이 한 번에 돌린 결과와 바이트 단위로 동일.
  - 해시 집합은 문서당 8바이트(+set 오버헤드)라 수백만 문서에서도 수백 MB 수준. 메모리가 부족하면 --no-dedup.

산출물: <out>.train.bin / <out>.val.bin (+ .json meta), <out>.stats.json, <out>.state.json, <out>.hashes.npy

CLI:
  python -m graphmoe.build --tokenizer file:tok/ko_math.json --out data/ko \\
      --hf-dataset HuggingFaceFW/fineweb-2 --hf-config kor_Hang --text-field text \\
      --max-tokens 1000000000 --val-frac 0.005 --val-tokens 2000000 --min-hangul 0.3 --resume
"""
import argparse
import hashlib
import itertools
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Callable, Iterator, Optional

import numpy as np

_HANGUL = re.compile(r"[가-힣]")
_LETTER = re.compile(r"[^\W\d_]")


@dataclass
class QualityFilter:
    """언어 무관의 가벼운 휴리스틱. 하나라도 걸리면 reason 문자열 반환(= 제외)."""
    min_chars: int = 100
    max_chars: int = 200_000
    max_dup_line_frac: float = 0.3      # 문서 내 중복 줄 비율 (줄이 5개 이상일 때만)
    max_char_run: int = 50              # 같은 문자가 이만큼 이상 연속이면 제외
    max_replacement_frac: float = 0.01  # U+FFFD(깨진 인코딩) 비율
    min_hangul_ratio: float = 0.0       # >0 이면 (한글 음절 / 전체 글자) 가 이 값 이상이어야 함 (한국어 소스용)

    def reason(self, t: str) -> Optional[str]:
        n = len(t)
        if n < self.min_chars:
            return "too_short"
        if n > self.max_chars:
            return "too_long"
        if t.count("�") / n > self.max_replacement_frac:
            return "replacement_chars"
        lines = [l.strip() for l in t.split("\n") if l.strip()]
        if len(lines) >= 5 and 1 - len(set(lines)) / len(lines) > self.max_dup_line_frac:
            return "dup_lines"
        if re.search(r"(.)\1{%d,}" % self.max_char_run, t, flags=re.S):
            return "char_run"
        if self.min_hangul_ratio > 0:
            letters = len(_LETTER.findall(t))
            if letters == 0 or len(_HANGUL.findall(t)) / letters < self.min_hangul_ratio:
                return "low_hangul"
        return None


def doc_hash(t: str) -> int:
    """공백만 정규화한 내용 해시 (64bit)."""
    norm = " ".join(unicodedata.normalize("NFC", t).split())
    return int.from_bytes(hashlib.blake2b(norm.encode("utf-8"), digest_size=8).digest(), "little")


def _atomic_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def build(docs_factory: Callable[[int], Iterator[str]], tok, out: str, *, val_frac: float = 0.005,
          val_tokens: Optional[int] = None, max_tokens: Optional[int] = None,
          qf: Optional[QualityFilter] = None, dedup: bool = True, resume: bool = False,
          ckpt_every: int = 50_000, log: Callable = print) -> dict:
    """docs_factory(n) -> 앞 n 개 문서를 건너뛴 이터레이터 (재개용). 통계 dict 반환."""
    qf = qf or QualityFilter()
    tr_p, va_p = out + ".train.bin", out + ".val.bin"
    st_p, hs_p = out + ".state.json", out + ".hashes.npy"
    dtype = np.uint16 if tok.vocab_size < 2 ** 16 else np.uint32
    meta = lambda n: {"dtype": np.dtype(dtype).name, "n_tokens": n, "vocab_size": tok.vocab_size}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    consumed, tokens, stats, seen = 0, {"train": 0, "val": 0}, Counter(), set()
    if resume and os.path.exists(st_p):
        s = json.load(open(st_p))
        consumed, tokens, stats = s["consumed"], s["tokens"], Counter(s["stats"])
        for p, size in ((tr_p, s["size_train"]), (va_p, s["size_val"])):
            with open(p, "r+b") as f:
                f.truncate(size)                       # 체크포인트 이후의 부분 쓰기 폐기
        if dedup and os.path.exists(hs_p):
            seen = set(np.load(hs_p).tolist())
        log(f"resume: {consumed:,} docs 이후부터 (train {tokens['train']:,} / val {tokens['val']:,} tokens)")
        if s.get("done"):
            return dict(stats, tokens=tokens, consumed=consumed)
    else:
        for p in (tr_p, va_p, st_p, hs_p):
            if os.path.exists(p):
                os.remove(p)

    f_tr, f_va = open(tr_p, "ab"), open(va_p, "ab")
    thresh = int(val_frac * 10000)

    def checkpoint(done=False):
        f_tr.flush(); f_va.flush()
        os.fsync(f_tr.fileno()); os.fsync(f_va.fileno())
        for p, n in ((tr_p, tokens["train"]), (va_p, tokens["val"])):
            _atomic_json(p + ".json", meta(n))
        if dedup:
            np.save(hs_p, np.fromiter(seen, dtype=np.uint64, count=len(seen)))
        _atomic_json(st_p, {"consumed": consumed, "tokens": tokens, "stats": dict(stats), "done": done,
                            "size_train": os.path.getsize(tr_p), "size_val": os.path.getsize(va_p)})

    try:
        last_ckpt = consumed
        for text in docs_factory(consumed):
            if consumed - last_ckpt >= ckpt_every:      # 루프 맨 앞: 직전 문서까지 완전히 처리된 일관 상태
                checkpoint()
                last_ckpt = consumed
                log(f"[{consumed:,} docs] train {tokens['train']:,} val {tokens['val']:,} tokens")
            consumed += 1
            stats["seen"] += 1
            r = qf.reason(text)
            if r:
                stats["drop_" + r] += 1
                continue
            h = doc_hash(text)
            if dedup:
                if h in seen:
                    stats["drop_duplicate"] += 1
                    continue
                seen.add(h)
            ids = np.asarray(tok.encode(text) + [tok.eos_id], dtype=dtype)
            if (h % 10000) < thresh:                    # 검증 쪽 (내용 해시로 결정)
                if val_tokens and tokens["val"] >= val_tokens:
                    stats["drop_val_full"] += 1         # train 으로 보내지 않는다 (누수 방지)
                    continue
                f_va.write(ids.tobytes()); tokens["val"] += len(ids); stats["kept_val"] += 1
            else:
                f_tr.write(ids.tobytes()); tokens["train"] += len(ids); stats["kept_train"] += 1
                if max_tokens and tokens["train"] >= max_tokens:
                    break
    finally:
        f_tr.close(); f_va.close()
    # 정상 종료 경로에서만 done 체크포인트 (예외 시에는 직전 체크포인트가 유지됨)
    f_tr, f_va = open(tr_p, "ab"), open(va_p, "ab")
    checkpoint(done=True)
    f_tr.close(); f_va.close()
    result = dict(stats, tokens=tokens, consumed=consumed)
    _atomic_json(out + ".stats.json", result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True, help="출력 접두사 (예: data/ko -> data/ko.train.bin, data/ko.val.bin)")
    ap.add_argument("files", nargs="*", help="로컬 텍스트 파일 (기본: 파일 1개 = 문서 1개)")
    ap.add_argument("--lines", action="store_true", help="로컬 파일의 비어있지 않은 줄 하나 = 문서 하나")
    ap.add_argument("--hf-dataset"); ap.add_argument("--hf-config"); ap.add_argument("--hf-data-dir")
    ap.add_argument("--split", default="train"); ap.add_argument("--text-field", default="text")
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--val-tokens", type=int, help="검증 토큰 상한 (넘으면 이후 val 문서는 버림)")
    ap.add_argument("--max-tokens", type=int, help="학습 토큰 목표 (도달하면 종료)")
    ap.add_argument("--min-chars", type=int, default=100)
    ap.add_argument("--min-hangul", type=float, default=0.0, help="한국어 소스용 최소 한글 비율 (예: 0.3)")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=50_000)
    a = ap.parse_args(argv)
    if bool(a.hf_dataset) == bool(a.files):
        ap.error("로컬 파일들 또는 --hf-dataset 중 하나만 지정")

    from .data import get_tokenizer, iter_hf_texts
    tok = get_tokenizer(a.tokenizer)

    def raw_docs() -> Iterator[str]:
        if a.hf_dataset:
            yield from iter_hf_texts(a.hf_dataset, a.split, a.text_field, a.hf_config, a.hf_data_dir)
            return
        for p in a.files:
            txt = open(p, encoding="utf-8").read()
            if a.lines:
                yield from (l for l in txt.split("\n") if l.strip())
            else:
                yield txt

    # 재개: 이미 처리한 문서 수만큼 다시 스트리밍해서 건너뜀(토큰화는 안 함). 정확성 우선, HF 는 네트워크 비용만 든다.
    factory = lambda n: itertools.islice(raw_docs(), n, None)
    qf = QualityFilter(min_chars=a.min_chars, min_hangul_ratio=a.min_hangul)
    res = build(factory, tok, a.out, val_frac=a.val_frac, val_tokens=a.val_tokens, max_tokens=a.max_tokens,
                qf=qf, dedup=not a.no_dedup, resume=a.resume, ckpt_every=a.ckpt_every)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n학습: --val {a.out}.val.bin 또는 --val-mix 에 {a.out}.val.bin 을 사용하세요.")


if __name__ == "__main__":
    main()
