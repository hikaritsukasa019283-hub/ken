import json
import os

import numpy as np
import pytest

from graphmoe.build import QualityFilter, build, doc_hash
from graphmoe.data import ByteTokenizer

TOK = ByteTokenizer()


def mk_docs(n=400, dup_every=10):
    """고유 문서 + 일부 정확 중복 + 일부 불량 문서."""
    docs = []
    for i in range(n):
        docs.append(f"문서 번호 {i} 입니다. " + "한국어 수학 문장 x^2+1=0 ".join(str(i) for _ in range(8)) + f" #{i}")
        if i % dup_every == 0:
            docs.append(docs[-1])                      # 정확 중복
        if i % 50 == 0:
            docs.append("짧음")                         # too_short
    return docs


def factory(docs):
    return lambda n: iter(docs[n:])


def read_docs(path):
    ids = np.fromfile(path, dtype=np.uint16)
    out, cur = [], []
    for t in ids.tolist():
        if t == TOK.eos_id:
            out.append(bytes(cur).decode("utf-8")); cur = []
        else:
            cur.append(t)
    return out


def test_filter_reasons():
    qf = QualityFilter(min_chars=10, min_hangul_ratio=0.3)
    good = "안녕하세요 한국어 문장입니다 " * 3
    assert qf.reason(good) is None
    assert qf.reason("짧음") == "too_short"
    assert qf.reason("a" * 60 + " 안녕하세요 한국어 문장입니다") == "char_run"
    assert qf.reason("�" * 5 + good) == "replacement_chars"
    assert qf.reason("english only text with no hangul at all " * 3) == "low_hangul"
    assert qf.reason("\n".join(["같은 줄입니다 반복 반복"] * 6)) == "dup_lines"
    assert QualityFilter(min_chars=10, max_chars=20).reason(good) == "too_long"


def test_split_dedup_no_leakage(tmp_path):
    docs = mk_docs()
    out = str(tmp_path / "d")
    r = build(factory(docs), TOK, out, val_frac=0.2, qf=QualityFilter(min_chars=20), log=lambda *_: None)
    tr, va = read_docs(out + ".train.bin"), read_docs(out + ".val.bin")
    assert r["drop_duplicate"] == 40 and r["drop_too_short"] == 8        # 중복 40, 짧은 문서 8
    assert not (set(map(doc_hash, tr)) & set(map(doc_hash, va)))         # 겹침 0
    assert len(tr) == len(set(tr)) and len(va) == len(set(va))           # 각 쪽 내부 중복 0
    assert 0.1 < len(va) / (len(tr) + len(va)) < 0.3                     # val_frac 0.2 근처
    assert json.load(open(out + ".val.bin.json"))["n_tokens"] == r["tokens"]["val"]
    # 해시 기반이라 입력 순서를 바꿔도 같은 문서는 같은 쪽
    out2 = str(tmp_path / "d2")
    build(factory(list(reversed(docs))), TOK, out2, val_frac=0.2, qf=QualityFilter(min_chars=20), log=lambda *_: None)
    assert set(read_docs(out2 + ".val.bin")) == set(va)


def test_val_tokens_cap_and_max_tokens(tmp_path):
    docs = mk_docs()
    out = str(tmp_path / "c")
    r = build(factory(docs), TOK, out, val_frac=0.5, val_tokens=3000, max_tokens=6000,
              qf=QualityFilter(min_chars=20), log=lambda *_: None)
    assert 3000 <= r["tokens"]["val"] < 3000 + 400                       # 상한 도달 후 val 중단
    assert r["tokens"]["train"] >= 6000 and r["tokens"]["train"] < 6000 + 400
    capped = read_docs(out + ".val.bin")
    train_set = set(read_docs(out + ".train.bin"))
    assert not (set(capped) & train_set)                                 # 상한 초과분이 train 으로 새지 않음


def test_resume_is_byte_identical(tmp_path):
    docs = mk_docs()
    qf = QualityFilter(min_chars=20)
    ref = str(tmp_path / "ref")
    r_ref = build(factory(docs), TOK, ref, val_frac=0.2, qf=qf, ckpt_every=40, log=lambda *_: None)

    out = str(tmp_path / "res")

    def crashing(n):                                    # 130 번째 문서에서 죽는 스트림
        for i, d in enumerate(docs[n:], start=n):
            if i == 130:
                raise RuntimeError("network down")
            yield d

    with pytest.raises(RuntimeError):
        build(crashing, TOK, out, val_frac=0.2, qf=qf, ckpt_every=40, log=lambda *_: None)
    st = json.load(open(out + ".state.json"))
    assert 0 < st["consumed"] <= 130 and not st["done"]                  # 체크포인트가 중간에 남음
    r = build(factory(docs), TOK, out, val_frac=0.2, qf=qf, ckpt_every=40, resume=True, log=lambda *_: None)
    for ext in (".train.bin", ".val.bin"):
        assert open(out + ext, "rb").read() == open(ref + ext, "rb").read(), ext
    assert r == r_ref
    assert json.load(open(out + ".state.json"))["done"]


def test_resume_when_done_is_noop(tmp_path):
    docs = mk_docs(60)
    out = str(tmp_path / "x")
    r1 = build(factory(docs), TOK, out, qf=QualityFilter(min_chars=20), log=lambda *_: None)
    before = open(out + ".train.bin", "rb").read()
    r2 = build(factory(docs), TOK, out, qf=QualityFilter(min_chars=20), resume=True, log=lambda *_: None)
    assert r1 == r2 and open(out + ".train.bin", "rb").read() == before


def test_cli_local_lines(tmp_path):
    from graphmoe.build import main
    f = tmp_path / "c.txt"
    f.write_text("\n".join(mk_docs(80)), encoding="utf-8")
    out = str(tmp_path / "cli")
    main(["--tokenizer", "byte", "--out", out, "--lines", "--min-chars", "20", "--val-frac", "0.1", str(f)])
    assert os.path.exists(out + ".train.bin") and os.path.exists(out + ".stats.json")
