import json
import random

from graphmoe.data import BinSampler, get_tokenizer, write_bin
from graphmoe.tokenizer import (SPECIALS, TrainedTokenizer, mix_corpus, padded_vocab, report,
                                train_tokenizer)

KO = ["안녕하세요 오늘은 수학 공부를 합니다", "한국어 문장은 조사와 어미가 많습니다", "이차방정식의 근을 구해봅시다"]
MATH = [r"\frac{a+b}{2} \geq \sqrt{ab}", r"\int_0^1 x^2 dx = \frac{1}{3}", r"x^{2} + 2x + 1 = (x+1)^{2}"]


def _docs(n, seed, base):
    r = random.Random(seed)
    return [" ".join(r.choice(base) for _ in range(5)) + f" {r.randint(0, 99999)}" for _ in range(n)]


def _srcs():
    return [{"name": "ko", "docs": _docs(400, 0, KO), "weight": 0.7},
            {"name": "math", "docs": _docs(400, 1, MATH), "weight": 0.3}]


def test_mix_corpus_ratio_and_holdout():
    gen, held, st = mix_corpus(_srcs(), total_chars=20000, holdout=10)
    n_ko = n_math = 0
    ko_chars = math_chars = 0
    for d in gen:
        if any(c in d for c in "안녕한국"):
            ko_chars += len(d)
        else:
            math_chars += len(d)
    assert len(held["ko"]) == 10 and len(held["math"]) == 10
    assert abs(ko_chars / (ko_chars + math_chars) - 0.7) < 0.05          # 문자 예산 비율 유지
    assert all(h not in _docs(400, 0, KO)[10:] for h in held["ko"][:1])    # 홀드아웃은 학습 구간과 분리


def test_train_roundtrip_digits_and_specials(tmp_path):
    out = str(tmp_path / "tok.json")
    gen, held, _ = mix_corpus(_srcs(), total_chars=60000, holdout=20)
    tok = train_tokenizer(gen, vocab_size=600, out_path=out)
    assert tok.vocab_size <= 600 and tok.eos_id == 0 and tok.tok.token_to_id(SPECIALS[1]) == 1
    rep = report(tok, held)
    assert rep["digits_split_ok"]
    for name, d in rep["domains"].items():
        assert d["roundtrip_ok"] == 1.0, (name, d)                        # 무손실 왕복
        assert d["chars_per_token"] > 1.5, (name, d)                      # 학습 도메인은 실제로 압축됨
    s = "미지의 문자 🙂 ∑_{k=1}^n k = n(n+1)/2 와 12345"
    assert tok.decode(tok.encode(s)) == s                                  # 학습에 없던 문자도 UNK 없이 복원
    assert tok.encode("12345") and len(tok.encode("12345")) == 5           # 숫자는 한 자리씩


def test_get_tokenizer_file_and_pipeline(tmp_path):
    out = str(tmp_path / "tok.json")
    gen, _, _ = mix_corpus(_srcs(), total_chars=30000, holdout=5)
    train_tokenizer(gen, 500, out)
    tok = get_tokenizer(f"file:{out}")
    assert isinstance(tok, TrainedTokenizer)
    p = str(tmp_path / "t.bin")
    n = write_bin(_docs(30, 5, KO), tok, p)                                # bin 빌드까지 연결
    meta = json.load(open(p + ".json"))
    assert meta["vocab_size"] == tok.vocab_size and meta["n_tokens"] == n
    x, y = BinSampler(p, 16, 0).get_batch(2)
    assert x.shape == (2, 16) and int(x.max()) < tok.vocab_size


def test_padded_vocab():
    assert padded_vocab(63987) == 64000 and padded_vocab(64000) == 64000 and padded_vocab(1) == 64


def test_get_tokenizer_bare_json_and_clear_errors(tmp_path, monkeypatch):
    import sys, types
    import pytest
    out = str(tmp_path / "tok.json")
    gen, _, _ = mix_corpus(_srcs(), total_chars=30000, holdout=5)
    train_tokenizer(gen, 500, out)
    assert isinstance(get_tokenizer(out), TrainedTokenizer)               # 접두사 없는 .json 경로도 인식
    assert isinstance(get_tokenizer("file:" + out), TrainedTokenizer)
    with pytest.raises(FileNotFoundError, match="토크나이저 파일을 찾을 수 없음"):
        get_tokenizer(str(tmp_path / "nope.json"))                        # 파일이 없으면 HF 탐색 대신 명확한 에러

    class Boom:
        @staticmethod
        def from_pretrained(name):
            raise OSError(f"Can't load tokenizer for '{name}'")
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=Boom))
    with pytest.raises(RuntimeError, match="file:"):                      # HF 실패 시 file: 안내 포함
        get_tokenizer("hf:some/missing-model")
