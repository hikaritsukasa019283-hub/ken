"""graphmoe 의 모델 정의(config, quant, attention, experts, model, budget)를 단일 .py 로 묶는다.

  python tools/bundle_model.py [출력경로]     (기본: export/graphmoe_single.py)

학습/데이터 도구(train, data, build, tokenizer, packing)는 제외 — '모델 구조' 만 담는다.
출력 파일은 자동 생성물이므로 직접 고치지 말고 graphmoe/*.py 를 고친 뒤 다시 실행할 것.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["config", "quant", "attention", "experts", "model", "budget"]

HEADER = '''"""Graph-MoE 7B — 단일 파일 모델 정의.

자동 생성: tools/bundle_model.py  (직접 수정 금지 — graphmoe/*.py 를 고친 뒤 재생성)
포함: config · quant · attention · experts · model · budget
실행: python graphmoe_single.py   -> 7B 스펙 파라미터/메모리 리포트 + tiny 모델 forward 확인
"""
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
'''

FOOTER = '''

# ======================================================================================
if __name__ == "__main__":
    report(GraphMoEConfig())                                   # 7B 스펙 수치 (모델은 만들지 않음)
    torch.manual_seed(0)
    m = GraphMoE(tiny_config()).eval()                          # 구조는 같고 규모만 축소한 확인용 모델
    ids = torch.randint(0, 256, (1, 16))
    out = m(ids)
    print(f"\\n[tiny forward] logits {tuple(out['logits'].shape)}  routes {tuple(out['routes'].shape)}"
          f"  params {sum(p.numel() for p in m.parameters()):,}")
'''


def strip(src: str) -> str:
    src = re.sub(r'\A\s*""".*?"""\s*', "", src, count=1, flags=re.S)       # 모듈 docstring
    src = re.sub(r"^(?:from|import) .*\n", "", src, flags=re.M)             # 최상위(0칸) import 줄
    src = re.sub(r'\nif __name__ == "__main__":.*\Z', "\n", src, flags=re.S)  # 모듈별 __main__ 블록
    return src.strip("\n")


def bundle(out_path) -> Path:
    parts = [HEADER]
    for name in ORDER:
        body = strip((ROOT / "graphmoe" / f"{name}.py").read_text(encoding="utf-8"))
        parts.append(f"\n\n# {'=' * 86}\n# {name}.py\n# {'=' * 86}\n{body}\n")
    parts.append(FOOTER)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = bundle(sys.argv[1] if len(sys.argv) > 1 else ROOT / "export" / "graphmoe_single.py")
    print(f"wrote {p} ({p.stat().st_size:,} bytes, {len(p.read_text().splitlines())} lines)")
