# Graph-MoE 7B 작업지침서 (사람·에이전트 공용)

> 이 문서만 읽으면 바로 일할 수 있게 쓴 문서. 작업 전 §0~§3 필독, 작업 후 §7 갱신.
> 마지막 갱신 기준: **"학습 직전(pre-training-ready)" 단계 완료**.

## 0. 한 줄 요약
스펙(아래 §1) 기반 Graph-MoE 의 **PyTorch 뼈대 + 학습 전 인프라가 완성**됐다. 아직 **본 학습은 하지 않았고**,
스펙 숫자 불일치(§5 D1) 때문에 7B 규모 학습 전에 **사용자 결정이 필요**하다.

## 1. 아키텍처 스펙 (원본, 변경 금지 — 바꾸려면 §5 에 결정으로 기록)
- 32 layers, d_model 2560, 20 Q heads / 4 KV heads (GQA), head_dim 128
- KV 공유 5:3 (8-layer 블록 중 5개가 K/V 생성, 3개는 재사용) + **8-bit KV cache**
- Shared SwiGLU 4096 (모든 레이어, 항상 활성)
- Per-layer Expert Graph: Domain 8×rank512 / Operation 8×rank384 / Control 4×rank192
- Router: Domain top-1, Operation top-1, Control top-0/1 → 논리 경로 256/layer, 물리 factor 20/layer
- 양자화: Graph 2-bit QAT, Shared FFN 2~3-bit, Attention 3-bit, Embedding 4-bit, Router/Norm FP16
- Default context 8K. 목표: ≈6.8B 물리, ≈2.4B 활성/token, 상주 RAM ≈4.3~4.8GB

## 2. 저장소 지도
```
graphmoe/
  config.py       GraphMoEConfig + 프리셋(7b / mini ≈0.11B / tiny). kv_owner(), ffn_bits() 레이어 맵
  quant.py        quantize_codes(공통 경로) / fake_quant_weight(STE) / QuantLinear / QuantEmbedding / RMSNorm
  attention.py    GQAttention, Int8KVCache, KVContext(owner→sharer K/V 전달), RoPE
  experts.py      ExpertGraph: Domain→Operation→Control 체인 + 라우터 + 정렬 기반 top-1 dispatch + aux loss
  model.py        Block = Attn + [SharedSwiGLU ∥ ExpertGraph]; GraphMoE(forward/generate/new_caches, grad_ckpt)
  budget.py       모델 생성 없이 파라미터/활성/RAM 계산 (스펙 대비 점검)   python -m graphmoe.budget
  packing.py      2/3/4-bit 실제 비트패킹, export_packed / load_packed (추론 커널의 참조 구현)
  data.py         토크나이저 인터페이스(byte / HF), .bin 생성, BinSampler, 합성데이터
  diagnostics.py  route_stats: tier 별 엔트로피·dead expert·ctrl on-rate·사용 경로 수
  train.py        AdamW + warmup/cosine + accum + bf16 + clip + ckpt/resume + 로깅
tests/            test_smoke.py, test_pretrain_ready.py   (python -m pytest -q tests, ~30s CPU)
docs/WORKGUIDE.md 이 문서
```

## 3. 설계 불변식 (깨면 안 되는 것 — 테스트가 지킨다)
1. **QAT == 배포 수치**: 학습의 fake-quant 와 export 의 dequant 가 같은 `quantize_codes` 를 쓴다. scale 은 fp16 으로 반올림.
   양자화 로직을 바꾸면 `test_export_matches_qat_weights` 를 반드시 통과시킬 것.
2. **KV 경로 일관성**: 학습(임시 텐서)과 추론(int8 cache)은 같은 int8 양자화를 거친다 → `test_cache_matches_full_forward`.
3. **K/V proj 는 owner 레이어에만** 존재. sharer 는 owner 의 K/V 를 읽는다 (`KVContext`).
4. **dispatch 는 토큰 독립**: 배치 구성과 무관하게 토큰당 결과 동일 → `test_dispatch_matches_per_token_reference`.
5. **budget.py 파라미터 수 == 실제 모델** (`test_param_count_matches_budget`, 7b 는 meta device 로 확인).
6. **Router / Norm / edge_* 는 양자화하지 않고 weight decay 제외** (`train.build_param_groups`).
7. Expert 체인: `v` 는 (domain,op), `w` 는 (domain,op,ctrl) 에 의존. ctrl 이 꺼진 토큰은 ctrl 연산을 건너뛴다.

## 4. 실행법
```bash
pip install torch numpy pytest            # transformers 는 HF 토크나이저 쓸 때만
python -m pytest -q tests                 # 전체 검증 (CPU, ~30s)
python -m graphmoe.budget                 # 7B 스펙 대비 파라미터/RAM 리포트
# 학습 파이프라인 스모크 (합성 데이터, 수 초)
python -m graphmoe.train --preset tiny --synthetic --steps 60 --batch 8 --seq-len 32 --lr 3e-3 --out runs/smoke
# 실데이터
python -m graphmoe.data --tokenizer hf:<name> --out data/train.bin corpus1.txt corpus2.txt
python -m graphmoe.train --preset mini --data data/train.bin --val data/val.bin --seq-len 1024 \
       --batch 8 --accum 4 --bf16 --grad-ckpt --steps 20000 --save-every 1000 --eval-every 500 --out runs/mini0
```
출력 `runs/<name>/log.jsonl` 의 `*_dead`(죽은 expert), `*_entropy`(0~1, 1=균등), `paths_used` 를 반드시 확인.

## 5. 열린 결정사항 (사용자 결정 필요 — 에이전트가 임의로 정하지 말 것)
| ID | 내용 | 현재 가정 |
|---|---|---|
| **D1** | **스펙 숫자 불일치**: 이 구조의 계산값은 총 **3.03B**/활성 **1.83B**/RAM **1.16GB** (목표 6.8B/2.4B/4.3~4.8GB). expert 가 저랭크라 레이어당 43M 뿐. 해결안: (a) rank 확대 (b) expert 수 확대 (c) expert 를 full-rank 로 (d) 목표 숫자를 현실값으로 수정 | 스펙 구조 그대로 구현, 숫자는 불일치 상태 |
| D2 | vocab 크기 / 토크나이저 / 학습 코퍼스 | vocab 64000, 토크나이저 미정(byte 는 테스트용) |
| D3 | RAM 4.3~4.8GB 의 정의 (런타임 버퍼·활성·패딩 포함 여부) | 가중치+KV(8K)만 계산 |
| D4 | "Shared SwiGLU" = 모든 레이어에 각각 있는 항상-활성 FFN (레이어 간 가중치 공유 아님) | 레이어별 독립 |
| D5 | "5:3 KV-sharing" = 8-layer 블록당 5 owner : 3 sharer | 20 owner / 12 sharer |
| D6 | Expert 합성 방식(§3-7 의 체인 + tier 별 up-proj 잔차 기여) | 이 방식 |

D1 이 풀리면 `config.py` 의 rank/expert 수만 바꾸면 되도록 짜여 있다 (`budget.py` 로 즉시 재계산).

## 6. 작업 보드 (다음 단계)
상태: ☐ 미착수. 각 항목은 독립 작업 가능. 완료 조건(DoD)을 만족해야 완료.

| ID | 작업 | 선행 | DoD |
|---|---|---|---|
| T1 | 실제 토크나이저 + 코퍼스 준비 → `.bin` | D2 | `data.py` 로 train/val bin 생성, `BinSampler` 로 로드 테스트 |
| T2 | 규모 확정(D1 반영) 및 `budget.py` 재검증 | D1,D3 | 목표 숫자와 ±5% 이내 or 사용자가 승인한 대체 숫자, 테스트 통과 |
| T3 | **QAT 스케줄**: FP(qat=False) 워밍업 → 4→3→2bit 점진 하향. 현재는 `cfg.qat` on/off 만 있음 | - | 스케줄 구현 + mini 에서 bit 하향 시 loss 급등 없음 로그 |
| T4 | 라우터 안정화: z-loss, router 노이즈, aux 계수 튜닝, ctrl 게이트 임계 | - | mini 학습에서 `*_dead`=0, entropy>0.8 유지 |
| T5 | mini(0.11B) 실데이터 파일럿 학습 + val ppl 곡선 | T1 | 재현 가능한 로그/체크포인트, 라우터 진단 리포트 |
| T6 | 다중 GPU (FSDP/DDP) + expert 병렬 | T2 | 2+ GPU 에서 단일 GPU 와 loss 일치(허용오차) |
| T7 | Grouped-GEMM 으로 expert dispatch 가속 | - | `test_dispatch_*` 통과 + 속도 개선 수치 |
| T8 | 추론 커널(packed 2/3/4bit matmul, int8 KV attention) | - | `packing.py` 참조 구현과 수치 일치 |
| T9 | 평가 하네스 (ppl + 다운스트림), 장문맥(>8K) / KV 정책 | T5 | 자동 평가 스크립트 |
| T10 | 배치 generate (padding mask, sampling, 정지 토큰) — 현재 greedy·동일 길이만 | - | 가변 길이 배치에서 단일 추론과 일치 |

## 7. 협업 규칙
- **브랜치**: 지정된 작업 브랜치에서만 작업. 다른 브랜치 push·PR 생성은 사용자 요청 시에만.
- **커밋 전 필수**: `python -m pytest -q tests` 전부 통과. 실패를 숨기려고 테스트를 약화/삭제하지 말 것.
- 스펙(§1)·불변식(§3)을 바꾸는 변경은 **§5 에 결정으로 기록**하고 사용자 승인을 받는다.
- 새 기능 = 새 테스트. 수치 로직은 반드시 기준 구현(reference)과 비교 테스트를 둔다.
- 작업 끝나면 이 문서 §6 에서 해당 항목을 ✅ 로 바꾸고, 발견한 사실/제약을 §8 에 한 줄 추가.
- 설명은 **결론 먼저, 근거(수치·테스트·로그)** 순서로 보고한다.

## 8. 알려진 제약 / 발견 사항 (누적)
- 풀사이즈 7B 는 CPU 에서 인스턴스화하지 말 것 (`torch.device("meta")` 로 구조만 확인). 실제 forward 검증은 tiny/mini.
- STE 형태 `w + (q-w).detach()` 는 값이 q 와 ulp 단위로 다를 수 있다 (테스트에서 round 후 비교).
- `grad_ckpt` 는 sharer 가 owner K/V 를 클로저로 읽기 때문에 owner→sharer 구간 activation 이 일부 유지된다 (메모리 절감 일부 제한).
- 라우터 초기엔 dead expert 가 많다 (tiny 60 step 실험: domain dead 1.0→0.25). aux loss 가 필요하며 T4 로 개선 여지.
- 학습 루프는 단일 디바이스 전용. 비유한(NaN/Inf) grad 스텝은 건너뛴다.
- `Int8KVCache` 는 batch 고정·max_seq_len 사전할당. 8K 초과 문맥 미지원.
