# Graph-MoE 7B 작업지침서 (사람·에이전트 공용)

> 이 문서만 읽으면 바로 일할 수 있게 쓴 문서. 작업 전 §0~§3 필독, 작업 후 §7 갱신.
> 마지막 갱신 기준: **"학습 직전(pre-training-ready)" 단계 완료**.

## 0. 한 줄 요약
스펙(아래 §1) 기반 Graph-MoE 의 **PyTorch 뼈대 + 학습 전 인프라가 완성**됐다. 아직 **본 학습은 하지 않았고**,
D1(숫자 불일치)은 rank 확대로 해결했고, 남은 결정은 §5 D2·D3.

## 1. 아키텍처 스펙 (원본, 변경 금지 — 바꾸려면 §5 에 결정으로 기록)
- 32 layers, d_model 2560, 20 Q heads / 4 KV heads (GQA), head_dim 128
- KV 공유 5:3 (8-layer 블록 중 5개가 K/V 생성, 3개는 재사용) + **8-bit KV cache**
- Shared SwiGLU 4096 (모든 레이어, 항상 활성)
- Per-layer Expert Graph: Domain 8×rank**1728** / Operation 8×rank**1280** / Control 4×rank**640**  (원 스펙 512/384/192 → D1 결정으로 확대)
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
  data.py         토크나이저 인터페이스(byte / HF), .bin 생성(로컬/HF 스트리밍), BinSampler, **MixSampler(비율 혼합)**, 합성데이터
  build.py        데이터 빌더: 품질 필터 + 정확 중복 제거 + 해시 기반 train/val 분리 + 재개(바이트 동일)
  tokenizer.py    한국어+수학 혼합 byte-level BPE 학습/로드/점검 (`file:<json>` 로 data.py 와 연결)
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
8. **MixSampler**: 샘플 단위로 소스를 가중치 확률로 뽑는다. 윈도우 길이가 seq_len 으로 고정이라 *샘플 비율 = 토큰 비율*. 시드 고정 시 재현 가능, `usage` 로 실제 비율 점검 (`test_mix_sampler_*`).

## 4. 실행법
```bash
pip install torch numpy pytest tokenizers  # transformers 는 HF 토크나이저 쓸 때만, datasets 는 HF 스트리밍 쓸 때만
python -m pytest -q tests                 # 전체 검증 (CPU, ~30s)
python -m graphmoe.budget                 # 7B 스펙 대비 파라미터/RAM 리포트
# 학습 파이프라인 스모크 (합성 데이터, 수 초)
python -m graphmoe.train --preset tiny --synthetic --steps 60 --batch 8 --seq-len 32 --lr 3e-3 --out runs/smoke
# 토크나이저 학습 (한국어+수학 혼합; total-chars 는 PC RAM 에 맞춰 — BPE 학습이 RAM 을 많이 씀)
python -m graphmoe.tokenizer train --out tok/ko_math.json --vocab-size 64000 --total-chars 100000000 \
       --src hf=HuggingFaceFW/fineweb-2,config=kor_Hang,field=text,weight=0.7 \
       --src hf=HuggingFaceTB/finemath,config=finemath-4plus,field=text,weight=0.25 --src file=extra.txt,weight=0.05
#   -> tok/ko_math.json.report.json 의 padded_vocab_size 를 cfg.vocab_size 로, 이후 --tokenizer file:tok/ko_math.json
# 데이터 빌드 (소스별로 1회씩: train/val 자동 분리, 중복 제거, 품질 필터, --resume 로 끊겨도 이어받기)
python -m graphmoe.build --tokenizer file:tok/ko_math.json --out data/ko --hf-dataset HuggingFaceFW/fineweb-2 \
       --hf-config kor_Hang --text-field text --max-tokens 1000000000 --val-frac 0.005 --val-tokens 2000000 --min-hangul 0.3 --resume
#   -> data/ko.train.bin, data/ko.val.bin, data/ko.stats.json (필터/중복 통계). 수학 소스는 --min-hangul 없이 따로 빌드
# 실데이터
python -m graphmoe.data --tokenizer hf:<name> --out data/train.bin corpus1.txt corpus2.txt
python -m graphmoe.train --preset mini --data data/train.bin --val data/val.bin --seq-len 1024 \
       --batch 8 --accum 4 --bf16 --grad-ckpt --steps 20000 --save-every 1000 --eval-every 500 --out runs/mini0
# 혼합 학습 (경로=가중치, 쉼표 구분; Windows 경로의 ':' 때문에 '=' 사용). log.jsonl 의 "mix" 로 실제 비율 확인
python -m graphmoe.train --preset mini --mix "data/ko.train.bin=0.7,data/math.train.bin=0.25,data/en.train.bin=0.05" \
       --val-mix "data/ko.val.bin=0.7,data/math.val.bin=0.25,data/en.val.bin=0.05" --seq-len 1024 --out runs/mix0
```
출력 `runs/<name>/log.jsonl` 의 `*_dead`(죽은 expert), `*_entropy`(0~1, 1=균등), `paths_used` 를 반드시 확인.

## 5. 열린 결정사항 (사용자 결정 필요 — 에이전트가 임의로 정하지 말 것)
| ID | 내용 | 현재 가정 |
|---|---|---|
| ~~D1~~ ✅ | **해결(rank 확대)**: 512/384/192 → 1728/1280/640. 계산값 총 **6.76B**(목표 6.8B, -0.6%) / 활성 **2.32B**(목표 2.4B, -3%). 원 스펙 rank 로는 총 3.03B/활성 1.83B 였음 | 적용 완료 |
| D2 | 데이터: 사용자 요구 = **한국어 + 수학 강화 사전학습**. 후보·검증 상태는 §9. (`Anthropic/hh-rlhf` 는 선호 데이터라 사전학습 부적합 → SFT/선호 단계용). 토크나이저는 자체 학습(T11 도구 완료, 실학습 대기) | 혼합 샘플러 완료. 한국어 웹 70 / 수학 25 / 기타 5 로 시작(근거 없는 시작값, T13 에서 비교). vocab 64000 은 가정값 |
| ~~D3~~ ✅ | **해결**: 사용자 PC RAM 이 작아 *작을수록 좋음*. 4.3~4.8GB 는 목표가 아니라 **상한**으로 해석. 현 계산값 ≈2.2GB (8K ctx) 로 상한 이내 → 추가 조정 불필요. 단 RAM 을 더 줄이고 싶으면 ctx 축소/KV 비트 하향 가능 | 상한 해석 적용 |
| D4 | "Shared SwiGLU" = 모든 레이어에 각각 있는 항상-활성 FFN (레이어 간 가중치 공유 아님) | 레이어별 독립 |
| D5 | "5:3 KV-sharing" = 8-layer 블록당 5 owner : 3 sharer | 20 owner / 12 sharer |
| D6 | Expert 합성 방식(§3-7 의 체인 + tier 별 up-proj 잔차 기여) | 이 방식 |

rank/expert 수는 `config.py` 만 바꾸면 되고 `budget.py` 로 즉시 재계산된다.

## 6. 작업 보드 (다음 단계)
상태: ☐ 미착수. 각 항목은 독립 작업 가능. 완료 조건(DoD)을 만족해야 완료.

| ID | 작업 | 선행 | DoD |
|---|---|---|---|
| T1 | HF 데이터셋 3종(§9)을 `.bin` 으로 빌드 (스트리밍·혼합 도구는 완료. **다운로드는 사용자 PC 에서** — 클라우드는 huggingface.co 차단) | T11 | ✅ **도구 완료** (byte-level BPE, 숫자 개별 분리, NFC, 혼합 비율 학습, 홀드아웃 점검 리포트). **남은 일: 실제 코퍼스로 vocab 64000 학습 — 사용자 PC 에서 실행**(HF 차단) 후 `cfg.vocab_size` 를 리포트의 padded 값으로 동기화 | - | `tok/*.json` + report.json (도메인별 chars/token, roundtrip 1.0) |
| T11 | **토크나이저 학습**: 한국어+수학(LaTeX/기호) 혼합 샘플로 BPE/Unigram 학습, vocab 을 cfg.vocab_size 와 일치시킴 | - | 학습 스크립트 + 한국어/수학 텍스트의 토큰/문자 비율 리포트, `HFTokenizer` 로 로드 |
| T12 | 한국어 수학 SFT 후보 3종의 라이선스·약관 확인(GPT-4o 합성 데이터 약관 포함) 후 후반 단계 혼합 | - | 라이선스 표 + 사용 가능 여부 결정 기록 |
| T13 | 혼합 비율 파일럿 비교(mini): ko:math 비율별 val ppl(한국어)·수학 val loss 곡선 | T1,T11 | 비율별 로그 + 권장 비율 |
| T14 | ✅ **완료**: `build.py` (train/val 내용해시 분리·겹침 0, 정확 중복 제거, 가벼운 품질 필터, val 토큰 상한, --resume 바이트 동일). **남은 일**: 유사(near) 중복 제거(MinHash 등), 필터 임계값을 실데이터 통계로 튜닝 | - | `test_build.py` 6개 통과 |
| T2 | ✅ 규모(D1)·RAM(D3) 확정 완료 (6.76B / 2.32B / 2.2GB) | - | `budget.py` + 테스트 통과 |
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
- **데이터 빌더**: 중복 제거는 *정확 일치*(공백 정규화 후 해시)만 한다 → 유사 중복·보일러플레이트는 train/val 사이에 남아 검증 손실이 낙관적일 수 있음. 해시 집합은 메모리에 올라감(문서당 수십 바이트) → RAM 이 부족하면 `--no-dedup`. 재개 시 HF 는 이미 처리한 문서 수만큼 *다시 스트리밍*(토큰화는 안 함, 네트워크만 소모). 품질 필터 임계값은 일반 휴리스틱이며 **실데이터로 튜닝 안 됨**(`stats.json` 의 drop 비율을 보고 조정). `--min-hangul` 은 한국어 소스에만.
- **토크나이저**: NFC 정규화를 하므로 자모 분리형 입력은 decode 시 음절형으로 돌아온다(무손실은 NFC 입력 기준). 학습 코퍼스 소스가 먼저 고갈되면 그 소스는 예산보다 적게 쓰이고 나머지가 계속 채운다(비율이 약간 달라짐 → `report.json.used_chars` 확인). `--src` 값에 쉼표/`=` 가 든 경로는 불가. 모델 `vocab_size` 는 토크나이저 vocab 이상이어야 하며 64 배수 권장(`padded_vocab`).
- **네트워크**: 클라우드 세션 환경이 `huggingface.co` 를 차단(프록시 403)해서 HF 데이터/토크나이저 실다운로드는 아직 미검증. 환경 설정의 Network access 에서 해당 호스트 허용 필요. 로컬 PC 에서는 영향 없음.
- **학습 메모리 ≠ 추론 메모리**: 추론 RAM 은 2.2GB 지만, 6.76B 학습은 파라미터×(fp32 가중치 4 + grad 4 + Adam 8) = 16B/param ≈ **108GB + activation** 이 필요하다. 사용자 로컬 PC 에서 7b 학습은 불가 → 로컬은 `mini`(0.11B ≈ 1.8GB + activation) 로 파이프라인 검증, 7b 는 클라우드 GPU(다중) 필요 (T6).
- 풀사이즈 7B 는 CPU 에서 인스턴스화하지 말 것 (`torch.device("meta")` 로 구조만 확인). 실제 forward 검증은 tiny/mini.
- STE 형태 `w + (q-w).detach()` 는 값이 q 와 ulp 단위로 다를 수 있다 (테스트에서 round 후 비교).
- `grad_ckpt` 는 sharer 가 owner K/V 를 클로저로 읽기 때문에 owner→sharer 구간 activation 이 일부 유지된다 (메모리 절감 일부 제한).
- 라우터 초기엔 dead expert 가 많다 (tiny 60 step 실험: domain dead 1.0→0.25). aux loss 가 필요하며 T4 로 개선 여지.
- 학습 루프는 단일 디바이스 전용. 비유한(NaN/Inf) grad 스텝은 건너뛴다.
- `Int8KVCache` 는 batch 고정·max_seq_len 사전할당. 8K 초과 문맥 미지원.

## 9. 데이터 후보 조사 (한국어 + 수학, 사전학습용)
> 출처: 웹 검색 결과(2차 자료). **huggingface.co 가 클라우드 환경에서 차단돼 원문 페이지는 직접 확인하지 못함** → 사용 전 각 데이터셋 카드에서 라이선스·필드명·설정명 재확인 필수. 표의 "미확인" 은 검색에서 못 찾은 것.

| 용도 | 데이터셋 | 규모 | 라이선스 | 비고 |
|---|---|---|---|---|
| 한국어 웹 (1순위) | `HuggingFaceFW/fineweb-2`, 설정 `kor_Hang` | 6,087만 문서 / 486억 **단어** / 213GB | ODC-By 1.0 | 단어≠토큰. 토큰 수는 토크나이저 확정 후 측정 |
| 한국어 웹 (대안) | CulturaX `ko` | 한국어 분량 미확인 | 미확인 (mC4/OSCAR 약관 승계로 기억 — 상업 이용 확인 필요) | |
| 한국어 (보조) | `wikimedia/wikipedia` ko 판 | 미확인 | CC-BY-SA (기억) | 검색으로 미검증 |
| 수학 | `nvidia/Nemotron-CC-Math-v1` | 3+ 1,330억 / 4+ 520억 토큰 | CC-BY 4.0 | 가장 큼. 영어 위주(미확인) |
| 수학 | `HuggingFaceTB/finemath` | 3+ 340억 / 4+ 96억 토큰 | 미확인 | 소형 파일럿엔 4+ 로 충분 |
| 수학 | `open-web-math/open-web-math` | 147억 토큰 | ODC-By 1.0 | |
| 한국어 수학 (SFT용, 사전학습엔 소규모) | `kuotient/orca-math-korean-dpo-pairs`, `nayohan/math-gpt-4o-200k-ko`, `youjunhyeok/PersonaHub-ko`(reasoning) | 각 10만~100만 건 | 미확인 (GPT-4o 합성 → 약관 확인, T12) | 대규모 한국어 수학 *사전학습* 코퍼스는 못 찾음 |

혼합 시작안(mini 파일럿): 한국어 웹 0.70 / 수학 0.25 / 영어·위키 0.05. 규모 감각: mini(0.11B) ≈ 20억 토큰(파라미터당 ~20토큰 경험칙), uint16 기준 약 4GB.
