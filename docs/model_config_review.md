# 검토 진행서 — ModelConfig 규모·MTP·vocab 결정

| 항목 | 내용 |
|---|---|
| 검토 일자 | 2026-09-01 |
| 검토 대상 | [model.py](../model.py) `ModelConfig` 전 항목 (n_layers/d_model/heads/ffn, mtp_*, vocab_size, rope/eps/id) |
| 학습 환경 | **Colab RTX PRO 6000 (96GB VRAM, bf16)** — 초기 문서의 T4 16GB 전제는 폐기 |
| 근거 자료 | 파라미터 실측(모델 인스턴스화), FLOPs 산정(6ND + dense attention), CPU 활성 메모리 실측, [seq_len_review.md](seq_len_review.md) 토큰 집계, 토크나이저 분절 실측 |
| 결론 | **24L/d768/12H/4KV/hd64/ffn2048 유지, mtp_n=16 유지(사용자 결정), vocab 10,240 → 32,768 재학습.** 총 파라미터 161.3M → 178.6M |

---

## 1. 검토 배경

`n_layers=24, d_model=768` 등 규모 계수가 데이터·하드웨어 기준으로 적정한지, MTP·vocab 등 나머지 계수가 근거를 갖는지 재검토했다. 검토 중 학습 환경이 T4 가 아니라 **RTX PRO 6000 96GB Colab 런타임**으로 확인되어, 메모리·시간을 이유로 한 축소 논거는 전부 폐기하고 데이터 예산만 남겼다.

## 2. 규모 — 파라미터 분해와 토큰 예산

파라미터 실측 (vocab 32,768 기준):

| 구성요소 | 파라미터 |
|---|---:|
| 블록 24개 (attention 37.8M + GeGLU 113.3M) | 151.1M |
| 임베딩(tied) 32,768 × 768 | 25.2M (14%) |
| MTP 헤드 | 2.4M |
| **합계** | **178.6M** (non-embedding 153.4M) |

토큰 예산 (32k 재학습 후 실측: train 전체 76.70M, 2048 초과 필터 −1.98M → **≈74.7M tokens**, 손실 대상 ≈77% ≈57.5M):

| 지표 | 값 |
|---|---:|
| 고유 토큰 / 파라미터 | 0.42 (Chinchilla 기준 ≈20 의 2% 수준) |
| 1 epoch | ≈1,140 steps (65,536 tok/step) |
| 4 epoch 총 소비 토큰 | ≈0.30B |
| 동급 모델 학습량 참고 | GPT-2 small(124M) ~10B, SmolLM2-135M 2T |

**판정: 데이터가 모델 대비 10배 이상 부족하지만, 규모는 유지한다.**

- 데이터가 고정된 조건에서 큰 모델은 토큰당 손실 감소가 더 빠르므로(Kaplan et al. 2020) 축소해서 얻는 것이 없다. 유일한 위험은 반복 학습 과적합이며, val `main_loss` 기준 조기 종료로 관리한다 (문헌상 4 epoch 까지는 반복 손해가 거의 없음 — Muennighoff et al. 2023).
- 96GB 환경에서 메모리·시간 제약 없음: 4 epoch ≈4,100 스텝, 스텝당 272 TFLOP, 유효 120 TFLOPS 가정 시 ≈2.6h.
- 근본 해법은 사전학습 코퍼스 추가(≥1B tokens)이나, 현 단계에서는 제공된 데이터만 사용하기로 결정(사용자).
- 개별 계수: `head_dim` 64·`ffn_hidden` 2048(GeGLU 관례 8/3·d)·`n_kv_heads` 4(GQA 3:1) 모두 표준값 유지. `max_seq_len`/`sliding_window`/`global_every`/rope theta/`rms_eps`/special id 도 유지 ([seq_len_review.md](seq_len_review.md) 및 토크나이저 실측 id 0~5 일치 확인).

## 3. MTP — mtp_n=16 유지 (사용자 결정) 와 그 비용

**비용 실측·산정** (batch 8 × accum 4 × seq 2048):

| 항목 | 값 |
|---|---|
| 스텝 FLOPs 중 MTP 비중 | vocab 10k 에서 41% → **32k 에서 64%** (offset 16개 × 언임베딩 T×768×V 반복이 지배) |
| CE 로짓 활성 메모리 | offset 당 fp32 log_softmax 출력 B×T×V×4B — 32k·B8 기준 17개(CE main+16) ≈34 GiB |
| CPU 실측 (B=1, T=2048, fp32, forward 후 보유 RSS) | mtp_n 0/4/16 = 4.71 / 5.15 / 6.78 GiB — offset 당 ≈0.13 GiB, 산정식과 부합 |
| 총 메모리 추정 (32k, B=8, checkpointing 없음) | 로짓 34 + MTP 은닉 3 + 트렁크 11 + 파라미터·Adam 2.7 ≈ **51 GiB** → 96GB 내 수용, **batch 16 은 ≈99 GiB 로 초과** |

**참고 문헌**: Gloeckle et al. 2024 는 n=4 최적·n=8 부터 악화, 1B 미만에서는 MTP 이득이 없거나 손해로 보고. DeepSeek-V3 는 depth 1. 선형 감쇠 가중(1.0→0.5, 합 12)으로 나누므로 offset 1 의 실질 가중은 `mtp_weight 0.2 × 1/12 ≈ 0.017`.

**결정**: 프로젝트 목표(16-토큰 MTP 단일 모델)에 따라 **mtp_n=16, mtp_weight=0.2 유지**. 이후(2026-09-01) [model.py](../model.py) `generate_speculative` 가 MTP 헤드를 draft 생성기로 쓰는 self-speculative decoding 을 추가해, mtp_n=16 은 학습 보조를 넘어 추론 가속 용도를 갖게 됐다(기본 `draft_k=8`, `return_stats` 로 offset별 수용률을 재서 튜닝). 비용은 위와 같이 수용하며, batch 는 8×4 로 고정한다. 효과 검증이 필요해지면 SFT 1 epoch × {16, 4, 0} ablation 을 val `main_loss` 로 비교한다(mtp_loss 는 n 에 따라 정의가 달라 비교 불가).

## 4. vocab 10,240 → 32,768 재학습

10k 토크나이저 실측 (val 등간격 5,587줄):

| 지표 | 10k 실측 | 32k 추정 | **32k 실측 (재학습 후)** |
|---|---:|---:|---:|
| 한글 1음절 piece | 1,645개 (상용 2,350자 미달) | 3~4천개 | **1,842개** |
| byte-fallback 토큰 비율 | 2.28% | ≈0.3~0.5% | **2.70%** (절대 개수는 동일) |
| 한글 토큰 중 1음절 비율 | 45.5% | 대폭 감소 | **29.2%** |
| chars/token | 2.18 | ≈2.7~2.9 | **2.58** |
| 같은 텍스트의 토큰 수 | 100% | ≈75~80% | **84.9%** (train 90.3M → 76.70M) |
| 한자 piece | 1개 | — | — |

**재학습 후 실측 해석 (2026-09-01)**: 압축 이득은 15%로 추정(20~25%)보다 작았고, 1음절 토큰 비율은 45.5% → 29.2% 로 크게 개선됐다. 반면 **byte-fallback 은 절대 개수가 그대로**라 줄어든 분모 때문에 비율로는 2.28% → 2.70% 로 올랐다. 원인은 vocab 크기가 아니라 `character_coverage=0.9995` — 커버리지에서 밀려난 희귀 문자(한자·이모지·기호·희귀 음절)는 vocab 을 늘려도 piece 를 받지 못한다. 1음절 piece 가 1,842개에 그친 이유도 같다. **채택 (2026-09-01)**: [tokenizer/train_tokenizer.py](../tokenizer/train_tokenizer.py) 의 `character_coverage` 기본값을 0.9999 로 올렸고, 사용자가 재학습을 완료했다.

**coverage 0.9999 최종 실측**: 한글 1음절 piece 1,842 → **2,239**, 한자 piece 0 → **369**, chars/token 2.57, train 전체 76.86M tokens(+0.2% — 희귀문자 piece 유입만큼 일반 텍스트 압축이 미세 양보). byte-fallback 은 2.70% → 2.46% 로만 줄었는데, **잔여분의 97%는 개행 문자**다(`<0x0A>` 등 1바이트 = 1토큰이라 압축 손해 없음): [tokenizer/train_tokenizer.py](../tokenizer/train_tokenizer.py) `extract_corpus` 가 코퍼스에서 `\n` 을 공백으로 치환해 토크나이저가 개행을 본 적이 없고, 챗 템플릿(`user\n`, `<end_of_turn>\n`)과 본문 개행이 전부 byte 로 인코딩된다. 개행을 제외한 **실질 미커버 문자는 0.07%** 로 목표 달성. 개행을 단일 토큰으로 만들려면 `user_defined_symbols` 에 `"\n"` 추가 + 코퍼스 개행 보존이 필요하나, 토큰 수 이득이 없어(1바이트→1토큰 동일) 채택하지 않는다.

vocab 구성(10k): 한글 1음절 1,645 / 2음절 3,301 / 3음절 1,858 / 4음절+ 1,089 / ascii 2,013 / byte 256 — 한글 슬롯 ~7,900개 안에서 음절 커버리지와 어절 커버리지가 경합하는 구조였다.

**근거**: (1) 음절 미달로 인한 byte-fallback 2.3% 는 학습 노이즈이자 생성 시 깨진 글자 위험, (2) 실효 컨텍스트·문자당 연산 20~25% 이득, (3) 동급 모델 비교(polyglot-ko 30k, KoGPT2 51k, SmolLM2-135M 49k) 및 vocab 스케일링 연구(Tao et al. 2024, non-emb 150M 급 최적 ≈16k~32k)에서 10k 는 하한 미달, (4) 한국어는 음절 수(11,172)가 커서 작은 vocab 의 손해가 영어보다 크다. 32,768(2^15)은 GEMM 정렬상 32,000 보다 유리.

**비용**: 임베딩 +17.3M(전부 룩업), MTP 언임베딩 FLOPs 49T→173T/step, 토크나이저·캐시 재생성. 캐시 파일명에 vocab 이 들어가도록 [data.py](../data.py) 를 수정해 구 캐시 오사용을 차단했다.

## 5. 반영 내역 (2026-09-01)

1. [model.py](../model.py) `vocab_size` 32768.
2. [tokenizer/train_tokenizer.py](../tokenizer/train_tokenizer.py) 기본 `--vocab-size` 32768, 검증 출력에 1음절 piece 수·byte-fallback·1음절 토큰 비율 추가 (10k 기준치 병기).
3. [data.py](../data.py) 캐시 파일명에 `_v{vocab}` 포함.
4. [train.py](../train.py) `--epochs`(윈도우 수 기반 스텝 환산), 시작 배너에 vocab·스케줄, eval 로그에 CUDA 최대 메모리.
5. [colab_train.ipynb](../colab_train.ipynb) RTX PRO 6000 기준으로 갱신: grad-checkpointing 제거, `--epochs 4`, batch 8×4 고정 사유 명시.
6. README·[seq_len_review.md](seq_len_review.md) 에 32k 전환 주석.

**미반영(후순위 기록)**: sliding window 가 학습(SDPA bool mask → dense T²)·추론([model.py](../model.py) `KVCache` 미트리밍) 어느 쪽에서도 실제 절감으로 이어지지 않음 — FlexAttention/캐시 트리밍은 필요 시 별도 작업. 사전학습 단계 추가도 데이터 확보 시 재론.

## 6. 실행 체크리스트 (토크나이저 재학습은 사용자 수행)

1. ~~`python tokenizer/train_tokenizer.py`~~ **완료 (coverage 0.9999 최종본)** — vocab 32768, id 0~5 정상, §4 실측 확인. `spm.model`/`spm.vocab` **커밋 필요**. 캐시 파일명에 토크나이저 모델 해시가 포함되므로([data.py](../data.py)) 구 캐시(해시 없는 `*_v32768*`, `*_v10240*`)는 참조되지 않는다 — 원하면 삭제.
2. ~~`docs/seq_stats.py train` 재집계~~ **완료 (0.9999 기준)** — 전체 76.86M tokens, p99 1,246, ≤2048 99.76%, 필터 후 ≈74.86M → **1 epoch ≈ 1,142 steps**.
3. ~~tiny smoke test~~ **완료** — 캐시 재생성 동작, 초기 main loss 10.42 ≈ ln(32768)=10.40 으로 정상.
4. Colab 에서 [colab_train.ipynb](../colab_train.ipynb) 실행 — 시작 배너의 "스케줄" 줄(≈1,142 steps/epoch)과 첫 eval 의 `mem`(추정 ≈51 GiB) 확인.
5. val `main_loss` 상승 전환 시 직전 체크포인트를 최종으로.
