# Korean sLLM

Gemma 3 계열 구조에 MTP(미래 16토큰 예측)를 단일 모델로 포함한 소형 한국어 챗 모델.

## 구조

- **토크나이저**: SentencePiece Unigram, NFKC 정규화, vocab 32,768, byte fallback ([tokenizer/spm.model](tokenizer/spm.model) 커밋됨)
- **모델** ([model.py](model.py)): 24 layers / d_model 768 / GQA(12Q·4KV) / QK-norm / RMSNorm pre+post 샌드위치 / sliding-window local:global 5:1 / GeGLU / 임베딩 tying — 약 179M 파라미터 (구성 근거: [docs/model_config_review.md](docs/model_config_review.md))
- **MTP**: 공유 경량 블록 + offset별 임베딩 → tied unembedding 으로 offset 1..16 미래 토큰을 병렬 예측. 손실 `L = CE_next + 0.2 · weighted_mean(CE_mtp)`. 추론에서는 `generate_speculative` 가 MTP 헤드를 draft 로 쓰는 self-speculative decoding 을 지원한다
- **데이터** ([data.py](data.py)): `{"user", "assistant"}` jsonl → 챗 템플릿 + assistant 구간만 손실 마스킹 → seq_len 초과 샘플 제외(`--max-sample-len`, 기본 = seq_len) → uint16 캐시 + 시퀀스 패킹

챗 템플릿:

```
<bos><start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n{assistant}<end_of_turn><eos>
```

## 사용법

### 1. 토크나이저 학습 (로컬, 1회)

```bash
python tokenizer/train_tokenizer.py   # train.jsonl 필요, tokenizer/spm.model 생성 (vocab 32,768)
```

주의: 커밋된 `spm.model` 이 구버전(vocab 10,240)이면 `train.py` 의 vocab assert 가 실패하므로 이 단계를 먼저 재실행하고 산출물을 커밋한다. 데이터 캐시는 파일명에 vocab 이 들어가 자동으로 재생성된다.

### 2. 모델 학습 (Colab, RTX PRO 6000 96GB 런타임 기준)

[colab_train.ipynb](colab_train.ipynb) 참조. 리포 clone → `pip install -r requirements.txt` → 학습:

```bash
python train.py --epochs 4 --batch-size 8 --grad-accum 4 --ckpt-dir /content/drive/MyDrive/korean_sllm_ckpt
```

메모리 추정 ≈51 GiB (mtp 16 로짓 포함, grad-checkpointing 불필요). T4 등 16GB GPU 에서는 `--grad-checkpointing` 과 함께 mtp_n 축소 없이는 돌지 않는다.

`train.jsonl` 이 없으면 `train.tar.xz` 를 자동으로 풀어 사용한다.

### 로컬 smoke test

```bash
python train.py --preset tiny --max-steps 30 --batch-size 2 --grad-accum 1 --eval-every 15 --save-every 30
```

## 데이터

`train.tar.xz`(301,669 샘플) / `val.tar.xz`(33,519 샘플): KoAlpaca, KULLM-v2, 한국어 위키 QA, 의학·보건의료 법령 QA. 원본 jsonl 은 용량 문제로 git 에 포함하지 않는다 (tar.xz 만 추적).

토큰 길이 분포와 학습 seq_len(2048) 결정 근거는 [docs/seq_len_review.md](docs/seq_len_review.md) 참조 (vocab 32k 실측: train 전체 76.70M tokens, 샘플 평균 254 / p99 1,243 tokens, 99.76% 가 2048 이내).
