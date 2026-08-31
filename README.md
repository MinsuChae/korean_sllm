# Korean sLLM

Gemma 3 계열 구조에 MTP(미래 16토큰 예측)를 단일 모델로 포함한 소형 한국어 챗 모델.

## 구조

- **토크나이저**: SentencePiece Unigram, NFKC 정규화, vocab 10,240, byte fallback ([tokenizer/spm.model](tokenizer/spm.model) 커밋됨)
- **모델** ([model.py](model.py)): 24 layers / d_model 768 / GQA(12Q·4KV) / QK-norm / RMSNorm pre+post 샌드위치 / sliding-window local:global 5:1 / GeGLU / 임베딩 tying — 약 160M 파라미터
- **MTP**: 공유 경량 블록 + offset별 임베딩 → tied unembedding 으로 offset 1..16 미래 토큰을 병렬 예측. 손실 `L = CE_next + 0.2 · weighted_mean(CE_mtp)` (학습 전용, 추론 미사용)
- **데이터** ([data.py](data.py)): `{"user", "assistant"}` jsonl → 챗 템플릿 + assistant 구간만 손실 마스킹 → uint16 캐시 + 시퀀스 패킹

챗 템플릿:

```
<bos><start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n{assistant}<end_of_turn><eos>
```

## 사용법

### 1. 토크나이저 학습 (로컬, 1회)

```bash
python tokenizer/train_tokenizer.py   # train.jsonl 필요, tokenizer/spm.model 생성
```

### 2. 모델 학습 (Colab)

[colab_train.ipynb](colab_train.ipynb) 참조. 리포 clone → `pip install -r requirements.txt` → 학습:

```bash
python train.py --grad-checkpointing --ckpt-dir /content/drive/MyDrive/korean_sllm_ckpt
```

`train.jsonl` 이 없으면 `train.tar.xz` 를 자동으로 풀어 사용한다.

### 로컬 smoke test

```bash
python train.py --preset tiny --max-steps 30 --batch-size 2 --grad-accum 1 --eval-every 15 --save-every 30
```

## 데이터

`train.tar.xz`(301,669 샘플) / `val.tar.xz`(33,519 샘플): KoAlpaca, KULLM-v2, 한국어 위키 QA, 의학·보건의료 법령 QA. 원본 jsonl 은 용량 문제로 git 에 포함하지 않는다 (tar.xz 만 추적).
