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

[colab_train_think_weight_01.ipynb](colab_train_think_weight_01.ipynb) 참조. 리포 clone → `pip install -r requirements.txt` → 학습:

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

## 일반 텍스트 사전학습 → 인스트럭션 학습

1. [colab_pretrain.ipynb](colab_pretrain.ipynb)를 Colab GPU에서 실행합니다.
   Drive의 `korean_sllm_data/pretrain/pretrain_{train,val}.json`에 `["문서 본문", ...]` 데이터를 준비하세요.
   문서 단위로 train/val을 분리하고 중복을 제거하세요. 사전학습 코퍼스는 별도로 준비해야 합니다.
2. 사전학습 결과는 `MyDrive/korean_sllm_data/pretrain/{best.pt,last.pt,spm.model}`에 저장됩니다.
3. [colab_train_think_weight_01.ipynb](colab_train_think_weight_01.ipynb)를 실행합니다.
   SFT 데이터는 `korean_sllm_data/sft/{train,val}.jsonl`의 `{"user": "질문", "assistant": "답변"}` 형식입니다.
   사전학습 `best.pt`의 모델 구조와 가중치를 읽고 optimizer/step/best loss를 초기화합니다.
   기존 think 내부 weight=0.1을 유지하며 결과는 `MyDrive/korean_sllm_sft`에 저장됩니다.

사전학습 노트북은 저장소의 `train_pretrain.py`와 `pretrain_data.py`를 직접 사용합니다.
이 파일들과 노트북 변경사항을 함께 GitHub에 올린 뒤 Colab에서 실행하세요.
SFT 노트북에는 변경된 학습 스크립트가 포함되어 있습니다.
모델 구조와 MTP 16개는 동일하며, 토크나이저 파일을 두 단계 사이에서 바꾸지 마세요.
체크포인트의 토크나이저 SHA-256으로 호환성을 검사합니다.

공용 스크립트의 로컬 사용 예:

```bash
python train_pretrain.py --data-dir ./pretrain_corpus --ckpt-dir ./checkpoints/pretrain --epochs 5 --session-epochs 0.25
python train.py --stage sft --init-from ./checkpoints/pretrain/best.pt --ckpt-dir ./checkpoints/sft --lr 5e-5 --epochs 2
python train.py --stage sft --resume ./checkpoints/sft/last.pt --ckpt-dir ./checkpoints/sft --lr 5e-5 --epochs 3
```

로컬 `train.py`의 SFT는 기존 assistant 마스킹을 사용합니다. think 가중치 0.1은 SFT 노트북의 패치 셀에서 적용됩니다.
`--init-from`과 `--resume`은 동시에 사용할 수 없습니다. 재개 시 데이터, 배치, LR 설정도 이전 실행과 맞추세요.
사전학습은 모델/옵티마이저/스케줄/데이터 위치/RNG 상태를 복원하며, 노트북을 같은 출력 폴더로 실행하면 0.25epoch씩 총 5epoch까지 이어갑니다.
SFT 재개는 모델/옵티마이저/스텝/AMP scaler를 복원하지만 데이터 순서와 RNG 상태까지 동일하게 복원하지는 않습니다.
사전학습 캐시는 원문과 토크나이저 해시로 구분하고, 긴 문서도 버리지 않고 패킹합니다.
사전학습은 모든 다음 토큰을 학습하며 문서 사이 attention을 허용합니다.
성능은 데이터 품질·양과 학습량에 따라 달라지므로 SFT-only 기준 모델과 동일한 평가셋에서 비교하세요.
