"""챗 템플릿 인코딩 + 시퀀스 패킹 데이터 파이프라인.

- {split}.jsonl 이 없으면 {split}.tar.xz 를 자동으로 풀어 사용한다 (Colab 흐름).
- 전처리 1회: jsonl 스트리밍 토크나이즈 -> 평탄한 uint16 토큰 배열 + uint8 손실 마스크를
  cache/ 에 저장하고, 이후에는 memmap 으로 읽는다. 파일명에 vocab 크기와 토크나이저
  모델 해시가 들어간다 ({split}_tokens_v32768_ab12cd34_max2048.npy) — 같은 vocab 으로
  재학습해도 구 캐시를 오사용하지 않는다.
- 손실 마스크: assistant 응답(+종료 토큰) 구간만 1, user/템플릿 구간은 0.
- 필터: max_sample_len 을 넘는 샘플은 캐시 단계에서 제외한다 (기본: train.py 가 seq_len 을 넘김).
- 패킹: 샘플들을 그대로 이어 붙여 seq_len 단위 비중첩 윈도우로 자른다.
"""

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import Dataset

TOKENIZER_PATH = Path(__file__).resolve().parent / "tokenizer" / "spm.model"


def load_tokenizer(path: Path = TOKENIZER_PATH) -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=str(path))


def ensure_dataset(root: Path, split: str) -> Path:
    """{split}.jsonl 경로를 반환. 없으면 {split}.tar.xz 에서 풀어낸다."""
    jsonl = root / f"{split}.jsonl"
    if jsonl.exists():
        return jsonl
    archive = root / f"{split}.tar.xz"
    if not archive.exists():
        raise FileNotFoundError(f"{jsonl} 도 {archive} 도 없습니다.")
    print(f"[data] {archive.name} 압축 해제 중...")
    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(root, filter="data")
    if not jsonl.exists():
        raise FileNotFoundError(f"{archive} 를 풀었지만 {jsonl.name} 이 없습니다.")
    return jsonl


def encode_sample(sp: spm.SentencePieceProcessor, user: str, assistant: str) -> tuple[list[int], list[int]]:
    prompt_ids = [sp.bos_id()] + sp.encode(f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n")
    answer_ids = sp.encode(f"{assistant}<end_of_turn>") + [sp.eos_id()]
    ids = prompt_ids + answer_ids
    mask = [0] * len(prompt_ids) + [1] * len(answer_ids)
    return ids, mask


def build_cache(root: Path, split: str, cache_dir: Path, sp: spm.SentencePieceProcessor,
                max_sample_len: int | None = None) -> tuple[Path, Path]:
    """토큰/마스크 캐시를 만들고 경로를 반환한다.

    max_sample_len 이 주어지면 인코딩 길이가 이를 넘는 샘플은 통째로 제외한다
    (윈도우보다 긴 샘플은 질문 없는 답변 조각만 남기므로; docs/seq_len_review.md 4-2 참조).
    토크나이저 식별자(vocab 크기 + 모델 내용 해시)와 필터 조건이 캐시 파일명에 들어가므로,
    토크나이저를 재학습하거나(같은 vocab 크기라도) 값이 바뀌면 구 캐시를 읽지 않고
    별도 캐시가 만들어진다.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    sp_hash = hashlib.md5(sp.serialized_model_proto()).hexdigest()[:8]
    suffix = f"_v{sp.get_piece_size()}_{sp_hash}" + (f"_max{max_sample_len}" if max_sample_len else "")
    tokens_path = cache_dir / f"{split}_tokens{suffix}.npy"
    mask_path = cache_dir / f"{split}_mask{suffix}.npy"
    if tokens_path.exists() and mask_path.exists():
        return tokens_path, mask_path

    jsonl = ensure_dataset(root, split)
    print(f"[data] {jsonl.name} 토크나이즈 중..." + (f" (>{max_sample_len} tokens 샘플 제외)" if max_sample_len else ""))
    tokens: list[int] = []
    masks: list[int] = []
    n = dropped = dropped_tokens = 0
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            user, assistant = obj.get("user", "").strip(), obj.get("assistant", "").strip()
            if not user or not assistant:
                continue
            ids, mask = encode_sample(sp, user, assistant)
            if max_sample_len and len(ids) > max_sample_len:
                dropped += 1
                dropped_tokens += len(ids)
                continue
            tokens.extend(ids)
            masks.extend(mask)
            n += 1
            if n % 50_000 == 0:
                print(f"  {n:,}줄 처리, {len(tokens):,} tokens")

    assert sp.get_piece_size() <= 65535, "uint16 범위 초과"
    np.save(tokens_path, np.asarray(tokens, dtype=np.uint16))
    np.save(mask_path, np.asarray(masks, dtype=np.uint8))
    print(f"[data] {split}: {n:,}줄 -> {len(tokens):,} tokens 캐시 저장")
    if dropped:
        total = n + dropped
        print(f"[data] {split}: {max_sample_len} tokens 초과 {dropped:,}줄 제외 "
              f"({dropped / total * 100:.2f}% 샘플, {dropped_tokens:,} tokens)")
    return tokens_path, mask_path


class PackedDataset(Dataset):
    def __init__(self, tokens_path: Path, mask_path: Path, seq_len: int):
        self.tokens = np.load(tokens_path, mmap_mode="r")
        self.mask = np.load(mask_path, mmap_mode="r")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.tokens) // self.seq_len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = idx * self.seq_len
        e = s + self.seq_len
        return {
            "input_ids": torch.from_numpy(self.tokens[s:e].astype(np.int64)),
            "loss_mask": torch.from_numpy(self.mask[s:e].astype(np.int64)),
        }


def make_dataset(root: Path, split: str, seq_len: int, cache_dir: Path,
                 sp: spm.SentencePieceProcessor | None = None,
                 max_sample_len: int | None = None) -> PackedDataset:
    sp = sp or load_tokenizer()
    tokens_path, mask_path = build_cache(root, split, cache_dir, sp, max_sample_len)
    return PackedDataset(tokens_path, mask_path, seq_len)
