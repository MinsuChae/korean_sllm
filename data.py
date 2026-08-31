"""챗 템플릿 인코딩 + 시퀀스 패킹 데이터 파이프라인.

- {split}.jsonl 이 없으면 {split}.tar.xz 를 자동으로 풀어 사용한다 (Colab 흐름).
- 전처리 1회: jsonl 스트리밍 토크나이즈 -> 평탄한 uint16 토큰 배열 + uint8 손실 마스크를
  cache/ 에 저장하고, 이후에는 memmap 으로 읽는다.
- 손실 마스크: assistant 응답(+종료 토큰) 구간만 1, user/템플릿 구간은 0.
- 패킹: 샘플들을 그대로 이어 붙여 seq_len 단위 비중첩 윈도우로 자른다.
"""

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


def build_cache(root: Path, split: str, cache_dir: Path, sp: spm.SentencePieceProcessor) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = cache_dir / f"{split}_tokens.npy"
    mask_path = cache_dir / f"{split}_mask.npy"
    if tokens_path.exists() and mask_path.exists():
        return tokens_path, mask_path

    jsonl = ensure_dataset(root, split)
    print(f"[data] {jsonl.name} 토크나이즈 중...")
    tokens: list[int] = []
    masks: list[int] = []
    n = 0
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
            tokens.extend(ids)
            masks.extend(mask)
            n += 1
            if n % 50_000 == 0:
                print(f"  {n:,}줄 처리, {len(tokens):,} tokens")

    assert sp.get_piece_size() <= 65535, "uint16 범위 초과"
    np.save(tokens_path, np.asarray(tokens, dtype=np.uint16))
    np.save(mask_path, np.asarray(masks, dtype=np.uint8))
    print(f"[data] {split}: {n:,}줄 -> {len(tokens):,} tokens 캐시 저장")
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
                 sp: spm.SentencePieceProcessor | None = None) -> PackedDataset:
    sp = sp or load_tokenizer()
    tokens_path, mask_path = build_cache(root, split, cache_dir, sp)
    return PackedDataset(tokens_path, mask_path, seq_len)
