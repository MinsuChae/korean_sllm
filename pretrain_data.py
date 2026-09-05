"""Plain-text JSON string array -> disk-backed causal-LM windows (no chat template)."""

import hashlib
import ijson
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, path: Path, seq_len: int, overlap: int = 256):
        if not 1 <= overlap < seq_len:
            raise ValueError("overlap은 1 이상 seq_len 미만이어야 합니다.")
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.stride = seq_len - overlap

    def __len__(self):
        remaining = max(0, len(self.tokens) - self.seq_len)
        return 1 + (remaining + self.stride - 1) // self.stride

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        # End-align the final window: retain every token without padding.
        # Only the final overlap may exceed the configured overlap.
        start = min(index * self.stride, max(0, len(self.tokens) - self.seq_len))
        ids = torch.from_numpy(self.tokens[start:start + self.seq_len].astype(np.int64))
        return {"input_ids": ids, "loss_mask": torch.ones_like(ids)}


def iter_texts(source):
    with Path(source).open("rb") as stream:
        events = ijson.parse(stream)
        if next(events, None) != ("", "start_array", None):
            raise ValueError(f"{source}: JSON 문자열 배열이 필요합니다.")
        for prefix, event, value in events:
            if prefix == "" and event == "end_array":
                # Consume the parser to reject trailing invalid JSON as well.
                if next(events, None) is not None:
                    raise ValueError(f"{source}: 배열 뒤에 데이터가 있습니다.")
                return
            if prefix != "item" or event != "string":
                raise ValueError(f"{source}: 모든 배열 원소는 문자열이어야 합니다.")
            yield value
        raise ValueError(f"{source}: 배열이 완성되지 않았습니다.")


def make_pretrain_dataset(root, split, seq_len, cache_dir, sp, overlap=256):
    source = Path(root) / f"pretrain_{split}.json"
    if not source.is_file():
        raise FileNotFoundError(f'{source}: ["본문", ...] 형식의 JSON 문자열 배열을 준비하세요.')
    if not 0 <= sp.bos_id() < sp.get_piece_size() or not 0 <= sp.eos_id() < sp.get_piece_size():
        raise ValueError("BOS/EOS가 있는 기존 토크나이저가 필요합니다.")
    if sp.get_piece_size() > 65536:
        raise ValueError("uint16 vocabulary overflow")
    digest = hashlib.sha256(sp.serialized_model_proto())
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"pretrain_v2_json_{split}_{digest.hexdigest()}.bin"
    if not target.exists():
        tmp = target.with_suffix(".tmp")
        documents = tokens = 0
        try:
            with tmp.open("wb") as out:
                for text in iter_texts(source):
                    ids = [sp.bos_id()] + sp.encode(text) + [sp.eos_id()]
                    np.asarray(ids, dtype=np.uint16).tofile(out)
                    documents += 1
                    tokens += len(ids)
            if tokens == 0:
                raise ValueError(f"{source}: 비어 있는 코퍼스")
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        print(f"[pretrain] {split}: {documents:,} documents / {tokens:,} tokens")
    dataset = PretrainDataset(target, seq_len, overlap)
    if len(dataset) == 0:
        raise ValueError(f"{source}: 최소 {seq_len} tokens가 필요합니다.")
    return dataset
