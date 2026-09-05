"""Plain-text JSONL -> disk-backed causal-LM windows (no chat template)."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, path: Path, seq_len: int):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def __len__(self):
        # One extra target token; adjacent windows overlap by exactly one token.
        return max(0, (len(self.tokens) - 1) // (self.seq_len - 1))

    def __getitem__(self, index):
        start = index * (self.seq_len - 1)
        ids = torch.from_numpy(self.tokens[start:start + self.seq_len].astype(np.int64))
        return {"input_ids": ids, "loss_mask": torch.ones_like(ids)}


def make_pretrain_dataset(root, split, seq_len, cache_dir, sp):
    source = Path(root) / f"{split}.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f'{source}: {{"text": "본문"}} 형식의 JSONL을 준비하세요.')
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
    target = cache_dir / f"pretrain_v1_{split}_{digest.hexdigest()}.bin"
    if not target.exists():
        tmp = target.with_suffix(".tmp")
        documents = tokens = 0
        try:
            with source.open(encoding="utf-8") as stream, tmp.open("wb") as out:
                for line_no, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{source}:{line_no}: 잘못된 JSON") from exc
                    text = obj.get("text") if isinstance(obj, dict) else None
                    if not isinstance(text, str):
                        raise ValueError(f"{source}:{line_no}: text 문자열이 필요합니다.")
                    if not text.strip():
                        continue
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
    dataset = PretrainDataset(target, seq_len)
    if len(dataset) == 0:
        raise ValueError(f"{source}: 최소 {seq_len} tokens가 필요합니다.")
    return dataset
