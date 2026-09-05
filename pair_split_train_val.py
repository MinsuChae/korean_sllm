"""data 의 각 jsonl 파일을 9:1 로 나눠 train/ 과 val/ 에 저장한다.

파일별로 두 번 읽는다.
  1) 유효한 라인 수를 센다.
  2) 시드 고정 셔플로 뽑은 검증 인덱스에 따라 라인을 스트리밍으로 나눠 쓴다.
큰 파일(수백 MB)도 메모리에 통째로 올리지 않기 위한 방식이다.
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def count_valid_lines(path: Path) -> tuple[int, int]:
    """(정상 파싱된 줄 수, 파싱 실패로 버릴 줄 수)"""
    ok = bad = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                bad += 1
            else:
                ok += 1
    return ok, bad


def split_file(path: Path, train_dir: Path, val_dir: Path, val_ratio: float, seed: int) -> tuple[int, int, int]:
    total, skipped = count_valid_lines(path)
    n_val = int(round(total * val_ratio))
    if total > 1:
        n_val = min(max(n_val, 1), total - 1)  # 양쪽 모두 최소 1개는 남긴다
    else:
        n_val = 0

    rng = random.Random(f"{seed}:{path.name}")
    indices = list(range(total))
    rng.shuffle(indices)
    val_indices = set(indices[:n_val])

    idx = 0
    with path.open(encoding="utf-8") as src, \
            (train_dir / path.name).open("w", encoding="utf-8") as ftrain, \
            (val_dir / path.name).open("w", encoding="utf-8") as fval:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                continue
            out = fval if idx in val_indices else ftrain
            out.write(line + "\n")
            idx += 1

    return total, n_val, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="data 를 train/val 로 9:1 분할")
    parser.add_argument("--src", default=str(ROOT / "pair_data"))
    parser.add_argument("--train", default=str(ROOT / "pair_train"))
    parser.add_argument("--val", default=str(ROOT / "pair_val"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src_dir = Path(args.src)
    train_dir = Path(args.train)
    val_dir = Path(args.val)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src_dir.glob("*.jsonl") if p.is_file())
    if not files:
        print(f"[!] {src_dir} 에 jsonl 파일이 없습니다.")
        return

    tot_all = tot_val = tot_skipped = 0
    for path in files:
        total, n_val, skipped = split_file(path, train_dir, val_dir, args.val_ratio, args.seed)
        tot_all += total
        tot_val += n_val
        tot_skipped += skipped
        note = f", 파싱 실패 {skipped}줄 제외" if skipped else ""
        print(f"{path.name}: 전체 {total:,} -> train {total - n_val:,} / val {n_val:,}{note}")

    print("-" * 60)
    print(f"파일 {len(files)}개, 전체 {tot_all:,}줄 -> train {tot_all - tot_val:,} / val {tot_val:,}")
    if tot_skipped:
        print(f"JSON 파싱 실패로 제외한 줄: {tot_skipped:,}")


if __name__ == "__main__":
    main()
