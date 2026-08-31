"""train/ 과 val/ 의 jsonl 파일들을 각각 train.jsonl, val.jsonl 하나로 합친다.

파일명 정렬 순서대로 스트리밍하며 이어 붙인다. 빈 줄과 JSON 파싱이 안 되는 줄은 버린다.
--shuffle 을 주면 합친 뒤 시드 고정으로 줄 순서를 섞는다(메모리에 전부 올라가므로 주의).
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def merge_dir(src_dir: Path, out_path: Path) -> tuple[int, int, int]:
    """(합친 줄 수, 읽은 파일 수, 버린 줄 수)"""
    files = sorted(p for p in src_dir.glob("*.jsonl") if p.is_file() and p.resolve() != out_path.resolve())
    written = skipped = 0

    with out_path.open("w", encoding="utf-8") as out:
        for path in files:
            n = bad = 0
            with path.open(encoding="utf-8") as src:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        bad += 1
                        continue
                    out.write(line + "\n")
                    n += 1
            written += n
            skipped += bad
            note = f", 파싱 실패 {bad}줄 제외" if bad else ""
            print(f"  {path.name}: {n:,}줄{note}")

    return written, len(files), skipped


def shuffle_file(path: Path, seed: int) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    random.Random(seed).shuffle(lines)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="train/ , val/ 을 각각 하나의 jsonl 로 병합")
    parser.add_argument("--train", default=str(ROOT / "train"))
    parser.add_argument("--val", default=str(ROOT / "val"))
    parser.add_argument("--out-dir", default=str(ROOT))
    parser.add_argument("--shuffle", action="store_true", help="병합 후 줄 순서를 섞는다")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, src in (("train", Path(args.train)), ("val", Path(args.val))):
        if not src.is_dir():
            print(f"[!] {src} 디렉토리가 없습니다. 건너뜁니다.")
            continue
        out_path = out_dir / f"{name}.jsonl"
        print(f"[{name}] {src} -> {out_path}")
        written, n_files, skipped = merge_dir(src, out_path)
        if args.shuffle and written:
            shuffle_file(out_path, args.seed)
            print("  (셔플 완료)")
        note = f", 제외 {skipped:,}줄" if skipped else ""
        print(f"  => 파일 {n_files}개, 총 {written:,}줄{note}\n")


if __name__ == "__main__":
    main()
