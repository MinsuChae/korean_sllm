"""pretrain_train/과 pretrain_val/의 JSON 문자열 배열을 각각 병합한다.

사용: python3 pretrain_merge_json.py
출력: pretrain_train.json, pretrain_val.json
파일명 정렬 순서와 각 배열의 순서를 유지하며 스트리밍으로 처리한다.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def iter_texts(path: Path, chunk_size: int = 1024 * 1024):
    """큰 JSON 문자열 배열을 청크 단위로 읽고 형식을 검증한다."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8-sig") as src:
        buffer = ""
        pos = 0
        eof = False

        def refill():
            nonlocal buffer, pos, eof
            chunk = src.read(chunk_size)
            buffer = buffer[pos:] + chunk
            pos = 0
            eof = not chunk

        def peek():
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos] in " \t\r\n":
                    pos += 1
                if pos < len(buffer):
                    return buffer[pos]
                if eof:
                    return ""
                refill()

        if peek() != "[":
            raise ValueError(f"{path}: 최상위 JSON 배열이 필요합니다.")
        pos += 1
        if peek() != "]":
            while True:
                if peek() != '"':
                    raise ValueError(f"{path}: 배열 원소는 문자열이어야 합니다.")
                while True:
                    try:
                        value, end = decoder.raw_decode(buffer, pos)
                        break
                    except json.JSONDecodeError as exc:
                        if eof:
                            raise ValueError(f"{path}: 잘못된 JSON 문자열") from exc
                        refill()
                pos = end
                yield value
                delimiter = peek()
                if delimiter == "]":
                    break
                if delimiter != ",":
                    raise ValueError(f"{path}: 원소 뒤에 쉼표 또는 닫는 괄호가 필요합니다.")
                pos += 1
        pos += 1
        if peek():
            raise ValueError(f"{path}: JSON 배열 뒤에 불필요한 내용이 있습니다.")


def merge_dir(src_dir: Path, out_path: Path) -> tuple[int, int]:
    files = sorted(p for p in src_dir.glob("*.json")
                   if p.is_file() and p.resolve() != out_path.resolve())
    if not files:
        raise ValueError(f"{src_dir}: 병합할 JSON 파일이 없습니다.")
    tmp = out_path.with_suffix(".json.tmp")
    written = 0
    try:
        with tmp.open("w", encoding="utf-8") as out:
            out.write("[")
            for path in files:
                count = 0
                for value in iter_texts(path):
                    out.write("\n  " if written == 0 else ",\n  ")
                    out.write(json.dumps(value, ensure_ascii=False))
                    written += 1
                    count += 1
                print(f"  {path.name}: {count:,}개", flush=True)
            out.write("\n]\n")
        tmp.replace(out_path)
    finally:
        tmp.unlink(missing_ok=True)
    return written, len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "pretrain_train")
    parser.add_argument("--val", type=Path, default=ROOT / "pretrain_val")
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    for src in (args.train, args.val):
        if not src.is_dir():
            parser.error(f"입력 디렉토리가 없습니다: {src}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, src in (("pretrain_train", args.train), ("pretrain_val", args.val)):
        out_path = args.out_dir / f"{name}.json"
        print(f"[{name}] {src} -> {out_path}", flush=True)
        written, n_files = merge_dir(src, out_path)
        print(f"  => 파일 {n_files}개, 총 {written:,}개\n", flush=True)


if __name__ == "__main__":
    main()
