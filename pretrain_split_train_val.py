"""pretrain_data의 하위 디렉토리를 각각 9:1로 분할해 JSON 문자열 배열로 저장.

사용: python3 pretrain_split_train_val.py
결과: pretrain_train/<디렉토리명>.json, pretrain_val/<디렉토리명>.json

각 디렉토리 안의 JSON/JSONL/TXT 파일을 재귀적으로 읽는다. TXT 파일 전체는
하나의 원본 레코드이며, JSON 배열의 각 원소도 하나의 원본 레코드다.
의학관련법 디렉토리는 예외로 모든 레코드를 train에 넣고 val은 빈 배열로 저장한다.
content/text 및 언어 필드는 분할 후 각각 문자열 원소로 저장한다. 따라서
같은 레코드의 ko/en은 항상 같은 split에 들어간다. 비율은 추출된 문자열 수가
아닌 원본 레코드 수 기준이다. 서로 다른 레코드 사이의 중복 제거는 하지 않는다.
JSON은 파일 하나씩 메모리에 읽고, 출력 배열은 스트리밍으로 작성한다.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
DEFAULT_FIELDS = ("content", "text", "ko", "en", "ja", "zh")


def iter_records(path):
    if path.suffix.lower() == ".txt":
        yield path.read_text(encoding="utf-8-sig")
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8-sig") as stream:
            for line_no, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_no}: 잘못된 JSON") from exc
    else:
        with path.open(encoding="utf-8-sig") as stream:
            records = json.load(stream)
        if not isinstance(records, list):
            raise ValueError(f"{path}: 최상위 JSON 배열이 필요합니다.")
        yield from records


def extract_texts(record, fields):
    if isinstance(record, str):
        values = [record]
    elif isinstance(record, dict):
        values = [record[field] for field in fields if field in record]
    else:
        raise ValueError(f"문자열 또는 객체 레코드가 필요합니다: {type(record).__name__}")
    texts = []
    for value in values:
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError("본문 필드는 문자열이어야 합니다.")
        if value.strip():
            texts.append(value)
    if isinstance(record, dict) and not any(field in record for field in fields):
        raise ValueError(f"본문 필드가 없습니다: {list(record)} (--fields로 지정 가능)")
    return texts


def iter_groups(files, fields):
    for path in files:
        for index, record in enumerate(iter_records(path), 1):
            try:
                yield extract_texts(record, fields)
            except ValueError as exc:
                raise ValueError(f"{path}: 레코드 {index}: {exc}") from exc


@contextmanager
def json_array_writer(path):
    """완성된 배열만 최종 경로로 교체한다."""
    tmp = path.with_suffix(".json.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            stream.write("[")
            first = True

            def write(text):
                nonlocal first
                stream.write("\n  " if first else ",\n  ")
                json.dump(text, stream, ensure_ascii=False)
                first = False

            yield write
            stream.write("\n]\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def split_directory(directory, train_dir, val_dir, val_ratio=0.1, seed=42,
                    fields=DEFAULT_FIELDS):
    files = sorted(p for p in directory.rglob("*")
                   if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".txt"})
    total = skipped = 0
    for texts in iter_groups(files, fields):
        if texts:
            total += 1
        else:
            skipped += 1
    n_val = min(max(round(total * val_ratio), 1), total - 1) if total > 1 else 0
    if directory.name == "의학관련법":
        n_val = 0
    rng = random.Random(f"{seed}:{directory.name}")
    val_indices = set(rng.sample(range(total), n_val))
    train_texts = val_texts = index = 0
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    with json_array_writer(train_dir / f"{directory.name}.json") as write_train, \
            json_array_writer(val_dir / f"{directory.name}.json") as write_val:
        for texts in iter_groups(files, fields):
            if not texts:
                continue
            is_val = index in val_indices
            write = write_val if is_val else write_train
            for text in texts:
                write(text)
            if is_val:
                val_texts += len(texts)
            else:
                train_texts += len(texts)
            index += 1
    return {"records": total, "train_records": total - n_val, "val_records": n_val,
            "train_texts": train_texts, "val_texts": val_texts, "empty_records": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=ROOT / "pretrain_data")
    parser.add_argument("--train", type=Path, default=ROOT / "pretrain_train")
    parser.add_argument("--val", type=Path, default=ROOT / "pretrain_val")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fields", nargs="+", default=DEFAULT_FIELDS,
                        help="추출할 본문 필드 목록 (기본: %(default)s)")
    parser.add_argument("--directories", nargs="+", help="처리할 하위 디렉토리 이름 (기본: 전체)")
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1:
        parser.error("--val-ratio는 0보다 크고 1보다 작아야 합니다.")
    src, train, val = (p.resolve() for p in (args.src, args.train, args.val))
    if not src.is_dir():
        parser.error(f"입력 디렉토리가 없습니다: {src}")
    for output in (train, val):
        if output == src or src in output.parents or output in src.parents:
            parser.error("출력 디렉토리와 입력 디렉토리는 서로 포함할 수 없습니다.")
    if train == val or train in val.parents or val in train.parents:
        parser.error("train과 val 디렉토리는 서로 포함할 수 없습니다.")
    directories = sorted(p for p in src.iterdir() if p.is_dir())
    if args.directories:
        missing = set(args.directories) - {p.name for p in directories}
        if missing:
            parser.error(f"입력에 없는 디렉토리: {sorted(missing)}")
        directories = [p for p in directories if p.name in args.directories]
    if not directories:
        parser.error("분할할 하위 디렉토리가 없습니다.")
    for directory in directories:
        print(f"처리 중: {directory.name}", flush=True)
        stats = split_directory(directory, train, val, args.val_ratio, args.seed,
                                tuple(dict.fromkeys(args.fields)))
        print(f"{directory.name}: 원본 {stats['records']:,}개 → "
              f"train {stats['train_records']:,} / val {stats['val_records']:,}; "
              f"출력 문자열 {stats['train_texts']:,} / {stats['val_texts']:,}; "
              f"빈 레코드 제외 {stats['empty_records']:,}개", flush=True)


if __name__ == "__main__":
    main()
