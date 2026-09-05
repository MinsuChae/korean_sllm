#!/usr/bin/env python3
"""Compress two files into separate .tar.xz archives using LZMA preset 9e."""

import argparse
import lzma
from pathlib import Path
import tarfile


def compress_file(source: Path) -> Path:
    destination = source.with_name(source.name + ".tar.xz")
    # Exclusive creation protects any existing archive from being overwritten.
    with destination.open("xb") as output:
        try:
            with lzma.LZMAFile(
                output,
                mode="w",
                format=lzma.FORMAT_XZ,
                preset=9 | lzma.PRESET_EXTREME,
            ) as compressed:
                # Stream the archive so the input file is not loaded into memory.
                with tarfile.open(fileobj=compressed, mode="w|") as archive:
                    archive.add(source, arcname=source.name, recursive=False)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="입력 파일 두 개를 각각 최대 압축 설정(9e)의 .tar.xz로 압축합니다."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        metavar="FILE",
        default=[Path("pair_train.jsonl"), Path("pair_val.jsonl")],
        help="입력 파일 두 개 (기본값: pair_train.jsonl pair_val.jsonl)",
    )
    args = parser.parse_args()

    if len(args.files) != 2:
        parser.error("파일 인자는 생략하거나 두 개를 입력하세요.")
    if args.files[0].resolve() == args.files[1].resolve():
        parser.error("서로 다른 파일 두 개를 입력하세요.")
    for source in args.files:
        if not source.is_file():
            parser.error(f"입력 파일을 찾을 수 없습니다: {source}")
        destination = source.with_name(source.name + ".tar.xz")
        if destination.exists():
            parser.error(f"출력 파일이 이미 존재합니다: {destination}")

    for source in args.files:
        print(f"압축 중: {source}", flush=True)
        try:
            destination = compress_file(source)
        except (OSError, lzma.LZMAError, tarfile.TarError) as error:
            parser.exit(1, f"압축 실패: {source}: {error}\n")
        print(f"완료: {destination}", flush=True)


if __name__ == "__main__":
    main()
