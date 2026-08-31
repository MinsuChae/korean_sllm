"""train.jsonl 로 SentencePiece Unigram(NFKC) 토크나이저를 학습한다. (로컬 실행용)

user/assistant 텍스트를 한 줄씩 뽑아 임시 코퍼스를 만든 뒤 학습하고,
산출물 spm.model / spm.vocab 을 이 파일과 같은 디렉토리에 저장한다.

vocab 32,768 근거 (docs/model_config_review.md): 이전 10,240 은 1음절 한글 piece 가 1,645개뿐이라
byte-fallback 이 토큰의 2.3%, 한글 토큰의 45% 가 음절 단위로 쪼개졌다. 32k 는 한글 음절·형태소를
더 담아 같은 텍스트의 토큰 수를 약 20~25% 줄이고, model.ModelConfig.vocab_size 와 일치해야 한다.

사용:
  python tokenizer/train_tokenizer.py [--input train.jsonl] [--vocab-size 32768]
"""

import argparse
import json
import os
import tempfile
import unicodedata
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
SPECIAL_TURN_TOKENS = ["<start_of_turn>", "<end_of_turn>"]

NUM_CPUS = os.cpu_count() or 1


def extract_corpus(jsonl_path: Path, corpus_path: Path) -> int:
    # 순수 I/O + json 파싱이라 단일 코어로도 수십 초면 끝난다.
    # (multiprocessing 으로 나눠 봤지만 문자열 pickling 비용이 더 커서 오히려 느렸음)
    n = 0
    with jsonl_path.open(encoding="utf-8") as src, corpus_path.open("w", encoding="utf-8") as out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("user", "assistant"):
                text = obj.get(key, "").strip()
                if text:
                    out.write(text.replace("\n", " ") + "\n")
                    n += 1
    return n


def train(corpus_path: Path, vocab_size: int, num_threads: int = NUM_CPUS) -> Path:
    spm.SentencePieceTrainer.train(
        num_threads=num_threads,  # EM 학습 단계를 모든 CPU 코어에 분산
        input=str(corpus_path),
        model_prefix=str(OUT_DIR / "spm"),
        model_type="unigram",
        vocab_size=vocab_size,
        normalization_rule_name="nfkc",
        # 0.9995 에서는 커버리지 밖 희귀 문자(한자·이모지·희귀 음절)가 vocab 을 늘려도
        # byte-fallback 으로 남았다(토큰의 2.7%). docs/model_config_review.md §4 참조.
        character_coverage=0.9999,
        byte_fallback=True,
        split_digits=True,
        remove_extra_whitespaces=False,
        allow_whitespace_only_pieces=True,
        max_sentence_length=32768,
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID, unk_id=UNK_ID,
        pad_piece="<pad>", bos_piece="<bos>", eos_piece="<eos>", unk_piece="<unk>",
        user_defined_symbols=SPECIAL_TURN_TOKENS,
    )
    return OUT_DIR / "spm.model"


def validate(model_path: Path, jsonl_path: Path) -> None:
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    print(f"\n[검증] vocab_size={sp.get_piece_size()}")
    for tok in SPECIAL_TURN_TOKENS:
        print(f"  {tok} -> id {sp.piece_to_id(tok)}")

    samples = [
        "안녕하세요, 국민건강보험법 제5조를 요약해 주세요.",
        "The quick brown fox jumps over 13 lazy dogs.",
        "혈압이 140/90 mmHg 이상이면 고혈압입니다. 😊 漢字",
    ]
    for text in samples:
        ids = sp.encode(text)
        decoded = sp.decode(ids)
        ok = decoded == unicodedata.normalize("NFKC", text)
        print(f"  [{'OK' if ok else 'DIFF'}] {len(ids):3d} tokens | {text[:40]}")

    # vocab 구성: 한글 1음절 piece 수 (10k 에서는 1,645개로 상용 2,350자에 못 미쳤음)
    def is_hangul(s: str) -> bool:
        return bool(s) and all("가" <= c <= "힣" for c in s)

    n_syllable_pieces = sum(
        1 for i in range(sp.get_piece_size()) if len(sp.id_to_piece(i).replace("▁", "")) == 1
        and is_hangul(sp.id_to_piece(i).replace("▁", "")))
    print(f"  한글 1음절 piece: {n_syllable_pieces:,}개")

    # 실제 데이터 등간격 2,000줄: 토큰 수, 압축률, byte-fallback 비율, 한글 1음절 토큰 비율
    with jsonl_path.open(encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    stride = max(total_lines // 2000, 1)
    n_tokens = n_chars = n_lines = n_byte = n_hangul = n_syllable = 0
    with jsonl_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % stride:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("user", "") + " " + obj.get("assistant", "")
            ids = sp.encode(text)
            n_tokens += len(ids)
            n_chars += len(text)
            n_lines += 1
            for tok in ids:
                piece = sp.id_to_piece(tok)
                if piece.startswith("<0x"):
                    n_byte += 1
                    continue
                core = piece.replace("▁", "")
                if is_hangul(core):
                    n_hangul += 1
                    n_syllable += len(core) == 1
    print(f"  실데이터 {n_lines}줄: 평균 {n_tokens / n_lines:.0f} tokens/sample, {n_chars / n_tokens:.2f} chars/token")
    print(f"  byte-fallback {n_byte / n_tokens * 100:.2f}% of tokens | 한글 토큰 중 1음절 {n_syllable / max(n_hangul, 1) * 100:.1f}%"
          f"  (10k 기준: 2.18 chars/token, byte-fallback 2.28%, 1음절 45.5%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="SentencePiece Unigram 토크나이저 학습")
    parser.add_argument("--input", default=str(ROOT / "train.jsonl"))
    parser.add_argument("--vocab-size", type=int, default=32768)
    args = parser.parse_args()

    jsonl_path = Path(args.input)
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "corpus.txt"
        n = extract_corpus(jsonl_path, corpus_path)
        print(f"코퍼스 {n:,}줄 추출 -> 학습 시작 (vocab={args.vocab_size})")
        model_path = train(corpus_path, args.vocab_size)
    print(f"저장: {model_path}")
    validate(model_path, jsonl_path)


if __name__ == "__main__":
    main()
