"""train.jsonl 로 SentencePiece Unigram(NFKC) 토크나이저를 학습한다. (로컬 실행용)

user/assistant 텍스트를 한 줄씩 뽑아 임시 코퍼스를 만든 뒤 학습하고,
산출물 spm.model / spm.vocab 을 이 파일과 같은 디렉토리에 저장한다.

사용:
  python tokenizer/train_tokenizer.py [--input train.jsonl] [--vocab-size 10240]
"""

import argparse
import json
import tempfile
import unicodedata
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
SPECIAL_TURN_TOKENS = ["<start_of_turn>", "<end_of_turn>"]


def extract_corpus(jsonl_path: Path, corpus_path: Path) -> int:
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


def train(corpus_path: Path, vocab_size: int) -> Path:
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(OUT_DIR / "spm"),
        model_type="unigram",
        vocab_size=vocab_size,
        normalization_rule_name="nfkc",
        character_coverage=0.9995,
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

    # 실제 데이터 100줄 평균 토큰 수
    n_tokens = n_chars = n_lines = 0
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if n_lines >= 100:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("user", "") + " " + obj.get("assistant", "")
            n_tokens += len(sp.encode(text))
            n_chars += len(text)
            n_lines += 1
    print(f"  실데이터 {n_lines}줄: 평균 {n_tokens / n_lines:.0f} tokens/sample, {n_chars / n_tokens:.2f} chars/token")


def main() -> None:
    parser = argparse.ArgumentParser(description="SentencePiece Unigram 토크나이저 학습")
    parser.add_argument("--input", default=str(ROOT / "train.jsonl"))
    parser.add_argument("--vocab-size", type=int, default=10240)
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
