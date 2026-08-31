"""train/*.jsonl 전체를 챗 템플릿으로 토크나이즈해 샘플 길이 분포를 소스별/전체로 집계한다."""
import json, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/pc/project/korean_sllm')
sys.path.insert(0, str(ROOT))
from data import load_tokenizer, encode_sample  # noqa: E402

_sp = None


def _init():
    global _sp
    _sp = load_tokenizer()


def _work(path_str):
    path = Path(path_str)
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            u, a = obj.get('user', '').strip(), obj.get('assistant', '').strip()
            if not u or not a:
                continue
            ids, mask = encode_sample(_sp, u, a)
            n_ans = sum(mask)
            rows.append((len(ids), len(ids) - n_ans, n_ans))
    return path.name, rows


def group(name: str) -> str:
    if name.startswith('koalpaca'): return 'KoAlpaca'
    if name.startswith('kullm'): return 'KULLM-v2'
    if name == 'pair_data_ko_wiki.jsonl': return 'ko_wiki QA'
    if name == 'pair_data_RM.jsonl': return 'pair_RM'
    if 'pair_data' in name: return '의학 QA'
    return '보건의료 법령 QA'


def main(split='train'):
    files = sorted(str(p) for p in (ROOT / split).glob('*.jsonl'))
    with Pool(processes=min(24, len(files)), initializer=_init) as pool:
        results = pool.map(_work, files)

    frames = []
    for name, rows in results:
        df = pd.DataFrame(rows, columns=['total', 'prompt', 'answer'])
        df['file'] = name
        df['group'] = group(name)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    q = [0.5, 0.9, 0.95, 0.99, 0.999]
    def summarize(d):
        s = d['total']
        out = {'n': len(s), 'tokens_M': s.sum() / 1e6, 'mean': s.mean(),
               'p50': s.quantile(.5), 'p90': s.quantile(.9), 'p95': s.quantile(.95),
               'p99': s.quantile(.99), 'p99.9': s.quantile(.999), 'max': s.max(),
               'ans_frac': d['answer'].sum() / s.sum()}
        for L in (512, 1024, 2048):
            out[f'<= {L}'] = (s <= L).mean() * 100
        return pd.Series(out)

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(f'=== {split}: 전체 ===')
    print(summarize(df).round(2).to_string())
    print(f'\n=== {split}: 소스 그룹별 ===')
    print(df.groupby('group').apply(summarize).round(1).sort_values('n', ascending=False).to_string())

    print(f'\n=== {split}: 길이 구간별 샘플/토큰 비중 ===')
    bins = [0, 128, 256, 512, 768, 1024, 1536, 2048, 4096, 10**9]
    labels = ['~128', '129-256', '257-512', '513-768', '769-1024', '1025-1536', '1537-2048', '2049-4096', '4097+']
    df['bin'] = pd.cut(df['total'], bins=bins, labels=labels)
    b = df.groupby('bin', observed=False)['total'].agg(samples='count', tokens='sum')
    b['samples_%'] = b['samples'] / b['samples'].sum() * 100
    b['tokens_%'] = b['tokens'] / b['tokens'].sum() * 100
    print(b.round(2).to_string())

    print(f'\n=== {split}: seq_len 후보별 패킹 지표 ===')
    total_tokens = df['total'].sum()
    mean_len = df['total'].mean()
    for L in (512, 1024, 2048):
        n_windows = total_tokens // L
        # 비중첩 윈도우 경계에 걸려 잘리는 샘플 비율 ≈ (경계 수)/(샘플 수) (샘플이 L보다 길면 추가로 잘림)
        split_frac = min(1.0, n_windows / len(df)) * 100
        over = (df['total'] > L).mean() * 100
        print(f'seq_len={L:5d}: windows {n_windows:,} | 윈도우당 평균 {L / mean_len:.1f}샘플 | '
              f'경계에 잘리는 샘플 ≈{split_frac:.1f}% | L 초과 샘플 {over:.2f}%')

    print(f'\n답변 길이(answer) 전체: mean {df.answer.mean():.0f}, p50 {df.answer.quantile(.5):.0f}, '
          f'p90 {df.answer.quantile(.9):.0f}, p99 {df.answer.quantile(.99):.0f}, max {df.answer.max()}')
    print(f'질문 길이(prompt) 전체: mean {df.prompt.mean():.0f}, p50 {df.prompt.quantile(.5):.0f}, '
          f'p90 {df.prompt.quantile(.9):.0f}, p99 {df.prompt.quantile(.99):.0f}, max {df.prompt.max()}')
    print(f'\n노트북(앞 2000샘플) 대응 확인 - KoAlpaca 앞 2000: '
          f"mean {df[df.group == 'KoAlpaca'].total.head(2000).mean():.1f}, max {df[df.group == 'KoAlpaca'].total.head(2000).max()}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'train')
