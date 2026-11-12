"""Review lengths in CNN tokens on the training partition, and how much max_len 200 or 300 truncates.
Run: python length_coverage.py --out-dir results/length_coverage
Reads only the training partition.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import tokenize

PERCENTILES = [50, 75, 90, 95, 98, 99]
MAX_LENS = [200, 300]


def main(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path('processed/split_metadata.json').read_text(encoding='utf-8'))
    train = pd.read_csv('processed/train.csv', keep_default_na=False)
    if len(train) != metadata['split_stats']['train']['rows']:
        raise ValueError('train.csv does not match split_metadata.json')
    lengths = np.array([len(tokenize(text)) for text in train.clean_text])
    stats = {'reviews': len(lengths), 'tokens': int(lengths.sum()), 'mean': float(lengths.mean()), 'max': int(lengths.max()),
             **{f'p{p}': float(np.percentile(lengths, p)) for p in PERCENTILES}}
    truncation = [{'max_len': m, 'reviews_truncated': int((lengths > m).sum()), 'reviews_truncated_pct': 100 * float((lengths > m).mean()),
                   'tokens_cut_pct': 100 * float(np.clip(lengths - m, 0, None).sum() / lengths.sum())} for m in MAX_LENS]
    pd.DataFrame([stats]).to_csv(out_dir / 'train_length_percentiles.csv', index=False, lineterminator='\n')
    pd.DataFrame(truncation).to_csv(out_dir / 'truncation.csv', index=False, lineterminator='\n')
    lines = [
        '# Review length coverage (training partition, CNN tokens)', '',
        f"{stats['reviews']:,} training reviews, {stats['tokens']:,} tokens, counted with the CNN tokenizer (`prepare_data.tokenize`). "
        'Every token takes a position, including ones that map to `<UNK>`.', '',
        '| Statistic | Tokens |', '|---|---|',
        f"| Mean | {stats['mean']:.1f} |", *[f"| {p}th percentile | {stats[f'p{p}']:.0f} |" for p in PERCENTILES], f"| Longest | {stats['max']:,} |", '',
        '| max_len | Reviews truncated | Share of reviews | Share of all tokens cut |', '|---|---|---|---|',
        *[f"| {row['max_len']} | {row['reviews_truncated']:,} | {row['reviews_truncated_pct']:.2f}% | {row['tokens_cut_pct']:.2f}% |" for row in truncation], '',
        'Truncation keeps the start of each review (padding and truncation are at the end).', '',
    ]
    (out_dir / 'length_coverage.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, default=Path('results/length_coverage'))
    main(parser.parse_args().out_dir)
