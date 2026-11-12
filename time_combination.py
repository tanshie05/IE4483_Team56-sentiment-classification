"""Rubric (f): CPU inference time of the submitted CNN + baseline combination, measured like time_inference.py.

Run on the same CPU node as results/cpu_inference/ (cpu-2, 4 threads):
    sbatch --nodelist=cpu-2 jobs/slurm/cpu_job.sh python time_combination.py --out-dir results/cpu_inference
Writes NEW files only: combination_inference.csv and combination_inference.md (cpu_inference.* are left untouched).

Same method as time_inference.py: full validation set (1,078 reviews), one warm-up pass, then the median of 5 runs, model
loading excluded, CNN batch size 64, the baseline scores the set in one call. Times the submitted final models
(results/final/, fitted on 7,182 reviews; speed does not depend on which reviews were used for fitting):
the baseline alone, the CNN alone, and the combination end to end (both pipelines + averaging + threshold), all in one job.
"""
import argparse
import json
import os
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch

import final_utils as fu
from time_inference import cpu_name
from train_cnn import DEFAULT_CONFIG, md_table, median_ms_per_1000

FINAL = Path('results/final')
ROUNDS = 7


def main(out_dir):
    threads = int(os.environ.get('SLURM_CPUS_PER_TASK', torch.get_num_threads()))
    torch.set_num_threads(threads)
    repeats = DEFAULT_CONFIG['timing_repeats']
    config = json.loads((FINAL / 'final_config.json').read_text(encoding='utf-8'))
    settings = fu.cnn_settings()
    vocab = fu.load_vocab(settings)
    texts = fu.load_labelled(['val']).clean_text
    n, batch = len(texts), settings['batch_size']
    cnn_path, base_path = FINAL / 'cnn_final.pt', FINAL / 'baseline_final.joblib'
    cnn = fu.build_cnn(settings, len(vocab), torch.load(cnn_path, map_location='cpu', weights_only=True), 'cpu')
    baseline = joblib.load(base_path)
    tfidf, clf = baseline.named_steps['tfidf'], baseline.named_steps['clf']
    threshold = config['combination_threshold']

    def cnn_forward(tokens):
        with torch.no_grad():
            return torch.cat([torch.sigmoid(cnn(tokens[i:i + batch])) for i in range(0, len(tokens), batch)]).numpy()

    def combination():
        p_cnn = cnn_forward(fu.encode_texts(texts, vocab, settings))
        p_base = clf.predict_proba(tfidf.transform(texts))[:, 1]
        return ((p_cnn + p_base) / 2 >= threshold).astype(int)

    tokens, features = fu.encode_texts(texts, vocab, settings), tfidf.transform(texts)
    measures = {
        'baseline_pre': lambda: tfidf.transform(texts),
        'baseline_model': lambda: clf.predict_proba(features),
        'cnn_pre': lambda: fu.encode_texts(texts, vocab, settings),
        'cnn_model': lambda: cnn_forward(tokens),
        'combination': combination,
    }
    # The node is shared, so a single median of 5 is noisy (a first attempt timed the combination below the CNN alone).
    # Each round is the same measurement as time_inference.py (warm-up + median of 5); rounds alternate between the
    # models so slow periods hit all of them, and the reported value is the median over ROUNDS rounds.
    rounds = {key: [] for key in measures}
    for _ in range(ROUNDS):
        for key, function in measures.items():
            rounds[key].append(median_ms_per_1000(function, n, repeats))
    timing = {key: float(np.median(values)) for key, values in rounds.items()}
    spread = {key: (min(values), max(values)) for key, values in rounds.items()}
    totals = {'baseline': [a + b for a, b in zip(rounds['baseline_pre'], rounds['baseline_model'])],
              'cnn': [a + b for a, b in zip(rounds['cnn_pre'], rounds['cnn_model'])], 'combination': rounds['combination']}
    # The timed combination must give the same predictions as the saved final pipeline would (sanity check).
    reference = ((fu.cnn_probabilities(cnn, tokens, batch, 'cpu') + baseline.predict_proba(texts)[:, 1]) / 2 >= threshold).astype(int)
    if not np.array_equal(combination(), reference):
        raise SystemExit('Timed combination does not reproduce the reference predictions')

    cnn_params = sum(p.numel() for p in cnn.parameters())
    base_params = clf.coef_.size + clf.intercept_.size
    cnn_kb, base_kb = cnn_path.stat().st_size / 1024, base_path.stat().st_size / 1024
    rows = [
        {'Model (final, fitted on 7,182 reviews)': 'TF-IDF + LR (exp04)', 'Parameters': f'{base_params:,}', 'Size on disk (KB)': f'{base_kb:.1f}',
         'Preprocess ms / 1,000': f"{timing['baseline_pre']:.1f}", 'Model ms / 1,000': f"{timing['baseline_model']:.1f}",
         'Total CPU ms / 1,000': f"{timing['baseline_pre'] + timing['baseline_model']:.1f}", 'Range over rounds': f"{min(totals['baseline']):.1f}–{max(totals['baseline']):.1f}"},
        {'Model (final, fitted on 7,182 reviews)': 'CNN (fine-tuned GloVe, pos_weight 1)', 'Parameters': f'{cnn_params:,}', 'Size on disk (KB)': f'{cnn_kb:.1f}',
         'Preprocess ms / 1,000': f"{timing['cnn_pre']:.1f}", 'Model ms / 1,000': f"{timing['cnn_model']:.1f}",
         'Total CPU ms / 1,000': f"{timing['cnn_pre'] + timing['cnn_model']:.1f}", 'Range over rounds': f"{min(totals['cnn']):.1f}–{max(totals['cnn']):.1f}"},
        {'Model (final, fitted on 7,182 reviews)': f'Combination (submitted; mean of both, threshold {threshold})', 'Parameters': f'{cnn_params + base_params:,}',
         'Size on disk (KB)': f'{cnn_kb + base_kb:.1f}', 'Preprocess ms / 1,000': '(included)', 'Model ms / 1,000': '(included)',
         'Total CPU ms / 1,000': f"{timing['combination']:.1f}", 'Range over rounds': f"{min(totals['combination']):.1f}–{max(totals['combination']):.1f}"},
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'combination_inference.csv', 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(','.join(rows[0]) + '\n')
        for row in rows:
            handle.write(','.join(f'"{v}"' for v in row.values()) + '\n')
    base_total = timing['baseline_pre'] + timing['baseline_model']
    lines = [
        '# CPU inference of the submitted combination (rubric f)', '',
        f"Same node and method as `cpu_inference.md`: {platform.node()}, {cpu_name()}, {threads} threads, Python {platform.python_version()}, "
        f"PyTorch {torch.__version__}, scikit-learn {sklearn.__version__}, Slurm job {os.environ.get('SLURM_JOB_ID', '-')}. Full validation set ({n:,} reviews): "
        f"one warm-up pass, then the median of {repeats} runs; model loading excluded. CNN batch size {batch}; the baseline scores the set in one call. "
        f'Because the node is shared, that measurement was repeated in {ROUNDS} rounds alternating between the three models; the table shows the median over rounds '
        '(each part separately; a total is the sum of its two medians) and the range of totals. All rows timed in the same job, on the submitted final models (`results/final/`).', '',
        *md_table(rows, list(rows[0])), '',
        f"- The combination is timed end to end (CNN tokenising + forward pass, TF-IDF + logistic regression, averaging, threshold): "
        f"{timing['combination']:.1f} ms per 1,000 reviews, {timing['combination'] / base_total:.1f}× the baseline alone and "
        f"{timing['combination'] / (timing['cnn_pre'] + timing['cnn_model']):.2f}× the CNN alone.",
        '- The final models have the same architecture and size as the validation-stage models in `cpu_inference.md`. Differences between the two files '
        '(e.g. the CNN alone) are timing noise on a shared node; compare rows within one file.',
        '- The combination can come out slightly below the CNN alone plus the baseline because its CNN part is timed inside a different round; the '
        'expected cost is about CNN + baseline.', '',
    ]
    (out_dir / 'combination_inference.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=Path('results/cpu_inference'))
    args = parser.parse_args()
    for name in ('combination_inference.csv', 'combination_inference.md'):
        if (args.out_dir / name).exists():
            raise SystemExit(f'{args.out_dir / name} exists; not overwriting results')
    main(args.out_dir)
