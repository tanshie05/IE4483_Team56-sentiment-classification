"""Rubric (f): CPU inference time, parameters and size for the baseline and three CNNs, all on one machine.
Run on a CPU node: python time_inference.py --out-dir results/cpu_inference
Times the full validation set: one warm-up pass, then the median of 5 runs; model loading is excluded.
Reads only the train and validation partitions (train only for the baseline's fit time).
"""
import argparse
import io
import json
import os
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

from train_cnn import DEFAULT_CONFIG, TextCNN, benchmark_baseline, benchmark_cnn, load_data, md_table, parameter_counts

SWEEP_CONFIG = Path('configs_gpu/cnn_hparam_sweep.json')
DECISION = Path('results/cnn/hparam_sweeps/hparam_sweep/sweep_decision.json')
GPU_RUNS = {'CNN random embeddings': ('results/cnn/embeddings/glove_runs.csv', 'random', {'embedding_init': 'random', 'freeze_embedding': False}),
            'CNN GloVe frozen': ('results/cnn/embeddings/glove_runs.csv', 'glove_frozen', {'embedding_init': 'glove', 'freeze_embedding': True})}


def cpu_name():
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith('model name'):
            return line.split(':', 1)[1].strip()
    return platform.processor()


def best_run(runs):
    # Same rule as the GPU runs: highest tuned validation macro-F1, then lowest seed.
    return runs.sort_values(['tuned_macro_f1', 'seed'], ascending=[False, True], kind='mergesort').iloc[0]


def spread(values):
    return f'{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}'


def final_cnn():
    """The final config from the sweep, with the weights of a sweep variant that has exactly those settings, if one exists."""
    sweep = json.loads(SWEEP_CONFIG.read_text(encoding='utf-8'))
    settings = json.loads(DECISION.read_text(encoding='utf-8'))['settings']
    runs = pd.read_csv(Path(sweep['out_dir']) / f"{sweep['variants_prefix']}_runs.csv")
    for variant, overrides in sweep['variants'].items():
        if all({**sweep, **overrides}.get(key, DEFAULT_CONFIG[key]) == value for key, value in settings.items()):
            return settings, variant, runs[runs.variant == variant]
    return settings, None, None


def main(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    threads = int(os.environ.get('SLURM_CPUS_PER_TASK', torch.get_num_threads()))
    torch.set_num_threads(threads)
    config = dict(DEFAULT_CONFIG)
    train, val, vocab, _ = load_data(config)
    rows = []

    baseline_path = Path('results/baseline/best_baseline_pipeline.joblib')
    baseline_config = json.loads(Path('results/baseline/best_baseline_config.json').read_text(encoding='utf-8'))
    pipeline = joblib.load(baseline_path)
    timing = benchmark_baseline(pipeline, train, val.clean_text, config, baseline_path)
    clf = pipeline.named_steps['clf']
    rows.append({'Model': 'TF-IDF + LR (exp04)', 'Feature format': 'sparse TF-IDF, 5,000 unigrams', 'Parameters': f'{clf.coef_.size + clf.intercept_.size:,}',
                 'Size on disk (KB)': f"{timing['size_kb']:.1f}", 'Preprocess ms / 1,000': f"{timing['transform_ms']:.1f}",
                 'Model ms / 1,000': f"{timing['predict_ms']:.1f}", 'Total CPU ms / 1,000': f"{timing['total_ms']:.1f}",
                 'Val macro-F1, tuned': f"{baseline_config['validation_at_threshold']['macro_f1']:.4f}",
                 'Val accuracy, tuned': f"{baseline_config['validation_at_threshold']['accuracy']:.4f}",
                 'CV macro-F1': f"{baseline_config['cv']['tuned_macro_f1']['mean']:.4f} ± {baseline_config['cv']['tuned_macro_f1']['std']:.4f}",
                 'Training': f"{timing['fit_seconds']:.2f} s fit on this CPU"})

    settings, variant, final_runs = final_cnn()
    cnns = [(name, path, variant_name, overrides) for name, (path, variant_name, overrides) in GPU_RUNS.items()]
    for name, path, variant_name, overrides in cnns + [('CNN final (fine-tuned GloVe)', None, variant, {'embedding_init': 'glove', 'freeze_embedding': False, **settings})]:
        model_config = {**config, **overrides}
        runs = final_runs if path is None else pd.read_csv(path).query('variant == @variant_name')
        model = TextCNN(len(vocab), model_config['embedding_dim'], model_config['kernel_sizes'], model_config['num_filters'], model_config['dropout'])
        if runs is not None and len(runs):
            run = best_run(runs)
            weights = Path(run.weights_file)
            if weights.exists():
                model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
                size_kb, weights_note = weights.stat().st_size / 1024, f'{weights} (seed {int(run.seed)})'
            else:
                # Per-run weights are not in the public repository. Speed does not depend on weight values, and the saved
                # state dict has the same size whatever the values, so untrained weights of the same shape are timed.
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                size_kb, weights_note = len(buffer.getvalue()) / 1024, f'{weights} not found; untrained weights of the same shape'
        else:
            size_kb, weights_note = None, 'untrained weights of the final architecture (speed does not depend on weight values)'
        model.embedding.weight.requires_grad_(not model_config['freeze_embedding'])
        counts = parameter_counts(model, model_config)
        timing = benchmark_cnn(model, val.clean_text, vocab, model_config)
        cv_note = '—'
        cv_folds = Path('results/cnn/cv/cv_folds.csv')
        if path is None and cv_folds.exists():
            folds = pd.read_csv(cv_folds).query("variant == 'final'")
            cv_note = spread(folds.tuned_macro_f1)
        rows.append({'Model': name, 'Feature format': f"{'random' if model_config['embedding_init'] == 'random' else 'GloVe 100d'} embeddings"
                     f"{', frozen' if model_config['freeze_embedding'] else ''}, max_len {model_config['max_len']}",
                     'Parameters': f"{counts['total_params']:,} ({counts['trainable_params']:,} trainable)",
                     'Size on disk (KB)': '—' if size_kb is None else f'{size_kb:.1f}', 'Preprocess ms / 1,000': f"{timing['preprocess_ms_per_1000']:.1f}",
                     'Model ms / 1,000': f"{timing['forward_ms_per_1000']:.1f}", 'Total CPU ms / 1,000': f"{timing['inference_ms_per_1000']:.1f}",
                     'Val macro-F1, tuned': spread(runs.tuned_macro_f1) if runs is not None else '—',
                     'Val accuracy, tuned': spread(runs.tuned_accuracy) if runs is not None else '—', 'CV macro-F1': cv_note,
                     'Training': f"{runs.total_train_seconds.median():.1f} s per run on the a6000 GPU (median)" if runs is not None else '—',
                     'weights': weights_note})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / 'cpu_inference.csv', index=False, lineterminator='\n')
    columns = [column for column in frame.columns if column != 'weights']
    lines = [
        '# Feature format vs accuracy and resources (rubric f)', '',
        f"All timings on one CPU node: {platform.node()}, {cpu_name()}, {threads} threads, Python {platform.python_version()}, PyTorch {torch.__version__}, "
        f"scikit-learn {sklearn.__version__}. Full validation set ({len(val):,} reviews): one warm-up pass, then the median of {config['timing_repeats']} runs; model loading excluded. "
        f"CNN batch size {config['batch_size']}; the baseline scores the set in one call.", '',
        *md_table(rows, columns), '',
        '- Accuracy and macro-F1 are validation means ± sample std over 5 seeds (CNNs, all on the a6000) or the single saved model (baseline). '
        'CV macro-F1 is the 5-fold mean ± std over folds, with the same folds for both models.',
        '- GPU training time is reported separately from CPU inference and is not comparable with the baseline\'s CPU fit time.',
        '- Size on disk is the saved state dict or joblib file. A frozen embedding is still stored, so frozen GloVe has the same file size.', '',
        'Weights timed:', '', *[f"- {row['Model']}: {row['weights']}" for row in rows if 'weights' in row and isinstance(row.get('weights'), str)], '',
    ]
    (out_dir / 'cpu_inference.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, default=Path('results/cpu_inference'))
    main(parser.parse_args().out_dir)
