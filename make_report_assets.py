"""Report assets for rubric parts (a) and (b): pipeline walkthrough, data figures, final-model learning curve, model card.
Run: python make_report_assets.py --out-dir results/figures
Reads the training partition, saved results and split_metadata.json only. Holdout and test files are never opened;
holdout class counts come from split_metadata.json.
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from prepare_data import tokenize
from train_cnn import (AXIS_LINE, DEFAULT_CONFIG, GRID_LINE, INK, MUTED, SERIES_STYLES, TextCNN, check_shapes, encode,
                       md_table, parameter_counts)

FINAL_RUNS = Path('results/cnn/imbalance_checkpoint')  # The final config's 5 validation runs (unweighted, val-loss checkpoint).
FINAL_VARIANT = 'unweighted_val_loss'
RECIPE = Path('results/cnn/cv/cv_recipe.json')
POSITIVE, NEGATIVE = SERIES_STYLES[0][0], SERIES_STYLES[1][0]


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc='left', fontsize=11, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(axis='y', color=GRID_LINE, linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(AXIS_LINE)


def cell(text):
    return str(text).replace('|', '\\|').replace('\n', ' ')


def pipeline_example(out_dir, train, vocab):
    pipeline = joblib.load('results/baseline/best_baseline_pipeline.joblib')
    tfidf = pipeline.named_steps['tfidf']
    analyzer, features = tfidf.build_analyzer(), tfidf.vocabulary_
    lengths = train.clean_text.map(lambda text: len(tokenize(text)))
    short = train[(lengths >= 12) & (lengths <= 30)]
    # First short negative review with a negation contraction, and the first short positive review, in file order.
    negative = short[(short.sentiments == 0) & short.clean_text.str.contains("n't")].iloc[0]
    positive = short[short.sentiments == 1].iloc[0]
    columns, steps = {}, []
    for label, row in (('Review A (negative, with negation)', negative), ('Review B (positive)', positive)):
        tfidf_tokens = analyzer(row.clean_text)
        cnn_tokens = tokenize(row.clean_text)
        ids = [vocab.get(token, 1) for token in cnn_tokens]
        padded = encode([row.clean_text], vocab, DEFAULT_CONFIG['max_len'])[0].tolist()
        columns[label] = {
            'Label': f'{row.sentiments} ({"negative" if row.sentiments == 0 else "positive"})',
            'Raw text (`reviews`)': cell(row.reviews),
            'Cleaned text (`clean_text`)': cell(row.clean_text),
            'TF-IDF tokens': ' '.join(f'`{token}`' if token in features else f'`{token}`†' for token in tfidf_tokens),
            'TF-IDF features used': f'{sum(token in features for token in set(tfidf_tokens))} distinct of the 5,000-feature vocabulary (sparse vector, length 5,000)',
            'CNN tokens': ' '.join(f'`{token}`' if token in vocab else f'`{token}`‡' for token in cnn_tokens),
            'Token IDs': ' '.join(map(str, ids)),
            f"Padded to max_len {DEFAULT_CONFIG['max_len']}: first 20 IDs": ' '.join(map(str, padded[:20])),
            'Padded sequence': f"length {len(padded)}: {len(ids)} token IDs, then {padded.count(0) if len(ids) < len(padded) else 0} zeros (`<PAD>` = 0)",
        }
        steps = list(columns[label])
    rows = [{'Step': step, **{label: values[step] for label, values in columns.items()}} for step in steps]
    lines = ['# Pipeline walkthrough: two real training reviews', '',
             'Both reviews come from `processed/train.csv` (the first short negative review containing a negation, and the first short positive review). '
             'TF-IDF tokens come from the saved baseline pipeline\'s analyzer; CNN tokens from `prepare_data.tokenize`; IDs from `processed/cnn_vocab.json`.', '',
             *md_table(rows, list(rows[0])), '',
             '† not among the baseline\'s 5,000 features, so ignored by TF-IDF. ‡ not in the CNN vocabulary, so mapped to `<UNK>` = 1. '
             'Cleaning only unescapes HTML, strips tags and collapses whitespace; the TF-IDF analyzer lowercases and keeps words of 2+ characters, contractions, `!` and `?`; '
             'the CNN tokenizer lowercases and also keeps digits and other punctuation.', '']
    (out_dir / 'pipeline_example.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def class_balance(out_dir, metadata):
    stats = metadata['split_stats']
    names = [('raw_train', 'Raw train.json'), ('deduplicated', 'After deduplication'), ('train', 'Train (70%)'), ('val', 'Validation (15%)'), ('holdout', 'Holdout (15%)')]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=180)
    y = np.arange(len(names))[::-1]
    negatives = np.array([stats[key]['negative'] for key, _ in names])
    positives = np.array([stats[key]['positive'] for key, _ in names])
    ax.barh(y, negatives, color=NEGATIVE, height=.55, label='Negative (0)', edgecolor='white', linewidth=2)
    ax.barh(y, positives, left=negatives, color=POSITIVE, height=.55, label='Positive (1)', edgecolor='white', linewidth=2)
    for row, (key, _) in zip(y, names):
        total = stats[key]['rows']
        ax.text(total + 60, row, f"{total:,} reviews · {100 - stats[key]['positive_pct']:.1f}% negative", va='center', fontsize=8.5, color=INK)
    ax.set_yticks(y, [label for _, label in names])
    ax.set_xlim(0, max(stats[key]['rows'] for key, _ in names) * 1.42)
    style(ax, 'Class balance by split', 'Reviews', '')
    ax.grid(axis='x', color=GRID_LINE, linewidth=.8)
    ax.grid(axis='y', visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc='lower right', labelcolor=INK)
    fig.text(.01, .01, 'Counts from processed/split_metadata.json. Every split keeps about 85% positive reviews (stratified split).', fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(out_dir / 'class_balance.png')
    plt.close(fig)


def length_histogram(out_dir, train):
    lengths = np.array([len(tokenize(text)) for text in train.clean_text])
    cap = 600
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=180)
    ax.hist(np.clip(lengths, 0, cap), bins=np.arange(0, cap + 10, 10), color=POSITIVE, edgecolor='white', linewidth=.6)
    top = ax.get_ylim()[1]
    for max_len, height in ((200, .92), (300, .68)):
        share = 100 * (lengths > max_len).mean()
        ax.axvline(max_len, color=INK, linewidth=1.2, linestyle='--' if max_len == 200 else '-')
        ax.text(max_len + 6, top * height, f'max_len {max_len}\n{share:.1f}% of reviews truncated', fontsize=8.5, color=INK, va='top')
    style(ax, 'Training review length in CNN tokens', 'Tokens per review (reviews over 600 are counted in the last bar)', 'Reviews')
    fig.text(.01, .01, f'{len(lengths):,} training reviews; median {np.median(lengths):.0f}, 95th percentile {np.percentile(lengths, 95):.0f}, longest {lengths.max():,}. '
             'Details: results/length_coverage/length_coverage.md.', fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(out_dir / 'token_length_histogram.png')
    plt.close(fig)


def final_learning_curve(out_dir, recipe):
    history = pd.read_csv(FINAL_RUNS / 'imbalance_history.csv').query('variant == @FINAL_VARIANT')
    runs = pd.read_csv(FINAL_RUNS / 'imbalance_runs.csv').query('variant == @FINAL_VARIANT')
    mean = history.groupby('epoch').mean(numeric_only=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5.0), dpi=170)
    for ax, series in ((left, [('train_eval_loss', 'Train loss (eval mode)', SERIES_STYLES[1]), ('val_loss', 'Validation loss', SERIES_STYLES[0])]),
                       (right, [('val_macro_f1', 'Validation macro-F1 at 0.5', SERIES_STYLES[2]), ('val_tuned_macro_f1', 'Validation macro-F1, tuned per epoch', SERIES_STYLES[0])])):
        for column, label, (color, line) in series:
            ax.plot(mean.index, mean[column], color=color, linestyle=line, linewidth=2, label=label)
        ax.axvline(recipe['epochs'], color=INK, linewidth=1.2)
        ax.text(recipe['epochs'] + .1, .45, f"retraining epochs = {recipe['epochs']}\n(median best epoch in CV)", fontsize=8, color=INK, va='center',
                transform=ax.get_xaxis_transform(), bbox=dict(facecolor='white', edgecolor='none', pad=2))
        ax.set_xticks(mean.index)
    checkpoint = runs.best_epoch.median()
    left.scatter(runs.best_epoch, runs.best_val_loss, s=46, color=SERIES_STYLES[0][0], edgecolor='white', linewidth=1.2, zorder=4, label=f'Checkpoint (lowest val loss), epoch {checkpoint:.0f} in every seed')
    style(left, 'Loss by epoch', 'Epoch', 'BCE loss (pos_weight = 1)')
    style(right, 'Validation macro-F1 by epoch', 'Epoch', 'Macro-F1')
    for ax in (left, right):
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc='upper center', bbox_to_anchor=(.5, -.16), ncol=2)
    fig.text(.01, .01, f'Final config (fine-tuned GloVe, pos_weight 1, checkpoint by val loss): mean of {runs.seed.nunique()} seeds on the a6000, trained on train, scored on validation.',
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(out_dir / 'final_learning_curve.png')
    plt.close(fig)


def model_card(out_dir, vocab, recipe):
    config = {**DEFAULT_CONFIG, **recipe['settings'], 'embedding_init': 'glove'}
    torch.manual_seed(42)
    model = TextCNN(len(vocab), config['embedding_dim'], config['kernel_sizes'], config['num_filters'], config['dropout']).eval()
    shapes = check_shapes(model, torch.ones(config['batch_size'], config['max_len'], dtype=torch.long), config)
    counts = parameter_counts(model, config)
    params = {f"Embedding({len(vocab)}, {config['embedding_dim']}, padding_idx=0)": counts['embedding_params'], 'Linear (raw logit)': counts['classifier_params'],
              **{f'Conv1d(kernel={size})': counts[f'conv{size}_params'] for size in config['kernel_sizes']}}
    # Each layer's input is the previous layer's output, except that every Conv1d branch reads the transposed
    # embedding and the concatenation reads the pooled output of every branch.
    branch_input = next(row['Output shape'] for row in shapes if row['Layer'] == 'Transpose')
    pooled = next(row['Output shape'] for row in shapes if row['Layer'].startswith('Global'))
    layers, previous = [], '—'
    for row in shapes:
        if row['Layer'].startswith('Conv1d'):
            previous = branch_input
        elif row['Layer'] == 'Concatenate':
            previous = f"{len(config['kernel_sizes'])} × {pooled}"
        layers.append({'Layer': row['Layer'], 'Input shape': previous, 'Output shape': row['Output shape'], 'Parameters': f"{params.get(row['Layer'], 0):,}"})
        previous = row['Output shape']
    runs = pd.read_csv(FINAL_RUNS / 'imbalance_runs.csv').query('variant == @FINAL_VARIANT')
    cv = pd.read_csv('results/cnn/cv/cv_folds.csv').query("variant == 'final'")
    training = [
        ('Loss', 'BCEWithLogitsLoss, pos_weight = 1 (unweighted)', '2×2 experiment: unweighted beat weighted (N_neg/N_pos ≈ 0.17) by +0.0128 macro-F1 in 5 of 5 seeds; `results/cnn/imbalance_checkpoint/`'),
        ('Optimiser', 'Adam', 'fixed (Kingma & Ba, 2015)'),
        ('Learning rate', f"{config['lr']}", 'sweep: 5e-4 / 1e-3 / 2e-3, tested, no reliable gain; `results/cnn/hparam_sweeps/hparam_sweep/`'),
        ('Batch size', f"{config['batch_size']}", 'completeness sweep: 32 / 64 / 128, tested, no reliable gain; `results/cnn/hparam_sweeps/hparam_sweep_2/`'),
        ('Embedding', f"GloVe 6B {config['embedding_dim']}d, fine-tuned", 'embedding experiment: fine-tuned > frozen > random; dimension 50 / 100 / 200 / 300 tested, no reliable gain; `results/cnn/embeddings/`, `results/cnn/hparam_sweeps/hparam_sweep_2/embedding_dim/`'),
        ('Dropout', f"{config['dropout']}", 'sweep: 0.3 / 0.5, tested, no reliable gain'),
        ('max_len', f"{config['max_len']}", 'sweep: 200 / 300, tested, no reliable gain; 300 truncates 1.2% of training reviews'),
        ('Kernels and filters', f"{config['kernel_sizes']} × {config['num_filters']}", 'Kim (2014); filters 50 / 100 / 200 and kernels [3]×300, [2, 3, 4], [3, 5, 7] tested, no reliable gain'),
        ('Epochs (validation runs)', f"up to {config['max_epochs']}, early stopping on validation loss, patience {config['patience']}", f"checkpoint epoch {runs.best_epoch.median():.0f} in all {len(runs)} seeds; patience 2 / 3 / 5 tested, identical results"),
        ('Epochs (final retraining)', f"{recipe['epochs']}", f"median best epoch over {recipe['runs']} CV runs (range {recipe['epochs_min']}–{recipe['epochs_max']}); selection rule 3"),
        ('Threshold', f"{recipe['threshold']:.2f}", f"median of {recipe['runs']} thresholds tuned on CV inner-validation splits (range {recipe['threshold_min']:.2f}–{recipe['threshold_max']:.2f}); selection rule 2. Predict positive when P(positive) ≥ threshold"),
        ('Seeds', '42, 43, 44, 45, 46 (validation runs); 42, 43, 44 per CV fold', 'mean ± sample std reported'),
        ('Device', f"NVIDIA RTX A6000, node {runs.node.iloc[0]}", 'every compared CNN run on the same GPU model'),
    ]
    results = [('Validation, 5 seeds', f"macro-F1 {runs.tuned_macro_f1.mean():.4f} ± {runs.tuned_macro_f1.std(ddof=1):.4f}, accuracy {runs.tuned_accuracy.mean():.4f} ± {runs.tuned_accuracy.std(ddof=1):.4f} (tuned threshold)"),
               ('5-fold CV on train + val', f"macro-F1 {cv.tuned_macro_f1.mean():.4f} ± {cv.tuned_macro_f1.std(ddof=1):.4f}, accuracy {cv.tuned_accuracy.mean():.4f} ± {cv.tuned_accuracy.std(ddof=1):.4f}; baseline exp04 0.8253 ± 0.0190 / 0.9066 ± 0.0106 on the same folds")]
    reproducibility = [
        'Fixed seeds for Python, NumPy and PyTorch, and a seeded DataLoader shuffle.',
        '`torch.use_deterministic_algorithms(True)`, cuDNN deterministic, cuDNN benchmark off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; no operation was refused on the GPU.',
        'Identical results when the same run is repeated: in the same process, in separate processes, and in separate jobs.',
        'Every result averages 5 seeds (mean ± sample std); 5-fold CV on the baseline\'s exact folds; paired bootstrap (2,000 resamples) for every comparison on the same reviews.',
        'Same-device rule: CNN runs are compared only with runs on the same GPU model; earlier laptop CPU results are kept as an archive.',
        'Cross-hardware finding: retraining the baseline on the server CPU changed its scores slightly and its selected configuration, so the official baseline is the saved laptop pipeline, which reproduces its metrics exactly anywhere.',
    ]
    lines = ['# Final model card: Text CNN', '',
             f"Kim (2014) Text CNN, final config chosen by the rules in `docs/report_tracker.md`. Total {counts['total_params']:,} parameters, all trainable; the embedding holds {counts['embedding_share_pct']:.1f}%. "
             f"Shapes use batch size {config['batch_size']} and are asserted by running one batch through the model.", '',
             '## Layers', '', *md_table(layers, list(layers[0])), '',
             '## Training and decision settings', '', *md_table([{'Setting': a, 'Value': b, 'How it was chosen': c} for a, b, c in training], ['Setting', 'Value', 'How it was chosen']), '',
             '## Results (holdout not yet used)', '', *md_table([{'Estimate': a, 'Result': b} for a, b in results], ['Estimate', 'Result']), '',
             '## Reproducibility measures', '', *[f'- {item}' for item in reproducibility], '']
    (out_dir / 'final_model_card.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def main(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path('processed/split_metadata.json').read_text(encoding='utf-8'))
    train = pd.read_csv('processed/train.csv', keep_default_na=False)
    if len(train) != metadata['split_stats']['train']['rows']:
        raise ValueError('train.csv does not match split_metadata.json')
    vocab = json.loads(Path(DEFAULT_CONFIG['vocab_path']).read_text(encoding='utf-8'))
    recipe = json.loads(RECIPE.read_text(encoding='utf-8'))
    pipeline_example(out_dir, train, vocab)
    class_balance(out_dir, metadata)
    length_histogram(out_dir, train)
    final_learning_curve(out_dir, recipe)
    model_card(out_dir, vocab, recipe)
    print('Saved:', sorted(path.name for path in out_dir.iterdir()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, default=Path('results/figures'))
    main(parser.parse_args().out_dir)
