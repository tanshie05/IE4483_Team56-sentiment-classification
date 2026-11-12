"""IE4483 Project 1: Text CNN (Kim 2014) reference configuration with randomly initialised embeddings.
Run: python train_cnn.py --config configs/cnn_reference.json
Reads only the train and validation partitions. Holdout and official test data are never loaded.
"""
import argparse
import copy
import io
import json
import os
import platform
import random
import statistics
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # Required for deterministic CUBLAS on a GPU.

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eval_utils import threshold_metrics, tune_threshold
from prepare_data import tokenize

DEFAULT_CONFIG = {
    'processed_dir': 'processed', 'vocab_path': 'processed/cnn_vocab.json', 'baseline_dir': 'results/baseline',
    'out_dir': 'reproduction/cnn_reference', 'run_name': 'cnn_reference',
    'embedding_dim': 100, 'max_len': 300, 'kernel_sizes': [3, 4, 5], 'num_filters': 100, 'dropout': .5,
    'lr': 1e-3, 'batch_size': 64, 'max_epochs': 20, 'patience': 3, 'seeds': [42, 43, 44], 'rerun_seed': 42,
    'device': 'auto', 'num_workers': 0, 'timing_repeats': 5, 'baseline_fit_repeats': 3, 'bootstrap_resamples': 2000,
    'embedding_init': 'random', 'freeze_embedding': False, 'glove_path': 'data/glove/glove.6B.100d.txt',
    'variants': None, 'reference_config': 'configs/cnn_reference.json', 'reference_dir': 'results/archive_cpu',
    # pos_weight 'balanced' means N_neg / N_pos from the training partition; a number sets it directly (1 = unweighted).
    # checkpoint_metric 'val_macro_f1' tunes the threshold on validation every epoch and keeps the best tuned macro-F1.
    'pos_weight': 'balanced', 'checkpoint_metric': 'val_loss',
    # Variant runs only. reference_check False skips the exact match against the saved laptop CPU results (other devices
    # never match them) and allows a different seed list. reuse maps a variant to {"dir", "prefix", "variant"} of
    # saved runs instead of retraining them. bootstrap_pairs lists [variant, comparator] pairs, where the comparator
    # may be "baseline exp04". grid lays variants out as {row: {column: variant}} for a two-factor table.
    'reference_check': True, 'variants_prefix': 'glove', 'report_title': None, 'reuse': None, 'bootstrap_pairs': None,
    'grid': None, 'save_all_weights': False,
    # One-factor sweep: {"reference": variant, "min_seeds": n, "factors": {name: {"setting": key, "variants": [...]}}}.
    # A value replaces the reference only if its mean tuned macro-F1 is higher and its paired bootstrap against the
    # reference lies entirely above 0 in at least min_seeds seeds.
    'sweep': None,
}
GRID_FACTORS = {'rows': 'Loss weighting', 'columns': 'Checkpoint and early stopping'}
# The reference palette's first four categorical slots, in order, with a line style each as a second cue.
SERIES_STYLES = [('#2a78d6', '-'), ('#eb6834', '--'), ('#1baf7a', ':'), ('#eda100', '-.')]
INK, MUTED, GRID_LINE, AXIS_LINE = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
VARIANT_STYLES = {'random': ('#2a6fb0', '-'), 'glove_frozen': ('#c8742b', '--'), 'glove_finetuned': ('#7a4fa3', ':')}
METRIC_KEYS = ['accuracy', 'macro_f1', 'neg_precision', 'neg_recall', 'neg_f1', 'pos_precision', 'pos_recall', 'pos_f1',
               'tn', 'fp', 'fn', 'tp']
TIMING_COLUMNS = ['total_train_seconds', 'optimizer_seconds', 'preprocess_ms_per_1000', 'forward_ms_per_1000',
                  'inference_ms_per_1000']


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_filters, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(embedding_dim, num_filters, size) for size in kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, tokens):
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = [torch.relu(conv(embedded)).max(dim=2).values for conv in self.convs]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1))).squeeze(1)


def check_shapes(model, tokens, config):
    """Runs one batch layer by layer, asserting each shape, and returns the table for the report."""
    batch, max_len = tokens.shape
    dim, filters = config['embedding_dim'], config['num_filters']
    table = []

    def record(layer, tensor, expected):
        if tuple(tensor.shape) != expected:
            raise ValueError(f'{layer}: expected {expected}, got {tuple(tensor.shape)}')
        table.append({'Layer': layer, 'Output shape': str(tuple(tensor.shape))})

    with torch.no_grad():
        record('Input token IDs', tokens, (batch, max_len))
        embedded = model.embedding(tokens)
        record(f'Embedding({model.embedding.num_embeddings}, {dim}, padding_idx=0)', embedded, (batch, max_len, dim))
        transposed = embedded.transpose(1, 2)
        record('Transpose', transposed, (batch, dim, max_len))
        pooled = []
        for size, conv in zip(config['kernel_sizes'], model.convs):
            convolved = conv(transposed)
            record(f'Conv1d(kernel={size})', convolved, (batch, filters, max_len - size + 1))
            activated = torch.relu(convolved)
            record(f'ReLU (kernel={size})', activated, (batch, filters, max_len - size + 1))
            pooled.append(activated.max(dim=2).values)
            record(f'Global max-pool (kernel={size})', pooled[-1], (batch, filters))
        joined = torch.cat(pooled, dim=1)
        record('Concatenate', joined, (batch, filters * len(pooled)))
        dropped = model.dropout(joined)
        record(f"Dropout({config['dropout']})", dropped, (batch, filters * len(pooled)))
        logits = model.classifier(dropped)
        record('Linear (raw logit)', logits, (batch, 1))
        record('Squeeze', logits.squeeze(1), (batch,))
        if model(tokens).shape != (batch,):
            raise ValueError('forward() does not match the layer-by-layer shapes')
    return table


def parameter_counts(model, config):
    counts = {'embedding_params': sum(p.numel() for p in model.embedding.parameters())}
    for size, conv in zip(config['kernel_sizes'], model.convs):
        counts[f'conv{size}_params'] = sum(p.numel() for p in conv.parameters())
    counts['conv_total_params'] = sum(p.numel() for p in model.convs.parameters())
    counts['classifier_params'] = sum(p.numel() for p in model.classifier.parameters())
    counts['total_params'] = sum(p.numel() for p in model.parameters())
    counts['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    counts['embedding_share_pct'] = 100 * counts['embedding_params'] / counts['total_params']
    return counts


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def encode(texts, vocab, max_len):
    # Unknown tokens map to <UNK>=1; padding (<PAD>=0) and truncation both happen at the end.
    rows = []
    for text in texts:
        ids = [vocab.get(token, 1) for token in tokenize(text)][:max_len]
        rows.append(ids + [0] * (max_len - len(ids)))
    return torch.tensor(rows, dtype=torch.long)


def score_split(model, tokens, labels, criterion, batch_size, device):
    """Returns the mean weighted loss and P(positive). Sigmoid is applied only here, at inference."""
    model.eval()
    total, scores = 0.0, []
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            batch = tokens[start:start + batch_size].to(device)
            target = labels[start:start + batch_size].to(device)
            logits = model(batch)
            total += criterion(logits, target).item() * len(batch)
            scores.append(torch.sigmoid(logits).cpu())
    return total / len(tokens), torch.cat(scores).numpy()


def contraction_parts(token):
    """GloVe splits contractions: can't -> ca + n't, it's -> it + 's. Returns None for other tokens."""
    if "'" not in token.strip("'"):
        return None
    if token.endswith("n't"):
        return token[:-3], "n't"
    left, right = token.rsplit("'", 1)
    return left, "'" + right


def load_glove(path, vocab, dim):
    """Keeps vectors needed by the vocabulary and the global std across all vectors."""
    wanted = set(vocab)
    wanted.update(part for token in vocab for part in (contraction_parts(token) or ()))
    vectors, total, squares, count = {}, 0.0, 0.0, 0
    with open(path, encoding='utf-8') as source:
        for line in source:
            word, *values = line.rstrip('\n').split(' ')
            if len(values) != dim:
                raise ValueError(f'Unexpected GloVe line for {word!r}: {len(values)} values')
            array = np.asarray(values, dtype=np.float64)
            total += array.sum()
            squares += (array ** 2).sum()
            count += dim
            if word in wanted:
                vectors[word] = array
    mean = total / count
    return vectors, float(np.sqrt(squares / count - mean ** 2))


def glove_sources(vocab, vectors):
    """Classifies each vocabulary token as direct, rescued through contraction parts, or random."""
    direct, rescued, random_tokens = [], {}, []
    for token in vocab:
        if token in ('<PAD>', '<UNK>'):
            continue
        parts = contraction_parts(token)
        if token in vectors:
            direct.append(token)
        elif parts and all(part in vectors for part in parts):
            rescued[token] = parts
        else:
            random_tokens.append(token)
    return direct, rescued, random_tokens


def glove_matrix(seed, vocab, vectors, sources, std, dim):
    # A separate generator leaves the global random stream, and so the conv initialisation, unchanged.
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.normal(0.0, std, size=(len(vocab), dim), generator=generator, dtype=torch.float32)
    direct, rescued, _ = sources
    for token in direct:
        matrix[vocab[token]] = torch.tensor(vectors[token], dtype=torch.float32)
    for token, parts in rescued.items():
        matrix[vocab[token]] = torch.tensor(np.mean([vectors[part] for part in parts], axis=0), dtype=torch.float32)
    matrix[vocab['<PAD>']] = 0.0
    return matrix


def train_seed(seed, config, data, vocab_size, device, glove=None):
    seed_everything(seed)
    model = TextCNN(vocab_size, config['embedding_dim'], config['kernel_sizes'], config['num_filters'], config['dropout']).to(device)
    if config['embedding_init'] == 'glove':
        model.embedding.weight.data.copy_(glove_matrix(seed, *glove, config['embedding_dim']).to(device))
    model.embedding.weight.requires_grad_(not config['freeze_embedding'])
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=config['lr'])
    pos_weight = effective_pos_weight(config, data)
    # Validation loss uses the same weighting as training.
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    loader = DataLoader(TensorDataset(data['train_x'], data['train_y']), batch_size=config['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker,
                        num_workers=config['num_workers'])
    if config['checkpoint_metric'] not in ('val_loss', 'val_macro_f1'):
        raise ValueError(f"checkpoint_metric must be val_loss or val_macro_f1, not {config['checkpoint_metric']!r}")
    # Higher is better: minus the validation loss, or the epoch's tuned validation macro-F1.
    history, best_score, best_state, best_epoch, waited, optimizer_seconds = [], float('-inf'), None, 0, 0, 0.0
    start_all = time.perf_counter()
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        running, start = 0.0, time.perf_counter()
        for batch, target in loader:
            batch, target = batch.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), target)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(batch)
        optimizer_seconds += time.perf_counter() - start
        train_eval_loss, train_scores = score_split(model, data['train_x'], data['train_y'], criterion, config['batch_size'], device)
        val_loss, val_scores = score_split(model, data['val_x'], data['val_y'], criterion, config['batch_size'], device)
        epoch_threshold, _ = tune_threshold(data['val_labels'], val_scores)
        row = {
            'seed': seed, 'epoch': epoch, 'train_loss': running / len(data['train_x']), 'train_eval_loss': train_eval_loss,
            'val_loss': val_loss, 'train_macro_f1': threshold_metrics(data['train_labels'], train_scores, .5)['macro_f1'],
            'val_macro_f1': threshold_metrics(data['val_labels'], val_scores, .5)['macro_f1'],
            'val_tuned_threshold': epoch_threshold,
            'val_tuned_macro_f1': threshold_metrics(data['val_labels'], val_scores, epoch_threshold)['macro_f1'],
        }
        history.append(row)
        print(f"seed {seed} epoch {epoch}: train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} val_macro_f1={row['val_macro_f1']:.4f} "
              f"tuned={row['val_tuned_macro_f1']:.4f}@{epoch_threshold:.2f}", flush=True)
        score = -val_loss if config['checkpoint_metric'] == 'val_loss' else row['val_tuned_macro_f1']
        if score > best_score:
            best_score, best_epoch, waited = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= config['patience']:
                break
    total_seconds = time.perf_counter() - start_all
    model.load_state_dict(best_state)
    _, scores = score_split(model, data['val_x'], data['val_y'], criterion, config['batch_size'], device)
    threshold, _ = tune_threshold(data['val_labels'], scores)
    frame = pd.DataFrame(history)
    at_best = frame[frame.epoch == best_epoch].iloc[0]
    if threshold != at_best.val_tuned_threshold:
        raise ValueError(f'Seed {seed}: the reloaded checkpoint tunes to {threshold}, not its stored {at_best.val_tuned_threshold}')
    best_f1 = frame.sort_values(['val_macro_f1', 'epoch'], ascending=[False, True], kind='mergesort').iloc[0]
    best_tuned = frame.sort_values(['val_tuned_macro_f1', 'epoch'], ascending=[False, True], kind='mergesort').iloc[0]
    result = {
        'seed': seed, 'pos_weight': pos_weight, 'checkpoint_metric': config['checkpoint_metric'],
        'best_epoch': best_epoch, 'epochs_run': len(history), 'best_f1_epoch': int(best_f1.epoch),
        'best_f1_epoch_macro_f1': float(best_f1.val_macro_f1), 'best_tuned_f1_epoch': int(best_tuned.epoch),
        'best_tuned_f1_epoch_macro_f1': float(best_tuned.val_tuned_macro_f1), 'best_train_loss': float(at_best.train_loss),
        'best_train_eval_loss': float(at_best.train_eval_loss), 'best_val_loss': float(at_best.val_loss),
        'best_train_macro_f1': float(at_best.train_macro_f1), 'best_val_macro_f1': float(at_best.val_macro_f1),
        'val_roc_auc': float(roc_auc_score(data['val_labels'], scores)), 'tuned_threshold': threshold,
        **{f't05_{key}': value for key, value in threshold_metrics(data['val_labels'], scores, .5).items()},
        **{f'tuned_{key}': value for key, value in threshold_metrics(data['val_labels'], scores, threshold).items()},
        'total_train_seconds': total_seconds, 'optimizer_seconds': optimizer_seconds,
    }
    return result, frame, model, scores


def effective_pos_weight(config, data):
    return data['pos_weight'] if config['pos_weight'] == 'balanced' else float(config['pos_weight'])


def median_ms_per_1000(function, count, repeats):
    function()  # Warm-up, not timed.
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return 1e6 * statistics.median(times) / count


def benchmark_cnn(model, val_texts, vocab, config):
    model = copy.deepcopy(model).to('cpu').eval()
    tokens = encode(val_texts, vocab, config['max_len'])

    def forward():
        with torch.no_grad():
            for start in range(0, len(tokens), config['batch_size']):
                torch.sigmoid(model(tokens[start:start + config['batch_size']]))

    preprocess_ms = median_ms_per_1000(lambda: encode(val_texts, vocab, config['max_len']), len(val_texts), config['timing_repeats'])
    forward_ms = median_ms_per_1000(forward, len(val_texts), config['timing_repeats'])
    return {'preprocess_ms_per_1000': preprocess_ms, 'forward_ms_per_1000': forward_ms, 'inference_ms_per_1000': preprocess_ms + forward_ms}


def benchmark_baseline(pipeline, train, val_texts, config, baseline_path):
    tfidf, clf = pipeline.named_steps['tfidf'], pipeline.named_steps['clf']
    features = tfidf.transform(val_texts)
    transform_ms = median_ms_per_1000(lambda: tfidf.transform(val_texts), len(val_texts), config['timing_repeats'])
    predict_ms = median_ms_per_1000(lambda: clf.predict_proba(features), len(val_texts), config['timing_repeats'])
    fit_times = []
    for _ in range(config['baseline_fit_repeats']):
        fresh = clone(pipeline)
        start = time.perf_counter()
        fresh.fit(train.clean_text, train.sentiments)
        fit_times.append(time.perf_counter() - start)
    return {'transform_ms': transform_ms, 'predict_ms': predict_ms, 'total_ms': transform_ms + predict_ms,
            'fit_seconds': statistics.median(fit_times), 'size_kb': baseline_path.stat().st_size / 1024}


def bootstrap_difference(labels, cnn_predictions, baseline_predictions, resamples):
    """Paired bootstrap over validation reviews: CNN minus baseline, 95% interval."""
    rng = np.random.default_rng(42)
    differences = {'macro_f1': [], 'accuracy': []}
    for _ in range(resamples):
        index = rng.integers(0, len(labels), len(labels))
        cnn = threshold_metrics(labels[index], cnn_predictions[index].astype(float), .5)
        base = threshold_metrics(labels[index], baseline_predictions[index].astype(float), .5)
        for key in differences:
            differences[key].append(cnn[key] - base[key])
    return {key: (float(np.mean(values)), *map(float, np.percentile(values, [2.5, 97.5]))) for key, values in differences.items()}


def plot_curves(path, history, runs):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=160)
    series = [('train_eval_loss', 'Train loss (eval mode)', '#c8742b'), ('val_loss', 'Validation loss', '#2a6fb0')]
    for column, label, color in series:
        for _, group in history.groupby('seed'):
            left.plot(group.epoch, group[column], color=color, alpha=.25, linewidth=1)
        mean = history.groupby('epoch')[column].mean()
        left.plot(mean.index, mean.values, color=color, linewidth=2, label=label)
    f1 = '#2a6fb0'
    for _, group in history.groupby('seed'):
        right.plot(group.epoch, group.val_macro_f1, color=f1, alpha=.25, linewidth=1)
    mean = history.groupby('epoch').val_macro_f1.mean()
    right.plot(mean.index, mean.values, color=f1, linewidth=2, label='Validation macro-F1 at 0.5')
    for number, run in enumerate(runs.itertuples()):
        loss_label = 'Best val-loss epoch' if number == 0 else None
        f1_label = 'Best val macro-F1 epoch' if number == 0 else None
        left.scatter(run.best_epoch, run.best_val_loss, s=42, color=f1, edgecolor='white', linewidth=1.2, zorder=3, label=loss_label)
        right.scatter(run.best_epoch, run.best_val_macro_f1, s=42, color=f1, edgecolor='white', linewidth=1.2, zorder=3, label=loss_label)
        right.scatter(run.best_f1_epoch, run.best_f1_epoch_macro_f1, s=48, facecolor='white', edgecolor='#333333', linewidth=1.4, zorder=4, label=f1_label)
    epochs = sorted(history.epoch.unique())
    for ax, title, ylabel in ((left, 'Loss by epoch', 'Weighted BCE loss'), (right, 'Validation macro-F1 by epoch', 'Macro-F1')):
        ax.set_title(title, loc='left', fontsize=11, color='#222222')
        ax.set_xlabel('Epoch', fontsize=9, color='#555555')
        ax.set_ylabel(ylabel, fontsize=9, color='#555555')
        ax.set_xticks(epochs)
        ax.grid(axis='y', color='#e6e6e6', linewidth=.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors='#555555', labelsize=8, length=0)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#bbbbbb')
        ax.legend(frameon=False, fontsize=8)
    fig.text(.01, .01, 'Bold lines: mean over seeds still running at that epoch. Faint lines: individual seeds.', fontsize=8, color='#555555')
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def md_table(rows, columns):
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns)]
    lines += ['| ' + ' | '.join(str(row[column]) for column in columns) + ' |' for row in rows]
    return lines


def write_report(path, config, runs, history, shapes, counts, timing, baseline_timing, baseline, bootstrap, rerun, environment, best):
    def spread(column):
        return f'{runs[column].mean():.4f} ± {runs[column].std(ddof=1):.4f}'

    per_seed = [{'Seed': row.seed, 'Best epoch': row.best_epoch, 'Epochs run': row.epochs_run,
                 'Accuracy at 0.5': f'{row.t05_accuracy:.4f}', 'Macro-F1 at 0.5': f'{row.t05_macro_f1:.4f}',
                 'Threshold': f'{row.tuned_threshold:.2f}', 'Accuracy tuned': f'{row.tuned_accuracy:.4f}',
                 'Macro-F1 tuned': f'{row.tuned_macro_f1:.4f}', 'Neg precision tuned': f'{row.tuned_neg_precision:.4f}',
                 'Neg recall tuned': f'{row.tuned_neg_recall:.4f}', 'ROC-AUC': f'{row.val_roc_auc:.4f}'}
                for row in runs.itertuples()]
    per_seed.append({'Seed': 'Mean ± std', 'Best epoch': spread('best_epoch').replace('.0000', ''), 'Epochs run': spread('epochs_run'),
                     'Accuracy at 0.5': spread('t05_accuracy'), 'Macro-F1 at 0.5': spread('t05_macro_f1'), 'Threshold': spread('tuned_threshold'),
                     'Accuracy tuned': spread('tuned_accuracy'), 'Macro-F1 tuned': spread('tuned_macro_f1'),
                     'Neg precision tuned': spread('tuned_neg_precision'), 'Neg recall tuned': spread('tuned_neg_recall'), 'ROC-AUC': spread('val_roc_auc')})
    classes = [{'Metric': label, 'At 0.5': spread(f't05_{key}'), 'Tuned': spread(f'tuned_{key}')}
               for label, key in (('Negative precision', 'neg_precision'), ('Negative recall', 'neg_recall'), ('Negative F1', 'neg_f1'),
                                  ('Positive precision', 'pos_precision'), ('Positive recall', 'pos_recall'), ('Positive F1', 'pos_f1'))]
    confusion = [{'Seed': row.seed, 'Threshold': label, 'True neg, pred neg': tn, 'True neg, pred pos': fp, 'True pos, pred neg': fn, 'True pos, pred pos': tp}
                 for row in runs.itertuples()
                 for label, tn, fp, fn, tp in (('0.50', row.t05_tn, row.t05_fp, row.t05_fn, row.t05_tp),
                                               (f'{row.tuned_threshold:.2f}', row.tuned_tn, row.tuned_fp, row.tuned_fn, row.tuned_tp))]
    agreement = [{'Seed': row.seed, 'Best val-loss epoch': row.best_epoch, 'Macro-F1 there': f'{row.best_val_macro_f1:.4f}',
                  'Best macro-F1 epoch': row.best_f1_epoch, 'Macro-F1 there ': f'{row.best_f1_epoch_macro_f1:.4f}',
                  'Agree': 'Yes' if row.best_epoch == row.best_f1_epoch else 'No'} for row in runs.itertuples()]
    fit = [{'Seed': row.seed, 'Train loss, best epoch': f'{row.best_train_eval_loss:.4f}', 'Val loss, best epoch': f'{row.best_val_loss:.4f}',
            'Train macro-F1': f'{row.best_train_macro_f1:.4f}', 'Val macro-F1': f'{row.best_val_macro_f1:.4f}',
            'Final val loss': f"{history[history.seed == row.seed].val_loss.iloc[-1]:.4f}"} for row in runs.itertuples()]
    agree_count = int((runs.best_epoch == runs.best_f1_epoch).sum())
    gap = (runs.best_train_macro_f1 - runs.best_val_macro_f1).mean()
    cv = baseline['cv']
    val = baseline['validation_at_threshold']
    compare = [
        {'Model': 'Baseline exp04, validation, tuned', 'Accuracy': f"{val['accuracy']:.4f}", 'Macro-F1': f"{val['macro_f1']:.4f}", 'Neg precision': f"{val['neg_precision']:.4f}",
         'Neg recall': f"{val['neg_recall']:.4f}", 'ROC-AUC': f"{baseline['validation_roc_auc']:.4f}"},
        {'Model': f'CNN, validation, tuned (mean ± std over {len(runs)} seeds)', 'Accuracy': spread('tuned_accuracy'), 'Macro-F1': spread('tuned_macro_f1'),
         'Neg precision': spread('tuned_neg_precision'), 'Neg recall': spread('tuned_neg_recall'), 'ROC-AUC': spread('val_roc_auc')},
        {'Model': 'Baseline exp04, validation, 0.5', 'Accuracy': f"{baseline['validation_at_0.5']['accuracy']:.4f}", 'Macro-F1': f"{baseline['validation_at_0.5']['macro_f1']:.4f}",
         'Neg precision': f"{baseline['validation_at_0.5']['neg_precision']:.4f}", 'Neg recall': f"{baseline['validation_at_0.5']['neg_recall']:.4f}", 'ROC-AUC': f"{baseline['validation_roc_auc']:.4f}"},
        {'Model': f'CNN, validation, 0.5 (mean ± std over {len(runs)} seeds)', 'Accuracy': spread('t05_accuracy'), 'Macro-F1': spread('t05_macro_f1'),
         'Neg precision': spread('t05_neg_precision'), 'Neg recall': spread('t05_neg_recall'), 'ROC-AUC': spread('val_roc_auc')},
        {'Model': 'Baseline exp04, nested CV (split variability)', 'Accuracy': f"{cv['tuned_accuracy']['mean']:.4f} ± {cv['tuned_accuracy']['std']:.4f}",
         'Macro-F1': f"{cv['tuned_macro_f1']['mean']:.4f} ± {cv['tuned_macro_f1']['std']:.4f}", 'Neg precision': f"{cv['tuned_neg_precision']['mean']:.4f} ± {cv['tuned_neg_precision']['std']:.4f}",
         'Neg recall': f"{cv['tuned_neg_recall']['mean']:.4f} ± {cv['tuned_neg_recall']['std']:.4f}", 'ROC-AUC': f"{cv['roc_auc']['mean']:.4f} ± {cv['roc_auc']['std']:.4f}"},
    ]
    boot_rows = [{'Seed': seed, 'Macro-F1 difference': f"{values['macro_f1'][0]:+.4f}", 'Macro-F1 95% interval': f"{values['macro_f1'][1]:+.4f} to {values['macro_f1'][2]:+.4f}",
                  'Accuracy difference': f"{values['accuracy'][0]:+.4f}", 'Accuracy 95% interval': f"{values['accuracy'][1]:+.4f} to {values['accuracy'][2]:+.4f}"}
                 for seed, values in bootstrap.items()]
    timing_rows = [
        {'Model (CPU)': 'CNN', 'Preprocess or TF-IDF transform': f"{timing['preprocess_ms_per_1000']:.1f}", 'Forward or predict_proba': f"{timing['forward_ms_per_1000']:.1f}",
         'Total ms per 1,000': f"{timing['inference_ms_per_1000']:.1f}", 'Train s (median)': f"{runs.total_train_seconds.median():.1f}", 'Size KB': f'{runs.model_size_kb.iloc[0]:.1f}'},
        {'Model (CPU)': 'Baseline exp04', 'Preprocess or TF-IDF transform': f"{baseline_timing['transform_ms']:.1f}", 'Forward or predict_proba': f"{baseline_timing['predict_ms']:.1f}",
         'Total ms per 1,000': f"{baseline_timing['total_ms']:.1f}", 'Train s (median)': f"{baseline_timing['fit_seconds']:.3f}", 'Size KB': f"{baseline_timing['size_kb']:.1f}"},
    ]
    count_rows = [{'Part': label, 'Parameters': f'{counts[key]:,}'} for label, key in
                  [('Embedding', 'embedding_params'), *[(f'Conv1d kernel {size}', f'conv{size}_params') for size in config['kernel_sizes']],
                   ('Conv1d total', 'conv_total_params'), ('Classifier', 'classifier_params'), ('Total', 'total_params'), ('Trainable', 'trainable_params')]]
    lines = [
        '# Text CNN reference configuration', '',
        'Validation set: 1,078 reviews. Holdout and official test data were not loaded. Vocabulary: `processed/cnn_vocab.json`, unchanged. '
        f"Sequences use max_len={config['max_len']}, padded and truncated at the end; that length was chosen from coverage only.", '',
        '## Accuracy and macro-F1 against the baseline (validation)', '',
        *md_table(compare, ['Model', 'Accuracy', 'Macro-F1', 'Neg precision', 'Neg recall', 'ROC-AUC']), '',
        'Tuned thresholds maximise macro-F1, not accuracy, and are optimistic because the same validation set selects them and scores them.', '',
        '### Paired bootstrap on validation reviews (CNN minus baseline, both tuned)', '',
        *md_table(boot_rows, ['Seed', 'Macro-F1 difference', 'Macro-F1 95% interval', 'Accuracy difference', 'Accuracy 95% interval']), '',
        f"{config['bootstrap_resamples']} paired resamples of the same reviews. An interval containing 0 means this validation set cannot separate the models. "
        'The ± across seeds measures initialisation randomness on one fixed split. The baseline CV ± measures split variability. They are different sources of uncertainty, '
        'and the CNN has no CV estimate yet, so neither ± supports a claim of significant improvement on its own.', '',
        '## Results by seed', '',
        *md_table(per_seed, list(per_seed[0])), '',
        f'### Per-class metrics (mean ± std over {len(runs)} seeds)', '',
        *md_table(classes, ['Metric', 'At 0.5', 'Tuned']), '',
        '### Confusion matrices', '',
        *md_table(confusion, list(confusion[0])), '',
        '## Learning curves', '',
        '![Learning curves](learning_curves.png)', '',
        '### Do validation loss and macro-F1 agree on the best epoch?', '',
        *md_table(agreement, ['Seed', 'Best val-loss epoch', 'Macro-F1 there', 'Best macro-F1 epoch', 'Macro-F1 there ', 'Agree']), '',
        f'They agree for {agree_count} of {len(runs)} seeds. The checkpoint is chosen by validation loss as specified; macro-F1 at 0.5 is recorded only.', '',
        '### Overfitting or underfitting', '',
        *md_table(fit, list(fit[0])), '',
        f'At the best epoch, mean train macro-F1 exceeds validation by {gap:.4f}. Train loss here is measured in eval mode, without dropout. '
        'Validation loss rising after the best epoch while train loss keeps falling is the overfitting signal.', '',
        '## Fair inference and training comparison (CPU)', '',
        *md_table(timing_rows, list(timing_rows[0])), '',
        f"Median of {config['timing_repeats']} runs over the full validation set, excluding model loading, after one warm-up pass. CNN batch size {config['batch_size']}; the baseline scores the set in one call. "
        f"CNN timing is the median across seeds. Train time for the CNN includes per-epoch train and validation scoring; the optimiser steps alone take a median {runs.optimizer_seconds.median():.1f} s. "
        f"Threads: {environment['torch_threads']}. Timings vary with machine load.", '',
        '## Model', '',
        '### Shape check (B = batch size 64)', '',
        *md_table(shapes, ['Layer', 'Output shape']), '',
        '### Parameters', '',
        *md_table(count_rows, ['Part', 'Parameters']), '',
        f"The embedding holds {counts['embedding_share_pct']:.1f}% of all parameters.", '',
        '## Reproducibility', '',
        f"- Environment: {environment['device']} ({environment['hardware']}), Python {environment['python']}, PyTorch {environment['torch']}, NumPy {environment['numpy']}, scikit-learn {environment['sklearn']}.",
        f"- `torch.use_deterministic_algorithms(True)` was on, with cuDNN deterministic set. No operation failed; no GPU was used{'' if environment['device'] == 'cpu' else ' except as recorded'}.",
        f"- Rerun of seed {config['rerun_seed']} in the same process: metrics and history identical: {rerun['metrics']}; all parameter tensors exactly equal: {rerun['tensors']}.",
        f"- Saved weights reload and reproduce validation scores: {rerun['reload']}.",
        f"- Saved weights are from seed {best}, the highest tuned validation macro-F1. That single-seed score is optimistic; use the mean above for comparisons.", '',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def setup_device(config):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if config['device'] == 'auto' else config['device'])


def load_data(config):
    processed_dir = Path(config['processed_dir'])
    metadata = json.loads((processed_dir / 'split_metadata.json').read_text(encoding='utf-8'))
    frames = {}
    for name in ('train', 'val'):
        frames[name] = pd.read_csv(processed_dir / f'{name}.csv', keep_default_na=False)
        if len(frames[name]) != metadata['split_stats'][name]['rows']:
            raise ValueError(f'{name}.csv does not match split_metadata.json')
    train, val = frames['train'], frames['val']
    vocab = json.loads(Path(config['vocab_path']).read_text(encoding='utf-8'))
    if len(vocab) != metadata['cnn_vocab_size'] or vocab['<PAD>'] != 0 or vocab['<UNK>'] != 1:
        raise ValueError('cnn_vocab.json does not match the expected vocabulary')

    n_neg, n_pos = int((train.sentiments == 0).sum()), int((train.sentiments == 1).sum())
    data = {
        'train_x': encode(train.clean_text, vocab, config['max_len']), 'val_x': encode(val.clean_text, vocab, config['max_len']),
        'train_y': torch.tensor(train.sentiments.to_numpy(), dtype=torch.float32), 'val_y': torch.tensor(val.sentiments.to_numpy(), dtype=torch.float32),
        'train_labels': train.sentiments.to_numpy(), 'val_labels': val.sentiments.to_numpy(), 'pos_weight': n_neg / n_pos,
    }
    return train, val, vocab, data


def write_coverage(path, vocab, sources, std, train, val):
    direct, rescued, random_tokens = sources
    types = len(direct) + len(rescued) + len(random_tokens)
    pretrained = set(direct) | set(rescued)
    counts = {name: Counter(token for text in frame.clean_text for token in tokenize(text)) for name, frame in (('train', train), ('val', val))}
    token_rows = []
    for name, counter in counts.items():
        total = sum(counter.values())
        covered = sum(count for token, count in counter.items() if token in pretrained)
        token_rows.append({'Split': name, 'Tokens': f'{total:,}', 'With a pretrained vector': f'{100 * covered / total:.2f}%', 'Random or <UNK>': f'{100 * (total - covered) / total:.2f}%'})
    uncovered = [(token, count) for token, count in counts['train'].most_common() if token not in pretrained][:20]
    negations = sorted(token for token in vocab if token.endswith("n't"))
    type_rows = [{'Source': 'Found directly in GloVe', 'Types': len(direct), 'Share': f'{100 * len(direct) / types:.2f}%'},
                 {'Source': 'Rescued through contraction parts', 'Types': len(rescued), 'Share': f'{100 * len(rescued) / types:.2f}%'},
                 {'Source': 'Random initialisation', 'Types': len(random_tokens), 'Share': f'{100 * len(random_tokens) / types:.2f}%'}]
    negation_rows = [{'Token': f'`{token}`', 'Source': f"mean of `{rescued[token][0]}` + `{rescued[token][1]}`" if token in rescued else ('direct' if token in direct else 'random')} for token in negations]
    plain = [{'Token': f'`{token}`', 'Source': 'direct' if token in direct else 'random'} for token in ('not', 'no', 'never', 'cannot', '!', '?')]
    lines = [
        '# GloVe coverage', '',
        f'GloVe 6B 100d against `cnn_vocab.json`. Shares exclude the two special tokens: `<PAD>` is a zero vector and `<UNK>` is random. Random vectors use mean 0 and std {std:.4f}, the std of all GloVe values.', '',
        '## Vocabulary types', '', *md_table(type_rows, ['Source', 'Types', 'Share']), '',
        '## Token coverage', '',
        'A token counts as covered only if it is in the vocabulary and has a pretrained vector. Validation words outside the vocabulary map to `<UNK>`, even if GloVe knows them.', '',
        *md_table(token_rows, ['Split', 'Tokens', 'With a pretrained vector', 'Random or <UNK>']), '',
        '## Negation words', '', *md_table(negation_rows + plain, ['Token', 'Source']), '',
        f'All rescued tokens ({len(rescued)}): ' + ', '.join(f'`{token}`' for token in sorted(rescued)) + '.', '',
        '## 20 most frequent uncovered train tokens', '',
        *md_table([{'Token': f'`{token}`', 'Train count': count, 'In vocabulary': 'Yes' if token in vocab else 'No (maps to <UNK>)'} for token, count in uncovered], ['Token', 'Train count', 'In vocabulary']), '',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def plot_variants(path, history):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=160)
    panels = (('train_eval_loss', 'Train loss (eval mode)', 'Weighted BCE loss'), ('val_loss', 'Validation loss', 'Weighted BCE loss'),
              ('val_macro_f1', 'Validation macro-F1 at 0.5', 'Macro-F1'))
    for ax, (column, title, ylabel) in zip(axes, panels):
        for variant, (color, style) in VARIANT_STYLES.items():
            mean = history[history.variant == variant].groupby('epoch')[column].mean()
            ax.plot(mean.index, mean.values, color=color, linestyle=style, linewidth=2, label=variant)
        ax.set_title(title, loc='left', fontsize=11, color='#222222')
        ax.set_xlabel('Epoch', fontsize=9, color='#555555')
        ax.set_ylabel(ylabel, fontsize=9, color='#555555')
        ax.grid(axis='y', color='#e6e6e6', linewidth=.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors='#555555', labelsize=8, length=0)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#bbbbbb')
        ax.legend(frameon=False, fontsize=8)
    fig.text(.01, .01, 'Lines show the mean over seeds still running at that epoch.', fontsize=8, color='#555555')
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def write_glove_report(path, config, runs, baseline, bootstrap, checks, environment, best_variant, best_seed):
    def spread(frame, column):
        return f'{frame[column].mean():.4f} ± {frame[column].std(ddof=1):.4f}'

    val, at_05, cv = baseline['validation_at_threshold'], baseline['validation_at_0.5'], baseline['cv']
    comparison = [{'Model': 'Baseline exp04', 'Accuracy, tuned': f"{val['accuracy']:.4f}", 'Macro-F1, tuned': f"{val['macro_f1']:.4f}", 'Accuracy, 0.5': f"{at_05['accuracy']:.4f}", 'Macro-F1, 0.5': f"{at_05['macro_f1']:.4f}",
                   'Neg precision, tuned': f"{val['neg_precision']:.4f}", 'Neg recall, tuned': f"{val['neg_recall']:.4f}", 'ROC-AUC': f"{baseline['validation_roc_auc']:.4f}"}]
    overfit, seeds, agreement, cost, classes, confusion = [], [], [], [], [], []
    for variant, frame in runs.groupby('variant', sort=False):
        comparison.append({'Model': f'CNN {variant}', 'Accuracy, tuned': spread(frame, 'tuned_accuracy'), 'Macro-F1, tuned': spread(frame, 'tuned_macro_f1'), 'Accuracy, 0.5': spread(frame, 't05_accuracy'), 'Macro-F1, 0.5': spread(frame, 't05_macro_f1'),
                           'Neg precision, tuned': spread(frame, 'tuned_neg_precision'), 'Neg recall, tuned': spread(frame, 'tuned_neg_recall'), 'ROC-AUC': spread(frame, 'val_roc_auc')})
        overfit.append({'Variant': variant, 'Best epoch': spread(frame, 'best_epoch'), 'Epochs run': spread(frame, 'epochs_run'), 'Train macro-F1': spread(frame, 'best_train_macro_f1'), 'Val macro-F1': spread(frame, 'best_val_macro_f1'),
                        'Gap': spread(frame, 'train_val_gap'), 'Train loss': spread(frame, 'best_train_eval_loss'), 'Val loss': spread(frame, 'best_val_loss')})
        cost.append({'Variant': variant, 'Trainable parameters': f'{int(frame.trainable_params.iloc[0]):,}', 'Total parameters': f'{int(frame.total_params.iloc[0]):,}', 'Train s (median)': f'{frame.total_train_seconds.median():.1f}'})
        for key, label in (('neg_precision', 'Neg precision'), ('neg_recall', 'Neg recall'), ('neg_f1', 'Neg F1'), ('pos_precision', 'Pos precision'), ('pos_recall', 'Pos recall'), ('pos_f1', 'Pos F1')):
            classes.append({'Variant': variant, 'Metric': label, 'At 0.5': spread(frame, f't05_{key}'), 'Tuned': spread(frame, f'tuned_{key}')})
        for row in frame.itertuples():
            seeds.append({'Variant': variant, 'Seed': row.seed, 'Best epoch': row.best_epoch, 'Accuracy, 0.5': f'{row.t05_accuracy:.4f}', 'Macro-F1, 0.5': f'{row.t05_macro_f1:.4f}', 'Threshold': f'{row.tuned_threshold:.2f}',
                          'Accuracy, tuned': f'{row.tuned_accuracy:.4f}', 'Macro-F1, tuned': f'{row.tuned_macro_f1:.4f}', 'ROC-AUC': f'{row.val_roc_auc:.4f}', 'Gap': f'{row.train_val_gap:.4f}', 'Train s': f'{row.total_train_seconds:.1f}'})
            agreement.append({'Variant': variant, 'Seed': row.seed, 'Best val-loss epoch': row.best_epoch, 'Macro-F1 at val-loss epoch': f'{row.best_val_macro_f1:.4f}', 'Best macro-F1 epoch': row.best_f1_epoch, 'Best macro-F1': f'{row.best_f1_epoch_macro_f1:.4f}'})
            for label, tn, fp, fn, tp in (('0.50', row.t05_tn, row.t05_fp, row.t05_fn, row.t05_tp), (f'{row.tuned_threshold:.2f}', row.tuned_tn, row.tuned_fp, row.tuned_fn, row.tuned_tp)):
                confusion.append({'Variant': variant, 'Seed': row.seed, 'Threshold': label, 'True neg, pred neg': tn, 'True neg, pred pos': fp, 'True pos, pred neg': fn, 'True pos, pred pos': tp})
    boot_rows = []
    for (variant, against, seed), values in bootstrap.items():
        boot_rows.append({'Variant': variant, 'Against': against, 'Seed': seed, 'Macro-F1 difference': f"{values['macro_f1'][0]:+.4f}", 'Macro-F1 95% interval': f"{values['macro_f1'][1]:+.4f} to {values['macro_f1'][2]:+.4f}",
                          'Accuracy difference': f"{values['accuracy'][0]:+.4f}", 'Accuracy 95% interval': f"{values['accuracy'][1]:+.4f} to {values['accuracy'][2]:+.4f}"})
    means = runs.groupby('variant', sort=False).tuned_macro_f1.mean()
    agree = int((runs.best_epoch == runs.best_f1_epoch).sum())
    lines = [
        '# Text CNN: pretrained GloVe embeddings', '',
        'One controlled experiment: only the embedding initialisation and whether it trains differ from `configs/cnn_reference.json`. Validation set: 1,078 reviews. Holdout and official test data were not loaded. Coverage details: `glove_coverage.md`.', '',
        '## Comparison (validation, mean ± sample std over 3 seeds for CNN)', '',
        *md_table(comparison, list(comparison[0])), '',
        f"Baseline nested CV for context: macro-F1 {cv['tuned_macro_f1']['mean']:.4f} ± {cv['tuned_macro_f1']['std']:.4f}, accuracy {cv['tuned_accuracy']['mean']:.4f} ± {cv['tuned_accuracy']['std']:.4f}. Tuned thresholds are optimistic because validation both selects and scores them. "
        'Seed ± is initialisation randomness on one split; CV ± is split variability. The CNN variants have no CV estimate.', '',
        '## Paired bootstrap on validation reviews (variant minus comparator, both tuned)', '',
        *md_table(boot_rows, list(boot_rows[0])), '',
        f"{config['bootstrap_resamples']} paired resamples, the same method as the reference report. Random-embedding comparisons pair the same seed. An interval containing 0 means this validation set cannot separate the two. These intervals say nothing beyond this validation set.", '',
        '## Overfitting (at the best val-loss epoch)', '',
        *md_table(overfit, list(overfit[0])), '', 'Gap is train macro-F1 minus validation macro-F1 at threshold 0.5. Train loss is measured in eval mode.', '',
        '## Learning curves', '', '![Learning curves](glove_learning_curves.png)', '',
        '## Validation loss vs macro-F1 best epoch', '',
        *md_table(agreement, list(agreement[0])), '', f'They agree for {agree} of {len(runs)} runs. The checkpoint is chosen by validation loss.', '',
        '## Parameters and training time', '', *md_table(cost, list(cost[0])), '',
        f"Train time is on {environment['device']}, includes per-epoch scoring, and varies with machine load.", '',
        '## Results by seed', '', *md_table(seeds, list(seeds[0])), '',
        '### Per-class metrics (mean ± std)', '', *md_table(classes, list(classes[0])), '',
        '### Confusion matrices', '', *md_table(confusion, list(confusion[0])), '',
        '## Reproducibility', '',
        f"- Random variant retrained because paired comparisons need each seed's predictions and only seed 43's weights were saved. It matches `results/archive_cpu/cnn_reference_runs.csv` on every metric: {checks['reference_metrics']}. Seed 43 tensors match the saved reference weights: {checks['reference_weights']}.",
        f"- Same-process rerun of seed {config['rerun_seed']}, metrics and tensors identical: " + ', '.join(f'{variant} {value}' for variant, value in checks['rerun'].items()) + '.',
        f"- Saved weights are {best_variant}, seed {best_seed}, the higher-scoring GloVe variant by mean tuned macro-F1 ({means[best_variant]:.4f}) and its best seed. That single-seed score is optimistic. They reload and reproduce their scores: {checks['reload']}.",
        f"- Environment: {environment['device']} ({environment['hardware']}), Python {environment['python']}, PyTorch {environment['torch']}. Deterministic algorithms on: {environment['deterministic_algorithms']}.", '',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def run_environment(device):
    """Where the runs happen: device, node, GPU card and Slurm job, saved with every run."""
    environment = {'device': str(device), 'hardware': torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor(),
                   'node': platform.node(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'gpu_bus_id': None, 'gpu_uuid': None,
                   'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                   'torch_threads': torch.get_num_threads(), 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(0)
        if hasattr(properties, 'pci_bus_id'):
            environment['gpu_bus_id'] = f'{properties.pci_domain_id:08X}:{properties.pci_bus_id:02X}:{properties.pci_device_id:02X}.0'
        if hasattr(properties, 'uuid'):
            environment['gpu_uuid'] = f'GPU-{properties.uuid}'
    return environment


def load_reused(spec, variant, seeds):
    """Saved runs of another experiment, renamed to this variant, with their validation scores."""
    source = Path(spec['dir'])
    runs = pd.read_csv(source / f"{spec['prefix']}_runs.csv")
    runs = runs[(runs.variant == spec['variant']) & runs.seed.isin(seeds)].reset_index(drop=True)
    history = pd.read_csv(source / f"{spec['prefix']}_history.csv")
    history = history[(history.variant == spec['variant']) & history.seed.isin(seeds)].reset_index(drop=True)
    if sorted(runs.seed) != sorted(seeds):
        raise ValueError(f"Reused runs {spec} have seeds {sorted(runs.seed)}, not all of {sorted(seeds)}")
    saved_scores = np.load(source / f"{spec['prefix']}_val_scores.npz")
    scores = {seed: saved_scores[f"{spec['variant']}_seed{seed}"] for seed in seeds}
    reused_from = f"{source}/{spec['variant']}"
    return runs.assign(variant=variant, reused_from=reused_from), history.assign(variant=variant), scores


def plot_gpu_variants(path, history):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), dpi=160)
    panels = (('train_eval_loss', 'Train loss (eval mode)', 'BCE loss, as trained'), ('val_loss', 'Validation loss', 'BCE loss, as trained'),
              ('val_macro_f1', 'Validation macro-F1 at threshold 0.5', 'Macro-F1'),
              ('val_tuned_macro_f1', 'Validation macro-F1 at each epoch\'s tuned threshold', 'Macro-F1'))
    variants = list(dict.fromkeys(history.variant))
    if len(variants) > len(SERIES_STYLES):
        raise ValueError(f'At most {len(SERIES_STYLES)} variants fit one figure')
    for ax, (column, title, ylabel) in zip(axes.flat, panels):
        for order, (variant, (color, style)) in enumerate(zip(variants, SERIES_STYLES)):
            mean = history[history.variant == variant].groupby('epoch')[column].mean()
            # Earlier variants draw on top, so a line that overlaps a later one stays visible.
            ax.plot(mean.index, mean.values, color=color, linestyle=style, linewidth=2, label=variant, zorder=10 - order)
        ax.set_title(title, loc='left', fontsize=11, color=INK)
        ax.set_xlabel('Epoch', fontsize=9, color=MUTED)
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
        ax.set_xticks(sorted(history.epoch.unique()))
        ax.grid(axis='y', color=GRID_LINE, linewidth=.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(AXIS_LINE)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.text(.01, .028, 'Lines show the mean over seeds still running at that epoch. Loss panels use each variant\'s own pos_weight, '
             'so weighted and unweighted losses are on different scales.', fontsize=8, color=MUTED)
    fig.text(.01, .008, 'Same embedding and pos_weight means identical training per seed; the checkpoint criterion only changes the kept epoch '
             'and the stop, so such lines overlap.', fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def grid_effects(runs, grid, column):
    """Per-seed main effects and interaction for a 2x2 grid: second row or column minus the first."""
    (row_a, cells_a), (row_b, cells_b) = grid.items()
    col_a, col_b = cells_a
    value = {(row, col): runs[runs.variant == variant].set_index('seed')[column] for row, cells in grid.items() for col, variant in cells.items()}
    rows_effect = ((value[row_b, col_a] - value[row_a, col_a]) + (value[row_b, col_b] - value[row_a, col_b])) / 2
    columns_effect = ((value[row_a, col_b] - value[row_a, col_a]) + (value[row_b, col_b] - value[row_b, col_a])) / 2
    interaction = (value[row_b, col_b] - value[row_b, col_a]) - (value[row_a, col_b] - value[row_a, col_a])
    return {f'{GRID_FACTORS["rows"]}: {row_b} minus {row_a}': rows_effect,
            f'{GRID_FACTORS["columns"]}: {col_b} minus {col_a}': columns_effect,
            'Interaction: column effect in the second row minus in the first': interaction}


def sweep_decision(config, runs, bootstrap):
    """Applies the sweep rule factor by factor and returns the resulting settings with the evidence for each value."""
    sweep = config['sweep']
    reference = sweep['reference']
    reference_mean = runs[runs.variant == reference].tuned_macro_f1.mean()
    rows, settings, changed = [], {}, []
    for factor, spec in sweep['factors'].items():
        # A factor may move several settings together, e.g. one kernel size with three times the filters.
        keys = spec.get('settings', [spec.get('setting')])
        settings.update({key: config[key] for key in keys})
        passing = []
        for variant in spec['variants']:
            frame = runs[runs.variant == variant]
            values = {key: config['variants'][variant].get(key, config[key]) for key in keys}
            value = values[keys[0]] if len(keys) == 1 else values
            row = {'factor': factor, 'setting': ', '.join(keys), 'value': value, 'values': values, 'variant': variant,
                   'tuned_macro_f1_mean': float(frame.tuned_macro_f1.mean()), 'tuned_macro_f1_std': float(frame.tuned_macro_f1.std(ddof=1)),
                   'tuned_accuracy_mean': float(frame.tuned_accuracy.mean()), 'tuned_accuracy_std': float(frame.tuned_accuracy.std(ddof=1))}
            if variant == reference:
                row.update(seeds_above=None, seeds_below=None, mean_difference=0.0, decision='reference')
            else:
                intervals = [bootstrap[variant, reference, seed]['macro_f1'] for seed in config['seeds']]
                row.update(seeds_above=sum(low > 0 for _, low, _ in intervals), seeds_below=sum(high < 0 for _, _, high in intervals),
                           mean_difference=float(np.mean([mean for mean, _, _ in intervals])))
                qualifies = row['tuned_macro_f1_mean'] > reference_mean and row['seeds_above'] >= sweep['min_seeds']
                row['decision'] = 'qualifies' if qualifies else 'tested, no reliable gain'
                if qualifies:
                    passing.append(row)
            rows.append(row)
        if passing:
            winner = max(passing, key=lambda row: row['tuned_macro_f1_mean'])
            winner['decision'] = 'adopted'
            settings.update(winner['values'])
            changed += keys
    return {'rule': f"replace the reference only if mean tuned macro-F1 is higher and the paired bootstrap macro-F1 interval vs {reference} "
                    f"is entirely above 0 in at least {sweep['min_seeds']} of {len(config['seeds'])} seeds",
            'reference': reference, 'settings': settings, 'changed': changed, 'table': rows}


def plot_sweep(path, history, sweep):
    factors = list(sweep['factors'].items())
    fig, axes = plt.subplots(len(factors), 2, figsize=(11, 3.2 * len(factors)), dpi=160, squeeze=False)
    for (factor, spec), (loss_ax, f1_ax) in zip(factors, axes):
        for order, (variant, (color, style)) in enumerate(zip(spec['variants'], SERIES_STYLES)):
            group = history[history.variant == variant]
            label = f'{variant} (reference)' if variant == sweep['reference'] and 'reference' not in variant else variant
            for ax, column in ((loss_ax, 'val_loss'), (f1_ax, 'val_tuned_macro_f1')):
                mean = group.groupby('epoch')[column].mean()
                ax.plot(mean.index, mean.values, color=color, linestyle=style, linewidth=2, label=label, zorder=10 - order)
        for ax, title, ylabel in ((loss_ax, f'{factor}: validation loss', 'BCE loss'), (f1_ax, f'{factor}: validation macro-F1, tuned per epoch', 'Macro-F1')):
            ax.set_title(title, loc='left', fontsize=10.5, color=INK)
            ax.set_xlabel('Epoch', fontsize=9, color=MUTED)
            ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
            ax.grid(axis='y', color=GRID_LINE, linewidth=.8)
            ax.set_axisbelow(True)
            ax.tick_params(colors=MUTED, labelsize=8, length=0)
            for side in ('top', 'right', 'left'):
                ax.spines[side].set_visible(False)
            ax.spines['bottom'].set_color(AXIS_LINE)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.text(.01, .005, 'Lines show the mean over seeds still running at that epoch. One factor changes per row; everything else stays at the reference.',
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, .02, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def write_gpu_report(path, config, runs, baseline, bootstrap, checks, environment, best_variant, best_seed):
    def spread(frame, column, digits=4):
        return f'{frame[column].mean():.{digits}f} ± {frame[column].std(ddof=1):.{digits}f}'

    seeds_used = len(config['seeds'])
    val, at_05, cv = baseline['validation_at_threshold'], baseline['validation_at_0.5'], baseline['cv']
    comparison = [{'Model': 'Baseline exp04 (saved laptop pipeline)', 'Threshold': f"{baseline['threshold']:.2f}", 'Accuracy, tuned': f"{val['accuracy']:.4f}", 'Macro-F1, tuned': f"{val['macro_f1']:.4f}",
                   'Accuracy, 0.5': f"{at_05['accuracy']:.4f}", 'Macro-F1, 0.5': f"{at_05['macro_f1']:.4f}", 'Neg precision, tuned': f"{val['neg_precision']:.4f}",
                   'Neg recall, tuned': f"{val['neg_recall']:.4f}", 'ROC-AUC': f"{baseline['validation_roc_auc']:.4f}"}]
    settings, overfit, agreement, cost, classes, seeds, confusion = [], [], [], [], [], [], []
    for variant, frame in runs.groupby('variant', sort=False):
        first = frame.iloc[0]
        source = first.reused_from if isinstance(first.get('reused_from'), str) else 'trained here'
        chosen = {**config, **config['variants'][variant]}
        settings.append({'Variant': variant, 'Embedding': chosen['embedding_init'], 'Embedding trains': 'No' if chosen['freeze_embedding'] else 'Yes',
                         'pos_weight': f'{first.pos_weight:.4f}', 'Checkpoint and early stopping': first.checkpoint_metric,
                         'lr': chosen['lr'], 'Dropout': chosen['dropout'], 'max_len': chosen['max_len'], 'Filters per kernel': chosen['num_filters'], 'Runs': source})
        comparison.append({'Model': f'CNN {variant}', 'Threshold': spread(frame, 'tuned_threshold', 2), 'Accuracy, tuned': spread(frame, 'tuned_accuracy'), 'Macro-F1, tuned': spread(frame, 'tuned_macro_f1'),
                           'Accuracy, 0.5': spread(frame, 't05_accuracy'), 'Macro-F1, 0.5': spread(frame, 't05_macro_f1'), 'Neg precision, tuned': spread(frame, 'tuned_neg_precision'),
                           'Neg recall, tuned': spread(frame, 'tuned_neg_recall'), 'ROC-AUC': spread(frame, 'val_roc_auc')})
        overfit.append({'Variant': variant, 'Best epoch': spread(frame, 'best_epoch', 1), 'Epochs run': spread(frame, 'epochs_run', 1), 'Train macro-F1': spread(frame, 'best_train_macro_f1'),
                        'Val macro-F1': spread(frame, 'best_val_macro_f1'), 'Gap': spread(frame, 'train_val_gap'), 'Train loss': spread(frame, 'best_train_eval_loss'), 'Val loss': spread(frame, 'best_val_loss')})
        cost.append({'Variant': variant, 'Trainable parameters': f'{int(first.trainable_params):,}', 'Total parameters': f'{int(first.total_params):,}',
                     'Train s per run (median)': f'{frame.total_train_seconds.median():.1f}', 'Train s per run (range)': f'{frame.total_train_seconds.min():.1f} to {frame.total_train_seconds.max():.1f}'})
        for key, label in (('neg_precision', 'Neg precision'), ('neg_recall', 'Neg recall'), ('neg_f1', 'Neg F1'), ('pos_precision', 'Pos precision'), ('pos_recall', 'Pos recall'), ('pos_f1', 'Pos F1')):
            classes.append({'Variant': variant, 'Metric': label, 'At 0.5': spread(frame, f't05_{key}'), 'Tuned': spread(frame, f'tuned_{key}')})
        for row in frame.itertuples():
            seeds.append({'Variant': variant, 'Seed': row.seed, 'Best epoch': row.best_epoch, 'Epochs run': row.epochs_run, 'Threshold': f'{row.tuned_threshold:.2f}',
                          'Accuracy, 0.5': f'{row.t05_accuracy:.4f}', 'Macro-F1, 0.5': f'{row.t05_macro_f1:.4f}', 'Accuracy, tuned': f'{row.tuned_accuracy:.4f}',
                          'Macro-F1, tuned': f'{row.tuned_macro_f1:.4f}', 'ROC-AUC': f'{row.val_roc_auc:.4f}', 'Gap': f'{row.train_val_gap:.4f}',
                          'Train s': f'{row.total_train_seconds:.1f}', 'Node': row.node, 'GPU bus ID': row.gpu_bus_id})
            agreement.append({'Variant': variant, 'Seed': row.seed, 'Checkpoint epoch': row.best_epoch, 'Best macro-F1 epoch (0.5)': row.best_f1_epoch,
                              'Best macro-F1 epoch (tuned)': row.best_tuned_f1_epoch, 'Macro-F1 at checkpoint (0.5)': f'{row.best_val_macro_f1:.4f}',
                              'Best macro-F1 (0.5)': f'{row.best_f1_epoch_macro_f1:.4f}'})
            for label, tn, fp, fn, tp in (('0.50', row.t05_tn, row.t05_fp, row.t05_fn, row.t05_tp), (f'{row.tuned_threshold:.2f}', row.tuned_tn, row.tuned_fp, row.tuned_fn, row.tuned_tp)):
                confusion.append({'Variant': variant, 'Seed': row.seed, 'Threshold': label, 'True neg, pred neg': tn, 'True neg, pred pos': fp, 'True pos, pred neg': fn, 'True pos, pred pos': tp})

    boot_rows, boot_summary = [], []
    for (variant, against, seed), values in bootstrap.items():
        boot_rows.append({'Variant': variant, 'Against': against, 'Seed': seed, 'Macro-F1 difference': f"{values['macro_f1'][0]:+.4f}", 'Macro-F1 95% interval': f"{values['macro_f1'][1]:+.4f} to {values['macro_f1'][2]:+.4f}",
                          'Accuracy difference': f"{values['accuracy'][0]:+.4f}", 'Accuracy 95% interval': f"{values['accuracy'][1]:+.4f} to {values['accuracy'][2]:+.4f}"})
    for variant, against in dict.fromkeys((variant, against) for variant, against, _ in bootstrap):
        pairs = [values for (v, a, _), values in bootstrap.items() if (v, a) == (variant, against)]

        def summary(key):
            above = sum(values[key][1] > 0 for values in pairs)
            below = sum(values[key][2] < 0 for values in pairs)
            return f"{np.mean([values[key][0] for values in pairs]):+.4f}", f'{above} above 0, {below} below 0, {len(pairs) - above - below} include 0'
        f1_mean, f1_count = summary('macro_f1')
        acc_mean, acc_count = summary('accuracy')
        boot_summary.append({'Variant': variant, 'Against': against, 'Mean macro-F1 difference': f1_mean, 'Macro-F1 intervals by seed': f1_count,
                             'Mean accuracy difference': acc_mean, 'Accuracy intervals by seed': acc_count})

    grid_lines = []
    if config['grid']:
        columns = list(next(iter(config['grid'].values())))
        for column_name, label in (('tuned_macro_f1', 'Macro-F1, tuned'), ('tuned_accuracy', 'Accuracy, tuned'), ('tuned_threshold', 'Tuned threshold')):
            table = [{GRID_FACTORS['rows']: row, **{col: spread(runs[runs.variant == variant], column_name, 2 if column_name == 'tuned_threshold' else 4) for col, variant in cells.items()}}
                     for row, cells in config['grid'].items()]
            grid_lines += [f'### {label} (mean ± sample std over {seeds_used} seeds)', '', f"Columns: {GRID_FACTORS['columns']}.", '', *md_table(table, [GRID_FACTORS['rows'], *columns]), '']
        for column_name, label in (('tuned_macro_f1', 'macro-F1'), ('tuned_accuracy', 'accuracy')):
            effects = [{'Effect': name, f'Tuned {label} (mean ± std over seeds)': f'{values.mean():+.4f} ± {values.std(ddof=1):.4f}', 'Seeds above 0': f'{int((values > 0).sum())} of {len(values)}'}
                       for name, values in grid_effects(runs, config['grid'], column_name).items()]
            grid_lines += [f'### Effects on tuned {label}', '', *md_table(effects, list(effects[0])), '']
        grid_lines += ['Effects are computed per seed (the same seed in every cell) and then summarised. A main effect averages the change over the other factor. '
                       'An interaction near 0 means the two factors add up; a large one means the effect of one depends on the other.', '']

    sweep_lines = []
    if config['sweep']:
        decision = sweep_decision(config, runs, bootstrap)
        by_variant = runs.groupby('variant', sort=False)
        table = [{'Factor': row['factor'], 'Value': ', '.join(f'{key} {value}' for key, value in row['values'].items()) if len(row['values']) > 1 else row['value'],
                  'Variant': row['variant'], 'Parameters': f"{int(by_variant.get_group(row['variant']).total_params.iloc[0]):,}",
                  'Macro-F1, tuned': f"{row['tuned_macro_f1_mean']:.4f} ± {row['tuned_macro_f1_std']:.4f}",
                  'Accuracy, tuned': f"{row['tuned_accuracy_mean']:.4f} ± {row['tuned_accuracy_std']:.4f}",
                  'Best epoch': spread(by_variant.get_group(row['variant']), 'best_epoch', 1),
                  'Gap': spread(by_variant.get_group(row['variant']), 'train_val_gap'),
                  'Train s (median)': f"{by_variant.get_group(row['variant']).total_train_seconds.median():.1f}",
                  'Macro-F1 diff vs reference': f"{row['mean_difference']:+.4f}",
                  'Bootstrap seeds above / below 0': '—' if row['seeds_above'] is None else f"{row['seeds_above']} / {row['seeds_below']}",
                  'Decision': row['decision']} for row in decision['table']]
        outcome = ', '.join(f'{setting} = {value}' for setting, value in decision['settings'].items())
        sweep_lines = ['## One-factor sweep and decision', '', f"Rule (fixed in `docs/report_tracker.md` before this run): {decision['rule']}.", '',
                       *md_table(table, list(table[0])), '',
                       f"Outcome: {outcome}. " + (f"Changed from the reference: {', '.join(decision['changed'])}." if decision['changed'] else 'No setting changed: every alternative is "tested, no reliable gain".'), '',
                       'Gap is train minus validation macro-F1 at threshold 0.5 at the checkpoint epoch. Train time is per run on the GPU.', '']
    uses_f1 = (runs.checkpoint_metric == 'val_macro_f1').any()
    title = config['report_title'] or 'Text CNN variants on GPU'
    lines = [
        f'# {title}', '',
        f"Validation set: 1,078 reviews. Holdout and official test data were not loaded. All runs: {environment['hardware']}, seeds {', '.join(map(str, config['seeds']))}. "
        f"Fixed settings: embedding dim {config['embedding_dim']}, kernels {config['kernel_sizes']} × {config['num_filters']} filters, dropout {config['dropout']}, Adam lr {config['lr']}, batch {config['batch_size']}, "
        f"max {config['max_epochs']} epochs, patience {config['patience']}, max_len {config['max_len']}.", '',
        '## Notes before reading', '',
        f"- Same device only: every CNN run here used {' and '.join(sorted(set(runs.device)))}. The earlier laptop CPU results in `results/archive_cpu/` are an archive and are not compared with these numbers.",
        '- The baseline is the saved exp04 pipeline produced on the laptop (`results/baseline/best_baseline_pipeline.joblib`), not retrained. It reproduces its saved validation metrics exactly on this machine.',
        '- Tuned thresholds are optimistic: the validation set both chooses each threshold and scores it.',
        *(['- Choosing checkpoints by validation macro-F1 reuses the validation set a second time (to pick the epoch as well as the threshold), which adds optimism to those variants\' scores. '
           'Their validation numbers favour them relative to loss-checkpointed variants; a holdout or cross-validation estimate is needed for a fair final comparison.'] if uses_f1 else []),
        '', '## Variants', '', *md_table(settings, list(settings[0])), '',
        *sweep_lines,
        *(['## 2×2 table', '', *grid_lines] if grid_lines else []),
        f'## Comparison (validation, mean ± sample std over {seeds_used} seeds for CNN)', '',
        *md_table(comparison, list(comparison[0])), '',
        f"Baseline nested CV for context: macro-F1 {cv['tuned_macro_f1']['mean']:.4f} ± {cv['tuned_macro_f1']['std']:.4f}, accuracy {cv['tuned_accuracy']['mean']:.4f} ± {cv['tuned_accuracy']['std']:.4f}. "
        'Seed ± is initialisation randomness on one split; CV ± is split variability. The CNN variants have no CV estimate.', '',
        '## Paired bootstrap on validation reviews (variant minus comparator, both at their tuned thresholds)', '',
        '### Summary over seeds', '', *md_table(boot_summary, list(boot_summary[0])), '',
        f"{config['bootstrap_resamples']} paired resamples of the same reviews per seed. CNN-vs-CNN comparisons pair the same seed. The baseline uses the saved exp04 pipeline's validation probabilities at threshold {baseline['threshold']:.2f}. "
        'An interval containing 0 means this validation set cannot separate the two. These intervals say nothing beyond this validation set.', '',
        '### By seed', '', *md_table(boot_rows, list(boot_rows[0])), '',
        '## Overfitting (at the checkpoint epoch)', '',
        *md_table(overfit, list(overfit[0])), '', 'Gap is train macro-F1 minus validation macro-F1 at threshold 0.5. Train loss is measured in eval mode, without dropout.', '',
        '## Learning curves', '', f"![Learning curves]({config['variants_prefix']}_learning_curves.png)", '',
        '## Checkpoint epoch vs best validation macro-F1 epoch', '',
        *md_table(agreement, list(agreement[0])), '',
        f"The checkpoint epoch equals the best 0.5-threshold macro-F1 epoch in {int((runs.best_epoch == runs.best_f1_epoch).sum())} of {len(runs)} runs, "
        f"and the best tuned macro-F1 epoch in {int((runs.best_epoch == runs.best_tuned_f1_epoch).sum())} of {len(runs)}.", '',
        '## Parameters and GPU training time', '', *md_table(cost, list(cost[0])), '',
        'Train time is on the GPU, includes per-epoch scoring of train and validation, and varies with load on the node. It is reported separately from CPU inference speed.', '',
        '## Results by seed', '', *md_table(seeds, list(seeds[0])), '',
        '### Per-class metrics (mean ± std)', '', *md_table(classes, list(classes[0])), '',
        '### Confusion matrices', '', *md_table(confusion, list(confusion[0])), '',
        '## Reproducibility', '',
        f"- Deterministic algorithms on: {environment['deterministic_algorithms']} (cuDNN deterministic, benchmark off, CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}). No operation was refused.",
        f"- Same-process rerun of seed {config['rerun_seed']}, metrics, history and tensors identical: " + (', '.join(f'{variant} {value}' for variant, value in checks['rerun'].items()) or 'no variant trained here') + '.',
        *([f"- Reused runs: {', '.join(sorted(set(runs.reused_from.dropna())))}. They are the same runs, not retrained." ] if 'reused_from' in runs and runs.reused_from.notna().any() else []),
        f"- Best by mean tuned macro-F1: {best_variant}; its best seed is {best_seed}. That single-seed score is optimistic. Its weights reload and reproduce their validation scores: {checks['reload']}.",
        f"- Weights for every run are listed in the `weights_file` column of `{config['variants_prefix']}_runs.csv`; validation probabilities are in `{config['variants_prefix']}_val_scores.npz`." if config['save_all_weights'] else '',
        f"- Environment: node {environment['node']}, {environment['hardware']} (bus {environment['gpu_bus_id']}, {environment['gpu_uuid']}), Slurm job {environment['slurm_job_id']}, "
        f"Python {environment['python']}, PyTorch {environment['torch']}, NumPy {environment['numpy']}.", '',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def run_variants(config):
    device = setup_device(config)
    out_dir, reference_dir, baseline_dir = Path(config['out_dir']), Path(config['reference_dir']), Path(config['baseline_dir'])
    prefix = config['variants_prefix']
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = json.loads(Path(config['reference_config']).read_text(encoding='utf-8'))
    differing = {key: value for key, value in reference.items() if config[key] != value and (config['reference_check'] or key != 'seeds')}
    if differing:
        raise ValueError(f'Settings differ from the reference config: {differing}')
    train, val, vocab, data = load_data(config)
    gloves = {}

    def glove_for(settings):
        """GloVe vectors for a variant's file and dimension, loaded once each, with a coverage report per file."""
        key = (settings['glove_path'], settings['embedding_dim'])
        if key not in gloves:
            vectors, std = load_glove(settings['glove_path'], vocab, settings['embedding_dim'])
            sources = glove_sources(vocab, vectors)
            suffix = '' if key == (config['glove_path'], config['embedding_dim']) else f"_{settings['embedding_dim']}d"
            write_coverage(out_dir / f'{prefix}_coverage{suffix}.md', vocab, sources, std, train, val)
            print(f"GloVe {settings['embedding_dim']}d direct={len(sources[0])} rescued={len(sources[1])} random={len(sources[2])} std={std:.4f}", flush=True)
            gloves[key] = (vocab, vectors, sources, std)
        return gloves[key]

    glove_for(config)
    environment = run_environment(device)
    print('Environment:', environment, flush=True)

    rows, histories, models, scores, rerun = [], [], {}, {}, {}
    reused = config['reuse'] or {}
    for variant, overrides in config['variants'].items():
        if variant in reused:
            print(f'--- {variant}: reusing {reused[variant]}', flush=True)
            reused_runs, reused_history, reused_scores = load_reused(reused[variant], variant, config['seeds'])
            rows += reused_runs.to_dict('records')
            histories.append(reused_history)
            scores.update({(variant, seed): values for seed, values in reused_scores.items()})
            continue
        settings = {**config, **overrides}
        variant_data = data if settings['max_len'] == config['max_len'] else {
            **data, 'train_x': encode(train.clean_text, vocab, settings['max_len']), 'val_x': encode(val.clean_text, vocab, settings['max_len'])}
        glove = glove_for(settings) if settings['embedding_init'] == 'glove' else None
        for seed in config['seeds']:
            print(f'--- {variant}', flush=True)
            result, frame, model, seed_scores = train_seed(seed, settings, variant_data, len(vocab), device, glove)
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            result = {'variant': variant, **result, 'train_val_gap': result['best_train_macro_f1'] - result['best_val_macro_f1'], 'model_size_kb': buffer.tell() / 1024, **parameter_counts(model, settings)}
            if not config['reference_check']:
                result.update({'device': environment['hardware'], 'node': environment['node'], 'gpu_bus_id': environment['gpu_bus_id'],
                               'gpu_uuid': environment['gpu_uuid'], 'slurm_job_id': environment['slurm_job_id']})
            if config['save_all_weights']:
                weights_file = out_dir / 'weights' / f'{variant}_seed{seed}.pt'
                weights_file.parent.mkdir(exist_ok=True)
                torch.save(model.state_dict(), weights_file)
                result['weights_file'] = str(weights_file)
            print(f"{variant} seed {seed}: checkpoint epoch {result['best_epoch']} of {result['epochs_run']}, threshold {result['tuned_threshold']:.2f}, "
                  f"tuned macro-F1 {result['tuned_macro_f1']:.4f}, accuracy {result['tuned_accuracy']:.4f}, {result['total_train_seconds']:.1f} s", flush=True)
            rows.append(result)
            histories.append(frame.assign(variant=variant))
            models[variant, seed], scores[variant, seed] = model, seed_scores
            if seed == config['rerun_seed'] and variant != 'random':
                again, again_frame, again_model, _ = train_seed(seed, settings, variant_data, len(vocab), device, glove)
                rerun[variant] = (all(again[key] == result[key] for key in again if key not in TIMING_COLUMNS) and again_frame.equals(frame)
                                  and all(torch.equal(left, right) for left, right in zip(model.state_dict().values(), again_model.state_dict().values())))
    runs, history = pd.DataFrame(rows), pd.concat(histories, ignore_index=True)

    reference_metrics = reference_weights = None
    if config['reference_check']:
        saved_runs = pd.read_csv(reference_dir / 'cnn_reference_runs.csv', float_precision='round_trip')
        columns = [column for column in saved_runs.columns if column in runs.columns and column not in TIMING_COLUMNS]
        fresh = runs[runs.variant == 'random'][columns].reset_index(drop=True)
        reference_metrics = bool((fresh.to_numpy(dtype=float) == saved_runs[columns].to_numpy(dtype=float)).all())
        saved_weights = torch.load(reference_dir / 'cnn_reference_best.pt', weights_only=True)
        reference_seed = json.loads((reference_dir / 'cnn_reference_config.json').read_text(encoding='utf-8'))['seed']
        reference_weights = all(torch.equal(tensor, saved_weights[name]) for name, tensor in models['random', reference_seed].state_dict().items())

    baseline = json.loads((baseline_dir / 'best_baseline_config.json').read_text(encoding='utf-8'))
    baseline_scores = joblib.load(baseline_dir / 'best_baseline_pipeline.joblib').predict_proba(val.clean_text)[:, 1]
    if threshold_metrics(data['val_labels'], baseline_scores, baseline['threshold']) != baseline['validation_at_threshold']:
        raise ValueError('The saved baseline pipeline no longer reproduces its saved validation metrics')
    baseline_predictions = baseline_scores >= baseline['threshold']
    tuned = {(row.variant, row.seed): scores[row.variant, row.seed] >= row.tuned_threshold for row in runs.itertuples()}
    pairs = config['bootstrap_pairs'] or [pair for variant in config['variants'] if variant != 'random' for pair in ((variant, 'random'), (variant, 'baseline exp04'))]
    bootstrap = {}
    for variant, against in pairs:
        for seed in config['seeds']:
            comparator = baseline_predictions if against == 'baseline exp04' else tuned[against, seed]
            bootstrap[variant, against, seed] = bootstrap_difference(data['val_labels'], tuned[variant, seed], comparator, config['bootstrap_resamples'])

    glove_runs = runs[runs.variant != 'random']
    best_variant = glove_runs.groupby('variant').tuned_macro_f1.mean().idxmax()
    best_row = glove_runs[glove_runs.variant == best_variant].sort_values(['tuned_macro_f1', 'seed'], ascending=[False, True], kind='mergesort').iloc[0]
    best_seed = int(best_row.seed)
    if config['save_all_weights']:
        weights_path = Path(best_row.weights_file)
    else:
        weights_path = out_dir / f'{best_variant}_best.pt'
        torch.save(models[best_variant, best_seed].state_dict(), weights_path)
    best_settings = {**config, **config['variants'][best_variant]}
    reloaded = TextCNN(len(vocab), best_settings['embedding_dim'], best_settings['kernel_sizes'], best_settings['num_filters'], best_settings['dropout']).to(device)
    reloaded.load_state_dict(torch.load(weights_path, weights_only=True))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(data['pos_weight'], dtype=torch.float32, device=device))
    reload_x = data['val_x'] if best_settings['max_len'] == config['max_len'] else encode(val.clean_text, vocab, best_settings['max_len'])
    _, reloaded_scores = score_split(reloaded, reload_x, data['val_y'], criterion, config['batch_size'], device)
    checks = {'reference_metrics': reference_metrics, 'reference_weights': reference_weights, 'rerun': {key: bool(value) for key, value in rerun.items()},
              'reload': bool(np.array_equal(reloaded_scores, scores[best_variant, best_seed]))}

    runs.to_csv(out_dir / f'{prefix}_runs.csv', index=False, lineterminator='\n')
    history.to_csv(out_dir / f'{prefix}_history.csv', index=False, lineterminator='\n')
    np.savez(out_dir / f'{prefix}_val_scores.npz', **{f'{variant}_seed{seed}': values for (variant, seed), values in scores.items()})
    for variant, overrides in config['variants'].items():
        settings = {**config, **overrides}
        (out_dir / f'{variant}_config.json').write_text(json.dumps({
            **{key: settings[key] for key in ('vocab_path', 'embedding_init', 'freeze_embedding', 'embedding_dim', 'max_len', 'kernel_sizes', 'num_filters', 'dropout', 'lr', 'batch_size', 'max_epochs', 'patience', 'seeds', 'checkpoint_metric')},
            'glove_path': settings['glove_path'] if settings['embedding_init'] == 'glove' else None,
            'glove_std': glove_for(settings)[3] if settings['embedding_init'] == 'glove' else None,
            'padding': 'end', 'truncating': 'end', 'pos_weight': effective_pos_weight(settings, data), 'vocab_size': len(vocab), 'device': str(device), 'environment': environment,
            'reused_from': reused.get(variant),
            'thresholds_by_seed': {str(row.seed): row.tuned_threshold for row in runs[runs.variant == variant].itertuples()},
            'saved_weights': {'file': str(weights_path) if config['save_all_weights'] else weights_path.name, 'seed': best_seed, 'threshold': float(best_row.tuned_threshold)} if variant == best_variant else None,
        }, indent=2), encoding='utf-8', newline='\n')
    if config['reference_check']:
        plot_variants(out_dir / f'{prefix}_learning_curves.png', history)
        write_glove_report(out_dir / f'{prefix}_report.md', config, runs, baseline, bootstrap, checks, environment, best_variant, best_seed)
    else:
        if config['sweep']:
            plot_sweep(out_dir / f'{prefix}_learning_curves.png', history, config['sweep'])
            decision = sweep_decision(config, runs, bootstrap)
            (out_dir / 'sweep_decision.json').write_text(json.dumps(decision, indent=2), encoding='utf-8', newline='\n')
            print('Sweep decision:', decision['settings'], 'changed:', decision['changed'], flush=True)
        else:
            plot_gpu_variants(out_dir / f'{prefix}_learning_curves.png', history)
        write_gpu_report(out_dir / f'{prefix}_report.md', config, runs, baseline, bootstrap, checks, environment, best_variant, best_seed)
    print('Checks:', checks)
    print('Outputs saved in:', out_dir.resolve())
    reference_ok = not config['reference_check'] or (reference_metrics and reference_weights)
    if not (reference_ok and checks['reload'] and all(checks['rerun'].values())):
        raise ValueError(f'Reproducibility check failed: {checks}')


def main(config):
    device = setup_device(config)
    out_dir, baseline_dir = Path(config['out_dir']), Path(config['baseline_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val, vocab, data = load_data(config)
    print(f"pos_weight={data['pos_weight']:.4f} device={device} threads={torch.get_num_threads()}", flush=True)

    seed_everything(config['seeds'][0])
    shape_model = TextCNN(len(vocab), config['embedding_dim'], config['kernel_sizes'], config['num_filters'], config['dropout']).to(device).eval()
    shapes = check_shapes(shape_model, data['train_x'][:config['batch_size']].to(device), config)
    counts = parameter_counts(shape_model, config)
    print(md_table(shapes, ['Layer', 'Output shape']) and '\n'.join(md_table(shapes, ['Layer', 'Output shape'])), flush=True)

    rows, histories, models, all_scores = [], [], {}, {}
    for seed in config['seeds']:
        result, frame, model, scores = train_seed(seed, config, data, len(vocab), device)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        result.update({'model_size_kb': buffer.tell() / 1024, **benchmark_cnn(model, val.clean_text, vocab, config), **counts})
        rows.append(result)
        histories.append(frame)
        models[seed], all_scores[seed] = model, scores
    runs, history = pd.DataFrame(rows), pd.concat(histories, ignore_index=True)

    # Rerun in the same process: every metric and saved tensor must match exactly.
    rerun_seed = config['rerun_seed']
    again, again_frame, again_model, _ = train_seed(rerun_seed, config, data, len(vocab), device)
    reference = runs[runs.seed == rerun_seed].iloc[0]
    metrics_equal = all(again[key] == reference[key] for key in again if key not in TIMING_COLUMNS) and again_frame.equals(history[history.seed == rerun_seed].reset_index(drop=True))
    tensors_equal = all(torch.equal(left, right) for left, right in zip(models[rerun_seed].state_dict().values(), again_model.state_dict().values()))

    best_row = runs.sort_values(['tuned_macro_f1', 'seed'], ascending=[False, True], kind='mergesort').iloc[0]
    best_seed = int(best_row.seed)
    weights_path = out_dir / f"{config['run_name']}_best.pt"
    torch.save(models[best_seed].state_dict(), weights_path)
    reloaded = TextCNN(len(vocab), config['embedding_dim'], config['kernel_sizes'], config['num_filters'], config['dropout']).to(device)
    reloaded.load_state_dict(torch.load(weights_path, weights_only=True))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(data['pos_weight'], dtype=torch.float32, device=device))
    _, reloaded_scores = score_split(reloaded, data['val_x'], data['val_y'], criterion, config['batch_size'], device)
    reload_equal = bool(np.array_equal(reloaded_scores, all_scores[best_seed]))
    rerun = {'metrics': bool(metrics_equal), 'tensors': bool(tensors_equal), 'reload': reload_equal}

    baseline_path = baseline_dir / 'best_baseline_pipeline.joblib'
    baseline_config = json.loads((baseline_dir / 'best_baseline_config.json').read_text(encoding='utf-8'))
    pipeline = joblib.load(baseline_path)
    baseline_scores = pipeline.predict_proba(val.clean_text)[:, 1]
    baseline_predictions = baseline_scores >= baseline_config['threshold']
    if abs(threshold_metrics(data['val_labels'], baseline_scores, baseline_config['threshold'])['macro_f1'] - baseline_config['validation_at_threshold']['macro_f1']) > 1e-12:
        raise ValueError('Saved baseline no longer reproduces its validation macro-F1')
    baseline_timing = benchmark_baseline(pipeline, train, val.clean_text, config, baseline_path)
    bootstrap = {int(row.seed): bootstrap_difference(data['val_labels'], all_scores[int(row.seed)] >= row.tuned_threshold, baseline_predictions, config['bootstrap_resamples'])
                 for row in runs.itertuples()}

    environment = {
        'device': str(device), 'hardware': torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor(),
        'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__,
        'torch_threads': torch.get_num_threads(), 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
    }
    runs.to_csv(out_dir / f"{config['run_name']}_runs.csv", index=False, lineterminator='\n')
    history.to_csv(out_dir / f"{config['run_name']}_history.csv", index=False, lineterminator='\n')
    plot_curves(out_dir / 'learning_curves.png', history, runs)
    timing = {key: float(runs[key].median()) for key in ('preprocess_ms_per_1000', 'forward_ms_per_1000', 'inference_ms_per_1000')}
    saved = {
        **{key: config[key] for key in ('vocab_path', 'embedding_dim', 'max_len', 'kernel_sizes', 'num_filters', 'dropout', 'lr', 'batch_size', 'max_epochs', 'patience')},
        'padding': 'end', 'truncating': 'end', 'pos_weight': data['pos_weight'], 'vocab_size': len(vocab),
        'seed': best_seed, 'best_epoch': int(best_row.best_epoch), 'threshold': float(best_row.tuned_threshold),
        'threshold_rule': 'predict 1 (positive) when P(positive) >= threshold', 'weights': weights_path.name,
        'device': str(device), 'environment': environment,
        'validation_at_threshold': {key: best_row[f'tuned_{key}'].item() for key in METRIC_KEYS},
        'validation_roc_auc': float(best_row.val_roc_auc),
    }
    (out_dir / f"{config['run_name']}_config.json").write_text(json.dumps(saved, indent=2), encoding='utf-8', newline='\n')
    write_report(out_dir / f"{config['run_name']}_report.md", config, runs, history, shapes, counts, timing, baseline_timing, baseline_config, bootstrap, rerun, environment, best_seed)
    print('Rerun checks:', rerun)
    print('Tuned accuracy and macro-F1 by seed:', runs[['seed', 'tuned_accuracy', 'tuned_macro_f1']].to_dict('records'))
    print('Outputs saved in:', out_dir.resolve())
    if not all(rerun.values()):
        raise ValueError(f'Reproducibility check failed: {rerun}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON file whose keys override the defaults in DEFAULT_CONFIG')
    parser.add_argument('--out-dir', help='Override the output directory')
    parser.add_argument('--device', help='Override the device, e.g. cuda or cpu; default auto')
    args = parser.parse_args()
    settings = dict(DEFAULT_CONFIG)
    if args.config:
        overrides = json.loads(Path(args.config).read_text(encoding='utf-8'))
        unknown = set(overrides) - set(settings)
        if unknown:
            raise ValueError(f'Unknown config keys: {sorted(unknown)}')
        settings.update(overrides)
    if args.out_dir:
        settings['out_dir'] = args.out_dir
    if args.device:
        settings['device'] = args.device
    (run_variants if settings['variants'] else main)(settings)
