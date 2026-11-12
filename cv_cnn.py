"""IE4483 Project 1: 5-fold cross-validation of the final Text CNN config on train + validation.
Uses the baseline's outer folds (StratifiedGroupKFold, 5 folds, shuffle, seed 42, grouped by clean_text, on the same
train + val frame), so fold for fold the CNN and the baseline's nested CV score the same reviews.
Run: python cv_cnn.py --config configs_gpu/cnn_cv.json
Reads only the train and validation partitions. Holdout and official test data are never loaded.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # Required for deterministic CUBLAS on a GPU.

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from eval_utils import threshold_metrics
from prepare_data import SEED
from train_baseline import OUTER_FOLDS
from train_cnn import (DEFAULT_CONFIG, effective_pos_weight, encode, glove_sources, load_data, load_glove, md_table,
                       run_environment, score_split, setup_device, train_seed)

# The first fold of a 6-way split of the outer training folds is the inner validation split: about 1/6, close to the
# original validation share (1,078 of 6,104 reviews). It drives early stopping and threshold tuning only.
INNER_SPLITS = 6
METRICS = ['macro_f1', 'accuracy', 'neg_precision', 'neg_recall', 'neg_f1']
CV_DEFAULTS = {'decision': None, 'cv_variants': None, 'cv_seeds': [42, 43, 44], 'cv_fold_limit': None, 'final_variant': 'final',
               'baseline_cv': 'results/baseline/baseline_cv.csv', 'baseline_experiment': 'exp04',
               # save_oof: also save every run's per-review probabilities (inner validation and outer test) to cv_oof_scores.csv.
               'save_oof': False}


def cv_splits(frame):
    texts, labels, groups = frame.clean_text.to_numpy(), frame.sentiments.to_numpy(), frame.clean_text.to_numpy()
    outer = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    for fold, (fit_idx, test_idx) in enumerate(outer.split(texts, labels, groups), 1):
        inner = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
        inner_fit, inner_val = next(inner.split(texts[fit_idx], labels[fit_idx], groups[fit_idx]))
        train_idx, val_idx = fit_idx[inner_fit], fit_idx[inner_val]
        for a, b in ((train_idx, val_idx), (train_idx, test_idx), (val_idx, test_idx)):
            if set(groups[a]) & set(groups[b]):
                raise ValueError('Identical cleaned reviews span CV splits')
        yield fold, train_idx, val_idx, test_idx


def fold_data(tokens, labels, train_idx, val_idx):
    train_labels, val_labels = labels[train_idx], labels[val_idx]
    return {'train_x': tokens[torch.as_tensor(train_idx)], 'val_x': tokens[torch.as_tensor(val_idx)],
            'train_y': torch.tensor(train_labels, dtype=torch.float32), 'val_y': torch.tensor(val_labels, dtype=torch.float32),
            'train_labels': train_labels, 'val_labels': val_labels,
            'pos_weight': int((train_labels == 0).sum()) / int((train_labels == 1).sum())}


def spread(values, digits=4):
    return f'{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}'


def write_report(path, config, settings, runs, folds, baseline_folds, recipe, check, environment, fold_sizes):
    final = config['final_variant']
    base = baseline_folds.set_index('fold')
    rows = [{'Model': f"Baseline {config['baseline_experiment']} (nested CV, saved laptop results)",
             **{label: spread(base[f'tuned_{key}']) for key, label in (('macro_f1', 'Macro-F1, tuned'), ('accuracy', 'Accuracy, tuned'),
                                                                     ('neg_precision', 'Neg precision, tuned'), ('neg_recall', 'Neg recall, tuned'))},
             'ROC-AUC': spread(base.roc_auc), 'Threshold': spread(base.inner_threshold, 2), 'Best epoch': '—'}]
    for variant, frame in folds.groupby('variant', sort=False):
        rows.append({'Model': f'CNN {variant}', 'Macro-F1, tuned': spread(frame.tuned_macro_f1), 'Accuracy, tuned': spread(frame.tuned_accuracy),
                     'Neg precision, tuned': spread(frame.tuned_neg_precision), 'Neg recall, tuned': spread(frame.tuned_neg_recall),
                     'ROC-AUC': spread(frame.roc_auc), 'Threshold': spread(frame.inner_threshold, 2), 'Best epoch': spread(frame.best_epoch, 1)})
    at_05 = [{'Model': f"Baseline {config['baseline_experiment']}", 'Macro-F1, 0.5': spread(base.t05_macro_f1), 'Accuracy, 0.5': spread(base.t05_accuracy)}]
    at_05 += [{'Model': f'CNN {variant}', 'Macro-F1, 0.5': spread(frame.t05_macro_f1), 'Accuracy, 0.5': spread(frame.t05_accuracy)}
              for variant, frame in folds.groupby('variant', sort=False)]
    paired = []
    for variant, frame in folds.groupby('variant', sort=False):
        for key, label in (('tuned_macro_f1', 'Macro-F1'), ('tuned_accuracy', 'Accuracy')):
            difference = frame.set_index('fold')[key] - base[key].loc[frame.fold.to_numpy()].to_numpy()
            paired.append({'CNN variant': variant, 'Metric': label, 'CNN minus baseline (mean ± std over folds)': f'{difference.mean():+.4f} ± {difference.std(ddof=1):.4f}',
                           'Folds where CNN is higher': f'{int((difference > 0).sum())} of {len(difference)}'})
    by_fold = [{'Variant': row.variant, 'Fold': row.fold, 'Macro-F1, tuned': f'{row.tuned_macro_f1:.4f}', 'Accuracy, tuned': f'{row.tuned_accuracy:.4f}',
                'Baseline macro-F1': f"{base.loc[row.fold, 'tuned_macro_f1']:.4f}", 'Baseline accuracy': f"{base.loc[row.fold, 'tuned_accuracy']:.4f}",
                'Threshold (seed mean)': f'{row.inner_threshold:.2f}', 'Best epoch (seed mean)': f'{row.best_epoch:.1f}'} for row in folds.itertuples()]
    per_run = [{'Variant': row.variant, 'Fold': row.fold, 'Seed': row.seed, 'Best epoch': row.best_epoch, 'Epochs run': row.epochs_run,
                'Threshold': f'{row.inner_threshold:.2f}', 'Macro-F1, tuned': f'{row.tuned_macro_f1:.4f}', 'Accuracy, tuned': f'{row.tuned_accuracy:.4f}',
                'Macro-F1, 0.5': f'{row.t05_macro_f1:.4f}', 'ROC-AUC': f'{row.roc_auc:.4f}', 'Train s': f'{row.total_train_seconds:.1f}'} for row in runs.itertuples()]
    lines = [
        '# Text CNN: 5-fold cross-validation (train + validation)', '',
        f"Final config from the hyperparameter sweep (`{config['decision']}`): " + ', '.join(f'{key} = {value}' for key, value in settings.items()) + '. '
        f"Fine-tuned GloVe 6B 100d, pos_weight = 1. Seeds per fold: {', '.join(map(str, config['cv_seeds']))}. Holdout and official test data were not loaded.", '',
        '## How the folds work', '',
        f"- Outer folds: the baseline's scheme exactly (StratifiedGroupKFold, {OUTER_FOLDS} folds, shuffle, random_state {SEED}, grouped by `clean_text`, on train + val in the same order), "
        'so each CNN fold scores the same reviews as the matching baseline fold.',
        f'- Inside each fold, the first part of a {INNER_SPLITS}-way StratifiedGroupKFold of the training folds is the inner validation split, used only for early stopping and threshold tuning. '
        'The model is then scored once on the outer fold.',
        "- The baseline's per-fold threshold came from 3-fold out-of-fold scores on its training folds, and the baseline was retrained on all of them; the CNN uses one inner split instead, "
        'because early stopping needs held-out data during training. Both thresholds come only from the training folds.',
        '- The vocabulary (`processed/cnn_vocab.json`) was built from the original training partition and is kept as is. Some of those reviews fall in CV test folds, so the test folds '
        'saw no labels but did shape which words have their own embedding. This affects every fold equally and only slightly favours the CNN.',
        "- The baseline's fold results are the saved laptop values in `results/baseline/baseline_cv.csv` (the official baseline). Retraining the baseline on the server gives slightly different numbers, see the README.", '',
        *md_table(fold_sizes, list(fold_sizes[0])), '',
        '## Results (outer folds; CNN values are averaged over seeds within each fold, then mean ± sample std over folds)', '',
        *md_table(rows, list(rows[0])), '', *md_table(at_05, list(at_05[0])), '',
        'The ± is split variability across the 5 folds, the same as the baseline nested CV ± (0.8253 ± 0.0190 macro-F1), so these rows compare like for like.', '',
        '## Paired by fold (same reviews)', '', *md_table(paired, list(paired[0])), '',
        'With only 5 folds this is descriptive: it shows how consistently one model leads, not a significance test.', '',
        '## Retraining recipe (selection rules 2 and 3 in `docs/report_tracker.md`)', '',
        f"- Threshold = median of the {recipe['runs']} inner-validation thresholds of `{final}`: **{recipe['threshold']:.2f}** (mean {recipe['threshold_mean']:.3f}, range {recipe['threshold_min']:.2f} to {recipe['threshold_max']:.2f}).",
        f"- Epochs = median best epoch over the same runs: **{recipe['epochs']}** (mean {recipe['epochs_mean']:.1f}, range {recipe['epochs_min']} to {recipe['epochs_max']}).",
        '- Saved in `cv_recipe.json`. These are used only when retraining on all labelled data; holdout has not been used.', '',
        *(['## Check: unweighted + validation macro-F1 checkpointing', '', check, ''] if check else []),
        '## By fold', '', *md_table(by_fold, list(by_fold[0])), '',
        '## Every run', '', *md_table(per_run, list(per_run[0])), '',
        '## Environment', '',
        f"- Node {environment['node']}, {environment['hardware']} (bus {environment['gpu_bus_id']}), Slurm job {environment['slurm_job_id']}, PyTorch {environment['torch']}. "
        f"Deterministic algorithms on: {environment['deterministic_algorithms']}.", '',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def main(config):
    device = setup_device(config)
    out_dir = Path(config['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    decision = json.loads(Path(config['decision']).read_text(encoding='utf-8'))
    settings = decision['settings']
    config = {**config, **settings}
    train, val, vocab, _ = load_data(config)
    frame = pd.concat([train, val], ignore_index=True)  # Same frame and order as train_baseline.py's nested CV.
    labels = frame.sentiments.to_numpy()
    tokens = encode(frame.clean_text, vocab, config['max_len'])
    vectors, std = load_glove(config['glove_path'], vocab, config['embedding_dim'])
    glove = (vocab, vectors, glove_sources(vocab, vectors), std)
    environment = run_environment(device)
    print('Final settings from the sweep:', settings, 'environment:', environment, flush=True)

    splits = list(cv_splits(frame))[:config['cv_fold_limit']]
    fold_sizes = [{'Fold': fold, 'Inner train': len(train_idx), 'Inner validation': len(val_idx), 'Outer test': len(test_idx),
                   'Outer test negatives': int((labels[test_idx] == 0).sum())} for fold, train_idx, val_idx, test_idx in splits]
    rows, histories, oof = [], [], []
    for variant, overrides in config['cv_variants'].items():
        variant_config = {**config, **overrides}
        for fold, train_idx, val_idx, test_idx in splits:
            data = fold_data(tokens, labels, train_idx, val_idx)
            for seed in config['cv_seeds']:
                print(f'--- {variant} fold {fold} seed {seed}', flush=True)
                result, history, model, inner_scores = train_seed(seed, variant_config, data, len(vocab), device, glove)
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(effective_pos_weight(variant_config, data), dtype=torch.float32, device=device))
                _, scores = score_split(model, tokens[torch.as_tensor(test_idx)], torch.tensor(labels[test_idx], dtype=torch.float32), criterion, config['batch_size'], device)
                if config['save_oof']:
                    for role, index, values in (('inner_val', val_idx, inner_scores), ('outer_test', test_idx, scores)):
                        oof.append(pd.DataFrame({'variant': variant, 'fold': fold, 'seed': seed, 'role': role, 'row': index,
                                                 'label': labels[index], 'prob': values}))
                tuned = threshold_metrics(labels[test_idx], scores, result['tuned_threshold'])
                default = threshold_metrics(labels[test_idx], scores, .5)
                row = {'variant': variant, 'fold': fold, 'seed': seed, 'best_epoch': result['best_epoch'], 'epochs_run': result['epochs_run'],
                       'inner_threshold': result['tuned_threshold'], 'roc_auc': float(roc_auc_score(labels[test_idx], scores)),
                       **{f'tuned_{key}': tuned[key] for key in METRICS}, **{f't05_{key}': default[key] for key in METRICS},
                       'inner_val_macro_f1': result['tuned_macro_f1'], 'total_train_seconds': result['total_train_seconds'],
                       'node': environment['node'], 'gpu_bus_id': environment['gpu_bus_id'], 'slurm_job_id': environment['slurm_job_id']}
                print(f"{variant} fold {fold} seed {seed}: epoch {row['best_epoch']}, threshold {row['inner_threshold']:.2f}, "
                      f"outer macro-F1 {row['tuned_macro_f1']:.4f}, accuracy {row['tuned_accuracy']:.4f}", flush=True)
                rows.append(row)
                histories.append(history.assign(variant=variant, fold=fold))
    runs = pd.DataFrame(rows)
    numeric = [column for column in runs.columns if column not in ('variant', 'fold', 'seed', 'node', 'gpu_bus_id', 'slurm_job_id')]
    folds = runs.groupby(['variant', 'fold'], sort=False)[numeric].mean().reset_index()

    final = runs[runs.variant == config['final_variant']]
    recipe = {'settings': settings, 'pos_weight': config['cv_variants'][config['final_variant']].get('pos_weight', config['pos_weight']),
              'checkpoint_metric': config['cv_variants'][config['final_variant']].get('checkpoint_metric', config['checkpoint_metric']),
              'runs': len(final), 'threshold': float(np.median(final.inner_threshold)), 'threshold_mean': float(final.inner_threshold.mean()),
              'threshold_min': float(final.inner_threshold.min()), 'threshold_max': float(final.inner_threshold.max()),
              'epochs': int(np.median(final.best_epoch)), 'epochs_median_exact': float(np.median(final.best_epoch)), 'epochs_mean': float(final.best_epoch.mean()),
              'epochs_min': int(final.best_epoch.min()), 'epochs_max': int(final.best_epoch.max()), 'environment': environment}
    check = None
    others = [variant for variant in config['cv_variants'] if variant != config['final_variant']]
    for other in others:
        a = folds[folds.variant == other].set_index('fold')
        b = folds[folds.variant == config['final_variant']].set_index('fold')
        difference = a.tuned_macro_f1 - b.tuned_macro_f1
        check = (f"`{other}` minus `{config['final_variant']}`, macro-F1 by fold: {difference.mean():+.4f} ± {difference.std(ddof=1):.4f}, higher in {int((difference > 0).sum())} of {len(difference)} folds. "
                 f"Accuracy: {(a.tuned_accuracy - b.tuned_accuracy).mean():+.4f}. This check cannot change the final config (rule 1).")
    baseline_cv = pd.read_csv(config['baseline_cv'])
    baseline_folds = baseline_cv[(baseline_cv.experiment == config['baseline_experiment']) & baseline_cv.fold.str.isdigit()].astype({'fold': int})
    runs.to_csv(out_dir / 'cv_runs.csv', index=False, lineterminator='\n')
    folds.to_csv(out_dir / 'cv_folds.csv', index=False, lineterminator='\n')
    pd.concat(histories, ignore_index=True).to_csv(out_dir / 'cv_history.csv', index=False, lineterminator='\n')
    if config['save_oof']:
        pd.concat(oof, ignore_index=True).to_csv(out_dir / 'cv_oof_scores.csv', index=False, lineterminator='\n', float_format='%.9g')
    (out_dir / 'cv_recipe.json').write_text(json.dumps(recipe, indent=2), encoding='utf-8', newline='\n')
    write_report(out_dir / 'cv_report.md', config, settings, runs, folds, baseline_folds, recipe, check, environment, fold_sizes)
    print('Recipe:', {key: recipe[key] for key in ('threshold', 'epochs')}, flush=True)
    print('Outputs saved in:', out_dir.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='JSON file whose keys override DEFAULT_CONFIG and the CV defaults')
    args = parser.parse_args()
    settings = {**DEFAULT_CONFIG, **CV_DEFAULTS}
    overrides = json.loads(Path(args.config).read_text(encoding='utf-8'))
    unknown = set(overrides) - set(settings)
    if unknown:
        raise ValueError(f'Unknown config keys: {sorted(unknown)}')
    settings.update(overrides)
    main(settings)
