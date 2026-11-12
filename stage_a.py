"""IE4483 Project 1: stage A, the CNN + baseline combination test (selection rule 8), on cross-validation only.

Run: python stage_a.py [--cnn-dir results/cnn/cv_oof] [--out-dir results/combination]
Needs the CNN CV rerun with saved out-of-fold probabilities (cv_cnn.py with save_oof, configs_gpu/cnn_cv_oof.json).
Uses train + val only, on exactly the CV folds of train_baseline.py and cv_cnn.py. Holdout and test are never loaded.

1. Checks the CNN rerun against the saved CV (results/cnn/cv/cv_folds.csv, cv_runs.csv); reports any difference.
2. Recomputes the baseline exp04 nested CV (train_baseline.cross_validate_config, step for step) keeping its
   probabilities: inner out-of-fold scores on each outer fold's training part and outer-test scores.
   Checks them against results/baseline/baseline_cv.csv.
3. Per outer fold, thresholds tuned as in CV (eval_utils.tune_threshold on the inner validation reviews, applied to the
   outer test fold):
   (a) CNN 3-seed average: mean of the 3 seeds' probabilities.
   (b) CNN + baseline combination: mean of one CNN seed's and the baseline's probabilities, per seed, then averaged
       over the 3 seeds within the fold, like the single CNN. This matches the model that would be submitted (one
       CNN, seed 42, plus the baseline). The inner validation reviews are the CNN's inner validation split; the
       baseline's probabilities there are its own inner out-of-fold scores, so both are held-out predictions.
   (b') information only: combination with the 3-seed CNN average.
4. Rule 8 (fixed before any result): the combination is adopted only if its CV mean macro-F1 beats BOTH single models
   by at least 0.01 AND it is higher than both in at least 4 of 5 folds. Single models: the CNN (per-seed scores
   averaged per fold, from the rerun) and the baseline (recomputed nested CV).
   Stage-A threshold (used only if adopted) = median of the 15 combination inner thresholds, as rule 2 does for the CNN.
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import final_utils as fu
from cv_cnn import cv_splits
from eval_utils import threshold_metrics, tune_threshold
from prepare_data import SEED
from train_baseline import INNER_FOLDS, OUTER_FOLDS

KEYS = ['macro_f1', 'accuracy', 'neg_precision', 'neg_recall', 'neg_f1']
MARGIN, MIN_FOLDS = 0.01, 4


def baseline_nested_cv(frame, settings):
    """train_baseline.cross_validate_config for exp04, keeping the probabilities (indices are rows of frame)."""
    texts, labels, groups = frame.clean_text.to_numpy(), frame.sentiments.to_numpy(), frame.clean_text.to_numpy()
    outer = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    inner_prob, outer_prob, rows = np.full(len(frame), np.nan), np.full(len(frame), np.nan), []
    per_fold_inner = {}
    for fold, (fit_idx, test_idx) in enumerate(outer.split(texts, labels, groups), 1):
        inner_scores = np.zeros(len(fit_idx))
        inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
        for inner_fit, inner_test in inner.split(texts[fit_idx], labels[fit_idx], groups[fit_idx]):
            model = fu.make_baseline(settings).fit(texts[fit_idx][inner_fit], labels[fit_idx][inner_fit])
            inner_scores[inner_test] = model.predict_proba(texts[fit_idx][inner_test])[:, 1]
        threshold, _ = tune_threshold(labels[fit_idx], inner_scores)
        model = fu.make_baseline(settings).fit(texts[fit_idx], labels[fit_idx])
        scores = model.predict_proba(texts[test_idx])[:, 1]
        outer_prob[test_idx] = scores
        per_fold_inner[fold] = pd.Series(inner_scores, index=fit_idx)
        tuned = threshold_metrics(labels[test_idx], scores, threshold)
        rows.append({'fold': fold, 'inner_threshold': threshold, 'roc_auc': roc_auc_score(labels[test_idx], scores),
                     **{f'tuned_{k}': tuned[k] for k in KEYS}})
    return pd.DataFrame(rows), per_fold_inner, outer_prob


def score(inner_labels, inner_probs, test_labels, test_probs):
    threshold, _ = tune_threshold(inner_labels, inner_probs)
    m = threshold_metrics(test_labels, test_probs, threshold)
    return {'inner_threshold': threshold, 'roc_auc': float(roc_auc_score(test_labels, test_probs)), **{k: m[k] for k in KEYS}}


def spread(values, digits=4):
    values = list(values)
    return f'{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}' if len(values) > 1 else f'{values[0]:.{digits}f}'


def md(rows):
    columns = list(rows[0])
    return ['| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns)] + ['| ' + ' | '.join(str(r[c]) for c in columns) + ' |' for r in rows]


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = fu.load_labelled(['train', 'val'])  # Same frame and order as both nested CVs.
    labels = frame.sentiments.to_numpy()
    splits = {fold: (train_idx, val_idx, test_idx) for fold, train_idx, val_idx, test_idx in cv_splits(frame)}
    oof = pd.read_csv(Path(args.cnn_dir) / 'cv_oof_scores.csv')
    oof = oof[oof.variant == 'final']
    folds = sorted(oof.fold.unique())
    seeds = sorted(oof.seed.unique())
    notes = []

    # 1. CNN rerun vs the saved CV.
    new_runs = pd.read_csv(Path(args.cnn_dir) / 'cv_runs.csv')
    old_runs = pd.read_csv(args.saved_cnn_runs)
    new_runs, old_runs = new_runs[new_runs.variant == 'final'], old_runs[old_runs.variant == 'final']
    merged = new_runs.merge(old_runs, on=['fold', 'seed'], suffixes=('_new', '_old'))
    run_cols = ['tuned_macro_f1', 'tuned_accuracy', 'roc_auc', 'inner_threshold', 'best_epoch']
    run_diff = {c: float((merged[f'{c}_new'] - merged[f'{c}_old']).abs().max()) for c in run_cols}
    new_folds = pd.read_csv(Path(args.cnn_dir) / 'cv_folds.csv')
    old_folds = pd.read_csv(args.saved_cnn_folds)
    fm = new_folds[new_folds.variant == 'final'].merge(old_folds[old_folds.variant == 'final'], on='fold', suffixes=('_new', '_old'))
    fold_diff = {c: float((fm[f'{c}_new'] - fm[f'{c}_old']).abs().max()) for c in ('tuned_macro_f1', 'tuned_accuracy', 'roc_auc', 'inner_threshold')}
    cnn_match = all(v == 0 for v in fold_diff.values()) and all(v == 0 for v in run_diff.values())
    print('CNN rerun matches the saved CV exactly:', cnn_match, '| max abs differences per fold:', fold_diff, '| per run:', run_diff, flush=True)

    # Recompute the CNN per-run scores from the saved probabilities: must equal the rerun's own cv_runs.csv.
    cnn_inner, cnn_outer, cnn_rows = {}, {}, []
    for fold in folds:
        train_idx, val_idx, test_idx = splits[fold]
        for seed in seeds:
            part = oof[(oof.fold == fold) & (oof.seed == seed)]
            inner = part[part.role == 'inner_val'].set_index('row').prob.reindex(val_idx).to_numpy()
            outer = part[part.role == 'outer_test'].set_index('row').prob.reindex(test_idx).to_numpy()
            if np.isnan(inner).any() or np.isnan(outer).any():
                raise ValueError(f'Missing CNN probabilities for fold {fold} seed {seed}')
            cnn_inner[fold, seed], cnn_outer[fold, seed] = inner, outer
            cnn_rows.append({'fold': fold, 'seed': seed, **score(labels[val_idx], inner, labels[test_idx], outer)})
    cnn_runs = pd.DataFrame(cnn_rows)
    check = cnn_runs.merge(new_runs, on=['fold', 'seed'])
    recompute_gap = float(max((check.macro_f1 - check.tuned_macro_f1).abs().max(), (check.inner_threshold_x - check.inner_threshold_y).abs().max()))
    if recompute_gap > 1e-6:
        raise ValueError(f'Saved CNN probabilities do not reproduce cv_runs.csv (gap {recompute_gap})')

    # 2. Baseline nested CV with probabilities.
    base = fu.baseline_settings()
    base_folds, base_inner, base_outer = baseline_nested_cv(frame, base)
    saved = fu.baseline_cv_table()
    saved = saved[(saved.experiment == base['experiment']) & saved.fold.astype(str).str.isdigit()].astype({'fold': int})
    bm = base_folds.merge(saved, on='fold', suffixes=('_new', '_old'))
    base_diff = {c: float((bm[f'{c}_new'] - bm[f'{c}_old']).abs().max()) for c in ('tuned_macro_f1', 'tuned_accuracy', 'roc_auc', 'inner_threshold')}
    base_match = all(v < 1e-12 for v in base_diff.values())
    print('Baseline recomputation matches results/baseline/baseline_cv.csv:', base_match, base_diff, flush=True)
    base_folds = base_folds[base_folds.fold.isin(folds)]

    # 3. Per-fold scores.
    rows, comb_thresholds = [], []
    for fold in folds:
        train_idx, val_idx, test_idx = splits[fold]
        y_in, y_out = labels[val_idx], labels[test_idx]
        b_in, b_out = base_inner[fold].reindex(val_idx).to_numpy(), base_outer[test_idx]
        if np.isnan(b_in).any():
            raise ValueError('Baseline inner scores missing on the CNN inner validation split')
        single_cnn = cnn_runs[cnn_runs.fold == fold]
        avg_in = np.mean([cnn_inner[fold, s] for s in seeds], axis=0)
        avg_out = np.mean([cnn_outer[fold, s] for s in seeds], axis=0)
        per_seed = []
        for seed in seeds:
            r = score(y_in, (cnn_inner[fold, seed] + b_in) / 2, y_out, (cnn_outer[fold, seed] + b_out) / 2)
            per_seed.append(r)
            comb_thresholds.append(r['inner_threshold'])
        entries = {
            'CNN (single seed, mean of seeds)': {k: single_cnn[k].mean() for k in KEYS + ['roc_auc', 'inner_threshold']},
            'Baseline exp04 (nested CV)': base_folds[base_folds.fold == fold].iloc[0].rename(lambda c: c.replace('tuned_', '')).to_dict(),
            '(a) CNN 3-seed average': score(y_in, avg_in, y_out, avg_out),
            '(b) CNN + baseline (per seed, mean of seeds)': {k: statistics.mean(r[k] for r in per_seed) for k in KEYS + ['roc_auc', 'inner_threshold']},
            "(b') CNN 3-seed average + baseline (information)": score(y_in, (avg_in + b_in) / 2, y_out, (avg_out + b_out) / 2),
        }
        for model, values in entries.items():
            rows.append({'model': model, 'fold': fold, **{k: float(values[k]) for k in KEYS + ['roc_auc', 'inner_threshold']}})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / 'stage_a_folds.csv', index=False, lineterminator='\n')

    # 4. Rule 8.
    pivot = table.pivot(index='fold', columns='model', values='macro_f1')
    comb, cnn, bl = pivot['(b) CNN + baseline (per seed, mean of seeds)'], pivot['CNN (single seed, mean of seeds)'], pivot['Baseline exp04 (nested CV)']
    gain_cnn, gain_base = comb.mean() - cnn.mean(), comb.mean() - bl.mean()
    folds_above_both = int(((comb > cnn) & (comb > bl)).sum())
    adopted = bool(gain_cnn >= MARGIN and gain_base >= MARGIN and folds_above_both >= MIN_FOLDS and len(folds) == 5)
    threshold = round(float(np.median(comb_thresholds)), 2)
    avg = pivot['(a) CNN 3-seed average']
    decision = {
        'rule': 'Rule 8: adopt the CNN + baseline combination only if its CV mean macro-F1 beats BOTH single models by at least 0.01 '
                'AND it is higher than both in at least 4 of 5 folds.',
        'adopted': adopted, 'status': 'adopted' if adopted else 'tested, no reliable gain',
        'combination_mean_macro_f1': float(comb.mean()), 'cnn_mean_macro_f1': float(cnn.mean()), 'baseline_mean_macro_f1': float(bl.mean()),
        'gain_over_cnn': float(gain_cnn), 'gain_over_baseline': float(gain_base), 'folds_above_both': folds_above_both, 'folds': len(folds),
        'combination_threshold': threshold, 'combination_thresholds': comb_thresholds,
        'cnn_3seed_average_mean_macro_f1': float(avg.mean()), 'cnn_3seed_average_gain_over_cnn': float(avg.mean() - cnn.mean()),
        'cnn_rerun_matches_saved_cv': cnn_match, 'cnn_rerun_max_abs_diff_folds': fold_diff, 'cnn_rerun_max_abs_diff_runs': run_diff,
        'baseline_recompute_matches_saved': base_match, 'baseline_recompute_max_abs_diff': base_diff,
    }
    (out_dir / 'stage_a_decision.json').write_text(json.dumps(decision, indent=2), encoding='utf-8', newline='\n')
    print('Rule 8:', decision['status'], {k: decision[k] for k in ('combination_mean_macro_f1', 'gain_over_cnn', 'gain_over_baseline', 'folds_above_both', 'combination_threshold')}, flush=True)

    summary = []
    for model, group in table.groupby('model', sort=False):
        summary.append({'Model': model, 'Macro-F1': spread(group.macro_f1), 'Accuracy': spread(group.accuracy),
                        'Neg precision': spread(group.neg_precision), 'Neg recall': spread(group.neg_recall), 'Neg F1': spread(group.neg_f1),
                        'ROC-AUC': spread(group.roc_auc), 'Threshold': spread(group.inner_threshold, 2)})
    by_fold = [{'Fold': f, 'CNN': f'{cnn[f]:.4f}', 'Baseline': f'{bl[f]:.4f}', '(a) CNN 3-seed avg': f'{avg[f]:.4f}',
                '(b) CNN + baseline': f'{comb[f]:.4f}', '(b) higher than both': 'yes' if comb[f] > cnn[f] and comb[f] > bl[f] else 'no'} for f in folds]
    lines = [
        '# Stage A: CNN + baseline combination on cross-validation (selection rule 8)', '',
        f'5-fold CV on train + val ({len(frame):,} reviews), the same outer folds as the baseline nested CV and the CNN CV; CNN seeds '
        f"{', '.join(map(str, seeds))}. Holdout and test not loaded. Each fold's threshold is tuned on that fold's inner validation reviews "
        '(macro-F1 sweep 0.05-0.95) and applied to its outer test fold; mean ± sample std over folds.', '',
        '## Checks', '',
        f"- CNN CV rerun with saved probabilities (`{args.cnn_dir}`) vs the saved CV (`results/cnn/cv/`): "
        f"{'identical' if cnn_match else 'NOT identical'}. Largest absolute difference per fold: "
        + ', '.join(f'{k} {v:.2e}' for k, v in fold_diff.items()) + '; per run: ' + ', '.join(f'{k} {v:.2e}' for k, v in run_diff.items()) + '.',
        f"- Baseline nested CV recomputed with probabilities vs `results/baseline/baseline_cv.csv`: {'identical' if base_match else 'NOT identical'}. "
        'Largest absolute difference: ' + ', '.join(f'{k} {v:.2e}' for k, v in base_diff.items()) + '.',
        '- The single-CNN rows below are recomputed from the saved probabilities and equal the rerun\'s `cv_runs.csv`.', '',
        '## Results (mean ± std over 5 folds)', '', *md(summary), '',
        '## Macro-F1 by fold', '', *md(by_fold), '',
        '## Rule 8 decision', '',
        f"Combination (b): mean macro-F1 {comb.mean():.4f}; minus CNN {gain_cnn:+.4f}, minus baseline {gain_base:+.4f} (needed: at least +{MARGIN} over both); "
        f'higher than both in {folds_above_both} of {len(folds)} folds (needed: {MIN_FOLDS}).', '',
        f"**{'Adopted' if adopted else 'Not adopted: tested, no reliable gain'}.** "
        + (f'The combination is scored on the holdout at threshold {threshold} (median of {len(comb_thresholds)} inner thresholds).' if adopted
           else 'Rules 4-7 apply unchanged; the combination is not scored on the holdout.'), '',
        f"(a) CNN 3-seed average: mean macro-F1 {avg.mean():.4f}, {avg.mean() - cnn.mean():+.4f} vs the single CNN. Information only: rule 5 fixes "
        'the official CNN as one seed, with no seed ensemble.', '',
        '(b) uses one CNN seed + the baseline per run (the model that would be submitted); (b\') uses the 3-seed CNN average and is information only.', '',
        'Files: `stage_a_folds.csv` (every model, every fold), `stage_a_decision.json`.',
    ]
    (out_dir / 'stage_a_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print('Outputs saved in:', out_dir.resolve(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cnn-dir', default='results/cnn/cv_oof')
    parser.add_argument('--saved-cnn-runs', default='results/cnn/cv/cv_runs.csv')
    parser.add_argument('--saved-cnn-folds', default='results/cnn/cv/cv_folds.csv')
    parser.add_argument('--out-dir', default='results/combination')
    main(parser.parse_args())
