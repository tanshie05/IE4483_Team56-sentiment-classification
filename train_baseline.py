"""IE4483 Project 1: TF-IDF + Logistic Regression baseline with threshold tuning and shortlist CV.
Run: python train_baseline.py --processed-dir processed --out-dir results/baseline
Reads only the train and validation partitions. Holdout and official test data are never loaded.
"""
import argparse
import io
import itertools
import json
import random
import statistics
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from eval_utils import THRESHOLDS, threshold_metrics, tune_threshold
from prepare_data import SEED, TFIDF_TOKEN_PATTERN

NGRAMS = {'unigram': (1, 1), 'unigram+bigram': (1, 2)}
MAX_FEATURES = [5000, 10000]
C_VALUES = [0.1, 1, 10]
EXTENDED_C_VALUES = [30, 100]
CLASS_WEIGHTS = [None, 'balanced']
MAX_ITER = 2000
INFERENCE_REPEATS = 5
TRAIN_REPEATS = 3
SHORTLIST_SIZE = 4
OUTER_FOLDS = 5
INNER_FOLDS = 3
# Set when any fit in the run, including CV fits, reaches MAX_ITER.
FIT_LOG = {'fits': 0, 'unconverged': 0}


def load_split(processed_dir, name, metadata):
    df = pd.read_csv(processed_dir / f'{name}.csv', keep_default_na=False)
    expected = metadata['split_stats'][name]['rows']
    if len(df) != expected:
        raise ValueError(f'{name}.csv has {len(df)} rows; split_metadata.json expects {expected}')
    return df


def evaluate(model, texts, labels):
    times = []
    for _ in range(INFERENCE_REPEATS):
        start = time.perf_counter()
        scores = model.predict_proba(texts)[:, 1]
        times.append(time.perf_counter() - start)
    inference_seconds = statistics.median(times)
    predictions = model.predict(texts)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        'val_accuracy': accuracy_score(labels, predictions),
        'val_macro_f1': f1_score(labels, predictions, average='macro', zero_division=0),
        'val_roc_auc': roc_auc_score(labels, scores),
        'val_neg_precision': precision[0], 'val_neg_recall': recall[0], 'val_neg_f1': f1[0],
        'val_pos_precision': precision[1], 'val_pos_recall': recall[1], 'val_pos_f1': f1[1],
        'val_tn': int(tn), 'val_fp': int(fp), 'val_fn': int(fn), 'val_tp': int(tp),
        'inference_seconds': inference_seconds,
        'inference_ms_per_1000': 1e6 * inference_seconds / len(labels),
    }, scores


def model_size_kb(model):
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.tell() / 1024


def always_positive(train, val):
    model = DummyClassifier(strategy='constant', constant=1)
    start = time.perf_counter()
    model.fit(train.clean_text, train.sentiments)
    train_seconds = time.perf_counter() - start
    metrics, _ = evaluate(model, val.clean_text, val.sentiments)
    return {
        'experiment': 'always_positive', 'model': 'always_positive', 'ngrams': '', 'max_features': '', 'n_features': 0,
        'C': '', 'class_weight': '', 'converged': '', 'n_iter': '', 'train_macro_f1': f1_score(
            train.sentiments, model.predict(train.clean_text), average='macro', zero_division=0),
        **metrics,
        'train_seconds': train_seconds, 'model_size_kb': model_size_kb(model),
    }


def build_pipeline(ngrams, max_features, c_value, class_weight):
    # Same TF-IDF settings as prepare_data.py except the n-gram range and feature cap under test.
    return Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, ngram_range=NGRAMS[ngrams], max_features=max_features, min_df=2,
                                  sublinear_tf=True, token_pattern=TFIDF_TOKEN_PATTERN)),
        ('clf', LogisticRegression(C=c_value, class_weight=class_weight, max_iter=MAX_ITER, random_state=SEED)),
    ])


def fit_logged(pipeline, texts, labels):
    pipeline.fit(texts, labels)
    FIT_LOG['fits'] += 1
    FIT_LOG['unconverged'] += int(pipeline.named_steps['clf'].n_iter_[0] >= MAX_ITER)
    return pipeline


def run_experiment(number, ngrams, max_features, c_value, class_weight, train, val):
    pipeline = build_pipeline(ngrams, max_features, c_value, class_weight)
    start = time.perf_counter()
    fit_logged(pipeline, train.clean_text, train.sentiments)
    train_seconds = time.perf_counter() - start
    n_iter = int(pipeline.named_steps['clf'].n_iter_[0])
    metrics, scores = evaluate(pipeline, val.clean_text, val.sentiments)
    labels = val.sentiments.to_numpy()
    threshold, _ = tune_threshold(labels, scores)
    row = {
        'experiment': f'exp{number:02d}', 'model': 'tfidf_logreg', 'ngrams': ngrams, 'max_features': max_features,
        'n_features': len(pipeline.named_steps['tfidf'].vocabulary_), 'C': c_value,
        'class_weight': class_weight or 'none', 'converged': n_iter < MAX_ITER, 'n_iter': n_iter,
        'train_macro_f1': f1_score(train.sentiments, pipeline.predict(train.clean_text), average='macro'),
        **metrics,
        'train_seconds': train_seconds, 'model_size_kb': model_size_kb(pipeline),
        'tuned_threshold': threshold,
        **{f'tuned_val_{key}': value for key, value in threshold_metrics(labels, scores, threshold).items()},
    }
    return row, pipeline, scores


def cross_validate_config(experiment, config, frame):
    """Nested CV: each outer fold's threshold comes from inner out-of-fold scores on its training folds only."""
    texts, labels, groups = frame.clean_text.to_numpy(), frame.sentiments.to_numpy(), frame.clean_text.to_numpy()
    rows = []
    outer = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    for fold, (fit_idx, test_idx) in enumerate(outer.split(texts, labels, groups), 1):
        if set(groups[fit_idx]) & set(groups[test_idx]):
            raise ValueError('Identical cleaned reviews span CV folds')
        inner_scores = np.zeros(len(fit_idx))
        inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
        for inner_fit, inner_test in inner.split(texts[fit_idx], labels[fit_idx], groups[fit_idx]):
            model = fit_logged(build_pipeline(*config), texts[fit_idx][inner_fit], labels[fit_idx][inner_fit])
            inner_scores[inner_test] = model.predict_proba(texts[fit_idx][inner_test])[:, 1]
        threshold, _ = tune_threshold(labels[fit_idx], inner_scores)
        model = fit_logged(build_pipeline(*config), texts[fit_idx], labels[fit_idx])
        scores = model.predict_proba(texts[test_idx])[:, 1]
        default = threshold_metrics(labels[test_idx], scores, .5)
        tuned = threshold_metrics(labels[test_idx], scores, threshold)
        row = {'experiment': experiment, 'fold': fold, 'inner_threshold': threshold,
               'roc_auc': roc_auc_score(labels[test_idx], scores)}
        for key in ('macro_f1', 'accuracy', 'neg_precision', 'neg_recall', 'neg_f1'):
            row[f'tuned_{key}'] = tuned[key]
            row[f't05_{key}'] = default[key]
        rows.append(row)
    folds = pd.DataFrame(rows)
    numeric = folds.drop(columns=['experiment', 'fold'])
    # std is the sample standard deviation across the outer folds.
    summary = [{'experiment': experiment, 'fold': 'mean', **numeric.mean().to_dict()},
               {'experiment': experiment, 'fold': 'std', **numeric.std(ddof=1).to_dict()}]
    return pd.concat([folds, pd.DataFrame(summary)], ignore_index=True)


def select_final(cv, logreg):
    """Highest CV mean macro-F1; within one std of it, prefer lower C, then fewer features."""
    means = cv[cv.fold == 'mean'].set_index('experiment').tuned_macro_f1
    stds = cv[cv.fold == 'std'].set_index('experiment').tuned_macro_f1
    top = means.idxmax()
    eligible = means[means >= means[top] - stds[top]].index
    info = logreg.set_index('experiment')
    chosen = min(eligible, key=lambda e: (float(info.loc[e, 'C']), int(info.loc[e, 'n_features']), -means[e], e))
    if chosen == top:
        reason = f'{chosen} has the highest CV mean macro-F1 and no simpler candidate lies within one std of it.'
    else:
        reason = (f'{top} has the highest CV mean macro-F1 ({means[top]:.4f} ± {stds[top]:.4f}), but {chosen} '
                  f'({means[chosen]:.4f}) is within one std and is more regularised or smaller, so it is preferred.')
    return chosen, top, list(eligible), reason


def plot_threshold_curve(path, curve, threshold, best_f1, default_f1, experiment):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=160)
    ax.plot(THRESHOLDS, curve, color='#2a6fb0', linewidth=2)
    ax.scatter([.5], [default_f1], s=45, color='white', edgecolor='#555555', linewidth=1.5, zorder=3)
    ax.scatter([threshold], [best_f1], s=55, color='#2a6fb0', edgecolor='white', linewidth=1.5, zorder=4)
    ax.annotate(f'Tuned {threshold:.2f}: {best_f1:.4f}', (threshold, best_f1), xytext=(0, 12),
                textcoords='offset points', ha='center', fontsize=9, color='#222222')
    ax.annotate(f'Default 0.50: {default_f1:.4f}', (.5, default_f1), xytext=(14, 12),
                textcoords='offset points', fontsize=9, color='#555555')
    ax.set_ylim(top=.95)
    ax.set_title(f'Validation macro-F1 by decision threshold ({experiment})', loc='left', fontsize=11, color='#222222')
    ax.set_xlabel('Threshold (predict positive when P(positive) ≥ threshold)', fontsize=9, color='#555555')
    ax.set_ylabel('Validation macro-F1', fontsize=9, color='#555555')
    ax.set_xlim(.05, .95)
    ax.grid(axis='y', color='#e6e6e6', linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors='#555555', labelsize=8, length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#bbbbbb')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def md_table(rows, columns):
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns)]
    lines += ['| ' + ' | '.join(str(row[column]) for column in columns) + ' |' for row in rows]
    return lines


def strategy_table(logreg):
    rows = {}
    strategies = [('(i) Unweighted, 0.5', 'none', ''), ('(ii) Balanced, 0.5', 'balanced', ''),
                  ('(iii) Unweighted, tuned', 'none', 'tuned_'), ('(iv) Balanced, tuned', 'balanced', 'tuned_')]
    for label, weight, prefix in strategies:
        group = logreg[logreg.class_weight == weight].sort_values(
            [f'{prefix}val_macro_f1', 'experiment'], ascending=[False, True], kind='mergesort')
        top = group.iloc[0]
        rows[label] = {
            'Strategy': label, 'Best config': top.experiment,
            'Best macro-F1': top[f'{prefix}val_macro_f1'], 'Best accuracy': top[f'{prefix}val_accuracy'],
            'Best neg precision': top[f'{prefix}val_neg_precision'], 'Best neg recall': top[f'{prefix}val_neg_recall'],
            'Mean macro-F1': group[f'{prefix}val_macro_f1'].mean(),
            'Mean neg precision': group[f'{prefix}val_neg_precision'].mean(),
            'Mean neg recall': group[f'{prefix}val_neg_recall'].mean(),
        }
    return rows


def write_report(path, results, ranked, best, majority, cv, selection, final_metrics, final_train_seconds, refit_size_kb):
    chosen, top, eligible, reason = selection
    logreg = results[results.model == 'tfidf_logreg'].copy()
    logreg['C'] = logreg.C.astype(float)
    logreg['gap'] = logreg.train_macro_f1 - logreg.val_macro_f1

    def fmt(value):
        return f'{value:.4f}' if isinstance(value, float) else value

    def per_class(metrics, label):
        return [{'Model': label, 'Class': name, 'Precision': f"{metrics[f'{key}_precision']:.4f}",
                 'Recall': f"{metrics[f'{key}_recall']:.4f}", 'F1': f"{metrics[f'{key}_f1']:.4f}"}
                for key, name in (('neg', 'negative (0)'), ('pos', 'positive (1)'))]

    def confusion(metrics, label):
        return {'Model': label, 'True neg, pred neg': metrics['tn'], 'True neg, pred pos': metrics['fp'],
                'True pos, pred neg': metrics['fn'], 'True pos, pred pos': metrics['tp']}

    majority_metrics = {key.removeprefix('val_'): majority[key] for key in majority.index if key.startswith('val_')}
    comparison = [
        {'Model': 'Always positive', 'Threshold': '-', 'Accuracy': f'{majority.val_accuracy:.4f}',
         'Macro-F1': f'{majority.val_macro_f1:.4f}', 'ROC-AUC': f'{majority.val_roc_auc:.4f}',
         'Neg precision': '0.0000', 'Neg recall': '0.0000'},
        {'Model': f'{best.experiment}, default', 'Threshold': '0.50', 'Accuracy': f'{best.val_accuracy:.4f}',
         'Macro-F1': f'{best.val_macro_f1:.4f}', 'ROC-AUC': f'{best.val_roc_auc:.4f}',
         'Neg precision': f'{best.val_neg_precision:.4f}', 'Neg recall': f'{best.val_neg_recall:.4f}'},
        {'Model': f'{best.experiment}, tuned', 'Threshold': f'{best.tuned_threshold:.2f}',
         'Accuracy': f"{final_metrics['accuracy']:.4f}", 'Macro-F1': f"{final_metrics['macro_f1']:.4f}",
         'ROC-AUC': f'{best.val_roc_auc:.4f}', 'Neg precision': f"{final_metrics['neg_precision']:.4f}",
         'Neg recall': f"{final_metrics['neg_recall']:.4f}"},
    ]

    c_order = sorted(logreg.C.unique())
    c_trend = []
    for weight in ('balanced', 'none'):
        for c_value in c_order:
            group = logreg[(logreg.class_weight == weight) & (logreg.C == c_value)]
            c_trend.append({'Weight': weight, 'C': f'{c_value:g}', 'Mean train macro-F1': f'{group.train_macro_f1.mean():.4f}',
                            'Mean val macro-F1': f'{group.val_macro_f1.mean():.4f}', 'Mean gap': f'{group.gap.mean():.4f}',
                            'Mean tuned val macro-F1': f'{group.tuned_val_macro_f1.mean():.4f}'})
    curves, peaks = [], []
    for (ngrams, cap, weight), group in logreg.groupby(['ngrams', 'max_features', 'class_weight'], sort=False):
        values = group.set_index('C').val_macro_f1
        peak = values.idxmax()
        peaks.append(peak)
        curves.append({'N-grams': ngrams, 'Cap': cap, 'Weight': weight, **{f'C={c:g}': f'{values[c]:.4f}' for c in c_order},
                       'Peak C': f'{peak:g}'})
    interior = sum(c_order[0] < peak < c_order[-1] for peak in peaks)
    balanced_means = logreg[logreg.class_weight == 'balanced'].groupby('C').val_macro_f1.mean()
    unweighted_means = logreg[logreg.class_weight == 'none'].groupby('C').val_macro_f1.mean()
    balanced_gap = logreg[logreg.class_weight == 'balanced'].groupby('C').gap.mean()
    unweighted_gap = logreg[logreg.class_weight == 'none'].groupby('C').gap.mean()

    strategies = strategy_table(logreg)
    strategy_rows = [{key: fmt(value) for key, value in row.items()} for row in strategies.values()]
    s1, s2, s3, s4 = (strategies[key] for key in strategies)
    combined_best = s4['Best macro-F1'] - max(s2['Best macro-F1'], s3['Best macro-F1'])
    combined_mean = s4['Mean macro-F1'] - max(s2['Mean macro-F1'], s3['Mean macro-F1'])
    verdict = 'complementary' if combined_best >= .01 else 'largely interchangeable at the best configuration'

    all_rows = [{'Rank': rank, 'ID': row.experiment, 'N-grams': row.ngrams, 'Cap': row.max_features,
                 'Features': row.n_features, 'C': row.C, 'Weight': row.class_weight,
                 'Macro-F1 at 0.5': f'{row.val_macro_f1:.4f}', 'Threshold': f'{row.tuned_threshold:.2f}',
                 'Tuned macro-F1': f'{row.tuned_val_macro_f1:.4f}', 'Tuned neg precision': f'{row.tuned_val_neg_precision:.4f}',
                 'Tuned neg recall': f'{row.tuned_val_neg_recall:.4f}', 'ROC-AUC': f'{row.val_roc_auc:.4f}',
                 'Train macro-F1': f'{row.train_macro_f1:.4f}'}
                for rank, row in enumerate(ranked.itertuples(), 1)]

    info = results.set_index('experiment')
    means, stds = cv[cv.fold == 'mean'].set_index('experiment'), cv[cv.fold == 'std'].set_index('experiment')
    cv_rows = []
    for experiment in means.sort_values('tuned_macro_f1', ascending=False).index:
        row = {'ID': experiment, 'Config': f"{info.loc[experiment, 'ngrams']}, cap {info.loc[experiment, 'max_features']}, "
                                            f"C={info.loc[experiment, 'C']}, {info.loc[experiment, 'class_weight']}"}
        for label, column in (('Macro-F1', 'tuned_macro_f1'), ('Accuracy', 'tuned_accuracy'), ('Neg precision', 'tuned_neg_precision'),
                              ('Neg recall', 'tuned_neg_recall'), ('ROC-AUC', 'roc_auc'), ('Macro-F1 at 0.5', 't05_macro_f1')):
            row[label] = f'{means.loc[experiment, column]:.4f} ± {stds.loc[experiment, column]:.4f}'
        row['Threshold'] = f"{means.loc[experiment, 'inner_threshold']:.2f} ± {stds.loc[experiment, 'inner_threshold']:.2f}"
        cv_rows.append(row)

    unigram_cap = logreg[(logreg.ngrams == 'unigram') & (logreg.max_features == 10000)].n_features.max()
    lines = [
        '# TF-IDF + Logistic Regression baseline', '',
        f'Validation set: {best.val_tn + best.val_fp + best.val_fn + best.val_tp} reviews. '
        'Holdout and official test data were not loaded. Every TF-IDF configuration was fitted on training data only.', '',
        '## Final selection rule (fixed before applying it)', '',
        f'1. Shortlist the top {SHORTLIST_SIZE} configurations by validation macro-F1 at their tuned threshold.',
        '2. Pick the highest CV mean macro-F1, using each outer fold\'s nested tuned threshold.',
        '3. If candidates are within one standard deviation of the top one (its std), prefer lower C, then fewer features.',
        '4. Refit on the train partition only and tune its threshold on validation.', '',
        f'**Selected: {best.experiment}** ({best.ngrams}, cap {best.max_features}, {best.n_features} features, C={best.C}, '
        f'class_weight={best.class_weight}, threshold {best.tuned_threshold:.2f}). {reason}', '',
        '## Selected model vs always-positive baseline (validation)', '',
        *md_table(comparison, ['Model', 'Threshold', 'Accuracy', 'Macro-F1', 'ROC-AUC', 'Neg precision', 'Neg recall']), '',
        f'Median train time over {TRAIN_REPEATS} fits: {final_train_seconds:.3f} s. Median inference over {INFERENCE_REPEATS} '
        f'predict_proba calls: {best.inference_ms_per_1000:.2f} ms per 1,000 reviews. Serialised pipeline: {refit_size_kb:.1f} KB. '
        'Timings depend on this machine.', '',
        '### Per-class metrics', '',
        *md_table(per_class(majority_metrics, 'Always positive')
                  + per_class({key.removeprefix('val_'): best[key] for key in best.index if key.startswith('val_')}, f'{best.experiment}, 0.50')
                  + per_class(final_metrics, f'{best.experiment}, {best.tuned_threshold:.2f}'),
                  ['Model', 'Class', 'Precision', 'Recall', 'F1']), '',
        '### Confusion matrices', '',
        *md_table([confusion(majority_metrics, 'Always positive'),
                   confusion({key.removeprefix('val_'): best[key] for key in best.index if key.startswith('val_')}, f'{best.experiment}, 0.50'),
                   confusion(final_metrics, f'{best.experiment}, {best.tuned_threshold:.2f}')],
                  ['Model', 'True neg, pred neg', 'True neg, pred pos', 'True pos, pred neg', 'True pos, pred pos']), '',
        '## C range: does macro-F1 peak inside the grid?', '',
        f'{interior} of {len(peaks)} n-gram, cap and weight combinations peak at an interior C (metrics at threshold 0.5). '
        f'Averaged over configurations, the peak is C={balanced_means.idxmax():g} for balanced and C={unweighted_means.idxmax():g} for unweighted.', '',
        *md_table(curves, ['N-grams', 'Cap', 'Weight', *[f'C={c:g}' for c in c_order], 'Peak C']), '',
        'Train-validation gap (regularisation vs overfitting). Gap is train macro-F1 minus validation macro-F1 at threshold 0.5.', '',
        *md_table(c_trend, ['Weight', 'C', 'Mean train macro-F1', 'Mean val macro-F1', 'Mean gap', 'Mean tuned val macro-F1']), '',
        f'Balanced mean gap moves from {balanced_gap.iloc[0]:.4f} at C={c_order[0]:g} to {balanced_gap.iloc[-1]:.4f} at C={c_order[-1]:g}; '
        f'unweighted from {unweighted_gap.iloc[0]:.4f} to {unweighted_gap.iloc[-1]:.4f}. Mean validation macro-F1 at the top C is '
        f'{balanced_means.iloc[-1]:.4f} (balanced) and {unweighted_means.iloc[-1]:.4f} (unweighted).', '',
        '## Weighting vs threshold tuning', '',
        *md_table(strategy_rows, ['Strategy', 'Best config', 'Best macro-F1', 'Best accuracy', 'Best neg precision', 'Best neg recall',
                                  'Mean macro-F1', 'Mean neg precision', 'Mean neg recall']), '',
        f'**Verdict: {verdict}.** At the best configuration, weighting alone gives {s2["Best macro-F1"]:.4f}, tuning alone {s3["Best macro-F1"]:.4f}, '
        f'and both {s4["Best macro-F1"]:.4f}, a gain of {combined_best:.4f} over the better single method. Over all configurations, the means are '
        f'{s1["Mean macro-F1"]:.4f} (neither), {s2["Mean macro-F1"]:.4f} (weighting), {s3["Mean macro-F1"]:.4f} (tuning) and {s4["Mean macro-F1"]:.4f} (both), '
        f'a mean gain of {combined_mean:.4f} from combining them. A gain under 0.01 is treated as interchangeable.', '',
        'Tuned-threshold validation scores are optimistic, because the same validation set both selects the threshold and is evaluated on it.', '',
        'Threshold plot for the selected model: `threshold_curve.png`.', '',
        f'## {OUTER_FOLDS}-fold CV on the shortlist (train + validation)', '',
        f'Method: nested CV with StratifiedGroupKFold keyed on clean_text. Each outer fold fits TF-IDF and LR on its training folds only and takes '
        f'its threshold from {INNER_FOLDS}-fold out-of-fold predictions inside those folds, so CV tuned scores do not reuse the evaluated fold. Values are mean ± sample std over outer folds. '
        'Because validation data is included and it helped choose the shortlist, these CV means are still slightly optimistic.', '',
        *md_table(cv_rows, ['ID', 'Config', 'Macro-F1', 'Accuracy', 'Neg precision', 'Neg recall', 'ROC-AUC', 'Macro-F1 at 0.5', 'Threshold']), '',
        f'Top CV mean: {top}. Within one std: {", ".join(eligible)}. Selected: {best.experiment}.', '',
        '## All experiments, ranked by tuned validation macro-F1', '',
        *md_table(all_rows, ['Rank', 'ID', 'N-grams', 'Cap', 'Features', 'C', 'Weight', 'Macro-F1 at 0.5', 'Threshold', 'Tuned macro-F1',
                             'Tuned neg precision', 'Tuned neg recall', 'ROC-AUC', 'Train macro-F1']), '',
        '## Fixed settings', '',
        '- `sublinear_tf=True`: damps repeated words, so a long review repeating one term does not dominate.',
        '- `min_df=2`: drops terms seen in one training review, which cannot generalise and are often typos or cut-off words.',
        "- Token pattern: keeps contractions such as `can't` and `isn't` plus `!` and `?`, so negation is not reduced to `can` or `isn`.",
        '- Stop words kept: standard stop lists contain `not`, `no` and similar words that carry sentiment.',
        f'- The 10,000 unigram cap does nothing: only {unigram_cap:,} eligible unigrams exist, so that run uses them all.', '',
        '## Verification and notes', '',
        f'- Convergence: {FIT_LOG["unconverged"]} of {FIT_LOG["fits"]} Logistic Regression fits, CV included, reached the {MAX_ITER}-iteration limit.',
        '- The saved pipeline reloads and reproduces its validation predictions (checked by the script).',
        f'- Negative validation reviews number {best.val_tn + best.val_fp}, so one review moves negative recall by {1 / (best.val_tn + best.val_fp):.2%}.',
        '- Holdout remains unused and is reserved for one final evaluation of this model.', '',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def main(processed_dir, out_dir):
    random.seed(SEED)
    np.random.seed(SEED)
    processed_dir, out_dir = Path(processed_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((processed_dir / 'split_metadata.json').read_text(encoding='utf-8'))
    train = load_split(processed_dir, 'train', metadata)
    val = load_split(processed_dir, 'val', metadata)
    if set(train.clean_text) & set(val.clean_text):
        raise ValueError('Train and validation partitions share cleaned reviews')

    rows, pipelines, val_scores, configs = [always_positive(train, val)], {}, {}, {}
    # The original 24 runs keep their numbers; extended C runs follow as exp25 onwards.
    grid = itertools.chain(itertools.product(NGRAMS, MAX_FEATURES, C_VALUES, CLASS_WEIGHTS),
                           itertools.product(NGRAMS, MAX_FEATURES, EXTENDED_C_VALUES, CLASS_WEIGHTS))
    for number, config in enumerate(grid, 1):
        ngrams, max_features, c_value, class_weight = config
        row, pipeline, scores = run_experiment(number, ngrams, max_features, c_value, class_weight, train, val)
        rows.append(row)
        pipelines[row['experiment']], val_scores[row['experiment']], configs[row['experiment']] = pipeline, scores, config
        print(row['experiment'], ngrams, max_features, f'C={c_value}', f'weight={class_weight}',
              f"macro-F1={row['val_macro_f1']:.4f}", f"tuned={row['tuned_val_macro_f1']:.4f}@{row['tuned_threshold']:.2f}",
              f"AUC={row['val_roc_auc']:.4f}")

    results = pd.DataFrame(rows)
    # Exact tuned macro-F1 ties go to the higher ROC-AUC, then the smaller model, then the earlier experiment.
    ranked = results[results.model == 'tfidf_logreg'].sort_values(
        ['tuned_val_macro_f1', 'val_roc_auc', 'n_features', 'experiment'], ascending=[False, False, True, True], kind='mergesort')
    shortlist = ranked.experiment.head(SHORTLIST_SIZE).tolist()
    print('Shortlist:', shortlist)

    frame = pd.concat([train, val], ignore_index=True)
    cv = pd.concat([cross_validate_config(e, configs[e], frame) for e in shortlist], ignore_index=True)
    cv.to_csv(out_dir / 'baseline_cv.csv', index=False, lineterminator='\n')
    selection = select_final(cv, results[results.model == 'tfidf_logreg'])
    chosen = selection[0]

    results['shortlisted'] = results.experiment.isin(shortlist)
    results['selected'] = results.experiment == chosen
    results.to_csv(out_dir / 'baseline_experiments.csv', index=False, lineterminator='\n')
    best = results[results.selected].iloc[0]
    majority = results[results.model == 'always_positive'].iloc[0]

    # Refit on the train partition only; timing uses the median of repeated fits.
    train_times = []
    for _ in range(TRAIN_REPEATS):
        final_pipeline = build_pipeline(*configs[chosen])
        start = time.perf_counter()
        fit_logged(final_pipeline, train.clean_text, train.sentiments)
        train_times.append(time.perf_counter() - start)
    labels = val.sentiments.to_numpy()
    threshold = float(best.tuned_threshold)
    final_scores = final_pipeline.predict_proba(val.clean_text)[:, 1]
    if not np.array_equal(final_scores, val_scores[chosen]):
        raise ValueError('Refit pipeline does not reproduce the experiment scores')
    final_metrics = threshold_metrics(labels, final_scores, threshold)

    best_path = out_dir / 'best_baseline_pipeline.joblib'
    joblib.dump(final_pipeline, best_path)
    reloaded = joblib.load(best_path).predict_proba(val.clean_text)[:, 1]
    if not np.array_equal(reloaded >= threshold, final_scores >= threshold):
        raise ValueError('Saved pipeline predictions do not match the fitted pipeline')

    _, curve = tune_threshold(labels, final_scores)
    plot_threshold_curve(out_dir / 'threshold_curve.png', curve, threshold, final_metrics['macro_f1'],
                         threshold_metrics(labels, final_scores, .5)['macro_f1'], chosen)

    means, stds = cv[cv.fold == 'mean'].set_index('experiment'), cv[cv.fold == 'std'].set_index('experiment')
    ngrams, max_features, c_value, class_weight = configs[chosen]
    config = {
        'experiment': chosen, 'seed': SEED, 'threshold': threshold,
        'threshold_rule': 'predict 1 (positive) when P(positive) >= threshold',
        'threshold_source': 'validation macro-F1 sweep 0.05 to 0.95 in steps of 0.01',
        'fitted_on': 'train partition only',
        'tfidf': {'lowercase': True, 'ngram_range': list(NGRAMS[ngrams]), 'max_features': max_features, 'min_df': 2,
                  'sublinear_tf': True, 'token_pattern': TFIDF_TOKEN_PATTERN, 'n_features': int(best.n_features)},
        'logistic_regression': {'C': c_value, 'class_weight': class_weight, 'max_iter': MAX_ITER, 'random_state': SEED},
        'selection_reason': selection[3],
        'validation_at_threshold': final_metrics,
        'validation_at_0.5': threshold_metrics(labels, final_scores, .5),
        'validation_roc_auc': float(best.val_roc_auc),
        'cv': {column: {'mean': float(means.loc[chosen, column]), 'std': float(stds.loc[chosen, column])}
               for column in means.columns if column != 'fold'},
    }
    (out_dir / 'best_baseline_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8', newline='\n')
    write_report(out_dir / 'baseline_report.md', results, ranked, best, majority, cv, selection, final_metrics,
                 statistics.median(train_times), model_size_kb(final_pipeline))
    print('Selected:', chosen, f'threshold={threshold:.2f}', f"validation macro-F1={final_metrics['macro_f1']:.4f}")
    print('Unconverged fits:', FIT_LOG['unconverged'], 'of', FIT_LOG['fits'])
    print('Outputs saved in:', out_dir.resolve())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', default='processed')
    parser.add_argument('--out-dir', default='results/baseline')
    args = parser.parse_args()
    main(args.processed_dir, args.out_dir)
