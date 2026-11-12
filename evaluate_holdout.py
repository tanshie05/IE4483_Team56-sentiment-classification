"""IE4483 Project 1: one holdout evaluation of the final CNN and the baseline (selection rules 4-8 in docs/report_tracker.md).

Official run (once):    python evaluate_holdout.py
Dry run (no holdout):   python evaluate_holdout.py --eval-split val --fit-splits train --out-dir <scratch folder>

Rule 5: the final CNN config retrained on train + val for the recipe's fixed epochs (no early stopping), seeds 42-46,
        recipe threshold; the official CNN is seed 42.
Rule 6: exp04 settings (not re-selected) retrained on train + val, threshold = median exp04 nested-CV inner threshold;
        the saved laptop pipeline (train only, threshold 0.41) is a secondary row.
Rule 4: submit the CNN unless its paired-bootstrap interval (CNN minus rule-6 baseline) for macro-F1 or accuracy
        lies entirely below 0.
Rule 8: the CNN + baseline combination is scored only if stage A adopted it (--stage-a results/combination/stage_a_decision.json);
        if adopted, it is submitted unless its paired-bootstrap interval against the official CNN lies entirely below 0.
Official test data is never loaded here.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

import final_utils as fu
from prepare_data import tokenize
from train_cnn import md_table, run_environment

MODEL_METRICS = ['accuracy', 'macro_f1', 'neg_precision', 'neg_recall', 'neg_f1', 'pos_precision', 'pos_recall', 'pos_f1']


def metric_row(name, fitted_on, result, extra=None):
    m = result['at_threshold']
    row = {'Model': name, 'Fitted on': fitted_on, 'Threshold': f"{result['threshold']:.2f}",
           **{key: f'{m[key]:.4f}' for key in MODEL_METRICS}, 'ROC-AUC': f"{result['roc_auc']:.4f}",
           'Confusion (TN FP FN TP)': f"{m['tn']} {m['fp']} {m['fn']} {m['tp']}"}
    if extra:
        row.update(extra)
    return row


def spread(values):
    return f'{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}' if len(values) > 1 else f'{values[0]:.4f}'


def interval_text(b):
    return f"{b['observed']:+.4f} [{b['low']:+.4f}, {b['high']:+.4f}]"


def main(args):
    start_all = time.perf_counter()
    device = fu.setup_deterministic(args.device)
    environment = run_environment(device)
    print('Environment:', environment, flush=True)
    out_dir = Path(args.out_dir)
    prefix = args.eval_split
    if prefix == 'holdout' and (out_dir / 'holdout_metrics.json').exists():
        raise SystemExit(f'{out_dir}/holdout_metrics.json exists: the holdout is evaluated only once')
    if prefix == 'holdout' and not args.stage_a:
        raise SystemExit('The holdout evaluation needs --stage-a (rule 8 must be decided first)')
    (out_dir / 'weights').mkdir(parents=True, exist_ok=True)
    fit_splits = args.fit_splits.split(',')
    if args.eval_split in fit_splits:
        raise ValueError('The evaluation split cannot be part of the fitting data')

    stage_a = json.loads(Path(args.stage_a).read_text(encoding='utf-8')) if args.stage_a else None
    cnn, base = fu.cnn_settings(), fu.baseline_settings()
    if args.dry_run_epochs is not None:
        if args.eval_split == 'holdout' or 'holdout' in fit_splits:
            raise SystemExit('--dry-run-epochs is for dry runs only; the holdout always uses the recipe epochs')
        cnn['epochs'] = args.dry_run_epochs
    print('CNN settings:', cnn, flush=True)
    print('Baseline settings:', base, flush=True)
    fit = fu.load_labelled(fit_splits)
    evaluation = fu.load_labelled([args.eval_split])
    labels = evaluation.sentiments.to_numpy()
    print(f'Fitting data: {"+".join(fit_splits)}, {len(fit)} reviews ({int((fit.sentiments == 0).sum())} negative). '
          f'Evaluation: {args.eval_split}, {len(evaluation)} reviews ({int((labels == 0).sum())} negative).', flush=True)

    # Rule 5: CNN seeds, fixed epochs, recipe threshold.
    vocab = fu.load_vocab(cnn)
    glove = fu.load_glove_init(cnn, vocab)
    fit_tokens = fu.encode_texts(fit.clean_text, vocab, cnn)
    eval_tokens = fu.encode_texts(evaluation.clean_text, vocab, cnn)
    probabilities, results, losses, train_seconds = {}, {}, {}, {}
    for seed in args.seeds:
        start = time.perf_counter()
        model, losses[seed] = fu.train_cnn_fixed(seed, cnn, fit_tokens, fit.sentiments.to_numpy(), len(vocab), device, glove)
        train_seconds[seed] = time.perf_counter() - start
        torch.save(model.state_dict(), out_dir / 'weights' / f'cnn_{"_".join(fit_splits)}_seed{seed}.pt')
        probabilities[f'cnn_seed{seed}'] = fu.cnn_probabilities(model, eval_tokens, cnn['batch_size'], device)
        results[f'cnn_seed{seed}'] = fu.full_metrics(labels, probabilities[f'cnn_seed{seed}'], cnn['threshold'])
        r = results[f'cnn_seed{seed}']['at_threshold']
        print(f'CNN seed {seed}: {train_seconds[seed]:.1f} s, macro-F1 {r["macro_f1"]:.4f}, accuracy {r["accuracy"]:.4f}', flush=True)
    official = f'cnn_seed{args.official_seed}'

    rerun = None
    if args.rerun_check:
        model, _ = fu.train_cnn_fixed(args.official_seed, cnn, fit_tokens, fit.sentiments.to_numpy(), len(vocab), device, glove,
                                      log=lambda _message: None)
        again = fu.cnn_probabilities(model, eval_tokens, cnn['batch_size'], device)
        rerun = bool(np.array_equal(again, probabilities[official]))
        print('Rerun of the official seed gives identical probabilities:', rerun, flush=True)
        if not rerun:
            raise RuntimeError('Determinism check failed: retraining the official seed changed its probabilities')
    reloaded = fu.build_cnn(cnn, len(vocab), torch.load(out_dir / 'weights' / f'cnn_{"_".join(fit_splits)}_seed{args.official_seed}.pt',
                                                         map_location=device), device)
    if not np.array_equal(fu.cnn_probabilities(reloaded, eval_tokens, cnn['batch_size'], device), probabilities[official]):
        raise RuntimeError('Reloaded official CNN weights do not reproduce its probabilities')

    # Rule 6: exp04 settings retrained on the fitting data; saved laptop pipeline as a secondary row.
    baseline = fu.make_baseline(base)
    start = time.perf_counter()
    baseline.fit(fit.clean_text, fit.sentiments.to_numpy())
    baseline_seconds = time.perf_counter() - start
    if baseline.named_steps['clf'].n_iter_[0] >= base['logistic_regression']['max_iter']:
        raise RuntimeError('The retrained baseline hit max_iter')
    joblib.dump(baseline, out_dir / 'weights' / f'baseline_{"_".join(fit_splits)}.joblib')
    probabilities['baseline'] = baseline.predict_proba(evaluation.clean_text)[:, 1]
    results['baseline'] = fu.full_metrics(labels, probabilities['baseline'], base['threshold'])
    saved = joblib.load(fu.SAVED_BASELINE)
    probabilities['saved_baseline'] = saved.predict_proba(evaluation.clean_text)[:, 1]
    results['saved_baseline'] = fu.full_metrics(labels, probabilities['saved_baseline'], base['saved_threshold'])

    if stage_a is not None:
        if args.combination_threshold is not None:
            raise SystemExit('Give either --stage-a or --combination-threshold, not both')
        args.combination_threshold = stage_a['combination_threshold'] if stage_a['adopted'] else None
        print('Stage A (rule 8):', stage_a['status'], flush=True)
    combination = args.combination_threshold is not None
    if combination:
        probabilities['combination'] = (probabilities[official] + probabilities['baseline']) / 2
        results['combination'] = fu.full_metrics(labels, probabilities['combination'], args.combination_threshold)

    predictions = {name: (probabilities[name] >= results[name]['threshold']).astype(int) for name in probabilities}

    # Rule 4 (and rule 8 if adopted): paired bootstrap on the same reviews.
    bootstrap = {'official CNN minus rule-6 baseline': fu.paired_bootstrap(labels, predictions[official], predictions['baseline'], args.resamples),
                 'official CNN minus saved laptop baseline (secondary)': fu.paired_bootstrap(labels, predictions[official], predictions['saved_baseline'], args.resamples)}
    for seed in args.seeds:
        if seed != args.official_seed:
            bootstrap[f'CNN seed {seed} minus rule-6 baseline (information only)'] = fu.paired_bootstrap(
                labels, predictions[f'cnn_seed{seed}'], predictions['baseline'], args.resamples)
    if combination:
        bootstrap['combination minus official CNN'] = fu.paired_bootstrap(labels, predictions['combination'], predictions[official], args.resamples)
    main_test = bootstrap['official CNN minus rule-6 baseline']
    cnn_clearly_worse = main_test['macro_f1']['entirely_below_0'] or main_test['accuracy']['entirely_below_0']
    chosen = 'baseline' if cnn_clearly_worse else 'cnn'
    reason = ('Rule 4: the official CNN is clearly worse than the rule-6 baseline (a paired-bootstrap interval lies entirely below 0), '
              'so the baseline is submitted.' if cnn_clearly_worse else
              'Rule 4: neither paired-bootstrap interval (CNN minus rule-6 baseline, macro-F1 and accuracy) lies entirely below 0, '
              'so the CNN (main model) is submitted.')
    if combination:
        versus_cnn = bootstrap['combination minus official CNN']
        if not versus_cnn['macro_f1']['entirely_below_0'] and not versus_cnn['accuracy']['entirely_below_0']:
            chosen, reason = 'combination', reason + ' Rule 8 adopted the combination in stage A and on the holdout it is not clearly worse than the official CNN, so the combination is submitted.'
        else:
            reason += ' Rule 8 adopted the combination in stage A, but on the holdout it is clearly worse than the official CNN, so it is not submitted.'
    else:
        reason += f" Rule 8: stage A found the combination '{stage_a['status']}', so it is not scored." if stage_a else ' The combination was not scored.'
    decision = {'eval_split': args.eval_split, 'fit_splits': fit_splits, 'chosen': chosen, 'reason': reason,
                'official_cnn_seed': args.official_seed, 'cnn_threshold': cnn['threshold'], 'cnn_epochs': cnn['epochs'],
                'baseline_threshold': base['threshold'], 'combination_threshold': args.combination_threshold,
                'stage_a': stage_a,
                'rule4_bootstrap': main_test}
    print('Decision:', chosen, '-', reason, flush=True)

    # Save everything.
    frame = evaluation[['reviews', 'clean_text', 'sentiments']].copy()
    frame.insert(0, f'{prefix}_row', np.arange(len(frame)))
    frame['n_tokens'] = [len(tokenize(text)) for text in evaluation.clean_text]
    for name in probabilities:
        frame[f'{name}_prob'] = probabilities[name]
        frame[f'{name}_pred'] = predictions[name]
    frame.to_csv(out_dir / f'{prefix}_predictions.csv', index=False, lineterminator='\n')
    np.savez(out_dir / f'{prefix}_probabilities.npz', labels=labels, **probabilities)
    cnn_seed_names = [f'cnn_seed{seed}' for seed in args.seeds]
    summary = {key: {'mean': statistics.mean(results[n]['at_threshold'][key] for n in cnn_seed_names),
                     'std': statistics.stdev(results[n]['at_threshold'][key] for n in cnn_seed_names) if len(cnn_seed_names) > 1 else 0.0}
               for key in MODEL_METRICS}
    summary['roc_auc'] = {'mean': statistics.mean(results[n]['roc_auc'] for n in cnn_seed_names),
                          'std': statistics.stdev(results[n]['roc_auc'] for n in cnn_seed_names) if len(cnn_seed_names) > 1 else 0.0}
    metrics = {'results': results, 'cnn_seed_summary': summary, 'bootstrap': bootstrap, 'decision': decision,
               'cnn_settings': cnn, 'baseline_settings': base, 'train_seconds': {'cnn': train_seconds, 'baseline': baseline_seconds},
               'cnn_train_losses': losses, 'rerun_identical': rerun, 'environment': environment,
               'fit_rows': len(fit), 'eval_rows': len(evaluation), 'resamples': args.resamples}
    (out_dir / f'{prefix}_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8', newline='\n')
    (out_dir / f'{prefix}_decision.json').write_text(json.dumps(decision, indent=2), encoding='utf-8', newline='\n')

    fitted_on = '+'.join(fit_splits)
    rows = [metric_row(f'CNN seed {seed}{" (official)" if seed == args.official_seed else ""}', fitted_on, results[f'cnn_seed{seed}']) for seed in args.seeds]
    rows.append(metric_row('Baseline exp04, rule 6', fitted_on, results['baseline']))
    rows.append(metric_row('Baseline exp04, saved laptop pipeline (secondary)', 'train', results['saved_baseline']))
    if combination:
        rows.append(metric_row('Combination (CNN seed 42 + rule-6 baseline)', fitted_on, results['combination']))
    columns = list(rows[0])
    rows_05 = [{'Model': row['Model'], **{key: f"{results[name]['at_0.5'][key]:.4f}" for key in ('accuracy', 'macro_f1', 'neg_precision', 'neg_recall', 'neg_f1')},
                'Confusion (TN FP FN TP)': ' '.join(str(results[name]['at_0.5'][k]) for k in ('tn', 'fp', 'fn', 'tp'))}
               for row, name in zip(rows, cnn_seed_names + ['baseline', 'saved_baseline'] + (['combination'] if combination else []))]
    boot_rows = [{'Comparison': name, 'Macro-F1 difference [95% interval]': interval_text(b['macro_f1']),
                  'Accuracy difference [95% interval]': interval_text(b['accuracy']),
                  'Entirely below 0 (F1 / acc)': f"{'yes' if b['macro_f1']['entirely_below_0'] else 'no'} / {'yes' if b['accuracy']['entirely_below_0'] else 'no'}",
                  'Entirely above 0 (F1 / acc)': f"{'yes' if b['macro_f1']['entirely_above_0'] else 'no'} / {'yes' if b['accuracy']['entirely_above_0'] else 'no'}"}
                 for name, b in bootstrap.items()]
    lines = [
        f'# {"Holdout" if prefix == "holdout" else "Dry run on " + prefix}: final CNN vs baseline (selection rules 4-8)', '',
        f'Evaluation set: {args.eval_split}, {len(evaluation):,} reviews ({int((labels == 0).sum())} negative, {100 * labels.mean():.1f}% positive). '
        f'Models fitted on {fitted_on} ({len(fit):,} reviews). Official test data not loaded. Run once; nothing below was tuned on this set.', '',
        f"Device: {environment['hardware']} on {environment['node']}, Slurm job {environment['slurm_job_id']}, GPU bus {environment['gpu_bus_id']}. "
        f"Deterministic algorithms: {environment['deterministic_algorithms']}. Rerun of the official seed identical: {rerun}.", '',
        '## Settings fixed in advance', '',
        f"- CNN (rule 5): fine-tuned GloVe 6B {cnn['embedding_dim']}d, kernels {cnn['kernel_sizes']} × {cnn['num_filters']}, dropout {cnn['dropout']}, "
        f"Adam lr {cnn['lr']}, batch {cnn['batch_size']}, max_len {cnn['max_len']}, pos_weight {cnn['pos_weight']}; {cnn['epochs']} fixed epochs, "
        f"no early stopping, threshold {cnn['threshold']} (`results/cnn/cv/cv_recipe.json`). Seeds {', '.join(map(str, args.seeds))}; official seed {args.official_seed}.",
        f"- Baseline (rule 6): exp04 settings (unigram TF-IDF, 5,000 features, min_df 2, sublinear_tf, LR C=1, class_weight balanced), "
        f"threshold {base['threshold']} = median of exp04's nested-CV inner thresholds {base['fold_thresholds']}. Secondary: saved laptop pipeline at {base['saved_threshold']}.",
        f"- Combination (rule 8): {'scored at threshold ' + str(args.combination_threshold) if combination else 'stage A: ' + (stage_a['status'] if stage_a else 'not given') + ', not scored'}.", '',
        '## Results at the assigned thresholds', '', *md_table(rows, columns), '',
        f"CNN, 5 seeds (mean ± sample std): macro-F1 {spread([results[n]['at_threshold']['macro_f1'] for n in cnn_seed_names])}, "
        f"accuracy {spread([results[n]['at_threshold']['accuracy'] for n in cnn_seed_names])}, "
        f"neg precision {spread([results[n]['at_threshold']['neg_precision'] for n in cnn_seed_names])}, "
        f"neg recall {spread([results[n]['at_threshold']['neg_recall'] for n in cnn_seed_names])}, "
        f"ROC-AUC {spread([results[n]['roc_auc'] for n in cnn_seed_names])}.", '',
        '## Results at threshold 0.5 (for reference)', '', *md_table(rows_05, list(rows_05[0])), '',
        f'## Paired bootstrap ({args.resamples:,} resamples of the same reviews, generator seed 42)', '', *md_table(boot_rows, list(boot_rows[0])), '',
        'Observed difference first, then the 95% percentile interval of the resampled differences.', '',
        '## Decision', '', f'**Submission model: {chosen}.** {reason}', '',
        f"Training time: CNN {statistics.median(train_seconds.values()):.1f} s per seed (median, {environment['device']}); baseline {baseline_seconds:.2f} s on the CPU.", '',
        f'Files: `{prefix}_predictions.csv` (text, label, every probability and prediction), `{prefix}_probabilities.npz`, `{prefix}_metrics.json`, '
        f'`{prefix}_decision.json`, `weights/`. Total run time {time.perf_counter() - start_all:.0f} s.',
    ]
    (out_dir / f'{prefix}_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print('Outputs saved in:', out_dir.resolve(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval-split', default='holdout', choices=['holdout', 'val'])
    parser.add_argument('--fit-splits', default='train,val')
    parser.add_argument('--out-dir', default='results/holdout')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--official-seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--combination-threshold', type=float, default=None,
                        help='Dry runs only: score the combination at this threshold')
    parser.add_argument('--stage-a', default=None, help='stage_a_decision.json: scores the combination if rule 8 adopted it')
    parser.add_argument('--no-rerun-check', dest='rerun_check', action='store_false')
    parser.add_argument('--dry-run-epochs', type=int, default=None, help='Dry runs only: fewer CNN epochs (refused with the holdout)')
    main(parser.parse_args())
