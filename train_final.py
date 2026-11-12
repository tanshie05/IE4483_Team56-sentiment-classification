"""IE4483 Project 1: final retraining (selection rule 7).

Official run:  python train_final.py                      # train + val + holdout -> results/final/
Dry run:       python train_final.py --fit-splits train --decision <dry-run decision.json> --out-dir <scratch folder>

Retrains, with the recipe and nothing re-tuned:
  - the CNN: final config, recipe epochs (7, no early stopping), seed 42, threshold 0.66;
  - the baseline: exp04 settings, rule-6 threshold (0.45).
Both are saved; the decision file from evaluate_holdout.py names the submission model (only it fills submission.csv;
the other gives test probabilities for the error analysis). Official test data is never loaded here.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import torch

import final_utils as fu
from train_cnn import run_environment


def main(args):
    device = fu.setup_deterministic(args.device)
    environment = run_environment(device)
    decision = json.loads(Path(args.decision).read_text(encoding='utf-8'))
    if decision['chosen'] == 'combination' and decision.get('combination_threshold') is None:
        raise ValueError('The decision names the combination but gives no combination threshold')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_splits = args.fit_splits.split(',')
    cnn, base = fu.cnn_settings(), fu.baseline_settings()
    if args.dry_run_epochs is not None:
        if 'holdout' in fit_splits:
            raise SystemExit('--dry-run-epochs is for dry runs only; the final model always uses the recipe epochs')
        cnn['epochs'] = args.dry_run_epochs
    if cnn['threshold'] != decision['cnn_threshold'] or base['threshold'] != decision['baseline_threshold']:
        raise ValueError('Thresholds differ from the ones used in the holdout decision')
    fit = fu.load_labelled(fit_splits)
    labels = fit.sentiments.to_numpy()
    print('Environment:', environment, flush=True)
    print(f'Fitting on {"+".join(fit_splits)}: {len(fit)} reviews, {int((labels == 0).sum())} negative. Submission model: {decision["chosen"]}', flush=True)

    vocab = fu.load_vocab(cnn)
    glove = fu.load_glove_init(cnn, vocab)
    tokens = fu.encode_texts(fit.clean_text, vocab, cnn)
    start = time.perf_counter()
    model, losses = fu.train_cnn_fixed(args.seed, cnn, tokens, labels, len(vocab), device, glove)
    cnn_seconds = time.perf_counter() - start
    torch.save(model.state_dict(), out_dir / 'cnn_final.pt')
    rerun = None
    if args.rerun_check:
        again, _ = fu.train_cnn_fixed(args.seed, cnn, tokens, labels, len(vocab), device, glove, log=lambda _message: None)
        rerun = all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), again.state_dict().values()))
        print('Rerun gives identical CNN weights:', rerun, flush=True)
        if not rerun:
            raise RuntimeError('Determinism check failed for the final CNN')

    baseline = fu.make_baseline(base)
    start = time.perf_counter()
    baseline.fit(fit.clean_text, labels)
    baseline_seconds = time.perf_counter() - start
    if baseline.named_steps['clf'].n_iter_[0] >= base['logistic_regression']['max_iter']:
        raise RuntimeError('The final baseline hit max_iter')
    joblib.dump(baseline, out_dir / 'baseline_final.joblib')

    config = {
        'submission_model': decision['chosen'], 'decision_file': str(args.decision), 'decision_reason': decision['reason'],
        'fitted_on': fit_splits, 'fit_rows': len(fit), 'fit_negatives': int((labels == 0).sum()),
        'cnn': {**cnn, 'seed': args.seed, 'weights': 'cnn_final.pt', 'vocab_size': len(vocab), 'train_losses': losses,
                'train_seconds': cnn_seconds, 'rerun_identical': rerun},
        'baseline': {**base, 'pipeline': 'baseline_final.joblib', 'train_seconds': baseline_seconds},
        'combination_threshold': decision.get('combination_threshold'),
        'threshold_rule': 'predict 1 (positive) when P(positive) >= threshold', 'environment': environment,
    }
    (out_dir / 'final_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8', newline='\n')
    print(f'CNN {cnn_seconds:.1f} s, baseline {baseline_seconds:.2f} s. Saved in {out_dir.resolve()}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fit-splits', default='train,val,holdout')
    parser.add_argument('--decision', default='results/holdout/holdout_decision.json')
    parser.add_argument('--out-dir', default='results/final')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--no-rerun-check', dest='rerun_check', action='store_false')
    parser.add_argument('--dry-run-epochs', type=int, default=None, help='Dry runs only: fewer CNN epochs (refused with the holdout)')
    main(parser.parse_args())
