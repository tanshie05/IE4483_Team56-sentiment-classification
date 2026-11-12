"""IE4483 Project 1: example candidates for the error analysis (rubric e).

Holdout groups:  python error_candidates.py holdout [--predictions results/holdout/holdout_predictions.csv] [--out results/holdout/error_candidates.md]
Test examples:   python error_candidates.py test [--probabilities results/final/submission_probabilities.csv] [--out results/final/test_examples.md]

Holdout: four groups by the official CNN (seed 42) and the rule-6 baseline (both fitted on train + val, each at its own
threshold): both right, both wrong, CNN right / baseline wrong, baseline right / CNN wrong; up to 8 reviews each.
Test: 3 confident and 3 borderline reviews for the submission model, with both models' probabilities (no labels).

The flags are keyword heuristics meant to point the reader at candidates; every flag must be confirmed by reading the review.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from prepare_data import tokenize

NEGATION = re.compile(r"\b(not|no|never|nothing|none|nobody|nowhere|neither|nor|cannot|without|hardly|barely)\b|n't\b", re.I)
CONTRAST = re.compile(r"\b(but|however|although|though|except|yet|otherwise|unfortunately|on the other hand)\b", re.I)
POSITIVE = re.compile(r"\b(great|good|love|loved|excellent|perfect|amazing|awesome|wonderful|best|nice|happy|recommend|fantastic|works well)\b", re.I)
NEGATIVE = re.compile(r"\b(bad|poor|terrible|awful|worst|broke|broken|disappoint\w*|return\w*|waste|useless|cheap|problem\w*|junk|refund)\b", re.I)
SARCASM = re.compile(r"\b(yeah right|thanks a lot|oh great|just great|so much for|what a joke|supposedly|real winner|thanks for nothing)\b", re.I)
LONG_TOKENS = 166  # 95th percentile of CNN token counts in the train partition (results/length_coverage/length_coverage.md)
MAX_LEN = 300


def flags(text, label=None):
    found = []
    if NEGATION.search(text):
        found.append('negation')
    positive, negative = bool(POSITIVE.search(text)), bool(NEGATIVE.search(text))
    if CONTRAST.search(text) or (positive and negative):
        found.append('mixed sentiment')
    # Possible sarcasm: an ironic phrase, or praise words (and no complaint words) in a review labelled negative
    # (for unlabelled test reviews, pass the predicted label). Exclamation marks alone are not counted: they are common in
    # sincere praise.
    if SARCASM.search(text) or (label == 0 and positive and not negative):
        found.append('possible sarcasm')
    n = len(tokenize(text))
    if n > MAX_LEN:
        found.append(f'long ({n} tokens, truncated at {MAX_LEN})')
    elif n > LONG_TOKENS:
        found.append(f'long ({n} tokens)')
    return found


def quote(text, limit=700):
    text = ' '.join(str(text).split())
    return text if len(text) <= limit else text[:limit] + f' […] ({len(text):,} characters in full)'


def pick(group, n, kind):
    """Deterministic choice. Wrong groups: most confident mistakes first. Right group: half negatives, confident and borderline."""
    if kind == 'both_right':
        neg, pos = group[group.sentiments == 0], group[group.sentiments == 1]
        chosen = [neg.sort_values('margin', ascending=False).head(2), neg.sort_values('margin').head(2),
                  pos.sort_values('margin', ascending=False).head(2), pos.sort_values('margin').head(2)]
        return pd.concat(chosen).drop_duplicates('row').head(n)
    return group.sort_values('wrongness', ascending=False).head(n)


def holdout(args):
    frame = pd.read_csv(args.predictions, keep_default_na=False)
    decision = json.loads(Path(args.predictions).with_name(Path(args.predictions).name.replace('_predictions.csv', '_decision.json')).read_text(encoding='utf-8'))
    seed = decision['official_cnn_seed']
    cnn_t, base_t = decision['cnn_threshold'], decision['baseline_threshold']
    frame = frame.rename(columns={frame.columns[0]: 'row'})
    cnn_p, base_p = frame[f'cnn_seed{seed}_prob'], frame['baseline_prob']
    cnn_ok = frame[f'cnn_seed{seed}_pred'] == frame.sentiments
    base_ok = frame['baseline_pred'] == frame.sentiments
    # Distance from each model's own threshold, in the direction of the true label (positive = right side).
    sign = frame.sentiments.map({1: 1, 0: -1})
    frame['cnn_margin'] = sign * (cnn_p - cnn_t)
    frame['base_margin'] = sign * (base_p - base_t)
    groups = {
        'both_right': ('Both models right', cnn_ok & base_ok, lambda f: f.assign(margin=f[['cnn_margin', 'base_margin']].min(axis=1))),
        'both_wrong': ('Both models wrong', ~cnn_ok & ~base_ok, lambda f: f.assign(wrongness=-(f.cnn_margin + f.base_margin))),
        'cnn_only': ('CNN right, baseline wrong', cnn_ok & ~base_ok, lambda f: f.assign(wrongness=-f.base_margin + f.cnn_margin)),
        'baseline_only': ('Baseline right, CNN wrong', ~cnn_ok & base_ok, lambda f: f.assign(wrongness=-f.cnn_margin + f.base_margin)),
    }
    split = decision['eval_split']
    lines = [f'# Error-analysis candidates ({split})', '',
             f'Models: official CNN (seed {seed}, threshold {cnn_t}) and baseline exp04 (rule 6, threshold {base_t}), both fitted on '
             f"{'+'.join(decision['fit_splits'])}; {len(frame):,} {split} reviews. P(pos) = predicted probability of positive; a model predicts "
             'positive when P(pos) ≥ its threshold. Label 1 = positive, 0 = negative.', '',
             'Wrong groups list the most confident mistakes first. The right group shows 2 confident and 2 borderline reviews of each class. '
             'Flags are keyword heuristics (negation words; contrast words or both praise and complaint words; an ironic phrase, or praise words without complaint words in a '
             f'negative review; more than {LONG_TOKENS} tokens = 95th percentile, more than {MAX_LEN} = truncated). Read each review before using a flag. '
             'Confident mistakes by both models are also candidates for label noise (rubric h).', '',
             '## Group sizes', '', '| Group | Reviews | Of which negative | Share of all |', '|---|---|---|---|']
    for key, (title, mask, _) in groups.items():
        lines.append(f'| {title} | {int(mask.sum())} | {int((frame.sentiments[mask] == 0).sum())} | {100 * mask.mean():.1f}% |')
    lines.append('')
    header, lines, flag_rows = lines, [], []
    for key, (title, mask, scorer) in groups.items():
        group = scorer(frame[mask].copy())
        counts = {name: 0 for name in ('negation', 'mixed sentiment', 'possible sarcasm', 'long')}
        for text, label in zip(group.reviews, group.sentiments):
            for f in flags(text, label):
                counts[f.split(' (')[0]] += 1
        flag_rows.append(f"| {title} | {len(group)} | " + ' | '.join(f'{counts[k]} ({100 * counts[k] / max(len(group), 1):.0f}%)' for k in counts) + ' |')
        lines += [f'## {title} ({int(mask.sum())} reviews; up to 8 shown)', '']
        chosen = pick(group, 8, key)
        if chosen.empty:
            lines += ['None.', '']
        for i, item in enumerate(chosen.itertuples(), 1):
            f = flags(item.reviews, item.sentiments)
            lines += [f'**{key.upper()}-{i}** · {split} row {item.row} · true label **{item.sentiments}** · CNN P(pos) {getattr(item, f"cnn_seed{seed}_prob"):.3f} → '
                      f'{getattr(item, f"cnn_seed{seed}_pred")} · baseline P(pos) {item.baseline_prob:.3f} → {item.baseline_pred} · {item.n_tokens} tokens · '
                      f"flags: {', '.join(f) if f else 'none'}", '', f'> {quote(item.reviews)}', '']
    lines = header + ['## How often each flag appears per group (all reviews in the group)', '',
                      '| Group | Reviews | Negation | Mixed sentiment | Possible sarcasm | Long |', '|---|---|---|---|---|---|', *flag_rows, ''] + lines
    Path(args.out).write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print('Wrote', args.out, {title: int(mask.sum()) for title, mask, _ in groups.values()})


def test(args):
    frame = pd.read_csv(args.probabilities, keep_default_na=False)
    config = json.loads((Path(args.final_dir) / 'final_config.json').read_text(encoding='utf-8'))
    chosen = config['submission_model']
    threshold = config['cnn']['threshold'] if chosen == 'cnn' else config['baseline']['threshold'] if chosen == 'baseline' else config['combination_threshold']
    p = frame[f'{chosen}_prob']
    frame['test_row'] = range(len(frame))
    frame['distance'] = (p - threshold).abs()
    confident = pd.concat([frame[frame.prediction == 0].sort_values('distance', ascending=False).head(1),
                           frame[frame.prediction == 1].sort_values('distance', ascending=False).head(2)])
    # Borderline: closest to the threshold; skip very short reviews so each example has something to discuss.
    borderline = frame[frame.reviews.str.split().str.len() >= 8].sort_values('distance').head(3)
    lines = ['# Test-set examples for the error analysis (no labels)', '',
             f"Submission model: {chosen} (threshold {threshold}), fitted on {'+'.join(config['fitted_on'])} ({config['fit_rows']:,} reviews). "
             f"CNN threshold {config['cnn']['threshold']}, baseline threshold {config['baseline']['threshold']}. Test labels are unknown, so these show "
             'what the model is sure or unsure about, not whether it is right.', '',
             f"Predicted test class balance: {int(frame.prediction.sum()):,} positive ({100 * frame.prediction.mean():.1f}%), "
             f"{int((frame.prediction == 0).sum()):,} negative. CNN and baseline agree on {100 * (frame.cnn_pred == frame.baseline_pred).mean():.1f}% of test reviews.", '']
    for title, chosen_rows in (('Confident (farthest from the threshold)', confident), ('Borderline (closest to the threshold)', borderline)):
        lines += [f'## {title}', '']
        for item in chosen_rows.itertuples():
            f = flags(item.reviews, item.prediction)
            lines += [f'**Test row {item.test_row}** · submitted prediction **{item.prediction}** · CNN P(pos) {item.cnn_prob:.3f} → {item.cnn_pred} · '
                      f"baseline P(pos) {item.baseline_prob:.3f} → {item.baseline_pred} · flags: {', '.join(f) if f else 'none'}", '', f'> {quote(item.reviews)}', '']
    Path(args.out).write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print('Wrote', args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)
    h = sub.add_parser('holdout')
    h.add_argument('--predictions', default='results/holdout/holdout_predictions.csv')
    h.add_argument('--out', default='results/holdout/error_candidates.md')
    t = sub.add_parser('test')
    t.add_argument('--probabilities', default='results/final/submission_probabilities.csv')
    t.add_argument('--final-dir', default='results/final')
    t.add_argument('--out', default='results/final/test_examples.md')
    args = parser.parse_args()
    holdout(args) if args.mode == 'holdout' else test(args)
