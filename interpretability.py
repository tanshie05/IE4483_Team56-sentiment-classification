"""IE4483 Project 1: interpretability of the saved models (CPU, read-only; no training, nothing existing is overwritten).

Run from the project root:  python interpretability.py [--out-dir results/interpretability]
(The script finds the project root itself, so it also runs from results/interpretability/.)

1. Baseline: top 20 positive and top 20 negative TF-IDF words by logistic-regression weight (CSV + bar chart of 15 each).
2. CNN: 12 filters (2 most positive and 2 most negative output weights per kernel size 3/4/5) and, for each, the 5
   distinct n-grams from the final model's training reviews (train + val + holdout, 7,182) that activate it most.
3. Examples (holdout rows 579, 898, 639, 439, 962, 16 and test row 456): for the CNN, each filter's contribution to the
   logit is (max-pooled activation) x (its output weight), located at the n-gram where the max-pool fired; contributions
   of filters that fired on the same n-gram are summed. For the baseline, each word's contribution is its TF-IDF value
   x its weight. Done for the FINAL models (fitted on train + val + holdout, so holdout rows were seen in training)
   and, where the files exist, for the holdout-stage models used in the report's error analysis (fitted on train + val,
   CNN seed 42), whose probabilities match results/holdout/holdout_predictions.csv.
Probabilities are recomputed and checked against the saved ones (CPU vs GPU differences up to about 1e-3 are allowed).
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in [HERE, *HERE.parents] if (p / 'final_utils.py').exists())
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import final_utils as fu  # noqa: E402
from prepare_data import clean_review, tokenize  # noqa: E402

HOLDOUT_ROWS = [579, 898, 639, 439, 962, 16]
TEST_ROWS = [456]
POSITIVE, NEGATIVE = '#2a78d6', '#e34948'  # diverging pair (blue = pushes positive, red = pushes negative)
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e1e0d9'
TOLERANCE = 1e-3


def load_cnn(weights, settings, vocab):
    return fu.build_cnn(settings, len(vocab), torch.load(weights, map_location='cpu', weights_only=True), 'cpu')


def tokens_of(text, max_len):
    return tokenize(text)[:max_len]


def conv_activations(model, ids):
    """ReLU(conv) outputs per kernel size: list of arrays (batch, filters, positions)."""
    with torch.no_grad():
        embedded = model.embedding(ids).transpose(1, 2)
        return [torch.relu(conv(embedded)).numpy() for conv in model.convs]


def ngram(tokens, start, size):
    return ' '.join(tokens[start:start + size])


# ---------------------------------------------------------------- baseline
def baseline_words(pipeline, n=20):
    names = pipeline.named_steps['tfidf'].get_feature_names_out()
    coef = pipeline.named_steps['clf'].coef_[0]
    order = np.argsort(coef)
    rows = [{'direction': 'positive', 'rank': r + 1, 'word': names[i], 'weight': coef[i]} for r, i in enumerate(order[::-1][:n])]
    rows += [{'direction': 'negative', 'rank': r + 1, 'word': names[i], 'weight': coef[i]} for r, i in enumerate(order[:n])]
    return pd.DataFrame(rows)


def baseline_chart(words, path, n=15):
    neg = words[words.direction == 'negative'].head(n).iloc[::-1]
    pos = words[words.direction == 'positive'].head(n).iloc[::-1]
    limit = max(abs(words.weight).max(), 1) * 1.08
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), sharex=False)
    for ax, frame, color, title in ((axes[0], neg, NEGATIVE, f'Top {n} words pushing NEGATIVE'),
                                    (axes[1], pos, POSITIVE, f'Top {n} words pushing POSITIVE')):
        ax.barh(range(len(frame)), frame.weight, color=color, height=0.72, edgecolor='white', linewidth=2)
        ax.set_yticks(range(len(frame)), frame.word, fontsize=9.5, color=INK)
        ax.set_xlim((-limit, 0) if color == NEGATIVE else (0, limit))
        ax.set_title(title, fontsize=11, color=INK, loc='left')
        ax.set_xlabel('Logistic-regression weight', fontsize=9, color=MUTED)
        ax.grid(axis='x', color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.tick_params(axis='both', colors=MUTED, length=0)
        for y, value in enumerate(frame.weight):
            ax.text(value + (0.08 if value > 0 else -0.08), y, f'{value:.2f}', va='center', ha='left' if value > 0 else 'right', fontsize=8, color=MUTED)
    axes[0].yaxis.tick_left()
    fig.suptitle('Baseline (TF-IDF + logistic regression, final model): the words with the largest weights',
                 fontsize=12, color=INK, x=0.01, ha='left')
    fig.text(0.01, 0.01, 'A word pushes towards its class in proportion to weight x TF-IDF value. Fitted on 7,182 reviews (train + val + holdout).',
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def baseline_contributions(pipeline, text, n=6):
    tfidf, clf = pipeline.named_steps['tfidf'], pipeline.named_steps['clf']
    x = tfidf.transform([text]).tocoo()
    names = tfidf.get_feature_names_out()
    contrib = pd.DataFrame({'word': names[x.col], 'tfidf': x.data, 'weight': clf.coef_[0][x.col]})
    contrib['contribution'] = contrib.tfidf * contrib.weight
    contrib = contrib.sort_values('contribution')
    top = pd.concat([contrib.tail(n).iloc[::-1].assign(direction='positive'), contrib.head(n).assign(direction='negative')])
    return top[top.contribution.abs() > 0], float(clf.intercept_[0])


# ---------------------------------------------------------------- CNN
def select_filters(model, settings, per_side=2):
    weights = model.classifier.weight.detach().numpy()[0]
    n = settings['num_filters']
    chosen = []
    for k_index, size in enumerate(settings['kernel_sizes']):
        block = weights[k_index * n:(k_index + 1) * n]
        order = np.argsort(block)
        for direction, picks in (('positive', order[::-1][:per_side]), ('negative', order[:per_side])):
            chosen += [{'kernel_size': size, 'k_index': k_index, 'filter': int(f), 'feature': k_index * n + int(f),
                        'output_weight': float(block[f]), 'direction': direction} for f in picks]
    return chosen


def filter_top_ngrams(model, settings, vocab, texts, chosen, top=5, batch=256):
    max_len = settings['max_len']
    token_lists = [tokens_of(t, max_len) for t in texts]
    ids = fu.encode_texts(texts, vocab, settings)
    best = {i: {} for i in range(len(chosen))}  # n-gram -> [max activation, reviews where it is the filter's max]
    for start in range(0, len(texts), batch):
        acts = conv_activations(model, ids[start:start + batch])
        for i, spec in enumerate(chosen):
            a = acts[spec['k_index']][:, spec['filter'], :]
            for row in range(a.shape[0]):
                length = len(token_lists[start + row])
                valid = max(length - spec['kernel_size'] + 1, 1)  # windows that start on a real token
                pos = int(np.argmax(a[row, :valid]))
                value = float(a[row, pos])
                if value <= 0:
                    continue
                gram = ngram(token_lists[start + row], pos, spec['kernel_size'])
                entry = best[i].setdefault(gram, [0.0, 0])
                entry[0] = max(entry[0], value)
                entry[1] += 1
    rows = []
    for i, spec in enumerate(chosen):
        ranked = sorted(best[i].items(), key=lambda item: -item[1][0])[:top]
        for rank, (gram, (value, count)) in enumerate(ranked, 1):
            rows.append({'kernel_size': spec['kernel_size'], 'filter': spec['filter'], 'output_weight': spec['output_weight'],
                         'direction': spec['direction'], 'rank': rank, 'ngram': gram, 'max_activation': value,
                         'contribution_to_logit': value * spec['output_weight'], 'reviews_where_top_ngram': count})
    return pd.DataFrame(rows)


def cnn_contributions(model, settings, vocab, text, n=6):
    tokens = tokens_of(text, settings['max_len'])
    ids = fu.encode_texts([text], vocab, settings)
    acts = conv_activations(model, ids)
    weights = model.classifier.weight.detach().numpy()[0]
    bias = float(model.classifier.bias.detach().numpy()[0])
    nf = settings['num_filters']
    spans = {}
    logit = bias
    for k_index, size in enumerate(settings['kernel_sizes']):
        a = acts[k_index][0]  # (filters, positions)
        positions = a.argmax(axis=1)  # same choice as max-pool (first maximum)
        for f in range(nf):
            value = float(a[f, positions[f]])
            contribution = value * float(weights[k_index * nf + f])
            logit += contribution
            if value <= 0:
                continue
            pos = int(positions[f])
            gram = ngram(tokens, pos, size) or '<padding>'
            key = (pos, size, gram)
            entry = spans.setdefault(key, [0.0, 0])
            entry[0] += contribution
            entry[1] += 1
    frame = pd.DataFrame([{'start_token': k[0], 'kernel_size': k[1], 'ngram': k[2], 'contribution': v[0], 'filters': v[1]}
                          for k, v in spans.items()]).sort_values('contribution')
    top = pd.concat([frame.tail(n).iloc[::-1].assign(direction='positive'), frame.head(n).assign(direction='negative')])
    return top, logit, bias


def explain(model_set, cnn, pipeline, settings, vocab, examples, saved):
    cnn_rows, base_rows, summary = [], [], []
    for split, row, text, label in examples:
        top, logit, bias = cnn_contributions(cnn, settings, vocab, text)
        p_cnn = float(1 / (1 + np.exp(-logit)))
        btop, intercept = baseline_contributions(pipeline, text)
        p_base = float(pipeline.predict_proba([text])[0, 1])
        ref = saved.get((split, row))
        if ref is not None and (abs(ref[0] - p_cnn) > TOLERANCE or abs(ref[1] - p_base) > TOLERANCE):
            raise SystemExit(f'{model_set} {split} row {row}: recomputed probabilities {p_cnn:.4f}/{p_base:.4f} differ from saved {ref}')
        common = {'models': model_set, 'split': split, 'row': row, 'label': label}
        cnn_rows += [{**common, 'cnn_prob': p_cnn, 'logit': logit, 'bias': bias, **r} for r in top.to_dict('records')]
        base_rows += [{**common, 'baseline_prob': p_base, 'intercept': intercept, **r} for r in btop.to_dict('records')]
        summary.append({**common, 'cnn_prob': round(p_cnn, 4), 'baseline_prob': round(p_base, 4),
                        'combination_prob': round((p_cnn + p_base) / 2, 4),
                        'saved_cnn_prob': None if ref is None else round(ref[0], 4), 'saved_baseline_prob': None if ref is None else round(ref[1], 4)})
    return cnn_rows, base_rows, summary


def main(out_dir):
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    out_dir.mkdir(parents=True, exist_ok=True)
    final = json.loads(Path('results/final/final_config.json').read_text(encoding='utf-8'))
    settings = {**fu.cnn_settings(), **{k: final['cnn'][k] for k in ('max_len', 'num_filters', 'kernel_sizes', 'embedding_dim', 'dropout', 'batch_size')}}
    vocab = fu.load_vocab(settings)
    cnn = load_cnn(Path('results/final') / final['cnn']['weights'], settings, vocab)
    pipeline = joblib.load(Path('results/final') / final['baseline']['pipeline'])

    words = baseline_words(pipeline)
    words.to_csv(out_dir / 'baseline_top_words.csv', index=False, lineterminator='\n')
    baseline_chart(words, out_dir / 'baseline_top_words.png')

    fitted = fu.load_labelled(final['fitted_on'])
    if len(fitted) != final['fit_rows']:
        raise SystemExit('Fitted reviews do not match final_config.json')
    chosen = select_filters(cnn, settings)
    filters = filter_top_ngrams(cnn, settings, vocab, fitted.clean_text.tolist(), chosen)
    filters.to_csv(out_dir / 'cnn_filter_ngrams.csv', index=False, lineterminator='\n')

    holdout = fu.load_labelled(['holdout']).reset_index(drop=True)
    test = pd.DataFrame(json.loads(Path('test.json').read_text(encoding='utf-8')))
    examples = [('holdout', r, holdout.clean_text[r], int(holdout.sentiments[r])) for r in HOLDOUT_ROWS]
    examples += [('test', r, clean_review(str(test.reviews[r])), None) for r in TEST_ROWS]

    sub = pd.read_csv('results/final/submission_probabilities.csv', keep_default_na=False)
    saved_final = {('test', r): (sub.cnn_prob[r], sub.baseline_prob[r]) for r in TEST_ROWS}
    cnn_rows, base_rows, summary = explain('final', cnn, pipeline, settings, vocab, examples, saved_final)

    hold_cnn, hold_base = Path('results/holdout/weights/cnn_train_val_seed42.pt'), Path('results/holdout/weights/baseline_train_val.joblib')
    if hold_cnn.exists() and hold_base.exists():
        preds = pd.read_csv('results/holdout/holdout_predictions.csv', keep_default_na=False).set_index('holdout_row')
        saved_hold = {('holdout', r): (preds.cnn_seed42_prob[r], preds.baseline_prob[r]) for r in HOLDOUT_ROWS}
        rows = explain('holdout_stage', load_cnn(hold_cnn, settings, vocab), joblib.load(hold_base), settings, vocab,
                       [e for e in examples if e[0] == 'holdout'], saved_hold)
        cnn_rows, base_rows, summary = cnn_rows + rows[0], base_rows + rows[1], summary + rows[2]
    else:
        print('Holdout-stage weights not found (not in the public repository); explaining the final models only.')

    pd.DataFrame(cnn_rows).to_csv(out_dir / 'example_cnn_ngrams.csv', index=False, lineterminator='\n')
    pd.DataFrame(base_rows).to_csv(out_dir / 'example_baseline_words.csv', index=False, lineterminator='\n')
    pd.DataFrame(summary).to_csv(out_dir / 'example_probabilities.csv', index=False, lineterminator='\n')
    print(pd.DataFrame(summary).to_string(index=False))
    print('Saved:', sorted(p.name for p in out_dir.iterdir()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=Path('results/interpretability'))
    main(parser.parse_args().out_dir)
