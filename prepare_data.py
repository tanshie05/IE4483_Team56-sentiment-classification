"""IE4483 Project 1: reproducible data splits and representation preparation.
Run: python prepare_data.py --data-dir PATH_TO_INPUT_FILES --out-dir processed
Optional: --train-file train.json --test-file test.json
"""
import argparse
import hashlib
import html
import json
import platform
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

SEED = 42
VAL_FRACTION = HOLDOUT_FRACTION = .15
# Default sklearn tokens split "can't" into "can"; keep contractions, ! and ? as features.
TFIDF_TOKEN_PATTERN = r"(?u)\b\w\w+(?:'\w+)?\b|[!?]"
SENTIMENT_PROBE = ["can't", "won't", "isn't", "don't", 'not', 'no', 'never', '!', '?']

def clean_review(value):
    text = html.unescape(str(value))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(s):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]", s.lower())


def class_stats(labels):
    counts = labels.value_counts().sort_index()
    return {
        'rows': int(len(labels)),
        'negative': int(counts.get(0, 0)),
        'positive': int(counts.get(1, 0)),
        'positive_pct': round(100 * float(labels.mean()), 2),
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_train_only_tfidf(vectorizer, train_text):
    """Check document frequencies and IDF values reproduce from the training partition alone."""
    counter = CountVectorizer(vocabulary=vectorizer.vocabulary_, ngram_range=vectorizer.ngram_range,
                              token_pattern=vectorizer.token_pattern, binary=True)
    doc_freq = np.asarray(counter.transform(train_text).sum(axis=0)).ravel()
    expected_idf = np.log((1 + len(train_text)) / (1 + doc_freq)) + 1
    if doc_freq.min() < vectorizer.min_df or not np.allclose(vectorizer.idf_, expected_idf):
        raise ValueError('TF-IDF vectorizer does not match the training partition')


def write_summary(path, stats, overlap, baseline, assignments_counts, tfidf_shape, vocab_size, oov):
    lines = [
        '# Preprocessing summary', '',
        f"Seed {SEED}; 70/15/15 stratified split of deduplicated cleaned reviews. "
        'Holdout is reserved for final evaluation; do not use it or the official test set for tuning.', '',
        '## Counts and class distribution', '',
        '| Set | Rows | Negative | Positive | Positive % |', '|---|---:|---:|---:|---:|',
    ]
    for name, s in stats.items():
        lines.append(f"| {name} | {s['rows']} | {s['negative']} | {s['positive']} | {s['positive_pct']} |")
    lines += [
        '', f"Raw rows by status: {assignments_counts}.", '',
        '## Class imbalance and majority baseline', '',
        f"Negative:positive is 1:{baseline['imbalance_ratio']} in the raw training data. "
        f"Always predicting the training majority class ({baseline['majority_class']}) scores:", '',
    ]
    lines += [f"- {name}: {acc:.2%} accuracy" for name, acc in baseline['accuracy'].items()]
    lines += [
        '', '## Checks', '',
        '- Cleaned-text duplicates across train/val/holdout: 0 (enforced).',
        '- TF-IDF document frequencies and IDF values reproduce from the training partition (enforced).',
        '- Every CNN vocabulary token occurs at least twice in the training partition (enforced).',
        f"- Negation and punctuation probe kept by both tokenizers: {' '.join(SENTIMENT_PROBE)} (enforced).", '',
        '## Overlap with official test set (reported only; test data is never used for fitting or tuning)', '',
        f"- {overlap['test_rows_matching_labelled_data']} of {overlap['test_rows']} test rows "
        f"({overlap['unique_texts']} unique cleaned texts) also appear in labelled data.",
        '- Unique shared texts by split: ' + ', '.join(f'{k} {v}' for k, v in overlap['by_split'].items()) + '.', '',
        '## Features', '',
        f"- TF-IDF training matrix: {tfidf_shape[0]} x {tfidf_shape[1]}.",
        f"- CNN vocabulary: {vocab_size} entries (0=PAD, 1=UNK). OOV token rate: "
        + ', '.join(f'{k} {v:.2%}' for k, v in oov.items()) + '.', '',
        '## Decisions', '',
        '- Cleaning changes only HTML and whitespace. Case and punctuation stay in clean_text; no stopword removal or stemming.',
        '- Identical cleaned reviews keep their first occurrence so no text spans two splits.',
        '- Official test rows are all kept, including duplicates, because each needs a prediction.',
        '- TF-IDF keeps contractions and ! ? as tokens; the CNN tokenizer also keeps other punctuation.',
        '- Near-duplicates are not removed. Reviews appearing cut off in the source data cannot be repaired.', '',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def main(data_dir, out_dir, train_file, test_file):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = data_dir / train_file, data_dir / test_file
    train = pd.DataFrame(json.loads(train_path.read_text(encoding='utf-8')))
    test = pd.DataFrame(json.loads(test_path.read_text(encoding='utf-8')))
    assert {'reviews', 'sentiments'} <= set(train.columns)
    assert 'reviews' in test.columns
    assert train.sentiments.isin([0, 1]).all()

    train['original_index'] = train.index
    train['clean_text'] = train.reviews.fillna('').map(clean_review)
    test['clean_text'] = test.reviews.fillna('').map(clean_review)
    raw = train
    print('Raw training rows:', len(train), '| test rows:', len(test))
    print('Raw label counts:', train.sentiments.value_counts().sort_index().to_dict())
    print('Blank training reviews:', (train.clean_text == '').sum())
    print('Exact duplicated reviews:', train.reviews.duplicated().sum())
    print('Training / official test exact overlap:', len(set(train.reviews) & set(test.reviews)))

    # Remove blank reviews; group identical cleaned texts together to avoid
    # train/validation/holdout leakage. Conflicting labels must be investigated.
    train = train.loc[train.clean_text.ne('')].copy()
    conflict_count = (train.groupby('clean_text').sentiments.nunique() > 1).sum()
    if conflict_count:
        raise ValueError(f'{conflict_count} normalized texts have conflicting labels; inspect before splitting')
    unique = train.drop_duplicates('clean_text', keep='first').reset_index(drop=True)
    dev, holdout = train_test_split(unique, test_size=HOLDOUT_FRACTION, stratify=unique.sentiments, random_state=SEED)
    train_part, val_part = train_test_split(dev, test_size=VAL_FRACTION/(1 - HOLDOUT_FRACTION), stratify=dev.sentiments, random_state=SEED)
    splits = {'train': train_part, 'val': val_part, 'holdout': holdout}
    for name, df in splits.items():
        df = df.reset_index(drop=True)
        df[['reviews', 'clean_text', 'sentiments']].to_csv(out_dir / f'{name}.csv', index=False, lineterminator='\n')
        print(name, 'rows:', len(df), 'label counts:', df.sentiments.value_counts().sort_index().to_dict())
        splits[name] = df
    test[['reviews', 'clean_text']].to_csv(out_dir / 'test_prepared.csv', index=False, lineterminator='\n')

    names = list(splits)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            shared = set(splits[first].clean_text) & set(splits[second].clean_text)
            if shared:
                raise ValueError(f'{len(shared)} cleaned texts appear in both {first} and {second}')
    print('Cleaned-text duplicates across internal splits: 0')

    # Maps every raw row to its split, including duplicates dropped before splitting.
    kept = pd.concat([df.assign(split=name, split_row=df.index) for name, df in splits.items()])
    kept = kept[['clean_text', 'split', 'split_row', 'original_index']].rename(columns={'original_index': 'kept_original_index'})
    assignments = raw[['original_index', 'clean_text']].merge(kept, on='clean_text', how='left')
    assignments['status'] = np.where(assignments.split.isna(), 'blank_removed',
                                     np.where(assignments.original_index == assignments.kept_original_index, 'kept', 'duplicate_removed'))
    assignments = assignments.drop(columns='clean_text').astype({'split_row': 'Int64', 'kept_original_index': 'Int64'})
    assignments.to_csv(out_dir / 'split_assignments.csv', index=False, lineterminator='\n')
    status_counts = assignments.status.value_counts().to_dict()

    # Test data is compared here for reporting only; it never reaches a fit below.
    test_texts = set(test.clean_text)
    labelled_texts = set(unique.clean_text)
    overlap = {
        'test_rows': int(len(test)),
        'test_rows_matching_labelled_data': int(test.clean_text.isin(labelled_texts).sum()),
        'unique_texts': len(test_texts & labelled_texts),
        'by_split': {name: len(test_texts & set(df.clean_text)) for name, df in splits.items()},
    }
    print('Cleaned-text overlap with official test set:', overlap)

    majority = int(splits['train'].sentiments.mode()[0])
    counts = raw.sentiments.value_counts()
    baseline = {
        'majority_class': majority,
        'imbalance_ratio': round(float(counts.get(1, 0) / counts.get(0, 1)), 2),
        'accuracy': {name: float((df.sentiments == majority).mean()) for name, df in splits.items()},
    }
    print('Majority-class baseline:', baseline)

    # Fit on the TRAINING partition only. Do not fit on validation, holdout, or test.
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=10000, min_df=2, sublinear_tf=True,
                                 token_pattern=TFIDF_TOKEN_PATTERN)
    X_train = vectorizer.fit_transform(splits['train'].clean_text)
    joblib.dump(vectorizer, out_dir / 'tfidf_vectorizer.joblib')
    print('TF-IDF train matrix shape:', X_train.shape)
    verify_train_only_tfidf(vectorizer, splits['train'].clean_text)

    # CNN vocabulary also built on training only. Index 0=PAD, 1=UNK.
    counts = Counter(token for review in splits['train'].clean_text for token in tokenize(review))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for token, freq in counts.most_common():
        if freq < 2 or len(vocab) >= 20000:
            break
        vocab[token] = len(vocab)
    (out_dir / 'cnn_vocab.json').write_text(json.dumps(vocab, indent=2), encoding='utf-8', newline='\n')
    print('CNN vocabulary size:', len(vocab))
    if any(counts[token] < 2 for token in vocab if token not in ('<PAD>', '<UNK>')):
        raise ValueError('CNN vocabulary contains tokens not supported by the training partition')
    print('TF-IDF and CNN vocabulary fit on training partition only: verified')

    probe = clean_review(' '.join(SENTIMENT_PROBE))
    analyzer = vectorizer.build_analyzer()
    missing = [token for token in SENTIMENT_PROBE if token not in analyzer(probe) or token not in tokenize(probe)]
    if missing:
        raise ValueError(f'Preprocessing lost sentiment tokens: {missing}')

    oov = {}
    for name, df in {**splits, 'test': test}.items():
        tokens = [token for review in df.clean_text for token in tokenize(review)]
        oov[name] = sum(token not in vocab for token in tokens) / len(tokens)

    stats = {'raw_train': class_stats(raw.sentiments), 'deduplicated': class_stats(unique.sentiments)}
    stats.update({name: class_stats(df.sentiments) for name, df in splits.items()})
    metadata = {
        'seed': SEED,
        'val_fraction': VAL_FRACTION,
        'holdout_fraction': HOLDOUT_FRACTION,
        'inputs': {path.name: sha256(path) for path in (train_path, test_path)},
        'split_stats': stats,
        'raw_row_status': status_counts,
        'majority_baseline': baseline,
        'official_test_overlap': overlap,
        'tfidf_params': {key: value for key, value in vectorizer.get_params().items()
                         if key in ('lowercase', 'ngram_range', 'max_features', 'min_df', 'sublinear_tf', 'token_pattern')},
        'tfidf_shape': list(X_train.shape),
        'cnn_vocab_size': len(vocab),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                     'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__, 'joblib': joblib.__version__},
    }
    (out_dir / 'split_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8', newline='\n')
    write_summary(out_dir / 'preprocessing_summary.md', stats, overlap, baseline, status_counts, X_train.shape, len(vocab), oov)
    print('Outputs saved in:', out_dir.resolve())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='.')
    parser.add_argument('--out-dir', default='processed')
    parser.add_argument('--train-file', default='train.json')
    parser.add_argument('--test-file', default='test.json')
    args = parser.parse_args()
    main(args.data_dir, args.out_dir, args.train_file, args.test_file)
