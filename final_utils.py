"""IE4483 Project 1: shared helpers for the holdout evaluation, final retraining and prediction (rules 5-7).

The CNN training loop repeats train_cnn.train_seed step for step (seeding, model creation, GloVe initialisation,
optimiser, seeded DataLoader shuffle), but runs a fixed number of epochs with no validation set and no early stopping,
as the retraining recipe requires. train_seed's per-epoch validation scoring runs in eval mode without gradients and
uses no random numbers, so leaving it out does not change the trained weights.
"""
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # Required for deterministic CUBLAS on a GPU.

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eval_utils import threshold_metrics
from train_baseline import build_pipeline
from train_cnn import TextCNN, encode, glove_matrix, glove_sources, load_glove, seed_everything, seed_worker

# IE4483_REPRO_DIR (set by run_all.sh) points every input at a reproduction folder instead of processed/ and results/.
# A reproduction folder has the same layout as results/ (baseline/, cnn/, ...) plus its own processed/.
_REPRO = os.environ.get('IE4483_REPRO_DIR')
PROCESSED = Path(_REPRO) / 'processed' if _REPRO else Path('processed')
_RESULTS = Path(_REPRO) if _REPRO else Path('results')
_CNN = _RESULTS / 'cnn'
_BASE = _RESULTS / 'baseline'
REFERENCE_CONFIG = _CNN / 'imbalance_checkpoint/unweighted_val_loss_config.json'
RECIPE = _CNN / 'cv/cv_recipe.json'
# The baseline's SETTINGS are fixed as exp04 and never re-selected: a reproduction re-runs the grid and, on other hardware,
# can select a different tied config (exp18 on the cluster). So the settings and the saved laptop pipeline always come from
# results/; only the nested-CV fold thresholds come from the reproduction's own baseline_cv.csv (same-hardware recompute).
BASELINE_CONFIG = Path('results/baseline/best_baseline_config.json')
SAVED_BASELINE = Path('results/baseline/best_baseline_pipeline.joblib')
BASELINE_CV = _BASE / 'baseline_cv.csv'
CNN_KEYS = ['vocab_path', 'embedding_init', 'freeze_embedding', 'embedding_dim', 'max_len', 'kernel_sizes', 'num_filters',
            'dropout', 'lr', 'batch_size', 'pos_weight', 'glove_path']
NGRAM_NAMES = {(1, 1): 'unigram', (1, 2): 'unigram+bigram'}


def cnn_settings():
    """Final CNN config (reference run of the 2x2, unchanged by every sweep) plus the CV retraining recipe."""
    reference = json.loads(REFERENCE_CONFIG.read_text(encoding='utf-8'))
    recipe = json.loads(RECIPE.read_text(encoding='utf-8'))
    settings = {key: reference[key] for key in CNN_KEYS}
    for key, value in recipe['settings'].items():
        if settings[key] != value:
            raise ValueError(f'Recipe setting {key}={value} differs from the reference config ({settings[key]})')
    if float(recipe['pos_weight']) != float(settings['pos_weight']) or recipe['checkpoint_metric'] != 'val_loss':
        raise ValueError('The recipe does not match the final config (pos_weight 1, checkpoint by validation loss)')
    settings.update(epochs=int(recipe['epochs']), threshold=float(recipe['threshold']))
    return settings


def baseline_cv_table():
    """The nested-CV table that holds exp04's folds. In a reproduction whose grid did not shortlist exp04 (the grid can
    select different tied configs on other hardware), exp04's nested CV is computed here with train_baseline's own
    cross_validate_config (same folds and seed) and saved next to the reproduction's baseline_cv.csv."""
    config = json.loads(BASELINE_CONFIG.read_text(encoding='utf-8'))
    cv = pd.read_csv(BASELINE_CV)
    if _REPRO and not (cv.experiment == config['experiment']).any():
        path = BASELINE_CV.with_name(f"baseline_cv_{config['experiment']}.csv")
        if not path.exists():
            from train_baseline import cross_validate_config
            tfidf, lr = config['tfidf'], config['logistic_regression']
            grid_config = (NGRAM_NAMES[tuple(tfidf['ngram_range'])], tfidf['max_features'], lr['C'], lr['class_weight'])
            print(f"{config['experiment']} is not in {BASELINE_CV}; computing its nested CV -> {path}", flush=True)
            cross_validate_config(config['experiment'], grid_config, load_labelled(['train', 'val'])).to_csv(path, index=False, lineterminator='\n')
        cv = pd.read_csv(path)
    return cv


def baseline_settings():
    """exp04 settings (never re-selected) and the rule-6 threshold: median of exp04's nested-CV inner thresholds."""
    config = json.loads(BASELINE_CONFIG.read_text(encoding='utf-8'))
    cv = baseline_cv_table()
    folds = cv[(cv.experiment == config['experiment']) & cv.fold.astype(str).str.isdigit()]
    if len(folds) != 5:
        raise ValueError(f"Expected 5 nested-CV folds for {config['experiment']} in {BASELINE_CV}, found {len(folds)}")
    thresholds = folds.inner_threshold.astype(float)
    if not _REPRO and abs(thresholds.mean() - config['cv']['inner_threshold']['mean']) > 1e-9:
        raise ValueError('baseline_cv.csv inner thresholds do not match best_baseline_config.json')
    return {'experiment': config['experiment'], 'tfidf': config['tfidf'], 'logistic_regression': config['logistic_regression'],
            'threshold': round(float(np.median(thresholds)), 2), 'fold_thresholds': thresholds.round(2).tolist(),
            'saved_threshold': float(config['threshold'])}


def make_baseline(settings):
    tfidf, lr = settings['tfidf'], settings['logistic_regression']
    pipeline = build_pipeline(NGRAM_NAMES[tuple(tfidf['ngram_range'])], tfidf['max_features'], lr['C'], lr['class_weight'])
    # build_pipeline must reproduce exp04 exactly; anything else would re-select the baseline.
    vectorizer, classifier = pipeline.named_steps['tfidf'], pipeline.named_steps['clf']
    expected = {'lowercase': tfidf['lowercase'], 'min_df': tfidf['min_df'], 'sublinear_tf': tfidf['sublinear_tf'],
                'token_pattern': tfidf['token_pattern'], 'max_features': tfidf['max_features']}
    actual = {key: getattr(vectorizer, key) for key in expected}
    if actual != expected or classifier.max_iter != lr['max_iter'] or classifier.random_state != lr['random_state']:
        raise ValueError(f'build_pipeline does not reproduce exp04: {actual} vs {expected}')
    return pipeline


def load_labelled(names):
    """Concatenates processed partitions in the given order (train, val, holdout)."""
    metadata = json.loads((PROCESSED / 'split_metadata.json').read_text(encoding='utf-8'))
    frames = []
    for name in names:
        frame = pd.read_csv(PROCESSED / f'{name}.csv', keep_default_na=False)
        if len(frame) != metadata['split_stats'][name]['rows']:
            raise ValueError(f'{name}.csv does not match split_metadata.json')
        frames.append(frame.assign(split=name))
    frame = pd.concat(frames, ignore_index=True)
    if frame.clean_text.duplicated().any():
        raise ValueError('Duplicate cleaned texts across the combined partitions')
    return frame


def load_vocab(settings):
    vocab = json.loads(Path(settings['vocab_path']).read_text(encoding='utf-8'))
    if vocab['<PAD>'] != 0 or vocab['<UNK>'] != 1:
        raise ValueError('Unexpected CNN vocabulary')
    return vocab


def load_glove_init(settings, vocab):
    vectors, std = load_glove(settings['glove_path'], vocab, settings['embedding_dim'])
    return vocab, vectors, glove_sources(vocab, vectors), std


def setup_deterministic(device_name='auto'):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device_name)


def train_cnn_fixed(seed, settings, tokens, labels, vocab_size, device, glove, log=print):
    """Trains for settings['epochs'] epochs on all given reviews; returns the final-epoch model and per-epoch losses."""
    seed_everything(seed)
    model = TextCNN(vocab_size, settings['embedding_dim'], settings['kernel_sizes'], settings['num_filters'], settings['dropout']).to(device)
    if settings['embedding_init'] != 'glove' or settings['freeze_embedding']:
        raise ValueError('The final config uses fine-tuned GloVe embeddings')
    model.embedding.weight.data.copy_(glove_matrix(seed, *glove, settings['embedding_dim']).to(device))
    model.embedding.weight.requires_grad_(True)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=settings['lr'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(settings['pos_weight']), dtype=torch.float32, device=device))
    targets = torch.tensor(labels, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tokens, targets), batch_size=settings['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker, num_workers=0)
    losses = []
    for epoch in range(1, settings['epochs'] + 1):
        model.train()
        running = 0.0
        for batch, target in loader:
            batch, target = batch.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), target)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(batch)
        losses.append(running / len(tokens))
        log(f'seed {seed} epoch {epoch}/{settings["epochs"]}: train_loss={losses[-1]:.4f}')
    return model, losses


def build_cnn(settings, vocab_size, state_dict, device):
    model = TextCNN(vocab_size, settings['embedding_dim'], settings['kernel_sizes'], settings['num_filters'], settings['dropout']).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def cnn_probabilities(model, tokens, batch_size, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            scores.append(torch.sigmoid(model(tokens[start:start + batch_size].to(device))).cpu())
    return torch.cat(scores).numpy().astype(np.float64)


def encode_texts(texts, vocab, settings):
    return encode(texts, vocab, settings['max_len'])


def full_metrics(labels, scores, threshold):
    """Metrics at the assigned threshold and at 0.5, plus ROC-AUC."""
    return {'threshold': threshold, 'roc_auc': float(roc_auc_score(labels, scores)),
            'at_threshold': threshold_metrics(labels, scores, threshold), 'at_0.5': threshold_metrics(labels, scores, .5)}


def paired_bootstrap(labels, predictions_a, predictions_b, resamples=2000, seed=42):
    """A minus B on the same reviews: mean difference, 95% percentile interval, and the observed difference."""
    rng = np.random.default_rng(seed)
    a, b = predictions_a.astype(float), predictions_b.astype(float)
    observed_a, observed_b = threshold_metrics(labels, a, .5), threshold_metrics(labels, b, .5)
    differences = {'macro_f1': [], 'accuracy': []}
    for _ in range(resamples):
        index = rng.integers(0, len(labels), len(labels))
        ma, mb = threshold_metrics(labels[index], a[index], .5), threshold_metrics(labels[index], b[index], .5)
        for key in differences:
            differences[key].append(ma[key] - mb[key])
    result = {}
    for key, values in differences.items():
        low, high = np.percentile(values, [2.5, 97.5])
        result[key] = {'observed': observed_a[key] - observed_b[key], 'mean': float(np.mean(values)), 'low': float(low), 'high': float(high),
                       'entirely_below_0': bool(high < 0), 'entirely_above_0': bool(low > 0)}
    return result
