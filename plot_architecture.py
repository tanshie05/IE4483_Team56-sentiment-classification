"""Draws the Text CNN architecture with tensor shapes and parameter counts, taken from the model itself.
Run: python plot_architecture.py --out results/figures/cnn_architecture.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyBboxPatch

from train_cnn import DEFAULT_CONFIG, INK, MUTED, SERIES_STYLES, TextCNN, check_shapes, parameter_counts

SURFACE, BOX_EDGE, ACCENT = '#fcfcfb', '#c3c2b7', SERIES_STYLES[0][0]


def symbolic(shape, batch):
    # The check runs one real batch; the figure shows the batch dimension as B.
    return shape.replace(f'({batch},', '(B,').replace(f'({batch})', '(B,)')


def main(out):
    config = dict(DEFAULT_CONFIG)
    vocab = json.loads(Path(config['vocab_path']).read_text(encoding='utf-8'))
    torch.manual_seed(42)
    model = TextCNN(len(vocab), config['embedding_dim'], config['kernel_sizes'], config['num_filters'], config['dropout']).eval()
    batch = config['batch_size']
    shapes = {row['Layer']: symbolic(row['Output shape'], batch) for row in check_shapes(model, torch.ones(batch, config['max_len'], dtype=torch.long), config)}
    counts = parameter_counts(model, config)
    kernels, filters, dim = config['kernel_sizes'], config['num_filters'], config['embedding_dim']

    fig, ax = plt.subplots(figsize=(9, 11), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 11)
    ax.axis('off')

    def box(x, y, width, title, shape, note=None, accent=False):
        ax.add_patch(FancyBboxPatch((x - width / 2, y - .38), width, .76, boxstyle='round,pad=0.02,rounding_size=0.08',
                                    facecolor='white', edgecolor=ACCENT if accent else BOX_EDGE, linewidth=1.6 if accent else 1.1))
        ax.text(x, y + .12, title, ha='center', va='center', fontsize=9.5, color=INK, fontweight='bold')
        ax.text(x, y - .14, shape if note is None else f'{shape}   ·   {note}', ha='center', va='center', fontsize=8.5, color=MUTED)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2 + .4), xytext=(x1, y1 - .4), arrowprops=dict(arrowstyle='-|>', color=MUTED, linewidth=1, shrinkA=0, shrinkB=0))

    centre, ys = 4.5, [10.3, 9.2, 8.1, 6.7, 5.5, 4.3, 3.2, 2.1, .95]
    box(centre, ys[0], 5.2, 'Input token IDs', shapes['Input token IDs'], f"max_len {config['max_len']}, padded and truncated at the end")
    box(centre, ys[1], 5.2, f'Embedding({len(vocab):,}, {dim}, padding_idx=0)', shapes[f'Embedding({len(vocab)}, {dim}, padding_idx=0)'], f"{counts['embedding_params']:,} parameters")
    box(centre, ys[2], 5.2, 'Transpose to channels first', shapes['Transpose'])
    arrow(centre, ys[0], centre, ys[1])
    arrow(centre, ys[1], centre, ys[2])
    columns = [1.6, 4.5, 7.4]
    for x, size in zip(columns, kernels):
        box(x, ys[3], 2.7, f'Conv1d, kernel {size}, {filters} filters', shapes[f'Conv1d(kernel={size})'], accent=True)
        ax.text(x, ys[3] - .56, f"{counts[f'conv{size}_params']:,} parameters", ha='center', va='center', fontsize=8, color=MUTED)
        box(x, ys[4], 2.7, 'ReLU, global max-pool', shapes[f'Global max-pool (kernel={size})'])
        arrow(centre, ys[2], x, ys[3])
        arrow(x, ys[3] - .25, x, ys[4])
        arrow(x, ys[4], centre, ys[5])
    box(centre, ys[5], 5.2, 'Concatenate', shapes['Concatenate'], f'{len(kernels)} × {filters} features')
    box(centre, ys[6], 5.2, f"Dropout(p={config['dropout']})", shapes[f"Dropout({config['dropout']})"], 'training only')
    box(centre, ys[7], 5.2, f'Linear({filters * len(kernels)}, 1), squeeze', shapes['Squeeze'], f"{counts['classifier_params']:,} parameters, one raw logit per review")
    box(centre, ys[8], 7.6, 'Training: BCEWithLogitsLoss (pos_weight)   ·   Inference: sigmoid → P(positive)',
        'predict positive when P(positive) ≥ threshold tuned on validation')
    for upper, lower in zip(ys[5:8], ys[6:9]):
        arrow(centre, upper, centre, lower)
    fig.text(.5, .015, f"Text CNN (Kim, 2014). B = batch size ({batch} in training). Total {counts['total_params']:,} parameters; "
             f"the embedding holds {counts['embedding_share_pct']:.1f}%. Shapes are asserted by running one batch through the model.",
             ha='center', fontsize=8, color=MUTED, wrap=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print('Saved', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('results/figures/cnn_architecture.png'))
    main(parser.parse_args().out)
