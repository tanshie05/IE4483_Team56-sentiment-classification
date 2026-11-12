"""Report figure: holdout macro-F1 and accuracy for the baseline, the official CNN (seed 42) and the submitted combination.
Reads results/holdout/holdout_metrics.json (no recomputation) and writes results/figures/holdout_results.png.
Run: python make_holdout_figure.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS = Path('results/holdout/holdout_metrics.json')
OUT = Path('results/figures/holdout_results.png')
# Fixed categorical order (reference palette slots 1-3): one colour per model, the same in both panels.
MODELS = [('baseline', 'Baseline\nTF-IDF + LR', '#2a78d6'),
          ('cnn_seed42', 'Text CNN\n(seed 42)', '#eb6834'),
          ('combination', 'CNN + baseline\n(submitted)', '#1baf7a')]
TEXT, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'

results = json.loads(METRICS.read_text(encoding='utf-8'))['results']
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True, facecolor=SURFACE)
for ax, (metric, title) in zip(axes, [('macro_f1', 'Macro-F1'), ('accuracy', 'Accuracy')]):
    ax.set_facecolor(SURFACE)
    values = [results[key]['at_threshold'][metric] for key, _, _ in MODELS]
    labels = [f"{name}\nthreshold {results[key]['threshold']:.2f}" for key, name, _ in MODELS]
    bars = ax.bar(range(len(MODELS)), values, width=0.6, color=[c for _, _, c in MODELS], edgecolor=SURFACE, linewidth=2, zorder=3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f'{value:.4f}', ha='center', va='bottom', fontsize=10, color=TEXT,
                fontweight='bold' if bar is bars[-1] else 'normal')
    ax.set_xticks(range(len(MODELS)), labels, fontsize=8.5, color=MUTED)
    ax.set_title(title, fontsize=11, color=TEXT, loc='left')
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis='y', colors=MUTED, labelsize=8.5, length=0)
    ax.tick_params(axis='x', length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
fig.suptitle('Holdout results (1,078 reviews, used once; models fitted on train + val)', fontsize=12, color=TEXT, x=0.02, ha='left')
fig.text(0.02, 0.01, 'Each model at its own threshold fixed before the holdout was read (rules 5, 6, 8). '
         'Combination = mean of the CNN and baseline probabilities.', fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(OUT, dpi=200, facecolor=SURFACE)
print('Wrote', OUT, {key: {m: round(results[key]['at_threshold'][m], 4) for m in ('macro_f1', 'accuracy')} for key, _, _ in MODELS})
