"""IE4483 Project 1: evaluation helpers shared by the sentiment models."""
import numpy as np

THRESHOLDS = np.arange(5, 96) / 100


def threshold_metrics(labels, scores, threshold):
    """Metrics when predicting 1 (positive) for P(positive) >= threshold."""
    predictions = scores >= threshold
    tp = int((predictions & (labels == 1)).sum())
    fp = int((predictions & (labels == 0)).sum())
    fn = int((~predictions & (labels == 1)).sum())
    tn = int((~predictions & (labels == 0)).sum())

    def prf(hit, false_alarm, miss):
        precision = hit / (hit + false_alarm) if hit + false_alarm else 0.0
        recall = hit / (hit + miss) if hit + miss else 0.0
        return precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    neg_precision, neg_recall, neg_f1 = prf(tn, fn, fp)
    pos_precision, pos_recall, pos_f1 = prf(tp, fp, fn)
    return {
        'accuracy': (tp + tn) / len(labels), 'macro_f1': (neg_f1 + pos_f1) / 2,
        'neg_precision': neg_precision, 'neg_recall': neg_recall, 'neg_f1': neg_f1,
        'pos_precision': pos_precision, 'pos_recall': pos_recall, 'pos_f1': pos_f1,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }


def tune_threshold(labels, scores):
    # Exact macro-F1 ties go to the threshold nearest 0.5, then the lower threshold.
    curve = [threshold_metrics(labels, scores, t)['macro_f1'] for t in THRESHOLDS]
    best = max(range(len(curve)), key=lambda i: (round(curve[i], 12), -abs(THRESHOLDS[i] - .5), -THRESHOLDS[i]))
    return float(THRESHOLDS[best]), curve
