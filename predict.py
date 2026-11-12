"""IE4483 Project 1: predictions from the saved final model(s).

Submission:  python predict.py --output reproduction/submission.csv
             Reads test.json, cleans it exactly as prepare_data.py does, predicts with the submission model named in
             results/final/final_config.json, and writes the predictions into a COPY of the lecturer's template
             (reviews untouched, same quoting and CRLF row endings). The template is rebuilt from test.json unless
             --template is given (check_submission.template_text_from_test). Never overwrites the template.
Dry run:     python predict.py --final-dir <dir> --input-csv processed/val.csv --output <file>
             Predicts a labelled CSV (clean_text column) and prints accuracy and macro-F1.
Both modes also save every model's probabilities next to the output (<output stem>_probabilities.csv).
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

import final_utils as fu
from check_submission import read_template, write_rows
from eval_utils import threshold_metrics
from prepare_data import clean_review


def load_models(final_dir, device):
    config = json.loads((final_dir / 'final_config.json').read_text(encoding='utf-8'))
    cnn = config['cnn']
    vocab = fu.load_vocab(cnn)
    if len(vocab) != cnn['vocab_size']:
        raise ValueError('Vocabulary size differs from the one the final CNN was trained with')
    model = fu.build_cnn(cnn, len(vocab), torch.load(final_dir / cnn['weights'], map_location=device), device)
    baseline = joblib.load(final_dir / config['baseline']['pipeline'])
    return config, vocab, model, baseline


def probabilities(texts, config, vocab, model, baseline, device):
    cnn_prob = fu.cnn_probabilities(model, fu.encode_texts(texts, vocab, config['cnn']), config['cnn']['batch_size'], device)
    baseline_prob = baseline.predict_proba(texts)[:, 1]
    frame = pd.DataFrame({'cnn_prob': cnn_prob, 'baseline_prob': baseline_prob})
    frame['cnn_pred'] = (frame.cnn_prob >= config['cnn']['threshold']).astype(int)
    frame['baseline_pred'] = (frame.baseline_prob >= config['baseline']['threshold']).astype(int)
    if config.get('combination_threshold') is not None:
        frame['combination_prob'] = (frame.cnn_prob + frame.baseline_prob) / 2
        frame['combination_pred'] = (frame.combination_prob >= config['combination_threshold']).astype(int)
    chosen = config['submission_model']
    frame['prediction'] = frame[f'{chosen}_pred']
    return frame


def main(args):
    device = fu.setup_deterministic(args.device)
    final_dir = Path(args.final_dir)
    config, vocab, model, baseline = load_models(final_dir, device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Submission model: {config['submission_model']} (fitted on {'+'.join(config['fitted_on'])}, {config['fit_rows']} reviews); "
          f"CNN threshold {config['cnn']['threshold']}, baseline threshold {config['baseline']['threshold']}")

    if args.input_csv:
        data = pd.read_csv(args.input_csv, keep_default_na=False)
        frame = probabilities(data.clean_text, config, vocab, model, baseline, device)
        pd.concat([data[['reviews', 'sentiments']], frame], axis=1).to_csv(output.with_name(output.stem + '_probabilities.csv'), index=False, lineterminator='\n')
        pd.DataFrame({'reviews': data.reviews, 'predictions': frame.prediction}).to_csv(output, index=False, lineterminator='\r\n')
        labels = data.sentiments.to_numpy()
        for name in ('cnn', 'baseline', 'combination'):
            if f'{name}_pred' in frame:
                m = threshold_metrics(labels, frame[f'{name}_pred'].to_numpy().astype(float), .5)
                print(f"{name}: accuracy {m['accuracy']:.4f}, macro-F1 {m['macro_f1']:.4f}")
        return

    if args.template and output.resolve() == Path(args.template).resolve():
        raise SystemExit('The output must be a copy; refusing to overwrite the template')
    template_text, rows = read_template(args.template, args.test_json)
    if write_rows(rows) != template_text:
        raise SystemExit('The CSV writer does not reproduce the template exactly; refusing to write')
    if rows[0] != ['reviews', 'predictions']:
        raise SystemExit(f'Unexpected template header {rows[0]}')
    test = pd.DataFrame(json.loads(Path(args.test_json).read_text(encoding='utf-8')))
    if [row[0] for row in rows[1:]] != test.reviews.astype(str).tolist():
        raise SystemExit('Template reviews are not test.json reviews in the same order')
    texts = test.reviews.fillna('').map(clean_review)
    prepared = fu.PROCESSED / 'test_prepared.csv'
    if prepared.exists() and pd.read_csv(prepared, keep_default_na=False).clean_text.tolist() != texts.tolist():
        raise SystemExit('Cleaned test reviews differ from processed/test_prepared.csv')
    frame = probabilities(texts, config, vocab, model, baseline, device)
    pd.concat([test[['reviews']], frame], axis=1).to_csv(output.with_name(output.stem + '_probabilities.csv'), index=False, lineterminator='\n')
    filled = [rows[0]] + [[row[0], str(int(p))] for row, p in zip(rows[1:], frame.prediction)]
    output.write_bytes(write_rows(filled).encode('utf-8'))
    positives = int(frame.prediction.sum())
    print(f'Wrote {output}: {len(frame)} predictions, {positives} positive ({100 * positives / len(frame):.1f}%), '
          f'{len(frame) - positives} negative. CNN and baseline agree on {100 * np.mean(frame.cnn_pred == frame.baseline_pred):.1f}% of reviews.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--final-dir', default='results/final')
    parser.add_argument('--template', default=None, help="The lecturer's template; default: rebuilt from --test-json")
    parser.add_argument('--test-json', default='test.json')
    parser.add_argument('--output', default='reproduction/submission.csv')
    parser.add_argument('--input-csv', default=None, help='Dry run: a labelled CSV with clean_text and sentiments columns')
    parser.add_argument('--device', default='auto')
    main(parser.parse_args())
