#!/bin/bash
# Runs the whole pipeline in order with the final settings, writing everything to a separate folder so the official
# results in results/ are never overwritten.
#
#   bash run_all.sh                       # CPU or GPU, picked automatically; output in reproduction/
#   DEVICE=cpu bash run_all.sh            # force CPU
#   REPRO_DIR=/tmp/repro bash run_all.sh  # choose the output folder
#
# Needs: the Python packages in requirements.txt, train.json and test.json in this folder, and
# data/glove/glove.6B.100d.txt (see README). The embedding-dimension sweep also needs glove.6B.50d/200d/300d.txt in
# data/glove/; it is skipped if they are missing.
#
# Expected runtime, from the measured epoch counts (about 1,390 CNN epochs in total) and per-epoch times on the
# NTU EEE cluster: about 0.4 s per epoch on an RTX A6000, about 5.3 s on 4 CPU threads (AMD EPYC 9655). A laptop CPU
# is usually slower. GloVe loading adds about half a minute per CNN step.
#   step                              GPU (RTX A6000)   CPU (4 threads)
#   1 prepare data                    < 1 min           < 1 min
#   2 baseline (40 runs + nested CV)  ~2 min (CPU)      ~2 min
#   3 CNN embedding variants          ~2 min            ~15 min
#   4 CNN 2x2 weighting x checkpoint  ~2 min            ~17 min
#   5 CNN hyperparameter sweep        ~3 min            ~30 min
#   6 CNN completeness sweep          ~3 min            ~33 min (6b embedding dimension: not yet measured)
#   7 CNN 5-fold CV + recipe          ~3 min            ~25 min
#   8 stage A (CV rerun), holdout,    ~4 min            ~30 min (CPU: estimate)
#     final model, predict
#   total                             ~19 min           ~2.5 h
# Exact numbers differ by machine: the same seed gives identical results on the same hardware, but a different CPU or
# GPU can change results slightly (see README, "Baseline on other hardware").
set -eo pipefail
cd "$(dirname "$0")"

R=${REPRO_DIR:-reproduction}
DEVICE=${DEVICE:-auto}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$R/configs"

echo "=== 1 Prepare data -> $R/processed"
python prepare_data.py --out-dir "$R/processed"

echo "=== 2 Baseline: TF-IDF + logistic regression -> $R/baseline"
python train_baseline.py --processed-dir "$R/processed" --out-dir "$R/baseline"

echo "=== Writing configs that point at $R"
python - "$R" "$DEVICE" <<'EOF'
import json, sys
from pathlib import Path
root, device = Path(sys.argv[1]), sys.argv[2]
baseline = json.loads((root / 'baseline' / 'best_baseline_config.json').read_text(encoding='utf-8'))
paths = {'processed_dir': str(root / 'processed'), 'vocab_path': str(root / 'processed' / 'cnn_vocab.json'), 'baseline_dir': str(root / 'baseline')}
for name in ('cnn_baseline_variants', 'cnn_imbalance_checkpoint', 'cnn_hparam_sweep', 'cnn_hparam_sweep_2', 'cnn_hparam_sweep_2_embedding', 'cnn_cv', 'cnn_cv_oof'):
    source = Path(f'configs_gpu/{name}.json')
    if not source.exists():
        continue
    config = json.loads(source.read_text(encoding='utf-8'))
    config.update(paths, device=device, out_dir=str(root / Path(config['out_dir']).relative_to('results')))
    for spec in (config.get('reuse') or {}).values():
        spec['dir'] = str(root / Path(spec['dir']).relative_to('results'))
    for overrides in (config.get('variants') or {}).values():
        if 'glove_path' in overrides:
            overrides['glove_path'] = str(Path('data/glove') / Path(overrides['glove_path']).name)
    if name in ('cnn_cv', 'cnn_cv_oof'):
        config.update(decision=str(root / 'cnn' / 'hparam_sweeps' / 'hparam_sweep' / 'sweep_decision.json'),
                      baseline_cv=str(root / 'baseline' / 'baseline_cv.csv'), baseline_experiment=baseline['experiment'])
    (root / 'configs' / f'{name}.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
EOF

echo "=== 3 CNN embedding variants (random, GloVe frozen, GloVe fine-tuned)"
python train_cnn.py --config "$R/configs/cnn_baseline_variants.json"
echo "=== 4 CNN 2x2: loss weighting x checkpoint criterion"
python train_cnn.py --config "$R/configs/cnn_imbalance_checkpoint.json"
echo "=== 5 CNN hyperparameter sweep (learning rate, dropout, max_len, filters)"
python train_cnn.py --config "$R/configs/cnn_hparam_sweep.json"
echo "=== 6 CNN completeness sweep (kernel sizes, batch size, patience)"
python train_cnn.py --config "$R/configs/cnn_hparam_sweep_2.json"
if [ -f "$R/configs/cnn_hparam_sweep_2_embedding.json" ] && ls data/glove/glove.6B.{50d,200d,300d}.txt >/dev/null 2>&1; then
    echo "=== 6b CNN embedding-dimension sweep (GloVe 50d, 200d, 300d)"
    python train_cnn.py --config "$R/configs/cnn_hparam_sweep_2_embedding.json"
else
    echo "=== 6b skipped: needs data/glove/glove.6B.50d.txt, 200d and 300d"
fi
echo "=== 7 CNN 5-fold cross-validation of the final config, and the retraining recipe"
python cv_cnn.py --config "$R/configs/cnn_cv.json"

echo "=== 8 Stage A (rule 8), holdout evaluation (rules 4-8), final retraining (rule 7), test predictions and submission check"
export IE4483_REPRO_DIR="$R"   # final_utils.py reads processed/, cnn/ and baseline/ from $R instead of processed/ and results/
python cv_cnn.py --config "$R/configs/cnn_cv_oof.json"   # step 7 again, saving out-of-fold probabilities
python stage_a.py --cnn-dir "$R/cnn/cv_oof" --saved-cnn-runs "$R/cnn/cv/cv_runs.csv" \
    --saved-cnn-folds "$R/cnn/cv/cv_folds.csv" --out-dir "$R/combination"
python evaluate_holdout.py --eval-split holdout --fit-splits train,val --out-dir "$R/holdout" --device "$DEVICE" \
    --stage-a "$R/combination/stage_a_decision.json"
python train_final.py --fit-splits train,val,holdout --decision "$R/holdout/holdout_decision.json" --out-dir "$R/final" --device "$DEVICE"
python predict.py --final-dir "$R/final" --test-json test.json --output "$R/submission.csv" --device "$DEVICE"
python check_submission.py --submission "$R/submission.csv" --test-json test.json
python error_candidates.py holdout --predictions "$R/holdout/holdout_predictions.csv" --out "$R/holdout/error_candidates.md"
python error_candidates.py test --probabilities "$R/submission_probabilities.csv" --final-dir "$R/final" --out "$R/final/test_examples.md"
echo "Done. Outputs in $R/"
