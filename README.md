# IE4483 Team 56: Sentiment Classification of Product Reviews

Binary sentiment classification of product reviews (0 = negative, 1 = positive) for the NTU IE4483 Artificial Intelligence and Data Mining mini project. The labelled data (`train.json`) has 7,401 reviews, 85% of them positive. It is split 70/15/15 into train / validation / holdout, stratified by label and grouped so that duplicate reviews never cross splits. The unlabelled test set (`test.json`) has 1,851 reviews. We compare two models: a **TF-IDF + logistic regression baseline** (5,000 unigrams, class-balanced) and a **Text CNN** (Kim, 2014) with fine-tuned GloVe 6B 100d embeddings (Pennington et al., 2014). The submitted model is their **combination**: the mean of the two models' P(positive), predicted positive when it is at least 0.57. The combination was adopted by a rule fixed in advance: on 5-fold CV it had to beat both single models by at least 0.01 macro-F1 and in at least 4 of 5 folds. The holdout, used once, confirmed it.

| Model | Validation macro-F1 / acc. | 5-fold CV macro-F1 / acc. | Holdout macro-F1 / acc. |
|---|---|---|---|
| TF-IDF + LR baseline | 0.8408 / 0.9193 | 0.8259 ± 0.0190 / 0.9069 | 0.8287 / 0.9072 |
| Text CNN (GloVe, fine-tuned) | 0.8434 ± 0.0069 / 0.9199 ± 0.0042 | 0.8260 ± 0.0187 / 0.9109 | 0.8287 / 0.9202 |
| **CNN + baseline combination (submitted)** | (not scored separately; adopted based on 5-fold CV) | **0.8388 ± 0.0209 / 0.9166** | **0.8546 / 0.9286** |

Validation: 1,078 reviews (CNN mean ± sample std over seeds 42–46). CV: 5 folds of train + val (CNN mean of seeds 42–44). Holdout: 1,078 reviews; models fitted on train + val, CNN seed 42 (seeds 42–46 give 0.8334 ± 0.0056). Every model uses a tuned decision threshold. The final models are retrained on train + val + holdout (7,182 reviews) before predicting the test set.

## Repository structure

```
.
├── *.py                       Pipeline scripts (see "Individual steps"); final_utils.py holds the shared final-stage code
├── run_all.sh                 Whole pipeline end to end, written to reproduction/
├── configs/                   CNN reference settings (checked by train_cnn.py)
├── configs_gpu/               Configs of every reported CNN experiment, plus the CPU smoke test
├── jobs/slurm/                Optional Slurm helpers (GPU/CPU job wrappers, environment setup)
├── train.json, test.json      Course-provided data
├── submission.csv             Final test predictions (lecturer's template format)
└── results/
    ├── baseline/              Baseline grid, nested CV, selected config and saved pipeline
    ├── cnn/embeddings/        Random vs frozen vs fine-tuned GloVe
    ├── cnn/imbalance_checkpoint/  Loss weighting × checkpoint criterion (2×2)
    ├── cnn/hparam_sweeps/     One-factor sweeps (lr, dropout, max_len, filters; kernels, batch, patience, embedding dim)
    ├── cnn/cv/, cnn/cv_oof/   5-fold CV of the final CNN, retraining recipe, out-of-fold probabilities
    ├── combination/           CV decision on the CNN + baseline combination (rule 8)
    ├── holdout/               One-time holdout metrics, decision, per-review predictions and probabilities
    ├── final/                 Final CNN weights, final baseline pipeline, config, test probabilities
    ├── cpu_inference/         CPU inference time, size and parameters per model
    ├── length_coverage/       Review-length percentiles and truncation at max_len
    ├── figures/               Architecture, class balance, length histogram, learning curve, holdout results
    └── archive_cpu/           Early laptop-CPU CNN runs (cited once; not compared with GPU runs)
```

Each experiment folder has `*_runs.csv` (one row per seed: metrics, threshold, timings), `*_history.csv` (per epoch), `*_config.json` (exact settings and environment) and `*_val_scores.npz` (validation probabilities). Per-run weights are not included; their file names remain in the `weights_file` column.

## Setup

Python 3.13 (tested with 3.13.18; the versions are pinned in `requirements.txt`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GloVe 6B is needed for every CNN step except the smoke test and `predict.py`. Download `glove.6B.zip` (822 MB) from <https://nlp.stanford.edu/projects/glove/> and extract `glove.6B.100d.txt` (347,116,733 bytes, 400,000 lines) into `data/glove/`. The optional embedding-dimension sweep (step 6b) also needs `glove.6B.50d.txt`, `glove.6B.200d.txt` and `glove.6B.300d.txt` there. Use Stanford's original files: some re-packaged copies add an extra `<unk>` line at the end.

## How to run

All commands run from the repository root, and every path is relative to it.

**Full pipeline.** `bash run_all.sh` runs every step with the final settings into `reproduction/`, so the committed `results/` are never overwritten. It picks CPU or GPU automatically; `DEVICE=cpu bash run_all.sh` forces CPU. Runtime: about 18 minutes on one NVIDIA RTX A6000 (measured, without step 6b), or about 2.5 hours on 4 CPU threads (estimated from per-epoch times).

**Individual steps** (each writes to its `results/` folder by default):

| Step | Command |
|---|---|
| 1 Data cleaning, splits, CNN vocabulary → `processed/` | `python prepare_data.py` |
| 2 Baseline grid + nested CV | `python train_baseline.py` |
| 3–6 CNN experiments | `python train_cnn.py --config configs_gpu/<name>.json` (`cnn_baseline_variants`, `cnn_imbalance_checkpoint`, `cnn_hparam_sweep`, `cnn_hparam_sweep_2`, `cnn_hparam_sweep_2_embedding`) |
| 7 CNN 5-fold CV and retraining recipe | `python cv_cnn.py --config configs_gpu/cnn_cv.json` (and `cnn_cv_oof.json` for out-of-fold probabilities) |
| 8a Combination decision on CV | `python stage_a.py` |
| 8b Holdout evaluation (once) | `python evaluate_holdout.py --eval-split holdout --fit-splits train,val --stage-a results/combination/stage_a_decision.json` |
| 8c Final retraining on all labelled data | `python train_final.py --fit-splits train,val,holdout` |
| Extras | `length_coverage.py`, `time_inference.py`, `time_combination.py` (CPU timing), `plot_architecture.py`, `make_report_assets.py`, `make_holdout_figure.py`, `error_candidates.py` |

`evaluate_holdout.py` refuses to run if `results/holdout/holdout_metrics.json` already exists, because the holdout is used only once. To rerun it, use `run_all.sh` or pass another `--out-dir`.

**Quick CPU smoke test** (about 30 s, no GloVe needed):

```bash
python prepare_data.py
python train_cnn.py --config configs_gpu/cnn_smoke.json   # 1 epoch, 1 seed -> reproduction/smoke_test/
```

**Regenerate and check the submission** from the saved final models (a few seconds on CPU, no GloVe needed):

```bash
python prepare_data.py                                         # optional: lets predict.py cross-check the cleaned test text
python predict.py --output reproduction/submission.csv
python check_submission.py --submission reproduction/submission.csv
cmp reproduction/submission.csv submission.csv && echo identical
```

`check_submission.py` parses the file as CSV (reviews contain quoted line breaks, so line counts are meaningless). It checks the header, 1,851 rows, the review order against `test.json`, 0/1 predictions and CRLF row endings. The lecturer's template is rebuilt from `test.json`, byte for byte; pass `--template FILE` to check against a copy of the original instead.

**Slurm (optional).** Set `PROJECT_DIR`, `ENV_PATH` and `CONDA_MODULE` at the top of the scripts in `jobs/slurm/` (or export them), run `mkdir -p logs`, then for example:

```bash
sbatch jobs/slurm/setup_env.sh                     # build the Conda environment on a CPU node
sbatch jobs/slurm/train_gpu.sh bash run_all.sh     # whole pipeline on one GPU
sbatch jobs/slurm/cpu_job.sh python time_inference.py
```

## Reproducibility notes

- Seeds: 42 for data splits and the baseline; 42–44 for CV; 42–46 for validation and holdout runs; 42 for the final CNN. Results are reported as mean ± sample std over seeds.
- PyTorch runs in deterministic mode: `torch.use_deterministic_algorithms(True)`, deterministic cuDNN, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set by the scripts and the Slurm wrapper). `train_cnn.py`, `evaluate_holdout.py` and `train_final.py` each rerun one seed in the same process and check that the metrics and weights are identical.
- A complete end-to-end run of `run_all.sh` on an RTX A6000 reproduced `submission.csv` byte for byte. It also reproduced the test probabilities, the holdout metrics of every model and the combination decision exactly. Regenerating the submission on CPU from `results/final/` also gives an identical `submission.csv`; CNN probabilities differ by at most 2e-4.
- Results can differ slightly across hardware. Retraining the baseline grid on a different CPU changed scores by up to 0.003 macro-F1 and selected a statistically tied configuration (exp18 instead of exp04). The baseline settings are therefore fixed as exp04, read from `results/baseline/best_baseline_config.json`, and retrained but never re-selected. CNN results are compared only between runs on the same device.
- Small differences are tested with a paired bootstrap (2,000 resamples, seed 42, 95% percentile interval).

## Authors

Team 56, NTU IE4483 Artificial Intelligence and Data Mining

- Tanshie Singhal
- Raghavi Goel
- Veeraj Mohanty

## References

- Kim, Y. (2014). Convolutional neural networks for sentence classification. *Proceedings of EMNLP 2014*, 1746–1751.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *Proceedings of EMNLP 2014*, 1532–1543.
- Blitzer, J., Dredze, M., & Pereira, F. (2007). Biographies, Bollywood, boom-boxes and blenders: Domain adaptation for sentiment classification. *Proceedings of ACL 2007*, 440–447.
