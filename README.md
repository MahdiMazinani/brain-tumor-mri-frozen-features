# Frozen Deep Features for Brain Tumor MRI Classification

Code and de-identified, prediction-level supporting results for the manuscript **“Frozen Deep Features for Brain Tumor MRI Classification: Classifier Selection and Patient-Grouped Evaluation.”**

## What is included

- A frozen-embedding pipeline: ImageNet-pretrained ResNet-18, DenseNet-121 and ViT-B/16 features; train-fitted standardization; optional PCA-95%; SVM, XGBoost, LightGBM, CatBoost and validation-weighted ensemble candidates.
- The patient-grouped five-fold planning script and controls.
- Final aggregate metrics, protocol records and de-identified OOF predictions.
- Scripts for patient-clustered bootstrap, deterministic majority vote, exact McNemar counts, added linear-probe controls, a matched ResNet-18 fine-tuning control, and runtime benchmarking.

## What is deliberately not included

No MRI images, MATLAB files, pretrained weights, native feature caches, checkpoints, private manifests, raw source paths, released participant identifiers, or old superseded experiments are redistributed. Obtain datasets from their original providers and comply with their terms. The local 7,023-image collection is treated as exploratory because patient identifiers were unavailable.

## Reproducibility scope

The five-fold evaluation is grouped by the released identifier field. This is not a verified distinct-person split. Results are exploratory internal evaluations: historical independence from prior work is not established. Model selection occurs on validation data and test evaluation is separate within each recorded fold, but a public code release does not change the study's independence limitations.

Published controls are same-protocol adaptations, not certified reproductions of each source publication. The archived results use no synthetic class-imbalance scenario and no content-based deduplication.

## Setup

The core pipeline files are in `src/`. On Windows CMD, set `PYTHONPATH` before running the pipeline scripts:

```bat
set PYTHONPATH=%CD%\src;%CD%\scripts
```

On Linux/macOS:

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts"
```

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate ; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Use the exact package versions recorded in the run environment where possible. CUDA, GPU driver, pretrained-weight availability and operating system affect runtime.

## Code guide

| File | Purpose |
|---|---|
| `src/data_manifest_and_split.py` | Validate image records and create manifests/splits |
| `scripts/create_patient_grouped_five_fold_plan.py` | Create five patient-grouped outer folds and validation holdouts |
| `src/image_preprocessing.py` | Fixed RGB, resize-224 and ImageNet normalization |
| `src/feature_cache_adapter.py` | Extract/validate frozen feature caches |
| `src/frozen_embedding_pipeline_core.py` | Model, representation and baseline implementations |
| `src/run_frozen_embedding_experiments.py` | Development selection and sealed final evaluation |
| `scripts/analyze_oof_predictions_patient_clustered.py` | Bootstrap, majority vote and McNemar analysis |
| `scripts/run_additional_linear_probe_and_finetuning_controls.py` | Optional extra LP/PCA and ResNet-18 fine-tuning controls |
| `scripts/benchmark_runtime.py` | Honest historical audit and prospective runtime measurement |
| `scripts/run_all_additional_experiments_windows.bat` | Windows runner for optional additional experiments |

## Verify existing public OOF outputs

```bash
python scripts/analyze_oof_predictions_patient_clustered.py \
  --oof results/oof_deidentified/oof_predictions_deidentified.csv \
  --out outputs/oof_analysis --resamples 2000
```

This regenerates majority-vote and clustered analyses from the de-identified hard predictions. It cannot perform mean-probability pooling because historical probability vectors were not retained. Slice-level McNemar is labeled diagnostic because slices within an identifier are dependent.

See [RESULTS_MAP.md](RESULTS_MAP.md), [splits/README.md](splits/README.md), and `docs/` before interpreting or extending results.
