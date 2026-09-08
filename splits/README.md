# Split assignments

The exact split assignments used for the grouped experiment contain source file paths and released dataset identifiers. They are not included in this public release until data-use and identifier-sharing permissions are confirmed. `results/oof_deidentified/oof_predictions_deidentified.csv` retains a stable research pseudonym and fold for every prediction, allowing verification of the published aggregate and patient-level analyses without disclosing the original identifier strings.

To reproduce the exact original folds after obtaining the dataset, use `scripts/create_patient_grouped_five_fold_plan.py` with the documented seed and the released labels/identifiers. Do not claim that regenerated assignments are bit-identical unless verified against the retained private manifest.
