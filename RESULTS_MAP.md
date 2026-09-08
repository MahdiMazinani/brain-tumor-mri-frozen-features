# Results map

| Manuscript content | Public artifact |
|---|---|
| Four-class final test metrics | `results/four_class_7023/final_test_metrics.csv` |
| Four-class paired comparisons | `results/four_class_7023/paired_comparisons.csv` |
| Five-fold grouped metrics | `results/patient_grouped_cv5/fold_metrics.csv`, `summary_metrics.csv` |
| CV protocol and independent metric audit | `results/patient_grouped_cv5/*.json` |
| Patient-clustered intervals and proposed-versus-linear comparison | `results/oof_deidentified/*.csv` |
| De-identified OOF labels/predictions | `results/oof_deidentified/oof_predictions_deidentified.csv` |

Values in the OOF-derived files are fractions unless the column name or manuscript table says percent. The de-identified file uses the standard analysis columns `path` and `patient_id`, but their values are nonreversible record labels and stable research pseudonyms; it has no raw MRI, source file path, released patient identifier, or reverse lookup.

The `additional_controls` directory currently contains only a read-only audit of historical timing fields. New control/fine-tuning results are intentionally not included because they have not been supplied as completed outputs.
