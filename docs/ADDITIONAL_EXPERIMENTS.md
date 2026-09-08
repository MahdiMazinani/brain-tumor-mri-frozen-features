# Optional additional experiments

The supplied scripts are designed to write to a new output directory, validate source manifests/cache bindings, and avoid modifying completed runs. They include (1) validation-tuned logistic regression with/without PCA-95%, and (2) a ResNet-18 fine-tuned control with the matched five-fold plan. These runs were proposed after the existing result review and must be described as exploratory. Do not add their results until the complete output folder, integrity files and logs have been reviewed.

The Windows batch runner defaults to a costly prospective full-grid timing benchmark. Change `RUN_FULL_GRID_TIMING=0` unless that specific timing measurement is required.
