# Milestone-based progression classes in early Parkinson's disease

Analysis code for the study *Longitudinal Clinical Milestone-Based Identification and Baseline Prediction of Progression Classes in Early Parkinson's Disease*.

The workflow builds six longitudinal clinical milestone histories from the Parkinson's Progression Markers Initiative (PPMI), fits censoring-aware discrete-time latent class models to identify rapid and slow progression classes, and develops and internally evaluates baseline prediction models (logistic regression, random forest, support vector machine and XGBoost) on a locked development/test split.

## Contents

- `code/run_all.py`: runs the pipeline in dependency order and finishes with cross-step consistency checks.
- `code/step_01_*.py` to `code/step_21_*.py`, `code/step_14_nomogram.r`: the analysis steps (milestones, event histories, latent class models, preprocessing and predictor selection, classifiers, calibration, nomogram, tables and figures, and the sensitivity analyses reported in the paper).
- `code/*_utils.py`, `code/baseline_preprocessing.py`, `code/figstyle.py`, `code/figure_style.py`: shared modules.
- `code/test_*.py`: unit tests for the milestone, MoCA and baseline-preprocessing logic.
- `code/validate_pipeline_outputs.py`, `code/generate_reproducibility_manifest.py`: output checks and input verification.
- `code/input_data_versions.csv`: the ten PPMI input files, with byte sizes and SHA-256 digests (no data).
- `code/PPMI_initial_feature_sets.csv`, `code/PPMI_feature_mapping.csv`: candidate predictors and display names.
- `code/random_seed_registry.csv`: random seeds used across the scripts.
- Final model parameters: `code/step_11_*_best_hyperparameters.csv`, `code/step_11_*_calibration_parameters.csv` (Platt scaling), and for logistic regression `code/step_11_lr_model_coefficients.csv`, `code/step_11_lr_scaler_parameters.csv`, `code/step_11_lr_nomogram_intercept.csv` and `code/step_13_lr_nomogram_parameters.csv` (full-precision coefficients on the original scale).

No PPMI data, participant-level derivatives, predictions or other results are included.

## Data

The analysis uses the PPMI curated data cut of 10 August 2026 (`PPMI_Curated_Data_Cut_Public_20260810.xlsx`) and nine PPMI study tables downloaded on 27 September 2026 (listed in `code/input_data_versions.csv`). Obtain them from the [PPMI data access portal](https://www.ppmi-info.org/access-data-specimens/download-data) after registering and accepting the PPMI Data Use Agreement. Place the workbook in `code/` and the tables in `code/tables/`.

Verify the inputs before running:

```bash
cd code
python -X utf8 generate_reproducibility_manifest.py
```

The command stops on a missing file or a size or checksum mismatch. Other data versions can be used but will not reproduce the reported numbers exactly.

## Software

```bash
conda env create -f code/environment-analysis.yml
conda activate ppmi-milestone-analysis
```

The reported run used Python 3.13.5 with the pinned packages in `code/environment-analysis.yml`. The nomogram used R 4.6.1 with `rms` 8.1.1 (see `code/r_session_info.txt`).

## Running

```bash
cd code
python -X utf8 run_all.py --list       # show the steps
python -X utf8 run_all.py --dry-run    # show what would run
python -X utf8 run_all.py              # run the Python pipeline
python -X utf8 run_all.py --include-r  # also draw the R nomogram
```

The runner also accepts `--only`, `--start-from`, `--to`, `--skip` and `--continue-on-error`. Each run writes a log under `code/pipeline_run_logs/`. Hyperparameter searches, nested cross-validation and bootstrap intervals make a full run computationally expensive.

Run the unit tests with:

```bash
cd code
python -m unittest
```

## License

The code is released under the MIT License (see `LICENSE`). This does not apply to PPMI data, which remain subject to the PPMI Data Use Agreement.
