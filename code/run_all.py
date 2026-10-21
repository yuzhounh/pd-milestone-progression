#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline Orchestrator: run_all.py

Run the full milestone-progression pipeline for early Parkinson's disease in dependency order (subtype identification and machine-learning prediction), followed by cross-step consistency checks.
Supports a full run, start/stop steps, single-step runs, a dry run and a structured run log.
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

# Avoid UnicodeEncodeError on Windows consoles with legacy encodings (e.g. cp1252 / gbk)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Directory of this script (code/)
CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent

# ==============================================================================
# Pipeline scripts in dependency order
# ==============================================================================
# Pipeline scripts in dependency order (steps 01-20, followed by QA/provenance)
# ==============================================================================
PIPELINE_STEPS: List[Tuple[str, str]] = [
    (
        "step_01_milestones.py",
        "Build the six longitudinal clinical milestone histories (raw PPMI -> milestones)",
    ),
    (
        "step_02_time_to_event.py",
        "Time to first milestone and baseline eligibility (time-to-event)",
    ),
    (
        "step_03_lca_model_selection.py",
        "Latent class model selection (AIC/BIC/entropy for k = 1-6)",
    ),
    (
        "step_04_lca_classification_quality.py",
        "Latent class classification quality (AvePP, OCC, etc.)",
    ),
    (
        "step_05_lca.py",
        "Fit the two-class latent class model and assign subtype labels (rapid vs. slow)",
    ),
    (
        "step_06_cumulative_incidence.py",
        "Cumulative incidence of each milestone by subtype",
    ),
    ("step_07_km_curve.py", "Kaplan-Meier curves of the six milestones by subtype"),
    (
        "step_08_lasso_calculate.py",
        "Training-only preprocessing and LASSO screening of candidates (maximum mean AUPRC)",
    ),
    (
        "step_09_backward_stepwise.py",
        "Backward elimination (P < 0.05) on the training set to fix the final predictors",
    ),
    (
        "step_09_nested_cv_sensitivity.py",
        "Repeated nested CV in the training set, redoing preprocessing, selection and LR tuning in each outer fold",
    ),
    (
        "step_09_functional_form_diagnostics.py",
        "Spline non-linearity and influential-point checks for the final LR predictors",
    ),
    ("step_10_lasso_plot.py", "LASSO coefficient paths and cross-validation curve"),
    ("step_11_lr.py", "Logistic regression: training, Optuna tuning and SHAP"),
    (
        "step_09_class_uncertainty_sensitivity.py",
        "Class-uncertainty sensitivity: high-confidence subset, posterior soft labels and pseudo-label sampling",
    ),
    ("step_11_rf.py", "Random forest: training, tuning and feature importance"),
    (
        "step_11_svm.py",
        "Support vector machine: training, tuning and test-set prediction",
    ),
    ("step_11_xgboost.py", "XGBoost: training, tuning and feature importance"),
    ("step_11_auc_curves.py", "Combined ROC and PR curves of the four classifiers"),
    (
        "step_12_bootstrap_ci_comparison.py",
        "Bootstrap (2000 resamples) confidence intervals and full performance table",
    ),
    (
        "step_13_lr_nomogram.py",
        "Nomogram logistic model with coefficients back-transformed to original units",
    ),
    ("step_14_nomogram.r", "Draw the nomogram with the R rms package"),
    ("step_15_or_forest.py", "Multivariable odds ratios and forest plot"),
    ("step_16_vif.py", "Variance inflation factors and predictor correlation heatmap"),
    (
        "step_17_shap_importance.py",
        "SHAP importance ranking compared across classifiers",
    ),
    (
        "step_18_baseline_tables.py",
        "Baseline characteristics (Table 1) and missing-data patterns",
    ),
    (
        "step_19_all_models_calibration_summary.py",
        "Calibration summary of the four prediction models",
    ),
    (
        "step_20_decision_curve_analysis.py",
        "Calibration, risk distribution, decision-curve and clinical-impact panels on the test set",
    ),
    (
        "validate_pipeline_outputs.py",
        "Check consistency of cohort, event times, split, predictions, calibration and feature sets",
    ),
    (
        "generate_reproducibility_manifest.py",
        "Verify input SHA-256 digests and write the software and output manifests",
    ),
]


def find_rscript_executable() -> Optional[str]:
    """Locate the Rscript executable."""
    import glob
    import shutil

    cmd = shutil.which("Rscript")
    if cmd:
        return cmd
    matches = glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe")
    if matches:
        return matches[-1]
    return None


def ensure_environment_prerequisites() -> None:
    """Check that the raw-data directory tables/ is present."""
    local_tables = CODE_DIR / "tables"
    if not local_tables.exists():
        print(
            f"[Warning] No tables/ raw-data directory found in {CODE_DIR}; check that the data are complete."
        )


def format_duration(seconds: float) -> str:
    """Format a duration for display."""
    if seconds < 60:
        return f"{seconds:5.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:2d}m {secs:04.1f}s"


def parse_step_index(identifier: str) -> Optional[int]:
    """Resolve a step index from a number, position or file name."""
    ident = identifier.strip().lower()
    # Try a plain number (1-indexed)
    if ident.isdigit():
        idx = int(ident) - 1
        if 0 <= idx < len(PIPELINE_STEPS):
            return idx

    # Try the script file name or a prefix
    for i, (script_name, _) in enumerate(PIPELINE_STEPS):
        base_name = script_name.lower()
        if ident == base_name or ident == base_name.replace(".py", ""):
            return i
        if ident in base_name:
            return i
    return None


def run_single_script(
    script_file: str, description: str, step_num: int, total_steps: int
) -> Tuple[bool, float, Optional[str]]:
    """Run one script in a separate subprocess."""
    script_path = CODE_DIR / script_file
    if not script_path.exists():
        return False, 0.0, f"File not found: {script_path}"

    banner = f"[{step_num:02d}/{total_steps:02d}] Running: {script_file}"
    desc_str = f"       Description: {description}"
    separator = "=" * 80

    print(f"\n{separator}")
    print(banner)
    print(desc_str)
    print(separator)

    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    if script_file.lower().endswith((".r", ".R")):
        rscript_bin = find_rscript_executable()
        if not rscript_bin:
            err_msg = "Rscript not found. To draw the nomogram, install R and add it to PATH or install it under C:\\Program Files\\R."
            print(f">>> [FAILED] {script_file}: {err_msg}")
            return False, 0.0, err_msg
        cmd = [rscript_bin, str(script_path)]
    else:
        cmd = [sys.executable, "-X", "utf8", str(script_path)]

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(CODE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        # Stream the output live
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                print(line, end="", flush=True)
            process.stdout.close()

        process.wait()
        elapsed = time.time() - start_time

        if process.returncode == 0:
            print(f">>> [OK] {script_file} finished (time: {format_duration(elapsed)})")
            return True, elapsed, None
        else:
            err_msg = f"process exited with code {process.returncode}"
            print(
                f">>> [FAILED] {script_file} ({err_msg}, time: {format_duration(elapsed)})"
            )
            return False, elapsed, err_msg

    except Exception as exc:
        elapsed = time.time() - start_time
        err_msg = f"subprocess error: {exc}"
        print(f">>> [ERROR] {script_file}: {err_msg}")
        return False, elapsed, err_msg


def capture_source_state() -> Dict[str, str]:
    """Capture the source revision and tracked-source changes before execution."""
    state = {
        "git_revision": "unavailable",
        "tracked_source_dirty": "unknown",
        "tracked_source_status": "",
    }
    try:
        state["git_revision"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                "code/*.py",
                "code/*.r",
                "code/*.R",
                "code/environment-analysis.yml",
                "code/random_seed_registry.csv",
            ],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
        state["tracked_source_dirty"] = "true" if status else "false"
        state["tracked_source_status"] = status
    except (OSError, subprocess.CalledProcessError) as exc:
        state["source_state_error"] = str(exc)
    return state


def write_execution_log(
    log_path: Path,
    execution_stats: List[Dict],
    run_started_utc: str,
    run_finished_utc: str,
    provenance: Optional[Dict[str, str]] = None,
    *,
    exclusive: bool = False,
) -> None:
    """Write a run summary; exclusive archives never replace an existing log."""
    provenance = provenance or {}
    fieldnames = [
        "run_started_utc",
        "run_finished_utc",
        "python_executable",
        "python_version",
        "platform",
        "step_index",
        "script",
        "status",
        "duration_seconds",
        "error",
        "run_id",
        "git_revision",
        "tracked_source_dirty",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open(
        "x" if exclusive else "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in execution_stats:
            writer.writerow(
                {
                    "run_started_utc": run_started_utc,
                    "run_finished_utc": run_finished_utc,
                    "python_executable": sys.executable,
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "step_index": item["index"],
                    "script": item["script"],
                    "status": "SUCCESS" if item["success"] else "FAILED",
                    "duration_seconds": f'{item["duration"]:.6f}',
                    "error": item["error"] or "",
                    "run_id": provenance.get("run_id", ""),
                    "git_revision": provenance.get("git_revision", "unavailable"),
                    "tracked_source_dirty": provenance.get(
                        "tracked_source_dirty", "unknown"
                    ),
                }
            )


def main():
    """Parse the command line, run the selected steps and write the run log."""
    parser = argparse.ArgumentParser(
        description="Run the early-PD milestone analysis pipeline (with final consistency checks)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--start-from",
        "-s",
        type=str,
        help="Start from this step (number or script name, e.g. step07)",
    )
    parser.add_argument(
        "--to",
        "-e",
        type=str,
        help="Stop after this step (number or script name, e.g. step15)",
    )
    parser.add_argument(
        "--only",
        "-o",
        type=str,
        help="Run a single step only (e.g. -o step_16_vif.py or -o 17)",
    )
    parser.add_argument(
        "--skip",
        "-k",
        type=str,
        nargs="+",
        help="Skip one or more scripts (e.g. --skip step_11_rf.py step_11_svm.py)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later steps after a failure (default: stop at the first failure)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the steps that would run, without running them",
    )
    parser.add_argument(
        "--include-r",
        action="store_true",
        help="Also run the R nomogram script (step_14_nomogram.r; requires R and the rms package)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="pipeline_run_log.csv",
        help="Path of the structured run log; relative to code/ (default: pipeline_run_log.csv)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all pipeline steps and what they do",
    )

    args = parser.parse_args()

    # List the steps
    if args.list:
        print(
            "\n=============================== Pipeline steps ==============================="
        )
        for i, (script_name, description) in enumerate(PIPELINE_STEPS, start=1):
            print(f"[{i:02d}] {script_name:<42} : {description}")
        print(
            "============================================================================\n"
        )
        return

    # Pre-flight checks
    ensure_environment_prerequisites()

    # Determine the steps to run
    start_idx = 0
    end_idx = len(PIPELINE_STEPS) - 1

    if args.only:
        idx = parse_step_index(args.only)
        if idx is None:
            print(f"[Error] Unknown step name or number: {args.only}")
            sys.exit(1)
        start_idx = idx
        end_idx = idx

    if args.start_from:
        idx = parse_step_index(args.start_from)
        if idx is None:
            print(f"[Error] Unknown start step: {args.start_from}")
            sys.exit(1)
        start_idx = idx

    if args.to:
        idx = parse_step_index(args.to)
        if idx is None:
            print(f"[Error] Unknown stop step: {args.to}")
            sys.exit(1)
        end_idx = idx

    if start_idx > end_idx:
        print(
            f"[Error] Start step ({start_idx + 1}) is after stop step ({end_idx + 1})"
        )
        sys.exit(1)

    # Filter the scripts to run
    scripts_to_run: List[Tuple[int, str, str]] = []
    skipped_names = set(args.skip) if args.skip else set()

    for idx in range(start_idx, end_idx + 1):
        script_name, desc = PIPELINE_STEPS[idx]
        if any(skip_item in script_name for skip_item in skipped_names):
            continue
        is_r_step = script_name.lower().endswith(".r")
        explicitly_selected_r = bool(args.only and start_idx == idx and end_idx == idx)
        if is_r_step and not (args.include_r or explicitly_selected_r):
            continue
        scripts_to_run.append((idx + 1, script_name, desc))

    # Dry run
    if args.dry_run:
        print("\n[Dry run] The following scripts would run in this order:")
        for step_idx, script_name, desc in scripts_to_run:
            print(
                f"  [{step_idx:02d}/{len(PIPELINE_STEPS):02d}] {script_name:<40} ({desc})"
            )
        print(f"\nNumber of planned steps: {len(scripts_to_run)}")
        return

    # Run the pipeline
    pipeline_start = time.time()
    run_started_utc = datetime.now(timezone.utc).isoformat()
    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = CODE_DIR / log_path
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    )
    provenance = {"run_id": run_id, **capture_source_state()}
    archive_dir = log_path.parent / "pipeline_run_logs"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{log_path.stem}_{run_id}.csv"
    with archive_path.with_suffix(".json").open("x", encoding="utf-8") as handle:
        json.dump(
            {
                **provenance,
                "run_started_utc": run_started_utc,
                "arguments": sys.argv[1:],
                "requested_steps": [script for _, script, _ in scripts_to_run],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    total_to_run = len(scripts_to_run)
    execution_stats: List[Dict] = []
    has_failure = False

    print("\n" + "=" * 80)
    print(f" Starting the PD milestone pipeline ({total_to_run} planned steps)")
    print(f" Working directory: {CODE_DIR}")
    print(f" Python interpreter: {sys.executable}")
    print("=" * 80)

    try:
        for current_i, (step_idx, script_name, desc) in enumerate(
            scripts_to_run, start=1
        ):
            step_start = time.time()
            try:
                success, elapsed, err_msg = run_single_script(
                    script_file=script_name,
                    description=desc,
                    step_num=current_i,
                    total_steps=total_to_run,
                )
            except BaseException as exc:
                execution_stats.append(
                    {
                        "index": step_idx,
                        "script": script_name,
                        "success": False,
                        "duration": time.time() - step_start,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                raise

            execution_stats.append(
                {
                    "index": step_idx,
                    "script": script_name,
                    "success": success,
                    "duration": elapsed,
                    "error": err_msg,
                }
            )

            if not success:
                has_failure = True
                if not args.continue_on_error:
                    print(
                        f"\n[Pipeline stopped] {script_name} failed; later steps were not run."
                    )
                    print(
                        "Hint: use '--continue-on-error' to carry on, or '--start-from' to retry from this step."
                    )
                    break
    finally:
        total_pipeline_time = time.time() - pipeline_start
        run_finished_utc = datetime.now(timezone.utc).isoformat()
        write_execution_log(
            archive_path,
            execution_stats,
            run_started_utc,
            run_finished_utc,
            provenance,
            exclusive=True,
        )
        write_execution_log(
            log_path, execution_stats, run_started_utc, run_finished_utc, provenance
        )

    # ==========================================================================
    # Pipeline summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print(" Pipeline summary")
    print("=" * 80)
    print(f"{'No.':<6} {'Script':<42} {'Status':<10} {'Time':<12}")
    print("-" * 80)

    for item in execution_stats:
        status_str = "SUCCESS" if item["success"] else "FAILED"
        print(
            f"[{item['index']:02d}]   {item['script']:<42} {status_str:<10} {format_duration(item['duration']):<12}"
        )

    print("=" * 80)
    success_count = sum(1 for item in execution_stats if item["success"])
    fail_count = sum(1 for item in execution_stats if not item["success"])
    print(
        f"Finished: {success_count}/{len(execution_stats)} succeeded, {fail_count} failed, "
        f"total time: {format_duration(total_pipeline_time)}"
    )
    print(f"Structured run log: {log_path}")
    print(f"Log of this run: {archive_path}")
    print("=" * 80 + "\n")

    if has_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
