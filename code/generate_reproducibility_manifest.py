"""Verify declared inputs and inventory reproducibility-critical artifacts."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import platform
import re
import subprocess
import sys
from pathlib import Path

from run_all import find_rscript_executable


CODE_DIR = Path(__file__).resolve().parent
INPUT_MANIFEST = CODE_DIR / "input_data_versions.csv"
OUTPUT_MANIFEST = CODE_DIR / "output_manifest.csv"
SOFTWARE_VERSIONS = CODE_DIR / "software_versions.csv"

PYTHON_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "lifelines",
    "stepmix",
    "optuna",
    "shap",
    "xgboost",
    "openpyxl",
    "joblib",
)

DIRECT_GENERATORS = {
    "PPMI_1_milestones.csv": "step_01_milestones.py",
    "PPMI_2_time_to_event.csv": "step_02_time_to_event.py",
    "PPMI_2_time_to_event_continuous.csv": "step_02_time_to_event.py",
    "PPMI_3_patients_subtypes_lca_k2.csv": "step_05_lca.py",
    "PPMI_4_train.csv": "step_08_lasso_calculate.py",
    "PPMI_4_test.csv": "step_08_lasso_calculate.py",
    "PPMI_4_train_unprocessed.csv": "step_08_lasso_calculate.py",
    "PPMI_4_test_unprocessed.csv": "step_08_lasso_calculate.py",
    "pipeline_validation_summary.csv": "validate_pipeline_outputs.py",
    "software_versions.csv": "generate_reproducibility_manifest.py",
    "random_seed_registry.csv": "manual registry; audited against source",
    "input_data_versions.csv": "manual registry; verified by generate_reproducibility_manifest.py",
    "step_20_decision_curve_results.csv": "step_20_decision_curve_analysis.py",
    "step_20_clinical_impact_results.csv": "step_20_decision_curve_analysis.py",
    "step_20_model_calibration_utility.pdf": "step_20_decision_curve_analysis.py",
    "workflow.svg": "manual workflow source",
    "workflow.pdf": "workflow.svg; Inkscape PDF export",
}

STEP_20_OUTPUTS = frozenset(
    name for name in DIRECT_GENERATORS if name.startswith("step_20_")
)

# These retained files describe the historical 637/274 split, not the current analysis.
HISTORICAL_OUTPUTS = frozenset(
    {
        "step_09_bic_bootstrap_differences.csv",
        "step_09_bic_performance_comparison.csv",
        "step_09_bic_selection_comparison.csv",
        "step_09_bic_stepwise_history.csv",
    }
)

STEP_09_GENERATORS = {
    "step_09_bic": "step_09_bic_sensitivity.py",
    "step_09_nested": "step_09_nested_cv_sensitivity.py",
    "step_09_class_uncertainty": "step_09_class_uncertainty_sensitivity.py",
    "step_09_functional": "step_09_functional_form_diagnostics.py",
    "step_09_nonlinearity": "step_09_functional_form_diagnostics.py",
    "step_09_influence": "step_09_functional_form_diagnostics.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs() -> list[Path]:
    verified: list[Path] = []
    with INPUT_MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            path = CODE_DIR / row["relative_path"]
            if not path.is_file():
                raise FileNotFoundError(f"Declared input is missing: {path}")
            actual_bytes = path.stat().st_size
            actual_hash = sha256_file(path)
            if actual_bytes != int(row["bytes"]):
                raise ValueError(f"Input byte-size mismatch: {path}")
            if actual_hash.lower() != row["sha256"].lower():
                raise ValueError(f"Input SHA-256 mismatch: {path}")
            verified.append(path)
    return verified


def write_software_versions() -> None:
    rows = [
        ("Python", platform.python_version(), sys.executable),
        ("Operating system", platform.platform(), "platform.platform()"),
    ]
    for package in PYTHON_PACKAGES:
        rows.append((package, importlib.metadata.version(package), "Python package"))

    rscript = find_rscript_executable()
    if rscript:
        expression = (
            "cat('R\\t', R.version.string, '\\n', sep=''); "
            "for (p in c('rms','Hmisc','survival')) "
            "cat(p, '\\t', as.character(packageVersion(p)), '\\n', sep='')"
        )
        completed = subprocess.run(
            [rscript, "-e", expression],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in completed.stdout.splitlines():
            if "\t" in line:
                component, version = line.split("\t", maxsplit=1)
                rows.append((component, version, "R runtime/package"))

    with SOFTWARE_VERSIONS.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "version", "source"])
        writer.writerows(rows)


def infer_generator(filename: str) -> str:
    if filename in DIRECT_GENERATORS:
        return DIRECT_GENERATORS[filename]
    if filename.startswith("step_20_"):
        return "not a declared current Step 20 output"
    for prefix, generator in STEP_09_GENERATORS.items():
        if filename.startswith(prefix):
            return generator
    if filename.startswith("step_09_"):
        return "step_09_backward_stepwise.py"
    if filename == "step_11_AUC_curves.pdf":
        return "step_11_auc_curves.py"
    for model in ("lr", "rf", "svm", "xgboost"):
        if filename.startswith(f"step_11_{model}_"):
            return f"step_11_{model}.py"
    match = re.match(r"step_(\d{2})_", filename)
    if not match:
        return "manual or source artifact"
    step = match.group(1)
    candidates = sorted(CODE_DIR.glob(f"step_{step}_*.py")) + sorted(
        CODE_DIR.glob(f"step_{step}_*.r")
    )
    if len(candidates) == 1:
        return candidates[0].name
    return "manual mapping required"


def artifact_role(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        return "machine-readable result/provenance table"
    if path.suffix.lower() == ".tex":
        return "generated manuscript table fragment"
    if path.suffix.lower() == ".pdf":
        return "analysis figure"
    return "analysis artifact"


def discover_artifacts() -> list[Path]:
    artifacts: set[Path] = set()
    for pattern in ("PPMI_[1-4]*.csv", "step_*.csv", "step_*.pdf", "step_*.tex"):
        artifacts.update(CODE_DIR.glob(pattern))
    for name in DIRECT_GENERATORS:
        path = CODE_DIR / name
        if path.is_file():
            artifacts.add(path)
    artifacts.discard(OUTPUT_MANIFEST)
    artifacts.discard(CODE_DIR / "pipeline_run_log.csv")
    # Keep legacy Step 20 files on disk, but do not attribute them to the current script.
    artifacts = {
        path
        for path in artifacts
        if path.name not in HISTORICAL_OUTPUTS
        and (not path.name.startswith("step_20_") or path.name in STEP_20_OUTPUTS)
    }
    return sorted(artifacts, key=lambda item: item.name.lower())


def write_output_manifest() -> int:
    artifacts = discover_artifacts()
    with OUTPUT_MANIFEST.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "artifact_path",
                "generator_script",
                "artifact_role",
                "bytes",
                "sha256",
            ]
        )
        for path in artifacts:
            writer.writerow(
                [
                    path.name,
                    infer_generator(path.name),
                    artifact_role(path),
                    path.stat().st_size,
                    sha256_file(path),
                ]
            )
    return len(artifacts)


def main() -> None:
    verified = verify_inputs()
    write_software_versions()
    artifact_count = write_output_manifest()
    print(f"Verified {len(verified)} declared inputs by byte size and SHA-256.")
    print(
        f"Recorded {artifact_count} reproducibility artifacts in {OUTPUT_MANIFEST.name}."
    )
    print(f"Recorded runtime versions in {SOFTWARE_VERSIONS.name}.")


if __name__ == "__main__":
    main()
