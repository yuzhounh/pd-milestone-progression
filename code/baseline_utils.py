"""Deterministic baseline selection and verified source corrections."""

import math
from pathlib import Path

import pandas as pd

from moca_utils import KEYS, resolve_moca_sources


BASELINE_SOURCE_COLUMN = "baseline_source_event"
BASELINE_EVENT_PRIORITY = {"BL": 0, "SC": 1}
CURATED_FILENAME = "PPMI_Curated_Data_Cut_Public_20260810.xlsx"
VERIFIED_CURATED_SOURCE = CURATED_FILENAME
CURATED_COLUMN_ALIASES = {
    "MIA_CAUDATE_BILAT": "MIA_CAUDATE_mean",
    "MIA_PUTAMEN_BILAT": "MIA_PUTAMEN_mean",
    "MIA_STRIATUM_BILAT": "MIA_STRIATUM_mean",
    "MIA_LOWPUT_EXPECTED": "mia_lowput_expected",
}
BMI_ORIGINAL_COLUMN = "BMI_source_value"
BMI_CORRECTION_METADATA = {
    "BMI_correction_rule": "invalid_height_137482_BL_v1",
    "BMI_correction_source_version": VERIFIED_CURATED_SOURCE,
    "BMI_correction_evidence": (
        "Vital_Signs_26Sep2026.csv; REC_ID=IA66449; PATNO=137482; "
        "EVENT_ID=BL; HTCM=5 cm; WGTKG=140.5 kg"
    ),
    "BMI_correction_reason": "Invalid source height; true baseline BMI unknown",
}
BMI_AUDIT_COLUMNS = [BMI_ORIGINAL_COLUMN, *BMI_CORRECTION_METADATA]


def normalize_curated_columns(curated: pd.DataFrame) -> pd.DataFrame:
    """Map renamed 20260810 curated columns to stable analysis names."""
    conflicts = [
        (source, target)
        for source, target in CURATED_COLUMN_ALIASES.items()
        if source in curated.columns and target in curated.columns
    ]
    if conflicts:
        raise ValueError(f"Curated data contain ambiguous renamed columns: {conflicts}")
    aliases = {
        source: target
        for source, target in CURATED_COLUMN_ALIASES.items()
        if source in curated.columns
    }
    return curated.rename(columns=aliases)


def load_curated_data(path=CURATED_FILENAME) -> pd.DataFrame:
    """Load the current curated workbook using stable downstream column names."""
    return normalize_curated_columns(pd.read_excel(path))


def select_baseline_records(curated: pd.DataFrame) -> pd.DataFrame:
    """Select one baseline row per participant, preferring BL over SC.

    Screening (SC) is used only when an enrolment baseline (BL) row is absent.
    Duplicate rows within the same participant and source visit are rejected so
    that selection never depends on source-file row order.
    """
    required = {"PATNO", "EVENT_ID"}
    missing = required - set(curated.columns)
    if missing:
        raise ValueError(
            f"Curated data are missing baseline-selection columns: {sorted(missing)}"
        )

    eligible = curated.loc[curated["EVENT_ID"].isin(BASELINE_EVENT_PRIORITY)].copy()
    if eligible.empty:
        raise ValueError("Curated data contain no BL or SC records")
    if eligible["PATNO"].isna().any():
        raise ValueError("BL/SC records contain missing PATNO values")

    duplicate_mask = eligible.duplicated(["PATNO", "EVENT_ID"], keep=False)
    if duplicate_mask.any():
        duplicate_keys = (
            eligible.loc[duplicate_mask, ["PATNO", "EVENT_ID"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Duplicate participant/source baseline rows require an explicit "
            f"resolution rule; examples: {duplicate_keys}"
        )

    eligible["_baseline_priority"] = eligible["EVENT_ID"].map(BASELINE_EVENT_PRIORITY)
    selected = (
        eligible.sort_values(["PATNO", "_baseline_priority"], kind="stable")
        .drop_duplicates("PATNO", keep="first")
        .drop(columns="_baseline_priority")
    )
    selected[BASELINE_SOURCE_COLUMN] = selected["EVENT_ID"]
    return selected


def prepare_baseline_records(
    curated: pd.DataFrame, *, source_path, raw_moca=None
) -> pd.DataFrame:
    """Select BL/SC records, apply verified BMI correction and optional MoCA fill.

    Only PATNO 137482 / BL / BMI 56200 in the audited curated release is
    corrected. Original values and correction provenance remain in dedicated
    columns. The filename is a release label, not a workbook-content hash.
    A different release label, unexpected old value, or incomplete provenance
    raises instead of overwriting a potentially corrected source. Reapplying
    without raw_moca is safe and does not lose the original BMI.
    With raw_moca supplied, the shared longitudinal source rule fills only the
    already-selected BL/SC key. It never substitutes SC for a missing BL score
    or borrows scores from later visits. Complete curated education evidence is
    supplied to the shared helper before selecting keys. Re-resolving MoCA from
    already-selected output is rejected because it may have lost earlier SC
    education evidence; pass the original full curated input. APOE is unchanged.
    """
    if Path(source_path).name != VERIFIED_CURATED_SOURCE:
        raise ValueError(
            "Verified BMI correction requires curated source "
            f"{VERIFIED_CURATED_SOURCE}; got {Path(source_path).name}"
        )
    resolved_moca_markers = {
        "moca_source",
        "moca_curated",
        "moca_raw_recorded",
        "moca_raw_reason",
    }
    if raw_moca is not None and resolved_moca_markers.intersection(curated.columns):
        raise ValueError(
            "MoCA resolution requires original full curated rows, not previously resolved output"
        )
    selected = select_baseline_records(curated)
    if "BMI" not in selected.columns:
        raise ValueError("Baseline data are missing BMI for source verification")

    present_audit_columns = set(BMI_AUDIT_COLUMNS) & set(selected.columns)
    if present_audit_columns and present_audit_columns != set(BMI_AUDIT_COLUMNS):
        raise ValueError("Incomplete BMI correction provenance columns")
    if not present_audit_columns:
        for column in BMI_AUDIT_COLUMNS:
            selected[column] = pd.Series(pd.NA, index=selected.index, dtype=object)

    target = pd.to_numeric(selected["PATNO"], errors="coerce").eq(137482) & selected[
        "EVENT_ID"
    ].eq("BL")
    if selected.loc[~target, BMI_AUDIT_COLUMNS].notna().any().any():
        raise ValueError("BMI correction provenance appears on an unaudited record")

    for row_index in selected.index[target]:
        bmi = selected.at[row_index, "BMI"]
        source_value = selected.at[row_index, BMI_ORIGINAL_COLUMN]
        prior_metadata = selected.loc[row_index, list(BMI_CORRECTION_METADATA)]
        already_corrected = (
            pd.isna(bmi)
            and _matches_verified_bmi(source_value)
            and all(
                pd.notna(prior_metadata[column]) and prior_metadata[column] == value
                for column, value in BMI_CORRECTION_METADATA.items()
            )
        )
        if already_corrected:
            continue
        if not _matches_verified_bmi(bmi):
            raise ValueError(
                "PATNO 137482 / BL BMI differs from audited value 56200; "
                "review source data before applying the correction"
            )
        if pd.notna(source_value) or prior_metadata.notna().any():
            raise ValueError(
                "Conflicting BMI correction provenance for PATNO 137482 / BL"
            )

        selected.at[row_index, BMI_ORIGINAL_COLUMN] = bmi
        selected.at[row_index, "BMI"] = float("nan")
        for column, value in BMI_CORRECTION_METADATA.items():
            selected.at[row_index, column] = value
    if raw_moca is not None:
        resolved = resolve_moca_sources(curated, raw_moca).set_index(KEYS)
        aligned = resolved.reindex(pd.MultiIndex.from_frame(selected[KEYS]))
        for column in aligned.columns:
            selected[column] = aligned[column].to_numpy()
    return selected


def _matches_verified_bmi(value) -> bool:
    """Allow floating-point storage error without accepting a different value."""
    if pd.isna(value):
        return False
    try:
        return math.isclose(float(value), 56200.0, rel_tol=0.0, abs_tol=1e-8)
    except (TypeError, ValueError):
        return False


def baseline_correction_audit(baseline: pd.DataFrame) -> pd.DataFrame:
    """Return the applied BMI correction records for a step's audit CSV."""
    return baseline.loc[
        baseline["BMI_correction_rule"].notna(),
        ["PATNO", BASELINE_SOURCE_COLUMN, "BMI", *BMI_AUDIT_COLUMNS],
    ].copy()
