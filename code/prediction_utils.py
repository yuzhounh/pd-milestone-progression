"""Validation and patient-level alignment for saved model predictions."""

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_PREDICTION_COLUMNS = {
    "PATNO",
    "split",
    "model_version",
    "y_true",
    "y_prob_raw",
    "y_prob",
    "calibration_method",
}


def load_aligned_prediction_frames(model_files):
    """Load prediction files and align every model to the same PATNO order."""
    frames = {}
    reference_labels = None

    for model, path in model_files.items():
        path = Path(path)
        frame = pd.read_csv(path)
        missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        if frame["PATNO"].isna().any() or frame["PATNO"].duplicated().any():
            raise ValueError(
                f"{path.name} must contain unique, non-missing PATNO values"
            )
        if set(frame["split"].dropna().unique()) != {"test"}:
            raise ValueError(f"{path.name} must contain only split='test' rows")
        if frame["model_version"].isna().any():
            raise ValueError(f"{path.name} contains missing model_version values")

        frame["y_true"] = frame["y_true"].astype(int)
        for column in ("y_prob_raw", "y_prob"):
            frame[column] = frame[column].astype(float)
            values = frame[column].to_numpy()
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError(f"{path.name} contains invalid {column} values")

        indexed = frame.set_index("PATNO", verify_integrity=True)
        if reference_labels is None:
            reference_labels = indexed["y_true"].copy()
        else:
            missing_patients = reference_labels.index.difference(indexed.index)
            extra_patients = indexed.index.difference(reference_labels.index)
            if len(missing_patients) or len(extra_patients):
                raise ValueError(
                    f"{path.name} PATNO mismatch: missing={len(missing_patients)}, "
                    f"extra={len(extra_patients)}"
                )
            indexed = indexed.loc[reference_labels.index]
            if not np.array_equal(
                indexed["y_true"].to_numpy(), reference_labels.to_numpy()
            ):
                raise ValueError(
                    f"{path.name} y_true values disagree after PATNO alignment"
                )

        frames[model] = indexed.reset_index()

    return frames
