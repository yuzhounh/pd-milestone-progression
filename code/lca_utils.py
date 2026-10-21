"""Shared utilities for censoring-aware discrete-time latent class models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from stepmix.stepmix import StepMix


N_INTERVALS = 6
NO_EVENT_CODE = N_INTERVALS + 1
MEASUREMENT = "binary_nan"
DEFAULT_INIT_SEEDS = (42, 142, 242, 342, 442, 542, 642, 742, 842, 942)


@dataclass(frozen=True)
class StableFit:
    model: StepMix
    log_likelihood: float
    best_seed: int
    converged_starts: int
    mean_stability_ari: float
    min_stability_ari: float


def load_lca_input(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Load and validate the annual event interval and follow-up data."""
    frame = pd.read_csv(path, index_col="PATNO")
    time_cols = [column for column in frame.columns if column.startswith("Time_")]
    censor_cols = ["Censor_" + column.removeprefix("Time_") for column in time_cols]

    if not time_cols:
        raise ValueError("No Time_* milestone columns were found.")
    missing_censor = sorted(set(censor_cols) - set(frame.columns))
    if missing_censor:
        raise ValueError(
            f"Domain-specific censoring columns are missing: {missing_censor}"
        )
    if not frame.index.is_unique:
        raise ValueError("PATNO must be unique in the LCA input.")
    if frame[time_cols + censor_cols].isna().any().any():
        raise ValueError(
            "Event intervals and domain-specific censoring must not contain missing values."
        )

    event_values = frame[time_cols].to_numpy(dtype=float)
    if not np.all(event_values == np.floor(event_values)):
        raise ValueError("Time_* values must be integer annual interval codes.")
    if not np.isin(event_values, np.arange(1, NO_EVENT_CODE + 1)).all():
        raise ValueError(f"Time_* values must be in 1..{NO_EVENT_CODE}.")

    censoring = frame[censor_cols].to_numpy(dtype=float)
    if (
        not np.isfinite(censoring).all()
        or ((censoring < 0) | (censoring > N_INTERVALS)).any()
    ):
        raise ValueError(
            f"Censor_* values must be finite and between 0 and {N_INTERVALS} years."
        )
    for time_col, censor_col in zip(time_cols, censor_cols):
        event = frame[time_col].between(1, N_INTERVALS)
        if not np.array_equal(
            np.ceil(frame.loc[event, censor_col].to_numpy(dtype=float)).astype(int),
            frame.loc[event, time_col].to_numpy(dtype=int),
        ):
            raise ValueError(
                f"{time_col} does not match the event horizon in {censor_col}."
            )
    if "max_followup" in frame.columns:
        followup = frame["max_followup"].to_numpy(dtype=float)
        if (
            not np.isfinite(followup).all()
            or ((followup < 0) | (followup > N_INTERVALS)).any()
        ):
            raise ValueError(
                f"max_followup must be finite and between 0 and {N_INTERVALS} years."
            )

    return frame, time_cols


def build_discrete_time_matrix(
    frame: pd.DataFrame,
    time_cols: Iterable[str],
    n_intervals: int = N_INTERVALS,
) -> pd.DataFrame:
    """Create event-history indicators with post-event/censoring cells masked.

    For each milestone and annual interval, 0 denotes remaining event-free through
    that interval and 1 denotes first occurrence in that interval. Intervals after
    the event or after the last observed follow-up are NaN and therefore omitted
    from the StepMix ``binary_nan`` likelihood.
    """
    blocks: list[pd.DataFrame] = []

    for time_col in time_cols:
        event_intervals = frame[time_col].to_numpy(dtype=int)
        censor_col = "Censor_" + time_col.removeprefix("Time_")
        followup = frame[censor_col].to_numpy(dtype=float)
        observed_intervals = np.minimum(
            np.floor(followup + 1e-12).astype(int), n_intervals
        )
        block = np.full((len(frame), n_intervals), np.nan, dtype=float)

        for row_idx, event_interval in enumerate(event_intervals):
            if 1 <= event_interval <= n_intervals:
                block[row_idx, : event_interval - 1] = 0.0
                block[row_idx, event_interval - 1] = 1.0
            elif event_interval == n_intervals + 1:
                block[row_idx, : observed_intervals[row_idx]] = 0.0
            else:
                raise ValueError(
                    f"Unexpected {time_col} interval code {event_interval} for row {row_idx}."
                )

        milestone = time_col.removeprefix("Time_")
        columns = [f"{milestone}_interval_{year}" for year in range(1, n_intervals + 1)]
        blocks.append(pd.DataFrame(block, index=frame.index, columns=columns))

    matrix = pd.concat(blocks, axis=1)
    if matrix.isna().all(axis=1).any():
        affected = matrix.index[matrix.isna().all(axis=1)].tolist()[:5]
        raise ValueError(
            f"Participants without any observable risk interval: {affected}"
        )
    return matrix


def calculate_entropy(probabilities: np.ndarray) -> float:
    """Return normalized classification entropy, where 1 is perfect separation."""
    n_components = probabilities.shape[1]
    if n_components == 1:
        return 1.0
    clipped = np.clip(probabilities, 1e-15, 1.0)
    return float(
        1.0 + np.sum(clipped * np.log(clipped)) / (len(clipped) * np.log(n_components))
    )


def fit_stable_model(
    matrix: pd.DataFrame,
    n_components: int,
    seeds: Iterable[int] = DEFAULT_INIT_SEEDS,
    max_iter: int = 1000,
) -> StableFit:
    """Fit independent random starts and retain the maximum-likelihood solution."""
    fitted: list[tuple[float, int, StepMix, np.ndarray]] = []

    for seed in seeds:
        model = StepMix(
            n_components=n_components,
            measurement=MEASUREMENT,
            n_init=1,
            max_iter=max_iter,
            random_state=seed,
            verbose=0,
            progress_bar=0,
        )
        model.fit(matrix)
        log_likelihood = float(model.score(matrix) * len(matrix))
        fitted.append((log_likelihood, seed, model, model.predict(matrix).astype(int)))

    fitted.sort(key=lambda item: item[0], reverse=True)
    best_log_likelihood, best_seed, best_model, best_labels = fitted[0]
    stability = [adjusted_rand_score(best_labels, item[3]) for item in fitted]

    return StableFit(
        model=best_model,
        log_likelihood=best_log_likelihood,
        best_seed=int(best_seed),
        converged_starts=sum(bool(item[2].converged_) for item in fitted),
        mean_stability_ari=float(np.mean(stability)),
        min_stability_ari=float(np.min(stability)),
    )


def class_expected_event_codes(
    model: StepMix,
    time_cols: Iterable[str],
    n_intervals: int = N_INTERVALS,
) -> pd.DataFrame:
    """Compute model-implied event-time expectations, including no event as year 7."""
    time_cols = list(time_cols)
    hazards = np.asarray(model.get_parameters()["measurement"]["pis"], dtype=float)
    expected_shape = (model.n_components, len(time_cols) * n_intervals)
    if hazards.shape != expected_shape:
        raise ValueError(
            f"Unexpected hazard parameter shape {hazards.shape}; expected {expected_shape}."
        )

    rows: list[dict[str, float]] = []
    for class_idx in range(model.n_components):
        row: dict[str, float] = {"Class": float(class_idx)}
        total = 0.0
        for milestone_idx, time_col in enumerate(time_cols):
            start = milestone_idx * n_intervals
            class_hazards = np.clip(
                hazards[class_idx, start : start + n_intervals], 0.0, 1.0
            )
            survival = 1.0
            expectation = 0.0
            for interval, hazard in enumerate(class_hazards, start=1):
                event_probability = survival * hazard
                expectation += interval * event_probability
                survival *= 1.0 - hazard
            expectation += (n_intervals + 1) * survival
            row[time_col] = float(expectation)
            total += expectation
        row["Expected_code_sum"] = float(total)
        rows.append(row)

    result = pd.DataFrame(rows)
    result["Class"] = result["Class"].astype(int)
    return result


def subtype_map_from_model(model: StepMix, time_cols: Iterable[str]) -> dict[int, str]:
    """Name the lower expected event-time class Rapid and the higher one Slow."""
    if model.n_components != 2:
        raise ValueError(
            "Rapid/Slow names are defined only for the prespecified two-class model."
        )
    expectations = class_expected_event_codes(model, time_cols).set_index("Class")
    rapid_label = int(expectations["Expected_code_sum"].idxmin())
    slow_label = int(expectations["Expected_code_sum"].idxmax())
    return {rapid_label: "Rapid", slow_label: "Slow"}


def build_posterior_table(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    model: StepMix,
    time_cols: Iterable[str],
) -> pd.DataFrame:
    """Append modal class, subtype name, and posterior probabilities to the input."""
    labels = model.predict(matrix).astype(int)
    probabilities = model.predict_proba(matrix)
    subtype_map = subtype_map_from_model(model, time_cols)

    posterior = frame.copy()
    posterior["Subtype_Label"] = labels
    posterior["Subtype"] = [subtype_map[int(label)] for label in labels]
    posterior["Subtype_Prob"] = probabilities.max(axis=1)
    for class_idx in range(probabilities.shape[1]):
        class_name = subtype_map.get(class_idx, f"Class_{class_idx}")
        posterior[f"Posterior_Class_{class_idx}_{class_name}"] = probabilities[
            :, class_idx
        ]
    return posterior


def subtype_posterior_column(frame: pd.DataFrame, subtype: str) -> str:
    """Return the unique saved posterior column for a named subtype."""
    suffix = "_" + subtype
    candidates = [
        column
        for column in frame.columns
        if column.startswith("Posterior_Class_") and column.endswith(suffix)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one posterior column for {subtype}, found {candidates}"
        )
    return candidates[0]
