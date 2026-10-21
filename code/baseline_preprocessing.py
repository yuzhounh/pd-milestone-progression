"""Leakage-safe baseline preprocessing with nominal race encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


RACE_COLUMN = "race"
RACE_LEVELS = {1: "White", 2: "Black", 3: "Asian", 4: "Other"}
RACE_REFERENCE = 1
RACE_DUMMY_COLUMNS = [
    f"race_{RACE_LEVELS[code]}" for code in RACE_LEVELS if code != RACE_REFERENCE
]


@dataclass(frozen=True)
class PreprocessedBaseline:
    train_filled: pd.DataFrame
    validation_filled: pd.DataFrame
    train_scaled: pd.DataFrame
    validation_scaled: pd.DataFrame
    continuous_columns: tuple[str, ...]
    discrete_columns: tuple[str, ...]
    race_imputation_code: int | None


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    converted = frame.apply(pd.to_numeric, errors="coerce")
    newly_missing = converted.isna() & frame.notna()
    if newly_missing.any().any():
        affected = sorted(newly_missing.columns[newly_missing.any()].tolist())
        raise ValueError(f"Non-numeric baseline values found in: {affected}")
    return converted


def _fill_columns(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = pd.DataFrame(index=train.index)
    validation_out = pd.DataFrame(index=validation.index)
    for column in columns:
        observed = train[column].dropna()
        if observed.empty:
            raise ValueError(f"Training fold has no observed values for {column}")
        if strategy == "median":
            fill_value = float(observed.median())
        elif strategy == "mode":
            fill_value = observed.mode(dropna=True).iloc[0]
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")
        train_out[column] = train[column].fillna(fill_value)
        validation_out[column] = validation[column].fillna(fill_value)
    return train_out, validation_out


def _encode_race(
    train: pd.Series,
    validation: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    allowed = set(RACE_LEVELS)
    for name, values in (("training", train), ("validation", validation)):
        invalid = sorted(set(values.dropna().astype(int)) - allowed)
        non_integer = values.dropna().ne(np.floor(values.dropna())).any()
        if invalid or non_integer:
            raise ValueError(f"Invalid race codes in {name} data: {invalid}")
    observed = train.dropna().astype(int)
    if observed.empty:
        raise ValueError("Training fold has no observed race values")
    mode = int(observed.mode().iloc[0])
    train_filled = train.fillna(mode).astype(int)
    validation_filled = validation.fillna(mode).astype(int)
    train_encoded = pd.DataFrame(index=train.index)
    validation_encoded = pd.DataFrame(index=validation.index)
    for code, label in RACE_LEVELS.items():
        if code == RACE_REFERENCE:
            continue
        column = f"race_{label}"
        train_encoded[column] = train_filled.eq(code).astype(float)
        validation_encoded[column] = validation_filled.eq(code).astype(float)
    return train_encoded, validation_encoded, mode


def fit_baseline_preprocessor(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> PreprocessedBaseline:
    """Fit imputation, race encoding, and scaling on ``train`` only."""
    if list(train.columns) != list(validation.columns):
        raise ValueError("Training and validation baseline columns must match in order")
    train = _numeric(train.copy())
    validation = _numeric(validation.copy())

    ordinary = [column for column in train.columns if column != RACE_COLUMN]
    discrete = [
        column for column in ordinary if train[column].nunique(dropna=True) <= 10
    ]
    continuous = [column for column in ordinary if column not in discrete]
    train_cont, validation_cont = _fill_columns(train, validation, continuous, "median")
    train_disc, validation_disc = _fill_columns(train, validation, discrete, "mode")

    train_filled = pd.concat([train_cont, train_disc], axis=1)
    validation_filled = pd.concat([validation_cont, validation_disc], axis=1)
    race_mode: int | None = None
    if RACE_COLUMN in train.columns:
        train_race, validation_race, race_mode = _encode_race(
            train[RACE_COLUMN], validation[RACE_COLUMN]
        )
        train_filled = pd.concat([train_filled, train_race], axis=1)
        validation_filled = pd.concat([validation_filled, validation_race], axis=1)

    output_order: list[str] = []
    for column in train.columns:
        if column == RACE_COLUMN:
            output_order.extend(RACE_DUMMY_COLUMNS)
        else:
            output_order.append(column)
    train_filled = train_filled[output_order].astype(float)
    validation_filled = validation_filled[output_order].astype(float)

    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_filled), index=train.index, columns=output_order
    )
    validation_scaled = pd.DataFrame(
        scaler.transform(validation_filled),
        index=validation.index,
        columns=output_order,
    )
    return PreprocessedBaseline(
        train_filled=train_filled,
        validation_filled=validation_filled,
        train_scaled=train_scaled,
        validation_scaled=validation_scaled,
        continuous_columns=tuple(continuous),
        discrete_columns=tuple(discrete),
        race_imputation_code=race_mode,
    )


def race_encoding_audit(result: PreprocessedBaseline) -> pd.DataFrame:
    """Return a compact, non-participant-level description of race encoding."""
    if result.race_imputation_code is None:
        return pd.DataFrame(
            [{"source_feature": RACE_COLUMN, "status": "not_in_candidate_set"}]
        )
    rows = []
    for code, label in RACE_LEVELS.items():
        rows.append(
            {
                "source_feature": RACE_COLUMN,
                "source_code": code,
                "source_label": label,
                "reference": code == RACE_REFERENCE,
                "output_feature": (
                    "reference" if code == RACE_REFERENCE else f"race_{label}"
                ),
                "training_imputation_code": result.race_imputation_code,
                "status": "encoded",
            }
        )
    return pd.DataFrame(rows)
