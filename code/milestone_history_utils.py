"""Shared milestone-state and observation-history utilities.

The production rules are deliberately explicit:

* milestone status is nullable (positive, negative, or unknown);
* R06/R08/R10/R12 are subannual visits at 2.5/3.5/4.5/5.5 years;
* ST timing comes from Part III ``INFODT`` relative to Part III baseline;
* ST can contribute a positive event or treatment evidence, but an ST negative
  never substitutes for a scheduled annual visit or extends censoring;
* only missing/unknown scheduled annual checkpoints create a permanent gap;
* censoring is calculated separately for every milestone domain.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


KEYS = ["PATNO", "EVENT_ID"]
DOMAINS = ("Walking", "MotorComp", "Cognition", "Autonomic", "Functional", "ADL")
PARTIV = ["NP4WDYSK", "NP4DYSKI", "NP4FLCTI", "NP4FLCTX"]
VISIT_TIMES = {
    "SC": -1.0,
    "BL": 0.0,
    "V01": 0.25,
    "V02": 0.5,
    "V03": 0.75,
    "V04": 1.0,
    "V05": 1.5,
    "V06": 2.0,
    "V07": 2.5,
    "R06": 2.5,
    "V08": 3.0,
    "V09": 3.5,
    "R08": 3.5,
    "V10": 4.0,
    "V11": 4.5,
    "R10": 4.5,
    "V12": 5.0,
    "R12": 5.5,
    "V13": 6.0,
}
ANNUAL_VISITS = {1: "V04", 2: "V06", 3: "V08", 4: "V10", 5: "V12", 6: "V13"}


def collapse_identical_visit_rows(
    frame: pd.DataFrame,
    value_columns: list[str],
    source_name: str,
) -> pd.DataFrame:
    """Collapse duplicate visit keys only when all analysis values agree."""

    duplicate_mask = frame.duplicated(KEYS, keep=False)
    if not duplicate_mask.any():
        return frame.copy()

    duplicate_rows = frame.loc[duplicate_mask, KEYS + value_columns]
    conflicts = (
        duplicate_rows.groupby(KEYS, dropna=False)[value_columns]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if conflicts.any():
        examples = [
            dict(zip(KEYS, key)) for key in conflicts[conflicts].index.tolist()[:10]
        ]
        raise ValueError(
            f"{source_name} duplicate visit rows disagree on analysis values; "
            f"examples: {examples}"
        )

    return frame.drop_duplicates(KEYS, keep="first").copy()


def _compare(series: pd.Series, operator: str, threshold: float) -> pd.Series:
    return (
        getattr(pd.to_numeric(series, errors="coerce"), operator)(threshold)
        .where(pd.to_numeric(series, errors="coerce").notna())
        .astype("boolean")
    )


def milestone_states(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate the six milestone domains with nullable Boolean logic."""

    def ge(name: str, value: float) -> pd.Series:
        return _compare(frame[name], "ge", value)

    def eq(name: str, value: float) -> pd.Series:
        return _compare(frame[name], "eq", value)

    return pd.DataFrame(
        {
            "Walking": ge("NP2WALK", 3)
            | ge("NP2FREZ", 3)
            | ge("NP3GAIT", 3)
            | eq("NP3FRZGT", 4)
            | ge("NP3PSTBL", 3)
            | ge("NHY", 4),
            "MotorComp": (ge("NP4WDYSK", 3) & ge("NP4DYSKI", 3))
            | ge("NP4FLCTI", 3)
            | ge("NP4FLCTX", 3),
            "Cognition": _compare(frame["moca"], "lt", 21)
            | ge("NP1COG", 3)
            | ge("NP1HALL", 3)
            | ge("NP1APAT", 3),
            "Autonomic": (ge("NP1URIN", 3) & (ge("SCAU8", 2) | ge("SCAU9", 2)))
            | (ge("SCAU15", 2) & (ge("Sys_Drop", 20) | ge("Dia_Drop", 10)))
            | eq("NP1LTHD", 4)
            | ge("SCAU16", 1),
            "Functional": _compare(frame["MSEADLG"], "lt", 80),
            "ADL": ge("NP2SWAL", 3)
            | ge("NP2EAT", 3)
            | ge("NP2DRES", 3)
            | ge("NP2HYGN", 3)
            | ge("NP3SPCH", 3),
        },
        index=frame.index,
    )


def resolve_visit_times(source: pd.DataFrame, part3: pd.DataFrame) -> pd.Series:
    """Return analysis time in years, using Part III dates only for ST.

    Part III can contain ON/OFF rows at one visit. A usable BL or ST key must
    therefore have exactly one nonmissing month across all of its rows. ST rows
    without an unambiguous paired Part III baseline, and ST rows outside the
    six-year window, are returned as missing and excluded from histories.
    """
    if source.duplicated(KEYS).any():
        raise ValueError("Milestone source has duplicate PATNO/EVENT_ID keys.")
    required = {*KEYS, "INFODT"}
    if not required.issubset(part3.columns):
        raise ValueError(
            f"Part III lacks ST timing columns: {sorted(required - set(part3.columns))}"
        )

    result = source["EVENT_ID"].map(VISIT_TIMES).astype(float)
    timing = part3.loc[part3.EVENT_ID.isin(["BL", "ST"]), KEYS + ["INFODT"]].copy()
    timing["date"] = pd.to_datetime(timing["INFODT"], format="%m/%Y", errors="coerce")
    grouped = timing.groupby(KEYS, sort=False).date.agg(["count", "nunique", "min"])
    unique = grouped.loc[
        grouped["count"].gt(0) & grouped["nunique"].eq(1), "min"
    ].unstack()
    if "BL" not in unique or "ST" not in unique:
        return result
    elapsed_months = (
        (unique["ST"].dt.year - unique["BL"].dt.year) * 12
        + unique["ST"].dt.month
        - unique["BL"].dt.month
    )
    st_years = (elapsed_months / 12.0).where(elapsed_months.between(1, 72))
    st_mask = source.EVENT_ID.eq("ST")
    result.loc[st_mask] = source.loc[st_mask, "PATNO"].map(st_years)
    return result


def _tokens(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def partiv_structural_negatives(
    source: pd.DataFrame,
    part3: pd.DataFrame,
    part4: pd.DataFrame,
    curated: pd.DataFrame,
    visit_time: pd.Series,
) -> pd.DataFrame:
    """Identify conservative visit-level Part IV structural negatives.

    A candidate needs an explicit current raw ``PDTRTMNT=0``, no contradictory
    treatment flag, no current or earlier recorded treatment/complication
    evidence, and an absent/blank/partially blank Part IV form without UR or an
    invalid value. Curated treatment fields provide positive evidence only.
    """
    if not visit_time.index.equals(source.index):
        raise ValueError("Visit time must be exactly aligned to the milestone source.")
    base = source[KEYS + PARTIV].copy()
    base["time"] = visit_time
    base = base.loc[base.time.notna() & base.time.between(-1, 6)].copy()
    base_index = pd.MultiIndex.from_frame(base[KEYS])

    raw4 = part4.loc[
        part4.EVENT_ID.isin(set(VISIT_TIMES) | {"ST"}), KEYS + PARTIV
    ].copy()
    if raw4.duplicated(KEYS).any():
        raise ValueError("Raw Part IV has duplicate visit keys.")
    raw4 = raw4.set_index(KEYS)
    tokens4 = raw4[PARTIV].apply(_tokens)
    numbers4 = tokens4.apply(pd.to_numeric, errors="coerce")
    blank4 = tokens4.isna() | tokens4.eq("").fillna(False)
    ur4 = (tokens4.eq("UR") | numbers4.eq(101)).fillna(False)
    valid4 = numbers4.isin([0, 1, 2, 3, 4])
    invalid4 = ~blank4 & ~ur4 & ~valid4
    flags4 = pd.DataFrame(
        {
            "partiv_missing_form": False,
            "partiv_blank_or_partial": blank4.any(axis=1),
            "partiv_ur": ur4.any(axis=1),
            "partiv_invalid": invalid4.any(axis=1),
            "partiv_nonzero": (valid4 & numbers4.gt(0)).any(axis=1),
        },
        index=raw4.index,
    )

    raw3 = part3.loc[part3.EVENT_ID.isin(set(VISIT_TIMES) | {"ST"})].copy()
    required3 = {*KEYS, "PDTRTMNT", "PDMEDYN", "DBSYN", "PDSTATE"}
    if not required3.issubset(raw3.columns):
        raise ValueError(
            f"Part III lacks treatment columns: {sorted(required3 - set(raw3.columns))}"
        )
    evidence = raw3[KEYS].copy()
    for column in ("PDTRTMNT", "PDMEDYN", "DBSYN"):
        tokens = _tokens(raw3[column])
        numbers = pd.to_numeric(tokens, errors="coerce")
        present = tokens.notna() & tokens.ne("")
        evidence[column + "_yes"] = numbers.eq(1)
        evidence[column + "_no"] = numbers.eq(0)
        evidence[column + "_invalid"] = present & ~numbers.isin([0, 1])
    evidence["onoff"] = _tokens(raw3.PDSTATE).isin(["ON", "OFF"])
    flags3 = evidence.groupby(KEYS, sort=False).any()
    flags3["explicit_no_treatment"] = flags3.PDTRTMNT_no & ~flags3.PDTRTMNT_yes
    flags3["treatment_invalid"] = flags3.filter(regex="_invalid$").any(axis=1)
    flags3["treatment_evidence"] = flags3[
        ["PDTRTMNT_yes", "PDMEDYN_yes", "DBSYN_yes", "onoff"]
    ].any(axis=1)

    curated_work = curated[KEYS].copy()
    curated_work["curated_evidence"] = False
    if "PDTRTMNT" in curated:
        curated_work["curated_evidence"] |= pd.to_numeric(
            curated.PDTRTMNT, errors="coerce"
        ).eq(1)
    if "LEDD" in curated:
        ledd = pd.to_numeric(curated.LEDD, errors="coerce")
        curated_work["curated_evidence"] |= ledd.gt(0) & np.isfinite(ledd)
    curated_flags = curated_work.groupby(KEYS, sort=False).any()

    flags = pd.DataFrame(index=base_index).join(flags3).join(flags4).join(curated_flags)
    flags["partiv_missing_form"] = ~flags.index.isin(flags4.index)
    bool_columns = flags.columns
    flags[bool_columns] = (
        flags[bool_columns].astype("boolean").fillna(False).astype(bool)
    )
    cleaned = base.set_index(KEYS)[PARTIV].apply(pd.to_numeric, errors="coerce")
    flags["partiv_nonzero"] |= (cleaned.isin([0, 1, 2, 3, 4]) & cleaned.gt(0)).any(
        axis=1
    )
    flags["current_evidence"] = (
        flags.treatment_evidence | flags.curated_evidence | flags.partiv_nonzero
    )

    timeline = base[KEYS + ["time"]].set_index(KEYS).join(flags)
    at_time = timeline.groupby(["PATNO", "time"], sort=True).current_evidence.any()
    prior = at_time.groupby(level="PATNO").transform(
        lambda values: values.shift(fill_value=False).cummax()
    )
    time_evidence = at_time.rename("time_evidence").reset_index()
    time_evidence["prior_evidence"] = prior.to_numpy(dtype=bool)
    timeline = timeline.reset_index().merge(
        time_evidence, on=["PATNO", "time"], validate="many_to_one"
    )
    timeline["structural_negative"] = (
        timeline.explicit_no_treatment
        & ~timeline.treatment_invalid
        & ~timeline.time_evidence
        & ~timeline.prior_evidence
        & (timeline.partiv_missing_form | timeline.partiv_blank_or_partial)
        & ~timeline.partiv_ur
        & ~timeline.partiv_invalid
    )
    result = timeline.set_index(KEYS)[
        ["structural_negative", "time_evidence", "prior_evidence"]
    ].reindex(pd.MultiIndex.from_frame(source[KEYS]))
    result.index = source.index
    return result.eq(True)


def build_domain_history(
    frame: pd.DataFrame,
    status: pd.Series,
    domain: str,
) -> pd.DataFrame:
    """Build first-event and domain-specific censoring histories.

    ``frame`` must contain unique PATNO/EVENT_ID keys and an analysis ``time``.
    Every participant needs a known-negative BL status. Missing/unknown annual
    visits truncate the main history only when a later observation or positive
    event makes that checkpoint relevant.
    """
    required = {*KEYS, "time"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"History frame lacks columns: {sorted(required - set(frame.columns))}"
        )
    if frame.duplicated(KEYS).any():
        raise ValueError("History frame has duplicate PATNO/EVENT_ID keys.")
    if not status.index.equals(frame.index):
        raise ValueError("Domain status must be exactly index-aligned.")
    if not pd.api.types.is_bool_dtype(status.dtype):
        raise ValueError("Domain status must use Boolean or nullable Boolean dtype.")

    work = frame[KEYS + ["time"]].copy()
    work["status"] = status
    rows: list[dict[str, object]] = []
    for identifier, visits in work.groupby("PATNO", sort=True):
        visits = visits.sort_values(["time", "EVENT_ID"], kind="stable")
        baseline = visits.loc[visits.EVENT_ID.eq("BL"), "status"]
        if len(baseline) != 1 or baseline.isna().any() or bool(baseline.iloc[0]):
            raise ValueError(
                "Every history participant must have one known-negative BL."
            )

        positive = visits.loc[visits.status.eq(True).fillna(False)]
        first_positive_time = float(positive.time.min()) if len(positive) else np.nan
        first_positive_visits = (
            "|".join(
                sorted(
                    positive.loc[
                        positive.time.eq(first_positive_time), "EVENT_ID"
                    ].astype(str)
                )
            )
            if np.isfinite(first_positive_time)
            else ""
        )
        event_boundary = (
            first_positive_time if np.isfinite(first_positive_time) else np.inf
        )
        max_time = float(visits.time.max())
        by_event = visits.set_index("EVENT_ID").status
        first_gap = np.nan
        gap_reason = ""
        for year, event_id in ANNUAL_VISITS.items():
            if year > max_time or year >= event_boundary:
                break
            if event_id not in by_event.index:
                first_gap, gap_reason = float(year), "absent_visit"
                break
            if pd.isna(by_event.loc[event_id]):
                first_gap, gap_reason = float(year), "unknown_state"
                break

        retained_event = bool(
            np.isfinite(first_positive_time) and not np.isfinite(first_gap)
        )
        negatives = visits.loc[visits.status.eq(False).fillna(False), "time"]
        if np.isfinite(first_gap):
            negatives = negatives.loc[negatives < first_gap]
        censor_time = first_positive_time if retained_event else float(negatives.max())
        rows.append(
            {
                "PATNO": identifier,
                "domain": domain,
                "event": retained_event,
                "event_time": first_positive_time if retained_event else np.nan,
                "censor_time": censor_time,
                "first_gap_year": first_gap,
                "first_gap_reason": gap_reason,
                "reference_first_positive": first_positive_time,
                "first_positive_visits": first_positive_visits,
                "event_lost_after_gap": bool(
                    np.isfinite(first_positive_time) and not retained_event
                ),
                "event_interval": (
                    math.ceil(first_positive_time) if retained_event else 7
                ),
                "observable_cells": (
                    math.ceil(first_positive_time)
                    if retained_event
                    else math.floor(censor_time + 1e-12)
                ),
            }
        )
    return pd.DataFrame(rows).set_index("PATNO")


def mask_st_negatives(frame: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """Keep ST positives and unknowns while preventing ST negatives from censoring."""
    result = states.copy()
    st = frame.EVENT_ID.eq("ST")
    for domain in result:
        result.loc[st & result[domain].eq(False).fillna(False), domain] = pd.NA
    return result
