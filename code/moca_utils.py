"""Resolve MoCA sources without rescoring recorded totals or imputing visits.

MCATOT includes the education adjustment in the PPMI MOCA schema; it must not
receive another point. The schema leaves the total blank for ambiguous education.
https://blackfynn-ppmi-schema.readthedocs-hosted.com/en/stable-documentation-branch/documentation_files/cognitive_assessments/tables/MOCA.html
Component sums validate its plausibility; they never replace a missing total.
The download label does not determine which source takes precedence.
"""

import pandas as pd


KEYS = ["PATNO", "EVENT_ID"]
RAW_MOCA_FILENAME = "Montreal_Cognitive_Assessment__MoCA__26Sep2026.csv"
COMPONENT_MAXIMA = dict.fromkeys(
    [
        "MCAALTTM",
        "MCACUBE",
        "MCACLCKC",
        "MCACLCKN",
        "MCACLCKH",
        "MCALION",
        "MCARHINO",
        "MCACAMEL",
        "MCAFDS",
        "MCABDS",
        "MCAVIGIL",
        "MCASER7",
        "MCASNTNC",
        "MCAVF",
        "MCAABSTR",
        "MCAREC1",
        "MCAREC2",
        "MCAREC3",
        "MCAREC4",
        "MCAREC5",
        "MCADATE",
        "MCAMONTH",
        "MCAYR",
        "MCADAY",
        "MCAPLACE",
        "MCACITY",
    ],
    1,
)
COMPONENT_MAXIMA.update(MCASER7=3, MCASNTNC=2, MCAABSTR=2)
RAW_COLUMNS = [*KEYS, "REC_ID", "INFODT", "MCATOT", *COMPONENT_MAXIMA]


def _validate_keys(frame: pd.DataFrame, name: str) -> None:
    missing = set(KEYS) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing key columns: {sorted(missing)}")
    blank = frame[KEYS].astype("string").apply(lambda column: column.str.strip().eq(""))
    if frame[KEYS].isna().any().any() or blank.any().any():
        raise ValueError(f"{name} contains missing PATNO/EVENT_ID keys")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{name} contains duplicate PATNO/EVENT_ID keys")


def prepare_raw_moca(raw: pd.DataFrame) -> pd.DataFrame:
    """Retain raw provenance and flag unusable totals without reconstructing them."""
    _validate_keys(raw, "Raw MoCA")
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Raw MoCA is missing columns: {sorted(missing)}")
    result = raw[KEYS].copy()
    for column in ["REC_ID", "INFODT", "MCATOT", *COMPONENT_MAXIMA]:
        result["moca_raw_" + column] = raw[column]
    result["moca_raw_present"] = True
    components = raw[list(COMPONENT_MAXIMA)].apply(pd.to_numeric, errors="coerce")
    component_valid = pd.DataFrame(
        {
            column: components[column].isin(range(maximum + 1))
            for column, maximum in COMPONENT_MAXIMA.items()
        }
    )
    total = pd.to_numeric(raw["MCATOT"], errors="coerce")
    total_missing = raw["MCATOT"].isna() | raw["MCATOT"].astype(
        "string"
    ).str.strip().eq("").fillna(False)
    component_missing = raw[list(COMPONENT_MAXIMA)].isna().any(axis=1)
    component_missing |= (
        raw[list(COMPONENT_MAXIMA)]
        .astype("string")
        .apply(lambda column: column.str.strip().eq(""))
        .any(axis=1)
    )
    component_sum = components.sum(axis=1, min_count=len(COMPONENT_MAXIMA))
    delta = total - component_sum
    consistent = delta.eq(0) | (delta.eq(1) & component_sum.lt(30))
    reason = pd.Series("usable_recorded_total", index=raw.index, dtype="string")
    reason.loc[~consistent] = "total_component_mismatch"
    reason.loc[~component_valid.all(axis=1)] = "component_invalid"
    reason.loc[component_missing] = "component_missing"
    reason.loc[~total.isin(range(31))] = "total_invalid"
    reason.loc[total_missing] = "total_missing"
    result["moca_raw_recorded"] = total
    result["moca_raw_component_sum"] = component_sum
    result["moca_raw_total_minus_components"] = delta
    result["moca_raw_reason"] = reason
    result["moca_raw_usable"] = total.where(reason.eq("usable_recorded_total"))
    return result


def _education_evidence(curated: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Use same-visit education or an earlier SC/BL record, never a future visit."""
    evidence = keys.copy()
    evidence["moca_education_years"] = float("nan")
    evidence["moca_education_source_event"] = pd.NA
    evidence["moca_education_source"] = "unavailable"
    if "EDUCYRS" not in curated:
        return evidence
    available = curated[[*KEYS, "EDUCYRS"]].copy()
    available["EDUCYRS"] = pd.to_numeric(available["EDUCYRS"], errors="coerce")
    available = available[
        available["EDUCYRS"].ge(0) & available["EDUCYRS"].lt(float("inf"))
    ]
    exact = available.set_index(KEYS)["EDUCYRS"].reindex(pd.MultiIndex.from_frame(keys))
    evidence["moca_education_years"] = exact.to_numpy()
    known = evidence["moca_education_years"].notna()
    evidence.loc[known, "moca_education_source_event"] = evidence.loc[known, "EVENT_ID"]
    evidence.loc[known, "moca_education_source"] = "same_visit"
    followup = (
        evidence["EVENT_ID"]
        .astype("string")
        .str.fullmatch(r"[VR]0*[1-9][0-9]*")
        .fillna(False)
    )
    for event in ("BL", "SC"):
        eligible = followup | (
            evidence["EVENT_ID"].eq("BL") if event == "SC" else False
        )
        values = available.loc[available["EVENT_ID"].eq(event)].set_index("PATNO")[
            "EDUCYRS"
        ]
        fill = evidence["PATNO"].map(values)
        use = evidence["moca_education_years"].isna() & eligible & fill.notna()
        evidence.loc[use, "moca_education_years"] = fill.loc[use]
        evidence.loc[use, "moca_education_source_event"] = event
        evidence.loc[use, "moca_education_source"] = "baseline_" + event
        evidence.loc[use & evidence["EVENT_ID"].eq("BL"), "moca_education_source"] = (
            "earlier_SC"
        )
    return evidence


def resolve_moca_sources(curated: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Resolve the key union: valid curated first, then reliable recorded raw.

    Curated moca is already the processed score. Its 101 unable-to-rate code is
    treated as missing, consistently with Step01; other invalid values fail.
    No education point is added to either source. Available curated education
    provides a cross-check, not proof that a conflicting original total is wrong.
    Such raw conflicts remain unavailable pending source reconciliation; unknown
    education does not invalidate an otherwise plausible recorded total.
    """
    _validate_keys(curated, "Curated MoCA")
    if "moca" not in curated.columns:
        raise ValueError("Curated MoCA is missing moca")
    cur = curated[KEYS].copy()
    score = pd.to_numeric(curated["moca"], errors="coerce")
    score = score.mask(score.eq(101))
    blank = curated["moca"].isna() | curated["moca"].astype("string").str.strip().eq(
        ""
    ).fillna(False)
    invalid = (score.notna() & ~score.isin(range(31))) | (
        score.isna() & ~blank & ~pd.to_numeric(curated["moca"], errors="coerce").eq(101)
    )
    if invalid.any():
        raise ValueError("Curated MoCA contains invalid nonmissing scores")
    cur["moca_curated"] = score
    cur["moca_curated_present"] = True
    result = cur.merge(
        prepare_raw_moca(raw), on=KEYS, how="outer", validate="one_to_one"
    )
    for column in ("moca_curated_present", "moca_raw_present"):
        result[column] = result[column].eq(True)
    result["moca_raw_reason"] = result["moca_raw_reason"].fillna("raw_row_absent")
    result = result.merge(
        _education_evidence(curated, result[KEYS]), on=KEYS, validate="one_to_one"
    )
    result["moca_raw_structural_reason"] = result["moca_raw_reason"]
    education_known = result["moca_education_years"].notna()
    component_sum = result["moca_raw_component_sum"]
    expected = component_sum + (
        result["moca_education_years"].le(12) & component_sum.lt(30)
    ).astype(int)
    result["moca_education_expected_total"] = expected.where(
        education_known & result["moca_raw_usable"].notna()
    )
    checkable = education_known & result["moca_raw_usable"].notna()
    result["moca_education_conflict"] = checkable & result["moca_raw_recorded"].ne(
        expected
    )
    result["moca_education_status"] = "not_checkable"
    result.loc[
        result["moca_raw_usable"].notna() & ~education_known, "moca_education_status"
    ] = "unknown"
    result.loc[checkable, "moca_education_status"] = "consistent"
    result.loc[result["moca_education_conflict"], "moca_education_status"] = "conflict"
    result.loc[result["moca_education_conflict"], "moca_raw_reason"] = (
        "education_conflict"
    )
    result.loc[result["moca_education_conflict"], "moca_raw_usable"] = float("nan")
    result["moca"] = result["moca_curated"].where(
        result["moca_curated"].notna(), result["moca_raw_usable"]
    )
    result["moca_source"] = "missing"
    result.loc[result["moca_raw_usable"].notna(), "moca_source"] = "raw_recorded"
    result.loc[result["moca_curated"].notna(), "moca_source"] = "curated"
    # Source disagreement is independent of eligibility for filling a blank.
    comparable = result["moca_curated"].notna() & result[
        "moca_raw_structural_reason"
    ].eq("usable_recorded_total")
    result["moca_score_disagreement"] = comparable & result["moca_curated"].ne(
        result["moca_raw_recorded"]
    )
    result["moca_threshold_disagreement"] = comparable & result["moca_curated"].lt(
        21
    ).ne(result["moca_raw_recorded"].lt(21))
    return result.sort_values(KEYS, kind="stable").reset_index(drop=True)


def moca_source_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract source metadata; these columns must never become model features."""
    return frame[
        [*KEYS, "moca", *[column for column in frame if column.startswith("moca_")]]
    ].copy()
