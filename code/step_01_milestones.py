"""Step 01: build visit-level clinical milestone indicators from the raw PPMI tables.

Cleans special codes, aggregates each source to one record per visit, joins the
sources on PATNO/EVENT_ID and derives the six milestone domains (walking and
balance, motor complications, cognition, autonomic dysfunction, functional
dependence, activities of daily living). Output: PPMI_1_milestones.csv.
"""

import pandas as pd
import numpy as np

from baseline_utils import CURATED_FILENAME, load_curated_data
from milestone_history_utils import collapse_identical_visit_rows
from moca_utils import (
    RAW_COLUMNS,
    RAW_MOCA_FILENAME,
    moca_source_audit,
    resolve_moca_sources,
)


VISIT_KEYS = ["PATNO", "EVENT_ID"]
STANDARD_ANALYSIS_VISITS = {
    "BL",
    "V01",
    "V02",
    "V03",
    "V04",
    "V05",
    "V06",
    "V07",
    "V08",
    "V09",
    "V10",
    "V11",
    "V12",
    "V13",
}


def require_columns(df, columns, source_name):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{source_name} lacks required columns: {missing}")


def validate_visit_keys(df, source_name, allow_duplicates=False):
    require_columns(df, VISIT_KEYS, source_name)
    if df[VISIT_KEYS].isna().any().any():
        raise ValueError(f"{source_name} has records with missing PATNO/EVENT_ID")
    duplicate_mask = df.duplicated(VISIT_KEYS, keep=False)
    if duplicate_mask.any() and not allow_duplicates:
        examples = df.loc[duplicate_mask, VISIT_KEYS].drop_duplicates().head(10)
        raise ValueError(
            f"{source_name} has duplicate PATNO/EVENT_ID and needs an explicit quality rule; "
            f"examples: {examples.to_dict('records')}"
        )


def clean_coded_numeric(df, allowed_values, source_name):
    """Convert coded columns to numbers, set 101 (Unable to Rate) to missing and reject codes outside the official list."""
    cleaned = df.copy()
    for column, valid_codes in allowed_values.items():
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        unable_to_rate = cleaned[column] == 101
        if unable_to_rate.any():
            print(
                f"  -> {source_name}.{column}: "
                f"set {int(unable_to_rate.sum())} values of 101 (Unable to Rate) to missing"
            )
            cleaned.loc[unable_to_rate, column] = np.nan
        invalid = cleaned[column].notna() & ~cleaned[column].isin(valid_codes)
        if invalid.any():
            values = sorted(cleaned.loc[invalid, column].unique().tolist())
            raise ValueError(
                f"{source_name}.{column} has values outside the official code list: {values}"
            )
    return cleaned


def key_index(df):
    return pd.MultiIndex.from_frame(df[VISIT_KEYS].drop_duplicates())


# ==========================================
# Step 1: load data
# ==========================================

# MDS-UPDRS Part I (non-motor experiences)
df_part1 = pd.read_csv("tables/MDS-UPDRS_Part_I_26Sep2026.csv")

# MDS-UPDRS Part I patient questionnaire
df_part1_pq = pd.read_csv("tables/MDS-UPDRS_Part_I_Patient_Questionnaire_26Sep2026.csv")

# MDS-UPDRS Part II (motor experiences of daily living)
df_part2 = pd.read_csv("tables/MDS_UPDRS_Part_II__Patient_Questionnaire_26Sep2026.csv")

# MDS-UPDRS Part III (motor examination)
df_part3 = pd.read_csv("tables/MDS-UPDRS_Part_III_26Sep2026.csv", low_memory=False)

# MDS-UPDRS Part IV (motor complications)
df_part4 = pd.read_csv("tables/MDS-UPDRS_Part_IV__Motor_Complications_26Sep2026.csv")

# SCOPA-AUT (autonomic function)
df_scopa = pd.read_csv("tables/SCOPA-AUT_26Sep2026.csv")

# Vital signs (blood pressure)
df_vital = pd.read_csv("tables/Vital_Signs_26Sep2026.csv")

# Schwab & England (functional dependence)
df_se = pd.read_csv(
    "tables/Modified_Schwab___England_Activities_of_Daily_Living_26Sep2026.csv"
)

# PPMI curated data (MoCA scores)
df_curated = load_curated_data(CURATED_FILENAME)
df_moca = pd.read_csv("tables/" + RAW_MOCA_FILENAME, low_memory=False)

# ==========================================
# Step 2: field checks, special-code cleaning and visit-level aggregation
# ==========================================

part1_cols = ["PATNO", "EVENT_ID", "NP1COG", "NP1HALL", "NP1APAT"]
part1_pq_cols = ["PATNO", "EVENT_ID", "NP1URIN", "NP1LTHD"]
part2_cols = [
    "PATNO",
    "EVENT_ID",
    "NP2WALK",
    "NP2FREZ",
    "NP2SWAL",
    "NP2EAT",
    "NP2DRES",
    "NP2HYGN",
]
part3_cols = ["PATNO", "EVENT_ID", "NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NHY", "NP3SPCH"]
part4_cols = ["PATNO", "EVENT_ID", "NP4WDYSK", "NP4DYSKI", "NP4FLCTI", "NP4FLCTX"]
scopa_cols = ["PATNO", "EVENT_ID", "SCAU8", "SCAU9", "SCAU15", "SCAU16"]
vitals_cols = ["SYSSUP", "SYSSTND", "DIASUP", "DIASTND"]
se_cols = ["PATNO", "EVENT_ID", "MSEADLG"]
curated_cols = ["PATNO", "EVENT_ID", "moca"]

source_requirements = {
    "MDS-UPDRS Part I": (df_part1, part1_cols),
    "MDS-UPDRS Part I-PQ": (df_part1_pq, part1_pq_cols),
    "MDS-UPDRS Part II": (df_part2, part2_cols),
    "MDS-UPDRS Part III": (df_part3, part3_cols),
    "MDS-UPDRS Part IV": (df_part4, part4_cols),
    "SCOPA-AUT": (df_scopa, scopa_cols),
    "Vital Signs": (df_vital, VISIT_KEYS + vitals_cols),
    "Schwab & England": (df_se, se_cols),
    "Curated MoCA": (df_curated, curated_cols),
    "Raw MoCA": (df_moca, RAW_COLUMNS),
}
for source_name, (source_df, required_cols) in source_requirements.items():
    require_columns(source_df, required_cols, source_name)
    validate_visit_keys(
        source_df,
        source_name,
        allow_duplicates=(source_name in {"MDS-UPDRS Part III", "SCOPA-AUT"}),
    )

mds_0_to_4 = [0, 1, 2, 3, 4]
df_part1_clean = clean_coded_numeric(
    df_part1[part1_cols],
    {col: mds_0_to_4 for col in part1_cols[2:]},
    "MDS-UPDRS Part I",
)
df_part1_pq_clean = clean_coded_numeric(
    df_part1_pq[part1_pq_cols],
    {col: mds_0_to_4 for col in part1_pq_cols[2:]},
    "MDS-UPDRS Part I-PQ",
)
df_part2_clean = clean_coded_numeric(
    df_part2[part2_cols],
    {col: mds_0_to_4 for col in part2_cols[2:]},
    "MDS-UPDRS Part II",
)
df_part3_clean = clean_coded_numeric(
    df_part3[part3_cols],
    {
        "NP3GAIT": mds_0_to_4,
        "NP3FRZGT": mds_0_to_4,
        "NP3PSTBL": mds_0_to_4,
        "NHY": [0, 1, 2, 3, 4, 5],
        "NP3SPCH": mds_0_to_4,
    },
    "MDS-UPDRS Part III",
)
df_part4_clean = clean_coded_numeric(
    df_part4[part4_cols],
    {col: mds_0_to_4 for col in part4_cols[2:]},
    "MDS-UPDRS Part IV",
)
df_scopa_clean = clean_coded_numeric(
    df_scopa[scopa_cols],
    {
        "SCAU8": [0, 1, 2, 3, 9],
        "SCAU9": [0, 1, 2, 3, 9],
        "SCAU15": [0, 1, 2, 3],
        "SCAU16": [0, 1, 2, 3],
    },
    "SCOPA-AUT",
)
df_scopa_clean = collapse_identical_visit_rows(
    df_scopa_clean, scopa_cols[2:], "SCOPA-AUT"
)
df_se_clean = clean_coded_numeric(
    df_se[se_cols], {"MSEADLG": list(range(0, 101))}, "Schwab & England"
)
resolved_moca = resolve_moca_sources(df_curated, df_moca)
# Keep source keys separate so source coverage describes the actual inputs.
df_curated_clean = df_curated[VISIT_KEYS].copy()
df_moca_clean = df_moca[VISIT_KEYS].copy()

# Part III can hold ON and OFF states at one visit; clean 101 first, then take the worst valid score.
df_part3_agg = df_part3_clean.groupby(VISIT_KEYS, as_index=False).max()

# The vitals file should have one record per visit; refuse to rely on row order if duplicated.
df_vital_clean = df_vital[VISIT_KEYS + vitals_cols].copy()
for col in vitals_cols:
    df_vital_clean[col] = pd.to_numeric(df_vital_clean[col], errors="coerce")
df_vital_clean["Sys_Drop"] = df_vital_clean["SYSSUP"] - df_vital_clean["SYSSTND"]
df_vital_clean["Dia_Drop"] = df_vital_clean["DIASUP"] - df_vital_clean["DIASTND"]
df_vital_clean = df_vital_clean[VISIT_KEYS + ["Sys_Drop", "Dia_Drop"]]

# ==========================================
# Step 3: build the visit table from the union of PATNO/EVENT_ID over all sources
# ==========================================

merge_sources = {
    "MDS-UPDRS Part III": df_part3_agg,
    "MDS-UPDRS Part I": df_part1_clean,
    "MDS-UPDRS Part I-PQ": df_part1_pq_clean,
    "MDS-UPDRS Part II": df_part2_clean,
    "MDS-UPDRS Part IV": df_part4_clean,
    "SCOPA-AUT": df_scopa_clean,
    "Vital Signs": df_vital_clean,
    "Schwab & England": df_se_clean,
    "Curated MoCA": df_curated_clean,
    "Raw MoCA": df_moca_clean,
}

visit_union = (
    pd.concat(
        [frame[VISIT_KEYS] for frame in merge_sources.values()], ignore_index=True
    )
    .drop_duplicates()
    .sort_values(VISIT_KEYS, kind="stable")
    .reset_index(drop=True)
)
base_keys = key_index(pd.concat([df_part3_agg[VISIT_KEYS], df_part1_clean[VISIT_KEYS]]))
union_key_count = len(visit_union)

audit_rows = []
for source_name, frame in merge_sources.items():
    source_keys = frame[VISIT_KEYS].drop_duplicates()
    source_index = key_index(source_keys)
    absent_from_old_base = ~source_index.isin(base_keys)
    absent_keys = source_keys.loc[absent_from_old_base]
    raw_df = source_requirements[source_name][0]
    duplicate_key_count = int(
        raw_df.duplicated(VISIT_KEYS, keep=False)
        .groupby([raw_df["PATNO"], raw_df["EVENT_ID"]])
        .any()
        .sum()
    )
    audit_rows.append(
        {
            "source": source_name,
            "source_rows": len(raw_df),
            "source_unique_visit_keys": len(source_keys),
            "duplicate_visit_keys_before_resolution": duplicate_key_count,
            "union_visit_keys": union_key_count,
            "union_coverage_percent": round(
                len(source_keys) / union_key_count * 100, 3
            ),
            "keys_absent_from_old_part1_part3_base": int(absent_from_old_base.sum()),
            "analysis_visit_keys_absent_from_old_base": int(
                absent_keys["EVENT_ID"].isin(STANDARD_ANALYSIS_VISITS).sum()
            ),
        }
    )

df_master = visit_union.copy()
for source_name, frame in merge_sources.items():
    df_master = df_master.merge(
        frame,
        on=VISIT_KEYS,
        how="left",
        validate="one_to_one",
    )

df_master = df_master.merge(
    resolved_moca, on=VISIT_KEYS, how="left", validate="one_to_one"
)
df_master["moca_source"] = df_master["moca_source"].fillna("missing")
for column in ("moca_raw_reason", "moca_raw_structural_reason"):
    df_master[column] = df_master[column].fillna("raw_row_absent")
for column in (
    "moca_curated_present",
    "moca_raw_present",
    "moca_education_conflict",
    "moca_score_disagreement",
    "moca_threshold_disagreement",
):
    df_master[column] = df_master[column].eq(True)
moca_source_audit(df_master).to_csv("step_01_moca_source_audit.csv", index=False)
pd.DataFrame(audit_rows).to_csv("step_01_source_match_audit.csv", index=False)
print(
    f"Union of visit keys: {union_key_count}; source-match audit saved to step_01_source_match_audit.csv"
)

# ==========================================
# Step 4: derive the milestones
# ==========================================

milestones = [
    "MS_Walking",
    "MS_MotorComp",
    "MS_Cognition",
    "MS_Autonomic",
    "MS_Functional",
    "MS_ADL",
]
for ms in milestones:
    df_master[ms] = 0

# Domain 1: walking and balance
c1 = df_master["NP2WALK"] >= 3
c2 = df_master["NP2FREZ"] >= 3
c3 = df_master["NP3GAIT"] >= 3
c4 = df_master["NP3FRZGT"] == 4
c5 = df_master["NP3PSTBL"] >= 3
c6 = df_master["NHY"] >= 4

df_master.loc[c1 | c2 | c3 | c4 | c5 | c6, "MS_Walking"] = 1

# Domain 2: motor complications
c_dys = (df_master["NP4WDYSK"] >= 3) & (df_master["NP4DYSKI"] >= 3)
c_fluc1 = df_master["NP4FLCTI"] >= 3
c_fluc2 = df_master["NP4FLCTX"] >= 3

df_master.loc[c_dys | c_fluc1 | c_fluc2, "MS_MotorComp"] = 1

# Domain 3: cognition
c_moca = (df_master["moca"] < 21) & (df_master["moca"].notna())
c_cog = df_master["NP1COG"] >= 3
c_hall = df_master["NP1HALL"] >= 3
c_apat = df_master["NP1APAT"] >= 3

df_master.loc[c_moca | c_cog | c_hall | c_apat, "MS_Cognition"] = 1

# Domain 4: autonomic dysfunction
# Urinary incontinence: UPDRS and SCOPA
c_urin_scopa = (df_master["SCAU8"] >= 2) | (df_master["SCAU9"] >= 2)
c_urin = (df_master["NP1URIN"] >= 3) & c_urin_scopa

# Orthostatic hypotension: symptoms and blood-pressure drop
c_oh_bp = (df_master["Sys_Drop"] >= 20) | (df_master["Dia_Drop"] >= 10)
c_oh = (df_master["SCAU15"] >= 2) & c_oh_bp

# Syncope: UPDRS = 4 or SCOPA >= 1
c_sync = (df_master["NP1LTHD"] == 4) | (df_master["SCAU16"] >= 1)

df_master.loc[c_urin | c_oh | c_sync, "MS_Autonomic"] = 1

# Domain 5: functional dependence
c_se = df_master["MSEADLG"] < 80
df_master.loc[c_se, "MS_Functional"] = 1

# Domain 6: activities of daily living
c_swal = df_master["NP2SWAL"] >= 3
c_eat = df_master["NP2EAT"] >= 3
c_dres = df_master["NP2DRES"] >= 3
c_hygn = df_master["NP2HYGN"] >= 3
c_spch = df_master["NP3SPCH"] >= 3

df_master.loc[c_swal | c_eat | c_dres | c_hygn | c_spch, "MS_ADL"] = 1

# ==========================================
# Step 5: save results
# ==========================================
df_master.to_csv("PPMI_1_milestones.csv", index=False)

print("Done: PPMI_1_milestones.csv written")
print("Preview:")
print(df_master[["PATNO", "EVENT_ID"] + milestones].head())
