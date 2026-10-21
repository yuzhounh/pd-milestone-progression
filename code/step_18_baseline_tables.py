"""Step 18: baseline characteristics by progression subtype (Table 1) and missing-data patterns."""

import math
import re
import warnings
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_utils import (
    CURATED_FILENAME,
    baseline_correction_audit,
    normalize_curated_columns,
    prepare_baseline_records,
)
from moca_utils import RAW_MOCA_FILENAME, moca_source_audit


warnings.filterwarnings("ignore")

SUBTYPE_PATH = Path("PPMI_3_patients_subtypes_lca_k2.csv")
LASSO_FEATURES_PATH = Path("step_08_selected_features.csv")
FEATURE_MAPPING_PATH = Path("PPMI_feature_mapping.csv")

MISSINGNESS_CSV = "step_18_baseline_missingness.csv"
BASELINE_COMPARE_CSV = "step_18_rapidslow_baseline_compare.csv"
GROUP_COL = "Subtype"

FEATURES = [
    "age_at_visit",
    "SEX",
    "EDUCYRS",
    "race",
    "fampd_bin",
    "BMI",
    "duration",
    "LEDD",
    "upsit_pctl",
    "moca",
    "DVT_SDM",
    "DVT_SFTANIM",
    "MSEADLG",
    "quip",
    "ess",
    "rem",
    "gds",
    "stai_state",
    "stai_trait",
    "scopa",
    "orthostasis",
    "NHY",
    "pigd",
    "td_pigd",
    "NP1COG",
    "updrs1_score",
    "updrs2_score",
    "updrs3_score",
    "APOE_e4",
    "mia_lowput_expected",
    "MIA_STRIATUM_mean",
]

UNIT_TYPE = {
    "age_at_visit": "years",
    "SEX": "binary",
    "EDUCYRS": "years",
    "race": "categorical",
    "fampd_bin": "binary",
    "BMI": "kg/m2",
    "duration": "months",
    "LEDD": "mg/day",
    "upsit_pctl": "percentile",
    "moca": "score",
    "DVT_SDM": "score",
    "DVT_SFTANIM": "score",
    "MSEADLG": "percent",
    "quip": "score",
    "ess": "score",
    "rem": "score",
    "gds": "score",
    "stai_state": "score",
    "stai_trait": "score",
    "scopa": "score",
    "orthostasis": "binary",
    "NHY": "stage",
    "pigd": "score",
    "td_pigd": "categorical",
    "NP1COG": "score",
    "updrs1_score": "score",
    "updrs2_score": "score",
    "updrs3_score": "score",
    "APOE_e4": "binary",
    "mia_lowput_expected": "SBR (unitless)",
    "MIA_STRIATUM_mean": "SBR (unitless)",
}

SECTION = {
    "age_at_visit": "Demographics",
    "SEX": "Demographics",
    "EDUCYRS": "Demographics",
    "race": "Demographics",
    "fampd_bin": "Demographics",
    "BMI": "Demographics",
    "duration": "Demographics",
    "LEDD": "Treatment and functional status",
    "MSEADLG": "Treatment and functional status",
    "updrs1_score": "Motor and non-motor scales",
    "updrs2_score": "Motor and non-motor scales",
    "updrs3_score": "Motor and non-motor scales",
    "NHY": "Motor and non-motor scales",
    "pigd": "Motor and non-motor scales",
    "td_pigd": "Motor and non-motor scales",
    "scopa": "Motor and non-motor scales",
    "orthostasis": "Motor and non-motor scales",
    "rem": "Motor and non-motor scales",
    "ess": "Motor and non-motor scales",
    "upsit_pctl": "Motor and non-motor scales",
    "quip": "Motor and non-motor scales",
    "gds": "Motor and non-motor scales",
    "stai_state": "Motor and non-motor scales",
    "stai_trait": "Motor and non-motor scales",
    "moca": "Cognitive features",
    "DVT_SDM": "Cognitive features",
    "DVT_SFTANIM": "Cognitive features",
    "NP1COG": "Cognitive features",
    "APOE_e4": "Biomarkers and imaging",
    "mia_lowput_expected": "Biomarkers and imaging",
    "MIA_STRIATUM_mean": "Biomarkers and imaging",
}

CATEGORICAL_VARS = {
    "SEX": {0: "Female (SEX=0)", 1: "Male (SEX=1; female=0)"},
    "race": {
        1: "White",
        2: "Black",
        3: "Asian",
        4: "Other (including multiracial)",
    },
    "fampd_bin": {0: "No family history of PD", 1: "Family history of PD"},
    "orthostasis": {0: "No orthostatic hypotension", 1: "Orthostatic hypotension"},
    "td_pigd": {
        1: "Tremor-dominant phenotype",
        2: "non-TD phenotype (PIGD/indeterminate)",
    },
    "APOE_e4": {0: "APOE e4 non-carrier", 1: "APOE e4 carrier"},
}


def load_core_lasso():
    return set(pd.read_csv(LASSO_FEATURES_PATH)["feature"].dropna())


def is_core_feature(feature, core_lasso):
    return feature in core_lasso or (
        feature == "race" and any(value.startswith("race_") for value in core_lasso)
    )


def load_feature_name_map(csv_path=FEATURE_MAPPING_PATH):
    mapping = pd.read_csv(csv_path)
    return dict(zip(mapping["Feature"], mapping["Abbreviation"]))


def find_curated_path():
    search_roots = [Path.cwd(), Path.home() / "Desktop"]
    for root in search_roots:
        if not root.exists():
            continue
        direct = root / CURATED_FILENAME
        if direct.exists():
            return direct
        matches = list(root.rglob(CURATED_FILENAME))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {CURATED_FILENAME}")


def col_to_index(cell_ref):
    letters = re.match(r"([A-Z]+)", cell_ref).group(1)
    idx = 0
    for char in letters:
        idx = idx * 26 + (ord(char) - ord("A") + 1)
    return idx - 1


def parse_value(cell, shared_strings):
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//a:t", ns))

    value_node = cell.find("a:v", ns)
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        return shared_strings[int(raw)]
    try:
        number = float(raw)
        return int(number) if number.is_integer() else number
    except ValueError:
        return raw


def read_xlsx_first_sheet(path):
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared_strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared_strings.append(
                    "".join(t.text or "" for t in si.findall(".//a:t", ns))
                )

        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = []
        max_col = 0
        for row in root.findall(".//a:sheetData/a:row", ns):
            values = {}
            for cell in row.findall("a:c", ns):
                ref = cell.attrib.get("r", "")
                if not ref:
                    continue
                col_idx = col_to_index(ref)
                values[col_idx] = parse_value(cell, shared_strings)
                max_col = max(max_col, col_idx)
            if values:
                rows.append([values.get(i) for i in range(max_col + 1)])

    headers = [
        str(x) if x is not None else f"Unnamed_{i}" for i, x in enumerate(rows[0])
    ]
    data_rows = []
    for row in rows[1:]:
        padded = row + [None] * (len(headers) - len(row))
        data_rows.append(padded[: len(headers)])
    return pd.DataFrame(data_rows, columns=headers)


def load_baseline_curated():
    curated_path = find_curated_path()
    try:
        df = pd.read_excel(curated_path)
    except ImportError:
        df = read_xlsx_first_sheet(curated_path)
    df = normalize_curated_columns(df)
    raw_moca = pd.read_csv(
        curated_path.parent / "tables" / RAW_MOCA_FILENAME, low_memory=False
    )
    baseline = prepare_baseline_records(df, source_path=curated_path, raw_moca=raw_moca)
    baseline_correction_audit(baseline).to_csv(
        "step_18_baseline_corrections.csv", index=False
    )
    moca_source_audit(baseline).to_csv("step_18_moca_source_audit.csv", index=False)
    return baseline


def construct_features(df):
    df = df.copy()
    if "fampd_bin" in df.columns:
        df["fampd_bin"] = df["fampd_bin"].map({1: 1, 2: 0})
    required_duration_cols = {"duration", "duration_yrs"}
    missing_duration_cols = sorted(required_duration_cols - set(df.columns))
    if missing_duration_cols:
        raise ValueError(
            f"Baseline data are missing duration columns: {missing_duration_cols}"
        )
    df["duration_months"] = pd.to_numeric(df["duration"], errors="coerce")
    df["duration_years"] = pd.to_numeric(df["duration_yrs"], errors="coerce")
    consistency_error = (df["duration_months"] / 12 - df["duration_years"]).abs()
    if consistency_error.dropna().gt(0.01).any():
        raise ValueError("duration (months) and duration_yrs (years) are inconsistent")
    df["duration"] = df["duration_months"]
    return df


def load_analysis_data():
    subtypes = pd.read_csv(SUBTYPE_PATH)[["PATNO", GROUP_COL]]
    subtypes["PATNO"] = pd.to_numeric(subtypes["PATNO"], errors="coerce").astype(
        "Int64"
    )
    curated = construct_features(load_baseline_curated())
    curated["PATNO"] = pd.to_numeric(curated["PATNO"], errors="coerce").astype("Int64")
    df = subtypes.merge(curated, on="PATNO", how="left")

    if "APOE_e4" in df.columns:
        apoe = pd.to_numeric(df["APOE_e4"], errors="coerce")
        if apoe.dropna().max() > 1:
            df["APOE_e4"] = np.where(apoe.notna(), (apoe > 0).astype(float), np.nan)
    return df


def p_to_text(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def median_iqr(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return ""
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    return f"{values.median():.2f} [{q1:.2f}-{q3:.2f}]"


def count_pct(values, level):
    values = values.dropna()
    if values.empty:
        return ""
    n = int((values == level).sum())
    return f"{n} ({n / len(values) * 100:.1f}%)"


def normal_two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def average_ranks(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def mannwhitneyu_p(x, y):
    x = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(y, errors="coerce").dropna().to_numpy(dtype=float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    values = np.concatenate([x, y])
    ranks = average_ranks(values)
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mean_u = n1 * n2 / 2.0
    _, counts = np.unique(values, return_counts=True)
    tie_term = np.sum(counts**3 - counts)
    n = n1 + n2
    var_u = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
    if var_u <= 0:
        return np.nan
    correction = 0.5 if u1 > mean_u else -0.5
    z = (u1 - mean_u - correction) / math.sqrt(var_u)
    return normal_two_sided_p(z)


def gammaincc(a, x):
    if x < 0 or a <= 0:
        return np.nan
    if x == 0:
        return 1.0
    eps = 1e-14
    max_iter = 200
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap = a
        summation = 1.0 / a
        delta = summation
        for _ in range(max_iter):
            ap += 1.0
            delta *= x / ap
            summation += delta
            if abs(delta) < abs(summation) * eps:
                break
        lower = summation * math.exp(-x + a * math.log(x) - gln)
        return max(0.0, min(1.0, 1.0 - lower))
    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    upper = math.exp(-x + a * math.log(x) - gln) * h
    return max(0.0, min(1.0, upper))


def chi2_contingency_p(table):
    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    if total == 0:
        return np.nan, np.array([])
    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / total
    valid = expected > 0
    chi2 = np.sum(((observed - expected) ** 2)[valid] / expected[valid])
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    p = gammaincc(dof / 2.0, chi2 / 2.0) if dof > 0 else np.nan
    return p, expected


def fisher_exact_2x2_p(table):
    observed = table.to_numpy(dtype=int)
    if observed.shape != (2, 2):
        return np.nan
    a, b = int(observed[0, 0]), int(observed[0, 1])
    c, d = int(observed[1, 0]), int(observed[1, 1])
    row1, row2 = a + b, c + d
    col1 = a + c
    n = row1 + row2

    def log_choose(nn, kk):
        if kk < 0 or kk > nn:
            return float("-inf")
        return math.lgamma(nn + 1) - math.lgamma(kk + 1) - math.lgamma(nn - kk + 1)

    def hypergeom_prob(x):
        return math.exp(
            log_choose(col1, x) + log_choose(n - col1, row1 - x) - log_choose(n, row1)
        )

    min_x = max(0, row1 - (n - col1))
    max_x = min(row1, col1)
    observed_prob = hypergeom_prob(a)
    p = 0.0
    for x in range(min_x, max_x + 1):
        prob = hypergeom_prob(x)
        if prob <= observed_prob + 1e-12:
            p += prob
    return min(1.0, p)


def categorical_p(df, var):
    clean = df.dropna(subset=[var])
    table = pd.crosstab(clean[GROUP_COL], clean[var])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan
    p_chi, expected = chi2_contingency_p(table)
    if table.shape == (2, 2) and expected.size and (expected < 5).any():
        return fisher_exact_2x2_p(table)
    return p_chi


def categorical_label(var, level, feature_name_map):
    label_map = CATEGORICAL_VARS.get(var)
    if label_map is None:
        if pd.isna(level):
            return ""
        if isinstance(level, float) and level.is_integer():
            return f"{feature_name_map.get(var, var)} = {int(level)}"
        return f"{feature_name_map.get(var, var)} = {level}"
    key = int(level) if isinstance(level, float) and level.is_integer() else level
    return label_map.get(key, f"{feature_name_map.get(var, var)} = {level}")


def build_missingness_table(df, core_lasso, feature_name_map):
    rows = []
    n = len(df)
    for idx, feature in enumerate(FEATURES, start=1):
        if feature in df.columns:
            missing_n = int(df[feature].isna().sum())
            non_missing_n = int(n - missing_n)
        else:
            missing_n = n
            non_missing_n = 0
        rows.append(
            {
                "Index": idx,
                "Variable": feature,
                "Definition": feature_name_map.get(feature, feature),
                "Unit/type": UNIT_TYPE.get(feature, ""),
                "LASSO selected": (
                    "Yes" if is_core_feature(feature, core_lasso) else "No"
                ),
                "Non-missing, n": non_missing_n,
                "Missing, n": missing_n,
                "Missing, %": round(missing_n / n * 100, 1),
            }
        )
    return pd.DataFrame(rows)


def build_baseline_compare_table(df, core_lasso, feature_name_map):
    rows = []
    slow_n = int((df[GROUP_COL] == "Slow").sum())
    rapid_n = int((df[GROUP_COL] == "Rapid").sum())

    for idx, var in enumerate(FEATURES, start=1):
        if var not in df.columns:
            continue
        if var in CATEGORICAL_VARS:
            p = categorical_p(df, var)
            levels = sorted(df[var].dropna().unique())
            if var in {"SEX", "fampd_bin", "orthostasis", "APOE_e4", "td_pigd"}:
                levels = [level for level in levels if level == 1]
            for level_i, level in enumerate(levels):
                rows.append(
                    {
                        "Index": idx,
                        "Section": SECTION.get(var, ""),
                        "Characteristic": categorical_label(
                            var, level, feature_name_map
                        ),
                        "Variable": var,
                        "Unit/type": UNIT_TYPE.get(var, ""),
                        "LASSO selected": (
                            "Yes" if is_core_feature(var, core_lasso) else "No"
                        ),
                        f"Slow (N={slow_n})": count_pct(
                            df.loc[df[GROUP_COL] == "Slow", var], level
                        ),
                        f"Rapid (N={rapid_n})": count_pct(
                            df.loc[df[GROUP_COL] == "Rapid", var], level
                        ),
                        "P value": p_to_text(p) if level_i == 0 else "",
                    }
                )
        else:
            p = mannwhitneyu_p(
                df.loc[df[GROUP_COL] == "Slow", var],
                df.loc[df[GROUP_COL] == "Rapid", var],
            )
            rows.append(
                {
                    "Index": idx,
                    "Section": SECTION.get(var, ""),
                    "Characteristic": feature_name_map.get(var, var),
                    "Variable": var,
                    "Unit/type": UNIT_TYPE.get(var, ""),
                    "LASSO selected": (
                        "Yes" if is_core_feature(var, core_lasso) else "No"
                    ),
                    f"Slow (N={slow_n})": median_iqr(
                        df.loc[df[GROUP_COL] == "Slow", var]
                    ),
                    f"Rapid (N={rapid_n})": median_iqr(
                        df.loc[df[GROUP_COL] == "Rapid", var]
                    ),
                    "P value": p_to_text(p),
                }
            )
    return pd.DataFrame(rows)


def main():
    """Build and save the baseline and missingness tables."""
    core_lasso = load_core_lasso()
    feature_name_map = load_feature_name_map()
    df = load_analysis_data()

    missingness = build_missingness_table(df, core_lasso, feature_name_map)
    baseline_compare = build_baseline_compare_table(df, core_lasso, feature_name_map)

    missingness.to_csv(MISSINGNESS_CSV, index=False, encoding="utf-8-sig")
    baseline_compare.to_csv(BASELINE_COMPARE_CSV, index=False, encoding="utf-8-sig")

    print(f"N = {len(df)}")
    print(df[GROUP_COL].value_counts().to_string())
    print(f"Saved: {MISSINGNESS_CSV}")
    print(f"Saved: {BASELINE_COMPARE_CSV}")


if __name__ == "__main__":
    main()
