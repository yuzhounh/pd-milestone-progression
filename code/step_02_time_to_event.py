"""Construct censoring-aware six-domain milestone event histories.

Run after ``step_01_milestones.py``. The script writes annual interval codes for
StepMix, exact nominal event times for survival plots, domain-specific analysis
horizons, and aggregate audit tables. It does not fit a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from baseline_utils import (
    BASELINE_SOURCE_COLUMN,
    CURATED_FILENAME,
    load_curated_data,
    select_baseline_records,
)
from milestone_history_utils import (
    DOMAINS,
    build_domain_history,
    mask_st_negatives,
    milestone_states,
    partiv_structural_negatives,
    resolve_visit_times,
)


MIN_FOLLOWUP_YEARS = 1.0
NO_EVENT_CODE = 7

print("Reading milestones, treatment status and baseline data...")
source = pd.read_csv("PPMI_1_milestones.csv")
curated = load_curated_data(CURATED_FILENAME)
part3 = pd.read_csv("tables/MDS-UPDRS_Part_III_26Sep2026.csv", low_memory=False)
part4 = pd.read_csv("tables/MDS-UPDRS_Part_IV__Motor_Complications_26Sep2026.csv")

if source.duplicated(["PATNO", "EVENT_ID"]).any():
    raise ValueError("PPMI_1_milestones.csv has duplicate PATNO/EVENT_ID")

# COHORT, disease duration and H&Y all use the BL visit as baseline.
baseline = select_baseline_records(curated)
required = {"PATNO", "COHORT", "duration_yrs", BASELINE_SOURCE_COLUMN}
missing = sorted(required - set(baseline.columns))
if missing:
    raise ValueError(f"Curated baseline lacks eligibility fields: {missing}")
cohort = pd.to_numeric(baseline.COHORT, errors="coerce")
duration = pd.to_numeric(baseline.duration_yrs, errors="coerce")
candidate_ids = set(baseline.loc[cohort.eq(1) & duration.le(2), "PATNO"])
bl_hy = source.loc[source.EVENT_ID.eq("BL"), ["PATNO", "NHY"]]
if bl_hy.PATNO.duplicated().any():
    raise ValueError("Milestone table has duplicate BL visits")
candidate_ids &= set(
    bl_hy.loc[pd.to_numeric(bl_hy.NHY, errors="coerce").le(2), "PATNO"]
)
print(
    f"  > COHORT=1, duration <= 2 years and baseline H&Y <= 2: {len(candidate_ids)} participants"
)

source = source.loc[source.PATNO.isin(candidate_ids)].copy()
source["time"] = resolve_visit_times(source, part3)
states = milestone_states(source)
for domain in DOMAINS:
    saved = pd.to_numeric(source["MS_" + domain], errors="raise").astype(int)
    if not np.array_equal(states[domain].fillna(False).astype(int), saved):
        raise ValueError(
            f"{domain}: three-state positives disagree with the step 01 binary flags"
        )

# A blank Part IV counts as a structural negative only if the patient is clearly untreated, with no conflict and no earlier treatment or complication evidence.
partiv = partiv_structural_negatives(source, part3, part4, curated, source.time)
states.loc[
    states.MotorComp.isna() & partiv.structural_negative,
    "MotorComp",
] = False

# All six domains must be known and negative at baseline.
baseline_mask = source.EVENT_ID.eq("BL")
baseline_states = states.loc[baseline_mask].copy()
baseline_states.index = source.loc[baseline_mask, "PATNO"].to_numpy()
baseline_positive = baseline_states.eq(True).any(axis=1)
baseline_complete_negative = baseline_states.notna().all(axis=1) & baseline_states.eq(
    False
).all(axis=1)
baseline_unknown = ~baseline_positive & ~baseline_complete_negative
strict_ids = set(baseline_states.index[baseline_complete_negative])
print(
    f"  > Known negative in all six domains at baseline: {len(strict_ids)} participants"
)

# SC visits are not part of the event history, nor are ST records with unclear dates or beyond 6 years.
analysis = source.loc[
    source.PATNO.isin(strict_ids) & source.time.notna() & source.time.between(0, 6)
].copy()
analysis_states = states.loc[analysis.index].copy()
analysis_states = mask_st_negatives(analysis, analysis_states)

# The common one-year minimum follow-up is applied before the domain-information filter; R/ST visits can show
# that this time was reached, but only a scheduled annual negative or a valid positive forms an event-history cell.
maximum_observed_time = analysis.groupby("PATNO").time.max()
one_year_ids = set(
    maximum_observed_time.index[maximum_observed_time.ge(MIN_FOLLOWUP_YEARS)]
)
analysis = analysis.loc[analysis.PATNO.isin(one_year_ids)].copy()
analysis_states = analysis_states.loc[analysis.index]
print(f"  > Observed for at least 1 year: {len(one_year_ids)} participants")

histories = {
    domain: build_domain_history(analysis, analysis_states[domain], domain)
    for domain in DOMAINS
}
observable_cells = sum(history.observable_cells for history in histories.values())
final_ids = observable_cells.index[observable_cells.gt(0)]
print(f"  > Contributing at least 1 event-history cell: {len(final_ids)} participants")

annual = pd.DataFrame(index=final_ids)
continuous = pd.DataFrame(index=final_ids)
annual.index.name = "PATNO"
continuous.index.name = "PATNO"
for domain, history in histories.items():
    selected = history.loc[final_ids]
    annual["Time_" + domain] = selected.event_interval.astype(int)
    continuous["Time_" + domain] = selected.event_time.fillna(NO_EVENT_CODE)
    annual["Censor_" + domain] = selected.censor_time.astype(float)
    continuous["Censor_" + domain] = selected.censor_time.astype(float)

# max_followup is kept only for cohort description; all models and KM curves use Censor_*.
max_followup = maximum_observed_time.reindex(final_ids).clip(upper=6).astype(float)
annual["max_followup"] = max_followup
continuous["max_followup"] = max_followup

annual.to_csv("PPMI_2_time_to_event.csv")
continuous.to_csv("PPMI_2_time_to_event_continuous.csv")

flow = pd.DataFrame(
    [
        ("clinical_candidates", len(candidate_ids)),
        ("baseline_positive_excluded", int(baseline_positive.sum())),
        ("baseline_unknown_excluded", int(baseline_unknown.sum())),
        ("strict_baseline_negative", len(strict_ids)),
        ("uniform_one_year", len(one_year_ids)),
        ("zero_event_history_cells_excluded", int(observable_cells.eq(0).sum())),
        ("final_analysis_cohort", len(final_ids)),
    ],
    columns=["stage", "participants_n"],
)
flow.to_csv("step_02_cohort_flow.csv", index=False)

audit_rows = []
for domain, history in histories.items():
    no_event = ~history.event
    audit_rows.append(
        {
            "domain": domain,
            "one_year_population_n": len(history),
            "final_population_n": len(final_ids),
            "retained_events_n": int(history.event.sum()),
            "events_lost_after_annual_gap_n": int(history.event_lost_after_gap.sum()),
            "retained_first_positive_at_remote_n": int(
                (
                    history.event
                    & history.first_positive_visits.str.contains(
                        r"R06|R08|R10|R12", regex=True
                    )
                ).sum()
            ),
            "retained_first_positive_at_st_n": int(
                (
                    history.event
                    & history.first_positive_visits.str.contains("ST", regex=False)
                ).sum()
            ),
            "absent_annual_first_gap_n": int(
                history.first_gap_reason.eq("absent_visit").sum()
            ),
            "unknown_annual_first_gap_n": int(
                history.first_gap_reason.eq("unknown_state").sum()
            ),
            "domain_zero_information_n": int(history.observable_cells.eq(0).sum()),
            "median_no_event_censor_year": float(
                history.loc[no_event, "censor_time"].median()
            ),
        }
    )
pd.DataFrame(audit_rows).to_csv("step_02_event_history_audit.csv", index=False)

print("-" * 50)
print("Event histories built")
print("Annual-interval output: PPMI_2_time_to_event.csv")
print("Continuous-time output: PPMI_2_time_to_event_continuous.csv")
print("Aggregate audits: step_02_cohort_flow.csv, step_02_event_history_audit.csv")
print(f"Final number of patients: {len(annual)}")
print("-" * 50)
