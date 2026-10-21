"""Plot descriptive Kaplan-Meier milestone risks; no between-class tests."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Plot font and minus sign
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = True

# ==========================================
# 1. Read the subtype assignments
# ==========================================
print("Reading subtype assignments...")
df_subtype_labels = pd.read_csv("PPMI_3_patients_subtypes_lca_k2.csv")
df_time = pd.read_csv("PPMI_2_time_to_event_continuous.csv")
df_subtypes = pd.merge(
    df_time, df_subtype_labels[["PATNO", "Subtype"]], on="PATNO", how="inner"
)
print(f"Read subtype data for {len(df_subtypes)} patients.")

# ==========================================
# 2. Cumulative incidence and multi-panel line plot
# ==========================================
print("\nComputing cumulative incidence and plotting...")

# Milestone columns
milestone_cols = [c for c in df_subtypes.columns if c.startswith("Time_")]

# Milestone abbreviations
abbr_map = {
    "Time_ADL": "ADL",
    "Time_Autonomic": "AUT",
    "Time_Cognition": "COG",
    "Time_Functional": "FD",
    "Time_MotorComp": "MC",
    "Time_Walking": "WB",
}

# Result containers
years = list(range(7))  # years 0-6
table4_data = {"Follow-years": years}

# Create the figure
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), dpi=300)
axes = axes.flatten()

# Group by subtype
df_slow = df_subtypes[df_subtypes["Subtype"] == "Slow"]
df_rapid = df_subtypes[df_subtypes["Subtype"] == "Rapid"]
n_slow = len(df_slow)
n_rapid = len(df_rapid)

# Colours
color_slow = "#1E88E5"  # blue: slow subtype
color_rapid = "#FF0D57"  # red: rapid subtype

from lifelines import KaplanMeierFitter

for i, col in enumerate(milestone_cols):
    short_name = abbr_map.get(col, col)
    censor_col = col.replace("Time_", "Censor_", 1)
    if censor_col not in df_subtypes.columns:
        raise ValueError(f"Missing domain-specific censoring column: {censor_col}")

    # Cumulative incidence 1 - S(t) from the Kaplan-Meier estimator, allowing for right censoring
    time_s = np.where(df_slow[col] < 7, df_slow[col], df_slow[censor_col])
    event_s = (df_slow[col] < 7).astype(int)
    kmf_s = KaplanMeierFitter()
    kmf_s.fit(time_s, event_observed=event_s)
    cum_slow = (kmf_s.cumulative_density_at_times(years).values * 100).tolist()

    time_r = np.where(df_rapid[col] < 7, df_rapid[col], df_rapid[censor_col])
    event_r = (df_rapid[col] < 7).astype(int)
    kmf_r = KaplanMeierFitter()
    kmf_r.fit(time_r, event_observed=event_r)
    cum_rapid = (kmf_r.cumulative_density_at_times(years).values * 100).tolist()

    # Store the values for export
    table4_data[f"{short_name}_Slow"] = [f"{x:.1f}%" for x in cum_slow]
    table4_data[f"{short_name}_Rapid"] = [f"{x:.1f}%" for x in cum_rapid]

    # Draw the lines
    ax = axes[i]
    if i < 6:
        ax.plot(
            years,
            cum_slow,
            "o-",
            label="Slow Subtype",
            color=color_slow,
            linewidth=2,
            markersize=5,
        )
        ax.plot(
            years,
            cum_rapid,
            "s-",
            label="Rapid Subtype",
            color=color_rapid,
            linewidth=2,
            markersize=5,
        )

        ax.fill_between(years, cum_slow, alpha=0.08, color=color_slow)
        ax.fill_between(years, cum_rapid, alpha=0.08, color=color_rapid)

        ax.set_title(f"{short_name}", fontsize=16, fontweight="bold", pad=10)
        ax.set_xlabel("Follow-up years", fontsize=12)
        ax.set_ylabel("KM-estimated milestone risk (%)", fontsize=12)
        ax.set_ylim(-2, 102)
        ax.set_xticks(years)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        if i == 0:
            ax.legend(frameon=False, loc="upper left", fontsize=10)

plt.tight_layout()
plt.savefig("step_06_cumulative_incidence.pdf", bbox_inches="tight")
print(f">>> Figure saved as step_06_cumulative_incidence.pdf")

# Export the table
df_table4 = pd.DataFrame(table4_data)
ordered_cols = ["Follow-years"]
for key in sorted(abbr_map.values()):
    ordered_cols.append(f"{key}_Slow")
    ordered_cols.append(f"{key}_Rapid")

final_cols = [c for c in ordered_cols if c in df_table4.columns]
df_table4 = df_table4[final_cols]

df_table4.to_csv("step_06_reproduction.csv", index=False, encoding="utf-8-sig")
print("\n>>> Table 4 preview:")
print(df_table4.head(7).to_string(index=False))
print(f"\nFull table saved as: step_06_reproduction.csv")


# Baseline characteristics are generated only by Step 18 using the BL-first rule.
print("\nDone.")
