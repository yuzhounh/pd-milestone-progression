"""Step 07: Kaplan-Meier curves of the six milestones by progression subtype."""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

import figstyle as fs

warnings.filterwarnings("ignore", category=UserWarning)


# Shared figure style (figstyle.py): fixed axes frames in mm, common font sizes.
# Three columns do not fit the standard 72 x 52 mm frame, so the frames are 50 x 36 mm
# (about the same aspect ratio).
AX_W, AX_H = 50.0, 36.0

fs.apply()


MILESTONE_CONFIGS = [
    ("Time_Walking", "Walking and balance"),
    ("Time_MotorComp", "Motor complications"),
    ("Time_Cognition", "Cognition"),
    ("Time_Autonomic", "Autonomic function"),
    ("Time_Functional", "Functional dependence"),
    ("Time_ADL", "Activities of daily living"),
]

PANEL_LABELS = ["A", "B", "C", "D", "E", "F"]
X_TICKS = np.arange(0, 7, 1)
RISK_TIME_POINTS = np.arange(0, 7, 1)
COLORS = {"Slow": "#1E88E5", "Rapid": "#D81B60"}


def load_data():
    df_time = pd.read_csv("PPMI_2_time_to_event_continuous.csv")
    df_sub = pd.read_csv("PPMI_3_patients_subtypes_lca_k2.csv")
    return pd.merge(df_time, df_sub[["PATNO", "Subtype"]], on="PATNO", how="inner")


def get_time_event(df, col):
    event = (df[col] < 7).astype(int)
    censor_col = col.replace("Time_", "Censor_", 1)
    if censor_col not in df.columns:
        raise ValueError(f"Missing domain-specific censoring column: {censor_col}")
    time = np.where(event == 1, df[col], df[censor_col].clip(upper=6))
    return time, event


def risk_counts(time, mask):
    time = pd.Series(time[mask])
    return [int((time >= point).sum()) for point in RISK_TIME_POINTS]


def plot_risk_table(ax, slow_counts, rapid_counts):
    """Number-at-risk table below the survival axes (x in data units, y in axes units)."""
    xform = ax.get_xaxis_transform()
    kw = {
        "transform": xform,
        "fontsize": fs.ANNOTATION_SIZE,
        "va": "center",
        "clip_on": False,
    }
    ax.text(-0.15, -0.40, "No. at risk", ha="left", **kw)
    ax.text(-0.35, -0.52, "Slow", ha="right", color=COLORS["Slow"], **kw)
    ax.text(-0.35, -0.64, "Rapid", ha="right", color=COLORS["Rapid"], **kw)
    for point, slow_n, rapid_n in zip(RISK_TIME_POINTS, slow_counts, rapid_counts):
        ax.text(point, -0.52, f"{slow_n}", ha="center", **kw)
        ax.text(point, -0.64, f"{rapid_n}", ha="center", **kw)


def style_survival_axis(ax, title, panel_label, show_ylabel, show_legend):
    ax.set_title(title)
    fs.panel_letter(ax, panel_label)
    ax.set_xlim(-0.15, 6.15)
    ax.set_ylim(0, 1.04)
    ax.set_xticks(X_TICKS)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(alpha=0.25)
    ax.set_xlabel("Years since baseline")
    ax.set_ylabel("Event-free probability" if show_ylabel else "")

    if show_legend:
        ax.legend(loc="lower left", frameon=True)
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()


def main():
    """Draw the six Kaplan-Meier panels and save the figure."""
    df = load_data()
    print(f"Loaded merged survival dataset: N={len(df)}")

    # hgap and bottom leave room for the x-axis label and the number-at-risk table.
    fig, axes_grid = fs.axes_grid(
        2, 3, ax_w=AX_W, ax_h=AX_H, left=15.0, wgap=14.0, hgap=36.0, bottom=30.0
    )

    for idx, (col, title) in enumerate(MILESTONE_CONFIGS):
        col_idx = idx % 3
        ax = axes_grid[idx // 3, col_idx]

        if col not in df.columns:
            ax.text(0.5, 0.5, f"Missing column: {col}", ha="center", va="center")
            continue

        time, event = get_time_event(df, col)
        mask_slow = df["Subtype"].eq("Slow")
        mask_rapid = df["Subtype"].eq("Rapid")

        kmf = KaplanMeierFitter()
        for group, mask in [("Slow", mask_slow), ("Rapid", mask_rapid)]:
            kmf.fit(time[mask], event_observed=event[mask], label=group)
            kmf.plot_survival_function(
                ax=ax,
                color=COLORS[group],
                linewidth=1.4,
                ci_alpha=0.12,
                censor_styles={"marker": "|", "ms": 3.8, "mew": 0.7},
            )

        style_survival_axis(
            ax,
            title=title,
            panel_label=PANEL_LABELS[idx],
            show_ylabel=col_idx == 0,
            show_legend=idx == 0,
        )

        slow_counts = risk_counts(time, mask_slow)
        rapid_counts = risk_counts(time, mask_rapid)
        plot_risk_table(ax, slow_counts, rapid_counts)

        print(
            f"{PANEL_LABELS[idx]} {title}: "
            f"Slow at risk {slow_counts}, Rapid at risk {rapid_counts}"
        )

    pdf_path = "step_07_km_curve.pdf"

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
