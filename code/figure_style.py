"""Shared publication styling for manuscript and supplementary figures."""

import matplotlib.pyplot as plt


MM_TO_INCH = 1 / 25.4
MAIN_DPI = 300
SUPPLEMENT_DPI = 600

# Match the main manuscript panels: width / height = 50 / 36, about 1.39.
SUBPLOT_BOX_ASPECT = 36 / 50

AXES_TITLE_SIZE = 7.2
AXES_LABEL_SIZE = 7.0
TICK_LABEL_SIZE = 6.2
LEGEND_SIZE = 6.1
PANEL_LABEL_SIZE = 10.0


def apply_publication_style(dpi=MAIN_DPI):
    """Apply the common font and line settings used across all figures."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
            "axes.titlesize": AXES_TITLE_SIZE,
            "axes.labelsize": AXES_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "lines.linewidth": 1.20,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
        }
    )


def add_panel_label(ax, label, x=-0.15, y=1.06, va="top"):
    """Add a consistently sized bold panel label."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va=va,
        ha="left",
    )
