"""Common style of the manuscript and supplementary figures.

All 2-D plots (forest plots, calibration and decision curves, SHAP dependence plots, ROC/PR
panels, LASSO path) use the same font sizes and the same axes frame, so that the figures look
alike in the paper. The frame (the axes rectangle, not the whole figure with its labels) has a
fixed size in millimetres and an aspect ratio of about 1.4 (width : height).

Sizes (points): axis labels 8, tick labels 7.5, panel titles 9, legend text 6.5 (smaller than the
labels), text annotations 7, panel letters 11 (bold).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, VPacker
from matplotlib.textpath import TextPath

MM = 1 / 25.4
AX_W = 72.0  # width of one axes frame in mm
AX_H = 52.0  # height of one axes frame in mm (aspect ratio 1.38)
LEGEND_SIZE = 6.5
ANNOTATION_SIZE = 7.0
LETTER_SIZE = 11


def apply() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "axes.unicode_minus": True,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": LEGEND_SIZE,
            "pdf.fonttype": 42,
        }
    )


def axes_grid(
    nrows: int,
    ncols: int,
    ax_w: float = AX_W,
    ax_h: float = AX_H,
    left: float = 15.0,
    right: float = 3.0,
    top: float = 7.0,
    bottom: float = 11.0,
    wgap: float = 17.0,
    hgap: float = 16.0,
):
    """Figure with an nrows x ncols grid of axes frames of exactly ax_w x ax_h mm.

    The margins (left, right, top, bottom) and gaps (wgap, hgap) are in mm and hold the tick
    labels, axis labels and titles, so labels never change the size of the frames.
    """
    width = left + ncols * ax_w + (ncols - 1) * wgap + right
    height = top + nrows * ax_h + (nrows - 1) * hgap + bottom
    fig = plt.figure(figsize=(width * MM, height * MM))
    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            x0 = left + c * (ax_w + wgap)
            y0 = bottom + (nrows - 1 - r) * (ax_h + hgap)
            axes[r, c] = fig.add_axes(
                [x0 / width, y0 / height, ax_w / width, ax_h / height]
            )
    return fig, axes


def panel_letter(ax, letter: str) -> None:
    ax.text(
        -0.13,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=LETTER_SIZE,
        fontweight="bold",
    )


def aligned_legend(ax, rows, loc, header=None, size=LEGEND_SIZE, handle_width=None):
    """Framed Arial legend whose columns line up: handle, left-aligned name, right-aligned value.

    rows: list of dicts with name, value ("" for none), color, kind ("line" or "marker"),
    and optionally linestyle and linewidth. header: optional (name, value) bold heading row.
    handle_width: optional legend-symbol width in points.
    """
    props = {"fontfamily": "Arial", "fontsize": size, "color": "black"}
    bold_props = {**props, "fontweight": "bold"}
    regular = FontProperties(family="Arial", size=size)
    bold = FontProperties(family="Arial", size=size, weight="bold")

    def width(label, prop):
        return TextPath((0, 0), label, prop=prop).get_extents().width if label else 0.0

    name_width = max(
        [width(r["name"], regular) for r in rows]
        + ([width(header[0], bold)] if header else [])
    )
    value_width = max(
        [width(r["value"], regular) for r in rows]
        + ([width(header[1], bold)] if header else [])
    )
    row_height = size * 1.55

    def cell(label, w, p, ha):
        box = DrawingArea(w, row_height, 0, 0)
        if label:
            box.add_artist(
                plt.Text(
                    0 if ha == "left" else w,
                    row_height / 2,
                    label,
                    ha=ha,
                    va="center",
                    **p,
                )
            )
        return box

    def make_row(handle_width, draw_handle, name, value, p):
        handle = DrawingArea(handle_width, row_height, 0, 0)
        draw_handle(handle)
        return HPacker(
            children=[
                handle,
                cell(name, name_width, p, "left"),
                cell(value, value_width, p, "right"),
            ],
            align="center",
            pad=0,
            sep=3,
        )

    if handle_width is None:
        handle_width = size * (3.4 if any(r["kind"] == "line" for r in rows) else 1.25)
    out = []
    if header:
        out.append(
            make_row(handle_width, lambda h: None, header[0], header[1], bold_props)
        )
    for r in rows:

        def draw(h, r=r):
            if r["kind"] == "line":
                h.add_artist(
                    Line2D(
                        [0, handle_width],
                        [row_height / 2] * 2,
                        color=r["color"],
                        linestyle=r.get("linestyle", "-"),
                        linewidth=r.get("linewidth", 1.4),
                    )
                )
            else:
                h.add_artist(
                    Line2D(
                        [handle_width / 2],
                        [row_height / 2],
                        color=r["color"],
                        marker="o",
                        markersize=4,
                        linestyle="",
                    )
                )

        out.append(make_row(handle_width, draw, r["name"], r["value"], props))
    anchored = AnchoredOffsetbox(
        loc=loc,
        child=VPacker(children=out, align="left", pad=0, sep=1),
        frameon=True,
        pad=0.4,
        borderpad=0.4,
    )
    anchored.patch.set_boxstyle("round,pad=0,rounding_size=0.2")
    anchored.patch.set_facecolor("white")
    anchored.patch.set_edgecolor("0.8")
    anchored.patch.set_alpha(0.8)
    ax.add_artist(anchored)
