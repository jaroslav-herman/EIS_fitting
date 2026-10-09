"""Central Matplotlib style used by all public plotting helpers."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.legend import Legend


PUBLICATION_RED = "red"
PUBLICATION_GRAY = "0.55"
PUBLICATION_ORANGE = "darkorange"
PUBLICATION_PURPLE = "mediumpurple"

FONT_SIZE = 12
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 14
LEGEND_FONT_SIZE = 13
TITLE_SIZE = 16
ANNOTATION_SIZE = 12

STYLE_RC_PARAMS: dict[str, Any] = {
    "text.usetex": True,
    "axes.formatter.use_locale": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "mathtext.rm": "Times New Roman",
    "font.size": FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "figure.titlesize": TITLE_SIZE,
    "axes.linewidth": 1.2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "lines.linewidth": 1.8,
    "lines.markersize": 4,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.facecolor": "white",
    "legend.framealpha": 1.0,
    "savefig.dpi": 1000,
    "savefig.bbox": "tight",
}


def configure_plot_style() -> None:
    """Install global defaults used when new Matplotlib artists are created."""
    plt.rcParams.update(STYLE_RC_PARAMS)


def _style_text(text: Any, size: float) -> None:
    text.set_fontfamily("serif")
    text.set_fontsize(size)


def style_legend(legend: Legend | None, *, frameon: bool | None = None) -> None:
    """Apply central legend defaults to an existing legend."""
    if legend is None:
        return
    if frameon is not None:
        legend.set_frame_on(frameon)
    for text in legend.get_texts():
        _style_text(text, LEGEND_FONT_SIZE)
    _style_text(legend.get_title(), LEGEND_FONT_SIZE)


def apply_plot_style(ax: Axes | None = None) -> Axes:
    """Configure defaults and style *ax*, including existing text artists.

    Explicit local artist settings applied after this call remain authoritative.
    """
    configure_plot_style()
    if ax is None:
        ax = plt.gca()

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)
    ax.tick_params(axis="both", which="major", direction="out", length=5, width=1.2)
    _style_text(ax.xaxis.label, AXIS_LABEL_SIZE)
    _style_text(ax.yaxis.label, AXIS_LABEL_SIZE)
    _style_text(ax.title, TITLE_SIZE)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        _style_text(label, TICK_LABEL_SIZE)
    for annotation in ax.texts:
        _style_text(annotation, ANNOTATION_SIZE)
    style_legend(ax.get_legend())
    return ax


# Make importing ``wepy`` install defaults before user code creates figures.
configure_plot_style()
