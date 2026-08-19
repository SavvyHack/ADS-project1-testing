"""Shared figure configuration for the report.

Every figure in ``plots/`` is produced through this module, so that one
palette, one font size, one figure width, and one output resolution apply
across the whole report.

The width is fixed to the LaTeX template's text width so that figures are
included at their natural size rather than scaled by ``\\includegraphics``,
which would scale their labels down with them.

Typical use::

    from plot_utils import new_figure, save_figure, set_plot_style

    set_plot_style()
    fig, ax = new_figure(height=2.8)
    ax.plot(x, y, color=airport_colour("JFK"))
    save_figure(fig, "fig_example")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

# --- Geometry --------------------------------------------------------------
#: Text width of the report template, in inches: ``\documentclass[11pt]``
#: on US Letter with 0.9 in margins leaves 6.7 in. Drawn a fraction narrower so
#: ``width=\textwidth`` never scales the raster up.
FIGURE_WIDTH_IN = 6.6

#: Width for a figure intended to sit in half a text column.
HALF_WIDTH_IN = 3.2

#: Export resolution, at the usual print threshold.
DPI = 300

#: Default destination for saved figures, resolved from this file rather than
#: from the working directory, so a notebook and a script agree on it.
DEFAULT_PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"

# --- Colour ----------------------------------------------------------------
# The Okabe-Ito qualitative palette, which is distinguishable under the three
# common forms of colour vision deficiency and survives greyscale printing.
# Reference: Okabe & Ito, "Color Universal Design" (2008).
PALETTE = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "grey": "#595959",
    "light_grey": "#BFBFBF",
}

#: One colour per airport, used everywhere the two are drawn together.
AIRPORT_COLOURS = {"JFK": PALETTE["blue"], "LGA": PALETTE["vermillion"]}

#: Colour for the citywide baseline series.
BASELINE_COLOUR = PALETTE["grey"]

#: Sequential colour map for choropleths and heatmaps. Perceptually uniform
#: and monotone in lightness, so it greyscales without collapsing.
SEQUENTIAL_CMAP = "viridis"

#: Diverging map for quantities with a meaningful zero, such as a night-minus-
#: day difference. Always pair with a norm centred on zero.
DIVERGING_CMAP = "RdBu_r"

#: Fill for map polygons with no data, or too little to report.
MISSING_COLOUR = "#EEEEEE"


def set_plot_style() -> None:
    """Apply the project's Matplotlib defaults.

    Called once near the top of each notebook. Font sizes are set for a figure
    included at its natural width next to 11 pt body text, and nothing in a
    figure is smaller than 8 pt. ``DejaVu Serif`` ships with Matplotlib, so no
    font has to be installed for the figures to reproduce elsewhere.
    """
    plt.rcParams.update({
        "figure.figsize": (FIGURE_WIDTH_IN, 3.4),
        "figure.dpi": 110,           # on-screen only; exports use DPI
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.titlelocation": "left",
        "axes.titlepad": 4.0,

        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.frameon": False,
        "image.cmap": SEQUENTIAL_CMAP,
    })


def new_figure(
    nrows: int = 1,
    ncols: int = 1,
    height: float = 3.0,
    width: float = FIGURE_WIDTH_IN,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Create a figure at the report's text width.

    Args:
        nrows: Number of subplot rows.
        ncols: Number of subplot columns.
        height: Figure height in inches. Roughly 2.6 for a single strip, 3.4
            for a standalone plot, and 2.4 per row for a grid.
        width: Figure width in inches. Defaults to the template text width.
        **kwargs: Passed to :func:`matplotlib.pyplot.subplots`.

    Returns:
        The figure and its axes, exactly as ``plt.subplots`` returns them.
    """
    return plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)


def save_figure(
    fig: Figure,
    name: str,
    plots_dir: Path | str | None = None,
    close: bool = False,
) -> Path:
    """Write a figure to ``plots/`` as a 300 dpi PNG.

    Args:
        fig: Figure to write.
        name: File stem, without an extension. Use the ``fig_`` prefix so that
            the directory sorts into report order.
        plots_dir: Destination directory. Defaults to ``plots/`` at the
            project root.
        close: Whether to close the figure afterwards. False by default, so
            the figure still renders inline in the notebook.

    Returns:
        The path written.
    """
    directory = Path(plots_dir) if plots_dir is not None else DEFAULT_PLOTS_DIR
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"Saved {path.relative_to(directory.parent)}")

    if close:
        plt.close(fig)
    return path


def airport_colour(airport: str) -> str:
    """Return the fixed colour for an airport code.

    Args:
        airport: ``"JFK"`` or ``"LGA"``.

    Returns:
        A hex colour. Unknown codes fall back to the neutral grey rather than
        raising, so a stray code never breaks a figure mid-notebook.
    """
    return AIRPORT_COLOURS.get(airport, BASELINE_COLOUR)


def thousands_axis(ax: Axes, axis: str = "y") -> None:
    """Format an axis with thousands separators.

    Args:
        ax: Axes to format.
        axis: ``"x"`` or ``"y"``.
    """
    formatter = FuncFormatter(lambda value, _: f"{value:,.0f}")
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(formatter)


def currency_axis(ax: Axes, axis: str = "y", decimals: int = 0) -> None:
    """Format an axis as US dollars.

    Args:
        ax: Axes to format.
        axis: ``"x"`` or ``"y"``.
        decimals: Decimal places to show.
    """
    formatter = FuncFormatter(lambda value, _: f"${value:,.{decimals}f}")
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(formatter)


def heatmap(
    ax: Axes,
    matrix: np.ndarray,
    row_labels: list,
    col_labels: list,
    cmap: str = SEQUENTIAL_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    norm=None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    tick_every: int = 3,
):
    """Draw a labelled heatmap on an existing axes.

    Written once here because the hour-by-weekday grids are drawn several
    times and must share a colour scale and tick spacing to be comparable.

    Args:
        ax: Axes to draw on.
        matrix: Two-dimensional array, rows by columns.
        row_labels: Labels for the rows, top to bottom.
        col_labels: Labels for the columns, left to right.
        cmap: Colour map name.
        vmin: Lower limit of the colour scale. Share it across panels that are
            meant to be compared. Ignored when ``norm`` is given.
        vmax: Upper limit of the colour scale. Ignored when ``norm`` is given.
        norm: Optional Matplotlib normalisation, for a diverging scale that
            must be anchored at zero without being forced symmetric
            (``TwoSlopeNorm``). Passing both ``norm`` and ``vmin``/``vmax`` is
            an error in Matplotlib, so the limits are dropped when a norm is
            given.
        title: Axes title.
        xlabel: Label for the horizontal axis.
        ylabel: Label for the vertical axis.
        tick_every: Show every nth column label, so that 24 hourly labels do
            not overlap.

    Returns:
        The :class:`~matplotlib.image.AxesImage`, for attaching a colour bar.
    """
    if norm is not None:
        image = ax.imshow(
            matrix, aspect="auto", origin="upper", cmap=cmap, norm=norm,
            interpolation="nearest",
        )
    else:
        image = ax.imshow(
            matrix, aspect="auto", origin="upper", cmap=cmap, vmin=vmin,
            vmax=vmax, interpolation="nearest",
        )

    ax.set_xticks(range(0, len(col_labels), tick_every))
    ax.set_xticklabels(
        [col_labels[i] for i in range(0, len(col_labels), tick_every)]
    )
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    return image
