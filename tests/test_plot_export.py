import csv

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

from plot_export import extract_displayed_series, write_displayed_csv


def test_extracts_displayed_transformed_lines_and_scatter():
    axes = Figure().add_subplot(111)
    axes.set_xlabel("Re(Z) / Ohm")
    axes.set_ylabel("-Im(Z) / Ohm")
    axes.plot([1, 2], [3, 4], label="measured")
    axes.plot([1, 2], [-3, -4], label="fit")
    axes.scatter([1.5], [3.5], label="fit points")

    series = extract_displayed_series(axes)

    assert [item[0] for item in series] == ["measured", "fit", "fit points"]
    assert series[1][2].tolist() == [-3, -4]
    assert series[2][1].tolist() == [1.5]


def test_writes_all_visible_series_and_skips_hidden_lines(tmp_path):
    axes = Figure().add_subplot(111)
    axes.plot([1, 2], [10, 20], label="visible")
    hidden, = axes.plot([1, 2], [30, 40], label="hidden")
    hidden.set_visible(False)

    path = tmp_path / "displayed.csv"
    assert write_displayed_csv(axes, path) == 2

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["Series", "Point", "X", "Y"]
    assert rows[1][0] == "visible"
    assert rows[-1][2:] == ["2", "20"]


def test_empty_axes_has_no_export_rows(tmp_path):
    axes = Figure().add_subplot(111)
    assert extract_displayed_series(axes) == []
    assert write_displayed_csv(axes, tmp_path / "empty.csv") == 0
