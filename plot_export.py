"""Export the numerical data currently displayed by a Matplotlib axes."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _label(artist, fallback: str) -> str:
    label = artist.get_label()
    if not label or str(label).startswith("_"):
        return fallback
    return str(label)


def extract_displayed_series(axes) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return visible plotted series as the x/y values held by their artists."""
    series: list[tuple[str, np.ndarray, np.ndarray]] = []
    line_number = 1
    for line in axes.lines:
        if not line.get_visible():
            continue
        x_values = np.asarray(line.get_xdata(), dtype=object)
        y_values = np.asarray(line.get_ydata(), dtype=object)
        if x_values.size == 0 or y_values.size == 0:
            continue
        size = min(x_values.size, y_values.size)
        series.append(
            (
                _label(line, f"Series {line_number}"),
                x_values[:size],
                y_values[:size],
            )
        )
        line_number += 1

    for collection in axes.collections:
        if not collection.get_visible():
            continue
        if hasattr(collection, "get_segments"):
            segments = collection.get_segments()
            if segments:
                for segment_number, segment in enumerate(segments, start=1):
                    values = np.asarray(segment)
                    if values.ndim != 2 or values.shape[1] < 2 or not values.size:
                        continue
                    series.append(
                        (
                            _label(collection, f"Series {line_number}")
                            + f" ({segment_number})",
                            values[:, 0],
                            values[:, 1],
                        )
                    )
                line_number += 1
                continue
        if hasattr(collection, "get_offsets"):
            offsets = np.asarray(collection.get_offsets())
            if offsets.ndim == 2 and offsets.shape[1] >= 2 and offsets.size:
                series.append(
                    (
                        _label(collection, f"Series {line_number}"),
                        offsets[:, 0],
                        offsets[:, 1],
                    )
                )
                line_number += 1
    return series


def write_displayed_csv(axes, path: str | Path) -> int:
    """Write visible axes series in long CSV form and return row count."""
    series = extract_displayed_series(axes)
    if not series:
        return 0
    x_label = axes.get_xlabel().strip() or "X"
    y_label = axes.get_ylabel().strip() or "Y"
    if x_label == y_label:
        y_label = "Y"
    rows_written = 0
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Series", "Point", x_label, y_label])
        for label, x_values, y_values in series:
            for point, (x_value, y_value) in enumerate(zip(x_values, y_values), start=1):
                writer.writerow([label, point, x_value, y_value])
                rows_written += 1
    return rows_written
