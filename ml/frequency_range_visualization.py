from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataset import SpectrumRecord


def plot_frequency_range_results(records: list[SpectrumRecord], predictions: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    voltage = pd.DataFrame({"voltage": [r.voltage for r in records], "f_min": [r.manual_f_min for r in records], "f_max": [r.manual_f_max for r in records], "sample": [r.sample_id for r in records]})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for sample, group in voltage.groupby("sample"):
        axes[0].scatter(group.voltage, np.log10(group.f_min), s=8, alpha=0.35, label=sample)
        axes[1].scatter(group.voltage, np.log10(group.f_max), s=8, alpha=0.35, label=sample)
    axes[0].set(xlabel="Voltage", ylabel="manual log10(f_min / Hz)")
    axes[1].set(xlabel="Voltage", ylabel="manual log10(f_max / Hz)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / "manual_range_vs_voltage.png", dpi=160)
    plt.close(fig)

    for model_name, frame in predictions.groupby("model_name"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(frame.manual_log_f_min, frame.predicted_log_f_min, s=7, alpha=0.3)
        axes[1].scatter(frame.manual_log_f_max, frame.predicted_log_f_max, s=7, alpha=0.3)
        for axis, label in zip(axes, ("f_min", "f_max")):
            low, high = axis.get_xlim()
            axis.plot([low, high], [low, high], "k--", linewidth=1)
            axis.set(xlabel=f"manual log10({label} / Hz)", ylabel=f"predicted log10({label} / Hz)")
        fig.suptitle(model_name)
        fig.tight_layout()
        fig.savefig(directory / f"predicted_vs_manual_{model_name}.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(frame.voltage, frame.error_log_f_min, s=7, alpha=0.3)
        axes[1].scatter(frame.voltage, frame.error_log_f_max, s=7, alpha=0.3)
        axes[0].set(xlabel="Voltage", ylabel="log10 f_min error")
        axes[1].set(xlabel="Voltage", ylabel="log10 f_max error")
        fig.tight_layout()
        fig.savefig(directory / f"prediction_error_vs_voltage_{model_name}.png", dpi=160)
        plt.close(fig)

    by_id = {r.spectrum_id: r for r in records}
    for index, (_, row) in enumerate(predictions.head(3).iterrows(), start=1):
        record = by_id.get(row["spectrum_id"])
        if record is None:
            continue
        fig, axis = plt.subplots(figsize=(6, 5))
        axis.plot(record.frequency, record.impedance.real, "o-", markersize=2, label="Re(Z)")
        axis.set_xscale("log")
        axis.axvspan(row.manual_f_min, row.manual_f_max, alpha=0.2, label="manual range")
        axis.axvspan(row.predicted_f_min, row.predicted_f_max, alpha=0.2, color="orange", label="predicted range")
        axis.set(xlabel="Frequency / Hz", ylabel="Re(Z) / Ohm", title=f"{record.sample_id}, {record.cycle}")
        axis.legend()
        fig.tight_layout()
        fig.savefig(directory / f"example_spectrum_{index}.png", dpi=160)
        plt.close(fig)
