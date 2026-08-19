"""Evaluate local point validity against saved expert masks."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import load_eisfit_projects
from .point_validity import detect_valid_points


SAMPLES = ("129", "140", "150", "157", "159", "181")
DEFAULT_THRESHOLDS = (2.0, 3.0, 4.0, 5.0, 6.0)


def _manual_mask(record, selected_indices: np.ndarray) -> np.ndarray | None:
    if record.cleaned_frequency is None:
        return None
    keys = Counter((float(f), float(r), float(i)) for f, r, i in zip(record.cleaned_frequency, record.cleaned_z_real, record.cleaned_z_imag))
    result = np.zeros(selected_indices.size, dtype=bool)
    for position, index in enumerate(selected_indices):
        key = (float(record.frequency[index]), float(record.z_real[index]), float(record.z_imag[index]))
        if keys[key] > 0:
            result[position] = True
            keys[key] -= 1
    return result


def _point_metrics(manual_valid: np.ndarray, automatic_valid: np.ndarray) -> dict:
    manual_rejected = ~manual_valid; automatic_rejected = ~automatic_valid
    tp = int(np.sum(manual_rejected & automatic_rejected)); tn = int(np.sum(manual_valid & automatic_valid))
    fp = int(np.sum(manual_valid & automatic_rejected)); fn = int(np.sum(manual_rejected & automatic_valid))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / len(manual_valid) if len(manual_valid) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "precision": precision, "recall": recall,
            "F1": f1, "specificity": specificity, "accuracy": accuracy}


def _disagreement_categories(frequency: np.ndarray, manual_valid: np.ndarray, automatic_valid: np.ndarray) -> Counter:
    differing = np.flatnonzero(manual_valid != automatic_valid)
    result = Counter()
    if differing.size == 0:
        return result
    groups = np.split(differing, np.where(np.diff(differing) > 1)[0] + 1)
    log_frequency = np.log10(frequency)
    low, high = np.quantile(log_frequency, (1 / 3, 2 / 3))
    for group in groups:
        result["adjacent_group" if group.size > 1 else "isolated_point"] += 1
        center = float(np.mean(log_frequency[group]))
        result["low_frequency" if center <= low else "high_frequency" if center >= high else "middle_frequency"] += 1
    return result


def _plot_example(record, frequency, impedance, manual_valid, automatic_valid, score, destination: Path, threshold: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination.parent.mkdir(parents=True, exist_ok=True)
    log_frequency = np.log10(frequency)
    manual_rejected = ~manual_valid; automatic_rejected = ~automatic_valid
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(impedance.real, -impedance.imag, "-", color="0.65", linewidth=0.8)
    axes[0].scatter(impedance.real[manual_rejected], -impedance.imag[manual_rejected], label="manual rejected", marker="x")
    axes[0].scatter(impedance.real[automatic_rejected], -impedance.imag[automatic_rejected], label="automatic rejected", facecolors="none", edgecolors="tab:red")
    axes[0].set_xlabel("Re(Z)"); axes[0].set_ylabel("-Im(Z)"); axes[0].set_title("Nyquist")
    axes[1].plot(log_frequency, impedance.real, label="Re(Z)")
    axes[1].plot(log_frequency, impedance.imag, label="Im(Z)")
    axes[1].scatter(log_frequency[automatic_rejected], impedance.real[automatic_rejected], color="tab:red", s=15)
    axes[1].set_xlabel("log10(frequency)"); axes[1].set_title("Impedance trajectory"); axes[1].legend()
    axes[2].plot(log_frequency, score, ".-")
    axes[2].axhline(threshold, color="tab:red", linestyle="--")
    axes[2].set_xlabel("log10(frequency)"); axes[2].set_ylabel("anomaly score"); axes[2].set_title("Local score")
    fig.suptitle(f"{record.sample_id} / cycle {record.cycle}")
    fig.tight_layout(); fig.savefig(destination, dpi=140); plt.close(fig)


def evaluate(projects: list[Path], output: Path, thresholds=DEFAULT_THRESHOLDS, seed: int = 42) -> dict:
    mapping = {str(path): path.name.split(".")[0] for path in projects}
    extraction = load_eisfit_projects(projects, mapping)
    output.mkdir(parents=True, exist_ok=True)
    rows = []; category_rows = []; skipped = []
    plotted = set()
    for record in extraction.records:
        selected = np.flatnonzero((record.frequency >= record.manual_f_min) & (record.frequency <= record.manual_f_max))
        manual_valid = _manual_mask(record, selected)
        if manual_valid is None or selected.size < 5:
            skipped.append({"spectrum_id": record.spectrum_id, "sample_id": record.sample_id, "reason": "missing_manual_mask_or_too_few_points"})
            continue
        frequency = record.frequency[selected]; impedance = record.impedance[selected]
        # Scores are threshold-independent in one robust local pass.  Compute
        # them once, then derive the complete sweep; this avoids five costly
        # detector passes over every spectrum.  Iterative refinement remains
        # available through detect_valid_points for later selected thresholds.
        _, score, diagnostics = detect_valid_points(frequency, impedance, threshold=max(thresholds), max_iterations=1)
        context_valid = np.isfinite(score)
        for threshold in thresholds:
            automatic_valid = context_valid & (score <= threshold)
            metrics = _point_metrics(manual_valid, automatic_valid)
            rows.append({"spectrum_id": record.spectrum_id, "sample_id": record.sample_id, "threshold": threshold,
                         "point_count": selected.size, "manual_rejected": int((~manual_valid).sum()),
                         "automatic_rejected": int((~automatic_valid).sum()), "differing_points": int(np.sum(manual_valid != automatic_valid)),
                         **metrics})
            categories = _disagreement_categories(frequency, manual_valid, automatic_valid)
            for category, count in categories.items():
                category_rows.append({"spectrum_id": record.spectrum_id, "sample_id": record.sample_id, "threshold": threshold,
                                      "category": category, "count": count})
            if threshold == 4.0 and len(plotted) < 6 and record.sample_id not in plotted:
                _plot_example(record, frequency, impedance, manual_valid, automatic_valid, score, output / "plots" / f"sample_{record.sample_id}.png", threshold)
                plotted.add(record.sample_id)
    frame = pd.DataFrame(rows); categories = pd.DataFrame(category_rows)
    frame.to_csv(output / "per_spectrum.csv", index=False); categories.to_csv(output / "disagreement_categories.csv", index=False)
    summary_rows = []
    for threshold, part in frame.groupby("threshold"):
        metrics = _point_metrics(np.repeat(True, 0), np.repeat(True, 0)) if part.empty else {key: int(part[key].sum()) for key in ("TP", "TN", "FP", "FN")}
        tp, tn, fp, fn = metrics.values()
        precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0; accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        summary_rows.append({"threshold": threshold, "TP": tp, "TN": tn, "FP": fp, "FN": fn, "precision": precision, "recall": recall,
                             "F1": f1, "specificity": specificity, "accuracy": accuracy,
                             "manual_rejection_percent": 100 * (part.manual_rejected.sum() / part.point_count.sum()),
                             "automatic_rejection_percent": 100 * (part.automatic_rejected.sum() / part.point_count.sum()),
                             "spectra_with_zero_automatic_rejections_percent": 100 * (part.automatic_rejected == 0).mean(),
                             "mean_automatic_rejections_per_spectrum": part.automatic_rejected.mean(),
                             "mean_differing_fraction": (part.differing_points / part.point_count).mean(), "spectra": len(part)})
    summary = pd.DataFrame(summary_rows); summary.to_csv(output / "threshold_summary.csv", index=False)
    per_sample = []
    for (sample, threshold), part in frame.groupby(["sample_id", "threshold"]):
        tp, tn, fp, fn = [int(part[key].sum()) for key in ("TP", "TN", "FP", "FN")]
        precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0; f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_sample.append({"sample_id": sample, "threshold": threshold, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                           "precision": precision, "recall": recall, "F1": f1, "specificity": specificity,
                           "accuracy": (tp + tn) / (tp + tn + fp + fn), "spectra": len(part)})
    pd.DataFrame(per_sample).to_csv(output / "per_sample_summary.csv", index=False)
    pd.DataFrame(skipped).to_csv(output / "skipped.csv", index=False)
    report = {"records_extracted": len(extraction.records), "exclusions": extraction.exclusion_counts, "evaluated_spectra": int(frame["spectrum_id"].nunique()) if not frame.empty else 0,
              "thresholds": list(map(float, thresholds)), "output": str(output), "detector_does_not_use_manual_mask": True,
              "threshold_sweep": "common one-pass anomaly scores; no repeated detector calls"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projects", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ml/analysis/point_validity"))
    parser.add_argument("--threshold", type=float, action="append", dest="thresholds")
    args = parser.parse_args()
    evaluate(args.projects, args.output, tuple(args.thresholds or DEFAULT_THRESHOLDS))


if __name__ == "__main__":
    main()
