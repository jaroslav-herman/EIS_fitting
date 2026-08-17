from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def prediction_metrics(frame: pd.DataFrame, classes: list[str]) -> dict[str, object]:
    if frame.empty:
        return {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan}
    true = frame["true_topology"].to_numpy()
    predicted = frame["predicted_topology"].to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=classes, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=classes, average="macro", zero_division=0)),
        "per_class": pd.DataFrame({"class": classes, "precision": precision, "recall": recall, "f1": f1, "support": support}),
        "confusion_matrix": confusion_matrix(true, predicted, labels=classes),
    }


def confidence_metrics(frame: pd.DataFrame, thresholds=(0.5, 0.7, 0.8, 0.9)) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        selected = frame[frame["confidence"] >= threshold]
        if len(selected) == 0:
            continue
        rows.append({"threshold": threshold, "count": len(selected), "accuracy": float(selected["correct"].mean())})
    return pd.DataFrame(rows)


def multiclass_brier(frame: pd.DataFrame, classes: list[str]) -> float | None:
    if frame.empty:
        return None
    values = []
    for _, row in frame.iterrows():
        target = np.asarray([float(row.get(f"probability_{c}", 0.0)) for c in classes])
        truth = np.asarray([float(row["true_topology"] == c) for c in classes])
        values.append(float(np.mean((target - truth) ** 2)))
    return float(np.mean(values))


def export_confusion_matrices(
    predictions: pd.DataFrame,
    directory,
    classes: list[str] | None = None,
) -> None:
    """Export one CSV per model/fold and one aggregate CSV per model."""
    directory = __import__("pathlib").Path(directory) / "confusion_matrices"
    directory.mkdir(parents=True, exist_ok=True)
    classes = classes or sorted(
        set(predictions["true_topology"].dropna())
        | set(predictions["predicted_topology"].dropna())
    )
    for model_name, model_frame in predictions.groupby("model_name"):
        model_directory = directory / str(model_name)
        model_directory.mkdir(parents=True, exist_ok=True)
        for fold, fold_frame in model_frame.groupby("validation_fold"):
            matrix = confusion_matrix(
                fold_frame["true_topology"],
                fold_frame["predicted_topology"],
                labels=classes,
            )
            pd.DataFrame(matrix, index=classes, columns=classes).rename_axis(
                "true_topology"
            ).to_csv(model_directory / f"held_out_{fold}.csv")
        matrix = confusion_matrix(
            model_frame["true_topology"],
            model_frame["predicted_topology"],
            labels=classes,
        )
        pd.DataFrame(matrix, index=classes, columns=classes).rename_axis(
            "true_topology"
        ).to_csv(model_directory / "aggregated.csv")


def regression_range_metrics(values, summary: bool = False) -> dict[str, float | bool]:
    """Calculate boundary and interval metrics in log-frequency space."""
    if isinstance(values, pd.DataFrame):
        frame = values
        if frame.empty:
            return {"count": 0}
        result: dict[str, float] = {"count": float(len(frame))}
        for column in ("error_log_f_min", "error_log_f_max"):
            result[f"mae_{column[6:]}"] = float(frame[column].abs().mean())
            result[f"rmse_{column[6:]}"] = float(np.sqrt(np.mean(frame[column] ** 2)))
        for threshold in (1.5, 2.0, 3.0):
            result[f"f_min_within_factor_{threshold:g}"] = float(frame[f"f_min_within_factor_{threshold:g}"].mean() * 100.0)
            result[f"f_max_within_factor_{threshold:g}"] = float(frame[f"f_max_within_factor_{threshold:g}"].mean() * 100.0)
        for threshold in (0.5, 0.75, 0.9):
            result[f"iou_above_{threshold:g}"] = float((frame["range_iou"] > threshold).mean() * 100.0)
        result["predicted_order_valid_percent"] = float(frame["predicted_order_valid"].mean() * 100.0)
        result["predicted_measured_range_valid_percent"] = float(frame["predicted_measured_range_valid"].mean() * 100.0)
        return result
    manual_min = float(values["manual_log_f_min"])
    manual_max = float(values["manual_log_f_max"])
    predicted_min = float(values["predicted_log_f_min"])
    predicted_max = float(values["predicted_log_f_max"])
    intersection = max(0.0, min(manual_max, predicted_max) - max(manual_min, predicted_min))
    union = max(manual_max, predicted_max) - min(manual_min, predicted_min)
    result = {
        "error_log_f_min": predicted_min - manual_min,
        "error_log_f_max": predicted_max - manual_max,
        "range_iou": intersection / union if union > 0 else 1.0,
        "predicted_order_valid": bool(predicted_min < predicted_max),
        "predicted_measured_range_valid": bool(
            10**predicted_min >= float(values["measured_f_min"])
            and 10**predicted_max <= float(values["measured_f_max"])
            and predicted_min < predicted_max
        ),
    }
    for threshold in (1.5, 2.0, 3.0):
        limit = np.log10(threshold)
        result[f"f_min_within_factor_{threshold:g}"] = bool(abs(result["error_log_f_min"]) <= limit)
        result[f"f_max_within_factor_{threshold:g}"] = bool(abs(result["error_log_f_max"]) <= limit)
    return result
