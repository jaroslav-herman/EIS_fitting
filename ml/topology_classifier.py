from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .dataset import SpectrumRecord
from .metrics import (
    confidence_metrics,
    export_confusion_matrices,
    multiclass_brier,
    prediction_metrics,
)
from .preprocessing import SpectrumPreprocessor


def _models(seed: int) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150, random_state=seed),
    }


@dataclass
class TopologyExperiment:
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    overall_metrics: dict[str, object]
    confidence: dict[str, pd.DataFrame] = field(default_factory=dict)
    excluded: pd.DataFrame = field(default_factory=pd.DataFrame)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.predictions.to_csv(directory / "predictions.csv", index=False)
        self.fold_metrics.to_csv(directory / "fold_metrics.csv", index=False)
        for name, frame in self.confidence.items():
            frame.to_csv(directory / f"confidence_{name}.csv", index=False)
        if not self.excluded.empty:
            self.excluded.to_csv(directory / "exclusions.csv", index=False)
        export_confusion_matrices(self.predictions, directory)
        summary_rows = []
        for model_name, values in self.overall_metrics.items():
            if isinstance(values, dict):
                row = {"model_name": model_name}
                row.update({name: value for name, value in values.items() if isinstance(value, (int, float))})
                brier = self.overall_metrics.get(f"{model_name}_brier")
                if isinstance(brier, (int, float)):
                    row["brier"] = brier
                summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(directory / "overall_metrics.csv", index=False)


def _records_for(records: list[SpectrumRecord], sample_ids: Iterable[str]) -> list[SpectrumRecord]:
    selected = set(sample_ids)
    return [record for record in records if record.sample_id in selected]


def _fit_predict(model, preprocessor, train, test, classes, model_name, fold, use_metadata):
    x_train = preprocessor.fit_transform(train)
    x_test = preprocessor.transform(test)
    labels = np.asarray([r.topology_label for r in train])
    model.fit(x_train, labels)
    probabilities = model.predict_proba(x_test)
    model_classes = list(model.classes_)
    rows = []
    for record, predicted, probability in zip(test, model.predict(x_test), probabilities):
        row = {
            "spectrum_id": record.spectrum_id,
            "source_project": record.source_project,
            "sample_id": record.sample_id,
            "cycle": record.cycle,
            "voltage": record.voltage,
            "current": record.current,
            "time": record.time,
            "device_setup": record.device_setup,
            "original_eec_topology": record.original_eec_topology,
            "electrochemical_topology": record.electrochemical_topology,
            "l0_required_in_manual_fit": record.l0_required_in_manual_fit,
            "true_topology": record.electrochemical_topology,
            "predicted_topology": str(predicted),
            "predicted_electrochemical_topology": str(predicted),
            "confidence": float(np.max(probability)),
            "model_name": model_name,
            "validation_fold": fold,
            "feature_mode": "spectrum_plus_metadata" if use_metadata else "spectrum_only",
            "spectrum_representation": preprocessor.spectrum_mode,
        }
        for label in classes:
            row[f"probability_{label}"] = float(probability[model_classes.index(label)]) if label in model_classes else 0.0
        row["correct"] = bool(row["true_topology"] == row["predicted_topology"])
        rows.append(row)
    return rows


def run_topology_experiment(
    records: list[SpectrumRecord],
    *,
    model_names: tuple[str, ...] = ("logistic_regression", "random_forest", "hist_gradient_boosting"),
    use_metadata: bool = False,
    spectrum_mode: str = "raw",
    grid_size: int = 64,
    seed: int = 42,
) -> TopologyExperiment:
    """Run leave-one-sample-out validation with fold-local preprocessing."""
    if spectrum_mode not in {"raw", "cleaned"}:
        raise ValueError("spectrum_mode must be 'raw' or 'cleaned'")
    records = [record for record in records if spectrum_mode == "raw" or (record.cleaned_frequency is not None and record.cleaned_z_real is not None and record.cleaned_z_imag is not None)]
    samples = sorted({record.sample_id for record in records})
    classes = sorted({record.electrochemical_topology for record in records})
    if len(samples) < 2:
        raise ValueError("At least two sample IDs are required for sample-based validation")
    all_rows: list[dict] = []
    fold_rows: list[dict] = []
    for held_out in samples:
        train = [r for r in records if r.sample_id != held_out]
        test = [r for r in records if r.sample_id == held_out]
        train_classes = {r.topology_label for r in train}
        for name in model_names:
            if name not in _models(seed):
                raise ValueError(f"Unknown model: {name}")
            if len(train_classes) < 2:
                continue
            rows = _fit_predict(_models(seed)[name], SpectrumPreprocessor(grid_size, use_metadata, spectrum_mode), train, test, classes, name, held_out, use_metadata)
            all_rows.extend(rows)
            fold_frame = pd.DataFrame(rows)
            metrics = prediction_metrics(fold_frame, classes)
            fold_rows.append({"model_name": name, "held_out_sample": held_out, **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}})
    predictions = pd.DataFrame(all_rows)
    overall: dict[str, object] = {}
    confidence: dict[str, pd.DataFrame] = {}
    for name in model_names:
        frame = predictions[predictions["model_name"] == name]
        overall[name] = prediction_metrics(frame, classes)
        confidence[name] = confidence_metrics(frame)
        overall[f"{name}_brier"] = multiclass_brier(frame, classes)
    return TopologyExperiment(predictions, pd.DataFrame(fold_rows), overall, confidence)
