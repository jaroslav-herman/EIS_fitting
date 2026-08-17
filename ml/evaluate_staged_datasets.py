"""Evaluate topology models from already materialized staged datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .metrics import multiclass_brier, prediction_metrics
from .topology_classifier import _models


CLASSES = ["ONE_PROCESS", "TWO_PROCESS"]
PIPELINES = {
    "topology_raw": "A_raw",
    "topology_manual": "B_manual_range",
    "topology_ml_rf": "C_ml_range_random_forest",
    "topology_ml_hgb": "C_ml_range_hist_gradient_boosting",
}


def evaluate(dataset_dir: Path, output_dir: Path, seed: int = 42) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, overall, folds = [], [], []
    matrix_dir = output_dir / "confusion_matrices"
    for filename, pipeline in PIPELINES.items():
        path = Path(dataset_dir) / f"{filename}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        metadata = pd.read_csv(Path(dataset_dir) / f"{filename}_metadata.csv")
        for fold in sorted(metadata.validation_fold.astype(str).unique()):
            selection = metadata.validation_fold.astype(str) == fold
            train = selection & (metadata.sample_id.astype(str) != fold)
            test = selection & (metadata.sample_id.astype(str) == fold)
            if train.sum() == 0 or test.sum() == 0 or len(set(metadata.loc[train, "canonical_topology"])) < 2:
                continue
            for model_name in ("random_forest", "hist_gradient_boosting"):
                model = _models(seed)[model_name]
                model.fit(data["X"][train], metadata.loc[train, "canonical_topology"])
                predicted = model.predict(data["X"][test]).astype(str)
                probabilities = model.predict_proba(data["X"][test])
                classes = list(model.classes_)
                part = metadata.loc[test].copy()
                part["pipeline"] = pipeline
                part["topology_model"] = model_name
                part["topology_prediction"] = predicted
                part["topology_correct"] = predicted == part["canonical_topology"].to_numpy()
                for label in CLASSES:
                    part[f"probability_{label}"] = [float(p[classes.index(label)]) if label in classes else 0.0 for p in probabilities]
                all_rows.append(part)
                values = prediction_metrics(part.rename(columns={"canonical_topology": "true_topology", "topology_prediction": "predicted_topology"}), CLASSES)
                folds.append({"pipeline": pipeline, "topology_model": model_name, "held_out_sample": fold,
                              **{k: v for k, v in values.items() if isinstance(v, (int, float, np.floating))},
                              "brier": multiclass_brier(part.rename(columns={"canonical_topology": "true_topology", "topology_prediction": "predicted_topology"}), CLASSES),
                              "count": int(len(part))})
    predictions = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if predictions.empty:
        raise RuntimeError("No complete staged folds were available")
    for (pipeline, model_name), part in predictions.groupby(["pipeline", "topology_model"]):
        values = prediction_metrics(part.rename(columns={"canonical_topology": "true_topology", "topology_prediction": "predicted_topology"}), CLASSES)
        overall.append({"pipeline": pipeline, "topology_model": model_name,
                        **{k: v for k, v in values.items() if isinstance(v, (int, float, np.floating))},
                        "brier": multiclass_brier(part.rename(columns={"canonical_topology": "true_topology", "topology_prediction": "predicted_topology"}), CLASSES),
                        "count": int(len(part))})
        destination = matrix_dir / pipeline / model_name
        destination.mkdir(parents=True, exist_ok=True)
        for label, subset in [("aggregated", part), *[(str(f), g) for f, g in part.groupby("validation_fold")]]:
            matrix = confusion_matrix(subset.canonical_topology, subset.topology_prediction, labels=CLASSES)
            pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(destination / f"{label}.csv")
    predictions.to_csv(output_dir / "per_spectrum_predictions.csv", index=False)
    pd.DataFrame(overall).to_csv(output_dir / "overall_metrics.csv", index=False)
    pd.DataFrame(folds).to_csv(output_dir / "per_fold_metrics.csv", index=False)
    report = {"datasets": sorted(predictions.pipeline.unique()), "rows": int(len(predictions)),
              "folds": sorted(predictions.validation_fold.astype(str).unique())}
    (output_dir / "report.json").write_text(pd.Series(report).to_json(indent=2), encoding="utf-8")
    print(pd.DataFrame(overall).to_string(index=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate(args.dataset_dir, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
