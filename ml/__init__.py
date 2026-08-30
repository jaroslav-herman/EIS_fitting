"""Machine-learning utilities for EIS topology experiments.

This package is deliberately independent of the Tkinter application and does
not alter the EIS project format or fitting code.
"""

from .dataset import SpectrumRecord, ExtractionReport, canonical_electrochemical_topology, load_eisfit_projects
from .frequency_range import FrequencyRangeExperiment, run_frequency_range_experiment
from .preprocessing import SpectrumPreprocessor
from .topology_classifier import TopologyExperiment, run_topology_experiment

__all__ = [
    "SpectrumRecord",
    "ExtractionReport",
    "load_eisfit_projects",
    "canonical_electrochemical_topology",
    "FrequencyRangeExperiment",
    "run_frequency_range_experiment",
    "SpectrumPreprocessor",
    "TopologyExperiment",
    "run_topology_experiment",
    "PipelineBundle",
    "train_bundle",
    "run_pipeline",
]


def __getattr__(name):
    if name in {"PipelineBundle", "train_bundle", "run_pipeline"}:
        from .number_aware_pipeline import PipelineBundle, run_pipeline, train_bundle
        return {"PipelineBundle": PipelineBundle, "train_bundle": train_bundle, "run_pipeline": run_pipeline}[name]
    raise AttributeError(name)
