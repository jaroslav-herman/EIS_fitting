from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .dataset import SpectrumRecord


@dataclass
class SpectrumPreprocessor:
    grid_size: int = 64
    use_metadata: bool = False
    spectrum_mode: str = "raw"
    frequency_grid_: np.ndarray | None = None
    fill_values_: np.ndarray | None = None
    metadata_mean_: np.ndarray | None = None
    metadata_scale_: np.ndarray | None = None
    feature_names_: list[str] | None = None

    def fit(self, records: list[SpectrumRecord]) -> "SpectrumPreprocessor":
        if not records:
            raise ValueError("Cannot fit preprocessing without spectra")
        minimum = min(float(np.min(np.log10(r.arrays(self.spectrum_mode)[0]))) for r in records)
        maximum = max(float(np.max(np.log10(r.arrays(self.spectrum_mode)[0]))) for r in records)
        if not maximum > minimum:
            raise ValueError("Training spectra do not span a frequency range")
        self.frequency_grid_ = np.linspace(minimum, maximum, self.grid_size)
        raw = np.vstack([self._spectrum_features(record) for record in records])
        self.fill_values_ = np.nanmean(raw, axis=0)
        self.fill_values_[~np.isfinite(self.fill_values_)] = 0.0
        self.feature_names_ = [f"logf_{i:03d}" for i in range(self.grid_size)] + [f"zreal_{i:03d}" for i in range(self.grid_size)] + [f"zimag_{i:03d}" for i in range(self.grid_size)]
        if self.use_metadata:
            metadata = np.asarray([[r.voltage, r.current, r.time] for r in records], dtype=float)
            self.metadata_mean_ = np.nanmean(metadata, axis=0)
            self.metadata_scale_ = np.nanstd(metadata, axis=0)
            self.metadata_mean_[~np.isfinite(self.metadata_mean_)] = 0.0
            self.metadata_scale_[~np.isfinite(self.metadata_scale_) | (self.metadata_scale_ <= 1e-12)] = 1.0
            self.feature_names_ += ["voltage", "current", "time"]
        return self

    def _spectrum_features(self, record: SpectrumRecord) -> np.ndarray:
        grid = self.frequency_grid_
        if grid is None:
            raise RuntimeError("Preprocessor has not been fitted")
        frequency, impedance = record.arrays(self.spectrum_mode)
        logf = np.log10(frequency)
        order = np.argsort(logf)
        logf, z = logf[order], impedance[order]
        scale = max(float(np.nanmedian(np.abs(z))), np.finfo(float).eps)
        real = np.interp(grid, logf, z.real, left=np.nan, right=np.nan) / scale
        imag = np.interp(grid, logf, z.imag, left=np.nan, right=np.nan) / scale
        # log-frequency is represented by the common grid; NaNs mark missing
        # regions and are filled only with training-fold statistics in transform.
        return np.concatenate([grid, real, imag])

    def transform(self, records: list[SpectrumRecord]) -> np.ndarray:
        if self.frequency_grid_ is None or self.fill_values_ is None:
            raise RuntimeError("Preprocessor has not been fitted")
        result = np.vstack([self._spectrum_features(r) for r in records])
        missing = ~np.isfinite(result)
        rows, columns = np.where(missing)
        result[rows, columns] = self.fill_values_[columns]
        if self.use_metadata:
            metadata = np.asarray([[r.voltage, r.current, r.time] for r in records], dtype=float)
            missing = ~np.isfinite(metadata)
            metadata[missing] = np.take(self.metadata_mean_, np.where(missing)[1])
            metadata = (metadata - self.metadata_mean_) / self.metadata_scale_
            result = np.hstack([result, metadata])
        return result

    def fit_transform(self, records: list[SpectrumRecord]) -> np.ndarray:
        return self.fit(records).transform(records)
