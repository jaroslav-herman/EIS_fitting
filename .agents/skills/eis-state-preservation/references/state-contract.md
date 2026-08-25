# State and invalidation contract

## Canonical ownership

- `CycleState`: one spectrum's measured arrays, point selection, frequency window, model/parameters, fit, DRT/KK caches, and metadata.
- `ProjectState`: source/control, available and active cycles, default parameters, and project-wide frequency window.
- `LoadedProject`: dataframe plus `ProjectState`; unloaded cycles are materialized through `load_cycle()`.
- `eis_project.py`: project JSON serialization/loading and exports. Current outer format/version is `eis-fitting-project` / `4`.

## Dependency matrix

| Change | Preserve | Clear or recompute |
| --- | --- | --- |
| Display mode, zoom, legend | all scientific state | artists/layout only |
| Metadata not used by a model | measurements, masks, fit, DRT/KK | explorer/export view |
| Manual point or outlier mask | measurements, parameters | fit; DRT and KK derived from included points |
| Frequency window | measurements, full-length masks, parameters | fit; DRT/KK when effective included mask changes |
| Initial parameter value/bound/fixed flag | measurements, masks, DRT/KK independent of EEC fit | EEC fit and percentage errors |
| Circuit/model | measurements and masks | parameter compatibility, EEC fit; use established model replacement invalidation |
| New fit result | measurements, masks | atomically replace fitted parameters, errors, smooth curve, and fit-at-data curve |
| DRT result | measurements, fit | store tau/gamma and the included-mask/cache inputs used |
| Raw spectrum identity | metadata only when still valid | all masks and derived results unless remapped by verified point identity |

`invalidate_drt_cache()` also clears KK fields. Use it when the active point set changes. `clear_fit()` clears fitted arrays and parameter errors but intentionally leaves measurements and initial settings.

## Persistence pairs

When adding or renaming a persisted field, inspect both `_cycle_to_dict()`/`_state_to_payload()` and `load_project_file()`. Validate these pairs together:

- `fit_frequency_hz` ↔ complex `fit_impedance` lengths.
- raw point count ↔ `manually_included`, `outliers`, fit-at-data, and saved included masks.
- DRT `tau` ↔ `gamma` lengths.
- parameter list ↔ fitted-parameter vector.
- selected `drt_label` ↔ corresponding saved Ridge or Hybrid arrays.

Project signatures for unsaved-change detection are derived from serialized state, so new persistent scientific fields must participate in the payload rather than living only on widgets.
