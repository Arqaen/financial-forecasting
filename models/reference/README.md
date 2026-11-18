# Archived feature snapshot

`archived_features.csv` is the snapshot used in the multi-horizon model comparison,
copied byte-for-byte from the companion study's `fuentes/variables_pit.csv`.
Observation-date and source columns are preserved. The verification report records
its SHA-256 checksum.

The reference training command overlays nonmissing archived values for the original
features on the lagged revised panel. It does not replace missing values with zero,
and it does not imply that every input is point-in-time. In particular, the ERP
definition and its source data remain those of the historical experiment.

Run `python -m scripts.train_selected_model` from the repository root to verify the
selected candidate's out-of-sample metrics and export the frozen 1962–2020 fit.
See the [main README](../../README.md#selected-model-and-experiments) for the choice
of model, economic accounting and outstanding limitations.
