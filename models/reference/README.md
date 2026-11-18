# Archived feature snapshot

`archived_features.csv` is the snapshot used in the multi-horizon model comparison,
copied byte-for-byte from the companion study's `fuentes/variables_pit.csv`.
Observation-date and source columns are preserved. The verification report records
its SHA-256 checksum.

The reference training command overlays available archived values on the lagged
revised panel. The equity risk premium is calculated separately from historical
P/E and DGS10. The archive-only sensitivity excludes it and leaves archive gaps
missing. [Coverage and results](../../docs/experiments/reference-robustness.md#point-in-time-inputs)

Run `python -m scripts.train_selected_model` from the repository root to verify the
selected candidate's out-of-sample metrics and export the frozen 1962–2020 fit.
See the [main README](../../README.md#selected-model-and-experiments) for the choice
of model, economic accounting and outstanding limitations.
