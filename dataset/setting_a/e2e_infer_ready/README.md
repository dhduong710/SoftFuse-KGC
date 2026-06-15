# PrimeKG E2E Infer-Ready Packages

This directory contains the three PrimeKG packages consumed by `infer.py`:

- `backbone_raw`
- `soft_support_raw`
- `retrieval_main`

Each package contains `train.json`, `valid.json`, `test.json`, and a
`prep_manifest.json`.

To rebuild these files, run:

```bash
python3 scripts/e2e/build_infer_ready.py
```

The script reuses `dataset/setting_a/backbone_ready/train.json` as the train
split. If that file is missing, rebuild the backbone-ready data package before
running the E2E preparation step.
