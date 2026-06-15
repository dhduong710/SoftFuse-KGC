# PrimeKG Backbone Candidates

This directory contains the raw backbone candidate lists used by
the first PrimeKG soft-support step:

- `valid_top20_raw.json`
- `test_top20_raw.json`

The optional train candidate file, `train_top20_raw.json`, is not checked in
here because the current soft-support scripts in `scripts/soft_support` only
consume the validation split. If a downstream training experiment requires it,
export it from the included aligned train evidence with:

```bash
python3 scripts/backbone/export_primekg_train_candidates.py
```

The script writes `dataset/setting_a/backbone_candidates/train_top20_raw.json`.
