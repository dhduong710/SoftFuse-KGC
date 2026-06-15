# PrimeKG Aligned Evidence

This directory stores candidate rows aligned with evidence subgraphs:

- `train_aligned_evidence.json`
- `valid_aligned_evidence.json`
- `test_aligned_evidence.json`

The manifest keeps the historical raw/evidence source paths for provenance.
Those source paths are not required by the current `scripts/soft_support`
pipeline. If a later training script needs a missing raw train candidate file,
run:

```bash
python3 scripts/backbone/export_primekg_train_candidates.py
```

The script writes `dataset/setting_a/backbone_candidates/train_top20_raw.json`.
