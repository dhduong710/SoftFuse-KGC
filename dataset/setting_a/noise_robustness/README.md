# Noise Robustness Variants

These variants are generated from `dataset/setting_a/e2e_infer_ready/retrieval_main`.

If `train.json` is absent for the larger noise variants in a lightweight checkout, rebuild the variants with:

```bash
python scripts/sensitivity/build_sensitivity_manifest.py
python scripts/sensitivity/build_noise_variants.py
```
