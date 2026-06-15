# PharmKG E2E Infer-Ready Packages

The row folders here are generated from the PharmKG SoftFuse transfer outputs.

If any `train.json` file is absent in a lightweight checkout, regenerate the packages with:

```bash
python scripts/pharmkg/e2e/prepare_e2e_ready.py
```

This requires `dataset/setting_c_pharmkg/softfuse_ready/train.json`, which can be rebuilt with `python scripts/pharmkg/build_softfuse_ready_package.py`.
