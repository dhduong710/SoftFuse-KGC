# PharmKG SoftFuse Ready Package

This directory stores the PharmKG rows used by the SoftFuse transfer pipeline.

If `train.json` is absent in a lightweight checkout, regenerate it with:

```bash
python scripts/pharmkg/build_softfuse_ready_package.py
```

That script reads `dataset/setting_c_pharmkg/splits`, `dataset/setting_c_pharmkg/graph`, and the R-GCN top-20 files under `dataset/setting_c_pharmkg/baseline_outputs/rgcn`.
