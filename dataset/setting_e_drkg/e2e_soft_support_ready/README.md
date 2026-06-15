# DRKG E2E Soft-Support Packages

These packages are generated from `dataset/setting_e_drkg/softfuse_ready`.

To rebuild the R-GCN rows used by the E2E runner:

```bash
python scripts/drkg/build_soft_support.py --source rgcn
python scripts/drkg/sweep_soft_support.py
```
