# repoDB E2E Soft-Support Packages

The E2E runner uses `rgcn_raw_display_control` for the backbone row and `rgcn_sweep_selected` for the soft-support row.

If any `train.json`, `valid.json`, or `test.json` file is absent, rebuild the packages with:

```bash
python scripts/repodb/build_soft_support.py
python scripts/repodb/build_raw_display_control.py
```
