# PrimeKG Test Eval Rows

This directory contains reviewer-safe test rows for:

- `backbone_raw`
- `ontology_raw`
- `soft_support_raw`
- `retrieval_main`

To rebuild the files, run these steps in order:

```bash
python3 scripts/soft_support/build_soft_support_test.py
python3 scripts/fuzzy_retrieval/build_fuzzy_retrieval_test.py
python3 scripts/evaluation/build_test_eval_ready.py
```

The final aggregation script is:

```bash
python3 scripts/evaluation/build_test_main_table.py
```
