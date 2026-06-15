# PrimeKG Validation Eval Rows

This directory contains validation rows for:

- `backbone_raw`
- `ontology_raw`
- `soft_support_raw`
- `retrieval_main`

To rebuild the files, run:

```bash
python3 scripts/evaluation/build_valid_eval_ready.py
```

The input candidate files come from `dataset/setting_a/backbone_candidates`,
`dataset/setting_a/ontology_control`,
`dataset/setting_a/soft_support_ranked_candidates`, and
`dataset/setting_a/fuzzy_retrieval`.
