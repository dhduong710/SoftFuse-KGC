# PrimeKG Fuzzy Retrieval

This directory stores PrimeKG confidence-aware fuzzy retrieval artifacts:

- `valid_path_features.json`
- `valid_fuzzy_retrieval_main.json`
- `test_path_features.json`
- `test_fuzzy_retrieval_main.json`

The selection scripts in `scripts/fuzzy_retrieval` use validation artifacts.
For supervised E2E training, use the packaged train split under
`dataset/setting_a/e2e_infer_ready/retrieval_main/train.json`.
