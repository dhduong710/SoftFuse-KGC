# PrimeKG Soft-Support Ranked Candidates

This directory stores PrimeKG candidate rankings after the soft-support
reranker:

- `valid_top20_soft_support_main.json`
- `test_top20_soft_support_main.json`
- validation-only variant files used to select the main soft-support row

The main scripts in `scripts/soft_support` operate on validation candidates and
do not require a local soft-support train candidate file. For supervised E2E
training, use the packaged train split under
`dataset/setting_a/e2e_infer_ready/soft_support_raw/train.json`.
