# Week 23 E2E PharmKG Day 1 Prepare Report

## Decision

PHARMKG_E2E_READY_PACKAGE_BUILT

## Embedding

- Path: `dataset/setting_c_pharmkg/e2e_infer_ready/entity_embeddings_rgcn.pt`
- Shape: `[7247, 128]`
- Required E2E argument: `--graph_num_rels 28`

## Ready rows

### backbone_raw

- Output dir: `dataset/setting_c_pharmkg/e2e_infer_ready/backbone_raw`
- train: 28960 rows
- valid: rows=500, Gold@20=0.07, Rank21=465, avg_subgraph_size=100, max_rel_id=27
- test: rows=500, Gold@20=0.092, Rank21=454, avg_subgraph_size=100, max_rel_id=27

### soft_support_raw

- Output dir: `dataset/setting_c_pharmkg/e2e_infer_ready/soft_support_raw`
- train: 28960 rows
- valid: rows=500, Gold@20=0.07, Rank21=465, avg_subgraph_size=100, max_rel_id=27
- test: rows=500, Gold@20=0.092, Rank21=454, avg_subgraph_size=100, max_rel_id=27

### fuzzy_retrieval_main

- Output dir: `dataset/setting_c_pharmkg/e2e_infer_ready/fuzzy_retrieval_main`
- train: 28960 rows
- valid: rows=500, Gold@20=0.07, Rank21=465, avg_subgraph_size=55, max_rel_id=27
- test: rows=500, Gold@20=0.092, Rank21=454, avg_subgraph_size=55, max_rel_id=27

## Day 2 command requirements

Use the exported embedding and graph relation count:

```bash
--kge_embedding_path dataset/setting_c_pharmkg/e2e_infer_ready/entity_embeddings_rgcn.pt \
--graph_num_rels 28
```
