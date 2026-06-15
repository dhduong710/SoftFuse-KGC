# Day 3 — Schema Validity

## 1. Goal
Check schema validity for candidate / path / evidence on the valid split.

## 2. Candidate validity summary
- total_queries_checked: 500
- total_candidates_checked: 10000
- candidate_type_violations: 0
- unknown_candidate_types: 0

## 3. Evidence triple validity summary
- total_queries_checked: 500
- total_evidence_triples_checked: 30230
- valid_evidence_triples: 19372
- invalid_evidence_triples: 10858
- missing_relation_rules: 0
- incomplete_relation_rules: 0

### Top invalid evidence patterns
- Disease -[indication]-> Drug: 6365
- Disease -[associated_with]-> Protein_or_Gene: 3703
- Protein_or_Gene -[target]-> Drug: 790

## 4. Path validity summary
- valid_mechanism_templates_loaded: 2
- invalid_explanation_relations_loaded: 0
- total_path_sequences_checked: 28484
- valid_mechanism_path_sequences: 139
- direct_task_edge_sequences: 1870
- blocked_explanation_sequences: 0
- unsupported_path_sequences: 26475
- queries_with_no_candidate_to_query_path: 23

### Top unsupported path patterns
- indication -> indication -> indication: 15106
- indication -> associated_with -> associated_with: 11311
- target -> target -> indication: 58

### Sample unsupported paths
- query=leukemia, lymphocytic, susceptibility to | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Hydrocortisone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Dexamethasone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Dexamethasone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Dexamethasone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Dexamethasone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Dexamethasone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Dexamethasone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Dexamethasone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Dexamethasone | path=indication -> indication -> indication
- query=streptococcal infection | candidate=Dexamethasone | path=indication -> indication -> indication

## 5. Interpretation
- Direct indication edge is counted separately from mechanism paths.
- Contraindication-style relations are blocked as treatment explanation evidence.
- Unsupported path sequences are not necessarily schema-invalid triples; they are template-unapproved explanation paths.
