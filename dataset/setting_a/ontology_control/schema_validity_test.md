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
- total_evidence_triples_checked: 29959
- valid_evidence_triples: 19155
- invalid_evidence_triples: 10804
- missing_relation_rules: 0
- incomplete_relation_rules: 0

### Top invalid evidence patterns
- Disease -[indication]-> Drug: 6606
- Disease -[associated_with]-> Protein_or_Gene: 3461
- Protein_or_Gene -[target]-> Drug: 737

## 4. Path validity summary
- valid_mechanism_templates_loaded: 2
- invalid_explanation_relations_loaded: 0
- total_path_sequences_checked: 28099
- valid_mechanism_path_sequences: 185
- direct_task_edge_sequences: 2126
- blocked_explanation_sequences: 0
- unsupported_path_sequences: 25788
- queries_with_no_candidate_to_query_path: 14

### Top unsupported path patterns
- indication -> indication -> indication: 14870
- indication -> associated_with -> associated_with: 10846
- target -> target -> indication: 72

### Sample unsupported paths
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Cortisone acetate | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Fusidic acid | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Betamethasone | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Betamethasone | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Betamethasone | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Betamethasone | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Betamethasone | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Betamethasone | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Betamethasone | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Betamethasone | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Betamethasone | path=indication -> indication -> indication
- query=lung abscess (disease) | candidate=Betamethasone | path=indication -> indication -> indication

## 5. Interpretation
- Direct indication edge is counted separately from mechanism paths.
- Contraindication-style relations are blocked as treatment explanation evidence.
- Unsupported path sequences are not necessarily schema-invalid triples; they are template-unapproved explanation paths.
