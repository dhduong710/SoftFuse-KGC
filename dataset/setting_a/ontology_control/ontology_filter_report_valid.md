# Day 4 — Ontology Candidate Build

## 1. Goal
Build the ontology-only candidate artifact using direct valid task edges and valid mechanism paths.

## 2. Inputs
- input_candidates: `dataset/setting_a/ontology_control/valid_top20_type_filtered_raw.json`
- input_evidence: `dataset/setting_a/aligned_evidence/valid_aligned_evidence.json`
- type_map_tsv: `dataset/setting_b/annotations/type_map.tsv`
- schema_rules_json: `dataset/setting_b/annotations/schema_rules.json`
- path_templates_yaml: `dataset/setting_b/annotations/path_templates.yaml`

## 3. Output
- ontology_candidate_artifact: `dataset/setting_a/ontology_control/valid_top20_ontology_raw.json`
- ontology_filter_report: `dataset/setting_a/ontology_control/ontology_filter_report_valid.json`

## 4. Summary
- total_queries: 500
- total_candidates_before: 10000
- total_candidates_after: 4398
- removed_unsupported_candidates: 8362
- candidates_kept_by_direct_support: 1454
- candidates_kept_by_mechanism_support: 184
- queries_with_any_direct_support: 269
- queries_with_any_mechanism_support: 130
- fallback_queries: 138
- empty_queries_before_fallback: 138
- gold_in_ontology_candidates: 7
- top1_changed_queries: 211
- top5_changed_queries: 334

## 5. Sample before/after
### Query: leukemia, lymphocytic, susceptibility to
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: streptococcal infection
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Ofloxacin', 'Ciprofloxacin']
- support_labels_top5_after: ['direct', 'direct', 'direct']
- fallback_used: False

### Query: dyspepsia
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: basal cell carcinoma
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Betamethasone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Betamethasone', 'Hydrocortisone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: obsessive-compulsive disorder
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Propranolol']
- support_labels_top5_after: ['mechanism']
- fallback_used: False

### Query: Norwegian scabies
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: hypertension
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Hydrocortisone', 'Norfloxacin', 'Dexamethasone']
- after_top5: ['Dexamethasone']
- support_labels_top5_after: ['mechanism']
- fallback_used: False

### Query: pneumococcal meningitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Norfloxacin']
- after_top5: ['Ampicillin']
- support_labels_top5_after: ['direct']
- fallback_used: False

### Query: obsessive-compulsive disorder
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Propranolol']
- support_labels_top5_after: ['mechanism']
- fallback_used: False

### Query: arthropathy
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Fusidic acid']
- support_labels_top5_after: ['direct']
- fallback_used: False

### Query: diffuse large B-cell lymphoma
- before_top5: ['Dexamethasone', 'Cortisone acetate', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- after_top5: ['Dexamethasone', 'Prednisone', 'Carmustine', 'Bleomycin', 'Methotrexate']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'mechanism']
- fallback_used: False

### Query: obsolete Hodgkin's granuloma
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Hydrocortisone acetate']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: migraine with or without aura, susceptibility to
- before_top5: ['Cortisone acetate', 'Fusidic acid', 'Hydrocortisone', 'Dexamethasone', 'Norfloxacin']
- after_top5: ['Cortisone acetate', 'Fusidic acid', 'Hydrocortisone', 'Dexamethasone', 'Norfloxacin']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: infectious anterior uveitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Hydrocortisone']
- support_labels_top5_after: ['direct', 'direct']
- fallback_used: False

### Query: chromomycosis
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Dexamethasone', 'Betamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Dexamethasone', 'Betamethasone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: proctitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- after_top5: ['Hydrocortisone']
- support_labels_top5_after: ['direct']
- fallback_used: False

### Query: nasopharyngitis
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: acquired thrombocytopenia
- before_top5: ['Cortisone acetate', 'Fusidic acid', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Prednisolone', 'Methylprednisolone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: hemoglobinopathy
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Hydroxyurea', 'Dexamethasone', 'Hydrocortisone acetate']
- support_labels_top5_after: ['direct', 'mechanism', 'mechanism']
- fallback_used: False

### Query: prostate cancer
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Hydrocortisone acetate', 'Doxorubicin']
- support_labels_top5_after: ['mechanism', 'mechanism', 'mechanism', 'mechanism', 'mechanism']
- fallback_used: False

### Query: chronic cutaneous lupus erythematosus
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Betamethasone', 'Triamcinolone', 'Methylprednisolone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: pulmonary emphysema
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: common cold
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Methdilazine']
- support_labels_top5_after: ['direct']
- fallback_used: False

### Query: classic Hodgkin lymphoma
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Triamcinolone', 'Betamethasone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: mental disorder
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Propranolol']
- support_labels_top5_after: ['mechanism']
- fallback_used: False

### Query: aspiration pneumonia (disease)
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Betamethasone']
- after_top5: ['Hydrocortisone', 'Betamethasone', 'Triamcinolone', 'Methylprednisolone', 'Prednisolone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: autoimmune thrombocytopenic
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Triamcinolone', 'Betamethasone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: respiratory tract infectious disease
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: arteriosclerosis disorder
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Dexamethasone']
- support_labels_top5_after: ['mechanism']
- fallback_used: False

### Query: gonococcal epididymo-orchitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Dexamethasone', 'Norfloxacin', 'Hydrocortisone']
- after_top5: ['Fusidic acid', 'Tetracycline']
- support_labels_top5_after: ['direct', 'direct']
- fallback_used: False

### Query: ovarian mucinous adenocarcinoma
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Doxorubicin', 'Thiotepa']
- support_labels_top5_after: ['direct', 'direct']
- fallback_used: False

### Query: visual epilepsy
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Prednisone']
- after_top5: ['Dexamethasone', 'Propranolol', 'Vinblastine']
- support_labels_top5_after: ['mechanism', 'mechanism', 'mechanism']
- fallback_used: False

### Query: scleroderma (disease)
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- after_top5: ['Propranolol']
- support_labels_top5_after: ['mechanism']
- fallback_used: False

### Query: common cold
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Methdilazine']
- support_labels_top5_after: ['direct']
- fallback_used: False

### Query: hemiparkinsonism-hemiatrophy syndrome
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: allergic asthma
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Methylprednisolone', 'Hydrocortisone', 'Betamethasone', 'Prednisolone', 'Hydrocortisone acetate']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: actinic keratosis (disease)
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Betamethasone', 'Triamcinolone', 'Prednisolone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: mental disorder
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Propranolol']
- support_labels_top5_after: ['mechanism']
- fallback_used: False

### Query: dysentery
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Ofloxacin', 'Ciprofloxacin']
- support_labels_top5_after: ['direct', 'direct']
- fallback_used: False

### Query: acute myeloid leukemia with minimal differentiation
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Doxorubicin']
- support_labels_top5_after: ['direct']
- fallback_used: False

### Query: pharyngitis
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Tetracycline', 'Methdilazine', 'Demeclocycline']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: adenocarcinoma of liver and intrahepatic biliary tract
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Dexamethasone', 'Doxorubicin']
- support_labels_top5_after: ['mechanism', 'mechanism']
- fallback_used: False

### Query: angioedema
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Triamcinolone', 'Betamethasone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: cerebral infarction
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: granulomatous slack skin disease
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone', 'Betamethasone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: nasopharyngitis
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: urethritis (disease)
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- after_top5: ['Fusidic acid', 'Ofloxacin']
- support_labels_top5_after: ['direct', 'direct']
- fallback_used: False

### Query: hereditary angioedema with C1Inh deficiency
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: brain edema
- before_top5: ['Cortisone acetate', 'Fusidic acid', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Prednisolone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: gastroesophageal reflux disease
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Betamethasone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True
