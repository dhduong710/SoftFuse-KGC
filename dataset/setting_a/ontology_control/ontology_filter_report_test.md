# Day 4 — Ontology Candidate Build

## 1. Goal
Build the ontology-only candidate artifact using direct valid task edges and valid mechanism paths.

## 2. Inputs
- input_candidates: `dataset/setting_a/ontology_control/test_top20_type_filtered_raw.json`
- input_evidence: `dataset/setting_a/aligned_evidence/test_aligned_evidence.json`
- type_map_tsv: `dataset/setting_b/annotations/type_map.tsv`
- schema_rules_json: `dataset/setting_b/annotations/schema_rules.json`
- path_templates_yaml: `dataset/setting_b/annotations/path_templates.yaml`

## 3. Output
- ontology_candidate_artifact: `dataset/setting_a/ontology_control/test_top20_ontology_raw.json`
- ontology_filter_report: `dataset/setting_a/ontology_control/ontology_filter_report_test.json`

## 4. Summary
- total_queries: 500
- total_candidates_before: 10000
- total_candidates_after: 4021
- removed_unsupported_candidates: 8259
- candidates_kept_by_direct_support: 1566
- candidates_kept_by_mechanism_support: 175
- queries_with_any_direct_support: 299
- queries_with_any_mechanism_support: 128
- fallback_queries: 114
- empty_queries_before_fallback: 114
- gold_in_ontology_candidates: 6
- top1_changed_queries: 213
- top5_changed_queries: 359

## 5. Sample before/after
### Query: lung abscess (disease)
- before_top5: ['Cortisone acetate', 'Fusidic acid', 'Dexamethasone', 'Norfloxacin', 'Betamethasone']
- after_top5: ['Benzylpenicillin']
- support_labels_top5_after: ['direct']
- fallback_used: False

### Query: acquired hyperprolactinemia
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: allergic rhinitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- after_top5: ['Dexamethasone', 'Triamcinolone', 'Prednisolone', 'Methylprednisolone', 'Hydrocortisone acetate']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: blepharoconjunctivitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Norfloxacin']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Prednisolone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: parkinsonian-pyramidal syndrome
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: pharyngitis
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Tetracycline', 'Methdilazine', 'Demeclocycline']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: diabetes mellitus (disease)
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Dexamethasone']
- support_labels_top5_after: ['mechanism']
- fallback_used: False

### Query: Trichinella spiralis infectious disease
- before_top5: ['Cortisone acetate', 'Fusidic acid', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Hydrocortisone', 'Betamethasone', 'Hydrocortisone acetate', 'Prednisone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: type 2 diabetes mellitus
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Dexamethasone']
- support_labels_top5_after: ['mechanism']
- fallback_used: False

### Query: spondyloarthropathy
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Triamcinolone', 'Betamethasone', 'Prednisolone', 'Prednisone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: atopic conjunctivitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone', 'Prednisolone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: squamous cell carcinoma of liver and intrahepatic biliary tract
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Dexamethasone', 'Doxorubicin']
- support_labels_top5_after: ['mechanism', 'mechanism']
- fallback_used: False

### Query: rat-bite fever
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Fusidic acid', 'Betamethasone', 'Hydrocortisone']
- after_top5: ['Benzylpenicillin']
- support_labels_top5_after: ['direct']
- fallback_used: False

### Query: tinea unguium
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: acute myeloid leukemia with NPM1 somatic mutations
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Doxorubicin']
- support_labels_top5_after: ['direct']
- fallback_used: False

### Query: common cold
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Methdilazine']
- support_labels_top5_after: ['direct']
- fallback_used: False

### Query: intrinsic asthma
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Betamethasone', 'Hydrocortisone acetate']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: classic Hodgkin lymphoma
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Triamcinolone', 'Betamethasone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: parkinsonian disorder
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Carmustine', 'Bleomycin', 'Vinblastine']
- support_labels_top5_after: ['mechanism', 'mechanism', 'mechanism']
- fallback_used: False

### Query: hypertensive disorder
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Dexamethasone']
- support_labels_top5_after: ['mechanism']
- fallback_used: False

### Query: trichinosis
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Methylprednisolone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: peptic esophagitis
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Betamethasone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: seborrheic dermatitis
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- after_top5: ['Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone', 'Fusidic acid']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: rheumatoid arthritis
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: amyloidosis (disease)
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Vinblastine']
- support_labels_top5_after: ['mechanism']
- fallback_used: False

### Query: neurotrophic keratopathy
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Norfloxacin']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Norfloxacin']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: acute gonococcal endometritis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- after_top5: ['Fusidic acid', 'Tetracycline', 'Ofloxacin', 'Demeclocycline']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: cystic fibrosis
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Propranolol']
- support_labels_top5_after: ['mechanism']
- fallback_used: False

### Query: chronic hepatitis C virus infection
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: acute gonococcal cervicitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- after_top5: ['Fusidic acid', 'Tetracycline', 'Ofloxacin', 'Ciprofloxacin']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: Addison disease
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone', 'Methylprednisolone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: Mycoplasma pneumoniae pneumonia
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- after_top5: ['Fusidic acid', 'Ofloxacin']
- support_labels_top5_after: ['direct', 'direct']
- fallback_used: False

### Query: neoplasm of mature B-cells
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Methylprednisolone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: atopic eczema
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Triamcinolone', 'Methylprednisolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Triamcinolone', 'Betamethasone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: bipolar disorder
- before_top5: ['Dexamethasone', 'Cortisone acetate', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Propranolol']
- support_labels_top5_after: ['mechanism']
- fallback_used: False

### Query: malignant Sertoli-Leydig cell tumor of ovary
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Doxorubicin', 'Thiotepa']
- support_labels_top5_after: ['direct', 'direct']
- fallback_used: False

### Query: esophagitis (disease)
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Betamethasone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: vasomotor rhinitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: early-onset parkinsonism-intellectual disability syndrome
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: ankylosing spondylitis
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: common cold
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Methdilazine']
- support_labels_top5_after: ['direct']
- fallback_used: False

### Query: acute lymphoblastic leukemia (disease)
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone', 'Prednisolone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: staphylococcus aureus infection
- before_top5: ['Cortisone acetate', 'Fusidic acid', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Fusidic acid', 'Ofloxacin', 'Ciprofloxacin', 'Demeclocycline', 'Benzylpenicillin']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: Norwegian scabies
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: seborrheic keratosis
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: ovarian mucinous adenocarcinoma
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Doxorubicin', 'Thiotepa']
- support_labels_top5_after: ['direct', 'direct']
- fallback_used: False

### Query: postmenopausal atrophic vaginitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- support_labels_top5_after: ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported']
- fallback_used: True

### Query: endometriosis of uterus
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Vinblastine']
- support_labels_top5_after: ['mechanism']
- fallback_used: False

### Query: acquired angioedema
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Triamcinolone', 'Methylprednisolone']
- support_labels_top5_after: ['direct', 'direct', 'direct', 'direct', 'direct']
- fallback_used: False

### Query: Mycoplasma pneumoniae pneumonia
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- after_top5: ['Fusidic acid', 'Ofloxacin']
- support_labels_top5_after: ['direct', 'direct']
- fallback_used: False
