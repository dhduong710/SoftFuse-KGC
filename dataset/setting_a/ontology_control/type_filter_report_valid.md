# Day 2 — Type Filtering

## 1. Goal
Filter candidates so that every remaining candidate is of type `Drug`.

## 2. Inputs
- input_candidates: `dataset/setting_a/backbone_candidates/valid_top20_raw.json`
- type_map_tsv: `dataset/setting_b/annotations/type_map.tsv`
- fallback_type_json: `dataset/setting_b/annotations/type_map.tsv`

## 3. Outputs
- filtered_candidates: `dataset/setting_a/ontology_control/valid_top20_type_filtered_raw.json`
- filter_report_json: `dataset/setting_a/ontology_control/type_filter_report_valid.json`

## 4. Summary
- total_queries: 500
- total_candidates_before: 10000
- total_candidates_after: 10000
- total_removed: 0
- empty_queries_after_filter: 0
- remaining_non_drug_candidates: 0
- top1_changed_queries: 0
- top5_changed_queries: 0

## 5. Removal breakdown by type

## 6. Type lookup source breakdown
- type_map.tsv: 10000

## 7. Empty-query examples
- none

## 8. Before/after samples
### Query: leukemia, lymphocytic, susceptibility to
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- removed_types: {}

### Query: streptococcal infection
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: dyspepsia
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: basal cell carcinoma
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Betamethasone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Betamethasone', 'Hydrocortisone']
- removed_types: {}

### Query: obsessive-compulsive disorder
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- removed_types: {}

### Query: Norwegian scabies
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: hypertension
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Hydrocortisone', 'Norfloxacin', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Hydrocortisone', 'Norfloxacin', 'Dexamethasone']
- removed_types: {}

### Query: pneumococcal meningitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Norfloxacin']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Norfloxacin']
- removed_types: {}

### Query: obsessive-compulsive disorder
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- removed_types: {}

### Query: arthropathy
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- removed_types: {}

### Query: diffuse large B-cell lymphoma
- before_top5: ['Dexamethasone', 'Cortisone acetate', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- after_top5: ['Dexamethasone', 'Cortisone acetate', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- removed_types: {}

### Query: obsolete Hodgkin's granuloma
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- removed_types: {}

### Query: migraine with or without aura, susceptibility to
- before_top5: ['Cortisone acetate', 'Fusidic acid', 'Hydrocortisone', 'Dexamethasone', 'Norfloxacin']
- after_top5: ['Cortisone acetate', 'Fusidic acid', 'Hydrocortisone', 'Dexamethasone', 'Norfloxacin']
- removed_types: {}

### Query: infectious anterior uveitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- removed_types: {}

### Query: chromomycosis
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Dexamethasone', 'Betamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Dexamethasone', 'Betamethasone']
- removed_types: {}

### Query: proctitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- removed_types: {}

### Query: nasopharyngitis
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: acquired thrombocytopenia
- before_top5: ['Cortisone acetate', 'Fusidic acid', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Fusidic acid', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone']
- removed_types: {}

### Query: hemoglobinopathy
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- removed_types: {}

### Query: prostate cancer
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- removed_types: {}

### Query: chronic cutaneous lupus erythematosus
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- removed_types: {}

### Query: pulmonary emphysema
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- removed_types: {}

### Query: common cold
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: classic Hodgkin lymphoma
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- removed_types: {}

### Query: mental disorder
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- removed_types: {}

### Query: aspiration pneumonia (disease)
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Betamethasone']
- removed_types: {}

### Query: autoimmune thrombocytopenic
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Triamcinolone']
- removed_types: {}

### Query: respiratory tract infectious disease
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: arteriosclerosis disorder
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- removed_types: {}

### Query: gonococcal epididymo-orchitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Dexamethasone', 'Norfloxacin', 'Hydrocortisone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Dexamethasone', 'Norfloxacin', 'Hydrocortisone']
- removed_types: {}

### Query: ovarian mucinous adenocarcinoma
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- removed_types: {}

### Query: visual epilepsy
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Prednisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Prednisone']
- removed_types: {}

### Query: scleroderma (disease)
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- removed_types: {}

### Query: common cold
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: hemiparkinsonism-hemiatrophy syndrome
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- removed_types: {}

### Query: allergic asthma
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- removed_types: {}

### Query: actinic keratosis (disease)
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- removed_types: {}

### Query: mental disorder
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- removed_types: {}

### Query: dysentery
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: acute myeloid leukemia with minimal differentiation
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- removed_types: {}

### Query: pharyngitis
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: adenocarcinoma of liver and intrahepatic biliary tract
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- removed_types: {}

### Query: angioedema
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Triamcinolone', 'Betamethasone']
- removed_types: {}

### Query: cerebral infarction
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- removed_types: {}

### Query: granulomatous slack skin disease
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- removed_types: {}

### Query: nasopharyngitis
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: urethritis (disease)
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- removed_types: {}

### Query: hereditary angioedema with C1Inh deficiency
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- removed_types: {}

### Query: brain edema
- before_top5: ['Cortisone acetate', 'Fusidic acid', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Fusidic acid', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone']
- removed_types: {}

### Query: gastroesophageal reflux disease
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Betamethasone']
- removed_types: {}
