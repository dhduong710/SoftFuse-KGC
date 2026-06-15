# Day 2 — Type Filtering

## 1. Goal
Filter candidates so that every remaining candidate is of type `Drug`.

## 2. Inputs
- input_candidates: `dataset/setting_a/backbone_candidates/test_top20_raw.json`
- type_map_tsv: `dataset/setting_b/annotations/type_map.tsv`
- fallback_type_json: `dataset/setting_b/annotations/type_map.tsv`

## 3. Outputs
- filtered_candidates: `dataset/setting_a/ontology_control/test_top20_type_filtered_raw.json`
- filter_report_json: `dataset/setting_a/ontology_control/type_filter_report_test.json`

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
### Query: lung abscess (disease)
- before_top5: ['Cortisone acetate', 'Fusidic acid', 'Dexamethasone', 'Norfloxacin', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Fusidic acid', 'Dexamethasone', 'Norfloxacin', 'Betamethasone']
- removed_types: {}

### Query: acquired hyperprolactinemia
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- removed_types: {}

### Query: allergic rhinitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- removed_types: {}

### Query: blepharoconjunctivitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Norfloxacin']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Norfloxacin']
- removed_types: {}

### Query: parkinsonian-pyramidal syndrome
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- removed_types: {}

### Query: pharyngitis
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: diabetes mellitus (disease)
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- removed_types: {}

### Query: Trichinella spiralis infectious disease
- before_top5: ['Cortisone acetate', 'Fusidic acid', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Fusidic acid', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone']
- removed_types: {}

### Query: type 2 diabetes mellitus
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- removed_types: {}

### Query: spondyloarthropathy
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- removed_types: {}

### Query: atopic conjunctivitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: squamous cell carcinoma of liver and intrahepatic biliary tract
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- removed_types: {}

### Query: rat-bite fever
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Fusidic acid', 'Betamethasone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Fusidic acid', 'Betamethasone', 'Hydrocortisone']
- removed_types: {}

### Query: tinea unguium
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: acute myeloid leukemia with NPM1 somatic mutations
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- removed_types: {}

### Query: common cold
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: intrinsic asthma
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- removed_types: {}

### Query: classic Hodgkin lymphoma
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Triamcinolone']
- removed_types: {}

### Query: parkinsonian disorder
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- removed_types: {}

### Query: hypertensive disorder
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: trichinosis
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- removed_types: {}

### Query: peptic esophagitis
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Betamethasone']
- removed_types: {}

### Query: seborrheic dermatitis
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- removed_types: {}

### Query: rheumatoid arthritis
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- removed_types: {}

### Query: amyloidosis (disease)
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- removed_types: {}

### Query: neurotrophic keratopathy
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Norfloxacin']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Norfloxacin']
- removed_types: {}

### Query: acute gonococcal endometritis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- removed_types: {}

### Query: cystic fibrosis
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- removed_types: {}

### Query: chronic hepatitis C virus infection
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: acute gonococcal cervicitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- removed_types: {}

### Query: Addison disease
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- removed_types: {}

### Query: Mycoplasma pneumoniae pneumonia
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- removed_types: {}

### Query: neoplasm of mature B-cells
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Methylprednisolone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Methylprednisolone', 'Triamcinolone']
- removed_types: {}

### Query: atopic eczema
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Triamcinolone', 'Methylprednisolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Triamcinolone', 'Methylprednisolone']
- removed_types: {}

### Query: bipolar disorder
- before_top5: ['Dexamethasone', 'Cortisone acetate', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Dexamethasone', 'Cortisone acetate', 'Methylprednisolone', 'Triamcinolone', 'Betamethasone']
- removed_types: {}

### Query: malignant Sertoli-Leydig cell tumor of ovary
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- removed_types: {}

### Query: esophagitis (disease)
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Betamethasone']
- removed_types: {}

### Query: vasomotor rhinitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- removed_types: {}

### Query: early-onset parkinsonism-intellectual disability syndrome
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- removed_types: {}

### Query: ankylosing spondylitis
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone', 'Triamcinolone']
- removed_types: {}

### Query: common cold
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: acute lymphoblastic leukemia (disease)
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- removed_types: {}

### Query: staphylococcus aureus infection
- before_top5: ['Cortisone acetate', 'Fusidic acid', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Fusidic acid', 'Dexamethasone', 'Hydrocortisone', 'Betamethasone']
- removed_types: {}

### Query: Norwegian scabies
- before_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Norfloxacin', 'Cortisone acetate', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: seborrheic keratosis
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Fusidic acid', 'Betamethasone']
- removed_types: {}

### Query: ovarian mucinous adenocarcinoma
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Hydrocortisone', 'Betamethasone']
- removed_types: {}

### Query: postmenopausal atrophic vaginitis
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Hydrocortisone', 'Dexamethasone']
- removed_types: {}

### Query: endometriosis of uterus
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Methylprednisolone', 'Triamcinolone', 'Hydrocortisone']
- removed_types: {}

### Query: acquired angioedema
- before_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Triamcinolone', 'Betamethasone']
- after_top5: ['Cortisone acetate', 'Dexamethasone', 'Hydrocortisone', 'Triamcinolone', 'Betamethasone']
- removed_types: {}

### Query: Mycoplasma pneumoniae pneumonia
- before_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- after_top5: ['Fusidic acid', 'Cortisone acetate', 'Norfloxacin', 'Dexamethasone', 'Hydrocortisone']
- removed_types: {}
