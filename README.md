# Tumour vs adjacent-normal classification in osteosarcoma: revision code

Analysis code for the revised manuscript. It covers patient-grouped cross-validation
with within-fold differential expression, nested cross-validation, label-permutation
testing, threshold/feature-number sensitivity, confidence intervals, simple baselines,
calibration and gene-selection stability, tissue-composition (skeletal-muscle)
analyses, external validation and clinical-covariate checks.

## Data

The data are public and are not included in this repository. Download them from GEO
and place them in a `data/` folder at the project root:

| Dataset | Files expected in `data/` |
|---|---|
| GSE218035 (training; 23 osteosarcoma, 13 adjacent normal) | `raw_counts_data.csv` (raw counts, genes × samples), `meta_revision_with_patient_id.csv` (columns `source_name`, `patient_id`), `metadata.csv` (SRA run table with `AGE`, `sex`, `Site`) |
| GSE99671 (external validation; 18 tumour, 18 paired normal bone) | `GSE99671/GSE99671_Osteosarcoma_filecounts.txt`, `GSE99671/GSE99671_series_matrix.txt` |
| GSE253548 (external validation; FFPE, samples annotated as `tissue: bone`) | `GSE253548/GSE253548_NormCountsSalmonFFPEOS_12012024.xlsx` and the GEO series matrix |

Two CPM matrices derived from the GSE218035 counts are also expected in `data/`:
`filtered_counts_before_edgeR_revision.csv` (genes × samples, used to build the folds)
and `combined_output.csv` (CPM with gene annotation columns `#ID`, `Gene_type`, `Symbol`,
used by the nested cross-validation).

## How to run

Open `Revision_Reviewer_final.Rproj` in RStudio before running the R scripts, so that
paths resolve from the project root. Python scripts locate `data/` relative to their
own location and can be run from any folder.

Run the steps in this order:

1. **Patient-grouped folds:** `python/outer/create_grouped_folds_clean.py`
   (StratifiedGroupKFold, 5 folds, grouped by patient).
2. **Within-fold edgeR:** `R/edger_result.R`
   (glmQLFTest on the training samples of each fold only).
3. **Main analysis (31 configurations):** `python/revision_fold_grouped/load_data_full_test_clean_id.py`
4. **Nested cross-validation:** `R/edger_nested_cv.R`, then
   `python/nested_cv/load_data_full_test_nested.py`
5. **Label-permutation test:** `R/permutation_test_clean.R`, then
   `python/revision_fold_grouped/permutation_test_clean.py`.
   Set the same `PERM_MODE` in both scripts. `"within_patient"` (main analysis, 430 permutations,
   seed 123) swaps the tumour/normal labels within each paired patient; `"sample"` shuffles all
   36 labels freely (sensitivity analysis, 100 permutations). Outputs go to
   `data/perm_out_grouped_<PERM_MODE>/`.
6. **Sensitivity to |log2FC| threshold and k:** `python/revision_fold_grouped/sensitivity_analysis.py`
7. **Confidence intervals, baselines, calibration and gene stability:**
   `python/revision_fold_grouped/extra_analyses.py`
   (patient-level bootstrap 95% CIs and pooled confusion matrices for the 31 configurations,
   majority-class and single-gene baselines, Brier score and calibration curves,
   and how often each gene is selected across the 5 folds)
8. **Tissue composition, clinical covariates and fixed-gene-set external analysis** (`python/confound_analysis/`):
   - `muscle_experiment.py`: muscle score alone, muscle-correlated DEGs removed, muscle-adjusted features
   - `up_only.py`: tumour-up-regulated genes only
   - `within_ext_cv.py`: patient-grouped CV within GSE99671 (reference)
   - `age_confound.py`: age, sex and site comparisons
   - `external_validation.py`: supplementary analysis with a fixed gene set
9. **Locked-pipeline external validation** (GSE99671 and GSE253548):
   - `R/map_ensembl_GSE253548.R`: maps the Ensembl IDs of GSE253548 to gene symbols
     (writes `data/GSE253548/ensembl_to_symbol.csv`); run once before the next script
   - `python/external_validation/locked_pipeline_external.py`: re-fits the 31 configurations on
     GSE218035 using the genes available in each external cohort and applies them unchanged to the
     external samples (threshold 0.5; external labels are used only for evaluation).
     GSE253548 samples are kept if they have at least 1,000 genes with at least 10 counts.
     Outputs go to `data/external_locked/`.

10. **Descriptive figures (all samples):** `R/figures/descriptive_figures.R`
   (volcano plot, DEG heatmap, PCA, KEGG/Reactome/GO enrichment, GSEA Hallmark).
   All enrichment analyses use Benjamini-Hochberg adjusted p < 0.05, with all genes
   tested by edgeR as the background; redundant GO terms are merged with `simplify()`.
   These use all 36 samples and are descriptive only; they are not used for model training.
11. **Permutation-test figure:** `python/figures/make_fig.py` (run after step 5; reads
   `data/perm_out_grouped_within_patient/`).

Results are written to sub-folders of `data/`.

## Requirements

Exact package versions used for the reported results are listed in `requirements.txt` (Python)
and `sessionInfo.txt` (R).

- **Python** (`pip install -r requirements.txt`): `pandas`, `numpy`, `scikit-learn`, `scipy`,
  `matplotlib`, `openpyxl` (to read the GSE253548 `.xlsx` file)
- **R 4.6.2 / Bioconductor:** `edgeR`, `limma`, `reticulate` (for `edger_nested_cv.R`), `readxl`,
  `org.Hs.eg.db`, `AnnotationDbi`; for the figures also `ggplot2`, `pheatmap`, `clusterProfiler`,
  `ReactomePA`, `msigdbr`, `enrichplot`, `GOSemSim`, `dplyr`, `tibble`
