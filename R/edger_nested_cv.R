# Run from the project root (open Revision_Reviewer_final.Rproj in RStudio first).

# ============================================================
# edger_nested_cv.R
# Nested Cross Validation - Inner Fold DEG Analysis
# ============================================================

library(edgeR)
library(reticulate)

# ------------------------------------------------------------
# 1. Define Paths
# ------------------------------------------------------------

# Project data directory
base_path <- "data/revision_folds_grouped"

# Raw counts file
raw_file <- "data/raw_counts_data.csv"

# edgeR output directory
output_path <- "data/Edger_nested_cv"

# Create output folder if it does not exist
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

# ------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------

N_OUTER_FOLDS <- 5
N_INNER_FOLDS <- 3

# ------------------------------------------------------------
# 3. Import Python Libraries
# ------------------------------------------------------------

sklearn_model_selection <- import(
  "sklearn.model_selection",
  convert = FALSE
)

builtins <- import_builtins()

# ------------------------------------------------------------
# 4. Read Raw Counts
# ------------------------------------------------------------

raw <- read.csv(
  raw_file,
  row.names = 1,
  check.names = FALSE
)

raw <- as.matrix(raw)
storage.mode(raw) <- "numeric"

cat("Raw Counts Dimensions:", dim(raw), "\n")

# ------------------------------------------------------------
# 5. Outer Folds
# ------------------------------------------------------------

for (outer_number in 1:N_OUTER_FOLDS) {
  
  cat("\n====================================\n")
  cat("OUTER FOLD:", outer_number, "\n")
  cat("====================================\n")
  
  fold_path <- file.path(
    base_path,
    paste0("fold_", outer_number)
  )
  
  # Read Outer Training Data
  tr_counts <- read.csv(
    file.path(fold_path, "training_counts.csv"),
    row.names = 1,
    check.names = FALSE
  )
  
  meta <- read.csv(
    file.path(fold_path, "training_meta.csv"),
    row.names = 1,
    check.names = FALSE
  )
  
  train_ids <- colnames(tr_counts)
  
  # Align metadata with training samples
  meta <- meta[train_ids, , drop = FALSE]
  
  # Check required metadata
  if (!"patient_id" %in% colnames(meta)) {
    stop("patient_id column is missing.")
  }
  
  if (!"source_name" %in% colnames(meta)) {
    stop("source_name column is missing.")
  }
  
  if (any(is.na(meta$patient_id))) {
    stop("Missing patient_id values.")
  }
  
  if (any(is.na(meta$source_name))) {
    stop("Missing source_name values.")
  }
  
  # Check sample IDs in raw counts
  if (!all(train_ids %in% colnames(raw))) {
    stop("Some training samples are missing from raw counts.")
  }
  
  # ----------------------------------------------------------
  # 6. Create Stratified Grouped Inner Folds
  # ----------------------------------------------------------
  
  set.seed(100 + outer_number)
  
  inner_cv <- sklearn_model_selection$StratifiedGroupKFold(
    n_splits = as.integer(N_INNER_FOLDS),
    shuffle = TRUE,
    random_state = as.integer(100 + outer_number)
  )
  
  X_dummy <- matrix(
    0,
    nrow = nrow(meta),
    ncol = 1
  )
  
  inner_splits <- inner_cv$split(
    X_dummy,
    as.character(meta$source_name),
    as.character(meta$patient_id)
  )
  
  # Convert Python generator to a list
  inner_splits <- builtins$list(inner_splits)
  
  if (length(inner_splits) != N_INNER_FOLDS) {
    stop("Unexpected number of inner folds.")
  }
  
  outer_output_path <- file.path(
    output_path,
    paste0("outer_fold_", outer_number)
  )
  
  dir.create(
    outer_output_path,
    recursive = TRUE,
    showWarnings = FALSE
  )
  
  ###############################################################
  
  # ----------------------------------------------------------
  # Run edgeR on the FULL OUTER TRAINING DATA
  # This is used for final model training after inner CV selection
  # ----------------------------------------------------------
  
  outer_counts <- raw[
    rownames(tr_counts),
    train_ids,
    drop = FALSE
  ]
  
  storage.mode(outer_counts) <- "integer"
  
  outer_group <- factor(
    meta$source_name,
    levels = c(
      "Normal_adjacent_tissue",
      "Osteosarcoma_sample"
    )
  )
  
  outer_dge <- DGEList(
    counts = outer_counts,
    group = outer_group
  )
  
  outer_dge <- calcNormFactors(outer_dge)
  
  outer_design <- model.matrix(~ outer_group)
  
  outer_dge <- estimateDisp(outer_dge, outer_design)
  
  outer_fit <- glmQLFit(outer_dge, outer_design)
  
  outer_res <- glmQLFTest(
    outer_fit,
    coef = 2
  )
  
  outer_tab <- topTags(
    outer_res,
    n = Inf
  )$table
  
  outer_deg <- outer_tab[
    outer_tab$FDR < 0.05 &
      abs(outer_tab$logFC) > 6,
    ,
    drop = FALSE
  ]
  
  cat("Outer Training DEGs:", nrow(outer_deg), "\n")
  
  write.csv(
    outer_tab,
    file.path(
      outer_output_path,
      "edgeR_outer_training_results.csv"
    )
  )
  
  write.csv(
    outer_deg,
    file.path(
      outer_output_path,
      "DEGs_outer_training.csv"
    )
  )
  
  ###############################################################
  
  
  # ----------------------------------------------------------
  # 7. Process Each Inner Fold
  # ----------------------------------------------------------
  
  for (inner_number in seq_len(N_INNER_FOLDS)) {
    
    cat("\n------------------------------------\n")
    cat("Outer Fold:", outer_number,
        "| Inner Fold:", inner_number, "\n")
    cat("------------------------------------\n")
    
    # Get Python split indices
    split_indices <- inner_splits[[inner_number]]
    
    inner_train_indices <- as.integer(
      py_to_r(split_indices[[1]])
    ) + 1L
    
    inner_validation_indices <- as.integer(
      py_to_r(split_indices[[2]])
    ) + 1L
    
    # Get sample IDs
    inner_train_ids <- train_ids[inner_train_indices]
    
    inner_validation_ids <- train_ids[
      inner_validation_indices
    ]
    
    # Prepare Inner Training Metadata
    inner_train_meta <- meta[
      inner_train_ids,
      ,
      drop = FALSE
    ]
    
    inner_validation_meta <- meta[
      inner_validation_ids,
      ,
      drop = FALSE
    ]
    
    # --------------------------------------------------------
    # 8. Check Class Distribution and Patient Separation
    # --------------------------------------------------------
    
    cat("Training class distribution:\n")
    print(table(inner_train_meta$source_name))
    
    cat("Validation class distribution:\n")
    print(table(inner_validation_meta$source_name))
    
    cat("Training patients:",
        length(unique(inner_train_meta$patient_id)), "\n")
    
    cat("Validation patients:",
        length(unique(inner_validation_meta$patient_id)), "\n")
    
    # Check patient overlap
    train_patients <- unique(inner_train_meta$patient_id)
    
    validation_patients <- unique(
      inner_validation_meta$patient_id
    )
    
    if (length(intersect(
      train_patients,
      validation_patients
    )) > 0) {
      stop("Patient overlap detected.")
    }
    
    # Check both classes in both splits
    if (length(unique(inner_train_meta$source_name)) != 2) {
      stop("Both classes are not present in Inner Training.")
    }
    
    if (length(unique(inner_validation_meta$source_name)) != 2) {
      stop("Both classes are not present in Inner Validation.")
    }
    
    # --------------------------------------------------------
    # 9. Extract Inner Training Raw Counts
    # --------------------------------------------------------
    
    if (!all(rownames(tr_counts) %in% rownames(raw))) {
      stop("Some genes are missing from raw counts.")
    }
    
    counts <- raw[
      rownames(tr_counts),
      inner_train_ids,
      drop = FALSE
    ]
    
    if (any(is.na(counts))) {
      stop("Missing values detected in raw counts.")
    }
    
    if (any(counts < 0) || any(counts != round(counts))) {
      stop("Raw counts must be non-negative integers.")
    }
    
    storage.mode(counts) <- "integer"
    
    # --------------------------------------------------------
    # 10. Run edgeR on Inner Training Only
    # --------------------------------------------------------
    
    g <- factor(
      inner_train_meta$source_name,
      levels = c(
        "Normal_adjacent_tissue",
        "Osteosarcoma_sample"
      )
    )
    
    cat("Training Samples:", ncol(counts), "\n")
    cat("Validation Samples:", length(inner_validation_ids), "\n")
    
    print(table(g))
    
    dge <- DGEList(
      counts = counts,
      group = g
    )
    
    dge <- calcNormFactors(dge)
    
    design <- model.matrix(~ g)
    
    dge <- estimateDisp(dge, design)
    
    fit <- glmQLFit(dge, design)
    
    res <- glmQLFTest(
      fit,
      coef = 2
    )
    
    tab <- topTags(
      res,
      n = Inf
    )$table
    
    # --------------------------------------------------------
    # 11. Extract Significant DEGs
    # --------------------------------------------------------
    
    deg <- tab[
      tab$FDR < 0.05 &
        abs(tab$logFC) > 6,
      ,
      drop = FALSE
    ]
    
    cat("Significant DEGs:", nrow(deg), "\n")
    
    # --------------------------------------------------------
    # 12. Save Results
    # --------------------------------------------------------
    
    inner_output_path <- file.path(
      outer_output_path,
      paste0("inner_fold_", inner_number)
    )
    
    dir.create(
      inner_output_path,
      recursive = TRUE,
      showWarnings = FALSE
    )
    
    write.csv(
      tab,
      file.path(
        inner_output_path,
        "edgeR_nested_cv_results.csv"
      )
    )
    
    write.csv(
      deg,
      file.path(
        inner_output_path,
        "DEGs_nested_cv.csv"
      )
    )
    
    writeLines(
      inner_train_ids,
      file.path(
        inner_output_path,
        "inner_train_samples.txt"
      )
    )
    
    writeLines(
      inner_validation_ids,
      file.path(
        inner_output_path,
        "inner_validation_samples.txt"
      )
    )
    
    write.csv(
      inner_train_meta,
      file.path(
        inner_output_path,
        "inner_train_meta.csv"
      )
    )
    
    write.csv(
      inner_validation_meta,
      file.path(
        inner_output_path,
        "inner_validation_meta.csv"
      )
    )
    
  }
}

cat("\n====================================\n")
cat("ALL NESTED CV EDGE R FOLDS COMPLETED\n")
cat("====================================\n")
