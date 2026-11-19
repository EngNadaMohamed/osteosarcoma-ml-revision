# Run from the project root (open Revision_Reviewer_final.Rproj in RStudio first).

library(edgeR)

# Project data directory
base_path <- "data/revision_folds_grouped"

# Raw counts file
raw_file <- "data/raw_counts_data.csv"

# edgeR output directory
output_path <- "data/Edger_result_grouped_6"

# Create output folder if it does not exist
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

raw <- read.csv(raw_file, row.names = 1, check.names = FALSE)

for (fold_number in 1:5) {
  cat("\n=== Fold", fold_number, "===\n")
  fold_path <- file.path(base_path, paste0("fold_", fold_number))
  
  tr_cpm <- read.csv(file.path(fold_path, "training_counts.csv"),
                     row.names = 1, check.names = FALSE)
  meta   <- read.csv(file.path(fold_path, "training_meta.csv"),
                     row.names = 1, check.names = FALSE)
  val_ids <- colnames(read.csv(file.path(fold_path, "validation_counts.csv"),
                               row.names = 1, check.names = FALSE, nrows = 1))
  
  train_ids <- colnames(tr_cpm)
  stopifnot(length(intersect(train_ids, val_ids)) == 0)  
  stopifnot(all(train_ids %in% rownames(meta)))
  
  counts <- raw[rownames(tr_cpm), train_ids]            
  meta   <- meta[train_ids, , drop = FALSE]
  g      <- factor(meta$source_name)
  cat("Train samples:", ncol(counts), "| Validation:", length(val_ids), "\n")
  print(table(g))
  
  # Create DGEList object
  dge    <- DGEList(counts = counts, group = g)
  dge    <- calcNormFactors(dge)
  
  # Create design matrix
  design <- model.matrix(~ g)
  
  dge    <- estimateDisp(dge, design)
  fit    <- glmQLFit(dge, design)
  
  res    <- glmQLFTest(fit, coef = 2)                      # Osteosarcoma vs Normal
  tab    <- topTags(res, n = Inf)$table
  deg    <- tab[tab$FDR < 0.05 & abs(tab$logFC) > 6, ]
  cat("Significant DEGs:", nrow(deg), "\n")
  
  write.csv(tab, file.path(output_path, paste0("Fold_", fold_number, "_edgeR_results.csv")))
  write.csv(deg, file.path(output_path, paste0("Fold_", fold_number, "_DEGs.csv")))
  writeLines(train_ids, file.path(output_path, paste0("Fold_", fold_number, "_train_samples.txt")))
}

cat("\nALL 5 FOLDS DONE\n")