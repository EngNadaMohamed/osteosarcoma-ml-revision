# Run from the project root (open Revision_Reviewer_final.Rproj in RStudio first).
# ============================================================
# Permutation test (R part): for every permutation, the class labels are
# permuted and edgeR is re-run on the training samples of each fold.
# The selected genes and permuted labels are saved; the 31 configurations
# are then evaluated in python/revision_fold_grouped/permutation_test_clean.py
#
# PERM_MODE
#   "within_patient": respects the patient structure. For the patients with a
#                     paired tumour and normal sample, the two labels are
#                     swapped at random (probability 0.5). Patients with a
#                     single sample keep their label. Class sizes stay
#                     23 / 13, and every patient keeps one tumour and one
#                     normal sample. This is the main analysis.
#   "sample":         labels shuffled freely across all 36 samples
#                     (the original analysis, kept as a sensitivity check).
# ============================================================
library(edgeR)

base     <- "data/revision_folds_grouped"
raw_file <- "data/raw_counts_data.csv"
meta_file <- "data/meta_revision_with_patient_id.csv"

PERM_MODE <- "within_patient"     # "within_patient" (main) or "sample"
N_PERM    <- 430                  # permutations reported in the manuscript (seed 123)
N_TOP     <- 80
LOGFC_CUT <- 6
set.seed(123)

out <- file.path("data", paste0("perm_out_grouped_", PERM_MODE))
dir.create(out, showWarnings = FALSE, recursive = TRUE)

rd <- function(f) read.csv(f, row.names = 1, check.names = FALSE)

raw <- rd(raw_file)

tr1       <- rd(file.path(base, "fold_1", "training_counts.csv"))
genes_all <- rownames(tr1)

fold_meta <- rbind(rd(file.path(base, "fold_1", "training_meta.csv")),
                   rd(file.path(base, "fold_1", "validation_meta.csv")))
samples <- rownames(fold_meta)
y <- setNames(as.character(fold_meta$source_name), samples)

pmeta   <- rd(meta_file)
patient <- setNames(as.character(pmeta[samples, "patient_id"]), samples)
stopifnot(!any(is.na(patient)))

stopifnot(all(genes_all %in% rownames(raw)), all(samples %in% colnames(raw)))
counts_all <- raw[genes_all, samples]

train_ids <- lapply(1:5, function(f)
  colnames(rd(file.path(base, paste0("fold_", f), "training_counts.csv"))))
stopifnot(all(unlist(train_ids) %in% samples))

# patients with exactly one tumour and one normal sample
pairs <- split(samples, patient)
pairs <- pairs[sapply(pairs, function(s) length(s) == 2 && length(unique(y[s])) == 2)]
cat("Permutation mode:", PERM_MODE, "\n")
cat("Paired patients:", length(pairs), "| single-sample patients:",
    length(unique(patient)) - length(pairs), "\n")
if (PERM_MODE == "within_patient")
  cat("Number of distinct label permutations:", 2^length(pairs), "\n")

permute_labels <- function() {
  if (PERM_MODE == "sample") {
    return(setNames(sample(unname(y)), names(y)))
  }
  y_new <- y
  for (s in pairs) {
    if (runif(1) < 0.5) y_new[s] <- rev(y[s])   # swap tumour/normal within patient
  }
  y_new
}

gene_rows  <- list()
label_rows <- list()
t0 <- Sys.time()

for (p in 1:N_PERM) {
  y_perm <- permute_labels()
  label_rows[[p]] <- data.frame(perm = p, sample = names(y_perm), label = unname(y_perm))

  for (f in 1:5) {
    tr <- train_ids[[f]]
    g  <- factor(y_perm[tr])
    if (nlevels(g) < 2 || min(table(g)) < 2) {
      cat("perm", p, "fold", f, ": skipped (one class too small)\n"); next
    }

    dge    <- DGEList(counts = counts_all[, tr], group = g)
    dge    <- normLibSizes(dge)
    design <- model.matrix(~ g)
    dge    <- estimateDisp(dge, design)
    fit    <- glmQLFit(dge, design)
    res    <- glmQLFTest(fit, coef = 2)
    tab    <- topTags(res, n = Inf)$table

    # Under permuted labels almost no gene passes FDR < 0.05, so genes are
    # ranked by p-value among those with |log2FC| > 6 (top 80).
    tab    <- tab[abs(tab$logFC) > LOGFC_CUT, ]
    tab    <- tab[order(tab$PValue), ]
    genes  <- head(rownames(tab), N_TOP)

    if (length(genes) > 0)
      gene_rows[[paste(p, f)]] <- data.frame(perm = p, fold = f, gene = genes)
  }

  if (p %% 10 == 0 || p == N_PERM) {
    write.csv(do.call(rbind, gene_rows),  file.path(out, "perm_genes.csv"),  row.names = FALSE)
    write.csv(do.call(rbind, label_rows), file.path(out, "perm_labels.csv"), row.names = FALSE)
    el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
    cat(sprintf("perm %d / %d done | %.1f min so far | about %.0f min left\n",
                p, N_PERM, el, el / p * (N_PERM - p)))
  }
}
cat("DONE. Saved to", out, "\n")
