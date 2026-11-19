# Run from the project root (open Revision_Reviewer_final.Rproj in RStudio first).

library(edgeR)

# Project data directory
base <- "data/revision_folds_grouped"

# Raw counts file
raw_file <- "data/raw_counts_data.csv"

# permutation output directory
base_path <- "data"

out      <- file.path(base_path, "perm_out_grouped")
dir.create(out, showWarnings = FALSE)

N_PERM <- 100       
N_TOP  <- 80
set.seed(123)

rd <- function(f) read.csv(f, row.names = 1, check.names = FALSE)

raw <- rd(raw_file)

tr1       <- rd(file.path(base, "fold_1", "training_counts.csv"))
genes_all <- rownames(tr1)

meta <- rbind(rd(file.path(base, "fold_1", "training_meta.csv")),
              rd(file.path(base, "fold_1", "validation_meta.csv")))
samples <- rownames(meta)
y <- setNames(as.character(meta$source_name), samples)

stopifnot(all(genes_all %in% rownames(raw)), all(samples %in% colnames(raw)))
counts_all <- raw[genes_all, samples]

train_ids <- lapply(1:5, function(f)
  colnames(rd(file.path(base, paste0("fold_", f), "training_counts.csv"))))
stopifnot(all(unlist(train_ids) %in% samples))

gene_rows  <- list()
label_rows <- list()

for (p in 1:N_PERM) {
  y_perm <- setNames(sample(unname(y)), names(y))
  label_rows[[p]] <- data.frame(perm = p, sample = names(y_perm), label = unname(y_perm))
  
  for (f in 1:5) {
    tr <- train_ids[[f]]
    g  <- factor(y_perm[tr])
    if (nlevels(g) < 2 || min(table(g)) < 2) {
      cat("perm", p, "fold", f, ": skipped (one class too small)\n"); next
    }
    
    dge    <- DGEList(counts = counts_all[, tr], group = g)
    dge    <- calcNormFactors(dge)
    design <- model.matrix(~ g)
    dge    <- estimateDisp(dge, design)
    fit    <- glmQLFit(dge, design)
    res    <- glmQLFTest(fit, coef = 2)
    tab    <- topTags(res, n = Inf)$table
    
    tab    <- tab[abs(tab$logFC) > 6, ]
    tab    <- tab[order(tab$PValue), ]
    genes  <- head(rownames(tab), N_TOP)
    
    if (length(genes) > 0)
      gene_rows[[paste(p, f)]] <- data.frame(perm = p, fold = f, gene = genes)
    cat("perm", p, "fold", f, ":", length(genes), "genes\n")
  }
  
  write.csv(do.call(rbind, gene_rows),  file.path(out, "perm_genes.csv"),  row.names = FALSE)
  write.csv(do.call(rbind, label_rows), file.path(out, "perm_labels.csv"), row.names = FALSE)
}
cat("DONE\n")