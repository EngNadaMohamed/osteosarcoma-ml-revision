# ============================================================
# descriptive_figures.R
# Descriptive figures for GSE218035: volcano plot, DEG heatmap,
# PCA, and pathway enrichment (KEGG, Reactome, GO, GSEA Hallmark).
#
# NOTE: these analyses use ALL 36 samples and are DESCRIPTIVE ONLY.
# They are not used for model training or evaluation; for the
# classifiers, differential expression is re-run within each
# training fold (see edger_result.R).
#
# Run from the project root (open Revision_Reviewer_final.Rproj).
# ============================================================

library(edgeR)
library(ggplot2)
library(pheatmap)
library(clusterProfiler)
library(org.Hs.eg.db)
library(ReactomePA)
library(msigdbr)
library(enrichplot)
library(dplyr)
library(tibble)

set.seed(42)

# ------------------------------------------------------------
# 0. Paths and settings
# ------------------------------------------------------------
raw_file  <- "data/raw_counts_data.csv"
meta_file <- "data/meta_revision_with_patient_id.csv"
gene_file <- "data/filtered_counts_before_edgeR_revision.csv"   # same gene set as the folds
out_dir   <- "data/descriptive_figures"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

FDR_CUT   <- 0.05
LOGFC_CUT <- 6

# ------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------
raw  <- read.csv(raw_file,  row.names = 1, check.names = FALSE)
meta <- read.csv(meta_file, row.names = 1, check.names = FALSE)
genes_used <- rownames(read.csv(gene_file, row.names = 1, check.names = FALSE, nrows = 1e6))

counts <- as.matrix(raw[genes_used, rownames(meta)])
stopifnot(all(colnames(counts) == rownames(meta)))

group <- factor(meta$source_name,
                levels = c("Normal_adjacent_tissue", "Osteosarcoma_sample"))

# ------------------------------------------------------------
# 2. edgeR on all samples (descriptive)
# ------------------------------------------------------------
dge    <- DGEList(counts = counts, group = group)
dge    <- normLibSizes(dge)
design <- model.matrix(~ group)
dge    <- estimateDisp(dge, design)
fit    <- glmQLFit(dge, design)
res    <- glmQLFTest(fit, coef = 2)          # Osteosarcoma vs Normal

edgeR_table <- topTags(res, n = Inf)$table
edgeR_deg   <- edgeR_table[edgeR_table$FDR < FDR_CUT &
                             abs(edgeR_table$logFC) > LOGFC_CUT, ]
cat("DEGs (all samples):", nrow(edgeR_deg),
    "| up:", sum(edgeR_deg$logFC > 0), "| down:", sum(edgeR_deg$logFC < 0), "\n")

write.csv(edgeR_table, file.path(out_dir, "edgeR_all_samples_results.csv"))
write.csv(edgeR_deg,   file.path(out_dir, "edgeR_all_samples_DEGs.csv"))

logCPM <- cpm(dge, log = TRUE, prior.count = 1)

# ------------------------------------------------------------
# 3. Volcano plot
# ------------------------------------------------------------
edgeR_table$significant <- edgeR_table$FDR < FDR_CUT &
  abs(edgeR_table$logFC) > LOGFC_CUT

p_volcano <- ggplot(edgeR_table, aes(x = logFC, y = -log10(FDR))) +
  geom_point(aes(color = significant), alpha = 0.6, size = 1) +
  scale_color_manual(values = c("gray70", "firebrick3")) +
  geom_vline(xintercept = c(-LOGFC_CUT, LOGFC_CUT), linetype = "dashed") +
  geom_hline(yintercept = -log10(FDR_CUT), linetype = "dashed") +
  labs(title = "Volcano plot (edgeR, all samples)",
       x = "log2 fold change (osteosarcoma vs adjacent normal)",
       y = "-log10 FDR") +
  theme_minimal() +
  theme(legend.position = "none")
ggsave(file.path(out_dir, "volcano_plot.pdf"), p_volcano, width = 6, height = 5)

# ------------------------------------------------------------
# 4. Heatmap of DEGs (z-scored log-CPM)
# ------------------------------------------------------------
scaled_data <- t(scale(t(logCPM[rownames(edgeR_deg), ])))
annotation_col <- data.frame(Group = meta$source_name, row.names = rownames(meta))
stopifnot(all(colnames(scaled_data) == rownames(annotation_col)))
rownames(scaled_data) <- substr(rownames(scaled_data), 1, 15)

pheatmap(scaled_data,
         annotation_col = annotation_col,
         show_rownames = TRUE, show_colnames = TRUE,
         fontsize_row = 4,
         cluster_rows = TRUE, cluster_cols = TRUE,
         scale = "none",
         main = "Differentially expressed genes (edgeR, all samples)",
         color = colorRampPalette(c("navy", "white", "firebrick3"))(100),
         filename = file.path(out_dir, "heatmap_DEGs.pdf"),
         width = 8, height = 10)

# ------------------------------------------------------------
# 5. PCA on log-CPM
# ------------------------------------------------------------
pca_res <- prcomp(t(logCPM), scale. = TRUE)
var_exp <- round(100 * summary(pca_res)$importance[2, 1:2], 1)
pca_df  <- data.frame(PC1 = pca_res$x[, 1], PC2 = pca_res$x[, 2],
                      Group = meta$source_name)

p_pca <- ggplot(pca_df, aes(PC1, PC2, color = Group)) +
  geom_point(size = 3) +
  scale_color_manual(values = c("steelblue", "firebrick3")) +
  labs(title = "PCA (edgeR log-CPM)",
       x = paste0("PC1 (", var_exp[1], "%)"),
       y = paste0("PC2 (", var_exp[2], "%)")) +
  theme_minimal()
ggsave(file.path(out_dir, "pca_plot.pdf"), p_pca, width = 6, height = 5)

# ------------------------------------------------------------
# 6. Map gene symbols to Entrez IDs (org.Hs.eg.db)
#    Background universe = all genes tested by edgeR (expressed
#    genes after filtering), not the whole genome.
# ------------------------------------------------------------
conv_deg <- bitr(rownames(edgeR_deg), fromType = "SYMBOL",
                 toType = "ENTREZID", OrgDb = org.Hs.eg.db)
entrez_ids <- unique(conv_deg$ENTREZID)

conv_universe <- bitr(rownames(edgeR_table), fromType = "SYMBOL",
                      toType = "ENTREZID", OrgDb = org.Hs.eg.db)
universe_ids <- unique(conv_universe$ENTREZID)

cat("DEGs mapped to Entrez:", length(entrez_ids),
    "| background universe:", length(universe_ids), "genes\n")

# Same multiple-testing rule for every enrichment analysis:
# Benjamini-Hochberg adjusted p < 0.05 (no extra q-value filter).
P_ADJ_METHOD <- "BH"
P_ADJ_CUT    <- 0.05

# ------------------------------------------------------------
# 7. KEGG, Reactome and GO (over-representation)
# ------------------------------------------------------------
kegg_results <- enrichKEGG(gene = entrez_ids, organism = "hsa",
                           keyType = "ncbi-geneid", universe = universe_ids,
                           pAdjustMethod = P_ADJ_METHOD,
                           pvalueCutoff = P_ADJ_CUT, qvalueCutoff = 1)
write.csv(as.data.frame(kegg_results), file.path(out_dir, "KEGG_results.csv"), row.names = FALSE)
ggsave(file.path(out_dir, "KEGG_dotplot.pdf"),
       dotplot(kegg_results, showCategory = 20, label_format = 50), width = 8, height = 7)

reactome_results <- enrichPathway(gene = entrez_ids, organism = "human",
                                  universe = universe_ids,
                                  pAdjustMethod = P_ADJ_METHOD,
                                  pvalueCutoff = P_ADJ_CUT, qvalueCutoff = 1)
write.csv(as.data.frame(reactome_results), file.path(out_dir, "Reactome_results.csv"), row.names = FALSE)
ggsave(file.path(out_dir, "Reactome_dotplot.pdf"),
       dotplot(reactome_results, showCategory = 20, label_format = 50), width = 8, height = 7)

go_results <- enrichGO(gene = entrez_ids, OrgDb = org.Hs.eg.db,
                       keyType = "ENTREZID", ont = "BP",
                       universe = universe_ids,
                       pAdjustMethod = P_ADJ_METHOD,
                       pvalueCutoff = P_ADJ_CUT, qvalueCutoff = 1)
write.csv(as.data.frame(go_results), file.path(out_dir, "GO_BP_results.csv"), row.names = FALSE)

# Overlapping GO terms (many terms share the same genes): keep one
# representative term per group of similar terms (semantic similarity > 0.7).
go_simple <- if (nrow(as.data.frame(go_results)) > 0) {
  simplify(go_results, cutoff = 0.7, by = "p.adjust", select_fun = min)
} else go_results
write.csv(as.data.frame(go_simple), file.path(out_dir, "GO_BP_results_simplified.csv"), row.names = FALSE)
cat("GO BP terms:", nrow(as.data.frame(go_results)),
    "| after removing redundant terms:", nrow(as.data.frame(go_simple)), "\n")
ggsave(file.path(out_dir, "GO_BP_dotplot.pdf"),
       dotplot(go_simple, showCategory = 30, label_format = 50), width = 8, height = 9)

# ------------------------------------------------------------
# 8. GSEA on Hallmark gene sets (all genes, ranked)
# ------------------------------------------------------------
ranked <- edgeR_table %>%
  rownames_to_column("SYMBOL") %>%
  filter(!is.na(logFC), !is.na(FDR)) %>%
  mutate(rank_metric = sign(logFC) * -log10(FDR))

conv_all <- bitr(ranked$SYMBOL, fromType = "SYMBOL",
                 toType = "ENTREZID", OrgDb = org.Hs.eg.db) %>%
  distinct(SYMBOL, .keep_all = TRUE)

gene_list <- ranked %>%
  inner_join(conv_all, by = "SYMBOL") %>%
  arrange(desc(rank_metric)) %>%
  distinct(ENTREZID, .keep_all = TRUE)
gene_list <- setNames(gene_list$rank_metric, gene_list$ENTREZID)

# msigdbr >= 10 uses `collection` and `ncbi_gene`; older versions use `category` and `entrez_gene`
hallmark_raw <- tryCatch(msigdbr(species = "Homo sapiens", collection = "H"),
                         error = function(e) msigdbr(species = "Homo sapiens", category = "H"))
entrez_col <- intersect(c("ncbi_gene", "entrez_gene"), colnames(hallmark_raw))[1]
hallmark <- hallmark_raw %>%
  dplyr::select(gs_name, entrez = all_of(entrez_col)) %>%
  filter(!is.na(entrez))
cat("Hallmark gene sets:", length(unique(hallmark$gs_name)), "\n")

gsea_results <- GSEA(gene_list, TERM2GENE = hallmark,
                     pAdjustMethod = P_ADJ_METHOD, pvalueCutoff = P_ADJ_CUT,
                     seed = TRUE)
gsea_df <- as.data.frame(gsea_results)
write.csv(gsea_df, file.path(out_dir, "GSEA_Hallmark_results.csv"), row.names = FALSE)

if (nrow(gsea_df) > 0) {
  ggsave(file.path(out_dir, "GSEA_dotplot.pdf"),
         dotplot(gsea_results, showCategory = 20, label_format = 50), width = 8, height = 7)
  ggsave(file.path(out_dir, "GSEA_ridgeplot.pdf"),
         ridgeplot(gsea_results), width = 8, height = 7)
  ggsave(file.path(out_dir, "GSEA_enrichment_map.pdf"),
         emapplot(pairwise_termsim(gsea_results), showCategory = 10), width = 8, height = 7)
  ggsave(file.path(out_dir, "GSEA_cnetplot.pdf"),
         cnetplot(gsea_results, showCategory = 5, foldChange = gene_list,
                  circular = TRUE, colorEdge = TRUE, node_label = "category"),
         width = 9, height = 9)
  for (gs in intersect(c("HALLMARK_DNA_REPAIR", "HALLMARK_APOPTOSIS",
                         "HALLMARK_MYOGENESIS"), gsea_df$ID)) {
    ggsave(file.path(out_dir, paste0("GSEA_", gs, ".pdf")),
           gseaplot2(gsea_results, geneSetID = gs), width = 7, height = 5)
  }
}

writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
cat("All figures and tables saved in:", out_dir, "\n")

# Summary of enrichment results (all BH-adjusted p < 0.05, same background)
cat("\nSignificant terms -> KEGG:", nrow(as.data.frame(kegg_results)),
    "| Reactome:", nrow(as.data.frame(reactome_results)),
    "| GO BP (simplified):", nrow(as.data.frame(go_simple)),
    "| GSEA Hallmark:", nrow(gsea_df), "\n")
head(gsea_df[, c("ID", "NES", "p.adjust")], 15)