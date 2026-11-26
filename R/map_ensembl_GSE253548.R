# Run from the project root (open Revision_Reviewer_final.Rproj in RStudio first).
# ============================================================
# Ensembl -> gene symbol table for GSE253548 (offline, org.Hs.eg.db).
# Output: data/GSE253548/ensembl_to_symbol.csv
# ============================================================
library(readxl)
library(AnnotationDbi)
library(org.Hs.eg.db)

xlsx <- "data/GSE253548/GSE253548_NormCountsSalmonFFPEOS_12012024.xlsx"
ids  <- read_excel(xlsx, range = cell_cols("A"))[[1]]
ens  <- unique(sub("\\..*$", "", ids))                 # drop version suffix

map <- AnnotationDbi::select(org.Hs.eg.db, keys = ens, keytype = "ENSEMBL",
                             columns = "SYMBOL")
map <- map[!is.na(map$SYMBOL) & !duplicated(map$ENSEMBL), ]
write.csv(map, "data/GSE253548/ensembl_to_symbol.csv", row.names = FALSE)
cat("Ensembl IDs:", length(ens), "| mapped to a symbol:", nrow(map), "\n")
