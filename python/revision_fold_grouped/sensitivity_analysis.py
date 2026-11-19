"""Sensitivity analysis for the DEG fold-change threshold and the number of
selected features (k). Reviewer comments #6 and #7.

Uses the per-fold edgeR tables that were computed on TRAINING samples only
(Fold_X_edgeR_results.csv contains every gene), so edgeR does not need to be
re-run: each threshold is applied to the same training-only statistics.
"""
import warnings
warnings.filterwarnings("ignore")

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

from train_model_clean_id import get_model_pool, build_combination

DATA = Path(__file__).resolve().parents[2] / "data"
FOLD_ROOT = DATA / "revision_folds_grouped"
DEG_ROOT = DATA / "Edger_result_grouped_6"
OUT = DATA / "sensitivity_results"
OUT.mkdir(parents=True, exist_ok=True)

LOGFC_GRID = [1, 2, 4, 6]
K_GRID = [5, 10, 20, 50]
FDR = 0.05
TOP_N = 80          # same cap as the main analysis
N_FOLDS = 5


def read(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def encode(series):
    return series.astype(str).str.contains("Osteosarcoma").astype(int)


# ------------------------------------------------------------
# 1. Load folds once
# ------------------------------------------------------------
folds = {}
for f in range(1, N_FOLDS + 1):
    d = FOLD_ROOT / f"fold_{f}"
    tr = np.log2(read(d / "training_counts.csv") + 1)       # log2(CPM + 1)
    va = np.log2(read(d / "validation_counts.csv") + 1)
    mtr = read(d / "training_meta.csv").loc[tr.columns]
    mva = read(d / "validation_meta.csv").loc[va.columns]
    tab = read(DEG_ROOT / f"Fold_{f}_edgeR_results.csv")    # all genes, training only
    folds[f] = (tr, va, encode(mtr["source_name"]), encode(mva["source_name"]), tab)

# ------------------------------------------------------------
# 2. Run every threshold x k x fold x 31 configurations
# ------------------------------------------------------------
rows, gene_info = [], []
for lfc in LOGFC_GRID:
    for f, (tr, va, ytr, yva, tab) in folds.items():
        deg = tab[(tab["FDR"] < FDR) & (tab["logFC"].abs() > lfc)]
        deg = deg.sort_values(["FDR", "PValue"])
        genes = [g for g in deg.index[:TOP_N] if g in tr.index and g in va.index]
        gene_info.append({
            "logFC_threshold": lfc, "Fold": f,
            "DEGs_passing": len(deg), "Genes_used": len(genes),
            "Pct_down_in_tumour": 100 * (deg.loc[genes, "logFC"] < 0).mean(),
        })

        X_tr, X_va = tr.loc[genes].T, va.loc[genes].T
        keep = X_tr.columns[X_tr.nunique() > 1]
        X_tr, X_va = X_tr[keep], X_va[keep]

        for k in K_GRID:
            if X_tr.shape[1] < k:
                print(f"logFC>{lfc} fold {f}: only {X_tr.shape[1]} genes, skip k={k}")
                continue
            for size in range(1, 6):
                for combo in combinations(get_model_pool().items(), size):
                    name = " + ".join(n.upper() for n, _ in combo)
                    m = build_combination(combo, k_best=k).fit(X_tr, ytr)
                    yp = m.predict(X_va)
                    pr = m.predict_proba(X_va)[:, list(m.classes_).index(1)]
                    rows.append({
                        "logFC_threshold": lfc, "k": k, "Fold": f, "Model": name,
                        "F1": f1_score(yva, yp, zero_division=0),
                        "AUC": roc_auc_score(yva, pr),
                        "BalAcc": balanced_accuracy_score(yva, yp),
                    })
        print(f"logFC>{lfc} fold {f} done", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "sensitivity_per_fold.csv", index=False)
genes_df = pd.DataFrame(gene_info)
genes_df.to_csv(OUT / "sensitivity_gene_counts.csv", index=False)

# ------------------------------------------------------------
# 3. Summaries
# ------------------------------------------------------------
per_model = (res.groupby(["logFC_threshold", "k", "Model"])[["F1", "AUC", "BalAcc"]]
             .mean().reset_index())
per_model.to_csv(OUT / "sensitivity_per_model.csv", index=False)

grid = (per_model.groupby(["logFC_threshold", "k"])
        .agg(Best_F1=("F1", "max"), Median_F1=("F1", "median"), Min_F1=("F1", "min"),
             Best_AUC=("AUC", "max"), Median_AUC=("AUC", "median"), Min_AUC=("AUC", "min"),
             Median_BalAcc=("BalAcc", "median"))
        .reset_index())
grid.to_csv(OUT / "sensitivity_summary.csv", index=False)

pd.set_option("display.width", 200)
print("\nGenes per fold:")
print(genes_df.groupby("logFC_threshold")[["DEGs_passing", "Genes_used", "Pct_down_in_tumour"]]
      .mean().round(1).to_string())
print("\nPerformance across the 31 configurations:")
print(grid.round(3).to_string(index=False))
print("\nAll results saved in:", OUT.resolve())