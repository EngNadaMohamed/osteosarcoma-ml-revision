"""External validation with a locked pipeline.

The complete pipeline is trained once on all 36 GSE218035 samples and then
applied, unchanged, to two independent cohorts (GSE99671, GSE253548):

  * genes: top 80 edgeR DEGs on all GSE218035 samples (FDR < 0.05,
    |log2FC| > 6), read from R/figures/descriptive_figures.R output
  * features: log2(CPM + 1)
  * StandardScaler and SelectKBest (k = 10) fitted on GSE218035 only
  * the same 31 configurations (default hyperparameters)
  * fixed threshold 0.5, osteosarcoma = positive class

Nothing from the external cohorts is used for gene selection, scaling,
training or threshold choice. If some of the 80 genes are not measured in an
external cohort, the locked model for that cohort is trained (on GSE218035
only) with the genes that are available, and the number is reported.

Inputs (data/):
  raw_counts_data.csv, meta_revision_with_patient_id.csv
  descriptive_figures/edgeR_all_samples_results.csv
  GSE99671/GSE99671_Osteosarcoma_filecounts.txt, GSE99671/GSE99671_series_matrix.txt
  GSE253548/GSE253548_NormCountsSalmonFFPEOS_12012024.xlsx,
  GSE253548/GSE253548_series_matrix.txt, GSE253548/ensembl_to_symbol.csv
    (made by R/map_ensembl_GSE253548.R)
Outputs: data/external_locked/
"""
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "python" / "revision_fold_grouped"))
from train_model_clean_id import build_combination, get_model_pool  # noqa: E402

warnings.filterwarnings("ignore")

DATA = PROJECT_DIR / "data"
OUT = DATA / "external_locked"
OUT.mkdir(parents=True, exist_ok=True)

N_TOP, K_BEST, FDR_CUT, LOGFC_CUT, THRESHOLD = 80, 10, 0.05, 6, 0.5
QC_MIN_GENES, QC_MIN_COUNT = 1000, 10        # GSE253548 RNA-quality filter (FFPE)


def log2cpm(counts):
    counts = counts.groupby(level=0).sum()
    return np.log2(counts / counts.sum(axis=0) * 1e6 + 1)


def series_field(path, key):
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith("!Sample_characteristics_ch1") and key in line:
            return [v.strip().strip('"').split(": ", 1)[1] for v in line.rstrip("\n").split("\t")[1:]]
    raise ValueError(f"{key} not found in {path}")


def series_titles(path):
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith("!Sample_title"):
            return [v.strip().strip('"') for v in line.rstrip("\n").split("\t")[1:]]
    raise ValueError(f"titles not found in {path}")


# ------------------------------------------------------------------
# 1. Training cohort: GSE218035 (all 36 samples) and the locked genes
# ------------------------------------------------------------------
raw = pd.read_csv(DATA / "raw_counts_data.csv", index_col=0)
meta = pd.read_csv(DATA / "meta_revision_with_patient_id.csv", index_col=0)
raw = raw[meta.index]
X_train_all = log2cpm(raw)
y_train = meta["source_name"].str.startswith("Osteo").astype(int)

edger = pd.read_csv(DATA / "descriptive_figures" / "edgeR_all_samples_results.csv", index_col=0)
edger = edger[(edger["FDR"] < FDR_CUT) & (edger["logFC"].abs() > LOGFC_CUT)]
locked_genes = list(edger.sort_values(["FDR", "PValue"]).index[:N_TOP])
pd.Series(locked_genes, name="gene").to_csv(OUT / "locked_genes_top80.csv", index=False)
print(f"Locked genes from GSE218035: {len(locked_genes)}")

# ------------------------------------------------------------------
# 2. External cohorts
# ------------------------------------------------------------------
cohorts = {}

# GSE99671: raw counts; labels from the series matrix (column order = GSM order;
# the series matrix fixes the swapped T/N column names of OSVN004)
d = DATA / "GSE99671"
c = pd.read_csv(d / "GSE99671_Osteosarcoma_filecounts.txt", sep="\t", index_col=0)
c = c.set_index("symbol").drop(columns="Chr")
lab = series_field(d / "GSE99671_series_matrix.txt", "tumor:")
titles = series_titles(d / "GSE99671_series_matrix.txt")
keep = [i for i, t in enumerate(titles) if t.startswith("OS")]   # the 36 tissue samples
# same patient, same order as the count columns (the T/N suffix of OSVN004 is swapped in the counts file)
lab = [lab[i] for i in keep]
assert len(lab) == c.shape[1] and all(titles[i][:7] == col[:7] for i, col in zip(keep, c.columns))
cohorts["GSE99671"] = (log2cpm(c), pd.Series([int(v == "TUMOR") for v in lab], index=c.columns))

# GSE253548: Salmon normalised counts (FFPE); RNA-quality filter fixed in advance
d = DATA / "GSE253548"
x = pd.read_excel(d / "GSE253548_NormCountsSalmonFFPEOS_12012024.xlsx", index_col=0)
x.index = x.index.str.split(".").str[0]
# QC on all quantified genes, before symbol mapping
qc = (x >= QC_MIN_COUNT).sum(axis=0) >= QC_MIN_GENES
mp = pd.read_csv(d / "ensembl_to_symbol.csv").set_index("ENSEMBL")["SYMBOL"]
x = x.loc[x.index.intersection(mp.index)]
x.index = mp[x.index].values
lab = series_field(d / "GSE253548_series_matrix.txt", "disease state:")
y = pd.Series([int(v == "TUMOUR") for v in lab], index=x.columns)
print(f"GSE253548: {qc.sum()} of {len(qc)} samples passed QC "
      f"({int(y[qc].sum())} tumour, {int((1 - y[qc]).sum())} normal)")
cohorts["GSE253548"] = (log2cpm(x.loc[:, qc]), y[qc])

# ------------------------------------------------------------------
# 3. Train on GSE218035 only, apply unchanged to each cohort
# ------------------------------------------------------------------
pool = get_model_pool()
combos = [cmb for s in range(1, len(pool) + 1) for cmb in combinations(pool.items(), s)]
rows, preds = [], []

for name, (X_ext, y_ext) in cohorts.items():
    genes = [g for g in locked_genes if g in X_ext.index and g in X_train_all.index]
    print(f"\n{name}: {len(y_ext)} samples ({int(y_ext.sum())} tumour) | "
          f"locked genes available: {len(genes)} / {len(locked_genes)}")
    Xtr = X_train_all.loc[genes].T
    Xte = X_ext.loc[genes, y_ext.index].T
    for combo in combos:
        cfg = " + ".join(n.upper() for n, _ in combo)
        model = build_combination(combo, k_best=min(K_BEST, len(genes))).fit(Xtr, y_train)
        p = model.predict_proba(Xte)[:, list(model.classes_).index(1)]
        yp = (p >= THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_ext, yp, labels=[0, 1]).ravel()
        rows.append({"Cohort": name, "Model": cfg, "N_genes": len(genes),
                     "AUC": roc_auc_score(y_ext, p),
                     "Sensitivity": tp / (tp + fn), "Specificity": tn / (tn + fp),
                     "BalAcc": balanced_accuracy_score(y_ext, yp),
                     "F1": f1_score(y_ext, yp, zero_division=0),
                     "TN": tn, "FP": fp, "FN": fn, "TP": tp})
        preds += [{"Cohort": name, "Model": cfg, "Sample": s, "y": a, "p": q}
                  for s, a, q in zip(Xte.index, y_ext, p)]

res = pd.DataFrame(rows)
res.to_csv(OUT / "locked_external_results.csv", index=False)
pd.DataFrame(preds).to_csv(OUT / "locked_external_predictions.csv", index=False)

pd.set_option("display.width", 200)
print("\nSummary over the 31 configurations (min / median / max):")
print(res.groupby("Cohort")[["AUC", "Sensitivity", "Specificity", "BalAcc"]]
      .agg(["min", "median", "max"]).round(3).to_string())
print("\nSingle classifiers and LR + RF:")
sel = res[~res.Model.str.contains(r"\+") | (res.Model == "LR + RF")]
print(sel[["Cohort", "Model", "AUC", "Sensitivity", "Specificity", "BalAcc",
           "TN", "FP", "FN", "TP"]].round(3).to_string(index=False))
print(f"\nSaved to: {OUT}")
