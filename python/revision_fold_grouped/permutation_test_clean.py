"""Label-permutation test with patient-structured permutations (430 permutations).

Uses permuted labels and per-fold DEGs produced by R/permutation_test_clean.R,
re-trains all 31 configurations on the same patient-grouped folds, and compares
the observed scores with the permutation null distribution.
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
# Must match PERM_MODE in R/permutation_test_clean.R
PERM_MODE = "within_patient"          # "within_patient" (main) or "sample"
PERM_DIR = DATA / f"perm_out_grouped_{PERM_MODE}"
REAL_BY_FOLD = DATA / "revision_fold_results" / "main_results_by_fold_id.csv"

K_BEST = 10
N_FOLDS = 5
MAX_PERMS = None

USE_LOG2 = True

METRICS = ["F1", "AUC", "BalAcc"]


def read(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


# ------------------------------------------------------------
# 1. Full expression matrix = training + validation samples of fold 1
# ------------------------------------------------------------
tr1 = read(FOLD_ROOT / "fold_1" / "training_counts.csv")
va1 = read(FOLD_ROOT / "fold_1" / "validation_counts.csv")
full = pd.concat([tr1, va1.loc[tr1.index]], axis=1)
if USE_LOG2:
    full = np.log2(full + 1)

# Same patient-grouped folds as the main analysis
train_ids, test_ids = {}, {}
for f in range(1, N_FOLDS + 1):
    d = FOLD_ROOT / f"fold_{f}"
    train_ids[f] = list(read(d / "training_counts.csv").columns)
    test_ids[f] = list(read(d / "validation_counts.csv").columns)

labels_df = pd.read_csv(PERM_DIR / "perm_labels.csv")
genes_df = pd.read_csv(PERM_DIR / "perm_genes.csv")
labels_df["sample"] = labels_df["sample"].astype(str)
genes_df["gene"] = genes_df["gene"].astype(str)

perms = sorted(labels_df["perm"].unique())
if MAX_PERMS:
    perms = perms[:MAX_PERMS]

# ------------------------------------------------------------
# 2. Evaluate all 31 configurations on every permutation
# ------------------------------------------------------------
rows = []
for p in perms:
    lab = labels_df[labels_df["perm"] == p].set_index("sample")["label"]
    y_all = lab.str.contains("Osteosarcoma").astype(int)

    for f in range(1, N_FOLDS + 1):
        genes = genes_df[(genes_df["perm"] == p) & (genes_df["fold"] == f)]["gene"]
        genes = [g for g in genes if g in full.index]
        if not genes:
            continue

        X_tr = full.loc[genes, train_ids[f]].T
        X_te = full.loc[genes, test_ids[f]].T
        keep = X_tr.columns[X_tr.nunique() > 1]
        X_tr, X_te = X_tr[keep], X_te[keep]
        y_tr, y_te = y_all.loc[X_tr.index], y_all.loc[X_te.index]

        if y_tr.nunique() < 2 or X_tr.shape[1] == 0:
            continue
        k = min(K_BEST, X_tr.shape[1])

        pool = get_model_pool()
        for size in range(1, len(pool) + 1):
            for combo in combinations(pool.items(), size):
                name = " + ".join(n.upper() for n, _ in combo)
                model = build_combination(combo, k_best=k).fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                prob = model.predict_proba(X_te)[:, list(model.classes_).index(1)]
                # AUC requires both classes in the test fold
                auc = roc_auc_score(y_te, prob) if y_te.nunique() == 2 else np.nan
                rows.append({
                    "Perm": p, "Fold": f, "Model": name,
                    "F1": f1_score(y_te, y_pred, zero_division=0),
                    "AUC": auc,
                    "BalAcc": balanced_accuracy_score(y_te, y_pred),
                    "N_genes": X_tr.shape[1],
                })
    print(f"perm {p} done", flush=True)
    pd.DataFrame(rows).to_csv(PERM_DIR /"perm_results_long.csv", index=False)

# ------------------------------------------------------------
# 3. Observed (real-label) results, read per fold so balanced accuracy can be computed
# ------------------------------------------------------------
real_fold = pd.read_csv(REAL_BY_FOLD)
sens = real_fold["TP"] / (real_fold["TP"] + real_fold["FN"])
spec = real_fold["TN"] / (real_fold["TN"] + real_fold["FP"])
real_fold["BalAcc"] = (sens + spec) / 2
real = real_fold.groupby("Model")[METRICS].mean()          # model x metric

# ------------------------------------------------------------
# 4. Permutation p-values (per configuration and best-of-31 max-statistic)
# ------------------------------------------------------------
res = pd.DataFrame(rows)
summary_rows, per_model_all = [], []

for m in METRICS:
    null_mean = res.groupby(["Perm", "Model"])[m].mean().unstack()   # perm x model
    null_best = null_mean.max(axis=1)                               # best configuration in each permutation
    n = len(null_best)

    real_best = real[m].max()
    p_best = (1 + (null_best >= real_best).sum()) / (n + 1)

    summary_rows.append({
        "Metric": m,
        "Real_best_of_31": real_best,
        "Null_best_of_31_mean": null_best.mean(),
        "Null_best_of_31_sd": null_best.std(),
        "Null_best_of_31_95th": null_best.quantile(0.95),
        "Null_best_of_31_max": null_best.max(),
        "p_value_best_of_31": p_best,
        "n_perm": n,
    })

    pm = pd.DataFrame({
        "Metric": m,
        "Real": real[m],
        "Null_mean": null_mean.mean(),
        "Null_sd": null_mean.std(),
        "Null_max": null_mean.max(),
    })
    pm["p_value"] = [
        (1 + (null_mean[mod] >= pm.loc[mod, "Real"]).sum()) / (n + 1)
        for mod in pm.index
    ]
    per_model_all.append(pm)
    null_best.to_csv(PERM_DIR / f"perm_null_best_{m}.csv")

summary = pd.DataFrame(summary_rows)
summary.to_csv(PERM_DIR / "perm_summary.csv", index=False)
pd.concat(per_model_all).to_csv(PERM_DIR / "perm_per_model_pvalues.csv")

print("\n" + summary.round(4).to_string(index=False))


