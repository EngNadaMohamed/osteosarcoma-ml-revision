"""Extra analyses requested by the reviewer, on the same patient-grouped
5-fold pipeline as main_id.py (within-fold edgeR DEGs, log2(CPM + 1),
StandardScaler -> SelectKBest(k=10) -> classifier, 31 configurations).

  1. Gene selection stability: how many of the 5 folds selected each gene
     (in the top-80 DEG list and in the final SelectKBest top 10).
  2. Performance with uncertainty: fold mean and SD, pooled out-of-fold
     metrics, patient-level bootstrap 95% confidence intervals and the
     pooled confusion matrix for every configuration.
  3. Simple baselines: majority class, and a single-gene logistic regression
     (the best gene by F statistic, chosen inside each training fold).
  4. Probability calibration: Brier score for every configuration and a
     calibration curve figure for selected configurations.

Run from PyCharm (same folder as main_id.py). Outputs go to
data/extra_analyses.
"""
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, roc_auc_score)

from main_id import DEG_ROOT, N_FOLDS, load_fold_data_id
from train_model_clean_id import build_combination, get_model_pool

warnings.filterwarnings("ignore")

# ============================================================
# Settings
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA = PROJECT_DIR / "data"
META = DATA / "meta_revision_with_patient_id.csv"
OUT = DATA / "extra_analyses"
OUT.mkdir(parents=True, exist_ok=True)

K_BEST = 10
TOP_N = 80
N_BOOT = 2000
SEED = 42
CALIBRATION_MODELS = ["LR", "LR + RF", "SVC + ET + LR + RF + GBM"]

meta = pd.read_csv(META, index_col=0)
meta.index = meta.index.astype(str)
patient = meta["patient_id"].astype(str)


def metrics(y, p, threshold=0.5):
    yp = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y, p) if len(set(y)) == 2 else np.nan,
        "F1": f1_score(y, yp, zero_division=0),
        "BalAcc": balanced_accuracy_score(y, yp),
        "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "Brier": brier_score_loss(y, p),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def bootstrap_ci(pred, n_boot=N_BOOT, seed=SEED):
    """Patient-level bootstrap: resample patients, keep all their samples."""
    rng = np.random.default_rng(seed)
    patients = pred["Patient"].unique()
    by_patient = {pt: d for pt, d in pred.groupby("Patient")}
    stats = {"AUC": [], "F1": [], "BalAcc": []}
    for _ in range(n_boot):
        pick = rng.choice(patients, size=len(patients), replace=True)
        b = pd.concat([by_patient[pt] for pt in pick])
        if b["y"].nunique() < 2:
            continue
        m = metrics(b["y"].to_numpy(), b["p"].to_numpy())
        for k in stats:
            stats[k].append(m[k])
    return {f"{k}_CI": f"{np.percentile(v, 2.5):.3f}-{np.percentile(v, 97.5):.3f}"
            for k, v in stats.items()}


# ============================================================
# Run all folds once and keep every out-of-fold prediction
# ============================================================
pool = get_model_pool()
combos = [c for size in range(1, len(pool) + 1)
          for c in combinations(pool.items(), size)]

pred_rows, fold_rows, gene_rows, base_rows = [], [], [], []

for fold in range(1, N_FOLDS + 1):
    X_tr, y_tr, X_va, y_va = load_fold_data_id(fold)

    # ---- 1. gene stability: top-80 list and SelectKBest top 10 ----
    deg = pd.read_csv(DEG_ROOT / f"Fold_{fold}_DEGs.csv", index_col=0)
    deg = deg.sort_values(["FDR", "PValue"]).head(TOP_N)
    sel = build_combination([("lr", pool["lr"])], k_best=K_BEST).fit(X_tr, y_tr)
    top10 = set(X_tr.columns[sel.named_steps["select"].get_support()])
    for g, r in deg.iterrows():
        gene_rows.append({"Gene": g, "Fold": fold, "logFC": r["logFC"],
                          "In_top10": g in top10})

    # ---- 2 and 4. the 31 configurations ----
    for combo in combos:
        name = " + ".join(n.upper() for n, _ in combo)
        model = build_combination(combo, k_best=K_BEST).fit(X_tr, y_tr)
        p = model.predict_proba(X_va)[:, list(model.classes_).index(1)]
        fold_rows.append({"Fold": fold, "Model": name,
                          **metrics(y_va.to_numpy(), p)})
        for s, yy, pp in zip(X_va.index, y_va, p):
            pred_rows.append({"Model": name, "Fold": fold, "Sample": s,
                              "Patient": patient[s], "y": yy, "p": pp})

    # ---- 3. baselines ----
    # majority class of the training fold (constant probability)
    p_major = np.full(len(y_va), y_tr.mean())
    # single best gene by F statistic, chosen on the training fold only
    F, _ = f_classif(X_tr, y_tr)
    best_gene = X_tr.columns[int(np.nanargmax(F))]
    lr = LogisticRegression(class_weight="balanced", max_iter=500,
                            random_state=SEED)
    lr.fit(X_tr[[best_gene]], y_tr)
    p_gene = lr.predict_proba(X_va[[best_gene]])[:, 1]
    for name, p, extra in [("Majority class", p_major, ""),
                           ("Single gene LR", p_gene, best_gene)]:
        for s, yy, pp in zip(X_va.index, y_va, p):
            base_rows.append({"Model": name, "Fold": fold, "Sample": s,
                              "Patient": patient[s], "y": yy, "p": pp,
                              "Gene": extra})
    print(f"Fold {fold} done. Single best gene: {best_gene}")

pred = pd.DataFrame(pred_rows)
per_fold = pd.DataFrame(fold_rows)
genes = pd.DataFrame(gene_rows)
base = pd.DataFrame(base_rows)

pred.to_csv(OUT / "oof_predictions_31.csv", index=False)
per_fold.to_csv(OUT / "per_fold_metrics_31.csv", index=False)

# ============================================================
# 1. Gene stability table
# ============================================================
stab = (genes.groupby("Gene")
        .agg(Folds_in_top80=("Fold", "nunique"),
             Folds_in_top10=("In_top10", "sum"),
             Mean_logFC=("logFC", "mean"))
        .reset_index())
stab["Direction"] = np.where(stab["Mean_logFC"] > 0, "up in tumour", "down in tumour")
stab = stab.sort_values(["Folds_in_top10", "Folds_in_top80", "Mean_logFC"],
                        ascending=[False, False, True])
stab.to_csv(OUT / "gene_selection_stability.csv", index=False)

# ============================================================
# 2. Performance with uncertainty (and 4. Brier score)
# ============================================================
summary = []
for name, d in pred.groupby("Model", sort=False):
    f = per_fold[per_fold["Model"] == name]
    pooled = metrics(d["y"].to_numpy(), d["p"].to_numpy())
    row = {"Model": name,
           "Fold_mean_F1": f["F1"].mean(), "Fold_SD_F1": f["F1"].std(),
           "Fold_mean_AUC": f["AUC"].mean(), "Fold_SD_AUC": f["AUC"].std(),
           **{f"Pooled_{k}": v for k, v in pooled.items()}}
    row.update(bootstrap_ci(d))
    summary.append(row)

for name, d in base.groupby("Model", sort=False):
    pooled = metrics(d["y"].to_numpy(), d["p"].to_numpy())
    row = {"Model": f"Baseline: {name}",
           **{f"Pooled_{k}": v for k, v in pooled.items()}}
    row.update(bootstrap_ci(d))
    summary.append(row)

summary = pd.DataFrame(summary)
summary.to_csv(OUT / "performance_with_ci.csv", index=False)

# ============================================================
# 4. Calibration curves
# ============================================================
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
for name in CALIBRATION_MODELS:
    d = pred[pred["Model"] == name]
    frac, mean_p = calibration_curve(d["y"], d["p"], n_bins=5,
                                     strategy="quantile")
    brier = brier_score_loss(d["y"], d["p"])
    ax.plot(mean_p, frac, "o-", label=f"{name} (Brier {brier:.3f})")
ax.set_xlabel("Mean predicted probability (tumour)")
ax.set_ylabel("Observed fraction of tumours")
ax.set_title("Calibration (pooled out-of-fold predictions)")
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "Figure_calibration.png", dpi=300)
fig.savefig(OUT / "Figure_calibration.pdf")

# ============================================================
# Print a short report
# ============================================================
pd.set_option("display.width", 200)
cols = ["Model", "Pooled_AUC", "AUC_CI", "Pooled_F1", "F1_CI",
        "Pooled_BalAcc", "Pooled_Brier", "Pooled_TN", "Pooled_FP",
        "Pooled_FN", "Pooled_TP"]
print("\nPerformance (pooled out-of-fold, patient-level bootstrap 95% CI):")
print(summary[cols].round(3).to_string(index=False))
print("\nSingle best gene chosen in each fold:",
      base.loc[base.Model == "Single gene LR"].groupby("Fold")["Gene"].first().tolist())
print("\nGenes selected in the top 10 in at least 3 of 5 folds:")
print(stab[stab["Folds_in_top10"] >= 3].round(2).to_string(index=False))
print(f"\nGenes ever in a top-80 list: {len(stab)}; "
      f"in all 5 folds: {(stab.Folds_in_top80 == 5).sum()}")
print(f"\nSaved to: {OUT}")



