"""Muscle-confound experiment on the grouped folds.

Uses the per-fold edgeR DEGs computed on training samples only, and the
same patient-grouped training/validation split as the main analysis.

Scenarios:
  A  baseline        : top-80 fold DEGs, log2(CPM+1)        (main pipeline)
  B  muscle-only     : single feature = canonical muscle marker score
  C  no-muscle DEGs  : fold DEGs minus genes correlated with the muscle score
                       (|r| > R_CUT, correlation computed on training samples only)
  D  muscle-adjusted : every gene residualised on the muscle score (fit on
                       training only), then SelectKBest from all genes
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# Paths (relative to the project root)
# ============================================================
DATA = Path(__file__).resolve().parents[2] / "data"
RAW = DATA / "raw_counts_data.csv"                       # raw counts, GSE218035
META = DATA / "meta_revision_with_patient_id.csv"        # source_name + patient_id
DEGDIR = DATA / "Edger_result_grouped_6"                 # per-fold edgeR results
OUT = DATA / "confound_results" / "muscle_experiment"

OUT.mkdir(parents=True, exist_ok=True)
R_CUT = 0.6
K = 10
MARKERS = ["MYH1", "MYH2", "MYH7", "DES", "MYL1", "MYLPF", "TNNT3",
           "TNNI2", "MYOD1", "MYOG", "MB", "TTN"]

raw = pd.read_csv(RAW, index_col=0)
meta = pd.read_csv(META, index_col=0)
raw = raw[meta.index]
logcpm = np.log2(raw / raw.sum(axis=0) * 1e6 + 1)          # per-sample, no leakage
y_all = meta["source_name"].str.startswith("Osteo").astype(int)


def pool():
    return {
        "svc": SVC(class_weight="balanced", probability=True, random_state=42),
        "et": ExtraTreesClassifier(class_weight="balanced", random_state=42),
        "lr": LogisticRegression(class_weight="balanced", max_iter=500, random_state=42),
        "rf": RandomForestClassifier(class_weight="balanced", random_state=42),
        "gbm": GradientBoostingClassifier(random_state=42),
    }


def build(combo, k):
    def pipe(m):
        steps = [("scaler", StandardScaler())]
        if k is not None:
            steps.append(("select", SelectKBest(f_classif, k=k)))
        steps.append(("clf", clone(m)))
        return Pipeline(steps)
    if len(combo) == 1:
        return pipe(combo[0][1])
    return VotingClassifier([(n, pipe(m)) for n, m in combo], voting="soft")


def muscle_score(train_ids, ids):
    """z-score markers using training mean/sd only, then average."""
    g = [m for m in MARKERS if m in logcpm.index]
    mu = logcpm.loc[g, train_ids].mean(axis=1)
    sd = logcpm.loc[g, train_ids].std(axis=1).replace(0, 1)
    return logcpm.loc[g, ids].sub(mu, axis=0).div(sd, axis=0).mean(axis=0)


rows, preds, removed_log = [], [], []
for fold in range(1, 6):
    train_ids = (DEGDIR / f"Fold_{fold}_train_samples.txt").read_text().split()
    val_ids = [s for s in meta.index if s not in train_ids]
    ytr, yva = y_all[train_ids], y_all[val_ids]

    deg = pd.read_csv(DEGDIR / f"Fold_{fold}_DEGs.csv", index_col=0)
    deg = deg.sort_values(["FDR", "PValue"])
    ms_tr = muscle_score(train_ids, train_ids)
    ms_va = muscle_score(train_ids, val_ids)

    # correlation of every DEG with muscle score, training samples only
    corr = logcpm.loc[deg.index, train_ids].T.corrwith(ms_tr)
    keep = corr.abs() <= R_CUT
    removed_log.append({"Fold": fold, "DEGs": len(deg),
                        "Muscle_correlated_removed": int((~keep).sum()),
                        "DEGs_remaining": int(keep.sum())})

    scen = {}
    g80 = deg.index[:80]
    scen["A_baseline"] = (logcpm.loc[g80, train_ids].T, logcpm.loc[g80, val_ids].T, K)
    scen["B_muscle_only"] = (ms_tr.to_frame("muscle"), ms_va.to_frame("muscle"), None)
    gC = deg.index[keep.values][:80]
    scen["C_no_muscle_DEGs"] = (logcpm.loc[gC, train_ids].T, logcpm.loc[gC, val_ids].T,
                                min(K, len(gC)))
    # D: residualise every gene on muscle score (slope/intercept from training)
    Xtr = logcpm.loc[:, train_ids].T
    Xva = logcpm.loc[:, val_ids].T
    xm = ms_tr - ms_tr.mean()
    beta = (Xtr.sub(Xtr.mean()).mul(xm, axis=0).sum() / (xm ** 2).sum())
    alpha = Xtr.mean() - beta * ms_tr.mean()
    Rtr = Xtr - np.outer(ms_tr, beta) - alpha.values
    Rva = Xva - np.outer(ms_va, beta) - alpha.values
    var = Rtr.var() > 1e-8
    scen["D_muscle_adjusted"] = (Rtr.loc[:, var], Rva.loc[:, var], K)

    for sname, (Xa, Xb, k) in scen.items():
        combos = [c for s in range(1, 6) for c in combinations(pool().items(), s)]
        if sname == "B_muscle_only":
            combos = [c for c in combos if len(c) == 1]
        for combo in combos:
            name = " + ".join(n.upper() for n, _ in combo)
            mdl = build(combo, k).fit(Xa, ytr)
            pr = mdl.predict_proba(Xb)[:, list(mdl.classes_).index(1)]
            yp = (pr >= 0.5).astype(int)
            rows.append({"Scenario": sname, "Fold": fold, "Model": name,
                         "F1": f1_score(yva, yp, zero_division=0),
                         "AUC": roc_auc_score(yva, pr),
                         "BalAcc": balanced_accuracy_score(yva, yp)})
            for sid, a, p_, q in zip(val_ids, yva, yp, pr):
                preds.append({"Scenario": sname, "Fold": fold, "Model": name,
                              "Sample_ID": sid, "Actual": a, "Pred": p_, "Prob": q})

res = pd.DataFrame(rows)
pr = pd.DataFrame(preds)
res.to_csv(OUT / "per_fold.csv", index=False)
pr.to_csv(OUT / "predictions.csv", index=False)
pd.DataFrame(removed_log).to_csv(OUT / "muscle_genes_removed.csv", index=False)

summ = (res.groupby(["Scenario", "Model"])
        .agg(Mean_F1=("F1", "mean"), Std_F1=("F1", "std"),
             Mean_AUC=("AUC", "mean"), Mean_BalAcc=("BalAcc", "mean"))
        .reset_index())
pooled = (pr.groupby(["Scenario", "Model"])
          .apply(lambda d: pd.Series({"Pooled_AUC": roc_auc_score(d.Actual, d.Prob)}),
                 include_groups=False).reset_index())
summ = summ.merge(pooled, on=["Scenario", "Model"])
summ.to_csv(OUT / "summary.csv", index=False)

pd.set_option("display.width", 200)
print(pd.DataFrame(removed_log).to_string(index=False))
print()
print(summ.groupby("Scenario")[["Mean_F1", "Mean_BalAcc", "Mean_AUC", "Pooled_AUC"]]
      .agg(["min", "median", "max"]).round(3).to_string())
print()
singles = summ[~summ.Model.str.contains(r"\+")]
print(singles.round(3).to_string(index=False))