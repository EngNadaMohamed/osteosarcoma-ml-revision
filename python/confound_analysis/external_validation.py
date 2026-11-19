"""External validation: train on GSE218035 (tumour vs adjacent soft tissue),
test on GSE99671 fresh-frozen (tumour vs paired NORMAL BONE).

Nothing from GSE99671 is used for gene selection or model fitting.

Feature sets (all chosen from GSE218035 only):
  paper_top10   : the 10 genes in Table 6 of the manuscript (muscle genes)
  muscle_score  : mean z-score of 12 canonical skeletal-muscle markers
  consensus_all : genes that were edgeR DEGs in >=3 of 5 training folds, top 10 by F
  consensus_up  : same, but only genes UP in tumour, top 10 by F

Scaling:
  train_scaler  : StandardScaler fitted on GSE218035 only
  per_dataset   : each gene z-scored within its own dataset (unsupervised)

Labels for GSE99671 come from the series-matrix characteristics ("tumor: TUMOR/NORMAL").
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.svm import SVC


# ============================================================
# Paths (relative to the project root)
# ============================================================
DATA = Path(__file__).resolve().parents[2] / "data"
RAW = DATA / "raw_counts_data.csv"                       # raw counts, GSE218035
META = DATA / "meta_revision_with_patient_id.csv"        # source_name + patient_id
DEGDIR = DATA / "Edger_result_grouped_6"                 # per-fold edgeR results
EXT_COUNTS = DATA / "GSE99671" / "GSE99671_Osteosarcoma_filecounts.txt"
EXT_SERIES = DATA / "GSE99671" / "GSE99671_series_matrix.txt"
OUT = DATA / "confound_results" / "external_validation"

OUT.mkdir(parents=True, exist_ok=True)
# Patients to leave out of GSE99671 (sensitivity run): e.g. {"OSVN004"}
EXCLUDE = set()

MARKERS = ["MYH1", "MYH2", "MYH7", "DES", "MYL1", "MYLPF", "TNNT3",
           "TNNI2", "MYOD1", "MYOG", "MB", "TTN"]
PAPER10 = ["AMPD1", "LDB3", "TNNC2", "MYOZ1", "COX6A2", "CASQ1", "PYGM",
           "CKM", "ASB10", "ACTA1"]


def logcpm(counts):
    return np.log2(counts / counts.sum(axis=0) * 1e6 + 1)


# ---------- training data: GSE218035 ----------
raw = pd.read_csv(RAW, index_col=0)
meta = pd.read_csv(META, index_col=0)
raw = raw[meta.index]
y_tr = meta["source_name"].str.startswith("Osteo").astype(int)

# ---------- external data: GSE99671 fresh-frozen ----------
ext = pd.read_csv(EXT_COUNTS, sep="\t", index_col=0).set_index("symbol").drop(columns="Chr")
lines = EXT_SERIES.read_text().splitlines()
titles = [l for l in lines if l.startswith("!Sample_title")][0].split("\t")[1:37]
chars = [l for l in lines if l.startswith("!Sample_characteristics_ch1")][0].split("\t")[1:37]
y_ext = pd.Series([1 if "TUMOR" in c else 0 for c in chars], index=ext.columns)
pid_ext = pd.Series([c[:7] for c in ext.columns], index=ext.columns)
keep_ext = ~pid_ext.isin(EXCLUDE)
ext, y_ext, pid_ext = ext.loc[:, keep_ext], y_ext[keep_ext], pid_ext[keep_ext]

common = raw.index.intersection(ext.index)
L_tr = logcpm(raw).loc[common]
L_ex = logcpm(ext).loc[common]

# ---------- feature sets (GSE218035 only) ----------
deg_count, deg_dir = {}, {}
for f in range(1, 6):
    d = pd.read_csv(DEGDIR / f"Fold_{f}_DEGs.csv", index_col=0)
    for g, lfc in d["logFC"].items():
        deg_count[g] = deg_count.get(g, 0) + 1
        deg_dir.setdefault(g, []).append(np.sign(lfc))
consensus = [g for g, n in deg_count.items() if n >= 3 and g in common]
cons_up = [g for g in consensus if np.mean(deg_dir[g]) > 0]


def top_by_f(genes, k=10):
    F, _ = f_classif(L_tr.loc[genes].T, y_tr)
    return list(pd.Series(F, index=genes).sort_values(ascending=False).index[:k])


sets = {
    "paper_top10": [g for g in PAPER10 if g in common],
    "consensus_all": top_by_f(consensus),
    "consensus_up": top_by_f(cons_up),
}
markers = [g for g in MARKERS if g in common]


def make_xy(genes, scaling):
    A, B = L_tr.loc[genes].T, L_ex.loc[genes].T
    if scaling == "train_scaler":
        mu, sd = A.mean(), A.std().replace(0, 1)
        return (A - mu) / sd, (B - mu) / sd
    return ((A - A.mean()) / A.std().replace(0, 1),
            (B - B.mean()) / B.std().replace(0, 1))


def muscle_xy(scaling):
    A, B = make_xy(markers, scaling)
    return A.mean(axis=1).to_frame("muscle"), B.mean(axis=1).to_frame("muscle")


def models():
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    base = {
        "SVC": SVC(class_weight="balanced", probability=True, random_state=42),
        "ET": ExtraTreesClassifier(class_weight="balanced", random_state=42),
        "LR": lr, "RF": rf,
        "GBM": GradientBoostingClassifier(random_state=42),
    }
    out = dict(base)
    out["LR + RF"] = VotingClassifier([("lr", lr), ("rf", rf)], voting="soft")
    out["ALL5"] = VotingClassifier(list(base.items()), voting="soft")
    return out


rows, scores = [], []
for scaling in ["train_scaler", "per_dataset"]:
    feats = {k: make_xy(v, scaling) for k, v in sets.items()}
    feats["muscle_score"] = muscle_xy(scaling)
    for fname, (Xa, Xb) in feats.items():
        for mname, m in models().items():
            m.fit(Xa, y_tr)
            p = m.predict_proba(Xb)[:, list(m.classes_).index(1)]
            yp = (p >= 0.5).astype(int)
            # paired concordance: tumour scored above its own normal bone
            s = pd.DataFrame({"p": p, "y": y_ext.values, "pid": pid_ext.values})
            piv = s.pivot_table(index="pid", columns="y", values="p")
            piv = piv.dropna()
            conc = float((piv[1] > piv[0]).mean())
            rows.append({
                "Scaling": scaling, "Features": fname, "Model": mname,
                "AUC": roc_auc_score(y_ext, p),
                "BalAcc": balanced_accuracy_score(y_ext, yp),
                "Sensitivity": float(((yp == 1) & (y_ext == 1)).sum() / (y_ext == 1).sum()),
                "Specificity": float(((yp == 0) & (y_ext == 0)).sum() / (y_ext == 0).sum()),
                "Paired_concordance": conc,
            })

res = pd.DataFrame(rows)
res.to_csv(OUT / "external_validation_results.csv", index=False)
pd.Series({k: ", ".join(v) for k, v in sets.items()} |
          {"muscle_score": ", ".join(markers)}).to_csv(OUT / "external_feature_sets.csv",
                                                       header=["genes"])

pd.set_option("display.width", 220)
print("genes in common:", len(common), "| external samples:", len(y_ext),
      "| excluded patients:", sorted(EXCLUDE) or "none")
for k, v in sets.items():
    print(f"{k}: {v}")
print()
print(res.round(3).to_string(index=False))