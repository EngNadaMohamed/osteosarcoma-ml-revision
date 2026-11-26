"""Permutation-test figure (null distribution of best-of-31 AUC, and observed vs
permuted scores for each of the 31 configurations)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
from pathlib import Path
DATA = Path(__file__).resolve().parents[2] / "data"
PERM = DATA / "perm_out_grouped_within_patient"          # output of permutation_test_clean.py
OUT = DATA / "figures"
OUT.mkdir(parents=True, exist_ok=True)

null=pd.read_csv(PERM / "perm_null_best_AUC.csv").iloc[:,1].values
pm=pd.read_csv(PERM / "perm_per_model_pvalues.csv")
NULLC,REALC="#9AA5B1","#C2410C"
fig,ax=plt.subplots(1,2,figsize=(10,4.2),gridspec_kw={"width_ratios":[1,1.35]})
a=ax[0]
a.hist(null,bins=np.arange(0.3,1.025,0.025),color=NULLC,edgecolor="white")
a.axvline(1.0,color=REALC,lw=2); a.axvline(np.quantile(null,.95),color="#334155",ls="--",lw=1)
a.text(0.99,a.get_ylim()[1]*0.95,"Observed\nAUC = 1.00",ha="right",va="top",color=REALC)
a.text(np.quantile(null,.95)-0.01,a.get_ylim()[1]*0.72,"95th pct\n%.2f"%np.quantile(null,.95),ha="right",color="#334155")
a.set_xlabel("Best-of-31 mean AUC (permuted labels)"); a.set_ylabel("Permutations (n = 100)")
a.set_title("A  Null distribution, best of 31 configurations\np = 0.0099",loc="left",fontsize=10)
b=ax[1]
for i,(m,lab) in enumerate([("AUC","AUC"),("BalAcc","Balanced accuracy"),("F1","F1")]):
    d=pm[pm.Metric==m]; y=np.full(len(d),i)+np.random.default_rng(0).uniform(-.18,.18,len(d))
    b.errorbar(d.Null_mean,y,xerr=d.Null_sd,fmt="o",ms=3,color=NULLC,elinewidth=.6,alpha=.8,label="Permuted (mean ± SD)" if i==0 else None)
    b.scatter(d.Real,y,s=14,color=REALC,zorder=3,label="Observed" if i==0 else None)
b.set_yticks([0,1,2]); b.set_yticklabels(["AUC","Balanced\naccuracy","F1"]); b.invert_yaxis()
b.set_xlim(0.3,1.05); b.set_xlabel("Score (mean over 5 patient-grouped folds)")
b.axvline(0.5,color="#CBD5E1",lw=.8,zorder=0)
b.set_title("B  Each of the 31 configurations: observed vs permuted\nall p = 0.0099",loc="left",fontsize=10)
b.legend(frameon=False,loc="lower left",fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "Figure_permutation_test.png", dpi=300)
fig.savefig(OUT / "Figure_permutation_test.pdf")
print("Saved:", OUT / "Figure_permutation_test.pdf")
