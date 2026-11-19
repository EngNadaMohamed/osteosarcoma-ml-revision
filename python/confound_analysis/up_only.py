import numpy as np, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

# ============================================================
# Paths (relative to the project root)
# ============================================================
DATA = Path(__file__).resolve().parents[2] / "data"
RAW = DATA / "raw_counts_data.csv"                       # raw counts, GSE218035
META = DATA / "meta_revision_with_patient_id.csv"        # source_name + patient_id
DEGDIR = DATA / "Edger_result_grouped_6"                 # per-fold edgeR results

raw=pd.read_csv(RAW,index_col=0); meta=pd.read_csv(META,index_col=0); raw=raw[meta.index]
lc=np.log2(raw/raw.sum(axis=0)*1e6+1); y=meta.source_name.str.startswith('Osteo').astype(int)
models={'LR':LogisticRegression(class_weight='balanced',max_iter=500),'SVC':SVC(class_weight='balanced',probability=True,random_state=42),'RF':RandomForestClassifier(class_weight='balanced',random_state=42)}
P=[]; sel={}
for f in range(1,6):
    tr=(DEGDIR/f'Fold_{f}_train_samples.txt').read_text().split(); va=[s for s in meta.index if s not in tr]
    d=pd.read_csv(DEGDIR/f'Fold_{f}_DEGs.csv',index_col=0).sort_values(['FDR','PValue'])
    up=d[d.logFC>0].index[:80]
    sk=SelectKBest(f_classif,k=10).fit(lc.loc[up,tr].T,y[tr])
    for g in up[sk.get_support()]: sel[g]=sel.get(g,0)+1
    for n,m in models.items():
        p=Pipeline([('s',StandardScaler()),('k',SelectKBest(f_classif,k=10)),('c',m)]).fit(lc.loc[up,tr].T,y[tr])
        pr=p.predict_proba(lc.loc[up,va].T)[:,1]
        for s,a,q in zip(va,y[va],pr): P.append((n,f,s,a,q))
P=pd.DataFrame(P,columns=['Model','Fold','S','A','Q'])
for n,d in P.groupby('Model'):
    fa=d.groupby('Fold').apply(lambda x: roc_auc_score(x.A,x.Q),include_groups=False).mean()
    yp=(d.Q>=.5).astype(int)
    print(n,'pooledAUC=%.3f foldAUC=%.3f F1=%.3f BalAcc=%.3f'%(roc_auc_score(d.A,d.Q),fa,f1_score(d.A,yp),balanced_accuracy_score(d.A,yp)))
    print('   wrong:',list(d[(d.Q>=.5).astype(int)!=d.A].S))
print('genes selected (folds):',sorted(sel.items(),key=lambda x:-x[1])[:15])