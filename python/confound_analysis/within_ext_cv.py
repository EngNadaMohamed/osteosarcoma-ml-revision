import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
DATA = Path(__file__).resolve().parents[2] / "data"
EXT_COUNTS = DATA / "GSE99671" / "GSE99671_Osteosarcoma_filecounts.txt"
EXT_SERIES = DATA / "GSE99671" / "GSE99671_series_matrix.txt"
C, S = EXT_COUNTS, EXT_SERIES
ext=pd.read_csv(C,sep='\t',index_col=0).set_index('symbol').drop(columns='Chr')
lines=S.read_text().splitlines()
ch=[l for l in lines if l.startswith('!Sample_characteristics_ch1')][0].split('\t')[1:37]
y=np.array([1 if 'TUMOR' in c else 0 for c in ch]); g=np.array([c[:7] for c in ext.columns])
L=np.log2(ext/ext.sum()*1e6+1); L=L[(ext>=10).sum(1)>=6].T
P=np.zeros(len(y))
for tr,va in StratifiedGroupKFold(5,shuffle=True,random_state=42).split(L,y,g):
    m=Pipeline([('s',StandardScaler()),('k',SelectKBest(f_classif,k=10)),('c',LogisticRegression(class_weight='balanced',max_iter=1000))]).fit(L.iloc[tr],y[tr])
    P[va]=m.predict_proba(L.iloc[va])[:,1]
print('Within-GSE99671 grouped 5-fold CV, LR, pooled AUC = %.3f'%roc_auc_score(y,P))