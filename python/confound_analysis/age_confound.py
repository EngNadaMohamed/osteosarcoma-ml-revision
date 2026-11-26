import sys, pandas as pd, numpy as np
from scipy import stats
from pathlib import Path

# ============================================================
# Paths (edit DATA if your project folder moves)
# ============================================================
DATA = Path(__file__).resolve().parents[2] / "data"
RAW = DATA / "raw_counts_data.csv"                       # raw counts, GSE218035
META = DATA / "meta_revision_with_patient_id.csv"        # source_name + patient_id
DEGDIR = DATA / "Edger_result_grouped_6"                 # per-fold edgeR results
SRA = DATA / "metadata.csv"                              # SRA run table (AGE, sex, Site)
OUT = DATA / "confound_results"
OUT.mkdir(parents=True, exist_ok=True)

s=pd.read_csv(SRA).set_index('Sample Name')
m=pd.read_csv(META,index_col=0).join(s[['AGE','sex','Site','Individual']])
assert (m.patient_id.astype(str)==m.Individual.astype(str)).all()
m['tumour']=m.source_name.str.startswith('Osteo')
tu,no=m[m.tumour],m[~m.tumour]
print('Age tumour median [IQR]:',tu.AGE.median(),tu.AGE.quantile([.25,.75]).tolist(),'| normal:',no.AGE.median(),no.AGE.quantile([.25,.75]).tolist())
print('Mann-Whitney age (sample level, descriptive only):', stats.mannwhitneyu(tu.AGE,no.AGE))
for c in ['sex','Site']:
    ct=pd.crosstab(m.tumour,m[c]); print(ct); print('Fisher',c,'(sample level, descriptive only)',stats.fisher_exact(ct))
pairs=m.groupby('patient_id').source_name.nunique(); pp=pairs[pairs==2].index
print('paired patients:',len(pp),'| identical age/sex/site within pair:',
      all(m[m.patient_id==p][['AGE','sex','Site']].nunique().max()==1 for p in pp))
# within the 13 paired patients only (perfectly matched design)
mp=m[m.patient_id.isin(pp)]
print('Mann-Whitney age, paired subset:', stats.mannwhitneyu(mp.AGE[mp.tumour],mp.AGE[~mp.tumour]))
print('Tumour-only patients ages:', sorted(m[~m.patient_id.isin(pp)].AGE))
r=pd.read_csv(RAW,index_col=0)[m.index]; lc=np.log2(r/r.sum()*1e6+1)
g=['MYH1','MYH2','MYH7','DES','MYL1','MYLPF','TNNT3','TNNI2','MYOD1','MYOG','MB','TTN']
z=lc.loc[g].sub(lc.loc[g].mean(1),axis=0).div(lc.loc[g].std(1),axis=0); m['muscle']=z.mean()
for lab,d in [('tumour',tu.index),('normal',no.index)]:
    print(lab,'Spearman age vs muscle score:',stats.spearmanr(m.loc[d,'AGE'],m.loc[d,'muscle']))
print(m.loc[['GSM6733226','GSM6733234','GSM6733244','GSM6733240'],['source_name','patient_id','AGE','sex','Site','muscle']].round(2))
m[['source_name','patient_id','AGE','sex','Site','muscle']].to_csv(OUT / 'sample_covariates.csv')
print('Saved:', OUT / 'sample_covariates.csv')

# ------------------------------------------------------------
# Patient-level comparison (one row per patient).
# All normal samples come from paired patients, so class differences in
# clinical variables can only come from the tumour-only patients.
# ------------------------------------------------------------
n_per_patient = m.groupby('patient_id').size()
pt = m.groupby('patient_id')[['AGE', 'sex', 'Site']].first()
pt['paired'] = n_per_patient == 2
print('\nPatient level: paired =', pt.paired.sum(), '| tumour-only =', (~pt.paired).sum())
print(pt.groupby('paired').AGE.describe())
print('Mann-Whitney age (tumour-only vs paired patients):',
      stats.mannwhitneyu(pt.AGE[~pt.paired], pt.AGE[pt.paired]))
for c in ['sex', 'Site']:
    ct = pd.crosstab(pt.paired, pt[c]); print(ct)
    print('Fisher', c, '(patient level):', stats.fisher_exact(ct))


