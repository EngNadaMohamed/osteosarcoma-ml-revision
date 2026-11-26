"""Add patient IDs to the GSE218035 sample metadata.

Input : data/meta_revision.csv   (one row per GSM sample, column `source_name`
                                   = tumour / normal label from GEO)
Output: data/meta_revision_with_patient_id.csv

Patient IDs are taken from the `individual` characteristic of each sample in
the GEO annotations of GSE218035. 36 samples, 23 patients, 13 of them with a
paired tumour and adjacent-normal sample. These IDs are used to group samples
by patient in all cross-validation splits.
"""
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
meta_path = DATA / "meta_revision.csv"

# Patient IDs from the `individual` field of the GSE218035 sample annotations
patient_ids = {
    "GSM6733214": 103,
    "GSM6733215": 105,
    "GSM6733216": 109,
    "GSM6733217": 110,
    "GSM6733218": 111,
    "GSM6733219": 112,
    "GSM6733220": 122,
    "GSM6733221": 129,
    "GSM6733222": 131,
    "GSM6733223": 141,
    "GSM6733224": 142,
    "GSM6733225": 148,
    "GSM6733226": 150,
    "GSM6733227": 151,
    "GSM6733228": 155,
    "GSM6733229": 157,
    "GSM6733230": 170,
    "GSM6733231": 88,
    "GSM6733232": 89,
    "GSM6733233": 90,
    "GSM6733234": 93,
    "GSM6733235": 98,
    "GSM6733236": 99,
    "GSM6733237": 103,
    "GSM6733238": 105,
    "GSM6733239": 109,
    "GSM6733240": 112,
    "GSM6733241": 122,
    "GSM6733242": 129,
    "GSM6733243": 131,
    "GSM6733244": 142,
    "GSM6733245": 150,
    "GSM6733246": 151,
    "GSM6733247": 155,
    "GSM6733248": 157,
    "GSM6733249": 99,
}

meta = pd.read_csv(meta_path, index_col=0)
meta.index = meta.index.astype(str)

missing = set(meta.index) - set(patient_ids)
extra = set(patient_ids) - set(meta.index)

if missing or extra:
    print("Missing patient IDs:", missing)
    print("Unexpected sample IDs:", extra)
    raise ValueError("Sample IDs do not match the verified mapping.")

meta["patient_id"] = meta.index.map(patient_ids)

print("Total samples:", len(meta))
print("Unique patients:", meta["patient_id"].nunique())
print("\nSamples per patient:")
print(meta.groupby("patient_id").size().to_string())

print("\nSample metadata:")
print(meta.to_string())

# Save a new file; preserve the original metadata
output_path = DATA / "meta_revision_with_patient_id.csv"
meta.to_csv(output_path)

print("\nSaved to:", output_path.resolve())
