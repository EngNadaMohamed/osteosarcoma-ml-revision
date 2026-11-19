from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# ============================================================
# 1. Paths
# ============================================================

# Project root: Revision_Reviewer_final
PROJECT_DIR = Path(__file__).resolve().parents[2]

# All input data is inside the data folder
DATA_DIR = PROJECT_DIR / "data"
COUNTS_FILE = DATA_DIR / "filtered_counts_before_edgeR_revision.csv"
META_FILE = DATA_DIR / "meta_revision_with_patient_id.csv"

# Create grouped folds inside data, preserving previous results
OUTPUT_DIR = DATA_DIR / "revision_folds_grouped"

# ============================================================
# 2. Read the original data
# ============================================================

counts = pd.read_csv(COUNTS_FILE, index_col=0)
meta = pd.read_csv(META_FILE, index_col=0)

meta.index = meta.index.astype(str)
counts.columns = counts.columns.astype(str)

required_columns = {"source_name", "patient_id"}

if not required_columns.issubset(meta.columns):
    raise ValueError(
        "Metadata must contain source_name and patient_id."
    )

if meta["patient_id"].isna().any():
    raise ValueError("Some samples have missing patient IDs.")

if meta.index.duplicated().any():
    raise ValueError("Duplicate sample IDs in metadata.")

if counts.columns.duplicated().any():
    raise ValueError("Duplicate sample IDs in counts.")

# Keep only samples present in both files, in metadata order
if set(counts.columns) != set(meta.index):
    raise ValueError(
        "Counts and metadata contain different sample IDs. "
        "Check the files before proceeding."
    )

counts = counts.loc[:, meta.index]

assert list(counts.columns) == list(meta.index)

print("Total samples:", len(meta))
print("Unique patients:", meta["patient_id"].nunique())
print("Class distribution:")
print(meta["source_name"].value_counts())


# ============================================================
# 3. Create patient-grouped, stratified folds
# ============================================================

X = counts.T
y = meta["source_name"]
groups = meta["patient_id"]

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Remove no files: write to a separate, new output directory.
fold_summary = []

for fold_number, (train_idx, val_idx) in enumerate(
    splitter.split(X, y, groups=groups),
    start=1
):
    train_samples = meta.index[train_idx]
    val_samples = meta.index[val_idx]

    train_meta = meta.loc[train_samples].copy()
    val_meta = meta.loc[val_samples].copy()

    train_counts = counts.loc[:, train_samples]
    val_counts = counts.loc[:, val_samples]

    # Critical check: no patient may appear in both sets
    train_patients = set(train_meta["patient_id"])
    val_patients = set(val_meta["patient_id"])


    ##################################################################
    # Identify paired and tumor-only patients
    patient_class_counts = (
        meta.groupby("patient_id")["source_name"].nunique()
    )
    paired_patients = set(
        patient_class_counts[patient_class_counts == 2].index
    )
    tumor_only_patients = set(
        patient_class_counts[patient_class_counts == 1].index
    )
    # Count patient types in each fold
    train_paired = train_patients.intersection(paired_patients)
    val_paired = val_patients.intersection(paired_patients)
    train_tumor_only = train_patients.intersection(tumor_only_patients)
    val_tumor_only = val_patients.intersection(tumor_only_patients)
    print("Training paired patients:", len(train_paired))
    print("Validation paired patients:", len(val_paired))
    print("Training tumor-only patients:", len(train_tumor_only))
    print("Validation tumor-only patients:", len(val_tumor_only))
    #################################################################


    overlap = train_patients.intersection(val_patients)

    if overlap:
        raise RuntimeError(
            f"Patient overlap in Fold {fold_number}: {overlap}"
        )

    fold_dir = OUTPUT_DIR / f"fold_{fold_number}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_counts.to_csv(fold_dir / "training_counts.csv")
    train_meta.to_csv(fold_dir / "training_meta.csv")

    val_counts.to_csv(fold_dir / "validation_counts.csv")
    val_meta.to_csv(fold_dir / "validation_meta.csv")

    fold_summary.append({
        "Fold": fold_number,
        "Training samples": len(train_samples),
        "Validation samples": len(val_samples),
        "Training patients": len(train_patients),
        "Validation patients": len(val_patients),
        "Patient overlap": len(overlap),
        "Training classes": train_meta["source_name"].value_counts().to_dict(),
        "Validation classes": val_meta["source_name"].value_counts().to_dict(),
    })

    print(f"\nFold {fold_number}")
    print("Training samples:", len(train_samples))
    print("Validation samples:", len(val_samples))
    print("Training patients:", len(train_patients))
    print("Validation patients:", len(val_patients))
    print("Patient overlap:", overlap)
    print("Validation classes:")
    print(val_meta["source_name"].value_counts().to_dict())


# ============================================================
# 4. Save summary
# ============================================================

summary = pd.DataFrame(fold_summary)
summary.to_csv(OUTPUT_DIR / "grouped_folds_summary.csv", index=False)

print("\nAll grouped folds created.")
print("Output folder:", OUTPUT_DIR.resolve())
print("\nSummary:")
print(summary.to_string(index=False))

##########################################################################

# Check that every sample appears in validation exactly once
validation_samples = []

for fold_number in range(1, 6):
    fold_dir = OUTPUT_DIR / f"fold_{fold_number}"

    val_meta = pd.read_csv(
        fold_dir / "validation_meta.csv",
        index_col=0
    )

    validation_samples.extend(val_meta.index.astype(str).tolist())

print("\nTotal validation entries:", len(validation_samples))
print("Unique validation samples:", len(set(validation_samples)))
print("Duplicated validation samples:",
      pd.Series(validation_samples).duplicated().sum())

expected_samples = set(meta.index.astype(str))
actual_samples = set(validation_samples)

print("Missing samples:", sorted(expected_samples - actual_samples))
print("Unexpected samples:", sorted(actual_samples - expected_samples))