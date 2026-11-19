from pathlib import Path

import numpy as np
import pandas as pd

from train_model_clean_id import evaluate_all_combinations

# ============================================================
# Configuration
# ============================================================

# Main project directory: Revision_Reviewer_final
PROJECT_DIR = Path(__file__).resolve().parents[2]

# Data directories
FOLD_ROOT = PROJECT_DIR / "data" / "revision_folds_grouped"
DEG_ROOT = PROJECT_DIR / "data" / "Edger_result_grouped_6"

print("PROJECT_DIR:", PROJECT_DIR)
print("FOLD_ROOT:", FOLD_ROOT)
print("FOLD_ROOT exists:", FOLD_ROOT.exists())
print("DEG_ROOT:", DEG_ROOT)
print("DEG_ROOT exists:", DEG_ROOT.exists())

N_FOLDS = 5

# ============================================================
# Load one fold
# ============================================================

def load_fold_data_id(fold_number):

    # 1. Locate the fold directory.
    fold_name = f"fold_{fold_number}".lower()

    fold_dirs = [
        path for path in FOLD_ROOT.iterdir()
        if path.is_dir()
        and path.name.lower() == fold_name
    ]

    if len(fold_dirs) != 1:
        raise FileNotFoundError(
            f"Could not uniquely locate Fold {fold_number} "
            f"inside {FOLD_ROOT}"
        )

    fold_dir = fold_dirs[0]

    # 2. Load training and validation data.
    train_counts = pd.read_csv(
        fold_dir / "training_counts.csv",
        index_col=0
    )

    train_meta = pd.read_csv(
        fold_dir / "training_meta.csv",
        index_col=0
    )

    validation_counts = pd.read_csv(
        fold_dir / "validation_counts.csv",
        index_col=0
    )

    validation_meta = pd.read_csv(
        fold_dir / "validation_meta.csv",
        index_col=0
    )

    # 3. Normalize identifiers to strings.
    for df in (
        train_counts,
        validation_counts,
        train_meta,
        validation_meta,
    ):
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)

    # 4. Check sample and metadata alignment.
    if not set(train_counts.columns).issubset(train_meta.index):
        raise ValueError(
            f"Training samples and metadata do not match "
            f"in Fold {fold_number}"
        )

    if not set(validation_counts.columns).issubset(
        validation_meta.index
    ):
        raise ValueError(
            f"Validation samples and metadata do not match "
            f"in Fold {fold_number}"
        )

    train_meta = train_meta.loc[train_counts.columns]
    validation_meta = validation_meta.loc[
        validation_counts.columns
    ]

    # 5. Locate the DEG file for this fold.
    deg_file = DEG_ROOT / f"Fold_{fold_number}_DEGs.csv"

    if not deg_file.exists():
        matches = [
            path for path in DEG_ROOT.glob("*.csv")
            if path.name.lower() == deg_file.name.lower()
        ]

        if len(matches) != 1:
            raise FileNotFoundError(
                f"DEG file not found: {deg_file}"
            )

        deg_file = matches[0]

    # 6. Locate and validate the training sample list.
    train_list_file = (
        DEG_ROOT / f"Fold_{fold_number}_train_samples.txt"
    )

    if not train_list_file.exists():
        raise FileNotFoundError(
            f"Training sample list not found: {train_list_file}"
        )

    train_list = train_list_file.read_text().split()

    samples_match = (
        set(train_list) == set(train_counts.columns)
    )

    print(f"\n========== Fold {fold_number} PATH CHECK ==========")
    print("FOLD_ROOT:", FOLD_ROOT.resolve())
    print("Fold directory:", fold_dir.resolve())
    print("DEG_ROOT:", DEG_ROOT.resolve())
    print("DEG file actually used:", deg_file.resolve())
    print("DEG file exists:", deg_file.exists())
    print("Training sample list:", train_list_file.resolve())
    print("Training sample list exists:", train_list_file.exists())
    print("Training sample count:", len(train_counts.columns))
    print("Saved training sample count:", len(train_list))
    print("Training sample IDs match:", samples_match)

    if not samples_match:
        missing_from_list = (
            set(train_counts.columns) - set(train_list)
        )
        extra_in_list = (
            set(train_list) - set(train_counts.columns)
        )

        raise ValueError(
            f"Fold {fold_number} training sample mismatch.\n"
            f"Missing from saved list: {sorted(missing_from_list)}\n"
            f"Extra in saved list: {sorted(extra_in_list)}"
        )

    # 7. Load and inspect the DEG file.
    deg_table = pd.read_csv(deg_file, index_col=0)

    required_columns = {"FDR", "logFC", "PValue"}

    if not required_columns.issubset(deg_table.columns):
        raise ValueError(
            f"Missing required columns in {deg_file}. "
            f"Found: {list(deg_table.columns)}"
        )

    print(f"\nChecking Fold {fold_number}: {deg_file.name}")
    print("Total genes in DEG file:", len(deg_table))

    valid_filter = (
        (deg_table["FDR"] < 0.05)
        & (deg_table["logFC"].abs() > 6)
    )

    print("Genes meeting both filters:", int(valid_filter.sum()))
    print("Any genes violating filters:", bool((~valid_filter).any()))

    # Stop rather than silently using genes that violate the filters.
    if not valid_filter.all():
        raise ValueError(
            f"Fold {fold_number}: DEG file contains genes "
            "that violate FDR < 0.05 or |logFC| > 6."
        )

    # 8. Select up to 80 genes by FDR, then PValue.
    deg_table = deg_table.sort_values(["FDR", "PValue"])
    deg_genes = deg_table.index.astype(str).tolist()[:80]

    if not deg_genes:
        raise ValueError(f"No DEGs found in {deg_file}")

    # 9. Keep genes available in both count matrices.
    common_genes = [
        gene for gene in deg_genes
        if gene in train_counts.index
        and gene in validation_counts.index
    ]

    if not common_genes:
        raise ValueError(
            f"No matching DEG genes found in Fold {fold_number}"
        )

    # 10. Convert gene-by-sample to sample-by-gene, then log2(CPM + 1).
    X_train = np.log2(train_counts.loc[common_genes].T + 1)
    X_validation = np.log2(validation_counts.loc[common_genes].T + 1)

    # 11. Remove constant genes based on training data only.
    variable_genes = X_train.columns[
        X_train.nunique() > 1
    ]

    X_train = X_train.loc[:, variable_genes]
    X_validation = X_validation.loc[:, variable_genes]

    # 12. Identify the label column.
    if "source_name" in train_meta.columns:
        label_column = "source_name"
    elif "label" in train_meta.columns:
        label_column = "label"
    else:
        raise ValueError(
            "Expected 'source_name' or 'label' in training metadata."
        )

    if label_column not in validation_meta.columns:
        raise ValueError(
            f"Missing '{label_column}' in validation metadata."
        )

    # 13. Encode labels: 0 = Normal, 1 = Osteosarcoma.
    def encode_labels(series):
        numeric = pd.to_numeric(series, errors="coerce")

        if (
            numeric.notna().all()
            and set(numeric.unique()).issubset({0, 1})
        ):
            return numeric.astype(int)

        encoded = []

        for value in series.astype(str).str.lower().str.strip():
            if (
                "osteosarcoma" in value
                or "cancer" in value
                or "tumor" in value
            ):
                encoded.append(1)
            elif "normal" in value:
                encoded.append(0)
            else:
                raise ValueError(f"Unrecognized label: {value}")

        return pd.Series(
            encoded, index=series.index, dtype=int
        )

    y_train = encode_labels(
        train_meta.loc[X_train.index, label_column]
    )

    y_validation = encode_labels(
        validation_meta.loc[X_validation.index, label_column]
    )

    if X_train.shape[1] < 10:
        raise ValueError(
            f"Fold {fold_number} has fewer than 10 variable genes."
        )

    print(f"\nFold {fold_number} data loaded successfully")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_validation)}")
    print(f"Selected genes available: {len(common_genes)}")
    print(f"Variable genes: {X_train.shape[1]}")

    return X_train, y_train, X_validation, y_validation