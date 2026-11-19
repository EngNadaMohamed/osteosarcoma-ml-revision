from pathlib import Path

import numpy as np
import pandas as pd


from train_model_nested import (
    evaluate_inner_fold,
    select_best_combination,
    evaluate_outer_fold,
)

# ============================================================
# CONFIGURATION
# ============================================================

# Project root: Revision_Reviewer_final
PROJECT_DIR = Path(__file__).resolve().parents[2]

# Data directory
DATA_DIR = PROJECT_DIR / "data"

# CPM expression matrix with gene annotation (#ID, Gene_type, Symbol)
CPM_FILE = DATA_DIR / "combined_output.csv"

# Existing outer grouped folds
FOLD_ROOT = DATA_DIR / "revision_folds_grouped"

# Nested edgeR results (Inner + Outer)
INNER_DEG_ROOT = DATA_DIR / "Edger_nested_cv"
OUTER_DEG_ROOT = DATA_DIR / "Edger_nested_cv"

# Python nested CV results
OUTPUT_DIR = DATA_DIR / "revision_fold_results"

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3

TOP_N_DEGS = 80
K_BEST = 10

FDR_THRESHOLD = 0.05
LOGFC_THRESHOLD = 6

# ============================================================
# GENERAL HELPERS
# ============================================================

def read_sample_list(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Sample list file not found: {file_path}"
        )

    return [
        line.strip()
        for line in file_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


# ============================================================
# READ FOLD FILES (gene x sample matrices)
# ============================================================

def read_counts(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Counts file not found: {file_path}"
        )

    # Fold files: genes as rows and samples as columns.
    df = pd.read_csv(
        file_path,
        index_col=0
    )

    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    if df.index.duplicated().any():
        raise ValueError(
            f"Duplicate gene IDs found in {file_path}"
        )

    if df.columns.duplicated().any():
        raise ValueError(
            f"Duplicate sample IDs found in {file_path}"
        )

    return df


# ============================================================
# READ CPM EXPRESSION DATA
# ============================================================


def read_cpm(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"CPM file not found: {file_path}"
        )

    # Read CPM file.
    df = pd.read_csv(file_path)

    # Validate required columns.
    required_columns = ["#ID", "Symbol"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' is missing "
                f"from CPM file."
            )

    # Clean gene symbols.
    df["Symbol"] = (
        df["Symbol"]
        .astype(str)
        .str.strip()
    )

    # Remove rows without valid gene symbols.
    df = df[
        df["Symbol"].notna()
        & (df["Symbol"] != "")
        & (df["Symbol"].str.lower() != "nan")
    ].copy()

    # Identify expression columns.
    annotation_columns = [
        "#ID",
        "Gene_type",
        "Symbol"
    ]

    expression_columns = [
        col for col in df.columns
        if col not in annotation_columns
    ]

    if not expression_columns:
        raise ValueError(
            "No sample expression columns found in CPM file."
        )

    # Convert expression values to numeric.
    df[expression_columns] = df[
        expression_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Validate expression values.
    if df[expression_columns].isna().any().any():
        raise ValueError(
            "CPM matrix contains missing or non-numeric "
            "expression values."
        )

    if (df[expression_columns] < 0).any().any():
        raise ValueError(
            "CPM matrix contains negative values."
        )

    # Check duplicate gene symbols.
    duplicate_count = (
        df["Symbol"].duplicated().sum()
    )

    print(
        "Duplicate gene symbols:",
        duplicate_count
    )

    # Aggregate duplicate gene symbols using mean CPM.
    # Aggregation is performed before log transformation.
    expression_data = (
        df.groupby("Symbol", sort=False)[expression_columns]
        .mean()
    )

    # Ensure gene and sample identifiers are strings.
    expression_data.index = (
        expression_data.index.astype(str)
    )

    expression_data.columns = (
        expression_data.columns.astype(str)
    )

    # Validate unique gene symbols.
    if expression_data.index.duplicated().any():
        raise ValueError(
            "Duplicate gene symbols remain in CPM matrix."
        )

    # Apply log2(CPM + 1) transformation.
    expression_data = np.log2(
        expression_data + 1
    )

    # Display final matrix information.
    print(
        "CPM matrix shape:",
        expression_data.shape
    )

    print(
        "Number of samples:",
        expression_data.shape[1]
    )

    return expression_data



# ============================================================
# READ METADATA
# ============================================================

def read_metadata(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        index_col=0
    )

    df.index = df.index.astype(str)

    if df.index.duplicated().any():
        raise ValueError(
            f"Duplicate sample IDs found in {file_path}"
        )

    return df


# ============================================================
# LABELS
# ============================================================

def get_labels(metadata):

    possible_columns = [
        "source_name",
        "label",
        "condition",
        "group",
        "class",
    ]

    label_column = None

    for column in possible_columns:

        if column in metadata.columns:
            label_column = column
            break

    if label_column is None:
        raise ValueError(
            "Could not find a label column in metadata. "
            f"Available columns: {metadata.columns.tolist()}"
        )

    raw_labels = (
        metadata[label_column]
        .astype(str)
        .str.lower()
    )

    encoded = []

    for value in raw_labels:

        if "normal" in value or "adjacent" in value:
            encoded.append(0)

        elif "osteosarcoma" in value or "tumor" in value:
            encoded.append(1)

        else:
            raise ValueError(
                f"Unknown class label: {value}"
            )

    return pd.Series(
        encoded,
        index=metadata.index,
        name="Target"
    )


# ============================================================
# PATIENT SEPARATION VALIDATION
# ============================================================

def validate_patient_separation(
    train_meta,
    validation_meta
):

    if "patient_id" not in train_meta.columns:
        raise ValueError(
            "patient_id column is missing from training metadata."
        )

    if "patient_id" not in validation_meta.columns:
        raise ValueError(
            "patient_id column is missing from validation metadata."
        )

    train_patients = set(
        train_meta["patient_id"].astype(str)
    )

    validation_patients = set(
        validation_meta["patient_id"].astype(str)
    )

    overlap = train_patients.intersection(
        validation_patients
    )

    if overlap:
        raise ValueError(
            f"Patient leakage detected: {overlap}"
        )


# ============================================================
# LOAD DEG TABLE
# ============================================================

def load_deg_table(deg_file, description):

    deg_file = Path(deg_file)

    if not deg_file.exists():
        raise FileNotFoundError(
            f"{description} DEG file not found:\n{deg_file}"
        )

    deg_df = pd.read_csv(deg_file)

    required_columns = [
        "FDR",
        "logFC",
        "PValue"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in deg_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {deg_file}: "
            f"{missing_columns}"
        )

    # Identify the gene column.
    possible_gene_columns = [
        "Gene",
        "gene",
        "gene_id",
        "GeneID",
        "Symbol",
        "X",
        "Unnamed: 0",
    ]

    gene_column = None

    for column in possible_gene_columns:

        if column in deg_df.columns:
            gene_column = column
            break

    # If no known gene column is found,
    # use the first column that is not a statistical column.
    if gene_column is None:

        excluded = {
            "FDR",
            "logFC",
            "logCPM",
            "F",
            "PValue",
            "P.Value",
        }

        candidates = [
            col
            for col in deg_df.columns
            if col not in excluded
        ]

        if not candidates:
            raise ValueError(
                f"Could not identify gene ID column "
                f"in {deg_file}"
            )

        gene_column = candidates[0]

    # Clean gene identifiers.
    deg_df[gene_column] = (
        deg_df[gene_column]
        .astype(str)
        .str.strip()
    )

    # Convert statistical columns to numeric.
    for column in required_columns:

        deg_df[column] = pd.to_numeric(
            deg_df[column],
            errors="coerce"
        )

    deg_df = deg_df.dropna(
        subset=[
            gene_column,
            "FDR",
            "logFC",
            "PValue"
        ]
    )

    # Remove invalid gene identifiers.
    deg_df = deg_df[
        (deg_df[gene_column] != "")
        & (deg_df[gene_column].str.lower() != "nan")
    ]

    # Apply DEG filters.
    filtered = deg_df[
        (deg_df["FDR"] < FDR_THRESHOLD)
        &
        (deg_df["logFC"].abs() > LOGFC_THRESHOLD)
    ].copy()

    # Sort by FDR and PValue.
    filtered = filtered.sort_values(
        by=["FDR", "PValue"],
        ascending=[True, True]
    )

    if filtered.empty:
        raise ValueError(
            f"No DEGs passed the required filters "
            f"in {deg_file}"
        )

    # Select top 80 genes.
    top_genes = (
        filtered[gene_column]
        .drop_duplicates()
        .head(TOP_N_DEGS)
        .tolist()
    )

    print(
        f"{description}: "
        f"DEGs passing filters = {len(filtered)}, "
        f"Selected top DEGs = {len(top_genes)}"
    )

    return top_genes


# ============================================================
# PREPARE ML DATA
# ============================================================

def prepare_ml_data(
    counts,
    metadata,
    train_samples,
    validation_samples,
    deg_file,
    description,
):

    missing_train = (
        set(train_samples) - set(counts.columns)
    )

    missing_validation = (
        set(validation_samples) - set(counts.columns)
    )

    if missing_train or missing_validation:
        raise ValueError(
            f"Samples missing from counts for {description}.\n"
            f"Missing training samples: {missing_train}\n"
            f"Missing validation samples: {missing_validation}"
        )

    missing_meta = (
        set(train_samples + validation_samples)
        - set(metadata.index)
    )

    if missing_meta:
        raise ValueError(
            f"Samples missing from metadata: {missing_meta}"
        )

    train_meta = metadata.loc[
        train_samples
    ].copy()

    validation_meta = metadata.loc[
        validation_samples
    ].copy()

    validate_patient_separation(
        train_meta,
        validation_meta
    )

    y_train = get_labels(train_meta)

    y_validation = get_labels(validation_meta)

    if y_train.nunique() != 2:
        raise ValueError(
            f"Training data does not contain both classes: "
            f"{description}"
        )

    if y_validation.nunique() != 2:
        raise ValueError(
            f"Validation data does not contain both classes: "
            f"{description}"
        )

    # Load DEGs selected using the training data only.
    selected_genes = load_deg_table(
        deg_file,
        description
    )

    # Match DEG symbols with CPM gene symbols.
    available_genes = [
        gene
        for gene in selected_genes
        if gene in counts.index
    ]

    missing_genes = [
        gene
        for gene in selected_genes
        if gene not in counts.index
    ]

    print(
        f"{description}: "
        f"Selected DEGs = {len(selected_genes)}, "
        f"Available genes = {len(available_genes)}, "
        f"Missing genes = {len(missing_genes)}"
    )

    print("\nFirst 10 selected DEG IDs:")
    print(selected_genes[:10])

    print("\nFirst 10 CPM gene IDs:")
    print(counts.index[:10].tolist())

    print("\nFirst 10 missing DEG IDs:")
    print(missing_genes[:10])

    print("\nFirst 10 matched DEG IDs:")
    print(available_genes[:10])

    if len(available_genes) < K_BEST:
        raise ValueError(
            f"Only {len(available_genes)} selected genes "
            f"available for {description}. "
            f"At least {K_BEST} are required."
        )

    # Extract training and validation expression data.
    X_train = counts.loc[
        available_genes,
        train_samples
    ].T.copy()

    X_validation = counts.loc[
        available_genes,
        validation_samples
    ].T.copy()

    # Remove constant genes using training data only.
    variable_genes = X_train.columns[
        X_train.nunique(dropna=False) > 1
    ].tolist()

    X_train = X_train[
        variable_genes
    ]

    X_validation = X_validation[
        variable_genes
    ]

    if X_train.shape[1] < K_BEST:
        raise ValueError(
            f"Only {X_train.shape[1]} variable genes remain "
            f"for {description}. K_BEST={K_BEST}."
        )

    X_train = X_train.astype(float)

    X_validation = X_validation.astype(float)

    return (
        X_train,
        y_train,
        X_validation,
        y_validation
    )


# ============================================================
# LOAD OUTER FOLD DATA
# ============================================================

def load_outer_fold(outer_fold):

    fold_dir = (
        FOLD_ROOT / f"fold_{outer_fold}"
    )

    if not fold_dir.exists():
        raise FileNotFoundError(
            f"Outer fold directory not found: {fold_dir}"
        )

    # Read the fold files only to obtain the training/validation sample IDs.
    raw_train_counts = read_counts(
        fold_dir / "training_counts.csv"
    )

    raw_validation_counts = read_counts(
        fold_dir / "validation_counts.csv"
    )

    train_meta = read_metadata(
        fold_dir / "training_meta.csv"
    )

    validation_meta = read_metadata(
        fold_dir / "validation_meta.csv"
    )

    # Read the CPM expression matrix.
    expression_data = read_cpm(
        CPM_FILE
    )

    train_samples = (
        raw_train_counts.columns.astype(str).tolist()
    )

    validation_samples = (
        raw_validation_counts.columns.astype(str).tolist()
    )

    missing_samples = (
        set(train_samples + validation_samples)
        - set(expression_data.columns)
    )

    if missing_samples:
        raise ValueError(
            f"Samples missing from CPM file: {missing_samples}"
        )

    if set(train_samples) != set(train_meta.index):
        raise ValueError(
            f"Training counts/metadata mismatch "
            f"in outer fold {outer_fold}"
        )

    if set(validation_samples) != set(validation_meta.index):
        raise ValueError(
            f"Validation counts/metadata mismatch "
            f"in outer fold {outer_fold}"
        )

    train_meta = train_meta.loc[
        train_samples
    ]

    validation_meta = validation_meta.loc[
        validation_samples
    ]

    validate_patient_separation(
        train_meta,
        validation_meta
    )

    train_expression = expression_data.loc[
        :,
        train_samples
    ]

    validation_expression = expression_data.loc[
        :,
        validation_samples
    ]

    return (
        train_expression,
        train_meta,
        validation_expression,
        validation_meta
    )


# ============================================================
# INNER CROSS-VALIDATION
# ============================================================

def run_inner_cv(
    outer_fold,
    outer_train_expression,
    outer_train_meta,
):

    print("\n" + "=" * 70)
    print(
        f"OUTER FOLD {outer_fold} - INNER CROSS VALIDATION"
    )
    print("=" * 70)

    inner_results_all = []

    outer_train_samples = set(
        outer_train_expression.columns
    )

    for inner_fold in range(
        1,
        N_INNER_FOLDS + 1
    ):

        print(
            f"\nProcessing Outer Fold {outer_fold}, "
            f"Inner Fold {inner_fold}"
        )

        inner_dir = (
            INNER_DEG_ROOT
            / f"outer_fold_{outer_fold}"
            / f"inner_fold_{inner_fold}"
        )

        inner_train_samples = read_sample_list(
            inner_dir / "inner_train_samples.txt"
        )

        inner_validation_samples = read_sample_list(
            inner_dir / "inner_validation_samples.txt"
        )

        if set(inner_train_samples).intersection(
            inner_validation_samples
        ):
            raise ValueError(
                f"Sample overlap in outer {outer_fold}, "
                f"inner {inner_fold}"
            )

        if (
            set(inner_train_samples)
            | set(inner_validation_samples)
        ) != outer_train_samples:
            raise ValueError(
                f"Inner split does not cover exactly the outer "
                f"training samples in outer {outer_fold}, "
                f"inner {inner_fold}."
            )

        inner_deg_file = (
            inner_dir / "DEGs_nested_cv.csv"
        )

        (
            X_train,
            y_train,
            X_validation,
            y_validation
        ) = prepare_ml_data(
            counts=outer_train_expression,
            metadata=outer_train_meta,
            train_samples=inner_train_samples,
            validation_samples=inner_validation_samples,
            deg_file=inner_deg_file,
            description=(
                f"Outer Fold {outer_fold}, "
                f"Inner Fold {inner_fold}"
            ),
        )

        print(
            f"Training samples: {len(X_train)} | "
            f"Validation samples: {len(X_validation)} | "
            f"Genes: {X_train.shape[1]}"
        )

        inner_results = evaluate_inner_fold(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            k_best=K_BEST,
        )

        inner_results_all.extend(
            inner_results
        )

    expected_results = (
        N_INNER_FOLDS * 31
    )

    if len(inner_results_all) != expected_results:
        raise ValueError(
            f"Expected {expected_results} inner results "
            f"for outer fold {outer_fold}, "
            f"but received {len(inner_results_all)}."
        )

    return inner_results_all


# ============================================================
# OUTER FOLD EVALUATION
# ============================================================

def run_outer_evaluation(
    outer_fold,
    selected_combo,
    train_expression,
    train_meta,
    validation_expression,
    validation_meta,
):

    print("\n" + "-" * 70)
    print(
        f"OUTER FOLD {outer_fold} - FINAL EVALUATION"
    )
    print("-" * 70)

    train_samples = (
        train_expression.columns.astype(str).tolist()
    )

    validation_samples = (
        validation_expression.columns.astype(str).tolist()
    )

    combined_expression = pd.concat(
        [
            train_expression,
            validation_expression
        ],
        axis=1
    )

    combined_meta = pd.concat(
        [
            train_meta,
            validation_meta
        ],
        axis=0
    )

    outer_deg_file = (
            OUTER_DEG_ROOT
            / f"outer_fold_{outer_fold}"
            / "DEGs_outer_training.csv"
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation
    ) = prepare_ml_data(
        counts=combined_expression,
        metadata=combined_meta,
        train_samples=train_samples,
        validation_samples=validation_samples,
        deg_file=outer_deg_file,
        description=(
            f"Outer Fold {outer_fold} final evaluation"
        ),
    )

    model, metrics, prediction_df = evaluate_outer_fold(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        selected_combo=selected_combo,
        outer_fold=outer_fold,
        k_best=K_BEST,
    )

    return (
        model,
        metrics,
        prediction_df
    )


# ============================================================
# MAIN NESTED CROSS-VALIDATION
# ============================================================

def run_nested_cv():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not FOLD_ROOT.exists():
        raise FileNotFoundError(
            f"Outer fold directory not found:\n{FOLD_ROOT}"
        )

    if not INNER_DEG_ROOT.exists():
        raise FileNotFoundError(
            f"Inner edgeR results directory not found:\n"
            f"{INNER_DEG_ROOT}"
        )

    for fold_number in range(1, N_OUTER_FOLDS + 1):

        outer_deg_file = (
                OUTER_DEG_ROOT
                / f"outer_fold_{fold_number}"
                / "DEGs_outer_training.csv"
        )

        if not outer_deg_file.exists():
            raise FileNotFoundError(
                f"Outer DEG file not found:\n{outer_deg_file}"
            )

    all_inner_results = []

    all_inner_summaries = []

    all_selected_models = []

    all_outer_metrics = []

    all_outer_predictions = []

    for outer_fold in range(
        1,
        N_OUTER_FOLDS + 1
    ):

        print("\n\n" + "#" * 80)

        print(
            f"STARTING OUTER FOLD "
            f"{outer_fold}/{N_OUTER_FOLDS}"
        )

        print("#" * 80)

        (
            outer_train_expression,
            outer_train_meta,
            outer_validation_expression,
            outer_validation_meta
        ) = load_outer_fold(
            outer_fold
        )

        # ----------------------------------------------------
        # 1. Inner cross-validation
        # ----------------------------------------------------

        inner_results = run_inner_cv(
            outer_fold=outer_fold,
            outer_train_expression=outer_train_expression,
            outer_train_meta=outer_train_meta,
        )

        inner_results_df = pd.DataFrame(
            inner_results
        )

        inner_results_df.to_csv(
            OUTPUT_DIR
            / f"inner_results_outer_fold_{outer_fold}.csv",
            index=False
        )

        all_inner_results.append(
            inner_results_df
        )

        # ----------------------------------------------------
        # 2. Select best model using inner results only
        # ----------------------------------------------------

        selected_combo, inner_summary = (
            select_best_combination(
                inner_results
            )
        )

        inner_summary_df = pd.DataFrame(
            inner_summary
        )

        inner_summary_df.to_csv(
            OUTPUT_DIR
            / f"inner_summary_outer_fold_{outer_fold}.csv",
            index=False
        )

        all_inner_summaries.append(
            inner_summary_df
        )

        all_selected_models.append({
            "Outer_Fold": outer_fold,
            "Selected_Combination": str(selected_combo),
        })

        print("\nSelected combination:")
        print(selected_combo)

        # ----------------------------------------------------
        # 3. Evaluate selected model on outer validation
        # ----------------------------------------------------

        model, metrics, prediction_df = (
            run_outer_evaluation(
                outer_fold=outer_fold,
                selected_combo=selected_combo,
                train_expression=outer_train_expression,
                train_meta=outer_train_meta,
                validation_expression=outer_validation_expression,
                validation_meta=outer_validation_meta,
            )
        )

        metrics_df = pd.DataFrame(
            [metrics]
        )

        metrics_df.to_csv(
            OUTPUT_DIR
            / f"outer_fold_{outer_fold}_metrics.csv",
            index=False
        )

        all_outer_metrics.append(
            metrics
        )

        if prediction_df is not None:

            prediction_df = prediction_df.copy()

            prediction_df["Outer_Fold"] = outer_fold

            prediction_df.to_csv(
                OUTPUT_DIR
                / f"outer_fold_{outer_fold}_predictions.csv",
                index=False
            )

            all_outer_predictions.append(
                prediction_df
            )

        print("\nOuter fold metrics:")

        print(
            metrics_df.to_string(index=False)
        )

        print(
            f"\nCompleted Outer Fold {outer_fold}"
        )

    # ========================================================
    # SAVE FINAL RESULTS
    # ========================================================

    combined_inner_results = pd.concat(
        all_inner_results,
        ignore_index=True
    )

    combined_inner_results.to_csv(
        OUTPUT_DIR / "all_inner_results.csv",
        index=False
    )

    combined_inner_summaries = pd.concat(
        all_inner_summaries,
        ignore_index=True
    )

    combined_inner_summaries.to_csv(
        OUTPUT_DIR / "all_inner_summaries.csv",
        index=False
    )

    selected_models_df = pd.DataFrame(
        all_selected_models
    )

    selected_models_df.to_csv(
        OUTPUT_DIR / "selected_models_per_outer_fold.csv",
        index=False
    )

    outer_metrics_df = pd.DataFrame(
        all_outer_metrics
    )

    outer_metrics_df.to_csv(
        OUTPUT_DIR / "outer_fold_results.csv",
        index=False
    )

    # Mean and standard deviation across outer folds.
    summary_records = []

    for metric in ["F1", "AUC", "Sensitivity", "Specificity", "Balanced_Accuracy"]:

        if metric not in outer_metrics_df.columns:
            continue

        values = pd.to_numeric(
            outer_metrics_df[metric],
            errors="coerce"
        )

        summary_records.append({
            "Metric": metric,
            "Mean": values.mean(),
            "Std": values.std(ddof=1),
            "Number_of_Outer_Folds": values.count(),
        })

    final_summary_df = pd.DataFrame(
        summary_records
    )

    final_summary_df.to_csv(
        OUTPUT_DIR / "final_outer_cv_summary.csv",
        index=False
    )

    if all_outer_predictions:

        combined_predictions = pd.concat(
            all_outer_predictions,
            ignore_index=True
        )

        combined_predictions.to_csv(
            OUTPUT_DIR / "all_outer_predictions.csv",
            index=False
        )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n\n" + "=" * 80)

    print(
        "NESTED CROSS-VALIDATION COMPLETED SUCCESSFULLY"
    )

    print("=" * 80)

    print("\nSelected models per outer fold:")

    print(
        selected_models_df.to_string(index=False)
    )

    print("\nOuter fold results:")

    print(
        outer_metrics_df.to_string(index=False)
    )

    print("\nFinal summary:")

    print(
        final_summary_df.to_string(index=False)
    )

    print(
        f"\nAll results saved in:\n{OUTPUT_DIR}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_nested_cv()