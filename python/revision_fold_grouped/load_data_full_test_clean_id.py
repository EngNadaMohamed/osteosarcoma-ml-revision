from pathlib import Path

import pandas as pd

from main_id import load_fold_data_id
from train_model_clean_id import evaluate_all_combinations

# ============================================================
# 1. Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = BASE_DIR / "data" / "revision_fold_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
K_BEST = 10

# ============================================================
# 2. Main analysis: real labels, fold-based evaluation
# ============================================================

def run_revision_main_analysis():

    print("\n" + "=" * 65)
    print("MAIN ANALYSIS: REAL LABELS AND REVISED OUTER FOLDS")
    print("=" * 65)

    all_results = []

    for fold_number in range(1, N_FOLDS + 1):

        print(f"\nProcessing Fold {fold_number}/{N_FOLDS}")

        X_train, y_train, X_validation, y_validation = (
            load_fold_data_id(fold_number)
        )

        fold_results = evaluate_all_combinations(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            fold_number=fold_number,
            k_best=K_BEST,
        )

        all_results.extend(fold_results)

    if not all_results:
        raise ValueError(
            "No results were generated. "
            "Please check the fold data and model evaluation."
        )

    results_df = pd.DataFrame(all_results)

    # Save results for every model and fold.
    fold_results_path = (
        OUTPUT_DIR / "main_results_by_fold_id.csv"
    )
    results_df.to_csv(fold_results_path, index=False)

    # Summarize model performance across the folds.
    summary_df = (
        results_df
        .groupby("Model")
        .agg(
            Mean_F1=("F1", "mean"),
            Std_F1=("F1", "std"),
            Mean_AUC=("AUC", "mean"),
            Std_AUC=("AUC", "std"),
            Number_of_Folds=("Fold", "nunique"),
        )
        .reset_index()
    )

    summary_df = summary_df.fillna(0)

    summary_df = summary_df.sort_values(
        "Mean_F1",
        ascending=False,
    )

    summary_path = (
        OUTPUT_DIR / "main_results_summary_id.csv"
    )
    summary_df.to_csv(summary_path, index=False)

    print("\nMain analysis results:")
    print(summary_df.to_string(index=False))

    print(f"\nFold results saved to: {fold_results_path}")
    print(f"Summary saved to: {summary_path}")

    return results_df, summary_df

# ============================================================
# 3. Entry point
# ============================================================

if __name__ == "__main__":
    run_revision_main_analysis()

    print("\nMain analysis has finished.")