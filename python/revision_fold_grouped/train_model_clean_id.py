from itertools import combinations

import pandas as pd
from sklearn.metrics import confusion_matrix
from pathlib import Path

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. Model pool: five classifiers with default hyperparameters
# (no hyperparameter tuning is performed).
# ============================================================

def get_model_pool():
    return {
        "svc": SVC(
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        "et": ExtraTreesClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        "lr": LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=42,
        ),
        "rf": RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        "gbm": GradientBoostingClassifier(
            random_state=42,
        ),
    }


# ============================================================
# 2. Build a model or soft-voting ensemble
# ============================================================

def build_combination(combo, k_best=10):

    def make_pipeline(model):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(
                score_func=f_classif,
                k=k_best,
            )),
            ("clf", clone(model)),
        ])

    if len(combo) == 1:
        return make_pipeline(combo[0][1])

    estimators = [
        (name, make_pipeline(model))
        for name, model in combo
    ]

    return VotingClassifier(
        estimators=estimators,
        voting="soft",
    )


# ============================================================
# 3. Evaluate all 31 combinations on one outer fold
# ============================================================

def evaluate_all_combinations(
    X_train,
    y_train,
    X_validation,
    y_validation,
    fold_number,
    k_best=10,
):
    model_pool = get_model_pool()
    fold_results = []

    # Save detailed predictions separately from summary metrics.
    PROJECT_DIR = Path(__file__).resolve().parents[2]
    OUTPUT_DIR = PROJECT_DIR / "data" / "revision_fold_results"
    output_dir = Path(OUTPUT_DIR) / "validation_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    y_validation = pd.Series(y_validation, index=X_validation.index)

    for combination_size in range(1, len(model_pool) + 1):

        for combo in combinations(
            model_pool.items(),
            combination_size,
        ):
            combination_name = " + ".join(
                name.upper() for name, _ in combo
            )

            model = build_combination(
                combo,
                k_best=k_best,
            )

            # Fit using training data only.
            model.fit(X_train, y_train)

            # Predict held-out validation samples.
            y_pred = model.predict(X_validation)
            probabilities = model.predict_proba(X_validation)

            # Find the probability column for class 1 explicitly.
            classes = list(model.classes_)
            if 1 not in classes:
                raise ValueError(
                    f"Class 1 not found in model classes: {classes}"
                )

            tumor_col = classes.index(1)
            y_probability = probabilities[:, tumor_col]

            f1 = f1_score(
                y_validation,
                y_pred,
                zero_division=0,
            )

            auc = roc_auc_score(
                y_validation,
                y_probability,
            )

            tn, fp, fn, tp = confusion_matrix(
                y_validation,
                y_pred,
                labels=[0, 1],
            ).ravel()

            fold_results.append({
                "Fold": fold_number,
                "Model": combination_name,
                "F1": f1,
                "AUC": auc,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            })

            # Save one row per validation sample.
            prediction_df = pd.DataFrame({
                "Fold": fold_number,
                "Model": combination_name,
                "Sample_ID": X_validation.index.astype(str),
                "Actual_Label": y_validation.to_numpy(),
                "Predicted_Label": y_pred,
                "Tumor_Probability": y_probability,
            })

            safe_name = (
                combination_name
                .replace(" + ", "_")
                .replace(" ", "_")
            )

            prediction_file = (
                output_dir
                / f"Fold_{fold_number}_{safe_name}.csv"
            )

            prediction_df.to_csv(
                prediction_file,
                index=False,
            )

            print(
                f"Fold {fold_number} | {combination_name} | "
                f"F1={f1:.4f} | AUC={auc:.4f} | "
                f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
            )

    return fold_results