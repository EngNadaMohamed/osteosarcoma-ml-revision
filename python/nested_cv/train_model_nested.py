from itertools import combinations

import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. Configuration
# ============================================================

RANDOM_STATE = 42
K_BEST = 10


# ============================================================
# 2. Model Pool
# Five classifiers with default hyperparameters (no tuning)
# ============================================================

def get_model_pool():

    return {
        "svc": SVC(
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),

        "et": ExtraTreesClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "lr": LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=RANDOM_STATE,
        ),

        "rf": RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "gbm": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }


# ============================================================
# 3. Build Model Pipeline
# StandardScaler -> SelectKBest -> Classifier
# ============================================================

def build_pipeline(classifier, k_best=K_BEST):

    return Pipeline([
        ("scaler", StandardScaler()),

        ("select", SelectKBest(
            score_func=f_classif,
            k=k_best,
        )),

        ("clf", clone(classifier)),
    ])


# ============================================================
# 4. Build One Combination
# Single model or Soft Voting Ensemble
# ============================================================

def build_combination(combo, k_best=K_BEST):

    model_pool = get_model_pool()

    estimators = [
        (
            name,
            build_pipeline(model_pool[name], k_best)
        )
        for name in combo
    ]

    if len(estimators) == 1:
        return estimators[0][1]

    return VotingClassifier(
        estimators=estimators,
        voting="soft",
    )


# ============================================================
# 5. Generate All 31 Combinations
# ============================================================

def get_all_combinations():

    model_names = list(get_model_pool().keys())

    all_combinations = []

    for size in range(1, len(model_names) + 1):

        all_combinations.extend(
            combinations(model_names, size)
        )

    if len(all_combinations) != 31:
        raise ValueError(
            "Expected exactly 31 model combinations."
        )

    return all_combinations


def combination_name(combo):

    return " + ".join(
        name.upper() for name in combo
    )


# ============================================================
# 6. Calculate Evaluation Metrics
# Positive class: Osteosarcoma = 1
# ============================================================

def calculate_metrics(model, X, y, fold_number, model_name):

    y = pd.Series(y, index=X.index).astype(int)

    y_pred = model.predict(X)

    probabilities = model.predict_proba(X)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            f"Class 1 not found. Model classes: {classes}"
        )

    tumor_column = classes.index(1)

    y_probability = probabilities[:, tumor_column]

    f1 = f1_score(
        y,
        y_pred,
        zero_division=0,
    )

    auc = roc_auc_score(
        y,
        y_probability,
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2

    return {
        "Fold": fold_number,
        "Model": model_name,
        "F1": f1,
        "AUC": auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Balanced_Accuracy": balanced_accuracy,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


# ============================================================
# 7. Evaluate All 31 Combinations on One Inner Fold
# ============================================================

def evaluate_inner_fold(
    X_train,
    y_train,
    X_validation,
    y_validation,
    outer_fold,
    inner_fold,
    k_best=K_BEST,
):

    results = []

    if X_train.shape[1] < k_best:
        raise ValueError(
            f"Outer {outer_fold}, Inner {inner_fold}: "
            f"Only {X_train.shape[1]} genes available; "
            f"SelectKBest requires {k_best}."
        )

    if set(pd.Series(y_train).unique()) != {0, 1}:
        raise ValueError(
            "Inner training data must contain both classes."
        )

    if set(pd.Series(y_validation).unique()) != {0, 1}:
        raise ValueError(
            "Inner validation data must contain both classes."
        )

    for combo in get_all_combinations():

        name = combination_name(combo)

        model = build_combination(
            combo,
            k_best=k_best,
        )

        model.fit(X_train, y_train)

        metrics = calculate_metrics(
            model=model,
            X=X_validation,
            y=y_validation,
            fold_number=inner_fold,
            model_name=name,
        )

        metrics["Outer_Fold"] = outer_fold
        metrics["Inner_Fold"] = inner_fold

        results.append(metrics)

        print(
            f"Outer {outer_fold} | "
            f"Inner {inner_fold} | "
            f"{name} | "
            f"F1={metrics['F1']:.4f} | "
            f"AUC={metrics['AUC']:.4f}"
        )

    return results


# ============================================================
# 8. Select Best Combination Using Inner CV Results
# Selection criterion: Mean F1
# ============================================================

def select_best_combination(inner_results):

    results_df = pd.DataFrame(inner_results)

    if results_df.empty:
        raise ValueError(
            "No inner CV results available."
        )

    summary = (
        results_df
        .groupby("Model")
        .agg(
            Mean_F1=("F1", "mean"),
            Std_F1=("F1", "std"),
            Mean_AUC=("AUC", "mean"),
            Std_AUC=("AUC", "std"),
            Number_of_Inner_Folds=("Inner_Fold", "nunique"),
        )
        .reset_index()
    )

    summary["Std_F1"] = summary["Std_F1"].fillna(0)
    summary["Std_AUC"] = summary["Std_AUC"].fillna(0)

    summary = summary.sort_values(
        ["Mean_F1", "Mean_AUC"],
        ascending=[False, False],
    ).reset_index(drop=True)

    best_model_name = summary.iloc[0]["Model"]

    best_combo = tuple(
        part.strip().lower()
        for part in best_model_name.split("+")
    )

    print("\nInner CV Model Selection")
    print(summary.to_string(index=False))

    print("\nSelected Combination:", best_model_name)

    return best_combo, summary


# ============================================================
# 9. Final Outer Fold Evaluation
# Train selected combination on full outer training data
# Evaluate once on outer validation data
# ============================================================

def evaluate_outer_fold(
    X_train,
    y_train,
    X_validation,
    y_validation,
    selected_combo,
    outer_fold,
    k_best=K_BEST,
):

    if X_train.shape[1] < k_best:
        raise ValueError(
            f"Outer Fold {outer_fold}: "
            f"Only {X_train.shape[1]} genes available."
        )

    model = build_combination(
        selected_combo,
        k_best=k_best,
    )

    model.fit(X_train, y_train)

    name = combination_name(selected_combo)

    metrics = calculate_metrics(
        model=model,
        X=X_validation,
        y=y_validation,
        fold_number=outer_fold,
        model_name=name,
    )

    metrics["Selected_Combination"] = name

    y_pred = model.predict(X_validation)
    probabilities = model.predict_proba(X_validation)

    tumor_column = list(model.classes_).index(1)

    prediction_df = pd.DataFrame({
        "Outer_Fold": outer_fold,
        "Sample_ID": X_validation.index.astype(str),
        "Actual_Label": pd.Series(
            y_validation,
            index=X_validation.index
        ).astype(int).to_numpy(),
        "Predicted_Label": y_pred,
        "Tumor_Probability": probabilities[:, tumor_column],
    })

    print(
        f"\nOUTER FOLD {outer_fold} FINAL RESULT | "
        f"{name} | "
        f"F1={metrics['F1']:.4f} | "
        f"AUC={metrics['AUC']:.4f}"
    )

    return model, metrics, prediction_df