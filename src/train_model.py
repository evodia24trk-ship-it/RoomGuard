"""
Train Decision Tree for room occupancy (2 features, 2 classes).
Steps (Bab III.3-III.5):
  1. Load combined_dataset_2000.csv (1500 public + 500 private)
  2. Check: missing values, duplicates, negative/out-of-range values
  3. Split stratified 70% train / 30% test (random_state=42) — per petunjuk
  4. Grid-search max_depth (3..8, None) with 5-fold CV on train, scoring F1-macro
     (macro-F1 is used for selection because classes are imbalanced:
      Empty 73.4% vs Occupied 26.6% — accuracy alone would hide minority errors)
  5. Evaluate on the untouched test set: accuracy, precision, recall,
     F1 per class + macro-F1, confusion matrix, feature importances
  6. Save model (joblib), metrics.json, plots (confusion matrix, tree, importances)
  Decision Tree needs no scaling / normalisation (threshold splits only),
  so no scaler is fitted — this is stated explicitly, not omitted.
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

FEATURES = ["Light_Lux", "Sound_dB"]
TARGET = "Occupied"

def main():
    df = pd.read_csv("dataset/combined_dataset_2000.csv")
    print(f"rows={len(df)} cols={list(df.columns)}")
    print("missing per column:\n", df.isna().sum().to_dict())
    # Exact-duplicate check is on the full record (timestamp+values+label):
    # identical (Light, Sound) pairs alone are NOT duplicates — the light
    # sensor only reports whole Lux values, so e.g. (0 Lux, ~37 dB) repeats.
    print("exact duplicate rows:", int(df.duplicated().sum()))
    assert df[FEATURES].isna().sum().sum() == 0
    assert ((df["Light_Lux"] >= 0) & (df["Sound_dB"] >= 0)).all()
    print("class balance:\n", df[TARGET].value_counts(normalize=True).round(4).to_dict())

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    print(f"train={len(X_train)} (Empty={(y_train==0).sum()}, Occupied={(y_train==1).sum()})")
    print(f"test ={len(X_test)} (Empty={(y_test==0).sum()}, Occupied={(y_test==1).sum()})")

    # min_samples_leaf >= 5: a leaf may never be decided by 1-4 samples alone.
    # Without this the tree memorises the noisy "hard cases" (lamp left on,
    # person sitting quietly in the dark) and the tree becomes unreadable.
    param_grid = {"max_depth": [3, 4, 5, 6, 7, 8, None],
                  "min_samples_leaf": [5, 10],
                  "criterion": ["gini", "entropy"]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(DecisionTreeClassifier(random_state=42, class_weight="balanced"),
                        param_grid, cv=cv, scoring="f1_macro")
    grid.fit(X_train, y_train)
    print("best params:", grid.best_params_, "CV macro-F1:", round(grid.best_score_, 4))
    print("CV results (top 5):")
    cv_df = pd.DataFrame(grid.cv_results_).sort_values("mean_test_score", ascending=False)
    print(cv_df[["params", "mean_test_score", "std_test_score"]].head(5).to_string(index=False))

    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "n_total": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "best_params": {k: (None if v is None else str(v)) for k, v in grid.best_params_.items()},
        "cv_macro_f1": round(float(grid.best_score_), 4),
        "test_accuracy": round(float(acc), 4),
        "test_precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "test_recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "test_f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "per_class": classification_report(y_test, y_pred, target_names=["Empty", "Occupied"],
                                           output_dict=True, zero_division=0),
        "confusion_matrix": cm.tolist(),  # [[TN, FP],[FN, TP]]
        "feature_importances": {f: round(float(v), 4) for f, v in zip(FEATURES, model.feature_importances_)},
        "tree_text": export_text(model, feature_names=FEATURES, class_names=["Empty", "Occupied"]),
        "train_class": {"Empty": int((y_train == 0).sum()), "Occupied": int((y_train == 1).sum())},
        "test_class": {"Empty": int((y_test == 0).sum()), "Occupied": int((y_test == 1).sum())},
    }
    print(classification_report(y_test, y_pred, target_names=["Empty", "Occupied"], digits=4))
    print("confusion matrix [[TN,FP],[FN,TP]]:\n", cm)
    print("feature importances:", metrics["feature_importances"])
    print("\n--- Decision Tree ---\n", metrics["tree_text"])

    joblib.dump(model, "model/decision_tree_model.pkl")
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Empty", "Occupied"]).plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set (n=600)")
    fig.tight_layout(); fig.savefig("model/confusion_matrix.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_tree(model, feature_names=FEATURES, class_names=["Empty", "Occupied"],
              filled=True, rounded=True, fontsize=9, ax=ax)
    ax.set_title("Decision Tree (best params: " + str(grid.best_params_) + ")")
    fig.tight_layout(); fig.savefig("model/decision_tree.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    imp.plot.barh(ax=ax); ax.set_xlabel("Importance"); ax.set_title("Feature Importance")
    fig.tight_layout(); fig.savefig("model/feature_importance.png", dpi=200); plt.close(fig)
    print("Saved: model/decision_tree_model.pkl, model/metrics.json, model/*.png")

if __name__ == "__main__":
    main()
