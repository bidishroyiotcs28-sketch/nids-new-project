"""Metrics and plots. Accuracy alone is misleading on imbalanced data, so we report
macro-F1, per-class results and the false alarm rate on normal traffic."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # works without a display (servers, Colab, scripts)
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, roc_auc_score)
from sklearn.inspection import permutation_importance
from . import config


def false_alarm_rate(y_true, y_pred, normal_idx: int) -> float:
    """Share of NORMAL traffic wrongly flagged as an attack. Must be low in practice."""
    mask = y_true == normal_idx
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] != normal_idx).mean())


def detection_rate(y_true, y_pred, normal_idx: int) -> float:
    """Share of ATTACK traffic flagged as some kind of attack (binary view: attack vs normal)."""
    mask = y_true != normal_idx
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] != normal_idx).mean())


def summarize(name, y_true, y_pred, normal_idx, train_time) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": p,
        "Macro Recall": r,
        "Macro F1": f1,
        "Detection Rate": detection_rate(y_true, y_pred, normal_idx),
        "False Alarm Rate": false_alarm_rate(y_true, y_pred, normal_idx),
        "Train Time (s)": train_time,
    }


def score_model(name, model, X_test, y_test, normal_idx, train_time, n_classes, eval_subset=None):
    """Score one trained model: the original metrics plus ROC-AUC and prediction time.

    Returns (summary_row, y_true_used, y_pred, y_proba). `eval_subset` lets the slow
    models (KNN/SVM) be scored on a capped test subsample; the row records how many.
    """
    if eval_subset is not None:
        X_test, y_test = eval_subset(name, X_test, y_test)
    t0 = time.time()
    y_pred = model.predict(X_test)
    pred_s = time.time() - t0
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    row = summarize(name, y_test, y_pred, normal_idx, train_time)
    row["ROC-AUC"] = float("nan")
    if y_proba is not None:
        try:   # macro one-vs-rest AUC over all classes
            row["ROC-AUC"] = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro",
                                           labels=list(range(n_classes)))
        except ValueError:
            pass
    row["Prediction Time (ms/1k flows)"] = pred_s / max(len(y_test), 1) * 1000 * 1000
    row["Eval Rows"] = len(y_test)
    return row, y_test, y_pred, y_proba


def save_report(name, y_true, y_pred, class_names):
    text = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    path = config.OUT_DIR / f"report_{name.replace(' ', '_')}.txt"
    path.write_text(text)
    return text


def plot_confusion(name, y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)           # row-normalised: recall per class
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (row-normalised) - {name}")
    plt.tight_layout()
    fig.savefig(config.OUT_DIR / f"confusion_{name.replace(' ', '_')}.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    melted = results.melt(id_vars="Model", value_vars=["Accuracy", "Macro F1", "Macro Recall"])
    sns.barplot(data=melted, x="Model", y="value", hue="variable", ax=ax)
    ax.set_ylim(max(0, melted["value"].min() - 0.05), 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    fig.savefig(config.OUT_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_class_distribution(raw_df: pd.DataFrame):
    counts = raw_df[config.LABEL_COL].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#4C72B0")
    ax.set_yscale("log")
    ax.set_ylabel("Flows (log scale)")
    ax.set_title("Class distribution (note the imbalance)")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(config.OUT_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(model, X_test, y_test, features, top_n=15, sample=3000):
    """Tree models: built-in importances. Otherwise: permutation importance on a sample."""
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=features)
        title = "Feature importance (model built-in)"
    else:
        idx = np.random.default_rng(config.RANDOM_STATE).choice(
            len(X_test), size=min(sample, len(X_test)), replace=False)
        r = permutation_importance(model, X_test.iloc[idx], y_test[idx], n_repeats=3,
                                   random_state=config.RANDOM_STATE, scoring="f1_macro")
        imp = pd.Series(r.importances_mean, index=features)
        title = "Feature importance (permutation)"
    # full ranking is also saved as CSV for the dashboard's feature-importance chart
    imp.sort_values(ascending=False).rename("importance").to_csv(config.OUT_DIR / "feature_importance.csv",
                                                                 index_label="feature")
    imp = imp.sort_values(ascending=False).head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax, color="#55A868")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(config.OUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    return imp[::-1]
