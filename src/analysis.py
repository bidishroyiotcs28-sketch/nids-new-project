"""Advanced model analysis: ROC / Precision-Recall curves, learning curves,
5-fold cross validation and hyper-parameter tuning.

Everything here returns plain Python dicts / lists so run_pipeline.py can dump
them to JSON in outputs/, and the dashboard pages can draw them without having
to retrain anything."""
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import (StratifiedKFold, cross_validate, learning_curve,
                                     RandomizedSearchCV, GridSearchCV)
from . import config, models

GRID_POINTS = 101   # every curve is resampled to this many points (small JSON, easy to plot)


# ---------------------------------------------------------------------------
# ROC and Precision-Recall curves
# ---------------------------------------------------------------------------
def roc_pr_curves(y_true, y_proba, class_names):
    """One-vs-rest ROC and PR curves for each class plus a macro average.

    Returns (roc, pr): dicts {"macro": {...}, "classes": {name: {...}}}.
    Classes missing from y_true are skipped.
    """
    fpr_grid = np.linspace(0, 1, GRID_POINTS)
    rec_grid = np.linspace(0, 1, GRID_POINTS)
    roc_cls, pr_cls, tprs, precs = {}, {}, [], []

    for i, name in enumerate(class_names):
        pos = (y_true == i).astype(int)
        if pos.sum() == 0 or pos.sum() == len(pos):
            continue
        fpr, tpr, _ = roc_curve(pos, y_proba[:, i])
        tpr_i = np.interp(fpr_grid, fpr, tpr)
        tpr_i[0] = 0.0
        roc_cls[name] = {"fpr": fpr_grid.tolist(), "tpr": tpr_i.tolist(), "auc": float(auc(fpr, tpr))}
        tprs.append(tpr_i)

        p, r, _ = precision_recall_curve(pos, y_proba[:, i])
        prec_i = np.interp(rec_grid, r[::-1], p[::-1])        # recall must be increasing for interp
        pr_cls[name] = {"recall": rec_grid.tolist(), "precision": prec_i.tolist(),
                        "ap": float(average_precision_score(pos, y_proba[:, i]))}
        precs.append(prec_i)

    mean_tpr = np.mean(tprs, axis=0)
    mean_prec = np.mean(precs, axis=0)
    roc = {"macro": {"fpr": fpr_grid.tolist(), "tpr": mean_tpr.tolist(),
                     "auc": float(np.mean([v["auc"] for v in roc_cls.values()]))},
           "classes": roc_cls}
    pr = {"macro": {"recall": rec_grid.tolist(), "precision": mean_prec.tolist(),
                    "ap": float(np.mean([v["ap"] for v in pr_cls.values()]))},
          "classes": pr_cls}
    return roc, pr


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------
def diagnose_fit(train_acc: float, val_acc: float) -> str:
    """Turn the final train/validation accuracies into a plain-English verdict."""
    if train_acc - val_acc > config.OVERFIT_GAP:
        return "Overfitting"
    if val_acc < config.UNDERFIT_SCORE:
        return "Underfitting"
    return "Well-fitted"


def compute_learning_curve(name, model, X, y):
    """Training vs validation accuracy as the training set grows (3-fold CV)."""
    Xs, ys = models.stratified_subsample(X, y, config.ANALYSIS_SAMPLE)
    sizes, tr, va = learning_curve(
        models.light_clone(model), Xs, ys, train_sizes=list(config.LEARNING_CURVE_SIZES),
        cv=StratifiedKFold(3, shuffle=True, random_state=config.RANDOM_STATE),
        scoring="accuracy", n_jobs=1, shuffle=True, random_state=config.RANDOM_STATE)
    out = {"train_sizes": sizes.tolist(),
           "train_mean": tr.mean(axis=1).tolist(), "train_std": tr.std(axis=1).tolist(),
           "val_mean": va.mean(axis=1).tolist(), "val_std": va.std(axis=1).tolist()}
    out["diagnosis"] = diagnose_fit(out["train_mean"][-1], out["val_mean"][-1])
    out["gap"] = out["train_mean"][-1] - out["val_mean"][-1]
    return out


# ---------------------------------------------------------------------------
# 5-fold cross validation
# ---------------------------------------------------------------------------
def compute_cv(name, model, X, y):
    """Stratified 5-fold CV. Reports accuracy and macro-F1 per fold, mean and std."""
    Xs, ys = models.stratified_subsample(X, y, config.ANALYSIS_SAMPLE)
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    res = cross_validate(models.light_clone(model), Xs, ys, cv=cv, n_jobs=1,
                         scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"})
    out = {"rows_used": int(len(ys))}
    for metric in ("accuracy", "f1_macro"):
        scores = res[f"test_{metric}"]
        out[metric] = {"folds": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std())}
    return out


# ---------------------------------------------------------------------------
# Hyper-parameter tuning
# ---------------------------------------------------------------------------
def tune_model(name, model, X, y, n_iter: int = 8):
    """Randomized search (cheaper than a full grid) with 3-fold CV on a train subsample.

    Returns (tuned_untrained_estimator, info_dict). Grids with <= n_iter combinations
    are searched exhaustively with GridSearchCV instead.
    """
    grid = models.PARAM_GRIDS[models.base_name(name)]
    n_combos = int(np.prod([len(v) for v in grid.values()]))
    Xs, ys = models.stratified_subsample(X, y, config.ANALYSIS_SAMPLE)
    cv = StratifiedKFold(3, shuffle=True, random_state=config.RANDOM_STATE)
    base = models.light_clone(model)

    t0 = time.time()
    if n_combos <= n_iter:
        search = GridSearchCV(base, grid, scoring="f1_macro", cv=cv, n_jobs=1)
        method = "GridSearchCV"
    else:
        search = RandomizedSearchCV(base, grid, n_iter=n_iter, scoring="f1_macro", cv=cv,
                                    n_jobs=1, random_state=config.RANDOM_STATE)
        method = "RandomizedSearchCV"
    if models.base_name(name) == "XGBoost":
        from sklearn.utils.class_weight import compute_sample_weight
        search.fit(Xs, ys, sample_weight=compute_sample_weight("balanced", ys))
    else:
        search.fit(Xs, ys)

    best_params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in search.best_params_.items()}
    info = {"method": method, "best_params": best_params, "best_cv_f1_macro": float(search.best_score_),
            "combinations_in_space": n_combos, "candidates_tried": min(n_combos, n_iter) if method.startswith("Rand") else n_combos,
            "search_seconds": time.time() - t0, "rows_used": int(len(ys))}
    tuned = models.clone(model).set_params(**search.best_params_)
    return tuned, info


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def exploration_sample(X_scaled: pd.DataFrame, y_names, n: int = config.EXPLORE_SAMPLE) -> pd.DataFrame:
    """Stratified sample of SCALED features + class name, for the PCA / K-Means page."""
    codes, uniques = pd.factorize(pd.Series(np.asarray(y_names)))
    Xs, cs = models.stratified_subsample(X_scaled.reset_index(drop=True), codes, n)
    out = Xs.copy()
    out["Label"] = uniques[cs]
    return out.reset_index(drop=True)
