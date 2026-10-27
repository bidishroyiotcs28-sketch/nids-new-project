"""Per-prediction explanations.

Preferred method: SHAP (TreeExplainer) for tree models (Decision Tree, Random
Forest, XGBoost). For every other model, or when the `shap` package is not
installed, we fall back to a model-agnostic *occlusion* score: how much does the
predicted-class probability drop when one feature is replaced by its typical
(background-mean) value? Both give a signed contribution per feature:
positive = pushes the model towards the explained class.
"""
import numpy as np
import pandas as pd
from . import models

try:
    import shap
    HAS_SHAP = True
except ImportError:  # the app still works, with the occlusion fallback
    shap = None
    HAS_SHAP = False


def get_explainer(model, model_name: str):
    """Build a SHAP TreeExplainer when possible, else return None (-> occlusion)."""
    if HAS_SHAP and models.base_name(model_name) in models.TREE_MODELS:
        try:
            return shap.TreeExplainer(model)
        except Exception:
            return None
    return None


def _shap_for_class(explainer, x: np.ndarray, class_idx: int) -> np.ndarray:
    """Return the SHAP vector (n_features,) for one class, across SHAP versions."""
    sv = explainer.shap_values(x.reshape(1, -1))
    if isinstance(sv, list):                      # older SHAP: list of (1, F) per class
        return np.asarray(sv[class_idx])[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:                              # newer SHAP: (1, F, n_classes)
        return sv[0, :, class_idx]
    return sv[0]                                  # binary / single-output


def _occlusion(model, x: np.ndarray, background: np.ndarray, class_idx: int, features) -> np.ndarray:
    """Probability drop when each feature is replaced by its background mean."""
    base = background.mean(axis=0)
    variants = np.tile(x, (len(x), 1))
    variants[np.arange(len(x)), np.arange(len(x))] = base
    # DataFrames, because the models were fitted with column names
    p_full = model.predict_proba(pd.DataFrame(x.reshape(1, -1), columns=list(features)))[0, class_idx]
    p_occ = model.predict_proba(pd.DataFrame(variants, columns=list(features)))[:, class_idx]
    return p_full - p_occ


def explain_prediction(model, model_name, x_scaled, background, features, class_idx,
                       explainer=None, top_n: int = 10):
    """Explain why `model` assigns class `class_idx` to one (already scaled) flow.

    Returns (DataFrame sorted by |contribution|, method_name). Columns:
    feature, contribution, value (scaled), direction.
    """
    x = np.asarray(x_scaled, dtype=float).ravel()
    method = "Occlusion (model-agnostic)"
    contrib = None
    if explainer is not None:
        try:
            contrib = _shap_for_class(explainer, x, class_idx)
            method = "SHAP (TreeExplainer)"
        except Exception:
            contrib = None
    if contrib is None:
        if not hasattr(model, "predict_proba"):
            raise ValueError("Model has no predict_proba; cannot explain.")
        contrib = _occlusion(model, x, np.asarray(background, dtype=float), class_idx, features)

    df = pd.DataFrame({"feature": list(features), "contribution": contrib, "value": x})
    df["direction"] = np.where(df["contribution"] >= 0, "pushes towards this class", "pushes away")
    df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index)
    return df.head(top_n).reset_index(drop=True), method
