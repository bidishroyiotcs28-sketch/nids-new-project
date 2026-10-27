"""The models we compare. Class imbalance is handled with class weights
(simple, no extra library, and no synthetic rows leaking into the test set).

Added: KNN, SVM (RBF) and Gaussian Naive Bayes, hyper-parameter search spaces,
and helpers that keep the slow models (KNN, SVM) tractable on big datasets."""
import time
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from . import config

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# scikit-learn >= 1.9 warns that SVC(probability=True) will be removed in 1.11.
# requirements.txt pins scikit-learn below 1.11, so the warning is just noise here.
warnings.filterwarnings("ignore", message=".*`probability` parameter was deprecated.*",
                        category=FutureWarning)

SEED = config.RANDOM_STATE
TREE_MODELS = ("Decision Tree", "Random Forest", "XGBoost")   # SHAP TreeExplainer works on these


def get_models(fast: bool = False) -> dict:
    """`fast=True` shrinks the models so a first run finishes in about a minute."""
    n_trees = 50 if fast else 200
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=20, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=n_trees, class_weight="balanced",
                                                n_jobs=-1, random_state=SEED),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                             max_iter=100, random_state=SEED),
        # --- added models ---
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        # probability=True gives the class probabilities needed for ROC-AUC / confidence
        "SVM": SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=SEED),
        "Naive Bayes": GaussianNB(),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=n_trees, max_depth=6, learning_rate=0.1,
                                          tree_method="hist", n_jobs=-1, random_state=SEED,
                                          eval_metric="mlogloss")
    return models


# Search spaces used by RandomizedSearchCV in run_pipeline.py (--tune)
PARAM_GRIDS = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100]},
    "Decision Tree": {"max_depth": [8, 12, 20, 30, None], "min_samples_leaf": [1, 2, 5, 10],
                      "criterion": ["gini", "entropy"]},
    "Random Forest": {"n_estimators": [100, 200, 300], "max_depth": [10, 20, 30, None],
                      "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2"]},
    "MLP": {"hidden_layer_sizes": [(32,), (64, 32), (128, 64)], "alpha": [1e-4, 1e-3, 1e-2]},
    "KNN": {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"]},
    "SVM": {"C": [0.5, 1, 5, 10], "gamma": ["scale", 0.01, 0.1]},
    "Naive Bayes": {"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
    "XGBoost": {"n_estimators": [100, 200, 300], "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.2], "subsample": [0.8, 1.0]},
}


def base_name(name: str) -> str:
    """'Random Forest (tuned)' -> 'Random Forest'."""
    return name.replace(" (tuned)", "")


def stratified_subsample(X, y, n_max: int, seed: int = SEED):
    """Return (X_sub, y_sub) with at most n_max rows, keeping class proportions.
    Every class keeps at least a handful of rows so metrics stay defined."""
    if len(y) <= n_max:
        return X, y
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        k = max(min(len(c_idx), 30), int(round(n_max * len(c_idx) / len(y))))
        idx.extend(rng.choice(c_idx, size=min(k, len(c_idx)), replace=False))
    idx = np.sort(np.array(idx))
    return X.iloc[idx], y[idx]


def fit(name, model, X_train, y_train):
    """Train one model and return the training time in seconds.
    XGBoost has no class_weight option, so it gets balanced sample weights.
    KNN / SVM are trained on a capped stratified subsample (see config)."""
    if base_name(name) in config.SLOW_MODELS:
        X_train, y_train = stratified_subsample(X_train, y_train, config.SLOW_TRAIN_MAX)
    start = time.time()
    if base_name(name) == "XGBoost":
        w = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, sample_weight=w)
    else:
        model.fit(X_train, y_train)
    return time.time() - start


def eval_subset(name, X_test, y_test):
    """Slow models are scored on a capped test subsample; others on the full test set."""
    if base_name(name) in config.SLOW_MODELS:
        return stratified_subsample(X_test, y_test, config.SLOW_EVAL_MAX)
    return X_test, y_test


def light_clone(model):
    """Clone for CV / learning curves. SVC's internal probability calibration is
    an extra 5-fold fit and is not needed for accuracy/F1, so switch it off."""
    m = clone(model)
    if isinstance(m, SVC):
        m.set_params(probability=False)
    return m
