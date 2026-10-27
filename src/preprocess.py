"""Cleaning and preprocessing. Every step that LEARNS something (correlated-feature
removal, scaling) is fitted on the TRAIN set only, to avoid data leakage."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from . import config


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip column names, fix inf/NaN, drop duplicates."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    n0 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"  cleaning: {n0:,} -> {len(df):,} rows")
    return df


def group_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map the raw attack names to a few families (see config.LABEL_GROUPS)."""
    def mapper(raw: str) -> str:
        r = str(raw).lower()
        for key, family in config.LABEL_GROUPS:
            if key in r:
                return family
        return "Unmapped"

    df = df.copy()
    df[config.LABEL_COL] = df[config.LABEL_COL].map(mapper)
    n_unmapped = (df[config.LABEL_COL] == "Unmapped").sum()
    if n_unmapped:
        print(f"  dropping {n_unmapped:,} rows whose label matches no group (e.g. Infiltration, Heartbleed)")
        df = df[df[config.LABEL_COL] != "Unmapped"]
    # drop classes that are far too rare to learn (and cannot be split stratified)
    counts = df[config.LABEL_COL].value_counts()
    keep = counts[counts >= config.MIN_CLASS_SAMPLES].index
    dropped = set(counts.index) - set(keep)
    if dropped:
        print(f"  dropping rare classes: {sorted(dropped)}")
    return df[df[config.LABEL_COL].isin(keep)]


def drop_constant_and_correlated(X_train: pd.DataFrame, threshold: float = config.CORR_THRESHOLD):
    """Return the list of feature names to keep (decided from TRAIN data only)."""
    keep = [c for c in X_train.columns if X_train[c].nunique() > 1]       # constant columns
    corr = X_train[keep].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = [c for c in upper.columns if (upper[c] > threshold).any()]  # redundant columns
    keep = [c for c in keep if c not in to_drop]
    print(f"  features: {X_train.shape[1]} -> {len(keep)} (removed constant/highly-correlated)")
    return keep


def prepare(df: pd.DataFrame):
    """Full pipeline. Returns a dict with everything training and the app need."""
    df = group_labels(clean(df))

    X = df.drop(columns=[config.LABEL_COL]).select_dtypes(include=[np.number])
    y_raw = df[config.LABEL_COL]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )

    X_test_full = X_test.copy()          # all numeric columns, unscaled (used by the analytics page)
    features = drop_constant_and_correlated(X_train)
    X_train, X_test = X_train[features], X_test[features]

    scaler = StandardScaler().fit(X_train)             # fit on train ONLY
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=features, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

    print("  class distribution (whole dataset):")
    print(y_raw.value_counts().to_string().replace("\n", "\n    "))

    return dict(
        X_train=X_train_s, X_test=X_test_s, y_train=y_train, y_test=y_test,
        scaler=scaler, label_encoder=le, features=features, raw_df=df, X_test_full=X_test_full,
    )


def preprocess_for_inference(df: pd.DataFrame, features, scaler) -> pd.DataFrame:
    """Used by the Streamlit app: apply the SAME steps to new, unseen traffic."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Uploaded file is missing required columns: {missing[:5]}...")
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.DataFrame(scaler.transform(X), columns=features, index=X.index)
