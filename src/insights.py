"""Pure-data helpers for the dashboard pages (no Streamlit / Plotly in here, so
they are easy to test and reuse). The pages only call these and draw the result."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from . import config


# ---------------------------------------------------------------------------
# Prediction + probability
# ---------------------------------------------------------------------------
def attack_probability(proba_row, class_names, normal_class: str = config.NORMAL_CLASS) -> dict:
    """Summarise one row of class probabilities.

    attack probability = 1 - P(normal): the chance that the flow is *some* attack,
    even when the model is unsure which kind.
    """
    proba_row = np.asarray(proba_row, dtype=float)
    pred_idx = int(proba_row.argmax())
    normal_idx = list(class_names).index(normal_class)
    return {"pred_idx": pred_idx, "pred_name": class_names[pred_idx],
            "confidence": float(proba_row[pred_idx]),
            "p_normal": float(proba_row[normal_idx]), "p_attack": float(1 - proba_row[normal_idx]),
            "is_attack": class_names[pred_idx] != normal_class}


# ---------------------------------------------------------------------------
# Network analytics
# ---------------------------------------------------------------------------
def normalize_protocol(series: pd.Series) -> pd.Series:
    """Map protocol numbers (6/17/1) or names ('tcp') to TCP / UDP / ICMP / Other."""
    def one(v):
        try:
            return config.PROTOCOL_NAMES.get(int(float(v)), "Other")
        except (TypeError, ValueError):
            s = str(v).strip().upper()
            return s if s in ("TCP", "UDP", "ICMP") else "Other"
    return series.map(one)


def traffic_summary(df: pd.DataFrame, normal_class: str = config.NORMAL_CLASS) -> dict:
    """Numbers for the analytics cards. `df` needs a 'Prediction' column.

    Packet totals come from 'Total Fwd Packets' + 'Total Backward Packets' when those
    columns exist; otherwise only flow counts are available (has_packets=False).
    """
    is_attack = df["Prediction"] != normal_class
    out = {"total_flows": int(len(df)), "attack_flows": int(is_attack.sum()),
           "normal_flows": int((~is_attack).sum())}
    out["attack_pct_flows"] = out["attack_flows"] / max(out["total_flows"], 1)

    pkt_cols = ["Total Fwd Packets", "Total Backward Packets"]
    out["has_packets"] = all(c in df.columns for c in pkt_cols)
    if out["has_packets"]:
        pkts = df[pkt_cols].sum(axis=1)
        out["total_packets"] = int(round(pkts.sum()))
        out["attack_packets"] = int(round(pkts[is_attack].sum()))
        out["normal_packets"] = int(round(pkts[~is_attack].sum()))
        out["attack_pct_packets"] = out["attack_packets"] / max(out["total_packets"], 1)

    counts = df.loc[is_attack, "Prediction"].value_counts()
    out["attack_counts"] = counts
    out["top_attack"] = (counts.index[0], int(counts.iloc[0])) if len(counts) else None

    out["protocol_counts"] = None
    if "Protocol" in df.columns:
        proto = normalize_protocol(df["Protocol"])
        out["protocol_counts"] = proto.value_counts()
        out["protocol_by_class"] = pd.crosstab(proto, np.where(is_attack, "Attack", "Normal"))
    return out


# ---------------------------------------------------------------------------
# PCA and K-Means (exploration page)
# ---------------------------------------------------------------------------
def split_explore(df: pd.DataFrame):
    """explore_sample.csv -> (scaled feature matrix, label Series)."""
    return df.drop(columns=["Label"]), df["Label"]


def pca_projection(X: pd.DataFrame, labels: pd.Series, normal_class: str = config.NORMAL_CLASS):
    """2-D PCA. Returns (DataFrame[PC1, PC2, Label, Group], explained_variance_ratio)."""
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    xy = pca.fit_transform(X)
    out = pd.DataFrame({"PC1": xy[:, 0], "PC2": xy[:, 1], "Label": labels.to_numpy()})
    out["Group"] = np.where(out["Label"] == normal_class, "Normal traffic", "Attack traffic")
    return out, pca.explained_variance_ratio_


def kmeans_report(X: pd.DataFrame, labels: pd.Series, k: int, normal_class: str = config.NORMAL_CLASS):
    """Fit K-Means (unsupervised: it never sees the labels) and describe each cluster.

    Labels are only used afterwards, to show what each cluster turned out to contain.
    """
    km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(X)
    cl = pd.Series(km.labels_, name="Cluster")
    lab = labels.reset_index(drop=True)

    rows = []
    for c in range(k):
        mask = cl == c
        counts = lab[mask].value_counts()
        rows.append({"Cluster": c, "Flows": int(mask.sum()), "Share": mask.mean(),
                     "Dominant class": counts.index[0] if len(counts) else "-",
                     "Purity": counts.iloc[0] / max(mask.sum(), 1) if len(counts) else 0.0,
                     "Attack share": float((lab[mask] != normal_class).mean()) if mask.any() else 0.0})
    stats = pd.DataFrame(rows)

    n_sil = min(len(X), 3000)
    sil = float(silhouette_score(X, km.labels_, sample_size=n_sil, random_state=config.RANDOM_STATE)) if k > 1 else float("nan")
    crosstab = pd.crosstab(cl, lab, normalize="index")
    return {"clusters": cl, "stats": stats, "silhouette": sil, "inertia": float(km.inertia_),
            "crosstab": crosstab}


def elbow(X: pd.DataFrame, k_values=range(2, 11)) -> pd.DataFrame:
    """Inertia for several k, used for the elbow chart."""
    rows = [{"k": k, "inertia": KMeans(n_clusters=k, n_init=3, random_state=config.RANDOM_STATE).fit(X).inertia_}
            for k in k_values]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prediction-pipeline walk-through
# ---------------------------------------------------------------------------
def pipeline_trace(raw_row: pd.Series, bundle: dict, model) -> dict:
    """Run ONE flow through every stage and keep each intermediate result."""
    features, scaler, le = bundle["features"], bundle["scaler"], bundle["label_encoder"]
    class_names = list(le.classes_)

    raw = raw_row.copy()
    kept = pd.to_numeric(raw.reindex(features), errors="coerce")
    n_bad = int((~np.isfinite(kept.to_numpy(dtype=float))).sum())
    cleaned = kept.replace([np.inf, -np.inf], np.nan).fillna(0)       # same as preprocess_for_inference

    scaled = (cleaned.to_numpy() - scaler.mean_) / scaler.scale_
    scaling = pd.DataFrame({"feature": features, "raw value": cleaned.to_numpy(),
                            "mean (train)": scaler.mean_, "std (train)": scaler.scale_,
                            "scaled z-score": scaled})

    proba = model.predict_proba(pd.DataFrame(scaled.reshape(1, -1), columns=list(features)))[0]
    info = attack_probability(proba, class_names)
    return {"n_raw_columns": int(len(raw)), "n_features": len(features),
            "dropped_columns": [c for c in raw.index if c not in features],
            "n_fixed_values": n_bad, "kept": kept, "cleaned": cleaned, "scaling": scaling,
            "scaled": scaled, "proba": proba, "class_names": class_names, "info": info,
            "encoding": pd.DataFrame({"class": class_names, "encoded as": range(len(class_names))})}
