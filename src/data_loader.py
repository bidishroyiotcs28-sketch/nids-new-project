"""Loading data: real CICIDS2017 CSVs, or a synthetic stand-in for testing the pipeline."""
import numpy as np
import pandas as pd
from pathlib import Path
from . import config


def load_cicids(folder: Path = config.RAW_DIR, max_rows_per_file: int = config.MAX_ROWS_PER_FILE) -> pd.DataFrame:
    """Read every CSV in `folder`, sample up to `max_rows_per_file` rows from each, and stack them."""
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {folder}. Download CICIDS2017 (MachineLearningCVE) "
            "and place the CSVs there, or run with --synthetic."
        )
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="latin-1", low_memory=False)
        df.columns = df.columns.str.strip()          # CICIDS column names have leading spaces
        if len(df) > max_rows_per_file:
            df = df.sample(max_rows_per_file, random_state=config.RANDOM_STATE)
        print(f"  loaded {f.name}: {len(df):,} rows")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Synthetic data: lets you run and understand the whole pipeline without the
# multi-GB download. Column names mimic CICIDS2017. It is NOT real traffic,
# so do not report its numbers as results.
# ---------------------------------------------------------------------------
# per class: (log-duration, log-fwd-pkts, log-bwd-pkts, log-pkt-len, P(SYN), log-IAT, share)
_PROFILES = {
    "Normal":     (11, 2.5, 2.4, 5.0, 0.2, 9.0, 0.60),
    "DoS":        (13, 3.5, 1.0, 4.0, 0.1, 7.0, 0.12),
    "DDoS":       (12, 3.0, 0.5, 3.5, 0.8, 5.0, 0.12),
    "PortScan":   (4,  0.7, 0.3, 2.5, 0.9, 3.0, 0.08),
    "BruteForce": (10, 2.0, 2.0, 4.5, 0.3, 8.0, 0.05),
    "WebAttack":  (12, 2.2, 2.6, 5.5, 0.2, 8.5, 0.03),
    "Bot":        (11, 1.8, 1.8, 3.8, 0.4, 7.5, 0.03),
}
# protocol mix per family: (P(TCP), P(UDP)); the remainder is ICMP
_PROTO_MIX = {"Normal": (0.75, 0.22), "DoS": (0.95, 0.03), "DDoS": (0.55, 0.35), "PortScan": (0.85, 0.05),
              "BruteForce": (1.0, 0.0), "WebAttack": (1.0, 0.0), "Bot": (0.9, 0.1)}
# family -> raw CICIDS2017 label name (so the label-grouping step gets exercised)
_RAW_NAME = {"Normal": "BENIGN", "DoS": "DoS Hulk", "DDoS": "DDoS", "PortScan": "PortScan",
             "BruteForce": "FTP-Patator", "WebAttack": "Web Attack \x96 Brute Force", "Bot": "Bot"}


def generate_synthetic(n_rows: int = 40_000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(_PROFILES)
    shares = np.array([_PROFILES[k][6] for k in names])
    labels = rng.choice(names, size=n_rows, p=shares / shares.sum())

    rows = []
    for lab in labels:
        dur, fwd, bwd, ln, syn, iat, _ = _PROFILES[lab]
        d = rng.lognormal(dur, 1.0)
        f = rng.lognormal(fwd, 0.6)
        b = rng.lognormal(bwd, 0.6)
        pl = rng.lognormal(ln, 0.5)
        p_tcp, p_udp = _PROTO_MIX[lab]
        proto = rng.choice([6, 17, 1], p=[p_tcp, p_udp, max(0.0, 1 - p_tcp - p_udp)])
        rows.append({
            "Protocol": int(proto),
            "Destination Port": int(rng.choice([80, 443, 22, 21, 8080, 53, int(rng.integers(1, 65535))])),
            "Flow Duration": d,
            "Total Fwd Packets": f,
            "Total Backward Packets": b,
            "Total Length of Fwd Packets": f * pl,
            "Total Length of Bwd Packets": b * pl * rng.uniform(0.5, 1.5),
            "Fwd Packet Length Mean": pl,
            "Bwd Packet Length Mean": pl * rng.uniform(0.3, 1.2),
            "Flow Bytes/s": (f + b) * pl / max(d, 1) * 1e6,
            "Flow Packets/s": (f + b) / max(d, 1) * 1e6,
            "Flow IAT Mean": rng.lognormal(iat, 0.8),
            "Flow IAT Std": rng.lognormal(iat - 0.5, 0.8),
            "Fwd IAT Total": d * rng.uniform(0.6, 1.0),
            "SYN Flag Count": float(rng.random() < syn),
            "ACK Flag Count": float(rng.random() < 0.6),
            "PSH Flag Count": float(rng.random() < 0.3),
            "Average Packet Size": pl * rng.uniform(0.9, 1.1),
            "Init_Win_bytes_forward": int(rng.choice([-1, 256, 8192, 29200, 65535])),
            "Active Mean": rng.lognormal(iat - 1, 1.0),
            "Idle Mean": rng.lognormal(iat + 1, 1.0),
            "Label": _RAW_NAME[lab],
        })
    df = pd.DataFrame(rows)
    # inject the dirt real CICIDS2017 has, so the cleaning step has real work to do
    df.loc[df.sample(frac=0.002, random_state=seed).index, "Flow Bytes/s"] = np.inf
    df.loc[df.sample(frac=0.002, random_state=seed + 1).index, "Flow Packets/s"] = np.nan
    df = pd.concat([df, df.sample(frac=0.01, random_state=seed)], ignore_index=True)  # duplicates
    return df
