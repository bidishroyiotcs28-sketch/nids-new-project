"""Central place for paths and settings. Change things here, not all over the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"          # put CICIDS2017 CSV files here
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CORR_THRESHOLD = 0.95        # drop one of two features correlated above this
MAX_ROWS_PER_FILE = 100_000  # sample cap per CSV so Colab doesn't run out of RAM
MIN_CLASS_SAMPLES = 500      # classes rarer than this are dropped (e.g. Heartbleed)

LABEL_COL = "Label"

# CICIDS2017 has 15 raw labels. We group them into a few clear families.
# Matching is done on lowercase substrings, in this order (the dataset has
# encoding glitches in the "Web Attack" names, so exact matching would fail).
# ORDER MATTERS: "ddos" contains the text "dos", so "ddos" must be checked first.
LABEL_GROUPS = [
    ("benign", "Normal"),
    ("ddos", "DDoS"),
    ("dos", "DoS"),             # Hulk, GoldenEye, slowloris, Slowhttptest
    ("portscan", "PortScan"),
    ("patator", "BruteForce"),  # FTP-Patator, SSH-Patator
    ("web attack", "WebAttack"),
    ("bot", "Bot"),
]
NORMAL_CLASS = "Normal"

# ---------------------------------------------------------------------------
# Settings added for the dashboard / advanced-analysis features
# ---------------------------------------------------------------------------
# KNN and SVM get very slow on hundreds of thousands of rows, so they are
# trained on a stratified subsample and scored on a capped test subsample.
SLOW_MODELS = ("KNN", "SVM")
SLOW_TRAIN_MAX = 15_000
SLOW_EVAL_MAX = 10_000

# CV, learning curves and tuning run on a stratified subsample of the TRAIN set.
ANALYSIS_SAMPLE = 8_000
CV_FOLDS = 5
LEARNING_CURVE_SIZES = (0.1, 0.25, 0.5, 0.75, 1.0)
TUNE_TOP_N = 2                 # how many of the best models --tune will tune

# Learning-curve diagnosis thresholds (on accuracy)
OVERFIT_GAP = 0.05             # train - validation above this  -> overfitting
UNDERFIT_SCORE = 0.90          # validation accuracy below this -> underfitting

# Samples saved for the dashboard pages
EXPLORE_SAMPLE = 5_000         # rows used for PCA / K-Means page
ANALYTICS_SAMPLE = 20_000      # test rows used for the analytics page
SHAP_BACKGROUND = 100          # background rows for SHAP / occlusion explanations

# Protocol numbers used by CICIDS ("Protocol" column, when present)
PROTOCOL_NAMES = {6: "TCP", 17: "UDP", 1: "ICMP"}
