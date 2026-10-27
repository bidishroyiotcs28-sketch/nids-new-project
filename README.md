# Network Attack Detection using Machine Learning

A complete Intrusion Detection System (IDS) project: it learns from labeled network flows
(CICIDS2017) to classify traffic as **Normal, DoS, DDoS, PortScan, BruteForce, WebAttack or Bot**,
compares eight models, explains its decisions, and ships a dark-theme dashboard.

## Quick start

```bash
pip install -r requirements.txt

# 1. Test everything with fake data first (no download, ~3-4 minutes)
python run_pipeline.py --synthetic --fast

# 2. Real run: put the CICIDS2017 CSVs in data/raw/, then
python run_pipeline.py            # add --fast for smaller models, --tune to tune the best models

# 3. Dashboard
streamlit run app.py
```

**Getting the real dataset:** search "CICIDS2017" (University of New Brunswick / Canadian Institute for
Cybersecurity). `MachineLearningCVE` (8 CSVs) works; the `GeneratedLabelledFlows` version also has a
`Protocol` column, which enables the TCP/UDP/ICMP charts. Put the CSVs in `data/raw/`. Each file is sampled
to at most 100,000 rows (`MAX_ROWS_PER_FILE` in `src/config.py`) so it fits in Colab memory.

> The synthetic data only exists so you can run and understand the code. Its numbers are not results.
> Report only numbers from the real dataset.

### Pipeline options

| Command | What it does |
|---|---|
| `python run_pipeline.py --synthetic --fast` | quick test run on fake data |
| `python run_pipeline.py` | real data, full-size models |
| `--tune` | also tune the 2 best models (RandomizedSearchCV) |
| `--skip-extras` | skip ROC/PR curves, cross validation and learning curves (original behaviour) |

## The dashboard (5 pages)

| Page | What you get |
|---|---|
| **Live Detection** (`app.py`) | Upload flows (or use the demo sample), pick a model, see predictions, confidence, attack probability, SHAP/occlusion explanation and mitigation steps for any flow |
| **Model Performance** | Metric comparison, bar charts, ROC and Precision-Recall curves, learning curves (over/under-fit verdicts), 5-fold CV, tuning results |
| **Network Analytics** | Total/normal/attack packets, attack %, protocol mix, attack types, feature importance |
| **Data Exploration** | Interactive PCA scatter (normal vs attack) and a K-Means clustering demo |
| **Prediction Pipeline** | One flow traced through input, preprocessing, encoding, scaling, model, prediction, probability, importance |

Every upgrade is explained (why / how / files / dependencies) in **[docs/FEATURES.md](docs/FEATURES.md)**.

## Project structure

| File | What it does | Study it for |
|---|---|---|
| `src/config.py` | Paths, seed, thresholds, label grouping, analysis settings | Where to change settings |
| `src/data_loader.py` | Loads CICIDS2017 CSVs, or generates fake data | Dataset handling |
| `src/preprocess.py` | Cleaning, label grouping, feature selection, split, scaling | **Leakage-free preprocessing** |
| `src/models.py` | The 8 models, search spaces, slow-model handling | Model choice |
| `src/evaluate.py` | Metrics, ROC-AUC, false alarm rate, plots | **Why accuracy alone is not enough** |
| `src/analysis.py` | ROC/PR curves, learning curves, cross validation, tuning | Model diagnostics |
| `src/explain.py` | SHAP / occlusion explanations | Explainable AI |
| `src/recommendations.py` | Mitigation playbooks | Turning detection into action |
| `src/insights.py` | Traffic stats, PCA, K-Means, pipeline trace | Data behind the pages |
| `src/ui.py` | Dark theme, cards, charts, cached loaders | Dashboard styling |
| `run_pipeline.py` | Runs load, preprocess, train, evaluate, analyse, save | The whole flow |
| `app.py`, `pages/*.py` | The Streamlit dashboard | Deployment |

## The pipeline, step by step

1. **Load.** Read the CSVs, strip the leading spaces from column names.
2. **Clean.** Replace `inf` with NaN, drop NaN rows and duplicates.
3. **Group labels.** 15 raw attack names become 7 families. Classes with fewer than 500 rows are dropped.
4. **Split first, then learn from train only.** Stratified 80/20 split, then remove constant/highly correlated
   features (>0.95) and fit the scaler on the training set only (no leakage).
5. **Handle imbalance** with class weights (`class_weight="balanced"`, or balanced sample weights for XGBoost).
6. **Train** Logistic Regression, Decision Tree, Random Forest, MLP, **KNN, SVM, Gaussian Naive Bayes** and
   XGBoost (if installed). KNN and SVM use capped subsamples because they are slow on large data.
7. **Evaluate** with accuracy, macro precision/recall/F1, **ROC-AUC**, **detection rate**, **false alarm rate**,
   prediction time, per-class reports and confusion matrices.
8. **Analyse** with ROC/PR curves, learning curves and 5-fold cross validation (and tuning with `--tune`).
9. **Explain** with feature importance (global) and SHAP/occlusion (per flow).
10. **Save** the best model by macro-F1 (plus all models) with its scaler, label encoder and feature list.

## What gets produced

`outputs/`: `results.csv`, `model_comparison.png`, `class_distribution.png`, `confusion_<model>.png`,
`report_<model>.txt`, `feature_importance.png/.csv`, `roc_curves.json`, `pr_curves.json`, `learning_curves.json`,
`cv_results.json`, `tuning_results.json` (with `--tune`), `analytics_sample.csv`, `explore_sample.csv`,
`demo_upload.csv`. Models: `models/best_model.joblib`, `models/all_models.joblib`.

## Concepts you should be able to explain in your viva

- **Why macro-F1 and not accuracy?** With ~80% normal traffic, a model that always says "normal" scores 80%
  accuracy and detects nothing. Macro-F1 weights every class equally.
- **Why a false alarm rate?** An IDS that flags 5% of normal traffic is unusable at scale.
- **What is data leakage and where do we avoid it?** Fitting the scaler or feature selection on all data lets the
  test set influence training. See `preprocess.prepare()`.
- **ROC vs Precision-Recall?** ROC can look rosy when negatives dominate; PR focuses on the rare attack class.
- **How do you tell over- from underfitting?** Learning curves: a big train/validation gap is overfitting; both
  low is underfitting.
- **Why cross validation?** One split can be lucky; the standard deviation across folds shows stability.
- **SHAP vs feature importance?** Importance is global (the whole model); SHAP is local (this one flow).
- **Why do tree ensembles do so well here?** Tabular data with mixed scales and non-linear thresholds.
- **Class weights vs SMOTE?** Class weights reweight the loss with no fake rows; SMOTE creates synthetic minority
  samples and must only be applied to the training set.
- **Is K-Means a classifier here?** No. It is unsupervised, never sees labels, and is only for exploration.
- **Limitations:** CICIDS2017 is lab-generated (2017); the model is supervised and cannot detect truly unknown
  attacks (an autoencoder could); KNN/SVM results come from subsamples.

## Ideas to make it your own (do at least two)

- Add a **binary mode** (Normal vs Attack) and compare with multi-class results.
- Add **SMOTE** via `imbalanced-learn` (train set only) and compare against class weights.
- Add an **Autoencoder** trained on normal traffic only for anomaly detection.
- Try a different dataset (UNSW-NB15) and check whether conclusions hold.
- Do **cross-dataset testing**: train on one day of CICIDS, test on another.
- Add LightGBM, or stream test rows into the app to simulate live traffic.

## Suggested report outline

Introduction, related work, dataset, preprocessing, models, results (use `results.csv`, ROC/PR curves, CV and the
confusion matrices), explainability, limitations, conclusion.
