# Feature guide: what was added, why, how, and where

This documents the 14 upgrades made on top of the original project. For each one:
**why** it exists, **how** it works, **which files** it touches, and **new dependencies**.

**Note on the original brief:** it mentions a Flask app. This project's front end was
always **Streamlit** (`app.py`), so the dashboard was built on Streamlit instead of adding a
second web framework. The original `app.py` behaviour (upload CSV, predictions, attack
filter, download) is unchanged.

## How the pieces fit

```
run_pipeline.py  --(trains, evaluates, analyses)-->  outputs/*.csv|json|png  +  models/*.joblib
                                                          |
            app.py + pages/*.py  <--(only read these files; nothing is retrained in the UI)
```

The dashboard is fast because all heavy work happens once in `run_pipeline.py`.

## New files at a glance

| File | Role |
|---|---|
| `src/analysis.py` | ROC/PR curves, learning curves, 5-fold CV, hyper-parameter tuning |
| `src/explain.py` | SHAP explanations with an occlusion fallback |
| `src/recommendations.py` | Mitigation playbooks per attack type |
| `src/insights.py` | Pure-data helpers for the pages (traffic stats, PCA, K-Means, pipeline trace) |
| `src/ui.py` | Dark theme CSS, animated cards, progress bars, cached loaders |
| `pages/1_Model_Performance.py` | Features 1-6 |
| `pages/2_Network_Analytics.py` | Feature 12 |
| `pages/3_Data_Exploration.py` | Features 8-9 |
| `pages/4_Prediction_Pipeline.py` | Feature 13 |
| `.streamlit/config.toml` | Dark theme |
| `docs/FEATURES.md` | This file |

Modified files: `app.py`, `run_pipeline.py`, `src/config.py`, `src/models.py`, `src/evaluate.py`,
`src/preprocess.py`, `src/data_loader.py`, `requirements.txt`, `README.md`.
Unchanged logic: the cleaning, label grouping, split, scaling and class-weight handling.

---

## 1. Model Performance dashboard
- **Why:** a single accuracy number hides trade-offs; evaluators want a side-by-side comparison.
- **How:** `evaluate.score_model()` now returns accuracy, macro precision/recall/F1, ROC-AUC (one-vs-rest macro),
  training time and prediction time (ms per 1,000 flows, so models are comparable). The page shows a table with
  progress bars, four metric bar charts (Accuracy, Precision, Recall, F1) and two timing charts (log scale).
- **Files:** `src/evaluate.py`, `run_pipeline.py`, `pages/1_Model_Performance.py`.
- **Dependencies:** `plotly`.

## 2. ROC curves
- **Why:** shows the detection / false-alarm trade-off across all thresholds.
- **How:** multi-class, so each class is scored one-vs-rest; curves are resampled to 101 points and averaged
  into a macro curve. The page shows all models together, or one model per class.
- **Files:** `src/analysis.py` (`roc_pr_curves`), `outputs/roc_curves.json`, `pages/1_Model_Performance.py`.

## 3. Precision-Recall curves
- **Why:** more honest than ROC when attack classes are rare.
- **How:** same one-vs-rest approach; reports average precision (AP).
- **Files:** same as above, `outputs/pr_curves.json`.

## 4. Learning curves
- **Why:** shows whether a model is underfitting, overfitting or well-fitted.
- **How:** `sklearn.learning_curve` (3-fold) on a stratified train subsample (8,000 rows, `config.ANALYSIS_SAMPLE`),
  five training-set sizes. Verdict rule (thresholds in `config.py`): train - validation accuracy > 0.05 is
  **Overfitting**; otherwise validation accuracy < 0.90 is **Underfitting**; else **Well-fitted**.
  These thresholds are a reasonable rule of thumb, not a law, so you can tune them.
- **Files:** `src/analysis.py`, `outputs/learning_curves.json`, `pages/1_Model_Performance.py`.

## 5. 5-fold cross validation
- **Why:** a single train/test split can be lucky; CV shows stability.
- **How:** stratified 5-fold on the same 8,000-row train subsample, scored on accuracy and macro-F1.
  Fold-wise scores, mean and standard deviation are shown in a table and a bar chart with error bars.
- **Files:** `src/analysis.py` (`compute_cv`), `outputs/cv_results.json`.

## 6. Hyper-parameter tuning
- **Why:** shows the models were optimised, not just run with defaults.
- **How:** `python run_pipeline.py --tune` tunes the best 2 models by macro-F1 with `RandomizedSearchCV`
  (a full `GridSearchCV` when the search space is tiny), 3-fold CV, scored on macro-F1. The tuned model is retrained
  on the full training set and added to the comparison as "(tuned)". The page shows best parameters, best CV score
  and the test-F1 change. Search spaces live in `models.PARAM_GRIDS`.
- **Files:** `src/analysis.py` (`tune_model`), `src/models.py`, `run_pipeline.py`, `outputs/tuning_results.json`.
- **Note:** this replaces the old Random-Forest-only `--tune`; the flag still exists.

## 7. More models: KNN, SVM, Gaussian Naive Bayes
- **Why:** a broader comparison (distance-based, margin-based, probabilistic).
- **How:** added to `models.get_models()` and run through the identical preprocessing and evaluation.
  **KNN and SVM are slow on hundreds of thousands of rows**, so they train on a stratified subsample of 15,000
  rows and are scored on a stratified 10,000-row test subsample (`config.SLOW_*`). The `Eval Rows` column in the
  results table shows this, so the comparison stays honest.
- **Files:** `src/models.py`, `src/config.py`.
- **Dependencies:** none beyond scikit-learn (`<1.11`, see requirements; `SVC(probability=True)` is removed in 1.11).

## 8. PCA visualisation
- **Why:** lets people *see* how separable normal and attack traffic are.
- **How:** PCA to 2 components on a stratified 5,000-flow sample of unseen test flows; interactive Plotly scatter
  (zoom, hover, legend toggling), coloured by normal/attack or by attack type.
- **Files:** `src/insights.py` (`pca_projection`), `pages/3_Data_Exploration.py`, `outputs/explore_sample.csv`.

## 9. K-Means clustering (demonstration only)
- **Why:** shows an unsupervised method next to the supervised classifier.
- **How:** K-Means on the scaled features (it never sees labels), with a slider for k. Shows the clusters in PCA
  space, a statistics table (size, dominant class, purity, attack share), silhouette score, an elbow curve and a
  cluster-vs-class heat-map. It does not touch the classifier.
- **Files:** `src/insights.py` (`kmeans_report`, `elbow`), `pages/3_Data_Exploration.py`.

## 10. Explainable AI (SHAP)
- **Why:** security teams need to know *why* a flow was flagged.
- **How:** for tree models (Decision Tree, Random Forest, XGBoost), `shap.TreeExplainer` gives exact per-feature
  contributions for the predicted class. For other models, or if `shap` is not installed, it automatically falls
  back to **occlusion**: how much the predicted-class probability drops when a feature is replaced by its typical
  value. Both are shown as a signed bar chart plus a table; the page states which method was used.
- **Files:** `src/explain.py`, `app.py`, `pages/4_Prediction_Pipeline.py`.
- **Dependencies:** `shap` (optional).

## 11. Attack probability
- **Why:** "DoS" alone does not say how sure the model is.
- **How:** `Confidence` = probability of the predicted class; `Attack Probability` = 1 - P(Normal). The page shows
  "Attack Probability: 96%" or "Normal Traffic Probability: 99%", a confidence figure and per-class progress bars.
  Both columns are added to the predictions table and the CSV download.
- **Files:** `src/insights.py` (`attack_probability`), `app.py`.

## 12. Network Analytics dashboard
- **Why:** an operations view of a batch of traffic.
- **How:** cards for total / normal / attack packets and attack percentage, the most frequent attack type, a donut
  chart, attack-type bars, protocol distribution (TCP/UDP/ICMP), class by protocol, and feature importance.
  Packet totals come from `Total Fwd Packets + Total Backward Packets`; if those columns are missing it falls back
  to flow counts. Data: your last upload from Live Detection, or the saved test sample.
- **Limitation:** the TCP/UDP/ICMP charts need a `Protocol` column. CICIDS2017's `GeneratedLabelledFlows` CSVs have
  it; the `MachineLearningCVE` CSVs do not, and the page says so instead of inventing data.
- **Files:** `src/insights.py` (`traffic_summary`), `pages/2_Network_Analytics.py`.

## 13. Prediction explanation page
- **Why:** makes the whole pipeline understandable in a viva.
- **How:** pick a flow and a model, and walk through 8 tabs: input, preprocessing, encoding, scaling, model,
  prediction, probability, feature importance. Every tab shows real intermediate values from the trained
  scaler / encoder / model (`insights.pipeline_trace`).
- **Files:** `pages/4_Prediction_Pipeline.py`, `src/insights.py`.

## 14. Attack recommendation system
- **Why:** detection is only useful if it leads to action.
- **How:** playbooks for DoS, DDoS, PortScan, BruteForce, WebAttack and Bot (plus Probe/R2L/U2R for NSL-KDD-style
  labels), each with a severity and 4-5 mitigation steps; unknown classes get a generic playbook.
  Shown next to every flagged flow.
- **Files:** `src/recommendations.py`, `app.py`, `pages/4_Prediction_Pipeline.py`.

## UI
Dark theme (`.streamlit/config.toml`), responsive card grid (CSS grid, wraps on small screens), fade-in and hover
animations, animated progress bars, icons, Plotly charts with a shared dark style (`src/ui.py`), and the built-in
Streamlit sidebar navigation (one page per file in `pages/`).

## Other changes you should know about
- `best_model.joblib` still has the same keys, plus `background` (reference rows for explanations).
  New `models/all_models.joblib` holds every model so you can switch models in the sidebar. On the full dataset the
  Random Forest makes this file large (hundreds of MB); lower `n_estimators` in `models.py` if disk is tight.
- `prepare()` also returns `X_test_full` (all numeric columns, unscaled) for the analytics page.
- The synthetic data now has a `Protocol` column so every chart has something to show.
- `requirements.txt` adds `plotly` and `shap`, and pins `scikit-learn<1.11`.
