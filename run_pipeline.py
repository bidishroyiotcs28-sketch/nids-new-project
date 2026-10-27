"""Run the whole project end to end.

    python run_pipeline.py --synthetic          # test the pipeline with fake data (1-3 min)
    python run_pipeline.py                      # real CICIDS2017 CSVs from data/raw/
    python run_pipeline.py --fast               # smaller models, quicker run
    python run_pipeline.py --tune               # also tune the best models (slower)
    python run_pipeline.py --skip-extras        # skip curves / CV / learning curves (original behaviour)

Everything the dashboard shows is produced here and saved in outputs/ and models/.
"""
import argparse
import time
import joblib
import numpy as np
import pandas as pd

from src import config, data_loader, preprocess, models, evaluate, analysis

# Raw columns kept for the analytics page even when feature selection dropped them
_ANALYTICS_EXTRA_COLS = ["Protocol", "Total Fwd Packets", "Total Backward Packets"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="use generated data instead of CICIDS2017")
    ap.add_argument("--fast", action="store_true", help="smaller models for a quick run")
    ap.add_argument("--tune", action="store_true",
                    help=f"hyper-parameter search on the {config.TUNE_TOP_N} best models")
    ap.add_argument("--skip-extras", action="store_true",
                    help="skip ROC/PR curves, cross validation and learning curves")
    args = ap.parse_args()
    extras = not args.skip_extras

    config.OUT_DIR.mkdir(exist_ok=True)
    config.MODEL_DIR.mkdir(exist_ok=True)

    # 1. LOAD ---------------------------------------------------------------
    print("\n[1/6] Loading data")
    if args.synthetic:
        df = data_loader.generate_synthetic()
        print(f"  generated {len(df):,} synthetic rows")
    else:
        df = data_loader.load_cicids()

    # 2. PREPROCESS ---------------------------------------------------------
    print("\n[2/6] Preprocessing")
    d = preprocess.prepare(df)
    X_train, X_test, y_train, y_test = d["X_train"], d["X_test"], d["y_train"], d["y_test"]
    le, features = d["label_encoder"], d["features"]
    class_names = list(le.classes_)
    n_classes = len(class_names)
    normal_idx = class_names.index(config.NORMAL_CLASS)
    print(f"  train: {X_train.shape}, test: {X_test.shape}, classes: {class_names}")
    evaluate.plot_class_distribution(d["raw_df"])

    # 3. TRAIN + EVALUATE ---------------------------------------------------
    print("\n[3/6] Training models (KNN/SVM use capped subsamples, see config.py)")
    trained, evals, rows = {}, {}, []
    for name, model in models.get_models(fast=args.fast).items():
        t = models.fit(name, model, X_train, y_train)
        row, y_true_e, y_pred, y_proba = evaluate.score_model(
            name, model, X_test, y_test, normal_idx, t, n_classes, eval_subset=models.eval_subset)
        rows.append(row)
        trained[name] = model
        evals[name] = (y_true_e, y_pred, y_proba)
        print(f"  {name:<20} acc={row['Accuracy']:.4f}  macroF1={row['Macro F1']:.4f}  "
              f"AUC={row['ROC-AUC']:.4f}  FAR={row['False Alarm Rate']:.4f}  ({t:.1f}s)")

    # 3b. HYPER-PARAMETER TUNING (optional) ------------------------------------
    tuning = {}
    if args.tune:
        top = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)["Model"].head(config.TUNE_TOP_N)
        print(f"\n  Tuning the best {len(top)} models: {', '.join(top)}")
        for name in top:
            tuned, info = analysis.tune_model(name, trained[name], X_train, y_train)
            tname = f"{name} (tuned)"
            t = models.fit(tname, tuned, X_train, y_train)
            row, y_true_e, y_pred, y_proba = evaluate.score_model(
                tname, tuned, X_test, y_test, normal_idx, t, n_classes, eval_subset=models.eval_subset)
            info["baseline_macro_f1"] = next(r["Macro F1"] for r in rows if r["Model"] == name)
            info["tuned_macro_f1"] = row["Macro F1"]
            tuning[name] = info
            rows.append(row)
            trained[tname] = tuned
            evals[tname] = (y_true_e, y_pred, y_proba)
            print(f"  {name:<20} best CV F1={info['best_cv_f1_macro']:.4f}  params={info['best_params']}")
        analysis.save_json(tuning, config.OUT_DIR / "tuning_results.json")

    # 4. EXTRA ANALYSIS: curves, cross validation, learning curves -----------------
    if extras:
        print("\n[4/6] ROC / PR curves, 5-fold cross validation, learning curves")
        roc_all, pr_all, cv_all, lc_all = {}, {}, {}, {}
        for name, model in trained.items():
            y_true_e, _, y_proba = evals[name]
            if y_proba is not None:
                roc_all[name], pr_all[name] = analysis.roc_pr_curves(y_true_e, y_proba, class_names)
            if "(tuned)" in name:      # CV / learning curves are shown for the base models
                continue
            t0 = time.time()
            cv_all[name] = analysis.compute_cv(name, model, X_train, y_train)
            lc_all[name] = analysis.compute_learning_curve(name, model, X_train, y_train)
            print(f"  {name:<20} CV acc={cv_all[name]['accuracy']['mean']:.4f}"
                  f"±{cv_all[name]['accuracy']['std']:.4f}  fit: {lc_all[name]['diagnosis']}"
                  f"  ({time.time() - t0:.0f}s)")
        analysis.save_json(roc_all, config.OUT_DIR / "roc_curves.json")
        analysis.save_json(pr_all, config.OUT_DIR / "pr_curves.json")
        analysis.save_json(cv_all, config.OUT_DIR / "cv_results.json")
        analysis.save_json(lc_all, config.OUT_DIR / "learning_curves.json")
    else:
        print("\n[4/6] Skipped extra analysis (--skip-extras)")

    # 5. REPORTS AND PLOTS --------------------------------------------------
    print("\n[5/6] Saving results")
    results = pd.DataFrame(rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)
    results.to_csv(config.OUT_DIR / "results.csv", index=False)
    evaluate.plot_model_comparison(results)
    for name in trained:
        y_true_e, y_pred, _ = evals[name]
        evaluate.save_report(name, y_true_e, y_pred, class_names)
        evaluate.plot_confusion(name, y_true_e, y_pred, class_names)
    show = ["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "ROC-AUC",
            "False Alarm Rate", "Train Time (s)", "Prediction Time (ms/1k flows)"]
    print("\n" + results[show].round(4).to_string(index=False))

    # 6. SAVE BEST MODEL + EXPLAIN + DASHBOARD DATA -------------------------
    print("\n[6/6] Best model, explainability and dashboard data")
    best_name = results.loc[0, "Model"]
    best_model = trained[best_name]
    imp = evaluate.plot_feature_importance(best_model, X_test, y_test, features)
    print(f"  best model (by macro F1): {best_name}")
    print("  top features:\n    " + imp.head(8).round(4).to_string().replace("\n", "\n    "))

    rng = np.random.default_rng(config.RANDOM_STATE)
    background = X_train.iloc[rng.choice(len(X_train), size=min(config.SHAP_BACKGROUND, len(X_train)),
                                         replace=False)].to_numpy()
    # same keys as before + `background` (used by the explanation page)
    joblib.dump(dict(model=best_model, scaler=d["scaler"], label_encoder=le, features=features,
                     model_name=best_name, background=background),
                config.MODEL_DIR / "best_model.joblib")
    # every model, so the app can let you switch between them
    joblib.dump(trained, config.MODEL_DIR / "all_models.joblib", compress=3)

    # unseen test rows (unscaled) for trying the app; Protocol/packet columns kept for analytics
    keep_cols = list(features) + [c for c in _ANALYTICS_EXTRA_COLS if c in d["X_test_full"].columns
                                  and c not in features]
    actual = pd.Series(le.inverse_transform(y_test), index=X_test.index)
    full = d["X_test_full"][keep_cols]
    full.iloc[:500].assign(Actual=actual.iloc[:500]).to_csv(config.OUT_DIR / "demo_upload.csv", index=False)

    # analytics sample: test flows scored by the best model
    sub_idx = np.sort(np.random.default_rng(config.RANDOM_STATE).choice(
        len(X_test), size=min(config.ANALYTICS_SAMPLE, len(X_test)), replace=False))
    Xa = X_test.iloc[sub_idx]
    proba = best_model.predict_proba(Xa)
    pred = le.inverse_transform(proba.argmax(axis=1))
    full.iloc[sub_idx].assign(Actual=actual.iloc[sub_idx].to_numpy(), Prediction=pred,
                              Confidence=proba.max(axis=1).round(4)) \
        .to_csv(config.OUT_DIR / "analytics_sample.csv", index=False)

    # PCA / K-Means page sample (scaled features + class name)
    analysis.exploration_sample(X_test, le.inverse_transform(y_test)) \
        .to_csv(config.OUT_DIR / "explore_sample.csv", index=False)

    print(f"\nDone. Look in: {config.OUT_DIR}  and  {config.MODEL_DIR}")
    print("Dashboard:  streamlit run app.py")


if __name__ == "__main__":
    main()
