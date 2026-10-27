"""Model Performance dashboard: metric comparison, ROC / PR curves, learning curves,
5-fold cross validation and hyper-parameter tuning results.
All numbers come from files written by run_pipeline.py (nothing is retrained here)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui

ui.setup_page("Model Performance", "📊")
ui.page_header("Model Performance", "How every trained model compares, and whether each one generalises.", "📊")

results = ui.require(ui.load_csv("results.csv"))
cmp_df = results.rename(columns={"Macro Precision": "Precision", "Macro Recall": "Recall", "Macro F1": "F1 Score"})
best = cmp_df.iloc[0]

ui.card_grid([
    ui.card("Best model (macro-F1)", best["Model"], "🏆", ui.ACCENT, delay=0.0),
    ui.card("Accuracy", f"{best['Accuracy']:.2%}", "🎯", ui.NORMAL, delay=0.08),
    ui.card("F1 Score", f"{best['F1 Score']:.3f}", "⚖️", ui.PURPLE, delay=0.16),
    ui.card("ROC-AUC", f"{best['ROC-AUC']:.3f}", "📈", ui.WARN, delay=0.24),
    ui.card("False alarm rate", f"{best['False Alarm Rate']:.2%}", "🔔", ui.ATTACK,
            sub="normal traffic wrongly flagged", delay=0.32),
])

tab_over, tab_roc, tab_pr, tab_lc, tab_cv, tab_tune = st.tabs(
    ["Overview", "ROC curves", "Precision-Recall", "Learning curves", "Cross validation", "Tuning"])

COLOR = {m: ui.COLORWAY[i % len(ui.COLORWAY)] for i, m in enumerate(cmp_df["Model"])}


def bar(metric: str, log: bool = False) -> go.Figure:
    d = cmp_df.sort_values(metric, ascending=log)
    fig = go.Figure(go.Bar(x=d["Model"], y=d[metric], marker_color=[COLOR[m] for m in d["Model"]],
                           text=[f"{v:.3f}" if v < 100 else f"{v:,.0f}" for v in d[metric]], textposition="outside"))
    if log:
        fig.update_yaxes(type="log")
    elif metric in ("Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"):
        lo = max(0.0, float(d[metric].min()) - 0.1)
        fig.update_yaxes(range=[lo, 1.02])
    fig.update_xaxes(tickangle=-25)
    return ui.style_fig(fig, height=340, title=metric)


# ---- overview --------------------------------------------------------------
with tab_over:
    table = cmp_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Train Time (s)",
                    "Prediction Time (ms/1k flows)", "Eval Rows"]]
    ui.show_df(table, hide_index=True, column_config={
        c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.3f")
        for c in ("Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC")
    } | {"Train Time (s)": st.column_config.NumberColumn(format="%.2f"),
         "Prediction Time (ms/1k flows)": st.column_config.NumberColumn(format="%.2f")})
    st.caption("Precision, Recall and F1 are macro-averaged over all classes. ROC-AUC is one-vs-rest macro. "
               "KNN and SVM are trained on a capped subsample and scored on a capped test subsample "
               "(see 'Eval Rows') because they are too slow on the full data.")
    c1, c2 = st.columns(2)
    with c1:
        ui.show_chart(bar("Accuracy"), key="bar_acc")
        ui.show_chart(bar("Recall"), key="bar_rec")
    with c2:
        ui.show_chart(bar("Precision"), key="bar_prec")
        ui.show_chart(bar("F1 Score"), key="bar_f1")
    c3, c4 = st.columns(2)
    with c3:
        ui.show_chart(bar("Train Time (s)", log=True), key="bar_train")
    with c4:
        ui.show_chart(bar("Prediction Time (ms/1k flows)", log=True), key="bar_pred")


# ---- ROC / PR --------------------------------------------------------------
def curve_tab(tab, data, kind: str):
    """kind = 'roc' or 'pr'. Shows all models' macro-average curves, or one model per class."""
    with tab:
        if data is None:
            st.info("Curve data not found. Run the pipeline without `--skip-extras`.")
            return
        x_key, y_key, score_key = ("fpr", "tpr", "auc") if kind == "roc" else ("recall", "precision", "ap")
        xt, yt, st_name = (("False positive rate", "True positive rate", "AUC") if kind == "roc"
                           else ("Recall", "Precision", "AP"))
        mode = st.radio("View", ["All models (macro average)", "One model, per class"], horizontal=True, key=f"mode_{kind}")
        fig = go.Figure()
        if mode.startswith("All"):
            for m, d in data.items():
                c = d["macro"]
                fig.add_trace(go.Scatter(x=c[x_key], y=c[y_key], mode="lines", name=f"{m} ({st_name} {c[score_key]:.3f})",
                                         line=dict(color=COLOR.get(m, ui.ACCENT), width=2.2)))
        else:
            m = st.selectbox("Model", list(data), key=f"model_{kind}")
            for i, (cls, c) in enumerate(data[m]["classes"].items()):
                fig.add_trace(go.Scatter(x=c[x_key], y=c[y_key], mode="lines", name=f"{cls} ({st_name} {c[score_key]:.3f})",
                                         line=dict(color=ui.COLORWAY[i % len(ui.COLORWAY)], width=2.2)))
        if kind == "roc":
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color="#556", dash="dash")))
        fig.update_xaxes(title=xt, range=[0, 1])
        fig.update_yaxes(title=yt, range=[0, 1.02])
        ui.show_chart(ui.style_fig(fig, height=520, title="ROC curves" if kind == "roc" else "Precision-Recall curves"),
                      key=f"fig_{kind}")
        st.caption("Multi-class problem: each class is scored one-vs-rest, then curves are averaged. "
                   + ("The closer to the top-left corner, the better." if kind == "roc"
                      else "The closer to the top-right corner, the better. Useful for rare attack classes."))


curve_tab(tab_roc, ui.load_json("roc_curves.json"), "roc")
curve_tab(tab_pr, ui.load_json("pr_curves.json"), "pr")

# ---- learning curves -------------------------------------------------------
with tab_lc:
    lc = ui.load_json("learning_curves.json")
    if lc is None:
        st.info("Learning-curve data not found. Run the pipeline without `--skip-extras`.")
    else:
        verdict_color = {"Well-fitted": ui.NORMAL, "Overfitting": ui.ATTACK, "Underfitting": ui.WARN}
        summary = pd.DataFrame([{"Model": m, "Train accuracy": d["train_mean"][-1], "Validation accuracy": d["val_mean"][-1],
                                 "Gap": d["gap"], "Verdict": d["diagnosis"]} for m, d in lc.items()])
        ui.show_df(summary, hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="%.3f") for c in ("Train accuracy", "Validation accuracy", "Gap")})
        st.caption("Overfitting = training accuracy is much higher than validation accuracy (large gap). "
                   "Underfitting = even validation accuracy is low. Well-fitted = high and close together. "
                   "Thresholds are set in src/config.py.")
        cols = st.columns(2)
        for i, (m, d) in enumerate(lc.items()):
            sizes = d["train_sizes"]
            fig = go.Figure()
            for key, label, color in (("train", "Training accuracy", ui.ACCENT), ("val", "Validation accuracy", ui.WARN)):
                mean, std = np.array(d[f"{key}_mean"]), np.array(d[f"{key}_std"])
                fig.add_trace(go.Scatter(x=sizes + sizes[::-1], y=list(mean + std) + list((mean - std)[::-1]), fill="toself",
                                         fillcolor="rgba(255,255,255,0.06)",
                                         line=dict(width=0), hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=sizes, y=mean, mode="lines+markers", name=label, line=dict(color=color, width=2.4)))
            fig.update_xaxes(title="training examples")
            fig.update_yaxes(title="accuracy")
            with cols[i % 2]:
                ui.show_chart(ui.style_fig(fig, height=320, title=m), key=f"lc_{m}")
                st.markdown(ui.badge(d["diagnosis"], verdict_color[d["diagnosis"]]) +
                            f" &nbsp;<span class='nids-note'>gap {d['gap']:.3f}</span>", unsafe_allow_html=True)

# ---- cross validation ------------------------------------------------------
with tab_cv:
    cv = ui.load_json("cv_results.json")
    if cv is None:
        st.info("Cross-validation data not found. Run the pipeline without `--skip-extras`.")
    else:
        metric = st.radio("Metric", ["accuracy", "f1_macro"], format_func=lambda s: "Accuracy" if s == "accuracy" else "Macro F1",
                          horizontal=True)
        n_folds = len(next(iter(cv.values()))[metric]["folds"])
        rows = [{"Model": m, **{f"Fold {i + 1}": d[metric]["folds"][i] for i in range(n_folds)},
                 "Mean": d[metric]["mean"], "Std": d[metric]["std"]} for m, d in cv.items()]
        table = pd.DataFrame(rows)
        ui.show_df(table, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.4f") for c in table.columns[1:]})
        fig = go.Figure()
        for m, d in cv.items():
            fig.add_trace(go.Bar(x=[m], y=[d[metric]["mean"]], error_y=dict(type="data", array=[d[metric]["std"]]),
                                 marker_color=COLOR.get(m, ui.ACCENT), name=m, showlegend=False))
            fig.add_trace(go.Scatter(x=[m] * n_folds, y=d[metric]["folds"], mode="markers", showlegend=False,
                                     marker=dict(color="white", size=7, opacity=0.8), hovertemplate="fold score %{y:.4f}<extra></extra>"))
        fig.update_yaxes(title="score", range=[max(0, min(r["Mean"] for r in rows) - 0.15), 1.02])
        ui.show_chart(ui.style_fig(fig, height=400, title=f"{n_folds}-fold cross validation (bars = mean ± std, dots = folds)"), key="cv_fig")
        rows_used = next(iter(cv.values()))["rows_used"]
        st.caption(f"Stratified {n_folds}-fold CV on a {rows_used:,}-row sample of the training set. "
                   "A small std means the model's performance is stable across different data splits.")

# ---- tuning ----------------------------------------------------------------
with tab_tune:
    tune = ui.load_json("tuning_results.json")
    if tune is None:
        st.info("No tuning results yet. Run `python run_pipeline.py --tune` to tune the best models.")
    else:
        for m, info in tune.items():
            st.markdown(f"#### {m}")
            ui.card_grid([
                ui.card("Best CV score (macro-F1)", f"{info['best_cv_f1_macro']:.4f}", "🧪", ui.ACCENT),
                ui.card("Test F1: before → after", f"{info['baseline_macro_f1']:.3f} → {info['tuned_macro_f1']:.3f}", "📊", ui.NORMAL),
                ui.card("Search", info["method"].replace("SearchCV", " search"), "🔎", ui.PURPLE,
                        sub=f"{info['candidates_tried']} of {info['combinations_in_space']} combinations, {info['search_seconds']:.0f}s"),
            ])
            st.markdown("**Best parameters**")
            ui.show_df(pd.DataFrame({"parameter": list(info["best_params"]), "value": [str(v) for v in info["best_params"].values()]}),
                       hide_index=True)
        st.caption("Tuning uses randomized search (or a full grid for tiny search spaces) with 3-fold CV on a training subsample. "
                   "The tuned model is then retrained on the full training set and added to the comparison as '(tuned)'.")
