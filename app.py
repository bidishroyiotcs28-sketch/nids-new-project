"""Live Detection page (main page of the dashboard).

    pip install -r requirements.txt
    streamlit run app.py

Run `python run_pipeline.py` first so models/best_model.joblib exists. Try the page
with outputs/demo_upload.csv (unseen test rows saved by the pipeline).

Other pages (Model Performance, Network Analytics, Data Exploration, Prediction
Pipeline) appear automatically in the sidebar from the pages/ folder.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, explain, insights, preprocess, recommendations, ui

ui.setup_page("Live Detection", "🛡️")
ui.page_header("Live Detection", "Upload network flows and see which ones are attacks, how sure the model is, "
               "why, and what to do about it.", "🛡️")

name, model, bundle = ui.model_selector()
scaler, le, features = bundle["scaler"], bundle["label_encoder"], bundle["features"]
class_names = list(le.classes_)
st.sidebar.caption(f"Classes: {', '.join(class_names)}")


@st.cache_resource
def get_explainer(_model, model_name: str):
    """SHAP TreeExplainer for tree models (cached); None -> occlusion fallback."""
    return explain.get_explainer(_model, model_name)


# ---- input -----------------------------------------------------------------
file = st.file_uploader("Upload a CSV of network flows", type="csv")
use_demo = st.checkbox("…or use the built-in demo sample (outputs/demo_upload.csv)", value=False)

df = None
if file is not None:
    df = pd.read_csv(file)
elif use_demo:
    df = ui.load_csv("demo_upload.csv")
    ui.require(df, "demo_upload.csv not found. Run `python run_pipeline.py` first.")

if df is None:
    st.info("Upload a CSV with the same columns the model was trained on, or tick the demo-sample box.")
    st.stop()

try:
    X = preprocess.preprocess_for_inference(df, features, scaler)
except ValueError as e:
    st.error(str(e))
    st.stop()

# ---- predict ---------------------------------------------------------------
has_proba = hasattr(model, "predict_proba")
Xf = pd.DataFrame(X, columns=features)
if has_proba:
    proba = model.predict_proba(Xf)
    pred_idx = proba.argmax(axis=1)
else:
    pred_idx = model.predict(Xf)
    proba = None
pred = le.inverse_transform(pred_idx)

out = df.copy()
out["Prediction"] = pred
if has_proba:
    out["Confidence"] = proba.max(axis=1).round(3)
    normal_col = class_names.index(config.NORMAL_CLASS)
    out["Attack Probability"] = (1 - proba[:, normal_col]).round(3)

# keep for the Network Analytics page
st.session_state["scored"] = out
st.session_state["scored_model"] = name

# ---- summary ---------------------------------------------------------------
n_attack = int((pred != config.NORMAL_CLASS).sum())
cards = [ui.card("Flows analysed", f"{len(out):,}", "📡", ui.ACCENT, delay=0.0),
         ui.card("Flagged as attack", f"{n_attack:,}", "🚨", ui.ATTACK, delay=0.08),
         ui.card("Attack share", f"{n_attack / len(out):.1%}", "📈", ui.WARN, delay=0.16)]
if has_proba:
    cards.append(ui.card("Avg. confidence", f"{out['Confidence'].mean():.1%}", "🎯", ui.NORMAL, delay=0.24))
ui.card_grid(cards)

# ---- predictions by class --------------------------------------------------
counts = pd.Series(pred).value_counts()
fig = go.Figure(go.Bar(x=counts.index, y=counts.values,
                       marker_color=[ui.NORMAL if c == config.NORMAL_CLASS else ui.ATTACK for c in counts.index],
                       text=counts.values, textposition="outside"))
fig.update_yaxes(title="flows")
ui.show_chart(ui.style_fig(fig, height=320, title="Predictions by class"))

# ---- flow table ------------------------------------------------------------
st.subheader("Flows")
only_attacks = st.checkbox("Show only attacks", value=True)
view = out[out["Prediction"] != config.NORMAL_CLASS] if only_attacks else out
ui.show_df(view)
st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv")

# ---- single-flow inspection ------------------------------------------------
st.subheader("🔍 Inspect one flow")
if len(view) == 0:
    st.success("No attacks detected in this file.")
    st.stop()

options = list(view.index[:500])
if len(view) > 500:
    st.caption("Showing the first 500 flows of the current view.")
idx = st.selectbox("Choose a flow", options,
                   format_func=lambda i: f"#{i}  -  {out.loc[i, 'Prediction']}"
                   + (f"  ({out.loc[i, 'Confidence']:.0%})" if has_proba else ""))

if not has_proba:
    st.warning(f"{name} cannot output probabilities, so confidence and explanations are unavailable.")
    st.stop()

pos = out.index.get_loc(idx)
info = insights.attack_probability(proba[pos], class_names)
accent = ui.ATTACK if info["is_attack"] else ui.NORMAL
headline = (f"Attack Probability: {info['p_attack']:.0%}" if info["is_attack"]
            else f"Normal Traffic Probability: {info['p_normal']:.0%}")

left, right = st.columns([1, 1])
with left:
    st.markdown(f'<div class="nids-verdict" style="--accent:{accent}"><div class="nids-note">PREDICTION</div>'
                f'<div class="big">{info["pred_name"]}</div>'
                f'<div class="nids-note">Confidence: <b>{info["confidence"]:.0%}</b> &nbsp;|&nbsp; {headline}</div></div>',
                unsafe_allow_html=True)
    st.markdown("**Class probabilities**")
    order = np.argsort(proba[pos])[::-1][:6]
    st.markdown("".join(ui.progress_bar(class_names[i], proba[pos][i],
                                        ui.NORMAL if class_names[i] == config.NORMAL_CLASS else ui.ATTACK)
                        for i in order), unsafe_allow_html=True)

with right:
    rec = recommendations.get_recommendations(info["pred_name"])
    sev_color = {"Critical": ui.ATTACK, "High": "#FF7A45", "Medium": ui.WARN, "None": ui.NORMAL}.get(rec["severity"], ui.WARN)
    st.markdown(f"**{rec['icon']} Recommended response** &nbsp; {ui.badge('Severity: ' + rec['severity'], sev_color)}",
                unsafe_allow_html=True)
    st.markdown("".join(f'<div class="nids-rec" style="--accent:{sev_color}">{step}</div>' for step in rec["steps"]),
                unsafe_allow_html=True)

# explanation
st.markdown("**Top contributing features**")
try:
    contrib, method = explain.explain_prediction(
        model, name, X.iloc[pos].to_numpy(), bundle["background"], features, info["pred_idx"],
        explainer=get_explainer(model, name))
    c1, c2 = st.columns([1, 1])
    with c1:
        ui.show_chart(ui.contribution_figure(contrib, method, info["pred_name"]))
    with c2:
        shown = contrib.assign(contribution=contrib["contribution"].round(4), value=contrib["value"].round(3))
        ui.show_df(shown[["feature", "contribution", "value", "direction"]].rename(columns={"value": "scaled value"}))
        st.caption(f"Method: {method}. Green bars push the model towards '{info['pred_name']}', red bars away from it."
                   + ("" if explain.HAS_SHAP else " (Install `shap` for exact SHAP values on tree models.)"))
except Exception as e:   # explanations are a bonus; never break detection
    st.info(f"Explanation unavailable for this model: {e}")
