"""Prediction Pipeline: follow ONE network flow through every stage, from raw input to
the final explanation, using the real trained objects (scaler, model, label encoder)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, explain, insights, recommendations, ui

ui.setup_page("Prediction Pipeline", "🧬")
ui.page_header("Prediction Pipeline", "Exactly what happens to a flow between 'raw CSV row' and 'attack / normal'.", "🧬")

name, model, bundle = ui.model_selector()
features = bundle["features"]

# ---- pick a flow -----------------------------------------------------------
demo = ui.load_csv("demo_upload.csv")
scored = st.session_state.get("scored")
choices = []
if scored is not None:
    choices.append("Flows from Live Detection")
if demo is not None:
    choices.append("Demo sample")
ui.require(choices, "No flows available. Run `python run_pipeline.py` or score a file on the Live Detection page.")
src = st.radio("Take a flow from", choices, horizontal=True)
data = scored if src.startswith("Flows") else demo
row_no = st.number_input("Row number", min_value=0, max_value=len(data) - 1, value=min(3, len(data) - 1), step=1)
raw_row = data.iloc[int(row_no)].drop(labels=["Prediction", "Confidence", "Attack Probability"], errors="ignore")

missing = [f for f in features if f not in raw_row.index]
ui.require(None if missing else raw_row, f"This data is missing required columns, e.g. {missing[:3]}")

t = insights.pipeline_trace(raw_row, bundle, model)
info = t["info"]

# ---- the diagram -----------------------------------------------------------
steps = ["Input Data", "Preprocessing", "Encoding", "Scaling", f"Model ({name})", "Prediction", "Probability", "Feature Importance"]
html = '<div class="nids-steps">'
for i, s in enumerate(steps):
    html += f'<div class="nids-step" style="animation-delay:{i * 0.07:.2f}s"><b>{i + 1}</b>{s}</div>'
    if i < len(steps) - 1:
        html += '<span class="nids-arrow">➜</span>'
st.markdown(html + "</div>", unsafe_allow_html=True)

tabs = st.tabs(["1 · Input", "2 · Preprocessing", "3 · Encoding", "4 · Scaling", "5 · Model",
                "6 · Prediction", "7 · Probability", "8 · Importance"])

with tabs[0]:
    st.markdown(f"One flow with **{t['n_raw_columns']}** columns, as it arrives in the CSV.")
    ui.show_df(raw_row.astype(str).rename("value").to_frame())

with tabs[1]:
    st.markdown("Cleaning and feature selection, using the same rules as training:")
    st.markdown(f"- Keep only the **{t['n_features']}** features the model was trained on "
                f"(constant and highly correlated columns were removed during training).")
    if t["dropped_columns"]:
        st.markdown(f"- Ignored columns: `{', '.join(map(str, t['dropped_columns']))}`")
    st.markdown(f"- Infinite / missing values replaced with 0: **{t['n_fixed_values']}** in this flow.")
    st.markdown("- Duplicate rows are only removed during training, not at prediction time.")
    ui.show_df(t["cleaned"].rename("cleaned value").to_frame())

with tabs[2]:
    st.markdown("All input features are already numeric, so none need encoding. "
                "The **target label** is encoded as an integer for training, then decoded back to a name for display.")
    ui.show_df(t["encoding"], hide_index=True)

with tabs[3]:
    st.markdown("Each feature is standardised with statistics learned **on the training set only**: "
                "`z = (value − mean) / std`. Distance- and gradient-based models (KNN, SVM, MLP, Logistic Regression) need this.")
    ui.show_df(t["scaling"].round(4), hide_index=True)

with tabs[4]:
    st.markdown(f"**{name}** receives the scaled vector of {t['n_features']} numbers.")
    params = {k: str(v) for k, v in list(model.get_params().items())[:15]}
    ui.show_df(pd.DataFrame({"hyper-parameter": list(params), "value": list(params.values())}), hide_index=True)

with tabs[5]:
    accent = ui.ATTACK if info["is_attack"] else ui.NORMAL
    st.markdown(f'<div class="nids-verdict" style="--accent:{accent}"><div class="nids-note">PREDICTED CLASS</div>'
                f'<div class="big">{info["pred_name"]}</div></div>', unsafe_allow_html=True)
    if "Actual" in raw_row.index:
        ok = raw_row["Actual"] == info["pred_name"]
        st.markdown(f"True label in the file: **{raw_row['Actual']}** &nbsp; {ui.badge('correct ✓' if ok else 'wrong ✗', ui.NORMAL if ok else ui.ATTACK)}",
                    unsafe_allow_html=True)
    rec = recommendations.get_recommendations(info["pred_name"])
    st.markdown(f"**{rec['icon']} {rec['title']}**")
    st.markdown("".join(f'<div class="nids-rec" style="--accent:{accent}">{s}</div>' for s in rec["steps"]), unsafe_allow_html=True)

with tabs[6]:
    headline = (f"Attack Probability: {info['p_attack']:.0%}" if info["is_attack"]
                else f"Normal Traffic Probability: {info['p_normal']:.0%}")
    st.markdown(f"### {headline}")
    order = t["proba"].argsort()[::-1]
    st.markdown("".join(ui.progress_bar(t["class_names"][i], t["proba"][i],
                                        ui.NORMAL if t["class_names"][i] == config.NORMAL_CLASS else ui.ATTACK) for i in order),
                unsafe_allow_html=True)
    st.caption("Attack probability = 1 − P(normal): the chance the flow is some kind of attack, even if the model is unsure which one.")

with tabs[7]:
    try:
        explainer = explain.get_explainer(model, name)
        contrib, method = explain.explain_prediction(model, name, t["scaled"], bundle["background"], features,
                                                     info["pred_idx"], explainer=explainer)
        ui.show_chart(ui.contribution_figure(contrib, method, info["pred_name"]), key="local_imp")
        st.caption("Local explanation: which features of THIS flow drove the decision. "
                   "See Network Analytics for the model's global feature importance.")
    except Exception as e:
        st.info(f"Explanation unavailable for this model: {e}")
