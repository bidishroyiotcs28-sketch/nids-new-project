"""Network Analytics dashboard: traffic totals, attack share, protocol mix,
attack types and global feature importance.

Data source: the flows you last scored on the Live Detection page, or the saved
test sample from the pipeline run."""
import plotly.graph_objects as go
import streamlit as st

from src import config, insights, ui

ui.setup_page("Network Analytics", "📡")
ui.page_header("Network Analytics", "A traffic-operations view of what the model sees.", "📡")

# ---- choose the data -------------------------------------------------------
uploaded = st.session_state.get("scored")
sample = ui.load_csv("analytics_sample.csv")
sources = []
if uploaded is not None:
    sources.append("Flows from Live Detection")
if sample is not None:
    sources.append("Saved test sample")
ui.require(sources, "No data yet. Run `python run_pipeline.py`, or score a file on the Live Detection page.")
source = st.radio("Data", sources, horizontal=True)
df = uploaded if source.startswith("Flows") else sample
if source.startswith("Flows"):
    st.caption(f"Scored with: {st.session_state.get('scored_model', 'selected model')}")
else:
    st.caption("Unseen test flows scored by the best model from the last pipeline run.")

s = insights.traffic_summary(df)

# ---- cards -----------------------------------------------------------------
if s["has_packets"]:
    cards = [ui.card("Total packets", f"{s['total_packets']:,}", "📦", ui.ACCENT, sub=f"{s['total_flows']:,} flows", delay=0.0),
             ui.card("Normal packets", f"{s['normal_packets']:,}", "✅", ui.NORMAL, sub=f"{s['normal_flows']:,} flows", delay=0.08),
             ui.card("Attack packets", f"{s['attack_packets']:,}", "🚨", ui.ATTACK, sub=f"{s['attack_flows']:,} flows", delay=0.16),
             ui.card("Attack percentage", f"{s['attack_pct_flows']:.1%}", "📈", ui.WARN,
                     sub=f"{s['attack_pct_packets']:.1%} of packets", delay=0.24)]
else:
    cards = [ui.card("Total flows", f"{s['total_flows']:,}", "📦", ui.ACCENT, sub="no packet-count columns in this file", delay=0.0),
             ui.card("Normal flows", f"{s['normal_flows']:,}", "✅", ui.NORMAL, delay=0.08),
             ui.card("Attack flows", f"{s['attack_flows']:,}", "🚨", ui.ATTACK, delay=0.16),
             ui.card("Attack percentage", f"{s['attack_pct_flows']:.1%}", "📈", ui.WARN, delay=0.24)]
top = s["top_attack"]
cards.append(ui.card("Most frequent attack", top[0] if top else "None", "🎯", ui.PURPLE,
                     sub=f"{top[1]:,} flows" if top else "no attacks detected", delay=0.32))
ui.card_grid(cards)

# ---- charts ----------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure(go.Pie(labels=["Normal", "Attack"], values=[s["normal_flows"], s["attack_flows"]], hole=0.62,
                           marker=dict(colors=[ui.NORMAL, ui.ATTACK]), textinfo="percent+label", sort=False))
    ui.show_chart(ui.style_fig(fig, height=340, title="Normal vs attack traffic (flows)"), key="donut")
with c2:
    counts = s["attack_counts"]
    if len(counts):
        fig = go.Figure(go.Bar(x=counts.values[::-1], y=counts.index[::-1], orientation="h", marker_color=ui.ATTACK,
                               text=counts.values[::-1], textposition="outside"))
        fig.update_xaxes(title="flows")
        ui.show_chart(ui.style_fig(fig, height=340, title="Attack types"), key="attack_types")
    else:
        st.success("No attacks in this data.")

c3, c4 = st.columns(2)
with c3:
    pc = s["protocol_counts"]
    if pc is not None:
        colors = {"TCP": ui.ACCENT, "UDP": ui.PURPLE, "ICMP": ui.WARN, "Other": "#556"}
        fig = go.Figure(go.Pie(labels=list(pc.index), values=list(pc.values), hole=0.5,
                               marker=dict(colors=[colors.get(p, "#556") for p in pc.index]), textinfo="percent+label"))
        ui.show_chart(ui.style_fig(fig, height=340, title="Protocol distribution"), key="proto")
    else:
        st.info("No 'Protocol' column in this data, so the TCP/UDP/ICMP split can't be shown. "
                "(The CICIDS2017 'GeneratedLabelledFlows' CSVs include it; 'MachineLearningCVE' does not.)")
with c4:
    pbc = s.get("protocol_by_class")
    if pbc is not None:
        fig = go.Figure()
        for col, color in (("Normal", ui.NORMAL), ("Attack", ui.ATTACK)):
            if col in pbc.columns:
                fig.add_trace(go.Bar(x=pbc.index, y=pbc[col], name=col, marker_color=color))
        fig.update_layout(barmode="stack")
        fig.update_yaxes(title="flows")
        ui.show_chart(ui.style_fig(fig, height=340, title="Traffic class by protocol"), key="proto_class")
    elif "Confidence" in df.columns:
        fig = go.Figure(go.Histogram(x=df["Confidence"], nbinsx=25, marker_color=ui.ACCENT))
        fig.update_xaxes(title="model confidence")
        ui.show_chart(ui.style_fig(fig, height=340, title="Prediction confidence"), key="conf")

# ---- feature importance ----------------------------------------------------
imp = ui.load_csv("feature_importance.csv")
if imp is not None:
    top_n = st.slider("Features to show", 5, min(30, len(imp)), min(15, len(imp)))
    d = imp.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(x=d["importance"], y=d["feature"], orientation="h",
                           marker=dict(color=d["importance"], colorscale=[[0, "#1E3A5F"], [1, ui.ACCENT]])))
    fig.update_xaxes(title="importance")
    ui.show_chart(ui.style_fig(fig, height=max(320, 28 * top_n + 80), title="Feature importance (best model)"), key="imp")
    st.caption("Built-in importance for tree models, permutation importance for the others.")
