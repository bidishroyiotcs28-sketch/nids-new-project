"""Data Exploration: PCA projection (normal vs attack) and a K-Means clustering demo.
K-Means is unsupervised and for demonstration only; it never replaces the classifier."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config, insights, ui

ui.setup_page("Data Exploration", "🧭")
ui.page_header("Data Exploration", "See the structure of the traffic data in two dimensions.", "🧭")

raw = ui.require(ui.load_csv("explore_sample.csv"))
X, labels = insights.split_explore(raw)
n_classes = labels.nunique()

tab_pca, tab_km = st.tabs(["PCA", "K-Means clustering"])


@st.cache_data
def cached_pca(n_rows: int):
    return insights.pca_projection(X, labels)


@st.cache_data
def cached_kmeans(k: int):
    return insights.kmeans_report(X, labels, k)


@st.cache_data
def cached_elbow(n_rows: int):
    return insights.elbow(X)


# ---- PCA -------------------------------------------------------------------
with tab_pca:
    proj, var = cached_pca(len(X))
    ui.card_grid([ui.card("PC1 variance", f"{var[0]:.1%}", "➊", ui.ACCENT),
                  ui.card("PC2 variance", f"{var[1]:.1%}", "➋", ui.PURPLE),
                  ui.card("Total shown", f"{var.sum():.1%}", "Σ", ui.NORMAL, sub=f"{X.shape[1]} features → 2 dimensions"),
                  ui.card("Points", f"{len(proj):,}", "•", ui.WARN, sub="stratified sample of unseen test flows")])
    by = st.radio("Colour by", ["Normal vs attack", "Attack type"], horizontal=True)
    if by.startswith("Normal"):
        fig = px.scatter(proj, x="PC1", y="PC2", color="Group", hover_data=["Label"], render_mode="webgl",
                         color_discrete_map={"Normal traffic": ui.NORMAL, "Attack traffic": ui.ATTACK}, opacity=0.6)
    else:
        fig = px.scatter(proj, x="PC1", y="PC2", color="Label", render_mode="webgl", opacity=0.65,
                         color_discrete_sequence=ui.COLORWAY)
    fig.update_traces(marker=dict(size=5))
    fig.update_xaxes(title=f"PC1 ({var[0]:.1%} of variance)")
    fig.update_yaxes(title=f"PC2 ({var[1]:.1%} of variance)")
    ui.show_chart(ui.style_fig(fig, height=560, title="PCA projection of network flows (interactive: zoom, hover, click legend)"), key="pca")
    st.caption("PCA squeezes all features into the two directions that keep the most variance. "
               "Overlap here does not mean the classifier fails: it only has 2 of the dimensions to separate classes.")

# ---- K-Means ---------------------------------------------------------------
with tab_km:
    st.markdown(f'<div class="nids-note">⚠️ Demonstration only. K-Means never sees the labels; they are used afterwards '
                f'just to describe what each cluster contains. The supervised classifier is unchanged.</div>', unsafe_allow_html=True)
    k = st.slider("Number of clusters (k)", 2, 10, min(max(n_classes, 2), 10))
    rep = cached_kmeans(k)
    stats = rep["stats"]
    ui.card_grid([ui.card("Clusters", k, "🧩", ui.ACCENT),
                  ui.card("Silhouette score", f"{rep['silhouette']:.3f}", "📐", ui.NORMAL, sub="−1 bad … +1 well separated"),
                  ui.card("Inertia", f"{rep['inertia']:,.0f}", "🎯", ui.PURPLE, sub="lower = tighter clusters"),
                  ui.card("Avg. purity", f"{(stats['Purity'] * stats['Flows']).sum() / stats['Flows'].sum():.1%}", "🧪", ui.WARN,
                          sub="share of the dominant class per cluster")])

    proj, var = cached_pca(len(X))
    plot_df = proj.assign(Cluster=rep["clusters"].astype(str).to_numpy())
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster", hover_data=["Label"], render_mode="webgl",
                     opacity=0.65, color_discrete_sequence=ui.COLORWAY,
                     category_orders={"Cluster": [str(i) for i in range(k)]})
    fig.update_traces(marker=dict(size=5))
    ui.show_chart(ui.style_fig(fig, height=500, title="K-Means clusters (shown in PCA space)"), key="km_scatter")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Cluster statistics**")
        ui.show_df(stats, hide_index=True, column_config={
            "Share": st.column_config.ProgressColumn("Share", min_value=0.0, max_value=1.0, format="%.2f"),
            "Purity": st.column_config.ProgressColumn("Purity", min_value=0.0, max_value=1.0, format="%.2f"),
            "Attack share": st.column_config.ProgressColumn("Attack share", min_value=0.0, max_value=1.0, format="%.2f")})
    with c2:
        el = cached_elbow(len(X))
        fig = go.Figure(go.Scatter(x=el["k"], y=el["inertia"], mode="lines+markers", line=dict(color=ui.ACCENT)))
        fig.add_vline(x=k, line_dash="dash", line_color=ui.WARN)
        fig.update_xaxes(title="k")
        fig.update_yaxes(title="inertia")
        ui.show_chart(ui.style_fig(fig, height=300, title="Elbow curve"), key="elbow")

    ct = rep["crosstab"]
    fig = go.Figure(go.Heatmap(z=ct.values, x=list(ct.columns), y=[f"Cluster {i}" for i in ct.index], colorscale="Tealgrn",
                               zmin=0, zmax=1, text=ct.round(2).values, texttemplate="%{text}"))
    ui.show_chart(ui.style_fig(fig, height=360, title="What each cluster contains (share of cluster by true class)"), key="km_heat")
