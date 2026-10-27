"""Shared look & feel and cached loaders for the Streamlit dashboard.

Nothing in here runs at import time; every page calls `setup_page()` first.
HTML snippets are built as single-line strings on purpose: Streamlit's markdown
turns indented / blank-line-separated HTML into a code block."""
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from . import config

# ---- palette ---------------------------------------------------------------
ACCENT = "#00D4FF"      # cyan
NORMAL = "#2ECC71"      # green  = normal traffic
ATTACK = "#FF4D5E"      # red    = attack traffic
WARN = "#FFB020"
PURPLE = "#A78BFA"
MUTED = "#8B9BB4"
COLORWAY = ["#00D4FF", "#FF4D5E", "#2ECC71", "#FFB020", "#A78BFA", "#FF7AC6", "#4DD0A0", "#7AA2FF", "#FFD166", "#C0C8D8"]

_CSS = """
<style>
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #14213d 0%, #0B1220 55%); }
[data-testid="stSidebar"] { background: #0E1628; border-right: 1px solid #1E2A44; }
.block-container { padding-top: 2rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: .2px; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes grow { from { width: 0; } }
@keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(0,212,255,.35);} 50% { box-shadow: 0 0 0 8px rgba(0,212,255,0);} }
.nids-hero { display:flex; align-items:center; gap:16px; margin-bottom: 6px; animation: fadeUp .5s ease both; }
.nids-hero .ico { font-size: 2.2rem; width:58px; height:58px; display:flex; align-items:center; justify-content:center;
  border-radius:16px; background: linear-gradient(135deg, rgba(0,212,255,.18), rgba(167,139,250,.18)); border:1px solid #1E2A44; animation: pulse 3s infinite; }
.nids-hero h1 { margin:0; font-size: 1.9rem; }
.nids-hero p { margin:0; color:#8B9BB4; }
.nids-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap:14px; margin: 14px 0 6px; }
.nids-card { background: linear-gradient(160deg, #131D33, #0F1830); border:1px solid #1E2A44; border-radius:14px;
  padding:16px 18px; animation: fadeUp .6s ease both; transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease; }
.nids-card:hover { transform: translateY(-3px); border-color: var(--accent); box-shadow: 0 8px 26px rgba(0,0,0,.35), 0 0 0 1px var(--accent); }
.nids-card .lbl { color:#8B9BB4; font-size:.78rem; text-transform:uppercase; letter-spacing:.8px; display:flex; gap:8px; align-items:center; }
.nids-card .val { font-size:1.75rem; font-weight:700; color: var(--accent); margin-top:4px; line-height:1.15; }
.nids-card .sub { color:#8B9BB4; font-size:.8rem; margin-top:4px; }
.nids-bar { margin: 8px 0 12px; }
.nids-bar .row { display:flex; justify-content:space-between; font-size:.85rem; margin-bottom:4px; color:#C8D3E6; }
.nids-bar .track { height:10px; background:#1E2A44; border-radius:8px; overflow:hidden; }
.nids-bar .fill { height:100%; border-radius:8px; animation: grow 1s ease both; }
.nids-badge { display:inline-block; padding:3px 12px; border-radius:999px; font-size:.78rem; font-weight:600; border:1px solid currentColor; }
.nids-verdict { border-radius:16px; padding:20px 24px; border:1px solid var(--accent); background: linear-gradient(160deg, rgba(0,0,0,.15), rgba(0,0,0,0));
  animation: fadeUp .5s ease both; }
.nids-verdict .big { font-size: 2rem; font-weight: 800; color: var(--accent); }
.nids-steps { display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin: 10px 0 18px; }
.nids-step { background:#131D33; border:1px solid #1E2A44; border-radius:12px; padding:8px 14px; font-size:.85rem; animation: fadeUp .6s ease both; }
.nids-step b { color:#00D4FF; margin-right:6px; }
.nids-arrow { color:#3A4B73; font-size:1.1rem; }
.nids-rec { background:#131D33; border-left:4px solid var(--accent); border-radius:8px; padding:10px 16px; margin:6px 0; animation: fadeUp .5s ease both; }
.nids-note { color:#8B9BB4; font-size:.85rem; }
</style>
"""


# ---------------------------------------------------------------------------
# page scaffolding
# ---------------------------------------------------------------------------
def setup_page(title: str, icon: str = "🛡️") -> None:
    """Must be the first Streamlit call on every page."""
    st.set_page_config(page_title=f"{title} | Network Attack Detection", page_icon=icon, layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    st.sidebar.markdown("### 🛡️ NetShield IDS")
    st.sidebar.caption("Machine-learning network attack detection")


def page_header(title: str, subtitle: str = "", icon: str = "🛡️") -> None:
    st.markdown(f'<div class="nids-hero"><div class="ico">{icon}</div><div><h1>{title}</h1><p>{subtitle}</p></div></div>',
                unsafe_allow_html=True)


def card(label: str, value, icon: str = "", accent: str = ACCENT, sub: str = "", delay: float = 0.0) -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (f'<div class="nids-card" style="--accent:{accent}; animation-delay:{delay:.2f}s">'
            f'<div class="lbl"><span>{icon}</span><span>{label}</span></div>'
            f'<div class="val">{value}</div>{sub_html}</div>')


def card_grid(cards) -> None:
    """Render a responsive row of cards (list of card() strings)."""
    st.markdown('<div class="nids-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def progress_bar(label: str, frac: float, color: str = ACCENT, text: str = "") -> str:
    frac = max(0.0, min(1.0, float(frac)))
    text = text or f"{frac:.1%}"
    return (f'<div class="nids-bar"><div class="row"><span>{label}</span><span>{text}</span></div>'
            f'<div class="track"><div class="fill" style="width:{frac * 100:.1f}%; background:{color}"></div></div></div>')


def badge(text: str, color: str = ACCENT) -> str:
    return f'<span class="nids-badge" style="color:{color}">{text}</span>'


# ---------------------------------------------------------------------------
# Streamlit version compatibility: from 1.50 on, `use_container_width=True` is
# deprecated in favour of `width="stretch"`; older versions only know the former.
# ---------------------------------------------------------------------------
def _version_tuple(v: str):
    parts = []
    for p in v.split(".")[:2]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


_NEW_WIDTH_API = _version_tuple(getattr(st, "__version__", "0.0")) >= (1, 50)


def _stretch(fn, *args, **kwargs):
    if _NEW_WIDTH_API:
        return fn(*args, width="stretch", **kwargs)
    return fn(*args, use_container_width=True, **kwargs)


def show_chart(fig, key=None):
    return _stretch(st.plotly_chart, fig, key=key)


def show_df(df, **kwargs):
    return _stretch(st.dataframe, df, **kwargs)


def style_fig(fig: go.Figure, height: int = 360, title: str = "") -> go.Figure:
    """Dark, transparent, consistent look for every Plotly chart."""
    fig.update_layout(template="plotly_dark", height=height, title=dict(text=title, x=0.01, font=dict(size=16)),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,29,51,0.55)",
                      colorway=COLORWAY, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      legend=dict(bgcolor="rgba(0,0,0,0)"), font=dict(color="#C8D3E6"))
    fig.update_xaxes(gridcolor="#1E2A44", zerolinecolor="#1E2A44")
    fig.update_yaxes(gridcolor="#1E2A44", zerolinecolor="#1E2A44")
    return fig


def contribution_figure(df: pd.DataFrame, method: str, class_name: str) -> go.Figure:
    """Horizontal bars of per-feature contributions (green = towards the class, red = away)."""
    d = df.iloc[::-1]
    fig = go.Figure(go.Bar(x=d["contribution"], y=d["feature"], orientation="h",
                           marker_color=[NORMAL if v >= 0 else ATTACK for v in d["contribution"]],
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title="contribution to the predicted class")
    return style_fig(fig, height=max(280, 34 * len(d) + 80), title=f"Why '{class_name}'? - {method}")


# ---------------------------------------------------------------------------
# cached loaders (keyed on file modification time, so re-running the pipeline refreshes them)
# ---------------------------------------------------------------------------
def _mtime(path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_resource
def _bundle(mtime: float):
    return joblib.load(config.MODEL_DIR / "best_model.joblib")


@st.cache_resource
def _all_models(mtime: float):
    return joblib.load(config.MODEL_DIR / "all_models.joblib")


@st.cache_data
def _csv(name: str, mtime: float):
    return pd.read_csv(config.OUT_DIR / name)


@st.cache_data
def _json(name: str, mtime: float):
    with open(config.OUT_DIR / name) as f:
        return json.load(f)


def load_bundle():
    p = config.MODEL_DIR / "best_model.joblib"
    return _bundle(_mtime(p)) if p.exists() else None


def load_all_models() -> dict:
    """All trained models; falls back to just the best one for older runs."""
    p = config.MODEL_DIR / "all_models.joblib"
    if p.exists():
        return _all_models(_mtime(p))
    b = load_bundle()
    return {b["model_name"]: b["model"]} if b else {}


def load_csv(name: str):
    p = config.OUT_DIR / name
    return _csv(name, _mtime(p)) if p.exists() else None


def load_json(name: str):
    p = config.OUT_DIR / name
    return _json(name, _mtime(p)) if p.exists() else None


def require(obj, message: str = "No results found. Run `python run_pipeline.py` first (add `--synthetic` to try it without the dataset)."):
    """Stop the page with a friendly message when a needed artifact is missing."""
    if obj is None or (hasattr(obj, "__len__") and len(obj) == 0):
        st.warning(message)
        st.stop()
    return obj


def model_selector(default_to_best: bool = True):
    """Sidebar model picker shared by the detection and pipeline pages. Returns (name, model, bundle)."""
    bundle = require(load_bundle(), "No trained model found. Run `python run_pipeline.py` first.")
    models_ = load_all_models()
    names = list(models_)
    best = bundle["model_name"]
    idx = names.index(best) if best in names else 0
    choice = st.sidebar.selectbox("Model", names, index=idx, key="selected_model",
                                  help="The best model by macro-F1 is selected by default.")
    st.sidebar.caption(f"Best model: **{best}**")
    return choice, models_[choice], bundle
