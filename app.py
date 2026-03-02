"""
╔══════════════════════════════════════════════════════════════════════╗
║        Monte Carlo Portfolio Risk Pricer — Streamlit Edition         ║
║  VaR · Expected Shortfall · Cholesky GBM · 10 000 trajectoires       ║
╚══════════════════════════════════════════════════════════════════════╝

Stack : Streamlit · NumPy (vectorisé) · yfinance · Plotly · SciPy

Fix v3 : st.form() — la validation Yahoo ne se déclenche qu'au clic
         "Lancer la simulation", jamais à chaque frappe de touche.
"""

from __future__ import annotations
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import gaussian_kde

# ══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Monte Carlo Risk Pricer",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace !important;
    background-color: #080c10 !important;
    color: #d0e8e0 !important;
}
.stApp {
    background: #080c10;
    background-image:
        radial-gradient(ellipse 120% 60% at 80% -10%, rgba(0,163,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 80% 40% at 10% 90%, rgba(0,220,180,0.05) 0%, transparent 55%);
}
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid rgba(0,220,180,0.12) !important;
}
[data-testid="stSidebar"] * { font-family: 'JetBrains Mono', monospace !important; }

input, textarea, select,
[data-baseweb="input"] input,
[data-baseweb="select"] div {
    background-color: #0d1117 !important;
    border: 1px solid rgba(0,220,180,0.2) !important;
    border-radius: 3px !important;
    color: #d0e8e0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
input:focus {
    border-color: #00dcb4 !important;
    box-shadow: 0 0 0 2px rgba(0,220,180,0.14) !important;
}
[data-testid="stNumberInput"] button {
    background: #111820 !important;
    border: 1px solid rgba(0,220,180,0.15) !important;
    color: #00dcb4 !important;
}
[data-testid="stSlider"] .rc-slider-track { background: #00dcb4 !important; }
[data-testid="stSlider"] .rc-slider-handle {
    background: #00dcb4 !important; border-color: #00dcb4 !important;
    box-shadow: 0 0 8px rgba(0,220,180,0.4) !important;
}

/* Form submit button */
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, rgba(0,220,180,0.18), rgba(0,163,255,0.12)) !important;
    border: 1px solid #00dcb4 !important;
    border-radius: 3px !important;
    color: #00dcb4 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    width: 100% !important;
    padding: 0.7rem 1rem !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    box-shadow: 0 0 32px rgba(0,220,180,0.4) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stMetric"] {
    background: #111820 !important;
    border: 1px solid rgba(0,220,180,0.13) !important;
    border-radius: 4px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] {
    color: rgba(208,232,224,0.5) !important;
    font-size: 10px !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.35rem !important;
    font-weight: 800 !important;
}
[data-testid="stMetricDelta"] { font-size: 11px !important; }

h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }
h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: #00dcb4 !important;
    letter-spacing: 0.12em !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid rgba(0,220,180,0.12) !important;
    padding-bottom: 6px !important;
    margin-top: 1.4rem !important;
}
.stSuccess { background:rgba(0,220,180,0.07) !important; border:1px solid rgba(0,220,180,0.3) !important; border-radius:3px !important; }
.stError   { background:rgba(255,62,94,0.07)  !important; border:1px solid rgba(255,62,94,0.3)  !important; border-radius:3px !important; }
.stWarning { background:rgba(255,153,0,0.07)  !important; border:1px solid rgba(255,153,0,0.3)  !important; border-radius:3px !important; }
.stInfo    { background:rgba(0,163,255,0.07)   !important; border:1px solid rgba(0,163,255,0.25) !important; border-radius:3px !important; }
[data-testid="stExpander"] {
    background: #111820 !important;
    border: 1px solid rgba(0,220,180,0.12) !important;
    border-radius: 4px !important;
}
[data-testid="stForm"] { border: none !important; padding: 0 !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080c10; }
::-webkit-scrollbar-thumb { background: rgba(0,220,180,0.2); border-radius:3px; }
hr { border-color: rgba(0,220,180,0.12) !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════

ANNUALIZATION  = {"1d": 252, "1wk": 52, "1mo": 12}
GRAN_LABELS    = {"1d": "Daily (1d)", "1wk": "Weekly (1wk)", "1mo": "Monthly (1mo)"}
HISTORY_YEARS  = 3
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
DEFAULT_WEIGHTS = [35.0,   25.0,   20.0,   15.0,    5.0  ]

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,12,16,0.6)",
    font=dict(family="'JetBrains Mono', monospace", color="#d0e8e0", size=11),
    margin=dict(l=60, r=20, t=36, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,220,180,0.15)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(0,220,180,0.07)", linecolor="rgba(0,220,180,0.15)",
               tickcolor="rgba(0,220,180,0.3)", zerolinecolor="rgba(0,220,180,0.1)"),
    yaxis=dict(gridcolor="rgba(0,220,180,0.07)", linecolor="rgba(0,220,180,0.15)",
               tickcolor="rgba(0,220,180,0.3)", zerolinecolor="rgba(0,220,180,0.1)"),
)

# ══════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.session_state.results = None

# ══════════════════════════════════════════════════════════════════════
#  QUANT ENGINE
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def validate_ticker_cached(ticker: str) -> dict:
    """
    Validation via yf.download — même méthode que la simulation.
    Cache 1h : un ticker valide ne sera jamais re-téléchargé inutilement.
    """
    try:
        raw = yf.download(
            ticker, period="5d", interval="1d",
            auto_adjust=True, progress=False, threads=False,
        )
        if raw is None or raw.empty:
            return {
                "valid": False,
                "error": (
                    f"'{ticker}' introuvable. "
                    "Tickers EU : MC.PA, AIR.PA, SAN.MC, AZN.L, NOVN.SW"
                ),
            }
        close = (raw["Close"][ticker] if isinstance(raw.columns, pd.MultiIndex)
                 else raw["Close"]).dropna()
        if close.empty:
            return {"valid": False, "error": f"'{ticker}' : données vides."}

        last_price = float(close.iloc[-1])
        name, currency, exchange = ticker, "", ""
        try:
            info     = yf.Ticker(ticker).info
            name     = info.get("longName") or info.get("shortName") or ticker
            currency = info.get("currency", "")
            exchange = info.get("exchange", "")
        except Exception:
            pass

        return {"valid": True, "name": name, "last_price": last_price,
                "currency": currency, "exchange": exchange}
    except Exception as e:
        return {"valid": False, "error": str(e)[:120]}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_log_returns(tickers: tuple, granularity: str):
    raw = yf.download(list(tickers), period=f"{HISTORY_YEARS}y",
                      interval=granularity, auto_adjust=True, progress=False, threads=True)
    prices = (raw["Close"].dropna(axis=0, how="all")
              if isinstance(raw.columns, pd.MultiIndex)
              else raw[["Close"]].rename(columns={"Close": tickers[0]}))
    min_obs = {"1d": 60, "1wk": 26, "1mo": 12}[granularity]
    prices  = prices.dropna(axis=1, thresh=min_obs).ffill().dropna()
    valid   = list(prices.columns)
    if not valid:
        raise ValueError("Aucune donnée valide. Vérifiez les tickers.")
    return np.log(prices / prices.shift(1)).dropna().values, valid


def cholesky_params(log_rets):
    mu    = log_rets.mean(axis=0)
    sigma = log_rets.std(axis=0, ddof=1)
    corr  = np.corrcoef(log_rets, rowvar=False) + np.eye(log_rets.shape[1]) * 1e-8
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        ev, evec = np.linalg.eigh(corr)
        L = np.linalg.cholesky(evec @ np.diag(np.clip(ev, 1e-8, None)) @ evec.T)
    return mu, sigma, L


def run_monte_carlo(capital, weights, mu, sigma, L, horizon, n_sims):
    """Simulation GBM 100 % vectorisée — zéro boucle Python."""
    rng    = np.random.default_rng()
    Z_raw  = rng.standard_normal((n_sims, horizon, len(weights)))
    Z_corr = np.einsum("ij,stj->sti", L, Z_raw)
    log_r  = (mu - 0.5 * sigma**2) + sigma * Z_corr
    prices = np.exp(np.cumsum(log_r, axis=1))
    port   = capital * (prices @ weights)
    return np.concatenate([np.full((n_sims, 1), capital), port], axis=1)


def compute_risk_metrics(paths, capital):
    final = paths[:, -1]
    pnl   = final - capital
    v95, v99 = float(-np.percentile(pnl, 5)), float(-np.percentile(pnl, 1))
    e95 = float(-pnl[pnl <= -v95].mean())
    e99 = float(-pnl[pnl <= -v99].mean())
    return dict(
        var95=v95, var99=v99, es95=e95, es99=e99,
        var95_pct=v95/capital*100, var99_pct=v99/capital*100,
        es95_pct=e95/capital*100,  es99_pct=e99/capital*100,
        mean=float(final.mean()), std=float(final.std()),
        min=float(final.min()),   max=float(final.max()),
        prob_loss=float((final < capital).mean() * 100),
    )


def build_surface(paths, n_t=30, n_v=60):
    t_idx = np.linspace(0, paths.shape[1]-1, n_t, dtype=int)
    vg    = np.linspace(paths.min()*0.98, paths.max()*1.02, n_v)
    Z = []
    for ti in t_idx:
        col = paths[:, ti]
        if col.std() < 1e-6:
            d = np.zeros(n_v); d[np.argmin(np.abs(vg - col[0]))] = 1.0
        else:
            d = gaussian_kde(col, bw_method="scott")(vg)
        Z.append(d)
    return t_idx.tolist(), vg.tolist(), Z

# ══════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════

def chart_spaghetti(paths, capital):
    S, T = paths.shape
    x    = list(range(T))
    idx  = np.random.choice(S, size=min(100, S), replace=False)
    fig  = go.Figure()
    for i, pi in enumerate(idx):
        fig.add_trace(go.Scatter(x=x, y=paths[pi].tolist(), mode="lines",
            line=dict(color=f"rgba(0,{155+int(65*i/len(idx))},{170+int(50*i/len(idx))},0.08)", width=0.7),
            showlegend=False, hoverinfo="skip"))
    p5  = np.percentile(paths, 5,  axis=0).tolist()
    p95 = np.percentile(paths, 95, axis=0).tolist()
    fig.add_trace(go.Scatter(x=x+x[::-1], y=p95+p5[::-1], fill="toself",
        fillcolor="rgba(0,163,255,0.07)", line=dict(color="rgba(0,0,0,0)"), name="Bande 5–95%"))
    fig.add_trace(go.Scatter(x=x, y=paths.mean(axis=0).tolist(), mode="lines",
        name="Chemin moyen", line=dict(color="#00dcb4", width=2.5)))
    fig.add_hline(y=capital, line=dict(color="rgba(208,232,224,0.25)", dash="dot", width=1),
                  annotation_text="Capital initial", annotation_font_color="rgba(208,232,224,0.4)")
    fig.update_layout(**PLOTLY_BASE, height=400,
        title=dict(text="01 · Trajectoires du portefeuille (100 chemins GBM)",
                   font=dict(size=12, color="#d0e8e0")),
        xaxis_title="Pas de temps (t)", yaxis_title="Valeur du portefeuille (€)")
    return fig


def chart_histogram(paths, capital, rm):
    idx    = np.random.choice(paths.shape[0], size=min(8000, paths.shape[0]), replace=False)
    finals = paths[idx, -1].tolist()
    fig    = go.Figure()
    fig.add_trace(go.Histogram(x=finals, nbinsx=80, name="Valeur finale",
        marker=dict(color="#00a3ff", opacity=0.5,
                    line=dict(color="rgba(0,163,255,0.06)", width=0.4))))
    for val, pct, color, label in [
        (capital - rm["var95"], rm["var95_pct"], "#ff9900", "VaR 95%"),
        (capital - rm["var99"], rm["var99_pct"], "#ff3e5e", "VaR 99%"),
    ]:
        fig.add_vline(x=val, line=dict(color=color, width=2),
            annotation_text=f"<b>{label}</b><br>−{pct:.2f}%",
            annotation_font=dict(color=color, size=10), annotation_position="top right")
    fig.add_vline(x=capital, line=dict(color="rgba(208,232,224,0.3)", width=1, dash="dot"),
                  annotation_text="P₀", annotation_font_color="rgba(208,232,224,0.4)")
    fig.update_layout(**PLOTLY_BASE, height=400, bargap=0.04,
        title=dict(text="02 · Distribution des valeurs terminales & Tail Risk",
                   font=dict(size=12, color="#d0e8e0")),
        xaxis_title="Valeur finale (€)", yaxis_title="Fréquence")
    return fig


def chart_surface_3d(paths):
    sx, sy, sz = build_surface(paths)
    fig = go.Figure(go.Surface(x=sx, y=sy, z=sz,
        colorscale=[[0.0,"rgba(8,12,16,0)"],[0.15,"rgba(0,163,255,0.25)"],
                    [0.45,"rgba(0,163,255,0.65)"],[0.75,"rgba(0,220,180,0.85)"],
                    [1.0,"rgba(0,220,180,1.0)"]],
        showscale=False, opacity=0.92,
        lighting=dict(ambient=0.6, diffuse=0.7, specular=0.1)))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'JetBrains Mono', monospace", color="#d0e8e0", size=10),
        margin=dict(l=0, r=0, t=36, b=0), height=520,
        title=dict(text="03 · Évolution de la densité de probabilité — Surface 3D KDE",
                   font=dict(size=12, color="#d0e8e0")),
        scene=dict(bgcolor="rgba(8,12,16,0.6)",
            xaxis=dict(title="Pas de temps", gridcolor="rgba(0,220,180,0.07)",
                       linecolor="rgba(0,220,180,0.15)", showbackground=False),
            yaxis=dict(title="Valeur (€)",   gridcolor="rgba(0,163,255,0.07)",
                       linecolor="rgba(0,163,255,0.15)", showbackground=False),
            zaxis=dict(title="Densité",      gridcolor="rgba(208,232,224,0.05)",
                       linecolor="rgba(208,232,224,0.1)", showbackground=False),
            camera=dict(eye=dict(x=-1.5, y=-1.8, z=1.1))))
    return fig

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR  —  st.form() : zéro rerun avant soumission
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:

    st.markdown("""
    <div style='padding:14px 0 18px;border-bottom:1px solid rgba(0,220,180,0.15);margin-bottom:18px;'>
        <div style='font-size:9px;letter-spacing:0.25em;color:#00dcb4;text-transform:uppercase;margin-bottom:6px;'>
            // Quantitative Risk Engine
        </div>
        <div style='font-family:"Syne",sans-serif;font-size:20px;font-weight:800;letter-spacing:-0.02em;
             background:linear-gradient(135deg,#fff 30%,#00dcb4 100%);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.15;'>
            Monte Carlo<br>Risk Pricer
        </div>
        <div style='font-size:10px;color:rgba(208,232,224,0.4);margin-top:5px;'>
            Cholesky GBM · VaR · ES · NumPy vectorisé
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    #  FORM — toute saisie est figée jusqu'au clic sur "Lancer"
    #  Streamlit ne rerun PAS le script tant que submitted est False.
    #  "AAP", "AAPL" incomplet ne déclenchera jamais de validation.
    # ════════════════════════════════════════════════════════════════
    with st.form(key="portfolio_form", clear_on_submit=False):

        # Capital
        st.markdown(
            "<div style='font-size:10px;letter-spacing:0.15em;color:rgba(208,232,224,0.5);"
            "text-transform:uppercase;margin-bottom:4px;'>Capital initial (€)</div>",
            unsafe_allow_html=True)
        capital = st.number_input(
            "Capital", min_value=1_000, max_value=100_000_000,
            value=100_000, step=10_000, label_visibility="collapsed")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Actifs
        st.markdown(
            "<div style='font-size:10px;letter-spacing:0.15em;color:rgba(208,232,224,0.5);"
            "text-transform:uppercase;margin-bottom:2px;'>Actifs · Ticker / Poids (%)</div>",
            unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:10px;color:rgba(208,232,224,0.3);margin-bottom:8px;'>"
            "Tickers EU : MC.PA, AIR.PA, SAN.MC, AZN.L</div>",
            unsafe_allow_html=True)

        n_assets = st.number_input("Nombre d'actifs", min_value=1, max_value=5, value=3, step=1)

        tickers_input = []
        weights_input = []
        for i in range(int(n_assets)):
            c1, c2 = st.columns([3, 2])
            with c1:
                t = st.text_input(
                    f"T{i+1}", value=DEFAULT_TICKERS[i],
                    key=f"t_{i}", placeholder=DEFAULT_TICKERS[i],
                    label_visibility="collapsed").strip().upper()
            with c2:
                w = st.number_input(
                    f"W{i+1}", min_value=0.0, max_value=100.0,
                    value=DEFAULT_WEIGHTS[i], step=0.5,
                    key=f"w_{i}", label_visibility="collapsed")
            tickers_input.append(t)
            weights_input.append(w)

        total_w = sum(weights_input)
        w_ok    = abs(total_w - 100) < 0.1
        w_color = "#00dcb4" if w_ok else ("#ff9900" if total_w < 100 else "#ff3e5e")
        st.markdown(
            f"<div style='text-align:right;font-size:12px;color:{w_color};margin-top:4px;'>"
            f"{'✓' if w_ok else '⚠'} Σ = {total_w:.1f}%</div>",
            unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Paramètres simulation
        st.markdown(
            "<div style='font-size:10px;letter-spacing:0.15em;color:rgba(208,232,224,0.5);"
            "text-transform:uppercase;margin-bottom:8px;'>Paramètres de simulation</div>",
            unsafe_allow_html=True)

        cg, cs = st.columns(2)
        with cg:
            granularity = st.selectbox(
                "Granularité", options=["1d", "1wk", "1mo"],
                format_func=lambda x: GRAN_LABELS[x], index=0)
        with cs:
            n_sims = st.selectbox(
                "Simulations", options=[1_000, 5_000, 10_000, 25_000, 50_000],
                format_func=lambda x: f"{x:,}", index=2)

        horizon = st.slider("Horizon (périodes)", min_value=5, max_value=504, value=252, step=1)
        ann     = ANNUALIZATION[granularity]
        st.markdown(
            f"<div style='font-size:10px;color:rgba(208,232,224,0.35);margin-top:-8px;'>"
            f"≈ {horizon/ann:.1f} an(s) · annualisation ×{ann}</div>",
            unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # ── Bouton submit ─────────────────────────────────────────────
        submitted = st.form_submit_button(
            "▶  Lancer la simulation",
            use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:9px;color:rgba(208,232,224,0.2);letter-spacing:0.08em;'>"
        "NumPy vectorisé · Cholesky GBM<br>yfinance · SciPy KDE · Plotly · Streamlit</div>",
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  HEADER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='padding-bottom:20px;border-bottom:1px solid rgba(0,220,180,0.12);margin-bottom:28px;'>
    <div style='font-size:9px;letter-spacing:0.25em;color:#00dcb4;text-transform:uppercase;margin-bottom:8px;'>
        // Quantitative Risk Engine v3.0 — Streamlit Edition
    </div>
    <h1 style='font-family:"Syne",sans-serif;font-size:2.2rem;font-weight:800;
               background:linear-gradient(135deg,#fff 30%,#00dcb4 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               margin:0;line-height:1.1;'>
        Monte Carlo Portfolio Risk Pricer
    </h1>
    <p style='color:rgba(208,232,224,0.45);font-size:12px;margin-top:8px;'>
        Cholesky-correlated GBM &nbsp;·&nbsp; Value at Risk 95/99% &nbsp;·&nbsp;
        Expected Shortfall (CVaR) &nbsp;·&nbsp; 10 000 trajectoires vectorisées NumPy
    </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  LOGIQUE DE SIMULATION  (uniquement si submitted)
# ══════════════════════════════════════════════════════════════════════

if submitted:
    st.session_state.results = None

    # ── Vérifications de base ─────────────────────────────────────────
    pairs = [(t.strip().upper(), w) for t, w in zip(tickers_input, weights_input) if t.strip()]
    if not pairs:
        st.error("❌ Aucun ticker saisi.")
        st.stop()
    if abs(sum(w for _, w in pairs) - 100) > 0.5:
        st.error(f"❌ Les poids doivent totaliser 100% (actuellement {sum(w for _,w in pairs):.1f}%).")
        st.stop()

    tickers_raw = [p[0] for p in pairs]
    weights_raw = [p[1] for p in pairs]

    # ── Validation Yahoo Finance ──────────────────────────────────────
    st.markdown("### Validation des tickers")
    prog = st.progress(0)

    validated, failed = [], []
    for i, (ticker, weight) in enumerate(zip(tickers_raw, weights_raw)):
        prog.progress((i + 1) / len(tickers_raw),
                      text=f"Vérification de **{ticker}** sur Yahoo Finance…")
        res = validate_ticker_cached(ticker)
        if res["valid"]:
            validated.append((ticker, weight, res))
        else:
            failed.append((ticker, res.get("error", "Introuvable")))

    prog.empty()

    # Cards de validation
    if tickers_raw:
        cols = st.columns(len(tickers_raw))
        for i, ticker in enumerate(tickers_raw):
            with cols[i]:
                ok = next((r for t, _, r in validated if t == ticker), None)
                err = next((e for t, e in failed if t == ticker), None)
                if ok:
                    price_str = f"{ok['last_price']:.2f} {ok.get('currency','')}"
                    name_str  = (ok.get("name") or ticker)[:24]
                    st.markdown(
                        f"<div style='background:rgba(0,220,180,0.07);border:1px solid rgba(0,220,180,0.28);"
                        f"border-radius:3px;padding:8px 10px;font-size:11px;text-align:center;'>"
                        f"<div style='color:#00dcb4;font-weight:600;'>✓ {ticker}</div>"
                        f"<div style='color:rgba(208,232,224,0.45);font-size:10px;margin-top:2px;'>{name_str}</div>"
                        f"<div style='color:#d0e8e0;margin-top:3px;font-size:12px;'>{price_str}</div>"
                        f"</div>", unsafe_allow_html=True)
                elif err:
                    st.markdown(
                        f"<div style='background:rgba(255,62,94,0.07);border:1px solid rgba(255,62,94,0.28);"
                        f"border-radius:3px;padding:8px 10px;font-size:11px;text-align:center;'>"
                        f"<div style='color:#ff3e5e;font-weight:600;'>✗ {ticker}</div>"
                        f"<div style='color:rgba(255,62,94,0.55);font-size:10px;margin-top:2px;'>"
                        f"{err[:45]}</div>"
                        f"</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if not validated:
        st.error("❌ Aucun ticker valide. Exemples : **AAPL**, **MSFT**, **NVDA**, **MC.PA**, **AIR.PA**")
        st.stop()
    if failed:
        st.warning(f"⚠ Tickers ignorés : **{', '.join(t for t,_ in failed)}**")

    # Recalcul des poids sur les tickers valides
    valid_tickers = [t for t, _, _ in validated]
    raw_w         = np.array([w for _, w, _ in validated], dtype=float)
    weights       = raw_w / raw_w.sum()

    # ── Données historiques ───────────────────────────────────────────
    with st.spinner("⬡  Téléchargement de 3 ans d'historique Yahoo Finance…"):
        try:
            log_rets, confirmed = fetch_log_returns(tuple(valid_tickers), granularity)
        except Exception as e:
            st.error(f"❌ Erreur données : {e}")
            st.stop()

    if set(confirmed) != set(valid_tickers):
        missing = set(valid_tickers) - set(confirmed)
        t2w     = dict(zip(valid_tickers, weights))
        new_w   = np.array([t2w[t] for t in confirmed])
        weights = new_w / new_w.sum()
        st.warning(f"⚠ Historique 3 ans insuffisant pour : **{', '.join(missing)}** — ignorés.")

    # ── Monte Carlo ───────────────────────────────────────────────────
    with st.spinner(f"⬡  Décomposition de Cholesky · {n_sims:,} simulations GBM vectorisées…"):
        t0 = time.perf_counter()
        mu, sigma, L = cholesky_params(log_rets)
        paths  = run_monte_carlo(capital, weights, mu, sigma, L, horizon, n_sims)
        rm     = compute_risk_metrics(paths, capital)
        elapsed_ms = (time.perf_counter() - t0) * 1000

    st.session_state.results = dict(
        paths=paths, rm=rm, capital=capital,
        tickers=confirmed, weights=weights,
        log_rets=log_rets, mu=mu, sigma=sigma,
        horizon=horizon, n_sims=n_sims, ann=ann,
        elapsed_ms=elapsed_ms,
    )

# ══════════════════════════════════════════════════════════════════════
#  AFFICHAGE
# ══════════════════════════════════════════════════════════════════════

if st.session_state.results is not None:
    r  = st.session_state.results
    paths, rm, cap = r["paths"], r["rm"], r["capital"]

    # Info bar
    st.markdown(
        f"<div style='display:flex;gap:16px;flex-wrap:wrap;padding:10px 16px;"
        f"background:#111820;border:1px solid rgba(0,220,180,0.12);border-radius:3px;"
        f"font-size:11px;margin-bottom:22px;align-items:center;'>"
        f"<span style='color:#00dcb4;font-weight:600;'>{' · '.join(r['tickers'])}</span>"
        f"<span style='color:rgba(208,232,224,0.2);'>|</span>"
        f"<span style='color:rgba(208,232,224,0.4);'>"
        f"{' / '.join(f'{w*100:.1f}%' for w in r['weights'])}</span>"
        f"<span style='color:rgba(208,232,224,0.2);'>|</span>"
        f"<span style='color:rgba(208,232,224,0.4);'>{r['n_sims']:,} sim · {r['horizon']} périodes</span>"
        f"<span style='color:rgba(208,232,224,0.2);'>|</span>"
        f"<span style='color:#00dcb4;'>⚡ {r['elapsed_ms']:.0f} ms</span>"
        f"</div>", unsafe_allow_html=True)

    # KPIs
    st.markdown("### Métriques de risque")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("VaR 95%",      f"{rm['var95']:,.0f} €",  delta=f"−{rm['var95_pct']:.2f}% capital",  delta_color="inverse")
    with c2: st.metric("VaR 99%",      f"{rm['var99']:,.0f} €",  delta=f"−{rm['var99_pct']:.2f}% capital",  delta_color="inverse")
    with c3: st.metric("ES 95% (CVaR)",f"{rm['es95']:,.0f} €",   delta=f"−{rm['es95_pct']:.2f}% capital",   delta_color="inverse")
    with c4: st.metric("ES 99% (CVaR)",f"{rm['es99']:,.0f} €",   delta=f"−{rm['es99_pct']:.2f}% capital",   delta_color="inverse")

    c5, c6, c7, c8 = st.columns(4)
    with c5: st.metric("Valeur moyenne", f"{rm['mean']:,.0f} €",  delta=f"{rm['mean']-cap:+,.0f} €", delta_color="normal")
    with c6: st.metric("Écart-type σ",   f"{rm['std']:,.0f} €")
    with c7: st.metric("Prob. de perte", f"{rm['prob_loss']:.1f}%", delta="P(P(T)<P₀)", delta_color="off")
    with c8: st.metric("Pire scénario",  f"{rm['min']:,.0f} €",  delta=f"{rm['min']-cap:+,.0f} €",  delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # Graphiques
    st.plotly_chart(chart_spaghetti(paths, cap), use_container_width=True)
    st.plotly_chart(chart_histogram(paths, cap, rm), use_container_width=True)
    st.plotly_chart(chart_surface_3d(paths), use_container_width=True)

    # Matrice corrélation
    if len(r["tickers"]) > 1:
        with st.expander("📐 Matrice de corrélation des log-rendements"):
            corr_df = pd.DataFrame(
                np.corrcoef(r["log_rets"], rowvar=False),
                index=r["tickers"], columns=r["tickers"]).round(3)
            fig_c = go.Figure(go.Heatmap(
                z=corr_df.values, x=r["tickers"], y=r["tickers"],
                colorscale=[[0,"#ff3e5e"],[0.5,"#111820"],[1,"#00dcb4"]],
                zmid=0, zmin=-1, zmax=1,
                text=corr_df.values.round(2), texttemplate="%{text}", showscale=True,
                colorbar=dict(tickfont=dict(color="#d0e8e0"))))
            fig_c.update_layout(**PLOTLY_BASE, height=280,
                title=dict(text="Corrélations historiques (3 ans)", font=dict(size=11)),
                margin=dict(l=60, r=60, t=40, b=40))
            st.plotly_chart(fig_c, use_container_width=True)

    # Paramètres calibrés
    with st.expander("⚙ Paramètres calibrés (μ, σ annualisés par actif)"):
        ann_v = r["ann"]
        df_p  = pd.DataFrame({
            "Ticker":           r["tickers"],
            "Poids (%)":        [round(float(w)*100, 2) for w in r["weights"]],
            "μ annualisé (%)":  [round(float(m)*100*ann_v, 2) for m in r["mu"]],
            "σ annualisée (%)": [round(float(s)*100*np.sqrt(ann_v), 2) for s in r["sigma"]],
        })
        st.dataframe(df_p.style.format({
            "Poids (%)": "{:.2f}", "μ annualisé (%)": "{:.2f}", "σ annualisée (%)": "{:.2f}"}),
            hide_index=True, use_container_width=True)

else:
    # Placeholder
    st.markdown("""
    <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;
                min-height:420px;border:1px dashed rgba(0,220,180,0.12);border-radius:4px;
                color:rgba(208,232,224,0.25);gap:14px;'>
        <div style='font-size:42px;opacity:0.3;'>⬡</div>
        <div style='font-size:11px;letter-spacing:0.18em;text-transform:uppercase;'>
            Configurez le portefeuille dans la barre latérale
        </div>
        <div style='font-size:10px;opacity:0.6;'>puis cliquez sur ▶ Lancer la simulation</div>
    </div>
    """, unsafe_allow_html=True)
