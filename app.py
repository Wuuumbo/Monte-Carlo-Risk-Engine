"""
╔══════════════════════════════════════════════════════════════════════╗
║        Monte Carlo Portfolio Risk Pricer — Streamlit Edition         ║
║  VaR · Expected Shortfall · Cholesky GBM · 10 000 trajectoires       ║
╚══════════════════════════════════════════════════════════════════════╝

Stack : Streamlit · NumPy (vectorisé) · yfinance · Plotly · SciPy
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
#  PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Monte Carlo Risk Pricer",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace !important;
    background-color: #080c10 !important;
    color: #d0e8e0 !important;
}

/* ── App background ── */
.stApp {
    background: #080c10;
    background-image:
        radial-gradient(ellipse 120% 60% at 80% -10%, rgba(0,163,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 80% 40% at 10% 90%, rgba(0,220,180,0.05) 0%, transparent 55%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid rgba(0,220,180,0.12) !important;
}
[data-testid="stSidebar"] * { font-family: 'JetBrains Mono', monospace !important; }

/* ── Inputs ── */
input, textarea, [data-baseweb="input"] input,
[data-baseweb="select"] div {
    background-color: #0d1117 !important;
    border: 1px solid rgba(0,220,180,0.18) !important;
    border-radius: 3px !important;
    color: #d0e8e0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
input:focus { border-color: #00dcb4 !important; box-shadow: 0 0 0 2px rgba(0,220,180,0.12) !important; }

/* ── Number input arrows ── */
[data-testid="stNumberInput"] button {
    background: #111820 !important;
    border: 1px solid rgba(0,220,180,0.15) !important;
    color: #00dcb4 !important;
}

/* ── Slider ── */
[data-testid="stSlider"] .rc-slider-track { background: #00dcb4 !important; }
[data-testid="stSlider"] .rc-slider-handle {
    background: #00dcb4 !important;
    border-color: #00dcb4 !important;
    box-shadow: 0 0 8px rgba(0,220,180,0.5) !important;
}

/* ── Select ── */
[data-baseweb="select"] {
    background: #0d1117 !important;
    border-color: rgba(0,220,180,0.18) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,220,180,0.15), rgba(0,163,255,0.1)) !important;
    border: 1px solid #00dcb4 !important;
    border-radius: 3px !important;
    color: #00dcb4 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
    width: 100% !important;
    padding: 0.6rem 1rem !important;
}
.stButton > button:hover {
    box-shadow: 0 0 24px rgba(0,220,180,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #111820 !important;
    border: 1px solid rgba(0,220,180,0.12) !important;
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
    font-size: 1.4rem !important;
    font-weight: 800 !important;
}
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* ── Section headers ── */
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

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #111820 !important;
    border: 1px solid rgba(0,220,180,0.12) !important;
    border-radius: 4px !important;
}

/* ── Success / Error / Info ── */
.stSuccess {
    background: rgba(0,220,180,0.07) !important;
    border: 1px solid rgba(0,220,180,0.30) !important;
    color: #00dcb4 !important;
    border-radius: 3px !important;
}
.stError {
    background: rgba(255,62,94,0.07) !important;
    border: 1px solid rgba(255,62,94,0.30) !important;
    border-radius: 3px !important;
}
.stWarning {
    background: rgba(255,153,0,0.07) !important;
    border: 1px solid rgba(255,153,0,0.30) !important;
    border-radius: 3px !important;
}
.stInfo {
    background: rgba(0,163,255,0.07) !important;
    border: 1px solid rgba(0,163,255,0.25) !important;
    border-radius: 3px !important;
}

/* ── Divider ── */
hr { border-color: rgba(0,220,180,0.12) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080c10; }
::-webkit-scrollbar-thumb { background: rgba(0,220,180,0.2); border-radius: 3px; }

/* ── KPI colors ── */
.var95  { color: #ff9900 !important; }
.var99  { color: #ff3e5e !important; }
.es     { color: #ff3e5e !important; }
.good   { color: #00dcb4 !important; }
.blue   { color: #00a3ff !important; }

/* ── Feedback pills for ticker validation ── */
.pill-valid {
    display:inline-block; padding:3px 10px; border-radius:2px; font-size:11px;
    background:rgba(0,220,180,0.08); border:1px solid rgba(0,220,180,0.3); color:#00dcb4;
}
.pill-invalid {
    display:inline-block; padding:3px 10px; border-radius:2px; font-size:11px;
    background:rgba(255,62,94,0.08); border:1px solid rgba(255,62,94,0.3); color:#ff3e5e;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════

ANNUALIZATION = {"1d": 252, "1wk": 52, "1mo": 12}
GRAN_LABELS   = {"1d": "Daily (1d)", "1wk": "Weekly (1wk)", "1mo": "Monthly (1mo)"}
HISTORY_YEARS = 3

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,12,16,0.6)",
    font=dict(family="'JetBrains Mono', monospace", color="#d0e8e0", size=11),
    margin=dict(l=60, r=20, t=30, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,220,180,0.15)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(0,220,180,0.07)", linecolor="rgba(0,220,180,0.15)",
               tickcolor="rgba(0,220,180,0.3)", zerolinecolor="rgba(0,220,180,0.1)"),
    yaxis=dict(gridcolor="rgba(0,220,180,0.07)", linecolor="rgba(0,220,180,0.15)",
               tickcolor="rgba(0,220,180,0.3)", zerolinecolor="rgba(0,220,180,0.1)"),
)


# ══════════════════════════════════════════════════════════════════════
#  QUANT ENGINE
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_and_validate(ticker: str, granularity: str) -> dict:
    """
    Download 5d of price data via yf.download to validate a ticker.
    Cached 1h so the sidebar doesn't re-request on every rerun.
    """
    try:
        raw = yf.download(
            ticker, period="5d", interval="1d",
            auto_adjust=True, progress=False, threads=False,
        )
        if raw is None or raw.empty:
            return {"valid": False, "error": f"'{ticker}' introuvable sur Yahoo Finance."}

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"][ticker].dropna()
        else:
            close = raw["Close"].dropna()

        if close.empty:
            return {"valid": False, "error": "Données de prix vides — ticker suspendu ?"}

        last_price = float(close.iloc[-1])

        # Best-effort metadata
        name, currency, exchange = ticker, None, None
        try:
            info     = yf.Ticker(ticker).info
            name     = info.get("longName") or info.get("shortName") or ticker
            currency = info.get("currency")
            exchange = info.get("exchange")
        except Exception:
            pass

        return {
            "valid": True,
            "name": name,
            "last_price": last_price,
            "currency": currency or "",
            "exchange": exchange or "",
        }
    except Exception as e:
        return {"valid": False, "error": str(e)[:120]}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_log_returns(tickers: tuple[str, ...], granularity: str) -> tuple[np.ndarray, list[str]]:
    """
    Download Adjusted-Close for all tickers and compute log-returns.
    Returns (log_rets [T×N], valid_tickers).
    """
    raw = yf.download(
        list(tickers),
        period=f"{HISTORY_YEARS}y",
        interval=granularity,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].dropna(axis=0, how="all")
    else:
        prices = raw[["Close"]].copy()
        prices.columns = list(tickers)

    min_obs = {"1d": 60, "1wk": 26, "1mo": 12}[granularity]
    prices  = prices.dropna(axis=1, thresh=min_obs).ffill().dropna()
    valid   = list(prices.columns)

    if not valid:
        raise ValueError("Aucune donnée valide récupérée. Vérifiez les tickers.")

    log_rets = np.log(prices / prices.shift(1)).dropna().values  # (T, N)
    return log_rets, valid


def cholesky_params(log_rets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute drift μ, volatility σ, and Cholesky factor L of the correlation matrix."""
    mu    = log_rets.mean(axis=0)
    sigma = log_rets.std(axis=0, ddof=1)
    corr  = np.corrcoef(log_rets, rowvar=False) + np.eye(log_rets.shape[1]) * 1e-8
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        eigval, eigvec = np.linalg.eigh(corr)
        L = np.linalg.cholesky(eigvec @ np.diag(np.clip(eigval, 1e-8, None)) @ eigvec.T)
    return mu, sigma, L


def run_monte_carlo(
    capital: float, weights: np.ndarray,
    mu: np.ndarray, sigma: np.ndarray, L: np.ndarray,
    horizon: int, n_sims: int,
) -> np.ndarray:
    """
    Fully vectorised GBM — ZERO boucle Python sur les simulations.

    Z_raw   (S, T, N)  ← i.i.d. normales standard
    Z_corr  (S, T, N)  ← corrélées par Cholesky  einsum('ij,stj→sti', L, Z_raw)
    r_it               ← (μᵢ - σᵢ²/2) + σᵢ·Z_corr  (correction Itô)
    P(t)               ← capital · Σ wᵢ · exp(cum_sum(r_it))
    """
    rng    = np.random.default_rng()
    Z_raw  = rng.standard_normal((n_sims, horizon, len(weights)))
    Z_corr = np.einsum("ij,stj->sti", L, Z_raw)
    log_r  = (mu - 0.5 * sigma**2) + sigma * Z_corr
    prices = np.exp(np.cumsum(log_r, axis=1))                       # (S, T, N)
    port   = capital * (prices @ weights)                            # (S, T)
    return np.concatenate([np.full((n_sims, 1), capital), port], axis=1)  # (S, T+1)


def risk_metrics(paths: np.ndarray, capital: float) -> dict:
    final  = paths[:, -1]
    pnl    = final - capital
    var95  = float(-np.percentile(pnl, 5))
    var99  = float(-np.percentile(pnl, 1))
    es95   = float(-pnl[pnl <= -var95].mean())
    es99   = float(-pnl[pnl <= -var99].mean())
    return dict(
        var95=var95, var99=var99, es95=es95, es99=es99,
        var95_pct=var95/capital*100, var99_pct=var99/capital*100,
        es95_pct=es95/capital*100,   es99_pct=es99/capital*100,
        mean=float(final.mean()),    std=float(final.std()),
        min=float(final.min()),      max=float(final.max()),
        prob_loss=float((final < capital).mean() * 100),
    )


def build_surface(paths: np.ndarray, n_t: int = 30, n_v: int = 60):
    """KDE density at n_t decimated time steps → 3D surface data."""
    S, Tp1  = paths.shape
    t_idx   = np.linspace(0, Tp1 - 1, n_t, dtype=int)
    vmin    = paths.min() * 0.98
    vmax    = paths.max() * 1.02
    vgrid   = np.linspace(vmin, vmax, n_v)
    Z = []
    for ti in t_idx:
        col = paths[:, ti]
        if col.std() < 1e-6:
            d = np.zeros(n_v); d[np.argmin(np.abs(vgrid - col[0]))] = 1.0
        else:
            d = gaussian_kde(col, bw_method="scott")(vgrid)
        Z.append(d)
    return t_idx.tolist(), vgrid.tolist(), Z  # Z: [n_t][n_v]


# ══════════════════════════════════════════════════════════════════════
#  PLOTLY CHARTS
# ══════════════════════════════════════════════════════════════════════

def chart_spaghetti(paths: np.ndarray, capital: float) -> go.Figure:
    S, T = paths.shape
    x    = list(range(T))
    idx  = np.random.choice(S, size=min(100, S), replace=False)
    fig  = go.Figure()

    # Sampled paths
    for i, pi in enumerate(idx):
        fig.add_trace(go.Scatter(
            x=x, y=paths[pi].tolist(), mode="lines",
            line=dict(color=f"rgba(0,{160+int(60*i/len(idx))},{180+int(40*i/len(idx))},0.09)", width=0.7),
            showlegend=False, hoverinfo="skip",
        ))

    # Mean path
    mean_path = paths.mean(axis=0).tolist()
    fig.add_trace(go.Scatter(
        x=x, y=mean_path, mode="lines", name="Chemin moyen",
        line=dict(color="#00dcb4", width=2.5),
    ))

    # 5th / 95th percentile band
    p5  = np.percentile(paths, 5,  axis=0).tolist()
    p95 = np.percentile(paths, 95, axis=0).tolist()
    fig.add_trace(go.Scatter(
        x=x+x[::-1], y=p95+p5[::-1], fill="toself",
        fillcolor="rgba(0,163,255,0.06)", line=dict(color="rgba(0,0,0,0)"),
        name="Intervalle 5–95%", showlegend=True,
    ))

    # Capital line
    fig.add_hline(y=capital, line=dict(color="rgba(208,232,224,0.25)", dash="dot", width=1),
                  annotation_text="Capital initial", annotation_font_color="rgba(208,232,224,0.4)")

    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text="Trajectoires du portefeuille (100 chemins GBM)", font=dict(size=12, color="#d0e8e0")),
        xaxis_title="Pas de temps (t)",
        yaxis_title="Valeur du portefeuille (€)",
        height=400,
    )
    return fig


def chart_histogram(paths: np.ndarray, capital: float, rm: dict) -> go.Figure:
    idx    = np.random.choice(paths.shape[0], size=min(8000, paths.shape[0]), replace=False)
    finals = paths[idx, -1].tolist()
    fig    = go.Figure()

    fig.add_trace(go.Histogram(
        x=finals, nbinsx=80, name="Valeur finale",
        marker=dict(color="#00a3ff", opacity=0.5,
                    line=dict(color="rgba(0,163,255,0.08)", width=0.4)),
    ))

    # VaR lines
    for val, pct, color, label in [
        (capital - rm["var95"], rm["var95_pct"], "#ff9900", "VaR 95%"),
        (capital - rm["var99"], rm["var99_pct"], "#ff3e5e", "VaR 99%"),
    ]:
        fig.add_vline(
            x=val, line=dict(color=color, width=1.8, dash="solid"),
            annotation_text=f"<b>{label}</b><br>−{pct:.2f}%",
            annotation_font=dict(color=color, size=10),
            annotation_position="top right",
        )

    fig.add_vline(x=capital, line=dict(color="rgba(208,232,224,0.3)", width=1, dash="dot"),
                  annotation_text="P₀", annotation_font_color="rgba(208,232,224,0.4)")

    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text="Distribution des valeurs terminales — Tail Risk", font=dict(size=12, color="#d0e8e0")),
        xaxis_title="Valeur finale du portefeuille (€)",
        yaxis_title="Fréquence",
        height=400, bargap=0.04,
    )
    return fig


def chart_surface_3d(paths: np.ndarray) -> go.Figure:
    sx, sy, sz = build_surface(paths)
    fig = go.Figure(go.Surface(
        x=sx, y=sy, z=sz,
        colorscale=[
            [0.0,  "rgba(8,12,16,0)"],
            [0.15, "rgba(0,163,255,0.25)"],
            [0.45, "rgba(0,163,255,0.6)"],
            [0.75, "rgba(0,220,180,0.8)"],
            [1.0,  "rgba(0,220,180,1.0)"],
        ],
        showscale=False, opacity=0.92,
        lighting=dict(ambient=0.6, diffuse=0.7, specular=0.1),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'JetBrains Mono', monospace", color="#d0e8e0", size=10),
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text="Évolution de la densité de probabilité dans le temps (KDE 3D)", font=dict(size=12, color="#d0e8e0")),
        scene=dict(
            bgcolor="rgba(8,12,16,0.6)",
            xaxis=dict(title="Pas de temps", gridcolor="rgba(0,220,180,0.07)",
                       linecolor="rgba(0,220,180,0.15)", showbackground=False),
            yaxis=dict(title="Valeur (€)",   gridcolor="rgba(0,163,255,0.07)",
                       linecolor="rgba(0,163,255,0.15)", showbackground=False),
            zaxis=dict(title="Densité",      gridcolor="rgba(208,232,224,0.05)",
                       linecolor="rgba(208,232,224,0.1)", showbackground=False),
            camera=dict(eye=dict(x=-1.5, y=-1.8, z=1.1)),
        ),
        height=520,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR — INPUTS
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── Header ──────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding:16px 0 20px; border-bottom:1px solid rgba(0,220,180,0.15); margin-bottom:20px;'>
        <div style='font-size:9px; letter-spacing:0.25em; color:#00dcb4; text-transform:uppercase; margin-bottom:6px;'>
            // Quantitative Risk Engine
        </div>
        <div style='font-family:"Syne",sans-serif; font-size:20px; font-weight:800; letter-spacing:-0.02em;
             background:linear-gradient(135deg,#fff 30%,#00dcb4 100%);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            Monte Carlo<br>Risk Pricer
        </div>
        <div style='font-size:10px; color:rgba(208,232,224,0.4); margin-top:6px;'>
            Cholesky GBM · VaR · ES · 10 000 chemins
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Capital ─────────────────────────────────────────────────────
    st.markdown("### Capital initial")
    capital = st.number_input(
        "Montant (€)", min_value=1_000, max_value=100_000_000,
        value=100_000, step=10_000, label_visibility="collapsed",
    )
    st.markdown(f"<div style='font-size:11px;color:rgba(208,232,224,0.4);margin-top:-8px;margin-bottom:8px;'>"
                f"{capital:,.0f} €</div>", unsafe_allow_html=True)

    # ── Assets ──────────────────────────────────────────────────────
    st.markdown("### Actifs du portefeuille")
    st.markdown("<div style='font-size:10px;color:rgba(208,232,224,0.4);margin-bottom:8px;'>"
                "Jusqu'à 5 tickers Yahoo Finance. Saisir le ticker puis valider avec Entrée.</div>",
                unsafe_allow_html=True)

    N_ASSETS = st.number_input("Nombre d'actifs", min_value=1, max_value=5, value=3, step=1)

    default_tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
    default_weights = [35.0,   25.0,   20.0,   15.0,    5.0  ]

    tickers_input = []
    weights_input = []
    validation_results = []

    for i in range(int(N_ASSETS)):
        col_t, col_w = st.columns([3, 2])
        with col_t:
            t = st.text_input(
                f"Ticker {i+1}", value=default_tickers[i],
                key=f"ticker_{i}", label_visibility="collapsed",
                placeholder=f"ex: {default_tickers[i]}",
            ).strip().upper()
        with col_w:
            w = st.number_input(
                f"Poids {i+1} (%)", min_value=0.0, max_value=100.0,
                value=default_weights[i], step=0.5,
                key=f"weight_{i}", label_visibility="collapsed",
            )
        tickers_input.append(t)
        weights_input.append(w)

        # Live validation pill
        if t:
            res = fetch_and_validate(t, "1d")
            validation_results.append((t, res))
            if res["valid"]:
                price_str = f"{res['last_price']:.2f} {res['currency']}" if res.get("last_price") else ""
                name_str  = res.get("name", t)[:28]
                st.markdown(
                    f"<div class='pill-valid'>✓ {name_str}"
                    f"{' · ' + price_str if price_str else ''}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='pill-invalid'>✗ {res.get('error','Introuvable')}</div>",
                    unsafe_allow_html=True,
                )
        else:
            validation_results.append((t, {"valid": False, "error": "Vide"}))

    # Weight sum indicator
    total_w = sum(weights_input)
    w_color = "#00dcb4" if abs(total_w - 100) < 0.1 else ("#ff9900" if total_w < 100 else "#ff3e5e")
    w_icon  = "✓" if abs(total_w - 100) < 0.1 else "⚠"
    st.markdown(
        f"<div style='text-align:right; font-size:12px; color:{w_color}; "
        f"margin-top:4px; margin-bottom:12px;'>"
        f"{w_icon} Σ = {total_w:.1f}%</div>",
        unsafe_allow_html=True,
    )

    # ── Simulation params ────────────────────────────────────────────
    st.markdown("### Paramètres de simulation")

    col_g, col_s = st.columns(2)
    with col_g:
        granularity = st.selectbox(
            "Granularité", options=["1d", "1wk", "1mo"],
            format_func=lambda x: GRAN_LABELS[x], index=0,
        )
    with col_s:
        n_sims = st.selectbox(
            "Simulations", options=[1_000, 5_000, 10_000, 25_000, 50_000],
            format_func=lambda x: f"{x:,}", index=2,
        )

    horizon = st.slider(
        "Horizon temporel (périodes)", min_value=5, max_value=504, value=252, step=1,
    )
    ann = ANNUALIZATION[granularity]
    st.markdown(
        f"<div style='font-size:10px;color:rgba(208,232,224,0.4);margin-top:-8px;'>"
        f"≈ {horizon/ann:.1f} an(s) · facteur annualisation = {ann}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Run button ───────────────────────────────────────────────────
    run = st.button("▶  Lancer la simulation", type="primary")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:9px;color:rgba(208,232,224,0.25);letter-spacing:0.1em;'>"
        "NumPy vectorisé · Cholesky GBM<br>yfinance · SciPy KDE · Plotly<br>"
        "Streamlit Cloud</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
#  MAIN AREA — HEADER
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='padding-bottom:20px; border-bottom:1px solid rgba(0,220,180,0.12); margin-bottom:28px;'>
    <div style='font-size:9px; letter-spacing:0.25em; color:#00dcb4; text-transform:uppercase; margin-bottom:8px;'>
        // Quantitative Risk Engine v2.0 — Streamlit Edition
    </div>
    <h1 style='font-family:"Syne",sans-serif; font-size:2.2rem; font-weight:800;
               background:linear-gradient(135deg,#fff 30%,#00dcb4 100%);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               margin:0; line-height:1.1;'>
        Monte Carlo Portfolio Risk Pricer
    </h1>
    <p style='color:rgba(208,232,224,0.45); font-size:12px; margin-top:8px;'>
        Cholesky-correlated GBM &nbsp;·&nbsp; Value at Risk &nbsp;·&nbsp; Expected Shortfall &nbsp;·&nbsp; 10 000 trajectoires vectorisées
    </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  RUN SIMULATION
# ══════════════════════════════════════════════════════════════════════

if run:
    # ── Input validation ─────────────────────────────────────────────
    valid_tickers   = [t for t, r in zip(tickers_input, [v[1] for v in validation_results]) if t and r["valid"]]
    invalid_tickers = [t for t, r in zip(tickers_input, [v[1] for v in validation_results]) if t and not r["valid"]]

    errors = []
    if not valid_tickers:
        errors.append("Aucun ticker valide. Vérifiez les noms (ex: AAPL, MC.PA, AIR.PA).")
    if abs(total_w - 100) > 0.5:
        errors.append(f"Les poids doivent totaliser 100% (actuellement {total_w:.1f}%).")
    if invalid_tickers:
        errors.append(f"Tickers invalides ignorés : {', '.join(invalid_tickers)}")

    for e in errors:
        if "invalides ignorés" in e:
            st.warning(e)
        elif "poids" in e or "Aucun" in e:
            st.error(e)
            st.stop()

    # Re-map weights to valid tickers only
    t2w       = dict(zip(tickers_input, weights_input))
    raw_w     = np.array([t2w[t] for t in valid_tickers], dtype=float)
    weights   = raw_w / raw_w.sum()

    # ── Fetch + compute ──────────────────────────────────────────────
    with st.spinner("⬡  Téléchargement des données Yahoo Finance…"):
        try:
            log_rets, confirmed_tickers = fetch_log_returns(tuple(valid_tickers), granularity)
        except Exception as e:
            st.error(f"Erreur de données : {e}")
            st.stop()

    # Re-align weights if some tickers had no 3y history
    if confirmed_tickers != valid_tickers:
        t2w2   = dict(zip(valid_tickers, weights))
        raw_w2 = np.array([t2w2.get(t, 0) for t in confirmed_tickers])
        weights = raw_w2 / raw_w2.sum()
        if set(confirmed_tickers) != set(valid_tickers):
            missing = set(valid_tickers) - set(confirmed_tickers)
            st.warning(f"Données insuffisantes sur 3 ans pour : {', '.join(missing)}. Ignorés.")

    t0 = time.perf_counter()

    with st.spinner(f"⬡  Décomposition de Cholesky + {n_sims:,} simulations GBM vectorisées…"):
        mu, sigma, L = cholesky_params(log_rets)
        paths = run_monte_carlo(capital, weights, mu, sigma, L, horizon, n_sims)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    rm = risk_metrics(paths, capital)

    # ── Portfolio info bar ───────────────────────────────────────────
    st.markdown(
        f"<div style='display:flex; gap:16px; flex-wrap:wrap; padding:10px 16px; "
        f"background:#111820; border:1px solid rgba(0,220,180,0.12); border-radius:3px; "
        f"font-size:11px; margin-bottom:20px; align-items:center;'>"
        f"<span style='color:rgba(208,232,224,0.35);'>Portefeuille</span>"
        f"<span style='color:#00dcb4;font-weight:600;'>{' · '.join(confirmed_tickers)}</span>"
        f"<span style='color:rgba(208,232,224,0.2);'>|</span>"
        f"<span style='color:rgba(208,232,224,0.35);'>Simulations</span>"
        f"<span style='color:#d0e8e0;'>{n_sims:,}</span>"
        f"<span style='color:rgba(208,232,224,0.2);'>|</span>"
        f"<span style='color:rgba(208,232,224,0.35);'>Horizon</span>"
        f"<span style='color:#d0e8e0;'>{horizon} périodes ({horizon/ann:.1f}a)</span>"
        f"<span style='color:rgba(208,232,224,0.2);'>|</span>"
        f"<span style='color:rgba(208,232,224,0.35);'>Durée</span>"
        f"<span style='color:#00dcb4;'>{elapsed_ms:.0f} ms</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── KPIs ─────────────────────────────────────────────────────────
    st.markdown("### Métriques de risque")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("VaR 95%",
                  f"{rm['var95']:,.0f} €",
                  delta=f"−{rm['var95_pct']:.2f}% du capital",
                  delta_color="inverse")
    with c2:
        st.metric("VaR 99%",
                  f"{rm['var99']:,.0f} €",
                  delta=f"−{rm['var99_pct']:.2f}% du capital",
                  delta_color="inverse")
    with c3:
        st.metric("ES 95% (CVaR)",
                  f"{rm['es95']:,.0f} €",
                  delta=f"−{rm['es95_pct']:.2f}% du capital",
                  delta_color="inverse")
    with c4:
        st.metric("ES 99% (CVaR)",
                  f"{rm['es99']:,.0f} €",
                  delta=f"−{rm['es99_pct']:.2f}% du capital",
                  delta_color="inverse")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        delta_mean = rm['mean'] - capital
        st.metric("Valeur finale moyenne",
                  f"{rm['mean']:,.0f} €",
                  delta=f"{delta_mean:+,.0f} €",
                  delta_color="normal")
    with c6:
        st.metric("Écart-type σ", f"{rm['std']:,.0f} €")
    with c7:
        st.metric("Prob. de perte", f"{rm['prob_loss']:.1f}%",
                  delta="P(P(T) < P₀)", delta_color="off")
    with c8:
        st.metric("Pire cas", f"{rm['min']:,.0f} €",
                  delta=f"{rm['min']-capital:+,.0f} €", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Graphique 1 : Spaghetti ───────────────────────────────────────
    st.markdown("### 01 · Trajectoires du portefeuille (Spaghetti Plot)")
    st.markdown("<div style='font-size:10px;color:rgba(208,232,224,0.3);margin-bottom:8px;'>"
                "100 chemins GBM échantillonnés · Décomposition de Cholesky · Bande percentile 5–95%</div>",
                unsafe_allow_html=True)
    st.plotly_chart(chart_spaghetti(paths, capital), use_container_width=True)

    # ── Graphique 2 : Histogramme ─────────────────────────────────────
    st.markdown("### 02 · Distribution des valeurs terminales & Tail Risk")
    st.markdown("<div style='font-size:10px;color:rgba(208,232,224,0.3);margin-bottom:8px;'>"
                "Distribution de 10 000 valeurs finales simulées · Seuils VaR 95% et 99% marqués</div>",
                unsafe_allow_html=True)
    st.plotly_chart(chart_histogram(paths, capital, rm), use_container_width=True)

    # ── Graphique 3 : Surface 3D ──────────────────────────────────────
    st.markdown("### 03 · Évolution de la densité de probabilité (Surface 3D)")
    st.markdown("<div style='font-size:10px;color:rgba(208,232,224,0.3);margin-bottom:8px;'>"
                "KDE gaussienne à chaque pas de temps · La variance s'évase visuellement (diffusion GBM)</div>",
                unsafe_allow_html=True)
    st.plotly_chart(chart_surface_3d(paths), use_container_width=True)

    # ── Matrice de corrélation ────────────────────────────────────────
    if len(confirmed_tickers) > 1:
        with st.expander("📐 Matrice de corrélation des rendements log"):
            corr_df = pd.DataFrame(
                np.corrcoef(log_rets, rowvar=False),
                index=confirmed_tickers, columns=confirmed_tickers,
            ).round(3)

            fig_corr = go.Figure(go.Heatmap(
                z=corr_df.values, x=confirmed_tickers, y=confirmed_tickers,
                colorscale=[[0,"#ff3e5e"],[0.5,"#111820"],[1,"#00dcb4"]],
                zmid=0, zmin=-1, zmax=1,
                text=corr_df.values.round(2),
                texttemplate="%{text}", showscale=True,
                colorbar=dict(tickcolor="#d0e8e0", tickfont=dict(color="#d0e8e0")),
            ))
            fig_corr.update_layout(
                **PLOTLY_BASE, height=300,
                title=dict(text="Corrélations historiques (log-rendements 3 ans)", font=dict(size=11)),
                margin=dict(l=60, r=60, t=40, b=40),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    # ── Paramètres calibrés ───────────────────────────────────────────
    with st.expander("⚙ Paramètres calibrés (μ, σ annualisés par actif)"):
        mu_ann    = mu * ann
        sigma_ann = sigma * np.sqrt(ann)
        params_df = pd.DataFrame({
            "Ticker":         confirmed_tickers,
            "Poids (%)":      [round(float(w)*100, 2) for w in weights],
            "μ annualisé (%)":  [round(float(m)*100, 2) for m in mu_ann],
            "σ annualisée (%)": [round(float(s)*100, 2) for s in sigma_ann],
        })
        st.dataframe(
            params_df.style.format({
                "Poids (%)": "{:.2f}",
                "μ annualisé (%)": "{:.2f}",
                "σ annualisée (%)": "{:.2f}",
            }),
            hide_index=True, use_container_width=True,
        )

else:
    # ── Placeholder ──────────────────────────────────────────────────
    st.markdown("""
    <div style='display:flex; flex-direction:column; align-items:center; justify-content:center;
                min-height:400px; border:1px dashed rgba(0,220,180,0.12); border-radius:4px;
                color:rgba(208,232,224,0.3); gap:12px;'>
        <div style='font-size:40px; opacity:0.35;'>⬡</div>
        <div style='font-size:11px; letter-spacing:0.15em; text-transform:uppercase;'>
            Configurez le portefeuille dans la barre latérale
        </div>
        <div style='font-size:10px; opacity:0.6;'>
            puis cliquez sur ▶ Lancer la simulation
        </div>
    </div>
    """, unsafe_allow_html=True)
