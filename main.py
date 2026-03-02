"""
Monte Carlo Portfolio Risk Pricer - Backend API
================================================
Senior Quant Dev implementation using vectorized NumPy (zero Python loops).
Calculates VaR & Expected Shortfall via Cholesky-decomposed correlated GBM paths.

Author  : Monte Carlo Risk Engine
Stack   : FastAPI + NumPy + yfinance
Deployed: Vercel Serverless Functions
"""

from __future__ import annotations

import logging
import time
from typing import Literal

import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Monte Carlo Portfolio Risk Pricer",
    description="VaR & ES computation via Cholesky-decomposed correlated GBM paths.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────────

class SimulationRequest(BaseModel):
    """Validated input parameters for the Monte Carlo engine."""

    capital: float = Field(..., gt=0, description="Initial portfolio capital (€)")
    tickers: list[str] = Field(..., min_length=1, max_length=5)
    weights: list[float] = Field(..., min_length=1, max_length=5)
    horizon: int = Field(..., ge=5, le=504, description="Number of periods to simulate")
    granularity: Literal["1d", "1wk", "1mo"] = Field("1d")
    n_simulations: int = Field(10_000, ge=1_000, le=50_000)

    @field_validator("tickers")
    @classmethod
    def upper_tickers(cls, v: list[str]) -> list[str]:
        return [t.strip().upper() for t in v]

    @model_validator(mode="after")
    def check_consistency(self) -> "SimulationRequest":
        if len(self.tickers) != len(self.weights):
            raise ValueError("tickers and weights must have the same length.")
        total = sum(self.weights)
        if not (99.0 <= total <= 101.0):
            raise ValueError(f"Weights must sum to 100 (got {total:.2f}).")
        return self


class RiskMetrics(BaseModel):
    var_95: float
    var_99: float
    es_95: float
    es_99: float
    var_95_pct: float
    var_99_pct: float
    es_95_pct: float
    es_99_pct: float
    mean_final: float
    std_final: float
    min_final: float
    max_final: float
    prob_loss: float


class SimulationResponse(BaseModel):
    risk_metrics: RiskMetrics
    # Spaghetti plot: sampled trajectories  shape [100, horizon+1]
    sample_paths: list[list[float]]
    # Histogram: distribution of final portfolio values  shape [n_simulations]
    final_values: list[float]
    # 3D surface: time × value_bins → density  compressed
    surface_x: list[float]          # time steps (decimated)
    surface_y: list[float]          # value axis bins
    surface_z: list[list[float]]    # density  shape [len(x), len(y)]
    # Meta
    tickers: list[str]
    weights: list[float]
    elapsed_ms: float
    annualization_factor: int


# ── Quant Engine ───────────────────────────────────────────────────────────────

ANNUALIZATION = {"1d": 252, "1wk": 52, "1mo": 12}
HISTORY_YEARS = 3  # Rolling lookback window


def fetch_log_returns(
    tickers: list[str], granularity: str
) -> tuple[np.ndarray, list[str]]:
    """
    Download Adjusted-Close prices from Yahoo Finance and compute log-returns.

    Returns
    -------
    log_rets : ndarray shape (T, N)  – T periods × N assets
    valid_tickers : list of tickers that actually downloaded cleanly
    """
    import pandas as pd

    raw = yf.download(
        tickers,
        period=f"{HISTORY_YEARS}y",
        interval=granularity,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex when >1 ticker, flat when single
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].dropna(axis=0, how="all")
    else:
        prices = raw[["Close"]].dropna()
        prices.columns = tickers

    # Keep only tickers with sufficient data
    min_obs = {"1d": 60, "1wk": 26, "1mo": 12}[granularity]
    prices = prices.dropna(axis=1, thresh=min_obs)
    valid = list(prices.columns)

    if not valid:
        raise ValueError("No valid price data retrieved. Check tickers and granularity.")

    prices = prices.ffill().dropna()
    log_rets = np.log(prices / prices.shift(1)).dropna().values  # (T, N)
    return log_rets, valid


def cholesky_correlation(log_rets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-asset drift & volatility plus the Cholesky factor of the
    correlation matrix (NOT covariance – we rescale separately for numerical
    stability and interpretability).

    Returns
    -------
    mu_dt   : (N,) daily drift (mean log-return)
    sigma   : (N,) daily volatility
    L       : (N, N) lower-triangular Cholesky factor of corr matrix
    """
    N = log_rets.shape[1]
    mu_dt = log_rets.mean(axis=0)          # shape (N,)
    sigma = log_rets.std(axis=0, ddof=1)   # shape (N,)

    # Correlation matrix (more stable than covariance for Cholesky)
    corr = np.corrcoef(log_rets, rowvar=False)  # (N, N)

    # Regularise: nudge diagonal to ensure positive-definiteness
    corr += np.eye(N) * 1e-8

    try:
        L = np.linalg.cholesky(corr)  # (N, N) lower-triangular
    except np.linalg.LinAlgError:
        # Fallback: nearest PSD via eigenvalue clipping
        eigval, eigvec = np.linalg.eigh(corr)
        eigval = np.clip(eigval, 1e-8, None)
        corr = eigvec @ np.diag(eigval) @ eigvec.T
        L = np.linalg.cholesky(corr)

    return mu_dt, sigma, L


def run_monte_carlo(
    capital: float,
    weights: np.ndarray,       # (N,)
    mu_dt: np.ndarray,         # (N,)
    sigma: np.ndarray,         # (N,)
    L: np.ndarray,             # (N, N)
    horizon: int,
    n_sims: int,
) -> np.ndarray:
    """
    Fully vectorised GBM simulation – ZERO Python for-loops.

    Geometry
    --------
    Z_raw  : (n_sims, horizon, N)  iid standard normals
    Z_corr : (n_sims, horizon, N)  correlated via Cholesky: Z_corr = Z_raw @ L.T
    r_it   : log-return of asset i at step t = mu_i - 0.5*sigma_i^2 + sigma_i*Z_corr

    Returns
    -------
    port_paths : (n_sims, horizon+1)  portfolio value at each time step
    """
    N = len(weights)
    rng = np.random.default_rng()  # PCG64, thread-safe

    # ── 1. Draw all random numbers at once ─────────────────────────────────
    Z_raw = rng.standard_normal((n_sims, horizon, N))   # (S, T, N)

    # ── 2. Correlate via Cholesky  Z_corr[s,t,:] = L @ Z_raw[s,t,:] ───────
    #       Using einsum avoids explicit Python loops:
    #       'ij,stj->sti'  →  Z_corr[s,t,i] = sum_j L[i,j] * Z_raw[s,t,j]
    Z_corr = np.einsum("ij,stj->sti", L, Z_raw)         # (S, T, N)

    # ── 3. GBM log-return increments ───────────────────────────────────────
    #       drift correction: mu - 0.5*sigma^2  (Itô's lemma)
    drift = mu_dt - 0.5 * sigma ** 2                    # (N,)
    log_ret_increments = drift + sigma * Z_corr          # (S, T, N)  broadcast

    # ── 4. Cumulative sum → price relatives ────────────────────────────────
    cum_log_rets = np.cumsum(log_ret_increments, axis=1) # (S, T, N)
    price_relatives = np.exp(cum_log_rets)               # (S, T, N)  ratio vs t=0

    # ── 5. Aggregate to portfolio value ────────────────────────────────────
    #       Portfolio value at t = capital * sum_i(w_i * P_i(t)/P_i(0))
    #       w is already normalised to 1
    port_at_t = capital * (price_relatives @ weights)    # (S, T)

    # Prepend initial capital column
    init_col = np.full((n_sims, 1), capital)
    port_paths = np.concatenate([init_col, port_at_t], axis=1)  # (S, T+1)

    return port_paths


def compute_risk_metrics(port_paths: np.ndarray, capital: float) -> RiskMetrics:
    """Compute VaR & ES from the distribution of terminal portfolio values."""
    final = port_paths[:, -1]           # (S,)
    pnl   = final - capital             # Profit & Loss

    # VaR: loss at quantile level (positive = loss)
    var_95_val = float(-np.percentile(pnl, 5))
    var_99_val = float(-np.percentile(pnl, 1))

    # ES (CVaR): mean of losses beyond VaR threshold
    es_95_val = float(-pnl[pnl <= -var_95_val].mean())
    es_99_val = float(-pnl[pnl <= -var_99_val].mean())

    return RiskMetrics(
        var_95=round(var_95_val, 2),
        var_99=round(var_99_val, 2),
        es_95=round(es_95_val, 2),
        es_99=round(es_99_val, 2),
        var_95_pct=round(var_95_val / capital * 100, 3),
        var_99_pct=round(var_99_val / capital * 100, 3),
        es_95_pct=round(es_95_val / capital * 100, 3),
        es_99_pct=round(es_99_val / capital * 100, 3),
        mean_final=round(float(final.mean()), 2),
        std_final=round(float(final.std()), 2),
        min_final=round(float(final.min()), 2),
        max_final=round(float(final.max()), 2),
        prob_loss=round(float((final < capital).mean() * 100), 2),
    )


def build_3d_surface(
    port_paths: np.ndarray,
    capital: float,
    n_time_pts: int = 30,
    n_value_bins: int = 60,
) -> tuple[list[float], list[float], list[float, list[float]]]:
    """
    Build the data for the 3D surface plot.

    At each (decimated) time step, compute a KDE/histogram of portfolio values
    to get density estimates → surface Z(time, value) = density.
    """
    from scipy.stats import gaussian_kde

    S, T_plus1 = port_paths.shape
    time_indices = np.linspace(0, T_plus1 - 1, n_time_pts, dtype=int)

    # Global value range across all paths
    global_min = port_paths.min() * 0.98
    global_max = port_paths.max() * 1.02
    value_grid = np.linspace(global_min, global_max, n_value_bins)

    surface_z = []
    for t_idx in time_indices:
        col = port_paths[:, t_idx]
        if col.std() < 1e-6:
            # t=0: all paths identical → Dirac-like spike
            density = np.zeros(n_value_bins)
            density[np.argmin(np.abs(value_grid - col[0]))] = 1.0
        else:
            kde = gaussian_kde(col, bw_method="scott")
            density = kde(value_grid)
        surface_z.append(density.tolist())

    return (
        time_indices.tolist(),
        value_grid.tolist(),
        surface_z,  # shape [n_time_pts, n_value_bins]
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "engine": "numpy-vectorized-cholesky-gbm"}


@app.post("/api/simulate", response_model=SimulationResponse)
def simulate(req: SimulationRequest):
    t0 = time.perf_counter()
    log.info(f"Simulation request: {req.tickers} | {req.n_simulations} paths | horizon={req.horizon}")

    # ── 1. Fetch market data ─────────────────────────────────────────────────
    try:
        log_rets, valid_tickers = fetch_log_returns(req.tickers, req.granularity)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Data fetch error: {e}")

    # Re-align weights to valid tickers (some may have failed)
    ticker_to_weight = dict(zip(req.tickers, req.weights))
    raw_weights = np.array([ticker_to_weight.get(t, 0.0) for t in valid_tickers])
    weights = raw_weights / raw_weights.sum()  # re-normalise

    # ── 2. Cholesky decomposition ─────────────────────────────────────────────
    mu_dt, sigma, L = cholesky_correlation(log_rets)

    # ── 3. Monte Carlo simulation (fully vectorised) ──────────────────────────
    port_paths = run_monte_carlo(
        capital=req.capital,
        weights=weights,
        mu_dt=mu_dt,
        sigma=sigma,
        L=L,
        horizon=req.horizon,
        n_sims=req.n_simulations,
    )

    # ── 4. Risk metrics ───────────────────────────────────────────────────────
    risk = compute_risk_metrics(port_paths, req.capital)

    # ── 5. Spaghetti plot sample (100 paths) ──────────────────────────────────
    sample_idx = np.random.choice(req.n_simulations, size=min(100, req.n_simulations), replace=False)
    sample_paths = port_paths[sample_idx].tolist()

    # ── 6. Final value histogram ──────────────────────────────────────────────
    # Downsample to 5000 points for network efficiency
    hist_idx = np.random.choice(req.n_simulations, size=min(5000, req.n_simulations), replace=False)
    final_values = port_paths[hist_idx, -1].tolist()

    # ── 7. 3D surface ─────────────────────────────────────────────────────────
    surface_x, surface_y, surface_z = build_3d_surface(port_paths, req.capital)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info(f"Simulation complete in {elapsed_ms:.1f} ms")

    return SimulationResponse(
        risk_metrics=risk,
        sample_paths=sample_paths,
        final_values=final_values,
        surface_x=surface_x,
        surface_y=surface_y,
        surface_z=surface_z,
        tickers=valid_tickers,
        weights=[round(float(w) * 100, 2) for w in weights],
        elapsed_ms=round(elapsed_ms, 1),
        annualization_factor=ANNUALIZATION[req.granularity],
    )
