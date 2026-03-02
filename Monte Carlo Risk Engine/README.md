# Monte Carlo Portfolio Risk Pricer
> **Quantitative Risk Engine** — VaR · Expected Shortfall · Cholesky GBM

---

## Architecture

```
monte-carlo-pricer/
├── api/
│   └── main.py          # FastAPI backend — NumPy vectorized Monte Carlo engine
├── index.html           # Single-file frontend — Plotly interactive dashboards
├── requirements.txt     # Python dependencies
├── vercel.json          # Vercel routing + build config
└── README.md
```

---

## Quantitative Methodology

### 1. Data Ingestion
- Yahoo Finance (`yfinance`) — Adjusted Close prices
- 3-year rolling lookback window
- Supports daily / weekly / monthly granularity

### 2. Parameter Estimation
```
μᵢ = E[log(Sᵢ(t)/Sᵢ(t-1))]          # Historical log-return mean
σᵢ = Std[log(Sᵢ(t)/Sᵢ(t-1))]        # Historical log-return volatility
Σ  = Corr matrix (N×N)               # Correlation between all assets
```

### 3. Cholesky Decomposition
```
Σ = L Lᵀ    →    L = cholesky(Σ)
```
Correlates independent standard normals to preserve realistic co-movement.

### 4. Vectorized GBM Simulation (Zero for-loops)
```python
Z_raw  ∈ ℝ^(S×T×N)    # i.i.d. standard normals  (drawn in one shot)
Z_corr = einsum('ij,stj→sti', L, Z_raw)  # Cholesky correlation
r_it   = (μᵢ - σᵢ²/2)Δt + σᵢ √Δt Z_corr  # Itô drift correction
P(t)   = P₀ · exp(Σ rᵢₜ)                  # Geometric Brownian Motion
```

### 5. Risk Measures
```
VaR_α   = -Quantile(PnL, 1-α)          # α = 95%, 99%
ES_α    = -E[PnL | PnL ≤ -VaR_α]       # Expected Shortfall (CVaR)
```

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run backend
uvicorn api.main:app --reload --port 8000

# Serve frontend
python -m http.server 3000
# → Open http://localhost:3000
# → Change API_BASE in index.html to 'http://localhost:8000/api'
```

## Deploy on Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

---

## Performance

| Config             | Time     |
|--------------------|----------|
| 10,000 paths · 252 steps · 5 assets | ~0.8–1.5s |
| 50,000 paths · 252 steps · 5 assets | ~3–5s     |

Pure NumPy vectorization: no Python loops → fits within Vercel's 10s timeout.

---

## Skills Demonstrated

- **Quantitative Finance**: GBM, Itô calculus, Cholesky decomposition, VaR, ES/CVaR
- **Python Engineering**: FastAPI, Pydantic v2, NumPy broadcasting, SciPy KDE
- **Financial Data**: yfinance integration, log-return estimation, correlation matrices
- **Software Architecture**: RESTful API design, serverless deployment (Vercel)
- **Data Visualization**: Plotly spaghetti plots, histograms with VaR overlays, 3D KDE surfaces
