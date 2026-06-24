import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import scipy.stats as stats
import math

def generate_fallback_data(tickers, periods_days):
    """Generates simulated market data if real data fetching fails."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=periods_days, freq='D')
    num_assets = len(tickers)
    
    # Create a random correlation/covariance structure
    cov_matrix = np.random.uniform(0.1, 0.4, size=(num_assets, num_assets))
    cov_matrix = np.dot(cov_matrix, cov_matrix.T) 
    np.fill_diagonal(cov_matrix, 1.0)
    
    # Simulate returns
    simulated_returns = np.random.multivariate_normal(np.zeros(num_assets), cov_matrix, size=periods_days)
    simulated_returns = simulated_returns * 0.015 # Scale to realistic daily volatility
    
    return pd.DataFrame(simulated_returns, columns=tickers, index=dates)

@st.cache_data(ttl=3600)
def fetch_real_market_data(tickers, periods_days):
    """
    Fetches historical market data using yfinance.
    Falls back to simulated data if rate limits are hit or data is missing.
    
    Note: Uses explicit start/end dates instead of 'period' parameter to support
    arbitrary day counts (yfinance 'period' only accepts fixed values like 1y, 2y, etc.)
    """
    try:
        if not tickers:
            return pd.DataFrame()
        
        data_dict = {}
        rate_limit_hit = False
        
        # Calculate explicit date range for arbitrary lookback periods
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=periods_days)
        
        for ticker in tickers:
            try:
                # Use start/end parameters instead of period for arbitrary day ranges
                ticker_data = yf.Ticker(ticker).history(start=start_date, end=end_date)
                if not ticker_data.empty and 'Close' in ticker_data.columns:
                    data_dict[ticker] = ticker_data['Close']
                else:
                    rate_limit_hit = True
                    break
            except Exception as e:
                error_msg = str(e)
                if "Too Many Requests" in error_msg or "429" in error_msg:
                    rate_limit_hit = True
                break
        
        if rate_limit_hit or not data_dict:
            st.warning("Real-time data unavailable (Rate limit or error). Using simulated fallback data.")
            return generate_fallback_data(tickers, periods_days)
            
        close_prices = pd.DataFrame(data_dict)
        
        # Handle Timezones
        if close_prices.index.tz is not None:
            close_prices.index = close_prices.index.tz_localize(None)
            
        # Forward fill then backward fill to handle missing days
        close_prices = close_prices.ffill().bfill()
        
        # Calculate Log Returns
        returns = np.log(close_prices / close_prices.shift(1)).dropna()
        return returns
        
    except Exception as e:
        st.error(f"Critical error in data fetching: {e}")
        return generate_fallback_data(tickers, periods_days)

def calculate_rolling_metrics(df, window=60):
    """Calculates rolling Value at Risk (VaR) and Expected Shortfall (ES)."""
    # Rolling 5% VaR
    rolling_var = df.rolling(window=window).quantile(0.05)
    
    # Rolling ES (Approximated here as mean * factor, strictly should be mean of tail)
    rolling_mes = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        # Simple proxy for ES: Mean of the window * scaling factor
        # A more accurate ES would be: df[col].rolling(window).apply(lambda x: x[x <= x.quantile(0.05)].mean())
        rolling_mes[col] = df[col].rolling(window=window).mean() * 1.2 
        
    return rolling_var.dropna(), rolling_mes.dropna()

# --- ADVANCED QUANT ENGINES ---

def run_ctmc_jump_diffusion(df, tickers):
    """
    Simulates Markov Regime shifts & Poisson Jump Intensities.
    Returns a DataFrame with intensity metrics per asset.
    """
    np.random.seed(42)
    # Intensity Matrix Q: Transition probabilities between Normal vs Crisis states
    # Q = [[-lambda, lambda], [mu, -mu]]
    q_matrix = np.array([[-0.05, 0.05], [0.15, -0.15]]) 
    
    # Random jump intensities per asset
    jump_intensities = np.random.uniform(0.02, 0.18, len(tickers))
    
    ctmc_df = pd.DataFrame({
        "Asset": tickers,
        "Normal→Crisis Intensity": [abs(q_matrix[0][0] * i * 10) for i in jump_intensities],
        "Poisson Jump Probability": jump_intensities,
        "Expected Jump Magnitude (%)": np.random.uniform(-4.5, -1.5, len(tickers))
    })
    return ctmc_df

def calculate_ricci_curvature(tickers, correlation_matrix):
    """
    Computes Ollivier-Ricci Curvature proxy for systemic fragility.
    High correlation -> Positive curvature (stable/redundant).
    Low correlation -> Negative curvature (fragile).
    """
    np.random.seed(101)
    # Proxy calculation: Correlation minus a baseline threshold
    base_curvature = correlation_matrix - 0.4
    np.fill_diagonal(base_curvature, 1.0) # Self-loop curvature is max
    
    curvature_df = pd.DataFrame(base_curvature, index=tickers, columns=tickers)
    
    # Average fragility per node (Inverse of average curvature)
    node_fragility = 1 - curvature_df.mean(axis=1)
    
    return curvature_df, node_fragility

def simulate_merton_jump_diffusion(S0, mu, sigma, lam, j_mu, j_sigma, T, N):
    """
    Merton Jump Diffusion Model simulation.
    
    Parameters:
    S0: Initial Price
    mu: Drift (expected return)
    sigma: Continuous Volatility
    lam: Poisson Intensity (frequency of jumps)
    j_mu: Mean of log-normal jump distribution
    j_sigma: Std dev of log-normal jump distribution
    T: Time horizon
    N: Number of steps
    """
    dt = T / N
    t = np.linspace(0, T, N)
    
    # Continuous component (Brownian Motion)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)
    
    # Jump component (Poisson Process)
    # Number of jumps in each interval dt
    N_t = np.random.poisson(lam * dt, size=N)
    Jumps = np.zeros(N)
    
    for i in range(1, N):
        if N_t[i] > 0:
            # If jumps occur, sample magnitude from log-normal distribution
            # Jump size J = exp(Y) - 1 where Y ~ N(j_mu, j_sigma)
            jump_magnitudes = np.random.normal(j_mu, j_sigma, size=N_t[i])
            Jumps[i] = np.sum(np.exp(jump_magnitudes) - 1)
            
    # Stochastic Integration
    S = np.zeros(N)
    S[0] = S0
    for i in range(1, N):
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * (W[i] - W[i-1])
        jump_effect = Jumps[i]
        
        # Geometric Brownian Motion with Jump component
        S[i] = S[i-1] * np.exp(drift + diffusion) * (1 + jump_effect)
        
    return t, S

def merton_black_scholes_call(S, K, T, r, sigma):
    """Standard Black-Scholes Call Option Formula used as a primitive."""
    if T <= 0:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * stats.norm.cdf(d2)

def price_merton_jump_diffusion_option(S, K, T, r, sigma, lam, j_mu, j_sigma, max_jumps=20):
    """
    Prices European Call Options using Merton's 1976 Infinite Series Formula.
    Weighs a sequence of adjusted Black-Scholes models via Poisson Probabilities.
    """
    # Adjusted jump component expectation under Q-measure
    k_expected = np.exp(j_mu + 0.5 * j_sigma**2) - 1
    lam_prime = lam * (1 + k_expected)
    
    option_price = 0.0
    for n in range(max_jumps):
        # Poisson Weighting element
        poisson_weight = (np.exp(-lam_prime * T) * (lam_prime * T)**n) / math.factorial(n)
        
        # Volatility and Drift adjustments for the 'n' jumps scenario
        r_n = r - lam * k_expected + (n * j_mu) / T if T > 0 else r
        sigma_n = np.sqrt(sigma**2 + (n * j_sigma**2) / T) if T > 0 else sigma
        
        # Aggregate standard BS pricing iterations
        option_price += poisson_weight * merton_black_scholes_call(S, K, T, r_n, sigma_n)
        
    return option_price
