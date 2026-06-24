import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from data_engine import price_merton_jump_diffusion_option

st.set_page_config(layout="wide")
st.title("📐 Ito Lemma Primitives & Merton Option Pricing Desk")
st.markdown("---")

st.markdown("""
### ⚡ Stochastic Calculus under Discontinuous Regimes
यह मॉड्यूल आपकी स्लाइड **"Ito Lemma and Jumps"** और **"Characteristic Function"** के टॉपिक्स को सीधे ऑप्शन मार्केट्स के रिस्क मैनेजमेंट से जोड़ता है।
""")

# Sidebar option space controls
st.sidebar.header("📋 Option & Asset Parameters")
S0 = st.sidebar.slider("Spot Price (S0)", 50.0, 150.0, 100.0)
r = st.sidebar.slider("Risk-Free Rate (r)", 0.01, 0.15, 0.05)
T_expiry = st.sidebar.slider("Time to Maturity (T in Years)", 0.1, 2.0, 0.5)

# Jump Specs - Using session state if available from Jump Processes page, else defaults
sigma = st.session_state.get('sigma', 0.2) if 'sigma' in st.session_state else 0.2
lam = st.session_state.get('lam', 3.0) if 'lam' in st.session_state else 3.0
j_mu = st.session_state.get('j_mu', -0.05) if 'j_mu' in st.session_state else -0.05
j_sigma = st.session_state.get('j_sigma', 0.05) if 'j_sigma' in st.session_state else 0.05

# Allow override of jump parameters on this page too for flexibility
st.sidebar.subheader("⚡ Override Jump Parameters")
sigma = st.sidebar.slider("Continuous Vol ($\sigma$)", 0.05, 0.8, sigma, key="ito_sigma")
lam = st.sidebar.slider("Poisson Intensity ($\lambda$)", 0.1, 10.0, lam, key="ito_lam")
j_mu = st.sidebar.slider("Mean Jump Magnitude", -0.2, 0.2, j_mu, key="ito_jmu")
j_sigma = st.sidebar.slider("Jump Volatility", 0.01, 0.3, j_sigma, key="ito_jsig")

tab1, tab2 = st.tabs(["Ito Stochastic Expansions", "Merton Pricing Matrix Desk"])

with tab1:
    st.subheader("🔗 Non-local Taylor Shifting Analysis")
    st.markdown(r"""
    पारंपरिक निरंतर इतो लेम्मा में केवल फर्स्ट और सेकंड-ऑर्डर डेरिवेटिव्स ($\frac{\partial f}{\partial S} dW_t + \frac{1}{2}\frac{\partial^2 f}{\partial S^2}dS_t^2$) का उपयोग होता है। 
    लेकिन जंप्स आने पर, थ्योरी के अनुसार हमें पूरा **Non-local shift operator** जोड़ना पड़ता है:
    """)
    st.latex(r"\Delta f(S_t) = f(S_{t-}Y) - f(S_{t-})")
    
    # Simulating Ito expansion drift mapping
    st.write("#### Drift Drag Mapping ($dX_t$ vs Standard Brownian Motion)")
    j_sizes = np.linspace(0.5, 1.5, 50)
    ito_drag = [np.log(y) for y in j_sizes]  # Log space scaling expansion
    linear_approx = [y - 1 for y in j_sizes]
    
    drag_df = pd.DataFrame({
        "Jump Magnitude (Y)": j_sizes,
        "Actual Ito Jump Transformation [log(Y)]": ito_drag,
        "Linear Error (Standard BS Assumption)": linear_approx
    }).set_index("Jump Magnitude (Y)")
    
    st.line_chart(drag_df)
    st.caption("Notice that for large jumps down (Y < 1), standard linear approximations collapse, highlighting the requirement for Non-local PIDE Operators.")

with tab2:
    st.subheader("🔮 Merton European Call Option Matrix")
    st.markdown("Merton Infinite Series-driven analytical pricing framework vs Black-Scholes Baseline:")
    
    strike_range = np.linspace(80, 120, 10)
    merton_prices = []
    bs_prices = []
    
    for K in strike_range:
        m_p = price_merton_jump_diffusion_option(S0, K, T_expiry, r, sigma, lam, j_mu, j_sigma)
        b_p = price_merton_jump_diffusion_option(S0, K, T_expiry, r, sigma, lam=0, j_mu=0, j_sigma=0) # Pure BS
        merton_prices.append(m_p)
        bs_prices.append(b_p)
        
    pricing_df = pd.DataFrame({
        "Strike Price (K)": strike_range,
        "Merton Price (With Jumps)": merton_prices,
        "Black-Scholes Price (No Jumps)": bs_prices,
        "Jump Risk Premium ($)": np.array(merton_prices) - np.array(bs_prices)
    })
    
    st.dataframe(pricing_df.style.background_gradient(cmap="Oranges", subset=["Jump Risk Premium ($)"]))
    
    fig_opt = px.line(pricing_df, x="Strike Price (K)", y=["Merton Price (With Jumps)", "Black-Scholes Price (No Jumps)"],
                      title="Option Pricing Discrepancy (Volatility Skew Source)")
    st.plotly_chart(fig_opt, use_container_width=True)
