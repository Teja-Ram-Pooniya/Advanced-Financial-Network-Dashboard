import numpy as np
import streamlit as st
from data_engine import fetch_real_market_data

st.set_page_config(
    page_title="Quant Network & Systemic Risk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Quant Network & Systemic Risk Analytics")
st.markdown("---")

# app.py की लाइन 18 के पास इसे बदलकर टेस्ट करें:
# app.py में ticker_options को टेस्ट करने के लिए ऐसे बदलें:
ticker_options = {
    "RELIANCE.NS": "RELIANCE.NS",
    "TCS.NS": "TCS.NS",
    "INFY.NS": "INFY.NS",
    "Apple (AAPL)": "AAPL"
}
selected_labels = st.sidebar.multiselect(
    "Select Assets for Analysis",
    options=list(ticker_options.keys()),
    default=list(ticker_options.keys())[:4]
)

tickers = [ticker_options[label] for label in selected_labels]
lookback_period = st.sidebar.slider("Historical Lookback (Days)", 200, 1500, 750)

if len(tickers) >= 2:
    with st.spinner("Fetching live market data via yfinance..."):
        df_returns = fetch_real_market_data(tickers, lookback_period)
        
    if not df_returns.empty:
        st.session_state['market_data'] = df_returns
        st.session_state['tickers'] = tickers
        
        st.success(f"Successfully loaded {df_returns.shape[0]} trading days for {len(tickers)} assets.")
        
        # Display Current Volatility Metrics
        cols = st.columns(len(tickers))
        for i, t in enumerate(tickers):
            with cols[i]:
                vol = df_returns[t].std() * np.sqrt(252) * 100 # Annualized Volatility
                st.metric(label=f"{t} Ann. Volatility", value=f"{vol:.2f}%")
    else:
        st.error("No data fetched. Please check your internet connection or tickers.")
else:
    st.warning("⚠️ Please select at least 2 assets to initialize the risk & network pipelines.")
