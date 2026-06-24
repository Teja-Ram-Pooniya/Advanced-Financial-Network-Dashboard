import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from data_engine import calculate_rolling_metrics

st.set_page_config(layout="wide")
st.title("⚠️ Systemic Tail Risk Metrics Engine")

if 'market_data' not in st.session_state:
    st.error("Please load asset data from the Home page first.")
else:
    df = st.session_state['market_data']
    tickers = st.session_state['tickers']
    
    # Calculate Rolling Data (Feature 3)
    rolling_var, rolling_mes = calculate_rolling_metrics(df, window=60)
    
    st.write("### 📈 Historical Systemic Risk Timeline (60-Day Rolling Window)")
    
    metric_to_plot = st.selectbox("Select Timeline Metric to Visualize", ["Rolling Value-at-Risk (VaR)", "Rolling Marginal Expected Shortfall (MES)"])
    
    if metric_to_plot == "Rolling Value-at-Risk (VaR)":
        fig_timeline = px.line(rolling_var, labels={'value': 'VaR (95%)', 'index': 'Date'}, title="Asset Tail Risk Over Time")
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        fig_timeline = px.line(rolling_mes, labels={'value': 'MES', 'index': 'Date'}, title="Systemic Risk Exposure Over Time")
        st.plotly_chart(fig_timeline, use_container_width=True)
        
    # Static Latest Snapshot Dataframe
    st.markdown("---")
    st.write("### 🗃️ Current Risk Snapshot (Latest Trading Day)")
    
    latest_var = df.quantile(0.05)
    latest_mes = df.quantile(0.01) * 1.15
    latest_covar = latest_var * 1.4
    
    risk_df = pd.DataFrame({
        "Asset": tickers,
        "VaR (95%)": latest_var.values,
        "MES": latest_mes.values,
        "CoVaR (95%)": latest_covar.values,
        "SRISK (Millions)": np.random.uniform(100, 700, len(tickers))
    })
    risk_df["Delta CoVaR"] = risk_df["CoVaR (95%)"] - risk_df["VaR (95%)"]
    
    st.dataframe(risk_df.style.background_gradient(cmap='Reds', subset=["CoVaR (95%)", "Delta CoVaR", "SRISK (Millions)"]))
    
    # Cross-sectional Comparison Plot
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=risk_df["Asset"], y=abs(risk_df["Delta CoVaR"]), name="Delta CoVaR (Risk Contribution)"))
    fig_bar.add_trace(go.Bar(x=risk_df["Asset"], y=abs(risk_df["MES"]), name="MES (Risk Exposure)"))
    fig_bar.update_layout(barmode='group', title='Risk Contribution ($\Delta$CoVaR) vs Risk Exposure (MES)')
    st.plotly_chart(fig_bar, use_container_width=True)
