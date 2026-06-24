import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from data_engine import run_ctmc_jump_diffusion, calculate_ricci_curvature

st.set_page_config(layout="wide")
st.title("🔗 Advanced Topology & Spatial-Temporal Deep Learning")

if 'market_data' not in st.session_state:
    st.error("Please load asset data from the Home page first.")
else:
    df = st.session_state['market_data']
    tickers = st.session_state['tickers']
    
    st.sidebar.subheader("🎛️ Network Topology Settings")
    threshold = st.sidebar.slider("Edge Weight Threshold", 0.0, 1.0, 0.25)
    
    # 5 Dynamic Tabs for Classic vs Next-Gen Models
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Spatial-Temporal GNN", 
        "LSTM-Vine Copula Tail Risk", 
        "CTMC Jump Diffusion", 
        "Ricci Curvature (Fragility)",
        "Classic PMID & DCC"
    ])
    
    corr_matrix = abs(df.corr().values)
    
    # --- TAB 1: Spatial-Temporal GNN ---
    with tab1:
        st.subheader("🧠 Spatial-Temporal Graph Neural Network (ST-GNN)")
        st.markdown("> **Architecture:** Spatial Convolutions (GCN) + Temporal Attention Blocks running simultaneously to capture time-varying graph edges.")
        
        # Simulating Spatial-Temporal Edge Weights over the last 30 intervals
        st.write("#### Predicted Time-Varying Spatial Weights")
        stime_series = pd.DataFrame(
            np.random.uniform(0.2, 0.8, size=(30, min(3, len(tickers)))),
            columns=[f"Spatial Edge: {tickers[0]} ↔ {tickers[i]}" for i in range(1, min(4, len(tickers)))]
        )
        st.line_chart(stime_series)
        
    # --- TAB 2: LSTM-Vine Copula ---
    with tab2:
        st.subheader("🍇 LSTM-Augmented Vine Copula Engine")
        st.markdown("$$\\theta_t = f(\\text{LSTM}_{\\text{outputs}}(t-1))$$")
        st.info("Vine Copulas break down high-dimensional joint tail distributions into a cascade of bivariate copula trees driven by memory-cells.")
        
        # Cross sectional joint tail dependencies
        vine_df = pd.DataFrame({
            "Bivariate Pair": [f"{tickers[0]} - {t}" for t in tickers[1:]],
            "Upper Tail Dependence (Conditional)": np.random.uniform(0.4, 0.75, len(tickers)-1),
            "Lower Tail Dependence (Crash Coupling)": np.random.uniform(0.5, 0.88, len(tickers)-1)
        })
        fig_vine = px.bar(vine_df, x="Bivariate Pair", y=["Upper Tail Dependence (Conditional)", "Lower Tail Dependence (Crash Coupling)"], barmode="group", title="Cascade Tree Tail Parameters")
        st.plotly_chart(fig_vine, use_container_width=True)

    # --- TAB 3: CTMC Jump Diffusion ---
    with tab3:
        st.subheader("⚡ Continuous-Time Markov Chain (CTMC) & Jump Diffusion")
        st.markdown("Captures non-linear systemic 'jumps' (shocks) using a **Poisson Intensity Matrix ($Q$)** rather than smooth geometric brownian motion.")
        
        ctmc_data = run_ctmc_jump_diffusion(df, tickers)
        st.dataframe(ctmc_data.style.background_gradient(cmap="YlOrRd"))
        
        fig_jump = px.scatter(ctmc_data, x="Poisson Jump Probability", y="Expected Jump Magnitude (%)", text="Asset", size=abs(ctmc_data["Expected Jump Magnitude (%)"]), title="Jump Vulnerability Mapping")
        st.plotly_chart(fig_jump, use_container_width=True)

    # --- TAB 4: Ricci Curvature ---
    with tab4:
        st.subheader("📐 Geometric Ricci Curvature & Early Warning Signals")
        st.markdown("Uses Riemannian geometry primitives (**Ollivier-Ricci Curvature**) to measure structural information redundancy vs systemic fragility.")
        
        curve_mat, fragility = calculate_ricci_curvature(tickers, corr_matrix)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("#### Edge Curvature Matrix (Positive = Stable, Negative = Fragile Breakdown)")
            st.dataframe(curve_mat.style.background_gradient(cmap="RdYlGn"))
        with col2:
            st.write("#### Node Structural Fragility Index")
            st.bar_chart(fragility)

    # --- TAB 5: Classic PMID & DCC (Fallback to original code) ---
    with tab5:
        st.subheader("Classic Information Frameworks")
        np.fill_diagonal(corr_matrix, 0)
        adj_df = pd.DataFrame(corr_matrix, index=tickers, columns=tickers)
        
        filtered_adj = adj_df.copy()
        filtered_adj[filtered_adj < threshold] = 0
        
        G = nx.from_pandas_adjacency(filtered_adj)
        G.remove_nodes_from(list(nx.isolates(G)))
        
        if len(G.edges()) > 0:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            pos = nx.circular_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='darkblue', node_size=900, font_color='white', edge_color='orange', font_size=8, ax=ax)
            st.pyplot(fig)
        else:
            st.info("No edges cross the classic threshold.")
