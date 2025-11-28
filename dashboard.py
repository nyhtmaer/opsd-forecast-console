"""
OPSD PowerDesk: Interactive Dashboard
Day-Ahead Forecasting, Anomaly Detection, and Live Monitoring Dashboard

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="OPSD PowerDesk Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with bold, fun, modern design
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #0F0F1E 0%, #1A0A2E 50%, #0F0F1E 100%);
        font-family: 'Poppins', 'Segoe UI', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        color: #FFFFFF;
        text-align: center;
        padding: 2rem 1rem;
        letter-spacing: 1px;
        text-shadow: none;
        animation: none;
        border-radius: 1rem;
        background: transparent;
        backdrop-filter: none;
        border: none;
        margin: 1rem auto;
        max-width: 95%;
        box-shadow: none;
    }
    
    h1, h2, h3 {
        letter-spacing: 1px;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        font-weight: 900;
    }
    
    h1 {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #FF006E 0%, #FB5607 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        border: none;
        padding: 0;
        background: none;
        color: #FF006E;
        text-shadow: 0 4px 20px rgba(255, 0, 110, 0.3);
    }
    
    h2 {
        font-size: 2.2rem;
        color: #FFBE0B;
        text-shadow: 0 4px 15px rgba(255, 190, 11, 0.3);
    }
    
    h3 {
        font-size: 1.6rem;
        color: #FB5607;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 0, 110, 0.1) 0%, rgba(251, 86, 7, 0.1) 100%);
        padding: 2rem;
        border-radius: 1.5rem;
        margin: 1rem 0;
        border: 2px solid rgba(255, 190, 11, 0.3);
        box-shadow: 0 8px 32px rgba(255, 0, 110, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        backdrop-filter: blur(15px);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 190, 11, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover {
        box-shadow: 0 16px 48px rgba(255, 0, 110, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.3);
        transform: translateY(-8px) scale(1.03);
        border-color: rgba(255, 190, 11, 0.6);
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .country-header {
        background: linear-gradient(135deg, #FF006E 0%, #FB5607 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 1.5rem;
        margin: 1.5rem 0;
        font-weight: 900;
        box-shadow: 0 10px 30px rgba(255, 0, 110, 0.3);
        letter-spacing: 1.5px;
        font-size: 1.4rem;
        border: 2px solid rgba(255, 190, 11, 0.4);
        text-transform: uppercase;
        transform: perspective(1000px) rotateX(0deg);
        transition: all 0.3s ease;
    }
    
    .country-header:hover {
        transform: perspective(1000px) rotateX(5deg) translateY(-2px);
        box-shadow: 0 15px 40px rgba(255, 0, 110, 0.4);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F0F1E 0%, #16213E 100%) !important;
        box-shadow: -8px 0 30px rgba(255, 0, 110, 0.2);
        border-right: 3px solid rgba(255, 0, 110, 0.3);
    }
    
    [data-testid="stSidebarContent"] {
        background: transparent !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #CCCCCC !important;
    }
    
    [data-testid="stSidebarContent"] * {
        color: #CCCCCC !important;
    }
    
    [data-testid="stSidebar"] label {
        font-weight: 900 !important;
        color: #FFBE0B !important;
        font-size: 1.1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    [data-testid="stSidebarContent"] label {
        font-weight: 900 !important;
        color: #FFBE0B !important;
        font-size: 1.1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FF006E !important;
        border: none !important;
        text-shadow: 0 2px 10px rgba(255, 0, 110, 0.3) !important;
    }
    
    [data-testid="stSidebarContent"] h1, [data-testid="stSidebarContent"] h2, [data-testid="stSidebarContent"] h3 {
        color: #FF006E !important;
        border: none !important;
        text-shadow: 0 2px 10px rgba(255, 0, 110, 0.3) !important;
    }
    
    /* Tabs styling */
    [data-testid="stTabs"] {
        margin: 2.5rem 0;
        background: linear-gradient(90deg, rgba(255, 0, 110, 0.08), rgba(251, 86, 7, 0.08));
        padding: 2rem;
        border-radius: 2rem;
        border: 2px solid rgba(255, 190, 11, 0.2);
        box-shadow: inset 0 2px 15px rgba(0, 0, 0, 0.3);
    }
    
    [role="tablist"] {
        background: linear-gradient(90deg, rgba(255, 0, 110, 0.1) 0%, rgba(251, 86, 7, 0.1) 50%, rgba(255, 190, 11, 0.1) 100%) !important;
        padding: 1.5rem !important;
        border-radius: 1.5rem !important;
        border: 2px solid rgba(255, 0, 110, 0.2) !important;
        box-shadow: 0 8px 24px rgba(255, 0, 110, 0.15) !important;
        gap: 1rem !important;
    }
    
    [role="tab"] {
        background: linear-gradient(135deg, #FF006E 0%, #FB5607 100%) !important;
        color: white !important;
        font-weight: 900 !important;
        font-size: 1.2rem !important;
        padding: 1rem 2rem !important;
        border-radius: 1rem !important;
        border: 2px solid rgba(255, 190, 11, 0.3) !important;
        transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        box-shadow: 0 6px 20px rgba(255, 0, 110, 0.2) !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }
    
    [role="tab"]:hover {
        transform: translateY(-5px) scale(1.08) !important;
        box-shadow: 0 12px 35px rgba(255, 0, 110, 0.4) !important;
        background: linear-gradient(135deg, #FFB347 0%, #FFBE0B 100%) !important;
    }
    
    [role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #FFBE0B 0%, #FF006E 100%) !important;
        border: 2px solid rgba(255, 255, 255, 0.6) !important;
        box-shadow: 0 10px 30px rgba(255, 190, 11, 0.4), inset 0 2px 5px rgba(255, 255, 255, 0.2) !important;
        color: #0F0F1E !important;
        font-weight: 900 !important;
    }
    
    /* Button hover effects */
    button {
        transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        border-radius: 1rem !important;
        font-weight: 900 !important;
        font-size: 1.1rem !important;
        background: linear-gradient(135deg, #FF006E, #FB5607) !important;
        color: white !important;
        border: 2px solid rgba(255, 190, 11, 0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        padding: 0.75rem 1.5rem !important;
        box-shadow: 0 6px 20px rgba(255, 0, 110, 0.25) !important;
    }
    
    button:hover {
        box-shadow: 0 12px 35px rgba(255, 0, 110, 0.4) !important;
        transform: translateY(-4px) scale(1.08) !important;
        background: linear-gradient(135deg, #FFBE0B, #FF006E) !important;
    }
    
    /* Card styling for metric boxes */
    [data-testid="stMetricContainer"] {
        background: linear-gradient(135deg, rgba(255, 0, 110, 0.15) 0%, rgba(255, 190, 11, 0.15) 100%);
        border-radius: 1.5rem;
        box-shadow: 0 8px 30px rgba(255, 0, 110, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        padding: 2.5rem 2rem;
        border: 2px solid rgba(255, 190, 11, 0.3);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    [data-testid="stMetricContainer"]:hover {
        transform: translateY(-8px);
        box-shadow: 0 16px 45px rgba(255, 0, 110, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.3);
        border-color: rgba(255, 190, 11, 0.6);
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 4px;
        background: linear-gradient(90deg, transparent, #FF006E 15%, #FB5607 40%, #FFBE0B 60%, #FB5607 80%, #FF006E 100%, transparent);
        margin: 3rem 0;
        border-radius: 3px;
        box-shadow: 0 4px 15px rgba(255, 0, 110, 0.3);
    }
    
    /* Table styling */
    table {
        border-radius: 1.5rem;
        overflow: hidden;
        box-shadow: 0 6px 25px rgba(255, 0, 110, 0.2);
        border: 2px solid rgba(255, 190, 11, 0.25);
        background: linear-gradient(135deg, rgba(255, 0, 110, 0.05), rgba(251, 86, 7, 0.05));
    }
    
    /* Radio button styling */
    [data-testid="stRadio"] {
        background: linear-gradient(90deg, rgba(255, 0, 110, 0.1), rgba(251, 86, 7, 0.1));
        padding: 2rem;
        border-radius: 1.5rem;
        border: 2px solid rgba(255, 0, 110, 0.2);
        box-shadow: 0 6px 20px rgba(255, 0, 110, 0.1);
    }
    
    /* Selectbox styling */
    [data-testid="stSelectbox"] {
        background: linear-gradient(135deg, rgba(255, 0, 110, 0.1) 0%, rgba(251, 86, 7, 0.1) 100%);
        border: 2px solid rgba(255, 190, 11, 0.2);
        border-radius: 1rem;
    }
    
    /* Markdown text */
    [data-testid="stMarkdownContainer"] {
        color: #E0E0E0;
    }
    
    /* Info boxes */
    [data-testid="stAlert"] {
        background: linear-gradient(135deg, rgba(255, 0, 110, 0.1) 0%, rgba(251, 86, 7, 0.1) 100%);
        border: 2px solid rgba(255, 190, 11, 0.3);
        border-radius: 1.2rem;
        box-shadow: 0 6px 20px rgba(255, 0, 110, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h2 class="main-header">⚡ OPSD PowerDesk Dashboard</h2>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center; font-size: 2rem; color: #FFBE0B; text-shadow: 0 4px 15px rgba(255, 190, 11, 0.3); margin: 1rem 0;">Day-Ahead Forecasting, Anomaly Detection & Live Monitoring</h2>', unsafe_allow_html=True)
st.markdown("---")

# LOAD DATA

@st.cache_data
def load_sarima_forecasts():
    """Load SARIMA forecast results"""
    forecasts = {}
    for country in ['DE', 'FR', 'IT']:
        try:
            df = pd.read_csv(f'results/phase4_sarima_de_fr_it_results/forecast_data_{country}.csv', parse_dates=['timestamp'])
            forecasts[country] = df
        except Exception:
            st.warning(f"Could not load SARIMA forecasts for {country}")
    return forecasts

@st.cache_data
def load_lstm_forecasts():
    """Load LSTM forecast results"""
    forecasts = {}
    for country in ['DE', 'FR', 'IT']:
        try:
            df = pd.read_csv(f'results/phase4b_lstm_de_fr_it_results/forecast_data_{country}.csv', parse_dates=['timestamp'])
            forecasts[country] = df
        except Exception:
            pass
    return forecasts

@st.cache_data
def load_gru_forecasts():
    """Load GRU forecast results"""
    forecasts = {}
    for country in ['DE', 'FR', 'IT']:
        try:
            df = pd.read_csv(f'results/phase4c_gru_de_fr_it_results/forecast_data_{country}.csv', parse_dates=['timestamp'])
            forecasts[country] = df
        except Exception:
            pass
    return forecasts

@st.cache_data
def load_rnn_forecasts():
    """Load Vanilla RNN forecast results"""
    forecasts = {}
    for country in ['DE', 'FR', 'IT']:
        try:
            df = pd.read_csv(f'results/phase4d_rnn_de_fr_it_results/forecast_data_{country}.csv', parse_dates=['timestamp'])
            forecasts[country] = df
        except Exception:
            pass
    return forecasts

@st.cache_data
def load_anomalies():
    """Load anomaly detection results"""
    anomalies = {}
    for country in ['DE', 'FR', 'IT']:
        try:
            df = pd.read_csv(f'outputs/{country}_anomalies.csv', parse_dates=['timestamp'])
            anomalies[country] = df
        except Exception:
            st.warning(f"Could not load anomalies for {country}")
    return anomalies

@st.cache_data
def load_live_simulation():
    """Load live monitoring simulation results for all 4 models"""
    live_data = {'SARIMA': {}, 'LSTM': {}, 'GRU': {}, 'RNN': {}}
    
    # Load SARIMA results
    for country in ['DE', 'FR', 'IT']:
        try:
            df = pd.read_csv(f'results/phase6_live_adaptation/{country}_live_simulation.csv', parse_dates=['timestamp'])
            live_data['SARIMA'][country] = df
        except Exception:
            pass
    
    # Load LSTM results
    for country in ['DE', 'FR', 'IT']:
        try:
            df = pd.read_csv(f'results/phase6_live_adaptation/{country}_lstm_live_simulation.csv', parse_dates=['timestamp'])
            live_data['LSTM'][country] = df
        except Exception:
            pass
    
    # Load GRU results
    for country in ['DE', 'FR', 'IT']:
        try:
            df = pd.read_csv(f'results/phase6_live_adaptation/{country}_gru_live_simulation.csv', parse_dates=['timestamp'])
            live_data['GRU'][country] = df
        except Exception:
            pass
    
    # Load Vanilla RNN results
    for country in ['DE', 'FR', 'IT']:
        try:
            df = pd.read_csv(f'results/phase6_live_adaptation/{country}_rnn_live_simulation.csv', parse_dates=['timestamp'])
            live_data['RNN'][country] = df
        except Exception:
            pass
    
    return live_data

@st.cache_data
def load_metrics():
    """Load model comparison metrics"""
    try:
        sarima_metrics = pd.read_csv('results/phase4_sarima_de_fr_it_results/metrics_comparison.csv', index_col=0)
        lstm_metrics = pd.read_csv('results/phase4b_lstm_de_fr_it_results/metrics_comparison.csv', index_col=0)
        gru_metrics = pd.read_csv('results/phase4c_gru_de_fr_it_results/metrics_comparison.csv', index_col=0)
        rnn_metrics = pd.read_csv('results/phase4d_rnn_de_fr_it_results/metrics_comparison.csv', index_col=0)
        return sarima_metrics, lstm_metrics, gru_metrics, rnn_metrics
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return None, None, None, None

# Load all data
with st.spinner("Loading data..."):
    sarima_forecasts = load_sarima_forecasts()
    lstm_forecasts = load_lstm_forecasts()
    gru_forecasts = load_gru_forecasts()
    rnn_forecasts = load_rnn_forecasts()
    anomalies = load_anomalies()
    live_data = load_live_simulation()
    sarima_metrics, lstm_metrics, gru_metrics, rnn_metrics = load_metrics()

# SIDEBAR

# Countries covered
countries = ['DE', 'FR', 'IT']
country_info = {
    'DE': {'name': 'Germany', 'flag': '🇩🇪'},
    'FR': {'name': 'France', 'flag': '🇫🇷'},
    'IT': {'name': 'Italy', 'flag': '🇮🇹'}
}

st.sidebar.markdown("**Countries Covered:**")
for code in countries:
    st.sidebar.markdown(f"{country_info[code]['flag']} {country_info[code]['name']} ({code})")
st.sidebar.markdown("---")

# Section selection
section = st.sidebar.radio(
    "NAVIGATION ",
    options=[
        "📊 Overview",
        "📈 Forecast Comparison",
        "🚨 Anomaly Detection",
        "🔄 Live Monitoring",
        "🏆 Model Comparison"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard visualizes day-ahead electric load forecasting results "
    "for Germany (DE), France (FR), and Italy (IT) using OPSD data. "
    "\n\n**Models:** SARIMA, LSTM, GRU, Vanilla RNN"
)

# SECTION 1: OVERVIEW

if section == "📊 Overview":
    st.header("📊 Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Countries Analyzed", "3", help="DE, FR, IT")
    
    with col2:
        st.metric("Models Tested", "4", help="SARIMA, LSTM, GRU, Vanilla RNN")
    
    with col3:
        st.metric("Live Simulation", "3,500 hrs", help="146 days of streaming data")
    
    with col4:
        if sarima_metrics is not None:
            best_mape = sarima_metrics['MAPE'].min()
            st.metric("Best MAPE", f"{best_mape:.2f}%", help="Germany|France|Italy SARIMA")
    
    st.markdown("---")
    
    # Key Findings
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Key Findings")
        st.markdown("""
        - **Excellent Accuracy:** SARIMA achieved 2.95-7.14% MAPE across all countries
        - **Neural Networks Excel:** GRU achieved best performance with 0.41 MASE for Germany|France|Italy
        - **Live Adaptation:** 10 successful model refits over 3,500 hours (146 days)
        - **Anomaly Detection:** 21 anomalies detected across 2,928 data points (0.72% rate)
        - **Data Coverage:** 50,398 hours spanning 6 years (2014-2020)
        """)
    
    with col2:
        st.subheader("📋 Assignment Compliance")
        st.markdown("""
        ✅ **Phase 1:** Data preprocessing & STL decomposition  
        ✅ **Phase 2:** ACF/PACF analysis & SARIMA grid search  
        ✅ **Phase 3:** Model building (SARIMA + 3 neural models)  
        ✅ **Phase 4:** 24-step forecasting with all metrics  
        ✅ **Phase 5:** Anomaly detection (z-score, CUSUM, ML)  
        ✅ **Phase 6:** Live monitoring (3,500 hours simulation)  
        ✅ **Phase 7:** Interactive dashboard (this app!)  
        """)
    
    # Dataset Overview
    st.markdown("---")
    st.subheader("📁 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Training Data**")
        st.write("• 40,319 hours (1,680 days)")
        st.write("• 80% of total data")
        st.write("• Used for model training")
    
    with col2:
        st.markdown("**Validation Data**")
        st.write("• 5,039 hours (210 days)")
        st.write("• 10% of total data")
        st.write("• Used for hyperparameter tuning")
    
    with col3:
        st.markdown("**Test Data**")
        st.write("• 5,040 hours (210 days)")
        st.write("• 10% of total data")
        st.write("• Used for final evaluation")

# SECTION 2: FORECAST COMPARISON

elif section == "📈 Forecast Comparison":
    st.header("📈 Forecast Comparison")
    
    if not countries:
        st.warning("Please select at least one country from the sidebar.")
    else:
        # Country tabs
        tabs = st.tabs(countries)
        
        for idx, country in enumerate(countries):
            with tabs[idx]:
                st.subheader(f"{country} - Day-Ahead Forecasts")
                
                # Get data
                sarima_df = sarima_forecasts.get(country)
                lstm_df = lstm_forecasts.get(country)
                
                if sarima_df is None:
                    st.error(f"No forecast data available for {country}")
                    continue
                
                # Create forecast plot with optimized layer ordering and styling
                fig = go.Figure()
                
                # Plot prediction interval first (background layer)
                fig.add_trace(go.Scatter(
                    x=sarima_df['timestamp'],
                    y=sarima_df['upper_bound_80%'],
                    fill=None,
                    mode='lines',
                    line=dict(color='rgba(0,119,182,0)', width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=sarima_df['timestamp'],
                    y=sarima_df['lower_bound_80%'],
                    fill='tonexty',
                    mode='lines',
                    line=dict(color='rgba(0,119,182,0)', width=0),
                    name='80% Prediction Interval',
                    fillcolor='rgba(0, 119, 182, 0.15)'
                ))
                
                # Neural network forecasts (thick, bright lines for high visibility)
                if lstm_df is not None:
                    fig.add_trace(go.Scatter(
                        x=lstm_df['timestamp'],
                        y=lstm_df['forecast'],
                        name='LSTM Forecast',
                        line=dict(color='#32CD32', width=2.5),
                        mode='lines'
                    ))
                
                gru_df = gru_forecasts.get(country)
                if gru_df is not None:
                    fig.add_trace(go.Scatter(
                        x=gru_df['timestamp'],
                        y=gru_df['forecast'],
                        name='GRU Forecast',
                        line=dict(color='#FF8C00', width=2.5),
                        mode='lines'
                    ))
                
                rnn_df = rnn_forecasts.get(country)
                if rnn_df is not None:
                    fig.add_trace(go.Scatter(
                        x=rnn_df['timestamp'],
                        y=rnn_df['forecast'],
                        name='Vanilla RNN Forecast',
                        line=dict(color='#FF1744', width=2.5),
                        mode='lines'
                    ))
                
                # SARIMA forecast (slightly thinner for differentiation)
                fig.add_trace(go.Scatter(
                    x=sarima_df['timestamp'],
                    y=sarima_df['forecast'],
                    name='SARIMA Forecast',
                    line=dict(color='#0077B6', width=2),
                    mode='lines'
                ))
                
                # Actual load on top (white for contrast on black background)
                fig.add_trace(go.Scatter(
                    x=sarima_df['timestamp'],
                    y=sarima_df['actual'],
                    name='Actual Load',
                    line=dict(color='white', width=2.5),
                    mode='lines'
                ))
                
                fig.update_layout(
                    title=f"{country} - Actual Load vs Forecasts",
                    xaxis_title="Timestamp",
                    yaxis_title="Load (MW)",
                    hovermode='x unified',
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor='#1A1A1A',
                    paper_bgcolor='#1A1A1A',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics table
                st.markdown("#### Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if sarima_metrics is not None and country in sarima_metrics.index:
                        st.markdown("**SARIMA Model**")
                        metrics_df = sarima_metrics.loc[[country]]
                        st.dataframe(metrics_df.style.format({
                            'MASE': '{:.4f}',
                            'sMAPE': '{:.2f}',
                            'MAPE': '{:.2f}',
                            'RMSE': '{:.2f}',
                            'MSE': '{:.2f}',
                            'PI_Coverage_80%': '{:.2f}'
                        }), use_container_width=True)
                
                with col2:
                    if lstm_metrics is not None and country in lstm_metrics.index:
                        st.markdown("**LSTM Model**")
                        lstm_df_metrics = lstm_metrics.loc[[country]]
                        st.dataframe(lstm_df_metrics.style.format({
                            'MASE': '{:.4f}',
                            'sMAPE_%': '{:.2f}',
                            'MAPE_%': '{:.2f}',
                            'RMSE_MW': '{:.2f}',
                            'MSE': '{:.2f}'
                        }), use_container_width=True)
                
                with col3:
                    if gru_metrics is not None and country in gru_metrics.index:
                        st.markdown("**GRU Model**")
                        gru_df_metrics = gru_metrics.loc[[country]]
                        st.dataframe(gru_df_metrics.style.format({
                            'MASE': '{:.4f}',
                            'sMAPE_%': '{:.2f}',
                            'MAPE_%': '{:.2f}',
                            'RMSE_MW': '{:.2f}',
                            'MSE': '{:.2f}'
                        }), use_container_width=True)
                
                with col4:
                    if rnn_metrics is not None and country in rnn_metrics.index:
                        st.markdown("**Vanilla RNN Model**")
                        rnn_df_metrics = rnn_metrics.loc[[country]]
                        st.dataframe(rnn_df_metrics.style.format({
                            'MASE': '{:.4f}',
                            'sMAPE_%': '{:.2f}',
                            'MAPE_%': '{:.2f}',
                            'RMSE_MW': '{:.2f}',
                            'MSE': '{:.2f}'
                        }), use_container_width=True)

# SECTION 3: ANOMALY DETECTION

elif section == "🚨 Anomaly Detection":
    st.header("🚨 Anomaly Detection Results")
    
    if not countries:
        st.warning("Please select at least one country from the sidebar.")
    else:
        # Country tabs
        tabs = st.tabs(countries)
        
        for idx, country in enumerate(countries):
            with tabs[idx]:
                st.subheader(f"{country} - Detected Anomalies")
                
                anom_df = anomalies.get(country)
                
                if anom_df is None:
                    st.error(f"No anomaly data available for {country}")
                    continue
                
                # Anomaly statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_points = len(anom_df)
                    st.metric("Total Points", f"{total_points:,}")
                
                with col2:
                    z_anomalies = anom_df['flag_z'].sum()
                    z_rate = (z_anomalies / total_points * 100)
                    st.metric("Z-score Anomalies", z_anomalies, f"{z_rate:.2f}%")
                
                with col3:
                    if 'flag_cusum' in anom_df.columns:
                        cusum_anomalies = anom_df['flag_cusum'].sum()
                        cusum_rate = (cusum_anomalies / total_points * 100)
                        st.metric("CUSUM Anomalies", cusum_anomalies, f"{cusum_rate:.2f}%")
                
                with col4:
                    max_z = anom_df['z_resid'].abs().max()
                    st.metric("Max |Z-score|", f"{max_z:.2f}")
                
                # Create anomaly visualization
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=(
                        "Load with Anomalies Flagged",
                        "Rolling Z-score of Residuals"
                    ),
                    vertical_spacing=0.12,
                    row_heights=[0.6, 0.4]
                )
                
                # Plot 1: Load with anomalies
                fig.add_trace(
                    go.Scatter(
                        x=anom_df['timestamp'],
                        y=anom_df['y_true'],
                        name='Actual Load',
                        line=dict(color='black', width=1),
                        mode='lines'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=anom_df['timestamp'],
                        y=anom_df['yhat'],
                        name='Forecast',
                        line=dict(color='blue', width=1),
                        mode='lines',
                        opacity=0.6
                    ),
                    row=1, col=1
                )
                
                # Highlight anomalies
                anomaly_points = anom_df[anom_df['flag_z'] == 1]
                if len(anomaly_points) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_points['timestamp'],
                            y=anomaly_points['y_true'],
                            name='Z-score Anomaly',
                            mode='markers',
                            marker=dict(color='red', size=8, symbol='circle')
                        ),
                        row=1, col=1
                    )
                
                # Plot 2: Z-scores
                fig.add_trace(
                    go.Scatter(
                        x=anom_df['timestamp'],
                        y=anom_df['z_resid'],
                        name='Z-score',
                        line=dict(color='steelblue', width=1),
                        mode='lines'
                    ),
                    row=2, col=1
                )
                
                # Add threshold lines
                fig.add_hline(y=3.0, line_dash="dash", line_color="red", row=2, col=1, 
                             annotation_text="Threshold (+3σ)")
                fig.add_hline(y=-3.0, line_dash="dash", line_color="red", row=2, col=1,
                             annotation_text="Threshold (-3σ)")
                fig.add_hline(y=0, line_dash="solid", line_color="gray", row=2, col=1,
                             opacity=0.5)
                
                fig.update_xaxes(title_text="Timestamp", row=2, col=1)
                fig.update_yaxes(title_text="Load (MW)", row=1, col=1)
                fig.update_yaxes(title_text="Z-score", row=2, col=1)
                
                fig.update_layout(
                    height=700,
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor='#1A1A1A',
                    paper_bgcolor='#1A1A1A',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly details table
                if len(anomaly_points) > 0:
                    st.markdown("#### Detected Anomaly Details")
                    anomaly_display = anomaly_points[['timestamp', 'y_true', 'yhat', 'z_resid']].copy()
                    anomaly_display.columns = ['Timestamp', 'Actual Load', 'Forecast', 'Z-score']
                    st.dataframe(
                        anomaly_display.style.format({
                            'Actual Load': '{:.2f}',
                            'Forecast': '{:.2f}',
                            'Z-score': '{:.2f}'
                        }),
                        use_container_width=True
                    )

# SECTION 4: LIVE MONITORING

elif section == "🔄 Live Monitoring":
    st.header("🔄 Live Monitoring Simulation Results")
    
    st.markdown("""
    Simulated **3,500 hours** (146 days) of live data streaming with **periodic model retraining**.
    All models were refitted every **336 hours** (2 weeks) using an expanding window for fair comparison.
    """)
    
    if not countries:
        st.warning("Please select at least one country from the sidebar.")
    else:
        # Model selector
        model_choice = st.selectbox(
            "Select Model to View",
            options=['SARIMA', 'LSTM', 'GRU', 'RNN', 'All Models (Comparison)'],
            index=4
        )
        
        # Summary metrics for all models
        st.subheader("📊 3,500-Hour Simulation Summary")
        
        # Create summary table
        summary_data = []
        for model_name in ['SARIMA', 'LSTM', 'GRU', 'RNN']:
            for country in countries:
                model_data = live_data.get(model_name, {})
                live_df = model_data.get(country)
                if live_df is not None:
                    summary_data.append({
                        'Model': model_name,
                        'Country': country,
                        'Avg MAPE (%)': live_df['mape_rolling_24h'].mean(),
                        'Avg MASE': live_df['mase_rolling_24h'].mean(),
                        'Avg RMSE': live_df['rmse_rolling_24h'].mean(),
                        'Refits': int(live_df['refit_flag'].sum())
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Display as table
            cols = st.columns(len(countries))
            for idx, country in enumerate(countries):
                with cols[idx]:
                    st.markdown(f"### {country}")
                    country_summary = summary_df[summary_df['Country'] == country][['Model', 'Avg MAPE (%)', 'Avg MASE', 'Refits']]
                    st.dataframe(
                        country_summary.style.format({
                            'Avg MAPE (%)': '{:.2f}',
                            'Avg MASE': '{:.4f}',
                            'Refits': '{:.0f}'
                        }).highlight_min(subset=['Avg MAPE (%)'], color='lightgreen'),
                        use_container_width=True
                    )
        
        st.markdown("---")
        
        # Performance evolution chart
        st.subheader("📈 Performance Evolution Over Time")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            metric_choice = st.radio(
                "Select Metric",
                options=['MAPE (%)', 'MASE (scaled)', 'RMSE (MW)'],
                horizontal=True
            )
        with col2:
            smooth_data = st.checkbox("Smooth curves (4-hour avg)", value=False)
        
        metric_map = {
            'MAPE (%)': 'mape_rolling_24h',
            'MASE (scaled)': 'mase_rolling_24h',
            'RMSE (MW)': 'rmse_rolling_24h'
        }
        
        ylabel_map = {
            'MAPE (%)': 'Error (%)',
            'MASE (scaled)': 'Scaled Error',
            'RMSE (MW)': 'Error (MW)'
        }
        
        color_map = {
            'SARIMA': '#E53935',
            'LSTM': '#1E88E5',
            'GRU': '#43A047',
            'RNN': '#FB8C00'
        }
        
        if model_choice == 'All Models (Comparison)':
            # Show all models for selected countries in separate subplots
            for country in countries:
                st.markdown(f"#### {country}")
                fig = go.Figure()
                
                for model_name in ['SARIMA', 'LSTM', 'GRU', 'RNN']:
                    model_data = live_data.get(model_name, {})
                    live_df = model_data.get(country)
                    if live_df is not None:
                        y_data = live_df[metric_map[metric_choice]]
                        
                        # Apply smoothing if requested
                        if smooth_data:
                            y_data = y_data.rolling(window=4, center=True).mean()
                        
                        fig.add_trace(go.Scatter(
                            x=live_df['timestamp'],
                            y=y_data,
                            name=model_name,
                            mode='lines',
                            line=dict(width=2.5, color=color_map.get(model_name)),
                            hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>'
                        ))
                
                fig.update_layout(
                    xaxis=dict(
                        title="Time Period",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128,128,128,0.2)',
                        tickformat='%b %d'
                    ),
                    yaxis=dict(
                        title=ylabel_map[metric_choice],
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128,128,128,0.2)',
                        zeroline=True,
                        zerolinewidth=2,
                        zerolinecolor='rgba(128,128,128,0.3)'
                    ),
                    height=450,
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    plot_bgcolor='#1A1A1A',
                    paper_bgcolor='#1A1A1A',
                    font=dict(color='white'),
                    margin=dict(l=60, r=40, t=40, b=60)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Show single model for all countries (each country in separate subplot)
            from plotly.subplots import make_subplots
            
            model_data = live_data.get(model_choice, {})
            country_colors = {'DE': '#FF6B6B', 'FR': '#4ECDC4', 'IT': '#95E1D3'}
            country_names = {'DE': 'Germany', 'FR': 'France', 'IT': 'Italy'}
            
            # Create subplots for each country
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=[f"{country} - {country_names[country]}" for country in countries],
                vertical_spacing=0.12,
                shared_xaxes=True
            )
            
            for row_idx, country in enumerate(countries, start=1):
                live_df = model_data.get(country)
                if live_df is not None:
                    y_data = live_df[metric_map[metric_choice]]
                    
                    # Apply smoothing if requested
                    if smooth_data:
                        y_data = y_data.rolling(window=4, center=True).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=live_df['timestamp'],
                        y=y_data,
                        name=country,
                        mode='lines',
                        line=dict(width=2.5, color=country_colors.get(country)),
                        hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>',
                        showlegend=(row_idx == 1)
                    ), row=row_idx, col=1)
                    
                    # Mark refits with vertical lines
                    refit_points = live_df[live_df['refit_flag'] == 1]
                    if len(refit_points) > 0 and not smooth_data:
                        for idx, refit_time in enumerate(refit_points['timestamp']):
                            fig.add_vline(
                                x=refit_time,
                                line_dash="dot",
                                line_color=country_colors.get(country),
                                opacity=0.4,
                                row=row_idx, col=1
                            )
            
            # Update axes for all subplots
            for i in range(1, 4):
                fig.update_yaxes(
                    title_text=ylabel_map[metric_choice],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='rgba(128,128,128,0.3)',
                    row=i, col=1
                )
            
            fig.update_xaxes(
                title_text="Time Period",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                tickformat='%b %d',
                row=3, col=1
            )
            
            fig.update_layout(
                height=800,
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                plot_bgcolor='#1A1A1A',
                paper_bgcolor='#1A1A1A',
                font=dict(color='white'),
                margin=dict(l=60, r=40, t=60, b=60)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model age chart (for single model view only)
        if model_choice != 'All Models (Comparison)':
            st.markdown("---")
            st.subheader("🔧 Model Freshness (Days Since Last Refit)")
            
            fig2 = go.Figure()
            
            model_data = live_data.get(model_choice, {})
            country_colors = {'DE': '#FF6B6B', 'FR': '#4ECDC4', 'IT': '#95E1D3'}
            
            for country in countries:
                live_df = model_data.get(country)
                if live_df is not None and 'model_age_hours' in live_df.columns:
                    # Convert hours to days for better readability
                    model_age_days = live_df['model_age_hours'] / 24
                    
                    fig2.add_trace(go.Scatter(
                        x=live_df['timestamp'],
                        y=model_age_days,
                        name=country,
                        mode='lines',
                        fill='tozeroy',
                        line=dict(width=2, color=country_colors.get(country)),
                        fillcolor=f"rgba{tuple(list(int(country_colors.get(country)[i:i+2], 16) for i in (1, 3, 5)) + [0.2])}",
                        hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Age: %{y:.1f} days<extra></extra>'
                    ))
            
            # Add refit threshold line (14 days = 336 hours)
            fig2.add_hline(
                y=14,
                line_dash="dash",
                line_color="rgba(255,0,0,0.5)",
                line_width=2,
                annotation_text="Refit Threshold (14 days)",
                annotation_position="right"
            )
            
            fig2.update_layout(
                xaxis=dict(
                    title="Time Period",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    tickformat='%b %d'
                ),
                yaxis=dict(
                    title="Days",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    range=[0, 15]
                ),
                hovermode='x unified',
                height=350,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                plot_bgcolor='#1A1A1A',
                paper_bgcolor='#1A1A1A',
                font=dict(color='white'),
                margin=dict(l=60, r=40, t=40, b=60)
            )
            
            st.plotly_chart(fig2, use_container_width=True)

# SECTION 5: MODEL COMPARISON

elif section == "🏆 Model Comparison":
    st.header("🏆 Model Performance Comparison")
    
    # Metrics comparison table
    st.subheader("📊 All Models - Test Set Metrics")
    
    if sarima_metrics is not None and lstm_metrics is not None and gru_metrics is not None and rnn_metrics is not None:
        # Combine metrics
        comparison = pd.DataFrame()
        
        for country in ['DE', 'FR', 'IT']:
            if country in sarima_metrics.index:
                sarima_row = sarima_metrics.loc[country].copy()
                sarima_row.name = f"{country} - SARIMA"
                comparison = pd.concat([comparison, pd.DataFrame([sarima_row])])
            
            if country in lstm_metrics.index:
                lstm_row = lstm_metrics.loc[country].copy()
                lstm_row.name = f"{country} - LSTM"
                if 'MAPE_%' in lstm_row.index:
                    lstm_row['MAPE'] = lstm_row['MAPE_%']
                if 'sMAPE_%' in lstm_row.index:
                    lstm_row['sMAPE'] = lstm_row['sMAPE_%']
                if 'RMSE_MW' in lstm_row.index:
                    lstm_row['RMSE'] = lstm_row['RMSE_MW']
                comparison = pd.concat([comparison, pd.DataFrame([lstm_row])])
            
            if country in gru_metrics.index:
                gru_row = gru_metrics.loc[country].copy()
                gru_row.name = f"{country} - GRU"
                if 'MAPE_%' in gru_row.index:
                    gru_row['MAPE'] = gru_row['MAPE_%']
                if 'sMAPE_%' in gru_row.index:
                    gru_row['sMAPE'] = gru_row['sMAPE_%']
                if 'RMSE_MW' in gru_row.index:
                    gru_row['RMSE'] = gru_row['RMSE_MW']
                comparison = pd.concat([comparison, pd.DataFrame([gru_row])])
            
            if country in rnn_metrics.index:
                rnn_row = rnn_metrics.loc[country].copy()
                rnn_row.name = f"{country} - Vanilla RNN"
                if 'MAPE_%' in rnn_row.index:
                    rnn_row['MAPE'] = rnn_row['MAPE_%']
                if 'sMAPE_%' in rnn_row.index:
                    rnn_row['sMAPE'] = rnn_row['sMAPE_%']
                if 'RMSE_MW' in rnn_row.index:
                    rnn_row['RMSE'] = rnn_row['RMSE_MW']
                comparison = pd.concat([comparison, pd.DataFrame([rnn_row])])
        
        # Display table
        display_cols = ['MASE', 'MAPE', 'sMAPE', 'RMSE', 'MSE']
        available_cols = [col for col in display_cols if col in comparison.columns]
        
        st.dataframe(
            comparison[available_cols].style.format({
                'MASE': '{:.4f}',
                'MAPE': '{:.2f}',
                'sMAPE': '{:.2f}',
                'RMSE': '{:.2f}',
                'MSE': '{:.2f}'
            }).highlight_min(axis=0, color='lightgreen'),
            use_container_width=True
        )
    
    # Visual comparison
    st.markdown("---")
    st.subheader("📊 Visual Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # MAPE comparison
        if sarima_metrics is not None and lstm_metrics is not None and gru_metrics is not None and rnn_metrics is not None:
            mape_data = []
            for country in ['DE', 'FR', 'IT']:
                if country in sarima_metrics.index:
                    mape_data.append({
                        'Country': country,
                        'Model': 'SARIMA',
                        'MAPE': sarima_metrics.loc[country, 'MAPE']
                    })
                if country in lstm_metrics.index:
                    mape_val = lstm_metrics.loc[country, 'MAPE_%'] if 'MAPE_%' in lstm_metrics.columns else lstm_metrics.loc[country, 'MAPE']
                    mape_data.append({
                        'Country': country,
                        'Model': 'LSTM',
                        'MAPE': mape_val
                    })
                if country in gru_metrics.index:
                    mape_val = gru_metrics.loc[country, 'MAPE_%'] if 'MAPE_%' in gru_metrics.columns else gru_metrics.loc[country, 'MAPE']
                    mape_data.append({
                        'Country': country,
                        'Model': 'GRU',
                        'MAPE': mape_val
                    })
                if country in rnn_metrics.index:
                    mape_val = rnn_metrics.loc[country, 'MAPE_%'] if 'MAPE_%' in rnn_metrics.columns else rnn_metrics.loc[country, 'MAPE']
                    mape_data.append({
                        'Country': country,
                        'Model': 'Vanilla RNN',
                        'MAPE': mape_val
                    })
            
            mape_df = pd.DataFrame(mape_data)
            fig = px.bar(
                mape_df,
                x='Country',
                y='MAPE',
                color='Model',
                barmode='group',
                title='MAPE Comparison by Country',
                labels={'MAPE': 'MAPE (%)'},
                color_discrete_map={'SARIMA': 'blue', 'LSTM': 'green', 'GRU': 'orange', 'Vanilla RNN': 'red'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # MASE comparison
        if sarima_metrics is not None and lstm_metrics is not None and gru_metrics is not None and rnn_metrics is not None:
            mase_data = []
            for country in ['DE', 'FR', 'IT']:
                if country in sarima_metrics.index:
                    mase_data.append({
                        'Country': country,
                        'Model': 'SARIMA',
                        'MASE': sarima_metrics.loc[country, 'MASE']
                    })
                if country in lstm_metrics.index:
                    mase_data.append({
                        'Country': country,
                        'Model': 'LSTM',
                        'MASE': lstm_metrics.loc[country, 'MASE']
                    })
                if country in gru_metrics.index:
                    mase_data.append({
                        'Country': country,
                        'Model': 'GRU',
                        'MASE': gru_metrics.loc[country, 'MASE']
                    })
                if country in rnn_metrics.index:
                    mase_data.append({
                        'Country': country,
                        'Model': 'Vanilla RNN',
                        'MASE': rnn_metrics.loc[country, 'MASE']
                    })
            
            mase_df = pd.DataFrame(mase_data)
            fig = px.bar(
                mase_df,
                x='Country',
                y='MASE',
                color='Model',
                barmode='group',
                title='MASE Comparison by Country',
                labels={'MASE': 'MASE (lower is better)'},
                color_discrete_map={'SARIMA': 'blue', 'LSTM': 'green', 'GRU': 'orange', 'Vanilla RNN': 'red'}
            )
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                         annotation_text="Baseline (MASE=1.0)")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Model Recommendations")
    
    if sarima_metrics is not None and lstm_metrics is not None and gru_metrics is not None and rnn_metrics is not None:
        # Calculate best model per country based on MAPE
        recommendations = {}
        
        for country in ['DE', 'FR', 'IT']:
            country_results = {}
            
            if country in sarima_metrics.index:
                country_results['SARIMA'] = {
                    'MAPE': sarima_metrics.loc[country, 'MAPE'],
                    'MASE': sarima_metrics.loc[country, 'MASE']
                }
            
            if country in lstm_metrics.index:
                mape = lstm_metrics.loc[country, 'MAPE_%'] if 'MAPE_%' in lstm_metrics.columns else lstm_metrics.loc[country, 'MAPE']
                country_results['LSTM'] = {
                    'MAPE': mape,
                    'MASE': lstm_metrics.loc[country, 'MASE']
                }
            
            if country in gru_metrics.index:
                mape = gru_metrics.loc[country, 'MAPE_%'] if 'MAPE_%' in gru_metrics.columns else gru_metrics.loc[country, 'MAPE']
                country_results['GRU'] = {
                    'MAPE': mape,
                    'MASE': gru_metrics.loc[country, 'MASE']
                }
            
            if country in rnn_metrics.index:
                mape = rnn_metrics.loc[country, 'MAPE_%'] if 'MAPE_%' in rnn_metrics.columns else rnn_metrics.loc[country, 'MAPE']
                country_results['Vanilla RNN'] = {
                    'MAPE': mape,
                    'MASE': rnn_metrics.loc[country, 'MASE']
                }
            
            # Find best model (lowest MAPE)
            if country_results:
                best_model = min(country_results.items(), key=lambda x: x[1]['MAPE'])
                recommendations[country] = {
                    'best_model': best_model[0],
                    'best_mape': best_model[1]['MAPE'],
                    'best_mase': best_model[1]['MASE'],
                    'all_results': country_results
                }
        
        # Display per-country recommendations
        col1, col2, col3 = st.columns(3)
        
        for col, country in zip([col1, col2, col3], ['DE', 'FR', 'IT']):
            with col:
                if country in recommendations:
                    rec = recommendations[country]
                    st.markdown(f"<div class='country-header'>{country_info[country]['flag']} {country_info[country]['name']} ({country})</div>", unsafe_allow_html=True)
                    st.success(f"✅ **{rec['best_model']}** - Best performance")
                    st.metric("MAPE", f"{rec['best_mape']:.2f}%", delta=None)
                    st.metric("MASE", f"{rec['best_mase']:.4f}", delta=None)
                    
                    # Show ranking
                    sorted_models = sorted(rec['all_results'].items(), key=lambda x: x[1]['MAPE'])
                    st.caption("**Ranking (by MAPE):**")
                    for i, (model, metrics) in enumerate(sorted_models, 1):
                        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                        st.caption(f"{emoji} {i}. {model}: {metrics['MAPE']:.2f}%")
        
        # Overall best model recommendation
        st.markdown("---")
        st.subheader("🏅 Overall Best Model")
        
        # Calculate average MAPE across all countries for each model
        model_averages = {}
        
        for model in ['SARIMA', 'LSTM', 'GRU', 'Vanilla RNN']:
            mapes = []
            for country in ['DE', 'FR', 'IT']:
                if country in recommendations and model in recommendations[country]['all_results']:
                    mapes.append(recommendations[country]['all_results'][model]['MAPE'])
            
            if mapes:
                model_averages[model] = {
                    'avg_mape': sum(mapes) / len(mapes),
                    'countries': len(mapes)
                }
        
        if model_averages:
            best_overall = min(model_averages.items(), key=lambda x: x[1]['avg_mape'])
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"### 🏆 **{best_overall[0]}**")
                st.metric("Average MAPE", f"{best_overall[1]['avg_mape']:.2f}%")
                st.caption(f"Evaluated across {best_overall[1]['countries']} countries")
            
            with col2:
                st.markdown("**Summary:**")
                sorted_overall = sorted(model_averages.items(), key=lambda x: x[1]['avg_mape'])
                
                for i, (model, metrics) in enumerate(sorted_overall, 1):
                    if i == 1:
                        st.success(f"🥇 **{model}**: {metrics['avg_mape']:.2f}% MAPE (Best)")
                    elif i == 2:
                        st.info(f"🥈 {model}: {metrics['avg_mape']:.2f}% MAPE")
                    elif i == 3:
                        st.info(f"🥉 {model}: {metrics['avg_mape']:.2f}% MAPE")
                    else:
                        st.warning(f"📊 {model}: {metrics['avg_mape']:.2f}% MAPE")
                
                # Key insights
                st.markdown("---")
                st.markdown("**Key Insights:**")
                winner_count = sum(1 for c in recommendations.values() if c['best_model'] == best_overall[0])
                st.write(f"• {best_overall[0]} achieves the best performance in **{winner_count} out of 3 countries**")
                st.write(f"• Average improvement over baseline: **{((model_averages.get('SARIMA', {}).get('avg_mape', 0) - best_overall[1]['avg_mape']) / model_averages.get('SARIMA', {}).get('avg_mape', 1) * 100):.1f}%**")
        st.write("LSTM: 4.50% MAPE, 0.755 MASE")

# FOOTER

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 2rem 0;'>
        <p><strong>OPSD PowerDesk Dashboard</strong></p>
        <p>Advanced Time Series Analysis Project | November 2025</p>
        <p>Data Source: Open Power System Data (OPSD) | Countries: Germany (DE), France (FR), Italy (IT)</p>
    </div>
    """,
    unsafe_allow_html=True
)
