import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# App configuration
st.set_page_config(
    page_title="EUR Forex Tracker",
    page_icon="💱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .currency-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: #e8f4fd;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">💱 EUR Forex Tracker</h1>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.title("⚙️ Settings")
    
    # Currency pairs to track
    currency_pairs = {
        "EUR/USD": "EURUSD=X",
        "EUR/CNY": "EURCNY=X", 
        "EUR/AUD": "EURAUD=X",
        "EUR/GBP": "EURGBP=X"
    }
    
    selected_pairs = st.sidebar.multiselect(
        "Select Currency Pairs:",
        options=list(currency_pairs.keys()),
        default=list(currency_pairs.keys())
    )
    
    # Date range
    period = st.sidebar.selectbox(
        "Time Period:",
        options=["1M", "3M", "6M", "1Y", "2Y", "5Y"],
        index=2
    )
    
    # Moving averages
    ma_periods = st.sidebar.multiselect(
        "Moving Averages:",
        options=[5, 10, 20, 50, 100, 200],
        default=[20, 50]
    )
    
    # Update frequency
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 minutes", value=True)
    
    if auto_refresh:
        st.sidebar.info("Next update: " + (datetime.now() + timedelta(minutes=5)).strftime("%H:%M:%S"))
    
    # Main content
    if not selected_pairs:
        st.warning("Please select at least one currency pair to display.")
        return
    
    # Fetch and display data
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_data = {}
    current_prices = {}
    
    for i, (pair_name, pair_code) in enumerate([(k, v) for k, v in currency_pairs.items() if k in selected_pairs]):
        status_text.text(f"Fetching data for {pair_name}...")
        try:
            data = fetch_forex_data(pair_code, period)
            if data is not None:
                all_data[pair_name] = data
                current_prices[pair_name] = data['Close'].iloc[-1]
            progress_bar.progress((i + 1) / len(selected_pairs))
        except Exception as e:
            st.error(f"Error fetching {pair_name}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_data:
        st.error("No data could be fetched. Please check your internet connection.")
        return
    
    # Display current prices
    st.subheader("📊 Current Exchange Rates")
    cols = st.columns(len(selected_pairs))
    
    for col, (pair_name, price) in zip(cols, current_prices.items()):
        with col:
            st.markdown(f"""
            <div class="currency-card">
                <h3>{pair_name}</h3>
                <h2>{price:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Display charts
    st.subheader("📈 Price Charts with Moving Averages")
    
    for pair_name, data in all_data.items():
        with st.expander(f"{pair_name} Chart", expanded=True):
            fig = create_chart(data, pair_name, ma_periods)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current", f"{data['Close'].iloc[-1]:.4f}")
            with col2:
                change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                st.metric("Total Change", f"{change:+.2f}%")
            with col3:
                st.metric("High", f"{data['High'].max():.4f}")
            with col4:
                st.metric("Low", f"{data['Low'].min():.4f}")
    
    # Auto-refresh logic
    if auto_refresh:
        st.sidebar.button("🔄 Refresh Now")
        st.balloons()

def fetch_forex_data(pair_code, period):
    """Fetch forex data from Yahoo Finance"""
    try:
        # Map period to days
        period_map = {
            "1M": "1mo",
            "3M": "3mo", 
            "6M": "6mo",
            "1Y": "1y",
            "2Y": "2y",
            "5Y": "5y"
        }
        
        data = yf.download(pair_code, period=period_map[period])
        if data.empty:
            return None
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def create_chart(data, pair_name, ma_periods):
    """Create interactive chart with moving averages"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{pair_name} Price', 'Volume'),
        row_width=[0.7, 0.3]
    )
    
    # Price line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    # Moving averages
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    for i, period in enumerate(ma_periods):
        if period <= len(data):
            ma = data['Close'].rolling(window=period).mean()
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=ma,
                    name=f'MA{period}',
                    line=dict(color=colors[i % len(colors)], width=1.5, dash='dash')
                ),
                row=1, col=1
            )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='#7f7f7f',
            opacity=0.5
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

if __name__ == "__main__":
    main()
