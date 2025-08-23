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
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 minutes", value=False)
    
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
    
    for i, pair_name in enumerate(selected_pairs):
        pair_code = currency_pairs[pair_name]
        status_text.text(f"Fetching data for {pair_name}...")
        try:
            data = fetch_forex_data(pair_code, period)
            if data is not None and not data.empty:
                all_data[pair_name] = data
                # Safely get the last close price
                if 'Close' in data.columns and len(data) > 0:
                    current_prices[pair_name] = data['Close'].iloc[-1]
                else:
                    current_prices[pair_name] = None
                    st.warning(f"No price data available for {pair_name}")
            else:
                current_prices[pair_name] = None
                st.warning(f"No data returned for {pair_name}")
            
            progress_bar.progress((i + 1) / len(selected_pairs))
            
        except Exception as e:
            st.error(f"Error fetching {pair_name}: {str(e)}")
            current_prices[pair_name] = None
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_data:
        st.error("No data could be fetched. Please check your internet connection or try again later.")
        return
    
    # Display current prices - with safe formatting
    st.subheader("📊 Current Exchange Rates")
    
    # Filter out None values
    valid_prices = {k: v for k, v in current_prices.items() if v is not None}
    
    if valid_prices:
        cols = st.columns(len(valid_prices))
        
        for col, (pair_name, price) in zip(cols, valid_prices.items()):
            with col:
                # Safe formatting with error handling
                try:
                    formatted_price = f"{float(price):.4f}"
                except (TypeError, ValueError):
                    formatted_price = "N/A"
                
                st.markdown(f"""
                <div class="currency-card">
                    <h3>{pair_name}</h3>
                    <h2>{formatted_price}</h2>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No current prices available for the selected pairs.")
    
    # Display charts
    st.subheader("📈 Price Charts with Moving Averages")
    
    for pair_name, data in all_data.items():
        with st.expander(f"{pair_name} Chart", expanded=True):
            try:
                fig = create_chart(data, pair_name, ma_periods)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display statistics with error handling
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_price = data['Close'].iloc[-1] if 'Close' in data.columns and len(data) > 0 else "N/A"
                    if isinstance(current_price, (int, float)):
                        st.metric("Current", f"{current_price:.4f}")
                    else:
                        st.metric("Current", current_price)
                
                with col2:
                    if 'Close' in data.columns and len(data) > 1:
                        change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                        st.metric("Total Change", f"{change:+.2f}%")
                    else:
                        st.metric("Total Change", "N/A")
                
                with col3:
                    if 'High' in data.columns:
                        st.metric("High", f"{data['High'].max():.4f}" if len(data) > 0 else "N/A")
                    else:
                        st.metric("High", "N/A")
                
                with col4:
                    if 'Low' in data.columns:
                        st.metric("Low", f"{data['Low'].min():.4f}" if len(data) > 0 else "N/A")
                    else:
                        st.metric("Low", "N/A")
                        
            except Exception as e:
                st.error(f"Error creating chart for {pair_name}: {str(e)}")
    
    # Auto-refresh logic
    if auto_refresh:
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()

def fetch_forex_data(pair_code, period):
    """Fetch forex data from Yahoo Finance with error handling"""
    try:
        # Map period to Yahoo Finance format
        period_map = {
            "1M": "1mo",
            "3M": "3mo", 
            "6M": "6mo",
            "1Y": "1y",
            "2Y": "2y",
            "5Y": "5y"
        }
        
        data = yf.download(pair_code, period=period_map[period], progress=False)
        
        # Validate data
        if data is None or data.empty:
            st.warning(f"No data returned for {pair_code}")
            return None
        
        # Check if we have the required columns
        required_columns = ['Close', 'High', 'Low', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.warning(f"Missing columns for {pair_code}: {missing_columns}")
            return None
            
        return data
        
    except Exception as e:
        st.error(f"Error downloading {pair_code}: {str(e)}")
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
    
    # Volume (if available)
    if 'Volume' in data.columns:
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
