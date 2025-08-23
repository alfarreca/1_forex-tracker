import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# App configuration
st.set_page_config(page_title="EUR Forex Tracker", page_icon="💱", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .currency-card { background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin: 10px 0; border-left: 4px solid #1f77b4; }
    .metric-card { background-color: #e8f4fd; border-radius: 8px; padding: 10px; text-align: center; margin: 5px; }
</style>
""", unsafe_allow_html=True)

# ---------- helpers ----------
def fmt_num(x, decimals=4, default="N/A"):
    """Format numbers robustly (works with numpy scalars, Pandas scalars/Series)."""
    try:
        # If it's a Series, try to extract a single value
        if isinstance(x, pd.Series):
            # prefer the last valid value if present
            x = x.dropna().iloc[-1] if not x.dropna().empty else np.nan
        # numpy scalar -> float
        if isinstance(x, (np.floating, np.integer)):
            x = float(x)
        # plain python number
        if isinstance(x, (int, float)):
            return f"{x:.{decimals}f}"
        return default
    except Exception:
        return default

def to_numeric_df(df, cols):
    """Coerce listed columns to numeric (errors -> NaN)."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fetch_forex_data(pair_code, period):
    """Fetch forex data from Yahoo Finance with error handling"""
    try:
        period_map = {"1M":"1mo","3M":"3mo","6M":"6mo","1Y":"1y","2Y":"2y","5Y":"5y"}
        data = yf.download(pair_code, period=period_map[period], progress=False)
        if data is None or data.empty:
            st.warning(f"No data returned for {pair_code}")
            return None

        # Ensure a DateTimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index, errors="coerce")

        # Coerce columns we use to numeric to avoid object/str dtypes
        data = to_numeric_df(data, ["Open","High","Low","Close","Adj Close","Volume"])
        return data
    except Exception as e:
        st.error(f"Error downloading {pair_code}: {str(e)}")
        return None

def create_chart(data, pair_name, ma_periods):
    """Create interactive chart with moving averages"""
    # Guard: drop rows without Close
    data = data.copy()
    data = data[pd.notna(data.get("Close"))]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=(f'{pair_name} Price', 'Volume'), row_width=[0.7, 0.3]
    )

    # Price line
    fig.add_trace(
        go.Scatter(x=data.index, y=data["Close"].astype(float), name="Price",
                   line=dict(color="#1f77b4", width=2)),
        row=1, col=1
    )

    # Moving averages
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    for i, p in enumerate(ma_periods):
        try:
            p = int(p)
        except Exception:
            continue
        if p <= len(data):
            ma = data["Close"].rolling(window=p, min_periods=p).mean()
            fig.add_trace(
                go.Scatter(x=data.index, y=ma.astype(float), name=f"MA{p}",
                           line=dict(color=colors[i % len(colors)], width=1.5, dash="dash")),
                row=1, col=1
            )

    # Volume (if available)
    if "Volume" in data.columns and data["Volume"].notna().any():
        fig.add_trace(
            go.Bar(x=data.index, y=data["Volume"].fillna(0).astype(float),
                   name="Volume", marker_color="#7f7f7f", opacity=0.5),
            row=2, col=1
        )

    fig.update_layout(
        height=600, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

# ---------- app ----------
def main():
    st.markdown('<h1 class="main-header">💱 EUR Forex Tracker</h1>', unsafe_allow_html=True)

    st.sidebar.title("⚙️ Settings")
    currency_pairs = {
        "EUR/USD": "EURUSD=X",
        "EUR/CNY": "EURCNY=X",
        "EUR/AUD": "EURAUD=X",
        "EUR/GBP": "EURGBP=X",
    }
    selected_pairs = st.sidebar.multiselect(
        "Select Currency Pairs:", options=list(currency_pairs.keys()),
        default=list(currency_pairs.keys())
    )
    period = st.sidebar.selectbox("Time Period:", options=["1M","3M","6M","1Y","2Y","5Y"], index=2)
    ma_periods = st.sidebar.multiselect(
        "Moving Averages:", options=[5,10,20,50,100,200], default=[20,50]
    )
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 minutes", value=False)
    if auto_refresh:
        st.sidebar.info("Next update: " + (datetime.now() + timedelta(minutes=5)).strftime("%H:%M:%S"))

    if not selected_pairs:
        st.warning("Please select at least one currency pair to display.")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    all_data, current_prices = {}, {}
    for i, pair_name in enumerate(selected_pairs):
        pair_code = currency_pairs[pair_name]
        status_text.text(f"Fetching data for {pair_name}...")
        try:
            data = fetch_forex_data(pair_code, period)
            if data is not None and not data.empty:
                all_data[pair_name] = data
                current_prices[pair_name] = data["Close"].dropna().iloc[-1] if "Close" in data else None
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

    # Current prices (robust formatting)
    st.subheader("📊 Current Exchange Rates")
    valid_prices = {k: v for k, v in current_prices.items() if v is not None}
    if valid_prices:
        cols = st.columns(len(valid_prices))
        for col, (pair_name, price) in zip(cols, valid_prices.items()):
            with col:
                st.markdown(f"""
                <div class="currency-card">
                    <h3>{pair_name}</h3>
                    <h2>{fmt_num(price, 4)}</h2>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No current prices available for the selected pairs.")

    # Charts + stats
    st.subheader("📈 Price Charts with Moving Averages")
    for pair_name, df in all_data.items():
        with st.expander(f"{pair_name} Chart", expanded=True):
            try:
                fig = create_chart(df, pair_name, ma_periods)
                st.plotly_chart(fig, use_container_width=True)

                col1, col2, col3, col4 = st.columns(4)
                # Current
                with col1:
                    current = df["Close"].dropna().iloc[-1] if "Close" in df else np.nan
                    st.metric("Current", fmt_num(current, 4))
                # Total Change
                with col2:
                    if "Close" in df and df["Close"].notna().sum() > 1:
                        first = df["Close"].dropna().iloc[0]
                        last  = df["Close"].dropna().iloc[-1]
                        change = (float(last) - float(first)) / float(first) * 100.0
                        st.metric("Total Change", fmt_num(change, 2) + "%")
                    else:
                        st.metric("Total Change", "N/A")
                # High / Low
                with col3:
                    hi = df["High"].max() if "High" in df else np.nan
                    st.metric("High", fmt_num(hi, 4))
                with col4:
                    lo = df["Low"].min() if "Low" in df else np.nan
                    st.metric("Low", fmt_num(lo, 4))
            except Exception as e:
                st.error(f"Error creating chart for {pair_name}: {str(e)}")

    if auto_refresh and st.sidebar.button("🔄 Refresh Now"):
        st.rerun()

if __name__ == "__main__":
    main()
