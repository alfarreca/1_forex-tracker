
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf

st.set_page_config(page_title="EUR Forex Tracker", page_icon="💱", layout="wide")

# --------------------------- Styling ---------------------------
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; text-align: center; margin-bottom: 0.5rem; }
    .currency-card { background:#f6f9ff; border-left:4px solid #1f77b4; border-radius:12px; padding:10px 14px; }
    .warn { color:#b22222; }
</style>
""", unsafe_allow_html=True)

# --------------------------- Helpers ---------------------------
PERIOD_MAP = {"1M":"1mo","3M":"3mo","6M":"6mo","1Y":"1y","2Y":"2y","5Y":"5y"}
PAIRS = {"EUR/USD":"EURUSD=X","EUR/CNY":"EURCNY=X","EUR/AUD":"EURAUD=X","EUR/GBP":"EURGBP=X"}

def fmt_num(x, decimals=4, default="N/A"):
    """Safe formatter that accepts pandas/numpy scalars and Series."""
    try:
        if isinstance(x, pd.Series):
            x = x.dropna().iloc[-1] if not x.dropna().empty else np.nan
        if pd.isna(x):
            return default
        return f"{float(x):.{decimals}f}"
    except Exception:
        return default

@st.cache_data(show_spinner=False, ttl=300)
def get_history(ticker: str, period: str) -> pd.DataFrame:
    """Robust downloader using Ticker.history (avoids some yf.download quirks)."""
    hist = yf.Ticker(ticker).history(period=PERIOD_MAP[period])
    if hist is None or hist.empty:
        return pd.DataFrame()
    # ensure DateTimeIndex and numeric columns
    hist.index = pd.to_datetime(hist.index, errors="coerce")
    for c in ["Open","High","Low","Close","Volume"]:
        if c in hist.columns:
            hist[c] = pd.to_numeric(hist[c], errors="coerce")
    # drop rows with no Close
    if "Close" in hist.columns:
        hist = hist[hist["Close"].notna()]
    return hist

def make_chart(df: pd.DataFrame, title: str, ma_periods):
    df = df.copy()
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(f"{title} Price", "Volume"), row_width=[0.25, 0.75]
    )

    # price
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"], name="Price"),
        row=1, col=1
    )

    # moving averages
    for p in ma_periods:
        try:
            p = int(p)
        except Exception:
            continue
        if p <= len(df):
            ma = df["Close"].rolling(p, min_periods=p).mean()
            fig.add_trace(
                go.Scatter(x=df.index, y=ma, name=f"MA{p}", line=dict(dash="dash")),
                row=1, col=1
            )

    # volume
    if "Volume" in df.columns and df["Volume"].notna().any():
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"].fillna(0), name="Volume", opacity=0.5),
            row=2, col=1
        )

    fig.update_layout(height=560, legend=dict(orientation="h", x=1, xanchor="right", y=1.1))
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

def section_metrics(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    last = df["Close"].iloc[-1] if not df.empty else np.nan
    with c1: st.metric("Current", fmt_num(last, 4))
    if len(df) > 1:
        first = df["Close"].iloc[0]
        change = (float(last) - float(first))/float(first)*100.0 if first else np.nan
        with c2: st.metric("Total Change", fmt_num(change, 2) + "%")
    else:
        with c2: st.metric("Total Change", "N/A")
    with c3: st.metric("High", fmt_num(df["High"].max() if "High" in df else np.nan, 4))
    with c4: st.metric("Low", fmt_num(df["Low"].min() if "Low" in df else np.nan, 4))

# --------------------------- UI ---------------------------
st.markdown('<div class="main-header">💱 EUR Forex Tracker</div>', unsafe_allow_html=True)

st.sidebar.title("⚙️ Settings")
selected = st.sidebar.multiselect("Select Currency Pairs:", list(PAIRS.keys()), default=list(PAIRS.keys()))
period = st.sidebar.selectbox("Time Period:", list(PERIOD_MAP.keys()), index=2)
ma_periods = st.sidebar.multiselect("Moving Averages:", [5,10,20,50,100,200], default=[20,50])
auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 minutes", value=False)
if auto_refresh:
    st.sidebar.caption("Next update " + (datetime.now() + timedelta(minutes=5)).strftime("%H:%M:%S"))

if not selected:
    st.info("Pick at least one pair on the left.")
    st.stop()

# --------------------------- Fetch ---------------------------
status = st.empty()
progress = st.progress(0)
data_map = {}
for i, name in enumerate(selected):
    status.info(f"Downloading {name} ...")
    try:
        df = get_history(PAIRS[name], period)
    except Exception as e:
        st.error(f"Error downloading {PAIRS[name]}: {e}")
        df = pd.DataFrame()
    if df.empty:
        st.warning(f"No data returned for {name}")
    else:
        data_map[name] = df
    progress.progress((i+1)/len(selected))

status.empty(); progress.empty()

if not data_map:
    st.error("No data available. Try a different period or refresh.")
    st.stop()

# --------------------------- Current Rates ---------------------------
st.subheader("📊 Current Exchange Rates")
cols = st.columns(len(data_map))
for col, (name, df) in zip(cols, data_map.items()):
    with col:
        last = df["Close"].iloc[-1] if not df.empty else np.nan
        st.markdown(f'<div class="currency-card"><h4>{name}</h4><h2>{fmt_num(last,4)}</h2></div>', unsafe_allow_html=True)

# --------------------------- Charts ---------------------------
st.subheader("📈 Price Charts with Moving Averages")
for name, df in data_map.items():
    with st.expander(f"{name} Chart", expanded=True):
        try:
            fig = make_chart(df, name, ma_periods)
            st.plotly_chart(fig, use_container_width=True)
            section_metrics(df)
        except Exception as e:
            st.error(f"Error creating chart for {name}: {e}")
