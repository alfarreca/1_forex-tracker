import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf

# ------------------------------------------------------------
# App setup
# ------------------------------------------------------------
st.set_page_config(page_title="EUR Forex Tracker", page_icon="💱", layout="wide")

# ---- DARK THEME CSS ----
st.markdown("""
<style>
    body, .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .main-header {
        font-size: 2.2rem; 
        text-align: center; 
        margin-bottom: .5rem;
        color: #1DB954;
    }
    .currency-card {
        background:#161b22;
        border-left:4px solid #1DB954;
        border-radius:12px; 
        padding:10px 14px;
        color: #fafafa;
    }
    .stMetric {
        background:#1c1f26;
        padding:8px;
        border-radius:8px;
    }
    /* Sidebar dark */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        color: #fafafa !important;
    }
    /* Sidebar labels */
    section[data-testid="stSidebar"] label {
        color: #fafafa !important;
    }
    /* Inputs & select boxes */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1c1f26;
        color: #fafafa;
    }
    .stMultiSelect div[data-baseweb="select"] > div {
        background-color: #1c1f26;
        color: #fafafa;
    }
    .stNumberInput input {
        background-color: #1c1f26;
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

PERIOD_MAP = {"1M":"1mo","3M":"3mo","6M":"6mo","1Y":"1y","2Y":"2y","5Y":"5y"}
PAIRS = {
    "EUR/USD":"EURUSD=X",   # USD per 1 EUR
    "EUR/CNY":"EURCNY=X",   # CNY per 1 EUR
    "EUR/AUD":"EURAUD=X",   # AUD per 1 EUR
    "EUR/GBP":"EURGBP=X",
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def fmt_num(x, decimals=4, default="N/A"):
    try:
        if isinstance(x, pd.Series):
            x = x.dropna().iloc[-1] if not x.dropna().empty else np.nan
        if pd.isna(x): return default
        return f"{float(x):.{decimals}f}"
    except Exception:
        return default

@st.cache_data(show_spinner=False, ttl=300)
def get_history(ticker: str, period: str) -> pd.DataFrame:
    hist = yf.Ticker(ticker).history(period=PERIOD_MAP[period])
    if hist is None or hist.empty:
        return pd.DataFrame()
    hist.index = pd.to_datetime(hist.index, errors="coerce")
    for c in ["Open","High","Low","Close","Volume"]:
        if c in hist.columns:
            hist[c] = pd.to_numeric(hist[c], errors="coerce")
    if "Close" in hist.columns:
        hist = hist[hist["Close"].notna()]
    return hist

def make_chart(df: pd.DataFrame, title: str, ma_periods, period: str):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(f"{title} Price", "Volume"), row_width=[0.25, 0.75]
    )
    # Price
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color="#1DB954")), row=1, col=1)
    # MAs
    for p in ma_periods:
        try: p = int(p)
        except: continue
        if len(df)==0: continue
        minp = 1 if period=="1M" else p
        ma = df["Close"].rolling(p, min_periods=minp).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ma, name=f"MA{p}", line=dict(dash="dot")), row=1, col=1)
    # Volume
    if "Volume" in df.columns and df["Volume"].notna().any():
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"].fillna(0), name="Volume", marker_color="#666"), row=2, col=1)
    fig.update_layout(
        template="plotly_dark",
        height=560,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.08),
        margin=dict(l=20,r=20,t=40,b=20)
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

def section_metrics(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    last = df["Close"].iloc[-1] if not df.empty else np.nan
    with c1: st.metric("Current", fmt_num(last, 4))
    if len(df)>1:
        first = df["Close"].iloc[0]
        change = (float(last)-float(first))/float(first)*100.0 if first else np.nan
        with c2: st.metric("Total Change", fmt_num(change,2)+"%")
    else: 
        with c2: st.metric("Total Change","N/A")
    with c3: st.metric("High", fmt_num(df["High"].max() if "High" in df else np.nan,4))
    with c4: st.metric("Low", fmt_num(df["Low"].min() if "Low" in df else np.nan,4))

# FX calculators
def fx_pl(close_series, amount, pair):
    s = close_series.dropna().astype(float)
    if s.empty or len(s)<2: return None
    if pair in ["EUR/USD","EUR/CNY","EUR/AUD"]:
        eur_series = amount / s
    else:
        return None
    start, end = eur_series.iloc[0], eur_series.iloc[-1]
    return pd.DataFrame({"EUR_Value": eur_series}), {
        "start_eur": start, "end_eur": end,
        "abs_pl_eur": float(end-start),
        "pct_pl": (end/start-1.0)*100.0
    }

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.markdown('<div class="main-header">💱 EUR Forex Tracker</div>', unsafe_allow_html=True)

st.sidebar.title("⚙️ Settings")
selected = st.sidebar.multiselect("Select Currency Pairs:", list(PAIRS.keys()), default=list(PAIRS.keys()))
period = st.sidebar.selectbox("Time Period:", list(PERIOD_MAP.keys()), index=0)
ma_periods = st.sidebar.multiselect("Moving Averages:", [5,10,20,50,100,200], default=[20,50])

usd_amount = cny_amount = aud_amount = None
if "EUR/USD" in selected: usd_amount = st.sidebar.number_input("USD amount (EUR/USD)", value=1000.0, step=100.0)
if "EUR/CNY" in selected: cny_amount = st.sidebar.number_input("CNY amount (EUR/CNY)", value=7000.0, step=500.0)
if "EUR/AUD" in selected: aud_amount = st.sidebar.number_input("AUD amount (EUR/AUD)", value=1500.0, step=100.0)

# ------------------------------------------------------------
# Fetch
# ------------------------------------------------------------
data_map={}
for name in selected:
    df = get_history(PAIRS[name], period)
    if not df.empty: data_map[name]=df
if not data_map: st.stop()

# ------------------------------------------------------------
# Current rates
# ------------------------------------------------------------
st.subheader("📊 Current Exchange Rates")
cols = st.columns(len(data_map))
for col,(name,df) in zip(cols,data_map.items()):
    with col:
        last=df["Close"].iloc[-1] if not df.empty else np.nan
        st.markdown(f'<div class="currency-card"><h4>{name}</h4><h2>{fmt_num(last,4)}</h2></div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# Charts + FX impact
# ------------------------------------------------------------
st.subheader("📈 Price Charts with Moving Averages")
for name,df in data_map.items():
    with st.expander(f"{name} Chart", expanded=True):
        fig=make_chart(df,name,ma_periods,period)
        st.plotly_chart(fig,use_container_width=True)
        section_metrics(df)

        # FX-only impact
        if name=="EUR/USD" and usd_amount:
            eur_df,stats=fx_pl(df["Close"],usd_amount,"EUR/USD")
            st.markdown("**💶 USD asset → EUR**")
        elif name=="EUR/CNY" and cny_amount:
            eur_df,stats=fx_pl(df["Close"],cny_amount,"EUR/CNY")
            st.markdown("**💶 CNY asset → EUR**")
        elif name=="EUR/AUD" and aud_amount:
            eur_df,stats=fx_pl(df["Close"],aud_amount,"EUR/AUD")
            st.markdown("**💶 AUD asset → EUR**")
        else: continue

        c1,c2,c3=st.columns(3)
        c1.metric("Start (EUR)", fmt_num(stats["start_eur"],2))
        c2.metric("End (EUR)", fmt_num(stats["end_eur"],2))
        sign="+" if stats["abs_pl_eur"]>=0 else ""
        c3.metric("P/L (FX only)", f"{sign}{fmt_num(stats['abs_pl_eur'],2)} EUR", f"{fmt_num(stats['pct_pl'],2)}%")

        fx_fig=go.Figure()
        fx_fig.add_trace(go.Scatter(x=eur_df.index,y=eur_df["EUR_Value"],name="EUR Value"))
        fx_fig.update_layout(template="plotly_dark",height=260,margin=dict(l=20,r=20,t=20,b=20))
        fx_fig.update_yaxes(title_text="EUR")
        st.plotly_chart(fx_fig,use_container_width=True)
