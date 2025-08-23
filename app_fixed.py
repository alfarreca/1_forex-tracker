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
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; text-align: center; margin-bottom: .5rem; }
    .currency-card { background:#f6f9ff; border-left:4px solid #1f77b4; border-radius:12px; padding:10px 14px; }
</style>
""", unsafe_allow_html=True)

PERIOD_MAP = {"1M":"1mo","3M":"3mo","6M":"6mo","1Y":"1y","2Y":"2y","5Y":"5y"}
PAIRS = {
    "EUR/USD":"EURUSD=X",   # USD per 1 EUR
    "EUR/CNY":"EURCNY=X",   # CNY per 1 EUR
    "EUR/AUD":"EURAUD=X",
    "EUR/GBP":"EURGBP=X",
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def fmt_num(x, decimals=4, default="N/A"):
    """Safe formatter for numpy/pandas scalars; accepts Series too."""
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
    """Use Ticker.history for robustness; coerce to numeric and drop empty rows."""
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
    """Price + MAs + Volume chart. For 1M we allow min_periods=1 so lines are visible."""
    df = df.copy()
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(f"{title} Price", "Volume"), row_width=[0.25, 0.75]
    )

    # Price
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"), row=1, col=1)

    # Moving averages
    for p in ma_periods:
        try:
            p = int(p)
        except Exception:
            continue
        if len(df) == 0:
            continue
        minp = 1 if period == "1M" else p
        ma = df["Close"].rolling(p, min_periods=minp).mean()
        fig.add_trace(
            go.Scatter(x=df.index, y=ma, name=f"MA{p}", line=dict(dash="dash")),
            row=1, col=1
        )

    # Volume
    if "Volume" in df.columns and df["Volume"].notna().any():
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"].fillna(0), name="Volume", opacity=0.5),
            row=2, col=1
        )

    fig.update_layout(height=560, legend=dict(orientation="h", x=1, xanchor="right", y=1.08))
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

# ---------- FX P/L calculators ----------
def fx_pl_from_eurusd(close_series: pd.Series, usd_amount: float = 1000.0):
    """
    FX-only impact for converting a USD amount to EUR using EURUSD=X (USD per EUR).
    EUR value = USD / EURUSD.
    """
    s = close_series.dropna().astype(float)
    if s.empty or len(s) < 2:
        return None
    eur_series = usd_amount / s
    start_eur = eur_series.iloc[0]
    end_eur   = eur_series.iloc[-1]
    abs_pl_eur = float(end_eur - start_eur)
    pct_pl     = (end_eur / start_eur - 1.0) * 100.0
    out = pd.DataFrame({"EUR_Value": eur_series})
    stats = {"start_eur": start_eur, "end_eur": end_eur,
             "abs_pl_eur": abs_pl_eur, "pct_pl": pct_pl}
    return out, stats

def fx_pl_from_eurcny(close_series: pd.Series, cny_amount: float = 7000.0):
    """
    FX-only impact for converting a CNY amount to EUR using EURCNY=X (CNY per EUR).
    EUR value = CNY / EURCNY.
    """
    s = close_series.dropna().astype(float)
    if s.empty or len(s) < 2:
        return None
    eur_series = cny_amount / s
    start_eur = eur_series.iloc[0]
    end_eur   = eur_series.iloc[-1]
    abs_pl_eur = float(end_eur - start_eur)
    pct_pl     = (end_eur / start_eur - 1.0) * 100.0
    out = pd.DataFrame({"EUR_Value": eur_series})
    stats = {"start_eur": start_eur, "end_eur": end_eur,
             "abs_pl_eur": abs_pl_eur, "pct_pl": pct_pl}
    return out, stats

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.markdown('<div class="main-header">💱 EUR Forex Tracker</div>', unsafe_allow_html=True)

st.sidebar.title("⚙️ Settings")
selected = st.sidebar.multiselect("Select Currency Pairs:", list(PAIRS.keys()), default=list(PAIRS.keys()))
period = st.sidebar.selectbox("Time Period:", list(PERIOD_MAP.keys()), index=0)
ma_periods = st.sidebar.multiselect("Moving Averages:", [5,10,20,50,100,200], default=[20,50])

# FX impact inputs (show only if relevant pair is selected)
usd_amount = None
cny_amount = None
if "EUR/USD" in selected:
    usd_amount = st.sidebar.number_input("USD amount for FX impact (EUR/USD)", min_value=100.0, value=1000.0, step=100.0)
if "EUR/CNY" in selected:
    cny_amount = st.sidebar.number_input("CNY amount for FX impact (EUR/CNY)", min_value=1000.0, value=7000.0, step=500.0)

auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 minutes", value=False)
if auto_refresh:
    st.sidebar.caption("Next update " + (datetime.now() + timedelta(minutes=5)).strftime("%H:%M:%S"))

if not selected:
    st.info("Pick at least one pair on the left.")
    st.stop()

# ------------------------------------------------------------
# Fetch
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Current rates
# ------------------------------------------------------------
st.subheader("📊 Current Exchange Rates")
cols = st.columns(len(data_map))
for col, (name, df) in zip(cols, data_map.items()):
    with col:
        last = df["Close"].iloc[-1] if not df.empty else np.nan
        st.markdown(f'<div class="currency-card"><h4>{name}</h4><h2>{fmt_num(last,4)}</h2></div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# Charts + metrics + FX impacts
# ------------------------------------------------------------
st.subheader("📈 Price Charts with Moving Averages")
for name, df in data_map.items():
    with st.expander(f"{name} Chart", expanded=True):
        try:
            fig = make_chart(df, name, ma_periods, period)
            st.plotly_chart(fig, use_container_width=True)
            section_metrics(df)

            # --- FX-only impact for EUR/USD (USD -> EUR) ---
            if name == "EUR/USD" and usd_amount:
                res = fx_pl_from_eurusd(df["Close"], usd_amount)
                if res is not None:
                    eur_df, stats = res
                    st.markdown("**💶 FX-only impact on a USD asset (converted to EUR)**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Start (EUR)", fmt_num(stats["start_eur"], 2))
                    c2.metric("End (EUR)",   fmt_num(stats["end_eur"], 2))
                    sign = "+" if stats["abs_pl_eur"] >= 0 else ""
                    c3.metric("P/L (FX only)", f"{sign}{fmt_num(stats['abs_pl_eur'], 2)} EUR",
                              f"{fmt_num(stats['pct_pl'], 2)}%")
                    fx_fig = go.Figure()
                    fx_fig.add_trace(go.Scatter(x=eur_df.index, y=eur_df["EUR_Value"],
                                                name=f"EUR value of ${int(usd_amount)}"))
                    fx_fig.update_layout(height=260, margin=dict(l=20,r=20,t=20,b=20))
                    fx_fig.update_yaxes(title_text="EUR")
                    st.plotly_chart(fx_fig, use_container_width=True)
                    st.caption("FX effect only — excludes any change in the underlying US asset price.")

            # --- FX-only impact for EUR/CNY (CNY -> EUR) ---
            if name == "EUR/CNY" and cny_amount:
                res = fx_pl_from_eurcny(df["Close"], cny_amount)
                if res is not None:
                    eur_df, stats = res
                    st.markdown("**💶 FX-only impact on a CNY asset (converted to EUR)**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Start (EUR)", fmt_num(stats["start_eur"], 2))
                    c2.metric("End (EUR)",   fmt_num(stats["end_eur"], 2))
                    sign = "+" if stats["abs_pl_eur"] >= 0 else ""
                    c3.metric("P/L (FX only)", f"{sign}{fmt_num(stats['abs_pl_eur'], 2)} EUR",
                              f"{fmt_num(stats['pct_pl'], 2)}%")
                    fx_fig = go.Figure()
                    fx_fig.add_trace(go.Scatter(x=eur_df.index, y=eur_df["EUR_Value"],
                                                name=f"EUR value of ¥{int(cny_amount)}"))
                    fx_fig.update_layout(height=260, margin=dict(l=20,r=20,t=20,b=20))
                    fx_fig.update_yaxes(title_text="EUR")
                    st.plotly_chart(fx_fig, use_container_width=True)
                    st.caption("FX effect only — excludes any change in the underlying Chinese asset price.")

        except Exception as e:
            st.error(f"Error creating chart for {name}: {e}")

# manual refresh button if auto-refresh is on
if auto_refresh and st.sidebar.button("🔄 Refresh Now"):
    st.rerun()
