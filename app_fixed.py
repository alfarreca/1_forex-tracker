import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# ------------------------------------------------------------
# App setup (theme comes from .streamlit/config.toml)
# ------------------------------------------------------------
st.set_page_config(page_title="EUR Forex Tracker", page_icon="💱", layout="wide")

PERIOD_MAP = {"1M":"1mo","3M":"3mo","6M":"6mo","1Y":"1y","2Y":"2y","5Y":"5y"}
PAIRS = {
    "EUR/USD": "EURUSD=X",   # USD per 1 EUR
    "EUR/CNY": "EURCNY=X",   # CNY per 1 EUR
    "EUR/AUD": "EURAUD=X",   # AUD per 1 EUR
    "EUR/GBP": "EURGBP=X",
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def fmt_num(x, decimals=4, default="N/A"):
    """Safe numeric formatter that handles pandas/NumPy scalars and Series."""
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
    """Robust price history fetch via yfinance, coerced to numeric."""
    hist = yf.Ticker(ticker).history(period=PERIOD_MAP[period])
    if hist is None or hist.empty:
        return pd.DataFrame()
    hist.index = pd.to_datetime(hist.index, errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in hist.columns:
            hist[c] = pd.to_numeric(hist[c], errors="coerce")
    if "Close" in hist.columns:
        hist = hist[hist["Close"].notna()]
    return hist

def make_chart(df: pd.DataFrame, title: str, ma_periods, period: str):
    """Main OHLC-derived chart with MAs + volume, dark via plotly template."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(f"{title} Price", "Volume"), row_width=[0.25, 0.75]
    )

    # Price
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"], name="Price"),
        row=1, col=1
    )

    # Moving averages
    for p in ma_periods:
        try:
            p = int(p)
        except Exception:
            continue
        if len(df) == 0:
            continue
        minp = 1 if period == "1M" else p  # show MAs on short windows
        ma = df["Close"].rolling(p, min_periods=minp).mean()
        fig.add_trace(
            go.Scatter(x=df.index, y=ma, name=f"MA{p}", line=dict(dash="dot")),
            row=1, col=1
        )

    # Volume
    if "Volume" in df.columns and df["Volume"].notna().any():
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"].fillna(0), name="Volume", opacity=0.5),
            row=2, col=1
        )

    # Dark charts (respect app theme colors)
    fig.update_layout(
        template="plotly_dark",
        height=560,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.08),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

def section_metrics(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    last = df["Close"].iloc[-1] if not df.empty else np.nan
    with c1: st.metric("Current", fmt_num(last, 4))
    if len(df) > 1:
        first = df["Close"].iloc[0]
        change = (float(last) - float(first)) / float(first) * 100.0 if first else np.nan
        with c2: st.metric("Total Change", fmt_num(change, 2) + "%")
    else:
        with c2: st.metric("Total Change", "N/A")
    with c3: st.metric("High", fmt_num(df["High"].max() if "High" in df else np.nan, 4))
    with c4: st.metric("Low", fmt_num(df["Low"].min() if "Low" in df else np.nan, 4))

# -------- FX-only impact calculators (amount in local ccy -> EUR) --------
def fx_pl_inverse_quote(close_series: pd.Series, amount_local: float):
    """
    For pairs quoted as LOCAL per EUR (EUR/USD, EUR/CNY, EUR/AUD):
    EUR value = LOCAL_amount / (LOCAL per EUR) = amount / price.
    Returns (series_df, stats_dict) or None if not enough data.
    """
    s = close_series.dropna().astype(float)
    if s.empty or len(s) < 2:
        return None
    eur_series = amount_local / s
    start, end = eur_series.iloc[0], eur_series.iloc[-1]
    stats = {
        "start_eur": float(start),
        "end_eur": float(end),
        "abs_pl_eur": float(end - start),
        "pct_pl": float((end / start - 1.0) * 100.0)
    }
    return pd.DataFrame({"EUR_Value": eur_series}), stats

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("💱 EUR Forex Tracker")

st.sidebar.header("⚙️ Settings")
selected = st.sidebar.multiselect("Select Currency Pairs", list(PAIRS.keys()), default=list(PAIRS.keys()))
period = st.sidebar.selectbox("Time Period", list(PERIOD_MAP.keys()), index=0)
ma_periods = st.sidebar.multiselect("Moving Averages", [5, 10, 20, 50, 100, 200], default=[20, 50])

# FX impact inputs only if relevant pairs are selected
usd_amount = cny_amount = aud_amount = None
if "EUR/USD" in selected:
    usd_amount = st.sidebar.number_input("USD amount (EUR/USD)", min_value=100.0, value=1000.0, step=100.0)
if "EUR/CNY" in selected:
    cny_amount = st.sidebar.number_input("CNY amount (EUR/CNY)", min_value=1000.0, value=7000.0, step=500.0)
if "EUR/AUD" in selected:
    aud_amount = st.sidebar.number_input("AUD amount (EUR/AUD)", min_value=100.0, value=1500.0, step=100.0)

# ------------------------------------------------------------
# Fetch
# ------------------------------------------------------------
data_map = {}
for name in selected:
    df = get_history(PAIRS[name], period)
    if not df.empty:
        data_map[name] = df

if not data_map:
    st.warning("No data available. Try a different period or selection.")
    st.stop()

# ------------------------------------------------------------
# Current rates
# ------------------------------------------------------------
st.subheader("📊 Current Exchange Rates")
cols = st.columns(len(data_map))
for col, (name, df) in zip(cols, data_map.items()):
    with col:
        last = df["Close"].iloc[-1] if not df.empty else np.nan
        st.metric(name, fmt_num(last, 4))

# ------------------------------------------------------------
# Charts + FX impact
# ------------------------------------------------------------
st.subheader("📈 Price Charts with Moving Averages")
for name, df in data_map.items():
    with st.expander(f"{name} Chart", expanded=True):
        fig = make_chart(df, name, ma_periods, period)
        st.plotly_chart(fig, use_container_width=True)
        section_metrics(df)

        # FX-only impact blocks (LOCAL -> EUR)
        if name == "EUR/USD" and usd_amount:
            res = fx_pl_inverse_quote(df["Close"], usd_amount)
            if res:
                eur_df, stats = res
                st.markdown("**💶 USD asset → EUR (FX-only)**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Start (EUR)", fmt_num(stats["start_eur"], 2))
                c2.metric("End (EUR)", fmt_num(stats["end_eur"], 2))
                sign = "+" if stats["abs_pl_eur"] >= 0 else ""
                c3.metric("P/L (FX)", f"{sign}{fmt_num(stats['abs_pl_eur'], 2)} EUR", f"{fmt_num(stats['pct_pl'], 2)}%")

                fx_fig = go.Figure()
                fx_fig.add_trace(go.Scatter(x=eur_df.index, y=eur_df["EUR_Value"], name=f"EUR value of ${int(usd_amount)}"))
                fx_fig.update_layout(template="plotly_dark", height=260, margin=dict(l=20, r=20, t=20, b=20))
                fx_fig.update_yaxes(title_text="EUR")
                st.plotly_chart(fx_fig, use_container_width=True)
                st.caption("FX effect only — excludes any change in the underlying US asset price.")

        if name == "EUR/CNY" and cny_amount:
            res = fx_pl_inverse_quote(df["Close"], cny_amount)
            if res:
                eur_df, stats = res
                st.markdown("**💶 CNY asset → EUR (FX-only)**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Start (EUR)", fmt_num(stats["start_eur"], 2))
                c2.metric("End (EUR)", fmt_num(stats["end_eur"], 2))
                sign = "+" if stats["abs_pl_eur"] >= 0 else ""
                c3.metric("P/L (FX)", f"{sign}{fmt_num(stats['abs_pl_eur'], 2)} EUR", f"{fmt_num(stats['pct_pl'], 2)}%")

                fx_fig = go.Figure()
                fx_fig.add_trace(go.Scatter(x=eur_df.index, y=eur_df["EUR_Value"], name=f"EUR value of ¥{int(cny_amount)}"))
                fx_fig.update_layout(template="plotly_dark", height=260, margin=dict(l=20, r=20, t=20, b=20))
                fx_fig.update_yaxes(title_text="EUR")
                st.plotly_chart(fx_fig, use_container_width=True)
                st.caption("FX effect only — excludes any change in the underlying Chinese asset price.")

        if name == "EUR/AUD" and aud_amount:
            res = fx_pl_inverse_quote(df["Close"], aud_amount)
            if res:
                eur_df, stats = res
                st.markdown("**💶 AUD asset → EUR (FX-only)**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Start (EUR)", fmt_num(stats["start_eur"], 2))
                c2.metric("End (EUR)", fmt_num(stats["end_eur"], 2))
                sign = "+" if stats["abs_pl_eur"] >= 0 else ""
                c3.metric("P/L (FX)", f"{sign}{fmt_num(stats['abs_pl_eur'], 2)} EUR", f"{fmt_num(stats['pct_pl'], 2)}%")

                fx_fig = go.Figure()
                fx_fig.add_trace(go.Scatter(x=eur_df.index, y=eur_df["EUR_Value"], name=f"EUR value of A${int(aud_amount)}"))
                fx_fig.update_layout(template="plotly_dark", height=260, margin=dict(l=20, r=20, t=20, b=20))
                fx_fig.update_yaxes(title_text="EUR")
                st.plotly_chart(fx_fig, use_container_width=True)
                st.caption("FX effect only — excludes any change in the underlying Australian asset price.")
