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
    "EUR/GBP": "EURGBP=X",   # GBP per 1 EUR
    "EUR/JPY": "EURJPY=X",   # JPY per 1 EUR
    "EUR/SEK": "EURSEK=X",   # SEK per 1 EUR
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

def transform_series(s: pd.Series, mode: str):
    """
    Transform a price series based on normalization mode.
    - 'Actual' -> unchanged, ylabel 'Price'
    - 'Normalized to 100' -> s / s0 * 100, ylabel 'Index (100=start)'
    - 'Percent change (%)' -> (s / s0 - 1) * 100, ylabel '% from start'
    """
    s = s.dropna().astype(float)
    if s.empty:
        return s, "Price"
    s0 = s.iloc[0]
    if mode == "Actual" or s0 == 0:
        return s, "Price"
    if mode == "Normalized to 100":
        return (s / s0) * 100.0, "Index (100 = start)"
    if mode == "Percent change (%)":
        return (s / s0 - 1.0) * 100.0, "% from start"
    return s, "Price"

def make_chart(df: pd.DataFrame, title: str, ma_periods, period: str, norm_mode: str):
    """Main chart with MAs + volume, dark via plotly template, with normalization modes."""
    price_transformed, y_label = transform_series(df["Close"], norm_mode)

    # For MAs: compute on raw prices, then transform with the SAME baseline as price
    ma_series = []
    for p in ma_periods:
        try:
            p = int(p)
        except Exception:
            continue
        if len(df) == 0:
            continue
        minp = 1 if period == "1M" else p
        ma_raw = df["Close"].rolling(p, min_periods=minp).mean()
        ma_t, _ = transform_series(ma_raw, norm_mode)
        ma_series.append((p, ma_t))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(f"{title} ({norm_mode})", "Volume"), row_width=[0.25, 0.75]
    )

    # Price
    fig.add_trace(go.Scatter(x=price_transformed.index, y=price_transformed, name="Price"), row=1, col=1)

    # MAs
    for p, m in ma_series:
        fig.add_trace(
            go.Scatter(x=m.index, y=m, name=f"MA{p}", line=dict(dash="dot")),
            row=1, col=1
        )

    # Volume (only meaningful in Actual scale, but we keep it for context)
    if "Volume" in df.columns and df["Volume"].notna().any():
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"].fillna(0), name="Volume", opacity=0.5),
            row=2, col=1
        )

    fig.update_layout(
        template="plotly_dark",
        height=560,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.08),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_yaxes(title_text=y_label, row=1, col=1)
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

# -------- FX-only impact (amount in local ccy -> EUR), for inverse-quoted pairs --------
def fx_pl_inverse_quote(close_series: pd.Series, amount_local: float):
    """
    For pairs quoted as LOCAL per EUR (EUR/USD, EUR/CNY, EUR/AUD, EUR/GBP, EUR/JPY, EUR/SEK):
    EUR value = LOCAL_amount / price   (since price = LOCAL per EUR).
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
norm_mode = st.sidebar.selectbox("Chart scale", ["Actual", "Normalized to 100", "Percent change (%)"], index=0)
show_comparison = st.sidebar.checkbox("Show normalized comparison chart", value=True)

# FX impact inputs (only if relevant pairs are selected)
usd_amount = cny_amount = aud_amount = gbp_amount = jpy_amount = sek_amount = None
if "EUR/USD" in selected:
    usd_amount = st.sidebar.number_input("USD amount (EUR/USD)", min_value=100.0, value=1000.0, step=100.0)
if "EUR/CNY" in selected:
    cny_amount = st.sidebar.number_input("CNY amount (EUR/CNY)", min_value=1000.0, value=7000.0, step=500.0)
if "EUR/AUD" in selected:
    aud_amount = st.sidebar.number_input("AUD amount (EUR/AUD)", min_value=100.0, value=1500.0, step=100.0)
if "EUR/GBP" in selected:
    gbp_amount = st.sidebar.number_input("GBP amount (EUR/GBP)", min_value=50.0, value=750.0, step=50.0)
if "EUR/JPY" in selected:
    jpy_amount = st.sidebar.number_input("JPY amount (EUR/JPY)", min_value=5000.0, value=100000.0, step=5000.0)
if "EUR/SEK" in selected:
    sek_amount = st.sidebar.number_input("SEK amount (EUR/SEK)", min_value=1000.0, value=9500.0, step=500.0)

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
# Optional: Normalized comparison across selected pairs
# ------------------------------------------------------------
if show_comparison and len(data_map) > 1:
    st.subheader("📊 Normalized Comparison (Index 100 = start)")
    comp_fig = go.Figure()
    for name, df in data_map.items():
        norm, _ = transform_series(df["Close"], "Normalized to 100")
        if norm.empty: 
            continue
        comp_fig.add_trace(go.Scatter(x=norm.index, y=norm, name=name))
    comp_fig.update_layout(template="plotly_dark", height=420, legend=dict(orientation="h", x=1, xanchor="right"))
    comp_fig.update_yaxes(title_text="Index (100 = start)")
    st.plotly_chart(comp_fig, use_container_width=True)

# ------------------------------------------------------------
# Per-pair charts + FX impact
# ------------------------------------------------------------
st.subheader("📈 Price Charts with Moving Averages")
for name, df in data_map.items():
    with st.expander(f"{name} Chart", expanded=True):
        fig = make_chart(df, name, ma_periods, period, norm_mode)
        st.plotly_chart(fig, use_container_width=True)
        section_metrics(df)

        # FX-only impact blocks (LOCAL -> EUR), shown regardless of normalization mode
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
                fx_fig.add_trace(go.Scatter(x=eur_df.index, y=eur_df["EUR_Value"], name=f"EUR value of ¥{int(cny_amount)} (CNY)"))
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

        if name == "EUR/GBP" and gbp_amount:
            res = fx_pl_inverse_quote(df["Close"], gbp_amount)
            if res:
                eur_df, stats = res
                st.markdown("**💶 GBP asset → EUR (FX-only)**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Start (EUR)", fmt_num(stats["start_eur"], 2))
                c2.metric("End (EUR)", fmt_num(stats["end_eur"], 2))
                sign = "+" if stats["abs_pl_eur"] >= 0 else ""
                c3.metric("P/L (FX)", f"{sign}{fmt_num(stats['abs_pl_eur'], 2)} EUR", f"{fmt_num(stats['pct_pl'], 2)}%")
                fx_fig = go.Figure()
                fx_fig.add_trace(go.Scatter(x=eur_df.index, y=eur_df["EUR_Value"], name=f"EUR value of £{int(gbp_amount)}"))
                fx_fig.update_layout(template="plotly_dark", height=260, margin=dict(l=20, r=20, t=20, b=20))
                fx_fig.update_yaxes(title_text="EUR")
                st.plotly_chart(fx_fig, use_container_width=True)
                st.caption("FX effect only — excludes any change in the underlying British asset price.")

        if name == "EUR/JPY" and jpy_amount:
            res = fx_pl_inverse_quote(df["Close"], jpy_amount)
            if res:
                eur_df, stats = res
                st.markdown("**💶 JPY asset → EUR (FX-only)**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Start (EUR)", fmt_num(stats["start_eur"], 2))
                c2.metric("End (EUR)", fmt_num(stats["end_eur"], 2))
                sign = "+" if stats["abs_pl_eur"] >= 0 else ""
                c3.metric("P/L (FX)", f"{sign}{fmt_num(stats['abs_pl_eur'], 2)} EUR", f"{fmt_num(stats['pct_pl'], 2)}%")
                fx_fig = go.Figure()
                fx_fig.add_trace(go.Scatter(x=eur_df.index, y=eur_df["EUR_Value"], name=f"EUR value of ¥{int(jpy_amount)} (JPY)"))
                fx_fig.update_layout(template="plotly_dark", height=260, margin=dict(l=20, r=20, t=20, b=20))
                fx_fig.update_yaxes(title_text="EUR")
                st.plotly_chart(fx_fig, use_container_width=True)
                st.caption("FX effect only — excludes any change in the underlying Japanese asset price.")

        if name == "EUR/SEK" and sek_amount:
            res = fx_pl_inverse_quote(df["Close"], sek_amount)
            if res:
                eur_df, stats = res
                st.markdown("**💶 SEK asset → EUR (FX-only)**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Start (EUR)", fmt_num(stats["start_eur"], 2))
                c2.metric("End (EUR)", fmt_num(stats["end_eur"], 2))
                sign = "+" if stats["abs_pl_eur"] >= 0 else ""
                c3.metric("P/L (FX)", f"{sign}{fmt_num(stats['abs_pl_eur'], 2)} EUR", f"{fmt_num(stats['pct_pl'], 2)}%")
                fx_fig = go.Figure()
                fx_fig.add_trace(go.Scatter(x=eur_df.index, y=eur_df["EUR_Value"], name=f"EUR value of kr{int(sek_amount)} (SEK)"))
                fx_fig.update_layout(template="plotly_dark", height=260, margin=dict(l=20, r=20, t=20, b=20))
                fx_fig.update_yaxes(title_text="EUR")
                st.plotly_chart(fx_fig, use_container_width=True)
                st.caption("FX effect only — excludes any change in the underlying Swedish asset price.")
