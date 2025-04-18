import yfinance as yf
import pandas as pd
import streamlit as st
import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import time
import streamlit.components.v1 as components

st.set_page_config(page_title="Stock Dashboard", layout="wide")

@st.cache_data
def load_watchlists():
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(sp500_url)[0]
    sp500 = sp500_table["Symbol"].tolist()
    sp500 = [s.replace(".", "-") for s in sp500]

    nasdaq_url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    nasdaq_table = pd.read_html(nasdaq_url)[4]
    nasdaq = nasdaq_table["Ticker"].tolist()

    asx200 = [
        "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WBC.AX", "WES.AX", "TLS.AX", "WOW.AX",
        "MQG.AX", "ANZ.AX", "FMG.AX", "WDS.AX", "GMG.AX", "TCL.AX", "QBE.AX", "BXB.AX",
        "SUN.AX", "REH.AX", "COL.AX", "ALL.AX", "REX.AX", "MIN.AX", "SHL.AX", "STO.AX",
        "RHC.AX", "APA.AX", "S32.AX", "ORG.AX", "ILU.AX", "CPU.AX", "JHX.AX", "IAG.AX",
        "AMP.AX", "CNU.AX", "TPG.AX", "COH.AX", "CAR.AX", "NCM.AX", "QAN.AX", "ALU.AX",
        "XRO.AX", "TWE.AX", "DMP.AX", "ALD.AX", "EDV.AX", "CWY.AX", "MPL.AX"
    ]

    return sp500, nasdaq, asx200

sp500, nasdaq, asx200 = load_watchlists()

index_choice = st.selectbox("Select Market Index to Scan:", ["S&P 500", "NASDAQ-100", "ASX 200"])
watchlist = sp500 if index_choice == "S&P 500" else nasdaq if index_choice == "NASDAQ-100" else asx200

if "selected_stocks" not in st.session_state:
    st.session_state.selected_stocks = []
if "scan_complete" not in st.session_state:
    st.session_state.scan_complete = False
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

REFRESH_INTERVAL = 600

def notify(title, message):
    js = f"""
        <script>
            if (Notification.permission === "granted") {{
                new Notification("{title}", {{body: "{message}"}});
            }} else if (Notification.permission !== "denied") {{
                Notification.requestPermission().then(function(permission) {{
                    if (permission === "granted") {{
                        new Notification("{title}", {{body: "{message}"}});
                    }}
                }});
            }}
        </script>
    """
    components.html(js)

# --- Enhanced Scoring Function ---
def advanced_score(data, return_contribs=False):
    score = 0
    weights = {
        "ma": 0.25,
        "rsi": 0.15,
        "macd": 0.20,
        "volume": 0.15,
        "trendline": 0.15,
        "stochastic": 0.10
    }
    contribs = {}

    data["MA_short"] = data["Close"].rolling(window=8).mean()
    data["MA_long"] = data["Close"].rolling(window=24).mean()
    if data["MA_long"].iloc[-1] != 0:
        ma_diff = data["MA_short"].iloc[-1] - data["MA_long"].iloc[-1]
        contribs["MA Crossover"] = weights["ma"] * (ma_diff / data["MA_long"].iloc[-1])
    else:
        contribs["MA Crossover"] = 0
    score += contribs["MA Crossover"]

    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_val = rsi.iloc[-1]
    contribs["RSI"] = weights["rsi"] * (1 - abs(rsi_val - 50) / 50)
    score += contribs["RSI"]

    exp1 = data["Close"].ewm(span=12, adjust=False).mean()
    exp2 = data["Close"].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_diff = macd.iloc[-1] - signal.iloc[-1]
    contribs["MACD"] = weights["macd"] * (macd_diff / (abs(macd.iloc[-1]) + 1e-5))
    score += contribs["MACD"]

    data["Volume_MA"] = data["Volume"].rolling(window=20).mean()
    vol_ratio = data["Volume"].iloc[-1] / (data["Volume_MA"].iloc[-1] + 1e-5)
    contribs["Volume Spike"] = weights["volume"] * min(1, vol_ratio / 1.5)
    score += contribs["Volume Spike"]

    data["local_max"] = data["High"].rolling(5, center=True).max()
    data["local_min"] = data["Low"].rolling(5, center=True).min()
    close = data["Close"].iloc[-1]
    contribs["Trend Break ↑"] = weights["trendline"] if close > data["local_max"].iloc[-2] else 0
    contribs["Trend Break ↓"] = weights["trendline"] if close < data["local_min"].iloc[-2] else 0
    score += max(contribs["Trend Break ↑"], contribs["Trend Break ↓"])

    low14 = data["Low"].rolling(window=14).min()
    high14 = data["High"].rolling(window=14).max()
    percent_k = 100 * ((data["Close"] - low14) / (high14 - low14 + 1e-5))
    percent_d = percent_k.rolling(window=3).mean()
    stoch_strength = percent_k.iloc[-1] - percent_d.iloc[-1]
    contribs["Stochastic"] = weights["stochastic"] * (stoch_strength / 100)
    score += contribs["Stochastic"]

    score = round(score * 100, 2)

    if return_contribs:
        return score, contribs

    return score

# --- SCANNING LOGIC USING ADVANCED SCORE ---
results = []
for ticker in watchlist:
    try:
        data = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if len(data) < 30 or data.isnull().values.any():
            continue
        score_val, contribs = advanced_score(data, return_contribs=True)
        signal = "BUY" if score_val > 65 else "HOLD" if score_val > 45 else "SELL"
        results.append({
            "Ticker": ticker,
            "Price": round(data["Close"].iloc[-1], 2),
            "RSI": round(100 - (100 / (1 + data["Close"].diff().gt(0).rolling(14).sum().iloc[-1])), 2),
            "Volume Spike": "Yes" if data["Volume"].iloc[-1] > 1.5 * data["Volume"].rolling(20).mean().iloc[-1] else "No",
            "Signal": signal,
            "Score": round(score_val, 2),
            "Confidence (%)": round(score_val, 2),
            "Time": datetime.datetime.now()
        })
        print(f"{ticker} Score: {score_val}% | Contribs: {contribs}")
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

results_df = pd.DataFrame(results).sort_values("Confidence (%)", ascending=False).head(20)

# --- DISPLAY RESULTS ---
if not results_df.empty:
    st.subheader("📊 Top 20 Scan Results")
    st.dataframe(results_df.style.format({"Price": "${:.2f}", "Score": "{:.2f}", "Confidence (%)": "{:.2f}"}), use_container_width=True)
    st.success("✅ Confidence Scores updated using advanced logic.")

    st.markdown("### ✅ Tracked Stocks")
    for ticker in st.session_state.selected_stocks:
        match = results_df[results_df["Ticker"] == ticker]
        if not match.empty:
            conf = match.iloc[0]["Confidence (%)"]
            emoji = "🟢" if conf > 70 else ("🟡" if conf > 50 else "⚪")
            st.write(f"{emoji} {ticker} - Confidence: {conf}%")
if not results_df.empty:
    # [Your results display logic...]
else:
    st.info("👆 Click 'Start Scan' to begin scanning the selected market.")

# --- Sidebar: Tracked Stocks ---
st.sidebar.markdown("### 📌 Tracked Stocks")
if "scan_results" in st.session_state and not st.session_state.scan_results.empty:
    top20_tickers = list(st.session_state.scan_results["Ticker"].unique())
else:
    top20_tickers = watchlist

selected_to_track = st.sidebar.multiselect("Monitor:", options=top20_tickers, default=st.session_state.selected_stocks, key="tracked_stocks")
if selected_to_track:
    st.session_state.selected_stocks = selected_to_track

# --- Auto-refresh Logic ---
if st.session_state.selected_stocks and (time.time() - st.session_state.last_refresh > REFRESH_INTERVAL):
    st.experimental_rerun()

# --- Main UI ---
st.title("📈 Advanced Stock Scanner")

if st.button("🔍 Start Scan"):
    results = []
    with st.spinner("Scanning stocks with advanced scoring logic..."):
        for ticker in watchlist:
            try:
                df = yf.download(ticker, period="30d", interval="1h", progress=False)
                if df.empty or len(df) < 25:
                    continue
                score = advanced_score(df)
                print(f"DEBUG: {ticker} Score = {score}")  # DEBUG LOG
                results.append({
                    "Ticker": ticker,
                    "Latest Price": round(df["Close"].iloc[-1], 2),
                    "Confidence (%)": score,
                    "Signal": "🟢 BUY" if score > 70 else "🟡 WATCH" if score > 50 else "⚪ HOLD"
                })
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

    st.session_state.scan_results = pd.DataFrame(results).sort_values(by="Confidence (%)", ascending=False).head(20).reset_index(drop=True)
    st.session_state.scan_complete = True
    st.session_state.last_refresh = time.time()
    notify("Scan Complete", f"Scanned {len(results)} stocks. Top pick: {st.session_state.scan_results.iloc[0]['Ticker']}")

if st.session_state.get("scan_complete") and "scan_results" in st.session_state:
    st.subheader("📊 Top 20 Scan Results")
    st.dataframe(st.session_state.scan_results, use_container_width=True)

    st.markdown("### 🔍 Confidence Breakdown by Stock")
    if not st.session_state.scan_results.empty:
        tickers = list(st.session_state.scan_results["Ticker"].dropna().unique())
        selected = st.selectbox("Select a stock to inspect weights:", options=tickers, key="confidence_breakdown")
        if selected:
            df = yf.download(selected, period="30d", interval="1h", progress=False)
            _, contribs = advanced_score(df, return_contribs=True)
            bar = go.Figure(go.Bar(
                x=list(contribs.values()),
                y=list(contribs.keys()),
                orientation='h',
                marker=dict(color="darkcyan")
            ))
            bar.update_layout(height=350, xaxis_title="Weight", yaxis_title="Indicator")
            st.plotly_chart(bar, use_container_width=True)

    fig = go.Figure(go.Bar(
        x=st.session_state.scan_results["Confidence (%)"],
        y=st.session_state.scan_results["Ticker"],
        orientation="h",
        marker_color="mediumseagreen"
    ))
    fig.update_layout(
        height=600,
        xaxis_title="Confidence (%)",
        yaxis_title="Ticker",
        margin=dict(l=100, r=20, t=30, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 Signal Visualization")
    for _, row in st.session_state.scan_results.iterrows():
        emoji = "🟢" if row["Confidence (%)"] > 70 else ("🟡" if row["Confidence (%)"] > 50 else "⚪")
        st.write(f"{emoji} {row['Ticker']} — {row['Signal']} — Confidence: {row['Confidence (%)']}%")
else:
    st.warning("⚠️ No scan results to show breakdown. Please run a scan first.")

# --- Advanced Scoring Function Inserted ---
def advanced_score(data, return_contribs=False):
    score = 0
    weights = {
        "ma": 0.25,
        "rsi": 0.15,
        "macd": 0.20,
        "volume": 0.15,
        "trendline": 0.15,
        "stochastic": 0.10
    }
    contribs = {}

    data["MA_short"] = data["Close"].rolling(window=8).mean()
    data["MA_long"] = data["Close"].rolling(window=24).mean()
    if data["MA_short"].iloc[-1] > data["MA_long"].iloc[-1]:
        score += weights["ma"]
        contribs["MA Crossover"] = weights["ma"]
    else:
        contribs["MA Crossover"] = 0

    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    if rsi.iloc[-1] < 30 or rsi.iloc[-1] > 70:
        score += weights["rsi"]
        contribs["RSI"] = weights["rsi"]
    else:
        contribs["RSI"] = 0

    exp1 = data["Close"].ewm(span=12, adjust=False).mean()
    exp2 = data["Close"].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    if macd.iloc[-1] > signal.iloc[-1]:
        score += weights["macd"]
        contribs["MACD"] = weights["macd"]
    else:
        contribs["MACD"] = 0

    data["Volume_MA"] = data["Volume"].rolling(window=20).mean()
    if data["Volume"].iloc[-1] > 1.5 * data["Volume_MA"].iloc[-1]:
        score += weights["volume"]
        contribs["Volume Spike"] = weights["volume"]
    else:
        contribs["Volume Spike"] = 0

    data["local_max"] = data["High"][(data["High"] == data["High"].rolling(5, center=True).max())]
    data["local_min"] = data["Low"][(data["Low"] == data["Low"].rolling(5, center=True).min())]
    if data["Close"].iloc[-1] > data["local_max"].shift(1).max():
        score += weights["trendline"]
        contribs["Trend Break ↑"] = weights["trendline"]
    elif data["Close"].iloc[-1] < data["local_min"].shift(1).min():
        score += weights["trendline"]
        contribs["Trend Break ↓"] = weights["trendline"]
    else:
        contribs["Trend Break ↑"] = 0
        contribs["Trend Break ↓"] = 0

    low14 = data["Low"].rolling(window=14).min()
    high14 = data["High"].rolling(window=14).max()
    percent_k = 100 * ((data["Close"] - low14) / (high14 - low14))
    percent_d = percent_k.rolling(window=3).mean()
    if percent_k.iloc[-1] > percent_d.iloc[-1]:
        score += weights["stochastic"]
        contribs["Stochastic"] = weights["stochastic"]
    else:
        contribs["Stochastic"] = 0

    if return_contribs:
        return round(score * 100, 2), contribs

    return round(score * 100, 2)

if st.button("🔍 Start Scan", key="scan") or (time.time() - st.session_state.last_refresh > REFRESH_INTERVAL and not st.session_state.scan_complete):
    st.session_state.scan_complete = False
    results = []
    with st.spinner("Scanning stocks..."):
        for stock in watchlist:
            result = analyze_stock(stock)
            if result:
                results.append(result)
    st.session_state.scan_results = pd.DataFrame(results)
    st.session_state.scan_complete = True
    st.session_state.last_refresh = time.time()
    notify("Scan Complete", f"Found {len(results)} candidates")

if st.session_state.get("scan_complete") and "scan_results" in st.session_state:
    df = st.session_state.scan_results
    st.subheader("📊 Top 20 Scan Results")
    top20 = df.sort_values(by="Confidence (%)", ascending=False).head(20)
    st.dataframe(top20, use_container_width=True)

    st.markdown("### 📈 Confidence Ranking")
    fig = go.Figure(go.Bar(
        x=top20["Confidence (%)"],
        y=top20["Ticker"],
        orientation="h",
        marker_color="mediumseagreen"
    ))
    fig.update_layout(height=600, xaxis_title="Confidence (%)", yaxis_title="Ticker")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔍 Signal Component Breakdown")
    if "MACD" in top20.columns:
        score_fig = go.Figure()
        score_fig.add_trace(go.Bar(name="MACD", x=top20["Ticker"], y=top20["MACD"]))
        score_fig.add_trace(go.Bar(name="MA Slope", x=top20["Ticker"], y=top20["MA Slope"]))
        score_fig.add_trace(go.Bar(name="Wick Strength", x=top20["Ticker"], y=top20["Wick Strength"]))
        score_fig.update_layout(barmode="group", xaxis_title="Ticker", yaxis_title="Score Value")
        st.plotly_chart(score_fig, use_container_width=True)

    st.markdown("### ✅ Tracked Stocks")
    for ticker in st.session_state.selected_stocks:
        match = top20[top20["Ticker"] == ticker]
        if not match.empty:
            conf = match.iloc[0]["Confidence (%)"]
            emoji = "🟢" if conf > 70 else ("🟡" if conf > 50 else "⚪")
            st.write(f"{emoji} {ticker} - Confidence: {conf}%")
else:
    st.info("👆 Click 'Start Scan' to begin scanning the selected market.")

    st.markdown("### ✅ Tracked Stocks")
    for ticker in st.session_state.selected_stocks:
        match = top20[top20["Ticker"] == ticker]
        if not match.empty:
            conf = match.iloc[0]["Confidence (%)"]
            emoji = "🟢" if conf > 70 else ("🟡" if conf > 50 else "⚪")
            st.write(f"{emoji} {ticker} - Confidence: {conf}%")
else:
    st.info("👆 Click 'Start Scan' to begin scanning the selected market.")

# Streamlit title and input
st.title("📈 Interactive Stock Dashboard")

ticker = st.text_input("Enter Stock Ticker Symbol", value="AAPL")

# Download data
data = yf.download(ticker, period="30d", interval="1h", auto_adjust=False)

# 🛠 Flatten multi-level columns if needed
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

# ✅ Check that we got valid data
if data.empty:
    st.error("❌ No data was returned. Try a different ticker.")
    st.stop()

required_cols = {"Open", "High", "Low", "Close"}
if not required_cols.issubset(set(data.columns)):
    st.error(f"❌ Missing expected columns. Got: {list(data.columns)}")
    st.stop()

# ✅ Now it's safe to clean up
data = data.dropna(subset=["Open", "High", "Low", "Close"]).copy()
data.index = pd.to_datetime(data.index)

# --- Candle Wick Analysis ---
data["UpperWick"] = data["High"] - data[["Close", "Open"]].max(axis=1)
data["LowerWick"] = data[["Close", "Open"]].min(axis=1) - data["Low"]
data["Body"] = abs(data["Close"] - data["Open"])

# Classify candle types
data["WickType"] = "Neutral"
data.loc[data["UpperWick"] > 2 * data["Body"], "WickType"] = "Long Upper"
data.loc[data["LowerWick"] > 2 * data["Body"], "WickType"] = "Long Lower"
data.loc[
    abs(data["UpperWick"] - data["LowerWick"]) / data["Body"].replace(0, 0.01) < 0.2,
    "WickType"
] = "Equal Wicks"

# ✅ TEMP: Preview last 10 candles with wick classification
st.write("🕯️ Last 10 Candles with Wick Type:")
st.write(data[["Open", "High", "Low", "Close", "WickType"]].tail(10))

# --- Calculate Moving Averages for Hourly Data ---
# Approx: 8 hours = short-term trend, 24 hours = 1 trading day
data["MA_short"] = data["Close"].rolling(window=8).mean()
data["MA_long"] = data["Close"].rolling(window=24).mean()

# --- Volume Moving Average (20-period, based on intraday scale) ---
data["Volume_MA_20"] = data["Volume"].rolling(window=20).mean()

# --- Flag unusually high volume spikes (greater than 1.5x the 20-period average) ---
data["HighVolumeSpike"] = data["Volume"] > (1.5 * data["Volume_MA_20"])

# --- Auto Trendlines (Support & Resistance) ---
window = 5
data["local_max"] = data["High"][(data["High"] == data["High"].rolling(window, center=True).max())]
data["local_min"] = data["Low"][(data["Low"] == data["Low"].rolling(window, center=True).min())]

# --- Detect Price Breakouts Above Resistance / Below Support ---
data["Crossed_Resistance"] = data["Close"] > data["local_max"].shift(1)
data["Crossed_Support"] = data["Close"] < data["local_min"].shift(1)

# ✅ MA crossover + volume spike (existing logic)
data.loc[
    (data["MA_short"] > data["MA_long"]) &
    (data["MA_short"].shift(1) <= data["MA_long"].shift(1)) &
    (data["HighVolumeSpike"]),
    "Signal"
] = "BUY"

data.loc[
    (data["MA_short"] < data["MA_long"]) &
    (data["MA_short"].shift(1) >= data["MA_long"].shift(1)) &
    (data["HighVolumeSpike"]),
    "Signal"
] = "SELL"

# --- Generate Buy/Sell Signals ---
data["Signal"] = ""

# MA crossover + volume spike
data.loc[
    (data["MA_short"] > data["MA_long"]) &
    (data["MA_short"].shift(1) <= data["MA_long"].shift(1)) &
    (data["HighVolumeSpike"]),
    "Signal"
] = "BUY"

data.loc[
    (data["MA_short"] < data["MA_long"]) &
    (data["MA_short"].shift(1) >= data["MA_long"].shift(1)) &
    (data["HighVolumeSpike"]),
    "Signal"
] = "SELL"

# Trendline breakouts
data.loc[data["Crossed_Resistance"], "Signal"] = "BREAKOUT ↑"
data.loc[data["Crossed_Support"], "Signal"] = "BREAKDOWN ↓"

# ✅ New: Price breaks above resistance = Breakout
data.loc[
    data["Crossed_Resistance"],
    "Signal"
] = "BREAKOUT ↑"

# ✅ New: Price breaks below support = Breakdown
data.loc[
    data["Crossed_Support"],
    "Signal"
] = "BREAKDOWN ↓"

# --- Signal Alerts Table ---
st.subheader("📋 Signal Alerts")

# Filter rows that have signals
signal_log = data[data["Signal"] != ""][["Signal", "Close"]].copy()
signal_log.rename(columns={"Close": "Price"}, inplace=True)
signal_log["Timestamp"] = signal_log.index

# Reorder columns
signal_log = signal_log[["Timestamp", "Signal", "Price"]]

# Show table in dashboard
st.dataframe(signal_log.tail(10), use_container_width=True)

# (No changes needed here unless you're moving imports higher in your script)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Combined Candlestick + Volume Chart ---
st.subheader("📊 Candlestick + Volume (Shared View)")

# Color volume bars based on price movement
data["VolumeColor"] = ["green" if c >= o else "red" for c, o in zip(data["Close"], data["Open"])]

# Create subplots with shared x-axis
fig_combined = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.05,
    subplot_titles=(f"{ticker} Candlestick", "Volume")
)

# Row 1: Candlestick chart
fig_combined.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Price"
), row=1, col=1)

# Buy/Sell markers
buy_signals = data[data["Signal"] == "BUY"]
sell_signals = data[data["Signal"] == "SELL"]

# BUY markers
fig_combined.add_trace(go.Scatter(
    x=buy_signals.index,
    y=buy_signals["Close"],
    mode="markers",
    marker=dict(symbol="triangle-up", color="lime", size=10),
    name="BUY"
), row=1, col=1)

# SELL markers
fig_combined.add_trace(go.Scatter(
    x=sell_signals.index,
    y=sell_signals["Close"],
    mode="markers",
    marker=dict(symbol="triangle-down", color="red", size=10),
    name="SELL"
), row=1, col=1)

# 🔔 Highlight Breakouts Above Resistance
fig_combined.add_trace(go.Scatter(
    x=data[data["Crossed_Resistance"]].index,
    y=data[data["Crossed_Resistance"]]["Close"],
    mode="markers",
    marker=dict(color="purple", size=10, symbol="star"),
    name="Breakout ↑"
), row=1, col=1)

# 🔔 Highlight Breakdowns Below Support
fig_combined.add_trace(go.Scatter(
    x=data[data["Crossed_Support"]].index,
    y=data[data["Crossed_Support"]]["Close"],
    mode="markers",
    marker=dict(color="black", size=10, symbol="star"),
    name="Breakdown ↓"
), row=1, col=1)

# ✅ Short-Term Moving Average (8h)
fig_combined.add_trace(go.Scatter(
    x=data.index,
    y=data["MA_short"],
    mode="lines",
    name="MA 8h",
    line=dict(color="orange", dash="dash")
), row=1, col=1)

# ✅ Long-Term Moving Average (24h)
fig_combined.add_trace(go.Scatter(
    x=data.index,
    y=data["MA_long"],
    mode="lines",
    name="MA 24h",
    line=dict(color="blue", dash="dot")
), row=1, col=1)

# Row 2: Volume bars
fig_combined.add_trace(go.Bar(
    x=data.index,
    y=data["Volume"],
    marker_color=data["VolumeColor"],
    name="Volume",
    opacity=0.6
), row=2, col=1)

# ✅ Volume MA (20-period)
fig_combined.add_trace(go.Scatter(
    x=data.index,
    y=data["Volume_MA_20"],
    mode="lines",
    line=dict(color="black", width=1, dash="dash"),
    name="Vol MA 20"
), row=2, col=1)

# 🔔 Highlight Volume Spikes (above 1.5x avg)
fig_combined.add_trace(go.Scatter(
    x=data[data["HighVolumeSpike"]].index,
    y=data[data["HighVolumeSpike"]]["Volume"],
    mode="markers",
    marker=dict(color="gold", size=8, symbol="circle"),
    name="High Volume Spike"
), row=2, col=1)

# Layout settings
fig_combined.update_layout(
    height=700,
    xaxis_rangeslider_visible=False,
    showlegend=True
)

# Display the combined chart
st.plotly_chart(fig_combined, use_container_width=True)

# --- Fibonacci Retracement ---
st.subheader("🔢 Fibonacci Retracement")

# Get the most recent swing high and low
recent_high = data["High"].rolling(10).max().iloc[-1]
recent_low = data["Low"].rolling(10).min().iloc[-1]

# Define Fibonacci levels
fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

fib_fig = go.Figure()

# Add candlestick chart
fib_fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"]
))

# Draw Fib lines
for level in fib_levels:
    fib_price = recent_high - (recent_high - recent_low) * level
    fib_fig.add_shape(
        type="line",
        x0=data.index[0],
        x1=data.index[-1],
        y0=fib_price,
        y1=fib_price,
        line=dict(color="blue", dash="dot")
    )
    fib_fig.add_annotation(
        x=data.index[-1],
        y=fib_price,
        text=f"{round(level * 100)}% ({fib_price:.2f})",
        showarrow=False,
        font=dict(size=10, color="blue")
    )

fib_fig.update_layout(
    title="Auto Fibonacci Retracement",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fib_fig, use_container_width=True)

# --- RSI Calculation ---
delta = data["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss
data["RSI"] = 100 - (100 / (1 + rs))

# --- 🔮 Live Signal Banner ---
latest = data.iloc[-1]

# Generate recommendation
if latest["Signal"] in ["BUY", "BREAKOUT ↑"]:
    if latest["RSI"] < 70 and latest["HighVolumeSpike"] and latest["WickType"] == "Long Lower":
        live_recommendation = "🟢 BUY (Momentum + Support Test)"
    elif latest["WickType"] == "Equal Wicks":
        live_recommendation = "⚪ HOLD (Indecision)"
    else:
        live_recommendation = "✅ BUY"

elif latest["Signal"] in ["SELL", "BREAKDOWN ↓"]:
    if latest["RSI"] > 30 and latest["HighVolumeSpike"] and latest["WickType"] == "Long Upper":
        live_recommendation = "🔻 SHORT (Rejection)"
    elif latest["WickType"] == "Equal Wicks":
        live_recommendation = "⚪ HOLD (Indecision)"
    else:
        live_recommendation = "🔴 SELL"
else:
    live_recommendation = "⚪ HOLD"

# Confidence scoring
confidence = 0
if latest["HighVolumeSpike"]: confidence += 1
if latest["WickType"] in ["Long Upper", "Long Lower"]: confidence += 1
if latest["RSI"] < 30 or latest["RSI"] > 70: confidence += 1
if latest["Signal"] in ["BREAKOUT ↑", "BREAKDOWN ↓"]: confidence += 1

confidence_label = {
    4: "🔥 High",
    3: "✅ Moderate",
    2: "⚠️ Low",
    1: "⚪ Weak",
    0: "⚪ No Confidence"
}.get(confidence, "⚪ Uncertain")

# Display top banner
st.markdown(f"""
<div style='
    background-color: #1f77b4;  /* Blue background */
    color: white;
    padding: 1.2em;
    font-size: 1.4em;
    font-weight: bold;
    text-align: center;
    border-radius: 8px;
    margin-top: 20px;
    margin-bottom: 20px;
'>
    {live_recommendation} — <span style='font-weight: normal;'>Confidence: {confidence_label}</span>
</div>
""", unsafe_allow_html=True)

# --- Trade Insight Message (based on recommendation logic) ---
trade_message = ""

if "BUY" in live_recommendation:
    if latest["WickType"] == "Long Lower" and latest["HighVolumeSpike"]:
        trade_message = "📈 Hammer candle with strong buy-side volume. RSI shows bullish recovery."
    elif latest["WickType"] == "Equal Wicks":
        trade_message = "⚠️ Indecision candle detected. May be waiting on confirmation."
    else:
        trade_message = "✅ Upward momentum with crossover and moderate strength."

elif "SHORT" in live_recommendation or "SELL" in live_recommendation:
    if latest["WickType"] == "Long Upper" and latest["HighVolumeSpike"]:
        trade_message = "📉 Rejection wick with selling pressure. Possible reversal forming."
    elif latest["WickType"] == "Equal Wicks":
        trade_message = "⚠️ Market stalling. Watching for breakdown confirmation."
    else:
        trade_message = "🔻 Bearish crossover and declining RSI suggests short opportunity."

elif "HOLD" in live_recommendation:
    trade_message = "⏸️ No dominant signal. Waiting for clearer setup."

# Display message
st.markdown(f"""
<div style='
    background-color: #f8f9fa;
    color: #333;
    padding: 1em;
    border-left: 5px solid #888;
    border-radius: 6px;
    margin-bottom: 20px;
    font-size: 1.05em;
'>
    {trade_message}
</div>
""", unsafe_allow_html=True)

# --- 🔴 Live Prediction (Current Bar Only) ---
st.subheader("🔮 Real-Time Recommendation")

latest = data.iloc[-1]  # Most recent bar

# Basic recommendation logic
if latest["Signal"] in ["BUY", "BREAKOUT ↑"]:
    if latest["RSI"] < 70 and latest["HighVolumeSpike"] and latest["WickType"] == "Long Lower":
        live_recommendation = "🟢 BUY (Momentum + Support Test)"
    elif latest["WickType"] == "Equal Wicks":
        live_recommendation = "⚪ HOLD (Indecision)"
    else:
        live_recommendation = "✅ BUY"

elif latest["Signal"] in ["SELL", "BREAKDOWN ↓"]:
    if latest["RSI"] > 30 and latest["HighVolumeSpike"] and latest["WickType"] == "Long Upper":
        live_recommendation = "🔻 SHORT (Rejection)"
    elif latest["WickType"] == "Equal Wicks":
        live_recommendation = "⚪ HOLD (Indecision)"
    else:
        live_recommendation = "🔴 SELL"
else:
    live_recommendation = "⚪ HOLD"

# Confidence scoring
confidence = 0
if latest["HighVolumeSpike"]:
    confidence += 1
if latest["WickType"] in ["Long Upper", "Long Lower"]:
    confidence += 1
if latest["RSI"] < 30 or latest["RSI"] > 70:
    confidence += 1
if latest["Signal"] in ["BREAKOUT ↑", "BREAKDOWN ↓"]:
    confidence += 1

confidence_label = {
    4: "🔥 High",
    3: "✅ Moderate",
    2: "⚠️ Low",
    1: "⚪ Weak",
    0: "⚪ No Confidence"
}.get(confidence, "⚪ Uncertain")

# Display current bar prediction in a compact table
live_table = pd.DataFrame({
    "Time": [latest.name],
    "Price": [latest["Close"]],
    "RSI": [round(latest["RSI"], 2)],
    "Signal": [latest["Signal"]],
    "Wick": [latest["WickType"]],
    "Volume Spike": ["Yes" if latest["HighVolumeSpike"] else "No"],
    "Recommendation": [live_recommendation],
    "Confidence": [confidence_label]
})

st.dataframe(live_table, use_container_width=True)

# --- Stochastic Oscillator ---
low14 = data["Low"].rolling(window=14).min()
high14 = data["High"].rolling(window=14).max()

data["%K"] = 100 * ((data["Close"] - low14) / (high14 - low14))
data["%D"] = data["%K"].rolling(window=3).mean()

# --- Plot RSI ---
st.subheader("📉 Relative Strength Index (RSI)")
fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.plot(data.index, data["RSI"], label="RSI", color="purple")
ax2.axhline(70, linestyle="--", color="red", label="Overbought")
ax2.axhline(30, linestyle="--", color="green", label="Oversold")
ax2.set_ylim(0, 100)
ax2.set_ylabel("RSI")
ax2.legend()
st.pyplot(fig2)

# --- Plot Stochastic Oscillator ---
st.subheader("🎯 Stochastic Oscillator")
fig3, ax3 = plt.subplots(figsize=(12, 3))
ax3.plot(data.index, data["%K"], label="%K", color="orange")
ax3.plot(data.index, data["%D"], label="%D", color="blue")
ax3.axhline(80, linestyle="--", color="red", label="Overbought")
ax3.axhline(20, linestyle="--", color="green", label="Oversold")
ax3.set_ylim(0, 100)
ax3.set_ylabel("Stochastic %")
ax3.legend()
st.pyplot(fig3)

# --- Predictive Recommendation Engine ---
st.subheader("🔮 Predictive Recommendation")

# Default state
recommendation = "HOLD"

# Use latest data point only
latest = data.iloc[-1]

# Conditions
if latest["Signal"] == "BUY" or latest["Signal"] == "BREAKOUT ↑":
    if latest["RSI"] < 70 and latest["HighVolumeSpike"]:
        recommendation = "RECOMMEND: BUY"

elif latest["Signal"] == "SELL" or latest["Signal"] == "BREAKDOWN ↓":
    if latest["RSI"] > 30 and latest["HighVolumeSpike"]:
        recommendation = "RECOMMEND: SHORT"

elif latest["RSI"] >= 70:
    recommendation = "RECOMMEND: SELL (Overbought)"

elif latest["RSI"] <= 30:
    recommendation = "RECOMMEND: BUY (Oversold)"

# --- Display recommendation in table ---
rec_table = pd.DataFrame({
    "Time": [latest.name],
    "Close Price": [latest["Close"]],
    "RSI": [round(latest["RSI"], 2)],
    "Signal": [latest["Signal"]],
    "Volume Spike": ["Yes" if latest["HighVolumeSpike"] else "No"],
    "Recommendation": [recommendation]
})

st.dataframe(rec_table, use_container_width=True)

# --- 🔮 Predictive AI Table (Multi-Row) ---
st.subheader("📋 Predictive Recommendations")

# Work on a fresh table using signal triggers
predictions = data[data["Signal"] != ""].copy()

# Add timestamp
predictions["Timestamp"] = predictions.index

# Add base recommendation logic
def get_recommendation(row):
    if row["Signal"] in ["BUY", "BREAKOUT ↑"]:
        if row["RSI"] < 70 and row["HighVolumeSpike"] and row["WickType"] == "Long Lower":
            return "🟢 BUY (Momentum + Support Test)"
        elif row["WickType"] == "Equal Wicks":
            return "⚪ HOLD (Indecision)"
        else:
            return "✅ BUY"
        
    elif row["Signal"] in ["SELL", "BREAKDOWN ↓"]:
        if row["RSI"] > 30 and row["HighVolumeSpike"] and row["WickType"] == "Long Upper":
            return "🔻 SHORT (Rejection)"
        elif row["WickType"] == "Equal Wicks":
            return "⚪ HOLD (Indecision)"
        else:
            return "🔴 SELL"
    
    return "⚪ HOLD"

predictions["Recommendation"] = predictions.apply(get_recommendation, axis=1)

# Add confidence score (basic logic)
def get_confidence(row):
    score = 0
    if row["HighVolumeSpike"]:
        score += 1
    if row["WickType"] in ["Long Lower", "Long Upper"]:
        score += 1
    if row["RSI"] < 30 or row["RSI"] > 70:
        score += 1
    if row["Signal"] in ["BREAKOUT ↑", "BREAKDOWN ↓"]:
        score += 1
    
    if score >= 3:
        return "🔥 High"
    elif score == 2:
        return "✅ Moderate"
    elif score == 1:
        return "⚠️ Low"
    else:
        return "⚪ Uncertain"

predictions["Confidence"] = predictions.apply(get_confidence, axis=1)

# Select and format columns
display_cols = ["Timestamp", "Signal", "Close", "RSI", "WickType", "Volume", "Recommendation", "Confidence"]
st.write("✅ Rows with signals detected:", len(predictions))
st.write(predictions[["Signal", "RSI", "WickType", "HighVolumeSpike"]].tail(5))
st.dataframe(predictions[display_cols].tail(12), use_container_width=True)

# --- Combined Signal View Table ---
st.subheader("🧠 Combined Signal Alignment")

def get_status_emoji(row):
    action = row["Action"]
    pred = row["Predictive"]
    
    if action == "✅ Entry":
        return "✅"
    elif action == "🟡 Monitor":
        return "👀"
    elif "SHORT" in pred:
        return "🔻"
    elif "SELL" in pred:
        return "🔴"
    elif "BUY" in pred:
        return "🟢"
    elif action == "⚪ No Action":
        return "🚫"
    else:
        return "❗️"

# Use last N rows where signals were recently triggered or recent candles
recent_rows = data.tail(15).copy()

# Build recommendation & confidence
def generate_live_and_predictive(row):
    signal = row.get("Signal", "")
    if signal in ["BUY", "BREAKOUT ↑"]:
        pred = "🟢 BUY"
    elif signal in ["SELL", "BREAKDOWN ↓"]:
        pred = "🔻 SHORT"
    else:
        pred = "⚪ NONE"

    if signal in ["BUY", "BREAKOUT ↑"]:
        if row["RSI"] < 70 and row["HighVolumeSpike"] and row["WickType"] == "Long Lower":
            live = "🟢 BUY (Momentum + Support Test)"
        elif row["WickType"] == "Equal Wicks":
            live = "⚪ HOLD (Indecision)"
        else:
            live = "✅ BUY"
    elif signal in ["SELL", "BREAKDOWN ↓"]:
        if row["RSI"] > 30 and row["HighVolumeSpike"] and row["WickType"] == "Long Upper":
            live = "🔻 SHORT (Rejection)"
        elif row["WickType"] == "Equal Wicks":
            live = "⚪ HOLD (Indecision)"
        else:
            live = "🔴 SELL"
    else:
        live = "⚪ HOLD"

    score = 0
    if row["HighVolumeSpike"]: score += 1
    if row["WickType"] in ["Long Upper", "Long Lower"]: score += 1
    if row["RSI"] < 30 or row["RSI"] > 70: score += 1
    if signal in ["BREAKOUT ↑", "BREAKDOWN ↓"]: score += 1

    confidence = {
        4: "🔥 High",
        3: "✅ Moderate",
        2: "⚠️ Low",
        1: "⚪ Weak",
        0: "⚪ None"
    }.get(score, "⚪ None")

    if "BUY" in pred and "BUY" in live:
        decision = "✅ Entry"
    elif "SHORT" in pred and "SHORT" in live:
        decision = "✅ Entry"
    elif pred != "⚪ NONE":
        decision = "🟡 Monitor"
    else:
        decision = "⚪ No Action"

    return pd.Series([pred, live, confidence, decision])

recent_rows[["Predictive", "Live", "Confidence", "Action"]] = recent_rows.apply(generate_live_and_predictive, axis=1)
recent_rows["Status"] = recent_rows.apply(get_status_emoji, axis=1)

 
# --- Style rows based on Action ---

def highlight_action(row):
    if row["Action"] == "✅ Entry":
        return ['background-color: #d4edda; color: black'] * len(row)  # light green
    elif row["Action"] == "🟡 Monitor":
        return ['background-color: #fff3cd; color: black'] * len(row)  # light yellow
    elif row["Action"] == "⚪ No Action":
        return ['background-color: #e2e3e5; color: black'] * len(row)  # light gray
    else:
        return ['background-color: #f8d7da; color: black'] * len(row)  # light red

# Create styled DataFrame
styled_table = recent_rows[["Status", "Predictive", "Live", "Confidence", "Action"]].tail(10).style.apply(highlight_action, axis=1)

# Display styled table
st.dataframe(styled_table, use_container_width=True)

# --- Auto Refresh Section ---
import time

refresh = st.selectbox("⏱️ Auto-refresh interval (seconds)", [None, 10, 30, 60], index=2)

if refresh:
    st.markdown(f"🔄 Auto-refresh set to every **{refresh} seconds**.")
    time.sleep(refresh)
    st.rerun()

