"""
Shared logic for the Market Terminal app and the nightly alert job:
instrument list, data download, indicators and signal scoring.
"""

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = {
    "JSE ETFs": [("STX40.JO", "Satrix 40"), ("STXSWX.JO", "Satrix SWIX Top 40"),
                 ("STXDIV.JO", "Satrix Dividend Plus"), ("STXPRO.JO", "Satrix Property"),
                 ("STX500.JO", "Satrix S&P 500"), ("STXNDQ.JO", "Satrix Nasdaq 100"),
                 ("STXWDM.JO", "Satrix MSCI World"), ("STXEMG.JO", "Satrix MSCI Emerging Mkts"),
                 ("SYG500.JO", "Sygnia Itrix S&P 500"), ("GLD.JO", "NewGold Gold ETF")],
    "US ETFs": [("SPY", "SPDR S&P 500"), ("QQQ", "Invesco Nasdaq 100"), ("VTI", "Vanguard Total US Market"),
                ("EEM", "iShares Emerging Markets"), ("GLD", "SPDR Gold Shares"),
                ("TLT", "iShares 20+ Yr Treasury"), ("XLE", "Energy Select Sector"),
                ("XLF", "Financial Select Sector")],
    "Global Indices": [("^GSPC", "S&P 500"), ("^IXIC", "Nasdaq Composite"), ("^DJI", "Dow Jones"),
                       ("^FTSE", "FTSE 100"), ("^N225", "Nikkei 225")],
    "Commodities": [("BZ=F", "Brent Crude"), ("CL=F", "WTI Crude"), ("BNO", "US Brent Oil ETF"),
                    ("USO", "US Oil ETF"), ("GC=F", "Gold"), ("PL=F", "Platinum"), ("SI=F", "Silver"),
                    ("NG=F", "Natural Gas")],
    "Currencies": [("USDZAR=X", "USD/ZAR"), ("EURZAR=X", "EUR/ZAR"), ("GBPZAR=X", "GBP/ZAR")],
    "Crypto": [("BTC-USD", "Bitcoin"), ("ETH-USD", "Ethereum")],
}
NAMES = {t: n for g in UNIVERSE.values() for t, n in g}
LABELS = {f"{n} ({t})": t for t, n in NAMES.items()}


def label(t):
    return f"{NAMES.get(t, t)} ({t})"


def is_jse(t):
    return t.upper().endswith(".JO")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def download_closes(tickers, period="1y"):
    """Daily closes for many tickers (columns = tickers). JSE converted to rand."""
    tickers = list(tickers)
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False, group_by="column")
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]].rename(columns={"Close": tickers[0]})
    closes = closes.copy()
    for t in closes.columns:
        if is_jse(t):
            closes[t] = closes[t] / 100
    return closes


def fetch_ohlc(ticker, period="2y", start=None):
    df = yf.download(ticker, period=None if start else period, start=start,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}. Yahoo may be busy; wait a minute and retry.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"]).copy()
    if is_jse(ticker):
        df[["Open", "High", "Low", "Close"]] /= 100
    return df


# --------------------------------------------------------------------------
# Indicators and signals
# --------------------------------------------------------------------------

def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def macd(c, fast=12, slow=26, sig=9):
    line = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    return line, line.ewm(span=sig, adjust=False).mean()


def score_series(c):
    """Daily score from -4 to +4: price vs 200d, 50d vs 200d, MACD vs signal, 3-month return."""
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    m, sig = macd(c)
    parts = [np.sign(c - sma200), np.sign(sma50 - sma200), np.sign(m - sig),
             np.sign(c / c.shift(63) - 1)]
    return sum(parts).where(sma200.notna())


def bias(score):
    if pd.isna(score):
        return "—"
    return ("STRONG BULLISH" if score >= 3 else "BULLISH" if score >= 1 else
            "STRONG BEARISH" if score <= -3 else "BEARISH" if score <= -1 else "NEUTRAL")


def last_cross(a, b, lookback):
    """(direction, days ago) of the latest cross of a over b within lookback, else None."""
    state = np.sign(a - b).dropna()
    state = state[state != 0]
    flips = state[state.diff().fillna(0) != 0].tail(1)
    if flips.empty:
        return None
    days = len(state) - state.index.get_loc(flips.index[0]) - 1
    return (int(flips.iloc[0]), days) if days <= lookback else None


def analyse(c):
    """Current readings and recent events for one price series."""
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    m, sig = macd(c)
    r = rsi(c)
    sc = score_series(c)
    events = []
    gc = last_cross(sma50, sma200, 20)
    if gc:
        events.append((gc[1], "Golden cross (50 > 200)" if gc[0] > 0 else "Death cross (50 < 200)"))
    mc = last_cross(m, sig, 5)
    if mc:
        events.append((mc[1], "MACD bullish cross" if mc[0] > 0 else "MACD bearish cross"))
    pc = last_cross(c, sma200, 10)
    if pc:
        events.append((pc[1], "Broke above 200-day" if pc[0] > 0 else "Fell below 200-day"))
    if r.iloc[-1] >= 70:
        events.append((0, f"Overbought (RSI {r.iloc[-1]:.0f})"))
    if r.iloc[-1] <= 30:
        events.append((0, f"Oversold (RSI {r.iloc[-1]:.0f})"))
    yr = c.tail(252)
    if c.iloc[-1] >= yr.max() * 0.995:
        events.append((0, "At 52-week high"))
    if c.iloc[-1] <= yr.min() * 1.005:
        events.append((0, "At 52-week low"))
    fwd = c.shift(-20) / c - 1
    bull = fwd[sc >= 2].dropna()
    edge = (bull.mean() - fwd.dropna().mean()) * 100 if len(bull) > 30 else np.nan
    return {"Score": sc.iloc[-1], "Bias": bias(sc.iloc[-1]),
            "Trend": "UP" if c.iloc[-1] > sma200.iloc[-1] else "DOWN",
            "50/200": "GOLDEN" if sma50.iloc[-1] > sma200.iloc[-1] else "DEATH",
            "MACD": "BULL" if m.iloc[-1] > sig.iloc[-1] else "BEAR",
            "RSI": r.iloc[-1],
            "3M %": (c.iloc[-1] / c.iloc[-64] - 1) * 100 if len(c) > 64 else np.nan,
            "Bull edge 20d %": edge, "events": sorted(events)}


def fresh_alerts(c):
    """Only events that happened on the latest close (for push notifications)."""
    out = []
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    m, sig = macd(c)
    r = rsi(c)
    for (a, b, up, down) in [(sma50, sma200, "Golden cross (50 > 200) 🟢", "Death cross (50 < 200) 🔴"),
                             (c, sma200, "Broke above 200-day 🟢", "Fell below 200-day 🔴"),
                             (m, sig, "MACD bullish cross 🟢", "MACD bearish cross 🔴")]:
        x = last_cross(a, b, 0)
        if x:
            out.append(up if x[0] > 0 else down)
    if len(r) > 2 and r.iloc[-2] < 70 <= r.iloc[-1]:
        out.append(f"Turned overbought (RSI {r.iloc[-1]:.0f}) ⚠️")
    if len(r) > 2 and r.iloc[-2] > 30 >= r.iloc[-1]:
        out.append(f"Turned oversold (RSI {r.iloc[-1]:.0f}) ⚠️")
    prev_high = c.iloc[-253:-1].max() if len(c) > 253 else c.iloc[:-1].max()
    prev_low = c.iloc[-253:-1].min() if len(c) > 253 else c.iloc[:-1].min()
    if c.iloc[-1] > prev_high:
        out.append("New 52-week high 🚀")
    if c.iloc[-1] < prev_low:
        out.append("New 52-week low 📉")
    return out


def scan(groups=None, period="5y"):
    """Run the signal scanner. Returns (table, alert feed, skipped tickers)."""
    groups = groups or UNIVERSE
    tickers = [t for g in groups.values() for t, _ in g]
    closes = download_closes(tuple(tickers), period)
    rows, feed, skipped = [], [], []
    for g, items in groups.items():
        for t, name in items:
            c = closes[t].dropna() if t in closes else pd.Series(dtype=float)
            if len(c) < 260:
                skipped.append(t)
                continue
            a = analyse(c)
            events = a.pop("events")
            for days, text in events:
                feed.append({"When": "today" if days == 0 else f"{days}d ago", "days": days,
                             "Name": name, "Ticker": t, "Signal": text, "Bias": a["Bias"]})
            rows.append({"Name": name, "Ticker": t, "Last": c.iloc[-1], **a, "Alerts": len(events)})
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("Score", ascending=False)
    feed = pd.DataFrame(feed)
    if not feed.empty:
        feed = feed.sort_values("days").drop(columns="days")
    return table, feed, skipped, closes
