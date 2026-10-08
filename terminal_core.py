"""
Shared logic for the Market Terminal app and the nightly alert job:
instrument list, data download, indicators and signal scoring.
"""

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = {
    "JSE Shares": [("NPN.JO", "Naspers"), ("PRX.JO", "Prosus"), ("CFR.JO", "Richemont"),
                   ("BTI.JO", "British American Tobacco"), ("ANH.JO", "AB InBev"),
                   ("BHG.JO", "BHP Group"), ("AGL.JO", "Anglo American"), ("GLN.JO", "Glencore"),
                   ("S32.JO", "South32"), ("KIO.JO", "Kumba Iron Ore"), ("EXX.JO", "Exxaro"),
                   ("SOL.JO", "Sasol"), ("GFI.JO", "Gold Fields"), ("ANG.JO", "AngloGold Ashanti"),
                   ("HAR.JO", "Harmony Gold"), ("SSW.JO", "Sibanye-Stillwater"), ("IMP.JO", "Impala Platinum"),
                   ("NPH.JO", "Northam Platinum"), ("FSR.JO", "FirstRand"), ("SBK.JO", "Standard Bank"),
                   ("CPI.JO", "Capitec"), ("ABG.JO", "Absa"), ("NED.JO", "Nedbank"), ("INL.JO", "Investec"),
                   ("SLM.JO", "Sanlam"), ("DSY.JO", "Discovery"), ("OMU.JO", "Old Mutual"),
                   ("OUT.JO", "OUTsurance"), ("MTN.JO", "MTN Group"), ("VOD.JO", "Vodacom"),
                   ("SHP.JO", "Shoprite"), ("WHL.JO", "Woolworths"), ("CLS.JO", "Clicks"),
                   ("MRP.JO", "Mr Price"), ("PPH.JO", "Pepkor"), ("TFG.JO", "TFG (Foschini)"),
                   ("BID.JO", "Bid Corporation"), ("BVT.JO", "Bidvest"), ("APN.JO", "Aspen"),
                   ("REM.JO", "Remgro"), ("MNP.JO", "Mondi"), ("GRT.JO", "Growthpoint"),
                   ("NRP.JO", "NEPI Rockcastle")],
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
# Markets that trade while the JSE is closed, used for the morning brief
OVERNIGHT = [("ES=F", "S&P 500 futures"), ("NQ=F", "Nasdaq 100 futures"), ("^GSPC", "S&P 500 (last close)"),
             ("^N225", "Nikkei 225"), ("^HSI", "Hang Seng"), ("^AXJO", "ASX 200"),
             ("BZ=F", "Brent crude"), ("GC=F", "Gold"), ("PL=F", "Platinum"),
             ("USDZAR=X", "USD/ZAR"), ("BTC-USD", "Bitcoin")]
TONE_KEYS = ["ES=F", "NQ=F", "^HSI", "^AXJO"]  # plus a stronger rand

# JSE share -> listing that trades after the JSE closes: (ticker, description, currency)
PAIRS = {"NPN.JO": ("0700.HK", "Tencent, Hong Kong", "HKD"), "PRX.JO": ("0700.HK", "Tencent, Hong Kong", "HKD"),
         "BHG.JO": ("BHP.AX", "BHP, Sydney", "AUD"), "S32.JO": ("S32.AX", "South32, Sydney", "AUD"),
         "GFI.JO": ("GFI", "Gold Fields, NYSE", "USD"), "ANG.JO": ("AU", "AngloGold, NYSE", "USD"),
         "HAR.JO": ("HMY", "Harmony, NYSE", "USD"), "SSW.JO": ("SBSW", "Sibanye, NYSE", "USD"),
         "SOL.JO": ("SSL", "Sasol, NYSE", "USD"), "BTI.JO": ("BTI", "BAT, NYSE", "USD"),
         "ANH.JO": ("BUD", "AB InBev, NYSE", "USD")}
FX_TO_ZAR = {"USD": "USDZAR=X", "HKD": "USDZAR=X", "AUD": "AUDZAR=X"}  # HKD is pegged to USD

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


# --------------------------------------------------------------------------
# Morning brief
# --------------------------------------------------------------------------

def download_ohlc_many(tickers, period="2y"):
    """Dict of DataFrames (Open/High/Low/Close, columns = tickers). JSE in rand."""
    tickers = list(tickers)
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False, group_by="column")
    out = {}
    for f in ("Open", "High", "Low", "Close"):
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw[f].copy()
        else:
            df = raw[[f]].rename(columns={f: tickers[0]})
        for t in df.columns:
            if is_jse(t):
                df[t] = df[t] / 100
        out[f] = df
    return out


def atr(h, l, c, n=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def daily_change(s):
    s = s.dropna()
    if len(s) < 2:
        return np.nan, np.nan
    return s.iloc[-1], (s.iloc[-1] / s.iloc[-2] - 1) * 100


def overnight_moves():
    tickers = [t for t, _ in OVERNIGHT] + ["AUDZAR=X"] + sorted({p[0] for p in PAIRS.values()})
    closes = download_closes(tuple(dict.fromkeys(tickers)), "1mo")
    rows = []
    for t, name in OVERNIGHT:
        last, chg = daily_change(closes[t]) if t in closes else (np.nan, np.nan)
        rows.append({"Market": name, "Ticker": t, "Last": last, "Change %": chg})
    table = pd.DataFrame(rows)

    chg = dict(zip(table["Ticker"], table["Change %"]))
    votes = [chg.get(k) for k in TONE_KEYS if not pd.isna(chg.get(k, np.nan))]
    ups = sum(v > 0 for v in votes)
    zar = chg.get("USDZAR=X", np.nan)
    if not pd.isna(zar):
        votes.append(-zar)
        ups += zar < 0  # USD/ZAR falling = rand stronger
    n = len(votes)
    tone = ("RISK-ON" if n and ups / n >= 0.7 else "RISK-OFF" if n and ups / n <= 0.3 else "MIXED")

    cues = []
    for jse, (ctr, desc, cur) in PAIRS.items():
        if ctr not in closes:
            continue
        _, c_chg = daily_change(closes[ctr])
        fx = FX_TO_ZAR[cur]
        _, fx_chg = daily_change(closes[fx]) if fx in closes else (np.nan, 0.0)
        if pd.isna(c_chg):
            continue
        fx_chg = 0.0 if pd.isna(fx_chg) else fx_chg
        implied = ((1 + c_chg / 100) * (1 + fx_chg / 100) - 1) * 100
        cues.append({"JSE share": NAMES[jse], "Ticker": jse, "Follows": desc,
                     "Their move %": c_chg, "Currency effect %": fx_chg, "Open cue %": implied})
    cues = pd.DataFrame(cues)
    if not cues.empty:
        cues = cues.sort_values("Open cue %", key=lambda x: -x.abs())
    return table, tone, ups, n, cues


def setups(groups=("JSE Shares", "JSE ETFs")):
    """Mechanical pre-open setups with reference levels for JSE instruments."""
    tickers = tuple(t for g in groups for t, _ in UNIVERSE[g])
    d = download_ohlc_many(tickers, "2y")
    rows = []
    for t in tickers:
        if t not in d["Close"]:
            continue
        c = d["Close"][t].dropna()
        if len(c) < 260:
            continue
        h, l = d["High"][t].reindex(c.index), d["Low"][t].reindex(c.index)
        a = analyse(c)
        r = a["RSI"]
        rng = atr(h, l, c).iloc[-1]
        hi20, lo20 = h.tail(20).max(), l.tail(20).min()
        sma50, sma200 = c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
        last, prev_hi, prev_lo = c.iloc[-1], h.iloc[-1], l.iloc[-1]
        setup = None
        if a["Score"] >= 3 and last >= hi20 - 0.5 * rng:
            setup, level, stop, side = "Breakout watch", hi20, hi20 - 1.5 * rng, 1
            note = "Strong uptrend near its 20-day high. Watch for a move above the level."
        elif a["Score"] >= 2 and r < 45 and last > sma200:
            setup, level, stop, side = "Pullback in uptrend", prev_hi, prev_hi - 1.5 * rng, 1
            note = f"Dip inside an uptrend (50-day avg {sma50:,.2f}). A move above yesterday's high shows buyers returning."
        elif a["Score"] <= -3 and last <= lo20 + 0.5 * rng:
            setup, level, stop, side = "Breakdown risk", lo20, lo20 + 1.5 * rng, -1
            note = "Strong downtrend near its 20-day low. Caution on buying; a break lower may continue."
        elif r <= 30 and a["Score"] <= -2:
            setup, level, stop, side = "Oversold (counter-trend)", prev_hi, prev_lo - 0.5 * rng, 1
            note = "Very oversold but still in a downtrend. Bounces happen, but this is the riskiest setup."
        if not setup:
            continue
        risk = abs(level - stop)
        rows.append({"Name": NAMES[t], "Ticker": t, "Setup": setup, "Bias": a["Bias"], "Score": a["Score"],
                     "Last": last, "Watch level": level, "Invalidation": stop,
                     "2R reference": level + side * 2 * risk, "ATR": rng, "RSI": r,
                     "Bull edge 20d %": a["Bull edge 20d %"], "Why": note,
                     "Alerts": "; ".join(txt for _, txt in a["events"])})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Score", key=lambda x: -x.abs())
    return df
