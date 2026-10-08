"""
Market Terminal: market monitor, signal scanner, charts and strategy backtests.
Educational tool only, not financial advice.

Run locally:  streamlit run app.py
"""

import contextlib
import io
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import strategy_backtester as sb
import terminal_core as core

sb.plot = lambda *a, **k: None  # the app draws interactive charts instead of PNG files

st.set_page_config(page_title="Market Terminal", page_icon="🟧", layout="wide")

BG, PANEL, ORANGE, GREEN, RED, GREY = "#000000", "#0d0d0d", "#ff9900", "#00c853", "#ff3d3d", "#9e9e9e"
FONT = "Consolas, Courier New, monospace"
SAST = ZoneInfo("Africa/Johannesburg")
BIAS_COL = {"STRONG BULLISH": GREEN, "BULLISH": "#69f0ae", "NEUTRAL": GREY,
            "BEARISH": "#ff8a80", "STRONG BEARISH": RED, "—": GREY}

st.markdown(f"""
<style>
  .block-container {{padding-top: 2.2rem; max-width: 1500px;}}
  h1, h2, h3 {{color: {ORANGE} !important; font-family: {FONT};}}
  .banner {{background:{BG}; color:{ORANGE}; font-family:{FONT}; padding:8px 12px;
            border-left:4px solid {ORANGE}; font-size:16px; font-weight:bold; margin:6px 0 10px 0;}}
  .tbl {{overflow-x:auto; margin-bottom: 12px;}}
  .tbl table {{border-collapse:collapse; width:100%;}}
  .stat {{display:inline-block; padding:6px 18px 6px 0; font-family:{FONT};}}
  .stat .k {{color:{GREY}; font-size:11px;}} .stat .v {{color:#fff; font-size:18px;}}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def banner(text):
    st.markdown(f"<div class='banner'>{text}</div>", unsafe_allow_html=True)


def style_fig(fig, title, height=520):
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL, height=height,
                      font=dict(family=FONT, color="#e0e0e0", size=12),
                      title=dict(text=title, font=dict(color=ORANGE, size=16)),
                      legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
                      margin=dict(l=50, r=20, t=70, b=40), hovermode="x unified")
    fig.update_xaxes(gridcolor="#1f1f1f")
    fig.update_yaxes(gridcolor="#1f1f1f")
    return fig


def show_fig(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


TABLE_HEAD = [{"selector": "th", "props": [("background-color", "#1a1a1a"), ("color", ORANGE),
               ("font-family", FONT), ("font-size", "12px"), ("padding", "6px 10px"),
               ("border-bottom", f"2px solid {ORANGE}"), ("text-align", "left")]},
              {"selector": "", "props": [("border-collapse", "collapse"), ("background", BG)]}]
CELL = {"background-color": BG, "color": "#e0e0e0", "font-family": FONT, "font-size": "12px",
        "border-bottom": "1px solid #222", "padding": "4px 10px"}


def num_colour(v):
    if pd.isna(v):
        return f"color:{GREY}"
    return f"color:{GREEN if v > 0 else RED if v < 0 else GREY}"


def word_colour(v):
    return f"color:{GREEN if v in ('UP', 'GOLDEN', 'BULL') else RED if v in ('DOWN', 'DEATH', 'BEAR') else '#e0e0e0'}"


def show_table(styler):
    st.markdown(f"<div class='tbl'>{styler.hide(axis='index').to_html()}</div>", unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner="Fetching prices…")
def cached_closes(tickers, period):
    return core.download_closes(tickers, period)


@st.cache_data(ttl=900, show_spinner="Fetching prices…")
def cached_ohlc(ticker, period=None, start=None):
    return core.fetch_ohlc(ticker, period or "2y", start)


@st.cache_data(ttl=900, show_spinner="Scanning markets…")
def cached_scan(group_name):
    groups = core.UNIVERSE if group_name == "All" else {group_name: core.UNIVERSE[group_name]}
    table, feed, skipped, _ = core.scan(groups)
    return table, feed, skipped


@st.cache_data(ttl=300, show_spinner="Checking overnight markets…")
def cached_overnight():
    return core.overnight_moves()


@st.cache_data(ttl=900, show_spinner="Scanning JSE setups…")
def cached_setups():
    return core.setups()


@st.cache_data(ttl=900, show_spinner="Loading history…")
def cached_prices(ticker, start, demo, demo_key):
    params = {"etf": {"mu": 0.09, "sigma": 0.16, "seed": 1}, "cc": {"mu": 0.10, "sigma": 0.18, "seed": 2},
              "oil": {"mu": 0.02, "sigma": 0.38, "seed": 3}}[demo_key]
    return sb.load_prices(ticker, start, date.today().isoformat(), demo, params)


def captured(fn, *args):
    """Run a backtest and capture its printed report."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args)
    return result, buf.getvalue()


def equity_chart(df, cols, title):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.05, subplot_titles=("Portfolio value", "Drawdown %"))
    for (c, name), col in zip(cols.items(), [ORANGE, "#29b6f6", GREEN, "#ab47bc"]):
        s = df[c].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s, name=name, line=dict(color=col, width=1.6)), 1, 1)
        dd = (s / s.cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd, showlegend=False, line=dict(color=col, width=1),
                                 fill="tozeroy"), 2, 1)
    return style_fig(fig, title, 600)


def data_error(e):
    st.error(f"{e}")
    st.caption("Yahoo Finance sometimes refuses requests from cloud servers. Wait a minute, then press "
               "**R** (or use the ⋮ menu → Rerun). Tick *Use demo data* on backtests to test offline.")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.markdown(f"<h2 style='margin-bottom:0'>🟧 MARKET TERMINAL</h2>"
                    f"<div style='color:{GREY};font-family:{FONT};font-size:12px'>"
                    f"{datetime.now(SAST):%a %d %b %Y %H:%M} SAST</div>", unsafe_allow_html=True)
page = st.sidebar.radio("Screen", ["🌅 Morning Brief", "📊 Market Monitor", "🚦 Signal Scanner", "📈 Security Chart",
                                   "🔀 Compare", "🧪 Backtests"], label_visibility="collapsed")
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
st.sidebar.caption("Data: Yahoo Finance, may be delayed ~15 min. JSE prices shown in rand. "
                   "Educational tool only, not financial advice.")
GROUPS = ["All"] + list(core.UNIVERSE)



# --------------------------------------------------------------------------
# Morning Brief
# --------------------------------------------------------------------------

if page == "🌅 Morning Brief":
    now = datetime.now(SAST)
    open_at = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if now.weekday() >= 5:
        status = "Weekend: the JSE is closed. Brief shows the last session."
    elif now < open_at:
        mins = int((open_at - now).total_seconds() // 60)
        status = f"JSE opens in {mins // 60}h {mins % 60:02d}m (opening auction 08:30–09:00)"
    elif now.hour < 17:
        status = "JSE is OPEN (continuous trading until 16:50, closing auction 16:50–17:00)"
    else:
        status = "JSE closed for the day. This becomes tomorrow's pre-open brief."
    banner(f"🌅 MORNING BRIEF &nbsp;|&nbsp; {now:%a %d %b %Y %H:%M} SAST &nbsp;|&nbsp; {status}")
    st.caption("Mechanical readings from overnight markets and indicators, to help you plan. "
               "Not buy or sell recommendations. Decide your own trades and size them for your risk.")

    try:
        ov, tone, ups, n, cues = cached_overnight()
    except Exception as e:
        data_error(e); st.stop()
    tone_col = {"RISK-ON": GREEN, "RISK-OFF": RED}.get(tone, ORANGE)
    tone_txt = {"RISK-ON": "Most overnight markets are up and/or the rand is firmer. Openings tend to be firmer on days like this.",
                "RISK-OFF": "Most overnight markets are down and/or the rand is weaker. Expect a softer open; be selective.",
                "MIXED": "Overnight signals disagree. The open may lack direction; let the first 30 minutes settle."}[tone]
    st.markdown(f"<div style='font-family:{FONT};font-size:22px;color:{tone_col};font-weight:bold'>{tone} "
                f"<span style='font-size:14px;color:{GREY}'>({ups}/{n} overnight signals positive)</span></div>"
                f"<div style='font-family:{FONT};color:#e0e0e0;margin-bottom:8px'>{tone_txt}</div>",
                unsafe_allow_html=True)

    left, right = st.columns([1, 1.25])
    with left:
        banner("OVERNIGHT MARKETS")
        show_table(ov.style.format({"Last": "{:,.2f}", "Change %": "{:+.2f}"}, na_rep="—")
                   .map(num_colour, subset=["Change %"]).set_properties(**CELL)
                   .set_properties(subset=["Market"], **{"color": "#fff", "font-weight": "bold"})
                   .set_properties(subset=["Ticker"], **{"color": ORANGE}).set_table_styles(TABLE_HEAD))
        st.caption("USD/ZAR down = rand stronger (usually good for banks and retailers, "
                   "less so for rand-hedge and mining shares).")
    with right:
        banner("JSE OPEN CUES — shares with offshore listings")
        if cues.empty:
            st.write("No offshore cues available right now.")
        else:
            show_table(cues.style.format({"Their move %": "{:+.2f}", "Currency effect %": "{:+.2f}",
                                          "Open cue %": "{:+.2f}"}, na_rep="—")
                       .map(num_colour, subset=["Their move %", "Currency effect %", "Open cue %"])
                       .set_properties(**CELL)
                       .set_properties(subset=["JSE share"], **{"color": "#fff", "font-weight": "bold"})
                       .set_properties(subset=["Ticker"], **{"color": ORANGE})
                       .set_properties(subset=["Open cue %"], **{"font-weight": "bold"})
                       .set_table_styles(TABLE_HEAD))
            st.caption("Open cue = the offshore listing's latest move plus the currency move into rand. "
                       "A rough guide to the opening direction, not a forecast: part of a US move happens "
                       "while the JSE is still open, and local news can override it.")

    banner("SETUPS TO WATCH — JSE shares & ETFs")
    try:
        su = cached_setups()
    except Exception as e:
        data_error(e); st.stop()
    if su.empty:
        st.write("No instruments meet the setup rules today. Quiet days are normal; no trade is also a position.")
    else:
        kinds = ["All"] + sorted(su["Setup"].unique())
        kind = st.radio("Setup type", kinds, horizontal=True)
        view = su if kind == "All" else su[su["Setup"] == kind]
        setup_col = {"Breakout watch": GREEN, "Pullback in uptrend": "#69f0ae",
                     "Breakdown risk": RED, "Oversold (counter-trend)": ORANGE}
        show_table(view.style
                   .format({"Score": "{:+.0f}", "Last": "{:,.2f}", "Watch level": "{:,.2f}",
                            "Invalidation": "{:,.2f}", "2R reference": "{:,.2f}", "ATR": "{:,.2f}",
                            "RSI": "{:.0f}", "Bull edge 20d %": "{:+.2f}"}, na_rep="—")
                   .set_properties(**CELL)
                   .set_properties(subset=["Name"], **{"color": "#fff", "font-weight": "bold"})
                   .set_properties(subset=["Ticker"], **{"color": ORANGE})
                   .set_properties(subset=["Watch level"], **{"color": "#fff", "font-weight": "bold"})
                   .map(lambda v: f"color:{setup_col.get(v, GREY)};font-weight:bold", subset=["Setup"])
                   .map(lambda v: f"color:{BIAS_COL.get(v, GREY)}", subset=["Bias"])
                   .map(num_colour, subset=["Score", "Bull edge 20d %"])
                   .set_properties(subset=["Why", "Alerts"], **{"color": GREY, "white-space": "normal",
                                                                "min-width": "220px"})
                   .set_table_styles(TABLE_HEAD))
        with st.expander("How to read the setup table"):
            st.markdown(
                "- **Watch level**: the price that would confirm the setup (a 20-day high or low, or yesterday's high).\n"
                "- **Invalidation**: 1.5× the average daily range (ATR) away from the watch level. If price gets "
                "there, the setup has failed. Many traders put their stop-loss around this point.\n"
                "- **2R reference**: twice the distance between watch level and invalidation, a common minimum "
                "reward-to-risk check. It is not a price target.\n"
                "- **Bull edge 20d %**: whether bullish readings have actually helped on this share over 5 years. "
                "Be sceptical of setups where it is near zero or negative.")

    banner("POSITION SIZE CALCULATOR")
    c1, c2, c3, c4 = st.columns(4)
    acct = c1.number_input("Account size (R)", value=50000.0, step=5000.0)
    risk_pct = c2.number_input("Risk per trade %", value=1.0, min_value=0.1, max_value=5.0, step=0.25)
    entry = c3.number_input("Entry price (R)", value=100.0, step=1.0)
    stop = c4.number_input("Stop-loss price (R)", value=95.0, step=1.0)
    per_share = abs(entry - stop)
    if per_share > 0 and entry > 0:
        risk_r = acct * risk_pct / 100
        shares = int(risk_r // per_share)
        cost = shares * entry
        capped = cost > acct
        if capped:
            shares = int(acct // entry); cost = shares * entry
        st.markdown(
            f"<div style='font-family:{FONT};font-size:16px'>Buy up to <b style='color:{ORANGE}'>{shares:,} shares</b> "
            f"(≈ R{cost:,.0f}). If the stop is hit you lose about "
            f"<b style='color:{RED}'>R{shares * per_share:,.0f}</b> "
            f"({shares * per_share / acct * 100:.2f}% of the account), before costs.</div>",
            unsafe_allow_html=True)
        if capped:
            st.caption("Capped by account size: the stop is so close that the full risk amount would need more money than you have.")
        if cost > acct * 0.25:
            st.caption("⚠️ This position is over 25% of the account. Consider a smaller size or wider diversification.")
    else:
        st.caption("Entry and stop must be different prices.")

    with st.expander("📋 Opening playbook (read before trading)", expanded=False):
        st.markdown(
            "1. **Check the tone and cues above.** They set expectations, not certainties.\n"
            "2. **Skip the opening auction rush.** The JSE auction runs 08:30–09:00 and the first 15–30 minutes "
            "after 09:00 are the most volatile, with wider spreads. Many traders wait for the opening range to form.\n"
            "3. **Use limit orders, not market orders**, especially on smaller shares where spreads are wide.\n"
            "4. **Decide entry, stop and size before you place the trade**, using the calculator. Risking 0.5–2% of "
            "the account per trade is a common range.\n"
            "5. **Never move a stop further away or average down on a losing trade.**\n"
            "6. **Check company news** (SENS announcements, results dates, ex-dividend dates). News beats indicators.\n"
            "7. **Know your costs.** Brokerage, STT (0.25% on JSE share purchases) and spreads add up fast on frequent trades.\n"
            "8. **Keep a journal** of every trade and why you took it. After 30+ trades you'll know if your approach works.")

elif page == "📊 Market Monitor":
    grp = st.selectbox("Group", GROUPS)
    groups = core.UNIVERSE if grp == "All" else {grp: core.UNIVERSE[grp]}
    tickers = tuple(t for g in groups.values() for t, _ in g)
    try:
        closes = cached_closes(tickers, "1y")
    except Exception as e:
        data_error(e); st.stop()

    def pct(s, n):
        return (s.iloc[-1] / s.iloc[-1 - n] - 1) * 100 if len(s) > n else np.nan

    rows, missing = [], []
    for g, items in groups.items():
        for t, name in items:
            s = closes[t].dropna() if t in closes else pd.Series(dtype=float)
            if len(s) < 2:
                missing.append(t); continue
            ytd = s[s.index.year == s.index[-1].year]
            rows.append({"Group": g, "Name": name, "Ticker": t, "Last": s.iloc[-1],
                         "1D %": pct(s, 1), "1W %": pct(s, 5), "1M %": pct(s, 21),
                         "YTD %": (s.iloc[-1] / ytd.iloc[0] - 1) * 100 if len(ytd) else np.nan,
                         "1Y %": (s.iloc[-1] / s.iloc[0] - 1) * 100,
                         "Off 52W High %": (s.iloc[-1] / s.max() - 1) * 100})
    if not rows:
        data_error("No market data returned."); st.stop()
    board = pd.DataFrame(rows)
    pct_cols = ["1D %", "1W %", "1M %", "YTD %", "1Y %", "Off 52W High %"]

    banner("MARKET MONITOR")
    up = (board["1D %"] > 0).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Advancing", up)
    c2.metric("Declining", (board["1D %"] < 0).sum())
    best = board.loc[board["1D %"].idxmax()]
    c3.metric("Top mover", best["Name"], f"{best['1D %']:+.2f}%")

    show_table(board.style
               .format({"Last": "{:,.2f}", **{c: "{:+.2f}" for c in pct_cols}}, na_rep="—")
               .map(num_colour, subset=pct_cols).set_properties(**CELL)
               .set_properties(subset=["Name"], **{"color": "#fff", "font-weight": "bold"})
               .set_properties(subset=["Ticker", "Group"], **{"color": ORANGE})
               .set_table_styles(TABLE_HEAD))
    if missing:
        st.caption("No data right now for: " + ", ".join(missing))

    hm = board.dropna(subset=["1D %"]).copy()
    hm["label"] = hm["Name"] + "<br>" + hm["1D %"].map("{:+.2f}%".format)
    fig = px.treemap(hm, path=["Group", "label"], color="1D %", range_color=[-3, 3],
                     color_continuous_scale=[[0, "#b71c1c"], [0.5, "#1a1a1a"], [1, "#1b5e20"]])
    fig.update_traces(textfont=dict(family=FONT, size=13), marker=dict(line=dict(color=BG, width=2)),
                      hovertemplate="%{label}<extra></extra>")
    style_fig(fig, "1-DAY HEATMAP", 560).update_layout(margin=dict(l=5, r=5, t=60, b=5),
                                                       coloraxis_showscale=False)
    show_fig(fig)


# --------------------------------------------------------------------------
# Signal Scanner
# --------------------------------------------------------------------------

elif page == "🚦 Signal Scanner":
    c1, c2 = st.columns([2, 1])
    grp = c1.selectbox("Group", GROUPS)
    only_alerts = c2.checkbox("Only show instruments with alerts")
    try:
        table, feed, skipped = cached_scan(grp)
    except Exception as e:
        data_error(e); st.stop()
    if table.empty:
        data_error("No data returned for the scan."); st.stop()
    if only_alerts:
        table = table[table["Alerts"] > 0]

    banner("SIGNAL SCANNER")
    with st.expander("How the score works"):
        st.markdown("Four checks, each **+1 / −1**: price vs its 200-day average, 50-day vs 200-day "
                    "(golden/death cross), MACD vs its signal line, and 3-month return. "
                    "**+3/+4** strong bullish, **−3/−4** strong bearish.\n\n"
                    "**Bull edge 20d %**: over the last 5 years, how much better this instrument did in the "
                    "20 days after a bullish reading than in an average 20 days. Near zero or negative means "
                    "the signal hasn't helped there.\n\n"
                    "These are mechanical indicator readings, not buy or sell advice.")
    show_table(table.style
               .format({"Last": "{:,.2f}", "Score": "{:+.0f}", "RSI": "{:.0f}", "3M %": "{:+.1f}",
                        "Bull edge 20d %": "{:+.2f}"}, na_rep="—")
               .set_properties(**CELL)
               .set_properties(subset=["Name"], **{"color": "#fff", "font-weight": "bold"})
               .set_properties(subset=["Ticker"], **{"color": ORANGE})
               .map(lambda v: f"color:{BIAS_COL.get(v, GREY)};font-weight:bold", subset=["Bias"])
               .map(word_colour, subset=["Trend", "50/200", "MACD"])
               .map(lambda v: f"color:{RED if v >= 70 else GREEN if v <= 30 else '#e0e0e0'}", subset=["RSI"])
               .map(num_colour, subset=["Score", "3M %", "Bull edge 20d %"])
               .set_table_styles(TABLE_HEAD))
    if skipped:
        st.caption("Not enough data for: " + ", ".join(skipped))

    banner("ALERT FEED — most recent first")
    if feed.empty:
        st.write("No new alerts.")
    else:
        bull_words = ("Golden", "bullish", "above", "Oversold", "high")
        show_table(feed.style.set_properties(**CELL)
                   .set_properties(subset=["Ticker"], **{"color": ORANGE})
                   .map(lambda v: f"color:{GREEN if any(k in v for k in bull_words) else RED};font-weight:bold",
                        subset=["Signal"])
                   .map(lambda v: f"color:{BIAS_COL.get(v, GREY)}", subset=["Bias"])
                   .set_table_styles(TABLE_HEAD))


# --------------------------------------------------------------------------
# Security Chart
# --------------------------------------------------------------------------

elif page == "📈 Security Chart":
    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox("Instrument", list(core.LABELS), index=0)
    custom = c1.text_input("…or type any Yahoo ticker (e.g. NPN.JO, AAPL)", "")
    period = c2.selectbox("Period", ["6mo", "1y", "2y", "5y", "10y", "max"], index=2)
    t = custom.strip().upper() or core.LABELS[choice]
    try:
        df = cached_ohlc(t, period)
    except Exception as e:
        data_error(e); st.stop()
    c = df["Close"]
    chg = (c.iloc[-1] / c.iloc[-2] - 1) * 100
    ytd = c[c.index.year == c.index[-1].year]
    a = core.analyse(c) if len(c) > 210 else None

    banner(f"{core.NAMES.get(t, t)} &nbsp;[{t}]")
    stats = {"LAST": f"{c.iloc[-1]:,.2f}", "CHG": f"{chg:+.2f}%", "52W HIGH": f"{c.tail(252).max():,.2f}",
             "52W LOW": f"{c.tail(252).min():,.2f}", "YTD": f"{(c.iloc[-1] / ytd.iloc[0] - 1) * 100:+.2f}%",
             "VOL (ANN)": f"{c.pct_change().tail(252).std() * 252 ** 0.5 * 100:.1f}%"}
    html = "".join(f"<div class='stat'><div class='k'>{k}</div><div class='v' style='color:"
                   f"{(GREEN if chg >= 0 else RED) if k == 'CHG' else '#fff'}'>{v}</div></div>" for k, v in stats.items())
    st.markdown(html, unsafe_allow_html=True)
    if a:
        alerts = " • ".join(txt for _, txt in a["events"]) or "no new alerts"
        col = GREEN if a["Score"] > 0 else RED if a["Score"] < 0 else GREY
        st.markdown(f"<div style='font-family:{FONT};padding:4px 0'>SIGNAL: <b style='color:{col}'>"
                    f"{a['Bias']} ({a['Score']:+.0f})</b> &nbsp;|&nbsp; {alerts}</div>", unsafe_allow_html=True)
    else:
        st.caption("Pick a longer period (2y+) to see signals.")

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.55, 0.13, 0.16, 0.16],
                        vertical_spacing=0.025)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=c,
                                 name=t, increasing_line_color=GREEN, decreasing_line_color=RED), 1, 1)
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA 50", line=dict(color=ORANGE, width=1.3)), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=sma200, name="SMA 200", line=dict(color="#29b6f6", width=1.3)), 1, 1)
    flip = np.sign(sma50 - sma200).diff().fillna(0)
    for val, name, col, sym in [(2, "Golden cross", GREEN, "triangle-up"), (-2, "Death cross", RED, "triangle-down")]:
        idx = flip[flip == val].index
        fig.add_trace(go.Scatter(x=idx, y=c[idx], mode="markers", name=name,
                                 marker=dict(symbol=sym, size=13, color=col, line=dict(color="#fff", width=1))), 1, 1)
    if df["Volume"].sum() > 0:
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=np.where(c >= df["Open"], GREEN, RED),
                             opacity=0.6, showlegend=False, name="Volume"), 2, 1)
    fig.add_trace(go.Scatter(x=df.index, y=core.rsi(c), name="RSI 14", line=dict(color="#ce93d8", width=1.2)), 3, 1)
    fig.add_hline(y=70, line_dash="dot", line_color=RED, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=GREEN, row=3, col=1)
    m, sg = core.macd(c)
    hist = m - sg
    fig.add_trace(go.Bar(x=df.index, y=hist, marker_color=np.where(hist >= 0, GREEN, RED), opacity=0.6,
                         showlegend=False, name="MACD hist"), 4, 1)
    fig.add_trace(go.Scatter(x=df.index, y=m, name="MACD", line=dict(color=ORANGE, width=1.1)), 4, 1)
    fig.add_trace(go.Scatter(x=df.index, y=sg, name="Signal", line=dict(color="#29b6f6", width=1.1)), 4, 1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_xaxes(rangeselector=dict(
        buttons=[dict(count=1, label="1M", step="month", stepmode="backward"),
                 dict(count=6, label="6M", step="month", stepmode="backward"),
                 dict(count=1, label="YTD", step="year", stepmode="todate"),
                 dict(count=1, label="1Y", step="year", stepmode="backward"), dict(step="all", label="ALL")],
        bgcolor="#1a1a1a", activecolor=ORANGE, font=dict(color="#fff")), row=1, col=1)
    show_fig(style_fig(fig, core.label(t), 900))


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------

elif page == "🔀 Compare":
    defaults = [core.label(t) for t in ("STX40.JO", "STX500.JO", "STXNDQ.JO", "GLD.JO", "USDZAR=X")]
    picks = st.multiselect("Instruments", list(core.LABELS), default=defaults)
    start = st.date_input("Start date", date(2021, 1, 1))
    banner(f"RELATIVE PERFORMANCE SINCE {start:%d %b %Y} — rebased to 100")
    fig, summary = go.Figure(), []
    for p in picks:
        t = core.LABELS[p]
        try:
            s = cached_ohlc(t, start=start.isoformat())["Close"]
        except Exception as e:
            st.warning(str(e)); continue
        r = s / s.iloc[0] * 100
        fig.add_trace(go.Scatter(x=r.index, y=r, name=core.NAMES[t], line=dict(width=1.6)))
        summary.append({"Name": core.NAMES[t], "Ticker": t, "Return %": r.iloc[-1] - 100})
    fig.add_hline(y=100, line_dash="dot", line_color=GREY)
    show_fig(style_fig(fig, "", 560))
    if summary:
        show_table(pd.DataFrame(summary).sort_values("Return %", ascending=False).style
                   .format({"Return %": "{:+.1f}"}).map(num_colour, subset=["Return %"])
                   .set_properties(**CELL).set_properties(subset=["Ticker"], **{"color": ORANGE})
                   .set_table_styles(TABLE_HEAD))


# --------------------------------------------------------------------------
# Backtests
# --------------------------------------------------------------------------

elif page == "🧪 Backtests":
    etf_labels = [core.label(t) for g in ("JSE ETFs", "US ETFs") for t, _ in core.UNIVERSE[g]]
    oil_labels = [core.label(t) for t in ("BNO", "USO", "BZ=F", "CL=F")]
    tab1, tab2, tab3 = st.tabs(["ETF strategies", "Covered calls", "Oil trend"])

    with tab1:
        with st.form("etf"):
            a, b, c = st.columns(3)
            pick = a.selectbox("ETF", etf_labels)
            start = b.date_input("Start", date(2015, 1, 1), key="etf_start")
            capital = c.number_input("Lump sum", value=100000, step=10000)
            monthly = a.number_input("Monthly contribution", value=2000, step=500)
            sma_days = b.slider("Trend filter (days)", 20, 300, 200, 10)
            cash_rate = c.number_input("Money market rate", value=0.07, format="%.3f")
            cost = a.number_input("Cost per trade", value=0.002, format="%.4f", key="etf_cost")
            demo = b.checkbox("Use demo data", key="etf_demo")
            go_btn = st.form_submit_button("▶ Run backtest", type="primary")
        if go_btn:
            t = core.LABELS[pick]
            try:
                prices = cached_prices(t, start.isoformat(), demo, "etf")
                res, report = captured(sb.run_etf, prices, capital, monthly, sma_days, cost, cash_rate,
                                       "DEMO" if demo else t)
            except Exception as e:
                data_error(e); st.stop()
            st.code(report.strip(), language=None)
            show_fig(equity_chart(res, {"lump_sum": "Buy & hold", "trend_filter": f"Trend filter ({sma_days}d)"},
                                  f"{core.NAMES[t]}: buy & hold vs trend filter"))
            fig = go.Figure(go.Scatter(x=res.index, y=res["monthly"], line=dict(color=GREEN), fill="tozeroy",
                                       name="Monthly contributions"))
            show_fig(style_fig(fig, f"Monthly contributions of {monthly:,.0f}", 400))
            st.download_button("💾 Download daily results (CSV)", res.to_csv().encode(), "etf_results.csv")

    with tab2:
        with st.form("cc"):
            a, b, c = st.columns(3)
            pick = a.selectbox("Underlying", etf_labels, index=etf_labels.index(core.label("SPY")))
            start = b.date_input("Start", date(2015, 1, 1), key="cc_start")
            capital = c.number_input("Capital", value=100000, step=10000, key="cc_cap")
            otm = a.slider("Strike above price", 0.0, 0.10, 0.03, 0.005, format="%.3f")
            hold = b.slider("Days per option", 5, 63, 21)
            iv_mult = c.number_input("Implied / realised vol", value=1.1, step=0.05)
            rate = a.number_input("Risk-free rate", value=0.04, format="%.3f")
            fee = b.number_input("Fee per contract", value=0.65)
            demo = c.checkbox("Use demo data", key="cc_demo")
            go_btn = st.form_submit_button("▶ Run backtest", type="primary")
        if go_btn:
            t = core.LABELS[pick]
            try:
                prices = cached_prices(t, start.isoformat(), demo, "cc")
                res, report = captured(sb.run_covered_call, prices, capital, otm, hold, iv_mult, rate, fee,
                                       "DEMO" if demo else t)
            except Exception as e:
                data_error(e); st.stop()
            st.code(report.strip(), language=None)
            st.caption("Premiums are estimated with Black-Scholes from recent volatility, so treat results as an approximation.")
            show_fig(equity_chart(res, {"buy_hold": "Buy & hold", "covered_call": "Covered call"},
                                  f"{core.NAMES[t]}: covered call ({otm * 100:.1f}% OTM) vs buy & hold"))
            st.download_button("💾 Download daily results (CSV)", res.to_csv().encode(), "covered_call_results.csv")

    with tab3:
        with st.form("oil"):
            a, b, c = st.columns(3)
            pick = a.selectbox("Market", oil_labels)
            start = b.date_input("Start", date(2015, 1, 1), key="oil_start")
            capital = c.number_input("Capital", value=100000, step=10000, key="oil_cap")
            fast = a.slider("Fast average", 5, 100, 20, 5)
            slow = b.slider("Slow average", 20, 300, 100, 10)
            stop_atr = c.slider("Trailing stop (x avg daily move, 0 = off)", 0.0, 15.0, 5.0, 0.5)
            allow_short = a.checkbox("Allow short trades")
            cost = b.number_input("Cost per trade", value=0.002, format="%.4f", key="oil_cost")
            demo = c.checkbox("Use demo data", key="oil_demo")
            go_btn = st.form_submit_button("▶ Run backtest", type="primary")
        if go_btn:
            t = core.LABELS[pick]
            try:
                prices = cached_prices(t, start.isoformat(), demo, "oil")
                res, report = captured(sb.run_oil, prices, capital, fast, slow, allow_short, cost, stop_atr,
                                       "DEMO" if demo else t)
            except Exception as e:
                data_error(e); st.stop()
            st.code(report.strip(), language=None)
            show_fig(equity_chart(res, {"buy_hold": "Buy & hold", "trend": "Trend strategy"},
                                  f"{core.NAMES[t]}: trend strategy vs buy & hold"))
            pos = res["position"]
            change = pos.diff().fillna(pos)
            fig = go.Figure(go.Scatter(x=prices.index, y=prices, name="Price", line=dict(color="#e0e0e0", width=1.2)))
            for n, col in [(fast, ORANGE), (slow, "#29b6f6")]:
                fig.add_trace(go.Scatter(x=prices.index, y=prices.rolling(n).mean(), name=f"SMA {n}",
                                         line=dict(color=col, width=1)))
            buys, sells = change[change > 0].index, change[change < 0].index
            fig.add_trace(go.Scatter(x=buys, y=prices[buys], mode="markers", name="Buy / cover",
                                     marker=dict(symbol="triangle-up", size=10, color=GREEN)))
            fig.add_trace(go.Scatter(x=sells, y=prices[sells], mode="markers", name="Sell / exit",
                                     marker=dict(symbol="triangle-down", size=10, color=RED)))
            show_fig(style_fig(fig, "Trade signals", 520))
            st.download_button("💾 Download daily results (CSV)", res.to_csv().encode(), "oil_results.csv")
