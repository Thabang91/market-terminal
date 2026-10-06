"""
Strategy Backtester: ETF investing, covered calls, and oil trend-following.

Tests trading ideas on historical data BEFORE risking real money.
Educational tool only, not financial advice. Past performance does not
guarantee future results.

Setup:
    pip install yfinance pandas numpy matplotlib

Examples:
    python strategy_backtester.py etf --ticker STX40.JO --monthly 2000
    python strategy_backtester.py etf --ticker SPY --start 2010-01-01
    python strategy_backtester.py covered-call --ticker SPY --otm 0.03
    python strategy_backtester.py oil --ticker BNO --fast 20 --slow 100
    python strategy_backtester.py all --demo      (synthetic data, no internet)

Ticker tips:
    JSE ETFs on Yahoo end in .JO  (STX40.JO, STX500.JO, STXNDQ.JO)
    US ETFs: SPY, VOO, QQQ
    Oil: BNO (Brent ETF), USO (WTI ETF), CL=F (WTI futures, continuous)
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def synthetic_prices(start, end, s0=100.0, mu=0.08, sigma=0.18, seed=42):
    """Random price path for testing without internet."""
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    dt = 1 / TRADING_DAYS
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * math.sqrt(dt), len(dates))
    return pd.Series(s0 * np.exp(np.cumsum(shocks)), index=dates, name="Close")


def load_prices(ticker, start, end, demo=False, demo_params=None):
    if demo:
        return synthetic_prices(start, end, **(demo_params or {}))
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance  (or use --demo)")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Yahoo may be busy (retry in a minute) or the ticker is wrong.")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns multi-index columns
        close = close.iloc[:, 0]
    close = close.dropna()
    close = close[close > 0]  # e.g. negative WTI futures price in April 2020
    if ticker.upper().endswith(".JO"):
        close = close / 100  # Yahoo quotes JSE shares in cents; convert to rand
    return close


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def irr(cashflows, final_value, end_date):
    """Annual money-weighted return for regular contributions (bisection)."""
    t = np.array([(d - end_date).days / 365.25 for d in cashflows.index])
    amounts = cashflows.values

    def npv(rate):  # value of contributions at end date vs final value
        return (amounts * (1 + rate) ** (-t)).sum() - final_value

    lo, hi = -0.99, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid) > 0:
            hi = mid
        else:
            lo = mid
    return mid


def contribution_metrics(values, cashflows):
    final = values.iloc[-1]
    total = cashflows.sum()
    return {"Final value": final, "Total contributed": total,
            "Profit": final - total,
            "Annual return (IRR) %": irr(cashflows, final, values.index[-1]) * 100}


def metrics(equity, contributions=None):
    equity = equity.dropna()
    rets = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    out = {"Final value": equity.iloc[-1]}
    if contributions is not None:
        out["Total contributed"] = contributions
        out["Profit"] = equity.iloc[-1] - contributions
    else:
        total = equity.iloc[-1] / equity.iloc[0] - 1
        out["Total return %"] = total * 100
        out["CAGR %"] = ((1 + total) ** (1 / years) - 1) * 100 if years > 0 else np.nan
    vol = rets.std() * math.sqrt(TRADING_DAYS)
    out["Volatility %"] = vol * 100
    out["Sharpe (rf=0)"] = (rets.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    peak = equity.cummax()
    out["Max drawdown %"] = ((equity / peak) - 1).min() * 100
    return out


def print_table(title, results):
    print(f"\n=== {title} ===")
    df = pd.DataFrame(results).T
    with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 200, "display.max_columns", None):
        print(df)


def plot(curves, title, filename):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, series in curves.items():
        ax.plot(series.index, series.values, label=name, linewidth=1.4)
    ax.set_title(title)
    ax.set_ylabel("Portfolio value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"Chart saved: {filename}")


# --------------------------------------------------------------------------
# Strategy 1: ETF investing (lump sum, monthly debit order, trend filter)
# --------------------------------------------------------------------------

def run_etf(prices, capital, monthly, sma_days, cost, cash_rate, label):
    daily_ret = prices.pct_change().fillna(0)

    # A) Lump sum buy-and-hold
    lump = capital * (1 + daily_ret).cumprod()

    # B) Monthly contributions (like a debit order into a TFSA)
    units, values, flows = 0.0, [], {}
    month_starts = set(prices.groupby(prices.index.to_period("M")).head(1).index)
    for date, price in prices.items():
        if date in month_starts:
            units += monthly * (1 - cost) / price
            flows[date] = monthly
        values.append(units * price)
    dca = pd.Series(values, index=prices.index)

    # C) Trend filter: hold ETF only when price is above its moving average,
    #    otherwise sit in cash (money market). Signal uses yesterday's data.
    sma = prices.rolling(sma_days).mean()
    in_market = (prices > sma).shift(1).fillna(False).astype(int)
    switches = in_market.diff().abs().fillna(0)
    cash_daily = (1 + cash_rate) ** (1 / TRADING_DAYS) - 1
    trend_ret = in_market * daily_ret + (1 - in_market) * cash_daily - switches * cost
    trend = capital * (1 + trend_ret).cumprod()

    print_table(f"ETF strategies on {label}", {
        "Lump sum buy & hold": metrics(lump),
        f"Trend filter ({sma_days}d SMA)": metrics(trend),
    })
    print_table(f"Monthly contributions of {monthly:,.0f} into {label}",
                {"Monthly contributions": contribution_metrics(dca, pd.Series(flows))})
    print(f"Trend filter: {int(switches.sum())} switches, in market "
          f"{in_market.mean() * 100:.0f}% of the time")
    plot({"Lump sum buy & hold": lump, f"Trend filter ({sma_days}d SMA)": trend},
         f"ETF lump sum vs trend filter: {label}", "etf_strategies.png")
    plot({"Monthly contributions": dca}, f"Monthly contributions: {label}",
         "etf_monthly_contributions.png")
    return pd.DataFrame({"lump_sum": lump, "trend_filter": trend, "monthly": dca})


# --------------------------------------------------------------------------
# Strategy 2: Covered calls
# --------------------------------------------------------------------------

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(S, K, T, r, sigma):
    """Black-Scholes price of a European call."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def run_covered_call(prices, capital, otm, hold_days, iv_mult, rate,
                     fee_per_contract, label):
    """
    Each cycle: own the shares, sell a call `otm` above the price expiring in
    `hold_days` trading days. Collect the premium. At expiry, if the price is
    above the strike, pay the difference (equivalent to shares being called
    away and bought back). Premiums and losses are rolled into shares.

    Historical option prices are not freely available, so premiums are
    ESTIMATED with Black-Scholes using recent realised volatility x iv_mult.
    Real implied volatility is usually a bit higher than realised, which is
    why iv_mult defaults to 1.1. Treat results as an approximation.
    """
    log_ret = np.log(prices / prices.shift(1))
    realised = log_ret.rolling(21).std() * math.sqrt(TRADING_DAYS)

    shares = capital / prices.iloc[0]
    cash = 0.0
    strike, expiry_idx = None, None
    premiums, assigned, cycles = 0.0, 0, 0
    equity = []

    for i, (date, S) in enumerate(prices.items()):
        # Expiry: settle the short call, reinvest everything into shares
        if strike is not None and i == expiry_idx:
            payoff = max(S - strike, 0.0) * shares
            if payoff > 0:
                assigned += 1
            cash -= payoff
            shares += cash / S
            cash = 0.0
            strike = None

        # Sell a new call
        if strike is None and i + hold_days < len(prices):
            vol = realised.iloc[i]
            vol = 0.20 if np.isnan(vol) else vol * iv_mult
            strike = S * (1 + otm)
            expiry_idx = i + hold_days
            prem = bs_call(S, strike, hold_days / TRADING_DAYS, rate, vol)
            contracts = shares / 100
            cash += prem * shares - fee_per_contract * max(contracts, 1)
            premiums += prem * shares
            cycles += 1

        # Mark to market: shares + cash - current value of the short call
        liability = 0.0
        if strike is not None:
            T_left = (expiry_idx - i) / TRADING_DAYS
            vol = realised.iloc[i]
            vol = 0.20 if np.isnan(vol) else vol * iv_mult
            liability = bs_call(S, strike, T_left, rate, vol) * shares
        equity.append(shares * S + cash - liability)

    cc = pd.Series(equity, index=prices.index)
    bh = capital * prices / prices.iloc[0]

    print_table(f"Covered calls on {label} ({otm * 100:.1f}% OTM, {hold_days}-day)", {
        "Buy & hold": metrics(bh),
        "Covered call": metrics(cc),
    })
    print(f"Cycles: {cycles} | Calls finished in the money: {assigned} "
          f"({assigned / max(cycles, 1) * 100:.0f}%) | Premium collected: {premiums:,.2f}")
    min_capital = prices.iloc[-1] * 100
    print(f"Note: one real contract = 100 shares. At today's price you need about "
          f"{min_capital:,.0f} per contract.")
    plot({"Buy & hold": bh, "Covered call": cc},
         f"Covered call vs buy & hold: {label}", "covered_call.png")
    return pd.DataFrame({"buy_hold": bh, "covered_call": cc})


# --------------------------------------------------------------------------
# Strategy 3: Oil trend-following
# --------------------------------------------------------------------------

def run_oil(prices, capital, fast, slow, allow_short, cost, stop_atr, label):
    """
    Moving-average crossover: long when fast SMA > slow SMA, otherwise flat
    (or short if allow_short). Optional trailing stop based on volatility.
    Signals use yesterday's close to avoid look-ahead bias.
    """
    daily_ret = prices.pct_change().fillna(0)
    sma_f = prices.rolling(fast).mean()
    sma_s = prices.rolling(slow).mean()
    raw = np.where(sma_f > sma_s, 1, -1 if allow_short else 0)
    raw = pd.Series(raw, index=prices.index)
    raw[sma_s.isna()] = 0

    # Volatility trailing stop: exit a long if price falls stop_atr x daily
    # range-proxy below the highest close since entry (and vice versa for shorts).
    position = raw.copy()
    if stop_atr > 0:
        atr = prices.diff().abs().rolling(14).mean()
        pos, extreme, stopped_sig = 0, None, None
        for i in range(len(prices)):
            sig, p = raw.iloc[i], prices.iloc[i]
            if stopped_sig is not None and sig == stopped_sig:
                position.iloc[i] = 0  # wait for a fresh signal after a stop
                continue
            stopped_sig = None
            if sig != pos:
                pos, extreme = sig, p
            if pos == 1:
                extreme = max(extreme, p)
                if p < extreme - stop_atr * atr.iloc[i]:
                    stopped_sig, pos = 1, 0
            elif pos == -1:
                extreme = min(extreme, p)
                if p > extreme + stop_atr * atr.iloc[i]:
                    stopped_sig, pos = -1, 0
            position.iloc[i] = pos

    held = position.shift(1).fillna(0)
    trades = held.diff().abs().fillna(0)
    strat_ret = held * daily_ret - trades * cost
    trend = capital * (1 + strat_ret).cumprod()
    bh = capital * (1 + daily_ret).cumprod()

    print_table(f"Oil trend-following on {label} (SMA {fast}/{slow})", {
        "Buy & hold": metrics(bh),
        "Trend strategy": metrics(trend),
    })
    in_trade = held != 0
    wins = (strat_ret[in_trade] > 0).mean() * 100 if in_trade.any() else 0
    print(f"Position changes: {int(trades.sum())} | Days in a trade: "
          f"{in_trade.mean() * 100:.0f}% | Winning days while in a trade: {wins:.0f}%")
    plot({"Buy & hold": bh, "Trend strategy": trend},
         f"Oil trend-following vs buy & hold: {label}", "oil_trend.png")
    return pd.DataFrame({"buy_hold": bh, "trend": trend, "position": held})


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Backtest ETF, covered call and oil strategies.")
    p.add_argument("strategy", choices=["etf", "covered-call", "oil", "all"])
    p.add_argument("--ticker", help="Yahoo Finance ticker (defaults depend on strategy)")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--capital", type=float, default=100_000)
    p.add_argument("--demo", action="store_true", help="Use synthetic data (no internet)")
    p.add_argument("--cost", type=float, default=0.002,
                   help="Cost per trade as fraction incl. brokerage & spread (0.002 = 0.2%%)")
    # ETF
    p.add_argument("--monthly", type=float, default=2000, help="Monthly contribution")
    p.add_argument("--sma", type=int, default=200, help="Trend filter moving average days")
    p.add_argument("--cash-rate", type=float, default=0.07, help="Annual money market rate")
    # Covered call
    p.add_argument("--otm", type=float, default=0.03, help="Strike %% above price (0.03 = 3%%)")
    p.add_argument("--hold-days", type=int, default=21, help="Trading days per option (21 ~ 1 month)")
    p.add_argument("--iv-mult", type=float, default=1.1, help="Implied / realised vol ratio")
    p.add_argument("--rate", type=float, default=0.04, help="Risk-free rate for option pricing")
    p.add_argument("--fee", type=float, default=0.65, help="Fee per option contract")
    # Oil
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=100)
    p.add_argument("--short", action="store_true", help="Allow short positions in oil")
    p.add_argument("--stop-atr", type=float, default=5.0,
                   help="Trailing stop in multiples of avg daily move (0 = off)")
    a = p.parse_args()

    end = a.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    run = [a.strategy] if a.strategy != "all" else ["etf", "covered-call", "oil"]
    defaults = {"etf": "STX40.JO", "covered-call": "SPY", "oil": "BNO"}
    demo_params = {"etf": {"mu": 0.09, "sigma": 0.16, "seed": 1},
                   "covered-call": {"mu": 0.10, "sigma": 0.18, "seed": 2},
                   "oil": {"mu": 0.02, "sigma": 0.38, "seed": 3}}

    for s in run:
        ticker = a.ticker if (a.ticker and a.strategy != "all") else defaults[s]
        label = "SYNTHETIC DEMO DATA" if a.demo else ticker
        prices = load_prices(ticker, a.start, end, a.demo, demo_params[s])
        if s == "etf":
            out = run_etf(prices, a.capital, a.monthly, a.sma, a.cost, a.cash_rate, label)
        elif s == "covered-call":
            out = run_covered_call(prices, a.capital, a.otm, a.hold_days, a.iv_mult,
                                   a.rate, a.fee, label)
        else:
            out = run_oil(prices, a.capital, a.fast, a.slow, a.short, a.cost,
                          a.stop_atr, label)
        out.to_csv(f"{s}_results.csv")
        print(f"Daily results saved: {s}_results.csv")

    print("\nReminder: backtests ignore taxes, may understate costs and slippage, "
          "and past results do not guarantee future returns.")


if __name__ == "__main__":
    main()
