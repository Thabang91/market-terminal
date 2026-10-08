"""
Morning brief sent to Telegram before the JSE opens.
Run by GitHub Actions at 08:15 SAST on weekdays (see .github/workflows/morning.yml).
Mechanical readings, not trading advice.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

import terminal_core as core
from alerts import send

ICON = {"Breakout watch": "🟢", "Pullback in uptrend": "🟩", "Breakdown risk": "🔴",
        "Oversold (counter-trend)": "🟠"}


def fmt(v, pct=True):
    return "—" if pd.isna(v) else (f"{v:+.2f}%" if pct else f"{v:,.2f}")


def main():
    now = datetime.now(ZoneInfo("Africa/Johannesburg"))
    ov, tone, ups, n, cues = core.overnight_moves()
    su = core.setups()

    tone_icon = {"RISK-ON": "🟢", "RISK-OFF": "🔴"}.get(tone, "🟠")
    lines = [f"🌅 <b>Morning Brief</b> · {now:%a %d %b %Y}",
             f"{tone_icon} Tone: <b>{tone}</b> ({ups}/{n} overnight signals positive)", "",
             "<b>Overnight</b>"]
    for _, r in ov.iterrows():
        if r["Ticker"] in ("ES=F", "NQ=F", "^HSI", "^AXJO", "BZ=F", "GC=F", "PL=F", "USDZAR=X"):
            extra = f" ({fmt(r['Last'], False)})" if r["Ticker"] == "USDZAR=X" else ""
            lines.append(f"  {r['Market']}: {fmt(r['Change %'])}{extra}")

    if not cues.empty:
        lines += ["", "<b>JSE open cues</b> (offshore move + currency)"]
        for _, r in cues.head(6).iterrows():
            lines.append(f"  {r['JSE share']}: {fmt(r['Open cue %'])} · via {r['Follows']}")

    lines += ["", "<b>Setups to watch</b>"]
    if su.empty:
        lines.append("  None today. Quiet days are normal.")
    else:
        for _, r in su.head(8).iterrows():
            lines.append(f"  {ICON.get(r['Setup'], '•')} <b>{r['Name']}</b> {r['Setup'].lower()}: "
                         f"watch {r['Watch level']:,.2f}, invalid {r['Invalidation']:,.2f}")
        if len(su) > 8:
            lines.append(f"  …and {len(su) - 8} more in the app.")

    lines += ["", "Auction 08:30–09:00. First 15–30 min are the most volatile; use limit orders.",
              "<i>Indicator readings, not advice. Size every trade for your risk.</i>"]
    send("\n".join(lines))


if __name__ == "__main__":
    main()
