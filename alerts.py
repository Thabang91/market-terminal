"""
Nightly alert job: scans every instrument and sends NEW signals to Telegram.
Run by GitHub Actions after the JSE closes (see .github/workflows/alerts.yml).

Local test:
    export TELEGRAM_TOKEN=...  TELEGRAM_CHAT_ID=...
    python alerts.py
"""

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

import terminal_core as core


def send(text):
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID not set; printing instead:\n")
        print(text)
        return
    for i in range(0, len(text), 3900):  # Telegram limit is 4096 characters per message
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": text[i:i + 3900], "parse_mode": "HTML",
                                "disable_web_page_preview": True}, timeout=30)
        r.raise_for_status()


def main():
    always_send = "--always" in sys.argv  # send a summary even when there are no new alerts
    tickers = tuple(t for g in core.UNIVERSE.values() for t, _ in g)
    closes = core.download_closes(tickers, "2y")
    if closes.empty:
        sys.exit("No data downloaded from Yahoo Finance.")

    lines, scores = [], []
    for t in tickers:
        c = closes[t].dropna() if t in closes else None
        if c is None or len(c) < 260:
            continue
        a = core.analyse(c)
        scores.append((a["Score"], core.NAMES[t]))
        for alert in core.fresh_alerts(c):
            lines.append(f"<b>{core.NAMES[t]}</b> ({t}) {c.iloc[-1]:,.2f}\n   {alert} · bias {a['Bias'].lower()}")

    header = f"🟧 <b>Market Terminal</b> · {datetime.now(ZoneInfo('Africa/Johannesburg')):%a %d %b %Y}\n"
    if lines:
        send(header + f"{len(lines)} new signal(s):\n\n" + "\n\n".join(lines) +
             "\n\n<i>Indicator readings, not advice.</i>")
    elif always_send:
        top = sorted(scores, reverse=True)[:3]
        bottom = sorted(scores)[:3]
        send(header + "No new signals today.\n\n"
             "Strongest: " + ", ".join(f"{n} ({s:+.0f})" for s, n in top) + "\n"
             "Weakest: " + ", ".join(f"{n} ({s:+.0f})" for s, n in bottom))
    else:
        print("No new signals today; nothing sent.")


if __name__ == "__main__":
    main()
