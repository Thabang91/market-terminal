# 🟧 Market Terminal

Personal market dashboard: market monitor, signal scanner, candlestick charts with RSI/MACD,
performance comparison, strategy backtests, and nightly Telegram alerts.
Educational tool only, not financial advice. Data from Yahoo Finance (personal use).

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit web app |
| `terminal_core.py` | Instrument list, data download, indicators, signal scoring |
| `strategy_backtester.py` | ETF, covered call and oil backtest engine |
| `alerts.py` | Nightly scan that sends new signals to Telegram |
| `.github/workflows/alerts.yml` | Runs `alerts.py` at 18:30 SAST on weekdays |
| `.streamlit/config.toml` | Dark terminal theme |

## 1. Run it on your computer (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 2. Put it on GitHub

1. Create a new repository on github.com, e.g. `market-terminal`. Choose **Private**.
2. Upload all the files, keeping the `.streamlit` and `.github` folders. Easiest from the
   command line:
   ```bash
   cd market-terminal
   git init && git add . && git commit -m "Market Terminal"
   git branch -M main
   git remote add origin https://github.com/<your-username>/market-terminal.git
   git push -u origin main
   ```
   (If you drag-and-drop in the GitHub website instead, check that the hidden `.streamlit`
   and `.github` folders came across.)

## 3. Deploy on Streamlit Community Cloud

1. Go to **share.streamlit.io** and sign in with GitHub.
2. Click **Create app → Deploy a public app from GitHub**.
3. Choose your repository, branch `main`, main file `app.py`. Pick a custom URL if you like.
4. Click **Deploy**. The first build takes a few minutes.
5. To keep it private: open the app's **Settings → Sharing** and restrict who can view it to
   your own email.

Every `git push` redeploys automatically, like Netlify.

## 4. Telegram alerts

1. In Telegram, message **@BotFather**, send `/newbot`, and follow the prompts. Copy the
   **token** it gives you.
2. Open your new bot and send it any message (e.g. "hi").
3. In a browser, open `https://api.telegram.org/bot<TOKEN>/getUpdates` and find
   `"chat":{"id": ...}`. That number is your **chat ID**.
4. In your GitHub repo: **Settings → Secrets and variables → Actions → New repository secret**.
   Add `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID`.
5. Test it: **Actions** tab → *Nightly market alerts* → **Run workflow**. You should get a
   message within a minute or two.

From then on it runs every weekday at 18:30 SAST and only messages you when something new
happens (golden/death crosses, 200-day breaks, MACD crosses, RSI turning overbought/oversold,
new 52-week highs/lows). Change the time in `.github/workflows/alerts.yml` (the cron is in UTC).

## Customising

* **Add instruments:** edit `UNIVERSE` in `terminal_core.py`. JSE shares use `.JO`
  (e.g. `NPN.JO` for Naspers). Any Yahoo ticker can also be typed into the chart screen.
* **Data errors:** Yahoo occasionally blocks cloud servers. Wait a minute and use
  🔄 Refresh data in the sidebar.
