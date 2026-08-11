import os
import requests
import pandas as pd
import yfinance as yf
import numpy as np
import mplfinance as mpf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

MIN_PRICE = 50.0
MAX_PRICE = 200.0
MAX_MA_DIST_PCT = 1.0   # Max % distance between 20 SMA & 200 SMA
MAX_ATR_PCT = 0.50      # Max 14-period ATR (%)

def send_to_discord(caption, photo_path=None):
    if not DISCORD_WEBHOOK_URL:
        print("No DISCORD_WEBHOOK_URL set. Skipping notification.")
        return
    
    try:
        if photo_path and os.path.exists(photo_path):
            with open(photo_path, "rb") as photo:
                files = {"file": (photo_path, photo, "image/png")}
                payload = {"content": caption}
                requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files)
        else:
            requests.post(DISCORD_WEBHOOK_URL, json={"content": caption})
    except Exception as e:
        print(f"Error sending to Discord: {e}")

def generate_chart(ticker, df):
    chart_data = df.tail(120).copy()
    sma20 = chart_data['Close'].rolling(20).mean()
    sma200 = chart_data['Close'].rolling(200).mean()

    add_plots = [
        mpf.make_addplot(sma20, color='blue', width=1.5),
        mpf.make_addplot(sma200, color='green', width=2.0)
    ]
    filename = f"{ticker}_2m.png"
    mpf.plot(
        chart_data,
        type='candle',
        style='yahoo',
        title=f"\n{ticker} - 2M Narrow State",
        addplot=add_plots,
        savefig=filename
    )
    return filename

def get_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url)[0]
        tickers = table['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers]
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "GOOGL", "TSLA"]

def scan_ticker(ticker):
    try:
        # yf.Ticker().history() provides clean 1D Series columns
        t_obj = yf.Ticker(ticker)
        df = t_obj.history(period="5d", interval="2m")
        if df.empty or len(df) < 200:
            return None

        close = df['Close']
        high = df['High']
        low = df['Low']

        latest_price = float(close.iloc[-1])
        if not (MIN_PRICE <= latest_price <= MAX_PRICE):
            return None

        sma20 = close.rolling(20).mean()
        sma200 = close.rolling(200).mean()

        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr14 = tr.rolling(14).mean()

        ma_dist_pct = (abs(sma20 - sma200) / close) * 100.0
        atr_pct = (atr14 / close) * 100.0

        narrow_mask = (ma_dist_pct <= MAX_MA_DIST_PCT) & (atr_pct <= MAX_ATR_PCT)

        # Check if Narrow State condition occurred in recent bars
        if bool(narrow_mask.iloc[-60:].any()):
            chart_file = generate_chart(ticker, df)
            return {
                "Ticker": ticker,
                "Price": round(latest_price, 2),
                "MA_Dist_%": round(float(ma_dist_pct.iloc[-1]), 3),
                "Chart": chart_file
            }
    except Exception as e:
        print(f"Error scanning {ticker}: {e}")
        return None
    return None

def main():
    tickers = get_tickers()
    send_to_discord(f"🔍 **Starting Daily Post-Market Narrow State Scan ({len(tickers)} stocks)...**")

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for res in executor.map(scan_ticker, tickers):
            if res:
                results.append(res)

    if results:
        send_to_discord(f"✅ **Found {len(results)} Narrow State Candidate(s):**")
        for item in results:
            caption = f"📊 **{item['Ticker']}** | Price: ${item['Price']} | 20/200 SMA Dist: {item['MA_Dist_%']}%"
            send_to_discord(caption, item['Chart'])
            if os.path.exists(item['Chart']):
                os.remove(item['Chart'])
    else:
        send_to_discord("ℹ️ No stocks met the Narrow State criteria today.")

if __name__ == "__main__":
    main()
