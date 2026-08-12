import os
import requests
import pandas as pd
import numpy as np

# Force headless rendering backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf

import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

MIN_PRICE = 50.0
MAX_PRICE = 200.0

def send_to_discord(caption, photo_path=None):
    if not DISCORD_WEBHOOK_URL:
        print("NO WEBHOOK URL set.")
        return
    
    try:
        if photo_path and os.path.exists(photo_path):
            with open(photo_path, "rb") as photo:
                files = {"file": (os.path.basename(photo_path), photo, "image/png")}
                payload = {"content": caption}
                res = requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files)
                print(f"Discord Upload Response: {res.status_code}")
        else:
            requests.post(DISCORD_WEBHOOK_URL, json={"content": caption})
    except Exception as e:
        print(f"Error sending to Discord: {e}")

def generate_chart(ticker, df):
    try:
        # Calculate SMAs on full dataset FIRST so 200 SMA is valid
        df_calc = df.copy()
        df_calc['SMA20'] = df_calc['Close'].rolling(20).mean()
        df_calc['SMA200'] = df_calc['Close'].rolling(200).mean()

        # Slice last 120 candles AFTER SMAs are calculated
        chart_data = df_calc.tail(120)

        add_plots = [
            mpf.make_addplot(chart_data['SMA20'], color='blue', width=1.5),
            mpf.make_addplot(chart_data['SMA200'], color='green', width=2.0)
        ]
        filename = f"{ticker}_2m.png"
        mpf.plot(
            chart_data,
            type='candle',
            style='yahoo',
            title=f"\n{ticker} - 2M Chart (20 SMA Blue / 200 SMA Green)",
            addplot=add_plots,
            savefig=filename
        )
        plt.close('all')
        return filename
    except Exception as e:
        print(f"Chart generation error for {ticker}: {e}")
        return None

def get_tickers():
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        df = pd.read_csv(url)
        tickers = df['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers]
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return ["AMD", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "INTC", "PYPL", "QCOM", "TXN"]

def scan_ticker(ticker):
    try:
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

        latest_ma_dist = float(ma_dist_pct.dropna().iloc[-1])
        latest_atr_pct = float(atr_pct.dropna().iloc[-1])

        return {
            "Ticker": ticker,
            "Price": round(latest_price, 2),
            "MA_Dist_%": round(latest_ma_dist, 3),
            "ATR_%": round(latest_atr_pct, 3),
            "df": df
        }
    except Exception as e:
        return None

def main():
    tickers = get_tickers()
    send_to_discord(f"🔍 **Scanning S&P 500 for Top 5 Narrowest 20/200 SMA Squeezes ($50-$200)...**")

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for res in executor.map(scan_ticker, tickers):
            if res:
                results.append(res)

    if results:
        sorted_results = sorted(results, key=lambda x: x["MA_Dist_%"])
        top_candidates = sorted_results[:5]

        send_to_discord(f"🎯 **Top {len(top_candidates)} Narrow State Candidates Today:**")
        for item in top_candidates:
            chart_file = generate_chart(item['Ticker'], item['df'])
            caption = f"📊 **{item['Ticker']}** | Price: ${item['Price']} | 20/200 SMA Dist: {item['MA_Dist_%']}% | ATR: {item['ATR_%']}%"
            send_to_discord(caption, chart_file)
            if chart_file and os.path.exists(chart_file):
                try:
                    os.remove(chart_file)
                except Exception:
                    pass
    else:
        send_to_discord("⚠️ No eligible stocks met the price criteria today.")

if __name__ == "__main__":
    main()
