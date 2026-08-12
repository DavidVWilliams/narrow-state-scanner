import os
import json
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

# --- Scan Settings ---
MIN_PRICE = 50.0
MAX_PRICE = 200.0
MIN_DAILY_VOLUME = 2_000_000  # Minimum 2 Million shares average daily volume
TOP_COUNT = 10                # Deliver Top 10 structural squeezes

def send_to_discord(caption, photo_path=None):
    if not DISCORD_WEBHOOK_URL:
        print("NO WEBHOOK URL set.")
        return
    
    try:
        if photo_path and os.path.exists(photo_path):
            filename = os.path.basename(photo_path)
            with open(photo_path, "rb") as photo:
                files = {
                    "files[0]": (filename, photo, "image/png")
                }
                payload = {
                    "content": caption,
                    "attachments": [
                        {
                            "id": 0,
                            "filename": filename
                        }
                    ]
                }
                res = requests.post(
                    DISCORD_WEBHOOK_URL,
                    data={"payload_json": json.dumps(payload)},
                    files=files
                )
                print(f"Discord Upload Response for {filename}: {res.status_code}")
        else:
            res = requests.post(DISCORD_WEBHOOK_URL, json={"content": caption})
            print(f"Discord Text Response: {res.status_code}")
    except Exception as e:
        print(f"Error sending to Discord: {e}")

def generate_chart(ticker, df, tier_label):
    try:
        df_calc = df.copy()
        df_calc['SMA20'] = df_calc['Close'].rolling(20).mean()
        df_calc['SMA200'] = df_calc['Close'].rolling(200).mean()

        chart_data = df_calc.tail(120).copy()

        if hasattr(chart_data.index, 'tz_localize') and chart_data.index.tz is not None:
            chart_data.index = chart_data.index.tz_localize(None)

        add_plots = [
            mpf.make_addplot(chart_data['SMA20'], color='blue', width=1.5),
            mpf.make_addplot(chart_data['SMA200'], color='red', width=2.0)
        ]
        filename = f"{ticker}_2m.png"
        mpf.plot(
            chart_data,
            type='candle',
            style='yahoo',
            title=f"\n{ticker} - {tier_label}",
            addplot=add_plots,
            savefig=dict(fname=filename, dpi=100)
        )
        plt.close('all')
        return filename
    except Exception as e:
        print(f"Chart generation error for {ticker}: {e}")
        return None

def get_tickers():
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        df_sp = pd.read_csv(url)
        sp_tickers = df_sp['Symbol'].tolist()
        
        extra_tickers = ["PLTR", "DIS", "QQQ", "SPY", "IWM", "TSLA", "NVDA", "AMD", "AMZN", "META", "GOOGL", "AAPL", "MSFT", "SOFI", "HOOD", "UBER", "ABNB", "COIN", "MARA", "RIOT", "DKNG", "SNAP", "SQ", "SHOP", "RBLX", "PALO"]
        
        all_tickers = list(set(sp_tickers + extra_tickers))
        return [t.replace('.', '-') for t in all_tickers]
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return ["PLTR", "DIS", "AMD", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "INTC", "PYPL", "QCOM"]

def scan_ticker(ticker):
    try:
        t_obj = yf.Ticker(ticker)
        # 7d ensures len(df) is never under 200 bars
        df = t_obj.history(period="7d", interval="2m", prepost=False)
        if df.empty or len(df) < 50:
            return None

        close = df['Close']
        open_p = df['Open']
        volume = df['Volume']

        latest_price = float(close.iloc[-1])
        if not (MIN_PRICE <= latest_price <= MAX_PRICE):
            return None

        avg_daily_volume = float(volume.resample('1D').sum().mean())
        if avg_daily_volume < MIN_DAILY_VOLUME:
            return None

        sma20 = close.rolling(20).mean()
        sma200 = close.rolling(200).mean()

        # --- Structural Metrics ---
        # 1. 200 SMA Flatness over last 30 bars (3:00 PM - 4:00 PM ET)
        sma200_flatness = float(abs(sma200.iloc[-1] - sma200.iloc[-30]) / latest_price * 100)

        # 2. 20 SMA Flatness over last 15 bars
        sma20_flatness = float(abs(sma20.iloc[-1] - sma20.iloc[-15]) / latest_price * 100)

        # 3. MA Convergence at close
        ma_convergence = float(abs(sma20.iloc[-1] - sma200.iloc[-1]) / latest_price * 100)

        # 4. Price Consolidation: Candle body range (max body high - min body low) over last 10 bars
        body_high = np.maximum(open_p, close)
        body_low = np.minimum(open_p, close)
        price_consolidation = float((body_high.iloc[-10:].max() - body_low.iloc[-10:].min()) / latest_price * 100)

        # 5. 20 SMA Wiggle/Path Length (eliminates saw-tooth/oscillating stocks)
        sma20_path = float(abs(sma20.diff()).iloc[-30:].sum() / latest_price * 100)

        # Structural Hard Filters
        if sma200_flatness > 0.08 or ma_convergence > 0.08 or price_consolidation > 0.25 or sma20_path > 0.25:
            return None

        # Categorize setup quality
        if sma200_flatness <= 0.05 and sma20_flatness <= 0.06 and ma_convergence <= 0.05:
            tier_num = 1
            tier_label = "🔥 Tier 1: Perfect Flat 200 & 20 Squeeze"
        elif sma200_flatness <= 0.05:
            tier_num = 2
            tier_label = "⚡ Tier 2: Flat 200 Magnet Squeeze"
        else:
            tier_num = 3
            tier_label = "⏱️ Tier 3: Consolidating Squeeze"

        # Structural Squeeze Score (Lower is flatter and more consolidated)
        structural_score = round(
            (5.0 * ma_convergence) + 
            (4.0 * sma200_flatness) + 
            (3.0 * sma20_flatness) + 
            (2.0 * price_consolidation) + 
            sma20_path, 4
        )

        return {
            "Ticker": ticker,
            "Price": round(latest_price, 2),
            "Avg_Volume": int(avg_daily_volume),
            "MA_Gap_%": round(ma_convergence, 3),
            "200_Flat_%": round(sma200_flatness, 4),
            "Body_Box_%": round(price_consolidation, 3),
            "Tier_Num": tier_num,
            "Tier_Label": tier_label,
            "Score": structural_score,
            "df": df
        }
    except Exception as e:
        return None

def main():
    tickers = get_tickers()
    send_to_discord(f"🔍 **Scanning {len(tickers)} stocks for Pure Structural Narrow States...**")

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for res in executor.map(scan_ticker, tickers):
            if res:
                results.append(res)

    if results:
        sorted_results = sorted(results, key=lambda x: (x["Tier_Num"], x["Score"]))
        top_candidates = sorted_results[:TOP_COUNT]

        send_to_discord(f"🎯 **Found {len(top_candidates)} Pure Structural Oliver Velez Candidates:**")

        for item in top_candidates:
            chart_file = generate_chart(item['Ticker'], item['df'], item['Tier_Label'])
            caption = f"📊 **{item['Ticker']}** | {item['Tier_Label']} | Price: ${item['Price']} | MA Gap: {item['MA_Gap_%']}% | Vol: {item['Avg_Volume']:,}"
            send_to_discord(caption, chart_file)
            if chart_file and os.path.exists(chart_file):
                try:
                    os.remove(chart_file)
                except Exception:
                    pass
    else:
        send_to_discord("ℹ️ No stocks met the structural criteria today.")

if __name__ == "__main__":
    main()
