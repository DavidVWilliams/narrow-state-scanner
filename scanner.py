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
TOP_COUNT = 10                # Max candidates to deliver

# --- Strict Overlap & Flatness Constraints ---
MAX_MA_DIST_PCT = 0.15        # MAs must stay glued together (max 0.15% gap in last 20 bars)
MAX_SMA20_RANGE_PCT = 0.10    # 20 SMA must be horizontal (max 0.10% move over last 30 bars)
MAX_SMA200_RANGE_PCT = 0.05   # 200 SMA must be horizontal (max 0.05% move over last 30 bars)
MAX_PRICE_BOX_PCT = 0.30      # All price bars must overlap in a tight 0.30% channel

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
        
        extra_tickers = ["DIS", "QQQ", "SPY", "IWM", "TSLA", "NVDA", "AMD", "AMZN", "META", "GOOGL", "AAPL", "MSFT", "PLTR", "SOFI", "HOOD", "UBER", "ABNB", "COIN", "MARA", "RIOT", "DKNG", "SNAP", "SQ", "SHOP", "RBLX", "PALO"]
        
        all_tickers = list(set(sp_tickers + extra_tickers))
        return [t.replace('.', '-') for t in all_tickers]
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return ["DIS", "AMD", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "INTC", "PYPL", "QCOM"]

def scan_ticker(ticker):
    try:
        t_obj = yf.Ticker(ticker)
        df = t_obj.history(period="5d", interval="2m")
        if df.empty or len(df) < 200:
            return None

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        latest_price = float(close.iloc[-1])
        if not (MIN_PRICE <= latest_price <= MAX_PRICE):
            return None

        # Filter by minimum 2M average daily volume
        avg_daily_volume = float(volume.resample('1D').sum().mean())
        if avg_daily_volume < MIN_DAILY_VOLUME:
            return None

        sma20 = close.rolling(20).mean()
        sma200 = close.rolling(200).mean()

        ma_dist_pct = (abs(sma20 - sma200) / close) * 100.0

        # 1. Check Maximum MA distance in last 20 bars
        max_ma_dist = float(ma_dist_pct.iloc[-20:].max())
        if max_ma_dist > MAX_MA_DIST_PCT:
            return None

        # 2. Check 20 SMA & 200 SMA flatness over last 30 bars
        sma20_range = float((sma20.iloc[-30:].max() - sma20.iloc[-30:].min()) / latest_price * 100)
        sma200_range = float((sma200.iloc[-30:].max() - sma200.iloc[-30:].min()) / latest_price * 100)

        # 3. Check Price Overlap Channel
        price_box_pct = float((high.iloc[-20:].max() - low.iloc[-20:].min()) / latest_price * 100)
        if price_box_pct > MAX_PRICE_BOX_PCT:
            return None

        # Categorize setup quality
        is_flat_20 = sma20_range <= MAX_SMA20_RANGE_PCT
        is_flat_200 = sma200_range <= MAX_SMA200_RANGE_PCT

        if is_flat_20 and is_flat_200:
            tier_num = 1
            tier_label = "🔥 Tier 1: Perfect Flat 20/200 Squeeze"
        elif is_flat_200:
            tier_num = 2
            tier_label = "⚡ Tier 2: Flat 200 & Trending 20"
        elif is_flat_20:
            tier_num = 3
            tier_label = "⏱️ Tier 3: Flat 20 & Sloping 200"
        else:
            return None

        score = round(max_ma_dist + price_box_pct, 4)

        return {
            "Ticker": ticker,
            "Price": round(latest_price, 2),
            "Avg_Volume": int(avg_daily_volume),
            "Max_MA_Dist_%": round(max_ma_dist, 3),
            "Price_Box_%": round(price_box_pct, 3),
            "Tier_Num": tier_num,
            "Tier_Label": tier_label,
            "Score": score,
            "df": df
        }
    except Exception as e:
        return None

def main():
    tickers = get_tickers()
    send_to_discord(f"🔍 **Scanning {len(tickers)} stocks for Strict Overlapping Squeezes ($50-$200, Vol > 2M)...**")

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for res in executor.map(scan_ticker, tickers):
            if res:
                results.append(res)

    if results:
        sorted_results = sorted(results, key=lambda x: (x["Tier_Num"], x["Score"]))
        top_candidates = sorted_results[:TOP_COUNT]

        send_to_discord(f"🎯 **Top {len(top_candidates)} Refined Narrow State Candidates (Vol > 2M):**")

        for item in top_candidates:
            chart_file = generate_chart(item['Ticker'], item['df'], item['Tier_Label'])
            caption = f"📊 **{item['Ticker']}** | {item['Tier_Label']} | Price: ${item['Price']} | MA Gap: {item['Max_MA_Dist_%']}% | Box: {item['Price_Box_%']}% | Vol: {item['Avg_Volume']:,}"
            send_to_discord(caption, chart_file)
            if chart_file and os.path.exists(chart_file):
                try:
                    os.remove(chart_file)
                except Exception:
                    pass
    else:
        send_to_discord("⚠️ No stocks met the strict overlapping squeeze criteria (Min Vol > 2M) today.")

if __name__ == "__main__":
    main()
