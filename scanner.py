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
MIN_DAILY_VOLUME = 1_000_000  # Min 1,000,000 shares average daily volume
TOP_COUNT = 10                # Max candidates to deliver

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

        ma_dist_pct = (abs(sma20 - sma200) / close) * 100.0

        # Slope measurements over the last 15 bars (30 minutes)
        sma20_slope = float(abs(sma20.iloc[-1] - sma20.iloc[-15]) / close.iloc[-1] * 100)
        sma200_slope = float(abs(sma200.iloc[-1] - sma200.iloc[-15]) / close.iloc[-1] * 100)

        # Body height relative to price over last 15-20 bars (measuring tight overlapping bars)
        body_height_pct = (abs(close - open_p) / close) * 100
        avg_body_15 = float(body_height_pct.iloc[-15:].mean())
        ma_dist_15 = float(ma_dist_pct.iloc[-15:].mean())

        # Oliver Velez Setup Conditions
        is_flat_20 = sma20_slope <= 0.08
        is_flat_200 = sma200_slope <= 0.08
        is_close_ma = ma_dist_15 <= 0.60
        is_tight_bars = avg_body_15 <= 0.20

        # Tier 1 (Best): Close & flat 20 and 200 with tight overlapping bars
        is_tier_1 = is_close_ma and is_flat_20 and is_flat_200 and is_tight_bars

        # Tier 2 (OK): Flat 200 with a close but slightly trending 20
        is_tier_2 = is_close_ma and is_flat_200 and (not is_flat_20)

        # Tier 3 (OK): Flat 20 with a sloping 200, if last 15-20 bars are tight & overlapping
        is_tier_3 = is_flat_20 and (not is_flat_200) and is_tight_bars

        if not (is_tier_1 or is_tier_2 or is_tier_3):
            return None

        if is_tier_1:
            tier_num = 1
            tier_label = "🔥 Tier 1: Flat 20/200 & Tight Bars (Best)"
        elif is_tier_2:
            tier_num = 2
            tier_label = "⚡ Tier 2: Flat 200 & Trending 20 (OK)"
        else:
            tier_num = 3
            tier_label = "⏱️ Tier 3: Flat 20, Sloping 200 & Tight Bars (OK)"

        score = round(ma_dist_15 + avg_body_15, 4)

        return {
            "Ticker": ticker,
            "Price": round(latest_price, 2),
            "Avg_Volume": int(avg_daily_volume),
            "MA_Dist_%": round(ma_dist_15, 3),
            "Tier_Num": tier_num,
            "Tier_Label": tier_label,
            "Score": score,
            "df": df
        }
    except Exception as e:
        return None

def main():
    tickers = get_tickers()
    send_to_discord(f"🔍 **Scanning {len(tickers)} stocks for Oliver Velez Narrow State Setup Tiers ($50-$200)...**")

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for res in executor.map(scan_ticker, tickers):
            if res:
                results.append(res)

    if results:
        # Sort by Tier Number (Tier 1 Best first), then by tightest Squeeze Score
        sorted_results = sorted(results, key=lambda x: (x["Tier_Num"], x["Score"]))
        top_candidates = sorted_results[:TOP_COUNT]

        send_to_discord(f"🎯 **Top {len(top_candidates)} Oliver Velez Refined Narrow State Candidates:**")

        for item in top_candidates:
            chart_file = generate_chart(item['Ticker'], item['df'], item['Tier_Label'])
            caption = f"📊 **{item['Ticker']}** | {item['Tier_Label']} | Price: ${item['Price']} | SMA Dist: {item['MA_Dist_%']}% | Vol: {item['Avg_Volume']:,}"
            send_to_discord(caption, chart_file)
            if chart_file and os.path.exists(chart_file):
                try:
                    os.remove(chart_file)
                except Exception:
                    pass
    else:
        send_to_discord("⚠️ No eligible stocks met the setup criteria today.")

if __name__ == "__main__":
    main()
