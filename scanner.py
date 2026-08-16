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

SCRIPT_VERSION = "v8.2"
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

# --- Scan Settings ---
MIN_PRICE = 35.0              # Price range $35 - $350
MAX_PRICE = 350.0
MIN_DAILY_VOLUME = 2_000_000  # Minimum 2 Million shares average daily volume
MIN_DAILY_ATR = 1.50          # Minimum Daily ATR of $1.50
TOP_COUNT = 50                # Deliver all qualifying candidates

# --- Strict Pinning & Squeeze Constraints (v8.2) ---
MAX_200_SLOPE_2H = 0.08       # 200 SMA must be flat over last 2 hours (<= 0.08%, rejects IWM [0.16%])
MAX_MA_GAP = 0.22             # 20 & 200 SMA gap <= 0.22% (captures CSCO [0.18%], NVDA [0.15%], TSLA, NFLX, PLTR, DIS)
MAX_PRICE_TO_200_DIST = 0.18  # Price must be within 0.18% of 200 SMA at 4:00 PM (Rejects MCD, TGT, IWM, V, SO)

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
            title=f"\n[{SCRIPT_VERSION}] {ticker} - {tier_label}",
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
        
        extra_tickers = ["CSCO", "NVDA", "TSLA", "NFLX", "PLTR", "DIS", "GOOG", "QQQ", "SPY", "IWM", "AMD", "AMZN", "META", "AAPL", "MSFT", "SOFI", "HOOD", "UBER", "ABNB", "COIN", "MARA", "RIOT", "DKNG", "SNAP", "SQ", "SHOP", "RBLX", "PALO"]
        
        all_tickers = list(set(sp_tickers + extra_tickers))
        return [t.replace('.', '-') for t in all_tickers]
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return ["CSCO", "NVDA", "TSLA", "NFLX", "PLTR", "DIS", "GOOG", "AMD", "AMZN", "META"]

def scan_ticker(ticker):
    try:
        t_obj = yf.Ticker(ticker)
        
        # 1. 14-Day Daily ATR Check (>= $1.50)
        df_daily = t_obj.history(period="6mo", interval="1d")
        if df_daily.empty or len(df_daily) < 14:
            return None
            
        d_tr = np.maximum(
            df_daily['High'] - df_daily['Low'],
            np.maximum(abs(df_daily['High'] - df_daily['Close'].shift(1)), abs(df_daily['Low'] - df_daily['Close'].shift(1)))
        )
        daily_atr = float(d_tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
        if daily_atr < MIN_DAILY_ATR:
            return None

        # 2. Intraday 2m data (RTH only)
        df = t_obj.history(period="7d", interval="2m", prepost=False)
        if df.empty or len(df) < 60:
            return None

        close = df['Close']
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

        # --- Rule 1: 2-Hour 200 SMA Flatness (Must be <= 0.08%, Rejects IWM & V multi-hour trends) ---
        sma200_2h_slope = float(abs(sma200.iloc[-1] - sma200.iloc[-60]) / latest_price * 100)
        if sma200_2h_slope > MAX_200_SLOPE_2H:
            return None

        # --- Rule 2: Moving Average Gap at Close (<= 0.22%) ---
        closing_ma_gap = float(ma_dist_pct.iloc[-15:].mean())
        if closing_ma_gap > MAX_MA_GAP:
            return None

        # --- Rule 3: 4:00 PM Price Pinning (<= 0.18%, Hard-rejects MCD, TGT, IWM, V, SO multi-bar moves) ---
        price_to_200_gap = float(abs(latest_price - sma200.iloc[-1]) / latest_price * 100)
        if price_to_200_gap > MAX_PRICE_TO_200_DIST:
            return None

        sma20_slope = float(abs(sma20.iloc[-1] - sma20.iloc[-15]) / latest_price * 100)

        # Volatility Compression on 2m candles
        bb_std = close.rolling(20).std().iloc[-1]
        bb_width_pct = (2.0 * bb_std / latest_price) * 100.0

        # Tier Classification
        is_tight_ma = closing_ma_gap <= 0.06
        is_flat_200 = sma200_2h_slope <= 0.04
        is_compressed = bb_width_pct <= 0.25

        if is_tight_ma and is_flat_200 and is_compressed:
            tier_num = 1
            tier_label = "🔥 Tier 1: Parallel Flat Squeeze (Best)"
        elif closing_ma_gap <= 0.16:
            tier_num = 2
            tier_label = "⚡ Tier 2: Parallel Narrow State Squeeze (OK)"
        else:
            tier_num = 3
            tier_label = "⏱️ Tier 3: Converging Squeeze (OK)"

        # Squeeze Score
        squeeze_score = round((5.0 * sma200_2h_slope) + (4.0 * closing_ma_gap) + (3.0 * price_to_200_gap) + bb_width_pct, 4)

        return {
            "Ticker": ticker,
            "Price": round(latest_price, 2),
            "Daily_ATR": round(daily_atr, 2),
            "Avg_Volume": int(avg_daily_volume),
            "SMA200_2H_Slope_%": round(sma200_2h_slope, 4),
            "MA_Gap_%": round(closing_ma_gap, 3),
            "Price_200_Gap_%": round(price_to_200_gap, 3),
            "Tier_Num": tier_num,
            "Tier_Label": tier_label,
            "Score": squeeze_score,
            "df": df
        }
    except Exception as e:
        return None

def main():
    tickers = get_tickers()
    send_to_discord(f"🔍 **[{SCRIPT_VERSION}] Scanning {len(tickers)} stocks for Pinned Narrow States ($35-$350, Vol > 2M)...**")

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for res in executor.map(scan_ticker, tickers):
            if res:
                results.append(res)

    if results:
        sorted_results = sorted(results, key=lambda x: (x["Tier_Num"], x["Score"]))
        top_candidates = sorted_results[:TOP_COUNT]

        send_to_discord(f"🎯 **[{SCRIPT_VERSION}] Found {len(top_candidates)} Pinned Narrow State Candidates:**")

        for item in top_candidates:
            chart_file = generate_chart(item['Ticker'], item['df'], item['Tier_Label'])
            caption = f"📊 **{item['Ticker']}** | {item['Tier_Label']} | Price: ${item['Price']} | 2H 200 Slope: {item['SMA200_2H_Slope_%']}% | MA Gap: {item['MA_Gap_%']}% | Vol: {item['Avg_Volume']:,}"
            send_to_discord(caption, chart_file)
            if chart_file and os.path.exists(chart_file):
                try:
                    os.remove(chart_file)
                except Exception:
                    pass
    else:
        send_to_discord(f"ℹ️ [{SCRIPT_VERSION}] No stocks met the criteria today.")

if __name__ == "__main__":
    main()
