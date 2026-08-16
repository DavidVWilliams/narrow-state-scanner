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

SCRIPT_VERSION = "v5.0"
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

# --- Scan Settings ---
MIN_PRICE = 50.0
MAX_PRICE = 400.0             # Price range $50 - $400
MIN_DAILY_VOLUME = 2_000_000  # Minimum 2 Million shares average daily volume
MIN_DAILY_ATR = 1.50          # Minimum Daily ATR of $1.50
TOP_COUNT = 50                # Deliver all qualifying candidates

# --- Strict Oliver Velez Congestion Constraints (v5.0) ---
MAX_200_SLOPE_2H = 0.07       # 200 SMA must be flat over last 2 hours (<= 0.07%, rejects V, XOM, IWM, PG)
MAX_CLOSING_MA_GAP = 0.18     # 20 SMA & 200 SMA gap at close <= 0.18% (captures CSCO & PLTR, rejects PG, TJX, JPM)
MAX_CONGESTION_BOX_PCT = 0.35 # Final 15-bar price box <= 0.35% (rejects V dump, XOM drop, IBKR swings, TJX breakdown)
MIN_OVERLAP_RATIO = 0.70      # >= 70% of the last 15 bars must overlap adjacent bars (strictly enforces congestion)

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
        
        extra_tickers = ["CSCO", "GOOG", "PLTR", "DIS", "QQQ", "SPY", "IWM", "TSLA", "NVDA", "AMD", "AMZN", "META", "AAPL", "MSFT", "SOFI", "HOOD", "UBER", "ABNB", "COIN", "MARA", "RIOT", "DKNG", "SNAP", "SQ", "SHOP", "RBLX", "PALO"]
        
        all_tickers = list(set(sp_tickers + extra_tickers))
        return [t.replace('.', '-') for t in all_tickers]
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return ["CSCO", "GOOG", "PLTR", "DIS", "AMD", "NVDA", "AMZN", "META", "TSLA", "INTC", "PYPL", "QCOM"]

def scan_ticker(ticker):
    try:
        t_obj = yf.Ticker(ticker)
        
        # 1. 14-Day Daily ATR using Welles Wilder Smoothing (>= $1.50)
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
        high = df['High']
        low = df['Low']
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

        # --- Rule 1: 200 SMA Flatness over last 60 bars (2:00 PM - 4:00 PM ET) -> Must be <= 0.07%
        sma200_2h_slope = float(abs(sma200.iloc[-1] - sma200.iloc[-60]) / latest_price * 100)
        if sma200_2h_slope > MAX_200_SLOPE_2H:
            return None

        # --- Rule 2: 20 SMA & 200 SMA Gap at close (last 10 bars <= 0.18%)
        closing_ma_gap = float(ma_dist_pct.iloc[-10:].mean())
        if closing_ma_gap > MAX_CLOSING_MA_GAP:
            return None

        # --- Rule 3: 15-Bar Tight Congestion Box (3:30 PM - 4:00 PM ET <= 0.35%)
        recent_highs = high.iloc[-15:]
        recent_lows = low.iloc[-15:]
        congestion_box_pct = float((recent_highs.max() - recent_lows.min()) / latest_price * 100)
        if congestion_box_pct > MAX_CONGESTION_BOX_PCT:
            return None

        # --- Rule 4: Consecutive Bar Overlap Check (>= 70% of last 15 bars must overlap adjacent bars)
        overlap_count = 0
        for i in range(-14, 0):
            prev_low, prev_high = low.iloc[i-1], high.iloc[i-1]
            curr_low, curr_high = low.iloc[i], high.iloc[i]
            if max(prev_low, curr_low) <= min(prev_high, curr_high):
                overlap_count += 1
        
        overlap_ratio = overlap_count / 14.0
        if overlap_ratio < MIN_OVERLAP_RATIO:
            return None

        sma20_slope = float(abs(sma20.iloc[-1] - sma20.iloc[-15]) / latest_price * 100)
        price_to_200_gap = float(abs(latest_price - sma200.iloc[-1]) / latest_price * 100)

        # Categorize setup tier
        is_flat_20 = sma20_slope <= 0.06
        is_pinned_ma = closing_ma_gap <= 0.08
        is_tight_box = congestion_box_pct <= 0.25

        if is_flat_20 and is_pinned_ma and is_tight_box:
            tier_num = 1
            tier_label = "🔥 Tier 1: Tight Flat Congestion Pin (Best)"
        elif is_pinned_ma or sma200_2h_slope <= 0.05:
            tier_num = 2
            tier_label = "⚡ Tier 2: Flat 200 Squeeze Congestion (OK)"
        else:
            tier_num = 3
            tier_label = "⏱️ Tier 3: Consolidating Congestion (OK)"

        score = round((5.0 * sma200_2h_slope) + (4.0 * closing_ma_gap) + (2.0 * price_to_200_gap) + congestion_box_pct, 4)

        return {
            "Ticker": ticker,
            "Price": round(latest_price, 2),
            "Daily_ATR": round(daily_atr, 2),
            "Avg_Volume": int(avg_daily_volume),
            "SMA200_2H_Slope_%": round(sma200_2h_slope, 4),
            "MA_Gap_%": round(closing_ma_gap, 3),
            "Congestion_Box_%": round(congestion_box_pct, 3),
            "Overlap_%": round(overlap_ratio * 100, 1),
            "Tier_Num": tier_num,
            "Tier_Label": tier_label,
            "Score": score,
            "df": df
        }
    except Exception as e:
        return None

def main():
    tickers = get_tickers()
    send_to_discord(f"🔍 **[{SCRIPT_VERSION}] Scanning {len(tickers)} stocks for Tight Congestion Zones ($50-$400, Vol > 2M)...**")

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for res in executor.map(scan_ticker, tickers):
            if res:
                results.append(res)

    if results:
        sorted_results = sorted(results, key=lambda x: (x["Tier_Num"], x["Score"]))
        top_candidates = sorted_results[:TOP_COUNT]

        send_to_discord(f"🎯 **[{SCRIPT_VERSION}] Found {len(top_candidates)} Tight Congestion Narrow State Candidates:**")

        for item in top_candidates:
            chart_file = generate_chart(item['Ticker'], item['df'], item['Tier_Label'])
            caption = f"📊 **{item['Ticker']}** | {item['Tier_Label']} | Price: ${item['Price']} | Box: {item['Congestion_Box_%']}% | Overlap: {item['Overlap_%']}% | Vol: {item['Avg_Volume']:,}"
            send_to_discord(caption, chart_file)
            if chart_file and os.path.exists(chart_file):
                try:
                    os.remove(chart_file)
                except Exception:
                    pass
    else:
        send_to_discord(f"ℹ️ [{SCRIPT_VERSION}] No stocks met the tight congestion criteria today.")

if __name__ == "__main__":
    main()
