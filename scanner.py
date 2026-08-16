import os
import json
import itertools
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

SCRIPT_VERSION = "v7.0"
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

# --- Scan Settings ---
MIN_PRICE = 50.0
MAX_PRICE = 400.0             # Price range $50 - $400
MIN_DAILY_VOLUME = 2_000_000  # Minimum 2 Million shares average daily volume
MIN_DAILY_ATR = 1.50          # Minimum Daily ATR of $1.50
TOP_COUNT = 50                # Deliver all qualified candidates

# --- Oliver Velez Strict Narrow State & Anti-Cascade Constraints (v7.0) ---
MAX_200_SLOPE_30M = 0.10      # 200 SMA must be flat (max 0.10% slope in last 30 bars)
MAX_20_SLOPE_15M = 0.08       # 20 SMA must be flat (max 0.08% slope, rejects plunging 20 SMAs like V & SO)
MAX_CLOSING_MA_GAP = 0.20     # Max 0.20% MA gap (captures CSCO, NVDA, TSLA, NFLX; rejects JPM & PG)
MAX_CLOSING_15M_BOX = 0.30    # Final 15-min price box <= 0.30% (rejects V dump, SO collapse, IWM rally)
MAX_DIRECTIONAL_STREAK = 4    # Max 4 consecutive same-color bars into close (rejects multi-bar cascades/rallies)

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
        if df.empty or len(df) < 50:
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

        # --- Rule 1: 200 SMA Slope over last 30 bars (Must be flat: <= 0.10%)
        sma200_slope = float(abs(sma200.iloc[-1] - sma200.iloc[-30]) / latest_price * 100)
        if sma200_slope > MAX_200_SLOPE_30M:
            return None

        # --- Rule 2: 20 SMA Slope over last 15 bars (Must be flat: <= 0.08%, rejects plunging 20 SMAs like V and SO)
        sma20_slope = float(abs(sma20.iloc[-1] - sma20.iloc[-15]) / latest_price * 100)
        if sma20_slope > MAX_20_SLOPE_15M:
            return None

        # --- Rule 3: Moving Average Gap over closing 15 bars (<= 0.20%, rejects JPM)
        closing_ma_gap = float(ma_dist_pct.iloc[-15:].mean())
        if closing_ma_gap > MAX_CLOSING_MA_GAP:
            return None

        # --- Rule 4: Final 15-Minute Price Box (<= 0.30%, rejects late dumps/collapses)
        closing_15m_box = float((high.iloc[-8:].max() - low.iloc[-8:].min()) / latest_price * 100)
        if closing_15m_box > MAX_CLOSING_15M_BOX:
            return None

        # --- Rule 5: Anti-Cascade Streak Check (Rejects multi-bar cascading declines like V & SO, and rallies like IWM)
        recent_closes = close.iloc[-8:].values
        diffs = np.diff(recent_closes)
        is_red = diffs < 0
        is_green = diffs > 0
        
        max_red_streak = max([sum(1 for _ in group) for val, group in itertools.groupby(is_red) if val], default=0)
        max_green_streak = max([sum(1 for _ in group) for val, group in itertools.groupby(is_green) if val], default=0)
        
        if max_red_streak > MAX_DIRECTIONAL_STREAK or max_green_streak > MAX_DIRECTIONAL_STREAK:
            return None

        # Tier Classification
        is_tight_ma = closing_ma_gap <= 0.06
        is_flat_200 = sma200_slope <= 0.05
        is_tight_box = closing_15m_box <= 0.20

        if is_tight_ma and is_flat_200 and is_tight_box:
            tier_num = 1
            tier_label = "🔥 Tier 1: Perfect Flat 200 & 20 Pin (Best)"
        elif closing_ma_gap <= 0.16:
            tier_num = 2
            tier_label = "⚡ Tier 2: Parallel Narrow State Squeeze (OK)"
        else:
            tier_num = 3
            tier_label = "⏱️ Tier 3: Converging Squeeze (OK)"

        squeeze_score = round((4.0 * closing_ma_gap) + (3.0 * sma200_slope) + (2.0 * sma20_slope) + closing_15m_box, 4)

        return {
            "Ticker": ticker,
            "Price": round(latest_price, 2),
            "Daily_ATR": round(daily_atr, 2),
            "Avg_Volume": int(avg_daily_volume),
            "SMA200_Slope_%": round(sma200_slope, 4),
            "MA_Gap_%": round(closing_ma_gap, 3),
            "Closing_15m_Box_%": round(closing_15m_box, 3),
            "Tier_Num": tier_num,
            "Tier_Label": tier_label,
            "Score": squeeze_score,
            "df": df
        }
    except Exception as e:
        return None

def main():
    tickers = get_tickers()
    send_to_discord(f"🔍 **[{SCRIPT_VERSION}] Scanning {len(tickers)} stocks for Oliver Velez Narrow States (Anti-Cascade Active)...**")

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for res in executor.map(scan_ticker, tickers):
            if res:
                results.append(res)

    if results:
        sorted_results = sorted(results, key=lambda x: (x["Tier_Num"], x["Score"]))
        top_candidates = sorted_results[:TOP_COUNT]

        send_to_discord(f"🎯 **[{SCRIPT_VERSION}] Found {len(top_candidates)} Oliver Velez Narrow State Candidates:**")

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
        send_to_discord(f"ℹ️ [{SCRIPT_VERSION}] No stocks met the criteria today.")

if __name__ == "__main__":
    main()
