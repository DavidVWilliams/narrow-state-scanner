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

SCRIPT_VERSION = "v2.6"
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

# --- Scan Settings (Price-Neutral) ---
MIN_PRICE = 50.0
MAX_PRICE = 400.0
MIN_DAILY_VOLUME = 2_000_000  # Minimum 2 Million shares (Pure share volume, no dollar bias)
MIN_DAILY_ATR = 1.50          # Minimum Daily ATR of $1.50
TOP_COUNT = 50                # Deliver Top 10 candidates

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
        
        extra_tickers = ["PLTR", "DIS", "QQQ", "SPY", "IWM", "TSLA", "NVDA", "AMD", "AMZN", "META", "GOOGL", "AAPL", "MSFT", "SOFI", "HOOD", "UBER", "ABNB", "COIN", "MARA", "RIOT", "DKNG", "SNAP", "SQ", "SHOP", "RBLX", "PALO"]
        
        all_tickers = list(set(sp_tickers + extra_tickers))
        return [t.replace('.', '-') for t in all_tickers]
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return ["PLTR", "DIS", "AMD", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "INTC", "PYPL", "QCOM"]

def scan_ticker(ticker):
    try:
        t_obj = yf.Ticker(ticker)
        
        # 1. Daily ATR Check (>= $1.50)
        df_daily = t_obj.history(period="1mo", interval="1d")
        if df_daily.empty or len(df_daily) < 14:
            return None
            
        d_tr = np.maximum(
            df_daily['High'] - df_daily['Low'],
            np.maximum(abs(df_daily['High'] - df_daily['Close'].shift(1)), abs(df_daily['Low'] - df_daily['Close'].shift(1)))
        )
        daily_atr = float(d_tr.rolling(14).mean().iloc[-1])
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

        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr14 = tr.rolling(14).mean()
        atr2m = float(atr14.iloc[-1])
        if atr2m <= 0: return None

        # --- ATR-Normalized Squeeze Metrics (Price-Neutral) ---
        # 1. MA Gap in cents relative to 2m ATR
        ma_gap_cents = float(abs(sma20.iloc[-1] - sma200.iloc[-1]))
        ma_gap_to_atr = ma_gap_cents / atr2m

        # 2. 200 SMA slope relative to Daily ATR
        sma200_slope_cents = float(abs(sma200.iloc[-1] - sma200.iloc[-30]))
        sma200_slope_to_atr = sma200_slope_cents / daily_atr

        # 3. 20 SMA slope relative to Daily ATR
        sma20_slope_cents = float(abs(sma20.iloc[-1] - sma20.iloc[-15]))
        sma20_slope_to_atr = sma20_slope_cents / daily_atr

        # 4. Price to 200 SMA distance relative to 2m ATR
        price_to_200_cents = float(abs(latest_price - sma200.iloc[-1]))
        price_to_200_to_atr = price_to_200_cents / atr2m

        # 5. Candle Body Range in final 15 mins relative to 2m ATR
        body_high = np.maximum(open_p, close)
        body_low = np.minimum(open_p, close)
        body_box_cents = float(body_high.iloc[-8:].max() - body_low.iloc[-8:].min())
        body_box_to_atr = body_box_cents / atr2m

        # Baseline filters (normalized to ATR)
        if ma_gap_to_atr > 0.60 or sma200_slope_to_atr > 0.10 or body_box_to_atr > 2.2:
            return None

        # --- Strict 5-Point Tier 1 Definition ---
        is_flat_200 = sma200_slope_to_atr <= 0.04
        is_flat_20 = sma20_slope_to_atr <= 0.04
        is_pinned_ma = ma_gap_to_atr <= 0.25
        is_tight_body = body_box_to_atr <= 1.20
        is_price_pinned = price_to_200_to_atr <= 0.35

        # 🔥 Tier 1: All 5 strict criteria must hold
        if is_flat_200 and is_flat_20 and is_pinned_ma and is_tight_body and is_price_pinned:
            tier_num = 1
            tier_label = "🔥 Tier 1: Pristine Flat 200 & 20 Squeeze (Best)"
        elif is_flat_200 and is_pinned_ma:
            tier_num = 2
            tier_label = "⚡ Tier 2: Flat 200 Magnet Squeeze (OK)"
        else:
            tier_num = 3
            tier_label = "⏱️ Tier 3: Consolidating Squeeze (OK)"

        # Price-neutral squeeze score
        score = round((4.0 * ma_gap_to_atr) + (3.0 * sma200_slope_to_atr) + (2.0 * price_to_200_to_atr) + body_box_to_atr, 4)

        return {
            "Ticker": ticker,
            "Price": round(latest_price, 2),
            "Daily_ATR": round(daily_atr, 2),
            "Avg_Volume": int(avg_daily_volume),
            "MA_Gap_Cents": round(ma_gap_cents, 2),
            "Tier_Num": tier_num,
            "Tier_Label": tier_label,
            "Score": score,
            "df": df
        }
    except Exception as e:
        return None

def main():
    tickers = get_tickers()
    send_to_discord(f"🔍 **[{SCRIPT_VERSION}] Scanning {len(tickers)} stocks ($50-$400, Vol > 2M, Price-Neutral ATR Model)...**")

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for res in executor.map(scan_ticker, tickers):
            if res:
                results.append(res)

    if results:
        sorted_results = sorted(results, key=lambda x: (x["Tier_Num"], x["Score"]))
        top_candidates = sorted_results[:TOP_COUNT]

        send_to_discord(f"🎯 **[{SCRIPT_VERSION}] Top {len(top_candidates)} Price-Neutral Narrow State Candidates:**")

        for item in top_candidates:
            chart_file = generate_chart(item['Ticker'], item['df'], item['Tier_Label'])
            caption = f"📊 **{item['Ticker']}** | {item['Tier_Label']} | Price: ${item['Price']} | Daily ATR: ${item['Daily_ATR']} | MA Gap: ${item['MA_Gap_Cents']} | Vol: {item['Avg_Volume']:,}"
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
