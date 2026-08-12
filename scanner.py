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
MAX_PRICE = 350.0             # Max price $350.00
MIN_DAILY_VOLUME = 2_000_000  # Minimum 2 Million shares average daily volume
MIN_DOLLAR_VOLUME = 500_000_000 # Minimum $500 Million daily turnover
MIN_DAILY_ATR = 1.50          # Minimum Daily ATR of $1.50
TOP_COUNT = 40                # Delivered candidate limit increased to 40

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
        
        # 1. Daily ATR check
        df_daily = t_obj.history(period="1mo", interval="1d")
        if df_daily.empty or len(df_daily) < 14:
            if ticker == "PLTR": send_to_discord("⚠️ PLTR Debug: Daily data empty")
            return None
            
        d_tr = np.maximum(
            df_daily['High'] - df_daily['Low'],
            np.maximum(abs(df_daily['High'] - df_daily['Close'].shift(1)), abs(df_daily['Low'] - df_daily['Close'].shift(1)))
        )
        daily_atr = float(d_tr.rolling(14).mean().iloc[-1])
        if daily_atr < MIN_DAILY_ATR:
            if ticker == "PLTR": send_to_discord(f"⚠️ PLTR Debug: Daily ATR ${daily_atr:.2f} < ${MIN_DAILY_ATR}")
            return None

        # 2. Intraday 2m data
        df = t_obj.history(period="7d", interval="2m", prepost=False)
        if df.empty or len(df) < 50:
            if ticker == "PLTR": send_to_discord("⚠️ PLTR Debug: Intraday 2m data empty")
            return None

        close = df['Close']
        volume = df['Volume']

        latest_price = float(close.iloc[-1])
        if not (MIN_PRICE <= latest_price <= MAX_PRICE):
            if ticker == "PLTR": send_to_discord(f"⚠️ PLTR Debug: Price ${latest_price} out of range")
            return None

        avg_daily_volume = float(volume.resample('1D').sum().mean())
        if avg_daily_volume < MIN_DAILY_VOLUME:
            if ticker == "PLTR": send_to_discord("⚠️ PLTR Debug: Volume too low")
            return None

        avg_dollar_volume = latest_price * avg_daily_volume
        if avg_dollar_volume < MIN_DOLLAR_VOLUME:
            if ticker == "PLTR": send_to_discord("⚠️ PLTR Debug: Dollar volume too low")
            return None

        sma20 = close.rolling(20).mean()
        sma200 = close.rolling(200).mean()

        ma_dist_pct = (abs(sma20 - sma200) / close) * 100.0

        sma200_slope = float(abs(sma200.iloc[-1] - sma200.iloc[-30]) / latest_price * 100)
        closing_ma_gap = float(ma_dist_pct.iloc[-5:].mean())
        price_to_200_gap = float(abs(latest_price - sma200.iloc[-1]) / latest_price * 100)

        # PLTR Status Report
        if ticker == "PLTR":
            send_to_discord(f"📊 **PLTR Status**: Price: ${latest_price:.2f} | Daily ATR: ${daily_atr:.2f} | 200 Slope: {sma200_slope:.4f}% | MA Gap: {closing_ma_gap:.3f}% | Price/200 Gap: {price_to_200_gap:.3f}%")

        if sma200_slope > 0.20 or closing_ma_gap > 0.25 or price_to_200_gap > 0.30:
            if ticker == "PLTR": send_to_discord("⚠️ PLTR Debug: Failed slope/gap filter")
            return None

        sma20_slope = float(abs(sma20.iloc[-1] - sma20.iloc[-15]) / latest_price * 100)

        is_flat_20 = sma20_slope <= 0.10

        if is_flat_20:
            tier_num = 1
            tier_label = "🔥 Tier 1: Perfect Flat 200 & 20 Pin"
        else:
            tier_num = 2
            tier_label = "⚡ Tier 2: Flat 200 Magnet Squeeze"

        squeeze_score = round((5.0 * sma200_slope) + (4.0 * closing_ma_gap) + (3.0 * price_to_200_gap), 4)

        return {
            "Ticker": ticker,
            "Price": round(latest_price, 2),
            "Daily_ATR": round(daily_atr, 2),
            "Avg_Volume": int(avg_daily_volume),
            "SMA200_Slope_%": round(sma200_slope, 4),
            "MA_Gap_%": round(closing_ma_gap, 3),
            "Price_200_Gap_%": round(price_to_200_gap, 3),
            "Tier_Num": tier_num,
            "Tier_Label": tier_label,
            "Score": squeeze_score,
            "df": df
        }
    except Exception as e:
        if ticker == "PLTR": send_to_discord(f"❌ PLTR Exception: {e}")
        return None

def main():
    tickers = get_tickers()
    send_to_discord(f"🔍 **Scanning {len(tickers)} stocks ($50-$400, Vol > 2M, Daily ATR > $1.50)...**")

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for res in executor.map(scan_ticker, tickers):
            if res:
                results.append(res)

    if results:
        sorted_results = sorted(results, key=lambda x: (x["Tier_Num"], x["Score"]))
        top_candidates = sorted_results[:TOP_COUNT]

        send_to_discord(f"🎯 **Found {len(top_candidates)} Candidates (Top {TOP_COUNT} Delivered):**")

        for item in top_candidates:
            chart_file = generate_chart(item['Ticker'], item['df'], item['Tier_Label'])
            caption = f"📊 **{item['Ticker']}** | {item['Tier_Label']} | Price: ${item['Price']} | Daily ATR: ${item['Daily_ATR']} | MA Gap: {item['MA_Gap_%']}% | Vol: {item['Avg_Volume']:,}"
            send_to_discord(caption, chart_file)
            if chart_file and os.path.exists(chart_file):
                try:
                    os.remove(chart_file)
                except Exception:
                    pass
    else:
        send_to_discord("ℹ️ No stocks met the criteria today.")

if __name__ == "__main__":
    main()
