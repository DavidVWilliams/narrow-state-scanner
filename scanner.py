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

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

def send_to_discord(caption, photo_path=None):
    if not DISCORD_WEBHOOK_URL:
        print("ERROR: DISCORD_WEBHOOK_URL environment variable is missing!")
        return
    
    try:
        if photo_path and os.path.exists(photo_path):
            with open(photo_path, "rb") as photo:
                files = {"file": ("chart.png", photo, "image/png")}
                payload = {"content": caption}
                response = requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files)
                print(f"Discord Response Status: {response.status_code}")
        else:
            response = requests.post(DISCORD_WEBHOOK_URL, json={"content": caption})
            print(f"Discord Response Status: {response.status_code}")
    except Exception as e:
        print(f"Error sending to Discord: {e}")

def main():
    print("Starting NVDA Test Run...")
    send_to_discord("🧪 **Test Run**: Fetching NVDA 2M chart to verify Discord connection...")

    ticker = "NVDA"
    t_obj = yf.Ticker(ticker)
    df = t_obj.history(period="5d", interval="2m")

    if df.empty:
        send_to_discord("❌ Test failed: Unable to fetch price data for NVDA from Yahoo Finance.")
        return

    # Generate NVDA 2M candlestick chart
    chart_data = df.tail(100).copy()
    sma20 = chart_data['Close'].rolling(20).mean()
    sma200 = chart_data['Close'].rolling(200).mean()

    add_plots = [
        mpf.make_addplot(sma20, color='blue', width=1.5),
        mpf.make_addplot(sma200, color='green', width=2.0)
    ]
    filename = "NVDA_test.png"
    mpf.plot(
        chart_data,
        type='candle',
        style='yahoo',
        title=f"\n{ticker} - 2M Test Chart (20 SMA Blue / 200 SMA Green)",
        addplot=add_plots,
        savefig=filename
    )
    plt.close('all')

    latest_price = round(float(df['Close'].iloc[-1]), 2)
    caption = f"📊 **NVDA Test Successful!** | Latest Price: ${latest_price} | Timeframe: 2-Min"
    send_to_discord(caption, filename)

    if os.path.exists(filename):
        os.remove(filename)

if __name__ == "__main__":
    main()
