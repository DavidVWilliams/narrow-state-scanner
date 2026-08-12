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
        print("ERROR: DISCORD_WEBHOOK_URL is missing!")
        return
    
    try:
        if photo_path and os.path.exists(photo_path):
            with open(photo_path, "rb") as photo:
                files = {"file": (os.path.basename(photo_path), photo, "image/png")}
                payload = {"content": caption}
                res = requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files)
                print(f"Discord Photo Upload Status: {res.status_code}")
                if res.status_code not in [200, 204]:
                    requests.post(DISCORD_WEBHOOK_URL, json={"content": f"{caption}\n*(Photo status code: {res.status_code})*"})
        else:
            res = requests.post(DISCORD_WEBHOOK_URL, json={"content": caption})
            print(f"Discord Text Status: {res.status_code}")
    except Exception as e:
        print(f"Error sending to Discord: {e}")

def main():
    send_to_discord("🧪 **Running Test Diagnostic...**")

    ticker = "NVDA"
    try:
        df = yf.download(ticker, period="5d", interval="2m", progress=False)
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        rows = len(df)
        send_to_discord(f"📈 **Data Fetched**: {ticker} returned {rows} candles.")

        if df.empty or rows < 10:
            send_to_discord("❌ Data dataframe was empty.")
            return

        latest_price = round(float(df['Close'].iloc[-1]), 2)

        # Generate chart
        chart_data = df.tail(100).copy()
        sma20 = chart_data['Close'].rolling(20).mean()
        sma200 = chart_data['Close'].rolling(200).mean()

        add_plots = [
            mpf.make_addplot(sma20, color='blue', width=1.5),
            mpf.make_addplot(sma200, color='green', width=2.0)
        ]
        filename = f"{ticker}_test.png"
        mpf.plot(
            chart_data,
            type='candle',
            style='yahoo',
            title=f"\n{ticker} - 2M Test Chart",
            addplot=add_plots,
            savefig=filename
        )
        plt.close('all')

        send_to_discord(f"📸 **Chart Generated!** Sending photo for {ticker} (${latest_price})...", filename)

        if os.path.exists(filename):
            os.remove(filename)

    except Exception as e:
        send_to_discord(f"❌ **Script Error**: `{str(e)}`")

if __name__ == "__main__":
    main()
