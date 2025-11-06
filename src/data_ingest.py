import os
import pandas as pd
import yfinance as yf
from .config import DATA_DIR, TICKERS

os.makedirs(DATA_DIR, exist_ok=True)

def download_prices(ticker, period="1y", interval="1d"):
    """
    Download price data for a single ticker and return a flat DataFrame.
    """
    print(f"Downloading {ticker} data from Yahoo Finance...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"Yahoo Finance returned no data for {ticker}")

    # 🔧 Flatten MultiIndex columns like ('Close', 'AAPL') → 'close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df.reset_index(inplace=True)
    df = df.rename(columns={"adj close": "close"})

    path = os.path.join(DATA_DIR, f"{ticker}_prices.parquet")
    df.to_parquet(path, index=False)
    print(f"Saved {ticker} data to {path}")
    return df

def load_prices(ticker):
    path = os.path.join(DATA_DIR, f"{ticker}_prices.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No local cache found for {ticker}. Please run download_prices() first.")
    return pd.read_parquet(path)

def download_all_prices(tickers=None):
    tickers = tickers or TICKERS
    for t in tickers:
        download_prices(t)
