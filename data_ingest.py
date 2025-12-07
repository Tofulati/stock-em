import os
import pandas as pd
import yfinance as yf
from config import DATA_DIR, TICKERS

os.makedirs(DATA_DIR, exist_ok=True)

def download_prices(ticker, period="1y", interval="1d"):
    """
    Download price data for a single ticker and return a DataFrame with proper datetime index.
    
    CRITICAL FIX: Ensures the returned DataFrame has a DatetimeIndex, not integer index.
    This prevents downstream date conversion issues.
    """
    print(f"Downloading {ticker} data from Yahoo Finance...")
    
    # Download data
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    
    if df.empty:
        raise ValueError(f"Yahoo Finance returned no data for {ticker}")
    
    # 🔧 Flatten MultiIndex columns like ('Close', 'AAPL') → 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    
    # CRITICAL: Ensure index is datetime, not reset to integer
    # The index from yfinance is already a DatetimeIndex, so keep it!
    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"  Warning: Index is not DatetimeIndex, attempting to convert...")
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        else:
            df.index = pd.to_datetime(df.index)
    
    # Standardize column names (lowercase)
    df.columns = [c.lower() for c in df.columns]
    
    # Rename 'adj close' to 'close' if it exists
    if 'adj close' in df.columns:
        df = df.rename(columns={'adj close': 'close'})
    
    # Verify we have the datetime index
    print(f"  Data shape: {df.shape}")
    print(f"  Index type: {type(df.index)}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    # Save with datetime index preserved
    path = os.path.join(DATA_DIR, f"{ticker}_prices.parquet")
    df.to_parquet(path, index=True)  # CRITICAL: index=True preserves datetime index
    print(f"  Saved {ticker} data to {path}")
    
    return df


def load_prices(ticker):
    """
    Load previously downloaded price data.
    Ensures datetime index is preserved.
    """
    path = os.path.join(DATA_DIR, f"{ticker}_prices.parquet")
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No local cache found for {ticker}. "
            f"Please run download_prices('{ticker}') first."
        )
    
    df = pd.read_parquet(path)
    
    # Verify datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"  Warning: Loaded data has non-datetime index. Converting...")
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        else:
            df.index = pd.to_datetime(df.index)
    
    print(f"Loaded {ticker} data: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    
    return df


def download_all_prices(tickers=None):
    """Download data for multiple tickers"""
    tickers = tickers or TICKERS
    
    results = {}
    for ticker in tickers:
        try:
            df = download_prices(ticker)
            results[ticker] = df
        except Exception as e:
            print(f"  Error downloading {ticker}: {e}")
            results[ticker] = None
    
    return results


if __name__ == '__main__':
    # Test the functions
    print("Testing data download...")
    
    # Download single ticker
    df = download_prices('AAPL', period='1y')
    
    print(f"\nVerification:")
    print(f"  Index is DatetimeIndex: {isinstance(df.index, pd.DatetimeIndex)}")
    print(f"  First 3 rows:")
    print(df.head(3))
    
    # Test loading
    print("\nTesting load...")
    df_loaded = load_prices('AAPL')
    print(f"  Loaded index is DatetimeIndex: {isinstance(df_loaded.index, pd.DatetimeIndex)}")