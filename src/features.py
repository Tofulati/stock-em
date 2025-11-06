import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .config import SEQ_LEN

def compute_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window).mean()
    ma_down = down.rolling(window).mean()
    rsi = 100 - 100 / (1 + ma_up / ma_down)
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def make_technical_features(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    # Core features
    df['ret1'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    window = min(252, len(df)//2)
    df['pct_from_52w_high'] = df['close'] / df['close'].rolling(window).max() - 1
    
    # New features
    df['rsi14'] = compute_rsi(df['close'], 14)
    df['macd'], df['macd_signal'] = compute_macd(df['close'])
    df['volatility20'] = df['ret1'].rolling(20).std()
    
    df = df.bfill().ffill()
    
    # Preserve original columns
    df.rename(columns={'close':'Close','open':'Open','volume':'Volume'}, inplace=True)
    
    print(f"[make_technical_features] After processing: {len(df)} rows remain")
    return df


def build_sequences(df, seq_len=SEQ_LEN, features=None):
    if features is None:
        features = ['ret1','ma5','ma20','pct_from_52w_high','Volume','PE','PB','DivYield','news_score']
    
    for f in features:
        if f not in df.columns:
            raise ValueError(f"Feature '{f}' not found in DataFrame columns: {df.columns}")
    
    arr = df[features].values
    X, idx = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i+seq_len])
        idx.append(df.index[i+seq_len])
    X = np.stack(X)
    return X, idx
