import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .config import SEQ_LEN

def make_technical_features(df):
    df = df.copy()

    # Lowercase column names for consistency
    df.columns = [c.lower() for c in df.columns]

    # Compute returns and moving averages
    df['ret1'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()

    # 52-week high percentage
    window = min(252, len(df)//2)
    df['pct_from_52w_high'] = df['close'] / df['close'].rolling(window).max() - 1

    # Fill NaNs
    df = df.bfill().ffill()

    # Preserve Open, Close, Volume for backtesting and features
    df.rename(columns={'close':'Close', 'open':'Open', 'volume':'Volume'}, inplace=True)

    print(f"[make_technical_features] After processing: {len(df)} rows remain")
    return df


def build_sequences(df, seq_len=SEQ_LEN, features=['ret1','ma5','ma20','pct_from_52w_high','Volume']):
    # Ensure all features exist
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
